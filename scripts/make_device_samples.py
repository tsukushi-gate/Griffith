"""配布バンドルから「デバイス用サンプル集」(入力 + ホストの期待出力)を作る。

実機(ESP32 等)で `mdl_infer()` / `mdl_infer_image()` を動かし、**ホスト側 ir/simulate の
整数出力と一致するか**(+ レイテンシ計測の入力)を確かめるためのデータを、バンドル 1 つと
データセットから汎用に生成する。期待出力の基準は常に ir/simulate(numpy の素朴実装)で、
生成した C ランタイムとの一致は `--check-c`(既定 ON)でその場でも確認する。

入力の種類(バンドルから自動判定):

    bits      バンドルに前処理記述子が無い(dense 系)。入力は符号化済みビット列。
              サンプルは **LSB-first で 8 ビット/バイトに詰めた** バイト列
              (ビット i → byte[i>>3] の bit (i&7))。C では mdl_infer() が
              「1 ビット 1 バイトの 0/1 配列」を要求するので、ヘッダの unpack_bits() で展開する。
    image_u8  バンドルに前処理記述子がある(画像系)。入力は生 u8 画像 [C][H][W](NCHW)。
              C では mdl_infer_image() にそのまま渡せる(前処理は整数で C 側が実行)。

データセット(`--dataset`、既定 auto):

    uci-har     UCI-HAR test から クラス均等 N 窓(既定 50)。バンドルの preproc(mean/std/閾値)
                で thermometer 符号化済みのビット列。ラベルは 0..5。
    mnist       MNIST test からクラス均等 N 枚(既定 50)。入力幅 784 は 28×28 二値化(閾値 0.5)、
                400 は 20×20(ボーダー除去)の二値ビット。前処理記述子を持つ画像バンドルなら
                生 u8 画像(H×W へ補間、チャネル複製)。MNIST が見つからなければ --mnist-root。
    synthetic   画像バンドル用の合成/公開可の入力。全 0・全 255・乱数・勾配・市松・PIL で描いた
                顔風イラスト。オプション --mnist-images K で MNIST の数字画像を K 枚混ぜる。
                顔検出(WIDER FACE)は実データが手元に無いので、ラベル無し(null)で
                「ホスト simulate の出力」を期待値にする(目的は整数一致とレイテンシ計測)。

期待出力(サンプルごと): ir/simulate の全出力スコア(int32)、そのハッシュ(FNV-1a 32bit を
int32 リトルエンディアンのバイト列に適用)、予測 index(分類=全体 argmax、検出=hm ヘッドの
argmax。同点は最小 index)。出力が大きい(顔検出 5,120 要素)ときヘッダには
ハッシュのみ載せる(JSON には全スコアを載せる)。

出力:

    <out-dir>/<model>.json        {format, model, input_kind, input_width, sample_bytes,
                                   n_out, labels, samples:[{index, name, label,
                                   bits|image(hex), expected_pred, expected_hash,
                                   expected_scores}]}
    <out-dir>/<model>_samples.h   C 埋め込み用(PREFIX_SAMPLES[N][BYTES] / PREFIX_LABELS /
                                  PREFIX_EXPECTED / PREFIX_EXPECTED_HASH /
                                  (小出力のみ)PREFIX_EXPECTED_SCORES と
                                  unpack_bits()・hash_scores() ヘルパ)

使い方:

    # 前段: uv run python scripts/fetch_demo_models.py all
    uv run python scripts/make_device_samples.py \\
        --bundle out_examples/demo_models/har_dlgn_k6000.bundle.json
    uv run python scripts/make_device_samples.py \\
        --bundle out_examples/demo_models/det_face_lut_k256.bundle.json --n 8
    # MNIST バンドル(examples/29_deploy_mnist.py の出力)
    uv run python scripts/make_device_samples.py --bundle <mnist>.bundle.json --dataset mnist

注意: 顔検出のように conv を含むバンドルの simulate は 1 回あたり数十秒・数 GB を使う
(展開後 5,500 万 LUT)。--sim-chunk(既定 4)でバッチ分割している。
"""

import argparse
import importlib.util
import json
import re
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from griffith_logic.deploy.bundle import load_bundle  # noqa: E402
from griffith_logic.deploy.preproc_desc import (  # noqa: E402
    build_preproc_descriptors,
    encode_bits,
)
from griffith_logic.export.lowering import lower_to_primitives  # noqa: E402
from griffith_logic.ir.schema import IRGraph  # noqa: E402
from griffith_logic.ir.serialize import ir_from_dict  # noqa: E402
from griffith_logic.ir.simulate import simulate  # noqa: E402

DEFAULT_OUT_DIR = REPO / 'out_examples' / 'demo_models' / 'samples'
FORMAT = 'mdl-device-samples/1'
# 出力がこの要素数以下ならヘッダに全スコアを載せる(N×n_out 要素の総数で判定)
HEADER_SCORES_MAX_ELEMS = 4096
_LOWERABLE = ('conv2d', 'or_pool2d', 'upsample2d', 'permute')


def _load_sibling(name: str):
    """scripts/<name>.py をファイルパスから import する(scripts はパッケージではない)。"""
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, REPO / 'scripts' / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# ビット詰め・ハッシュ・選択(純関数。tests/deploy/test_demo_models_tools.py が検証)
# ---------------------------------------------------------------------------
def pack_bits_lsb(bits: np.ndarray) -> np.ndarray:
    """{0,1} の [N,W] → LSB-first で詰めた uint8 [N, ceil(W/8)](余りビットは 0)。"""
    bits = np.asarray(bits, dtype=np.uint8)
    assert bits.ndim == 2
    return np.packbits(bits, axis=1, bitorder='little')


def unpack_bits_lsb(packed: np.ndarray, width: int) -> np.ndarray:
    """pack_bits_lsb の逆変換。[N, B] uint8 → {0,1} の [N, width]。"""
    packed = np.asarray(packed, dtype=np.uint8)
    return np.unpackbits(packed, axis=1, bitorder='little')[:, :width]


def fnv1a32_scores(scores) -> int:
    """int32 スコア列を **リトルエンディアンの 4 バイト列** と見て FNV-1a(32bit)を取る。

    C 側(ヘッダの hash_scores())と同じ式。バイト順に依存しないよう値からバイトを作る。
    """
    h = 0x811C9DC5
    for v in np.asarray(scores, dtype=np.int64).ravel():
        u = int(v) & 0xFFFFFFFF
        for shift in (0, 8, 16, 24):
            h ^= (u >> shift) & 0xFF
            h = (h * 0x01000193) & 0xFFFFFFFF
    return h


def select_class_balanced(y: np.ndarray, n: int, seed: int = 0) -> list[int]:
    """クラス均等に n 個の index を選ぶ(決定的)。

    各クラスから seed 固定の乱択で取り、**クラス巡回順**(0,1,..,K-1,0,1,..)に並べる
    ので、先頭から何件で切っても概ねクラス均等になる。クラスの在庫が尽きたら残りで続ける。
    """
    y = np.asarray(y)
    rng = np.random.default_rng(seed)
    classes = sorted(int(c) for c in np.unique(y))
    pools = {c: list(rng.permutation(np.nonzero(y == c)[0])) for c in classes}
    out: list[int] = []
    while len(out) < n and any(pools.values()):
        for c in classes:
            if pools[c] and len(out) < n:
                out.append(int(pools[c].pop(0)))
    return out


def expected_pred(scores_row: np.ndarray, readout: dict | None) -> int:
    """予測 index。検出は hm ヘッド(readout.heads.hm の offset..)の argmax、他は全体の argmax。"""
    scores_row = np.asarray(scores_row)
    if readout and readout.get('kind') == 'detection':
        hm = readout['heads']['hm']
        n = int(hm['n_out']) * int(hm['h']) * int(hm['w'])
        seg = scores_row[int(hm['offset']):int(hm['offset']) + n]
        return int(np.argmax(seg))
    return int(np.argmax(scores_row))


# ---------------------------------------------------------------------------
# データセットアダプタ: SampleSet(入力 + ラベル)を返す
# ---------------------------------------------------------------------------
@dataclass
class SampleSet:
    kind: str                       # 'bits' | 'image_u8'
    inputs: np.ndarray              # bits: [N,W] {0,1} / image_u8: [N,C,H,W] uint8
    labels: list                    # int | None
    names: list                     # str | None
    dataset: str


def har_samples(bundle: dict, n: int, seed: int, har_root: Path, *,
                download: bool = True) -> SampleSet:
    """UCI-HAR test からクラス均等 n 窓を、バンドルの preproc で thermometer 符号化する。"""
    fdm = _load_sibling('fetch_demo_models')
    p = bundle.get('preproc') or {}
    if p.get('kind') != 'thermometer' or 'thresholds' not in p:
        raise SystemExit('uci-har: バンドルの preproc が thermometer(mean/std/thresholds)でない')
    _, _, x_te, y_te = fdm.load_har_raw(Path(har_root), download=download)
    idx = select_class_balanced(y_te, n, seed)
    bits = fdm.encode_thermometer(x_te[idx], p['mean'], p['std'], p['thresholds'])
    labels = [int(y_te[i]) for i in idx]
    names = [f'test[{i}]' for i in idx]
    return SampleSet('bits', bits, labels, names, 'uci-har test(クラス均等)')


def _mnist_test(root: str | None, width: int | None, binarize: bool):
    """MNIST test セット。width=400 は 20×20、それ以外は 28×28。"""
    from griffith_logic.datasets.vision import load_mnist, load_mnist20x20

    thr = 0.5 if binarize else None
    if width == 400:
        _, test = load_mnist20x20(root, binarize_threshold=thr, download=False)
    else:
        _, test = load_mnist(root, binarize_threshold=thr, download=False)
    return test


def _resize_u8(img: np.ndarray, h: int, w: int) -> np.ndarray:
    """[H0,W0] float(0..1) → u8 [h,w](PIL バイリニア)。"""
    from PIL import Image

    u8 = np.clip(np.rint(img * 255.0), 0, 255).astype(np.uint8)
    if u8.shape != (h, w):
        u8 = np.asarray(Image.fromarray(u8).resize((w, h), Image.BILINEAR))
    return u8


def mnist_samples(bundle: dict, n: int, seed: int, root: str | None) -> SampleSet:
    """MNIST test からクラス均等 n 枚。dense なら二値ビット、画像バンドルなら生 u8 画像。"""
    graph = ir_from_dict(bundle['ir'])
    width = int(graph.inputs[0].width)
    image_mode = bundle.get('preproc_desc') is not None
    if image_mode:
        plan = build_preproc_descriptors(bundle['preproc'])
        test = _mnist_test(root, None, binarize=False)
    else:
        if width not in (784, 400):
            raise SystemExit(f'mnist: 入力幅 {width} は 784(28×28)/400(20×20)のどちらでもない')
        test = _mnist_test(root, width, binarize=True)
    ys = np.asarray(test.targets)
    idx = select_class_balanced(ys, n, seed)
    labels = [int(ys[i]) for i in idx]
    names = [f'mnist_test[{i}]' for i in idx]
    if image_mode:
        c, h, w = plan['image_channels'], plan['height'], plan['width']
        imgs = np.stack([
            np.repeat(_resize_u8(test[i][0].numpy()[0], h, w)[None], c, axis=0)
            for i in idx])
        return SampleSet('image_u8', imgs, labels, names, 'mnist test(クラス均等、生画像)')
    bits = np.stack([test[i][0].numpy().reshape(-1) for i in idx]).astype(np.uint8)
    assert bits.shape[1] == width
    return SampleSet('bits', bits, labels, names,
                     f'mnist test(クラス均等、{"20x20" if width == 400 else "28x28"}二値)')


def draw_synthetic_face(size: int, cx: float, cy: float, r: float, *,
                        tone=(224, 172, 140), bg=(96, 120, 150)) -> np.ndarray:
    """PIL で顔風イラスト(楕円の輪郭・目・口)を描く。u8 [3,size,size]。

    独自に描いた図形なので出典・ライセンス上の問題が無い。実顔ではないため検出される保証は
    無く、期待値は simulate の出力そのものを使う(整数一致の入力として使う)。
    """
    from PIL import Image, ImageDraw

    img = Image.new('RGB', (size, size), bg)
    d = ImageDraw.Draw(img)
    d.ellipse([cx - r * 0.8, cy - r, cx + r * 0.8, cy + r], fill=tone, outline=(80, 50, 40))
    ex, ey, er = r * 0.32, cy - r * 0.2, r * 0.11
    for sx in (-1, 1):
        d.ellipse([cx + sx * ex - er * 1.4, ey - er, cx + sx * ex + er * 1.4, ey + er],
                  fill=(250, 250, 250), outline=(40, 30, 30))
        d.ellipse([cx + sx * ex - er * 0.6, ey - er * 0.6, cx + sx * ex + er * 0.6,
                   ey + er * 0.6], fill=(30, 30, 40))
    d.line([cx, cy - r * 0.1, cx - r * 0.06, cy + r * 0.25], fill=(150, 100, 80), width=2)
    d.arc([cx - r * 0.38, cy + r * 0.15, cx + r * 0.38, cy + r * 0.6], 20, 160,
          fill=(150, 40, 50), width=max(2, int(r * 0.06)))
    return np.asarray(img, dtype=np.uint8).transpose(2, 0, 1).copy()


def synthetic_image_samples(plan: dict, n: int, seed: int, *, mnist_images: int = 0,
                            mnist_root: str | None = None) -> SampleSet:
    """画像バンドル向けの合成入力(全 0/全 255/乱数/勾配/市松/顔風イラスト)+ 任意で MNIST 数字。

    生成順は固定(先頭ほど基本ケース)なので、n を変えても先頭の内容は変わらない。
    """
    c, h, w = plan['image_channels'], plan['height'], plan['width']
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w]
    gens: list[tuple[str, np.ndarray]] = [
        ('zeros', np.zeros((c, h, w), np.uint8)),
        ('ones', np.full((c, h, w), 255, np.uint8)),
        ('noise0', rng.integers(0, 256, (c, h, w), dtype=np.uint8)),
        ('gradient_x', np.repeat((xx * 255 // max(w - 1, 1)).astype(np.uint8)[None], c, 0)),
        ('gradient_y', np.repeat((yy * 255 // max(h - 1, 1)).astype(np.uint8)[None], c, 0)),
        ('checker8', np.repeat((((yy // 8 + xx // 8) % 2) * 255).astype(np.uint8)[None], c, 0)),
        ('face_center', _fit_channels(draw_synthetic_face(h, w * 0.5, h * 0.5, h * 0.36), c)),
        ('face_small', _fit_channels(draw_synthetic_face(
            h, w * 0.3, h * 0.35, h * 0.16, tone=(200, 150, 120), bg=(70, 90, 70)), c)),
        ('noise1', rng.integers(0, 256, (c, h, w), dtype=np.uint8)),
        ('noise2', rng.integers(0, 256, (c, h, w), dtype=np.uint8)),
    ]
    imgs = [g for _, g in gens]
    names = [nm for nm, _ in gens]
    k = 3
    while len(imgs) < n - mnist_images:        # 足りなければ乱数画像を追加
        imgs.append(rng.integers(0, 256, (c, h, w), dtype=np.uint8))
        names.append(f'noise{k}')
        k += 1
    imgs, names = imgs[:max(n - mnist_images, 0)], names[:max(n - mnist_images, 0)]
    if mnist_images > 0:
        try:
            test = _mnist_test(mnist_root, None, binarize=False)
            ys = np.asarray(test.targets)
            for i in select_class_balanced(ys, mnist_images, seed):
                imgs.append(np.repeat(_resize_u8(test[i][0].numpy()[0], h, w)[None], c, 0))
                names.append(f'mnist_test[{i}]')
        except (RuntimeError, OSError, FileNotFoundError) as exc:
            print(f'  [注意] MNIST を混ぜられない(スキップ): {exc}')
    arr = np.stack(imgs).astype(np.uint8)
    return SampleSet('image_u8', arr, [None] * len(arr), names,
                     '合成画像(+MNIST数字): ラベル無し、期待値は simulate 出力')


def _fit_channels(chw: np.ndarray, c: int) -> np.ndarray:
    """RGB [3,H,W] を c チャネルへ(c=1 なら輝度、c>3 は先頭を複製)。"""
    if chw.shape[0] == c:
        return chw
    if c == 1:
        gray = (0.299 * chw[0] + 0.587 * chw[1] + 0.114 * chw[2]).round()
        return gray.astype(np.uint8)[None]
    return np.concatenate([chw] * ((c + 2) // 3), axis=0)[:c]


# ---------------------------------------------------------------------------
# 期待出力(ホスト ir/simulate)と C ランタイム照合
# ---------------------------------------------------------------------------
def to_bits(bundle: dict, ss: SampleSet) -> np.ndarray:
    """SampleSet → IR 入力ビット [N, in_width](画像は整数前処理記述子で符号化)。"""
    if ss.kind == 'bits':
        return ss.inputs
    plan = build_preproc_descriptors(bundle['preproc'])
    return encode_bits(plan, ss.inputs)


def compute_expected(graph: IRGraph, bits: np.ndarray, *, chunk: int = 4) -> np.ndarray:
    """ir/simulate の出力 [N, n_out](int64)。conv 系は展開を 1 回だけ行い chunk で分割実行する。"""
    g = lower_to_primitives(graph) if any(n.type in _LOWERABLE for n in graph.nodes) else graph
    out = [simulate(g, bits[i:i + chunk]) for i in range(0, len(bits), chunk)]
    return np.concatenate(out, axis=0).astype(np.int64)


def check_against_c(bundle: dict, ss: SampleSet, bits: np.ndarray,
                    expected: np.ndarray) -> int:
    """生成 C をビルドして全サンプルを実行し、simulate と整数厳密一致を確認する(一致件数を返す)。"""
    from griffith_logic.deploy.verify import build_c_model, run_c_model

    image_entry = ss.kind == 'image_u8'
    n_out = expected.shape[1]
    with tempfile.TemporaryDirectory(prefix='mdl_samples_') as tmp:
        exe = build_c_model(bundle, tmp, image_entry=image_entry)
        for i in range(len(expected)):
            payload = (ss.inputs[i].astype(np.uint8).tobytes() if image_entry
                       else bits[i].astype(np.uint8).tobytes())
            got = run_c_model(exe, payload, n_out)
            if not np.array_equal(got, expected[i]):
                bad = np.nonzero(got != expected[i])[0]
                raise SystemExit(f'C ⇔ simulate 不一致: サンプル{i} 要素{int(bad[0])} '
                                 f'C={int(got[bad[0]])} simulate={int(expected[i][bad[0]])}')
    return len(expected)


# ---------------------------------------------------------------------------
# JSON / C ヘッダ
# ---------------------------------------------------------------------------
def build_doc(name: str, bundle: dict, ss: SampleSet, bits: np.ndarray,
              expected: np.ndarray) -> dict:
    """サンプル集の JSON ドキュメント(仕様は docstring)を組み立てる。"""
    graph = ir_from_dict(bundle['ir'])
    readout = bundle.get('readout')
    if ss.kind == 'bits':
        packed = pack_bits_lsb(bits)
        payloads = [bytes(p).hex() for p in packed]
        sample_bytes = int(packed.shape[1])
        key = 'bits'
        shape = None
    else:
        flat = ss.inputs.reshape(len(ss.inputs), -1).astype(np.uint8)
        payloads = [bytes(p).hex() for p in flat]
        sample_bytes = int(flat.shape[1])
        key = 'image'
        shape = list(ss.inputs.shape[1:])
    samples = []
    for i in range(len(expected)):
        samples.append({
            'index': i, 'name': ss.names[i], 'label': ss.labels[i],
            key: payloads[i],
            'expected_pred': expected_pred(expected[i], readout),
            'expected_hash': fnv1a32_scores(expected[i]),
            'expected_scores': [int(v) for v in expected[i]],
        })
    return {
        'format': FORMAT, 'model': name, 'dataset': ss.dataset,
        'input_kind': ss.kind, 'input_width': int(graph.inputs[0].width),
        'input_shape': shape, 'sample_bytes': sample_bytes,
        'encoding': 'hex', 'bit_order': 'lsb-first' if ss.kind == 'bits' else None,
        'n_out': int(expected.shape[1]),
        'pred_rule': ('hm ヘッドの argmax' if readout and readout.get('kind') == 'detection'
                      else '全出力の argmax(同点は最小 index)'),
        'hash_rule': 'FNV-1a 32bit over int32 little-endian bytes of all scores',
        'labels': bundle.get('labels'),
        'n_samples': len(samples), 'samples': samples,
    }


def payload_matrix(doc: dict) -> np.ndarray:
    """JSON の各サンプルの入力(hex)を [N, sample_bytes] の uint8 行列へ。"""
    key = 'bits' if doc['input_kind'] == 'bits' else 'image'
    return np.array([list(bytes.fromhex(s[key])) for s in doc['samples']], dtype=np.uint8)


def _c_ident_prefix(name: str) -> str:
    return re.sub(r'[^0-9A-Za-z]+', '_', name).strip('_').upper() + '_'


def _fmt_hex_rows(data: bytes, per_line: int = 24) -> str:
    rows = []
    for off in range(0, len(data), per_line):
        rows.append('    ' + ', '.join(f'0x{b:02x}' for b in data[off:off + per_line]) + ',')
    return '\n'.join(rows)


def render_header(doc: dict, prefix: str | None = None) -> str:
    """サンプル集 JSON → C ヘッダ(static const 配列 + ヘルパ)。

    prefix は識別子の接頭辞(既定: モデル名の大文字 + '_')。複数モデルのヘッダを
    同じ翻訳単位に include できるように全識別子に付ける('' で接頭辞なし = SAMPLES/EXPECTED)。
    """
    pf = _c_ident_prefix(doc['model']) if prefix is None else prefix
    lo = pf.lower()
    n, sb, n_out = doc['n_samples'], doc['sample_bytes'], doc['n_out']
    key = 'bits' if doc['input_kind'] == 'bits' else 'image'
    has_scores = n * n_out <= HEADER_SCORES_MAX_ELEMS
    parts = [
        f'/* {doc["model"]} デバイス用サンプル集(make_device_samples.py が生成。手編集禁止)\n'
        f' * dataset: {doc["dataset"]}\n'
        f' * 入力: {doc["input_kind"]}'
        + (' (LSB-first パック。unpack_bits() で 1ビット1バイトへ展開して mdl_infer へ)'
           if key == 'bits' else ' (u8 NCHW。mdl_infer_image へそのまま)') + '\n'
        f' * 期待値は ir/simulate の整数出力。{doc["pred_rule"]} / ハッシュ={doc["hash_rule"]} */',
        f'#ifndef {pf}SAMPLES_H\n#define {pf}SAMPLES_H\n\n#include <stdint.h>\n',
        f'#define {pf}N_SAMPLES {n}',
        f'#define {pf}SAMPLE_BYTES {sb}',
        f'#define {pf}INPUT_WIDTH {doc["input_width"]}  /* IR 入力ビット数 = MDL_INPUT_WIDTH */',
        f'#define {pf}N_OUT {n_out}',
        f'#define {pf}HAS_EXPECTED_SCORES {1 if has_scores else 0}',
        f'#define {pf}INPUT_IS_IMAGE {0 if key == "bits" else 1}\n',
        f'static const uint8_t {pf}SAMPLES[{n}][{sb}] = {{',
    ]
    for s in doc['samples']:
        parts.append('  {\n' + _fmt_hex_rows(bytes.fromhex(s[key])) + '\n  },')
    parts.append('};\n')
    labels = ', '.join(str(-1 if s['label'] is None else s['label']) for s in doc['samples'])
    preds = ', '.join(str(s['expected_pred']) for s in doc['samples'])
    hashes = ', '.join(f'0x{s["expected_hash"]:08x}u' for s in doc['samples'])
    parts += [
        f'/* 正解ラベル(無いサンプルは -1) */\n'
        f'static const int16_t {pf}LABELS[{n}] = {{ {labels} }};',
        f'/* ホスト simulate の予測 index */\n'
        f'static const int16_t {pf}EXPECTED[{n}] = {{ {preds} }};',
        f'/* 全出力スコアの FNV-1a(int32 LE)。{lo}hash_scores() と同じ式 */\n'
        f'static const uint32_t {pf}EXPECTED_HASH[{n}] = {{ {hashes} }};',
    ]
    if has_scores:
        rows = ',\n'.join(
            '  {' + ', '.join(str(v) for v in s['expected_scores']) + '}'
            for s in doc['samples'])
        parts.append(f'static const int32_t {pf}EXPECTED_SCORES[{n}][{n_out}] = {{\n{rows}\n}};')
    parts.append(
        f'\n/* LSB-first パックのビット列 → 1ビット1バイトの 0/1 配列(mdl_infer 用) */\n'
        f'static inline void {lo}unpack_bits(const uint8_t *packed, uint8_t *bits, int width) {{\n'
        f'    int i;\n'
        f'    for (i = 0; i < width; i++) bits[i] = (uint8_t)((packed[i >> 3] >> (i & 7)) & 1u);\n'
        f'}}\n\n'
        f'/* 出力スコア(int32)を FNV-1a(リトルエンディアン 4 バイト)でハッシュする */\n'
        f'static inline uint32_t {lo}hash_scores(const int32_t *scores, int n) {{\n'
        f'    uint32_t h = 0x811c9dc5u;\n'
        f'    int i, b;\n'
        f'    for (i = 0; i < n; i++) {{\n'
        f'        uint32_t u = (uint32_t)scores[i];\n'
        f'        for (b = 0; b < 4; b++) {{\n'
        f'            h ^= (u >> (8 * b)) & 0xffu;\n'
        f'            h *= 0x01000193u;\n'
        f'        }}\n'
        f'    }}\n'
        f'    return h;\n'
        f'}}\n\n#endif /* {pf}SAMPLES_H */\n')
    return '\n'.join(parts)


def parse_header(text: str, prefix: str) -> dict:
    """render_header の出力から配列を読み戻す(往復テスト・書き出し後の自己検証用)。"""
    def block(name: str) -> str:
        m = re.search(rf'{re.escape(prefix + name)}\[[^=]*=\s*\{{(.*?)\n?\}};', text, re.S)
        if not m:
            raise ValueError(f'{prefix + name} がヘッダに無い')
        return m.group(1)

    def macro(name: str) -> int:
        m = re.search(rf'#define\s+{re.escape(prefix + name)}\s+(-?\d+)', text)
        if not m:
            raise ValueError(f'#define {prefix + name} が無い')
        return int(m.group(1))

    n, sb, n_out = macro('N_SAMPLES'), macro('SAMPLE_BYTES'), macro('N_OUT')
    raw = [int(v, 16) for v in re.findall(r'0x([0-9a-fA-F]{2})(?![0-9a-fA-F])', block('SAMPLES'))]
    samples = np.array(raw, dtype=np.uint8).reshape(n, sb)
    labels = [int(v) for v in block('LABELS').replace('{', ' ').replace('}', ' ').split(',')]
    preds = [int(v) for v in block('EXPECTED').split(',')]
    hashes = [int(v.strip().rstrip('uU'), 16) for v in block('EXPECTED_HASH').split(',')]
    out = {'n': n, 'sample_bytes': sb, 'n_out': n_out, 'samples': samples,
           'labels': labels, 'expected': preds, 'hashes': hashes, 'scores': None}
    if macro('HAS_EXPECTED_SCORES'):
        nums = [int(v) for v in re.findall(r'-?\d+', block('EXPECTED_SCORES'))]
        out['scores'] = np.array(nums, dtype=np.int64).reshape(n, n_out)
    return out


# ---------------------------------------------------------------------------
# 駆動
# ---------------------------------------------------------------------------
def infer_name(bundle_path: Path) -> str:
    stem = bundle_path.name
    for suf in ('.bundle.json', '.json'):
        if stem.endswith(suf):
            return stem[:-len(suf)]
    return bundle_path.stem


def resolve_dataset(choice: str, bundle: dict) -> str:
    """--dataset auto をバンドルの中身から決める。"""
    if choice != 'auto':
        return choice
    p = bundle.get('preproc') or {}
    if p.get('kind') == 'thermometer' and p.get('bit_order') == 'ch,t,thr':
        return 'uci-har'
    graph = ir_from_dict(bundle['ir'])
    width = int(graph.inputs[0].width)
    if bundle.get('preproc_desc') is not None and (bundle.get('readout') or {}).get(
            'kind') == 'detection':
        return 'synthetic'
    if bundle.get('preproc_desc') is None and width in (784, 400):
        return 'mnist'
    raise SystemExit('--dataset auto で決められない。--dataset uci-har|mnist|synthetic を指定して')


def make_samples(bundle_path: Path, out_dir: Path, *, dataset: str = 'auto',
                 n: int | None = None, seed: int = 0, har_root: Path | None = None,
                 mnist_root: str | None = None, mnist_images: int = 0, chunk: int = 4,
                 check_c: bool = True, download: bool = True) -> dict:
    """バンドル 1 つからサンプル集(JSON + ヘッダ)を書き出す。サマリ dict を返す。"""
    bundle = load_bundle(str(bundle_path), validate=False)
    name = infer_name(bundle_path)
    ds = resolve_dataset(dataset, bundle)
    if ds == 'uci-har':
        ss = har_samples(bundle, n or 50, seed, har_root or REPO / 'data-uci-har',
                         download=download)
    elif ds == 'mnist':
        ss = mnist_samples(bundle, n or 50, seed, mnist_root)
    elif ds == 'synthetic':
        if bundle.get('preproc_desc') is None:
            raise SystemExit('synthetic は画像バンドル(前処理記述子あり)専用')
        plan = build_preproc_descriptors(bundle['preproc'])
        ss = synthetic_image_samples(plan, n or 10, seed, mnist_images=mnist_images,
                                     mnist_root=mnist_root)
    else:
        raise SystemExit(f'未対応の --dataset: {ds}')
    print(f'  {name}: dataset={ss.dataset} 入力={ss.kind} N={len(ss.inputs)}', flush=True)

    graph = ir_from_dict(bundle['ir'])
    bits = to_bits(bundle, ss)
    assert bits.shape[1] == int(graph.inputs[0].width), 'エンコード後の幅がIR入力幅と違う'
    expected = compute_expected(graph, bits, chunk=chunk)
    n_checked = check_against_c(bundle, ss, bits, expected) if check_c else 0

    doc = build_doc(name, bundle, ss, bits, expected)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f'{name}.json'
    json_path.write_text(json.dumps(doc, separators=(',', ':')), encoding='utf-8')
    header_path = out_dir / f'{name}_samples.h'
    header = render_header(doc)
    header_path.write_text(header, encoding='utf-8')
    # 書き出したヘッダを読み戻して JSON と突き合わせる(往復の自己検証)
    back = parse_header(header, _c_ident_prefix(name))
    want = payload_matrix(doc)
    assert np.array_equal(back['samples'], want), 'ヘッダの入力が JSON と一致しない'
    assert back['hashes'] == [s['expected_hash'] for s in doc['samples']]
    print(f'  wrote {json_path} ({json_path.stat().st_size:,}B)')
    print(f'  wrote {header_path} ({header_path.stat().st_size:,}B)'
          + (f'  C ⇔ simulate 整数一致 {n_checked}/{len(expected)}' if check_c else ''))
    return {'model': name, 'json': str(json_path), 'header': str(header_path),
            'n_samples': len(expected), 'input_kind': ss.kind,
            'sample_bytes': doc['sample_bytes'], 'checked_c': n_checked}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description='配布バンドルからデバイス用サンプル集(入力+ホスト simulate の期待出力)を作る',
        epilog='詳細はこのファイルの docstring を参照。')
    ap.add_argument('--bundle', required=True, action='append',
                    help='配布バンドル(*.bundle.json)。複数回指定可')
    ap.add_argument('--out-dir', default=str(DEFAULT_OUT_DIR),
                    help='出力先(既定: out_examples/demo_models/samples)')
    ap.add_argument('--dataset', default='auto',
                    choices=['auto', 'uci-har', 'mnist', 'synthetic'],
                    help='入力データ(既定 auto: バンドルから判定)')
    ap.add_argument('-n', '--n', type=int, default=None,
                    help='サンプル数(既定: uci-har/mnist 50、synthetic 10)')
    ap.add_argument('--seed', type=int, default=0, help='サンプル選択・乱数の seed')
    ap.add_argument('--har-root', default=str(REPO / 'data-uci-har'),
                    help='UCI-HAR のディレクトリ(既定: data-uci-har)')
    ap.add_argument('--no-download', action='store_true', help='UCI-HAR を自動取得しない')
    ap.add_argument('--mnist-root', default=None,
                    help='MNIST のディレクトリ(既定: repo 直下の data-mnist)')
    ap.add_argument('--mnist-images', type=int, default=0,
                    help='synthetic に混ぜる MNIST 数字画像の枚数(既定 0)')
    ap.add_argument('--sim-chunk', type=int, default=4,
                    help='simulate のバッチ分割サイズ(conv 系は 1 枚あたり数 GB 使う。既定 4)')
    ap.add_argument('--no-check-c', action='store_true',
                    help='生成 C ランタイムとの整数一致確認を省略する')
    args = ap.parse_args(argv)
    for b in args.bundle:
        print(f'=== {b} ===', flush=True)
        make_samples(Path(b), Path(args.out_dir), dataset=args.dataset, n=args.n,
                     seed=args.seed, har_root=Path(args.har_root), mnist_root=args.mnist_root,
                     mnist_images=args.mnist_images, chunk=args.sim_chunk,
                     check_c=not args.no_check_c, download=not args.no_download)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
