"""前処理記述子のPythonミラー。

TS側の真実源:
    tools/webbench/src/engines/gpuPreproc.ts::buildPreprocDescriptors
    tools/webbench/src/engines/cpuPreproc.ts::buildCpuPreprocPlan(THERMO_STEP畳み込み)

「画像(u8)→ 二値チャネル」の符号化を、**浮動小数点演算を含まないデータ**
(記述子表+256要素の量子化表+256ビットLUT)へコンパイルする。ターゲット側
ランタイムは記述子を舐めるだけでよく、mean/std/閾値の実数演算を持たない。

なぜ整数化できるか(議論をそのまま踏襲):
    入力が u8 なので画素値は f32(k/255) の256通りしかない。
    I(k) := f32(k/255) * 2^31 は厳密な整数(0 <= I(k) <= 2^31)であり、
    edge/curv/chroma/彩度が計算する量はすべて I の整数結合になる:

        edge   : D = I(a) - I(b)
        curv   : D = I(a) - 2*I(m) + I(c)      (|D| <= 2^32、符号付き34bit)
        chroma : D = I(a) - I(b)
        彩度   : D = I(kmax) - I(kmin)

    比較も t*2^31 が2冪倍で厳密なので整数化できる:

        d >  t  ⟺  D >= floor(t*2^31) + 1
        d < -t  ⟺  D <= ceil(-t*2^31) - 1

    thermometer だけは (x-mean)*(1/std) > th に任意の実係数が入るので整数化
    できないが、入力が256通りしかないので **256ビットLUT** にすれば定義上
    ビット一致する。さらに std>0 なら k について単調なので、ほぼ必ず
    「k >= k0」の1閾値へ畳める(THERMO_STEP)。畳めたかどうかはLUTを総なめして
    実際に階段関数か検査してから決める。

境界処理は replicate(端の値を複製)。`encoding/image_features.py::_shift` と
同じ規約。
"""

import math

import numpy as np

# 固定小数のスケール(2^31)。
SCALE = 2 ** 31

# 記述子1チャネルぶんのワード数(w0..w7=演算指定、w8..w15=thermometer LUT)。
PREPROC_DESC_STRIDE = 16

PREPROC_OP_THERMO = 0
PREPROC_OP_EDGE = 1
PREPROC_OP_CURV = 2
PREPROC_OP_CHROMA = 3
PREPROC_OP_SAT = 4
# CPU/C専用: 単調なthermometerを「k >= k0」の1閾値へ畳んだもの(k0はw2)。
PREPROC_OP_THERMO_STEP = 5

#: 入力符号化の op コード。**ここが単一情報源**で、C(`MDL_PREPROC_OP_*`)と
#: GPU(`MDL_G_OP_*`)の定義がこれと一致することをテストで固定している
#: (以前は3系統に手書きされていて、符号化を1つ足すのに3箇所直す必要があった)。
PREPROC_OPS = {
    'THERMO': PREPROC_OP_THERMO,
    'EDGE': PREPROC_OP_EDGE,
    'CURV': PREPROC_OP_CURV,
    'CHROMA': PREPROC_OP_CHROMA,
    'SAT': PREPROC_OP_SAT,
    'THERMO_STEP': PREPROC_OP_THERMO_STEP,
}

# 比較の向き。
DIR_GE = 0  # D >= cut(d > t 相当)
DIR_LE = 1  # D <= cut(d < -t 相当)

SUPPORTED_KINDS = ('image_thermometer', 'image_lowlevel')


def quant_table() -> list[int]:
    """I(k) = f32(k/255) * 2^31 の256要素表。整数でなければ前提が崩れている。"""
    out = []
    for k in range(256):
        s = float(np.float32(k / 255)) * SCALE
        if s != int(s) or s < 0 or s > 0xffffffff:
            raise ValueError(f'I({k})={s} がu32の整数でない(固定小数の前提崩れ)')
        out.append(int(s))
    return out


def cut_greater(t: float) -> int:
    """d > t ⟺ D >= cut。"""
    s = float(t) * SCALE
    if not math.isfinite(s) or abs(s) > 2 ** 52:
        raise ValueError(f'閾値 {t} が固定小数の表現範囲外')
    return math.floor(s) + 1


def cut_less(t: float) -> int:
    """d < -t ⟺ D <= cut。"""
    s = -float(t) * SCALE
    if not math.isfinite(s) or abs(s) > 2 ** 52:
        raise ValueError(f'閾値 {t} が固定小数の表現範囲外')
    return math.ceil(s) - 1


def _i32(value: int) -> int:
    """u32のビット列をi32として再解釈する(cpuPreproc.tsの `| 0` と同じ)。"""
    v = int(value) & 0xffffffff
    return v - 0x100000000 if v >= 0x80000000 else v


def _cut_words(cut: int) -> tuple[int, int]:
    """安全整数 → 64bit二の補数の (lo, hi)。どちらもi32として格納する。"""
    if abs(cut) > 2 ** 53:
        raise ValueError(f'cut={cut} が安全整数の範囲外')
    lo = cut % (2 ** 32)
    hi = cut // (2 ** 32)
    return _i32(lo), _i32(hi % (2 ** 32))


def cut_of(desc: list[int], base: int) -> int:
    """記述子のcut2語を安全整数へ戻す(cpuPreproc.ts::cutOf と同じ式)。"""
    lo = desc[base + 5] & 0xffffffff
    hi = desc[base + 6]
    return hi * 4294967296 + lo


class _DescWriter:
    def __init__(self):
        self.rows: list[list[int]] = []

    def push_thermo(self, c: int, mean: float, std: float, th: float) -> None:
        """256ビットLUTを、CPU側とまったく同じ式で埋める。"""
        inv_std = 1.0 / float(std)
        w = [0] * PREPROC_DESC_STRIDE
        w[0] = PREPROC_OP_THERMO
        w[1] = int(c)
        for k in range(256):
            z = (float(np.float32(k / 255)) - float(mean)) * inv_std
            if z > float(th):
                w[8 + (k >> 5)] |= 1 << (k & 31)
        self.rows.append([_i32(v) for v in w])

    def push_cmp(self, op: int, a: int, b: int, dy: int, dx: int,
                 cut: int, direction: int) -> None:
        lo, hi = _cut_words(cut)
        w = [0] * PREPROC_DESC_STRIDE
        w[0] = op
        w[1] = int(a)
        w[2] = int(b)
        w[3] = int(dy)
        w[4] = int(dx)
        w[5] = lo
        w[6] = hi
        w[7] = int(direction)
        self.rows.append(w)


def _thermo_step_index(row: list[int]) -> int | None:
    """256ビットLUTが「k >= k0」の階段関数なら k0(全0なら256)、違えばNone。"""
    def bit_at(k: int) -> int:
        return (row[8 + (k >> 5)] >> (k & 31)) & 1

    first = 256
    for k in range(256):
        if bit_at(k) == 1:
            first = k
            break
    for k in range(256):
        if bit_at(k) != (1 if k >= first else 0):
            return None
    return first


def build_preproc_descriptors(p: dict, *, fold_thermo_step: bool = True) -> dict:
    """preproc宣言 → 記述子表(整数のみ)。

    チャネル順は tools/webbench/src/convpreproc.ts::encodeImageAuto と厳密に
    同じでなければならない(image_thermometer は rgb-major、image_lowlevel は
    [thermo(thr-major)][edge][curv][chroma pair][saturation])。

    Args:
        p: バンドルの `preproc` 節
        fold_thermo_step: 単調なthermometerを THERMO_STEP(u8比較1個)へ畳む。
            Falseなら常に256ビットLUT参照のまま(比較用)

    Returns:
        {'desc', 'qval', 'channels', 'image_channels', 'height', 'width',
         'stride', 'counts'} のdict。descは長さ channels*16 のintリスト
    """
    kind = p.get('kind')
    if kind not in SUPPORTED_KINDS:
        raise ValueError(f'未対応の preproc.kind={kind!r}(対応: {SUPPORTED_KINDS})')

    mean = [float(v) for v in p['mean']]
    std = [float(v) for v in p['std']]
    c_img = len(mean)
    if c_img < 1 or len(std) != c_img:
        raise ValueError(f'mean/stdの長さ不一致(mean={c_img}, std={len(std)})')
    if c_img > 4:
        raise ValueError(f'画像チャネル数 {c_img} は未対応(RGBAの4まで)')
    for s in std:
        if not s > 0:
            raise ValueError(f'std={s} が正でない(単調性の前提崩れ)')

    thresholds = [[float(t) for t in row] for row in p['thresholds']]
    n_thr = len(thresholds[0]) if thresholds else 0
    for c in range(c_img):
        if len(thresholds[c]) != n_thr:
            raise ValueError(f'thresholds[{c}] の長さが不揃い')

    w = _DescWriter()
    if kind == 'image_thermometer':
        # rgb-major: ch = c*T + k
        for c in range(c_img):
            for k in range(n_thr):
                w.push_thermo(c, mean[c], std[c], thresholds[c][k])
    else:
        edge_thr = [float(t) for t in p.get('edge_thresholds') or ()]
        curv_thr = [float(t) for t in p.get('curv_thresholds') or ()]
        chroma_thr = [float(t) for t in p.get('chroma_thresholds') or ()]
        edge_off = p.get('edge_offsets') or ()
        curv_off = p.get('curv_offsets') or ()
        chroma_pairs = p.get('chroma_pairs') or ()
        # thermo は thr-major(kが外、cが内)
        for k in range(n_thr):
            for c in range(c_img):
                w.push_thermo(c, mean[c], std[c], thresholds[c][k])
        for dy, dx in edge_off:
            for t in edge_thr:
                for c in range(c_img):
                    w.push_cmp(PREPROC_OP_EDGE, c, 0, dy, dx, cut_greater(t), DIR_GE)
                for c in range(c_img):
                    w.push_cmp(PREPROC_OP_EDGE, c, 0, dy, dx, cut_less(t), DIR_LE)
        for dy, dx in curv_off:
            for t in curv_thr:
                for c in range(c_img):
                    w.push_cmp(PREPROC_OP_CURV, c, 0, dy, dx, cut_greater(t), DIR_GE)
                for c in range(c_img):
                    w.push_cmp(PREPROC_OP_CURV, c, 0, dy, dx, cut_less(t), DIR_LE)
        for ci, cj in chroma_pairs:
            for t in chroma_thr:
                w.push_cmp(PREPROC_OP_CHROMA, ci, cj, 0, 0, cut_greater(t), DIR_GE)
                w.push_cmp(PREPROC_OP_CHROMA, ci, cj, 0, 0, cut_less(t), DIR_LE)
        # 彩度(max-min)は全chromaペアの後に閾値順でまとめて置く
        for t in chroma_thr:
            w.push_cmp(PREPROC_OP_SAT, 0, 0, 0, 0, cut_greater(t), DIR_GE)

    counts = {'thermo_step': 0, 'thermo_lut': 0, 'cmp': 0}
    for row in w.rows:
        if row[0] != PREPROC_OP_THERMO:
            counts['cmp'] += 1
            continue
        k0 = _thermo_step_index(row) if fold_thermo_step else None
        if k0 is None:
            counts['thermo_lut'] += 1
        else:
            row[0] = PREPROC_OP_THERMO_STEP
            row[2] = k0
            counts['thermo_step'] += 1

    desc: list[int] = []
    for row in w.rows:
        desc.extend(row)
    return {
        'desc': desc,
        'qval': quant_table(),
        'channels': len(w.rows),
        'image_channels': c_img,
        'height': int(p['height']),
        'width': int(p['width']),
        'stride': PREPROC_DESC_STRIDE,
        'counts': counts,
    }


def _shift_replicate(a: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """out[y,x] = a[y+dy, x+dx](端は複製)。image_features._shift と同規約。"""
    h, w = a.shape[-2], a.shape[-1]
    ys = np.clip(np.arange(h) + dy, 0, h - 1)
    xs = np.clip(np.arange(w) + dx, 0, w - 1)
    return a[..., ys[:, None], xs[None, :]]


def run_preproc_descriptors(plan: dict, images_u8: np.ndarray) -> np.ndarray:
    """記述子を実行して二値チャネルを作る(整数のみ、浮動小数点演算なし)。

    Args:
        plan: build_preproc_descriptors の戻り値
        images_u8: [N, C_img, H, W] の uint8 画像

    Returns:
        [N, channels, H, W] の uint8(0/1)
    """
    if images_u8.dtype != np.uint8:
        raise ValueError(f'images_u8 は uint8 であること(実際は {images_u8.dtype})')
    if images_u8.ndim != 4:
        raise ValueError(f'images_u8 は [N,C,H,W](実際は {images_u8.shape})')
    n, c_img, h, w = images_u8.shape
    if (c_img, h, w) != (plan['image_channels'], plan['height'], plan['width']):
        raise ValueError(
            f'画像形状 {(c_img, h, w)} が記述子 '
            f"{(plan['image_channels'], plan['height'], plan['width'])} と違う"
        )

    desc = plan['desc']
    qval = np.asarray(plan['qval'], dtype=np.int64)
    q = qval[images_u8.astype(np.int64)]        # [N,C,H,W] の I(k)
    out = np.zeros((n, plan['channels'], h, w), dtype=np.uint8)

    for ch in range(plan['channels']):
        base = ch * PREPROC_DESC_STRIDE
        op = desc[base]
        ca, cb = desc[base + 1], desc[base + 2]
        dy, dx = desc[base + 3], desc[base + 4]

        if op == PREPROC_OP_THERMO_STEP:
            out[:, ch] = (images_u8[:, ca] >= cb).astype(np.uint8)
            continue
        if op == PREPROC_OP_THERMO:
            lut = np.zeros(256, dtype=np.uint8)
            for k in range(256):
                lut[k] = (desc[base + 8 + (k >> 5)] >> (k & 31)) & 1
            out[:, ch] = lut[images_u8[:, ca]]
            continue

        qc = q[:, ca]
        if op == PREPROC_OP_EDGE:
            d = _shift_replicate(qc, dy, dx) - qc
        elif op == PREPROC_OP_CURV:
            d = (_shift_replicate(qc, -dy, -dx) - 2 * qc
                 + _shift_replicate(qc, dy, dx))
        elif op == PREPROC_OP_CHROMA:
            d = qc - q[:, cb]
        elif op == PREPROC_OP_SAT:
            d = (qval[images_u8.max(axis=1).astype(np.int64)]
                 - qval[images_u8.min(axis=1).astype(np.int64)])
        else:
            raise ValueError(f'未知の前処理op={op}(チャネル{ch})')

        cut = cut_of(desc, base)
        out[:, ch] = (d <= cut if desc[base + 7] == DIR_LE else d >= cut).astype(np.uint8)

    return out


def encode_bits(plan: dict, images_u8: np.ndarray) -> np.ndarray:
    """記述子実行の結果を [N, channels*H*W] のNCHW flattenビットへ。

    IRの入力幅の並び(bit index = ((ch*H)+y)*W+x)に合わせる。
    """
    bits = run_preproc_descriptors(plan, images_u8)
    return bits.reshape(bits.shape[0], -1)


def build_pair_threshold(cut: int, dir_le: int, qval: list[int] | None = None) -> list[int]:
    """差分比較を「u8のしきい値比較」へ還元した256要素の表。

    I(k) は k について厳密に単調増加するので、

        I(a) - I(b) >= cut   ⟺   a >= thr[b]
        I(a) - I(b) <= cut   ⟺   a <  thr[b]

    と b ごとの1つのしきい値へ**厳密に**落ちる。thr は b について単調なので
    2ポインタで O(512) で作れる。Cランタイム(mdl_kernels.c::build_pair_threshold)
    と同一の表を返す。GPUバックエンドではこれを生成時に焼き込んで、
    デバイス側から64bit演算を消すのに使う。

    Returns:
        長さ256のintリスト。値域は [0, 256]
    """
    q = quant_table() if qval is None else list(qval)
    out: list[int] = []
    a = 0
    if not dir_le:
        for b in range(256):
            need = cut + q[b]
            while a < 256 and q[a] < need:
                a += 1
            out.append(a)
    else:
        for b in range(256):
            limit = cut + q[b]
            while a < 256 and q[a] <= limit:
                a += 1
            out.append(a)
    return out


def threshold_tables(plan: dict) -> list[int]:
    """全チャネルぶんの256要素しきい値表を1本に連結する(値域 [0,256])。

    GPUバックエンド(CUDA/Metal)が生成時に焼き込むための表。CPU版は
    `mdl_kernels.c::build_pair_threshold` が実行時に同じものを作る。
    THERMO 系と CURV はこの表を使わないので0で埋める(添字を揃えるため)。
    """
    out: list[int] = []
    desc = plan['desc']
    for ch in range(plan['channels']):
        base = ch * PREPROC_DESC_STRIDE
        op = desc[base]
        if op in (PREPROC_OP_THERMO, PREPROC_OP_THERMO_STEP, PREPROC_OP_CURV):
            out.extend([0] * 256)
        else:
            out.extend(build_pair_threshold(cut_of(desc, base), desc[base + 7],
                                            plan['qval']))
    return out
