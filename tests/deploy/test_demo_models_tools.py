"""デモ用モデル取得・デバイス用サンプル生成スクリプトの軽量テスト(ネットワーク不要)。

対象: scripts/fetch_demo_models.py と scripts/make_device_samples.py の純関数部分
(thermometer 符号化・ビット詰め・FNV ハッシュ・クラス均等選択・C ヘッダの往復・
summary 整形・model.h マクロ解析・ELF サイズ分類)。wandb・実データ・長時間処理は触らない。
"""

import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]


def _load_script(name: str):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, REPO / 'scripts' / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope='module')
def fdm():
    return _load_script('fetch_demo_models')


@pytest.fixture(scope='module')
def mds():
    return _load_script('make_device_samples')


# ---------------------------------------------------------------------------
# thermometer 符号化(HAR 契約: bit = ((ch*L)+t)*T+thr、z > thr)
# ---------------------------------------------------------------------------
def _toy_windows(seed=0, n=40, c=9, l=16):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, c, l)).astype(np.float32) * 3 + 1


def test_thermometer_params_shape_and_sorted(fdm):
    x = _toy_windows()
    p = fdm.har_thermometer_params(x, 4)
    assert p['mean'].shape == (9,) and p['std'].shape == (9,)
    assert p['thresholds'].shape == (9, 4)
    assert (np.diff(p['thresholds'], axis=1) >= 0).all()          # 閾値は昇順


def test_thermometer_monotone_and_bit_order(fdm):
    x = _toy_windows(l=8)
    p = fdm.har_thermometer_params(x, 4)
    bits = fdm.encode_thermometer(x, p['mean'], p['std'], p['thresholds'])
    assert bits.shape == (len(x), 9 * 8 * 4)
    assert set(np.unique(bits)) <= {0, 1}
    cube = bits.reshape(len(x), 9, 8, 4)
    # thermometer: 閾値が上がるほどビットは立ちにくい(thr 軸で非増加)
    assert (np.diff(cube.astype(np.int8), axis=3) <= 0).all()
    # ビット順の契約: index = ((ch*L)+t)*T + thr を素朴に再計算して一致
    z = (x.astype(np.float64) - p['mean'][None, :, None]) / p['std'][None, :, None]
    for ch, t, thr in [(0, 0, 0), (3, 5, 2), (8, 7, 3)]:
        want = (z[:, ch, t] > p['thresholds'][ch, thr]).astype(np.uint8)
        got = bits[:, (ch * 8 + t) * 4 + thr]
        assert np.array_equal(want, got)


def test_thermometer_quantile_levels(fdm):
    """等分位: 閾値 i の上側に約 (T-i)/(T+1) の train 値がある。"""
    x = _toy_windows(n=400, l=32)
    p = fdm.har_thermometer_params(x, 4)
    bits = fdm.encode_thermometer(x, p['mean'], p['std'], p['thresholds'])
    frac = bits.reshape(len(x), 9, 32, 4).mean(axis=(0, 1, 2))
    np.testing.assert_allclose(frac, [4 / 5, 3 / 5, 2 / 5, 1 / 5], atol=0.02)


# ---------------------------------------------------------------------------
# ビット詰め・ハッシュ・選択
# ---------------------------------------------------------------------------
@pytest.mark.parametrize('width', [1, 7, 8, 9, 31, 33, 4608])
def test_pack_unpack_roundtrip(mds, width):
    rng = np.random.default_rng(width)
    bits = rng.integers(0, 2, size=(5, width), dtype=np.uint8)
    packed = mds.pack_bits_lsb(bits)
    assert packed.shape == (5, (width + 7) // 8)
    assert np.array_equal(mds.unpack_bits_lsb(packed, width), bits)


def test_pack_is_lsb_first(mds):
    bits = np.zeros((1, 16), np.uint8)
    bits[0, 0] = 1        # byte0 の bit0
    bits[0, 9] = 1        # byte1 の bit1
    assert mds.pack_bits_lsb(bits).tolist() == [[0x01, 0x02]]


def test_fnv1a32_known_vectors(mds):
    # 空列 = オフセット基底値
    assert mds.fnv1a32_scores([]) == 0x811C9DC5
    # int32 LE のバイト列に対する FNV-1a を素朴実装と突き合わせる(負数・大きい値を含む)
    vals = [0, 1, -1, 123456, -2147483648, 2147483647]
    h = 0x811C9DC5
    for v in vals:
        for byte in (v & 0xFFFFFFFF).to_bytes(4, 'little'):
            h = ((h ^ byte) * 0x01000193) & 0xFFFFFFFF
    assert mds.fnv1a32_scores(np.array(vals, np.int32)) == h


def test_select_class_balanced(mds):
    y = np.repeat(np.arange(6), [50, 40, 30, 60, 45, 70])
    idx = mds.select_class_balanced(y, 50, seed=0)
    assert len(idx) == 50 and len(set(idx)) == 50
    counts = np.bincount(y[idx], minlength=6)
    assert counts.max() - counts.min() <= 1                     # 均等(余りは 1 件差)
    assert idx == mds.select_class_balanced(y, 50, seed=0)      # 決定的
    # 先頭 12 件はクラス巡回順で各クラス 2 件ずつ
    assert np.bincount(y[idx[:12]], minlength=6).tolist() == [2] * 6


def test_select_class_balanced_exhausted_class(mds):
    y = np.array([0, 0, 0, 0, 1])
    idx = mds.select_class_balanced(y, 5, seed=1)
    assert sorted(y[idx].tolist()) == [0, 0, 0, 0, 1]


def test_expected_pred_rules(mds):
    scores = np.array([1, 9, 9, 3, 100, 2])
    assert mds.expected_pred(scores, None) == 4
    ro = {'kind': 'detection', 'heads': {'hm': {'offset': 0, 'n_out': 1, 'h': 2, 'w': 2}}}
    assert mds.expected_pred(scores, ro) == 1        # hm = scores[0:4] の argmax(同点は最小)


# ---------------------------------------------------------------------------
# C ヘッダの往復(+ 実際に cc でコンパイルして値が読めること)
# ---------------------------------------------------------------------------
def _fake_doc(mds, *, kind='bits', n=5, sb=7, n_out=6, with_label=True, seed=3):
    rng = np.random.default_rng(seed)
    key = 'bits' if kind == 'bits' else 'image'
    samples = []
    for i in range(n):
        scores = rng.integers(-5000, 5000, size=n_out)
        samples.append({
            'index': i, 'name': f's{i}', 'label': (i % n_out) if with_label else None,
            key: bytes(rng.integers(0, 256, size=sb, dtype=np.uint8)).hex(),
            'expected_pred': int(np.argmax(scores)),
            'expected_hash': mds.fnv1a32_scores(scores),
            'expected_scores': [int(v) for v in scores],
        })
    return {
        'format': mds.FORMAT, 'model': 'toy-model.v1', 'dataset': 'toy',
        'input_kind': kind, 'input_width': sb * 8 - 3, 'input_shape': None,
        'sample_bytes': sb, 'encoding': 'hex', 'bit_order': 'lsb-first', 'n_out': n_out,
        'pred_rule': 'argmax', 'hash_rule': 'fnv', 'labels': None,
        'n_samples': n, 'samples': samples,
    }


def test_header_roundtrip(mds):
    doc = _fake_doc(mds)
    text = mds.render_header(doc)
    pf = mds._c_ident_prefix(doc['model'])
    assert pf == 'TOY_MODEL_V1_'
    back = mds.parse_header(text, pf)
    assert back['n'] == 5 and back['sample_bytes'] == 7 and back['n_out'] == 6
    assert np.array_equal(back['samples'], mds.payload_matrix(doc))
    assert back['labels'] == [s['label'] for s in doc['samples']]
    assert back['expected'] == [s['expected_pred'] for s in doc['samples']]
    assert back['hashes'] == [s['expected_hash'] for s in doc['samples']]
    assert np.array_equal(back['scores'],
                          np.array([s['expected_scores'] for s in doc['samples']]))


def test_header_unlabeled_and_empty_prefix(mds):
    doc = _fake_doc(mds, with_label=False)
    text = mds.render_header(doc, prefix='')
    assert 'static const uint8_t SAMPLES[5][7]' in text
    back = mds.parse_header(text, '')
    assert back['labels'] == [-1] * 5                       # ラベル無しは -1


def test_header_omits_scores_when_large(mds):
    doc = _fake_doc(mds, kind='image', n=3, sb=16, n_out=2000)
    text = mds.render_header(doc)
    pf = mds._c_ident_prefix(doc['model'])
    assert f'#define {pf}HAS_EXPECTED_SCORES 0' in text
    assert 'EXPECTED_SCORES[' not in text
    back = mds.parse_header(text, pf)
    assert back['scores'] is None
    assert back['hashes'] == [s['expected_hash'] for s in doc['samples']]


@pytest.mark.skipif(shutil.which('cc') is None, reason='Cコンパイラが無い')
def test_header_compiles_and_helpers_match_python(mds, tmp_path):
    """生成ヘッダを C99 でコンパイルし、unpack_bits / hash_scores が Python と一致する。"""
    doc = _fake_doc(mds, n=4, sb=5, n_out=6)
    doc['input_width'] = 37
    (tmp_path / 'toy_samples.h').write_text(mds.render_header(doc, prefix='T_'))
    (tmp_path / 'main.c').write_text(r'''
#include <stdio.h>
#include "toy_samples.h"
int main(void) {
    int i, j;
    uint8_t bits[T_INPUT_WIDTH];
    for (i = 0; i < T_N_SAMPLES; i++) {
        t_unpack_bits(T_SAMPLES[i], bits, T_INPUT_WIDTH);
        for (j = 0; j < T_INPUT_WIDTH; j++) printf("%d", bits[j]);
        printf(" %u %d\n", (unsigned)t_hash_scores(T_EXPECTED_SCORES[i], T_N_OUT),
               (int)T_EXPECTED[i]);
    }
    return 0;
}
''')
    exe = tmp_path / 'toy'
    r = subprocess.run(['cc', '-std=c99', '-Wall', '-Werror', '-o', str(exe),
                        str(tmp_path / 'main.c')], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    out = subprocess.run([str(exe)], capture_output=True, text=True, check=True).stdout
    mat = mds.payload_matrix(doc)
    want_bits = mds.unpack_bits_lsb(mat, 37)
    for i, line in enumerate(out.strip().splitlines()):
        bit_str, h, pred = line.split()
        assert bit_str == ''.join(str(int(b)) for b in want_bits[i])
        assert int(h) == doc['samples'][i]['expected_hash']
        assert int(pred) == doc['samples'][i]['expected_pred']


# ---------------------------------------------------------------------------
# 合成画像・summary・model.h・ELF
# ---------------------------------------------------------------------------
def test_synthetic_images_deterministic_and_prefix_stable(mds):
    plan = {'image_channels': 3, 'height': 32, 'width': 32}
    a = mds.synthetic_image_samples(plan, 12, seed=0)
    b = mds.synthetic_image_samples(plan, 12, seed=0)
    c = mds.synthetic_image_samples(plan, 6, seed=0)
    assert a.inputs.shape == (12, 3, 32, 32) and a.inputs.dtype == np.uint8
    assert np.array_equal(a.inputs, b.inputs)
    assert np.array_equal(a.inputs[:6], c.inputs)                # n を変えても先頭は不変
    assert a.names[:2] == ['zeros', 'ones']
    assert a.inputs[0].max() == 0 and a.inputs[1].min() == 255
    assert a.labels == [None] * 12


def test_synthetic_gray_channel_fit(mds):
    plan = {'image_channels': 1, 'height': 16, 'width': 16}
    ss = mds.synthetic_image_samples(plan, 8, seed=1)
    assert ss.inputs.shape == (8, 1, 16, 16)


def test_parse_model_h_macros(fdm):
    text = '''#define MDL_CTX_BYTES 12000
#define MDL_INPUT_WIDTH 4608   /* 入力 */
#  define MDL_OUTPUT_WIDTH 6
#define MDL_NUM_LABELS 6
#define MDL_MODEL_MODEL_H
#define NOT_MDL 3
'''
    m = fdm.parse_model_h_macros(text)
    assert m == {'MDL_CTX_BYTES': 12000, 'MDL_INPUT_WIDTH': 4608,
                 'MDL_OUTPUT_WIDTH': 6, 'MDL_NUM_LABELS': 6}


def _entry(name, gates, bundle, c, ctx, width, acc, dram_ok):
    return {
        'name': name, 'gates': {'gates_lut_layer': gates, 'gates_conv_weight_shared': None},
        'sizes': {'bundle_json_bytes': bundle, 'model_c_bytes': c},
        'macros': {'MDL_CTX_BYTES': ctx, 'MDL_INPUT_WIDTH': width},
        'accuracy': {'sim_first2900': acc},
        'esp32': {'fits_dram_320KB': dram_ok, 'fits_flash_4MB': True},
    }


def test_format_summary_table(fdm):
    summary = {'models': {
        'a': _entry('a', 36000, 565210, 400462, 12000, 4608, 0.89931, True),
        'b': _entry('b', 0, 271122, 121387, 1544000, 2834432, None, False),
    }}
    text = fdm.format_summary_table(summary)
    lines = text.splitlines()
    assert len(lines) == 4                                       # ヘッダ + 区切り + 2 行
    assert '| a | 36,000 | 565,210 | 400,462 | 12,000 | 4,608 | 0.89931 | DRAM/flash とも可 |' \
        in lines[2]
    assert '1,544,000' in lines[3] and 'flash可・DRAM不可' in lines[3] and '| - |' in lines[3]


def test_esp32_verdict_bits_and_image(fdm):
    har = fdm.esp32_verdict({'MDL_CTX_BYTES': 12000, 'MDL_INPUT_WIDTH': 4608,
                             'MDL_OUTPUT_WIDTH': 6}, 'bits', None, 400462)
    assert har['ram_working_bytes'] == 12000 + 4608 + 24
    assert har['fits_dram_320KB'] and har['fits_flash_4MB']
    face = fdm.esp32_verdict(
        {'MDL_CTX_BYTES': 1544000, 'MDL_INPUT_WIDTH': 2834432, 'MDL_OUTPUT_WIDTH': 5120,
         'MDL_IMAGE_CHANNELS': 3, 'MDL_INPUT_HEIGHT': 128, 'MDL_INPUT_WIDTH_PX': 128},
        'image_u8', {'flash': 150000, 'data': 0, 'bss': 0}, 121387)
    assert face['ram_breakdown']['input_buffer'] == 3 * 128 * 128      # 画像入力は u8 画像のみ
    assert not face['fits_dram_320KB'] and face['fits_flash_4MB']
    assert face['flash_estimate_bytes'] == 150000


def test_elf32_section_sizes(fdm):
    """最小の ELF32 を手組みして、flash/data/bss の分類を確かめる。"""
    import struct

    # セクション: (type, flags, size) を .text(ALLOC|EXEC) .rodata(ALLOC) .data(ALLOC|WRITE)
    # .bss(NOBITS, ALLOC|WRITE) と、非 ALLOC の .comment
    secs = [(1, 6, 100), (1, 2, 40), (1, 3, 8), (8, 3, 64), (1, 0, 999)]
    shoff = 0x40
    hdr = bytearray(0x40)
    hdr[0:4] = b'\x7fELF'
    hdr[4], hdr[5] = 1, 1
    struct.pack_into('<I', hdr, 0x20, shoff)
    struct.pack_into('<HH', hdr, 0x2E, 40, len(secs))
    body = bytearray()
    for t, fl, sz in secs:
        body += struct.pack('<IIIIIIIIII', 0, t, fl, 0, 0, sz, 0, 0, 0, 0)
    sizes = fdm.elf32_section_sizes(bytes(hdr) + bytes(body))
    assert sizes == {'flash': 140, 'data': 8, 'bss': 64}


def test_elf32_rejects_non_elf(fdm):
    with pytest.raises(ValueError):
        fdm.elf32_section_sizes(b'not an elf file at all' * 4)


def test_infer_name(mds):
    assert mds.infer_name(Path('out/har_dlgn_k6000.bundle.json')) == 'har_dlgn_k6000'
    assert mds.infer_name(Path('x/model.json')) == 'model'


def test_resolve_dataset(mds):
    har = {'preproc': {'kind': 'thermometer', 'bit_order': 'ch,t,thr'},
           'ir': {'format': 'x'}}
    assert mds.resolve_dataset('auto', har) == 'uci-har'
    assert mds.resolve_dataset('mnist', har) == 'mnist'          # 明示指定は尊重


# ---------------------------------------------------------------------------
# 小さな dense バンドルでの end-to-end(simulate 期待値 ⇔ C ランタイム)
# ---------------------------------------------------------------------------
@pytest.mark.skipif(shutil.which('cc') is None, reason='Cコンパイラが無い')
def test_make_samples_end_to_end_dense(webbench, mds, fdm, tmp_path):
    """合成 dense IR + HAR 形の thermometer preproc で、サンプル生成〜C 一致確認まで通す。"""
    from griffith_logic.deploy.bundle import build_bundle, save_bundle
    from griffith_logic.ir.serialize import ir_to_dict

    c, l, t, k = 9, 8, 4, 6
    graph = webbench.build_dense_ir(seed=3, in_width=c * l * t, widths=[24, 18], k=k,
                                    name='toy_har')
    rng = np.random.default_rng(0)
    x_tr = rng.normal(size=(60, c, l)).astype(np.float32)
    params = fdm.har_thermometer_params(x_tr, t)
    doc = {'ir': ir_to_dict(graph), 'preproc': fdm.har_preproc(params, t),
           'labels': list(fdm.HAR_LABELS),
           'readout': {'kind': 'class_scores', 'k': k, 'spatial': None}}
    bundle = build_bundle(doc)
    path = tmp_path / 'toy_har.bundle.json'
    save_bundle(bundle, path)

    # HAR ローダを差し替え(実データ不要): 合成の test 窓とラベル
    x_te = rng.normal(size=(30, c, l)).astype(np.float32)
    y_te = np.arange(30) % k
    orig = fdm.load_har_raw
    fdm.load_har_raw = lambda root, download=True: (x_tr, None, x_te, y_te)
    try:
        info = mds.make_samples(path, tmp_path / 'out', dataset='auto', n=12,
                                har_root=tmp_path, check_c=True, download=False)
    finally:
        fdm.load_har_raw = orig
    assert info['n_samples'] == 12 and info['checked_c'] == 12
    import json
    doc_out = json.loads((tmp_path / 'out' / 'toy_har.json').read_text())
    assert doc_out['input_kind'] == 'bits' and doc_out['input_width'] == c * l * t
    assert doc_out['sample_bytes'] == (c * l * t + 7) // 8
    assert [s['label'] for s in doc_out['samples']][:6] == [0, 1, 2, 3, 4, 5]
    # ヘッダは make_samples 内で読み戻し検証済み。ファイルが存在することだけ確認
    assert (tmp_path / 'out' / 'toy_har_samples.h').exists()


# ---------------------------------------------------------------------------
# k2400(版固定)・resynth(別名出力)の扱い
# ---------------------------------------------------------------------------
def test_har_k2400_spec_pinned_and_listed(fdm):
    spec = fdm.SPECS['har_dlgn_k2400']
    assert spec.kind == 'har'
    # :latest は別 run(test_acc 0.8352)。学習時 0.8903 の run g10xuhzw は v0
    assert spec.artifact == 'uci-har-4t_k2400_l4_n2_e50000-ir:v0'
    assert spec.filename == 'uci-har-4t_k2400_l4_n2_e50000.ir.json'
    assert fdm.KIND_MODELS['har'] == ['har_dlgn_k6000', 'har_dlgn_k2400', 'har_dlgn_k600']
    assert fdm.KIND_MODELS['all'][:3] == fdm.KIND_MODELS['har']
    assert set(fdm.KIND_MODELS['all']) == set(fdm.SPECS)


def test_artifact_dir_name_separates_pinned_versions(fdm):
    # :latest / 版なしは従来どおり(既存キャッシュを壊さない)
    assert fdm.artifact_dir_name('foo-ir:latest') == 'foo-ir'
    assert fdm.artifact_dir_name('foo-ir') == 'foo-ir'
    # 版固定は :latest のキャッシュと衝突しない別ディレクトリ
    assert fdm.artifact_dir_name('foo-ir:v0') == 'foo-ir-v0'
    assert fdm.artifact_name('foo-ir:v0') == 'foo-ir'
    assert fdm.artifact_dir_name(fdm.SPECS['har_dlgn_k6000'].artifact) \
        == 'uci-har-4t_k6000_l6_n2_e50000-ir'


def test_wandb_source_has_no_default_and_errors_clearly(fdm, monkeypatch):
    """取得元の entity/project は既定値を持たず、未指定なら指定方法を添えて止まる。"""
    assert fdm.WANDB_ENTITY == '' and fdm.WANDB_PROJECT == ''
    monkeypatch.delenv('WANDB_ENTITY', raising=False)
    monkeypatch.delenv('WANDB_PROJECT', raising=False)
    with pytest.raises(SystemExit, match='WANDB_ENTITY'):
        fdm.resolve_wandb_source(None, None)
    with pytest.raises(SystemExit, match='WANDB_PROJECT'):
        fdm.resolve_wandb_source('ent', None)
    # 引数 > 環境変数
    monkeypatch.setenv('WANDB_ENTITY', 'env-ent')
    monkeypatch.setenv('WANDB_PROJECT', 'env-proj')
    assert fdm.resolve_wandb_source(None, None) == ('env-ent', 'env-proj')
    assert fdm.resolve_wandb_source('arg-ent', 'arg-proj') == ('arg-ent', 'arg-proj')


def test_fetch_artifact_cache_hit_needs_no_wandb_source(fdm, monkeypatch, tmp_path):
    """キャッシュが揃っていれば取得元の指定なしで(wandb に触れず)動く。"""
    monkeypatch.delenv('WANDB_ENTITY', raising=False)
    monkeypatch.delenv('WANDB_PROJECT', raising=False)
    spec = fdm.SPECS['har_dlgn_k600']
    dest = tmp_path / fdm.artifact_dir_name(spec.artifact)
    dest.mkdir()
    (dest / spec.filename).write_text('{}', encoding='utf-8')
    (dest / 'provenance.json').write_text('{"artifact": "x"}', encoding='utf-8')
    path, prov = fdm.fetch_artifact(spec, tmp_path, offline=False)
    assert path == dest / spec.filename and prov == {'artifact': 'x'}


def test_resynth_variant_renames_without_touching_base(fdm):
    base = fdm.SPECS['det_face_lut_k256']
    assert fdm.resynth_variant(base, None) is base
    v = fdm.resynth_variant(base, 'lossless')
    assert v.name == 'det_face_lut_k256_resynth'
    assert fdm.SPECS['det_face_lut_k256'].name == 'det_face_lut_k256'   # 元の spec は不変
    assert v.artifact == base.artifact and v.filename == base.filename   # 取得元は同じ
    assert 'resynth' in v.title


def test_apply_resynth_passthrough_and_rejects_unknown(fdm):
    doc = {'ir': {}}
    assert fdm.apply_resynth(doc, None, 5) == (doc, None)
    with pytest.raises(SystemExit):
        fdm.apply_resynth(doc, 'lossy', 5)


def test_apply_resynth_noop_on_dense_and_writes_stats(webbench, fdm, tmp_path):
    """conv2d の無いモデルでは再合成は恒等。統計(所要時間付き)が <name>.resynth.json に残る。"""
    import json

    from griffith_logic.ir.serialize import ir_to_dict

    graph = webbench.build_dense_ir(seed=1, in_width=32, widths=[16, 12], k=4, name='toy')
    doc = {'ir': ir_to_dict(graph), 'preproc': {'kind': 'identity'},
           'labels': ['a', 'b', 'c', 'd'], 'readout': {'kind': 'class_scores', 'k': 4,
                                                       'spatial': None}}
    new_doc, st = fdm.apply_resynth(doc, 'lossless', 5)
    assert new_doc['ir'] == doc['ir'] and new_doc is not doc
    assert st['mode'] == 'lossless' and st['max_cost'] == 5 and st['seconds'] >= 0
    assert st['gates_before'] == st['gates_after'] and st['n_trees_changed'] == 0

    fdm.write_model_and_bundle(doc, tmp_path, 'toy_resynth', resynth='lossless')
    side = json.loads((tmp_path / 'toy_resynth.resynth.json').read_text())
    assert side['mode'] == 'lossless'
    assert (tmp_path / 'toy_resynth.bundle.json').exists()
    # resynth なしでは統計ファイルを作らない
    fdm.write_model_and_bundle(doc, tmp_path, 'toy_plain')
    assert not (tmp_path / 'toy_plain.resynth.json').exists()


def _toy_dense_doc(webbench):
    from griffith_logic.ir.serialize import ir_to_dict

    graph = webbench.build_dense_ir(seed=1, in_width=32, widths=[16, 12], k=4, name='toy')
    return {'ir': ir_to_dict(graph), 'preproc': {'kind': 'identity'},
            'labels': ['a', 'b', 'c', 'd'],
            'readout': {'kind': 'class_scores', 'k': 4, 'spatial': None}}


def test_write_model_and_bundle_applies_compress_by_default(webbench, fdm, tmp_path):
    """既定(fast)で CLI と同じ回路圧縮が入り、about.compress に残る。

    model.json は圧縮前の素の doc(= CLI の入力)のまま。
    """
    import json

    doc = _toy_dense_doc(webbench)
    bundle = fdm.write_model_and_bundle(doc, tmp_path, 'toy')
    comp = bundle['about']['compress']
    assert comp['level'] == 'fast'
    assert set(comp) >= {'level', 'applied', 'skipped'}
    saved = json.loads((tmp_path / 'toy.bundle.json').read_text())
    assert saved['about']['compress']['level'] == 'fast'
    model = json.loads((tmp_path / 'toy.model.json').read_text())
    assert 'compress' not in (model.get('about') or {})   # 圧縮前の doc


def test_write_model_and_bundle_compress_none_records_nothing(webbench, fdm, tmp_path):
    bundle = fdm.write_model_and_bundle(_toy_dense_doc(webbench), tmp_path, 'toy_none',
                                        compress='none')
    assert 'compress' not in (bundle.get('about') or {})


def test_write_model_and_bundle_resynth_not_run_twice(webbench, fdm, tmp_path):
    """--resynth 済みなら level=max でも再合成を繰り返さず、applied の先頭に 'resynth'。"""
    bundle = fdm.write_model_and_bundle(_toy_dense_doc(webbench), tmp_path, 'toy_rs',
                                        resynth='lossless', compress='max')
    comp = bundle['about']['compress']
    assert comp['level'] == 'max'
    assert comp['applied'][0] == 'resynth'
    assert comp['applied'].count('resynth') == 1
