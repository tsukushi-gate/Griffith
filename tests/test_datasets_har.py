"""UCI-HAR ローダと温度計符号化のテスト(ネットワーク不要)。

合成の txt を tmp_path に UCI の配置どおり書き出し、`load_uci_har` が
正しい軸・ラベル・キャッシュで読み戻すことを確かめる。実データは使わない。
"""

import zipfile

import numpy as np
import pytest

from griffith_logic.datasets import (
    UCI_HAR_CHANNELS,
    UCI_HAR_LABELS,
    fit_thermometer_thresholds,
    load_uci_har,
    thermometer_encode,
)
from griffith_logic.datasets import har as har_mod

N_TRAIN, N_TEST, WIN = 5, 3, 128


def _write_split(dataset_dir, split, n, offset):
    """1 分割ぶんの合成ファイルを書く。チャネル c・窓 i・時刻 t の値が一意に決まる。"""
    sig = dataset_dir / split / 'Inertial Signals'
    sig.mkdir(parents=True, exist_ok=True)
    expected = np.zeros((n, len(UCI_HAR_CHANNELS), WIN), dtype=np.float32)
    for c, ch in enumerate(UCI_HAR_CHANNELS):
        # 値 = チャネル番号*1000 + 窓番号*10 + 時刻/128(float32 で厳密に往復できる範囲)
        arr = (c * 1000 + (np.arange(n)[:, None] + offset) * 10
               + np.arange(WIN)[None, :] / 128).astype(np.float32)
        expected[:, c, :] = arr
        np.savetxt(sig / f'{ch}_{split}.txt', arr, fmt='%.7e')
    labels = (np.arange(n) % 6) + 1  # 公式は 1..6
    np.savetxt(dataset_dir / split / f'y_{split}.txt', labels, fmt='%d')
    subjects = np.arange(n) + 1 + offset
    np.savetxt(dataset_dir / split / f'subject_{split}.txt', subjects, fmt='%d')
    return expected, labels - 1, subjects


@pytest.fixture
def har_root(tmp_path):
    """展開済みの合成 'UCI HAR Dataset' を持つ root と期待値を返す。"""
    root = tmp_path / 'data-uci-har'
    dataset_dir = root / 'UCI HAR Dataset'
    train = _write_split(dataset_dir, 'train', N_TRAIN, 0)
    test = _write_split(dataset_dir, 'test', N_TEST, 100)
    return root, train, test


def test_load_roundtrip_shapes_and_labels(har_root):
    root, (xtr, ytr, _), (xte, yte, _) = har_root
    x_train, y_train, x_test, y_test = load_uci_har(str(root), download=False)
    assert x_train.shape == (N_TRAIN, 9, WIN) and x_train.dtype == np.float32
    assert x_test.shape == (N_TEST, 9, WIN)
    np.testing.assert_allclose(x_train, xtr, rtol=1e-6)
    np.testing.assert_allclose(x_test, xte, rtol=1e-6)
    # ラベルは 0 始まりの int64
    assert y_train.dtype == np.int64
    np.testing.assert_array_equal(y_train, ytr)
    np.testing.assert_array_equal(y_test, yte)
    assert y_train.min() >= 0 and y_train.max() < len(UCI_HAR_LABELS)


def test_channel_order_follows_inertial_signal_names(har_root):
    """軸 1 は UCI_HAR_CHANNELS の順(合成データは値の千の位がチャネル番号)。"""
    root, _, _ = har_root
    x_train, *_ = load_uci_har(str(root), download=False)
    for c in range(len(UCI_HAR_CHANNELS)):
        assert np.all(np.floor(x_train[:, c, :] / 1000) == c)


def test_cache_is_written_and_reused(har_root):
    root, _, _ = har_root
    first = load_uci_har(str(root), download=False)
    cache = root / 'har_raw.npz'
    assert cache.exists()
    # 元の txt を読めなくしても(別名へ付け替え)キャッシュだけで同じ結果が返る
    (root / 'UCI HAR Dataset').rename(root / 'moved')
    second = load_uci_har(str(root), download=False)
    for a, b in zip(first, second, strict=True):
        np.testing.assert_array_equal(a, b)


def test_return_subjects(har_root):
    root, (_, _, str_), (_, _, ste) = har_root
    out = load_uci_har(str(root), download=False, return_subjects=True)
    assert len(out) == 6
    np.testing.assert_array_equal(out[4], str_)
    np.testing.assert_array_equal(out[5], ste)
    # キャッシュ済みでも subject を取り出せる
    again = load_uci_har(str(root), download=False, return_subjects=True)
    np.testing.assert_array_equal(again[4], str_)


def test_missing_data_without_download_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_uci_har(str(tmp_path / 'nothing'), download=False)


def test_extracts_nested_zip_without_network(tmp_path, har_root, monkeypatch):
    """UCI 配布どおりの二重 zip を置けば、ダウンロードせずに展開して読める。"""
    src_root, (xtr, ytr, _), _ = har_root
    dataset_dir = src_root / 'UCI HAR Dataset'
    inner = tmp_path / 'inner.zip'
    with zipfile.ZipFile(inner, 'w') as zf:
        for p in sorted(dataset_dir.rglob('*')):
            if p.is_file():
                zf.write(p, p.relative_to(src_root).as_posix())
    outer = tmp_path / 'outer.zip'
    with zipfile.ZipFile(outer, 'w') as zf:
        zf.write(inner, 'UCI HAR Dataset.zip')

    root = tmp_path / 'fresh'
    root.mkdir()
    (root / har_mod._ZIP_NAME).write_bytes(outer.read_bytes())

    def _no_network(*_a, **_k):
        raise AssertionError('ネットワークに触れてはいけない')

    monkeypatch.setattr(har_mod.urllib.request, 'urlopen', _no_network)
    x_train, y_train, _, _ = load_uci_har(str(root), download=False)
    np.testing.assert_allclose(x_train, xtr, rtol=1e-6)
    np.testing.assert_array_equal(y_train, ytr)


def _nested_zip_bytes(tmp_path, har_root):
    """合成データを UCI 配布どおりの二重 zip にしたバイト列。"""
    src_root = har_root[0]
    dataset_dir = src_root / 'UCI HAR Dataset'
    inner = tmp_path / 'inner_dl.zip'
    with zipfile.ZipFile(inner, 'w') as zf:
        for p in sorted(dataset_dir.rglob('*')):
            if p.is_file():
                zf.write(p, p.relative_to(src_root).as_posix())
    outer = tmp_path / 'outer_dl.zip'
    with zipfile.ZipFile(outer, 'w') as zf:
        zf.write(inner, 'UCI HAR Dataset.zip')
    return outer.read_bytes()


def _fake_urlopen(payload):
    import io

    def _open(*_a, **_k):
        return io.BytesIO(payload)
    return _open


def test_download_is_checked_against_sha256(tmp_path, har_root, monkeypatch):
    """ダウンロードした zip は SHA-256 が一致したときだけ使う。"""
    import hashlib

    payload = _nested_zip_bytes(tmp_path, har_root)
    monkeypatch.setattr(har_mod.urllib.request, 'urlopen', _fake_urlopen(payload))
    root = tmp_path / 'dl_bad'
    with pytest.raises(ValueError, match='SHA-256'):
        load_uci_har(str(root), download=True)
    assert not (root / har_mod._ZIP_NAME).exists()        # 不一致の取得物は使わない

    monkeypatch.setattr(har_mod, 'UCI_HAR_SHA256', hashlib.sha256(payload).hexdigest())
    root = tmp_path / 'dl_ok'
    x_train, y_train, _, _ = load_uci_har(str(root), download=True)
    np.testing.assert_allclose(x_train, har_root[1][0], rtol=1e-6)


def test_official_sha256_constant_is_well_formed():
    assert len(har_mod.UCI_HAR_SHA256) == 64
    int(har_mod.UCI_HAR_SHA256, 16)


@pytest.mark.parametrize('evil', ['../escape.txt', '/abs/escape.txt',
                                  'UCI HAR Dataset/../../escape.txt'])
def test_extract_rejects_zip_slip(tmp_path, evil):
    """展開先が置き場の外を指すエントリがあれば、何も展開せずに止める。"""
    root = tmp_path / 'root'
    root.mkdir()
    zip_path = root / har_mod._ZIP_NAME
    with zipfile.ZipFile(zip_path, 'w') as zf:
        zf.writestr('UCI HAR Dataset/ok.txt', 'ok')
        zf.writestr(evil, 'pwned')
    with pytest.raises(ValueError, match='外を指しています'):
        load_uci_har(str(root), download=False)
    assert not (tmp_path / 'escape.txt').exists()
    assert not (root / 'UCI HAR Dataset').exists()          # 何も展開していない


def test_extract_rejects_symlink_entry(tmp_path):
    root = tmp_path / 'root'
    root.mkdir()
    zip_path = root / har_mod._ZIP_NAME
    info = zipfile.ZipInfo('UCI HAR Dataset/link')
    info.external_attr = (0o120777 << 16)
    with zipfile.ZipFile(zip_path, 'w') as zf:
        zf.writestr(info, '/etc/passwd')
    with pytest.raises(ValueError, match='外を指しています'):
        load_uci_har(str(root), download=False)


# --- 温度計符号化 -----------------------------------------------------------


def _signal(rng, n=20, c=3, length=16):
    return rng.normal(size=(n, c, length)).astype(np.float32) * np.arange(1, c + 1)[None, :, None]


def test_thermometer_shape_and_dtype():
    x = _signal(np.random.default_rng(0))
    bits = thermometer_encode(x, n_thresholds=4)
    assert bits.shape == (20, 3, 16, 4)
    assert bits.dtype == np.float32
    assert set(np.unique(bits)) <= {0.0, 1.0}
    assert bits.reshape(20, -1).shape == (20, 3 * 16 * 4)


def test_thermometer_is_monotone_in_threshold_axis():
    """温度計: 高い閾値のビットが立つなら、低い閾値のビットは必ず立っている。"""
    x = _signal(np.random.default_rng(1))
    bits = thermometer_encode(x, n_thresholds=5)
    assert np.all(np.diff(bits, axis=-1) <= 0)


def test_thermometer_is_monotone_in_value():
    """入力値が大きいほど立つビット数(=温度計の目盛り)は単調に増える。"""
    x = np.linspace(-3, 3, 200, dtype=np.float32).reshape(1, 1, 200)
    thr = np.array([[-1.0, 0.0, 1.0]], dtype=np.float32)
    level = thermometer_encode(x, thresholds=thr).sum(-1)[0, 0]
    assert np.all(np.diff(level) >= 0)
    assert level[0] == 0 and level[-1] == 3


def test_thermometer_uses_strict_greater_than():
    x = np.array([[[0.0, 0.5, 1.0]]], dtype=np.float32)
    bits = thermometer_encode(x, thresholds=np.array([0.5], dtype=np.float32))
    np.testing.assert_array_equal(bits[0, 0, :, 0], [0.0, 0.0, 1.0])


def test_fit_thresholds_are_train_quantiles_per_channel():
    rng = np.random.default_rng(2)
    x = _signal(rng, n=200, c=2, length=32)
    thr = fit_thermometer_thresholds(x, n_thresholds=4)
    assert thr.shape == (2, 4)
    assert np.all(np.diff(thr, axis=1) >= 0)
    for c in range(2):
        want = np.quantile(x[:, c, :], [0.2, 0.4, 0.6, 0.8])
        np.testing.assert_allclose(thr[c], want, rtol=1e-5)
    # 分位点で切るので、各ビットは訓練データ上でおおよそ 80/60/40/20% 立つ
    rate = thermometer_encode(x, thresholds=thr).mean(axis=(0, 2))
    np.testing.assert_allclose(rate, [[0.8, 0.6, 0.4, 0.2]] * 2, atol=0.02)


def test_thresholds_default_equals_self_fit_and_train_thresholds_apply_to_test():
    rng = np.random.default_rng(3)
    x_train, x_test = _signal(rng), _signal(rng) + 5.0
    thr = fit_thermometer_thresholds(x_train, 3)
    np.testing.assert_array_equal(
        thermometer_encode(x_train, 3), thermometer_encode(x_train, thresholds=thr))
    # 訓練由来の閾値だとテストの分布ずれ(+5)がそのまま符号に出る(自己フィットだと隠れる)
    assert thermometer_encode(x_test, thresholds=thr).mean() > 0.9


def test_thermometer_rejects_bad_inputs():
    x = _signal(np.random.default_rng(4))
    with pytest.raises(ValueError):
        thermometer_encode(x[0])  # 3 次元でない
    with pytest.raises(ValueError):
        thermometer_encode(x, thresholds=np.zeros((2, 3), dtype=np.float32))  # C 不一致
    with pytest.raises(ValueError):
        thermometer_encode(x, thresholds=np.array([1.0, 0.0], dtype=np.float32))  # 昇順でない
    with pytest.raises(ValueError):
        fit_thermometer_thresholds(x, 0)
