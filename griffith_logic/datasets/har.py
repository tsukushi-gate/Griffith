"""UCI-HAR(スマホ加速度・ジャイロによる行動認識)の公開ローダと符号化ヘルパ。

UCI HAR は、30人の被験者がウエストにスマートフォンを付けて 6 種類の行動
(歩く・階段を上る・階段を下る・座る・立つ・横になる)をしている間の
加速度センサ/ジャイロの記録である。2.56 秒(50Hz × 128 サンプル)の窓ごとに
9 チャネルの生信号が並ぶ。

- 生信号 9ch × 128step(float32)をそのまま返す。公式が配っている 561 次元の
  手作り特徴量は使わない(論理ゲート網へは「生信号 → 符号化 → ビット列」で渡すため)
- ラベルは 0..5 の整数(公式の 1..6 から 1 引いてある)
- 初回は UCI の静的ファイル(約 60MB の zip)をダウンロードして展開し、
  `har_raw.npz` にキャッシュする。2 回目以降はキャッシュだけを読むので高速。
  ダウンロードした zip は SHA-256(`UCI_HAR_SHA256`)を照合してから使い、
  展開時は各エントリの展開先がデータ置き場の外に出ないこと(zip slip)を確かめる

出典とライセンス: Jorge Reyes-Ortiz, Davide Anguita, Alessandro Ghio, Luca Oneto,
Xavier Parra, "Human Activity Recognition Using Smartphones",
UCI Machine Learning Repository, 2013(https://doi.org/10.24432/C54S4K)。
CC BY 4.0 で配布されている。データ自体は同梱せず、利用時に UCI から取得する。

`thermometer_encode` は、実数の信号を「閾値を超えたか」の 0/1 ビットへ落とす
**温度計符号化**である。閾値が複数あるので、値が大きいほど 1 が下から順に積み上がる
(`0001 → 0011 → 0111 → 1111` のように単調に増える)。論理ゲート網(`LutLayer`)は
0/1 しか受け取れないので、実数センサ値を扱うにはこの符号化が入口になる。
"""

from __future__ import annotations

import hashlib
import shutil
import stat
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath

import numpy as np

from .vision import resolve_data_root

#: UCI Machine Learning Repository の静的配布 URL(約 60MB の zip)。
UCI_HAR_URL = (
    'https://archive.ics.uci.edu/static/public/240/'
    'human+activity+recognition+using+smartphones.zip'
)

#: `UCI_HAR_URL` の配布 zip(外側)の SHA-256。同 URL から取得して実測した値
#: (61,005,872 バイト。中身は `UCI HAR Dataset.names` と内側の `UCI HAR Dataset.zip`
#: = SHA-256 2045e435c955214b38145fb5fa00776c72814f01b203fec405152dac7d5bfeb0)。
#: ダウンロードした zip がこれと違えば使わずに止める。
UCI_HAR_SHA256 = 'c00b803081a5c797cd5e4b83700a9810b38d53d9d84e01917e090e1fdbc81031'

#: Inertial Signals のファイル名の前半(チャネル順がそのまま返り値の軸 1 になる)。
UCI_HAR_CHANNELS = (
    'body_acc_x', 'body_acc_y', 'body_acc_z',
    'body_gyro_x', 'body_gyro_y', 'body_gyro_z',
    'total_acc_x', 'total_acc_y', 'total_acc_z',
)

#: 0..5 のラベルに対応する行動名(公式の activity_labels.txt と同じ順)。
UCI_HAR_LABELS = (
    'WALKING', 'WALKING_UPSTAIRS', 'WALKING_DOWNSTAIRS',
    'SITTING', 'STANDING', 'LAYING',
)

_DIRNAME = 'data-uci-har'
_DATASET_DIR = 'UCI HAR Dataset'
_ZIP_NAME = 'UCI_HAR_Dataset.zip'
_CACHE_NAME = 'har_raw.npz'
_CACHE_KEYS = ('x_train', 'y_train', 'x_test', 'y_test')
_SUBJECT_KEYS = ('subject_train', 'subject_test')


def _read_split(dataset_dir: Path, split: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """展開済みディレクトリから 1 分割ぶん(train か test)を読む。"""
    sig_dir = dataset_dir / split / 'Inertial Signals'
    channels = [
        np.loadtxt(sig_dir / f'{ch}_{split}.txt', dtype=np.float32)
        for ch in UCI_HAR_CHANNELS
    ]
    x = np.stack(channels, axis=1)  # (N, 9, 128)
    y = np.loadtxt(dataset_dir / split / f'y_{split}.txt', dtype=np.int64) - 1
    subject = np.loadtxt(dataset_dir / split / f'subject_{split}.txt', dtype=np.int64)
    return x, y.reshape(-1), subject.reshape(-1)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def _download(url: str, dest: Path, sha256: str | None = None) -> None:
    """`url` を `dest` へ取得する(途中で切れた半端なファイルを残さない)。

    ``sha256`` を渡すと、取得したファイルのハッシュが一致したときだけ ``dest`` に
    置く。一致しなければ ``dest`` は作らず ``ValueError``(取得物は ``.part`` の
    まま残り、次の取得で上書きされる)。
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_suffix(dest.suffix + '.part')
    with urllib.request.urlopen(url) as resp, open(partial, 'wb') as f:  # noqa: S310
        shutil.copyfileobj(resp, f)
    if sha256 is not None:
        got = _sha256(partial)
        if got != sha256:
            raise ValueError(
                f'{url} から取得したファイルの SHA-256 が一致しません'
                f'(期待 {sha256}、実際 {got})。配布物が変わったか、取得が壊れています。'
                f'取得物は {partial} に残してあります')
    partial.replace(dest)


def _safe_extractall(zf: zipfile.ZipFile, root: Path) -> None:
    """zip の全エントリの展開先が ``root`` の中に収まることを確かめてから展開する。

    絶対パス・ドライブ指定・``..`` を含むエントリ、``root`` の外を指すエントリ、
    シンボリックリンクのエントリがあれば、何も展開せずに ``ValueError``(zip slip 対策)。
    """
    base = root.resolve()
    for info in zf.infolist():
        name = info.filename.replace('\\', '/')
        parts = PurePosixPath(name).parts
        bad = (name.startswith('/') or '..' in parts
               or (parts and ':' in parts[0])
               or stat.S_ISLNK(info.external_attr >> 16))
        target = (base / name).resolve()
        if bad or (target != base and base not in target.parents):
            raise ValueError(
                f'zip のエントリ {info.filename!r} の展開先が {root} の外を指しています'
                '(不正な zip。展開を中止しました)')
    zf.extractall(root)


def _extract(zip_path: Path, root: Path) -> Path:
    """zip を展開して 'UCI HAR Dataset' ディレクトリのパスを返す。

    UCI の配布 zip は「外側の zip の中に `UCI HAR Dataset.zip` がもう 1 枚入っている」
    二重構造なので、見つからなければ内側の zip も展開する。
    """
    dataset_dir = root / _DATASET_DIR
    with zipfile.ZipFile(zip_path) as zf:
        _safe_extractall(zf, root)
    if not dataset_dir.is_dir():
        for inner in sorted(root.glob('*.zip')):
            if inner == zip_path:
                continue
            with zipfile.ZipFile(inner) as zf:
                _safe_extractall(zf, root)
            if dataset_dir.is_dir():
                break
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f'{zip_path} を展開しても {dataset_dir} が現れませんでした')
    return dataset_dir


def _ensure_dataset_dir(root: Path, download: bool) -> Path:
    """'UCI HAR Dataset' ディレクトリを用意する(必要なら zip 取得→展開)。

    ダウンロードした zip は `UCI_HAR_SHA256` と照合する。利用者が自分で置いた
    zip(``root/UCI_HAR_Dataset.zip``)はそのまま使う(展開時の zip slip 検査は行う)。
    """
    dataset_dir = root / _DATASET_DIR
    if dataset_dir.is_dir():
        return dataset_dir
    zip_path = root / _ZIP_NAME
    if not zip_path.exists():
        if not download:
            raise FileNotFoundError(
                f'{dataset_dir} も {zip_path} も見つかりません'
                '(download=True で取得できます)'
            )
        _download(UCI_HAR_URL, zip_path, sha256=UCI_HAR_SHA256)
    return _extract(zip_path, root)


def load_uci_har(
    root: str | None = None,
    download: bool = True,
    return_subjects: bool = False,
):
    """UCI-HAR の生 Inertial Signals(9ch × 128step)を読む。

    既定の置き場所は repo 直下の `data-uci-har/`(`resolve_data_root` と同じ解決規則)。
    `har_raw.npz` があればそれだけを読む。無ければ `UCI HAR Dataset/` の
    テキストから読んで `har_raw.npz` を作り、そのディレクトリも無ければ
    zip(約 60MB)をダウンロードして展開する。

    Args:
        root: データ置き場。None なら `data-uci-har/` を自動解決する
        download: 手元に無いときダウンロードしてよいか。False なら
            見つからない場合 `FileNotFoundError` を送出する
        return_subjects: True なら被験者 ID(1..30)の配列も返す

    Returns:
        `(x_train, y_train, x_test, y_test)`。`x_*` は float32 の
        `(N, 9, 128)`(チャネル順は `UCI_HAR_CHANNELS`)、`y_*` は 0..5 の int64。
        `return_subjects=True` のときは末尾に `subject_train, subject_test`
        (int64、各窓を撮った被験者の ID)を足した 6 要素のタプルを返す。

    Raises:
        FileNotFoundError: データが無く `download=False` のとき
    """
    data_root = Path(resolve_data_root(_DIRNAME, root))
    cache = data_root / _CACHE_NAME

    if cache.exists():
        z = np.load(cache)
        have_subjects = all(k in z.files for k in _SUBJECT_KEYS)
        if all(k in z.files for k in _CACHE_KEYS) and (have_subjects or not return_subjects):
            out = [z[k] for k in _CACHE_KEYS]
            if return_subjects:
                out += [z[k] for k in _SUBJECT_KEYS]
            return tuple(out)

    dataset_dir = _ensure_dataset_dir(data_root, download)
    x_train, y_train, s_train = _read_split(dataset_dir, 'train')
    x_test, y_test, s_test = _read_split(dataset_dir, 'test')
    data_root.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache, x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test,
        subject_train=s_train, subject_test=s_test,
    )
    out = [x_train, y_train, x_test, y_test]
    if return_subjects:
        out += [s_train, s_test]
    return tuple(out)


def fit_thermometer_thresholds(x_train: np.ndarray, n_thresholds: int = 4) -> np.ndarray:
    """訓練データの分位点から、チャネルごとの温度計閾値を決める。

    各チャネルの全サンプル・全時刻の値を 1 つの分布とみなし、
    `i / (n_thresholds + 1)`(i = 1..n_thresholds)分位点を閾値にする。
    4 閾値なら 20 / 40 / 60 / 80 パーセンタイルである。分位点で切るので、
    どのビットも訓練データ上でおおよそ偏りなく立つ(情報を使い切りやすい)。

    Args:
        x_train: `(N, C, L)` の実数信号。**訓練分割だけ**を渡すこと
            (テストの分布を閾値に混ぜると評価が楽観的になる)
        n_thresholds: 1 チャネルあたりの閾値数(ビット数)

    Returns:
        float32 の `(C, n_thresholds)`。各行は昇順。
    """
    if n_thresholds < 1:
        raise ValueError(f'n_thresholds は 1 以上が必要です(got {n_thresholds})')
    if x_train.ndim != 3:
        raise ValueError(f'x_train は (N, C, L) の 3 次元が必要です(got {x_train.shape})')
    qs = np.arange(1, n_thresholds + 1) / (n_thresholds + 1)
    flat = x_train.transpose(1, 0, 2).reshape(x_train.shape[1], -1)
    return np.quantile(flat, qs, axis=1).T.astype(np.float32)


def thermometer_encode(
    x: np.ndarray,
    n_thresholds: int = 4,
    thresholds: np.ndarray | None = None,
) -> np.ndarray:
    """実数信号を温度計符号(単調に積み上がる 0/1 ビット)へ変換する。

    チャネル `c` の値 `v` に対し、閾値 `θ[c, 0] <= θ[c, 1] <= ...` ごとに
    `bit_t = (v > θ[c, t])` を並べる。閾値が昇順なので、`bit_t = 1` なら
    それより低い閾値のビットはすべて 1 になる(**温度計**のように下から埋まる)。

    閾値の決め方は 2 通り。`thresholds` を渡せばそれを使う(通常は
    `fit_thermometer_thresholds(x_train, n)` で**訓練データの分位点**から作った
    ものを、訓練・テスト両方に同じ値で適用する)。`None` のときは **入力 `x` 自身の
    分位点**を使う(`fit_thermometer_thresholds` と同じ式)。テスト分割にこれを
    使うと分布がリークするので、評価には必ず訓練由来の閾値を渡すこと。

    Args:
        x: `(N, C, L)` の実数信号(float 配列)
        n_thresholds: `thresholds` が None のときの閾値数。`thresholds` を
            渡す場合は無視される
        thresholds: `(C, T)` または `(T,)` の昇順の閾値。`(T,)` は全チャネル共通

    Returns:
        float32 の `(N, C, L, T)`(値は 0.0 / 1.0)。
        `reshape(N, -1)` すると `C × L × T` ビットの 1 本の入力線になる
        (並びは チャネル → 時刻 → 閾値 の順)。
    """
    x = np.asarray(x)
    if x.ndim != 3:
        raise ValueError(f'x は (N, C, L) の 3 次元が必要です(got {x.shape})')
    if thresholds is None:
        thr = fit_thermometer_thresholds(x, n_thresholds)
    else:
        thr = np.asarray(thresholds, dtype=np.float32)
        if thr.ndim == 1:
            thr = np.broadcast_to(thr, (x.shape[1], thr.shape[0]))
        if thr.ndim != 2 or thr.shape[0] != x.shape[1]:
            raise ValueError(
                f'thresholds は (C, T) か (T,) が必要です(C={x.shape[1]}, got {thr.shape})'
            )
        if np.any(np.diff(thr, axis=1) < 0):
            raise ValueError('thresholds は各チャネルで昇順である必要があります')
    bits = x[..., None] > thr[None, :, None, :]
    return bits.astype(np.float32)
