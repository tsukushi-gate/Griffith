#!/usr/bin/env python3
"""
WIDER FACE データセット取得スクリプト。

HuggingFace `CUHK-CSE/wider_face`（repo_type='dataset'）から
data/WIDER_train.zip(1.47GB) / data/WIDER_val.zip(363MB) /
data/wider_face_split.zip(3.6MB)（--include-test 指定時は
data/WIDER_test.zip も）を取得し、<root>/widerface/ へ展開する。

torchvision.datasets.WIDERFace._check_integrity() は split に関係なく
WIDER_train/WIDER_val/WIDER_test/wider_face_split の4つ全ての存在を
要求するため、--include-test 未指定時は WIDER_test を空ディレクトリとして
作成する（さもないと RuntimeError: Dataset not found になる）。

zipのtop-levelエントリは WIDER_train/ WIDER_val/ wider_face_split/ なので
extractall(<root>/widerface/) でそのまま正しい構造になる。

`load_dataset()` は使わない（GPU機Dの datasets==4.8.4 はスクリプトローダー
廃止のため CUHK-CSE/wider_face のようなカスタムローダースクリプト付き
データセットで失敗する）。huggingface_hub がimportできない環境向けに
urllib.request による直接取得フォールバックを持つ。

ライセンス: WIDER FACE は CC BY-NC-ND 4.0（非商用・研究目的限定）。
    http://shuoyang1213.me/WIDERFACE/

使い方:
    uv run python scripts/fetch_widerface.py [--root data-widerface]
                                              [--include-test] [--purge-cache]
"""

import argparse
import os
import shutil
import zipfile

HF_REPO_ID = 'CUHK-CSE/wider_face'
HF_REPO_TYPE = 'dataset'

# repo内パス（'data/' サブフォルダ配下）
FILES = {
    'WIDER_train.zip': 'data/WIDER_train.zip',
    'WIDER_val.zip': 'data/WIDER_val.zip',
    'wider_face_split.zip': 'data/wider_face_split.zip',
}
TEST_FILE = {'WIDER_test.zip': 'data/WIDER_test.zip'}

# 展開済みかどうかの冪等判定用（既知のjpg枚数）
EXPECTED_JPG_COUNTS = {'WIDER_train': 12880, 'WIDER_val': 3226}

LICENSE_NOTICE = (
    'WIDER FACE データセットは CC BY-NC-ND 4.0 ライセンスです。\n'
    '非商用・研究目的でのみ使用してください。\n'
    '詳細: http://shuoyang1213.me/WIDERFACE/'
)


def _count_jpgs(dir_path: str) -> int:
    if not os.path.isdir(dir_path):
        return 0
    n = 0
    for _root, _dirs, files in os.walk(dir_path):
        n += sum(1 for f in files if f.lower().endswith('.jpg'))
    return n


def _dir_size_gb(dir_path: str) -> float:
    total = 0
    for root, _dirs, files in os.walk(dir_path):
        for f in files:
            fp = os.path.join(root, f)
            if os.path.exists(fp):
                total += os.path.getsize(fp)
    return total / (1024 ** 3)


def _download_via_hf_hub(filename_in_repo: str, download_dir: str) -> str:
    from huggingface_hub import hf_hub_download
    return hf_hub_download(
        repo_id=HF_REPO_ID, repo_type=HF_REPO_TYPE,
        filename=filename_in_repo, local_dir=download_dir,
    )


def _download_via_urllib(filename_in_repo: str, download_dir: str) -> str:
    import urllib.request
    url = f'https://huggingface.co/datasets/{HF_REPO_ID}/resolve/main/{filename_in_repo}'
    dest_path = os.path.join(download_dir, os.path.basename(filename_in_repo))
    os.makedirs(download_dir, exist_ok=True)
    print(f'ダウンロード(urllibフォールバック): {url}')
    urllib.request.urlretrieve(url, dest_path)  # noqa: S310 (信頼済みHFホスト)
    return dest_path


def _fetch_zip(filename_in_repo: str, download_dir: str) -> str:
    """1zipファイルを取得し、ローカルパスを返す。"""
    try:
        import huggingface_hub  # noqa: F401
    except ImportError:
        return _download_via_urllib(filename_in_repo, download_dir)

    try:
        return _download_via_hf_hub(filename_in_repo, download_dir)
    except Exception as e:
        print(f'huggingface_hub経由の取得に失敗({e!r})。urllibフォールバックへ切替')
        return _download_via_urllib(filename_in_repo, download_dir)


def _extract(zip_path: str, extract_root: str) -> None:
    print(f'展開中: {zip_path} -> {extract_root}/')
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(extract_root)


def fetch_widerface(root: str, include_test: bool = False, purge_cache: bool = False) -> None:
    widerface_dir = os.path.join(root, 'widerface')
    os.makedirs(widerface_dir, exist_ok=True)
    download_dir = os.path.join(root, '_hf_download_cache')

    train_dir = os.path.join(widerface_dir, 'WIDER_train')
    val_dir = os.path.join(widerface_dir, 'WIDER_val')
    split_dir = os.path.join(widerface_dir, 'wider_face_split')
    test_dir = os.path.join(widerface_dir, 'WIDER_test')

    ready_marker = os.path.join(root, '.widerface_ready')

    already_ready = (
        os.path.exists(ready_marker)
        and _count_jpgs(train_dir) == EXPECTED_JPG_COUNTS['WIDER_train']
        and _count_jpgs(val_dir) == EXPECTED_JPG_COUNTS['WIDER_val']
        and os.path.isdir(split_dir)
        and os.path.isdir(test_dir)
    )
    if already_ready:
        print(f'既に展開済み: {widerface_dir}/ (skip)')
    else:
        files_to_get = dict(FILES)
        if include_test:
            files_to_get.update(TEST_FILE)

        for name, filename_in_repo in files_to_get.items():
            print(f'取得中: {name}')
            zip_path = _fetch_zip(filename_in_repo, download_dir)
            _extract(zip_path, widerface_dir)

        if not include_test:
            os.makedirs(test_dir, exist_ok=True)
            print(
                f'--include-test 未指定のため空ディレクトリを作成: {test_dir}/ '
                '(torchvision.datasets.WIDERFace._check_integrity() が'
                '存在チェックのみ行うため)'
            )

        with open(ready_marker, 'w') as f:
            f.write('ok\n')

    if purge_cache and os.path.isdir(download_dir):
        print(f'ダウンロードキャッシュを削除: {download_dir}/')
        shutil.rmtree(download_dir)

    n_train = _count_jpgs(train_dir)
    n_val = _count_jpgs(val_dir)

    print('=' * 60)
    print(f'WIDER_train: {n_train}枚 ({_dir_size_gb(train_dir):.2f} GB)')
    print(f'WIDER_val:   {n_val}枚 ({_dir_size_gb(val_dir):.2f} GB)')
    if include_test:
        n_test = _count_jpgs(test_dir)
        print(f'WIDER_test:  {n_test}枚 ({_dir_size_gb(test_dir):.2f} GB)')
    else:
        print('WIDER_test:  未取得(空ディレクトリのみ、--include-test で取得可)')
    print(f'ルート: {os.path.abspath(widerface_dir)}')
    print('-' * 60)
    print(LICENSE_NOTICE)
    print('=' * 60)


def main() -> None:
    parser = argparse.ArgumentParser(
        description='WIDER FACE データセット取得(HuggingFace CUHK-CSE/wider_face)',
    )
    parser.add_argument('--root', type=str, default='data-widerface',
                        help='取得先ルートディレクトリ (default: data-widerface)')
    parser.add_argument('--include-test', action='store_true',
                        help='data/WIDER_test.zip(GT非公開、推論専用)も取得する')
    parser.add_argument('--purge-cache', action='store_true',
                        help='展開後、ダウンロード済みzipのキャッシュを削除する')
    args = parser.parse_args()

    fetch_widerface(
        root=args.root, include_test=args.include_test,
        purge_cache=args.purge_cache,
    )


if __name__ == '__main__':
    main()
