#!/usr/bin/env python
"""webbench モデルJSONの conv2d 木を再合成して `<name>_dc.json` を書く。

本番モデルJSONは**上書きしない**。出力は必ず別名(既定 `_dc` サフィックス)。

    # lossless(全パターン care)。精度は厳密不変
    uv run python scripts/dc_resynth_webbench.py seg --mode lossless

    # don't care 再合成。キャリブレーション画像で観測したタップパターンのみ care
    uv run python scripts/dc_resynth_webbench.py seg --mode dc --calib 256

検証(lossy でも必ず実施):
    - キャリブレーションセット上で元IRと**出力整数が完全一致**すること
    - 再合成木が care set 上で元の木と一致すること(resynth_tree が構成的に保証。
      本スクリプトは care 外まで含めた実データ照合で二重に確認する)
"""
import argparse
import gzip
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / 'experiments'))

from griffith_logic.export.dc_resynth import (  # noqa: E402
    ResynthStats,
    forward_graph,
    resynth_node_params,
)
from griffith_logic.export.estimate import estimate_resources  # noqa: E402
from griffith_logic.export.optimize import (  # noqa: E402
    build_conv_program,
    spec_from_node_params,
)
from griffith_logic.ir.serialize import ir_from_dict  # noqa: E402

MODELS = {
    'det': 'det_multimnist_lut_k256.json',
    'detmono': 'det_multimnist_mono_lut_k256.json',
    'detface': 'det_face_lut_k256.json',
    'detreal': 'det_fewclass_lut_k64.json',
    'seg': 'seg_convdlgn_k128.json',
}
MODEL_DIR = REPO / 'tools' / 'webbench' / 'public' / 'models'


# ---------------------------------------------------------------------------
# キャリブレーション入力
# ---------------------------------------------------------------------------
def encode_inputs(imgs, preproc: dict) -> np.ndarray:
    """(N,C,H,W)[0,1] → (N,in_width) uint8。thermometer 系は rgb-major。"""
    if preproc['kind'] == 'image_lowlevel':
        from griffith_logic.encoding.image_features import encode_image_features
        bits = encode_image_features(
            imgs,
            n_thresholds=int(preproc['n_thresholds']),
            edge_thresholds=tuple(preproc['edge_thresholds']),
            curv_thresholds=tuple(preproc['curv_thresholds']),
            chroma_thresholds=tuple(preproc['chroma_thresholds']),
        )
        return bits.reshape(bits.shape[0], -1).numpy().astype(np.uint8)
    mean = np.asarray(preproc['mean'])[None, :, None, None]
    std = np.asarray(preproc['std'])[None, :, None, None]
    thr = np.asarray(preproc['thresholds'])
    z = (imgs.numpy() - mean) / std
    bits = (z[:, :, None] > thr[None, :, :, None, None]).astype(np.uint8)
    return bits.reshape(bits.shape[0], -1)


def make_dataset(task: str, size: int, *, train: bool, n: int, seed: int = 7):
    if task in ('det', 'detmono'):
        from data.detection import MultiMNISTDetection
        color = task == 'det'
        return MultiMNISTDetection(
            mnist_root=str(REPO / 'data-mnist'), train=train, canvas_size=size,
            n_thresholds=0, num_samples=max(n, 1), seed=seed,
            color=color, use_rgb=color, cifar_root=str(REPO / 'data-cifar10'))
    if task == 'detface':
        from data.detection_face import WiderFaceDetection
        return WiderFaceDetection(
            root=str(REPO / 'data-widerface'), train=train, img_size=size,
            n_thresholds=0, use_rgb=True, download=False, cache_index=False)
    if task == 'detreal':
        from data.detection_fewclass import FewClassVOCDetection
        return FewClassVOCDetection(
            root=str(REPO / 'data-voc'), train=train, img_size=size,
            n_thresholds=0, use_rgb=True, download=False, cache_index=False)
    if task == 'seg':
        from griffith_logic.datasets.segmentation import OxfordPetBinarySeg
        return OxfordPetBinarySeg(
            root=str(REPO / 'data-oxford-pet'),
            split='trainval' if train else 'test',
            img_size=size, encoding='float', border='foreground', download=False)
    raise ValueError(task)


def take_images(task: str, size: int, *, train: bool, n: int, seed: int = 7):
    import torch
    ds = make_dataset(task, size, train=train, n=n, seed=seed)
    imgs = [ds[i][0] for i in range(min(n, len(ds)))]
    return torch.stack(imgs)


def iter_encoded(task: str, preproc: dict, *, train: bool, n: int,
                 batch: int, seed: int = 7):
    """符号化済み入力をバッチ単位で生成する(全枚数を同時に持たない)。

    det/detface は 1枚あたり 1.6-2.8Mbit あり、256枚を一括符号化すると
    数百MBになるため必ずチャンク化する。
    """
    import torch
    size = int(preproc['height'])
    ds = make_dataset(task, size, train=train, n=n, seed=seed)
    total = min(n, len(ds))
    for i in range(0, total, batch):
        imgs = torch.stack([ds[j][0] for j in range(i, min(i + batch, total))])
        yield encode_inputs(imgs, preproc)


# ---------------------------------------------------------------------------
# 再合成
# ---------------------------------------------------------------------------
def conv_gate_total(nodes) -> int:
    """全conv2dノードの A1 縮約後ゲート数の合計。"""
    return sum(build_conv_program(spec_from_node_params(n['params'])).stats.gates_after
               for n in nodes if n['type'] == 'conv2d')


def resynth_doc(doc: dict, care: dict | None, *, max_cost: int, verbose=True,
                keep_demo_samples: bool = False):
    """docのconv2dノードを再合成した新docと合計統計を返す(docは変更しない)。

    ``demo_samples`` は既定で落とす。理由は再配布条件:
    ``tools/webbench/scripts/strip_restricted.mjs`` は WIDER FACE / VOC 由来の
    埋め込み画像を**ファイル名の明示リスト**で除去するため、``_dc`` などの
    別名ファイルはその網に掛からない。再合成モデルは解析・採用検討用の
    生成物であり demo_samples は不要なので、既定で空にして事故を防ぐ
    (採用時は本番docの ``ir`` を差し替える運用)。
    """
    new_doc = json.loads(json.dumps(doc))
    if not keep_demo_samples:
        n = len(new_doc.get('demo_samples') or [])
        if n:
            new_doc['demo_samples'] = []
            new_doc['demo_samples_removed'] = {
                'count': n,
                'reason': '再合成モデル(別名ファイル)は strip_restricted.mjs の '
                          '除去対象名に入らないため、既定で埋め込み画像を持たない',
            }
    total = ResynthStats()
    for node in new_doc['ir']['nodes']:
        if node['type'] != 'conv2d':
            continue
        cs = None if care is None else care[node['id']]
        new_params, st = resynth_node_params(node['params'], cs, max_cost=max_cost)
        node['params'] = new_params
        total.merge(st)
        if verbose:
            print(f"    {node['id']:26s} ゲート {st.cost_before:5d} -> {st.cost_after:5d} "
                  f"(木 {st.n_changed}/{st.n_trees} 変更, "
                  f"care {st.care_patterns / max(st.n_trees, 1):.1f}/"
                  f"{st.care_total / max(st.n_trees, 1):.0f})")
    return new_doc, total


def refresh_resources(doc: dict) -> None:
    """resources を再計算する。

    json_bytes/gzip_bytes は build_model_doc と同じ規約(demo_samples を
    含めない doc に対する測定)にそろえる。配信実体のサイズは別途
    ``full_bytes`` として全docに対して測る。
    """
    graph = ir_from_dict(doc['ir'])
    res = estimate_resources(graph)
    doc['resources'] = res
    core = {k: v for k, v in doc.items() if k != 'demo_samples'}
    blob = json.dumps(core, separators=(',', ':')).encode()
    res['json_bytes'] = len(blob)
    res['gzip_bytes'] = len(gzip.compress(blob, 6))
    full = json.dumps(doc, separators=(',', ':')).encode()
    res['full_bytes'] = len(full)
    res['full_gzip_bytes'] = len(gzip.compress(full, 6))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('tasks', nargs='*', default=list(MODELS))
    ap.add_argument('--mode', choices=('lossless', 'dc'), default='dc')
    ap.add_argument('--calib', type=int, default=256, help='キャリブレーション画像数')
    ap.add_argument('--verify', type=int, default=32, help='整数一致検証に使う画像数')
    ap.add_argument('--max-cost', type=int, default=4)
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--suffix', default=None, help='既定: lossless→_ll / dc→_dc')
    ap.add_argument('--keep-demo-samples', action='store_true',
                    help='埋め込みサンプル画像を残す(再配布条件に注意)')
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()
    suffix = args.suffix or ('_ll' if args.mode == 'lossless' else '_dc')

    for task in args.tasks:
        src = MODEL_DIR / MODELS[task]
        doc = json.loads(src.read_text())
        graph = ir_from_dict(doc['ir'])
        conv_ids = {n.id for n in graph.nodes if n.type == 'conv2d'}
        print(f'\n=== {task} ({src.name}) mode={args.mode} ===')

        care = None
        n_cal = 0
        if args.mode == 'dc':
            print(f'  キャリブレーション {args.calib}枚 → care set 採取')
            care = {}
            for xb in iter_encoded(task, doc['preproc'], train=True,
                                   n=args.calib, batch=args.batch):
                n_cal += len(xb)
                _, c = forward_graph(graph, xb, care_nodes=conv_ids)
                for k, v in c.items():
                    if k not in care:
                        care[k] = [set(s) for s in v]
                    else:
                        for a, b in zip(care[k], v, strict=True):
                            a |= b
            print(f'    実枚数 {n_cal}')

        before = conv_gate_total(doc['ir']['nodes'])
        new_doc, total = resynth_doc(doc, care, max_cost=args.max_cost,
                                     keep_demo_samples=args.keep_demo_samples)
        after = conv_gate_total(new_doc['ir']['nodes'])
        print(f'  conv2dゲート合計: {before} -> {after} '
              f'({(1 - after / before) * 100:.1f}%減、木 {total.n_changed}/{total.n_trees} 変更)')

        # ---- 検証: 実データ上で元IRと出力整数が一致するか ----
        new_graph = ir_from_dict(new_doc['ir'])
        checks = [('calib', True, n_cal)] if n_cal else []
        if args.verify > 0:
            checks.append(('test', False, args.verify))
        for name, train, cnt in checks:
            same = tot = 0
            for xb in iter_encoded(task, doc['preproc'], train=train, n=cnt,
                                   batch=args.batch, seed=7 if train else 99):
                a, _ = forward_graph(graph, xb)
                b, _ = forward_graph(new_graph, xb)
                same += int((np.asarray(a) == np.asarray(b)).all(axis=1).sum())
                tot += len(xb)
            print(f'  出力整数一致({name}, n={tot}): {same}/{tot} '
                  f'= {same / max(tot, 1) * 100:.1f}%')

        refresh_resources(new_doc)
        r = new_doc['resources']
        print(f"  json {doc['resources']['json_bytes']:,}B -> {r['json_bytes']:,}B / "
              f"gzip {doc['resources']['gzip_bytes']:,}B -> {r['gzip_bytes']:,}B")
        if not args.dry_run:
            dst = MODEL_DIR / MODELS[task].replace('.json', f'{suffix}.json')
            assert dst != src, '本番モデルJSONを上書きしようとしている'
            dst.write_text(json.dumps(new_doc, separators=(',', ':')))
            print(f'  wrote {dst} ({dst.stat().st_size:,}B)')


if __name__ == '__main__':
    main()
