"""don't-careカバレッジ解析+勾配ガイドflip修復(実タスクcheckpoint対応)。

各LutLayerニューロンの「観測入力パターン(4通り中いくつ踏まれたか)」を
訓練データで集計する。観測されないパターン=don't care(NullaNet/ReducedLUT)
であり、その率がタスク・アーキ横断の**hard表現の自由度の代理指標**になる。
`--repair`で勾配ガイド付きflip修復(analysis/guided_flip)を適用し、
hard精度/BPCのbefore/afterを測る。

対応ソース(checkpointのmetadata['args']からモデルを再構築):
    dense  — experiments/train.py のdense DLGN(MNIST/CIFAR系)
    cnn    — experiments/train_cnn.py のConv-DLGN(dense尾部のLutLayerを解析)
    charlm — experiments/train_kd_charlm_rddlgn.py の
             rddlgn / attention / nextchar student

Usage:
    uv run python experiments/dontcare_coverage.py --source dense \
        --ckpt checkpoints/xxx/best.pt --repair --limit 4096
    uv run python experiments/dontcare_coverage.py --source cnn \
        --ckpt checkpoints/cnn_mnist_*/best.pt --repair

wandb不使用。結果はstdout+JSON保存。
"""

from __future__ import annotations

import argparse
import json
import time
from argparse import Namespace
from datetime import datetime, timedelta, timezone

import numpy as np
import torch
import torch.nn.functional as F
from checkpoint_utils import load_checkpoint

from griffith_logic.analysis.guided_flip import guided_flip_repair
from griffith_logic.layers import LutLayer

JST = timezone(timedelta(hours=9))


# ---------------------------------------------------------------------------
# モデル再構築
# ---------------------------------------------------------------------------

def load_model_and_data(source: str, ckpt: str, device: str, limit: int,
                        batch_size: int | None = None):
    """(model, eval_batches, task_kind)を返す。

    eval_batches: [(x, y)] hard評価/勾配収集/修復に使う固定バッチ列
    task_kind: 'classification' | 'charlm'
    """
    data = torch.load(ckpt, map_location=device, weights_only=False)
    meta = data.get('metadata', {})
    margs = dict(meta.get('args', {}))

    if source == 'dense':
        import train as train_mod
        args = Namespace(**margs)
        args.device = device
        if batch_size:
            args.batch_size = batch_size
        model = train_mod.build_model(args).to(device)
        load_checkpoint(ckpt, model, device=device)
        train_loader, test_loader = train_mod.load_dataset(args)
        batches = _take_batches(train_loader, limit)
        test_batches = _take_batches(test_loader, limit)
        return model, batches, test_batches, 'classification', margs

    if source == 'cnn':
        import train_cnn as cnn_mod
        # 旧世代checkpointのmetadataに無い新引数はデフォルト埋め
        defaults = {'wide_last_layer': False, 'residual_init_dense': False,
                    'residual_init': False, 'batch_size': 256}
        args = Namespace(**{**defaults, **margs})
        args.device = device
        if batch_size:
            args.batch_size = batch_size
        builder = (cnn_mod.build_mnist_model if args.dataset == 'mnist'
                   else cnn_mod.build_cifar10_model)
        built = builder(args)
        model = built[0] if isinstance(built, tuple) else built
        model = model.to(device)
        load_checkpoint(ckpt, model, device=device)
        loaded = cnn_mod.load_dataset(args)  # ThresholdBinarize込み
        tl, el = loaded[0], loaded[1]
        return (model, _take_batches(tl, limit), _take_batches(el, limit),
                'classification', margs)

    raise ValueError(f'未知ソース: {source}')


def _take_batches(loader, limit: int):
    batches, n = [], 0
    for x, y in loader:
        batches.append((x, y))
        n += len(x)
        if n >= limit:
            break
    return batches


# ---------------------------------------------------------------------------
# カバレッジ集計
# ---------------------------------------------------------------------------

@torch.no_grad()
def coverage_stats(model: torch.nn.Module, batches, device: str) -> dict:
    """各LutLayerの観測パターンカバレッジを集計する。

    フックでeval(hard)経路の各層入力を捕捉し、ニューロンごとに
    4パターンの観測有無を数える。
    """
    layers = [m for m in model.modules() if isinstance(m, LutLayer)]
    seen = [torch.zeros(la.out_dim, la.n_entries, dtype=torch.bool,
                        device=device)
            for la in layers]

    handles = []
    for li, la in enumerate(layers):
        def hook(mod, inp, out, li=li, la=la):
            x = inp[0]
            n = la.n_inputs
            g = x[:, la.indices.T.reshape(-1)].reshape(
                x.shape[0], la.out_dim, n)
            addr = torch.zeros(x.shape[0], la.out_dim,
                               dtype=torch.long, device=x.device)
            for j in range(n):
                addr += (g[..., j].long() << j)
            for k in range(la.n_entries):
                seen[li][:, k] |= (addr == k).any(0)
        handles.append(la.register_forward_hook(hook))

    model.eval()
    for x, y in batches:
        model(x.to(device).round())
    for h in handles:
        h.remove()

    per_layer = []
    for li, la in enumerate(layers):
        s = seen[li].float()
        cov = s.mean().item()                      # 観測パターン率
        dc = 1.0 - cov                             # don't-care率
        full = (s.sum(-1) == 4).float().mean().item()
        per_layer.append({
            'layer': li, 'out_dim': la.out_dim,
            'pattern_coverage': cov, 'dont_care_rate': dc,
            'fully_observed_neuron_frac': full,
        })
    return {'per_layer': per_layer,
            'mean_dont_care_rate': float(np.mean(
                [r['dont_care_rate'] for r in per_layer]))}


# ---------------------------------------------------------------------------
# 修復
# ---------------------------------------------------------------------------

@torch.no_grad()
def hard_accuracy(model, batches, device: str) -> float:
    model.eval()
    correct, total = 0, 0
    for x, y in batches:
        pred = model(x.to(device).round()).argmax(-1).cpu()
        correct += (pred == y).sum().item()
        total += len(y)
    return correct / total


def run_repair(model, train_batches, device: str, *, topk: int,
               passes: int, eval_limit: int = 2048,
               eval_chunk: int = 256, grad_limit: int = 64) -> dict:
    x_grad, y_grad = train_batches[0]
    x_grad = x_grad[:grad_limit].to(device)
    y_grad = y_grad[:grad_limit].to(device)

    def loss_on_batch(m):
        return F.cross_entropy(m(x_grad.round()), y_grad)

    # 採否は訓練バッチ複数での(エラー数, CE)。巨大モデル(CIFAR conv尾部)の
    # OOM回避のためミニバッチ評価する。
    xs = torch.cat([b[0] for b in train_batches])[:eval_limit]
    ys = torch.cat([b[1] for b in train_batches])[:eval_limit]

    @torch.no_grad()
    def eval_metrics(m):
        m.eval()
        err, nll, n = 0.0, 0.0, 0
        for i in range(0, len(xs), eval_chunk):
            xb = xs[i:i + eval_chunk].to(device).round()
            yb = ys[i:i + eval_chunk].to(device)
            logits = m(xb)
            err += float((logits.argmax(-1) != yb).sum().item())
            nll += float(F.cross_entropy(logits, yb, reduction='sum').item())
            n += len(yb)
        return err, nll / max(1, n)

    rep = guided_flip_repair(model, loss_on_batch, eval_metrics,
                             topk=topk, passes=passes)
    return {
        'metrics_before': list(rep.metrics_before),
        'metrics_after': list(rep.metrics_after),
        'n_proposed': rep.n_proposed, 'n_accepted': rep.n_accepted,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, choices=['dense', 'cnn'])
    parser.add_argument('--ckpt', required=True)
    parser.add_argument('--device', default='cuda'
                        if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--limit', type=int, default=4096,
                        help='カバレッジ/修復に使う訓練サンプル数上限')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='ローダのバッチサイズ上書き(OOM対策)')
    parser.add_argument('--repair', action='store_true')
    parser.add_argument('--topk', type=int, default=64)
    parser.add_argument('--passes', type=int, default=6)
    parser.add_argument('--grad-limit', type=int, default=64,
                        help='勾配収集に使うサンプル数(巨大conv尾部のOOM対策)')
    parser.add_argument('--eval-chunk', type=int, default=256)
    parser.add_argument('--out', type=str, default=None)
    args = parser.parse_args()

    print(f'開始 {datetime.now(JST):%H:%M:%S} source={args.source} '
          f'ckpt={args.ckpt}', flush=True)
    model, train_b, test_b, kind, margs = load_model_and_data(
        args.source, args.ckpt, args.device, args.limit,
        batch_size=args.batch_size)

    t0 = time.time()
    cov = coverage_stats(model, train_b, args.device)
    print(f'カバレッジ({time.time() - t0:.0f}s): '
          f'mean don\'t-care率={cov["mean_dont_care_rate"]:.3f}')
    for r in cov['per_layer']:
        print(f'  layer{r["layer"]} out={r["out_dim"]:>7,} '
              f'coverage={r["pattern_coverage"]:.3f} '
              f'full_obs={r["fully_observed_neuron_frac"]:.3f}')

    result = {'source': args.source, 'ckpt': args.ckpt,
              'coverage': cov, 'meta_args': {
                  k: v for k, v in margs.items()
                  if isinstance(v, int | float | str | bool | type(None))}}

    acc_before = hard_accuracy(model, test_b, args.device)
    result['test_hard_acc_before'] = acc_before
    print(f'test hard acc (before) = {acc_before:.4f}')

    if args.repair:
        t0 = time.time()
        rep = run_repair(model, train_b, args.device,
                         topk=args.topk, passes=args.passes,
                         grad_limit=args.grad_limit,
                         eval_chunk=args.eval_chunk)
        acc_after = hard_accuracy(model, test_b, args.device)
        result['repair'] = rep
        result['test_hard_acc_after'] = acc_after
        print(f'修復({time.time() - t0:.0f}s): '
              f'train(err,CE) {rep["metrics_before"]} -> '
              f'{rep["metrics_after"]}  '
              f'test hard acc {acc_before:.4f} -> {acc_after:.4f} '
              f'({acc_after - acc_before:+.4f}) '
              f'accepted={rep["n_accepted"]}/{rep["n_proposed"]}')

    out = args.out or f'dontcare_coverage_{args.source}_results.json'
    with open(out, 'w') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f'結果JSON保存: {out}')


if __name__ == '__main__':
    main()
