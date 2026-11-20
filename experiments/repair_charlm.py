"""char-LM checkpointへの勾配ガイド付きflip修復(①の本命)。

学習済みstudent(rddlgn / attention / nextchar)のcheckpointをロードし、
don't-careカバレッジ集計+guided_flip修復を適用して、実テキスト
val windowsでの**hard BPC** before/afterを測る。

採否指標: 訓練windowsサブセットでの(次文字エラー数, CE)辞書式。
最終評価: evaluate_bpc(train_kd_charlm_rddlgn と同一実装)のhard版。

Usage:
    uv run python experiments/repair_charlm.py \
        --ckpt checkpoints/kd_charlm_xxx/best.pt --topk 128 --passes 8

wandb不使用。結果はstdout+JSON保存。
"""

from __future__ import annotations

import argparse
import json
import time
from argparse import Namespace
from datetime import datetime, timedelta, timezone

import torch
import torch.nn.functional as F
from checkpoint_utils import load_checkpoint

from griffith_logic.analysis.guided_flip import guided_flip_repair
from griffith_logic.layers import LutLayer

JST = timezone(timedelta(hours=9))


def load_student(ckpt: str, device: str):
    import train_kd_charlm_rddlgn as ch

    data = torch.load(ckpt, map_location=device, weights_only=False)
    margs = dict(data.get('metadata', {}).get('args', {}))
    args = Namespace(**margs)
    args.device = device

    if args.student == 'rddlgn':
        config = ch.build_config_rddlgn(args)
        model = ch.RDDLGN(config, device=device).to(device)
    elif args.student == 'attention':
        config = ch.build_config_attention(args)
        model = ch.DLGNAttentionTranslator(config, device=device).to(device)
    elif args.student == 'nextchar':
        hidden = [int(v) for v in args.nextchar_hidden.split(',')]
        model = ch.NextCharBaseline(
            vocab_size=ch.STUDENT_VOCAB_SIZE, d_emb=args.d_emb,
            hidden_sizes=hidden, n_inputs=args.n_inputs, device=device,
        ).to(device)
    else:
        raise ValueError(f'修復未対応student: {args.student}')
    load_checkpoint(ckpt, model, device=device)
    return model, args, ch


def make_batch_fns(ch, model, args, windows: torch.Tensor, device: str,
                   n_eval: int, n_grad: int):
    """(loss_on_batch, eval_metrics, hard_bpc_fn)を構築する。"""
    grad_w = windows[:n_grad].to(device)
    eval_w = windows[:n_eval].to(device)

    def _forward_tf(m, batch):
        src = batch[:, :ch.SRC_LEN]
        tgt_target = batch[:, ch.SRC_LEN:ch.WINDOW]
        bos = torch.full((batch.shape[0], 1), ch.BOS_ID,
                         dtype=torch.long, device=batch.device)
        tgt_input = torch.cat([bos, batch[:, ch.SRC_LEN:ch.WINDOW - 1]],
                              dim=1)
        if args.student == 'nextchar':
            logits, _ = m(src)
            return logits[:, :ch.TEACHER_VOCAB], tgt_target[:, 0]
        logits, _ = m(src, tgt_input)
        return (logits[:, :, :ch.TEACHER_VOCAB].reshape(-1, ch.TEACHER_VOCAB),
                tgt_target.reshape(-1))

    def loss_on_batch(m):
        logits, target = _forward_tf(m, grad_w)
        return F.cross_entropy(logits, target)

    @torch.no_grad()
    def eval_metrics(m):
        m.eval()
        logits, target = _forward_tf(m, eval_w)
        err = float((logits.argmax(-1) != target).sum().item())
        return err, float(F.cross_entropy(logits, target).item())

    def hard_bpc(m, val_windows):
        m.eval()
        with torch.no_grad():
            return ch.evaluate_bpc(m, args.student, val_windows, device)

    return loss_on_batch, eval_metrics, hard_bpc


@torch.no_grad()
def coverage_summary(model, sample_windows, args, ch, device) -> dict:
    """LutLayerの観測パターンカバレッジ(概要のみ)。"""
    layers = [m for m in model.modules()
              if isinstance(m, LutLayer) and m.n_inputs == 2]
    seen = [torch.zeros(la.out_dim, 4, dtype=torch.bool, device=device)
            for la in layers]
    handles = []
    for li, la in enumerate(layers):
        def hook(mod, inp, out, li=li, la=la):
            x = inp[0]
            g = x[:, la.indices.T.reshape(-1)].reshape(
                x.shape[0], la.out_dim, 2)
            addr = (g[..., 0] + 2 * g[..., 1]).long()
            for k in range(4):
                seen[li][:, k] |= (addr == k).any(0)
        handles.append(la.register_forward_hook(hook))
    model.eval()
    batch = sample_windows.to(device)
    src = batch[:, :ch.SRC_LEN]
    bos = torch.full((batch.shape[0], 1), ch.BOS_ID, dtype=torch.long,
                     device=device)
    tgt_input = torch.cat([bos, batch[:, ch.SRC_LEN:ch.WINDOW - 1]], dim=1)
    if args.student == 'nextchar':
        model(src)
    else:
        model(src, tgt_input)
    for h in handles:
        h.remove()
    per = [{'layer': li, 'out_dim': la.out_dim,
            'pattern_coverage': seen[li].float().mean().item()}
           for li, la in enumerate(layers)]
    return {'per_layer': per}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ckpt', required=True)
    parser.add_argument('--device', default='cuda'
                        if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--topk', type=int, default=128)
    parser.add_argument('--passes', type=int, default=8)
    parser.add_argument('--n-grad-windows', type=int, default=256)
    parser.add_argument('--n-eval-windows', type=int, default=1024)
    parser.add_argument('--eval-max-chars', type=int, default=100_000)
    parser.add_argument('--out', type=str, default=None)
    args = parser.parse_args()

    print(f'開始 {datetime.now(JST):%H:%M:%S} ckpt={args.ckpt}', flush=True)
    model, margs, ch = load_student(args.ckpt, args.device)

    val_windows = ch.load_tiny_shakespeare_val(
        max_chars=args.eval_max_chars)
    # 修復用windows: valの先頭は最終評価と重複させない(後半から取る)
    n_repair = args.n_eval_windows
    repair_windows = val_windows[-n_repair:]
    eval_windows = val_windows[:max(1, len(val_windows) - n_repair)]

    cov = coverage_summary(model, repair_windows[:64], margs, ch,
                           args.device)
    for r in cov['per_layer']:
        print(f'  layer{r["layer"]} out={r["out_dim"]:>7,} '
              f'coverage={r["pattern_coverage"]:.3f}', flush=True)

    loss_fn, eval_fn, hard_bpc = make_batch_fns(
        ch, model, margs, repair_windows, args.device,
        n_eval=args.n_eval_windows, n_grad=args.n_grad_windows)

    bpc_before = hard_bpc(model, eval_windows)
    print(f'hard BPC (before) = {bpc_before:.4f}', flush=True)

    t0 = time.time()
    rep = guided_flip_repair(model, loss_fn, eval_fn,
                             topk=args.topk, passes=args.passes)
    bpc_after = hard_bpc(model, eval_windows)
    print(f'修復({time.time() - t0:.0f}s): '
          f'(err,CE) {rep.metrics_before} -> {rep.metrics_after}  '
          f'hard BPC {bpc_before:.4f} -> {bpc_after:.4f} '
          f'({bpc_after - bpc_before:+.4f})  '
          f'accepted={rep.n_accepted}/{rep.n_proposed}', flush=True)

    out = args.out or 'repair_charlm_results.json'
    with open(out, 'w') as f:
        json.dump({
            'ckpt': args.ckpt, 'student': margs.student,
            'coverage': cov,
            'hard_bpc_before': bpc_before, 'hard_bpc_after': bpc_after,
            'delta_bpc': bpc_after - bpc_before,
            'n_accepted': rep.n_accepted, 'n_proposed': rep.n_proposed,
            'topk': args.topk, 'passes': args.passes,
            'timestamp': datetime.now(JST).isoformat(),
        }, f, ensure_ascii=False, indent=2)
    print(f'結果JSON保存: {out}')


if __name__ == '__main__':
    main()
