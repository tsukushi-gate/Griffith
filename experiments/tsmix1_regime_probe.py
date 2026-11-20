"""
 候補(iv)の検証: 崩壊ギャップは「条件付けの regime」で向きが変わるか

背景（レポート §4.11 の図D）:
    論理版の本走で、**BLEU の崩壊ギャップ(soft − hard)は縮む**のに
    **teacher forcing の accuracy ギャップは開く**という食い違いが観測された。
    埋め込み二値化正則化のランプで説明する仮説は反証済み（符号が逆・寄与が
    main loss の 0.025%・soft acc が上昇）。

検証する候補(iv)（**これも仮説であり、断定しない**）:
    2つの指標は条件付けの regime が違う。
      - accuracy は **teacher forcing**（各位置が正解 prefix に条件付けられる）
      - BLEU は **greedy decode の自由生成**（モデル自身の出力 prefix に
        条件付けられ、誤差が累積する = exposure bias）
    hard は argmax の決定的出力なので、学習が進むほど**自由生成の軌道が安定**し、
    自由生成側のギャップだけが縮む——という説明が成り立ちうる。

**事前登録した判定基準**（実行前に決め、実測に合わせて動かさない）:
    - 自由生成 acc ギャップが **BLEU ギャップと同方向（縮小）**
      → 候補(iv)を支持。「regime の違いで説明できる」と書ける
    - 自由生成 acc ギャップが **teacher forcing acc ギャップと同方向（拡大）**
      → 候補(iv)は否定。regime ではなく指標そのものの性質が原因
    - どちらとも言えない → **未解明のまま残す**
    判定は「学習の前半→後半でギャップが増えたか減ったか」の符号と、
    ギャップ系列と step の相関 r の符号で行う。

方針:
    - **既存の BLEU 4条件を1つも変えない**（tokenize引数なし=13a / `max_len=tgt.shape[1]`
      / `max_samples=1000` / 空ref除外）。自由生成 accuracy は**別指標として追加**するだけ。
    - `logic_seq_mixer.py` / `train_logic_seq_mixer.py` は **import して再利用するだけ**で
      一切変更しない（走行中 run と同じコードのため）。
    - 本走の checkpoint があれば `--checkpoint` でそれを測る。無ければローカル MPS で
      小規模モデルを学習しながら3種のギャップを追跡する（**機序が起こりうるかの
      プローブであって、本走スケールの再現ではない**ことをレポートに明記すること）。

使用例:
    # ローカル小規模プローブ（WMT'14 vocab1600 キャッシュを流用・MPS）
    uv run python experiments/tsmix1_regime_probe.py \\
        --data-dir ./data-wmt14 --vocab-size 1600 -ni 2000 -ef 400

    # 本走 checkpoint を1点だけ測る
    uv run python experiments/tsmix1_regime_probe.py \\
        --checkpoint checkpoints/<run_dir>/best.pt --data-dir ./data-wmt14
"""

import argparse
import json
import os
import random
import sys
import time
from dataclasses import asdict, replace
from datetime import datetime, timedelta, timezone

JST = timezone(timedelta(hours=9))

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))

from data import TOKEN_PAD
from env_utils import load_dotenv
from train_logic_seq_mixer import (
    compute_bleu,
    evaluate,
    get_emb_reg_weight,
    greedy_decode,
    load_n,
    prepare_batch,
)
from wandb_utils import finish as wandb_finish
from wandb_utils import init_wandb, log_metrics, normalize_tags

from griffith_logic.models import LogicSeqMixer, LogicSeqMixerConfig

load_dotenv()


@torch.no_grad()
def generation_accuracy(model, loader, device, max_samples=1000, use_soft=False,
                        chunk_size=0) -> float:
    """**自由生成**でのトークン一致率（teacher forcing ではない）。

    `train_logic_seq_mixer.compute_bleu` と**同じ経路**で greedy decode し
    （BOS開始 / 毎ステップ prefix 全体を再デコード / `logits[:, -1].argmax`）、
    その出力を `[BOS, tgt, EOS][1:]` と位置ごとに突き合わせて PAD を除いた
    一致率を返す。BLEU の計算条件には**一切触れない**（別指標として追加するだけ）。

    `evaluate()`（teacher forcing 版）と**同じ集合・同じマスク**で測るので、
    2つの値の差はそのまま「条件付けの regime の違い」になる。
    """
    was_training = model.training
    model.train(mode=use_soft)

    correct = 0
    total = 0
    count = 0

    for src, tgt in loader:
        if count >= max_samples:
            break
        src = src.to(device)
        tgt = tgt.to(device)
        src_batch, tgt_full = prepare_batch(src, tgt)
        labels = tgt_full[:, 1:]

        take = min(src.shape[0], max_samples - count)
        preds = _chunked_decode(model, src_batch[:take], labels.shape[1], chunk_size)

        mask = labels[:take] != TOKEN_PAD
        correct += ((preds == labels[:take]) & mask).sum().item()
        total += mask.sum().item()
        count += take

    model.train(mode=was_training)
    return correct / total if total > 0 else 0.0


def _chunked_decode(model, src, max_len, chunk_size):
    """greedy_decode をチャンク分割して呼ぶ（結果は分割前と厳密に同一）。"""
    if chunk_size is None or chunk_size <= 0 or chunk_size >= src.shape[0]:
        return greedy_decode(model, src, max_len=max_len)
    outs = []
    for start in range(0, src.shape[0], chunk_size):
        outs.append(greedy_decode(model, src[start:start + chunk_size],
                                  max_len=max_len))
    return torch.cat(outs, dim=0)


def pearson(xs, ys) -> float | None:
    """標本ピアソン相関（点が2未満・分散0なら None）。"""
    n = len(xs)
    if n < 2 or len(ys) != n:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((a - mx) * (b - my) for a, b in zip(xs, ys, strict=True))
    den = (sum((a - mx) ** 2 for a in xs) * sum((b - my) ** 2 for b in ys)) ** 0.5
    return None if den == 0 else num / den


def measure(model, loader, tokenizer, device, max_samples, chunk_size) -> dict:
    """1時点の3種の指標を soft/hard 両方で測る。"""
    tf_soft = evaluate(model, loader, device, use_soft=True, chunk_size=chunk_size)
    tf_hard = evaluate(model, loader, device, use_soft=False, chunk_size=chunk_size)
    gen_soft = generation_accuracy(model, loader, device, max_samples=max_samples,
                                   use_soft=True, chunk_size=chunk_size)
    gen_hard = generation_accuracy(model, loader, device, max_samples=max_samples,
                                   use_soft=False, chunk_size=chunk_size)
    bleu_soft = compute_bleu(model, loader, tokenizer, device,
                             use_soft=True, chunk_size=chunk_size)
    bleu_hard = compute_bleu(model, loader, tokenizer, device,
                             use_soft=False, chunk_size=chunk_size)
    row = {
        'tf_acc/soft': tf_soft, 'tf_acc/hard': tf_hard,
        'tf_acc/gap': tf_soft - tf_hard,
        'gen_acc/soft': gen_soft, 'gen_acc/hard': gen_hard,
        'gen_acc/gap': gen_soft - gen_hard,
    }
    if bleu_soft is not None and bleu_hard is not None:
        row.update({'bleu/soft': bleu_soft, 'bleu/hard': bleu_hard,
                    'bleu/gap': bleu_soft - bleu_hard})
    return row


# --- 単一チェックポイント用の事前登録パラメータ（実行前に決めて動かさない） ---
# 「ほぼ一致」とみなす相対差の上限。|gen_gap - tf_gap| / |tf_gap| がこれ未満なら一致。
_SINGLE_POINT_TOL = 0.20
# 縮退解の検知しきい値。本走の soft acc は 0.31 前後なので、これを大きく下回る
# モデルは定数出力に張り付いた縮退解の疑いがあり、判定に使わない
# （小規模プローブが tf soft acc 0.0816 / hard 0.5001 固定で失敗した経験に基づく）。
_DEGENERATE_SOFT_ACC = 0.15


def verdict_single_point(row: dict) -> dict:
    """**単一チェックポイント**での判定（水準の比較のみ。推移は判定しない）。

    単一点で言えないこと:
        候補(iv) の元の問い（BLEUギャップは縮むのに accギャップは開くという
        **推移**の食い違い）は判定できない。推移には複数 step の checkpoint が
        必要だが、保存されるのは `latest.pt` / `best.pt` だけである。

    単一点でも言えること（事前登録した3分岐）:
        1. 自由生成ギャップ ≈ teacher forcing ギャップ（相対差 < 20%）
           → soft/hard の差は regime に依存しない。候補(iv)は「BLEUとaccが
             逆方向に動く理由」としては**弱い**。説明は候補(i)（BLEUのn-gram
             連なり依存性）側に寄る。
        2. 自由生成ギャップが**明確に小さい**（相対差 20% 以上）
           → 離散化の影響が自由生成で緩和されている。BLEU（自由生成で測る）の
             ギャップが acc（teacher forcing で測る）より縮みやすいことの説明に
             なり、候補(iv)を**部分的に支持**する。
        3. 自由生成ギャップが**明確に大きい**
           → 誤差累積が離散化で悪化している。候補(iv)とは逆向きだが、
             それ自体が報告価値のある知見。
    """
    tf_gap = row['tf_acc/gap']
    gen_gap = row['gen_acc/gap']
    out: dict[str, object] = {
        'scope': '水準の比較のみ。推移（BLEUギャップとaccギャップの動く向きの'
                 '食い違い）は単一チェックポイントでは判定不能',
        'tf_acc/gap': tf_gap,
        'gen_acc/gap': gen_gap,
        'diff(gen - tf)': gen_gap - tf_gap,
        'tolerance': _SINGLE_POINT_TOL,
    }
    if row['tf_acc/soft'] < _DEGENERATE_SOFT_ACC:
        out['candidate_iv_single_point'] = (
            f'判定不能: soft acc {row["tf_acc/soft"]:.4f} が縮退しきい値 '
            f'{_DEGENERATE_SOFT_ACC} を下回る（定数出力に張り付いた解の疑い）')
        return out

    denom = max(abs(tf_gap), 1e-9)
    rel = abs(gen_gap - tf_gap) / denom
    out['relative_diff'] = rel
    if rel < _SINGLE_POINT_TOL:
        out['candidate_iv_single_point'] = (
            f'分岐1: 両ギャップがほぼ一致（相対差 {rel:.1%} < {_SINGLE_POINT_TOL:.0%}）。'
            'soft/hardの差はregimeに依存せず、候補(iv)は弱い。'
            '説明は候補(i)（BLEUのn-gram連なり依存性）側に寄る')
    elif abs(gen_gap) < abs(tf_gap):
        out['candidate_iv_single_point'] = (
            f'分岐2: 自由生成ギャップが明確に小さい（相対差 {rel:.1%}）。'
            '離散化の影響が自由生成で緩和されており、候補(iv)を部分的に支持する')
    else:
        out['candidate_iv_single_point'] = (
            f'分岐3: 自由生成ギャップが明確に大きい（相対差 {rel:.1%}）。'
            '誤差累積が離散化で悪化している。候補(iv)とは逆向きだが報告価値がある')
    return out


def verdict(rows: list[dict]) -> dict:
    """事前登録した判定基準を機械的に適用する（後付けを避けるため）。

    点が1つ（`--checkpoint` で1点だけ測った場合）なら
    :func:`verdict_single_point` の水準判定へ委譲する。
    """
    if len(rows) == 1:
        return verdict_single_point(rows[0])
    steps = [r['step'] for r in rows]
    out: dict[str, object] = {}
    for key in ('tf_acc/gap', 'gen_acc/gap', 'bleu/gap'):
        vals = [r[key] for r in rows if key in r]
        sub = [s for s, r in zip(steps, rows, strict=True) if key in r]
        if len(vals) < 2:
            out[key] = {'n': len(vals), 'judgement': '点が足りず判定不能'}
            continue
        out[key] = {
            'n': len(vals), 'first': vals[0], 'last': vals[-1],
            'delta': vals[-1] - vals[0], 'r_vs_step': pearson(sub, vals),
        }

    tf = out.get('tf_acc/gap', {})
    gen = out.get('gen_acc/gap', {})
    if not isinstance(tf, dict) or not isinstance(gen, dict) \
            or 'delta' not in tf or 'delta' not in gen:
        out['candidate_iv'] = '判定不能（点が足りない）'
        return out
    tf_d, gen_d = tf['delta'], gen['delta']
    if gen_d < 0 and tf_d > 0:
        out['candidate_iv'] = ('支持: 自由生成ギャップは縮小・teacher forcingは拡大'
                               'で、BLEUギャップの向きと一致する')
    elif gen_d > 0 and tf_d > 0:
        out['candidate_iv'] = ('否定: 自由生成ギャップも teacher forcing と同じく'
                               '拡大しており、regimeでは説明できない')
    else:
        out['candidate_iv'] = 'どちらとも言えない（未解明のまま残す）'
    return out


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description='候補(iv): regimeの違いで崩壊ギャップの向きが変わるか')
    p.add_argument('--checkpoint', type=str, default=None,
                   help='本走 checkpoint を1点だけ測る（未指定ならローカル学習プローブ）')
    p.add_argument('--data-dir', type=str, default='./data-wmt14')
    p.add_argument('--vocab-size', type=int, default=1600,
                   help='ローカルプローブで使う語彙（キャッシュ済みのもの）')
    p.add_argument('--seq-len', type=int, default=16)
    p.add_argument('--d-channels', type=int, default=192)
    p.add_argument('--m-mult', type=int, default=2,
                   help='出力ヘッド最終層 = vocab_size × これ')
    p.add_argument('-bs', '--batch-size', type=int, default=64)
    p.add_argument('-ni', '--num-iterations', type=int, default=2000)
    p.add_argument('-ef', '--eval-freq', type=int, default=400)
    p.add_argument('--lr', type=float, default=0.05)
    p.add_argument('--max-samples', type=int, default=200,
                   help='自由生成accとBLEUのサンプル数（ローカルプローブ用に既定を絞る。'
                        '本走比較時は1000にすること）')
    p.add_argument('--eval-chunk-size', type=int, default=8)
    p.add_argument('--num-test-samples', type=int, default=512)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', type=str, default=None)
    p.add_argument('--out', type=str, default='results/tsmix1d/regime_probe.json')
    p.add_argument('--no-wandb', action='store_true')
    return p


def main() -> None:
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except (AttributeError, OSError):
        pass

    args = build_parser().parse_args()
    if args.device is None:
        if torch.cuda.is_available():
            args.device = 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            args.device = 'mps'
        else:
            args.device = 'cpu'
    print(f'device={args.device}')
    print(f'JST: {datetime.now(JST).strftime("%Y-%m-%d %H:%M:%S")}')

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    from data.wmt14 import download_and_tokenize
    train_set, test_set, tokenizer = download_and_tokenize(
        vocab_size=args.vocab_size, seq_len=args.seq_len, data_dir=args.data_dir)
    vocab = tokenizer.vocab_size

    test_subset = torch.utils.data.Subset(
        test_set, range(min(args.num_test_samples, len(test_set))))
    train_loader = DataLoader(train_set, batch_size=args.batch_size,
                              shuffle=True, drop_last=True)
    test_loader = DataLoader(test_subset, batch_size=args.batch_size,
                             shuffle=False, drop_last=True)

    config = replace(
        LogicSeqMixerConfig.tiny(),
        vocab_size=vocab, d_emb=1024, seq_len=args.seq_len,
        d_channels=args.d_channels, channel_hidden=args.d_channels * 2,
        n_pos_bits=64, n_enc_blocks=2, n_dec_blocks=2,
        token_kernels=(3, 8, 16), context_mode='gather',
        m_layer_sizes=(vocab * args.m_mult,), tau=2.0,
    )
    model = LogicSeqMixer(config, device=args.device).to(args.device)
    print(f'params={model.count_parameters()["total"]:,} '
          f'gates={model.count_gates()["total"]:,}')

    if args.checkpoint:
        state = torch.load(args.checkpoint, map_location=args.device,
                           weights_only=False)
        model.load_state_dict(state['model'] if 'model' in state else state)
        print(f'checkpoint を読み込み: {args.checkpoint}')

    use_wandb = False
    if not args.no_wandb:
        use_wandb = init_wandb(
            run_name=f'tsmix1d_regime_probe_v{vocab}_d{args.d_channels}',
            config_dict={**vars(args), **asdict(config)},
            model=model, device=args.device,
            tags=['tsmix-1d', 'regime-probe']
            + normalize_tags(model='logic_seq_mixer', dataset='wmt14',
                             seed=args.seed, series='TSMIX-1'),
            group='tsmix-1', job_type='eval',
            notes='候補(iv)検証: teacher forcing acc / 自由生成 acc / BLEU の'
                  '崩壊ギャップが同じ向きに動くか',
            hypothesis='自由生成accギャップはBLEUギャップと同方向(縮小)に動き、'
                       'teacher forcing accギャップ(拡大)とは逆になる',
        )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=0.001, betas=(0.9, 0.999), eps=1e-8)

    rows: list[dict] = []
    start = time.time()

    def snapshot(step: int) -> None:
        row = {'step': step}
        row.update(measure(model, test_loader, tokenizer, args.device,
                           args.max_samples, args.eval_chunk_size))
        rows.append(row)
        elapsed = time.time() - start
        eta = datetime.now(JST) + timedelta(
            seconds=elapsed / max(step, 1) * (args.num_iterations - step))
        print(f'[step {step:,}/{args.num_iterations:,}] '
              f'tf_gap={row["tf_acc/gap"]:+.4f} gen_gap={row["gen_acc/gap"]:+.4f} '
              + (f'bleu_gap={row["bleu/gap"]:+.3f} ' if 'bleu/gap' in row else '')
              + f'[{timedelta(seconds=int(elapsed))}, 終了予定 '
                f'{eta.strftime("%H:%M:%S")}]', flush=True)
        if use_wandb:
            log_metrics(row, step=step)

    snapshot(0)
    if not args.checkpoint:
        for step, (src, tgt) in enumerate(
                load_n(train_loader, args.num_iterations), start=1):
            src = src.to(args.device)
            tgt = tgt.to(args.device)
            src_batch, tgt_full = prepare_batch(src, tgt)
            model.train()
            logits, emb_loss = model(src_batch, tgt_full[:, :-1])
            loss = F.cross_entropy(
                logits.reshape(-1, vocab), tgt_full[:, 1:].reshape(-1),
                label_smoothing=0.1, ignore_index=TOKEN_PAD,
            ) + get_emb_reg_weight(step) * emb_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if step % args.eval_freq == 0 or step == args.num_iterations:
                snapshot(step)

    result = {'rows': rows, 'verdict': verdict(rows),
              'args': vars(args), 'config': asdict(config)}
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as fh:
        json.dump(result, fh, ensure_ascii=False, indent=2)

    print('\n=== 事前登録した判定基準による結論 ===')
    for key, val in result['verdict'].items():
        print(f'  {key}: {val}')
    print(f'保存: {os.path.abspath(args.out)}')

    if use_wandb:
        import wandb
        artifact = wandb.Artifact('tsmix1d_regime_probe', type='result')
        artifact.add_file(args.out)
        wandb.log_artifact(artifact)
        wandb_finish()


if __name__ == '__main__':
    main()
