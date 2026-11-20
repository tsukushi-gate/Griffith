"""
アライメント行列の学習過程の可視化

Bahdanau 式のアライメント図(縦=出力トークン位置・横=入力トークン位置・
明るさ=重み)と**同型**の図を、LogicSeqMixer の2種類の経路について作る。

可視化対象:
    (a) 純 Mixer アーム(attention 不使用 = `--cross-attention none`)の
        **静的 Attention マスク**。本モデルの token 混合は明示的な T×S 重み行列を
        持たない(窓に対する LUT スタック)ので、**存在しない重み行列を描かない**。
        代わりに autograd で **実効的な影響行列(感度マップ)**

            sens[t, s] = mean_batch  Σ_d | ∂ logit[t, y_t] / ∂ src_emb[s, d] |

        を実測して描く。**これは重み行列そのものではなく勾配感度である**。
        `context_mode='gather'` では decoder が enc_states[t] を見る構造なので
        対角が構造的に立つはずで、`broadcast` との差が同じ手法で比較できる。

    (b) cross アーム(`--cross-attention logic_single_head`)の学習ヘッドの
        attention 重み。`LogicAttention.last_attention`((B,H,S_q,S_k))を
        forward 後に取得してそのまま描く。入力依存なので**検証サンプル1文を固定**し、
        実トークンIDを軸ラベルにする。soft(`model.train()`)と
        hard(`model.eval()`、argmax one-hot)の両方を出す。

定量判定(図の印象論で終わらせないための必須指標):
    正解アライメント行列(copy=対角 / reverse=反対角 / delayed_recall=先頭列)との
    Pearson 相関 `align/corr_target` と、行ごとの argmax 一致率
    `align/argmax_acc` を step ごとに wandb へログする。対照として恒等(対角)・
    反対角・正解の3本すべての相関を同時に記録するので、reverse で
    「対角 → 反対角」に組み替わる過程が数値でも追える。

    **収束の判定基準(実行前に宣言。実測に合わせて緩めない)**:
        `argmax_acc == 1.0` かつ `corr_target >= 0.5` を満たしたとき「収束」。

使用例:
    # reverse(本命) × cross アーム
    uv run python experiments/train_tsmix1b_alignment.py \\
        --task reverse --cross-attention logic_single_head -ni 3000

    # reverse × 純 Mixer アーム(gather / broadcast)
    uv run python experiments/train_tsmix1b_alignment.py \\
        --task reverse --context-mode gather -ni 3000
"""

import argparse
import json
import os
import random
import sys
import time
from dataclasses import asdict
from datetime import datetime, timedelta, timezone

JST = timezone(timedelta(hours=9))

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from data import TASK_DATASETS, TOKEN_EOS
from env_utils import load_dotenv
from train_logic_seq_mixer import evaluate, prepare_batch, train_step
from wandb_utils import (
    LossLogPacer,
    add_experiment_args,
    count_logic_gates,
    init_wandb,
    log_artifact,
    log_images,
    log_metrics,
    normalize_tags,
    progress_metrics,
)
from wandb_utils import (
    finish as wandb_finish,
)

from griffith_logic.models import LogicSeqMixer, LogicSeqMixerConfig

load_dotenv()

# 収束判定のしきい値(実行前に宣言。実測に合わせて動かさない)
CONVERGE_ARGMAX_ACC = 1.0
CONVERGE_CORR = 0.5


# ============================================================
# 正解アライメント
# ============================================================

def target_alignment(task: str, n_out: int, n_src: int) -> np.ndarray:
    """正解アライメント行列 (n_out, n_src) を返す。無効行は全0。

    decoder の出力位置 t は tgt_full[1:] = [tgt..., EOS] の t 番目を予測する。
    EOS 位置には対応する src 位置が無いので**行を無効**(全0)にし、指標の
    計算から外す(存在しない対応を正解扱いしないため)。

    task:
        'copy'           — 対角(出力 t ← src t)
        'reverse'        — 反対角(出力 t ← src S-1-t)
        'delayed_recall' — 先頭列(出力 0 ← src 0)。tgt 長が1なので
                           「シフト対角」は1行に退化する(実態に即して1行だけ立てる)
    """
    mat = np.zeros((n_out, n_src), dtype=np.float64)
    if task == 'copy':
        for t in range(min(n_out, n_src)):
            mat[t, t] = 1.0
    elif task == 'reverse':
        for t in range(min(n_out, n_src)):
            mat[t, n_src - 1 - t] = 1.0
    elif task == 'delayed_recall':
        mat[0, 0] = 1.0
    else:
        raise ValueError(f'正解アライメントが定義されていないタスク: {task}')
    return mat


def canonical_patterns(n_out: int, n_src: int) -> dict[str, np.ndarray]:
    """対照用の標準パターン(対角・反対角)。行の有効性は呼び出し側で揃える。"""
    diag = np.zeros((n_out, n_src))
    anti = np.zeros((n_out, n_src))
    for t in range(min(n_out, n_src)):
        diag[t, t] = 1.0
        anti[t, n_src - 1 - t] = 1.0
    return {'diag': diag, 'anti_diag': anti}


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    """フラット化した2行列の Pearson 相関(定数ベクトルなら 0.0)。"""
    a = a.astype(np.float64).ravel()
    b = b.astype(np.float64).ravel()
    a = a - a.mean()
    b = b - b.mean()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom < 1e-12:
        return 0.0
    return float(np.dot(a, b) / denom)


def row_normalize(mat: np.ndarray) -> np.ndarray:
    """各行の総和を1にする(全0行はそのまま)。行ごとのスケール差を除く。"""
    s = mat.sum(axis=1, keepdims=True)
    return np.divide(mat, s, out=np.zeros_like(mat), where=s > 0)


def alignment_metrics(mat: np.ndarray, task: str, valid_rows: list[int]) -> dict:
    """測定行列と正解アライメントの一致度。

    Args:
        mat: (n_out, n_src) の実測行列(感度マップ or attention重み)
        task: 正解アライメントを決めるタスク名
        valid_rows: 指標に使う行(EOS 行など対応の無い行を除いたもの)

    Returns:
        corr_target / corr_diag / corr_anti_diag / argmax_acc / n_rows
    """
    n_out, n_src = mat.shape
    tgt = target_alignment(task, n_out, n_src)
    pats = canonical_patterns(n_out, n_src)

    rows = np.array(valid_rows, dtype=int)
    m = row_normalize(mat.astype(np.float64))[rows]
    t = tgt[rows]

    argmax_hit = (m.argmax(axis=1) == t.argmax(axis=1))
    return {
        'corr_target': _pearson(m, t),
        'corr_diag': _pearson(m, pats['diag'][rows]),
        'corr_anti_diag': _pearson(m, pats['anti_diag'][rows]),
        'argmax_acc': float(argmax_hit.mean()),
        'n_rows': int(len(rows)),
    }


# ============================================================
# 測定(感度マップ / attention重み)
# ============================================================

def sensitivity_map(model, src, tgt_input, tgt_label) -> np.ndarray:
    """**実効的な影響行列**(感度マップ) (n_out, n_src) を autograd で実測する。

    sens[t, s] = mean_batch Σ_d |∂ logit[t, y_t] / ∂ src_emb[s, d]|

    **重み行列そのものではない**。本モデルの token 混合は T×S の重み行列を
    持たないため、「出力位置 t の正解トークンのロジットが src 位置 s の
    埋め込みビットにどれだけ依存しているか」を勾配で測って代用する。
    soft モード(`model.train()`)で計算する。
    """
    was_training = model.training
    model.train()

    src_emb = model.src_embedding(src).detach().clone().requires_grad_(True)
    enc_states = model.encoder(src_emb)
    tgt_emb = model.tgt_embedding(tgt_input)
    logits = model.decoder(tgt_emb, enc_states)     # (B, T, V)

    n_out = logits.shape[1]
    n_src = src.shape[1]
    out = np.zeros((n_out, n_src), dtype=np.float64)
    for t in range(n_out):
        scalar = logits[:, t, :].gather(1, tgt_label[:, t:t + 1]).sum()
        grad, = torch.autograd.grad(
            scalar, src_emb, retain_graph=(t < n_out - 1))
        out[t] = grad.abs().sum(dim=-1).mean(dim=0).detach().float().cpu().numpy()

    model.train(mode=was_training)
    return out


@torch.no_grad()
def attention_map(model, src, tgt_input, use_soft: bool, sample: int = 0):
    """cross attention の重み (n_out, n_src) を固定サンプルから取得する。

    `LogicAttention.attend` が forward のたびに `_last_attn` を更新するので、
    forward 直後に `last_attention` を読む。use_soft=False(model.eval())では
    decoder が `hard=True` を渡すため argmax の one-hot になる。
    """
    if model.decoder.cross_attn is None:
        return None
    was_training = model.training
    model.train(mode=use_soft)
    model(src, tgt_input)
    attn = model.decoder.cross_attn.last_attention   # (B, H, S_q, S_k)
    model.train(mode=was_training)
    if attn is None:
        return None
    return attn[sample, 0].detach().float().cpu().numpy()


# ============================================================
# 描画
# ============================================================

_CMAP = 'viridis'


def _tick_labels(ids: list[int]) -> list[str]:
    """トークンIDの軸ラベル(特殊トークンは記号化)。"""
    names = {0: 'PAD', 1: 'BOS', 2: 'EOS'}
    return [names.get(i, str(i)) for i in ids]


def draw_heatmap(ax, mat: np.ndarray, *, row_labels=None, col_labels=None,
                 title: str = '', normalize: bool = True):
    """1枚のアライメント風ヒートマップを ax へ描く(縦=出力・横=入力)。"""
    data = row_normalize(mat) if normalize else mat
    im = ax.imshow(data, cmap=_CMAP, aspect='auto', interpolation='nearest',
                   vmin=0.0)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel('input position (src)', fontsize=8)
    ax.set_ylabel('output position (tgt)', fontsize=8)
    if col_labels is not None:
        ax.set_xticks(range(len(col_labels)))
        ax.set_xticklabels(col_labels, fontsize=6, rotation=90)
    if row_labels is not None:
        ax.set_yticks(range(len(row_labels)))
        ax.set_yticklabels(row_labels, fontsize=6)
    return im


def save_single(path: str, mat: np.ndarray, title: str,
                row_labels=None, col_labels=None) -> str:
    fig, ax = plt.subplots(figsize=(4.2, 4.0), dpi=140)
    im = draw_heatmap(ax, mat, row_labels=row_labels, col_labels=col_labels,
                      title=title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def save_panel(path: str, frames: list[tuple[int, np.ndarray]], suptitle: str,
               caption: str, row_labels=None, col_labels=None) -> str:
    """step 0 → 中間 → 収束 の3枚組パネル。"""
    n = len(frames)
    fig, axes = plt.subplots(1, n, figsize=(3.6 * n, 4.3), dpi=140)
    if n == 1:
        axes = [axes]
    for ax, (step, mat) in zip(axes, frames, strict=True):
        im = draw_heatmap(ax, mat, row_labels=row_labels, col_labels=col_labels,
                          title=f'step {step:,}')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(suptitle, fontsize=11)
    fig.text(0.5, 0.005, caption, ha='center', fontsize=7, wrap=True)
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))
    fig.savefig(path)
    plt.close(fig)
    return path


def save_gif(path: str, png_paths: list[str], duration_ms: int = 500) -> str | None:
    """PNG 群から GIF を作る(Pillow のみ。imageio 不要)。"""
    if len(png_paths) < 2:
        return None
    from PIL import Image
    frames = [Image.open(p).convert('P', palette=Image.ADAPTIVE)
              for p in png_paths]
    frames[0].save(path, save_all=True, append_images=frames[1:],
                   duration=duration_ms, loop=0)
    return path


# ============================================================
# 設定
# ============================================================

def build_config(args) -> LogicSeqMixerConfig:
    """tiny() を土台に、CLI で指定された寸法・アーム設定を反映する。"""
    config = LogicSeqMixerConfig.tiny()
    config.vocab_size = args.vocab_size
    config.seq_len = args.seq_len
    config.d_emb = args.d_emb
    config.d_channels = args.d_channels
    config.channel_hidden = args.channel_hidden
    config.n_pos_bits = args.n_pos_bits
    config.token_kernels = tuple(int(x) for x in args.token_kernels.split(','))
    config.m_layer_sizes = tuple(int(x) for x in args.m_sizes.split(','))
    config.cross_attention = args.cross_attention
    config.context_mode = args.context_mode
    config.__post_init__()
    return config


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description='アライメント行列の学習過程の可視化')
    p.add_argument('--task', type=str, default='reverse',
                   choices=['copy', 'reverse', 'delayed_recall'],
                   help='タスク (default: reverse)')
    p.add_argument('--num-samples', type=int, default=20000)
    p.add_argument('--num-test-samples', type=int, default=1024)

    p.add_argument('--vocab-size', type=int, default=32)
    p.add_argument('--seq-len', type=int, default=16)
    p.add_argument('--d-emb', type=int, default=64)
    p.add_argument('--d-channels', type=int, default=72)
    p.add_argument('--channel-hidden', type=int, default=144)
    p.add_argument('--n-pos-bits', type=int, default=16)
    p.add_argument('--token-kernels', type=str, default='3,4,8')
    p.add_argument('--m-sizes', type=str, default='128,96')
    p.add_argument('--context-mode', type=str, default='broadcast',
                   choices=['broadcast', 'gather', 'concat_all'])
    p.add_argument('--cross-attention', type=str, default='none',
                   choices=['none', 'logic_single_head'])

    p.add_argument('--lr', type=float, default=0.05)
    p.add_argument('--weight-decay', type=float, default=0.001)
    p.add_argument('-bs', '--batch-size', type=int, default=64)
    p.add_argument('-ni', '--num-iterations', type=int, default=3000)
    p.add_argument('--label-smoothing', type=float, default=0.1)
    p.add_argument('--align-every', type=int, default=500,
                   help='アライメント図を作る step 間隔 (default: 500)')
    p.add_argument('--align-batch', type=int, default=8,
                   help='感度マップを平均する検証サンプル数 (default: 8)')
    p.add_argument('--eval-freq', type=int, default=500)
    p.add_argument('--eval-chunk-size', type=int, default=8)

    p.add_argument('--out-dir', type=str, default='results/tsmix1b')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', type=str, default=None,
                   help='未指定なら自動検出 (cuda→mps→cpu)')
    p.add_argument('--no-wandb', action='store_true')
    p.add_argument('--job-type', type=str, default='ablation',
                   choices=['train', 'ablation', 'sweep', 'eval', 'smoke'])
    add_experiment_args(p)
    return p


def arm_label(config: LogicSeqMixerConfig) -> str:
    """アーム名(ファイル名・図タイトル用)。"""
    if config.cross_attention == 'logic_single_head':
        return 'cross'
    return config.context_mode


# ============================================================
# メイン
# ============================================================

def main():
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except (AttributeError, OSError):
        pass

    parser = build_parser()
    args = parser.parse_args()

    if args.device is None:
        if torch.cuda.is_available():
            args.device = 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            args.device = 'mps'
        else:
            args.device = 'cpu'
    print(f'device={args.device}')

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    config = build_config(args)
    arm = arm_label(config)
    tag = f'{args.task}_{arm}'
    out_dir = os.path.join(args.out_dir, tag)
    os.makedirs(out_dir, exist_ok=True)

    print(f'\n=== アライメント可視化 ({tag}) ===')
    print(f'task={args.task} arm={arm} context_mode={config.context_mode} '
          f'cross_attention={config.cross_attention}')
    print(f'vocab={config.vocab_size} seq_len={config.seq_len} '
          f'd_emb={config.d_emb} d_channels={config.d_channels}')
    print(f'出力先: {os.path.abspath(out_dir)}')

    dataset_cls = TASK_DATASETS[args.task]
    train_set = dataset_cls(config.vocab_size, config.seq_len, args.num_samples)
    test_set = dataset_cls(config.vocab_size, config.seq_len, args.num_test_samples)
    train_loader = DataLoader(train_set, batch_size=args.batch_size,
                              shuffle=True, drop_last=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size,
                             shuffle=False, drop_last=True)

    model = LogicSeqMixer(config, device=args.device).to(args.device)
    gates = model.count_gates()
    print(f'総ロジックゲート数: {gates["total"]:,}')

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay,
                                  betas=(0.9, 0.999), eps=1e-8)

    # 可視化に使う固定検証バッチ(shuffle=False なので毎回同一)
    fixed_src, fixed_tgt = next(iter(test_loader))
    fixed_src = fixed_src[:args.align_batch].to(args.device)
    fixed_tgt = fixed_tgt[:args.align_batch].to(args.device)
    _, fixed_tgt_full = prepare_batch(fixed_src, fixed_tgt)
    fixed_tgt_input = fixed_tgt_full[:, :-1]
    fixed_tgt_label = fixed_tgt_full[:, 1:]

    n_out = fixed_tgt_input.shape[1]
    n_src = fixed_src.shape[1]
    # EOS 行(対応する src 位置が無い)を指標から外す
    label0 = fixed_tgt_label[0].tolist()
    valid_rows = [t for t in range(n_out) if label0[t] != TOKEN_EOS]
    if args.task == 'delayed_recall':
        valid_rows = [0]
    print(f'アライメント行列: {n_out}(out) x {n_src}(src)  '
          f'指標対象行={valid_rows}')

    col_labels = _tick_labels(fixed_src[0].tolist())
    row_labels = _tick_labels(label0)

    # wandb
    run_name = (f'tsmix1b_align_{args.task}_{arm}_v{config.vocab_size}'
                f'_L{config.seq_len}_ni{args.num_iterations}')
    use_wandb = False
    if not args.no_wandb:
        use_wandb = init_wandb(
            run_name=run_name,
            config_dict={**vars(args), **asdict(config)},
            model=model,
            device=args.device,
            tags=(['logic_seq_mixer', 'tsmix-1b', 'alignment', args.task,
                   f'arm:{arm}']
                  + normalize_tags(model='logic_seq_mixer', dataset=args.task,
                                   seed=args.seed, series='TSMIX-1')),
            group='tsmix-1',
            job_type=args.job_type,
            notes=args.notes,
            hypothesis=args.hypothesis,
        )
        if use_wandb:
            log_metrics(count_logic_gates(model), step=0)

    history: list[dict] = []
    sens_pngs: list[str] = []
    sens_frames: list[tuple[int, np.ndarray]] = []
    attn_soft_frames: list[tuple[int, np.ndarray]] = []
    attn_hard_frames: list[tuple[int, np.ndarray]] = []

    def snapshot(step: int) -> dict:
        """この step のアライメント行列を測定・保存し、指標を返す。"""
        record: dict = {'step': step}
        images, captions = [], []

        sens = sensitivity_map(model, fixed_src, fixed_tgt_input, fixed_tgt_label)
        sens_frames.append((step, sens))
        m = alignment_metrics(sens, args.task, valid_rows)
        record['sens'] = m
        path = os.path.join(out_dir, f'sens_step{step:06d}.png')
        save_single(
            path, sens,
            f'{args.task} / {arm} / sensitivity (step {step:,})\n'
            f'corr_target={m["corr_target"]:.3f} argmax={m["argmax_acc"]:.3f}',
            row_labels=row_labels, col_labels=col_labels)
        sens_pngs.append(path)
        images.append(path)
        captions.append(f'感度マップ(勾配|∂logit/∂src_emb|・重み行列ではない) step={step}')

        if config.cross_attention == 'logic_single_head':
            for use_soft, store in ((True, attn_soft_frames),
                                    (False, attn_hard_frames)):
                attn = attention_map(model, fixed_src, fixed_tgt_input,
                                     use_soft=use_soft)
                mode = 'soft' if use_soft else 'hard'
                store.append((step, attn))
                am = alignment_metrics(attn, args.task, valid_rows)
                record[f'attn_{mode}'] = am
                ap = os.path.join(out_dir, f'attn_{mode}_step{step:06d}.png')
                save_single(
                    ap, attn,
                    f'{args.task} / cross attention ({mode}) step {step:,}\n'
                    f'corr_target={am["corr_target"]:.3f} '
                    f'argmax={am["argmax_acc"]:.3f}',
                    row_labels=row_labels, col_labels=col_labels)
                images.append(ap)
                captions.append(f'cross attention重み({mode}) step={step}')

        if use_wandb:
            from PIL import Image
            log_images(f'align/{tag}', [Image.open(p) for p in images],
                       captions=captions, step=max(step, 1))
            metrics = {}
            for kind, mm in record.items():
                if kind == 'step':
                    continue
                for k, v in mm.items():
                    metrics[f'align/{kind}/{k}'] = v
            # 主要指標のショートカット(感度マップ基準)
            metrics['align/corr_target'] = record['sens']['corr_target']
            metrics['align/argmax_acc'] = record['sens']['argmax_acc']
            log_metrics(metrics, step=max(step, 1))

        line = ' | '.join(
            f'{kind}: corr_t={mm["corr_target"]:+.3f} '
            f'corr_diag={mm["corr_diag"]:+.3f} '
            f'corr_anti={mm["corr_anti_diag"]:+.3f} '
            f'argmax={mm["argmax_acc"]:.3f}'
            for kind, mm in record.items() if kind != 'step')
        print(f'[align step {step:,}] {line}', flush=True)
        history.append(record)
        return record

    # step 0(ランダム初期化)の対照を必ず取る
    snapshot(0)

    print(f'\n訓練開始: {args.num_iterations:,} イテレーション')
    train_start = time.time()
    pacer = LossLogPacer()
    step = 0
    acc_soft = acc_hard = float('nan')

    while step < args.num_iterations:
        for src, tgt in train_loader:
            if step >= args.num_iterations:
                break
            src = src.to(args.device)
            tgt = tgt.to(args.device)
            src_batch, tgt_full = prepare_batch(src, tgt)
            total_loss, main_loss, emb_loss = train_step(
                model, src_batch, tgt_full, optimizer, step,
                label_smoothing=args.label_smoothing)
            step += 1

            elapsed = time.time() - train_start
            is_last = step == args.num_iterations

            # SKILL.md §11: train loss は壁時計で最低5分に1回
            if pacer.should_log(step, elapsed, is_last=is_last):
                if pacer.resolved_now:
                    print(pacer.describe(), flush=True)
                if use_wandb:
                    log_metrics({
                        'iteration': step,
                        'loss/total': total_loss,
                        'loss/main': main_loss,
                        'loss/emb_reg': emb_loss,
                        'lr': optimizer.param_groups[0]['lr'],
                        **progress_metrics(step, step, args.num_iterations,
                                           elapsed),
                    }, step=step)

            if step % args.eval_freq == 0 or is_last:
                acc_soft = evaluate(model, test_loader, args.device,
                                    use_soft=True,
                                    chunk_size=args.eval_chunk_size)
                acc_hard = evaluate(model, test_loader, args.device,
                                    use_soft=False,
                                    chunk_size=args.eval_chunk_size)
                sec_per_iter = elapsed / max(step, 1)
                remaining = sec_per_iter * (args.num_iterations - step)
                eta = (datetime.now(JST) + timedelta(seconds=remaining))
                print(f'[iter {step:,}/{args.num_iterations:,}] '
                      f'loss={total_loss:.4f} acc(soft)={acc_soft:.4f} '
                      f'acc(hard)={acc_hard:.4f} '
                      f'[{str(timedelta(seconds=int(elapsed)))}<'
                      f'{str(timedelta(seconds=int(remaining)))}, '
                      f'{step / max(elapsed, 1e-9):.1f}it/s, '
                      f'終了予定 {eta.strftime("%H:%M:%S")}]', flush=True)
                if use_wandb:
                    log_metrics({'acc/soft': acc_soft, 'acc/hard': acc_hard,
                                 'iteration': step}, step=step)

            if step % args.align_every == 0 or is_last:
                snapshot(step)

    total_time = str(timedelta(seconds=int(time.time() - train_start)))
    print(f'\n訓練完了 ({total_time})  acc(soft)={acc_soft:.4f} '
          f'acc(hard)={acc_hard:.4f}')

    # ---- 3枚組パネル(step 0 → 中間 → 最終)と GIF ----
    def pick3(frames):
        if len(frames) <= 3:
            return frames
        return [frames[0], frames[len(frames) // 2], frames[-1]]

    # 図中テキストは日本語フォント未導入環境で豆腐になるため英語で書く
    # (日本語の説明はレポート本文側に置く)
    sens_caption = (
        'Sensitivity map = mean_batch sum_d |d logit[t, y_t] / d src_emb[s, d]|,'
        ' measured by autograd. This is NOT the mixing weight matrix itself:'
        ' token mixing here is a LUT stack over a sliding window and has no'
        ' explicit T x S weight matrix. Rows are normalized to sum to 1.')
    attn_caption = (
        'Actual cross-attention weights from LogicAttention.last_attention'
        ' (one fixed validation sample). soft = model.train(),'
        ' hard = model.eval() argmax one-hot.')

    artifacts: list[str] = []
    panel_sens = os.path.join(out_dir, 'panel_sensitivity.png')
    save_panel(panel_sens, pick3(sens_frames),
               f'{args.task} / {arm}: static attention mask '
               f'(gradient sensitivity) over training',
               sens_caption, row_labels=row_labels, col_labels=col_labels)
    artifacts.append(panel_sens)

    if attn_soft_frames:
        panel_soft = os.path.join(out_dir, 'panel_attention_soft.png')
        save_panel(panel_soft, pick3(attn_soft_frames),
                   f'{args.task} / cross attention (soft) over training',
                   attn_caption, row_labels=row_labels, col_labels=col_labels)
        artifacts.append(panel_soft)
        panel_hard = os.path.join(out_dir, 'panel_attention_hard.png')
        save_panel(panel_hard, pick3(attn_hard_frames),
                   f'{args.task} / cross attention (hard) over training',
                   attn_caption, row_labels=row_labels, col_labels=col_labels)
        artifacts.append(panel_hard)

    gif_path = save_gif(os.path.join(out_dir, 'sensitivity.gif'), sens_pngs)
    if gif_path:
        artifacts.append(gif_path)

    # ---- 収束判定(実行前に宣言したしきい値。緩めない) ----
    final = history[-1]
    verdicts = {}
    for kind, mm in final.items():
        if kind == 'step':
            continue
        verdicts[kind] = bool(
            mm['argmax_acc'] >= CONVERGE_ARGMAX_ACC
            and mm['corr_target'] >= CONVERGE_CORR)

    summary = {
        'task': args.task,
        'arm': arm,
        'context_mode': config.context_mode,
        'cross_attention': config.cross_attention,
        'gates': gates,
        'seq_len': config.seq_len,
        'vocab_size': config.vocab_size,
        'num_iterations': args.num_iterations,
        'device': args.device,
        'acc_soft': acc_soft,
        'acc_hard': acc_hard,
        'converge_criteria': {'argmax_acc': CONVERGE_ARGMAX_ACC,
                              'corr_target': CONVERGE_CORR},
        'converged': verdicts,
        'history': history,
        'valid_rows': valid_rows,
        'figures': [os.path.abspath(p) for p in artifacts],
    }
    json_path = os.path.join(out_dir, 'metrics.json')
    with open(json_path, 'w') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    artifacts.append(json_path)
    print(f'\n収束判定 (argmax_acc>={CONVERGE_ARGMAX_ACC} かつ '
          f'corr_target>={CONVERGE_CORR}): {verdicts}')
    print(f'metrics: {os.path.abspath(json_path)}')

    if use_wandb:
        log_metrics({
            'align/final_corr_target': final['sens']['corr_target'],
            'align/final_argmax_acc': final['sens']['argmax_acc'],
            'acc/soft': acc_soft, 'acc/hard': acc_hard,
        }, step=args.num_iterations)
        log_artifact(artifacts, f'tsmix1b-align-{tag}', type='result',
                     aliases=['latest'],
                     metadata={k: v for k, v in summary.items()
                               if k != 'history'})
        wandb_finish()


if __name__ == '__main__':
    main()
