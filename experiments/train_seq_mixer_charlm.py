"""
decoder-only Mixer 文字LM 訓練スクリプト(Tiny Shakespeare)

目的は **「Mixer(attention不使用の静的causal Toeplitz混合)で LLM 的な自己回帰生成が
成立するか」を最小コストで実証すること**であり、規模は追わない。

社内比較点:
    run (group=rddlgn_charlm, val_loss=4.7266, 1,954ゲート)。
    これは既存の実測値なので**再走せず**そのまま比較対象に使う
    (``docs/experiments-ledger.md`` の記録)。

同一プロトコルの保証:
    - データローダ(load_shakespeare)・評価(evaluate の hard/soft)・サンプル生成
      (sample_text)・引数・wandb・checkpoint の作法は
      ``experiments/train_rddlgn_charlm.py`` と**同一物をコピー**している
      (唯一の相違はモデルと group/tags)。同スクリプト自身も
      ``train_logic_gpt.py`` からの複製であり、この系列の慣行に従う。
    - bpc(bits per character)は ``val_loss / ln(2)`` で算出する
      (``griffith_logic.models.logic_seq_mixer_lm.bits_per_character``)。
      既存スクリプトの val_loss は「1文字あたりの nats」なのでこの式が正しい。
      社内比較点にも同じ式を適用する。

使用例:
    # スモーク(wandbオフライン)
    WANDB_MODE=offline uv run python experiments/train_seq_mixer_charlm.py \\
        --max-steps 10 --eval-batches 5 --run-dir _seqmixer_charlm_smoke

    # 本走(MPS、20000ステップ)
    uv run python experiments/train_seq_mixer_charlm.py \\
        --notes "decoder-only Mixer char-LM 最小実証" \\
        --hypothesis "..." --seed 0
"""

import argparse
import math
import os
import sys
import time
from dataclasses import asdict
from datetime import datetime, timedelta, timezone

JST = timezone(timedelta(hours=9))

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from checkpoint_utils import load_checkpoint, save_checkpoint, setup_checkpoint_dir
from env_utils import load_dotenv
from train_rddlgn_charlm import _DEFAULT_DATA_DIR, cycle, load_shakespeare
from wandb_utils import (
    add_experiment_args,
    init_wandb,
    log_checkpoint_artifact,
    log_metrics,
    normalize_tags,
)
from wandb_utils import finish as wandb_finish

from griffith_logic.models.logic_seq_mixer_lm import (
    LogicSeqMixerLM,
    LogicSeqMixerLMConfig,
    bits_per_character,
)

load_dotenv()

# ハートビート(訓練が生きていることを示すloss行)の最小間隔[秒]
_HEARTBEAT_SEC = 300


# ============================================================
# 評価・生成 — train_rddlgn_charlm.py と同一物をコピー
# ============================================================

@torch.no_grad()
def evaluate(
    model, loader, device, max_batches: int | None = 20, mode: str = 'hard',
) -> float:
    """val loss(1文字あたりのnats)を返す。max_batches=Noneで全val評価。

    mode='hard': `model.eval()` で評価(LogicEmbedding/LutLayer が hard{0,1})。
    mode='soft': `model.train()` で評価(sigmoid/softmax の連続緩和)。
        soft と hard の差が離散化ギャップ Δ_disc = loss_hard - loss_soft。

    注意: 部分評価(max_batches指定)はvalの先頭スライスに固定されるため、
    全val評価との数値比較には使えない。本走は --eval-batches 0(=全val)を使うこと。
    """
    was_training = model.training
    if mode == 'soft':
        model.train()
    else:
        model.eval()
    total_loss = 0.0
    total_tokens = 0
    for i, (x, y) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        vocab = logits.size(-1)
        loss = F.cross_entropy(
            logits.reshape(-1, vocab), y.reshape(-1), reduction='sum',
        )
        total_loss += loss.item()
        total_tokens += y.numel()
    model.train(mode=was_training)
    return total_loss / total_tokens if total_tokens > 0 else 0.0


@torch.no_grad()
def sample_text(model, meta: dict, device, n_chars: int = 200,
                seed_char: str = '\n') -> str:
    """学習済みモデルから n_chars 文字を greedy 生成しデコードする。"""
    was_training = model.training
    model.eval()
    stoi = meta['stoi']
    itos = meta['itos']
    start_id = stoi.get(seed_char, 0)
    idx = torch.tensor([[start_id]], dtype=torch.long, device=device)
    out = model.generate(idx, max_new_tokens=n_chars, greedy=True)
    ids = out[0].tolist()
    model.train(mode=was_training)
    return ''.join(itos[i] for i in ids)


@torch.no_grad()
def sample_text_multinomial(model, meta: dict, device, n_chars: int = 200,
                            seed_char: str = '\n') -> str:
    """多項サンプリング版(greedyは同じ文字の反復に陥りやすいので定性確認用)。"""
    was_training = model.training
    model.eval()
    stoi = meta['stoi']
    itos = meta['itos']
    start_id = stoi.get(seed_char, 0)
    idx = torch.tensor([[start_id]], dtype=torch.long, device=device)
    out = model.generate(idx, max_new_tokens=n_chars, greedy=False)
    ids = out[0].tolist()
    model.train(mode=was_training)
    return ''.join(itos[i] for i in ids)


def parse_int_list(s: str) -> list[int]:
    """カンマ区切りの整数リストをパース。例: '650,585' -> [650, 585]"""
    return [int(x.strip()) for x in s.split(',')]


def parse_int_tuple(s: str) -> tuple[int, ...]:
    """カンマ区切りの整数タプルをパース。例: '3,8,16' -> (3, 8, 16)"""
    return tuple(int(x.strip()) for x in s.split(','))


# ============================================================
# メイン
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='decoder-only Mixer 文字LM 訓練(/ Tiny Shakespeare)'
    )
    # 訓練ハイパラ(既定は train_rddlgn_charlm.py と同一プロトコル)
    parser.add_argument('--max-steps', type=int, default=20000)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--context-len', type=int, default=64)
    parser.add_argument('--lr', type=float, default=3.0e-4)
    parser.add_argument('--weight-decay', type=float, default=0.0)
    parser.add_argument('--eval-freq', type=int, default=500)
    parser.add_argument('--eval-batches', type=int, default=None,
                        help='評価バッチ数の上限(0=全val)。未指定は全val評価')
    parser.add_argument('--aux-lambda', type=float, default=0.0,
                        help='埋め込み二値化正則化の重み(既定0.0=純粋な対照群)')
    # モデル構成(既定は RDDLGN char-LM の1,954ゲートへ揃えた1,967ゲート設計)
    parser.add_argument('--d-emb', type=int, default=32)
    parser.add_argument('--d-channels', type=int, default=96)
    parser.add_argument('--channel-hidden', type=int, default=192)
    parser.add_argument('--n-blocks', type=int, default=2)
    parser.add_argument('--channel-depth', type=int, default=1)
    parser.add_argument('--token-kernels', type=str, default='3,8,16')
    parser.add_argument('--token-depth', type=int, default=2)
    parser.add_argument('--m-sizes', type=str, default='650,585',
                        help='出力ヘッド(最終層はvocab_sizeの倍数)')
    parser.add_argument('--n-pos-bits', type=int, default=16)
    parser.add_argument('--positional', type=str, default='learned_bits',
                        choices=['learned_bits', 'none'])
    parser.add_argument('--residual-mode', type=str, default='xor',
                        choices=['xor', 'none'])
    parser.add_argument('--gate', type=str, default='and', choices=['and', 'none'])
    parser.add_argument('--n-inputs', type=int, default=2, choices=[2, 4, 6])
    parser.add_argument('--parameterization', type=str, default='softmax',
                        choices=['softmax', 'independent', 'iwp', 'warp', 'raw'])
    parser.add_argument('--tau', type=float, default=1.0,
                        help='GroupSum温度 (default: 1.0)')
    # 環境・記録
    parser.add_argument('--data-dir', type=str, default=_DEFAULT_DATA_DIR)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', type=str, default=None,
                        help='デバイス (default: 自動検出 cuda→mps→cpu)')
    parser.add_argument('--no-wandb', action='store_true')
    parser.add_argument('--resume', type=str, default=None)
    add_experiment_args(parser)
    parser.add_argument('--run-dir', type=str, default=None)
    args = parser.parse_args()

    # デバイス自動検出(cuda → mps → cpu)
    if args.device is None:
        if torch.cuda.is_available():
            args.device = 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            args.device = 'mps'
        else:
            args.device = 'cpu'
    device = args.device
    print(f'device={device}')

    torch.manual_seed(args.seed)

    batch_size = args.batch_size
    context_len = args.context_len
    max_steps = args.max_steps
    eval_freq = args.eval_freq
    if args.eval_batches is not None:
        eval_batches = None if args.eval_batches == 0 else args.eval_batches
    else:
        eval_batches = None
    aux_lambda = args.aux_lambda

    # データ(train_rddlgn_charlm.py と同一のローダを import して使う)
    train_loader, val_loader, meta = load_shakespeare(
        batch_size=batch_size, context_len=context_len, data_dir=args.data_dir,
    )
    vocab_size = meta['vocab_size']
    print(f'\nデータ: Tiny Shakespeare  vocab_size={vocab_size} '
          f'context_len={context_len}')

    # モデル
    config = LogicSeqMixerLMConfig(
        vocab_size=vocab_size,
        d_emb=args.d_emb,
        context_len=context_len,
        d_channels=args.d_channels,
        channel_hidden=args.channel_hidden,
        n_blocks=args.n_blocks,
        channel_depth=args.channel_depth,
        token_kernels=parse_int_tuple(args.token_kernels),
        token_depth=args.token_depth,
        gate=args.gate,
        residual_mode=args.residual_mode,
        m_layer_sizes=parse_int_list(args.m_sizes),
        positional=args.positional,
        n_pos_bits=args.n_pos_bits,
        n_inputs=args.n_inputs,
        parameterization=args.parameterization,
        tau=args.tau,
    )
    model = LogicSeqMixerLM(config, device=device).to(device)
    gates = model.count_gates()
    counts = model.count_parameters()
    print('\ndecoder-only Mixer char-LM (/ attentionなし)')
    print(f'  {model.extra_repr()}')
    print('  ゲート数(LUTニューロン):')
    for name, cnt in gates.items():
        print(f'    {name}: {cnt:,}')
    print('  パラメータ数:')
    for name, cnt in counts.items():
        print(f'    {name}: {cnt:,}')
    print(f'  社内比較点 RDDLGN char-LM: 1,954ゲート / 33,344パラメータ '
          f'(diff={gates["total"] - 1954:+,}ゲート)')

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
    )

    # チェックポイント
    if args.run_dir:
        run_dir = os.path.join('checkpoints', args.run_dir)
        os.makedirs(run_dir, exist_ok=True)
        latest_path = os.path.join(run_dir, 'latest.pt')
        best_path = os.path.join(run_dir, 'best.pt')
    else:
        run_dir, latest_path, best_path = setup_checkpoint_dir('seq_mixer_charlm')

    start_step = 0
    best_val_loss = float('inf')
    resume_wandb_run_id = None
    if args.resume:
        print(f'チェックポイントを読み込み: {args.resume}')
        ckpt_info = load_checkpoint(args.resume, model, optimizer, device=device)
        start_step = ckpt_info['step']
        best_val_loss = ckpt_info['best_metric']
        resume_wandb_run_id = ckpt_info.get('metadata', {}).get('wandb_run_id')
        extra_state = ckpt_info.get('extra_state', {})
        if 'torch_rng' in extra_state:
            torch.set_rng_state(extra_state['torch_rng'])
        print(f'  ステップ {start_step} から再開 (best_val_loss={best_val_loss:.4f})')

    # wandb
    use_wandb = False
    run_name = (f'seq_mixer_charlm_dc{config.d_channels}'
                f'_nb{config.n_blocks}_s{max_steps}')
    if not args.no_wandb:
        wandb_config = {
            'model_cfg': asdict(config), 'batch_size': batch_size,
            'context_len': context_len, 'lr': args.lr,
            'weight_decay': args.weight_decay, 'max_steps': max_steps,
            'aux_lambda': aux_lambda, 'seed': args.seed,
            'gates': gates, 'params': counts,
            'baseline_run_id': None,
            'baseline_val_loss': 4.7266,
            'baseline_gates': 1954,
        }
        use_wandb = init_wandb(
            run_name=run_name,
            config_dict=wandb_config,
            model=model,
            device=device,
            tags=['tsmix-1e', 'seq-mixer', 'charlm', 'shakespeare', 'decoder-only']
            + normalize_tags(series='TSMIX-1e', model='logic_seq_mixer_lm',
                             dataset='shakespeare', seed=args.seed),
            group='tsmix-1',
            job_type='train',
            resume_run_id=resume_wandb_run_id,
            notes=args.notes,
            hypothesis=args.hypothesis,
        )

    wandb_run_id = None
    if use_wandb:
        try:
            import wandb
            wandb_run_id = wandb.run.id
        except Exception:
            pass

    ckpt_metadata = {'config': asdict(config), 'wandb_run_id': wandb_run_id}

    print(f'\n訓練開始: {max_steps}ステップ (評価={eval_freq}ステップごと、'
          f'開始ステップ={start_step + 1})  開始 '
          f'{datetime.now(JST).strftime("%Y-%m-%d %H:%M:%S")} JST')
    print('=' * 78)
    loss_fn = nn.CrossEntropyLoss()
    data_iter = cycle(train_loader)
    train_start = time.time()
    last_heartbeat = train_start
    model.train()

    for step in range(start_step + 1, max_steps + 1):
        x, y = next(data_iter)
        x = x.to(device)
        y = y.to(device)

        logits = model(x)  # (B,T,V)
        main_loss = loss_fn(logits.reshape(-1, vocab_size), y.reshape(-1))
        aux = model.pop_aux_loss()
        loss = main_loss + aux_lambda * aux

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        is_eval_step = (step % eval_freq == 0 or step == max_steps or step == 1)

        # ハートビート: 評価ステップでなくても5分に1回は必ずloss行を出す
        now = time.time()
        if not is_eval_step and now - last_heartbeat >= _HEARTBEAT_SEC:
            last_heartbeat = now
            elapsed = now - train_start
            sec_per_step = elapsed / max(1, step - start_step)
            remaining = sec_per_step * (max_steps - step)
            eta = datetime.now(JST) + timedelta(seconds=remaining)
            print(
                f'[heartbeat step {step:>6d}/{max_steps}] '
                f'train_loss={main_loss.item():.4f} '
                f'[{timedelta(seconds=int(elapsed))}<'
                f'{timedelta(seconds=int(remaining))}, '
                f'終了予定 {eta.strftime("%H:%M:%S")}]',
                flush=True,
            )

        if is_eval_step:
            last_heartbeat = time.time()
            val_loss = evaluate(
                model, val_loader, device, max_batches=eval_batches, mode='hard',
            )
            val_soft_loss = evaluate(
                model, val_loader, device, max_batches=eval_batches, mode='soft',
            )
            discretization_gap = val_loss - val_soft_loss
            val_bpc = bits_per_character(val_loss)
            val_soft_bpc = bits_per_character(val_soft_loss)
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss

            extra_state = {'torch_rng': torch.get_rng_state()}
            save_checkpoint(latest_path, model, optimizer, None, step,
                            best_val_loss, ckpt_metadata, extra_state=extra_state)
            if is_best:
                save_checkpoint(best_path, model, optimizer, None, step,
                                best_val_loss, ckpt_metadata,
                                extra_state=extra_state)
            if use_wandb:
                log_checkpoint_artifact(
                    latest_path, run_name, stage='eval',
                    metadata={'step': step, 'val_loss': val_loss},
                )

            elapsed = time.time() - train_start
            sec_per_step = elapsed / max(1, step - start_step)
            remaining = sec_per_step * (max_steps - step)
            eta = datetime.now(JST) + timedelta(seconds=remaining)
            aux_val = float(aux.detach()) if torch.is_tensor(aux) else float(aux)

            print(
                f'[step {step:>6d}/{max_steps}] '
                f'train_loss={main_loss.item():.4f} '
                f'val_loss={val_loss:.4f} '
                f'val_bpc={val_bpc:.4f} '
                f'val_soft={val_soft_loss:.4f} '
                f'gap={discretization_gap:.4f} '
                f'aux={aux_val:.4f}'
                f'{"  ★ BEST" if is_best else ""}'
                f'  [{timedelta(seconds=int(elapsed))}<'
                f'{timedelta(seconds=int(remaining))}, '
                f'終了予定 {eta.strftime("%H:%M:%S")}]',
                flush=True,
            )
            if use_wandb:
                log_metrics({
                    'step': step,
                    'train_loss': main_loss.item(),
                    'train_bpc': bits_per_character(main_loss.item()),
                    'val_loss': val_loss,
                    'val_bpc': val_bpc,
                    'val_soft_loss': val_soft_loss,
                    'val_soft_bpc': val_soft_bpc,
                    'discretization_gap': discretization_gap,
                    'best_val_loss': best_val_loss,
                    'best_val_bpc': bits_per_character(best_val_loss),
                    'aux_loss': aux_val,
                })

    total_time = str(timedelta(seconds=int(time.time() - train_start)))
    print('=' * 78)
    print(f'訓練完了: best_val_loss={best_val_loss:.4f} '
          f'(bpc={bits_per_character(best_val_loss):.4f})  総時間: {total_time}')
    print(f' 社内比較点 (RDDLGN char-LM): val_loss=4.7266 '
          f'(bpc={bits_per_character(4.7266):.4f})  ※既存実測値・再走していない')
    print(f'  一様分布の理論値: val_loss={math.log(vocab_size):.4f} '
          f'(bpc={bits_per_character(math.log(vocab_size)):.4f})')

    # サンプル生成(greedy / multinomial)を標準出力と wandb の両方へ
    sample_greedy = sample_text(model, meta, device, n_chars=200)
    sample_multi = sample_text_multinomial(model, meta, device, n_chars=200)
    print('\n---- サンプル生成(200文字, greedy) ----')
    print(sample_greedy)
    print('---- サンプル生成(200文字, multinomial) ----')
    print(sample_multi)
    print('---- ここまで ----')
    if use_wandb:
        try:
            import wandb
            wandb.log({
                'sample_text': wandb.Html(f'<pre>{sample_greedy}</pre>'),
                'sample_text_multinomial': wandb.Html(f'<pre>{sample_multi}</pre>'),
            })
            table = wandb.Table(columns=['mode', 'text'])
            table.add_data('greedy', sample_greedy)
            table.add_data('multinomial', sample_multi)
            wandb.log({'samples': table})
        except Exception as e:  # noqa: BLE001 (記録失敗は訓練を止めない)
            print(f'wandbへのサンプルログに失敗: {e}')

    if use_wandb:
        log_checkpoint_artifact(
            best_path, run_name, stage='final',
            metadata={'config': asdict(config), 'gates': gates,
                      'best_val_loss': best_val_loss,
                      'best_val_bpc': bits_per_character(best_val_loss),
                      'max_steps': max_steps},
            model=model,
        )
        wandb_finish()


if __name__ == '__main__':
    main()
