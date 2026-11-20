"""
LogicLadderGPT 訓練スクリプト(D1: 自己回帰言語モデルの論理ゲート化 / Tiny Shakespeare)

`griffith_logic.models.LogicLadderGPT`(causal decoder-only Transformer、attention段階
論理化)を char-level 言語モデルとして訓練する。レシピ
`configs/recipes/logic_gpt_shakespeare.yaml` の --variant で L0(fp/fp 基準)と
L3-C(射影ゲート化+MUX選択器+論理FFN 本命)を切替える。

Tiny Shakespeare のDL・語彙構築・窓生成は本スクリプト内で自作する(attention_ladder
への実行時依存を作らない)。データは相対パス ./data-shakespeare/ に落とす。

使用例:
    # L0 基準(20000ステップ)
    uv run python experiments/train_logic_gpt.py --config configs/recipes/logic_gpt_shakespeare.yaml

    # L3-C 本命
    uv run python experiments/train_logic_gpt.py --config configs/recipes/logic_gpt_shakespeare.yaml --variant L3-C

    # スモーク(CPUで2分以内、wandb無し)
    uv run python experiments/train_logic_gpt.py --config configs/recipes/logic_gpt_shakespeare.yaml --max-steps 20 --no-wandb

    # Modal 実行
    uv run modal run scripts/modal_app.py --script train_logic_gpt.py --args "--config configs/recipes/logic_gpt_shakespeare.yaml --variant L3-C"
"""

import argparse
import os
import sys
import time
import urllib.request
from datetime import datetime, timedelta, timezone

JST = timezone(timedelta(hours=9))

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from checkpoint_utils import load_checkpoint, save_checkpoint, setup_checkpoint_dir
from env_utils import load_dotenv
from wandb_utils import (
    add_experiment_args,
    init_wandb,
    log_checkpoint_artifact,
    log_metrics,
    normalize_tags,
)
from wandb_utils import finish as wandb_finish

from griffith_logic.models import LogicLadderGPT, LogicLadderGPTConfig

load_dotenv()

# ============================================================
# Tiny Shakespeare(char-level LM)— 自作(attention_ladder非依存)
# ============================================================

SHAKESPEARE_URL = (
    'https://raw.githubusercontent.com/karpathy/char-rnn/'
    'master/data/tinyshakespeare/input.txt'
)
_DEFAULT_DATA_DIR = './data-shakespeare'
_SHAKESPEARE_SUBPATH = os.path.join('tinyshakespeare', 'input.txt')
_TRAIN_RATIO = 0.9


def prepare_shakespeare(data_dir: str = _DEFAULT_DATA_DIR) -> str:
    """Tiny Shakespeare の input.txt を用意しパスを返す(無ければDL)。

    DL先は data_dir(相対パス。Modalでは永続Volumeに載る)。DL失敗時は捏造せず
    明確に例外を投げる。
    """
    path = os.path.join(data_dir, _SHAKESPEARE_SUBPATH)
    if os.path.isfile(path) and os.path.getsize(path) > 0:
        return path

    os.makedirs(os.path.dirname(path), exist_ok=True)
    try:
        with urllib.request.urlopen(SHAKESPEARE_URL, timeout=60) as resp:
            data = resp.read()
    except Exception as e:  # noqa: BLE001 (DL失敗は原因を明示して再送出)
        raise RuntimeError(
            f'Tiny Shakespeare のダウンロードに失敗しました: {SHAKESPEARE_URL} '
            f'(原因: {e})'
        ) from e
    if not data:
        raise RuntimeError(
            f'Tiny Shakespeare のダウンロード内容が空です: {SHAKESPEARE_URL}'
        )
    with open(path, 'wb') as f:
        f.write(data)
    return path


class _CharWindowDataset(Dataset):
    """スライディング窓でchar-LMの(x, y)ペアを生成する。

    x = tokens[i : i+context_len]、y = tokens[i+1 : i+context_len+1]。
    """

    def __init__(self, data: torch.Tensor, context_len: int) -> None:
        self._data = data
        self._context_len = context_len

    def __len__(self) -> int:
        return max(0, self._data.numel() - self._context_len)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        chunk = self._data[idx: idx + self._context_len + 1]
        return chunk[:-1], chunk[1:]


def load_shakespeare(
    batch_size: int,
    context_len: int,
    data_dir: str = _DEFAULT_DATA_DIR,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader, dict]:
    """Tiny Shakespeare(char-level LM)の(train_loader, val_loader, meta)を返す。

    char語彙(stoi/itos, 語彙数V〜65)を構築し 90/10 でtrain/val分割、
    スライディング窓(context_len)で次トークン予測データを作る。
    """
    path = prepare_shakespeare(data_dir)
    with open(path, encoding='utf-8') as f:
        text = f.read()

    chars = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    vocab_size = len(chars)

    ids = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    split = int(ids.numel() * _TRAIN_RATIO)
    train_data = ids[:split]
    val_data = ids[split:]

    train_set = _CharWindowDataset(train_data, context_len)
    val_set = _CharWindowDataset(val_data, context_len)

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        drop_last=True, num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False,
        drop_last=False, num_workers=num_workers,
    )
    meta = {
        'task': 'lm',
        'seq_len': context_len,
        'vocab_size': vocab_size,
        'stoi': stoi,
        'itos': itos,
    }
    return train_loader, val_loader, meta


# ============================================================
# レシピ読み込み
# ============================================================

def load_recipe(config_path: str, variant: str | None) -> tuple[dict, dict, str]:
    """yamlレシピから (model_cfg, train_cfg, variant名) を返す。"""
    with open(config_path, encoding='utf-8') as f:
        recipe = yaml.safe_load(f)

    variants = recipe.get('variants', {})
    if not variants:
        raise ValueError(f'レシピに variants がありません: {config_path}')
    chosen = variant or recipe.get('default_variant')
    if chosen not in variants:
        raise ValueError(
            f"variant='{chosen}' はレシピに存在しません(候補: {list(variants)})"
        )
    model_cfg = dict(variants[chosen]['model'])
    train_cfg = dict(recipe.get('train', {}))
    return model_cfg, train_cfg, chosen


def cycle(loader):
    """DataLoaderを無限に巡回するジェネレータ(ステップ数ベース訓練用)。"""
    while True:
        yield from loader


# ============================================================
# 評価・生成
# ============================================================

@torch.no_grad()
def evaluate(
    model, loader, device, max_batches: int | None = 20, mode: str = 'hard',
) -> float:
    """val loss を平均で返す。max_batches=Noneで全val評価。

    mode='hard': `model.eval()` で評価(LutLayer/Binarize が hard{0,1}、選択器も
        argmax。従来の val_loss と同一)。
    mode='soft': `model.train()` で評価(LutLayer が soft[0,1]、選択器はGumbel緩和)。
        ladder harness._evaluate(mode='soft') と同義。soft と hard の差が
        離散化ギャップ Δ_disc = loss_hard - loss_soft(Mind the Gap)。
    どちらも no_grad。評価後は元の train/eval 状態に復帰する。

    注意: 部分評価(max_batches指定)はvalの先頭スライスに固定されるため、
    ladder実験(全val評価)との数値比較には使えない(に0.31 nat上振れを実測)。
    本走はレシピ train.eval_batches: null(=全val)を使うこと。
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


# ============================================================
# メイン
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='LogicLadderGPT 訓練スクリプト(Tiny Shakespeare / char-LM)'
    )
    parser.add_argument('--config', type=str, required=True,
                        help='レシピyaml(configs/recipes/logic_gpt_shakespeare.yaml)')
    parser.add_argument('--variant', type=str, default=None,
                        help='L0 / L3-C 等(未指定でレシピの default_variant)')
    parser.add_argument('--max-steps', type=int, default=None,
                        help='最大ステップ数(レシピ train.max_steps を上書き)')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='バッチサイズ(レシピ train.batch_size を上書き)')
    parser.add_argument('--lr', type=float, default=None,
                        help='学習率(レシピ train.lr を上書き)')
    parser.add_argument('--eval-freq', type=int, default=None,
                        help='評価頻度(レシピ train.eval_freq を上書き)')
    parser.add_argument('--eval-batches', type=int, default=None,
                        help='評価バッチ数の上限(0=全val。未指定時はレシピ'
                             ' train.eval_batches、それも無ければ20=高速部分評価)')
    parser.add_argument('--data-dir', type=str, default=_DEFAULT_DATA_DIR,
                        help=f'データ保存先 (default: {_DEFAULT_DATA_DIR})')
    parser.add_argument('--seed', type=int, default=0, help='ランダムシード')
    parser.add_argument('--device', type=str, default=None,
                        help='デバイス (default: 自動検出)')
    parser.add_argument('--log-dir', type=str, default='logs',
                        help='ログ出力先 (default: logs)')
    parser.add_argument('--no-wandb', action='store_true',
                        help='wandbを無効化する(スモーク用)')
    parser.add_argument('--resume', type=str, default=None,
                        help='チェックポイントから訓練再開(latest.pt等のパス指定)')
    add_experiment_args(parser)
    parser.add_argument('--run-dir', type=str, default=None,
                        help='チェックポイント保存先ディレクトリ名'
                             '(checkpoints/<NAME>/固定。未指定時はタイムスタンプ方式)')
    args = parser.parse_args()

    # デバイス自動検出
    if args.device is None:
        if torch.cuda.is_available():
            args.device = 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            args.device = 'mps'
        else:
            args.device = 'cpu'
    device = args.device
    print(f'デバイス: {device}')

    torch.manual_seed(args.seed)

    # レシピ読み込み(argsで上書き)
    model_cfg, train_cfg, variant = load_recipe(args.config, args.variant)
    batch_size = args.batch_size or int(train_cfg.get('batch_size', 64))
    context_len = int(train_cfg.get('context_len', 64))
    lr = args.lr if args.lr is not None else float(train_cfg.get('lr', 3.0e-4))
    weight_decay = float(train_cfg.get('weight_decay', 0.0))
    max_steps = args.max_steps or int(train_cfg.get('max_steps', 20000))
    eval_freq = args.eval_freq or int(train_cfg.get('eval_freq', 500))
    # 評価バッチ数: 0またはNone(レシピ null)=全val評価(ladderプロトコル準拠)
    if args.eval_batches is not None:
        eval_batches = None if args.eval_batches == 0 else args.eval_batches
    else:
        _eb = train_cfg.get('eval_batches', 20)
        eval_batches = None if _eb in (None, 0) else int(_eb)
    temp_start = float(train_cfg.get('temp_start', 1.0))
    temp_end = float(train_cfg.get('temp_end', 1.0))
    aux_lambda = float(train_cfg.get('aux_lambda', 0.0))

    # データ
    train_loader, val_loader, meta = load_shakespeare(
        batch_size=batch_size, context_len=context_len, data_dir=args.data_dir,
    )
    vocab_size = meta['vocab_size']
    print(f'\nデータ: Tiny Shakespeare  vocab_size={vocab_size} '
          f'context_len={context_len}')

    # モデル
    config = LogicLadderGPTConfig.from_model_cfg(
        model_cfg, vocab_size=vocab_size, seq_len=context_len,
    )
    model = LogicLadderGPT(config).to(device)
    counts = model.count_parameters()
    print(f'\nvariant={variant}  attention={config.attention} ffn={config.ffn}')
    print(f'  d_model={config.d_model} n_heads={config.n_heads} '
          f'n_layers={config.n_layers} mlp_ratio={config.mlp_ratio}')
    if config.attention_kwargs:
        print(f'  attention_kwargs={config.attention_kwargs}')
    print(f'  総パラメータ数: {counts["total"]:,}')

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay,
    )

    # チェックポイント(--run-dir指定時はcheckpoints/<NAME>/固定、未指定時はタイムスタンプ方式)
    if args.run_dir:
        run_dir = os.path.join('checkpoints', args.run_dir)
        os.makedirs(run_dir, exist_ok=True)
        latest_path = os.path.join(run_dir, 'latest.pt')
        best_path = os.path.join(run_dir, 'best.pt')
    else:
        run_dir, latest_path, best_path = setup_checkpoint_dir(f'logic_gpt_{variant}')

    # --resume: チェックポイントからモデル/optimizer/ステップ/best_metricを復元
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
    run_name = (f'logic_gpt_{variant}_d{config.d_model}'
                f'_l{config.n_layers}_s{max_steps}')
    if not args.no_wandb:
        wandb_config = {
            'variant': variant, 'model_cfg': model_cfg, 'train_cfg': train_cfg,
            'batch_size': batch_size, 'context_len': context_len, 'lr': lr,
            'weight_decay': weight_decay, 'max_steps': max_steps,
            'seed': args.seed,
        }
        use_wandb = init_wandb(
            run_name=run_name,
            config_dict=wandb_config,
            model=model,
            device=device,
            tags=['logic_gpt', variant, 'shakespeare',
                  f'attn-{config.attention}', f'ffn-{config.ffn}']
            + normalize_tags(model='logic_gpt', dataset='shakespeare', seed=args.seed),
            group='logic_gpt',
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

    ckpt_metadata = {
        'variant': variant, 'config_path': args.config, 'wandb_run_id': wandb_run_id,
    }

    # 訓練ループ(ステップ数ベース)
    print(f'\n訓練開始: {max_steps}ステップ (評価={eval_freq}ステップごと、'
          f'開始ステップ={start_step + 1})')
    print('=' * 70)
    loss_fn = nn.CrossEntropyLoss()
    data_iter = cycle(train_loader)
    train_start = time.time()
    model.train()

    for step in range(start_step + 1, max_steps + 1):
        # 温度アニーリング: 指数減衰(ladder harness._anneal_tau と同一式)。
        # 線形減衰では高温期間が長くなり離散化ギャップが残る(に
        # L3-C全val ppl 15.8 vs ladder 12.1 の差として実測)ため指数に揃える
        progress = (step - 1) / max(1, max_steps - 1)
        tau = temp_start * (temp_end / temp_start) ** progress
        model.set_temperature(tau)

        x, y = next(data_iter)
        x = x.to(device)
        y = y.to(device)

        logits = model(x)  # (B,T,V)
        main_loss = loss_fn(logits.reshape(-1, vocab_size), y.reshape(-1))
        aux = model.pop_aux_loss()  # 選択器の補助損失(fpでは0.0)
        loss = main_loss + aux_lambda * aux

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ハートビート(毎ステップ経過/残り/終了予定を計算、eval時に表示)
        if step % eval_freq == 0 or step == max_steps or step == 1:
            # hard(離散化)= 従来の val_loss。best判定はこちらで行う
            val_loss = evaluate(
                model, val_loader, device, max_batches=eval_batches, mode='hard',
            )
            # soft(連続緩和)= model.train()評価。離散化ギャップを測る
            val_soft_loss = evaluate(
                model, val_loader, device, max_batches=eval_batches, mode='soft',
            )
            # 離散化ギャップ Δ_disc = loss_hard - loss_soft(Mind the Gap)。
            # softは通常hardより良い(loss小)ため正値。ゲート崩壊で0に近づく
            discretization_gap = val_loss - val_soft_loss
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss

            # チェックポイント(eval毎にlatest.ptを必ず保存、best更新時はbest.ptにも保存)
            extra_state = {'torch_rng': torch.get_rng_state()}
            save_checkpoint(latest_path, model, optimizer, None, step,
                            best_val_loss, ckpt_metadata, extra_state=extra_state)
            if is_best:
                save_checkpoint(best_path, model, optimizer, None, step,
                                best_val_loss, ckpt_metadata, extra_state=extra_state)
            if use_wandb:
                log_checkpoint_artifact(
                    latest_path, run_name, stage='eval',
                    metadata={'step': step, 'val_loss': val_loss},
                )

            elapsed = time.time() - train_start
            sec_per_step = elapsed / step
            remaining = sec_per_step * (max_steps - step)
            eta = datetime.now(JST) + timedelta(seconds=remaining)
            elapsed_str = str(timedelta(seconds=int(elapsed)))
            remain_str = str(timedelta(seconds=int(remaining)))
            aux_val = float(aux.detach()) if torch.is_tensor(aux) else float(aux)

            print(
                f'[step {step:>6d}/{max_steps}] '
                f'train_loss={main_loss.item():.4f} '
                f'val_loss={val_loss:.4f} '
                f'val_soft={val_soft_loss:.4f} '
                f'gap={discretization_gap:.4f} '
                f'aux={aux_val:.4f} tau={tau:.3f}'
                f'{"  ★ BEST" if is_best else ""}'
                f'  [{elapsed_str}<{remain_str}, 終了予定 {eta.strftime("%H:%M:%S")}]'
            )
            if use_wandb:
                log_metrics({
                    'step': step,
                    'train_loss': main_loss.item(),
                    'val_loss': val_loss,
                    'val_soft_loss': val_soft_loss,
                    'discretization_gap': discretization_gap,
                    'best_val_loss': best_val_loss,
                    'aux_loss': aux_val,
                    'tau': tau,
                })

    total_time = str(timedelta(seconds=int(time.time() - train_start)))
    print('=' * 70)
    print(f'訓練完了: best_val_loss={best_val_loss:.4f}  (総時間: {total_time})')

    # サンプル生成(200文字)をログ + wandb へ
    sample = sample_text(model, meta, device, n_chars=200)
    print('\n---- サンプル生成(200文字, greedy) ----')
    print(sample)
    print('---- ここまで ----')
    if use_wandb:
        try:
            import wandb
            wandb.log({'sample_text': wandb.Html(f'<pre>{sample}</pre>')})
        except Exception:
            pass

    # 重みを wandb Artifacts へ
    if use_wandb:
        log_checkpoint_artifact(
            best_path, run_name, stage='final',
            metadata={'variant': variant, 'model_cfg': model_cfg,
                      'best_val_loss': best_val_loss, 'max_steps': max_steps},
            model=model,
        )
        wandb_finish()


if __name__ == '__main__':
    main()
