"""attention梯子の学習/評価ハーネス。

シード固定・デバイス選択・学習ループ・評価・CSV出力・wandb連携、および
マルチシード集計(mean±std)を担う。L0(全FP基準)では soft評価 = hard評価
なので、離散化ギャップ(discretization_gap)は常に0.0とする。
"""

from __future__ import annotations

import csv
import os
import random
import sys
from collections.abc import Iterator, Sequence

import numpy as np
import torch
import torch.nn.functional as F

# ------------------------------------------------------------
# import規約: experiments/ 直下(wandb_utils等)と src/ を優先的にパス追加
# (train_boolean_attention.py と同じ流儀。src/を先頭に置き data 名衝突を回避)
# ------------------------------------------------------------
_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
_EXP_DIR = os.path.abspath(os.path.join(_SRC_DIR, '..', '..'))  # experiments/
for _p in (_EXP_DIR, _SRC_DIR):
    if _p in sys.path:
        sys.path.remove(_p)
    sys.path.insert(0, _p)

import yaml  # noqa: E402
from data import load_shakespeare, load_smnist  # noqa: E402
from metrics import (  # noqa: E402
    attention_entropy,
    classification_accuracy,
    discretization_gap,
    lm_loss_ppl,
)
from models import build_model  # noqa: E402
from wandb_utils import finish as wandb_finish  # noqa: E402
from wandb_utils import init_wandb, log_metrics  # noqa: E402

# ------------------------------------------------------------
# 定数
# ------------------------------------------------------------
DEFAULT_SEEDS: list[int] = [0, 1, 2]
SMOKE_MAX_STEPS = 100        # スモーク時の学習ステップ上限
SMOKE_EVAL_BATCHES = 20      # スモーク時の評価バッチ数上限
LM_NUM_EVALS = 20            # LM学習中の評価回数(等間隔)
# L3の温度アニーリング既定(cfg['train']のtemp_start/temp_endで上書き可)。
# 指数減衰: tau = temp_start * (temp_end/temp_start)**(step/(total_steps-1))
_DEFAULT_TEMP_START = 2.0
_DEFAULT_TEMP_END = 0.5
# CSV列: eval_mode は 'soft'/'hard'(ラダー段のval2行) / '-'(train行, fp/fp段のval)
_CSV_HEADER = [
    'seed', 'epoch_or_step', 'split', 'eval_mode',
    'loss', 'ppl', 'acc', 'attn_entropy',
]
_EVAL_MODE_NONE = '-'
# 集計対象の指標(loss/ppl/acc/attn_entropy はラダー段では hard 主指標)
_SUMMARY_METRICS = ['loss', 'ppl', 'acc', 'attn_entropy', 'discretization_gap']
# taskキー(config) → データローダ
_CLASSIFICATION_TASKS = ('smnist', 'classification')
_LM_TASKS = ('shakespeare', 'lm')


# ============================================================
# 環境ユーティリティ
# ============================================================

def set_seed(seed: int) -> None:
    """torch/numpy/random のシードを固定する。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> str:
    """利用可能なデバイスを 'mps' > 'cuda' > 'cpu' の優先度で返す。"""
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return 'mps'
    if torch.cuda.is_available():
        return 'cuda'
    return 'cpu'


# ============================================================
# 内部ヘルパー
# ============================================================

def _compute_loss(logits: torch.Tensor, y: torch.Tensor, task: str) -> torch.Tensor:
    """タスク別の学習損失(cross entropy)を返す。"""
    if task == 'classification':
        return F.cross_entropy(logits, y)
    # lm: (B,T,V) vs (B,T)
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))


def _iter_train_steps(
    loader: torch.utils.data.DataLoader,
    total_steps: int,
) -> Iterator[tuple[int, tuple]]:
    """train_loader を巡回して total_steps 分の (step, batch) を返す。"""
    step = 0
    while step < total_steps:
        for batch in loader:
            yield step, batch
            step += 1
            if step >= total_steps:
                return


@torch.no_grad()
def _evaluate(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    task: str,
    device: str,
    *,
    mode: str = 'hard',
    max_batches: int | None = None,
) -> dict:
    """val上で loss/acc/ppl と attention_entropy を計算する。

    mode='hard': `model.eval()` で評価(LutLayer/Binarize が hard{0,1})。
    mode='soft': `model.train()` で評価(LutLayer が soft[0,1])。どちらも no_grad。
    soft評価は dropout=0 前提(configで担保)。評価後は元の train/eval 状態に復帰する。
    """
    was_training = model.training
    if mode == 'soft':
        model.train()
    else:
        model.eval()
    causal = task == 'lm'

    total_loss = 0.0
    total_items = 0      # classification: サンプル数
    total_tokens = 0     # lm: トークン数
    total_correct = 0.0
    ent_sum = 0.0
    ent_count = 0

    for i, (x, y) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        x = x.to(device)
        y = y.to(device)
        logits = model(x)

        if task == 'classification':
            loss = F.cross_entropy(logits, y)
            n = y.size(0)
            total_loss += loss.item() * n
            total_correct += classification_accuracy(logits, y) * n
            total_items += n
        else:  # lm
            batch_loss, _ = lm_loss_ppl(logits, y)
            ntok = y.numel()
            total_loss += batch_loss * ntok
            total_tokens += ntok

        ent_sum += attention_entropy(model.last_attn_weights, causal=causal)
        ent_count += 1

    if task == 'classification':
        avg_loss = total_loss / max(total_items, 1)
        acc: float | None = total_correct / max(total_items, 1)
        ppl: float | None = None
    else:
        avg_loss = total_loss / max(total_tokens, 1)
        acc = None
        ppl = float(np.exp(min(avg_loss, 20.0)))

    attn_ent = ent_sum / max(ent_count, 1)

    model.train(was_training)  # 元の状態に復帰

    return {'loss': avg_loss, 'acc': acc, 'ppl': ppl, 'attn_entropy': attn_ent}


def _resolve_task_kind(task: str) -> str:
    """configのtaskキーを内部タスク種別(classification/lm)に写像する。"""
    if task in _CLASSIFICATION_TASKS:
        return 'classification'
    if task in _LM_TASKS:
        return 'lm'
    raise ValueError(f'未知のtask: {task}')


def _wandb_config(cfg: dict, seed: int, device: str) -> dict:
    """wandbに渡すフラットな設定辞書を作る。"""
    flat: dict = {
        'seed': seed,
        'device': device,
        'stage': cfg.get('stage'),
        'task': cfg.get('task'),
    }
    for key, value in (cfg.get('model') or {}).items():
        flat[f'model/{key}'] = value
    for key, value in (cfg.get('train') or {}).items():
        flat[f'train/{key}'] = value
    return flat


def _save_config(run_dir: str, cfg: dict, seed: int, device: str) -> None:
    """実効設定を config.yaml として保存する(meta等の巨大辞書は含めない)。"""
    out = {k: v for k, v in cfg.items()}
    out['seed'] = seed
    out['device'] = device
    with open(os.path.join(run_dir, 'config.yaml'), 'w', encoding='utf-8') as f:
        yaml.safe_dump(out, f, allow_unicode=True, sort_keys=False)


def _csv_cell(value: float | int | None) -> str | float | int:
    """CSVセル値。Noneは空文字にする。"""
    return '' if value is None else value


def _anneal_tau(
    step: int, total_steps: int, temp_start: float, temp_end: float,
) -> float:
    """指数減衰の温度。step0で temp_start、最終stepで temp_end。"""
    frac = step / max(total_steps - 1, 1)
    return temp_start * (temp_end / temp_start) ** frac


# ============================================================
# 1シード学習
# ============================================================

def train_one(
    cfg: dict,
    seed: int,
    *,
    smoke: bool = False,
    out_root: str,
) -> dict:
    """1シード分の学習/評価を実行し、最終val指標dictを返す。

    classificationはepochs、lmはmax_steps(なければepochs)で学習する。smoke時は
    1エポック/最大~100stepに短縮しwandbを無効化する。metrics.csv と config.yaml を
    run_dir に出力する。
    """
    set_seed(seed)
    device = get_device()

    task = cfg['task']
    task_kind = _resolve_task_kind(task)
    stage = cfg.get('stage', 'L0')
    model_cfg = dict(cfg['model'])
    train_cfg = dict(cfg.get('train') or {})

    # ラダー段(soft/hard離散化ギャップを計測)か否か。
    # attention がFP以外、または ffn が論理ゲート化されているとき離散化が生じる。
    attention = str(model_cfg.get('attention', 'fp'))
    ffn = str(model_cfg.get('ffn', 'fp'))
    is_ladder = attention != 'fp' or ffn == 'logic'

    run_name = f'{stage}_{task}'
    run_dir = os.path.join(out_root, f'{run_name}_seed{seed}')
    os.makedirs(run_dir, exist_ok=True)

    # --- データ ---
    if task_kind == 'classification':
        batch_size = int(train_cfg.get('batch_size', 128))
        train_loader, val_loader, meta = load_smnist(batch_size)
    else:
        batch_size = int(train_cfg.get('batch_size', 64))
        context_len = int(train_cfg.get('context_len', 64))
        train_loader, val_loader, meta = load_shakespeare(
            batch_size, context_len=context_len,
        )

    # --- モデル/最適化 ---
    model = build_model(model_cfg, meta).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg.get('lr', 1e-3)),
        weight_decay=float(train_cfg.get('weight_decay', 0.0)),
    )

    # --- ステップ数/評価間隔 ---
    steps_per_epoch = len(train_loader)
    if task_kind == 'classification':
        epochs = int(train_cfg.get('epochs', 10))
        if smoke:
            epochs = 1
        total_steps = epochs * steps_per_epoch
        eval_every = steps_per_epoch  # エポック境界で評価
    else:  # lm
        if 'max_steps' in train_cfg:
            total_steps = int(train_cfg['max_steps'])
        else:
            total_steps = int(train_cfg.get('epochs', 1)) * steps_per_epoch
        eval_every = max(1, total_steps // LM_NUM_EVALS)

    if smoke:
        total_steps = min(total_steps, SMOKE_MAX_STEPS)
    eval_batches = SMOKE_EVAL_BATCHES if smoke else None

    _save_config(run_dir, cfg, seed, device)

    # --- wandb ---
    wandb_cfg = cfg.get('wandb') or {}
    use_wandb = bool(wandb_cfg.get('enabled', False)) and not smoke
    if use_wandb:
        use_wandb = init_wandb(
            run_name=f'{run_name}_seed{seed}',
            config_dict=_wandb_config(cfg, seed, device),
            model=model,
            device=device,
            tags=wandb_cfg.get('tags'),
            group=wandb_cfg.get('group'),
        )

    csv_rows: list[list] = []

    def _train_row(marker: int, train_loss: float | None) -> None:
        if train_loss is None:
            return
        csv_rows.append([
            seed, marker, 'train', _EVAL_MODE_NONE,
            _csv_cell(train_loss), '', '', '',
        ])

    def _val_row(marker: int, eval_mode: str, val: dict) -> None:
        csv_rows.append([
            seed, marker, 'val', eval_mode,
            _csv_cell(val['loss']), _csv_cell(val['ppl']),
            _csv_cell(val['acc']), _csv_cell(val['attn_entropy']),
        ])

    def _gap(soft: dict, hard: dict) -> float:
        # 分類: soft_acc - hard_acc / LM: hard_loss - soft_loss(hard悪化を正にする)
        if task_kind == 'classification':
            return discretization_gap(soft['acc'], hard['acc'])
        return discretization_gap(hard['loss'], soft['loss'])

    def _record_eval(marker: int, train_loss: float | None) -> dict:
        """評価を実行しCSV行/wandbへ記録する。

        返り値は {'hard','soft','gap'}。fp/fp段は soft=hard の1回評価・gap=0.0。
        ラダー段は hard(主指標)/soft の2回評価と実測gapを算出する。
        """
        hard = _evaluate(
            model, val_loader, task_kind, device,
            mode='hard', max_batches=eval_batches,
        )
        if is_ladder:
            soft = _evaluate(
                model, val_loader, task_kind, device,
                mode='soft', max_batches=eval_batches,
            )
            gap = _gap(soft, hard)
            _train_row(marker, train_loss)
            _val_row(marker, 'soft', soft)
            _val_row(marker, 'hard', hard)
        else:
            # fp/fp: soft=hardなので1回評価。eval_mode='-'で1行のみ記録。
            soft = hard
            gap = 0.0
            _train_row(marker, train_loss)
            _val_row(marker, _EVAL_MODE_NONE, hard)

        if use_wandb:
            # 主指標は hard。ラダー段は soft と gap も併記する。
            payload = {
                'val/loss': hard['loss'],
                'val/attn_entropy': hard['attn_entropy'],
            }
            if hard['acc'] is not None:
                payload['val/acc'] = hard['acc']
            if hard['ppl'] is not None:
                payload['val/ppl'] = hard['ppl']
            if is_ladder:
                payload['val/discretization_gap'] = gap
                payload['val/soft_loss'] = soft['loss']
                if soft['acc'] is not None:
                    payload['val/soft_acc'] = soft['acc']
                if soft['ppl'] is not None:
                    payload['val/soft_ppl'] = soft['ppl']
            if train_loss is not None:
                payload['train/loss'] = train_loss
            log_metrics(payload, step=marker)

        return {'hard': hard, 'soft': soft, 'gap': gap}

    # --- 初期評価(学習前ベースライン, epoch_or_step=0) ---
    final_val = _record_eval(0, train_loss=None)

    # --- L3: 温度アニーリング/補助損失の設定(選択器norm時のみ有効) ---
    # set_temperature/pop_aux_loss は LadderAttention 側で softmax/fp段は no-op。
    # そのため L0-L2 では呼んでも既存挙動を壊さない(tau/auxが影響しない)。
    temp_start = float(train_cfg.get('temp_start', _DEFAULT_TEMP_START))
    temp_end = float(train_cfg.get('temp_end', _DEFAULT_TEMP_END))
    aux_lambda = float(train_cfg.get('aux_lambda', 0.0))
    can_anneal = hasattr(model, 'set_temperature')
    can_aux = hasattr(model, 'pop_aux_loss')

    # --- 学習ループ ---
    running_loss = 0.0
    running_n = 0
    for step, (x, y) in _iter_train_steps(train_loader, total_steps):
        model.train()
        # 温度アニーリング(選択器normのみ反映。softmax/fp段は no-op)
        if can_anneal:
            tau = _anneal_tau(step, total_steps, temp_start, temp_end)
            model.set_temperature(tau)
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        loss = _compute_loss(logits, y, task_kind)
        # 補助損失(選択器normのみ非ゼロ。lam=0 or aux=0 なら loss 不変)
        if can_aux:
            aux = model.pop_aux_loss()
            if aux_lambda != 0.0:
                loss = loss + aux_lambda * aux

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_n += 1

        is_eval_step = (step + 1) % eval_every == 0 or (step + 1) == total_steps
        if is_eval_step:
            train_loss = running_loss / max(running_n, 1)
            running_loss = 0.0
            running_n = 0
            final_val = _record_eval(step + 1, train_loss=train_loss)

    # --- CSV書き出し ---
    csv_path = os.path.join(run_dir, 'metrics.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(_CSV_HEADER)
        writer.writerows(csv_rows)

    if use_wandb:
        wandb_finish()

    # 主指標は hard(受け入れ基準の判定に使う)。soft と実測gapも返す。
    # fp/fp段は soft=hard・gap=0.0。
    hard = final_val['hard']
    soft = final_val['soft']
    result = {
        'seed': seed,
        'loss': hard['loss'],
        'acc': hard['acc'],
        'ppl': hard['ppl'],
        'attn_entropy': hard['attn_entropy'],
        'soft_loss': soft['loss'],
        'soft_acc': soft['acc'],
        'soft_ppl': soft['ppl'],
        'soft_attn_entropy': soft['attn_entropy'],
        'discretization_gap': final_val['gap'],
    }
    return result


# ============================================================
# マルチシード実行/集計
# ============================================================

def _aggregate(per_seed: list[dict]) -> dict:
    """各指標の mean/std を計算する(値がNoneの指標はスキップ)。"""
    summary: dict = {}
    for metric in _SUMMARY_METRICS:
        values = [
            float(r[metric]) for r in per_seed
            if r.get(metric) is not None
        ]
        if not values:
            continue
        mean = float(np.mean(values))
        std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        summary[metric] = {'mean': mean, 'std': std}
    return summary


def run_stage(
    cfg: dict,
    *,
    out_root: str,
    seeds: Sequence[int] | None = None,
    smoke: bool = False,
) -> dict:
    """複数シードで train_one を実行し、最終指標の mean±std を集計する。

    summary.csv(per-seed行 + mean/std行)を out_root/{stage}_{task}/ に書く。
    """
    if seeds is None:
        seeds = cfg.get('seeds') or DEFAULT_SEEDS
    seeds = list(seeds)

    task = cfg['task']
    stage = cfg.get('stage', 'L0')

    per_seed: list[dict] = []
    for seed in seeds:
        per_seed.append(train_one(cfg, seed, smoke=smoke, out_root=out_root))

    summary = _aggregate(per_seed)

    # --- summary.csv ---
    summary_dir = os.path.join(out_root, f'{stage}_{task}')
    os.makedirs(summary_dir, exist_ok=True)
    summary_path = os.path.join(summary_dir, 'summary.csv')
    header = ['seed', *_SUMMARY_METRICS]
    with open(summary_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        # per-seed行
        for r in per_seed:
            writer.writerow([r['seed']] + [_csv_cell(r.get(m)) for m in _SUMMARY_METRICS])
        # 集計行(mean/std)
        writer.writerow(['mean'] + [
            _csv_cell(summary.get(m, {}).get('mean')) for m in _SUMMARY_METRICS
        ])
        writer.writerow(['std'] + [
            _csv_cell(summary.get(m, {}).get('std')) for m in _SUMMARY_METRICS
        ])

    return {
        'stage': stage,
        'task': task,
        'seeds': seeds,
        'per_seed': per_seed,
        'summary': summary,
        'summary_csv': summary_path,
    }
