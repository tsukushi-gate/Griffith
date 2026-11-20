"""ベンチマークスイート。`griffith bench` の実装。

レシピ(configs/recipes/*.yaml)または保存済みチェックポイント(.pt)を対象に
  (a) hard eval精度(離散化回路の精度)
  (b) ゲート数(LutLayer/LutConv2dのLUT数。experiments/wandb_utils.py の
      count_logic_gates と同一集計のパッケージ内再実装)
  (c) forwardレイテンシ(CPU、ウォームアップ5回+20回計測の中央値、batch=1/128)
を計測し、benchmarks/results/<日付>_<name>.json へ固定スキーマ
  {name, timestamp, config,
   metrics: {hard_acc, total_gates, latency_ms_b1, latency_ms_b128},
   env: {python, torch, platform}}
で出力する。

レシピ指定時は短時間学習→bestチェックポイントを計測する。--skip-train は
ランダム初期化のまま計測(レイテンシ/ゲート数のみ。hard_accはnull)。
計測は再現性のためCPU固定(学習のみレシピのdevice設定に従う)。
"""

from __future__ import annotations

import json
import platform
import statistics
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import torch
import torch.nn as nn

from .config import TrainConfig, dump_config, load_config
from .runner import ResolvedData, _build_model, _hard_eval, _resolve_data
from .tasks import make_task

JST = timezone(timedelta(hours=9))

DEFAULT_OUT_DIR = 'benchmarks/results'

# レイテンシ計測条件(結果JSONのlatency_ms_b1/b128の前提)
LATENCY_WARMUP = 5
LATENCY_ITERS = 20
LATENCY_BATCH_LARGE = 128


def count_gates(model: nn.Module) -> int:
    """モデル内のロジックゲート(LUT)総数を数える。

    LutConv2d は num_gates(ツリー内部ノード数)、LutLayer は num_neurons
    (1ニューロン=1ゲート)を合算する。
    """
    from ..conv import LutConv2d
    from ..layers.lut import LutLayer

    total = 0
    for m in model.modules():
        if isinstance(m, LutConv2d):
            total += m.num_gates
        elif isinstance(m, LutLayer):
            total += m.num_neurons
    return total


def measure_latency_ms(
    model: nn.Module, in_width: int, batch_size: int,
) -> float:
    """CPU forwardレイテンシ(ms)を計測する。

    固定seedの二値ランダム入力でウォームアップ LATENCY_WARMUP 回の後、
    LATENCY_ITERS 回計測して中央値を返す(外れ値の影響を抑える)。
    """
    gen = torch.Generator().manual_seed(0)
    x = torch.randint(
        0, 2, (batch_size, in_width), generator=gen,
    ).to(torch.float32)
    was_training = model.training
    model.eval()
    with torch.no_grad():
        for _ in range(LATENCY_WARMUP):
            model(x)
        times_ms = []
        for _ in range(LATENCY_ITERS):
            start = time.perf_counter()
            model(x)
            times_ms.append((time.perf_counter() - start) * 1000.0)
    model.train(mode=was_training)
    return statistics.median(times_ms)


def _load_checkpoint_model(
    path: str, *, allow_unsafe: bool = False,
) -> tuple[TrainConfig, ResolvedData, nn.Module]:
    """チェックポイントからCPU上にモデルを復元する(runner保存スキーマ前提)。

    読み込みは既定で weights_only=True。テンソル以外の復元が必要な
    チェックポイントは allow_unsafe=True か GRIFFITH_ALLOW_UNSAFE_CKPT=1 が必要。
    """
    # 遅延import(convert パッケージは training.tasks に依存するため循環回避)
    from griffith_logic.convert.teacher import load_checkpoint_safely

    data = load_checkpoint_safely(
        path, allow_unsafe=allow_unsafe,
        what=f'ベンチ対象チェックポイント {path}',
    )
    cfg_dict = data.get('metadata', {}).get('config')
    if cfg_dict is None:
        raise ValueError(
            f'チェックポイントに設定(metadata.config)がありません: {path}'
            '(griffith train / run_training が保存したチェックポイントを指定してください)'
        )
    cfg = TrainConfig.model_validate(cfg_dict)
    cfg.run.device = 'cpu'  # 計測はCPU固定
    resolved = _resolve_data(cfg.data, 'cpu')
    model = _build_model(cfg.model, resolved, 'cpu')
    model.load_state_dict(data['model_state_dict'])
    return cfg, resolved, model


def run_bench(
    source: str,
    *,
    skip_train: bool = False,
    out_dir: str = DEFAULT_OUT_DIR,
    name: str | None = None,
    allow_unsafe_checkpoint: bool = False,
) -> dict:
    """ベンチマークを実行し、結果JSONを out_dir に保存する。

    Args:
        source: レシピYAML(.yaml/.yml)またはチェックポイント(.pt)のパス
        skip_train: レシピ指定時に学習せずランダム初期化のまま計測する
            (hard_accはnull。レイテンシ/ゲート数の計測用)
        out_dir: 結果JSONの出力先ディレクトリ
        name: 結果のname(省略時はsourceのファイル名から導出)
        allow_unsafe_checkpoint: weights_only=Trueで読めないチェックポイントの
            pickle読込を明示的に許可する(信頼できるファイルのみ)

    Returns:
        {'payload': 保存したJSONの内容, 'path': 保存先パス}
    """
    src = Path(source)
    if not src.is_file():
        raise FileNotFoundError(f'ベンチ対象が見つかりません: {source}')
    bench_name = name or src.stem
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    hard_acc: float | None
    if src.suffix in ('.yaml', '.yml'):
        cfg = load_config(str(src))
        if skip_train:
            resolved = _resolve_data(cfg.data, 'cpu')
            model = _build_model(cfg.model, resolved, 'cpu')
            hard_acc = None
        else:
            from .runner import run_training

            result = run_training(cfg, out_dir=str(out / f'train_{bench_name}'))
            cfg, resolved, model = _load_checkpoint_model(
                result['checkpoint_best'],
                allow_unsafe=allow_unsafe_checkpoint,
            )
            hard_acc = _hard_eval(
                model, resolved.eval_loader,
                make_task(resolved.kind, cfg.run.loss), 'cpu',
            )['acc']
    else:
        cfg, resolved, model = _load_checkpoint_model(
            str(src), allow_unsafe=allow_unsafe_checkpoint,
        )
        hard_acc = _hard_eval(
            model, resolved.eval_loader,
            make_task(resolved.kind, cfg.run.loss), 'cpu',
        )['acc']

    metrics = {
        'hard_acc': hard_acc,
        'total_gates': count_gates(model),
        'latency_ms_b1': measure_latency_ms(model, resolved.in_width, 1),
        'latency_ms_b128': measure_latency_ms(
            model, resolved.in_width, LATENCY_BATCH_LARGE
        ),
    }
    payload = {
        'name': bench_name,
        'timestamp': datetime.now(JST).isoformat(),
        'config': dump_config(cfg),
        'metrics': metrics,
        'env': {
            'python': platform.python_version(),
            'torch': torch.__version__,
            'platform': platform.platform(),
        },
    }
    path = out / f'{datetime.now(JST).strftime("%Y-%m-%d")}_{bench_name}.json'
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + '\n',
        encoding='utf-8',
    )
    return {'payload': payload, 'path': str(path)}


def cmd_bench(
    source: str,
    *,
    skip_train: bool = False,
    out_dir: str = DEFAULT_OUT_DIR,
    name: str | None = None,
    allow_unsafe_checkpoint: bool = False,
) -> int:
    """`griffith bench` の本体(戻り値=exit code)"""
    res = run_bench(
        source, skip_train=skip_train, out_dir=out_dir, name=name,
        allow_unsafe_checkpoint=allow_unsafe_checkpoint,
    )
    m = res['payload']['metrics']
    acc = 'null(--skip-train)' if m['hard_acc'] is None else f'{m["hard_acc"]:.4f}'
    print('--- ベンチマーク結果 ---')
    print(f'name={res["payload"]["name"]}  hard_acc={acc}  '
          f'total_gates={m["total_gates"]}')
    print(f'latency(CPU中央値): batch=1 {m["latency_ms_b1"]:.3f}ms / '
          f'batch={LATENCY_BATCH_LARGE} {m["latency_ms_b128"]:.3f}ms')
    print(f'出力: {res["path"]}')
    return 0
