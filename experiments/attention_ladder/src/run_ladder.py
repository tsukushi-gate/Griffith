"""attention梯子を1プロセスで順次実行するドライバ。

複数のconfig(段)を1ジョブで直列に `harness.run_stage` へ流す。用途は
**Modalで1ジョブ=梯子全体**を回すこと(modal_app が runpy で本ファイルを実行し、
cwd=/workspace, ソース実体=/root。config は _resolve_config と同じ流儀で解決)。

使用例:
    uv run python experiments/attention_ladder/src/run_ladder.py \
        --configs "l0_shakespeare,l0base_shakespeare,l1_shakespeare,l2_shakespeare,l3c_shakespeare"

    # ローカルスモーク(tiny学習 + wandb無効, seed0のみ)
    uv run python experiments/attention_ladder/src/run_ladder.py \
        --configs "l1_shakespeare,l2_shakespeare,l3c_shakespeare" \
        --smoke --no-wandb --seeds 0

    # Modal(1ジョブ=梯子全体)
    uv run modal run scripts/modal_app.py --background \
        --script attention_ladder/src/run_ladder.py \
        --args "--configs l0_shakespeare,l0base_shakespeare,l1_shakespeare,l2_shakespeare,l3c_shakespeare"
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import traceback

# ------------------------------------------------------------
# import規約: experiments/ と src/ をパス追加(train.py と同じ流儀)
# ------------------------------------------------------------
_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
_EXP_DIR = os.path.abspath(os.path.join(_SRC_DIR, '..', '..'))  # experiments/
for _p in (_EXP_DIR, _SRC_DIR):
    if _p in sys.path:
        sys.path.remove(_p)
    sys.path.insert(0, _p)

import yaml  # noqa: E402
from env_utils import load_dotenv  # noqa: E402
from harness import run_stage  # noqa: E402
from train import _parse_seeds, _resolve_config  # noqa: E402

# configs は experiments/attention_ladder/configs/ 配下(cwd相対で解決可能な形)。
# _resolve_config が cwd相対 → ソースツリー基準(/root)の順で再解決する。
_CONFIGS_REL = os.path.join('experiments', 'attention_ladder', 'configs')
# 既定出力ルート(Modal規約: cwd=/workspace の相対パスが永続Volumeに載る)
_DEFAULT_OUT = 'results'
# スモーク時のサマリ表示に使う主指標(LMは acc=None のため自動で除外される)
_REPORT_METRICS = ('loss', 'ppl', 'acc', 'attn_entropy', 'discretization_gap')


def _resolve_config_name(name: str) -> str:
    """config名(拡張子なし可)を configs/配下のパスに解決する。

    train._resolve_config と同じ流儀で cwd相対 → ソースツリー基準の順に探索する。
    ローカル(cwd=リポジトリルート)でもModal(cwd=/workspace, 実体=/root)でも通す。
    """
    stem = name.strip()
    if stem.endswith(('.yaml', '.yml')):
        stem = os.path.splitext(stem)[0]
    rel = os.path.join(_CONFIGS_REL, f'{stem}.yaml')
    return _resolve_config(rel)


def _apply_overrides(cfg: dict, args: argparse.Namespace) -> None:
    """CLIフラグ(smoke/no-wandb/max-steps)を1つのcfgに反映する。"""
    cfg.setdefault('train', {})
    if args.max_steps is not None:
        cfg['train']['max_steps'] = args.max_steps
    if args.no_wandb or args.smoke:
        cfg.setdefault('wandb', {})
        cfg['wandb']['enabled'] = False


def _print_stage_summary(result: dict, elapsed: float) -> None:
    """1段の要約(stage/task/seeds と各指標 mean±std, 所要秒)を表示する。"""
    print(f"stage={result['stage']} task={result['task']} "
          f"seeds={result['seeds']} elapsed={elapsed:.1f}s")
    summary = result['summary']
    for metric in _REPORT_METRICS:
        stat = summary.get(metric)
        if stat is None:
            continue
        print(f"  {metric}: {stat['mean']:.4f} ± {stat['std']:.4f}")
    print(f"  summary_csv: {result['summary_csv']}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='attention梯子を1プロセスで順次実行するドライバ'
    )
    parser.add_argument('--configs', required=True,
                        help='カンマ区切りのconfig名(例 "l0_shakespeare,l1_shakespeare")')
    parser.add_argument('--out', default=_DEFAULT_OUT, help='出力ルート')
    parser.add_argument('--seeds', default=None,
                        help='"0,1,2" 形式(省略時は各cfgのseeds)')
    parser.add_argument('--smoke', action='store_true',
                        help='tiny学習+wandb無効(スモークテスト)')
    parser.add_argument('--no-wandb', action='store_true', help='wandbを無効化')
    parser.add_argument('--max-steps', type=int, default=None,
                        help='全configのmax_stepsを上書き')
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    load_dotenv()

    names = [n.strip() for n in args.configs.split(',') if n.strip()]
    seeds = _parse_seeds(args.seeds)

    # (config名, 成否, メッセージ) を貯めて最後に全段の成否を要約する
    outcomes: list[tuple[str, bool, str]] = []

    for idx, name in enumerate(names, start=1):
        print(f"\n========== [{idx}/{len(names)}] config={name} ==========")
        started = time.perf_counter()
        try:
            config_path = _resolve_config_name(name)
            with open(config_path, encoding='utf-8') as f:
                cfg = yaml.safe_load(f)
            _apply_overrides(cfg, args)

            result = run_stage(
                cfg, out_root=args.out, seeds=seeds, smoke=args.smoke,
            )
            elapsed = time.perf_counter() - started
            _print_stage_summary(result, elapsed)
            outcomes.append((name, True, f"stage={result['stage']}"))
        except Exception as e:  # noqa: BLE001 (1段失敗しても次段へ進む)
            elapsed = time.perf_counter() - started
            print(f"[FAIL] config={name} ({elapsed:.1f}s): {e}")
            traceback.print_exc()
            outcomes.append((name, False, f'{type(e).__name__}: {e}'))

    # --- 全段の成否要約 ---
    n_ok = sum(1 for _, ok, _ in outcomes if ok)
    print(f"\n========== ladder summary ({n_ok}/{len(outcomes)} ok) ==========")
    for name, ok, msg in outcomes:
        mark = 'OK  ' if ok else 'FAIL'
        print(f"  [{mark}] {name}: {msg}")

    # 1段でも失敗があれば非0終了(Modalジョブが失敗を検知できる)
    return 0 if n_ok == len(outcomes) else 1


if __name__ == '__main__':
    sys.exit(main())
