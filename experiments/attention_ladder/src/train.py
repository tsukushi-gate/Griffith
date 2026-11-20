"""attention梯子 L0 のCLIエントリ。

使用例:
    uv run python experiments/attention_ladder/src/train.py \
        --config experiments/attention_ladder/configs/l0_smnist.yaml
    uv run python experiments/attention_ladder/src/train.py \
        --config .../configs/l0_smnist.yaml --smoke --no-wandb
"""

from __future__ import annotations

import argparse
import os
import sys

# ------------------------------------------------------------
# import規約: src/ と experiments/ をパス追加(harnessと同じ流儀)
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

# 既定はcwd相対(Modal規約: modal_app.py が os.chdir('/workspace') するため
# 相対パス出力が永続Volumeに載る。ローカルはリポジトリルートから実行する前提)
_DEFAULT_OUT = os.path.join('experiments', 'attention_ladder', 'results')
# ソースツリーのルート(ローカル=リポジトリルート / Modal=/root)。config解決に使う
_SRC_REPO_ROOT = os.path.dirname(_EXP_DIR)


def _parse_seeds(text: str | None) -> list[int] | None:
    """"0,1,2" 形式のシード列をパースする。省略時はNone。"""
    if not text:
        return None
    return [int(s.strip()) for s in text.split(',') if s.strip()]


def _resolve_config(path: str) -> str:
    """config パスを解決する。

    Modalでは cwd=/workspace だがソース実体は /root にあるため、cwd相対で
    見つからない場合はソースツリー(_SRC_REPO_ROOT)基準で再解決する。同じ
    `--config experiments/attention_ladder/configs/*.yaml` 引数がローカルでも
    Modalでも通るようにする。
    """
    if os.path.isfile(path):
        return path
    alt = os.path.join(_SRC_REPO_ROOT, path)
    if os.path.isfile(alt):
        return alt
    raise FileNotFoundError(f'config が見つかりません: {path} (alt: {alt})')


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='attention梯子 L0 学習')
    parser.add_argument('--config', required=True, help='YAML設定ファイル')
    parser.add_argument('--seeds', default=None,
                        help='"0,1,2" 形式(省略時はcfgのseeds)')
    parser.add_argument('--smoke', action='store_true',
                        help='tiny学習+wandb無効(スモークテスト)')
    parser.add_argument('--no-wandb', action='store_true', help='wandbを無効化')
    parser.add_argument('--out', default=_DEFAULT_OUT, help='出力ルート')
    parser.add_argument('--epochs', type=int, default=None,
                        help='cfgのepochsを上書き')
    parser.add_argument('--max-steps', type=int, default=None,
                        help='cfgのmax_stepsを上書き')
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    load_dotenv()

    config_path = _resolve_config(args.config)
    with open(config_path, encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    # CLIによるcfg上書き
    cfg.setdefault('train', {})
    if args.epochs is not None:
        cfg['train']['epochs'] = args.epochs
    if args.max_steps is not None:
        cfg['train']['max_steps'] = args.max_steps
    if args.no_wandb or args.smoke:
        cfg.setdefault('wandb', {})
        cfg['wandb']['enabled'] = False

    seeds = _parse_seeds(args.seeds)

    result = run_stage(cfg, out_root=args.out, seeds=seeds, smoke=args.smoke)

    # --- 結果サマリの表示 ---
    print('\n=== stage summary ===')
    print(f"stage={result['stage']} task={result['task']} seeds={result['seeds']}")
    for metric, stat in result['summary'].items():
        print(f"  {metric}: {stat['mean']:.4f} ± {stat['std']:.4f}")
    print(f"summary_csv: {result['summary_csv']}")


if __name__ == '__main__':
    main()
