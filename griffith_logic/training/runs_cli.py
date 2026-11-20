"""run比較集計CLI。

griffith runs list    — .wandb_runs.jsonl(experiments/wandb_utils.pyが追記)を
                  markdown表で表示する(wandb通信なし)
griffith runs compare — wandb公開API(wandb.Api)でrun summary/configを取得して
                  比較表を生成。APIキー無しはjsonlの情報のみへ縮退する
"""

from __future__ import annotations

import json
import os
from pathlib import Path

RUNS_JSONL = '.wandb_runs.jsonl'

# 比較表から除外するconfigキー(wandb内部・環境情報)
_CONFIG_SKIP_PREFIXES = ('_', 'gpu_', 'cuda_', 'params/', 'gates/')
_CONFIG_SKIP_KEYS = {'device'}


def read_runs_jsonl(path: str | Path | None = None) -> list[dict]:
    """.wandb_runs.jsonl を読む(既定はカレントディレクトリ)。壊れた行は無視。"""
    p = Path(path) if path is not None else Path.cwd() / RUNS_JSONL
    if not p.is_file():
        return []
    entries = []
    for line in p.read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return entries


def format_runs_table(entries: list[dict]) -> str:
    """run一覧のmarkdown表を組み立てる。"""
    lines = [
        '| run_id | name | group | started_at |',
        '|--------|------|-------|------------|',
    ]
    for e in entries:
        lines.append(
            f'| {e.get("run_id", "-")} | {e.get("run_name", "-")} '
            f'| {e.get("group") or "-"} | {e.get("started_at", "-")} |'
        )
    return '\n'.join(lines)


def cmd_list(n: int = 20, jsonl_path: str | None = None) -> int:
    """griffith runs list: 新しい順にn件表示する。"""
    entries = read_runs_jsonl(jsonl_path)
    if not entries:
        print(f'{RUNS_JSONL} が見つからないか空です。'
              'wandb有効で学習すると自動追記されます(experiments/wandb_utils.py)')
        return 0
    latest = list(reversed(entries))[:n]
    print(format_runs_table(latest))
    return 0


# ============================================================
# griffith runs compare
# ============================================================

def _wandb_api_available() -> bool:
    """wandb公開APIが使えるか(実通信なしの事前判定)。"""
    if not os.environ.get('WANDB_API_KEY'):
        return False
    try:
        import wandb  # noqa: F401
    except ImportError:
        return False
    return True


def _fetch_run_rows(run_ids: list[str]) -> list[dict]:
    """wandb.Api()でrunごとのsummary/configを取得する。取得失敗はerrorキー。"""
    import wandb

    api = wandb.Api()
    project = os.environ.get('WANDB_PROJECT', 'griffith-logic')
    entity = os.environ.get('WANDB_ENTITY') or api.default_entity

    rows = []
    for rid in run_ids:
        try:
            run = api.run(f'{entity}/{project}/{rid}')
            summary = {
                k: v for k, v in dict(run.summary).items()
                if isinstance(v, int | float) and not k.startswith('_')
            }
            config = {
                k: v for k, v in dict(run.config).items()
                if not k.startswith(_CONFIG_SKIP_PREFIXES)
                and k not in _CONFIG_SKIP_KEYS
                and isinstance(v, int | float | str | bool)
            }
            rows.append({
                'run_id': rid, 'name': run.name, 'state': run.state,
                'summary': summary, 'config': config,
            })
        except Exception as e:  # noqa: BLE001 — 1件の取得失敗で全体を止めない
            rows.append({'run_id': rid, 'error': str(e)})
    return rows


def _format_value(v) -> str:
    if isinstance(v, float):
        return f'{v:.4f}'
    return str(v)


def build_compare_table(
    rows: list[dict], metrics: list[str] | None = None, fmt: str = 'md',
) -> str:
    """run比較表(行=メトリクス/差分config、列=run)を組み立てる。"""
    ok_rows = [r for r in rows if 'error' not in r]
    header = ['項目'] + [r['run_id'] for r in rows]

    # メトリクス行: --metrics指定はその順、省略時は全summaryキーの和集合
    if metrics:
        metric_keys = metrics
    else:
        metric_keys = sorted({k for r in ok_rows for k in r['summary']})

    body: list[list[str]] = []
    body.append(['name'] + [r.get('name', '-') for r in rows])
    body.append(['state'] + [r.get('state', r.get('error', '-')) for r in rows])
    for key in metric_keys:
        body.append(
            [key]
            + [_format_value(r['summary'].get(key, '-')) if 'error' not in r else '-'
               for r in rows]
        )

    # config差分行: runs間で値が異なるキーのみ
    config_keys = sorted({k for r in ok_rows for k in r['config']})
    for key in config_keys:
        values = [r['config'].get(key) for r in ok_rows]
        if len({json.dumps(v, sort_keys=True) for v in values}) <= 1:
            continue
        body.append(
            [f'config: {key}']
            + [_format_value(r['config'].get(key, '-')) if 'error' not in r else '-'
               for r in rows]
        )

    if fmt == 'csv':
        lines = [','.join(header)]
        lines += [','.join(str(c) for c in row) for row in body]
        return '\n'.join(lines)

    lines = ['| ' + ' | '.join(header) + ' |']
    lines.append('|' + '|'.join('---' for _ in header) + '|')
    lines += ['| ' + ' | '.join(str(c) for c in row) + ' |' for row in body]
    return '\n'.join(lines)


def build_degraded_table(run_ids: list[str], fmt: str = 'md') -> str:
    """縮退動作: jsonlの情報のみでrun一覧表を作る(WANDB_API_KEY無し)。"""
    index = {e.get('run_id'): e for e in read_runs_jsonl()}
    header = ['run_id', 'name', 'group', 'tags', 'started_at']
    body = []
    for rid in run_ids:
        e = index.get(rid)
        if e is None:
            body.append([rid, '(jsonl未記録)', '-', '-', '-'])
        else:
            body.append([
                rid, e.get('run_name', '-'), e.get('group') or '-',
                ' '.join(e.get('tags') or []) or '-', e.get('started_at', '-'),
            ])
    if fmt == 'csv':
        return '\n'.join([','.join(header)] + [','.join(row) for row in body])
    lines = ['| ' + ' | '.join(header) + ' |']
    lines.append('|' + '|'.join('---' for _ in header) + '|')
    lines += ['| ' + ' | '.join(row) + ' |' for row in body]
    return '\n'.join(lines)


def cmd_compare(
    run_ids: list[str], metrics: list[str] | None = None, fmt: str = 'md',
) -> int:
    """griffith runs compare: wandb APIで比較表を生成(キー無しはjsonl縮退)。"""
    if not _wandb_api_available():
        print('WANDB_API_KEY未設定のためwandb APIは使わず、'
              '.wandb_runs.jsonl の情報のみで表示します(縮退動作)')
        print(build_degraded_table(run_ids, fmt))
        return 0
    rows = _fetch_run_rows(run_ids)
    print(build_compare_table(rows, metrics, fmt))
    return 0
