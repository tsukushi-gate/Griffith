"""wandbレポート自動生成。

griffith report <run_id...> [--title ...]

wandb-workspaces(旧wandb.apis.reports)が利用可能なら、学習曲線パネル+
run比較(config差分表相当のRun comparer)+結論欄のテンプレレポートを
wandb上に生成してURLを表示する。

ライブラリ未導入・APIキー無し・通信失敗時はmarkdownレポートを
reports/auto/<日付>_<title>.md へローカル生成する
(手書きレポートの構造を踏襲: 目的/設定/結果/結論。
メトリクス表は取得できる範囲で自動埋め、結論はTODOプレースホルダ)。
"""

from __future__ import annotations

import os
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .runs_cli import _wandb_api_available, build_compare_table, read_runs_jsonl

JST = timezone(timedelta(hours=9))

DEFAULT_DEST = 'reports/auto'

# 学習曲線パネルに載せる既定メトリクス(griffith train / エディタ両系の主要キー)
DEFAULT_PANEL_METRICS = ('loss', 'hard_acc', 'best_hard_acc')


def _slugify(title: str) -> str:
    """ファイル名に使えない文字を'-'へ置換する(日本語は保持)。"""
    return re.sub(r'[\s/\\:*?"<>|]+', '-', title.strip()).strip('-') or 'report'


def try_wandb_report(run_ids: list[str], title: str) -> str | None:
    """wandb-workspacesでレポートを生成しURLを返す。不可ならNone。

    ライブラリ未導入・APIキー無し・API変更等の失敗はすべてNone
    (markdownフォールバックへ)。
    """
    if not _wandb_api_available():
        return None
    try:
        import wandb_workspaces.reports.v2 as wr
    except ImportError:
        return None

    project = os.environ.get('WANDB_PROJECT', 'griffith-logic')
    entity = os.environ.get('WANDB_ENTITY')
    try:
        if not entity:
            import wandb

            entity = wandb.Api().default_entity
        runset = wr.Runset(
            entity=entity, project=project,
            filters=f'ID in {list(run_ids)!r}',
        )
        report = wr.Report(
            entity=entity, project=project, title=title,
            description=f'griffith report による自動生成(対象run: {", ".join(run_ids)})',
            blocks=[
                wr.H1('学習曲線'),
                wr.PanelGrid(
                    runsets=[runset],
                    panels=[
                        wr.LinePlot(x='step', y=[m]) for m in DEFAULT_PANEL_METRICS
                    ],
                ),
                wr.H1('config差分'),
                wr.PanelGrid(runsets=[runset], panels=[wr.RunComparer()]),
                wr.H1('結論'),
                wr.P('TODO: 仮説の検証結果と次のアクションを記述する'),
            ],
        )
        report.save()
        return report.url
    except Exception:  # noqa: BLE001 — 生成失敗はmarkdownフォールバックへ
        return None


def _metrics_section(run_ids: list[str]) -> str:
    """結果セクション: wandb APIが使えれば比較表を自動埋め、不可ならTODO。"""
    if _wandb_api_available():
        try:
            from .runs_cli import _fetch_run_rows

            return build_compare_table(_fetch_run_rows(run_ids), fmt='md')
        except Exception:  # noqa: BLE001 — 通信失敗時はプレースホルダへ
            pass
    return (
        'TODO: メトリクスを記入する(WANDB_API_KEY設定後に '
        '`griffith runs compare ' + ' '.join(run_ids) + '` で取得できる)'
    )


def _settings_section(run_ids: list[str]) -> str:
    """設定セクション: .wandb_runs.jsonl から分かる範囲のrun情報表。"""
    index = {e.get('run_id'): e for e in read_runs_jsonl()}
    lines = [
        '| run_id | name | group | started_at |',
        '|--------|------|-------|------------|',
    ]
    for rid in run_ids:
        e = index.get(rid) or {}
        lines.append(
            f'| {rid} | {e.get("run_name", "-")} | {e.get("group") or "-"} '
            f'| {e.get("started_at", "-")} |'
        )
    return '\n'.join(lines)


def build_markdown_report(run_ids: list[str], title: str) -> str:
    """markdownレポート本文を組み立てる(手書きレポートの構造踏襲)。"""
    today = datetime.now(JST).strftime('%Y-%m-%d')
    runs = ', '.join(run_ids)
    return f"""# {title}

> 生成日: {today}
> 対象run: {runs}
> 生成コマンド: `griffith report {' '.join(run_ids)} --title "{title}"`(自動生成)

---

## 1. 目的

TODO: この実験群の仮説・目的を記述する

## 2. 設定

{_settings_section(run_ids)}

各runの完全な設定は `outputs/<run>/config.yaml`(`griffith train` の保存先)または
wandbのrunページを参照。

## 3. 結果

{_metrics_section(run_ids)}

## 4. 結論

TODO: 仮説の検証結果・expected_metricsの照合結果・次のアクションを記述する
"""


def generate_markdown_report(
    run_ids: list[str], title: str, dest: str = DEFAULT_DEST,
) -> Path:
    """markdownレポートを reports/auto/<日付>_<title>.md へ書き出す。"""
    today = datetime.now(JST).strftime('%Y-%m-%d')
    dest_dir = Path(dest)
    dest_dir.mkdir(parents=True, exist_ok=True)
    path = dest_dir / f'{today}_{_slugify(title)}.md'
    path.write_text(build_markdown_report(run_ids, title), encoding='utf-8')
    return path


# ============================================================
# griffith report --group <g>: グループ横断のデータパック(非LLM・決定的)
# ============================================================

# 台帳(experiments-ledger.md)のパス。finish_wandb が追記する台帳を転記する。
LEDGER_PATH = 'docs/experiments-ledger.md'

# データパックに載せるメトリクスの表示順(先頭優先。残りは名前順で続ける)
_GROUP_METRIC_ORDER = ('val_loss', 'best_val_loss', 'hard_acc', 'eval_acc', 'loss')

# 縮退時にconfig差分列へ回す除外プレフィクス/キー(runs_cliと同基準)
_CONFIG_SKIP_PREFIXES = ('_', 'gpu_', 'cuda_', 'params/', 'gates/')
_CONFIG_SKIP_KEYS = {'device'}


def _fetch_group_run_rows(group: str, limit: int = 50) -> list[dict]:
    """wandb公開APIでgroup内の全run(最大limit件)を取得する。

    行の形は degraded(jsonl)経路と共通化する:
    {run_id, name, state, created_at, notes, hypothesis, summary, config}
    """
    import wandb

    api = wandb.Api()
    project = os.environ.get('WANDB_PROJECT', 'griffith-logic')
    entity = os.environ.get('WANDB_ENTITY') or api.default_entity
    runs = api.runs(f'{entity}/{project}', filters={'group': group})

    rows: list[dict] = []
    for run in runs:
        if len(rows) >= limit:
            break
        summary = {
            k: v for k, v in dict(run.summary).items()
            if isinstance(v, int | float) and not isinstance(v, bool)
            and not str(k).startswith('_')
        }
        config = {
            k: v for k, v in dict(run.config).items()
            if not str(k).startswith(_CONFIG_SKIP_PREFIXES)
            and k not in _CONFIG_SKIP_KEYS
            and isinstance(v, int | float | str | bool)
        }
        rows.append({
            'run_id': run.id,
            'name': run.name,
            'state': run.state,
            'created_at': str(getattr(run, 'created_at', '') or ''),
            'notes': run.notes or '',
            'hypothesis': dict(run.config).get('experiment/hypothesis', ''),
            'summary': summary,
            'config': config,
        })
    return rows


def _group_rows_from_jsonl(group: str) -> list[dict]:
    """WANDB_API_KEY無し時の縮退: .wandb_runs.jsonl からgroup内runを組み立てる。

    initレコードとfinishレコードをrun_idで突き合わせる。configは非記録のため空。
    """
    inits: dict[str, dict] = {}
    finishes: dict[str, dict] = {}
    for e in read_runs_jsonl():
        rid = e.get('run_id')
        if not rid:
            continue
        if e.get('event') == 'finish':
            finishes[rid] = e
        else:
            inits[rid] = e

    rows: list[dict] = []
    for rid, init in inits.items():
        if init.get('group') != group:
            continue
        fin = finishes.get(rid, {})
        final_metrics = fin.get('final_metrics') or {}
        summary = {
            k: v for k, v in final_metrics.items()
            if isinstance(v, int | float) and not isinstance(v, bool)
        }
        rows.append({
            'run_id': rid,
            'name': init.get('run_name', '-'),
            'state': fin.get('status', '-'),
            'created_at': init.get('started_at', ''),
            'notes': init.get('notes') or '',
            'hypothesis': init.get('hypothesis') or '',
            'summary': summary,
            'config': {},
        })
    return rows


def _md_table(header: list[str], body: list[list[str]]) -> str:
    """markdown表を組み立てる。"""
    lines = ['| ' + ' | '.join(header) + ' |']
    lines.append('|' + '|'.join('---' for _ in header) + '|')
    lines += ['| ' + ' | '.join(str(c) for c in row) + ' |' for row in body]
    return '\n'.join(lines)


def _fmt(v) -> str:
    if isinstance(v, float):
        return f'{v:.4f}'
    return str(v)


def _esc(v) -> str:
    """表セル用にパイプ・改行を無害化する。空は '-'。"""
    text = str(v) if v not in (None, '') else '-'
    return text.replace('|', '\\|').replace('\n', ' ').strip()


def _group_run_list_table(rows: list[dict]) -> str:
    body = [
        [_esc(r['run_id']), _esc(r['name']), _esc(r.get('created_at')),
         _esc(r.get('notes')), _esc(r.get('hypothesis'))]
        for r in rows
    ]
    return _md_table(['run ID', 'name', '日時', 'notes', 'hypothesis'], body)


def _ordered_metric_keys(rows: list[dict]) -> list[str]:
    all_keys = {k for r in rows for k in r['summary']}
    ordered = [k for k in _GROUP_METRIC_ORDER if k in all_keys]
    rest = sorted(all_keys - set(ordered))
    return ordered + rest


def _group_metrics_table(rows: list[dict]) -> str:
    keys = _ordered_metric_keys(rows)
    if not keys:
        return '(メトリクス未記録)'
    header = ['メトリクス'] + [r['run_id'] for r in rows]
    body = [
        [key] + [_fmt(r['summary'].get(key, '-')) for r in rows]
        for key in keys
    ]
    return _md_table(header, body)


def _group_config_diff_table(rows: list[dict]) -> str:
    config_keys = sorted({k for r in rows for k in r['config']})
    header = ['config'] + [r['run_id'] for r in rows]
    body: list[list[str]] = []
    for key in config_keys:
        values = [r['config'].get(key) for r in rows]
        # run間で値が一致するキーは差分でないので除外
        import json as _json
        if len({_json.dumps(v, sort_keys=True, default=str) for v in values}) <= 1:
            continue
        body.append([key] + [_fmt(r['config'].get(key, '-')) for r in rows])
    if not body:
        return '(config差分なし。縮退経路ではconfigは非記録)'
    return _md_table(header, body)


def _ledger_rows_for(run_ids: list[str], ledger_path: str = LEDGER_PATH) -> str:
    """台帳から対象run_idを含む行を転記する。無ければ注記を返す。"""
    p = Path(ledger_path)
    if not p.is_file():
        return '(台帳が見つかりません。訓練を1回完走すると自動生成されます)'
    idset = set(run_ids)
    matched = [
        line for line in p.read_text(encoding='utf-8').splitlines()
        if line.startswith('|') and any(rid in line for rid in idset)
    ]
    if not matched:
        return '(台帳に該当runの行がまだありません)'
    header = (
        '| 日付 | run ID | group | job_type | hypothesis | notes | final | 判定 |\n'
        '|------|--------|-------|----------|------------|-------|-------|------|'
    )
    return header + '\n' + '\n'.join(matched)


def build_group_data_pack(group: str, rows: list[dict], degraded: bool) -> str:
    """グループ横断のデータパック本文を組み立てる(非LLM・決定的)。"""
    today = datetime.now(JST).strftime('%Y-%m-%d')
    source = 'jsonl縮退(WANDB_API_KEY無し。configは非記録)' if degraded else 'wandb API'
    run_ids = [r['run_id'] for r in rows]
    return f"""# データパック: {group}

> 生成日: {today}
> group: `{group}` / run数: {len(rows)}(最大50)
> 生成コマンド: `griffith report --group {group}`(自動生成・非LLM)
> データ源: {source}

このファイルは experiment-report スキルの唯一の根拠(error_list)。
ここに無い数値をレポートに書いてはならない。

---

## 1. run一覧

{_group_run_list_table(rows)}

## 2. config差分(全run横断)

{_group_config_diff_table(rows)}

## 3. 最終メトリクス

{_group_metrics_table(rows)}

## 4. 実験台帳(該当行の転記)

{_ledger_rows_for(run_ids)}
"""


def generate_group_data_pack(group: str, dest: str = DEFAULT_DEST) -> Path:
    """データパックを reports/auto/<日付>_<group>.data.md へ書き出す。"""
    if _wandb_api_available():
        try:
            rows = _fetch_group_run_rows(group)
            degraded = False
        except Exception:  # noqa: BLE001 — 通信失敗はjsonl縮退へ
            rows = _group_rows_from_jsonl(group)
            degraded = True
    else:
        rows = _group_rows_from_jsonl(group)
        degraded = True

    today = datetime.now(JST).strftime('%Y-%m-%d')
    dest_dir = Path(dest)
    dest_dir.mkdir(parents=True, exist_ok=True)
    path = dest_dir / f'{today}_{_slugify(group)}.data.md'
    path.write_text(build_group_data_pack(group, rows, degraded), encoding='utf-8')
    return path


def cmd_report_group(group: str, dest: str = DEFAULT_DEST) -> int:
    """griffith report --group: グループ横断データパックを生成する。"""
    path = generate_group_data_pack(group, dest)
    print('データパックを生成しました(experiment-reportスキルの根拠にする):')
    print(str(path))
    return 0


def cmd_report(run_ids: list[str], title: str | None, dest: str = DEFAULT_DEST) -> int:
    """griffith report: wandbレポート生成、不可ならmarkdownフォールバック。"""
    title = title or f'実験レポート {"+".join(run_ids)}'
    url = try_wandb_report(run_ids, title)
    if url is not None:
        print(f'wandbレポートを生成しました: {url}')
        return 0
    path = generate_markdown_report(run_ids, title, dest)
    print('wandb-workspacesが利用できないためmarkdownレポートを生成しました'
          '(結論はTODOを埋めること):')
    print(str(path))
    return 0
