"""エディタ学習runの永続レジストリ。

セッション終了時に outputs/editor_runs/<sid>.json へ
ハイパラ・metrics系列(間引き済み)・最終結果・IRスナップショットを保存し、
セッション横断のrun比較(RunsPanel)を可能にする。
"""

from __future__ import annotations

import json
import time
from pathlib import Path

# 保存する系列の最大点数(超過分は等間隔間引き)
SERIES_MAX_POINTS = 500

SUMMARY_KEYS = (
    'run_id', 'started_at', 'ended_at', 'state', 'dataset', 'method', 'loss',
    'optimizer', 'lr', 'batch_size', 'epochs', 'final_step', 'final_loss',
    'final_acc', 'model_name', 'wandb_url',
)


def _downsample(xs: list, max_points: int = SERIES_MAX_POINTS) -> list:
    if len(xs) <= max_points:
        return xs
    stride = len(xs) / max_points
    return [xs[int(i * stride)] for i in range(max_points)]


def build_record(session, ir_snapshot: dict | None) -> dict:
    """TrainingSessionから保存レコードを構築する。"""
    series = getattr(session, 'series', [])
    steps = _downsample([s[0] for s in series])
    loss = _downsample([s[1] for s in series])
    acc = _downsample([s[2] for s in series])
    metadata = (ir_snapshot or {}).get('metadata', {})
    return {
        'run_id': session.sid,
        'started_at': getattr(session, 'started_at', None),
        'ended_at': time.time(),
        'state': session.state,
        'dataset': session.dataset,
        'method': session.method,
        'loss': session.task.loss_name,
        'optimizer': getattr(session, 'optimizer_name', 'adam'),
        'lr': session.lr,
        'batch_size': session.batch_size,
        'epochs': session.epochs,
        'final_step': session.step,
        'final_loss': session.last_loss,
        'final_acc': session.last_acc,
        'model_name': str(metadata.get('name', 'model')),
        # wandb連携(opt-in)。未連携時はNone=RunsPanelのリンク非表示
        'wandb_url': getattr(session, 'wandb_url', None),
        'series': {'steps': steps, 'loss': loss, 'acc': acc},
        'ir': ir_snapshot,
    }


def save_run(record: dict, root: Path | str) -> Path:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{record['run_id']}.json"
    path.write_text(json.dumps(record, ensure_ascii=False))
    return path


def list_runs(root: Path | str) -> list[dict]:
    """runサマリ一覧(新しい順)。壊れたファイルはスキップする。"""
    root = Path(root)
    if not root.exists():
        return []
    out: list[dict] = []
    for path in root.glob('*.json'):
        try:
            doc = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        out.append({k: doc.get(k) for k in SUMMARY_KEYS})
    out.sort(key=lambda r: r.get('ended_at') or 0, reverse=True)
    return out


def load_run(root: Path | str, run_id: str) -> dict | None:
    path = Path(root) / f'{run_id}.json'
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None
