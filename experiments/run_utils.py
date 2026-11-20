"""統合ラン管理ユーティリティ。

ランごとに全出力（チェックポイント、ログ、メトリクス、可視化）を
1つのディレクトリにまとめて管理する。

ディレクトリ構造:
    outputs/{YYYYMMDD_HHMMSS}_{run_name}/
    ├── config.json              # ハイパーパラメータ・モデル構成
    ├── checkpoints/
    │   ├── best.pt
    │   └── latest.pt
    ├── logs/
    │   └── train.log
    ├── metrics.jsonl            # 1行1評価ステップ (JSON Lines)
    └── visualizations/          # 可視化画像
"""

import json
import os
from datetime import datetime, timedelta, timezone

JST = timezone(timedelta(hours=9))


class RunDir:
    """統合ランディレクトリの管理。"""

    def __init__(self, run_name: str, base_dir: str = 'outputs'):
        timestamp = datetime.now(JST).strftime('%Y%m%d_%H%M%S')
        self.name = f'{timestamp}_{run_name}'
        self.root = os.path.join(base_dir, self.name)
        self.ckpt_dir = os.path.join(self.root, 'checkpoints')
        self.log_dir = os.path.join(self.root, 'logs')
        self.vis_dir = os.path.join(self.root, 'visualizations')

        self.latest_path = os.path.join(self.ckpt_dir, 'latest.pt')
        self.best_path = os.path.join(self.ckpt_dir, 'best.pt')
        self.config_path = os.path.join(self.root, 'config.json')
        self.metrics_path = os.path.join(self.root, 'metrics.jsonl')
        self.log_path = os.path.join(self.log_dir, 'train.log')

        for d in [self.ckpt_dir, self.log_dir, self.vis_dir]:
            os.makedirs(d, exist_ok=True)

        self._log_file = open(self.log_path, 'a', buffering=1)

    def save_config(self, config: dict) -> None:
        """ハイパーパラメータ・モデル構成をJSONで保存。"""
        serializable = {}
        for k, v in config.items():
            try:
                json.dumps(v)
                serializable[k] = v
            except (TypeError, ValueError):
                serializable[k] = str(v)
        with open(self.config_path, 'w') as f:
            json.dump(serializable, f, indent=2, ensure_ascii=False)

    def log(self, msg: str) -> None:
        """タイムスタンプ付きでログを書き込む。"""
        ts = datetime.now(JST).strftime('%Y-%m-%d %H:%M:%S')
        self._log_file.write(f'[{ts}] {msg}\n')

    def log_metrics(self, metrics: dict) -> None:
        """メトリクスをJSON Linesに追記。"""
        with open(self.metrics_path, 'a') as f:
            f.write(json.dumps(metrics, ensure_ascii=False) + '\n')

    def save_visualizations(self, images: list, prefix: str,
                            step: int | str) -> list[str]:
        """可視化画像を保存。

        Args:
            images: PIL.Imageのリスト
            prefix: ファイル名のプレフィックス（例: 'hard', 'soft'）
            step: ステップ番号またはラベル

        Returns:
            保存したファイルパスのリスト
        """
        paths = []
        for i, img in enumerate(images):
            filename = f'{prefix}_step{step}_{i}.png'
            path = os.path.join(self.vis_dir, filename)
            img.save(path)
            paths.append(path)
        return paths

    def save_wandb_run_id(self, run_id: str) -> None:
        """wandbのrun IDを保存して紐付ける。"""
        with open(os.path.join(self.root, 'wandb_run_id.txt'), 'w') as f:
            f.write(run_id + '\n')

    def close(self) -> None:
        """リソースを解放する。"""
        self._log_file.close()

    def __repr__(self) -> str:
        return f'RunDir({self.root})'
