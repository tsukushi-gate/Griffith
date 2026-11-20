"""MIXER-1: ローカルMPS runの進捗をチェックポイントから読む(1行/run)。

train_mlp_mixer.py の標準出力はファイルへリダイレクトするとブロックバッファされ、
step行がすぐ見えない。latest.pt は評価ごとに保存されるので、そこから
step / best_hard_acc / 構成を読んで進捗を出す。

使い方: uv run python experiments/mixer1_local_progress.py
"""

import glob
import os
import sys
from datetime import datetime, timedelta, timezone

JST = timezone(timedelta(hours=9))

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def main() -> None:
    paths = sorted(glob.glob('checkpoints/mlp_mixer_*/latest.pt'))
    if not paths:
        print('(latest.pt が見つかりません)')
        return
    now = datetime.now(JST)
    for path in paths:
        try:
            data = torch.load(path, map_location='cpu', weights_only=False)
        except Exception as e:  # 保存中の競合等は握りつぶして次へ
            print(f'{path}: 読めません ({type(e).__name__})')
            continue
        meta = data.get('metadata', {})
        args = meta.get('args', {})
        cfg = meta.get('config', {})
        tag = args.get('run_tag') or os.path.basename(os.path.dirname(path))
        age = now - datetime.fromtimestamp(os.path.getmtime(path), JST)
        print(
            f'{tag:24} step={data.get("step", "?"):>6}/{args.get("max_steps", "?")} '
            f'best_hard={data.get("best_metric", float("nan")):.4f} '
            f'gates={meta.get("n_gates", "?"):>6} '
            f'dc={cfg.get("d_channels", "?")} depth={cfg.get("depth", "?")} '
            f'nb={cfg.get("n_blocks", "?")} '
            f'更新={int(age.total_seconds())}秒前'
        )


if __name__ == '__main__':
    main()
