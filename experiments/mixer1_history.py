"""MIXER-1: wandbから学習曲線(hard_acc vs step)の履歴を取得してJSONへ保存する。

`mixer1_collect.py` が最終値だけを集めるのに対し、こちらは時系列を取る。
「予算を増やせばまだ伸びるか」を読むための一次データ。

使い方:
    uv run python experiments/mixer1_history.py
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from env_utils import load_dotenv

load_dotenv()

MNIST_TRAIN_SIZE = 60000

# 曲線に載せるrun(run_tag)。Mixer主要構成 + 同条件の純MLP対照。
_WANTED_PREFIXES = (
    'dc64_d2', 'dc256_d2', 'dc384_d4', 'dc512_d4',
    'e_dc64_d2_head128', 'e_dc256_d2_head128', 'e_dc256_d4_head64',
    'e_dc256_d2_head256', 'e_dc256_d2_head512', 'e_dc384_d4_head128',
    'g_dc256_d2_head128_s60k', 'g_dc384_d2_head256_s60k',
    'mnist_k',  # 純MLP DLGN基準線
)


def steps_to_epochs(step: int, batch_size: int, train_size: int = MNIST_TRAIN_SIZE) -> float:
    """step数をepoch換算する(1 epoch = train_size / batch_size step)。"""
    return step * batch_size / train_size


def main() -> None:
    parser = argparse.ArgumentParser(description='MIXER-1 学習曲線の取得')
    parser.add_argument('--group', default='mixer-1')
    parser.add_argument('--out', default='results/mixer1/history.json')
    parser.add_argument('--entity', default=os.environ.get('WANDB_ENTITY'))
    parser.add_argument('--project', default=os.environ.get('WANDB_PROJECT'))
    parser.add_argument('--samples', type=int, default=500)
    args = parser.parse_args()

    import wandb

    api = wandb.Api()
    path = f'{args.entity}/{args.project}'
    runs = api.runs(path, filters={'group': args.group})

    out = []
    for run in runs:
        full = api.run(f'{path}/{run.id}')
        cfg = full.config
        tag = cfg.get('run_tag') or (run.name.rsplit('_', 2)[0] if run.name else '')
        if not any(tag.startswith(p) for p in _WANTED_PREFIXES):
            continue
        if cfg.get('dataset') != 'mnist':
            continue

        # 3スクリプトを取り違えない(train_cnn.py の run名も 'mnist_k...' で始まるため、
        # d_channels の有無だけで判定すると CDDLGN が純MLP基準線に混ざる)
        if cfg.get('d_channels'):
            model_kind = 'mixer'
        elif cfg.get('tree_depth') is not None:
            model_kind = 'cddlgn'
        elif cfg.get('num_neurons') is not None:
            model_kind = 'mlp_dlgn'
        else:
            model_kind = 'unknown'
        batch_size = cfg.get('batch_size') or 128
        gates = cfg.get('n_gates') or cfg.get('gates/total')

        # mixer側は hard_acc / step、train.py側は test_acc / iteration
        keys = ['hard_acc', 'step', 'test_acc', 'iteration', '_step']
        points = []
        for row in full.scan_history(keys=None, page_size=1000):
            acc = row.get('hard_acc')
            if acc is None:
                acc = row.get('test_acc')
            step = row.get('step')
            if step is None:
                step = row.get('iteration')
            if acc is None or step is None:
                continue
            points.append({
                'step': int(step),
                'hard_acc': float(acc),
                'epoch': round(steps_to_epochs(int(step), batch_size), 3),
            })
        if not points:
            continue
        points.sort(key=lambda p: p['step'])

        out.append({
            'run_id': run.id, 'run_tag': tag, 'state': run.state,
            'model_kind': model_kind,
            'n_gates': gates, 'batch_size': batch_size,
            'head_group_width': cfg.get('head_group_width'),
            'd_channels': cfg.get('d_channels'),
            'max_steps': cfg.get('max_steps') or cfg.get('num_iterations'),
            'device': cfg.get('device'),
            'points': points,
        })
        print(f'{tag:28} gates={str(gates):>6} 点数={len(points):>4} '
              f'最終hard={points[-1]["hard_acc"]:.4f} '
              f'({points[-1]["epoch"]:.1f} epoch)')

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f'\n保存: {args.out}  ({len(out)} runs)')


if __name__ == '__main__':
    main()
