"""MIXER-1: wandb group `mixer-1` のrunを集計してJSON/CSVへ落とす。

レポートの数値は必ずこの出力から転記する(手打ち・推測を禁止するための単一情報源)。

使い方:
    uv run python experiments/mixer1_collect.py
    uv run python experiments/mixer1_collect.py --group mixer-1 --out results/mixer1
"""

import argparse
import csv
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from env_utils import load_dotenv

load_dotenv()

# 収集するconfig/summaryのキー
_CONFIG_KEYS = (
    'run_tag', 'dataset', 'device', 'gpu_name', 'n_gates', 'n_params',
    'd_channels', 'depth', 'n_blocks', 'n_inputs', 'max_steps', 'lr',
    'batch_size', 'embed_hidden', 'embed_connections', 'embed_input_coverage',
    'head_group_width', 'head_input_coverage',
    'parameterization', 'seed',
    # 純MLP DLGN基準線(train.py)側の構成
    'num_neurons', 'num_layers', 'num_iterations',
    # CDDLGN(train_cnn.py)側の構成
    'tree_depth', 'k', 'n_thresholds', 'output_gate_factor',
)
_SUMMARY_KEYS = (
    'step', 'best_acc', 'hard_acc', 'soft_acc', 'discretization_gap',
    'train_loss', 'steps_per_sec', '_runtime',
)

# train.py(純MLP DLGN基準線)は別のメトリクス名を使うので対応付ける。
# train.py は test_acc(hard eval)と test_acc_soft を明示的に分けてログする。
_ALIASES = {
    'best_acc': 'best_test_acc',
    'hard_acc': 'test_acc',
    'soft_acc': 'test_acc_soft',
    'step': 'iteration',
    'train_loss': 'loss',
}


def main() -> None:
    parser = argparse.ArgumentParser(description='MIXER-1 run集計')
    parser.add_argument('--group', default='mixer-1', help='wandb group')
    # --out の既定は **--group から導出**する(results/<group名のハイフン除去>)。
    # 固定既定にしていると `--group mixer-2` だけ指定したときに mixer-1 の
    # 集計結果を上書きしてしまう(実際に一度やらかした)。
    parser.add_argument('--out', default=None,
                        help='出力ディレクトリ(既定は --group から導出: mixer-2 → results/mixer2)')
    parser.add_argument('--entity', default=os.environ.get('WANDB_ENTITY'))
    parser.add_argument('--project', default=os.environ.get('WANDB_PROJECT'))
    args = parser.parse_args()
    if args.out is None:
        args.out = os.path.join('results', args.group.replace('-', ''))

    import wandb

    api = wandb.Api()
    path = f'{args.entity}/{args.project}'
    runs = api.runs(path, filters={'group': args.group})

    def _config_of(run) -> dict:
        """runのconfigを取り出す。

        filters付き api.runs() が返すrunは config が空dictのことがあるため、
        その場合は run idで単体取得し直して完全なconfigを得る。
        """
        cfg = dict(run.config or {})
        if cfg:
            return cfg
        return dict(api.run(f'{path}/{run.id}').config or {})

    rows = []
    for run in runs:
        cfg = _config_of(run)
        summ = dict(run.summary)
        row = {'run_id': run.id, 'name': run.name, 'state': run.state}
        for k in _CONFIG_KEYS:
            row[k] = cfg.get(k)
        # train.py(純MLP DLGN基準線)は n_gates を持たないので gates/total で補う。
        # ゲート数 = num_neurons × num_layers。
        if row.get('n_gates') is None:
            row['n_gates'] = cfg.get('gates/total')
        # 同様に run_tag が無いスクリプトは run名から識別子を作る
        if row.get('run_tag') is None:
            row['run_tag'] = run.name.rsplit('_', 2)[0] if run.name else None
        # モデル種別を明示する。3スクリプトを取り違えないこと:
        #   train_mlp_mixer.py → d_channels を持つ      → mixer
        #   train_cnn.py(CDDLGN) → tree_depth を持つ    → cddlgn
        #   train.py(純MLP DLGN) → num_neurons を持つ   → mlp_dlgn
        if cfg.get('d_channels'):
            row['model_kind'] = 'mixer'
        elif cfg.get('tree_depth') is not None:
            row['model_kind'] = 'cddlgn'
        elif cfg.get('num_neurons') is not None:
            row['model_kind'] = 'mlp_dlgn'
        else:
            row['model_kind'] = 'unknown'
        for k in _SUMMARY_KEYS:
            v = summ.get(k)
            if v is None and k in _ALIASES:
                v = summ.get(_ALIASES[k])
            row[k] = v if isinstance(v, int | float | str | None) else str(v)
        # soft/hardが揃っていればgapを補完する(train.py側はgapをログしない)
        if (row.get('discretization_gap') is None
                and isinstance(row.get('soft_acc'), float)
                and isinstance(row.get('hard_acc'), float)):
            row['discretization_gap'] = row['soft_acc'] - row['hard_acc']
        rows.append(row)

    # ゲート数昇順(未取得はNoneなので後ろへ)
    rows.sort(key=lambda r: (r.get('n_gates') is None, r.get('n_gates') or 0))

    os.makedirs(args.out, exist_ok=True)
    json_path = os.path.join(args.out, 'runs.json')
    with open(json_path, 'w') as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    csv_path = os.path.join(args.out, 'runs.csv')
    fields = ['run_id', 'name', 'state', 'model_kind', *_CONFIG_KEYS, *_SUMMARY_KEYS]
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)

    def _fmt(value, spec: str) -> str:
        """数値なら書式化、欠損なら '?' を返す。"""
        if isinstance(value, int | float) and not isinstance(value, bool):
            return format(value, spec)
        return '?'

    hdr = (f'{"run_tag":26} {"gates":>6} {"cov":>5} {"dev":>4} {"state":>8} '
           f'{"step":>6} {"best_hard":>9} {"soft":>7} {"gap":>7}')
    print(hdr)
    print('-' * len(hdr))
    for r in rows:
        print(
            f'{str(r.get("run_tag")):26} {str(r.get("n_gates")):>6} '
            f'{_fmt(r.get("embed_input_coverage"), ".3f"):>5} '
            f'{str(r.get("device")):>4} {r["state"]:>8} '
            f'{str(r.get("step")):>6} '
            f'{_fmt(r.get("best_acc"), ".4f"):>9} '
            f'{_fmt(r.get("soft_acc"), ".4f"):>7} '
            f'{_fmt(r.get("discretization_gap"), "+.4f"):>7}'
        )
    print(f'\n保存: {json_path} / {csv_path}  ({len(rows)} runs)')


if __name__ == '__main__':
    main()
