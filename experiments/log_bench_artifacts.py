"""ベンチの測定JSONをwandb Artifactとして記録する汎用ユーティリティ。

A/Bドライバ(bench_backward_w.py / bench_mixing_ab.py / bench_lut_kernel_ab.py)は
子プロセスを交互起動する構造上、自前でwandb runを持つと子のrunと二重になる。
そこで**測定と記録を分離**し、走り終えた結果JSONを本スクリプトでまとめて
Artifact 化する(「成果物はローカル保存だけにせず必ずwandbにも残す」の担保)。

wandbの認証はModalコンテナ側のsecretにあるため、通常はModal上で実行する:

    uv run modal run scripts/modal_app.py --script log_bench_artifacts.py \\
        --args "--name kernel2-backward-w-ab --group kernel-2 \\
                --files results/kernel2/backward_w_20260909_170533.json \\
                --metrics model/speedup=3.598"

パスは実行時のカレント(Modalでは /workspace)からの相対で解決する。
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def parse_metric(spec: str) -> tuple[str, float]:
    key, _, value = spec.partition('=')
    if not key or not value:
        raise argparse.ArgumentTypeError(f'--metrics は key=value 形式: {spec}')
    return key, float(value)


def parse_args():
    p = argparse.ArgumentParser(description='測定JSONをwandb Artifact化する')
    p.add_argument('--files', nargs='+', required=True, help='記録するファイル群')
    p.add_argument('--name', required=True, help='Artifact名 兼 run名')
    p.add_argument('--group', default='bench', help='wandb group')
    p.add_argument('--type', default='benchmark', help='Artifact種別')
    p.add_argument('--tags', nargs='*', default=[], help='追加タグ')
    p.add_argument('--metrics', nargs='*', default=[],
                   help='summaryへ入れる key=value(数値)を複数指定できる')
    p.add_argument('--notes', default=None)
    p.add_argument('--device', default='cuda')
    return p.parse_args()


def main():
    args = parse_args()
    from wandb_utils import finish, init_wandb, log_artifact, log_metrics

    missing = [f for f in args.files if not os.path.isfile(f)]
    if missing:
        raise SystemExit(f'ファイルが見つからない: {missing}')

    metrics = dict(parse_metric(m) for m in args.metrics)

    ok = init_wandb(
        run_name=args.name,
        config_dict={'artifact_name': args.name, 'files': args.files},
        device=args.device,
        tags=['bench', *args.tags],
        group=args.group,
        job_type='eval',
        notes=args.notes,
    )
    if not ok:
        raise SystemExit('wandbの初期化に失敗した(結果JSONはローカルに残っている)')

    if metrics:
        log_metrics(metrics)
    log_artifact(args.files, name=args.name, type=args.type,
                 metadata={'files': args.files})
    finish()
    print(f'wandbへ記録した: {args.name} ({len(args.files)} files)')


if __name__ == '__main__':
    main()
