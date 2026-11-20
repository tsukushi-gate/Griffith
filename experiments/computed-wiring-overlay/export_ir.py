"""学習済みの計算型配線ネットを IR へ書き出し、配線が閉形式どおりか全数検証する。

overlay(mdl-fpga)側は配線表を持たない。ゲート記述は op(4 bit)だけで、
前層アドレスは「層ごとのずらし量の剰余アキュムレータ」で作る。したがって
**IR の indices が本当に閉形式どおりか**を渡す前に確かめておく必要がある。

やること:
    1. run の config.yaml + チェックポイントからモデルを組み直す(データは読まない)
    2. export_ir → validate(target='hdl') → save_ir で <name>.ir.json を書く
       IR の metadata['wiring'] に層ごとの kind / s_a / s_b を併記する
    3. 各 lut_layer の indices を閉形式で作り直して**全数一致**を確かめる
           rot           : idx_a[g] = (g + s_a) mod W_in, idx_b[g] = (g + s_b) mod W_in
           stretched_rot : base[g] = (2*g*W_in + W_out) // (2*W_out) を上の g の代わりに使う
       閉形式で書けない層(perm / 乱択)は「規則なし」と記録する(overlay に載らない)
    4. 検証結果を <name>.ir_verification.json に残す

使用例:
    uv run python experiments/computed-wiring-overlay/export_ir.py \
        outputs/computed-wiring-overlay/cwo_mnist_48k_p6_s1
    uv run python experiments/computed-wiring-overlay/export_ir.py <run_dir> \
        --checkpoint best --out-dir build/ir

既定のチェックポイントは latest(= 最終エポック)。C3 の判定指標が最終エポックの
hard_acc なので、報告した精度と同じ重みを書き出すため。
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import torch

JST = timezone(timedelta(hours=9))

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(REPO_ROOT))

from griffith_logic.datasets.vision import VISION_DATASETS  # noqa: E402
from griffith_logic.export import export_ir  # noqa: E402
from griffith_logic.ir.serialize import save_ir  # noqa: E402
from griffith_logic.ir.validate import validate  # noqa: E402
from griffith_logic.training.config import TrainConfig, load_config  # noqa: E402
from griffith_logic.training.runner import ResolvedData, _build_model  # noqa: E402


def closed_form_indices(kind: str, in_dim: int, out_dim: int,
                        s_a: int, s_b: int) -> np.ndarray:
    """rot / stretched_rot の添字を numpy で素朴に組み直す(library とは独立の計算)。"""
    g = np.arange(out_dim, dtype=np.int64)
    if kind == 'rot':
        base = g
    elif kind == 'stretched_rot':
        # 四捨五入(半整数は切り上げ)を整数演算で: floor(g·in/out + 1/2)
        base = (2 * g * in_dim + out_dim) // (2 * out_dim)
    else:
        raise ValueError(f'閉形式を持たない配線種: {kind}')
    return np.stack([(base + s_a) % in_dim, (base + s_b) % in_dim])


def layer_specs(cfg: TrainConfig) -> list[dict]:
    """設定から層ごとの配線指定 {kind, s_a, s_b} を取り出す(IR の metadata 用)。"""
    conn = cfg.model.connections
    if isinstance(conn, str):
        return [{'kind': conn, 's_a': None, 's_b': None}
                for _ in range(cfg.model.layers)]
    if conn.kind == 'per_layer':
        return [{'kind': layer.kind, 's_a': layer.s_a, 's_b': layer.s_b}
                for layer in conn.layers]
    if conn.kind == 'rot':
        return [{'kind': 'rot', 's_a': conn.s_a[i], 's_b': conn.s_b[i]}
                for i in range(cfg.model.layers)]
    return [{'kind': conn.kind, 's_a': None, 's_b': None}
            for _ in range(cfg.model.layers)]


def load_model(run_dir: Path, checkpoint: str):
    """run の config.yaml + チェックポイントからモデルを組み直す(CPU、データ読込なし)。"""
    config_path = run_dir / 'config.yaml'
    if not config_path.is_file():
        raise FileNotFoundError(f'config.yaml がありません: {config_path}')
    cfg = load_config(str(config_path), [])

    spec = VISION_DATASETS.get(cfg.data.name)
    if spec is None:
        raise ValueError(
            f'視覚データセットのメタ情報がありません: {cfg.data.name}')
    data = ResolvedData(
        name=spec.name, kind='classification', in_width=spec.in_width,
        out_width=spec.classes, train_loader=None, eval_loader=None,
    )
    model = _build_model(cfg.model, data, 'cpu', seed=cfg.run.seed)

    ckpt_path = run_dir / 'checkpoints' / f'{checkpoint}.pt'
    if not ckpt_path.is_file():
        raise FileNotFoundError(f'チェックポイントがありません: {ckpt_path}')
    blob = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    model.load_state_dict(blob['model_state_dict'])
    model.eval()
    return cfg, data, model, ckpt_path, blob


def verify_layers(graph, specs: list[dict]) -> list[dict]:
    """IR の lut_layer の indices を閉形式と全数照合する。"""
    lut_nodes = [n for n in graph.nodes if n.type == 'lut_layer']
    if len(lut_nodes) != len(specs):
        raise ValueError(
            f'IR の lut_layer 数({len(lut_nodes)})と設定の層数({len(specs)})が違う')
    rows = []
    for i, (node, spec) in enumerate(zip(lut_nodes, specs, strict=True)):
        indices = np.asarray(node.params['indices'], dtype=np.int64)
        in_dim = int(node.params['in_width'])
        out_dim = int(node.params['out_width'])
        row = {
            'layer': i,
            'node_id': node.id,
            'kind': spec['kind'],
            's_a': spec['s_a'],
            's_b': spec['s_b'],
            'in_width': in_dim,
            'out_width': out_dim,
            'n_inputs': int(node.params['n_inputs']),
            'n_gates': out_dim,
        }
        if spec['kind'] in ('rot', 'stretched_rot'):
            expected = closed_form_indices(
                spec['kind'], in_dim, out_dim, int(spec['s_a']), int(spec['s_b'])
            )
            matched = int((indices == expected).sum())
            row.update({
                'closed_form': True,
                'slots': int(expected.size),
                'matched': matched,
                'all_match': matched == expected.size,
                'pair_difference': int((int(spec['s_b']) - int(spec['s_a']))
                                       % in_dim),
            })
        else:
            row.update({
                'closed_form': False,
                'slots': int(indices.size),
                'matched': None,
                'all_match': None,
                'note': '閉形式を持たない配線(overlay のアドレス生成では作れない)',
            })
        rows.append(row)
    return rows


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description='学習済みネットを IR へ書き出し、配線が閉形式どおりか検証する')
    parser.add_argument('run_dir', help='griffith train / run_calibration.py の出力ディレクトリ')
    parser.add_argument('--checkpoint', default='latest', choices=['latest', 'best'],
                        help='書き出すチェックポイント(既定: latest = 最終エポック)')
    parser.add_argument('--out-dir', help='IR の出力先(既定: run_dir)')
    parser.add_argument('--name', help='IR の名前(既定: run_dir 名)')
    args = parser.parse_args(argv)

    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir) if args.out_dir else run_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    name = args.name or run_dir.name

    cfg, data, model, ckpt_path, blob = load_model(run_dir, args.checkpoint)
    specs = layer_specs(cfg)

    graph = export_ir(
        model,
        input_width=data.in_width,
        name=name,
        metadata={
            'wiring': {
                'source': 'computed-wiring-overlay',
                # ASCII のみ(ir.serialize.save_ir はロケール既定の encoding で
                # 書くので、非 ASCII の metadata は cp932 環境で落ちる)
                'rule': ('rot: idx[g] = (g + s) mod W_in / stretched_rot: '
                         'idx[g] = ((2*g*W_in + W_out) // (2*W_out) + s) '
                         'mod W_in'),
                'layers': specs,
            },
            'run': {
                'run_name': cfg.run.run_name,
                'seed': cfg.run.seed,
                'dataset': cfg.data.name,
                'checkpoint': str(ckpt_path),
                'step': blob.get('step'),
                'best_hard_acc': blob.get('best_metric'),
            },
        },
    )
    warnings = validate(graph, target='hdl')
    ir_path = save_ir(graph, str(out_dir / f'{name}.ir.json'))
    rows = verify_layers(graph, specs)

    checked = [r for r in rows if r['closed_form']]
    all_match = bool(checked) and all(r['all_match'] for r in checked)
    report = {
        'generated_at': datetime.now(JST).isoformat(),
        'ir_path': ir_path,
        'run_dir': str(run_dir),
        'checkpoint': str(ckpt_path),
        'validate_target': 'hdl',
        'validate_warnings': warnings,
        'total_gates': sum(r['n_gates'] for r in rows),
        'layers': rows,
        'closed_form_layers': len(checked),
        'all_indices_match_closed_form': all_match,
        'metadata_wiring': graph.metadata['wiring'],
    }
    report_path = out_dir / f'{name}.ir_verification.json'
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')

    print(f'IR: {ir_path}')
    print(f'検証: {report_path}')
    print(f'層数 {len(rows)} / ゲート {report["total_gates"]:,} / '
          f'validate(hdl) 警告 {len(warnings)} 件')
    for row in rows:
        if row['closed_form']:
            print(f'  層 {row["layer"]} {row["kind"]}(s_a={row["s_a"]}, '
                  f's_b={row["s_b"]}) {row["in_width"]}→{row["out_width"]}: '
                  f'{row["matched"]}/{row["slots"]} 一致 '
                  f'{"OK" if row["all_match"] else "不一致あり"}')
        else:
            print(f'  層 {row["layer"]} {row["kind"]}: 閉形式なし(検証対象外)')
    if all_match:
        verdict = '閉形式を持つ全層が一致'
    elif not checked:
        verdict = ('閉形式を持つ層が無い'
                   '(この配線は overlay のアドレス生成では作れない)')
    else:
        verdict = '閉形式と一致しない層がある'
    print(f'判定: {verdict}')
    return 0 if all_match else 1


if __name__ == '__main__':
    sys.exit(main())
