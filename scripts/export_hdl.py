"""学習済みチェックポイント → IR + Verilog エクスポートCLI。

使用例:
    # チェックポイントからIRとVerilogを生成
    uv run python scripts/export_hdl.py checkpoints/xxx/best.pt \\
        --arch "16000x4:n2" --in-width 400 --classes 10 -o out/

    # IR JSONから直接Verilog生成
    uv run python scripts/export_hdl.py model.json -o out/

    # iverilogで生成Verilogをgolden modelと照合(要iverilog)
    uv run python scripts/export_hdl.py model.json -o out/ --verify
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def build_model_from_checkpoint(ckpt_path: str, arch: str, in_width: int,
                                classes: int, tau: float,
                                allow_unsafe: bool = False):
    """--arch "WIDTHxLAYERS:nN[:param]" 形式からモデルを再構築してロード。

    チェックポイントは既定で weights_only=True で読む。
    テンソル以外の復元が必要な場合のみ --allow-unsafe-checkpoint を使う。
    """
    import torch.nn as nn

    from griffith_logic import GroupSum, LutLayer
    from griffith_logic.convert.teacher import load_checkpoint_safely

    spec, n_part = arch.split(':')[0], arch.split(':')[1]
    width, layers = (int(v) for v in spec.split('x'))
    n_inputs = int(n_part.lstrip('n'))
    param = arch.split(':')[2] if arch.count(':') >= 2 else 'softmax'

    mods = []
    d = in_width
    for _ in range(layers):
        mods.append(LutLayer(d, width, n_inputs=n_inputs, parameterization=param))
        d = width
    mods.append(GroupSum(k=classes, tau=tau))
    model = nn.Sequential(*mods)

    data = load_checkpoint_safely(
        ckpt_path, allow_unsafe=allow_unsafe,
        what=f'エクスポート対象チェックポイント {ckpt_path}',
    )
    state = data.get('model_state_dict', data) if isinstance(data, dict) else data
    model.load_state_dict(state)
    return model


def main():
    parser = argparse.ArgumentParser(description='IR/HDLエクスポート')
    parser.add_argument('source', help='チェックポイント(.pt) または IR(.json)')
    parser.add_argument('-o', '--outdir', default='out_hdl')
    parser.add_argument('--arch', help='例: "16000x4:n2:softmax"(.pt時に必須)')
    parser.add_argument('--in-width', type=int, help='入力ビット幅(.pt時に必須)')
    parser.add_argument('--classes', type=int, default=10)
    parser.add_argument('--tau', type=float, default=10.0)
    parser.add_argument('--name', default=None)
    parser.add_argument(
        '--allow-unsafe-checkpoint', action='store_true',
        help='weights_only=Trueで読めない.ptのpickle読込を許可(信頼できるファイルのみ)',
    )
    parser.add_argument('--verify', action='store_true',
                        help='iverilogでgolden modelと照合')
    parser.add_argument('--verify-vectors', type=int, default=256)
    parser.add_argument('--format', choices=['verilog', 'vhdl', 'both'],
                        default='verilog', help='生成するHDL形式')
    args = parser.parse_args()

    from griffith_logic.export import export_ir
    from griffith_logic.export.verilog import write_verilog
    from griffith_logic.export.vhdl import write_vhdl
    from griffith_logic.ir import load_ir, save_ir, validate

    os.makedirs(args.outdir, exist_ok=True)
    name = args.name or os.path.splitext(os.path.basename(args.source))[0]

    if args.source.endswith('.json'):
        ir = load_ir(args.source)
    else:
        if not (args.arch and args.in_width):
            parser.error('.ptからの変換には --arch と --in-width が必要です')
        model = build_model_from_checkpoint(
            args.source, args.arch, args.in_width, args.classes, args.tau,
            allow_unsafe=args.allow_unsafe_checkpoint,
        )
        model.eval()
        ir = export_ir(model, input_width=args.in_width, name=name)
        ir_path = save_ir(ir, os.path.join(args.outdir, f'{name}.json'))
        print(f'IR: {ir_path}')

    warnings = validate(ir, target='hdl')
    for w in warnings:
        print(f'警告: {w}')

    if args.format in ('verilog', 'both'):
        v_path = write_verilog(ir, os.path.join(args.outdir, f'{name}.v'))
        print(f'Verilog: {v_path}')
    if args.format in ('vhdl', 'both'):
        vh_path = write_vhdl(ir, os.path.join(args.outdir, f'{name}.vhdl'))
        print(f'VHDL: {vh_path}')

    if args.verify:
        import numpy as np

        from griffith_logic.export.verify import (
            ghdl_available,
            iverilog_available,
            run_ghdl_equivalence,
            run_iverilog_equivalence,
        )
        rng = np.random.default_rng(0)
        x = rng.integers(0, 2, (args.verify_vectors, ir.inputs[0].width))
        failed = False

        if args.format in ('verilog', 'both'):
            if iverilog_available():
                result = run_iverilog_equivalence(
                    ir, x, os.path.join(args.outdir, 'verify'),
                )
                print('Verilog照合: '
                      + ('一致(ALLPASS)' if result['passed'] else '不一致!'))
                failed |= not result['passed']
            else:
                print('iverilogが見つからないためVerilog照合をスキップ')

        if args.format in ('vhdl', 'both'):
            if ghdl_available():
                result = run_ghdl_equivalence(
                    ir, x, os.path.join(args.outdir, 'verify_vhdl'),
                )
                print('VHDL照合: '
                      + ('一致(ALLPASS)' if result['passed'] else '不一致!'))
                failed |= not result['passed']
            else:
                print('ghdlが見つからないためVHDL照合をスキップ')

        if failed:
            sys.exit(1)


if __name__ == '__main__':
    main()
