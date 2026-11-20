"""MIXER-1: CDDLGN(train_cnn.py の LogicTreeNet)のゲート数を構成別に算出する。

LogicMixerとの公平比較のため、`-k` ごとに
  - ユニークゲート数(重み共有を1回だけ数える = count_logic_gates)
  - 空間展開後ゲート数(出力ピクセルごとに展開した実回路規模)
の両方を出す。convは空間方向に重みを共有するので、この2つは大きく異なる。

使い方: uv run python experiments/mixer1_cddlgn_gates.py
"""

import argparse
import json
import os
import sys
import types

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))

import torch
from wandb_utils import count_logic_gates

from griffith_logic.conv import LutConv2d


def _args_for(k: int, dataset: str) -> types.SimpleNamespace:
    """train_cnn.py の build_*_model が参照する属性だけを持つ簡易args。"""
    return types.SimpleNamespace(
        k=k, tree_depth=3, n_inputs=2, n_thresholds=3,
        input_channels_per_tree=2, output_gate_factor=2,
        connections='random', parameterization='softmax', selection='soft',
        dataset=dataset, device='cpu', tau=None, grad_factor=1.0,
        # train_cnn.py の argparse 既定値に合わせる(いずれも store_true = 既定False)
        wide_last_layer=False, residual_init_dense=False,
    )


def conv_expanded_gates(model: torch.nn.Module, input_shape: tuple[int, ...]) -> int:
    """convを空間展開したときのゲート数。各LutConv2dの num_gates × 出力H×W の合計。

    convは空間方向に重みを共有するので、ユニーク数と展開後で大きく異なる。
    出力サイズはダミー入力を1回流してフックで実測する(stride/paddingの取り違え防止)。
    Dense層は空間展開されないので、ここには含めない(呼び出し側で足す)。
    """
    total = 0
    handles = []

    def _hook(module, _inp, out):
        nonlocal total
        # out: (B, C, H, W)
        total += int(module.num_gates) * int(out.shape[-2]) * int(out.shape[-1])

    for m in model.modules():
        if isinstance(m, LutConv2d):
            handles.append(m.register_forward_hook(_hook))
    model.eval()
    with torch.no_grad():
        model(torch.rand(1, *input_shape))
    for h in handles:
        h.remove()
    return total


def main() -> None:
    parser = argparse.ArgumentParser(description='CDDLGNのゲート数算出')
    parser.add_argument('--dataset', default='mnist', choices=['mnist', 'cifar-10'])
    parser.add_argument('--ks', type=int, nargs='+', default=[2, 4, 8, 16, 32, 64])
    parser.add_argument('--out', default='results/mixer1/cddlgn_gates.json')
    args = parser.parse_args()

    from train_cnn import build_cifar10_model, build_mnist_model

    # モデルの入力はthermometer符号化後のチャンネル(n_thresholds × 元チャンネル)。
    # build_*_model が in_ch = n_thresholds として組むので、それに合わせる。
    n_thresholds = _args_for(1, args.dataset).n_thresholds
    if args.dataset == 'mnist':
        build, shape = build_mnist_model, (n_thresholds, 28, 28)
    else:
        build, shape = build_cifar10_model, (3 * n_thresholds, 32, 32)

    rows = []
    hdr = (f'{"k":>4} {"conv_uniq":>10} {"dense":>9} {"total_uniq":>11} '
           f'{"conv_exp":>11} {"total_exp":>12} {"params":>11}')
    print(f'dataset={args.dataset}')
    print(hdr)
    print('-' * len(hdr))
    for k in args.ks:
        built = build(_args_for(k, args.dataset))
        model = built[0] if isinstance(built, tuple) else built
        model = model.to('cpu')
        info = count_logic_gates(model)
        total_uniq = int(info['gates/total'])
        conv_uniq = int(info.get('gates/conv', 0))
        dense = int(info.get('gates/dense', 0))
        conv_exp = conv_expanded_gates(model, shape)
        # Dense層は空間展開されないのでそのまま加算する
        total_exp = conv_exp + dense
        n_params = sum(p.numel() for p in model.parameters())
        rows.append({
            'dataset': args.dataset, 'k': k,
            'conv_unique_gates': conv_uniq, 'dense_gates': dense,
            'total_unique_gates': total_uniq,
            'conv_expanded_gates': conv_exp, 'total_expanded_gates': total_exp,
            'n_params': n_params,
        })
        print(f'{k:>4} {conv_uniq:>10,} {dense:>9,} {total_uniq:>11,} '
              f'{conv_exp:>11,} {total_exp:>12,} {n_params:>11,}')

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    print(f'\n保存: {args.out}')


if __name__ == '__main__':
    main()
