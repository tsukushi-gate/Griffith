"""MIXER-2: LogicMixerのブロックごとの中間活性統計を実測する(学習なし・1バッチ)。

目的: n_blocks≥6で学習が始まらない原因が「soft-XOR残差 a+b-2ab の深さ方向の
0.5収縮(=情報が消える)」かどうかを、推測でなく実測で確定させる。

測るもの(すべて soft経路 = model.train() のまま forward):
  - mean / std: 各ブロック出力の平均・標準偏差
  - dist_half: |h - 0.5| の平均(0に近いほど「全部0.5」= 情報が無い)
  - frac_near_half: |h-0.5| < 0.05 の割合(0.5に潰れた素子の割合)
  - binary_frac: |h-0.5| > 0.45 の割合(ほぼ0/1に確定している素子の割合)
埋め込み直後(block 0 の入力)も同じ指標で出し、深さ方向の推移を見る。

使い方:
    uv run python experiments/mixer2_block_stats.py
    uv run python experiments/mixer2_block_stats.py --n-blocks 2 4 6 8 --batch-size 256
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch

from griffith_logic.models.mlp_mixer import LogicMixer, LogicMixerConfig


def act_stats(h: torch.Tensor) -> dict:
    """活性テンソルの統計。0.5への収縮を見るための指標を含む。"""
    d = (h - 0.5).abs()
    return {
        'mean': float(h.mean()),
        'std': float(h.std()),
        'dist_half': float(d.mean()),
        'frac_near_half': float((d < 0.05).float().mean()),
        'binary_frac': float((d > 0.45).float().mean()),
    }


@torch.no_grad()
def probe(n_blocks: int, batch: torch.Tensor, *, d_channels: int, patch_size: int,
          head_group_width: int, depth: int, seed: int) -> dict:
    torch.manual_seed(seed)
    cfg = LogicMixerConfig(
        patch_size=patch_size, d_channels=d_channels, n_blocks=n_blocks,
        depth=depth, head_group_width=head_group_width,
    )
    model = LogicMixer(cfg)
    model.train()  # soft(緩和)経路で測る

    # forward を段階的に再現して各ブロック出力を拾う
    patches = model._patchify(batch)
    coded = model.thermometer(patches)
    h = model.patch_embed(coded.reshape(-1, coded.shape[-1]))
    h = h.reshape(batch.shape[0], cfg.d_tokens, cfg.d_channels)

    layers = [{'stage': 'embed', 'index': 0, **act_stats(h)}]
    for i, block in enumerate(model.blocks, start=1):
        h = block(h)
        layers.append({'stage': 'block', 'index': i, **act_stats(h)})

    return {
        'n_blocks': n_blocks, 'd_channels': d_channels, 'patch_size': patch_size,
        'depth': depth, 'head_group_width': head_group_width, 'seed': seed,
        'layers': layers,
    }


def grad_norms(n_blocks: int, batch: torch.Tensor, labels: torch.Tensor, *,
               d_channels: int, patch_size: int, head_group_width: int,
               depth: int, seed: int) -> dict:
    """初期化直後の1バッチで、ブロックごとのパラメータ勾配ノルムを測る。

    活性の0.5収縮は全構成で起きる(=学習可否を区別しない)ことが分かったため、
    「深さとともに勾配が指数的に消えるか」を直接測る。ブロック1(入力に近い側)の
    勾配が深さとともに落ちるなら、それが n_blocks≥6 で脱出できない理由になる。
    """
    import torch.nn.functional as F

    torch.manual_seed(seed)
    cfg = LogicMixerConfig(
        patch_size=patch_size, d_channels=d_channels, n_blocks=n_blocks,
        depth=depth, head_group_width=head_group_width,
    )
    model = LogicMixer(cfg)
    model.train()
    logits = model(batch)
    loss = F.cross_entropy(logits, labels)
    model.zero_grad()
    loss.backward()

    per_block = []
    for i, block in enumerate(model.blocks, start=1):
        total = 0.0
        for p in block.parameters():
            if p.grad is not None:
                total += float((p.grad ** 2).sum())
        per_block.append({'index': i, 'grad_norm': total ** 0.5})
    embed_norm = sum(
        float((p.grad ** 2).sum()) for p in model.patch_embed.parameters()
        if p.grad is not None) ** 0.5
    head_norm = sum(
        float((p.grad ** 2).sum()) for p in model.head_lut.parameters()
        if p.grad is not None) ** 0.5
    return {
        'n_blocks': n_blocks, 'loss': float(loss),
        'embed_grad_norm': embed_norm, 'head_grad_norm': head_norm,
        'blocks': per_block,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description='ブロックごとの活性統計')
    parser.add_argument('--n-blocks', type=int, nargs='+', default=[2, 4, 6, 8])
    parser.add_argument('--d-channels', type=int, default=256)
    parser.add_argument('--patch-size', type=int, default=7)
    parser.add_argument('--depth', type=int, default=4)
    parser.add_argument('--head-group-width', type=int, default=16)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--out', default='results/mixer2/block_stats.json')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    batch = torch.rand(args.batch_size, 1, 28, 28)

    rows = []
    for nb in args.n_blocks:
        r = probe(nb, batch, d_channels=args.d_channels, patch_size=args.patch_size,
                  head_group_width=args.head_group_width, depth=args.depth,
                  seed=args.seed)
        rows.append(r)
        print(f'\n=== n_blocks={nb} (dc={args.d_channels}, depth={args.depth}, '
              f'ps={args.patch_size}) ===')
        print(f'{"stage":>7} {"mean":>7} {"std":>7} {"|h-.5|":>8} '
              f'{"near.5%":>8} {"binary%":>8}')
        for lay in r['layers']:
            name = lay['stage'] if lay['stage'] == 'embed' else f'blk{lay["index"]}'
            print(f'{name:>7} {lay["mean"]:>7.4f} {lay["std"]:>7.4f} '
                  f'{lay["dist_half"]:>8.4f} {lay["frac_near_half"] * 100:>7.1f}% '
                  f'{lay["binary_frac"] * 100:>7.1f}%')

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    print(f'\n保存: {args.out}')

    # 深さ方向の収縮率(最終ブロックの |h-0.5| / 埋め込み直後の |h-0.5|)
    print('\n--- 0.5への収縮(最終ブロック / 埋め込み直後) ---')
    for r in rows:
        first = r['layers'][0]['dist_half']
        last = r['layers'][-1]['dist_half']
        ratio = last / first if first > 0 else float('nan')
        print(f'n_blocks={r["n_blocks"]:>2}: {first:.4f} -> {last:.4f} '
              f'(比 {ratio:.3f})')

    # 勾配ノルム(初期化直後1バッチ)。活性の収縮は全構成で起きるため、
    # 学習可否を分けるのはこちらの可能性が高い。
    print('\n--- 初期化直後の勾配ノルム(ブロック別) ---')
    labels = torch.randint(0, 10, (batch.shape[0],))
    grows = []
    for nb in args.n_blocks:
        g = grad_norms(nb, batch, labels, d_channels=args.d_channels,
                       patch_size=args.patch_size,
                       head_group_width=args.head_group_width,
                       depth=args.depth, seed=args.seed)
        grows.append(g)
        blocks = ' '.join(f'b{b["index"]}={b["grad_norm"]:.2e}' for b in g['blocks'])
        print(f'n_blocks={nb:>2} loss={g["loss"]:.4f} head={g["head_grad_norm"]:.2e} '
              f'embed={g["embed_grad_norm"]:.2e} | {blocks}')

    gpath = args.out.replace('.json', '_grads.json')
    with open(gpath, 'w') as f:
        json.dump(grows, f, ensure_ascii=False, indent=2)
    print(f'保存: {gpath}')


if __name__ == '__main__':
    main()
