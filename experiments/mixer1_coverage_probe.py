"""MIXER-1 C: パッチ埋め込みの入力被覆率を構成ごとに実測する(学習なし)。

get_random_connections は randperm ベースで、タップ数 out_dim×n_inputs を
入力へ可能な限り均等に配る。したがって被覆率は乱数に依らず
    min(out_dim×n_inputs, in_dim) / in_dim
に決まる。'unique' との差は「被覆」ではなく「同一入力の重複割り当て方」だけで、
被覆を1.0にする唯一の手段は1段目のタップ数を増やすこと(=2段埋め込み)。

使い方: uv run python experiments/mixer1_coverage_probe.py
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from griffith_logic.models.mlp_mixer import LogicMixer, LogicMixerConfig

# (dataset, image_size, patch_size, in_channels, encode_bits)
_SPECS = {
    'mnist': (28, 7, 1, 8),
    'cifar10': (32, 8, 3, 8),
}


def probe(dataset: str, d_channels: int, embed_hidden: int | None,
          embed_connections: str) -> dict:
    image_size, patch_size, in_channels, encode_bits = _SPECS[dataset]
    cfg = LogicMixerConfig(
        image_size=image_size, patch_size=patch_size, in_channels=in_channels,
        encode_bits=encode_bits, d_channels=d_channels, n_blocks=1, depth=1,
        embed_hidden=embed_hidden, embed_connections=embed_connections,
    )
    model = LogicMixer(cfg)
    first = model.patch_embed[0]
    in_bits = int(first.in_dim)
    taps = int(first.out_dim) * cfg.n_inputs
    head = model.head_lut
    head_in = int(head.in_dim)
    head_taps = int(head.out_dim) * cfg.n_inputs
    return {
        'dataset': dataset, 'd_channels': d_channels,
        'embed_hidden': embed_hidden, 'embed_connections': embed_connections,
        'embed_in_bits': in_bits, 'first_stage_taps': taps,
        'coverage': round(model.embed_input_coverage(), 6),
        'theoretical_coverage': round(min(taps, in_bits) / in_bits, 6),
        # 分類ヘッド側の絞り(埋め込みより深刻。d_channelsを増やすほど悪化)
        'head_in_bits': head_in, 'head_taps': head_taps,
        'head_coverage': round(model.head_input_coverage(), 6),
    }


def main() -> None:
    rows = []
    for dc in (64, 128, 196, 256, 384):
        rows.append(probe('mnist', dc, None, 'random'))
    rows.append(probe('mnist', 64, 256, 'unique'))
    rows.append(probe('mnist', 256, None, 'unique'))
    for dc in (256,):
        rows.append(probe('cifar10', dc, None, 'random'))
        rows.append(probe('cifar10', dc, 768, 'unique'))

    hdr = (f'{"dataset":8} {"d_ch":>5} {"embed_hidden":>12} {"conn":>7} '
           f'{"in_bits":>8} {"taps":>6} {"coverage":>9} '
           f'{"head_in":>8} {"head_tap":>8} {"head_cov":>9}')
    print(hdr)
    print('-' * len(hdr))
    for r in rows:
        assert abs(r['coverage'] - r['theoretical_coverage']) < 1e-6, r
        print(f'{r["dataset"]:8} {r["d_channels"]:>5} '
              f'{str(r["embed_hidden"]):>12} {r["embed_connections"]:>7} '
              f'{r["embed_in_bits"]:>8} {r["first_stage_taps"]:>6} '
              f'{r["coverage"]:>9.3f} '
              f'{r["head_in_bits"]:>8} {r["head_taps"]:>8} '
              f'{r["head_coverage"]:>9.3f}')

    os.makedirs('results/mixer1', exist_ok=True)
    out = 'results/mixer1/embed_coverage.json'
    with open(out, 'w') as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    print(f'\n保存: {out}')


if __name__ == '__main__':
    main()
