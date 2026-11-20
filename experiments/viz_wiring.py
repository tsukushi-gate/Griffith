"""論理ゲート配線の可視化CLI(VIZ-1)。

DLGN系モデルの配線構造を抽出して2種類の図(全配線図/抽象図)をPNG出力する。
生成した図はローカル保存に加えて必ず wandb にもログする(--no-wandb で無効化)。

使用例:
    # 純DLGN(直列のみ)のデモ図を生成
    uv run python experiments/viz_wiring.py --demo plain

    # 入力バイパス(IALGN型 spine+anchor)のデモ図を生成
    uv run python experiments/viz_wiring.py --demo bypass

    # 学習済みチェックポイントの配線を可視化(抽象図のみ)
    uv run python experiments/viz_wiring.py \\
        --ckpt checkpoints/input_bypass_mnist20x20/xxx/best.pt --mode abstract
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from env_utils import load_dotenv
from wandb_utils import (
    add_experiment_args,
    init_wandb,
    log_images,
    normalize_tags,
)
from wandb_utils import (
    finish as wandb_finish,
)

from griffith_logic import GroupSum, LutLayer
from griffith_logic.analysis.wiring_viz import (
    extract_wiring_spec,
    plot_abstract_wiring,
    plot_full_wiring,
)

load_dotenv()

DEMO_MODES = ('plain', 'bypass', 'hybrid-expand', 'hybrid-compress')

# HYBRID-1 の代表構成(experiments/train_hybrid_front.py の run 名に対応)。
# 配線は初期化で決まるため学習は不要(seed固定で構築するだけでよい)。
HYBRID_DEMOS = {
    # m-front2-expand-bypass: hard 0.9703 / 560ゲートの主役構成
    'hybrid-expand': dict(front_widths=[256, 64], dlgn_widths=[80, 160, 320],
                          expand_fill='bypass'),
    # m-front1-compress: 後段DLGNを圧縮方向に使った対照(180ゲート)
    'hybrid-compress': dict(front_widths=[256], dlgn_widths=[120, 60],
                            expand_fill='none'),
}


# ============================================================
# デモモデルの構築
# ============================================================

def build_demo_plain(in_dim: int = 400, k: int = 200, num_layers: int = 4,
                     num_classes: int = 10) -> torch.nn.Module:
    """純DLGN(experiments/train.py の build_model と同じ直列構造)。"""
    layers: list[torch.nn.Module] = [torch.nn.Flatten()]
    layers.append(LutLayer(in_dim=in_dim, out_dim=k, n_inputs=2,
                           connections='unique'))
    for _ in range(num_layers - 1):
        layers.append(LutLayer(in_dim=k, out_dim=k, n_inputs=2,
                               connections='random'))
    layers.append(GroupSum(num_classes, tau=10.0))
    return torch.nn.Sequential(*layers)


class DemoBypassDLGN(torch.nn.Module):
    """IALGN型(spine保持+anchor挿入)入力バイパスDLGNの小型版。

    experiments/train_input_bypass.py の InputBypassDLGN(bypass_mode='ialgn',
    anchor_source='dense')と**同じプール構成**を持つ自己完結の再現モデル。
    可視化デモを訓練スクリプト側の実装変更から独立させるために別実装にしてある。

        隠れ層 h_pos の入力プール = concat([h_prev(k), x(in_dim), o_0(k), ...])
        slot0 = 前層の同インデックス(spine)、slot1 = anchor(x または過去層)
    """

    def __init__(self, in_dim: int = 400, k: int = 200, num_layers: int = 4,
                 num_classes: int = 10, seed: int = 0):
        super().__init__()
        rng = torch.Generator().manual_seed(seed)
        self.flatten = torch.nn.Flatten()
        self.layer0 = LutLayer(in_dim=in_dim, out_dim=k, n_inputs=2,
                               connections='unique')
        hidden = []
        for h_pos in range(num_layers - 1):
            n_past = h_pos                      # o_0 .. o_{h_pos-1}
            pool_width = k + in_dim + n_past * k
            # anchor候補: 元入力 + 過去の中間層(dense)
            candidates = [(k, in_dim)] + [(k + in_dim + j * k, k)
                                          for j in range(n_past)]
            total = sum(size for _, size in candidates)
            local = torch.randint(0, total, (k,), generator=rng,
                                  dtype=torch.int64)
            anchor = torch.empty_like(local)
            cursor = 0
            for off, size in candidates:
                m = (local >= cursor) & (local < cursor + size)
                anchor[m] = local[m] - cursor + off
                cursor += size
            conn = torch.stack([torch.arange(k, dtype=torch.int64), anchor])
            hidden.append(LutLayer(in_dim=pool_width, out_dim=k, n_inputs=2,
                                   connections=conn))
        self.hidden = torch.nn.ModuleList(hidden)
        self.group_sum = GroupSum(num_classes, tau=10.0)
        self.in_dim = in_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.flatten(x)
        h = self.layer0(x)
        acts = [h]
        for h_pos, layer in enumerate(self.hidden):
            inp = torch.cat([h, x] + acts[:h_pos], dim=-1)
            h = layer(inp)
            acts.append(h)
        return self.group_sum(h)


# ============================================================
# チェックポイントからの復元
# ============================================================

def build_demo_hybrid(demo: str, in_dim: int = 784, num_classes: int = 10,
                      seed: int = 0) -> torch.nn.Module:
    """HYBRID-1 の代表構成の HybridFrontDLGN を構築する(学習なし)。

    `HybridFrontDLGN.wiring_spec()` が実装済みなので、抽出は自動推論ではなく
    そのメソッドに委譲される。
    """
    from train_hybrid_front import HybridFrontDLGN

    cfg = HYBRID_DEMOS[demo]
    return HybridFrontDLGN(
        in_dim=in_dim, num_classes=num_classes, n_inputs=2, tau=10.0,
        seed=seed, **cfg,
    )


class _StubLut(torch.nn.Module):
    """配線抽出に必要な属性だけを持つLutLayerのスタブ。

    チェックポイントには `*.indices`(配線)と `*.rev_starts`(長さ in_dim+1)が
    バッファとして保存されているため、**モデルクラスを再構築せずに**
    in_dim / out_dim / n_inputs / indices を復元できる。
    """

    def __init__(self, indices: torch.Tensor, in_dim: int):
        super().__init__()
        self.register_buffer('indices', indices)
        self.in_dim = in_dim
        self.n_inputs, self.out_dim = int(indices.shape[0]), int(indices.shape[1])


class _StubModel(torch.nn.Module):
    """スタブLutLayerを定義順に並べただけの容器(extract_wiring_spec 用)。"""

    def __init__(self, luts: list[_StubLut]):
        super().__init__()
        self.luts = torch.nn.ModuleList(luts)


def load_model_from_ckpt(path: str) -> torch.nn.Module:
    """チェックポイントから配線抽出用のスタブモデルを組み立てる。

    対応形式は checkpoint_utils.save_checkpoint 形式(`model_state_dict` を持つ dict)
    または state_dict そのもの。GroupSum はパラメータ/バッファを持たないため
    復元できず、抽象図の最終ノードは省略される(既知の制約)。
    """
    obj = torch.load(path, map_location='cpu', weights_only=False)
    state = obj.get('model_state_dict', obj) if isinstance(obj, dict) else obj
    assert isinstance(state, dict), f'state_dict を取り出せません: {path}'

    prefixes = [key[: -len('.indices')] for key in state
                if key.endswith('.indices')]
    assert prefixes, (
        f'{path} に .indices バッファが見つかりません。'
        'LutLayerを含むモデルのチェックポイントを指定してください'
    )
    luts = []
    for p in prefixes:
        indices = state[f'{p}.indices'].to(torch.int64)
        rev = state.get(f'{p}.rev_starts')
        if rev is None:
            # rev_starts が無い形式では in_dim を配線の最大値から下限推定する
            in_dim = int(indices.max()) + 1
            print(f'警告: {p}.rev_starts が無いため in_dim を {in_dim} と推定します')
        else:
            in_dim = int(rev.numel()) - 1
        luts.append(_StubLut(indices, in_dim))
    print(f'チェックポイント復元: LutLayer {len(luts)}層 '
          f'({", ".join(prefixes)})')
    return _StubModel(luts)


# ============================================================
# メイン
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='論理ゲート配線の可視化')
    src = parser.add_mutually_exclusive_group()
    src.add_argument('--demo', type=str, default=None, choices=list(DEMO_MODES),
                     help='デモモデルを構築して可視化する')
    src.add_argument('--ckpt', type=str, default=None,
                     help='チェックポイントから配線を復元して可視化する')
    parser.add_argument('--mode', type=str, default='both',
                        choices=['full', 'abstract', 'both'],
                        help='生成する図 (default: both)')
    parser.add_argument('--out-dir', type=str, default='results/wiring_viz',
                        help='PNGの出力先 (default: results/wiring_viz)')
    parser.add_argument('--max-gates-per-layer', type=int, default=None,
                        help='全配線図で描くゲート数の上限(既定: 全描画)')
    parser.add_argument('-k', '--num-neurons', type=int, default=200,
                        help='デモの層あたりニューロン数 (default: 200)')
    parser.add_argument('-l', '--num-layers', type=int, default=4,
                        help='デモのLUT層数 (default: 4)')
    parser.add_argument('--in-dim', type=int, default=400,
                        help='デモの入力次元 (default: 400 = mnist20x20)')
    parser.add_argument('--seed', type=int, default=0, help='ランダムシード')
    parser.add_argument('--no-wandb', action='store_true',
                        help='wandbへのログを無効化する')
    parser.add_argument('--wandb-group', type=str, default='wiring-viz',
                        help='wandb group (default: wiring-viz)')
    add_experiment_args(parser)
    args = parser.parse_args()

    if args.demo is None and args.ckpt is None:
        args.demo = 'plain'

    torch.manual_seed(args.seed)

    if args.ckpt is not None:
        model = load_model_from_ckpt(args.ckpt)
        tag = os.path.splitext(os.path.basename(args.ckpt))[0]
    elif args.demo == 'plain':
        model = build_demo_plain(args.in_dim, args.num_neurons,
                                 args.num_layers)
        tag = f'demo-plain_k{args.num_neurons}_l{args.num_layers}'
    elif args.demo in HYBRID_DEMOS:
        # ハイブリッドの入力次元は MNIST(784)が既定。--in-dim で上書き可
        in_dim = 784 if args.in_dim == 400 else args.in_dim
        model = build_demo_hybrid(args.demo, in_dim=in_dim, seed=args.seed)
        tag = f'demo-{args.demo}'
    else:
        model = DemoBypassDLGN(args.in_dim, args.num_neurons,
                               args.num_layers, seed=args.seed)
        tag = f'demo-bypass_k{args.num_neurons}_l{args.num_layers}'

    spec = extract_wiring_spec(model)
    print('層: ' + ', '.join(f'{n.name}({n.width})' for n in spec.layers))
    for e in spec.edges:
        print(f'  配線 {e.src} -> {e.dst}: {e.n_wires:,}本')

    os.makedirs(args.out_dir, exist_ok=True)
    outputs: list[tuple[str, str]] = []
    if args.mode in ('full', 'both'):
        p = plot_full_wiring(
            spec, os.path.join(args.out_dir, f'{tag}_full.png'),
            max_gates_per_layer=args.max_gates_per_layer,
        )
        outputs.append(('full', p))
    if args.mode in ('abstract', 'both'):
        p = plot_abstract_wiring(
            spec, os.path.join(args.out_dir, f'{tag}_abstract.png'))
        outputs.append(('abstract', p))
    for kind, path in outputs:
        print(f'{kind}: {path}')

    if args.no_wandb:
        return

    use_wandb = init_wandb(
        run_name=f'wiring-viz_{tag}',
        config_dict={**vars(args),
                     'n_layers': len(spec.layers),
                     'n_edges': len(spec.edges),
                     'total_wires': sum(e.n_wires for e in spec.edges)},
        tags=['wiring-viz'] + normalize_tags(
            model=(args.demo or 'ckpt'), seed=args.seed, series='VIZ-1'),
        group=args.wandb_group,
        job_type='analysis',
        notes=args.notes,
        hypothesis=args.hypothesis,
    )
    if not use_wandb:
        print('警告: wandbが無効のためログをスキップしました')
        return

    from PIL import Image

    for kind, path in outputs:
        with Image.open(path) as im:
            log_images(f'wiring/{kind}', [im.convert('RGB')], captions=[tag])
    import wandb

    wandb.run.summary['total_wires'] = sum(e.n_wires for e in spec.edges)
    wandb.run.summary['layers'] = [f'{n.name}:{n.width}' for n in spec.layers]
    print(f'WANDB_RUN_ID={wandb.run.id}')
    wandb_finish()


if __name__ == '__main__':
    main()
