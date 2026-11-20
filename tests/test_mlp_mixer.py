"""LogicMixer(MLP-Mixerの論理ゲート化)のユニットテスト。

高速スイート内で数秒に収める。
"""

import os
import sys

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from griffith_logic.layers.lut import LutLayer
from griffith_logic.models.mlp_mixer import LogicMixer, LogicMixerConfig


# ============================================================
# 1. shape契約
# ============================================================
class TestShapes:
    @pytest.mark.parametrize(
        'image_size,patch_size,in_channels,encode_bits,d_channels,'
        'n_blocks,depth,n_inputs,num_classes',
        [
            (28, 7, 1, 8, 64, 4, 2, 2, 10),   # 契約デフォルト相当(MNIST)
            (8, 4, 1, 2, 8, 1, 1, 2, 2),      # 極小構成
            (12, 4, 3, 3, 16, 2, 1, 4, 5),    # CIFAR風(3ch)+n_inputs=4
        ],
    )
    def test_forward_shape(
        self, seed, image_size, patch_size, in_channels, encode_bits,
        d_channels, n_blocks, depth, n_inputs, num_classes,
    ):
        cfg = LogicMixerConfig(
            image_size=image_size, patch_size=patch_size, in_channels=in_channels,
            encode_bits=encode_bits, d_channels=d_channels, n_blocks=n_blocks,
            depth=depth, n_inputs=n_inputs, num_classes=num_classes,
        )
        model = LogicMixer(cfg)
        batch = 3
        x = torch.rand(batch, in_channels, image_size, image_size)
        out = model(x)
        assert out.shape == (batch, num_classes)

    def test_d_tokens_derived(self, seed):
        cfg = LogicMixerConfig(image_size=28, patch_size=7)
        assert cfg.d_tokens == 16
        assert cfg.side == 4

    def test_image_size_not_divisible_raises(self, seed):
        with pytest.raises(ValueError):
            LogicMixerConfig(image_size=28, patch_size=6)

    def test_wrong_input_shape_raises(self, seed):
        cfg = LogicMixerConfig(image_size=8, patch_size=4, d_channels=8, n_blocks=1, depth=1)
        model = LogicMixer(cfg)
        with pytest.raises(ValueError):
            model(torch.rand(2, 1, 10, 10))


# ============================================================
# 1b. パッチ埋め込みのfan-in制御(MIXER-1 C)
# ============================================================
class TestPatchEmbedFanIn:
    def test_default_is_single_stage(self, seed):
        """既定は従来どおり1段埋め込み(後方互換)。"""
        model = LogicMixer(LogicMixerConfig())
        assert len(model.patch_embed) == 1
        assert model.patch_embed[0].in_dim == 49 * 8
        assert model.patch_embed[0].out_dim == 64

    def test_default_coverage_is_partial(self, seed):
        """既定構成(392入力/64ゲート×2)は入力の約2/3を構造的に捨てる。

        get_random_connections は randperm ベースでタップを入力へ均等に配るため、
        被覆率は乱数に依らず min(タップ数, 入力ビット数)/入力ビット数 に決まる。
        """
        model = LogicMixer(LogicMixerConfig())
        assert model.embed_input_coverage() == pytest.approx(128 / 392)

    @pytest.mark.parametrize('d_channels,expected', [
        (64, 128 / 392),    # タップ不足: 67.3%の入力が到達しない
        (128, 256 / 392),   # まだ不足
        (196, 1.0),         # 392タップでちょうど全被覆
        (256, 1.0),         # 以降はrandomでも全入力が到達する
    ])
    def test_coverage_is_deterministic_in_width(self, seed, d_channels, expected):
        """被覆率は幅だけで決まる。d_channels>=196では2段化しなくても1.0。"""
        model = LogicMixer(LogicMixerConfig(d_channels=d_channels))
        assert model.embed_input_coverage() == pytest.approx(expected)

    def test_unique_equals_random_coverage_at_same_width(self, seed):
        """同じ幅なら'unique'と'random'の被覆率は変わらない(差は重複の配り方のみ)。

        被覆を上げる唯一の手段は1段目のタップ数を増やすこと(=2段埋め込み)。
        """
        rnd = LogicMixer(LogicMixerConfig(d_channels=256))
        uniq = LogicMixer(LogicMixerConfig(d_channels=256, embed_connections='unique'))
        assert rnd.embed_input_coverage() == pytest.approx(uniq.embed_input_coverage())

    def test_two_stage_unique_covers_all_inputs(self, seed):
        """2段化+unique接続なら全入力ビットが埋め込みへ到達する。"""
        cfg = LogicMixerConfig(embed_hidden=256, embed_connections='unique')
        model = LogicMixer(cfg)
        assert len(model.patch_embed) == 2
        assert model.embed_input_coverage() == pytest.approx(1.0)
        out = model(torch.rand(2, 1, 28, 28))
        assert out.shape == (2, 10)

    def test_unique_without_enough_fan_in_raises(self, seed):
        """1段目のタップ数が入力ビット数に足りない'unique'指定は弾く。"""
        with pytest.raises(ValueError, match='unique'):
            LogicMixerConfig(embed_connections='unique')  # 64×2=128 < 392

    def test_invalid_embed_connections_raises(self, seed):
        with pytest.raises(ValueError):
            LogicMixerConfig(embed_connections='dense')


# ============================================================
# 1c. 分類ヘッドのfan-in(MIXER-1: 幅を増やすほど絞りが悪化する)
# ============================================================
class TestHeadFanIn:
    @pytest.mark.parametrize('d_channels,expected', [
        (64, 320 / 1024),    # 0.312
        (128, 320 / 2048),   # 0.156
        (256, 320 / 4096),   # 0.078: 入力の92%がヘッドへ到達しない
        (384, 320 / 6144),   # 0.052
    ])
    def test_head_coverage_degrades_with_width(self, seed, d_channels, expected):
        """既定head_group_width=16では、d_channelsを増やすほどヘッド被覆率が下がる。"""
        model = LogicMixer(LogicMixerConfig(d_channels=d_channels))
        assert model.head_input_coverage() == pytest.approx(expected)

    def test_wider_head_improves_coverage(self, seed):
        """head_group_widthを増やせばヘッド被覆率は上がる。"""
        narrow = LogicMixer(LogicMixerConfig(d_channels=256))
        wide = LogicMixer(LogicMixerConfig(d_channels=256, head_group_width=128))
        assert wide.head_input_coverage() > narrow.head_input_coverage()
        assert wide.head_input_coverage() == pytest.approx(2560 / 4096)

    def test_head_group_width_changes_shapes_only_internally(self, seed):
        """head_group_widthを変えても出力は(B, num_classes)のまま。"""
        model = LogicMixer(LogicMixerConfig(d_channels=64, head_group_width=32))
        assert model(torch.rand(2, 1, 28, 28)).shape == (2, 10)

    def test_invalid_head_group_width_raises(self, seed):
        with pytest.raises(ValueError):
            LogicMixerConfig(head_group_width=0)


# ============================================================
# 2. 勾配疎通: 全LutLayerにnon-zero勾配
# ============================================================
class TestGradientFlow:
    def test_all_lut_layers_get_nonzero_grad(self, seed):
        cfg = LogicMixerConfig(
            image_size=8, patch_size=4, in_channels=1, encode_bits=2,
            d_channels=8, n_blocks=2, depth=2, n_inputs=2, num_classes=2,
        )
        model = LogicMixer(cfg)
        model.train()
        x = torch.rand(6, 1, 8, 8)
        y = torch.randint(0, cfg.num_classes, (6,))

        logits = model(x)
        loss = F.cross_entropy(logits, y)
        loss.backward()

        lut_layers = [m for m in model.modules() if isinstance(m, LutLayer)]
        assert len(lut_layers) > 0
        for i, layer in enumerate(lut_layers):
            assert layer.weights.grad is not None, f'lut_layers[{i}]: 勾配なし'
            assert torch.count_nonzero(layer.weights.grad) > 0, (
                f'lut_layers[{i}]: 勾配が全てゼロ'
            )


# ============================================================
# 3. hard離散性: eval時、中間featureが{0,1}
# ============================================================
class TestHardDiscreteness:
    def test_eval_intermediate_features_are_binary(self, seed):
        cfg = LogicMixerConfig(
            image_size=8, patch_size=4, in_channels=1, encode_bits=2,
            d_channels=8, n_blocks=2, depth=2, n_inputs=2, num_classes=2,
        )
        model = LogicMixer(cfg)
        model.eval()

        captured = []

        def hook(_module, _inputs, output):
            captured.append(output.detach())

        handles = [
            m.register_forward_hook(hook)
            for m in model.modules() if isinstance(m, LutLayer)
        ]
        try:
            x = torch.rand(4, 1, 8, 8)
            with torch.no_grad():
                model(x)
        finally:
            for h in handles:
                h.remove()

        assert len(captured) > 0
        for i, out in enumerate(captured):
            is_binary = bool(((out == 0.0) | (out == 1.0)).all())
            assert is_binary, f'LutLayer出力[{i}]がhard eval時に{{0,1}}でない'


# ============================================================
# 4. 収束(L4級): 縦縞 vs 横縞トイ画像でhard acc=100%
# ============================================================
def _stripe_dataset(image_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    """縦縞(label=0) vs 横縞(label=1)の2クラストイ画像データセット。

    0/1反転でバリエーションを持たせた4枚の合成画像を返す。
    """
    coords = torch.arange(image_size)
    col_parity = (coords % 2).float().view(1, image_size).expand(image_size, image_size)
    row_parity = (coords % 2).float().view(image_size, 1).expand(image_size, image_size)
    imgs, labels = [], []
    for invert in (0.0, 1.0):
        imgs.append((col_parity + invert) % 2.0)  # 縦縞
        labels.append(0)
        imgs.append((row_parity + invert) % 2.0)  # 横縞
        labels.append(1)
    x = torch.stack(imgs).unsqueeze(1)  # (4,1,H,W)
    y = torch.tensor(labels, dtype=torch.long)
    return x, y


def _train_until_hard_acc_100(model, x, y, *, lr, max_steps, check_every=20):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for step in range(max_steps):
        model.train()
        opt.zero_grad()
        loss = F.cross_entropy(model(x), y)
        loss.backward()
        opt.step()
        if (step + 1) % check_every == 0:
            model.eval()
            with torch.no_grad():
                acc = (model(x).argmax(-1) == y).float().mean().item()
            if acc == 1.0:
                return 1.0
    model.eval()
    with torch.no_grad():
        return (model(x).argmax(-1) == y).float().mean().item()


class TestConvergence:
    def test_stripe_classification_hard_acc(self, seed):
        torch.manual_seed(42)
        cfg = LogicMixerConfig(
            image_size=8, patch_size=4, in_channels=1, encode_bits=2,
            d_channels=8, n_blocks=1, depth=1, n_inputs=2, num_classes=2,
        )
        model = LogicMixer(cfg)
        x, y = _stripe_dataset(cfg.image_size)

        acc = _train_until_hard_acc_100(model, x, y, lr=0.1, max_steps=800)
        assert acc == 1.0, f'縦縞/横縞トイ分類 hard acc={acc}'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
