import sys
from types import SimpleNamespace

import torch

from griffith_logic.layers.dwn import DWNLayer
from griffith_logic.models.llvit import (
    ConditionalSum,
    LLViTChannelMixer,
    LLViTConfig,
    build_deit_tiny_cifar10,
    build_llvit_cifar10,
    calibrate_thermometers,
    calibrate_thermometers_from_teacher,
    init_from_timm_deit_tiny,
    set_conditional_sum_quantized,
)


def tiny_config(**overrides) -> LLViTConfig:
    cfg = dict(
        image_size=32,
        patch_size=16,
        dim=12,
        depth=1,
        heads=3,
        mlp_ratio=2,
        thermometer_bits=2,
        lut_hidden=24,
        lut_out=12,
        lut_inputs=2,
        lut_mapping='learnable',
        lut_efd='official',
    )
    cfg.update(overrides)
    return LLViTConfig(**cfg)


def test_llvit_paper_config_defaults():
    cfg = LLViTConfig()
    meta = cfg.paper_metadata()
    assert cfg.image_size == 224
    assert cfg.patch_size == 16
    assert cfg.dim == 192
    assert cfg.depth == 12
    assert cfg.heads == 3
    assert cfg.tokens == 197
    assert meta['channel_mixer_luts'] == [768, 192]
    assert meta['thermometer_bits'] == 8
    assert meta['lut_inputs'] == 6
    assert meta['lut_mapping'] == 'learnable'
    assert meta['lut_efd'] == 'official'
    assert meta['encoded_value'] == 'int4'


def test_llvit_forward_backward_no_nan():
    torch.manual_seed(0)
    model = build_llvit_cifar10(tiny_config())
    x = torch.rand(2, 3, 32, 32)
    y = torch.tensor([1, 2])
    out = model(x)
    assert out.shape == (2, 10)
    loss = torch.nn.functional.cross_entropy(out, y)
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads
    assert all(torch.isfinite(g).all() for g in grads)


def test_baseline_keeps_mha_and_replaces_only_channel_mixer():
    baseline = build_deit_tiny_cifar10(tiny_config())
    llvit = build_llvit_cifar10(tiny_config())
    assert not any(isinstance(m, DWNLayer) for m in baseline.modules())
    assert sum(isinstance(m, DWNLayer) for m in llvit.modules()) == 2
    block = llvit.blocks[0]
    assert hasattr(block, 'attn')
    assert isinstance(block.channel_mixer, LLViTChannelMixer)


def test_conditional_sum_int4_eval_shape_and_error():
    layer = ConditionalSum(4, 3, quant_bits=4)
    with torch.no_grad():
        layer.encoded_values.copy_(torch.tensor([
            [-1.0, -0.25, 0.25],
            [0.5, -0.5, 1.0],
            [0.0, 0.25, -0.75],
            [1.0, 0.0, -0.25],
        ]))
    x = torch.ones(2, 4)
    train_out = layer(x)
    layer.eval()
    layer.use_quantized_eval = True
    quant_out = layer(x)
    assert quant_out.shape == train_out.shape
    assert torch.isfinite(quant_out).all()
    assert (quant_out - train_out).abs().max().item() <= 4.0 / 7.0 + 1e-6


def test_set_conditional_sum_quantized_toggles_all_layers():
    model = build_llvit_cifar10(tiny_config(depth=2))
    set_conditional_sum_quantized(model, True)
    layers = [m for m in model.modules() if isinstance(m, ConditionalSum)]
    assert len(layers) == 2
    assert all(m.use_quantized_eval for m in layers)


def test_calibrate_thermometers_fits_thresholds():
    model = build_llvit_cifar10(tiny_config(thermometer_mode='distributive'))
    data = torch.utils.data.TensorDataset(
        torch.rand(4, 3, 32, 32), torch.zeros(4, dtype=torch.long),
    )
    loader = torch.utils.data.DataLoader(data, batch_size=2)
    stats = calibrate_thermometers(
        model, loader, torch.device('cpu'), max_batches=1, max_samples_per_mixer=16,
    )
    mixer = next(m for m in model.modules() if isinstance(m, LLViTChannelMixer))
    assert mixer.thermometer.thresholds is not None
    assert stats['thermometer/calibrated_mixers'] == 1
    assert 0.0 <= stats['thermometer/bit_saturation_mean'] <= 1.0


def test_calibrate_thermometers_from_teacher_inputs():
    cfg = tiny_config(thermometer_mode='distributive')
    student = build_llvit_cifar10(cfg)
    teacher = build_deit_tiny_cifar10(cfg)
    data = torch.utils.data.TensorDataset(
        torch.rand(4, 3, 32, 32), torch.zeros(4, dtype=torch.long),
    )
    loader = torch.utils.data.DataLoader(data, batch_size=2)
    stats = calibrate_thermometers_from_teacher(
        student,
        teacher,
        loader,
        torch.device('cpu'),
        max_batches=1,
        max_samples_per_mixer=16,
    )
    mixer = next(m for m in student.modules() if isinstance(m, LLViTChannelMixer))
    assert mixer.thermometer.thresholds is not None
    assert stats['thermometer/calibrated_mixers'] == 1
    assert 0.0 <= stats['thermometer/bit_saturation_mean'] <= 1.0


def test_timm_deit_tiny_init_maps_qkv_and_mlp(monkeypatch):
    cfg = tiny_config()
    dim = cfg.dim
    hidden = cfg.mlp_hidden

    class FakeTimmModel:
        def state_dict(self):
            return {
                'cls_token': torch.full((1, 1, dim), 0.1),
                'pos_embed': torch.full((1, cfg.tokens, dim), 0.2),
                'patch_embed.proj.weight': torch.full(
                    (dim, 3, cfg.patch_size, cfg.patch_size), 0.3,
                ),
                'patch_embed.proj.bias': torch.full((dim,), 0.4),
                'blocks.0.norm1.weight': torch.full((dim,), 0.5),
                'blocks.0.norm1.bias': torch.full((dim,), 0.6),
                'blocks.0.attn.qkv.weight': torch.full((dim * 3, dim), 0.7),
                'blocks.0.attn.qkv.bias': torch.full((dim * 3,), 0.8),
                'blocks.0.attn.proj.weight': torch.full((dim, dim), 0.9),
                'blocks.0.attn.proj.bias': torch.full((dim,), 1.0),
                'blocks.0.norm2.weight': torch.full((dim,), 1.1),
                'blocks.0.norm2.bias': torch.full((dim,), 1.2),
                'blocks.0.mlp.fc1.weight': torch.full((hidden, dim), 1.3),
                'blocks.0.mlp.fc1.bias': torch.full((hidden,), 1.4),
                'blocks.0.mlp.fc2.weight': torch.full((dim, hidden), 1.5),
                'blocks.0.mlp.fc2.bias': torch.full((dim,), 1.6),
                'norm.weight': torch.full((dim,), 1.7),
                'norm.bias': torch.full((dim,), 1.8),
                'head.weight': torch.zeros(1000, dim),
                'head.bias': torch.zeros(1000),
            }

    fake_timm = SimpleNamespace(
        create_model=lambda model_name, pretrained: FakeTimmModel(),
    )
    monkeypatch.setitem(sys.modules, 'timm', fake_timm)

    model = build_deit_tiny_cifar10(cfg)
    stats = init_from_timm_deit_tiny(model, model_name='fake_deit', pretrained=True)

    assert stats['init_from_timm/copied'] == 18
    assert stats['init_from_timm/shape_mismatch'] == 2
    assert torch.allclose(model.blocks[0].attn.in_proj_weight, torch.full((dim * 3, dim), 0.7))
    assert torch.allclose(model.blocks[0].attn.out_proj.bias, torch.full((dim,), 1.0))
    assert torch.allclose(model.blocks[0].channel_mixer.net[0].weight, torch.full((hidden, dim), 1.3))
    assert torch.allclose(model.blocks[0].channel_mixer.net[3].bias, torch.full((dim,), 1.6))
