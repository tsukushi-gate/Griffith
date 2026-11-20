"""LL-ViT reproduction building blocks.

This module is intentionally separate from the editor samples.  The editor can
display these structures, but paper reproduction and long CIFAR-10 runs should
use the training script under ``experiments/``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

import torch
import torch.nn as nn

from ..encoding.thermometer import ThermometerEncoder
from ..layers.dwn import DWNLayer, LearnableMapping


@dataclass(slots=True)
class LLViTConfig:
    """Configuration matching the LL-ViT CIFAR-10 target by default."""

    image_size: int = 224
    patch_size: int = 16
    in_channels: int = 3
    num_classes: int = 10
    dim: int = 192
    depth: int = 12
    heads: int = 3
    mlp_ratio: int = 4
    thermometer_bits: int = 8
    thermometer_mode: Literal['uniform', 'distributive'] = 'uniform'
    lut_hidden: int = 768
    lut_out: int = 192
    lut_inputs: int = 6
    lut_mapping: Literal['random', 'learnable'] = 'learnable'
    lut_efd: Literal['paper', 'official'] = 'official'
    lut_ste: bool = True
    conditional_sum_bits: int = 4
    dropout: float = 0.0
    attention_dropout: float = 0.0

    @property
    def patch_tokens(self) -> int:
        side = self.image_size // self.patch_size
        return side * side

    @property
    def tokens(self) -> int:
        return self.patch_tokens + 1

    @property
    def mlp_hidden(self) -> int:
        return self.dim * self.mlp_ratio

    def paper_metadata(self) -> dict:
        return {
            'image_size': self.image_size,
            'patch_size': self.patch_size,
            'dim': self.dim,
            'encoders': self.depth,
            'heads': self.heads,
            'token_count': self.tokens,
            'channel_mixer_luts': [self.lut_hidden, self.lut_out],
            'thermometer_bits': self.thermometer_bits,
            'encoded_value': f'int{self.conditional_sum_bits}',
            'lut_inputs': self.lut_inputs,
            'lut_mapping': self.lut_mapping,
            'lut_efd': self.lut_efd,
        }


class ConditionalSum(nn.Module):
    """Dense encoded-value summation after binary LUT outputs.

    During training this uses fp32 encoded values.  When
    ``use_quantized_eval=True`` and the module is in eval mode, the encoded
    values are symmetrically quantized to int4-style levels and dequantized for
    forward execution.
    """

    def __init__(self, in_features: int, out_features: int | None = None,
                 quant_bits: int = 4):
        super().__init__()
        if quant_bits < 2:
            raise ValueError(f'quant_bits must be >= 2: {quant_bits}')
        self.in_features = in_features
        self.out_features = out_features or in_features
        self.quant_bits = quant_bits
        self.use_quantized_eval = False
        self.encoded_values = nn.Parameter(torch.empty(in_features, self.out_features))
        nn.init.normal_(self.encoded_values, mean=0.0, std=0.02)

    def quantized_encoded_values(self) -> tuple[torch.Tensor, torch.Tensor]:
        levels = 2 ** (self.quant_bits - 1) - 1
        max_abs = self.encoded_values.detach().abs().max().clamp_min(1e-8)
        scale = max_abs / levels
        q = torch.round(self.encoded_values / scale).clamp(-levels, levels)
        return q * scale, scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.in_features:
            raise ValueError(
                f'expected last dim {self.in_features}, got {x.shape[-1]}'
            )
        if not self.training and self.use_quantized_eval:
            values, _ = self.quantized_encoded_values()
        else:
            values = self.encoded_values
        return x @ values

    def extra_repr(self) -> str:
        return (
            f'in_features={self.in_features}, out_features={self.out_features}, '
            f'quant_bits={self.quant_bits}'
        )


class LLViTChannelMixer(nn.Module):
    """LL-ViT channel mixer: thermometer -> DWN -> DWN -> conditional sum."""

    def __init__(self, config: LLViTConfig):
        super().__init__()
        if config.lut_out != config.dim:
            raise ValueError(
                'The first reproduction target expects lut_out == dim '
                f'({config.lut_out} != {config.dim})'
            )
        if config.lut_hidden != config.mlp_hidden:
            raise ValueError(
                'The LL-ViT CIFAR-10 target expects lut_hidden == dim*4 '
                f'({config.lut_hidden} != {config.mlp_hidden})'
            )
        self.config = config
        self.thermometer = ThermometerEncoder(
            config.thermometer_bits, mode=config.thermometer_mode,
        )
        self.lut0 = DWNLayer(
            config.dim * config.thermometer_bits,
            config.lut_hidden,
            n_inputs=config.lut_inputs,
            mapping=config.lut_mapping,
            efd=config.lut_efd,
            ste=config.lut_ste,
        )
        self.lut1 = DWNLayer(
            config.lut_hidden,
            config.lut_out,
            n_inputs=config.lut_inputs,
            mapping=config.lut_mapping,
            efd=config.lut_efd,
            ste=config.lut_ste,
        )
        self.conditional_sum = ConditionalSum(
            config.lut_out, config.dim, quant_bits=config.conditional_sum_bits,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, tokens, dim = x.shape
        if dim != self.config.dim:
            raise ValueError(f'expected dim {self.config.dim}, got {dim}')
        h = x.reshape(batch * tokens, dim)
        h = self.thermometer(h)
        h = self.lut0(h)
        h = self.lut1(h)
        h = self.conditional_sum(h)
        return h.reshape(batch, tokens, dim)


class PatchEmbed(nn.Module):
    def __init__(self, config: LLViTConfig):
        super().__init__()
        self.config = config
        self.proj = nn.Conv2d(
            config.in_channels,
            config.dim,
            kernel_size=config.patch_size,
            stride=config.patch_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.proj(x)
        return h.flatten(2).transpose(1, 2)


class DeiTMLP(nn.Module):
    def __init__(self, config: LLViTConfig):
        super().__init__()
        hidden = config.mlp_hidden
        self.net = nn.Sequential(
            nn.Linear(config.dim, hidden),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(hidden, config.dim),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LLViTBlock(nn.Module):
    """Transformer encoder block with preserved MHA token mixer."""

    def __init__(self, config: LLViTConfig, *, channel_mixer: str = 'llvit'):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.dim)
        self.attn = nn.MultiheadAttention(
            config.dim,
            config.heads,
            dropout=config.attention_dropout,
            batch_first=True,
        )
        self.drop = nn.Dropout(config.dropout)
        self.norm2 = nn.LayerNorm(config.dim)
        if channel_mixer == 'llvit':
            self.channel_mixer = LLViTChannelMixer(config)
        elif channel_mixer == 'mlp':
            self.channel_mixer = DeiTMLP(config)
        else:
            raise ValueError(f'unknown channel_mixer: {channel_mixer}')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        attn, _ = self.attn(h, h, h, need_weights=False)
        x = x + self.drop(attn)
        return x + self.channel_mixer(self.norm2(x))


class LLViT(nn.Module):
    """Tiny ViT backbone used for baseline and LL-ViT reproduction."""

    def __init__(self, config: LLViTConfig, *, channel_mixer: str = 'llvit'):
        super().__init__()
        if config.image_size % config.patch_size != 0:
            raise ValueError('image_size must be divisible by patch_size')
        if config.dim % config.heads != 0:
            raise ValueError('dim must be divisible by heads')
        self.config = config
        self.channel_mixer_kind = channel_mixer
        self.patch_embed = PatchEmbed(config)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, config.tokens, config.dim))
        self.pos_drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(
            [LLViTBlock(config, channel_mixer=channel_mixer) for _ in range(config.depth)]
        )
        self.norm = nn.LayerNorm(config.dim)
        self.head = nn.Linear(config.dim, config.num_classes)
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        h = self.patch_embed(x)
        cls = self.cls_token.expand(h.shape[0], -1, -1)
        h = torch.cat([cls, h], dim=1) + self.pos_embed
        h = self.pos_drop(h)
        for block in self.blocks:
            h = block(h)
        h = self.norm(h)
        return h[:, 0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.forward_features(x))


def llvit_paper_config(**overrides) -> LLViTConfig:
    data = asdict(LLViTConfig())
    data.update(overrides)
    return LLViTConfig(**data)


def build_llvit_cifar10(config: LLViTConfig | None = None) -> LLViT:
    return LLViT(config or LLViTConfig(), channel_mixer='llvit')


def build_deit_tiny_cifar10(config: LLViTConfig | None = None) -> LLViT:
    return LLViT(config or LLViTConfig(), channel_mixer='mlp')


def init_from_timm_deit_tiny(
    model: LLViT,
    *,
    model_name: str = 'deit_tiny_patch16_224',
    pretrained: bool = True,
) -> dict[str, int | str | bool]:
    """Initialize the local LLViT-shaped model from a timm DeiT checkpoint.

    The reproduction code keeps a local module structure so the baseline
    checkpoint can later seed LL-ViT by name.  timm uses a fused qkv projection,
    while this module uses ``nn.MultiheadAttention``; this mapper copies the
    matching tensors and skips classifier or MLP tensors when their target shape
    does not exist, as in LL-ViT channel mixers.
    """

    try:
        import timm
    except Exception as e:  # pragma: no cover - optional dependency guard
        raise RuntimeError(
            'timm is required for --init-from-timm. Run uv sync with the '
            'cpu or cu128 extra before launching LL-ViT reproduction.'
        ) from e

    source_model = timm.create_model(model_name, pretrained=pretrained)
    source = source_model.state_dict()
    target = model.state_dict()
    updates: dict[str, torch.Tensor] = {}
    candidates: dict[str, str] = {
        'cls_token': 'cls_token',
        'pos_embed': 'pos_embed',
        'patch_embed.proj.weight': 'patch_embed.proj.weight',
        'patch_embed.proj.bias': 'patch_embed.proj.bias',
        'norm.weight': 'norm.weight',
        'norm.bias': 'norm.bias',
        'head.weight': 'head.weight',
        'head.bias': 'head.bias',
    }
    for i in range(model.config.depth):
        src = f'blocks.{i}'
        dst = f'blocks.{i}'
        candidates.update({
            f'{src}.norm1.weight': f'{dst}.norm1.weight',
            f'{src}.norm1.bias': f'{dst}.norm1.bias',
            f'{src}.attn.qkv.weight': f'{dst}.attn.in_proj_weight',
            f'{src}.attn.qkv.bias': f'{dst}.attn.in_proj_bias',
            f'{src}.attn.proj.weight': f'{dst}.attn.out_proj.weight',
            f'{src}.attn.proj.bias': f'{dst}.attn.out_proj.bias',
            f'{src}.norm2.weight': f'{dst}.norm2.weight',
            f'{src}.norm2.bias': f'{dst}.norm2.bias',
            f'{src}.mlp.fc1.weight': f'{dst}.channel_mixer.net.0.weight',
            f'{src}.mlp.fc1.bias': f'{dst}.channel_mixer.net.0.bias',
            f'{src}.mlp.fc2.weight': f'{dst}.channel_mixer.net.3.weight',
            f'{src}.mlp.fc2.bias': f'{dst}.channel_mixer.net.3.bias',
        })

    missing_source = 0
    missing_target = 0
    shape_mismatch = 0
    for src_key, dst_key in candidates.items():
        if src_key not in source:
            missing_source += 1
            continue
        if dst_key not in target:
            missing_target += 1
            continue
        value = source[src_key]
        if target[dst_key].shape != value.shape:
            shape_mismatch += 1
            continue
        updates[dst_key] = value.detach().to(dtype=target[dst_key].dtype)

    model.load_state_dict(updates, strict=False)
    return {
        'init_from_timm/model': model_name,
        'init_from_timm/pretrained': pretrained,
        'init_from_timm/copied': len(updates),
        'init_from_timm/missing_source': missing_source,
        'init_from_timm/missing_target': missing_target,
        'init_from_timm/shape_mismatch': shape_mismatch,
    }


def set_conditional_sum_quantized(model: nn.Module, enabled: bool) -> None:
    for module in model.modules():
        if isinstance(module, ConditionalSum):
            module.use_quantized_eval = enabled


def llvit_channel_mixers(model: nn.Module) -> list[LLViTChannelMixer]:
    return [m for m in model.modules() if isinstance(m, LLViTChannelMixer)]


def _channel_mixer_inputs_to_stats(
    mixers: list[LLViTChannelMixer],
    samples: dict[int, list[torch.Tensor]],
) -> dict[str, float | int]:
    stat_values: list[torch.Tensor] = []
    saturation: list[torch.Tensor] = []
    calibrated = 0
    for idx, mixer in enumerate(mixers):
        chunks = samples.get(idx, [])
        if not chunks:
            continue
        data = torch.cat(chunks, dim=0)
        mixer.thermometer.fit(data)
        thresholds = mixer.thermometer.thresholds.detach().float()
        encoded = mixer.thermometer.encode(data).float()
        stat_values.append(thresholds.reshape(-1))
        saturation.append(encoded.mean(dim=(0, 1)))
        calibrated += 1

    if not stat_values:
        return {'thermometer/calibrated_mixers': 0}
    all_thresholds = torch.cat(stat_values)
    all_saturation = torch.cat(saturation) if saturation else torch.tensor([0.0])
    return {
        'thermometer/calibrated_mixers': calibrated,
        'thermometer/threshold_mean': float(all_thresholds.mean().item()),
        'thermometer/threshold_std': float(all_thresholds.std(unbiased=False).item()),
        'thermometer/threshold_min': float(all_thresholds.min().item()),
        'thermometer/threshold_max': float(all_thresholds.max().item()),
        'thermometer/bit_saturation_mean': float(all_saturation.mean().item()),
        'thermometer/bit_saturation_min': float(all_saturation.min().item()),
        'thermometer/bit_saturation_max': float(all_saturation.max().item()),
    }


@torch.no_grad()
def calibrate_thermometers(
    model: nn.Module,
    loader,
    device: torch.device | str,
    *,
    max_batches: int = 8,
    max_samples_per_mixer: int = 8192,
) -> dict[str, float | int]:
    """Fit each LL-ViT thermometer from LayerNorm/channel-mixer inputs."""

    mixers = llvit_channel_mixers(model)
    if not mixers or max_batches <= 0:
        return {'thermometer/calibrated_mixers': 0}
    for mixer in mixers:
        thermo = mixer.thermometer
        if thermo.thresholds is None and thermo.mode == 'distributive':
            steps = torch.linspace(-1.0, 1.0, thermo.bits + 2)[1:-1]
            thermo.thresholds = steps.unsqueeze(1).expand(thermo.bits, mixer.config.dim)

    samples: dict[int, list[torch.Tensor]] = {idx: [] for idx in range(len(mixers))}
    remaining = {idx: max_samples_per_mixer for idx in range(len(mixers))}

    def hook(module, inputs, _output):
        if not inputs:
            return
        key = mixer_index[id(module)]
        quota = remaining.get(key, 0)
        if quota <= 0:
            return
        x = inputs[0].detach().reshape(-1, module.config.dim).cpu()
        take = min(quota, x.shape[0])
        if take > 0:
            samples[key].append(x[:take])
            remaining[key] = quota - take

    mixer_index = {id(m): idx for idx, m in enumerate(mixers)}
    handles = [m.register_forward_hook(hook) for m in mixers]
    was_training = model.training
    model.eval()
    try:
        for batch_idx, (images, _labels) in enumerate(loader):
            if batch_idx >= max_batches:
                break
            model(images.to(device, non_blocking=True))
            if all(v <= 0 for v in remaining.values()):
                break
    finally:
        for handle in handles:
            handle.remove()
        model.train(was_training)

    return _channel_mixer_inputs_to_stats(mixers, samples)


@torch.no_grad()
def calibrate_thermometers_from_teacher(
    student: nn.Module,
    teacher: nn.Module,
    loader,
    device: torch.device | str,
    *,
    max_batches: int = 8,
    max_samples_per_mixer: int = 8192,
) -> dict[str, float | int]:
    """Fit LL-ViT thermometers from the baseline/teacher channel-mixer inputs."""

    mixers = llvit_channel_mixers(student)
    teacher_mixers = [
        module for name, module in teacher.named_modules()
        if name.endswith('channel_mixer')
    ]
    if not mixers or max_batches <= 0:
        return {'thermometer/calibrated_mixers': 0}
    if len(teacher_mixers) < len(mixers):
        raise ValueError(
            f'teacher has {len(teacher_mixers)} channel mixers, '
            f'but student has {len(mixers)} LL-ViT mixers'
        )

    samples: dict[int, list[torch.Tensor]] = {idx: [] for idx in range(len(mixers))}
    remaining = {idx: max_samples_per_mixer for idx in range(len(mixers))}

    def make_hook(idx: int):
        def hook(_module, inputs, _output):
            if not inputs:
                return
            quota = remaining.get(idx, 0)
            if quota <= 0:
                return
            x = inputs[0].detach().reshape(-1, mixers[idx].config.dim).cpu()
            take = min(quota, x.shape[0])
            if take > 0:
                samples[idx].append(x[:take])
                remaining[idx] = quota - take
        return hook

    handles = [
        teacher_mixers[idx].register_forward_hook(make_hook(idx))
        for idx in range(len(mixers))
    ]
    was_training = teacher.training
    teacher.eval()
    try:
        for batch_idx, (images, _labels) in enumerate(loader):
            if batch_idx >= max_batches:
                break
            teacher(images.to(device, non_blocking=True))
            if all(v <= 0 for v in remaining.values()):
                break
    finally:
        for handle in handles:
            handle.remove()
        teacher.train(was_training)

    return _channel_mixer_inputs_to_stats(mixers, samples)


def split_llvit_param_groups(
    model: nn.Module,
    *,
    ann_lr: float,
    logic_lr: float,
    weight_decay: float,
) -> list[dict]:
    """Split ANN and logic/DWN parameters for optimizer construction."""

    logic_ids: set[int] = set()
    for module in model.modules():
        if isinstance(module, DWNLayer | LearnableMapping | ConditionalSum):
            logic_ids.update(id(p) for p in module.parameters(recurse=False))
            if isinstance(module, DWNLayer):
                logic_ids.update(id(p) for p in module.parameters(recurse=True))

    logic_params = []
    ann_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.endswith('bias') or 'norm' in name or name.endswith('pos_embed'):
            no_decay_params.append(param)
        elif id(param) in logic_ids:
            logic_params.append(param)
        else:
            ann_params.append(param)

    groups: list[dict] = []
    if ann_params:
        groups.append({'params': ann_params, 'lr': ann_lr, 'weight_decay': weight_decay})
    if logic_params:
        groups.append({'params': logic_params, 'lr': logic_lr, 'weight_decay': 0.0})
    if no_decay_params:
        groups.append({'params': no_decay_params, 'lr': ann_lr, 'weight_decay': 0.0})
    return groups


def llvit_component_summary(model: nn.Module) -> dict[str, int | str]:
    dwn_layers = [m for m in model.modules() if isinstance(m, DWNLayer)]
    conditional = [m for m in model.modules() if isinstance(m, ConditionalSum)]
    summary: dict[str, int | str] = {
        'dwn_layers': len(dwn_layers),
        'conditional_sum_layers': len(conditional),
        'dwn_luts': sum(m.n_luts for m in dwn_layers),
        'conditional_sum_values': sum(m.encoded_values.numel() for m in conditional),
    }
    lut_inputs = sorted({m.n_inputs for m in dwn_layers})
    if lut_inputs:
        summary['lut_inputs'] = '+'.join(f'LUT{n}' for n in lut_inputs)
    return summary


__all__ = [
    'ConditionalSum',
    'DeiTMLP',
    'LLViT',
    'LLViTBlock',
    'LLViTChannelMixer',
    'LLViTConfig',
    'PatchEmbed',
    'build_deit_tiny_cifar10',
    'build_llvit_cifar10',
    'calibrate_thermometers_from_teacher',
    'calibrate_thermometers',
    'init_from_timm_deit_tiny',
    'llvit_component_summary',
    'llvit_channel_mixers',
    'llvit_paper_config',
    'set_conditional_sum_quantized',
    'split_llvit_param_groups',
]
