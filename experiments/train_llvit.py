"""LL-ViT CIFAR-10 reproduction training entrypoint.

Gate order:
  1. fake-data/smoke forward-backward
  2. 32/512 sample overfit
  3. DeiT/I-ViT-T baseline CIFAR-10
  4. LL-ViT CIFAR-10 full run

Example:
  uv run python experiments/train_llvit.py --model llvit --fake-data \
    --epochs 1 --batch-size 2 --smoke-steps 2 \
    --image-size 32 --dim 24 --depth 1 --heads 3 \
    --thermometer-bits 2 --lut-hidden 96 --lut-out 24 --lut-inputs 2
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
EXP_DIR = Path(__file__).resolve().parent
if str(EXP_DIR) not in sys.path:
    sys.path.insert(0, str(EXP_DIR))

from checkpoint_utils import load_checkpoint, save_checkpoint, setup_checkpoint_dir
from wandb_utils import (
    finish_wandb,
    init_wandb,
    log_checkpoint_artifact,
    log_metrics,
    log_model_checkpoint,
)

from griffith_logic.models.llvit import (
    LLViTConfig,
    build_deit_tiny_cifar10,
    build_llvit_cifar10,
    calibrate_thermometers,
    calibrate_thermometers_from_teacher,
    init_from_timm_deit_tiny,
    llvit_component_summary,
    set_conditional_sum_quantized,
    split_llvit_param_groups,
)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _load_yaml_defaults(argv: list[str]) -> tuple[dict[str, Any], list[str]]:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument('--config')
    known, rest = pre.parse_known_args(argv)
    if not known.config:
        return {}, argv
    path = Path(known.config)
    data = yaml.safe_load(path.read_text(encoding='utf-8')) or {}
    defaults: dict[str, Any] = {'config': known.config}
    for key, value in data.items():
        if isinstance(value, dict):
            continue
        defaults[key.replace('-', '_')] = value
    return defaults, rest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    argv = list(sys.argv[1:] if argv is None else argv)
    defaults, argv = _load_yaml_defaults(argv)
    p = argparse.ArgumentParser(description='Train LL-ViT / DeiT tiny on CIFAR-10')
    p.add_argument('--config', help='Flat YAML recipe. CLI arguments override it.')
    p.add_argument('--model', choices=['baseline', 'llvit'], default='llvit')
    p.add_argument('--dataset', choices=['cifar10'], default='cifar10')
    p.add_argument('--data-root', default='data-llvit')
    p.add_argument('--output-dir', default='checkpoints/llvit')
    p.add_argument('--run-dir', default=None)
    p.add_argument('--resume-checkpoint', default=None)
    p.add_argument('--download', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--fake-data', action='store_true', default=False)
    p.add_argument('--fake-train-size', type=int, default=64)
    p.add_argument('--fake-val-size', type=int, default=32)
    p.add_argument('--overfit-samples', type=int, default=0)
    p.add_argument('--limit-train', type=int, default=0)
    p.add_argument('--limit-val', type=int, default=0)

    p.add_argument('--image-size', type=int, default=224)
    p.add_argument('--patch-size', type=int, default=16)
    p.add_argument('--dim', type=int, default=192)
    p.add_argument('--depth', type=int, default=12)
    p.add_argument('--heads', type=int, default=3)
    p.add_argument('--mlp-ratio', type=int, default=4)
    p.add_argument('--thermometer-bits', type=int, default=8)
    p.add_argument('--thermometer-mode', choices=['uniform', 'distributive'], default='uniform')
    p.add_argument('--lut-hidden', type=int, default=768)
    p.add_argument('--lut-out', type=int, default=192)
    p.add_argument('--lut-inputs', type=int, default=6)
    p.add_argument('--lut-mapping', choices=['random', 'learnable'], default='learnable')
    p.add_argument('--lut-efd', choices=['paper', 'official'], default='official')

    p.add_argument('--epochs', type=int, default=300)
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--num-workers', type=int, default=4)
    p.add_argument('--lr', type=float, default=5e-4)
    p.add_argument('--ann-lr', type=float, default=5e-4)
    p.add_argument('--logic-lr', type=float, default=2e-2)
    p.add_argument('--weight-decay', type=float, default=0.05)
    p.add_argument('--warmup-epochs', type=int, default=5)
    p.add_argument('--min-lr-factor', type=float, default=0.01)
    p.add_argument('--label-smoothing', type=float, default=0.1)
    p.add_argument('--mixup', type=float, default=0.8)
    p.add_argument('--cutmix', type=float, default=1.0)
    p.add_argument('--randaugment', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--random-erasing', type=float, default=0.25)
    p.add_argument('--amp', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--grad-clip', type=float, default=1.0)
    p.add_argument('--ema-decay', type=float, default=0.0)
    p.add_argument(
        '--calibrate-thermometer-source',
        choices=['student', 'teacher'],
        default='teacher',
    )
    p.add_argument('--calibrate-thermometer-batches', type=int, default=8)
    p.add_argument('--calibrate-thermometer-samples', type=int, default=8192)
    p.add_argument('--init-from-baseline', default=None)
    p.add_argument('--init-from-llvit', default=None)
    p.add_argument('--init-from-llvit-ema', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--init-from-timm', action='store_true', default=False)
    p.add_argument('--timm-model', default='deit_tiny_patch16_224')
    p.add_argument('--timm-pretrained', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--distill-checkpoint', default=None)
    p.add_argument('--distill-alpha', type=float, default=0.0)
    p.add_argument('--feature-distill-alpha', type=float, default=0.0)
    p.add_argument('--distill-temperature', type=float, default=2.0)
    p.add_argument('--channel-mixer-pretrain-epochs', type=int, default=0)
    p.add_argument('--channel-mixer-pretrain-batches', type=int, default=0)
    p.add_argument('--channel-mixer-pretrain-lr', type=float, default=0.0)
    p.add_argument('--channel-mixer-pretrain-weight-decay', type=float, default=0.0)
    p.add_argument('--smoke-steps', type=int, default=0)
    p.add_argument('--eval-interval', type=int, default=1)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda', 'mps'])
    p.add_argument('--conditional-sum-quant-eval', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--expected-acc', type=float, default=0.0)
    p.add_argument('--stop-at-acc', type=float, default=0.0)

    p.add_argument('--wandb', action=argparse.BooleanOptionalAction, default=False)
    p.add_argument('--run-name', default=None)
    p.add_argument('--wandb-group', default='llvit-cifar10')
    p.set_defaults(**defaults)
    return p.parse_args(argv)


def setup_llvit_checkpoint_paths(args: argparse.Namespace, run_name: str) -> tuple[str, str, str]:
    if args.run_dir:
        run_dir = args.run_dir
        os.makedirs(run_dir, exist_ok=True)
        return run_dir, os.path.join(run_dir, 'latest.pt'), os.path.join(run_dir, 'best.pt')
    return setup_checkpoint_dir(run_name, args.output_dir)


def resolve_resume_checkpoint(args: argparse.Namespace, latest_path: str) -> str | None:
    # For Modal preemption restarts, prefer the stable run directory's latest.pt
    # once it exists. The original resume checkpoint is only a bootstrap source.
    if args.run_dir and os.path.exists(latest_path):
        return latest_path
    if args.resume_checkpoint:
        return args.resume_checkpoint
    return None


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(name: str) -> torch.device:
    if name == 'auto':
        if torch.cuda.is_available():
            return torch.device('cuda')
        if torch.backends.mps.is_available():
            return torch.device('mps')
        return torch.device('cpu')
    return torch.device(name)


def build_config(args: argparse.Namespace) -> LLViTConfig:
    return LLViTConfig(
        image_size=args.image_size,
        patch_size=args.patch_size,
        num_classes=10,
        dim=args.dim,
        depth=args.depth,
        heads=args.heads,
        mlp_ratio=args.mlp_ratio,
        thermometer_bits=args.thermometer_bits,
        thermometer_mode=args.thermometer_mode,
        lut_hidden=args.lut_hidden,
        lut_out=args.lut_out,
        lut_inputs=args.lut_inputs,
        lut_mapping=args.lut_mapping,
        lut_efd=args.lut_efd,
    )


def build_model(args: argparse.Namespace, config: LLViTConfig) -> nn.Module:
    if args.model == 'baseline':
        return build_deit_tiny_cifar10(config)
    return build_llvit_cifar10(config)


def _torchvision():
    try:
        import torchvision
        from torchvision import transforms
    except Exception as e:  # pragma: no cover - environment dependent
        raise RuntimeError(
            'torchvision is required for train_llvit.py. '
            'Install the cpu/cu128 extra before running CIFAR-10 experiments.'
        ) from e
    return torchvision, transforms


def build_transforms(args: argparse.Namespace):
    _, transforms = _torchvision()
    train_ops: list[Any] = [
        transforms.RandomResizedCrop(args.image_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
    ]
    if args.randaugment:
        train_ops.append(transforms.RandAugment(num_ops=2, magnitude=9))
    train_ops.extend([
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    if args.random_erasing > 0:
        train_ops.append(transforms.RandomErasing(p=args.random_erasing))
    val_ops = [
        transforms.Resize(int(args.image_size * 256 / 224)),
        transforms.CenterCrop(args.image_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
    return transforms.Compose(train_ops), transforms.Compose(val_ops)


def maybe_subset(dataset, n: int):
    if n <= 0 or n >= len(dataset):
        return dataset
    return torch.utils.data.Subset(dataset, list(range(n)))


def build_loaders(args: argparse.Namespace):
    torchvision, _ = _torchvision()
    train_tf, val_tf = build_transforms(args)
    if args.fake_data:
        train_set = torchvision.datasets.FakeData(
            size=args.fake_train_size,
            image_size=(3, args.image_size, args.image_size),
            num_classes=10,
            transform=train_tf,
        )
        val_set = torchvision.datasets.FakeData(
            size=args.fake_val_size,
            image_size=(3, args.image_size, args.image_size),
            num_classes=10,
            transform=val_tf,
        )
    else:
        train_set = torchvision.datasets.CIFAR10(
            args.data_root, train=True, download=args.download, transform=train_tf,
        )
        val_set = torchvision.datasets.CIFAR10(
            args.data_root, train=False, download=args.download, transform=val_tf,
        )
    if args.overfit_samples > 0:
        train_set = maybe_subset(train_set, args.overfit_samples)
        val_set = maybe_subset(train_set, args.overfit_samples)
    else:
        train_set = maybe_subset(train_set, args.limit_train)
        val_set = maybe_subset(val_set, args.limit_val)

    workers = 0 if args.fake_data else args.num_workers
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    val_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader


def soft_cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return -(targets * F.log_softmax(logits, dim=1)).sum(dim=1).mean()


def smooth_one_hot(labels: torch.Tensor, classes: int, smoothing: float) -> torch.Tensor:
    off = smoothing / classes
    on = 1.0 - smoothing + off
    out = torch.full((labels.shape[0], classes), off, device=labels.device)
    out.scatter_(1, labels.unsqueeze(1), on)
    return out


def rand_bbox(size: torch.Size, lam: float) -> tuple[int, int, int, int]:
    width = size[-1]
    height = size[-2]
    cut_ratio = math.sqrt(1.0 - lam)
    cut_w = int(width * cut_ratio)
    cut_h = int(height * cut_ratio)
    cx = random.randint(0, width)
    cy = random.randint(0, height)
    x1 = max(cx - cut_w // 2, 0)
    y1 = max(cy - cut_h // 2, 0)
    x2 = min(cx + cut_w // 2, width)
    y2 = min(cy + cut_h // 2, height)
    return x1, y1, x2, y2


def apply_batch_aug(
    images: torch.Tensor,
    labels: torch.Tensor,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, torch.Tensor]:
    targets = smooth_one_hot(labels, 10, args.label_smoothing)
    use_cutmix = args.cutmix > 0 and random.random() < 0.5
    use_mixup = args.mixup > 0 and not use_cutmix
    if not (use_cutmix or use_mixup):
        return images, targets

    index = torch.randperm(images.shape[0], device=images.device)
    if use_cutmix:
        lam = torch.distributions.Beta(args.cutmix, args.cutmix).sample().item()
        x1, y1, x2, y2 = rand_bbox(images.size(), lam)
        mixed = images.clone()
        mixed[:, :, y1:y2, x1:x2] = images[index, :, y1:y2, x1:x2]
        area = (x2 - x1) * (y2 - y1)
        lam = 1.0 - area / float(images.shape[-1] * images.shape[-2])
        return mixed, targets * lam + targets[index] * (1.0 - lam)

    lam = torch.distributions.Beta(args.mixup, args.mixup).sample().item()
    return images * lam + images[index] * (1.0 - lam), targets * lam + targets[index] * (1.0 - lam)


def accuracy(logits: torch.Tensor, labels: torch.Tensor, topk=(1, 5)) -> list[float]:
    maxk = min(max(topk), logits.shape[1])
    _, pred = logits.topk(maxk, dim=1)
    pred = pred.t()
    correct = pred.eq(labels.reshape(1, -1).expand_as(pred))
    out = []
    for k in topk:
        kk = min(k, logits.shape[1])
        out.append(correct[:kk].reshape(-1).float().sum().mul_(100.0 / labels.shape[0]).item())
    return out


def build_scheduler(args: argparse.Namespace, optimizer, steps_per_epoch: int):
    total = max(1, args.epochs * steps_per_epoch)
    warmup = max(0, args.warmup_epochs * steps_per_epoch)

    def lr_lambda(step: int) -> float:
        if warmup > 0 and step < warmup:
            return max(1e-8, float(step + 1) / warmup)
        progress = (step - warmup) / max(1, total - warmup)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
        return args.min_lr_factor + (1.0 - args.min_lr_factor) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


class ModelEma:
    """Minimal EMA state for evaluation stability."""

    def __init__(self, model: nn.Module, decay: float):
        self.decay = decay
        self.shadow = {
            k: v.detach().clone()
            for k, v in model.state_dict().items()
            if torch.is_floating_point(v)
        }
        self.backup: dict[str, torch.Tensor] = {}

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for k, v in model.state_dict().items():
            if k not in self.shadow or not torch.is_floating_point(v):
                continue
            value = v.detach().to(self.shadow[k].device)
            self.shadow[k].mul_(self.decay).add_(value, alpha=1.0 - self.decay)

    @torch.no_grad()
    def store(self, model: nn.Module) -> None:
        self.backup = {
            k: v.detach().clone()
            for k, v in model.state_dict().items()
            if k in self.shadow
        }

    @torch.no_grad()
    def copy_to(self, model: nn.Module) -> None:
        state = model.state_dict()
        for k, v in self.shadow.items():
            state[k].copy_(v.to(state[k].device))

    @torch.no_grad()
    def restore(self, model: nn.Module) -> None:
        state = model.state_dict()
        for k, v in self.backup.items():
            state[k].copy_(v.to(state[k].device))
        self.backup = {}

    def state_dict(self) -> dict[str, Any]:
        return {
            'decay': self.decay,
            'shadow': {k: v.detach().cpu() for k, v in self.shadow.items()},
        }

    def load_state_dict(self, state: dict[str, Any], device: torch.device) -> None:
        self.decay = float(state.get('decay', self.decay))
        shadow = state.get('shadow', {})
        if not isinstance(shadow, dict):
            return
        self.shadow = {
            k: v.detach().to(device).clone()
            for k, v in shadow.items()
            if torch.is_tensor(v)
        }
        self.backup = {}


def git_metadata() -> dict[str, str | bool]:
    def run_git(args: list[str]) -> str:
        try:
            return subprocess.check_output(
                ['git', *args], stderr=subprocess.DEVNULL, text=True,
            ).strip()
        except Exception:
            return ''

    diff_stat = run_git(['diff', '--stat'])
    return {
        'git_hash': run_git(['rev-parse', '--short', 'HEAD']),
        'git_branch': run_git(['branch', '--show-current']),
        'git_dirty': bool(diff_stat or run_git(['status', '--short'])),
        'git_diff_stat': diff_stat,
    }


def init_from_baseline(model: nn.Module, checkpoint: str, device: torch.device) -> dict[str, int]:
    data = torch.load(checkpoint, map_location=device, weights_only=False)
    source = data.get('model_state_dict', data)
    target = model.state_dict()
    copied: dict[str, torch.Tensor] = {}
    skipped = 0
    for key, value in source.items():
        if '.channel_mixer.' in key:
            skipped += 1
            continue
        if key in target and target[key].shape == value.shape:
            copied[key] = value
        else:
            skipped += 1
    model.load_state_dict(copied, strict=False)
    return {'init_from_baseline/copied': len(copied), 'init_from_baseline/skipped': skipped}


def init_from_llvit_checkpoint(
    model: nn.Module,
    checkpoint: str,
    device: torch.device,
    *,
    use_ema: bool = True,
) -> dict[str, int | str]:
    data = torch.load(checkpoint, map_location=device, weights_only=False)
    source = data.get('model_state_dict', data)
    target = model.state_dict()
    modules = dict(model.named_modules())
    copied: dict[str, torch.Tensor] = {}
    skipped = 0

    def copy_value(key: str, value: Any) -> bool:
        if not torch.is_tensor(value):
            return False
        if key in target and target[key].shape == value.shape:
            copied[key] = value.to(device=target[key].device, dtype=target[key].dtype)
            return True
        if key.endswith('.thresholds'):
            module_name = key.rsplit('.', 1)[0]
            module = modules.get(module_name)
            if module is not None and hasattr(module, 'thresholds'):
                module.thresholds = value.detach().to(device)
                return True
        return False

    for key, value in source.items():
        if not copy_value(key, value):
            skipped += 1
    ema_copied = 0
    source_kind = 'model'
    ema_state = data.get('extra_state', {}).get('ema_state_dict')
    if use_ema and isinstance(ema_state, dict):
        shadow = ema_state.get('shadow', {})
        if isinstance(shadow, dict):
            for key, value in shadow.items():
                if copy_value(key, value):
                    ema_copied += 1
            if ema_copied:
                source_kind = 'model+ema'
    model.load_state_dict(copied, strict=False)
    return {
        'init_from_llvit/copied': len(copied),
        'init_from_llvit/ema_copied': ema_copied,
        'init_from_llvit/skipped': skipped,
        'init_from_llvit/source': source_kind,
    }


def build_teacher(
    args: argparse.Namespace,
    config: LLViTConfig,
    device: torch.device,
) -> nn.Module | None:
    if (
        not args.distill_checkpoint
        or (
            args.distill_alpha <= 0
            and args.feature_distill_alpha <= 0
            and args.channel_mixer_pretrain_epochs <= 0
            and not (
                args.model == 'llvit'
                and args.calibrate_thermometer_source == 'teacher'
            )
        )
    ):
        return None
    teacher = build_deit_tiny_cifar10(config).to(device)
    load_checkpoint(args.distill_checkpoint, teacher, device=str(device))
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    return teacher


def _channel_mixer_modules(model: nn.Module) -> list[nn.Module]:
    return [
        module for name, module in model.named_modules()
        if name.endswith('channel_mixer')
    ]


def _checkpoint_extra_state(ema: ModelEma | None) -> dict[str, Any]:
    return {'ema_state_dict': ema.state_dict()} if ema is not None else {}


def _load_ema_from_checkpoint(ema: ModelEma | None, resume: dict, device: torch.device) -> bool:
    if ema is None:
        return False
    extra = resume.get('extra_state', {})
    state = extra.get('ema_state_dict') if isinstance(extra, dict) else None
    if not isinstance(state, dict):
        return False
    ema.load_state_dict(state, device)
    return True


def pretrain_channel_mixers(
    model: nn.Module,
    teacher: nn.Module,
    loader,
    device: torch.device,
    args: argparse.Namespace,
) -> dict[str, float | int]:
    """Fit LL-ViT channel mixers directly to teacher MLP inputs/outputs.

    This is a local distillation phase before classifier fine-tuning.  It keeps
    MHA/patch/head fixed and optimizes only the LL-ViT DWN channel mixers
    against the baseline MLP transfer function on real augmented images.
    """

    epochs = int(args.channel_mixer_pretrain_epochs)
    if epochs <= 0:
        return {}
    if teacher is None:
        raise ValueError('--channel-mixer-pretrain-epochs requires --distill-checkpoint')

    student_mixers = _channel_mixer_modules(model)
    teacher_mixers = _channel_mixer_modules(teacher)
    if len(student_mixers) != len(teacher_mixers):
        raise ValueError(
            f'channel mixer count mismatch: student={len(student_mixers)} '
            f'teacher={len(teacher_mixers)}'
        )

    params = [
        p for mixer in student_mixers
        for p in mixer.parameters()
        if p.requires_grad
    ]
    if not params:
        return {'pretrain/channel_mixer_steps': 0}

    optimizer = torch.optim.AdamW(
        params,
        lr=args.channel_mixer_pretrain_lr or args.logic_lr,
        weight_decay=args.channel_mixer_pretrain_weight_decay,
    )
    amp_enabled = bool(args.amp and device.type == 'cuda')
    try:
        scaler = torch.amp.GradScaler('cuda', enabled=amp_enabled)
    except TypeError:  # older torch
        scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    teacher_inputs: list[torch.Tensor] = []
    teacher_outputs: list[torch.Tensor] = []

    def teacher_hook(_module, inputs, output):
        if not inputs:
            return
        teacher_inputs.append(inputs[0].detach())
        teacher_outputs.append(output.detach())

    hooks = [m.register_forward_hook(teacher_hook) for m in teacher_mixers]
    was_training = model.training
    teacher_was_training = teacher.training
    model.train()
    teacher.eval()
    steps = 0
    last_loss = 0.0
    total_loss = 0.0
    total_batches = 0
    try:
        for epoch in range(epochs):
            for batch_idx, (images, _labels) in enumerate(loader):
                if args.channel_mixer_pretrain_batches and batch_idx >= args.channel_mixer_pretrain_batches:
                    break
                images = images.to(device, non_blocking=True)
                teacher_inputs.clear()
                teacher_outputs.clear()
                with torch.no_grad():
                    teacher(images)

                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=amp_enabled):
                    losses = []
                    for mixer, source, target in zip(
                        student_mixers, teacher_inputs, teacher_outputs, strict=False,
                    ):
                        pred = mixer(source.to(device))
                        losses.append(F.mse_loss(pred, target.to(device, dtype=pred.dtype)))
                    if not losses:
                        continue
                    loss = torch.stack(losses).mean()
                if torch.isnan(loss):
                    raise RuntimeError('channel mixer pretrain loss became NaN')
                scaler.scale(loss).backward()
                if args.grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                steps += 1
                last_loss = float(loss.detach().item())
                total_loss += last_loss
                total_batches += 1
    finally:
        for hook in hooks:
            hook.remove()
        model.train(was_training)
        teacher.train(teacher_was_training)

    mean_loss = total_loss / max(1, total_batches)
    return {
        'pretrain/channel_mixer_epochs': epochs,
        'pretrain/channel_mixer_steps': steps,
        'pretrain/channel_mixer_loss': mean_loss,
        'pretrain/channel_mixer_last_loss': last_loss,
    }


def train_one_epoch(
    model: nn.Module,
    loader,
    optimizer,
    scheduler,
    scaler,
    device: torch.device,
    args: argparse.Namespace,
    epoch: int,
    global_step: int,
    *,
    teacher: nn.Module | None = None,
    ema: ModelEma | None = None,
) -> tuple[dict[str, float], int, bool]:
    model.train()
    if teacher is not None:
        teacher.eval()
    total_loss = 0.0
    total = 0
    top1_sum = 0.0
    student_features: list[torch.Tensor] = []
    teacher_features: list[torch.Tensor] = []

    def student_hook(_module, _inputs, output):
        student_features.append(output)

    def teacher_hook(_module, _inputs, output):
        teacher_features.append(output.detach())

    hooks = []
    if teacher is not None and args.feature_distill_alpha > 0:
        hooks.extend(m.register_forward_hook(student_hook) for m in _channel_mixer_modules(model))
        hooks.extend(m.register_forward_hook(teacher_hook) for m in _channel_mixer_modules(teacher))
    try:
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            images, targets = apply_batch_aug(images, labels, args)
            student_features.clear()
            teacher_features.clear()

            optimizer.zero_grad(set_to_none=True)
            amp_enabled = bool(args.amp and device.type == 'cuda')
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=amp_enabled):
                teacher_logits = None
                if teacher is not None:
                    with torch.no_grad():
                        teacher_logits = teacher(images)
                logits = model(images)
                supervised_loss = soft_cross_entropy(logits, targets)
                loss = supervised_loss
                distill_loss = torch.zeros((), device=device)
                feature_loss = torch.zeros((), device=device)
                if teacher_logits is not None and args.distill_alpha > 0:
                    temp = args.distill_temperature
                    distill_loss = F.kl_div(
                        F.log_softmax(logits / temp, dim=1),
                        F.softmax(teacher_logits / temp, dim=1),
                        reduction='batchmean',
                    ) * (temp * temp)
                    loss = (1.0 - args.distill_alpha) * supervised_loss + (
                        args.distill_alpha * distill_loss
                    )
                if teacher_features and args.feature_distill_alpha > 0:
                    pairs = zip(student_features, teacher_features, strict=False)
                    losses = [
                        F.mse_loss(student, target.to(student.device, dtype=student.dtype))
                        for student, target in pairs
                        if student.shape == target.shape
                    ]
                    if losses:
                        feature_loss = torch.stack(losses).mean()
                        loss = loss + args.feature_distill_alpha * feature_loss
            if torch.isnan(loss):
                raise RuntimeError('loss became NaN')
            scaler.scale(loss).backward()
            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            old_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            optimizer_stepped = (not scaler.is_enabled()) or scaler.get_scale() >= old_scale
            if ema is not None:
                ema.update(model)
            if optimizer_stepped:
                scheduler.step()

            batch = labels.shape[0]
            total += batch
            total_loss += loss.item() * batch
            top1, _ = accuracy(logits.detach(), labels, topk=(1, 5))
            top1_sum += top1 * batch
            global_step += 1
            if args.wandb:
                log_metrics({
                    'train/loss_step': loss.item(),
                    'train/supervised_loss_step': supervised_loss.item(),
                    'train/distill_loss_step': distill_loss.item(),
                    'train/feature_loss_step': feature_loss.item(),
                    'train/top1_step': top1,
                    'lr': scheduler.get_last_lr()[0],
                    'epoch': epoch,
                }, step=global_step)
            if args.smoke_steps and global_step >= args.smoke_steps:
                return {
                    'train_loss': total_loss / max(1, total),
                    'train_top1': top1_sum / max(1, total),
                }, global_step, True
        return {
            'train_loss': total_loss / max(1, total),
            'train_top1': top1_sum / max(1, total),
        }, global_step, False
    finally:
        for hook in hooks:
            hook.remove()


@torch.no_grad()
def evaluate(model: nn.Module, loader, device: torch.device) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total = 0
    top1_sum = 0.0
    top5_sum = 0.0
    criterion = nn.CrossEntropyLoss()
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, labels)
        batch = labels.shape[0]
        top1, top5 = accuracy(logits, labels, topk=(1, 5))
        total += batch
        total_loss += loss.item() * batch
        top1_sum += top1 * batch
        top5_sum += top5 * batch
    return {
        'val_loss': total_loss / max(1, total),
        'top1': top1_sum / max(1, total),
        'top5': top5_sum / max(1, total),
    }


def evaluate_with_ema(
    model: nn.Module,
    loader,
    device: torch.device,
    ema: ModelEma | None,
) -> dict[str, float]:
    if ema is None:
        return evaluate(model, loader, device)
    ema.store(model)
    ema.copy_to(model)
    try:
        metrics = evaluate(model, loader, device)
    finally:
        ema.restore(model)
    return {f'ema_{k}' if k in {'top1', 'top5', 'val_loss'} else k: v
            for k, v in metrics.items()}


def model_stats(model: nn.Module, config: LLViTConfig) -> dict[str, float | int | str]:
    from griffith_logic.layers.dwn import LearnableMapping

    train_params = sum(p.numel() for p in model.parameters())
    aux_mapping_params = sum(
        p.numel() for m in model.modules() if isinstance(m, LearnableMapping)
        for p in m.parameters()
    )
    params = train_params - aux_mapping_params
    # Rough ViT MAC estimate; LL-ViT paper reports hardware-level GMACs separately.
    t = config.tokens
    d = config.dim
    patch = config.patch_tokens * d * (config.patch_size ** 2) * config.in_channels
    attn = config.depth * (4 * t * d * d + 2 * t * t * d)
    mlp = 0 if model.channel_mixer_kind == 'llvit' else config.depth * 2 * t * d * config.mlp_hidden
    return {
        'param_count': params,
        'train_param_count': train_params,
        'aux_mapping_param_count': aux_mapping_params,
        'model_size_mb': round(params * 4 / (1024 ** 2), 3),
        'gmacs_estimate': round((patch + attn + mlp) / 1e9, 4),
        **llvit_component_summary(model),
    }


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = parse_args(argv)
    seed_everything(args.seed)
    device = resolve_device(args.device)
    config = build_config(args)
    model = build_model(args, config).to(device)
    train_loader, val_loader = build_loaders(args)
    init_stats: dict[str, int | str | bool] = {}
    if args.init_from_timm:
        init_stats.update(init_from_timm_deit_tiny(
            model,
            model_name=args.timm_model,
            pretrained=args.timm_pretrained,
        ))
        model.to(device)
    if args.model == 'llvit' and args.init_from_baseline:
        init_stats.update(init_from_baseline(model, args.init_from_baseline, device))
    if args.model == 'llvit' and args.init_from_llvit:
        init_stats.update(init_from_llvit_checkpoint(
            model,
            args.init_from_llvit,
            device,
            use_ema=args.init_from_llvit_ema,
        ))
    teacher = build_teacher(args, config, device)
    calibration_stats: dict[str, float | int] = {}
    if args.model == 'llvit' and args.calibrate_thermometer_batches > 0:
        if args.calibrate_thermometer_source == 'teacher' and teacher is not None:
            calibration_stats = calibrate_thermometers_from_teacher(
                model,
                teacher,
                train_loader,
                device,
                max_batches=args.calibrate_thermometer_batches,
                max_samples_per_mixer=args.calibrate_thermometer_samples,
            )
            calibration_stats['thermometer/calibration_source_teacher'] = 1
        else:
            calibration_stats = calibrate_thermometers(
                model,
                train_loader,
                device,
                max_batches=args.calibrate_thermometer_batches,
                max_samples_per_mixer=args.calibrate_thermometer_samples,
            )
            calibration_stats['thermometer/calibration_source_teacher'] = 0
    groups = (
        split_llvit_param_groups(
            model,
            ann_lr=args.ann_lr,
            logic_lr=args.logic_lr,
            weight_decay=args.weight_decay,
        )
        if args.model == 'llvit'
        else [{'params': model.parameters(), 'lr': args.lr, 'weight_decay': args.weight_decay}]
    )
    optimizer = torch.optim.AdamW(groups)
    scheduler = build_scheduler(args, optimizer, len(train_loader))
    amp_enabled = bool(args.amp and device.type == 'cuda')
    try:
        scaler = torch.amp.GradScaler('cuda', enabled=amp_enabled)
    except TypeError:  # older torch
        scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    stats = model_stats(model, config)

    run_name = args.run_name or f'llvit-cifar10-{args.model}'
    run_dir, latest_path, best_path = setup_llvit_checkpoint_paths(args, run_name)
    resume_path = resolve_resume_checkpoint(args, latest_path)
    resume_stats: dict[str, float | int | str | bool] = {}
    resume_data: dict[str, Any] | None = None
    best_top1 = -1.0
    global_step = 0
    start_epoch = 0
    if resume_path:
        resume_data = load_checkpoint(resume_path, model, optimizer, scheduler, device=str(device))
        global_step = int(resume_data.get('step', 0))
        best_top1 = float(resume_data.get('best_metric', -1.0))
        start_epoch = min(args.epochs, global_step // max(1, len(train_loader)))
        resume_stats = {
            'resume_checkpoint': resume_path,
            'resume_global_step': global_step,
            'resume_best_top1': best_top1,
            'resume_start_epoch': start_epoch,
        }
    ema = ModelEma(model, args.ema_decay) if args.ema_decay > 0 else None
    if resume_data is not None:
        resume_stats['resume_ema_restored'] = _load_ema_from_checkpoint(ema, resume_data, device)
    if args.wandb:
        cfg = vars(args) | {
            'llvit_config': asdict(config),
            'paper_config': config.paper_metadata(),
            'modal_gpu': os.environ.get('MODAL_GPU', ''),
            'run_dir': run_dir,
            'latest_checkpoint': latest_path,
            'best_checkpoint': best_path,
            **stats,
            **init_stats,
            **calibration_stats,
            **resume_stats,
            **git_metadata(),
        }
        init_wandb(
            run_name,
            cfg,
            model=model,
            device=str(device),
            tags=['llvit', args.model],
            group=args.wandb_group,
        )
        if calibration_stats:
            log_metrics(calibration_stats, step=0)

    pretrain_stats: dict[str, float | int] = {}
    if (
        args.model == 'llvit'
        and args.channel_mixer_pretrain_epochs > 0
        and not resume_path
    ):
        pretrain_stats = pretrain_channel_mixers(model, teacher, train_loader, device, args)
        if args.wandb and pretrain_stats:
            log_metrics(pretrain_stats, step=0)
        save_checkpoint(
            latest_path, model, optimizer, scheduler, global_step, best_top1,
            {
                'args': vars(args),
                'config': asdict(config),
                'stats': stats,
                'init_stats': init_stats,
                'calibration_stats': calibration_stats,
                'pretrain_stats': pretrain_stats,
            },
            extra_state=_checkpoint_extra_state(ema),
        )
    elif args.model == 'llvit' and args.channel_mixer_pretrain_epochs > 0 and resume_path:
        pretrain_stats = {'pretrain/channel_mixer_skipped_resume': 1}

    stop = False
    last_metrics: dict[str, float] = {}

    for epoch in range(start_epoch, args.epochs):
        train_metrics, global_step, stop = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            scaler,
            device,
            args,
            epoch,
            global_step,
            teacher=teacher,
            ema=ema,
        )
        should_eval = ((epoch + 1) % args.eval_interval == 0) or stop or epoch == args.epochs - 1
        if should_eval:
            set_conditional_sum_quantized(model, False)
            val_metrics = evaluate(model, val_loader, device)
            ema_metrics = evaluate_with_ema(model, val_loader, device, ema) if ema else {}
            metrics = train_metrics | val_metrics | {'epoch': float(epoch + 1)}
            metrics.update(ema_metrics)
            if args.model == 'llvit' and args.conditional_sum_quant_eval:
                set_conditional_sum_quantized(model, True)
                q_metrics = evaluate(model, val_loader, device)
                set_conditional_sum_quantized(model, False)
                metrics['conditional_sum_int4_acc'] = q_metrics['top1']
                metrics['conditional_sum_int4_loss'] = q_metrics['val_loss']
            score = max(
                float(val_metrics['top1']),
                float(metrics.get('ema_top1', float('-inf'))),
            )
            metrics['best_top1'] = max(best_top1, score)
            last_metrics = metrics
            if args.wandb:
                log_metrics(metrics, step=global_step)
            print(json.dumps({'epoch': epoch + 1, **metrics}, ensure_ascii=False))
            if score > best_top1:
                best_top1 = score
                save_checkpoint(
                    best_path, model, optimizer, scheduler, global_step, best_top1,
                    {
                        'args': vars(args),
                        'config': asdict(config),
                        'stats': stats,
                        'init_stats': init_stats,
                        'calibration_stats': calibration_stats,
                        'pretrain_stats': pretrain_stats,
                    },
                    extra_state=_checkpoint_extra_state(ema),
                )
            if args.stop_at_acc and score >= args.stop_at_acc:
                stop = True
        save_checkpoint(
            latest_path, model, optimizer, scheduler, global_step, best_top1,
            {
                'args': vars(args),
                'config': asdict(config),
                'stats': stats,
                'init_stats': init_stats,
                'calibration_stats': calibration_stats,
                'pretrain_stats': pretrain_stats,
            },
            extra_state=_checkpoint_extra_state(ema),
        )
        if stop:
            break

    if args.wandb and os.path.exists(best_path):
        log_checkpoint_artifact(best_path, run_name, stage='final', metadata={'best_top1': best_top1}, model=model)
    result: dict[str, Any] = {
        'run_dir': run_dir,
        'best_top1': best_top1,
        'global_step': global_step,
        'last_metrics': last_metrics,
        'stats': stats,
        'init_stats': init_stats,
        'calibration_stats': calibration_stats,
        'pretrain_stats': pretrain_stats,
        'paper_config': config.paper_metadata(),
    }
    print('RESULT ' + json.dumps(result, ensure_ascii=False))
    if args.wandb:
        finish_wandb()
    if args.expected_acc and best_top1 < args.expected_acc:
        raise SystemExit(f'expected acc {args.expected_acc} but got {best_top1}')
    return result


if __name__ == '__main__':
    main()
