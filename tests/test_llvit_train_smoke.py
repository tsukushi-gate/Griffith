import torch

from experiments.checkpoint_utils import save_checkpoint
from experiments.train_llvit import main
from griffith_logic.models.llvit import LLViTConfig, build_deit_tiny_cifar10


def test_train_llvit_fake_data_smoke(tmp_path):
    result = main([
        '--model', 'llvit',
        '--fake-data',
        '--fake-train-size', '4',
        '--fake-val-size', '4',
        '--epochs', '1',
        '--batch-size', '2',
        '--smoke-steps', '1',
        '--image-size', '32',
        '--patch-size', '16',
        '--dim', '12',
        '--depth', '1',
        '--heads', '3',
        '--mlp-ratio', '2',
        '--thermometer-bits', '2',
        '--thermometer-mode', 'distributive',
        '--lut-hidden', '24',
        '--lut-out', '12',
        '--lut-inputs', '2',
        '--output-dir', str(tmp_path),
        '--no-wandb',
    ])
    assert result['global_step'] == 1
    assert result['paper_config']['dim'] == 12
    assert result['stats']['dwn_layers'] == 2


def test_train_llvit_resumes_from_stable_run_dir(tmp_path):
    run_dir = tmp_path / 'resume-run'
    common = [
        '--model', 'llvit',
        '--fake-data',
        '--fake-train-size', '4',
        '--fake-val-size', '4',
        '--batch-size', '2',
        '--image-size', '32',
        '--patch-size', '16',
        '--dim', '12',
        '--depth', '1',
        '--heads', '3',
        '--mlp-ratio', '2',
        '--thermometer-bits', '2',
        '--thermometer-mode', 'distributive',
        '--lut-hidden', '24',
        '--lut-out', '12',
        '--lut-inputs', '2',
        '--run-dir', str(run_dir),
        '--ema-decay', '0.9',
        '--no-wandb',
    ]
    first = main([*common, '--epochs', '1'])
    assert first['global_step'] == 2
    assert (run_dir / 'latest.pt').exists()
    checkpoint = torch.load(run_dir / 'latest.pt', map_location='cpu', weights_only=False)
    assert 'ema_state_dict' in checkpoint['extra_state']

    resumed = main([*common, '--epochs', '2', '--resume-checkpoint', str(run_dir / 'latest.pt')])
    assert resumed['global_step'] == 4
    assert resumed['last_metrics']['epoch'] == 2.0
    assert resumed['pretrain_stats'] == {}


def test_train_llvit_init_from_llvit_uses_ema_weights(tmp_path):
    source_dir = tmp_path / 'source-run'
    common = [
        '--model', 'llvit',
        '--fake-data',
        '--fake-train-size', '4',
        '--fake-val-size', '4',
        '--batch-size', '2',
        '--image-size', '32',
        '--patch-size', '16',
        '--dim', '12',
        '--depth', '1',
        '--heads', '3',
        '--mlp-ratio', '2',
        '--thermometer-bits', '2',
        '--thermometer-mode', 'distributive',
        '--lut-hidden', '24',
        '--lut-out', '12',
        '--lut-inputs', '2',
        '--ema-decay', '0.9',
        '--no-wandb',
    ]
    main([*common, '--epochs', '1', '--run-dir', str(source_dir)])

    initialized = main([
        *common,
        '--epochs', '1',
        '--smoke-steps', '1',
        '--run-dir', str(tmp_path / 'initialized-run'),
        '--init-from-llvit', str(source_dir / 'best.pt'),
        '--calibrate-thermometer-batches', '0',
    ])

    assert initialized['global_step'] == 1
    assert initialized['init_stats']['init_from_llvit/source'] == 'model+ema'
    assert initialized['init_stats']['init_from_llvit/copied'] > 0
    assert initialized['init_stats']['init_from_llvit/ema_copied'] > 0


def test_train_llvit_channel_mixer_pretrain_runs(tmp_path):
    config = LLViTConfig(
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
    )
    teacher = build_deit_tiny_cifar10(config)
    optimizer = torch.optim.AdamW(teacher.parameters(), lr=1e-3)
    teacher_path = tmp_path / 'teacher.pt'
    save_checkpoint(str(teacher_path), teacher, optimizer, None, 0, 0.0, {})

    result = main([
        '--model', 'llvit',
        '--fake-data',
        '--fake-train-size', '4',
        '--fake-val-size', '4',
        '--epochs', '1',
        '--batch-size', '2',
        '--smoke-steps', '1',
        '--image-size', '32',
        '--patch-size', '16',
        '--dim', '12',
        '--depth', '1',
        '--heads', '3',
        '--mlp-ratio', '2',
        '--thermometer-bits', '2',
        '--lut-hidden', '24',
        '--lut-out', '12',
        '--lut-inputs', '2',
        '--distill-checkpoint', str(teacher_path),
        '--channel-mixer-pretrain-epochs', '1',
        '--channel-mixer-pretrain-batches', '1',
        '--calibrate-thermometer-batches', '1',
        '--output-dir', str(tmp_path),
        '--no-wandb',
    ])
    assert result['pretrain_stats']['pretrain/channel_mixer_steps'] == 1
    assert result['pretrain_stats']['pretrain/channel_mixer_loss'] >= 0
