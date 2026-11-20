"""`griffith deploy bundle model.pt`(学習済みチェックポイント直接入力)のテスト。

等価性の鎖: PyTorch hard eval ⇔ バンドルのIRを simulate ⇔ (cc があれば) C。
加えて、従来経路(export_ir → webbench doc → bundle)と同じIRになることを確認する。
"""

import itertools
import json
import shutil

import numpy as np
import pytest
import torch

from griffith_logic import cli
from griffith_logic.deploy.from_checkpoint import doc_from_checkpoint
from griffith_logic.export.extract import export_ir
from griffith_logic.ir.serialize import ir_from_dict, ir_to_dict
from griffith_logic.ir.simulate import simulate
from griffith_logic.training.config import dump_config, load_config, save_config
from griffith_logic.training.runner import (
    _build_model,
    _resolve_data_shape,
    load_model_from_checkpoint,
)

XOR_TRAIN_ARGS = [
    'data.name=logic:xor',
    'data.batch_size=32',
    'model.kind=logic_mlp',
    'model.k=8',
    'model.layers=2',
    'optim.lr=0.05',
    'run.epochs=3',
    'run.seed=0',
    'run.device=cpu',
]


@pytest.fixture(scope='module')
def xor_ckpt(tmp_path_factory):
    out = tmp_path_factory.mktemp('xor_run')
    assert cli.main(['train', *XOR_TRAIN_ARGS, '--out-dir', str(out)]) == 0
    return out / 'checkpoints' / 'best.pt'


def _save_mnist_ckpt(path, *, with_config=True, seed=0):
    """データセットを読まずに作れる分類チェックポイント(mnist形状、未学習)"""
    torch.manual_seed(seed)
    cfg = load_config(None, ['data.name=mnist', 'model.k=20', 'model.layers=2',
                             'model.tau=1.0'])
    model = _build_model(cfg.model, _resolve_data_shape(cfg.data), 'cpu')
    metadata = {'config': dump_config(cfg), 'run_name': 'mnist_test'} if with_config else {}
    torch.save({'model_state_dict': model.state_dict(), 'step': 7,
                'metadata': metadata}, str(path))
    return cfg


def _bundle(src, out, *extra):
    assert cli.main(['deploy', 'bundle', str(src), '-o', str(out), *extra]) == 0
    return json.loads(out.read_text(encoding='utf-8'))


def _hard_outputs(model, x_bits):
    with torch.no_grad():
        return model(torch.from_numpy(x_bits).to(torch.float32)).numpy()


class TestXorCheckpoint:
    def test_bundle_matches_pytorch_hard_eval(self, xor_ckpt, tmp_path):
        bundle = _bundle(xor_ckpt, tmp_path / 'xor.bundle.json')
        model, _, shape, _ = load_model_from_checkpoint(str(xor_ckpt))
        x = np.array(list(itertools.product([0, 1], repeat=shape.in_width)),
                     dtype=np.uint8)
        sim = simulate(ir_from_dict(bundle['ir']), x)
        np.testing.assert_array_equal(sim, _hard_outputs(model, x).round())
        assert bundle['readout']['kind'] == 'bits'
        assert bundle['about']['source_ckpt'] == str(xor_ckpt)

    def test_same_ir_as_conventional_path(self, xor_ckpt, tmp_path):
        """従来経路(export_ir → doc → bundle)とIRが一致する"""
        model, _, shape, _ = load_model_from_checkpoint(str(xor_ckpt))
        doc = {'ir': ir_to_dict(export_ir(model, input_width=shape.in_width))}
        src = tmp_path / 'model.json'
        src.write_text(json.dumps(doc), encoding='utf-8')
        via_json = _bundle(src, tmp_path / 'a.bundle.json')
        via_pt = _bundle(xor_ckpt, tmp_path / 'b.bundle.json')
        assert via_pt['ir']['nodes'] == via_json['ir']['nodes']
        assert via_pt['ir']['edges'] == via_json['ir']['edges']

    def test_default_output_name(self, xor_ckpt, tmp_path):
        src = tmp_path / 'model.pt'
        shutil.copy(xor_ckpt, src)
        assert cli.main(['deploy', 'bundle', str(src)]) == 0
        assert (tmp_path / 'model.bundle.json').exists()

    @pytest.mark.skipif(shutil.which('cc') is None, reason='cc がない')
    def test_verify_c_passes(self, xor_ckpt, tmp_path):
        out = tmp_path / 'xor.bundle.json'
        _bundle(xor_ckpt, out)
        assert cli.main(['deploy', 'verify-c', str(out)]) == 0


class TestClassificationCheckpoint:
    def test_readout_labels_and_scores(self, tmp_path):
        ckpt = tmp_path / 'mnist.pt'
        cfg = _save_mnist_ckpt(ckpt)
        bundle = _bundle(ckpt, tmp_path / 'mnist.bundle.json')
        assert bundle['readout'] == {'kind': 'class_scores', 'k': 10, 'spatial': None}
        assert bundle['labels'] == [str(i) for i in range(10)]
        assert bundle['about']['run_name'] == 'mnist_test'
        assert bundle['about']['step'] == 7

        model, _, shape, _ = load_model_from_checkpoint(str(ckpt))
        rng = np.random.default_rng(0)
        x = rng.integers(0, 2, size=(16, shape.in_width), dtype=np.uint8)
        sim = simulate(ir_from_dict(bundle['ir']), x)
        # GroupSum出力 = 整数カウント / tau
        np.testing.assert_array_equal(
            sim, (_hard_outputs(model, x) * cfg.model.tau).round())

    def test_config_option_for_checkpoint_without_metadata(self, tmp_path):
        ckpt = tmp_path / 'raw.pt'
        cfg = _save_mnist_ckpt(ckpt, with_config=False)
        with pytest.raises(ValueError, match='--config'):
            doc_from_checkpoint(ckpt)
        yaml = tmp_path / 'train.yaml'
        save_config(cfg, str(yaml))
        bundle = _bundle(ckpt, tmp_path / 'raw.bundle.json', '--config', str(yaml))
        assert bundle['readout']['k'] == 10

    def test_mismatched_config_is_rejected(self, tmp_path):
        ckpt = tmp_path / 'mnist.pt'
        _save_mnist_ckpt(ckpt)
        cfg = load_config(None, ['data.name=mnist', 'model.k=30', 'model.layers=2'])
        yaml = tmp_path / 'wrong.yaml'
        save_config(cfg, str(yaml))
        with pytest.raises(ValueError, match='不整合'):
            doc_from_checkpoint(ckpt, config=str(yaml))
