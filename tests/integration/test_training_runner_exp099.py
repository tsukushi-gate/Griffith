"""統合: 先行実験で追加した学習ランナーの配管。

- regularization の λ 線形ランプと reg_penalty / reg_lambda のログ
- λ=0 のとき学習が正則化なしと bitwise 不変
- 層ごとのマスク・初期化・規則配線をレシピから組めること
- run.export_netlist / run.save_eval_history の出力（形式と ir.simulate 一致）
（モデル級・統合テストは CONTRIBUTING の規約どおり tests/integration/ に置く）
"""

import json
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from griffith_logic.ir.simulate import simulate
from griffith_logic.layers.lut import LutLayer
from griffith_logic.training.config import load_config
from griffith_logic.training.runner import (
    ResolvedData,
    _build_model,
    _netlist_dict,
    run_training,
)

BASE_OVERRIDES = [
    'data.name=logic:xor',
    'data.batch_size=4',
    'model.kind=logic_mlp',
    'model.k=8',
    'model.layers=2',
    'optim.lr=0.05',
    'run.epochs=4',
    'run.seed=0',
    'run.device=cpu',
    'run.wandb=false',
]


def _cfg(*extra):
    return load_config(None, [*BASE_OVERRIDES, *extra])


def _weights(ckpt_path):
    data = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    return {k: v.clone() for k, v in data['model_state_dict'].items()}


class TestRegularizationPlumbing:
    def test_lambda_zero_is_bitwise_identical_to_no_regularization(self, tmp_path):
        base = run_training(_cfg(), out_dir=str(tmp_path / 'base'))
        regged = run_training(
            _cfg('regularization.kind=irreducibility',
                 'regularization.lambda=0.0'),
            out_dir=str(tmp_path / 'reg0'),
        )
        assert base['final_loss'] == regged['final_loss']
        assert base['hard_acc'] == regged['hard_acc']
        w0 = _weights(base['checkpoint_latest'])
        w1 = _weights(regged['checkpoint_latest'])
        assert set(w0) == set(w1)
        for k in w0:
            assert torch.equal(w0[k], w1[k]), k
        # λ=0 でも指標としての reg_penalty は記録される
        assert regged['reg_lambda'] == 0.0
        assert 0.0 <= regged['reg_penalty'] <= 1.0
        assert 'reg_penalty' not in base

    def test_lambda_changes_training(self, tmp_path):
        base = run_training(_cfg(), out_dir=str(tmp_path / 'base'))
        regged = run_training(
            _cfg('regularization.kind=irreducibility',
                 'regularization.lambda=1.0',
                 'regularization.ramp_frac=0.0'),
            out_dir=str(tmp_path / 'reg1'),
        )
        w0 = _weights(base['checkpoint_latest'])
        w1 = _weights(regged['checkpoint_latest'])
        assert any(not torch.equal(w0[k], w1[k]) for k in w0)
        assert regged['reg_lambda'] == 1.0

    def test_ramp_reaches_lambda_by_half(self, tmp_path):
        """ramp_frac=0.5 なら学習後半は λ に張り付く"""
        result = run_training(
            _cfg('regularization.kind=irreducibility',
                 'regularization.lambda=0.5',
                 'regularization.ramp_frac=0.5'),
            out_dir=str(tmp_path / 'ramp'),
        )
        assert abs(result['reg_lambda'] - 0.5) < 1e-12

    def test_ramp_frac_one_never_reaches_lambda(self, tmp_path):
        """ramp_frac=1.0 なら最終ステップでもまだ λ 未満（線形ランプの確認）"""
        result = run_training(
            _cfg('regularization.kind=irreducibility',
                 'regularization.lambda=0.5',
                 'regularization.ramp_frac=1.0'),
            out_dir=str(tmp_path / 'ramp1'),
        )
        assert 0.0 < result['reg_lambda'] < 0.5

    def test_ramp_frac_zero_is_constant(self, tmp_path):
        result = run_training(
            _cfg('regularization.kind=irreducibility',
                 'regularization.lambda=0.25',
                 'regularization.ramp_frac=0.0'),
            out_dir=str(tmp_path / 'const'),
        )
        assert result['reg_lambda'] == 0.25

    def test_config_roundtrip_keeps_regularization(self, tmp_path):
        result = run_training(
            _cfg('regularization.kind=irreducibility',
                 'regularization.lambda=0.125',
                 'regularization.ramp_frac=0.25'),
            out_dir=str(tmp_path / 'rt'),
        )
        reloaded = load_config(result['config_path'], [])
        assert reloaded.regularization is not None
        assert reloaded.regularization.lambda_ == 0.125
        assert reloaded.regularization.ramp_frac == 0.25


class TestRecipeModelBuilding:
    """レシピ(ModelConfig)から層ごとのマスク・初期化・規則配線が組めること"""

    @staticmethod
    def _data(in_width=16, out_width=2, kind='classification'):
        return ResolvedData(
            name='synthetic', kind=kind, in_width=in_width,
            out_width=out_width, train_loader=None, eval_loader=None,
        )

    @staticmethod
    def _luts(model):
        return [m for m in model.modules() if isinstance(m, LutLayer)]

    def _build(self, *overrides, in_width=16, seed=0):
        cfg = load_config(None, [
            'model.kind=dlgn', 'model.k=8', 'model.layers=3',
            *overrides,
        ])
        return _build_model(cfg.model, self._data(in_width), 'cpu', seed=seed)

    def test_defaults_match_legacy_construction(self):
        """既定値だけのレシピは従来の LutLayer 引数と同じ結果になる"""
        torch.manual_seed(0)
        model = self._build()
        luts = self._luts(model)
        assert len(luts) == 3
        for layer in luts:
            assert layer.selection == 'soft'
            assert layer.gate_mask == ()
            assert layer.init_mode is None
            assert layer.grad_factor == 1.0

    def test_masks_differ_between_hidden_and_final(self):
        model = self._build(
            'model.gate_mask_hidden=[0,15,3,5,10,12]',
            'model.gate_mask_final=[3,5,10,12]',
        )
        luts = self._luts(model)
        assert luts[0].gate_mask == (0, 3, 5, 10, 12, 15)
        assert luts[1].gate_mask == (0, 3, 5, 10, 12, 15)
        assert luts[-1].gate_mask == (3, 5, 10, 12)

    def test_selection_init_and_grad_factor_forwarded(self):
        model = self._build(
            'model.selection=hard_st', 'model.init_mode=ri_randn',
            'model.grad_factor=2.0',
        )
        for layer in self._luts(model):
            assert layer.selection == 'hard_st'
            assert layer.init_mode == 'ri_randn'
            assert layer.init_scale == 5.0     # レシピ既定
            assert layer.grad_factor == 2.0
            # ri_randn(α=5): op3 がほぼ全ゲートで argmax
            frac = (layer.weights.argmax(-1) == 3).float().mean().item()
            assert frac >= 0.9

    def test_init_scale_reaches_layers(self):
        """レシピ model.init_scale=1.0（先行実験の実値）が層に届く"""
        model = self._build('model.init_mode=ri_randn', 'model.init_scale=1.0')
        base = self._build('model.init_mode=randn')
        for layer, ref in zip(self._luts(model), self._luts(base), strict=True):
            assert layer.init_scale == 1.0
            expected = ref.weights.detach().clone()
            expected[:, 3] += 1.0
            assert torch.equal(layer.weights, expected)
            # α=1 では op3 の初期選択率が α=5 よりずっと低い
            frac = (layer.weights.argmax(-1) == 3).float().mean().item()
            assert frac < 0.9

    def test_init_scale_ignored_without_init_mode(self):
        """init_mode 未指定（従来動作）では init_scale を LutLayer へ渡さない"""
        torch.manual_seed(0)   # init_mode=None は generator を使わない（従来動作）
        a = self._build()
        torch.manual_seed(0)
        b = self._build('model.init_scale=1.0')
        for la, lb in zip(self._luts(a), self._luts(b), strict=True):
            assert la.init_scale == lb.init_scale == 5.0
            assert torch.equal(la.weights, lb.weights)

    def test_rot_connections_per_layer(self):
        model = self._build(
            'model.connections.kind=rot',
            'model.connections.s_a=[3,0,0]',
            'model.connections.s_b=[5,4,20]',
        )
        luts = self._luts(model)
        for i, (layer, s_a, s_b) in enumerate(
                zip(luts, [3, 0, 0], [5, 4, 20], strict=True)):
            in_dim = 16 if i == 0 else 8
            g = torch.arange(layer.out_dim)
            assert torch.equal(layer.indices[0], (g + s_a) % in_dim)
            assert torch.equal(layer.indices[1], (g + s_b) % in_dim)

    def test_perm_connections_are_balanced(self):
        model = self._build('model.connections.kind=perm', in_width=16)
        luts = self._luts(model)
        for i, layer in enumerate(luts):
            in_dim = 16 if i == 0 else 8
            counts = torch.bincount(layer.indices.reshape(-1), minlength=in_dim)
            total = 2 * layer.out_dim
            assert int(counts.min()) >= total // in_dim
            assert int(counts.max()) <= -(-total // in_dim)

    def test_wiring_is_independent_of_init_mode_and_mask(self):
        """腕どうしが同一配線を共有できる(配線用generatorが別ストリーム)"""
        base = self._build('model.connections.kind=perm',
                           'model.init_mode=ri_randn')
        other = self._build('model.connections.kind=perm',
                            'model.init_mode=rri',
                            'model.gate_mask_hidden=[0,15,3,5,10,12]',
                            'model.gate_mask_final=[3,5,10,12]')
        for a, b in zip(self._luts(base), self._luts(other), strict=True):
            assert torch.equal(a.indices, b.indices)

    def test_wiring_depends_on_seed(self):
        a = self._build('model.connections.kind=perm', seed=0)
        b = self._build('model.connections.kind=perm', seed=1)
        assert not torch.equal(
            self._luts(a)[0].indices, self._luts(b)[0].indices)

    def test_init_is_reproducible_across_runs(self):
        a = self._build('model.init_mode=rri', seed=7)
        b = self._build('model.init_mode=rri', seed=7)
        for la, lb in zip(self._luts(a), self._luts(b), strict=True):
            assert torch.equal(la.weights, lb.weights)


class TestNetlistFormat:
    @staticmethod
    def _dlgn(in_width=8, k=6, classes=2, layers=2, seed=0, *extra):
        cfg = load_config(None, [
            'model.kind=dlgn', f'model.k={k}', f'model.layers={layers}',
            'model.selection=hard_st', 'model.tau=3.0',
            'data.name=cifar-10-3-thresholds',
            'data.params.interleave_thresholds=true',
            *extra,
        ])
        data = ResolvedData(
            name='synthetic', kind='classification', in_width=in_width,
            out_width=classes, train_loader=None, eval_loader=None,
        )
        torch.manual_seed(seed)
        model = _build_model(cfg.model, data, 'cpu', seed=seed)
        return cfg, data, model

    def test_flat_format(self):
        cfg, data, model = self._dlgn()
        net = _netlist_dict(model, cfg, data)
        assert set(net) == {'input_bits', 'binarization', 'layers', 'output'}
        assert net['input_bits'] == 8
        assert 'cifar-10-3-thresholds' in net['binarization']
        assert 'interleave_thresholds' in net['binarization']
        assert len(net['layers']) == 2
        for li, layer in enumerate(net['layers']):
            assert len(layer) == 6
            in_dim = 8 if li == 0 else 6
            for gate in layer:
                assert set(gate) == {'op', 'a', 'b'}
                assert 0 <= gate['op'] <= 15
                assert 0 <= gate['a'] < in_dim
                assert 0 <= gate['b'] < in_dim
        assert net['output'] == {'classes': 2, 'group_size': 3, 'tau': 3.0}
        json.dumps(net)  # JSONシリアライズ可能

    def test_matches_ir_simulate(self):
        """平坦ネットリストの自前評価が ir.simulate とビット厳密に一致する"""
        from griffith_logic.export import export_ir

        cfg, data, model = self._dlgn(in_width=8, k=6, classes=2,
                                      layers=3, seed=3)
        net = _netlist_dict(model, cfg, data)
        graph = export_ir(model, input_width=8)

        rng = np.random.default_rng(0)
        x = rng.integers(0, 2, size=(32, 8))
        golden = simulate(graph, x)      # (batch, classes) 整数スコア

        group_size = net['output']['group_size']
        bits = x.copy()
        for layer in net['layers']:
            nxt = np.zeros((x.shape[0], len(layer)), dtype=np.int64)
            for o, gate in enumerate(layer):
                a = bits[:, gate['a']]
                b = bits[:, gate['b']]
                nxt[:, o] = (gate['op'] >> (3 - (2 * a + b))) & 1
            bits = nxt
        mine = np.zeros((x.shape[0], net['output']['classes']), dtype=np.int64)
        for i in range(bits.shape[1]):
            mine[:, i // group_size] += bits[:, i]
        assert np.array_equal(mine, golden)

    def test_masked_ops_absent_from_netlist(self):
        cfg = load_config(None, [
            'model.kind=dlgn', 'model.k=8', 'model.layers=2',
            'model.selection=hard_st',
            'model.gate_mask_hidden=[0,15,3,5,10,12]',
            'model.gate_mask_final=[3,5,10,12]',
        ])
        data = ResolvedData(
            name='synthetic', kind='classification', in_width=8, out_width=2,
            train_loader=None, eval_loader=None,
        )
        torch.manual_seed(0)
        model = _build_model(cfg.model, data, 'cpu', seed=0)
        net = _netlist_dict(model, cfg, data)
        hidden_ops = {g['op'] for g in net['layers'][0]}
        final_ops = {g['op'] for g in net['layers'][-1]}
        assert hidden_ops.isdisjoint({0, 15, 3, 5, 10, 12})
        assert final_ops.isdisjoint({3, 5, 10, 12})


class TestRunArtifacts:
    def test_netlist_written_and_default_off(self, tmp_path):
        run_training(_cfg('run.export_netlist=true'),
                     out_dir=str(tmp_path / 'net'))
        path = tmp_path / 'net' / 'netlist.json'
        assert path.is_file()
        net = json.loads(path.read_text('utf-8'))
        assert net['input_bits'] == 2
        assert len(net['layers']) == 2
        assert net['output'] is None  # logic_mlp は GroupSum を持たない

        run_training(_cfg(), out_dir=str(tmp_path / 'nonet'))
        assert not (tmp_path / 'nonet' / 'netlist.json').exists()

    def test_eval_history_and_summary(self, tmp_path):
        result = run_training(
            _cfg('run.eval_interval=2', 'run.save_eval_history=true'),
            out_dir=str(tmp_path / 'hist'),
        )
        history = json.loads(
            (tmp_path / 'hist' / 'eval_history.json').read_text('utf-8'))
        assert len(history) >= 2
        for row in history:
            assert set(row) == {
                'step', 'epoch', 'hard_acc', 'soft_acc', 'loss',
                'reg_penalty', 'reg_lambda',
            }
            assert 0.0 <= row['hard_acc'] <= 1.0
            assert 0.0 <= row['soft_acc'] <= 1.0
        assert [r['step'] for r in history] == sorted(r['step'] for r in history)
        assert history[-1]['hard_acc'] == result['hard_acc']

        summary = json.loads(
            (tmp_path / 'hist' / 'summary.json').read_text('utf-8'))
        assert set(summary) == {
            'hard_acc_final', 'hard_acc_best', 'soft_acc_final', 'op_hist',
            'steps', 'wall_clock_s',
        }
        assert summary['hard_acc_final'] == result['hard_acc']
        assert summary['hard_acc_best'] == result['best_hard_acc']
        assert summary['soft_acc_final'] == history[-1]['soft_acc']
        assert summary['steps'] == result['steps']
        assert summary['wall_clock_s'] > 0
        # LutLayer は 2 層(隠れ幅 8 と logic:xor の出力 1)。和がゲート数
        assert [sum(h) for h in summary['op_hist']] == [8, 1]
        assert all(len(h) == 16 for h in summary['op_hist'])

    def test_history_and_summary_default_off(self, tmp_path):
        result = run_training(_cfg(), out_dir=str(tmp_path / 'off'))
        assert not (tmp_path / 'off' / 'eval_history.json').exists()
        assert not (tmp_path / 'off' / 'summary.json').exists()
        assert 'soft_acc' not in result

    def test_soft_eval_does_not_change_training(self, tmp_path):
        """save_eval_history=true でも学習結果は bitwise 不変(eval本体を壊さない)"""
        off = run_training(_cfg(), out_dir=str(tmp_path / 'off'))
        on = run_training(_cfg('run.save_eval_history=true'),
                          out_dir=str(tmp_path / 'on'))
        assert off['hard_acc'] == on['hard_acc']
        assert off['final_loss'] == on['final_loss']
        w0 = _weights(off['checkpoint_latest'])
        w1 = _weights(on['checkpoint_latest'])
        for k in w0:
            assert torch.equal(w0[k], w1[k]), k

    def test_soft_eval_restores_selection(self):
        """soft評価は selection を元に戻す(hard_st のまま学習が続く)"""
        from griffith_logic.training.runner import _resolve_data, _soft_eval
        from griffith_logic.training.tasks import make_task

        cfg = _cfg('model.selection=hard_st')
        data = _resolve_data(cfg.data, 'cpu')
        model = _build_model(cfg.model, data, 'cpu', seed=0)
        task = make_task(data.kind, None)
        model.train()
        acc = _soft_eval(model, data.eval_loader, task, 'cpu')['acc']
        assert 0.0 <= acc <= 1.0
        for layer in model.modules():
            if isinstance(layer, LutLayer):
                assert layer.selection == 'hard_st'
        assert model.training is True


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
