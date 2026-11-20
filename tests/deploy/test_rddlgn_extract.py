"""RDDLGN → IR抽出の疎通。

export_rddlgn_classifier_ir がN-layers(state 0個・output_mode='stacked')と
K-layers(state 1個・output_mode='final_state')の2つのrecurrentノードを
正しく作ること、そのIRがvalidate()を通ること、そしてIR→torch(build_torch)
往復がsimulateと整合すること(build_torchは全く別クラスなので「再抽出して
元IRと一致」ではなく「rebuilt torchのforward結果がsimulateと一致」を
往復性の基準にする — _RecurrentModule/_BodyGraphModuleに対応する抽出器は
存在しない、これらはIR→torch方向専用の往復検証用モジュールのため)。
init_modeの検証(zero/one/gaussian)もここで固定する。
"""

import numpy as np
import pytest
import torch

from griffith_logic.export.rddlgn_ir import (
    export_rddlgn_classifier_ir,
    export_rddlgn_encoder_ir,
)
from griffith_logic.ir.build import build_torch
from griffith_logic.ir.serialize import load_ir, save_ir
from griffith_logic.ir.simulate import simulate
from griffith_logic.ir.validate import validate
from griffith_logic.models.rddlgn import RDDLGNClassifier, RDDLGNConfig

from .conftest import build_rddlgn_tiny_classifier


class TestEncoderStructure:
    """N-layers/K-layersがそれぞれ正しいrecurrent形状で抽出される。"""

    def test_two_recurrent_nodes(self):
        model, cfg = build_rddlgn_tiny_classifier(seed=0)
        graph, d_emb, k_out = export_rddlgn_encoder_ir(model.encoder, cfg.seq_len)
        recs = [n for n in graph.nodes if n.type == 'recurrent']
        assert len(recs) == 2
        n_rec, k_rec = recs
        assert n_rec.params['scan_vars'] == [
            {'id': 'sv_step', 'role': 'step', 'width': d_emb}
        ]
        assert n_rec.params['state_update'] == {}
        assert n_rec.params['output_mode'] == 'stacked'
        assert n_rec.params['steps'] == cfg.seq_len

        assert k_rec.params['output_mode'] == 'final_state'
        assert k_rec.params['steps'] == cfg.seq_len
        roles = {sv['id']: sv['role'] for sv in k_rec.params['scan_vars']}
        assert roles == {'sv_state': 'state', 'sv_step': 'step'}
        assert list(k_rec.params['state_update'].keys()) == ['sv_state']
        assert d_emb == model.embedding.embedding.embedding_dim
        assert k_out == cfg.k_layer_sizes[-1]

    def test_body_contains_only_allowed_node_types(self):
        """body内はscan_var/lut_layer/concatのみ(schema.py規約)。"""
        model, cfg = build_rddlgn_tiny_classifier(seed=0)
        graph, _, _ = export_rddlgn_encoder_ir(model.encoder, cfg.seq_len)
        for node in graph.nodes:
            if node.type != 'recurrent':
                continue
            body_types = {n.type for n in node.params['body'].nodes}
            assert body_types <= {'scan_var', 'lut_layer', 'concat'}

    def test_validate_passes(self):
        model, cfg = build_rddlgn_tiny_classifier(seed=1)
        graph = export_rddlgn_classifier_ir(model, cfg.seq_len)
        assert validate(graph) == []

    def test_input_output_widths(self):
        model, cfg = build_rddlgn_tiny_classifier(seed=2, n_classes=4)
        graph = export_rddlgn_classifier_ir(model, cfg.seq_len)
        d_emb = model.embedding.embedding.embedding_dim
        assert graph.inputs[0].width == cfg.seq_len * d_emb
        assert graph.outputs[0].width == 4


class TestInitModeHandling:
    """のhard eval boolean化規則をIR側でも確定する。"""

    def test_zero_mode_gives_all_zero_state(self):
        model, cfg = build_rddlgn_tiny_classifier(seed=0)
        assert model.encoder.init_mode == 'zero'
        graph = export_rddlgn_classifier_ir(model, cfg.seq_len)
        k_rec = model_node(graph, 'enc_k_rec')
        state_node = k_rec.params['body'].node_by_id('sv_state')
        assert state_node.params['init'] == {
            'kind': 'const', 'bits': [0] * len(state_node.params['init']['bits'])
        }

    def test_one_mode_gives_all_one_state(self):
        torch.manual_seed(0)
        cfg = RDDLGNConfig.tiny()
        cfg.init_mode = 'one'
        model = RDDLGNClassifier(cfg, n_classes=3)
        model.eval()
        graph = export_rddlgn_classifier_ir(model, cfg.seq_len)
        k_rec = model_node(graph, 'enc_k_rec')
        state_node = k_rec.params['body'].node_by_id('sv_state')
        bits = state_node.params['init']['bits']
        assert bits == [1] * len(bits)

    def test_gaussian_mode_raises_not_implemented(self):
        torch.manual_seed(0)
        cfg = RDDLGNConfig.tiny()
        cfg.init_mode = 'gaussian'
        model = RDDLGNClassifier(cfg, n_classes=3)
        model.eval()
        with pytest.raises(NotImplementedError, match='gaussian'):
            export_rddlgn_classifier_ir(model, cfg.seq_len)


class TestSaveLoadRoundTrip:
    """serialize.py::save_ir/load_ir がrecurrent.params['body']を再帰的に
    npz退避・復元できる(schema.pyのin-memory契約: params['body']は
    常にIRGraphインスタンス)。"""

    def test_round_trip_preserves_simulate_output(self, tmp_path):
        model, cfg = build_rddlgn_tiny_classifier(seed=3)
        graph = export_rddlgn_classifier_ir(model, cfg.seq_len)

        json_path = save_ir(graph, str(tmp_path / 'rddlgn.json'),
                            inline_threshold=8)   # 小さい閾値でnpz退避を強制
        assert (tmp_path / 'rddlgn.npz').exists(), (
            'inline_threshold=8ならindices等がnpzへ退避されるはず')
        reloaded = load_ir(json_path)
        assert validate(reloaded) == []

        rng = np.random.default_rng(0)
        x = rng.integers(0, 2, size=(3, graph.inputs[0].width)).astype(np.int64)
        want = simulate(graph, x)
        got = simulate(reloaded, x)
        assert np.array_equal(got, want)

    def test_body_is_irgraph_instance_after_reload(self, tmp_path):
        """in-memory契約: params['body']はIRGraph(dictのままではない)。"""
        from griffith_logic.ir.schema import IRGraph

        model, cfg = build_rddlgn_tiny_classifier(seed=4)
        graph = export_rddlgn_classifier_ir(model, cfg.seq_len)
        json_path = save_ir(graph, str(tmp_path / 'm.json'))
        reloaded = load_ir(json_path)
        for node in reloaded.nodes:
            if node.type == 'recurrent':
                assert isinstance(node.params['body'], IRGraph)


class TestBuildTorchRoundTrip:
    """build_torch(IR) のforward結果がsimulateと整合する(往復検証。
    IR→torch方向専用のモジュール(_RecurrentModule/_BodyGraphModule)に
    対応する抽出器は無いため、「再抽出して一致」ではなく
    「rebuilt torchのforward ⇔ simulate」を往復性の基準にする)。
    """

    def test_matches_simulate_up_to_tau_scaling(self):
        model, cfg = build_rddlgn_tiny_classifier(seed=5)
        graph = export_rddlgn_classifier_ir(model, cfg.seq_len)
        rebuilt = build_torch(graph)
        rebuilt.eval()

        rng = np.random.default_rng(1)
        x = rng.integers(0, 2, size=(4, graph.inputs[0].width)).astype(np.float32)
        with torch.no_grad():
            out = rebuilt(torch.tensor(x))
        want = simulate(graph, x.astype(np.int64))
        assert np.array_equal(out.argmax(-1).numpy(), want.argmax(-1))
        # group_sumのtauはsimulateでは無視される(argmax不変)ので、
        # 比率が全要素で一定であることまで確認する(数値そのものの整合)。
        ratio = out.numpy() / want
        assert np.allclose(ratio, ratio.flat[0])


def model_node(graph, node_id):
    return graph.node_by_id(node_id)
