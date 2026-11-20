"""統合: RDDLGN デコーダの 1 ステップ IR。

- 1 ステップ IR を simulate で T ステップ帰還させた貪欲生成が `simulate_rddlgn_greedy_bits`
  (recurrent body 経由の golden)と一致する(argmax はテスト側で「最初の最大値」)
- include_l=True(外部入力 = 埋め込みビット)/ False(= L-layers 出力)の両方
- 契約(export/step_contract.py)を通る、validate(target='hdl') を通る
- 既存の encoder IR の形は不変(FPGA 側 recurrent_encoder_to_step の前提)
"""

import numpy as np
import pytest
import torch

from griffith_logic.export.rddlgn_ir import (
    _embed_hard_bits,
    export_rddlgn_decoder_step_ir,
    export_rddlgn_encoder_ir,
    simulate_rddlgn_greedy_bits,
)
from griffith_logic.export.step_contract import check_step_ir, step_ir_manifest
from griffith_logic.ir.simulate import simulate
from griffith_logic.ir.validate import validate
from griffith_logic.models.rddlgn import RDDLGN, RDDLGNConfig

SRC_LEN, MAX_NEW, BOS, EOS = 4, 6, 1, 2


def _model(seed=0):
    torch.manual_seed(seed)
    cfg = RDDLGNConfig.tiny()
    model = RDDLGN(cfg).eval()
    return model, cfg


def _argmax_first(scores):
    return int(np.argmax(scores))        # 最初の最大値(numpy 既定)


def _greedy_with_step_ir(model, src_bits, step_ir, spec, *, include_l):
    """1 ステップ IR を T 回帰還させる貪欲生成(FPGA ラッパの意味論をテスト側で再現)"""
    embed = _embed_hard_bits(model.src_embedding).astype(np.int64)
    enc_ir, _, _ = export_rddlgn_encoder_ir(model.encoder, SRC_LEN, id_prefix='enc_')
    context = simulate(enc_ir, src_bits[None, :])[0].astype(np.int64)
    p = np.asarray(spec['init_bits'], dtype=np.int64)
    k, gs = spec['group_sum']['k'], spec['group_sum']['group_size']
    tokens, prev = [], BOS
    for _ in range(MAX_NEW):
        if include_l:
            ext = embed[prev]
        else:
            with torch.no_grad():
                l_t = model.decoder.l_layers(torch.tensor(embed[prev][None, :], dtype=torch.float32))
            ext = np.round(l_t.numpy()[0]).astype(np.int64)
        x = np.concatenate([ext, context, p])[None, :]
        y = simulate(step_ir, x)[0]
        p = y[:spec['state_width']]
        m = y[spec['state_width']:]
        scores = m.reshape(k, gs).sum(axis=1)
        tok = _argmax_first(scores)
        tokens.append(tok)
        if tok == EOS:
            break
        prev = tok
    return tokens


class TestDecoderStep:
    @pytest.mark.parametrize('include_l', [True, False])
    def test_step_feedback_matches_greedy_golden(self, include_l):
        model, cfg = _model()
        step_ir, spec = export_rddlgn_decoder_step_ir(model, include_l=include_l)
        validate(step_ir, target='hdl')
        d_emb = cfg.d_emb
        rng = np.random.default_rng(0)
        src_bits = rng.integers(0, 2, size=SRC_LEN * d_emb).astype(np.int64)
        want = simulate_rddlgn_greedy_bits(model, src_bits, bos_token=BOS, eos_token=EOS,
                                           max_new_tokens=MAX_NEW)
        got = _greedy_with_step_ir(model, src_bits, step_ir, spec, include_l=include_l)
        assert got == want
        assert len(want) >= 1

    def test_shape_and_contract(self):
        model, cfg = _model()
        step_ir, spec = export_rddlgn_decoder_step_ir(model)
        p_out = model.decoder.p_out_dim
        ctx = cfg.k_layer_sizes[-1]
        assert spec['state_width'] == p_out
        assert spec['ext_width'] == cfg.d_emb + ctx
        assert spec['layout']['emb'] == [0, cfg.d_emb]
        assert spec['layout']['state'] == [cfg.d_emb + ctx, cfg.d_emb + ctx + p_out]
        assert step_ir.inputs[0].width == cfg.d_emb + ctx + p_out
        assert step_ir.outputs[0].width == p_out + spec['readout_width']
        assert step_ir.nodes[-1].type == 'concat'          # group_sum 終端ではない
        assert not any(n.type in ('recurrent', 'scan_var', 'group_sum') for n in step_ir.nodes)
        cs = check_step_ir(step_ir, spec['state_width'])
        assert cs.state_in == tuple(spec['state_in'])
        assert cs.state_out == tuple(spec['state_out'])
        manifest = step_ir_manifest(cs, spec['init_bits'], steps=MAX_NEW)
        assert manifest['state_in'] == spec['state_in']
        assert manifest['init'] == '0x0'                   # init_mode='zero'
        assert spec['group_sum']['k'] * spec['group_sum']['group_size'] == spec['readout_width']

    def test_encoder_ir_unchanged_by_step_export(self):
        model, _ = _model()
        before = export_rddlgn_encoder_ir(model.encoder, SRC_LEN, id_prefix='enc_')[0]
        export_rddlgn_decoder_step_ir(model)
        after = export_rddlgn_encoder_ir(model.encoder, SRC_LEN, id_prefix='enc_')[0]
        assert [n.id for n in before.nodes] == [n.id for n in after.nodes]
        assert [n.type for n in before.nodes] == ['recurrent', 'recurrent']
