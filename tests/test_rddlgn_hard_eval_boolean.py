"""RDDLGN hard eval時の再帰隠れ状態がBoolean(超立方体の頂点)であることの検証。

`_init_hidden`(griffith_logic/models/rddlgn.py)がeval時にも
リテラル0.5やsigmoid(N(0,scale^2))の分数値を返していたバグの修正テスト。
修正前はK-layers/P-layersの再帰初期状態が超立方体の頂点(0/1)ではなかったため、
「.eval()すれば完全なBoolean論理回路として振る舞う」というRDDLGN論文(collapse)
およびデプロイキットのビット厳密等価の前提が崩れていた。
"""

import pytest
import torch

from griffith_logic.models.rddlgn import (
    RDDLGN,
    RDDLGNClassifier,
    RDDLGNConfig,
    _init_hidden,
)

DEVICE = 'cpu'
INIT_MODES = ['zero', 'one', 'gaussian']


def _assert_boolean(x: torch.Tensor, label: str) -> None:
    unique = torch.unique(x)
    assert all(v in (0.0, 1.0) for v in unique.tolist()), (
        f'{label}: eval時の再帰状態が{{0.0, 1.0}}以外の値を含む: {unique.tolist()}'
    )


def _tiny_config() -> RDDLGNConfig:
    return RDDLGNConfig.tiny()


# ---------------------------------------------------------------------------
# _init_hidden 単体テスト
# ---------------------------------------------------------------------------

class TestInitHiddenUnit:

    @pytest.mark.parametrize('mode', INIT_MODES)
    def test_eval_is_boolean(self, mode):
        x = _init_hidden(4, 16, DEVICE, mode=mode, scale=0.5, training=False)
        _assert_boolean(x, f'_init_hidden(mode={mode}, training=False)')

    def test_eval_zero_collapses_to_zero(self):
        # リポジトリ全体の閾値規則は strict `>`(griffith_logic/binarize.py の
        # BinarizeFunction.forward/Binarize.forward、layers/lut.pyの重み離散化、
        # 本ファイルのLogicEmbedding.forward eval経路がすべて`> 0.5`）。
        # ちょうど閾値と一致する0.5は0側に倒れるため、init_mode='zero'は0.0になる。
        x = _init_hidden(4, 16, DEVICE, mode='zero', scale=0.5, training=False)
        assert torch.all(x == 0.0)

    def test_eval_one_stays_one(self):
        x = _init_hidden(4, 16, DEVICE, mode='one', scale=0.5, training=False)
        assert torch.all(x == 1.0)

    def test_train_zero_is_half_unchanged(self):
        """train時の挙動(0.5のまま)は一切変わらないことを確認する。"""
        x = _init_hidden(4, 16, DEVICE, mode='zero', scale=0.5, training=True)
        assert torch.all(x == 0.5)

    def test_default_training_arg_preserves_backward_compat(self):
        # training引数を省略する既存呼び出し(experiments/ap_trainer_rddlgn.py等)は
        # 従来通り連続値を返す(デフォルトtraining=True)。
        x = _init_hidden(4, 16, DEVICE, mode='zero', scale=0.5)
        assert torch.all(x == 0.5)


# ---------------------------------------------------------------------------
# RDDLGN / RDDLGNClassifier 統合テスト: 全タイムステップでBoolean
# ---------------------------------------------------------------------------

class TestRDDLGNHardEvalBoolean:

    @pytest.mark.parametrize('init_mode', INIT_MODES)
    def test_encoder_k_states_boolean_all_t(self, init_mode):
        """K-layers(エンコーダの再帰)の出力が全タイムステップでBoolean。"""
        config = _tiny_config()
        config.init_mode = init_mode
        model = RDDLGN(config, device=DEVICE)
        model.eval()

        captured: list[torch.Tensor] = []

        def hook(module, inputs, output):
            captured.append(output.detach().clone())

        handle = model.encoder.k_layers.register_forward_hook(hook)
        batch = 2
        src = torch.randint(0, config.vocab_size, (batch, config.seq_len))
        tgt = torch.randint(0, config.vocab_size, (batch, config.seq_len))
        with torch.no_grad():
            model(src, tgt)
        handle.remove()

        assert len(captured) == config.seq_len
        for t, k_t in enumerate(captured):
            _assert_boolean(k_t, f'encoder k_layers output t={t} (init_mode={init_mode})')

    @pytest.mark.parametrize('init_mode', INIT_MODES)
    def test_decoder_p_states_boolean_all_t(self, init_mode):
        """P-layers(デコーダの自己回帰)の出力が全タイムステップでBoolean。"""
        config = _tiny_config()
        config.init_mode = init_mode
        model = RDDLGN(config, device=DEVICE)
        model.eval()

        batch = 2
        src = torch.randint(0, config.vocab_size, (batch, config.seq_len))
        tgt = torch.randint(0, config.vocab_size, (batch, config.seq_len))
        with torch.no_grad():
            _, _, hidden_states = model(src, tgt, return_hidden=True)
        p_stack = hidden_states[1]  # (batch, tgt_len, p_out_dim)

        assert p_stack.shape[1] == config.seq_len
        for t in range(p_stack.shape[1]):
            _assert_boolean(p_stack[:, t, :], f'decoder p_layers output t={t} (init_mode={init_mode})')

    @pytest.mark.parametrize('init_mode', INIT_MODES)
    def test_classifier_encoder_k_states_boolean_all_t(self, init_mode):
        """RDDLGNClassifierでも同じ経路(RDDLGNEncoder)を使うため同様に検証する。"""
        config = _tiny_config()
        config.init_mode = init_mode
        model = RDDLGNClassifier(config, n_classes=5, device=DEVICE)
        model.eval()

        captured: list[torch.Tensor] = []

        def hook(module, inputs, output):
            captured.append(output.detach().clone())

        handle = model.encoder.k_layers.register_forward_hook(hook)
        batch = 2
        tokens = torch.randint(0, config.vocab_size, (batch, config.seq_len))
        with torch.no_grad():
            model(tokens)
        handle.remove()

        assert len(captured) == config.seq_len
        for t, k_t in enumerate(captured):
            _assert_boolean(
                k_t, f'classifier encoder k_layers output t={t} (init_mode={init_mode})'
            )

    def test_train_mode_initial_state_not_binarized(self):
        """train時は修正が発動せず、init_mode='zero'の初期状態は0.5のまま。

        RDDLGNEncoder.forwardが渡す最初のk_prev(=k_layersへの入力の末尾
        k_out_dim列)を forward_pre_hook で直接観測する。
        """
        config = _tiny_config()
        config.init_mode = 'zero'
        model = RDDLGN(config, device=DEVICE)
        model.train()

        captured_inputs: list[torch.Tensor] = []

        def pre_hook(module, inputs):
            captured_inputs.append(inputs[0].detach().clone())

        handle = model.encoder.k_layers.register_forward_pre_hook(pre_hook)
        batch = 2
        src = torch.randint(0, config.vocab_size, (batch, config.seq_len))
        tgt = torch.randint(0, config.vocab_size, (batch, config.seq_len))
        model(src, tgt)
        handle.remove()

        assert len(captured_inputs) == config.seq_len
        k_out_dim = model.encoder.k_out_dim
        first_k_prev = captured_inputs[0][:, -k_out_dim:]
        assert torch.all(first_k_prev == 0.5)
