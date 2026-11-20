"""回復学習(convert/recover.py)のテスト。

L1相当(忠実性): rebuild直後のhard forwardが元転写IRのsimulateとビット一致。
L4(回復): 意図的に劣化させた転写IRをrecover()でhard acc=100%へ回復。

高速スイート2分以内維持のため、XOR(2入力)の極小規模に抑える。
"""

import copy
import itertools

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from griffith_logic.convert.recover import (
    RECOVER_PARAMETERIZATIONS,
    RecoverConfig,
    rebuild_trainable,
    recover,
)
from griffith_logic.convert.transcribe import TranscribeSpec, transcribe_mlp
from griffith_logic.ir.serialize import tt_from_hex, tt_to_hex
from griffith_logic.ir.simulate import simulate
from griffith_logic.layers.lut import LutLayer

# L4=学習収束テストは全てlearning(CI除外、ローカルmake testでは実行)
pytestmark = pytest.mark.learning


def _xor_data():
    """XOR(2入力)の完全データセット(x, y)。"""
    x = torch.tensor(list(itertools.product([0.0, 1.0], repeat=2)))
    y = torch.tensor([int(int(a) ^ int(b)) for a, b in x.tolist()])
    return x, y


def _xor_transcribe_ir(hidden: int = 8):
    """XOR用MLPを転写した小さなIRを返す(較正データ付き)。"""
    torch.manual_seed(0)
    x, _ = _xor_data()
    x_np = x.numpy().astype(np.int64)
    mlp = nn.Sequential(nn.Linear(2, hidden), nn.Linear(hidden, 2))
    return transcribe_mlp(
        mlp, calib_inputs=x_np, spec=TranscribeSpec(n_inputs=2),
    )


def _flip_hidden_bits(graph, mask: int = 0b0001):
    """隠れlut_layerの真理値表を数ビット反転させて劣化IRを作る。"""
    g = copy.deepcopy(graph)
    for node in g.nodes:
        if (node.type == 'lut_layer'
                and node.train_info.get('kind') == 'transcribe_hidden'):
            tts = tt_from_hex(node.params['truth_tables'])
            node.params['truth_tables'] = tt_to_hex([t ^ mask for t in tts])
    return g


# ============================================================
# 1. 忠実性(L1相当): rebuild直後のhard forward == 元IRのsimulate
# ============================================================

@pytest.mark.parametrize('param', RECOVER_PARAMETERIZATIONS)
def test_rebuild_fidelity(param):
    """3パラメータ化それぞれで、rebuild直後(eval)のhard forwardが
    元転写IRのsimulateと全入力でビット厳密に一致する。"""
    graph = _xor_transcribe_ir()
    x = np.array(list(itertools.product([0, 1], repeat=2)), dtype=np.int64)
    sim = simulate(graph, x)  # (B, k) 整数スコア

    model = rebuild_trainable(graph, param)
    model.eval()
    with torch.no_grad():
        out = model(torch.tensor(x, dtype=torch.float32)).numpy()

    # スコアそのものがビット厳密に一致(argmaxだけでなく値まで)
    np.testing.assert_array_equal(out, sim.astype(np.float32))


def test_rebuild_rejects_unsupported_parameterization():
    """任意真理値表を厳密表現できないパラメータ化は拒否する。"""
    graph = _xor_transcribe_ir()
    with pytest.raises(ValueError):
        rebuild_trainable(graph, 'softmax')
    with pytest.raises(ValueError):
        rebuild_trainable(graph, 'iwp')


# ============================================================
# 1b. selection伝播: rebuild_trainableがtrain_info.selectionを書き込む
# ============================================================

def test_rebuild_writes_selection_into_train_info():
    """selection引数がlut_layerノードのtrain_info.selectionへ反映される。"""
    graph = _xor_transcribe_ir()

    model_soft = rebuild_trainable(graph, 'warp', selection='soft')
    model_hard = rebuild_trainable(graph, 'warp', selection='hard_st')

    lut_layers_soft = [m for m in model_soft.modules() if isinstance(m, LutLayer)]
    lut_layers_hard = [m for m in model_hard.modules() if isinstance(m, LutLayer)]
    assert lut_layers_soft and lut_layers_hard
    assert all(layer.selection == 'soft' for layer in lut_layers_soft)
    assert all(layer.selection == 'hard_st' for layer in lut_layers_hard)


def test_rebuild_default_selection_is_soft():
    """selection省略時は既定'soft'(既存呼び出しの後方互換)。"""
    graph = _xor_transcribe_ir()
    model = rebuild_trainable(graph, 'independent')
    lut_layers = [m for m in model.modules() if isinstance(m, LutLayer)]
    assert lut_layers and all(layer.selection == 'soft' for layer in lut_layers)


def test_rebuild_hard_st_requires_compatible_parameterization():
    """hard_stはLutLayer側でsoftmax/warpのみ対応。独立エントリ系は非対応
    パラメータ化としてエラーになる(rebuild_trainable経由でも同様に伝播)。
    """
    graph = _xor_transcribe_ir()
    with pytest.raises((ValueError, AssertionError)):
        rebuild_trainable(graph, 'independent', selection='hard_st')
    with pytest.raises((ValueError, AssertionError)):
        rebuild_trainable(graph, 'raw', selection='hard_st')


@pytest.mark.parametrize('selection', LutLayer.SELECTIONS)
def test_rebuild_fidelity_with_selection(selection):
    """warpパラメータ化+各selectionで、rebuild直後(eval)のhard forwardが
    元転写IRのsimulateとビット厳密に一致する(eval経路はselection非依存)。

    これはL5三段一致規律のcritical invariant: selection='hard_st'でも
    build直後のhard forward(=eval)が崩れないことを担保する。
    """
    graph = _xor_transcribe_ir()
    x = np.array(list(itertools.product([0, 1], repeat=2)), dtype=np.int64)
    sim = simulate(graph, x)

    model = rebuild_trainable(graph, 'warp', selection=selection)
    model.eval()
    with torch.no_grad():
        out = model(torch.tensor(x, dtype=torch.float32)).numpy()

    np.testing.assert_array_equal(out, sim.astype(np.float32))


# ============================================================
# 2. 回復(L4): 劣化IR → recover() → hard acc = 100%
# ============================================================

@pytest.mark.parametrize('param', RECOVER_PARAMETERIZATIONS)
def test_recover_restores_hard_accuracy(param):
    """真理値表を反転して劣化させた転写IRを、recoverでhard acc=100%へ回復。

    seed固定・数秒以内で完了する規模に抑える。
    """
    torch.manual_seed(0)
    graph = _xor_transcribe_ir()
    degraded = _flip_hidden_bits(graph)

    x, y = _xor_data()
    x_np = x.numpy().astype(np.int64)
    y_np = y.numpy()

    # 劣化により hard acc < 100% であることを確認
    acc_before = (simulate(degraded, x_np).argmax(-1) == y_np).mean()
    assert acc_before < 1.0

    ds = TensorDataset(x, y)
    cfg = RecoverConfig(parameterization=param, epochs=200, lr=0.05, tau=4.0)
    recovered, metrics = recover(
        degraded, DataLoader(ds, batch_size=4, shuffle=True),
        DataLoader(ds, batch_size=4), cfg,
    )

    assert metrics['hard_acc_after'] == 1.0
    # 回復後IRのsimulateでも100%
    assert (simulate(recovered, x_np).argmax(-1) == y_np).mean() == 1.0


# ============================================================
# 3. metricsのbefore/afterが単調(after >= before)
# ============================================================

def test_metrics_monotonic():
    """回復後のhard精度が回復前を下回らない。"""
    torch.manual_seed(0)
    degraded = _flip_hidden_bits(_xor_transcribe_ir())
    x, y = _xor_data()
    ds = TensorDataset(x, y)
    _, metrics = recover(
        degraded, DataLoader(ds, batch_size=4, shuffle=True),
        DataLoader(ds, batch_size=4),
        RecoverConfig(parameterization='independent', epochs=200,
                      lr=0.05, tau=4.0),
    )
    assert metrics['hard_acc_after'] >= metrics['hard_acc_before']
    assert metrics['soft_acc_after'] >= metrics['soft_acc_before']


# ============================================================
# 4. loss_fn注入の動作(ダミーlossで学習が回る)
# ============================================================

def test_custom_loss_fn_injection():
    """loss_fn注入経路が動作する(将来KD損失注入口)。"""
    torch.manual_seed(0)
    degraded = _flip_hidden_bits(_xor_transcribe_ir())
    x, y = _xor_data()
    ds = TensorDataset(x, y)

    calls = {'n': 0}

    def dummy_loss(output, target):
        calls['n'] += 1
        return nn.functional.cross_entropy(output, target)

    recovered, metrics = recover(
        degraded, DataLoader(ds, batch_size=4, shuffle=True),
        DataLoader(ds, batch_size=4),
        RecoverConfig(parameterization='independent', epochs=3, lr=0.05),
        loss_fn=dummy_loss,
    )
    assert calls['n'] > 0  # 注入lossが実際に呼ばれた
    assert 'hard_acc_after' in metrics


def test_invalid_config_parameterization():
    """RecoverConfigは非対応パラメータ化を構築時に拒否する。"""
    with pytest.raises(ValueError):
        RecoverConfig(parameterization='softmax')


# ============================================================
# 5. RecoverConfig.selection
# ============================================================

def test_config_selection_default_and_validation():
    """selectionの既定は'soft'。LutLayer.SELECTIONS外の値は拒否する。"""
    cfg = RecoverConfig()
    assert cfg.selection == 'soft'
    with pytest.raises(ValueError):
        RecoverConfig(selection='bogus')


def test_recover_end_to_end_with_hard_st_selection():
    """selection='hard_st'(+warp)でrecover()が最後まで動作し、
    hard_acc_afterが劣化前のacc_before以上まで回復する。
    """
    torch.manual_seed(0)
    degraded = _flip_hidden_bits(_xor_transcribe_ir())
    x, y = _xor_data()
    ds = TensorDataset(x, y)

    cfg = RecoverConfig(
        parameterization='warp', selection='hard_st',
        epochs=50, lr=0.05, tau=4.0,
    )
    recovered, metrics = recover(
        degraded, DataLoader(ds, batch_size=4, shuffle=True),
        DataLoader(ds, batch_size=4), cfg,
    )
    assert metrics['hard_acc_after'] >= metrics['hard_acc_before']
