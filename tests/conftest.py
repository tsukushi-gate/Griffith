"""pytest共通設定: マーカー・seed・許容誤差の一元管理。

マーカー(pyproject.toml [tool.pytest.ini_options] で登録):
    slow     — 実行に時間がかかるテスト
    cuda     — CUDA GPUが必要(非対応環境では自動skip)
    paper    — 論文再現テスト(分単位、夜間/Modal実行)
    hdl      — HDLシミュレータ(iverilog等)が必要
    hdl_vhdl — GHDLが必要

高速スイート(常時グリーン基準):
    uv run pytest -m "not slow and not cuda and not paper and not hdl and not hdl_vhdl"
"""

import os
import random
import sys

import numpy as np
import pytest
import torch
from hypothesis import settings

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# ============================================================
# hypothesisプロファイル
#   ci : 高速スイート2分以内維持のためexamples控えめ。deadline=None必須
#        (CI環境の速度揺らぎによるフレーク禁止)。print_blobで再現コマンドを表示
#   dev: ローカルでの網羅探索用(HYPOTHESIS_PROFILE=dev で切替)
# ============================================================

settings.register_profile(
    'ci', deadline=None, max_examples=25, derandomize=False, print_blob=True,
)
settings.register_profile(
    'dev', deadline=None, max_examples=100, print_blob=True,
)
settings.load_profile(os.environ.get('HYPOTHESIS_PROFILE', 'ci'))

# ============================================================
# 許容誤差の全体規約(L1/L2テストで共通使用)
# ============================================================

TOL_F64_EQ = 1e-12     # float64での式一致(forward)
TOL_F64_GRAD = 1e-10   # float64での勾配一致(参照実装との照合)
TOL_F32_EQ = 1e-6      # float32での式一致
GRADCHECK_KW = dict(atol=1e-6, rtol=1e-3)  # torch.autograd.gradcheck 標準設定


def assert_all_finite(tensor: torch.Tensor, ctx: str) -> None:
    """テンソルの全要素が有限(NaN/infなし)であることを検証する。

    Args:
        tensor: 検査対象テンソル
        ctx: 失敗時に表示する文脈(レイヤ名・ケース名など)
    """
    if torch.isfinite(tensor).all():
        return
    n_nan = int(torch.isnan(tensor).sum())
    n_inf = int(torch.isinf(tensor).sum())
    raise AssertionError(
        f'{ctx}: 非有限値を検出 '
        f'(NaN={n_nan}, inf={n_inf}, shape={tuple(tensor.shape)})'
    )


def pytest_collection_modifyitems(config, items):
    """@cudaマーカーのテストをCUDA非対応環境で自動skip"""
    if torch.cuda.is_available():
        return
    skip_cuda = pytest.mark.skip(reason='CUDA未対応環境')
    for item in items:
        if 'cuda' in item.keywords:
            item.add_marker(skip_cuda)


@pytest.fixture
def seed():
    """torch/numpy/randomのseedを42に固定"""
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    return 42
