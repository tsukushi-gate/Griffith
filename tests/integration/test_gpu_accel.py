"""training.gpu_accel(CUDA Graphs自動適用レイヤ)の契約テスト。

CPU側: CPU実行(またはmode='off'/環境変数無効化)では完全な素通し(eager)であり、
手書きの学習ループとloss系列・最終パラメータがビット単位で一致することを検証する。

CUDA側(@cuda): 実際にcapture/replayする経路について
    - eagerとのloss系列・最終パラメータの一致
    - 呼び出し1回 = 更新1回(optimizerのstepカウンタ)
    - 端数バッチ(入力shapeの変化)時のeagerフォールバックとreplay再開
    - capture失敗時の巻き戻し(乱数生成器の状態も含む)
    - 返すlossが呼び出しごとに別テンソルであること
    - lrをテンソルで渡せばLRスケジューラが再生にも反映されること / floatなら警告
を検証する。
"""

import os
import sys
import warnings

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))  # conftest用

from conftest import TOL_F32_EQ

from griffith_logic.group_sum import GroupSum
from griffith_logic.layers.lut import LutLayer
from griffith_logic.training import make_train_step, wants_capturable

IN_DIM, OUT_DIM, CLASSES, BATCH, STEPS = 16, 32, 4, 8, 6


def _build(seed: int):
    torch.manual_seed(seed)
    model = torch.nn.Sequential(
        LutLayer(IN_DIM, OUT_DIM, n_inputs=2),
        GroupSum(CLASSES, tau=1.0),
    )
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    return model, opt


def _batches(seed: int):
    g = torch.Generator().manual_seed(seed + 1000)
    return [
        (torch.rand(BATCH, IN_DIM, generator=g),
         torch.randint(0, CLASSES, (BATCH,), generator=g))
        for _ in range(STEPS)
    ]


def _run_manual(seed: int):
    model, opt = _build(seed)
    losses = []
    for x, y in _batches(seed):
        opt.zero_grad(set_to_none=True)
        loss = torch.nn.functional.cross_entropy(model(x), y)
        loss.backward()
        opt.step()
        losses.append(loss.detach().clone())
    return losses, model


def _run_accel(seed: int, mode: str):
    model, opt = _build(seed)

    def compute_loss(x, y):
        return torch.nn.functional.cross_entropy(model(x), y)

    step, info = make_train_step(model, opt, compute_loss, mode=mode)
    losses = [step(x, y).detach().clone() for x, y in _batches(seed)]
    return losses, model, info


class TestCpuPassthrough:
    """CPUでは素通し: 手書きループとビット一致"""

    def _assert_identical(self, mode: str):
        ref_losses, ref_model = _run_manual(seed=0)
        acc_losses, acc_model, info = _run_accel(seed=0, mode=mode)
        assert info['enabled'] is False
        for a, b in zip(ref_losses, acc_losses):
            assert torch.equal(a, b)
        for p, q in zip(ref_model.parameters(), acc_model.parameters()):
            assert torch.equal(p, q)

    def test_auto_on_cpu_is_bitwise_identical(self):
        self._assert_identical('auto')

    def test_off_is_bitwise_identical(self):
        self._assert_identical('off')


class TestControls:
    def test_env_var_disables(self, monkeypatch):
        monkeypatch.setenv('MDL_NO_CUDA_GRAPHS', '1')
        _, _, info = _run_accel(seed=1, mode='force')
        assert info['enabled'] is False
        assert wants_capturable('force') is False

    def test_wants_capturable(self, monkeypatch):
        monkeypatch.delenv('MDL_NO_CUDA_GRAPHS', raising=False)
        assert wants_capturable('auto') is True
        assert wants_capturable('force') is True
        assert wants_capturable('off') is False

    def test_caller_reasons_disable(self):
        model, opt = _build(seed=2)
        step, info = make_train_step(
            model, opt,
            lambda x, y: torch.nn.functional.cross_entropy(model(x), y),
            mode='force', reasons=['tau annealが有効'],
        )
        assert info['enabled'] is False
        x, y = _batches(seed=2)[0]
        assert torch.isfinite(step(x, y))


# ----------------------------------------------------------------------
# CUDA経路(実際にcapture/replayする)
# ----------------------------------------------------------------------

CUDA_BATCH, CUDA_STEPS = 64, 8


def _require_kernel():
    from griffith_logic.cuda import cuda_available
    if not cuda_available():
        pytest.skip('CUDAカーネルのコンパイル失敗')


def _build_cuda(seed: int, *, lr_tensor: bool = True, gumbel: bool = False):
    torch.manual_seed(seed)
    model = torch.nn.Sequential(
        LutLayer(IN_DIM, OUT_DIM, n_inputs=2, device='cuda', gumbel_noise=gumbel),
        LutLayer(OUT_DIM, OUT_DIM, n_inputs=4, device='cuda', parameterization='iwp'),
        GroupSum(CLASSES, tau=1.0),
    ).cuda()
    lr = torch.tensor(1e-2, device='cuda') if lr_tensor else 1e-2
    opt = torch.optim.AdamW(model.parameters(), lr=lr, capturable=True)
    return model, opt


def _cuda_batches(sizes):
    g = torch.Generator(device='cuda').manual_seed(1234)
    return [
        (torch.rand(b, IN_DIM, device='cuda', generator=g),
         torch.randint(0, CLASSES, (b,), device='cuda', generator=g))
        for b in sizes
    ]


def _run_eager_cuda(batches, *, seed=0, scheduler=False, gumbel=False, item=False):
    model, opt = _build_cuda(seed, gumbel=gumbel)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=2, gamma=0.5) if scheduler else None
    torch.manual_seed(99)
    losses = []
    for x, y in batches:
        opt.zero_grad()
        loss = F.cross_entropy(model(x), y)
        if item:
            loss.item()
        loss.backward()
        opt.step()
        if sched is not None:
            sched.step()
        losses.append(loss.detach().clone())
    return losses, model, opt


def _run_graph_cuda(batches, *, seed=0, scheduler=False, gumbel=False, item=False):
    model, opt = _build_cuda(seed, gumbel=gumbel)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=2, gamma=0.5) if scheduler else None

    def compute_loss(x, y):
        loss = F.cross_entropy(model(x), y)
        if item:
            loss.item()  # capture中は同期できないため、captureを失敗させる
        return loss

    step, info = make_train_step(model, opt, compute_loss, mode='force')
    torch.manual_seed(99)
    losses = []
    for x, y in batches:
        losses.append(step(x, y))  # cloneしない(返り値が上書きされないことも見る)
        if sched is not None:
            sched.step()
    return losses, model, opt, info, step


def _assert_same(ref_losses, ref_model, losses, model):
    assert len(ref_losses) == len(losses)
    for i, (a, b) in enumerate(zip(ref_losses, losses)):
        assert torch.allclose(a, b, atol=TOL_F32_EQ, rtol=0), (
            f'step {i}: loss不一致 eager={a.item():.8f} graph={b.item():.8f}'
        )
    for p, q in zip(ref_model.parameters(), model.parameters()):
        assert torch.allclose(p, q, atol=TOL_F32_EQ, rtol=0), (
            f'最終パラメータ不一致 max diff {(p - q).abs().max():.3e}'
        )


def _closure_state(step):
    """make_train_step内部のstate辞書(白箱テスト用)。"""
    for cell in step.__closure__ or ():
        value = cell.cell_contents
        if isinstance(value, dict) and 'pre_warmup_model_state' in value:
            return value
    raise AssertionError('step_fnのクロージャにstate辞書が見つからない')


@pytest.mark.cuda
class TestCudaGraphPath:
    def test_matches_eager_loss_and_params(self):
        _require_kernel()
        batches = _cuda_batches([CUDA_BATCH] * CUDA_STEPS)
        ref_losses, ref_model, _ = _run_eager_cuda(batches)
        losses, model, _, info, _ = _run_graph_cuda(batches)
        assert info['enabled'] is True, info
        _assert_same(ref_losses, ref_model, losses, model)

    def test_one_call_is_one_update(self):
        _require_kernel()
        batches = _cuda_batches([CUDA_BATCH] * CUDA_STEPS)
        _, model, opt, info, _ = _run_graph_cuda(batches)
        assert info['enabled'] is True, info
        for p in model.parameters():
            assert int(opt.state[p]['step'].item()) == CUDA_STEPS

    def test_ragged_batch_falls_back_and_resumes_replay(self):
        _require_kernel()
        # warmup 3回 → capture → 端数バッチ(shape違い)→ replay再開
        sizes = [CUDA_BATCH] * 5 + [CUDA_BATCH - 7] + [CUDA_BATCH] * 2
        batches = _cuda_batches(sizes)
        ref_losses, ref_model, _ = _run_eager_cuda(batches)
        losses, model, opt, info, _ = _run_graph_cuda(batches)
        assert info['enabled'] is True, info
        assert info['captured_shape']['x'] == (CUDA_BATCH, IN_DIM)
        _assert_same(ref_losses, ref_model, losses, model)
        for p in model.parameters():
            assert int(opt.state[p]['step'].item()) == len(sizes)

    def test_capture_failure_rolls_back_including_rng(self):
        """capture失敗時、warmup前の状態(乱数生成器含む)へ戻してeagerで続行する。

        Gumbelノイズ(乱数)を使うモデルで、compute_loss内の.item()により
        captureを失敗させる。巻き戻し後の再実行が乱数系列も含めてeagerと
        一致すること、例外が外へ出ないこと、呼び出し1回=更新1回が保たれることを見る。
        """
        _require_kernel()
        batches = _cuda_batches([CUDA_BATCH] * CUDA_STEPS)
        ref_losses, ref_model, _ = _run_eager_cuda(batches, gumbel=True, item=True)
        losses, model, opt, info, step = _run_graph_cuda(batches, gumbel=True, item=True)
        assert info['attempted'] is True
        assert info['enabled'] is False
        assert info['summary'].startswith('capture failed')
        _assert_same(ref_losses, ref_model, losses, model)
        for p in model.parameters():
            assert int(opt.state[p]['step'].item()) == CUDA_STEPS
        state = _closure_state(step)
        assert state['pre_warmup_model_state'] is None
        assert state['warmup_batches'] == []

    def test_snapshot_released_after_capture(self):
        _require_kernel()
        batches = _cuda_batches([CUDA_BATCH] * CUDA_STEPS)
        _, _, _, info, step = _run_graph_cuda(batches)
        assert info['enabled'] is True, info
        state = _closure_state(step)
        assert state['pre_warmup_model_state'] is None
        assert state['pre_warmup_optim_state'] is None
        assert state['pre_warmup_rng_state'] is None
        assert state['warmup_batches'] == []

    def test_returned_losses_are_not_overwritten(self):
        _require_kernel()
        batches = _cuda_batches([CUDA_BATCH] * CUDA_STEPS)
        _, _, _, info, _ = _run_graph_cuda(batches)
        losses, _, _, info, _ = _run_graph_cuda(batches)
        assert info['enabled'] is True, info
        replayed = losses[3:]  # capture以降
        ptrs = {t.data_ptr() for t in replayed}
        assert len(ptrs) == len(replayed), '再生時のlossが同じテンソルを使い回している'
        assert len({round(t.item(), 7) for t in replayed}) > 1

    def test_tensor_lr_follows_scheduler(self):
        """lrをテンソルで渡せば、LRスケジューラの変更が再生にも反映される。"""
        _require_kernel()
        batches = _cuda_batches([CUDA_BATCH] * CUDA_STEPS)
        ref_losses, ref_model, ref_opt = _run_eager_cuda(batches, scheduler=True)
        with warnings.catch_warnings():
            warnings.simplefilter('error')  # テンソルlrでは警告が出ないこと
            losses, model, opt, info, _ = _run_graph_cuda(batches, scheduler=True)
        assert info['enabled'] is True, info
        _assert_same(ref_losses, ref_model, losses, model)
        assert float(opt.param_groups[0]['lr']) == float(ref_opt.param_groups[0]['lr'])

    def test_float_lr_warns(self):
        _require_kernel()
        model, opt = _build_cuda(0, lr_tensor=False)
        with pytest.warns(UserWarning, match='lr が float'):
            make_train_step(
                model, opt, lambda x, y: F.cross_entropy(model(x), y), mode='force',
            )
