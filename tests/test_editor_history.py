"""学習履歴ストア(HistoryStore)のユニットテスト。

MNIST非依存の合成テンソルのみ使用する(常時グリーン基準の高速枠)。
"""

import time

import pytest
import torch

from tools.editor.backend.capture import (
    build_snapshot,
    decide_capture_level,
    total_weights_bytes,
)
from tools.editor.backend.history import HistoryStore, LayerRecord, StepSnapshot


def _snap(step: int, *, out_dim: int = 8, keyframe: bool = False,
          seed: int | None = None) -> StepSnapshot:
    """合成スナップショット。weightsはstepから決定的に生成(bitwise照合用)。"""
    gen = torch.Generator().manual_seed(seed if seed is not None else step)
    weights = torch.randn(out_dim, 16, generator=gen)
    return StepSnapshot(
        step=step,
        epoch=0,
        loss=1.0 / (step + 1),
        acc=step / 100.0,
        ts=time.time(),
        method='bp',
        layers={'n1_lutlayer': LayerRecord(weights=weights)},
        opt_state={'dummy': step} if keyframe else None,
        rng=torch.get_rng_state().clone() if keyframe else None,
    )


class TestHistoryStoreRAM:
    def test_append_get_roundtrip_bitwise(self, tmp_path):
        store = HistoryStore('t1', tmp_path)
        for i in range(1, 21):
            store.append(_snap(i))
        for i in (1, 10, 20):
            got = store.get(i)
            assert got is not None and got.step == i
            expected = _snap(i).layers['n1_lutlayer'].weights
            assert torch.equal(got.layers['n1_lutlayer'].weights, expected)
        store.close()

    def test_get_floor_rounding(self, tmp_path):
        store = HistoryStore('t2', tmp_path, stride=2)
        for i in range(0, 11):
            store.append(_snap(i))
        # stride=2なので奇数stepは非capture → 直近下位へ丸め
        got = store.get(5)
        assert got is not None and got.step == 4
        assert store.get(0) is not None
        store.close()

    def test_index_and_keyframes(self, tmp_path):
        store = HistoryStore('t3', tmp_path)
        for i in range(0, 21):
            store.append(_snap(i, keyframe=(i % 10 == 0)))
        idx = store.index()
        assert idx['min_step'] == 0
        assert idx['max_step'] == 20
        assert idx['count'] == 21
        assert idx['keyframes'] == [0, 10, 20]
        assert store.nearest_keyframe(15) == 10
        assert store.nearest_keyframe(9) == 0
        store.close()


class TestHistoryStoreDisk:
    def test_ring_overflow_to_disk_and_reload(self, tmp_path):
        # 1スナップショット≈512B(8×16×4B)。リング予算を絞って強制退避させる
        store = HistoryStore('t4', tmp_path, ring_bytes=4096, chunk_steps=4)
        for i in range(1, 41):
            store.append(_snap(i))
        store.flush()
        # 古いstepはディスクチャンクから読める(bitwise一致)
        got = store.get(1)
        assert got is not None and got.step == 1
        assert torch.equal(
            got.layers['n1_lutlayer'].weights, _snap(1).layers['n1_lutlayer'].weights
        )
        # チャンクファイルが実際に生成されている
        chunks = list(store.dir.glob('chunk_*.pt'))
        assert len(chunks) > 0
        # 最新はRAMリングから読める
        got_new = store.get(40)
        assert got_new is not None and got_new.step == 40
        store.close()

    def test_truncate_after(self, tmp_path):
        store = HistoryStore('t5', tmp_path, ring_bytes=4096, chunk_steps=4)
        for i in range(1, 31):
            store.append(_snap(i, keyframe=(i % 10 == 0)))
        store.truncate_after(15)
        idx = store.index()
        assert idx['max_step'] == 15
        assert store.nearest_keyframe(30) == 10  # 20のkeyframeは切り捨て済み
        # 15より後は直近下位(15)へ丸まる
        got = store.get(25)
        assert got is not None and got.step == 15
        # 切り捨て分はbranches/へ退避されている
        branch_dir = store.dir / 'branches'
        assert branch_dir.exists() and any(branch_dir.iterdir())
        store.close()

    def test_disk_budget_escalates_stride(self, tmp_path):
        # ディスク予算をほぼゼロにして、チャンク退避時にstride倍化を誘発する
        store = HistoryStore(
            't6', tmp_path, ring_bytes=2048, chunk_steps=2, max_disk_mb=0
        )
        for i in range(1, 33):
            store.append(_snap(i))
        assert store.stride > 1
        store.close()


class TestCaptureLevel:
    def _lut_layer(self, out_dim: int, n_entries: int = 16):
        class _Fake:
            def __init__(self):
                self.weights = torch.zeros(out_dim, n_entries)
                self.indices = torch.zeros(2, out_dim, dtype=torch.int64)

        return _Fake()

    def test_decide_capture_level(self):
        small = [('n1', self._lut_layer(100))]
        assert decide_capture_level(small) == ('full', 1)
        # 8MB超〜64MB以下 → weights(float32×16entriesで約24MB)
        mid = [('n1', self._lut_layer(400_000))]
        level, stride = decide_capture_level(mid)
        assert level == 'weights' and stride > 1
        big = [('n1', self._lut_layer(2_000_000))]
        assert decide_capture_level(big)[0] == 'metrics'

    def test_total_weights_bytes(self):
        layers = [('n1', self._lut_layer(10)), ('n2', self._lut_layer(20))]
        assert total_weights_bytes(layers) == (10 + 20) * 16 * 4

    def test_build_snapshot_levels(self):
        layer = self._lut_layer(8)
        layers = [('n1_lutlayer', layer)]
        # metricsレベルはkeyframe以外None
        assert build_snapshot(
            step=1, epoch=0, loss=1.0, acc=0.5, method='bp',
            lut_layers=layers, level='metrics',
        ) is None
        # fullレベル: weights(f32)+grad(f16)+keyframeでindices/rng/opt_state
        grad = torch.randn(8, 16)
        snap = build_snapshot(
            step=2, epoch=0, loss=1.0, acc=0.5, method='bp',
            lut_layers=layers, level='full',
            grads={'n1_lutlayer': grad},
            keyframe=True, opt_state={'k': 1},
        )
        assert snap is not None
        rec = snap.layers['n1_lutlayer']
        assert rec.weights.dtype == torch.float32
        assert rec.grad is not None and rec.grad.dtype == torch.float16
        assert rec.indices is not None
        assert snap.is_keyframe and snap.rng is not None
        # weightsレベル: gradを渡してもトレースは持たない
        snap_w = build_snapshot(
            step=3, epoch=0, loss=1.0, acc=0.5, method='bp',
            lut_layers=layers, level='weights', trace={'x': 1},
        )
        assert snap_w is not None and snap_w.trace is None


class TestSnapshotRestore:
    def test_restore_weights_into_layer(self, tmp_path):
        """スナップショットからの重み書き戻し(branch復元の基本操作)。"""
        layer = TestCaptureLevel()._lut_layer(8)
        layers = [('n1_lutlayer', layer)]
        layer.weights.normal_()
        snap = build_snapshot(
            step=1, epoch=0, loss=1.0, acc=0.5, method='bp',
            lut_layers=layers, level='full',
        )
        assert snap is not None
        original = snap.layers['n1_lutlayer'].weights.clone()
        layer.weights.normal_()  # 学習が進んだ想定
        assert not torch.equal(layer.weights, original)
        with torch.no_grad():
            layer.weights.copy_(snap.layers['n1_lutlayer'].weights)
        assert torch.equal(layer.weights, original)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
