"""学習履歴ストア(タイムマシン)。

学習スレッドが毎captured stepの重みスナップショットを append し、
REST/WS側が任意stepの状態を取り出す。真実源はこのストア。

設計(プラン確定事項):
  - ゲート確率分布は weights から決定的に導出できるため分布は保存しない
  - RAMリングはバイト予算制(小規模モデルなら全ステップがRAMに載る)
  - 溢れた分は CHUNK_STEPS 件ずつ writerスレッドが非同期にディスク退避
  - keyframe(opt_state/rng/indices付き)は branch再学習の復元起点
  - ディスク予算超過時は capture stride を2倍へ段階的に間引く
"""

from __future__ import annotations

import queue
import threading
from dataclasses import dataclass, field
from pathlib import Path

import torch

from griffith_logic.training.checkpoint_utils import torch_load

RING_BYTES_DEFAULT = 256 * 2**20   # RAMリングのバイト予算
CHUNK_STEPS_DEFAULT = 64           # ディスク退避のチャンク件数
MAX_DISK_MB_DEFAULT = 2048         # ディスク使用量の上限
KEYFRAME_INTERVAL_DEFAULT = 200    # keyframe間隔(step)
CHUNK_CACHE_SIZE = 4               # ディスクチャンクのLRUキャッシュ数


@dataclass
class LayerRecord:
    """1層分の記録。weightsは必須、他は capture level / 学習手法に応じてオプション。"""

    weights: torch.Tensor                 # (out_dim, n_entries|n_inputs) float32 CPU
    grad: torch.Tensor | None = None      # 同形 float16 — BP/BinaryBrain(full時)
    delta: torch.Tensor | None = None     # 同形 float16 — AP法の更新量(後続フェーズ)
    act_mean: torch.Tensor | None = None  # (out_dim,) float16 — probe平均活性
    indices: torch.Tensor | None = None   # (n_inputs, out_dim) int64 — keyframeのみ

    def nbytes(self) -> int:
        total = 0
        for t in (self.weights, self.grad, self.delta, self.act_mean, self.indices):
            if t is not None:
                total += t.numel() * t.element_size()
        return total


@dataclass
class StepSnapshot:
    """1ステップ分のスナップショット。methodは最初から3手法対応('bp'|'ap'|'bb')。"""

    step: int
    epoch: int
    loss: float
    acc: float
    ts: float
    method: str
    layers: dict[str, LayerRecord] = field(default_factory=dict)
    trace: dict | None = None       # backpropトレース(grad_flowフレームと同形、full時)
    opt_state: dict | None = None   # keyframeのみ: optimizer.state_dict()のCPUコピー
    rng: torch.Tensor | None = None  # keyframeのみ: torch.get_rng_state()

    @property
    def is_keyframe(self) -> bool:
        return self.opt_state is not None

    def nbytes(self) -> int:
        total = sum(rec.nbytes() for rec in self.layers.values())
        if self.rng is not None:
            total += self.rng.numel() * self.rng.element_size()
        if self.opt_state is not None:
            # Adam状態はweightsの約2倍(exp_avg/exp_avg_sq)を概算で計上
            total += 2 * sum(
                rec.weights.numel() * rec.weights.element_size()
                for rec in self.layers.values()
            )
        return total


@dataclass
class _ChunkMeta:
    start: int
    end: int
    path: Path
    steps: list[int]
    nbytes: int


# ディスク退避チャンクは本モジュール自身が `torch.save` した dataclass の list。
# 許可リストへ登録することで `weights_only=True`(pickleの任意コード実行なし)の
# ままロードできる(/ 監査B4)。
torch.serialization.add_safe_globals([LayerRecord, StepSnapshot])


class HistoryStore:
    """スレッド安全な履歴ストア(RAMリング+ディスクチャンク退避)。"""

    def __init__(
        self,
        sid: str,
        root_dir: Path | str,
        *,
        ring_bytes: int = RING_BYTES_DEFAULT,
        chunk_steps: int = CHUNK_STEPS_DEFAULT,
        max_disk_mb: int = MAX_DISK_MB_DEFAULT,
        keyframe_interval: int = KEYFRAME_INTERVAL_DEFAULT,
        stride: int = 1,
    ):
        self.sid = sid
        self.dir = Path(root_dir) / sid
        self.dir.mkdir(parents=True, exist_ok=True)
        self.ring_bytes = ring_bytes
        self.chunk_steps = chunk_steps
        self.max_disk_bytes = max_disk_mb * 2**20
        self.keyframe_interval = keyframe_interval
        self.stride = stride  # ディスク予算超過で段階的に2倍される

        self._lock = threading.RLock()
        self._ring: dict[int, StepSnapshot] = {}      # step昇順(3.7+のdictは挿入順)
        self._ring_nbytes = 0
        self._pending: list[StepSnapshot] = []        # リングから溢れ、チャンク化待ち
        self._chunks: list[_ChunkMeta] = []           # ディスク退避済み(start昇順)
        self._inflight: dict[str, list[StepSnapshot]] = {}  # 書込中チャンク(メモリ参照可)
        self._disk_nbytes = 0
        self._keyframes: list[int] = []
        self._chunk_cache: dict[str, list[StepSnapshot]] = {}  # LRU(挿入順で管理)

        self._write_queue: queue.Queue = queue.Queue()
        self._closed = False  # close()の冪等化ガード(二重closeでのjoin無限待ちを防ぐ)
        self._writer = threading.Thread(
            target=self._writer_loop, name=f'history-writer-{sid}', daemon=True
        )
        self._writer.start()

    # --- 書き込み ---

    def append(self, snap: StepSnapshot) -> None:
        """captured stepを追加する。stride非整合はスキップ(keyframeは常に保存)。"""
        if snap.step % self.stride != 0 and not snap.is_keyframe and snap.step != 0:
            return
        with self._lock:
            self._ring[snap.step] = snap
            self._ring_nbytes += snap.nbytes()
            if snap.is_keyframe and snap.step not in self._keyframes:
                self._keyframes.append(snap.step)
            self._evict_locked()

    def _evict_locked(self) -> None:
        """リングのバイト予算超過分を古い順にpendingへ移し、チャンク単位で退避する。"""
        while self._ring_nbytes > self.ring_bytes and len(self._ring) > 1:
            oldest_step = next(iter(self._ring))
            snap = self._ring.pop(oldest_step)
            self._ring_nbytes -= snap.nbytes()
            self._pending.append(snap)
        while len(self._pending) >= self.chunk_steps:
            chunk, self._pending = (
                self._pending[: self.chunk_steps],
                self._pending[self.chunk_steps:],
            )
            self._enqueue_chunk_locked(chunk)

    def _enqueue_chunk_locked(self, snaps: list[StepSnapshot]) -> None:
        est = sum(s.nbytes() for s in snaps)
        if self._disk_nbytes + est > self.max_disk_bytes:
            self._escalate_locked()
        path = self.dir / f'chunk_{snaps[0].step:08d}.pt'
        meta = _ChunkMeta(
            start=snaps[0].step,
            end=snaps[-1].step,
            path=path,
            steps=[s.step for s in snaps],
            nbytes=est,
        )
        self._chunks.append(meta)
        self._disk_nbytes += est
        self._inflight[str(path)] = snaps
        self._write_queue.put((path, snaps))

    def _escalate_locked(self) -> None:
        """ディスク予算超過: strideを2倍にし、最古チャンクを1/2に間引いてcompactする。"""
        self.stride *= 2
        if not self._chunks:
            return
        oldest = self._chunks[0]
        snaps = self._load_chunk(oldest)
        kept = [s for s in snaps if s.step % self.stride == 0 or s.is_keyframe]
        if len(kept) == len(snaps):
            kept = snaps[::2]  # stride整合で減らない場合は機械的に半減
        self._disk_nbytes -= oldest.nbytes
        oldest.steps = [s.step for s in kept]
        oldest.nbytes = sum(s.nbytes() for s in kept)
        oldest.end = kept[-1].step if kept else oldest.start
        self._disk_nbytes += oldest.nbytes
        self._chunk_cache.pop(str(oldest.path), None)
        self._inflight[str(oldest.path)] = kept
        self._write_queue.put((oldest.path, kept))

    def _writer_loop(self) -> None:
        while True:
            item = self._write_queue.get()
            if item is None:
                # sentinelもjoin対象のため必ず消化する。task_doneを怠ると
                # unfinished_tasksが残り、以後のjoin()が永久ブロックする
                self._write_queue.task_done()
                return
            path, snaps = item
            try:
                torch.save(snaps, path)
            finally:
                # 保存が例外でも_inflight解放とtask_doneは必ず行う(同型ハング予防)
                with self._lock:
                    self._inflight.pop(str(path), None)
                self._write_queue.task_done()

    def flush(self) -> None:
        """書き込みキューが掃けるまでブロックする(テスト・終了処理用)。"""
        self._write_queue.join()

    def close(self) -> None:
        """writerスレッドを畳む(冪等)。

        sentinelは投入済みチャンク書き込みより後にFIFOで処理されるため、
        flush()でブロックせずとも永続性は保たれる(=リクエストスレッドを止めない)。
        二重closeは無害(2回目以降は即return)。close()後にappend()してはならない。
        """
        if self._closed:
            return
        self._closed = True
        self._write_queue.put(None)

    # --- 読み出し ---

    def get(self, step: int) -> StepSnapshot | None:
        """step以下で最も近いcaptured snapshotを返す(なければNone)。"""
        with self._lock:
            # リング内(最も新しい区間)
            ring_steps = list(self._ring.keys())
            candidates: list[int] = [s for s in ring_steps if s <= step]
            if candidates:
                return self._ring[candidates[-1]]
            # pending(リングとチャンクの間)
            pending_hit = None
            for snap in self._pending:
                if snap.step <= step:
                    pending_hit = snap
            if pending_hit is not None:
                return pending_hit
            # ディスクチャンク(start降順で探す)
            for meta in reversed(self._chunks):
                if meta.start <= step:
                    snaps = self._load_chunk(meta)
                    hit = None
                    for snap in snaps:
                        if snap.step <= step:
                            hit = snap
                    if hit is not None:
                        return hit
            return None

    def _load_chunk(self, meta: _ChunkMeta) -> list[StepSnapshot]:
        key = str(meta.path)
        with self._lock:
            if key in self._inflight:
                return self._inflight[key]
            if key in self._chunk_cache:
                snaps = self._chunk_cache.pop(key)
                self._chunk_cache[key] = snaps  # LRU: 末尾へ移動
                return snaps
        snaps = torch_load(str(meta.path), map_location='cpu')
        with self._lock:
            self._chunk_cache[key] = snaps
            while len(self._chunk_cache) > CHUNK_CACHE_SIZE:
                self._chunk_cache.pop(next(iter(self._chunk_cache)))
        return snaps

    def index(self) -> dict:
        with self._lock:
            steps_min: int | None = None
            steps_max: int | None = None
            count = 0
            for meta in self._chunks:
                count += len(meta.steps)
                if meta.steps:
                    steps_min = meta.steps[0] if steps_min is None else steps_min
                    steps_max = meta.steps[-1]
            for snap in self._pending:
                count += 1
                steps_min = snap.step if steps_min is None else steps_min
                steps_max = snap.step
            for s in self._ring:
                count += 1
                steps_min = s if steps_min is None else steps_min
                steps_max = s
            return {
                'min_step': steps_min,
                'max_step': steps_max,
                'stride': self.stride,
                'keyframes': list(self._keyframes),
                'count': count,
                'ram_bytes': self._ring_nbytes,
                'disk_bytes': self._disk_nbytes,
            }

    def nearest_keyframe(self, step: int) -> int | None:
        with self._lock:
            below = [k for k in self._keyframes if k <= step]
            return below[-1] if below else None

    # --- branch(巻き戻し再学習) ---

    def truncate_after(self, step: int) -> None:
        """step より後の履歴を切り捨てる(切り捨て分はbranches/へ退避)。"""
        self.flush()
        with self._lock:
            branch_dir = self.dir / 'branches'
            # リング/pendingから除去
            removed: list[StepSnapshot] = []
            for s in [k for k in self._ring if k > step]:
                snap = self._ring.pop(s)
                self._ring_nbytes -= snap.nbytes()
                removed.append(snap)
            keep_pending = [s for s in self._pending if s.step <= step]
            removed.extend(s for s in self._pending if s.step > step)
            self._pending = keep_pending
            # チャンク: 完全に後方のものはbranches/へ移動、跨ぐものは切って再保存
            kept_chunks: list[_ChunkMeta] = []
            for meta in self._chunks:
                if meta.start > step:
                    branch_dir.mkdir(parents=True, exist_ok=True)
                    if meta.path.exists():
                        meta.path.rename(branch_dir / meta.path.name)
                    self._disk_nbytes -= meta.nbytes
                    self._chunk_cache.pop(str(meta.path), None)
                elif meta.end > step:
                    snaps = self._load_chunk(meta)
                    cut = [s for s in snaps if s.step <= step]
                    dropped = [s for s in snaps if s.step > step]
                    if dropped:
                        branch_dir.mkdir(parents=True, exist_ok=True)
                        torch.save(dropped, branch_dir / f'{meta.path.stem}_cut.pt')
                    self._disk_nbytes -= meta.nbytes
                    meta.steps = [s.step for s in cut]
                    meta.end = cut[-1].step if cut else meta.start
                    meta.nbytes = sum(s.nbytes() for s in cut)
                    self._disk_nbytes += meta.nbytes
                    self._chunk_cache.pop(str(meta.path), None)
                    torch.save(cut, meta.path)
                    kept_chunks.append(meta)
                else:
                    kept_chunks.append(meta)
            self._chunks = kept_chunks
            if removed:
                branch_dir.mkdir(parents=True, exist_ok=True)
                torch.save(removed, branch_dir / f'ram_after_{step:08d}.pt')
            self._keyframes = [k for k in self._keyframes if k <= step]
