"""事前学習MLP→検証済みIR変換の非同期セッション(ROADMAP-v5)。

`convert_pretrained_mlp`を threading.Thread で実行し、progressコールバック
から状態(phase/progress/metrics)をロック保護で更新する。REST(main.py)は
statusポーリングで進捗を読む(v1はライブWS配信なし。学習曲線はmetrics内の配列)。

状態遷移:
    preparing → pruning → quantizing → lowering → verifying → done
                                                            ↘ error(任意phaseで)
    cancel要求時は各phase境界で should_stop を見て 'cancelled' へ抜ける。

done時、生成IRの反映は backend 側(main.py)がセッション完了を検知して
STATE へ書き込む(ConversionSession は STATE を直接触らない=責務分離)。
"""

from __future__ import annotations

import threading
import time
import uuid

import numpy as np
import torch
from torch import nn

from griffith_logic.convert.pipeline import (
    ConvertPipelineSpec,
    convert_pretrained_mlp,
)
from griffith_logic.convert.teacher import resolve_teacher
from griffith_logic.ir.serialize import ir_to_dict

# 変換状態のphase。'preparing'/'done'/'error'/'cancelled' はセッション制御用、
# pruning/quantizing/lowering/verifying はパイプライン本体のphase。
CONVERT_PHASES = (
    'preparing', 'pruning', 'quantizing', 'lowering', 'verifying',
    'done', 'error', 'cancelled',
)


class ConversionCancelled(Exception):
    """cancel要求によりphase境界で中断したことを表す内部例外。"""


class ConversionSession:
    """1回の変換実行を表すセッション(threading.Threadでパイプラインを回す)。

    teacher解決・データセット解決(preparing)→ convert_pretrained_mlp 実行。
    progressコールバックからphase/progress/metricsを更新する(状態はロック保護)。

    Args:
        source: teacher指定文字列('ckpt:...' 等、resolve_teacherが解釈)。
        dataset: エディタのデータセット解決キー('mnist' 等)。
        spec: ConvertPipelineSpec(プリセット+overrides解決済み)。
        device: 枝刈り/QATの計算デバイス。
        teacher_resolver: resolve_teacher の注入口(テストで合成MLPに差し替える)。
            (source, device) -> Teacher(.model が nn.Module)。
        loaders: (train_loader, eval_loader) を注入する場合に指定(テスト用)。
            省略時は dataset から二値化MNISTローダを構築する。
        binarize_threshold: データセット二値化閾値(既定0.5)。
    """

    def __init__(
        self,
        source: str,
        dataset: str,
        spec: ConvertPipelineSpec,
        *,
        device: str = 'cpu',
        teacher_resolver=None,  # noqa: ANN001
        loaders=None,  # noqa: ANN001
        binarize_threshold: float = 0.5,
    ) -> None:
        self.convert_id = uuid.uuid4().hex[:12]
        self.source = source
        self.dataset = dataset
        self.spec = spec
        self.device = device
        self._teacher_resolver = teacher_resolver or resolve_teacher
        self._loaders = loaders
        self.binarize_threshold = binarize_threshold

        self._lock = threading.Lock()
        self.phase: str = 'preparing'
        self.progress: float = 0.0
        self.metrics: dict = {}
        self.error: str | None = None
        self.ir: dict | None = None  # done時に生成IR(JSON dict)。反映後はNoneへ
        self.ir_reflected = False    # 生成IRをSTATEへ反映済みか(has_ir判定に使う)
        self.should_stop = False
        self.started_at = time.time()
        self.ended_at: float | None = None

        self._thread = threading.Thread(target=self._run, daemon=True)

    # ------------------------------------------------------------------
    # 状態(ロック保護)
    # ------------------------------------------------------------------

    def _set(self, *, phase: str | None = None, progress: float | None = None,
             metrics: dict | None = None, error: str | None = None,
             ir: dict | None = None) -> None:
        with self._lock:
            if phase is not None:
                self.phase = phase
            if progress is not None:
                self.progress = float(progress)
            if metrics is not None:
                self.metrics = metrics
            if error is not None:
                self.error = error
            if ir is not None:
                self.ir = ir

    def status(self) -> dict:
        """ConvertStatus 相当のスナップショット(ロック保護でコピー)。"""
        with self._lock:
            return {
                'convert_id': self.convert_id,
                'phase': self.phase,
                'progress': round(self.progress, 4),
                'metrics': dict(self.metrics),
                'error': self.error,
                # 生成IRがある(反映待ち)or 反映済みなら has_ir=True
                'has_ir': self.ir is not None or self.ir_reflected,
                'source': self.source,
                'dataset': self.dataset,
            }

    @property
    def is_done(self) -> bool:
        with self._lock:
            return self.phase in ('done', 'error', 'cancelled')

    # ------------------------------------------------------------------
    # 制御
    # ------------------------------------------------------------------

    def start(self) -> None:
        self._thread.start()

    def cancel(self) -> None:
        """cancelフラグを立てる。パイプラインは次のphase境界で中断する。"""
        self.should_stop = True

    def join(self, timeout: float | None = None) -> None:
        self._thread.join(timeout)

    def _check_cancel(self) -> None:
        if self.should_stop:
            raise ConversionCancelled()

    # ------------------------------------------------------------------
    # 実行本体(スレッド)
    # ------------------------------------------------------------------

    def _progress_cb(self, phase: str, prog: float, metrics: dict) -> None:
        """convert_pretrained_mlp のprogressコールバック。

        phase境界(進捗更新のたび)でcancelを見る。metricsは最新スナップショット。
        """
        self._set(phase=phase, progress=prog, metrics=metrics)
        self._check_cancel()

    def _resolve_loaders(self):  # noqa: ANN201
        """データセットから (train_loader, eval_loader) を得る(二値化0/1)。"""
        if self._loaders is not None:
            return self._loaders
        return _binarized_dataset_loaders(
            self.dataset, self.spec.batch_size, self.binarize_threshold)

    def _run(self) -> None:
        try:
            # --- preparing: teacher + データセット解決 ---
            self._set(phase='preparing', progress=0.0)
            self._check_cancel()
            teacher = self._teacher_resolver(self.source, device=self.device)
            model = teacher.model
            if not isinstance(model, nn.Module):
                raise ValueError('teacher.model が nn.Module ではない')
            self._check_cancel()
            train_loader, eval_loader = self._resolve_loaders()
            self._check_cancel()

            # --- pipeline本体 ---
            ir, metrics = convert_pretrained_mlp(
                model, train_loader, eval_loader, self.spec,
                progress=self._progress_cb, device=self.device,
            )
            ir_dict = ir_to_dict(ir)
            self._set(phase='done', progress=1.0, metrics=metrics, ir=ir_dict)
        except ConversionCancelled:
            self._set(phase='cancelled', error='ユーザによりキャンセルされました')
        except Exception as e:  # noqa: BLE001 — 変換失敗を状態へ集約
            self._set(phase='error', error=f'{type(e).__name__}: {e}')
        finally:
            self.ended_at = time.time()


class ConversionManager:
    """変換セッションのレジストリ(convert_id→ConversionSession)。"""

    def __init__(self) -> None:
        self.sessions: dict[str, ConversionSession] = {}

    def create(self, session: ConversionSession) -> ConversionSession:
        self.sessions[session.convert_id] = session
        session.start()
        return session

    def get(self, convert_id: str) -> ConversionSession | None:
        return self.sessions.get(convert_id)


# ============================================================
# データセット解決(二値化0/1ローダ)
# ============================================================

def _binarized_dataset_loaders(dataset: str, batch_size: int,
                               binarize_threshold: float):  # noqa: ANN201
    """データセットキーから二値化(0/1)train/eval DataLoaderを構築する。

    convert_pretrained_mlp は 0/1 特徴量を要求する(binarize_threshold=0.5)。
    'mnist' は griffith_logic.datasets.vision.load_mnist を二値化で使う。
    未対応データセット(fashion 等、'結果待ち)は分かりやすいValueError。
    """
    if dataset == 'mnist':
        from griffith_logic.datasets.vision import load_mnist
        train_set, test_set = load_mnist(binarize_threshold=binarize_threshold)
        return (
            _flat_binary_loader(train_set, batch_size, shuffle=True),
            _flat_binary_loader(test_set, batch_size, shuffle=False),
        )
    raise ValueError(
        f'変換未対応のデータセット: {dataset}'
        "(現状 'mnist' のみ。fashion 等は'結果待ち)"
    )


def _flat_binary_loader(dataset, batch_size: int, *, shuffle: bool):  # noqa: ANN001,ANN201
    """(画像, ラベル)データセットを (N,in) 0/1 float32 + int64ラベルのローダにする。

    load_mnist の transform は (1,28,28) の0/1テンソルを返すため、flattenして
    (784,) 特徴量にする。pipeline側で0/1検証されるので二値性はここで壊さない。
    """
    xs = np.empty((len(dataset), 28 * 28), dtype=np.float32)
    ys = np.empty((len(dataset),), dtype=np.int64)
    for i in range(len(dataset)):
        img, label = dataset[i]
        xs[i] = img.reshape(-1).numpy().astype(np.float32)
        ys[i] = int(label)
    tds = torch.utils.data.TensorDataset(
        torch.from_numpy(xs), torch.from_numpy(ys))
    return torch.utils.data.DataLoader(
        tds, batch_size=batch_size, shuffle=shuffle)
