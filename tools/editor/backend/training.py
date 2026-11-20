"""訓練セッション管理。

訓練は threading.Thread で実行する(torchはGIL解放するのでthreadで十分)。
スレッド → WebSocket送信ループへの受け渡しは
  - metrics: queue.Queue(毎ステップ)
  - layer_summary: レイヤごとlatest-wins辞書(最大5Hzで更新)
の2経路。asyncio側(main.py)が吸い出してクライアントへ配信する。

step(n)による「nステップ進めて自動pause」と、HistoryStore への
毎ステップスナップショット保存(capture levelはモデル規模から自動決定)。
"""

import copy
import queue
import threading
import time
import uuid
from pathlib import Path

import torch
import torchvision

from .capture import build_snapshot, decide_capture_level
from .extract import layer_summary, lut_layers_with_ids
from .gradcap import GradCapture, build_grad_flow
from .history import HistoryStore

PROJECT_ROOT = Path(__file__).resolve().parents[3]
MNIST_ROOT = str(PROJECT_ROOT / 'data-mnist')
HISTORY_ROOT = PROJECT_ROOT / 'outputs' / 'editor_history'
RUNS_ROOT = PROJECT_ROOT / 'outputs' / 'editor_runs'

METRICS_QUEUE_SIZE = 10000     # 溢れたら古いフレームから捨てる
SUMMARY_INTERVAL_SEC = 0.2     # layer_summary計算の最小間隔(5Hz)
DETAIL_INTERVAL_SEC = 0.5      # detail系フレーム計算の既定間隔(2Hz)
EVAL_MNIST_SAMPLES = 2000      # MNIST評価はtest splitの固定サブセット
EVAL_BATCH_SIZE = 512          # hard eval時のバッチサイズ


class TrainingSession:
    """1回の訓練実行を表すセッション。"""

    def __init__(self, model: torch.nn.Module, *, dataset: str = 'mnist',
                 lr: float = 0.01, batch_size: int = 128, epochs: int = 10,
                 method: str = 'bp', loss: str | None = None,
                 optimizer_name: str = 'adam', dataset_params: dict | None = None,
                 logic_lr: float | None = None, ann_lr: float | None = None,
                 ir_snapshot: dict | None = None,
                 loader_factory=None, history_root: Path | str | None = None,
                 runs_root: Path | str | None = None,
                 eval_interval_steps: int | None = None,
                 eval_loader_factory=None, use_wandb: bool = False):
        # 既定パスは呼び出し時に解決する(テストのmonkeypatchを有効にするため)
        history_root = HISTORY_ROOT if history_root is None else history_root
        runs_root = RUNS_ROOT if runs_root is None else runs_root
        assert method in ('bp', 'ap', 'bb'), f'未対応の学習手法: {method}'
        self.sid = uuid.uuid4().hex[:12]
        self.model = model
        self.dataset = dataset
        self.lr = lr
        self.logic_lr = logic_lr
        self.ann_lr = ann_lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.method = method  # 'bp' | 'ap' | 'bb'(steppers.py)
        self.optimizer_name = optimizer_name

        # データセット解決+タスク(損失/メトリクス)決定。
        # loader_factory注入時(テスト)はデータセット解決を省略しclassification既定
        from griffith_logic.training.tasks import make_task

        from .datasets import resolve_dataset

        if loader_factory is not None:
            self.resolved_dataset = None
            self.task = make_task('classification', loss)
        else:
            try:
                self.resolved_dataset = resolve_dataset(
                    dataset, batch_size=batch_size, params=dataset_params,
                )
            except ValueError:
                # 取込データセットへフォールバック
                from . import dataset_import

                self.resolved_dataset = dataset_import.resolve_imported(
                    dataset, batch_size=batch_size, params=dataset_params,
                )
            self.task = make_task(self.resolved_dataset.kind, loss)
        if method == 'ap' and self.task.kind != 'classification':
            raise ValueError('AP法はclassificationタスク(GroupSum終端)専用です')

        self.state = 'starting'  # starting/running/paused/stopped/finished/error
        self.error: str | None = None
        self.step = 0
        self.epoch = 0
        self.total_steps: int | None = None
        self.last_loss: float | None = None
        self.last_acc: float | None = None

        # hard eval(離散化評価)。既定は1エポック毎+セッション終了時
        self.eval_interval_steps = eval_interval_steps
        self._eval_loader_factory = eval_loader_factory or self._default_eval_loader
        self._eval_loader = None       # 初回評価時に生成して再利用
        self.last_val_loss: float | None = None
        self.last_val_acc: float | None = None
        self.last_val_step: int | None = None
        self.last_confusion: list[list[int]] | None = None  # classificationのみ
        # 選択ギャップ(§4-1): soft(連続緩和)評価精度とhard(離散化)評価精度の乖離
        self.last_val_acc_soft: float | None = None
        self.last_selection_gap: float | None = None

        # runレジストリ用: metrics系列とIRスナップショット
        self.started_at = time.time()
        self.series: list[tuple[int, float, float]] = []  # (step, loss, acc)
        self.ir_snapshot = ir_snapshot
        self.runs_root = Path(runs_root)

        # wandb連携: opt-in。未認証・未インストールでもwandb_url=Noneで無害。
        # URLはrunレジストリへ保存され、RunsPanelのwandbリンクに使われる。
        self.use_wandb = use_wandb
        self.wandb_url: str | None = None
        self._wandb_run = None

        self.metrics_queue: queue.Queue = queue.Queue(maxsize=METRICS_QUEUE_SIZE)
        self.summary_lock = threading.Lock()
        self.latest_summaries: dict[str, dict] = {}  # layer名 → 最新layer_summary
        self.subscribed: set[str] = set()            # WS購読中のレイヤ名

        # detail購読(subscribe v2)。latest-wins辞書はsummary_lockで保護
        self.detail_subscription: dict = {
            'gate_distribution': set(),
            'grad_flow': False,
            'activations': False,
            'ap_delta': False,
        }
        self.detail_interval_sec = DETAIL_INTERVAL_SEC
        self.latest_details: dict[str, dict] = {}  # key(type:layer) → 最新フレーム
        self._last_acts: dict[str, torch.Tensor] = {}  # forward hookの最新活性平均
        self._grad_capture: GradCapture | None = None  # fullレベル時のみattach
        self._last_trace: dict | None = None  # 直近stepのgrad_flowトレース

        # capture level(full/weights/metrics)はモデル規模から自動決定
        self.lut_layers = lut_layers_with_ids(model)
        self.capture_level, stride = decide_capture_level(self.lut_layers)
        self.history = HistoryStore(
            self.sid, history_root, stride=max(stride, 1)
        )

        # データローダは注入可能(テストは合成データ、既定はMNIST)
        self._loader_factory = loader_factory or self._default_loader

        self._pause_event = threading.Event()  # set = 実行継続
        self._pause_event.set()
        self._stop_event = threading.Event()
        self._step_budget = -1  # -1=無制限 / >0 = 残りnステップで自動pause
        self._thread = threading.Thread(
            target=self._run, name=f'train-{self.sid}', daemon=True
        )

    # --- 制御 ---

    def start(self):
        self._thread.start()

    def pause(self):
        if self.state in ('running', 'starting'):
            self._pause_event.clear()

    def resume(self):
        if self.state == 'paused':
            self.state = 'running'
        self._step_budget = -1  # step実行の残量をリセット(無制限へ)
        self._pause_event.set()

    def step_n(self, n: int):
        """nステップだけ進めて自動pauseする(paused状態でのみ許可)。"""
        if self.state != 'paused':
            raise RuntimeError(f'step実行はpaused状態でのみ可能(現在: {self.state})')
        self._step_budget = n
        self._pause_event.set()

    def restore_to(self, step: int) -> dict:
        """履歴のstepへ巻き戻して再学習の起点にする(branchモード)。

        - weights: スナップショットから厳密復元
        - optimizer: 直近keyframe(≤step)のopt_stateから復元、なければ状態リセット
          (モメンタム消失。近似であることをレスポンスで明示する)
        - RNG: keyframeにあれば復元(データ順序の厳密再現は保証しない)
        - 復元先より後の履歴は truncate(branches/へ退避)
        """
        if self.state != 'paused':
            raise RuntimeError(f'restoreはpaused状態でのみ可能(現在: {self.state})')
        snap = self.history.get(step)
        if snap is None:
            raise ValueError(f'step {step} 以前の履歴が存在しない')

        with torch.no_grad():
            for name, layer in self.lut_layers:
                rec = snap.layers.get(name)
                if rec is not None:
                    layer.weights.copy_(rec.weights.to(layer.weights.dtype))
                    if rec.indices is not None:
                        layer.indices.copy_(rec.indices)

        # optimizer状態: 直近keyframeから復元(なければモメンタムをリセット)
        optimizer_restored = False
        kf_step = self.history.nearest_keyframe(snap.step)
        optimizer = getattr(self, 'optimizer', None)
        if optimizer is not None:
            if kf_step is not None:
                kf = self.history.get(kf_step)
                if kf is not None and kf.opt_state is not None:
                    try:
                        optimizer.load_state_dict(copy.deepcopy(kf.opt_state))
                        optimizer_restored = True
                    except (ValueError, KeyError, RuntimeError):
                        optimizer_restored = False
                if kf is not None and kf.rng is not None:
                    torch.set_rng_state(kf.rng.clone())
            if not optimizer_restored:
                optimizer.state.clear()  # in-placeでモメンタムをリセット

        self.history.truncate_after(snap.step)

        # カウンタ・差分基準・サマリを復元後の状態へ揃える
        self.step = snap.step
        self.epoch = snap.epoch
        self.last_loss = snap.loss
        self.last_acc = snap.acc
        if hasattr(self, '_prev_weights'):
            self._prev_weights = {
                name: layer.weights.detach().clone()
                for name, layer in self.lut_layers
            }
        self._update_summaries(self.lut_layers)

        return {
            'restored_step': snap.step,
            'state': self.state,
            'optimizer_restored': optimizer_restored,
            'history_truncated_from': snap.step + 1,
        }

    def stop(self):
        self._stop_event.set()
        self._pause_event.set()  # pause中でも抜けられるように

    @property
    def is_active(self) -> bool:
        return self.state in ('starting', 'running', 'paused')

    def status(self) -> dict:
        idx = self.history.index()
        return {
            'session_id': self.sid,
            'state': self.state,
            'started_at': self.started_at,  # 並列ラン一覧の並び順・表示用
            'step': self.step,
            'epoch': self.epoch,
            'total_steps': self.total_steps,
            # 数値の直近train loss(旧'loss'。現在'loss'は損失名のため改名)
            'train_loss': self.last_loss,
            'acc': self.last_acc,
            'error': self.error,
            # 追加キー(既存キーは不変=後方互換)
            'step_budget': self._step_budget,
            'method': self.method,
            # 追加キー: タスク/データセット情報
            'dataset': self.dataset,
            'task': self.task.kind,
            'loss': self.task.loss_name,
            # 追加キー: 直近hard eval(離散化)の結果
            'last_val_acc': self.last_val_acc,
            # §4-1 追加キー: 直近soft評価精度と選択ギャップ(soft - hard)
            'last_val_acc_soft': self.last_val_acc_soft,
            'last_selection_gap': self.last_selection_gap,
            'capture': {'level': self.capture_level, 'stride': self.history.stride},
            'history': {
                'min_step': idx['min_step'],
                'max_step': idx['max_step'],
                'count': idx['count'],
            },
        }

    # --- 訓練スレッド ---

    def _put_metrics(self, frame: dict):
        """metricsをキューへ。満杯なら最古を捨てて必ず入れる。"""
        try:
            self.metrics_queue.put_nowait(frame)
        except queue.Full:
            try:
                self.metrics_queue.get_nowait()
            except queue.Empty:
                pass
            self.metrics_queue.put_nowait(frame)

    def _update_summaries(self, lut_layers: list[tuple[str, object]]):
        """購読中レイヤのlayer_summaryをlatest-wins辞書へ書き込む。"""
        subscribed = set(self.subscribed)
        if not subscribed:
            return
        for name, layer in lut_layers:
            if name not in subscribed:
                continue
            summary = layer_summary(name, layer, self.step)
            with self.summary_lock:
                self.latest_summaries[name] = summary

    def _has_detail_subscription(self) -> bool:
        sub = self.detail_subscription
        return bool(
            sub['gate_distribution']
            or sub['activations']
            or sub['grad_flow']
            or sub['ap_delta']
        )

    def _update_details(self):
        """detail購読(gate_distribution/activations/grad_flow)のフレームを更新する。"""
        from .monitor import build_activations, build_gate_distribution

        sub = self.detail_subscription
        for name, layer in self.lut_layers:
            if name in sub['gate_distribution']:
                frame = build_gate_distribution(
                    name,
                    layer.weights,
                    layer.parameterization,
                    int(layer.n_inputs),
                    self.step,
                )
                with self.summary_lock:
                    self.latest_details[f'gate_distribution:{name}'] = frame
            if sub['activations'] and name in self._last_acts:
                frame = build_activations(name, self._last_acts[name], self.step)
                with self.summary_lock:
                    self.latest_details[f'activations:{name}'] = frame
        # grad_flowはモデル全体で1フレーム(直近captured stepのトレース)
        if sub['grad_flow'] and self._last_trace is not None:
            with self.summary_lock:
                self.latest_details['grad_flow'] = self._last_trace
        # AP法の更新量delta(.grad非経由、Stepperが捕捉)
        if sub['ap_delta']:
            from .monitor import build_ap_delta

            stepper = getattr(self, 'stepper', None)
            deltas = getattr(stepper, 'last_deltas_by_name', None)
            if deltas:
                for name, delta in deltas.items():
                    frame = build_ap_delta(name, delta, self.step)
                    with self.summary_lock:
                        self.latest_details[f'ap_delta:{name}'] = frame

    # --- wandb連携(opt-in) ---

    def _init_wandb(self):
        """opt-in時のみwandbを初期化しrun URLを控える。

        experiments/wandb_utils.init_wandb の規約に従う(project=WANDB_PROJECT、
        group='editor')。WANDB_API_KEY未設定・wandb未インストールなら
        init_wandbがFalseを返し、wandb_url=Noneのまま無害に継続する。
        """
        if not self.use_wandb:
            return
        try:
            from experiments.wandb_utils import init_wandb

            model_name = 'model'
            if self.ir_snapshot:
                model_name = str(
                    self.ir_snapshot.get('metadata', {}).get('name', 'model')
                )
            config = {
                'source': 'editor',
                'dataset': self.dataset,
                'method': self.method,
                'loss': self.task.loss_name,
                'optimizer': self.optimizer_name,
                'lr': self.lr,
                'logic_lr': self.logic_lr,
                'ann_lr': self.ann_lr,
                'batch_size': self.batch_size,
                'epochs': self.epochs,
            }
            enabled = init_wandb(
                f'editor_{model_name}', config, model=self.model, device='cpu',
                tags=['editor'], group='editor', job_type='train',
            )
            if enabled:
                import wandb

                self._wandb_run = wandb.run
                self.wandb_url = wandb.run.url
        except Exception:  # noqa: BLE001 — wandb連携はベストエフォート
            self._wandb_run = None
            self.wandb_url = None

    def _log_wandb(self, frame: dict):
        """metricsフレームをwandbへログする(wandb無効時は何もしない)。"""
        if self._wandb_run is None:
            return
        try:
            from experiments.wandb_utils import log_metrics

            payload = {'train/loss': frame['loss'], 'train/acc': frame['acc']}
            for key in ('val_loss', 'val_acc', 'val_acc_soft', 'selection_gap'):
                if frame.get(key) is not None:
                    payload[f'eval/{key}'] = frame[key]
            log_metrics(payload, step=frame['step'])
        except Exception:  # noqa: BLE001 — ログ失敗は学習を止めない
            pass

    def _finish_wandb(self):
        """wandb runを明示終了する(wandb無効時は何もしない)。"""
        if self._wandb_run is None:
            return
        try:
            from experiments.wandb_utils import finish_wandb

            finish_wandb()
        except Exception:  # noqa: BLE001
            pass
        finally:
            self._wandb_run = None

    def _run(self):
        try:
            self._init_wandb()
            self._train_loop()
            # セッション終了時のhard eval。state遷移前に実施し、
            # WS送信ループがsession_endを送る前にvalフレームをキューへ積む
            self._final_eval()
            self.state = 'stopped' if self._stop_event.is_set() else 'finished'
        except Exception as e:  # noqa: BLE001 — 状態遷移でクライアントへ伝える
            self.error = f'{type(e).__name__}: {e}'
            self.state = 'error'
            self._put_metrics({'type': 'error', 'message': self.error})
        finally:
            self._save_run_record()
            self._finish_wandb()

    def _save_run_record(self):
        """runレジストリへ永続化。失敗しても学習スレッドは殺さない。"""
        try:
            from .runs import build_record, save_run

            save_run(build_record(self, self.ir_snapshot), self.runs_root)
        except Exception:  # noqa: BLE001 — レジストリ書き込みはベストエフォート
            pass

    def _default_loader(self):
        """解決済みデータセットのローダ(テストは合成データのloader_factoryを注入)。"""
        if self.resolved_dataset is not None:
            return self.resolved_dataset.loader_factory()
        # フォールバック(通常到達しない): 従来のMNIST直接ロード
        transforms = torchvision.transforms.ToTensor()
        train_set = torchvision.datasets.MNIST(
            MNIST_ROOT, train=True, download=True, transform=transforms
        )
        return torch.utils.data.DataLoader(
            train_set, batch_size=self.batch_size, shuffle=True, drop_last=True
        )

    # --- hard eval ---

    def _default_eval_loader(self):
        """hard評価用ローダ。mnist=test split固定2000件 / logic系=全列挙。

        loader_factory注入時(テスト等、resolved_dataset=None)はNone=評価なし。
        """
        if self.resolved_dataset is None:
            return None
        if self.dataset == 'mnist':
            transforms = torchvision.transforms.ToTensor()
            test_set = torchvision.datasets.MNIST(
                MNIST_ROOT, train=False, download=True, transform=transforms
            )
            subset = torch.utils.data.Subset(test_set, range(EVAL_MNIST_SAMPLES))
            return torch.utils.data.DataLoader(subset, batch_size=EVAL_BATCH_SIZE)
        # logic:<key> / truth_table: 全列挙(repeat水増し済みでも各パターン等重み
        # のため指標は真の全列挙と一致する)
        return self.resolved_dataset.loader_factory()

    def _eval_due(self, batch_i: int, n_batches: int | None) -> bool:
        """このstepで評価すべきか。既定(eval_interval_steps=None)は1エポック毎。"""
        if self.eval_interval_steps is not None:
            return self.step % self.eval_interval_steps == 0
        return n_batches is not None and batch_i == n_batches - 1

    def _eval_pass(
        self, loader, *, want_confusion: bool,
    ) -> tuple[float, float, torch.Tensor | None] | None:
        """1パス評価(model.training状態は呼び出し側が設定)。

        返り値 (loss平均, acc平均, 混同行列 or None)。空loaderはNone。
        """
        total_loss = 0.0
        total_acc = 0.0
        total = 0
        confusion: torch.Tensor | None = None
        with torch.no_grad():
            for x, y in loader:
                x = x.to(torch.float32)
                out = self.model(x)
                n = int(x.shape[0])
                total_loss += float(self.task.loss_fn(out, y).item()) * n
                total_acc += float(self.task.metric_fn(out, y)) * n
                total += n
                if want_confusion and self.task.kind == 'classification':
                    n_classes = int(out.shape[-1])
                    if confusion is None:
                        confusion = torch.zeros(
                            n_classes, n_classes, dtype=torch.long
                        )
                    idx = y.to(torch.long) * n_classes + out.argmax(dim=-1)
                    confusion += torch.bincount(
                        idx, minlength=n_classes * n_classes
                    ).reshape(n_classes, n_classes)
        if total == 0:
            return None
        return total_loss / total, total_acc / total, confusion

    def _run_eval(self) -> dict | None:
        """hard eval(離散化)とsoft eval(連続緩和)を計測する(§4-1)。

        hard(model.eval())=離散化ギャップ込みの精度、soft(model.train())=
        連続緩和のままの精度。両者の乖離 selection_gap = soft_acc - hard_acc が
        「hard化で失う精度」=このドメイン最大のデバッグ課題(§4-1)。
        classificationタスクでは混同行列も併せて更新する。
        評価不能(データ取得失敗等)は学習を止めずNoneを返す。
        """
        if self._eval_loader is None:
            try:
                self._eval_loader = self._eval_loader_factory()
            except Exception:  # noqa: BLE001 — 評価失敗は学習を止めない
                self._eval_loader = None
        loader = self._eval_loader
        if loader is None:
            return None

        model = self.model
        was_training = model.training
        # forward hook(活性平均・GradCapture)は評価バッチでも発火するため、
        # 学習step由来の捕捉状態を退避して評価後に戻す
        # (混在するとbuild_grad_flowでバッチ寸法が食い違う)
        saved_acts = dict(self._last_acts)
        gc = self._grad_capture
        saved_gc = (
            (dict(gc.x_in), dict(gc.grad_out), dict(gc.grad_x))
            if gc is not None else None
        )
        try:
            model.eval()  # LutLayerは推論時に離散化される(離散化ギャップ込み)
            hard = self._eval_pass(loader, want_confusion=True)
            model.train()  # 連続緩和のまま(=soft精度)
            soft = self._eval_pass(loader, want_confusion=False)
        finally:
            if was_training:
                model.train()
            else:
                model.eval()
            self._last_acts.clear()
            self._last_acts.update(saved_acts)
            if gc is not None and saved_gc is not None:
                gc.x_in.clear()
                gc.x_in.update(saved_gc[0])
                gc.grad_out.clear()
                gc.grad_out.update(saved_gc[1])
                gc.grad_x.clear()
                gc.grad_x.update(saved_gc[2])

        if hard is None:
            return None
        hard_loss, hard_acc, confusion = hard
        self.last_val_loss = round(hard_loss, 6)
        self.last_val_acc = round(hard_acc, 6)
        self.last_val_step = self.step
        if confusion is not None:
            self.last_confusion = [[int(v) for v in row] for row in confusion]
        result = {'val_loss': self.last_val_loss, 'val_acc': self.last_val_acc}
        if soft is not None:
            _, soft_acc, _ = soft
            self.last_val_acc_soft = round(soft_acc, 6)
            self.last_selection_gap = round(soft_acc - hard_acc, 6)
            result['val_acc_soft'] = self.last_val_acc_soft
            result['selection_gap'] = self.last_selection_gap
        return result

    def uncertain_gates(self, top_n: int = 20) -> list[dict]:
        """確信度が低い(=ゲート分布エントロピー最大)ニューロンTop-N(§4-1)。

        全LutLayer横断でニューロン毎エントロピーを集め、降順にtop_n件返す。
        各要素: layer_id / neuron_index / entropy / gate_id。
        """
        from .extract import get_gate_ids, layer_entropy

        scored: list[dict] = []
        for name, layer in self.lut_layers:
            ent = layer_entropy(layer)
            if ent is None:
                continue
            gate_ids = get_gate_ids(layer)
            ent_list = ent.tolist()
            for i, e in enumerate(ent_list):
                scored.append({
                    'layer_id': name,
                    'neuron_index': int(i),
                    'entropy': round(float(e), 4),
                    'gate_id': int(gate_ids[i]) if i < len(gate_ids) else 0,
                })
        scored.sort(key=lambda r: r['entropy'], reverse=True)
        return scored[: max(top_n, 0)]

    def _final_eval(self):
        """セッション終了時のhard eval。直近stepで評価済みならスキップ。"""
        if self.step == 0 or self.last_val_step == self.step:
            return
        try:
            val = self._run_eval()
        except Exception:  # noqa: BLE001 — 終了時評価はベストエフォート
            return
        if val:
            # 最終stepのmetricsフレームを複製してvalキーを載せる
            # (フロントは同一stepのフレームをin-placeマージする)
            frame = {
                'type': 'metrics',
                'step': self.step,
                'epoch': self.epoch,
                'total_steps': self.total_steps,
                'loss': round(self.last_loss or 0.0, 6),
                'acc': round(self.last_acc or 0.0, 6),
                'weight_delta': {},
                **val,
            }
            self._put_metrics(frame)
            self._log_wandb(frame)  # 終了時evalもwandbへ

    def _capture_step(self, epoch: int):
        """更新後のスナップショットを履歴ストアへ積む。"""
        keyframe = self.step % self.history.keyframe_interval == 0
        grads = None
        deltas = None
        trace = None
        if self.capture_level == 'full':
            grads = {
                name: layer.weights.grad
                for name, layer in self.lut_layers
                if layer.weights.grad is not None
            }
            # AP法は.gradが無いため、Stepperが捕捉した更新量deltaを保存する
            stepper = getattr(self, 'stepper', None)
            if stepper is not None and hasattr(stepper, 'last_deltas_by_name'):
                deltas = stepper.last_deltas_by_name
            # 誤差逆伝播トレース(層分解済みgrad_flow)。過去stepの逆伝播リプレイも
            # 可能にするためスナップショットへ同形で保存する
            if self._grad_capture is not None and self.method != 'ap':
                trace = build_grad_flow(
                    self.step, self.last_loss or 0.0,
                    self.lut_layers, self._grad_capture,
                )
                self._last_trace = trace
        optimizer = getattr(self, 'optimizer', None)
        # 平均活性(forward hookの最新値)もスナップショットへ載せる
        # (replay時のactivationsパネル用。fullレベルのみ)
        act_means = (
            dict(self._last_acts)
            if self.capture_level == 'full' and self._last_acts else None
        )
        snap = build_snapshot(
            step=self.step,
            epoch=epoch,
            loss=self.last_loss or 0.0,
            acc=self.last_acc or 0.0,
            method=self.method,
            lut_layers=self.lut_layers,
            level=self.capture_level,
            grads=grads,
            deltas=deltas,
            act_means=act_means,
            trace=trace,
            keyframe=keyframe,
            opt_state=(
                copy.deepcopy(optimizer.state_dict())
                if keyframe and optimizer is not None else None
            ),
        )
        if snap is not None:
            self.history.append(snap)

    def _train_loop(self):
        device = 'cpu'  # v0.1: スレッド実行の安定性優先でCPU固定
        model = self.model.to(device)
        model.train()

        loader = self._loader_factory()
        self.total_steps = (
            len(loader) * self.epochs if hasattr(loader, '__len__') else None
        )

        # 学習1ステップはStepper抽象(BP/AP/BB)。TaskSpec/optimizer注入
        from .steppers import make_stepper

        lut_layers = self.lut_layers
        stepper = make_stepper(
            self.method, model, self.lr, lut_layers,
            task=self.task, optimizer_name=self.optimizer_name,
            logic_lr=self.logic_lr, ann_lr=self.ann_lr,
        )
        self.stepper = stepper
        self.optimizer = stepper.optimizer  # branch復元から参照(APはNone)
        # weight_delta計算用の前回重み(branch復元時に外から再初期化するためインスタンス保持)
        self._prev_weights = {
            name: layer.weights.detach().clone() for name, layer in lut_layers
        }
        prev_weights = self._prev_weights

        # 中間活性フック(fullレベルのみ)。各層出力のバッチ平均を保持する
        hooks = []
        if self.capture_level == 'full':
            def make_act_hook(layer_name: str):
                def hook(_module, _inp, out):
                    with torch.no_grad():
                        self._last_acts[layer_name] = out.detach().mean(dim=0).cpu()
                return hook

            for name, layer in lut_layers:
                hooks.append(layer.register_forward_hook(make_act_hook(name)))

            # 勾配フロー捕捉(誤差逆伝播リプレイ用)
            self._grad_capture = GradCapture()
            self._grad_capture.attach(lut_layers)

        try:
            self._train_epochs(loader, stepper, lut_layers)
        finally:
            for h in hooks:
                h.remove()
            if self._grad_capture is not None:
                self._grad_capture.detach()

    def _train_epochs(self, loader, stepper, lut_layers):
        prev_weights = self._prev_weights
        self.state = 'running'
        last_summary_t = 0.0
        last_detail_t = 0.0

        for epoch in range(self.epochs):
            self.epoch = epoch
            n_batches = len(loader) if hasattr(loader, '__len__') else None
            for batch_i, (x, y) in enumerate(loader):
                if self._stop_event.is_set():
                    return
                if not self._pause_event.is_set():
                    self.state = 'paused'
                    while not self._pause_event.wait(timeout=0.2):
                        if self._stop_event.is_set():
                            return
                    if self._stop_event.is_set():
                        return
                    self.state = 'running'

                x = x.to(torch.float32)  # device=cpu固定(v0.1)

                # 学習1ステップ(BP: experiments/train.pyと同一手順 / AP / BB)
                loss_value, acc = stepper.step(x, y)

                self.step += 1

                # weight_delta = 前回送信時からの ||ΔW||
                # (branch復元でself._prev_weightsが差し替わるため毎回参照し直す)
                prev_weights = self._prev_weights
                weight_delta = {}
                with torch.no_grad():
                    for name, layer in lut_layers:
                        w = layer.weights.detach()
                        weight_delta[name] = round(
                            torch.norm(w - prev_weights[name]).item(), 6
                        )
                        prev_weights[name].copy_(w)

                self.last_loss = loss_value
                self.last_acc = acc
                # binary_vectorタスクの追加メトリクス(exact_match等)も載せる。
                # 未知キーはフロントが無視するため後方互換
                extra = {
                    k: round(float(v), 6)
                    for k, v in getattr(stepper, 'last_metrics', {}).items()
                    if k != 'acc'
                }
                # hard eval: 既定は1エポック毎(最終バッチ後)。
                # 評価直後のmetricsフレームのみval_loss/val_accキーが載る
                val_extra = (
                    (self._run_eval() or {})
                    if self._eval_due(batch_i, n_batches) else {}
                )
                frame = {
                    'type': 'metrics',
                    'step': self.step,
                    'epoch': epoch,
                    'total_steps': self.total_steps,
                    'loss': round(self.last_loss, 6),
                    'acc': round(acc, 6),
                    'weight_delta': weight_delta,
                    **extra,
                    **val_extra,
                }
                self._put_metrics(frame)
                self._log_wandb(frame)  # opt-in時のみwandbへ
                # runレジストリ用の系列(保存時に間引き)
                self.series.append((self.step, self.last_loss, acc))

                # スナップショットを履歴ストアへ(タイムマシン)
                self._capture_step(epoch)

                # step実行: 残量を消費し、0になったら自動pause
                if self._step_budget > 0:
                    self._step_budget -= 1
                    if self._step_budget == 0:
                        self._step_budget = -1
                        self._pause_event.clear()
                        # 停止直後の表示が最新になるようサマリ/detailを即時更新
                        self._update_summaries(lut_layers)
                        self._update_details()
                        last_summary_t = time.time()
                        last_detail_t = last_summary_t

                # layer_summaryは最大5Hz
                now = time.time()
                if now - last_summary_t >= SUMMARY_INTERVAL_SEC:
                    self._update_summaries(lut_layers)
                    last_summary_t = now

                # detail系(gate_distribution/activations)は既定2Hz
                if (
                    self._has_detail_subscription()
                    and now - last_detail_t >= self.detail_interval_sec
                ):
                    self._update_details()
                    last_detail_t = now


class SessionManager:
    """訓練セッションの生成・参照。

    複数セッションの同時並列実行を許可する(各セッションは独立したモデルの
    コピーを学習するため互いに干渉しない、train_startでdeepcopy)。終了済み
    セッションは一覧表示のため直近数件を残し、上限超過分だけ履歴を畳んで刈る。
    """

    MAX_TERMINATED_KEPT = 12  # 一覧に残す終了済みセッション数の上限(超過分を刈る)

    def __init__(self):
        self.sessions: dict[str, TrainingSession] = {}
        self._lock = threading.Lock()  # prune→createを直列化(同時createへの保険)

    def active_session(self) -> TrainingSession | None:
        for s in self.sessions.values():
            if s.is_active:
                return s
        return None

    def _prune_terminated(self) -> None:
        """終了済み(非アクティブ かつ スレッド停止)セッションのうち、上限を超えた
        古い分だけ辞書から除去し履歴writerを一度だけ畳む(冪等close)。

        直近MAX_TERMINATED_KEPT件は並列ラン一覧の表示のため残す。稼働中・終了処理中
        (is_active、または_final_eval等でスレッド稼働中)には触れない=履歴の
        読取/書込整合とタイムマシン再生を壊さない。dictは挿入順(=生成順、古い順)。
        """
        terminated = [
            sid for sid, s in self.sessions.items()
            if not s.is_active and not s._thread.is_alive()
        ]
        excess = len(terminated) - self.MAX_TERMINATED_KEPT
        for sid in terminated[: max(excess, 0)]:  # 古い順に超過分だけ畳む
            self.sessions[sid].history.close()
            del self.sessions[sid]

    def create(self, model, **kwargs) -> TrainingSession:
        with self._lock:
            self._prune_terminated()
            session = TrainingSession(model, **kwargs)
            self.sessions[session.sid] = session
            session.start()
            return session

    def get(self, sid: str) -> TrainingSession | None:
        return self.sessions.get(sid)

    def list_sessions(self) -> list[dict]:
        """全ライブセッションの状態一覧(新しい順)。並列ラン一覧UI用。"""
        return sorted(
            (s.status() for s in self.sessions.values()),
            key=lambda st: st.get('started_at') or 0,
            reverse=True,
        )
