"""エディタAPI契約のpydanticスキーマ(段階1)。

WSフレームと主要RESTレスポンスの形をpydanticモデルとして固定する。
役割は3つ:
  1. 契約の文書化(dict構築コードに散らばった形の単一情報源)
  2. TS型自動生成の入力(scripts/gen_editor_types.py → frontend/src/api/generated/)
  3. スキーマと実装のドリフト検出(tests/test_editor_schemas.py が実レスポンスをvalidate)

正とする定義:
  - フロントの手書き型 src/api/rest.ts / src/store/trainingStore.ts
  - 実装のdict構築 main.py / training.py / monitor.py / gradcap.py / runs.py

注意:
  - 実装のdict構築コードは変更しない(このモジュールは検証・生成専用)
  - torch非依存(型生成をCIの軽量環境で回せるようpydanticのみでimport可能に保つ)
  - extra='forbid' で「実装が知らないキーを足した」ドリフトも検出する
    (MetricsFrameのみstepper追加メトリクスが動的に載るためextra許可)
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

# N系列で追加したAPIのモデルを各Routerファイルからimportして登録する
# (モデル定義の重複コピー禁止。実装は各router内のBaseModelのまま、ここではALL_MODELSに
# 積むだけ)。evaluate.py/nas.py等はtorch依存のimportチェーンを持つため、本モジュールの
# 「torch非依存」原則は当初の統計/履歴系モデルに限った話であり、CI(frontend.yml e2eジョブ)
# は `uv sync --extra cpu --extra editor` 済みの環境でgen-typesを回すため実害はない。
from griffith_logic.training.config import TrainConfig  # noqa: E402
from tools.editor.backend.dataset_import import (  # noqa: E402
    DatasetConfigUpdate,
    EncodingPreviewRequest,
    EncodingPreviewResponse,
)
from tools.editor.backend.dataset_samples import (  # noqa: E402
    DatasetSampleItem,
    DatasetSamplesResponse,
)
from tools.editor.backend.evaluate import (  # noqa: E402
    EvalItem,
    EvalSource,
    EvaluateRequest,
    EvaluateResponse,
)
from tools.editor.backend.nas import (  # noqa: E402
    NasIrResponse,
    NasStartRequest,
    NasStartResponse,
    NasStatusResponse,
    NasTrialRow,
    NasTrialsResponse,
)
from tools.editor.backend.network_transfer import (  # noqa: E402
    SkippedItem,
    TransferRequest,
    TransferResponse,
)
from tools.editor.backend.project import ProjectSchema  # noqa: E402
from tools.editor.backend.projects_router import (  # noqa: E402
    CreateProjectRequest,
    UpdateProjectRequest,
)
from tools.editor.backend.stats import StatsResponse  # noqa: E402


class _StrictModel(BaseModel):
    """未知キーを拒否する基底(スキーマと実装のドリフト検出用)。"""

    model_config = ConfigDict(extra='forbid')


class U8Envelope(_StrictModel):
    """u8量子化テンソルエンベロープ(WS/REST共通、monitor.encode_u8 と対応)。"""

    encoding: Literal['u8']
    shape: list[int]
    scale: float
    data: str  # base64


# --- WSフレーム(/ws/train/{sid} サーバ→クライアント) ---

class MetricsFrame(BaseModel):
    """毎stepのmetricsフレーム(training.py _put_metrics)。

    val_loss/val_acc はhard eval直後のstepのみ載るスパースなキー。
    stepperの追加メトリクス(exact_match等)が動的に載るためextraを許可する
    (未知キーはフロントが無視する後方互換契約)。
    """

    model_config = ConfigDict(extra='allow')

    type: Literal['metrics']
    step: int
    epoch: int
    total_steps: int | None = None
    loss: float
    acc: float
    weight_delta: dict[str, float]
    val_loss: float | None = None
    val_acc: float | None = None
    exact_match: float | None = None  # binary_vectorタスク
    # §4-1: hard evalと同stepにのみ載る選択ギャップ関連キー(スパース)
    val_acc_soft: float | None = None
    selection_gap: float | None = None


class LayerSummaryFrame(_StrictModel):
    """レイヤ要約フレーム(extract.layer_summary、最大5Hz差分送信)。"""

    type: Literal['layer_summary']
    layer: str
    step: int
    n_inputs: int
    gate_ids: list[int]
    confidence: list[float]
    # §4-1: レイヤ別の迷い統計(ニューロン毎の正規化迷い[0,1]の平均・最大)。
    # softmax=カテゴリカルエントロピー/ln(K)、independent系=二値エントロピー/ln(2)。
    # warp等で確率解釈できない層はNone。
    entropy_mean: float | None
    entropy_max: float | None


class GateDistributionFrame(_StrictModel):
    """ゲート確率分布フレーム(monitor.build_gate_distribution)。

    dense: probs(u8エンベロープ)あり / aggregate: hist(argmaxヒスト)あり。
    """

    type: Literal['gate_distribution']
    layer: str
    step: int
    n_inputs: int
    parameterization: str
    mode: Literal['dense', 'aggregate']
    probs: U8Envelope | None = None
    hist: list[int] | None = None
    entropy_mean: float | None


class ActivationsFrame(_StrictModel):
    """中間活性フレーム(monitor.build_activations、fullキャプチャ時)。"""

    type: Literal['activations']
    layer: str
    step: int
    mean: U8Envelope
    saturation: float


class GradFlowLayerTrace(_StrictModel):
    """grad_flowの1層分トレース(gradcap.build_grad_flow)。order昇順=損失側→入力側。"""

    layer: str
    order: int
    grad_out: U8Envelope | None = None
    grad_w: U8Envelope | None = None
    edge_grad: U8Envelope | None = None


class GradFlowFrame(_StrictModel):
    """誤差逆伝播トレースフレーム(gradcap.build_grad_flow)。

    履歴スナップショットのtrace(HistoryStepView.trace)も同形。
    """

    type: Literal['grad_flow']
    step: int
    loss: float
    layers: list[GradFlowLayerTrace]
    layer_grad_norm: dict[str, float]


class ApDeltaFrame(_StrictModel):
    """AP法更新量フレーム(monitor.build_ap_delta)。"""

    type: Literal['ap_delta']
    layer: str
    step: int
    delta_norm: float
    delta: U8Envelope | None = None


class CaptureConfigFrame(_StrictModel):
    """WS接続直後に送るcapture設定(monitor.build_capture_config)。"""

    type: Literal['capture_config']
    level: str  # 'full' | 'weights' | 'metrics'
    stride: int
    method: str  # 'bp' | 'ap' | 'bb'
    snapshot_bytes: int
    detail_max_elems: int


class CaptureInfo(_StrictModel):
    """capture level/stride のペア(TrainStatus.capture / HistoryIndex.capture)。"""

    level: str
    stride: int


class HistoryIndex(_StrictModel):
    """履歴インデックス(GET /api/train/{sid}/history、history.HistoryStore.index)。"""

    min_step: int | None
    max_step: int | None
    stride: int
    keyframes: list[int]
    count: int
    ram_bytes: int | None = None
    disk_bytes: int | None = None
    capture: CaptureInfo | None = None


class HistoryIndexFrame(HistoryIndex):
    """履歴インデックスのWSプッシュフレーム(RESTのHistoryIndexと同形+type)。"""

    type: Literal['history_index']


class HistoryCounts(_StrictModel):
    """TrainStatus.history のカウンタ(min/max_step, count)。"""

    min_step: int | None
    max_step: int | None
    count: int


class TrainStatus(_StrictModel):
    """訓練セッション状態(GET /api/train/{sid}/status 等、training.py status)。

    lossは現在タスクの損失名(例 'ce'/'bce')。数値の直近train lossは
    train_loss(旧'loss'。dictキー重複で損失名に上書きされていたのを分離)。
    現在の追加キーは旧バックエンド互換のためoptional。
    """

    session_id: str
    state: str  # starting/running/paused/stopped/finished/error
    started_at: float | None = None  # 並列ラン一覧の並び順・表示用
    step: int
    epoch: int
    total_steps: int | None = None
    loss: str | None
    train_loss: float | None = None
    acc: float | None
    error: str | None
    step_budget: int | None = None
    method: str | None = None
    capture: CaptureInfo | None = None
    history: HistoryCounts | None = None
    dataset: str | None = None
    task: str | None = None
    last_val_acc: float | None = None
    last_val_acc_soft: float | None = None  # §4-1
    last_selection_gap: float | None = None  # §4-1


class SessionEndFrame(TrainStatus):
    """セッション終了フレーム(main.py ws_train。TrainStatusと同形+type)。"""

    type: Literal['session_end']


class StepMetrics(_StrictModel):
    """HistoryStepView.metrics(スナップショットのstep/epoch/loss/acc)。"""

    step: int
    epoch: int
    loss: float
    acc: float


class HistoryLayerView(_StrictModel):
    """履歴ビューの1層分(main.py train_history_step。fieldsクエリで疎に載る)。"""

    gate_ids: list[int] | None = None
    confidence: list[float] | None = None
    n_inputs: int | None = None
    gate_distribution: U8Envelope | None = None
    grad_norm: U8Envelope | None = None
    act_mean: U8Envelope | None = None


class HistoryStepView(_StrictModel):
    """任意stepのスナップショットビュー(GET /api/train/{sid}/history/{step})。"""

    resolved_step: int
    method: str
    metrics: StepMetrics
    layers: dict[str, HistoryLayerView]
    trace: GradFlowFrame | None = None  # fields=trace指定時のみ(fullキャプチャ)


class DatasetInfo(_StrictModel):
    """データセットカタログの1件(GET /api/datasets、datasets.list_datasets)。"""

    name: str
    label: str
    in_width: int
    out_width: int
    kind: str  # 'classification' | 'binary_vector'
    input_spec: dict[str, Any] | None = None
    output_spec: dict[str, Any] | None = None
    # 取込データセットならTrue。組込カタログは省略(False)
    imported: bool = False


class NodeCapabilities(_StrictModel):
    """IR node meta.capabilities の標準形(v0.3相当)。"""

    python_executable: bool
    trainable: bool
    ir_buildable: bool
    gate_expandable: bool
    hdl_exportable: bool


class PythonModelImportResponse(_StrictModel):
    """POST /api/models/import/python のレスポンス。"""

    ir: dict
    input_width: int
    model_ref: dict | None = None
    spec: dict


class CapabilityReason(_StrictModel):
    """capabilityの可否とUI表示用理由。"""

    capability: str
    enabled: bool
    reason: str | None = None


class HierarchyNode(_StrictModel):
    """Python由来モデルの階層表示ノード(L3/L2/L1/L0)。"""

    id: str
    label: str
    level: str
    kind: str
    parent: str | None = None
    detail: str | None = None
    shape: dict[str, Any] | None = None
    capabilities: dict[str, bool] | None = None
    lowering: dict[str, Any] | None = None
    paper_note: str | None = None


class HierarchyResponse(_StrictModel):
    """GET /api/models/current/hierarchy のレスポンス。"""

    levels: list[str]
    mode: str
    nodes: list[HierarchyNode]
    paper_ref: dict[str, Any] | None = None
    dataset_presets: dict[str, dict[str, Any]] | None = None


class LoweringView(_StrictModel):
    """GET /api/models/current/lower のレスポンス。"""

    node_id: str
    level: str
    title: str
    description: str | None = None
    capabilities: dict[str, bool]
    shape: dict[str, Any] | None = None
    children: list[HierarchyNode]
    lowering: dict[str, Any] | None = None
    virtual_ir_node: dict[str, Any] | None = None
    summary: dict[str, Any] | None = None
    actions: list[str]


class UncertainGate(_StrictModel):
    """確信度が低いゲート1件(GET /api/train/{sid}/uncertain_gates、§4-1)。

    entropyは正規化迷い[0,1](層間・パラメータ化間で比較可能)。
    """

    layer_id: str
    neuron_index: int
    entropy: float
    gate_id: int


class UncertainGatesResponse(_StrictModel):
    """低確信ゲートTop-Nレスポンス(§4-1)。gatesはentropy降順。"""

    step: int
    gates: list[UncertainGate]


class ResourceEstimate(_StrictModel):
    """IRのリソース見積り(export.estimate.estimate_resources)。

    lut_counts/node_countsはJSON化でキーが文字列になる(dict[str, int])。
    """

    total_luts: int
    lut_counts: dict[str, int]  # n_inputs別のLUT個数(キーはn_inputsの文字列)
    logic_depth: int
    wire_count: int
    group_sums: int
    node_counts: dict[str, int]  # ノード型別の個数(展開後)
    lowered: bool


class PreflightBlocker(_StrictModel):
    """HDL出力を妨げる要因1件(§4-3)。

    kind: 'blackbox' | 'add_sum' | 'structure'。
    node_idは該当ノード(構造エラー等で特定不能ならNone)。
    """

    node_id: str | None
    kind: str
    message: str


class VerifyAvailability(_StrictModel):
    """HDL等価性検証ツールの利用可否(verify.iverilog_available/ghdl_available)。"""

    iverilog: bool
    ghdl: bool


class PreflightResponse(_StrictModel):
    """HDL出荷プリフライト(GET /api/models/current/preflight、§4-3)。

    hdl_exportableはblockersが空のときのみ真。estimateはHDL不可の構造
    エラー時はNone(見積り計算が破綻するため)。
    """

    hdl_exportable: bool
    blockers: list[PreflightBlocker]
    warnings: list[str]
    estimate: ResourceEstimate | None
    draft_node_count: int  # 未離散化(train_info.initialized=False)ノード数
    verify_available: VerifyAvailability


class HdlExportResponse(_StrictModel):
    """HDLエクスポート(POST /api/export/hdl、hdl.export_hdl)。"""

    source: str
    filename: str
    lang: str  # 'verilog' | 'vhdl'
    estimate: ResourceEstimate


class HdlVerifyResponse(_StrictModel):
    """HDL等価性検証(POST /api/export/hdl/verify、§4-3)。

    availableが偽(処理系未導入)のときpassedはNone。
    """

    available: bool
    passed: bool | None
    n_vectors: int
    stdout: str
    lang: str


class ConvertStartRequest(_StrictModel):
    """変換開始リクエスト(POST /api/convert/start)。

    source: teacher指定('ckpt:<path>' 等、convert.teacher.resolve_teacherが解釈)。
    dataset: 較正/評価データセットキー(省略時はpreset既定)。空文字はpreset既定。
    preset: convert_presets のプリセット名(例 'mnist-intlower-v1')。
    overrides: ConvertPipelineSpecフィールドの部分上書き(空dict=上書きなし)。
    """

    source: str
    preset: str
    dataset: str = ''
    overrides: dict = {}


class ConvertStatus(_StrictModel):
    """変換セッション状態(GET /api/convert/{cid}/status、convert.ConversionSession)。

    phase: preparing/pruning/quantizing/lowering/verifying/done/error/cancelled。
    progress: 現phase内の進捗0-1。metrics: 各段acc・回路統計・学習曲線配列。
    has_ir: done時にTrue(生成IRはSTATEへ反映済み、GET /api/models/current/irで取得)。
    """

    convert_id: str
    phase: str
    progress: float
    metrics: dict
    error: str | None
    has_ir: bool
    source: str
    dataset: str


class ConvertPresetInfo(_StrictModel):
    """変換プリセット1件(GET /api/convert/presets、convert_presets.list_presets)。"""

    name: str
    label: str
    description: str
    dataset: str
    teacher_hint: str
    spec: dict


class RunSummary(_StrictModel):
    """runサマリ(GET /api/runs、runs.SUMMARY_KEYS)。壊れたrunはキー欠落=None。"""

    run_id: str
    started_at: float | None
    ended_at: float | None
    state: str | None
    dataset: str | None
    method: str | None
    loss: str | None
    optimizer: str | None
    lr: float | None
    batch_size: int | None
    epochs: int | None
    final_step: int | None
    final_loss: float | None
    final_acc: float | None
    model_name: str | None
    wandb_url: str | None


# 型生成(scripts/gen_editor_types.py)の対象モデル一覧
ALL_MODELS: list[type[BaseModel]] = [
    U8Envelope,
    MetricsFrame,
    LayerSummaryFrame,
    GateDistributionFrame,
    ActivationsFrame,
    GradFlowLayerTrace,
    GradFlowFrame,
    ApDeltaFrame,
    CaptureConfigFrame,
    CaptureInfo,
    HistoryIndex,
    HistoryIndexFrame,
    HistoryCounts,
    TrainStatus,
    SessionEndFrame,
    StepMetrics,
    HistoryLayerView,
    HistoryStepView,
    DatasetInfo,
    NodeCapabilities,
    PythonModelImportResponse,
    CapabilityReason,
    HierarchyNode,
    HierarchyResponse,
    LoweringView,
    UncertainGate,
    UncertainGatesResponse,
    ResourceEstimate,
    PreflightBlocker,
    VerifyAvailability,
    PreflightResponse,
    HdlExportResponse,
    HdlVerifyResponse,
    ConvertStartRequest,
    ConvertStatus,
    ConvertPresetInfo,
    RunSummary,
    # --- N系列APIの追加分(各routerからimportして登録) ---
    StatsResponse,  # GET /api/stats
    DatasetSampleItem,  # GET /api/datasets/{id}/samples
    DatasetSamplesResponse,
    DatasetConfigUpdate,  # PUT /api/datasets/imported/{id}/config
    EncodingPreviewRequest,  # POST /api/datasets/encoding-preview
    EncodingPreviewResponse,
    ProjectSchema,  # プロジェクトCRUD(POST/GET/PUT /api/projects)
    CreateProjectRequest,
    UpdateProjectRequest,
    TrainConfig,  # GET /api/config/schema(TrainConfig.model_json_schema()そのもの)
    EvalSource,  # POST /api/evaluate
    EvaluateRequest,
    EvalItem,
    EvaluateResponse,
    NasStartRequest,  # POST/GET /api/nas/*
    NasStartResponse,
    NasStatusResponse,
    NasTrialRow,
    NasTrialsResponse,
    NasIrResponse,
    TransferRequest,  # POST /api/networks/transfer
    SkippedItem,
    TransferResponse,
]
