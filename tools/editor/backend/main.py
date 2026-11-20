"""ビジュアルエディタのFastAPIバックエンド。

起動方法:
    uv run uvicorn tools.editor.backend.main:app --port 8765

REST:
    GET  /api/health 死活確認(server_epoch=再起動検出用)
    GET  /api/datasets 学習データセットカタログ
    POST /api/models/demo            デモモデル生成 → IR JSON
    POST /api/models/import          チェックポイント読み込み → IR JSON
    POST /api/models/import/python   Pythonモデル読み込み → 表示IR JSON
    GET  /api/models/current/ir      現在のIR取得
    PUT  /api/models/current/ir      IR差し替え(validate→422 / 訓練中409)
    POST /api/models/current/nodes/{node_id}/truth_table
                                     単一ニューロンの真理値表編集(IR更新+ビルド済みならweights in-place更新。
                                     訓練running中409/paused許可)
    POST /api/models/current/nodes/{node_id}/indices
                                     単一ニューロンの接続編集(IR更新+ビルド済みならindices+逆引き再計算)
    POST /api/models/current/build   IR→モデル構築+再エクスポートIR
    GET  /api/models/current/hierarchy
                                     Python由来モデル/IRの階層表示メタデータ(read-only)
    GET  /api/models/current/lower   ?node_id=...&level=L0|L1|L2 のlowering表示(read-only)
    POST /api/models/current/codegen IR→Pythonコード生成
    POST /api/models/current/save    IRをファイルへ保存
    POST /api/models/ir/load         IRファイル読み込み → IR JSON
    POST /api/train/start            訓練開始 → session_id(各ランは編集中モデルの独立コピーを学習)
    GET  /api/train/sessions         全ライブセッション(並列ラン)の状態一覧
    POST /api/train/{sid}/apply      当該ランの学習済み重みを編集中モデルへ採用+IR再エクスポート
    POST /api/train/{sid}/pause      一時停止
    POST /api/train/{sid}/resume     再開
    POST /api/train/{sid}/stop       停止
    GET  /api/train/{sid}/status 状態取得(step_budget/capture/history付き)
    GET  /api/train/{sid}/confusion 最新hard evalの混同行列(classificationのみ)
    GET  /api/train/{sid}/uncertain_gates  確信度が低いゲートTop-N(§4-1)
    POST /api/train/{sid}/step       nステップ進めて自動pause(paused時のみ)
    GET  /api/train/{sid}/history 履歴インデックス(min/max_step, keyframes等)
    GET  /api/train/{sid}/history/{step}
                                     任意stepのスナップショットビュー(read-onlyスクラブ、
                                     ?layers=&fields=metrics,summary,gate_distribution,trace)
    POST /api/train/{sid}/restore    branch復元(weights厳密+optimizerはkeyframe近似、
                                     paused時のみ。復元先より後の履歴はtruncate)
    GET  /api/convert/presets 変換プリセット一覧
    POST /api/convert/start 事前学習MLP→IR変換を非同期開始→{convert_id}
                                     (学習中409 / preset・overrides不正400)
    GET  /api/convert/{cid}/status   変換状態(done検知でIRをSTATEへ反映)
    POST /api/convert/{cid}/cancel   変換キャンセル要求(各phase境界で中断)

WebSocket:
    /ws/train/{sid}
        サーバ→クライアント: {"type":"metrics",...} / {"type":"layer_summary",...}
        クライアント→サーバ: {"type":"subscribe","layers":[...]}
"""

import asyncio
import copy
import json
import queue
import re
import time
from pathlib import Path

import torch
import torch.nn as nn
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from griffith_logic.ir import (
    build_torch,
    ir_from_dict,
    ir_to_dict,
    load_ir,
    save_ir,
    validate,
    weights_from_truth_tables,
)

from .codegen import generate_code
from .edit import generate_wiring, set_neuron_indices, set_neuron_truth_table
from .security import (
    WorkspacePathError,
    allowed_hosts,
    is_host_allowed,
    is_origin_allowed,
    resolve_workspace_path,
)
from .extract import (
    gate_probs_from_weights,
    lut_layers_with_ids,
    summary_from_weights,
)
from .monitor import build_capture_config, encode_u8
from .demo import (
    MNIST_IN_DIM,
    PROJECT_ROOT,
    create_demo_model,
    export_model_ir,
    load_model_from_checkpoint,
)
from .training import SessionManager, TrainingSession
from .python_importer import (
    annotate_capabilities,
    generate_python_ref_code,
    hierarchy_from_ir,
    load_python_editor_model,
    lowering_view_from_ir,
    preflight_python_import,
)

METRICS_DRAIN_BATCH = 500        # 1ループで吸い出すmetricsの最大件数
SEND_LOOP_INTERVAL_SEC = 0.05    # 送信ループのポーリング間隔
SUMMARY_SEND_INTERVAL_SEC = 0.2  # layer_summary送信の最小間隔(5Hz)
HISTORY_INDEX_INTERVAL_SEC = 1.0  # history_indexプッシュの最小間隔

app = FastAPI(title='griffith-logic editor backend')

# 許可ホスト(ループバック + GRIFFITH_EDITOR_ALLOWED_HOSTS)。モジュール読み込み時に
# 1度だけ確定させる(リクエスト毎のenv参照を避ける)。
ALLOWED_HOSTS = allowed_hosts()

# 状態を変えうるメソッド。ブラウザはこれらに必ずOriginを付けるためCSRF判定に使う。
_STATE_CHANGING_METHODS = frozenset({'POST', 'PUT', 'DELETE', 'PATCH'})

# ヘッダが複数値だったことを表す番兵(Noneや文字列と区別する)。
_AMBIGUOUS_HEADER = object()


def _single_header(request: Request, name: str):
    """ヘッダを1個だけ取り出す。無ければ None、複数あれば `_AMBIGUOUS_HEADER`。

    Host/Originの重複は正当なブラウザでは起こらず、リクエストスマグリングで
    前段プロキシと後段の解釈を食い違わせる手口の兆候なので、検証を素通りさせずに
    拒否する(`headers.get` は先頭だけを返すため明示的に見る必要がある)。
    """
    values = request.headers.getlist(name)
    if not values:
        return None
    if len(values) > 1:
        return _AMBIGUOUS_HEADER
    return values[0]


@app.middleware('http')
async def guard_browser_requests(request: Request, call_next):
    """Host/Origin検証(R6)。ローカル専用ツールの前提を機械的に守る。

    - Host不一致 → 421 Misdirected Request(DNSリバインディング対策の本命)
    - 状態変更メソッドでOriginがループバック外 → 403(CSRF対策)
    """
    host = _single_header(request, 'host')
    if host is _AMBIGUOUS_HEADER or not is_host_allowed(host, ALLOWED_HOSTS):
        return JSONResponse(
            status_code=421,
            content={'detail': 'このサーバはローカル(localhost/127.0.0.1)からのみ利用できます'},
        )
    if request.method in _STATE_CHANGING_METHODS:
        origin = _single_header(request, 'origin')
        if origin is _AMBIGUOUS_HEADER or not is_origin_allowed(origin, ALLOWED_HOSTS):
            return JSONResponse(
                status_code=403,
                content={'detail': '許可されていないOriginからのリクエストです'},
            )
    return await call_next(request)


app.add_middleware(
    CORSMiddleware,
    allow_origins=['http://localhost:5173', 'http://127.0.0.1:5173'],
    allow_methods=['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'],
    allow_headers=['Content-Type', 'Authorization'],
)


class AppState:
    """プロセス内の共有状態(現在のモデル・IRと訓練セッション)。"""

    def __init__(self):
        self.model = None
        self.input_width: int | None = None
        self.ir: dict | None = None       # 現在のIR(JSON dict、単一モデル前提)
        self.ir_dirty: bool = False       # IRがモデルより新しい(要build)
        self.sessions = SessionManager()
        from .convert import ConversionManager
        self.conversions = ConversionManager()  # 変換セッション
        # プロセス起動時刻。フロントが再起動(=セッション消失)を検出する
        self.server_epoch = time.time()


STATE = AppState()

# 推論プローブとHDLエクスポートはAPIRouterで分離実装
from . import config_schema, dataset_import, dataset_samples, evaluate, hdl, nas, network_transfer, probe, projects_router, stats, templates  # noqa: E402 — STATE定義後にconfigure注入するため

app.include_router(probe.configure(STATE))
app.include_router(hdl.configure(STATE))
app.include_router(stats.configure(STATE))
app.include_router(projects_router.configure(STATE))
app.include_router(dataset_samples.router)
app.include_router(config_schema.configure(STATE))
app.include_router(dataset_import.router)
app.include_router(evaluate.configure(STATE))
app.include_router(nas.configure(STATE))
app.include_router(network_transfer.router)
app.include_router(templates.configure(STATE))


# --- リクエストスキーマ ---

class DemoRequest(BaseModel):
    k: int = Field(default=2000, ge=10)
    layers: int = Field(default=3, ge=1, le=16)


class ImportRequest(BaseModel):
    path: str


class PythonImportRequest(BaseModel):
    path: str
    symbol: str | None = None
    mode: str = Field(default='auto', pattern='^(auto|display)$')


class TrainStartRequest(BaseModel):
    dataset: str = 'mnist'  # 'mnist' | 'logic:<key>' | 'truth_table'
    lr: float = Field(default=0.01, gt=0)
    logic_lr: float | None = Field(default=None, gt=0)
    ann_lr: float | None = Field(default=None, gt=0)
    batch_size: int = Field(default=128, ge=1)
    epochs: int = Field(default=10, ge=1)
    method: str = Field(default='bp', pattern='^(bp|ap|bb)$')  #
    loss: str | None = None          # None=タスク既定(ce/bce)
    optimizer: str = Field(default='adam', pattern='^(adam|sgd)$')
    dataset_params: dict | None = None  # truth_table用(table/n_inputs/n_outputs)
    wandb: bool = False  # wandb連携(opt-in。未認証環境でも無害)


class IRPutRequest(BaseModel):
    ir: dict


class IRPathRequest(BaseModel):
    path: str


class TruthTableEditRequest(BaseModel):
    neuron: int = Field(ge=0)
    tt_hex: str


class IndicesEditRequest(BaseModel):
    neuron: int = Field(ge=0)
    slots: list[int]


class WiringGenerateRequest(BaseModel):
    """配線生成。kind と spec は core.connections.make_connections の引数。"""
    kind: str
    in_width: int = Field(ge=1)
    out_width: int = Field(ge=1)
    n_inputs: int = Field(ge=1)
    spec: dict = Field(default_factory=dict)
    dedupe: bool = False


class StepRequest(BaseModel):
    n: int = Field(default=1, ge=1, le=1000)


class RestoreRequest(BaseModel):
    step: int = Field(ge=0)
    mode: str = 'branch'  # v1はbranchのみ(閲覧はGET /history/{step}で行う)


class ConvertStartRequest(BaseModel):
    """変換開始リクエスト(schemas.ConvertStartRequestと同形)。"""

    source: str
    preset: str
    dataset: str = ''            # 空文字はpreset既定を使う
    overrides: dict = Field(default_factory=dict)


# --- REST ---

@app.get('/api/health')
def health():
    return {'status': 'ok', 'server_epoch': STATE.server_epoch}


@app.get('/api/datasets')
def datasets_catalog():
    """学習データセットカタログ。フロントのdatasetセレクタが使う。"""
    from .datasets import list_datasets

    from . import dataset_import

    # 組込カタログ+取込データセット。フロント側でも合成するがAPI単体でも完全な一覧を返す
    return list_datasets() + dataset_import.list_imported_datasets()


@app.post('/api/models/demo')
def models_demo(req: DemoRequest):
    """デモモデルを生成してセッションに保持し、IRを返す。"""
    try:
        model = create_demo_model(k=req.k, layers=req.layers)
    except AssertionError as e:
        raise HTTPException(status_code=400, detail=str(e))
    STATE.model = model
    STATE.input_width = MNIST_IN_DIM
    ir = export_model_ir(model, MNIST_IN_DIM, name=f'demo_mnist_k{req.k}_l{req.layers}')
    ir = annotate_capabilities(ir)
    STATE.ir = ir
    STATE.ir_dirty = False
    return {'ir': ir, 'input_width': MNIST_IN_DIM}


@app.post('/api/models/import')
def models_import(req: ImportRequest):
    """チェックポイントを読み込んでIRを返す(v0.1: Sequential直列のみ)。"""
    ckpt_path = _resolve_path(req.path, suffixes=('.pt', '.pth', '.ckpt'))
    try:
        model = load_model_from_checkpoint(str(ckpt_path))
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f'ファイルが存在しない: {req.path}')
    except Exception as e:  # noqa: BLE001 — torch.load失敗を400で返す
        raise HTTPException(status_code=400, detail=f'読み込み失敗: {e}')

    from griffith_logic import LutLayer
    lut = next((m for m in model.modules() if isinstance(m, LutLayer)), None)
    if lut is None:
        raise HTTPException(status_code=400, detail='LutLayerを含まないモデルは未対応')
    input_width = lut.in_dim

    try:
        ir = export_model_ir(model, input_width, name=req.path)
        ir = annotate_capabilities(ir)
    except Exception as e:  # noqa: BLE001 — 非直列モデル等
        raise HTTPException(status_code=400, detail=f'IR抽出失敗: {e}')
    STATE.model = model
    STATE.input_width = input_width
    STATE.ir = ir
    STATE.ir_dirty = False
    return {'ir': ir, 'input_width': input_width}


@app.post('/api/models/import/python')
def models_import_python(req: PythonImportRequest):
    """Pythonファイルのeditor_model/build_modelを読み込み、表示IRを返す。

    実行の正はPythonモデル、IRはエディタ表示・I/O検証・コード生成の契約として
    保持する。blackboxを含むためIR単体からのbuildは不可。
    """
    _reject_if_training()
    try:
        preflight_python_import(req.path, symbol=req.symbol)
        result = load_python_editor_model(req.path, symbol=req.symbol, mode=req.mode)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except (ValueError, TypeError, AttributeError, ImportError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    STATE.model = result.model
    STATE.input_width = result.input_width
    STATE.ir = result.ir
    STATE.ir_dirty = False
    return {
        'ir': result.ir,
        'input_width': result.input_width,
        'model_ref': result.ir.get('metadata', {}).get('editor', {}).get('model_ref'),
        'spec': result.spec,
    }


# --- IR編集(E2) ---

def _resolve_path(path: str, *, suffixes: tuple[str, ...] | None = None,
                  must_exist: bool = False) -> Path:
    """ワークスペース内包チェック付きでパスを解決する(B3)。

    相対パスはリポジトリルート基準。`../` 脱出・ワークスペース外の絶対パス・
    シンボリックリンク経由の脱出は400で拒否する。
    """
    try:
        return resolve_workspace_path(
            path, suffixes=suffixes, workspace_root=PROJECT_ROOT, must_exist=must_exist
        )
    except WorkspacePathError as e:
        raise HTTPException(status_code=400, detail=str(e))


def _parse_ir_or_422(doc: dict):
    """dict→IRGraph+validate。不正なら422を送出する。"""
    try:
        graph = ir_from_dict(doc)
        warnings = validate(graph)
    except (ValueError, KeyError, TypeError, AssertionError) as e:
        raise HTTPException(status_code=422, detail=f'IRが不正: {e}')
    return graph, warnings


def _model_ref_of(doc: dict | None) -> dict | None:
    if not isinstance(doc, dict):
        return None
    editor = doc.get('metadata', {}).get('editor')
    if not isinstance(editor, dict):
        return None
    ref = editor.get('model_ref')
    return ref if isinstance(ref, dict) else None


def _ir_execution_contract(doc: dict | None) -> dict | None:
    """Python由来IRの実行契約比較用に、表示metaを除いた構造だけを取り出す。"""
    if not isinstance(doc, dict):
        return None
    nodes = []
    for node in doc.get('nodes', []):
        if not isinstance(node, dict):
            return None
        nodes.append({
            'id': node.get('id'),
            'type': node.get('type'),
            'params': node.get('params'),
            'train_info': node.get('train_info'),
        })
    return {
        'inputs': doc.get('inputs'),
        'outputs': doc.get('outputs'),
        'nodes': nodes,
        'edges': doc.get('edges'),
    }


def _merge_editor_meta(old_graph, new_ir: dict) -> dict:
    """build再エクスポート後のIRへ旧IRのエディタmeta(座標等)を引き継ぐ。

    buildは構造を変えない1:1再エクスポートなので、トポロジカル順の
    対応でmeta.editorを移す(idはn{i}_...へ張り替わっていてよい)。
    """
    # export_irはnn.Flattenをノード化しないため、flattenノードを除いて対応づける
    old_nodes = [n for n in old_graph.topological_order() if n.type != 'flatten']
    for old_node, new_node in zip(old_nodes, new_ir['nodes']):
        editor = old_node.meta.get('editor') if isinstance(old_node.meta, dict) else None
        if editor:
            new_node['meta'] = {**new_node.get('meta', {}), 'editor': editor}
    editor_md = old_graph.metadata.get('editor')
    if editor_md:
        new_ir['metadata'] = {**new_ir.get('metadata', {}), 'editor': editor_md}
    return new_ir


def _reexport_dag_ir(graph, gmod) -> dict:
    """DAGモデルの確定重みを元IRへ書き戻す(ノードid不変)。

    Sequential前提のexport_irはDAGに使えないため、lut_layerノードのみ
    構築済みモジュールからtruth_tables/indicesを再抽出して確定する
    (draftはbuild_torchのランダム初期化で確定済み)。
    """
    from griffith_logic import LutLayer
    from griffith_logic.export.extract import truth_tables_from_lut_layer
    from griffith_logic.ir.serialize import tt_to_hex

    doc = ir_to_dict(graph)
    for nd in doc['nodes']:
        if nd['type'] != 'lut_layer':
            continue
        key = gmod._key_of.get(nd['id'])
        module = gmod.node_modules[key] if key else None
        if not isinstance(module, LutLayer):
            continue
        tables = truth_tables_from_lut_layer(module)
        nd['params']['truth_tables'] = tt_to_hex(tables.tolist())
        nd['params']['indices'] = module.indices.cpu().numpy().astype(int).tolist()
        # draftフラグを除去(確定済み。serial経路のexport_irと同じく持たない)
        (nd.get('train_info') or {}).pop('initialized', None)
    return doc


def _build_current_model() -> dict:
    """STATE.irからモデルを構築し、再エクスポートしたIRでSTATEを更新する。

    draftノード(initialized=False)はランダム初期化で確定する。
    直列: ノードidは export_ir 規約(n{i}_...)へ張り替わる。
    DAG(IRGraphModule): ノードidは不変のまま重みのみIRへ書き戻す。
    """
    graph, _ = _parse_ir_or_422(STATE.ir)
    try:
        core = build_torch(graph)
    except (ValueError, NotImplementedError) as e:
        raise HTTPException(status_code=422, detail=f'build失敗: {e}')
    input_width = graph.inputs[0].width
    from griffith_logic.ir import IRGraphModule
    if isinstance(core, IRGraphModule):
        # MNIST画像入力(batch,1,28,28)対応のためFlattenを前置(2D入力には恒等)
        model = nn.Sequential(nn.Flatten(), core)
        model.eval()
        ir = _reexport_dag_ir(graph, core)
    else:
        model = nn.Sequential(nn.Flatten(), *core)
        model.eval()
        ir = export_model_ir(model, input_width,
                             name=str(graph.metadata.get('name', 'model')))
        ir = _merge_editor_meta(graph, ir)
    ir = annotate_capabilities(ir)
    STATE.model = model
    STATE.input_width = input_width
    STATE.ir = ir
    STATE.ir_dirty = False
    return ir


def _validate_trainable(req: 'TrainStartRequest'):
    """選択データセットとモデルI/O幅の整合を開始前に検証する。

    classification: 出力=クラス数のGroupSum終端を想定。
    binary_vector: 出力=ビット幅の生出力(lut_layer/binarize終端)を想定。
    """
    from .datasets import resolve_dataset

    spec = (
        ((STATE.ir or {}).get('metadata') or {})
        .get('editor', {})
        .get('model_ref', {})
        .get('spec', {})
    )
    caps = spec.get('capabilities') if isinstance(spec, dict) else None
    if isinstance(caps, dict) and caps.get('trainable') is False:
        raise HTTPException(
            status_code=400,
            detail=(
                'このPythonモデルはエディタ内の軽量学習対象ではありません。'
                'LL-ViT再現は experiments/train_llvit.py またはModalジョブで実行してください。'
            ),
        )

    try:
        resolved = resolve_dataset(
            req.dataset, batch_size=req.batch_size, params=req.dataset_params,
        )
    except ValueError:
        # 組込カタログに無ければ取込データセットへフォールバック
        from . import dataset_import

        try:
            resolved = dataset_import.resolve_imported(
                req.dataset, batch_size=req.batch_size, params=req.dataset_params,
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    input_width = STATE.input_width
    output_width = None
    if STATE.ir is not None and STATE.ir.get('outputs'):
        output_width = STATE.ir['outputs'][0].get('width')
    if input_width != resolved.in_width or output_width != resolved.out_width:
        hint = (
            '小規模サンプルは logic:xor 等の論理データセットで学習できます。'
            if req.dataset == 'mnist' else ''
        )
        raise HTTPException(
            status_code=400,
            detail=(
                f'データセット {req.dataset} には'
                f'入力幅{resolved.in_width}・出力{resolved.out_width}のモデルが必要です'
                f'(現在: 入力{input_width}, 出力{output_width})。{hint}'
            ),
        )
    return resolved


def _reject_if_training():
    """並列ランは編集中モデルの独立コピーを学習するため、学習中でも編集/buildは
    安全(実行中のランに影響しない)。よって拒否しない(並列ワークフローの前提)。
    ※関数はガード撤廃後も呼び出し側の互換のため残す。"""
    return


@app.get('/api/models/current/ir')
def get_current_ir():
    if STATE.ir is None:
        raise HTTPException(
            status_code=404, detail='IR未設定。先に /api/models/demo 等を呼ぶこと'
        )
    return {'ir': STATE.ir, 'dirty': STATE.ir_dirty}


@app.put('/api/models/current/ir')
def put_current_ir(req: IRPutRequest):
    _reject_if_training()
    graph, warnings = _parse_ir_or_422(req.ir)
    old_ref = _model_ref_of(STATE.ir)
    new_ref = _model_ref_of(req.ir)
    preserve_python_model = (
        STATE.model is not None
        and old_ref is not None
        and new_ref is not None
        and old_ref.get('kind') == 'python'
        and old_ref.get('path') == new_ref.get('path')
        and old_ref.get('symbol') == new_ref.get('symbol')
        and _ir_execution_contract(STATE.ir) == _ir_execution_contract(req.ir)
    )
    STATE.ir = annotate_capabilities(req.ir, python_executable=preserve_python_model)
    STATE.ir_dirty = not preserve_python_model
    if not preserve_python_model:
        STATE.model = None  # 旧モデルは無効。次のbuild/train開始時に再構築する
    STATE.input_width = graph.inputs[0].width if graph.inputs else None
    return {'ok': True, 'warnings': warnings}


# --- 単一ニューロン編集 ---

def _reject_if_running():
    """単一ニューロン編集ガード。並列ランは独立コピーを学習するため、編集中モデルの
    in-place編集は実行中ランに影響しない。よって拒否しない(並列ワークフローの前提)。
    ※関数はガード撤廃後も呼び出し側の互換のため残す。"""
    return


def _get_ir_lut_node(node_id: str) -> dict:
    """STATE.irからlut_layerノードを取得する(なければ404/型違い422)。"""
    if STATE.ir is None:
        raise HTTPException(status_code=404, detail='IR未設定')
    node = next((n for n in STATE.ir['nodes'] if n['id'] == node_id), None)
    if node is None:
        raise HTTPException(status_code=404, detail=f'ノードが存在しない: {node_id}')
    if node['type'] != 'lut_layer':
        raise HTTPException(
            status_code=422, detail=f'{node_id}はlut_layerではない: {node["type"]}'
        )
    return node


def _model_lut_layer(node_id: str):
    """ビルド済みモデルから該当LutLayerを返す(未構築・不一致ならNone)。"""
    if STATE.model is None:
        return None
    for name, layer in lut_layers_with_ids(STATE.model):
        if name == node_id:
            return layer
    return None


@app.post('/api/models/current/nodes/{node_id}/truth_table')
def edit_node_truth_table(node_id: str, req: TruthTableEditRequest):
    _reject_if_running()
    node = _get_ir_lut_node(node_id)
    p = node['params']
    if req.neuron >= int(p['out_width']):
        raise HTTPException(
            status_code=422,
            detail=f'neuronが範囲外: {req.neuron} (out_width={p["out_width"]})',
        )
    try:
        tt = int(req.tt_hex, 16)
    except ValueError:
        raise HTTPException(status_code=422, detail=f'tt_hexが不正: {req.tt_hex}')

    n_inputs = int(p['n_inputs'])
    param = node['train_info'].get('parameterization', 'softmax')
    effective = 'softmax16g' if (param == 'softmax' and n_inputs == 2) else param
    try:
        # 表現可能性の検証(iwp/softmax N>2/値域外はここでValueError)
        weights_from_truth_tables(effective, n_inputs, [tt])
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    p['truth_tables'][req.neuron] = hex(tt)  # IRが真実源
    layer = _model_lut_layer(node_id)
    if layer is not None:
        set_neuron_truth_table(layer, req.neuron, tt)
    else:
        STATE.ir_dirty = True
    return {'ok': True, 'model_updated': layer is not None, 'tt_hex': hex(tt)}


@app.post('/api/models/current/nodes/{node_id}/indices')
def edit_node_indices(node_id: str, req: IndicesEditRequest):
    _reject_if_running()
    node = _get_ir_lut_node(node_id)
    p = node['params']
    in_width = int(p['in_width'])
    if req.neuron >= int(p['out_width']):
        raise HTTPException(
            status_code=422,
            detail=f'neuronが範囲外: {req.neuron} (out_width={p["out_width"]})',
        )
    if len(req.slots) != int(p['n_inputs']):
        raise HTTPException(
            status_code=422,
            detail=f'slots数({len(req.slots)}) != n_inputs({p["n_inputs"]})',
        )
    if any(not 0 <= int(v) < in_width for v in req.slots):
        raise HTTPException(
            status_code=422,
            detail=f'slotsが入力幅(in_width={in_width})を超えている: {req.slots}',
        )

    for j, v in enumerate(req.slots):  # IRが真実源
        p['indices'][j][req.neuron] = int(v)
    layer = _model_lut_layer(node_id)
    if layer is not None:
        set_neuron_indices(layer, req.neuron, [int(v) for v in req.slots])
    else:
        STATE.ir_dirty = True
    return {'ok': True, 'model_updated': layer is not None}


@app.post('/api/wiring/generate')
def wiring_generate(req: WiringGenerateRequest):
    """配線種と引数から lut_layer の indices を生成する。

    STATE に触れない純関数(ライブラリの make_connections と同じ生成器)。フロントは
    返った indices を lut_layer.params.indices へ書き込み、train_info.connections /
    wiring に種と引数を残す(draft なら build 時に同じ関数で再生成される)。
    """
    try:
        return generate_wiring(
            req.kind, req.in_width, req.out_width, req.n_inputs, req.spec,
            dedupe=req.dedupe,
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))


@app.post('/api/models/current/build')
def models_build():
    _reject_if_training()
    if STATE.ir is None:
        raise HTTPException(status_code=400, detail='IR未設定')
    ir = _build_current_model()
    return {'ir': ir, 'input_width': STATE.input_width}


@app.get('/api/models/current/hierarchy')
def models_hierarchy():
    """現在モデルの階層表示メタデータを返す(read-only)。"""
    if STATE.ir is None:
        raise HTTPException(status_code=404, detail='IR未設定')
    return hierarchy_from_ir(STATE.ir)


@app.get('/api/models/current/lower')
def models_lower(node_id: str, level: str = 'L0'):
    """階層ノードを指定レベルへloweringした表示データを返す(read-only)。"""
    if STATE.ir is None:
        raise HTTPException(status_code=404, detail='IR未設定')
    try:
        return lowering_view_from_ir(STATE.ir, node_id=node_id, level=level)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post('/api/models/current/codegen')
def models_codegen():
    if STATE.ir is None:
        raise HTTPException(status_code=400, detail='IR未設定')
    graph, _ = _parse_ir_or_422(STATE.ir)
    model_ref = (
        graph.metadata.get('editor', {}).get('model_ref')
        if isinstance(graph.metadata.get('editor'), dict) else None
    )
    if isinstance(model_ref, dict) and model_ref.get('kind') == 'python':
        code = generate_python_ref_code(model_ref)
        name = re.sub(r'[^0-9A-Za-z_]+', '_', str(graph.metadata.get('name', 'model')))
        return {'code': code, 'filename': f"{name.strip('_') or 'model'}_ref.py"}
    try:
        code = generate_code(graph)
    except NotImplementedError as e:
        raise HTTPException(status_code=422, detail=f'コード生成失敗: {e}')
    name = re.sub(r'[^0-9A-Za-z_]+', '_', str(graph.metadata.get('name', 'model')))
    return {'code': code, 'filename': f"{name.strip('_') or 'model'}.py"}


@app.post('/api/models/current/save')
def models_save(req: IRPathRequest):
    if STATE.ir is None:
        raise HTTPException(status_code=400, detail='IR未設定')
    graph, _ = _parse_ir_or_422(STATE.ir)
    try:
        json_path = save_ir(graph, str(_resolve_path(req.path, suffixes=('.json',))))
    except OSError as e:
        raise HTTPException(status_code=400, detail=f'保存失敗: {e}')
    return {'path': json_path}


@app.post('/api/models/ir/load')
def models_ir_load(req: IRPathRequest):
    _reject_if_training()
    path = _resolve_path(req.path, suffixes=('.json',))
    if not path.exists():
        raise HTTPException(status_code=404, detail=f'ファイルが存在しない: {path}')
    try:
        graph = load_ir(str(path))
        validate(graph)
        ir = annotate_capabilities(ir_to_dict(graph))
    except (ValueError, KeyError, TypeError, AssertionError) as e:
        raise HTTPException(status_code=422, detail=f'IR読み込み失敗: {e}')
    STATE.ir = ir
    STATE.ir_dirty = True
    STATE.model = None
    STATE.input_width = graph.inputs[0].width if graph.inputs else None
    return {'ir': ir, 'input_width': STATE.input_width}


@app.post('/api/train/start')
def train_start(req: TrainStartRequest):
    if STATE.model is None and STATE.ir is not None:
        _build_current_model()  # PUTされたIRから自動build
    if STATE.model is None:
        raise HTTPException(
            status_code=400, detail='モデル未設定。先に /api/models/demo を呼ぶこと'
        )
    _validate_trainable(req)
    # 各ランは編集中モデルの独立コピーを学習する(並列ランが互いの重みを壊さない、
    # かつ学習が編集中モデルを破壊的に書き換えない=非破壊探索)。採用は /apply で明示。
    try:
        run_model = copy.deepcopy(STATE.model)
    except Exception as e:  # noqa: BLE001 — 複製不能なモデルは明確に返す
        raise HTTPException(status_code=422, detail=f'モデルの複製に失敗: {e}')
    try:
        session = STATE.sessions.create(
            run_model, dataset=req.dataset, lr=req.lr,
            batch_size=req.batch_size, epochs=req.epochs,
            method=req.method, loss=req.loss,
            optimizer_name=req.optimizer, dataset_params=req.dataset_params,
            logic_lr=req.logic_lr, ann_lr=req.ann_lr,
            ir_snapshot=STATE.ir, use_wandb=req.wandb,
        )
    except (RuntimeError, ValueError) as e:
        status = 409 if isinstance(e, RuntimeError) else 400
        raise HTTPException(status_code=status, detail=str(e))
    return {'session_id': session.sid, 'state': session.state}


@app.get('/api/train/sessions')
def train_sessions():
    """全ライブセッション(並列ラン)の状態一覧。アクティブラン一覧UIのポーリング用。"""
    return {'sessions': STATE.sessions.list_sessions()}


def _get_session_or_404(sid: str) -> TrainingSession:
    session = STATE.sessions.get(sid)
    if session is None:
        raise HTTPException(status_code=404, detail=f'セッションが存在しない: {sid}')
    return session


def _apply_session(session: TrainingSession) -> dict:
    """セッションの学習済みモデルを編集中モデル(STATE.model)へ採用し、IRを再エクスポートする。

    学習は独立コピー上で行うため既定では編集中モデルを書き換えない。本APIでフォーカス中
    ランの重みを明示的に採用し、キャンバス表示・エクスポート/codegenへ反映する。
    """
    from griffith_logic.ir import IRGraphModule

    model = copy.deepcopy(session.model)  # 以後の編集がランに影響しないよう複製
    graph, _ = _parse_ir_or_422(STATE.ir)
    if len(lut_layers_with_ids(model)) != len(lut_layers_with_ids(STATE.model)):
        raise HTTPException(
            status_code=422,
            detail='モデル構造が学習開始時から変化したため重みを適用できません',
        )
    input_width = STATE.input_width or (graph.inputs[0].width if graph.inputs else None)
    core = model[1] if isinstance(model, nn.Sequential) and len(model) > 1 else model
    if isinstance(core, IRGraphModule):
        ir = _reexport_dag_ir(graph, core)
    else:
        ir = export_model_ir(
            model, input_width, name=str(graph.metadata.get('name', 'model'))
        )
        ir = _merge_editor_meta(graph, ir)
    ir = annotate_capabilities(ir)
    STATE.model = model
    STATE.input_width = input_width
    STATE.ir = ir
    STATE.ir_dirty = False
    return ir


@app.post('/api/train/{sid}/pause')
def train_pause(sid: str):
    session = _get_session_or_404(sid)
    session.pause()
    return session.status()


@app.post('/api/train/{sid}/resume')
def train_resume(sid: str):
    session = _get_session_or_404(sid)
    session.resume()
    return session.status()


@app.post('/api/train/{sid}/stop')
def train_stop(sid: str):
    session = _get_session_or_404(sid)
    session.stop()
    return session.status()


@app.post('/api/train/{sid}/apply')
def train_apply(sid: str):
    """当該ランの学習済み重みを編集中モデルへ採用し、再エクスポートしたIRを返す。"""
    session = _get_session_or_404(sid)
    ir = _apply_session(session)
    return {'ok': True, 'ir': ir}


@app.get('/api/train/{sid}/status')
def train_status(sid: str):
    return _get_session_or_404(sid).status()


@app.get('/api/train/{sid}/confusion')
def train_confusion(sid: str):
    """最新hard evalの混同行列(classificationタスクのみ)。

    binary_vectorタスクや評価前は404(フロントはパネル非表示にする)。
    """
    session = _get_session_or_404(sid)
    if session.last_confusion is None:
        raise HTTPException(
            status_code=404,
            detail='混同行列がまだない(classificationタスクのeval後に生成される)',
        )
    return {
        'step': session.last_val_step,
        'classes': len(session.last_confusion),
        'matrix': session.last_confusion,
        'val_acc': session.last_val_acc,
    }


@app.get('/api/train/{sid}/uncertain_gates')
def train_uncertain_gates(sid: str, top_n: int = 20):
    """確信度が低いゲートTop-N(選択ギャップ・ダッシュボード §4-1)。

    全LutLayer横断でゲート分布エントロピー最大のニューロンを返す。
    フロントはクリックで該当ノードを選択・キャンバスジャンプする。
    """
    session = _get_session_or_404(sid)
    return {'step': session.step, 'gates': session.uncertain_gates(top_n)}


@app.post('/api/train/{sid}/step')
def train_step(sid: str, req: StepRequest):
    """nステップだけ進めて自動pauseする(paused状態でのみ許可)。"""
    session = _get_session_or_404(sid)
    try:
        session.step_n(req.n)
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e))
    return session.status()


@app.get('/api/train/{sid}/history')
def train_history(sid: str):
    """履歴インデックス(スクラブUIのタイムライン構築用)。"""
    session = _get_session_or_404(sid)
    idx = session.history.index()
    idx['capture'] = {
        'level': session.capture_level,
        'stride': session.history.stride,
    }
    return idx


def _layer_meta_from_ir(node_id: str) -> tuple[str, int]:
    """IRからレイヤのparameterization/n_inputsを引く(なければ既定値)。"""
    if STATE.ir is not None:
        node = next((n for n in STATE.ir['nodes'] if n['id'] == node_id), None)
        if node is not None and node['type'] == 'lut_layer':
            param = node['train_info'].get('parameterization', 'softmax')
            return str(param), int(node['params']['n_inputs'])
    return 'softmax', 2


@app.get('/api/train/{sid}/history/{step}')
def train_history_step(
    sid: str, step: int,
    layers: str | None = None,
    fields: str = 'metrics,summary',
):
    """任意stepのスナップショットビュー(read-onlyスクラブ)。

    非capture stepは直近下位のcaptured stepへ丸め、resolved_stepで返す。
    分布・要約はスナップショットのweightsからオンデマンド導出する。
    running中でも呼べる(モデル状態には一切触れない)。
    """
    session = _get_session_or_404(sid)
    snap = session.history.get(step)
    if snap is None:
        raise HTTPException(status_code=404, detail=f'step {step} 以前の履歴がない')

    want = {f.strip() for f in fields.split(',') if f.strip()}
    layer_filter = (
        {name.strip() for name in layers.split(',') if name.strip()}
        if layers else None
    )

    out: dict = {
        'resolved_step': snap.step,
        'method': snap.method,
        'metrics': {
            'step': snap.step,
            'epoch': snap.epoch,
            'loss': snap.loss,
            'acc': snap.acc,
        },
        'layers': {},
    }

    for name, rec in snap.layers.items():
        if layer_filter is not None and name not in layer_filter:
            continue
        param, n_inputs = _layer_meta_from_ir(name)
        entry: dict = {}
        if 'summary' in want:
            gate_ids, confidence = summary_from_weights(rec.weights, param, n_inputs)
            entry['gate_ids'] = [int(g) for g in gate_ids]
            entry['confidence'] = [round(float(c), 4) for c in confidence]
            entry['n_inputs'] = n_inputs
        if 'gate_distribution' in want:
            probs = gate_probs_from_weights(rec.weights, param, n_inputs)
            entry['gate_distribution'] = (
                encode_u8(probs, scale=1.0) if probs is not None else None
            )
        if rec.grad is not None:
            # ニューロン別の勾配ノルム(非負なのでu8化可能)
            entry['grad_norm'] = encode_u8(rec.grad.to(torch.float32).norm(dim=-1))
        if rec.act_mean is not None:
            entry['act_mean'] = encode_u8(rec.act_mean.to(torch.float32), scale=1.0)
        out['layers'][name] = entry

    if 'trace' in want:
        out['trace'] = snap.trace
    return out


@app.get('/api/runs')
def runs_list():
    """セッション横断のrunサマリ一覧(新しい順)。"""
    from .runs import list_runs
    from .training import RUNS_ROOT

    return {'runs': list_runs(RUNS_ROOT)}


@app.get('/api/runs/{run_id}')
def runs_get(run_id: str):
    """runの全記録(metrics系列・IRスナップショット込み)。"""
    from .runs import load_run
    from .training import RUNS_ROOT

    doc = load_run(RUNS_ROOT, run_id)
    if doc is None:
        raise HTTPException(status_code=404, detail=f'runが存在しない: {run_id}')
    return doc


@app.post('/api/train/{sid}/restore')
def train_restore(sid: str, req: RestoreRequest):
    """branch復元: 指定stepへ巻き戻して再学習の起点にする(paused時のみ)。"""
    session = _get_session_or_404(sid)
    if req.mode != 'branch':
        raise HTTPException(status_code=422, detail=f'未対応のmode: {req.mode}')
    try:
        return session.restore_to(req.step)
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


# --- 変換パイプライン(事前学習MLP→IR) ---

def _reflect_conversion_ir(session) -> bool:
    """完了した変換セッションの生成IRをSTATEへ反映する(1回だけ)。

    done かつ未反映(session.ir あり)なら STATE.model=None / STATE.ir=<ir> /
    STATE.ir_dirty=True にする。学習中は _reject_if_training で409を送出する。
    反映後は session.ir を None にして二重反映を防ぐ。
    反映した(またはdone以外)を bool で返す。
    """
    if session.phase != 'done' or session.ir is None:
        return False
    _reject_if_training()  # 学習中は反映せず409(モデル差し替えは訓練中不可)
    graph, _ = _parse_ir_or_422(session.ir)
    STATE.ir = session.ir
    STATE.ir_dirty = True
    STATE.model = None  # 変換IRは未build。次のbuild/train開始時に構築する
    STATE.input_width = graph.inputs[0].width if graph.inputs else None
    session.ir = None  # 反映済み(二重反映防止)
    session.ir_reflected = True
    return True


@app.get('/api/convert/presets')
def convert_presets():
    """変換プリセット一覧。ConvertWizardが選択肢として使う。"""
    from .convert_presets import list_presets

    return {'presets': list_presets()}


@app.post('/api/convert/start')
def convert_start(req: ConvertStartRequest):
    """事前学習MLP→IR変換を非同期開始し convert_id を返す。

    学習セッション実行中は409(モデル状態を差し替えるため)。source不正・
    preset不正・overrides不正は400。

    注: 編集系の_reject_if_trainingは並列ラン対応でno-op化済みだが、変換は
    完了時にSTATE.model/irを差し替えるため、実行中セッションがある間は
    本ハンドラ固有のガードで409にする(mainマージ時の意味整合)。
    """
    if STATE.sessions.active_session() is not None:
        raise HTTPException(
            status_code=409,
            detail='学習セッション実行中は変換を開始できません(モデル状態を差し替えるため)。'
                   '学習の停止後に再実行してください。',
        )
    from .convert import ConversionSession
    from .convert_presets import preset_dataset, resolve_spec

    try:
        spec = resolve_spec(req.preset, req.overrides)
        dataset = req.dataset or preset_dataset(req.preset)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    session = ConversionSession(
        source=req.source, dataset=dataset, spec=spec, device='cpu',
    )
    STATE.conversions.create(session)
    return {'convert_id': session.convert_id}


def _get_conversion_or_404(cid: str):
    session = STATE.conversions.get(cid)
    if session is None:
        raise HTTPException(status_code=404, detail=f'変換セッションが存在しない: {cid}')
    return session


@app.get('/api/convert/{cid}/status')
def convert_status(cid: str):
    """変換セッション状態。done検知時に生成IRをSTATEへ反映する。"""
    session = _get_conversion_or_404(cid)
    # done時は初回statusでSTATEへ反映(学習中なら_reject_if_trainingが409)
    if session.phase == 'done' and session.ir is not None:
        _reflect_conversion_ir(session)
    return session.status()


@app.post('/api/convert/{cid}/cancel')
def convert_cancel(cid: str):
    """変換セッションのキャンセル要求(各phase境界で中断)。"""
    session = _get_conversion_or_404(cid)
    session.cancel()
    return session.status()


# --- WebSocket ---

async def _recv_loop(ws: WebSocket, session: TrainingSession):
    """クライアントからのsubscribe要求を処理する(v2: detail購読対応)。

    detail省略時は現行動作と完全同一(後方互換)。
    """
    while True:
        text = await ws.receive_text()
        try:
            msg = json.loads(text)
        except json.JSONDecodeError:
            continue
        if msg.get('type') == 'subscribe':
            session.subscribed = set(msg.get('layers', []))
            detail = msg.get('detail') or {}
            session.detail_subscription = {
                'gate_distribution': set(detail.get('gate_distribution') or []),
                'grad_flow': bool(detail.get('grad_flow')),
                'activations': bool(detail.get('activations')),
                'ap_delta': bool(detail.get('ap_delta')),
            }
            interval_ms = msg.get('detail_interval_ms')
            if isinstance(interval_ms, int | float) and interval_ms > 0:
                session.detail_interval_sec = max(0.1, float(interval_ms) / 1000.0)


@app.websocket('/ws/train/{sid}')
async def ws_train(ws: WebSocket, sid: str):
    # WebSocketのOrigin検証(R6c)。HTTPミドルウェアはハンドシェイクのHostは見るが、
    # Originは状態変更メソッド判定に入らないためWS側で明示的に検証する。ブラウザは
    # クロスオリジンのWS接続に必ずOriginを付けるので、ここで弾けばCSWSH
    # (Cross-Site WebSocket Hijacking)を遮断できる。
    if not is_origin_allowed(ws.headers.get('origin'), ALLOWED_HOSTS):
        await ws.close(code=4403, reason='許可されていないOriginからの接続です')
        return
    session = STATE.sessions.get(sid)
    if session is None:
        await ws.accept()
        await ws.close(code=4004, reason=f'セッションが存在しない: {sid}')
        return
    await ws.accept()

    # 接続直後にcapture設定を通知(旧クライアントは未知typeとして無視する)
    await ws.send_text(json.dumps(build_capture_config(session)))

    recv_task = asyncio.create_task(_recv_loop(ws, session))
    last_sent_step: dict[str, int] = {}  # layer → 送信済みsummaryのstep
    last_summary_send = 0.0
    last_sent_detail: dict[str, int] = {}  # detailキー → 送信済みstep
    last_detail_send = 0.0
    last_hist_key: tuple | None = None   # 履歴インデックスの差分送信キー
    last_hist_send = 0.0

    try:
        while True:
            # metricsキューを吸い出して送信
            for _ in range(METRICS_DRAIN_BATCH):
                try:
                    frame = session.metrics_queue.get_nowait()
                except queue.Empty:
                    break
                await ws.send_text(json.dumps(frame))

            # layer_summaryは最大5Hzで差分送信(latest-wins)
            now = time.time()
            if now - last_summary_send >= SUMMARY_SEND_INTERVAL_SEC:
                with session.summary_lock:
                    pending = [
                        s for layer, s in session.latest_summaries.items()
                        if layer in session.subscribed
                        and last_sent_step.get(layer) != s['step']
                    ]
                for s in pending:
                    await ws.send_text(json.dumps(s))
                    last_sent_step[s['layer']] = s['step']
                if pending:
                    last_summary_send = now

            # 履歴インデックスは1Hzで差分プッシュ(タイムラインのポーリング代替)
            if now - last_hist_send >= HISTORY_INDEX_INTERVAL_SEC:
                idx = session.history.index()
                key = (idx['min_step'], idx['max_step'], idx['count'],
                       session.history.stride)
                if key != last_hist_key:
                    await ws.send_text(json.dumps({
                        'type': 'history_index',
                        **idx,
                        'capture': {
                            'level': session.capture_level,
                            'stride': session.history.stride,
                        },
                    }))
                    last_hist_key = key
                last_hist_send = now

            # detail系(gate_distribution/activations等)はセッション設定間隔で差分送信
            if now - last_detail_send >= session.detail_interval_sec:
                with session.summary_lock:
                    detail_pending = [
                        (key, f) for key, f in session.latest_details.items()
                        if last_sent_detail.get(key) != f.get('step')
                    ]
                for key, f in detail_pending:
                    await ws.send_text(json.dumps(f))
                    last_sent_detail[key] = f.get('step', -1)
                if detail_pending:
                    last_detail_send = now

            if recv_task.done():
                break  # クライアント切断(receiveが例外/完了)
            # セッション終了後もキューを掃き切ってから状態を通知して終了
            if not session.is_active and session.metrics_queue.empty():
                await ws.send_text(json.dumps({
                    'type': 'session_end', **session.status(),
                }))
                break
            await asyncio.sleep(SEND_LOOP_INTERVAL_SEC)
    except (WebSocketDisconnect, RuntimeError):
        pass  # 切断
    finally:
        recv_task.cancel()
