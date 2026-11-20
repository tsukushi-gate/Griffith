"""共通学習ランナー。

TrainConfig(config.py)を受けて
  seed固定 → データ解決 → モデル構築 → 学習ループ(soft loss/acc +
  hard eval=離散化評価) → config.yaml/チェックポイント保存
までを一括実行する。experiments/train.py はこの新経路では変更しない
(数値回帰リスク回避。ランナーが datasets/vision.py を使う)。

設計判断:
  - チェックポイント保存は本モジュール内の最小実装(torch.save)。
    保存dictは griffith_logic/training/checkpoint_utils.py と同一スキーマ
    (model_state_dict / optimizer_state_dict / scheduler_state_dict /
    step / best_metric / metadata)のため、既存の load_checkpoint でも読める。
  - wandbは griffith_logic.training.wandb_utils をパッケージ内importで
    再利用する(init_wandbのrun ID記録・Artifactポリシー等の重複実装を避ける)。
    旧版は experiments/wandb_utils.py を cwd 相対でファイルロードして
    いたが、pip install した環境では成立せず、かつ cwd 汚染による任意コード
    実行の経路になっていたため廃止した。
"""

from __future__ import annotations

import json
import random
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from ..analysis.structure import (
    StructMetricsTracker,
    soft_eval_acc,
    struct_final_report,
)
from ..core.connections import (
    WIRING_SPEC_KEYS,
    butterfly_strides,
    get_permutation_connections,
    get_random_connections,
    get_rot_connections,
    get_stretched_rot_connections,
    get_uniform_random_connections,
    local2d_offset_schedule,
    make_connections,
)
from ..group_sum import GroupSum
from ..layers.lut import LutLayer
from . import wandb_utils
from .checkpoint_utils import torch_load
from .config import TrainConfig, check_expected_metrics, dump_config, save_config
from .tasks import TaskSpec, make_task

JST = timezone(timedelta(hours=9))

# 進捗printの間隔(秒)
PROGRESS_INTERVAL_SEC = 10.0

# 小さい論理データセットの1エポック目安行数(repeatで水増し)
LOGIC_TARGET_ROWS = 256

# データローダが解釈するparamsのうち、ランナー側で消費する予約キー
_RESERVED_DATA_PARAMS = ('limit',)

# 配線用 generator を重み初期化用と分けるためのseedオフセット(先行実験)。
# 別ストリームにすることで init_mode / gate_mask を変えても配線が変わらない
_WIRING_SEED_OFFSET = 1_000_003


def _auto_device() -> str:
    """デバイス自動検出(cuda→mps→cpu)"""
    if torch.cuda.is_available():
        return 'cuda'
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


def _set_seed(seed: int) -> None:
    """torch/numpy/randomのseedを固定する"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


@dataclass
class ResolvedData:
    """解決済みデータセット(タスク種別・入出力幅・ローダ)"""

    name: str
    kind: str  # 'classification' | 'binary_vector'
    in_width: int
    out_width: int  # classificationはクラス数
    train_loader: DataLoader
    eval_loader: DataLoader


def _resolve_data(cfg, device: str) -> ResolvedData:
    """DataConfigを解決する。

    - 'logic:<key>' — LOGIC_DATASETS(binary_vector)。paramsのrepeatで
      水増し行数を指定(省略時はLOGIC_TARGET_ROWS目安で自動)
    - それ以外      — VISION_DATASETS(classification)。paramsはローダへ
      透過(binarize_threshold等)。予約キー limit で先頭N件に絞る
    """
    params = dict(cfg.params)

    if cfg.name.startswith('logic:'):
        from ..datasets.logic import LOGIC_DATASETS

        key = cfg.name.split(':', 1)[1]
        if key not in LOGIC_DATASETS:
            raise ValueError(
                f'未対応の論理データセット: {key}(対応: {sorted(LOGIC_DATASETS)})'
            )
        probe = LOGIC_DATASETS[key]()
        repeat = int(params.pop('repeat', 0)) or max(
            1, LOGIC_TARGET_ROWS // len(probe.dataset)
        )
        if params:
            raise ValueError(f'logic:*で未対応のparams: {sorted(params)}')
        spec = LOGIC_DATASETS[key](repeat=repeat)
        train_loader = spec.loader(
            min(cfg.batch_size, len(spec.dataset)), shuffle=True
        )
        # hard evalは全2^nパターン一括(repeatなし)
        eval_loader = probe.loader()
        return ResolvedData(
            name=cfg.name, kind='binary_vector',
            in_width=spec.n_inputs, out_width=spec.n_outputs,
            train_loader=train_loader, eval_loader=eval_loader,
        )

    from ..datasets.vision import VISION_DATASETS

    if cfg.name not in VISION_DATASETS:
        raise ValueError(
            f'未対応のデータセット: {cfg.name}'
            f'(対応: logic:<key> / {sorted(VISION_DATASETS)})'
        )
    spec = VISION_DATASETS[cfg.name]
    limit = params.pop('limit', None)
    train_set, test_set = spec.load(**params)
    if limit is not None:
        limit = int(limit)
        train_set = Subset(train_set, range(min(limit, len(train_set))))
        test_set = Subset(test_set, range(min(limit, len(test_set))))

    pin_memory = device == 'cuda'
    train_loader = DataLoader(
        train_set, batch_size=cfg.batch_size, shuffle=True,
        drop_last=True, pin_memory=pin_memory,
    )
    eval_loader = DataLoader(
        test_set, batch_size=cfg.batch_size, shuffle=False,
        pin_memory=pin_memory,
    )
    return ResolvedData(
        name=cfg.name, kind='classification',
        in_width=spec.in_width, out_width=spec.classes,
        train_loader=train_loader, eval_loader=eval_loader,
    )


# 閉形式の構造化固定配線。乱数を引かず、extra の引数だけで決まる
_STRUCTURED_KINDS = ('rot_n', 'butterfly', 'local1d', 'local2d')


def _structured_connections(
    kind: str, extra: dict, n_inputs: int, in_dim: int, out_dim: int, device: str,
) -> torch.Tensor:
    """構造化固定配線 1 層分。extra は LayerConnectionsConfig の該当欄。

    生成は `core.connections.make_connections`(配線種の単一の入口)に委ね、
    LayerConnectionsConfig が持つ全欄のうち種が受け付けるものだけを渡す。
    extra['dedupe'] が真なら重複ピンを潰す。生成した入力数が model.n_inputs と
    違えば ValueError。
    """
    if kind not in _STRUCTURED_KINDS:
        raise ValueError(f'未知の構造化配線種: {kind!r}')
    spec = {
        k: v for k, v in extra.items()
        if k in WIRING_SPEC_KEYS[kind] and v is not None
    }
    return make_connections(
        kind, in_dim, out_dim, n_inputs, device=device,
        dedupe=bool(extra.get('dedupe')), **spec)


def _layer_extra(spec) -> dict:
    """LayerConnectionsConfig の構造化配線用の欄を dict にする。"""
    return dict(
        shifts=spec.shifts, step=spec.step, stride=spec.stride, offsets=spec.offsets,
        grid=spec.grid, bits_per_cell=spec.bits_per_cell, dedupe=spec.dedupe,
    )


def _structured_layer_spec(conn, cfg, layer_idx: int) -> tuple[str, dict]:
    """全層指定の butterfly / local1d / local2d を層 layer_idx の 1 層分に展開する。

    butterfly: 層 0 = gather(rot_n、shifts 0..n−1、step n = 内部調査リポジトリ先行実験
    `bf6_gather`)、層 l ≥ 1 = stride_l(strides 省略時 n^(l−1)、order で並び)。
    local1d: 層ごとの strides。
    local2d: 層 0 = 画像格子 × bits_per_cell × layer0_offsets、隠れ層 = 正方格子に
    ring 列を assign で割当(先行実験 `l2d6_offsets`)。
    """
    n, n_layers = cfg.n_inputs, cfg.layers
    if conn.kind == 'butterfly':
        if layer_idx == 0:
            return 'rot_n', dict(shifts=list(range(n)), step=n, dedupe=conn.dedupe)
        strides = conn.strides
        if strides is None:
            strides = butterfly_strides(n_layers - 1, n, conn.order)
        return 'butterfly', dict(stride=strides[layer_idx - 1], dedupe=conn.dedupe)
    if conn.kind == 'local1d':
        return 'local1d', dict(stride=conn.strides[layer_idx], dedupe=conn.dedupe)
    # local2d
    if layer_idx == 0:
        offsets = conn.layer0_offsets
        if offsets is None:
            offsets = [(0, 0)] if n == conn.bits_per_cell else [(1, 0), (0, 1)]
        return 'local2d', dict(
            offsets=offsets, grid=conn.grid, bits_per_cell=conn.bits_per_cell,
            dedupe=conn.dedupe)
    schedule = local2d_offset_schedule(n_layers - 1, n, conn.assign)
    return 'local2d', dict(
        offsets=schedule[layer_idx - 1], grid=None, bits_per_cell=1, dedupe=conn.dedupe)


def _connections_from_kind(
    kind: str, s_a, s_b, n_inputs: int, in_dim: int, out_dim: int,
    generator, device: str, step: int = 1, extra: dict | None = None,
) -> torch.Tensor:
    """配線種1つ分の生成(先行実験)。s_a / s_b は rot 系だけが使う。

    **配線用 generator を消費するのは perm / random_seeded / random_uniform
    だけ**で、rot / stretched_rot は乱数を引かない。generator は層順に
    消費されるので、層0が同じ種なら腕をまたいで層0の配線は一致する。
    step は rot の歩進(既定 1 で従来と bitwise 同一)。構造化配線
    (rot_n / butterfly / local1d / local2d)は extra の引数だけで決まり乱数を引かない。
    """
    if kind == 'rot':
        return get_rot_connections(in_dim, out_dim, s_a, s_b, device, step=step)
    if kind == 'stretched_rot':
        return get_stretched_rot_connections(in_dim, out_dim, s_a, s_b, device)
    if kind == 'random_seeded':
        return get_random_connections(
            in_dim, out_dim, n_inputs, device, generator)
    if kind == 'random_uniform':
        return get_uniform_random_connections(
            in_dim, out_dim, n_inputs, device, generator)
    if kind in _STRUCTURED_KINDS:
        return _structured_connections(
            kind, extra or {}, n_inputs, in_dim, out_dim, device)
    # perm: s_a / s_b は無視する(ConnectionsConfig の docstring 参照)
    return get_permutation_connections(in_dim, out_dim, generator, device)


def _resolve_connections(
    cfg, layer_idx: int, in_dim: int, out_dim: int,
    generator, device: str,
):
    """ModelConfig.connections を LutLayer へ渡す形へ解決する(先行実験)。

    文字列('random'/'unique')はそのまま素通し(従来動作)。ConnectionsConfig の
    場合だけ生成器を呼んで明示 Tensor を返す。`kind='per_layer'` では
    層番号 layer_idx の LayerConnectionsConfig を見て層ごとに種を変える(先行実験)。
    全層指定の構造化配線(butterfly / local1d / local2d)は
    `_structured_layer_spec` で層ごとの 1 層指定に展開する。
    """
    conn = cfg.connections
    if isinstance(conn, str):
        return conn
    if conn.kind == 'per_layer':
        spec = conn.layers[layer_idx]
        return _connections_from_kind(
            spec.kind, spec.s_a, spec.s_b, cfg.n_inputs,
            in_dim, out_dim, generator, device,
            step=spec.step, extra=_layer_extra(spec),
        )
    if conn.kind in ('butterfly', 'local1d', 'local2d'):
        kind, extra = _structured_layer_spec(conn, cfg, layer_idx)
        return _structured_connections(
            kind, extra, cfg.n_inputs, in_dim, out_dim, device)
    s_a = None if conn.s_a is None else conn.s_a[layer_idx]
    s_b = None if conn.s_b is None else conn.s_b[layer_idx]
    step = 1 if conn.step is None else conn.step[layer_idx]
    return _connections_from_kind(
        conn.kind, s_a, s_b, cfg.n_inputs, in_dim, out_dim, generator, device,
        step=step,
    )


def _lut_kwargs(
    cfg, layer_idx: int, n_layers: int, in_dim: int, out_dim: int,
    device: str, wiring_gen, init_gen,
) -> dict:
    """層ごとの LutLayer 引数(最終層だけマスクが違う、配線は層ごとに生成)。

    既定値のみのレシピでは従来の llkw と同じ引数になり bitwise 不変。
    """
    kw = dict(
        n_inputs=cfg.n_inputs,
        device=device,
        connections=_resolve_connections(
            cfg, layer_idx, in_dim, out_dim, wiring_gen, device,
        ),
        parameterization=cfg.parameterization,
        estimator=cfg.estimator,
        residual_init=cfg.residual_init,
        ri_mu=cfg.ri_mu,
        ri_sigma=cfg.ri_sigma,
        selection=cfg.selection,
        grad_factor=cfg.grad_factor,
    )
    is_final = layer_idx == n_layers - 1
    mask = cfg.gate_mask_final if is_final else cfg.gate_mask_hidden
    if mask is not None:
        kw['gate_mask'] = mask
    if cfg.init_mode is not None:
        kw['init_mode'] = cfg.init_mode
        kw['init_scale'] = cfg.init_scale
        kw['generator'] = init_gen
    return kw


def _build_model(
    cfg, data: ResolvedData, device: str, *, seed: int = 0,
) -> nn.Sequential:
    """ModelConfigからモデルを構築する(タスク種別との整合を検証)。

    seed は先行実験の規則配線(perm)と init_mode 用の generator を導く。
    **配線用と初期化用で別ストリーム**にしてあるので、init_mode や
    gate_mask を変えても配線は変わらない(腕どうしが同一配線を共有できる)。
    """
    wiring_gen = torch.Generator().manual_seed(seed + _WIRING_SEED_OFFSET)
    init_gen = torch.Generator().manual_seed(seed)

    if cfg.kind == 'conv_dlgn':
        return _build_conv_dlgn_model(cfg, data, device, wiring_gen)

    layers: list[nn.Module] = [nn.Flatten()]

    if cfg.kind == 'dlgn':
        if data.kind != 'classification':
            raise ValueError(
                f'dlgn(GroupSum終端)はclassification用です。'
                f'{data.name}({data.kind})には model.kind=logic_mlp を指定してください'
            )
        if cfg.k % data.out_width != 0:
            raise ValueError(
                f'ニューロン数({cfg.k})はクラス数({data.out_width})で'
                f'割り切れる必要があります'
            )
        d = data.in_width
        for i in range(cfg.layers):
            layers.append(LutLayer(d, cfg.k, **_lut_kwargs(
                cfg, i, cfg.layers, d, cfg.k, device, wiring_gen, init_gen,
            )))
            d = cfg.k
        layers.append(GroupSum(data.out_width, tau=cfg.tau, assign=cfg.group_assign))
    else:  # logic_mlp
        if data.kind != 'binary_vector':
            raise ValueError(
                f'logic_mlp(生ビット出力)はbinary_vector用です。'
                f'{data.name}({data.kind})には model.kind=dlgn を指定してください'
            )
        d = data.in_width
        for i in range(cfg.layers - 1):
            layers.append(LutLayer(d, cfg.k, **_lut_kwargs(
                cfg, i, cfg.layers, d, cfg.k, device, wiring_gen, init_gen,
            )))
            d = cfg.k
        layers.append(LutLayer(d, data.out_width, **_lut_kwargs(
            cfg, cfg.layers - 1, cfg.layers, d, data.out_width, device,
            wiring_gen, init_gen,
        )))

    return nn.Sequential(*layers).to(device)


def _build_conv_dlgn_model(cfg, data: ResolvedData, device: str, wiring_gen) -> nn.Module:
    """`model.kind='conv_dlgn'`を models/conv_dlgn.build_conv_dlgn で組む。

    入力は DataLoader の (B, C, H, W) をそのまま受ける(先頭に Flatten を置かない)ので、
    `model.conv.in_shape` の要素数がデータの入力幅と一致することを検査する。
    random8 の葉座標・sin の RI・頭部 random 配線は配線用 generator(run.seed 由来)で引く。
    """
    from ..models.conv_dlgn import ConvDLGNConfig, build_conv_dlgn

    if data.kind != 'classification':
        raise ValueError(
            f'conv_dlgn(GroupSum終端)はclassification用です。'
            f'{data.name}({data.kind})には使えません'
        )
    cc = cfg.conv
    in_shape = tuple(int(v) for v in cc.in_shape)
    if int(np.prod(in_shape)) != data.in_width:
        raise ValueError(
            f'model.conv.in_shape {in_shape} の要素数 {int(np.prod(in_shape))} が'
            f'データの入力幅 {data.in_width} と一致しません'
        )
    mcfg = ConvDLGNConfig(
        in_shape=in_shape, channels=tuple(int(v) for v in cc.channels),
        sampler=cc.sampler, kernel_size=cc.kernel_size, tree_depth=cc.tree_depth,
        padding=cc.padding, input_channels_per_tree=cc.input_channels_per_tree,
        downsample=cc.downsample,
        head_dims=None if cc.head_dims is None else tuple(int(v) for v in cc.head_dims),
        head_wiring=cc.head_wiring, head_scale=cc.head_scale,
        num_classes=data.out_width, tau=cfg.tau, group_assign=cfg.group_assign,
        parameterization=cfg.parameterization, estimator=cfg.estimator,
        residual_init=cfg.residual_init, ri_mu=cfg.ri_mu, ri_sigma=cfg.ri_sigma,
        selection=cfg.selection,
    )
    return build_conv_dlgn(mcfg, device=device, generator=wiring_gen).to(device)


def _resolve_data_shape(cfg) -> ResolvedData:
    """DataConfigからタスク種別・入出力幅だけを解決する(ローダは作らない)。

    デプロイ時のモデル再構築用。データセット本体のダウンロード/読み込みを
    しないので、学習データの無い環境でもチェックポイントを書き出せる。
    """
    if cfg.name.startswith('logic:'):
        from ..datasets.logic import LOGIC_DATASETS

        key = cfg.name.split(':', 1)[1]
        if key not in LOGIC_DATASETS:
            raise ValueError(
                f'未対応の論理データセット: {key}(対応: {sorted(LOGIC_DATASETS)})'
            )
        probe = LOGIC_DATASETS[key]()
        return ResolvedData(
            name=cfg.name, kind='binary_vector',
            in_width=probe.n_inputs, out_width=probe.n_outputs,
            train_loader=None, eval_loader=None,
        )

    from ..datasets.vision import VISION_DATASETS

    if cfg.name not in VISION_DATASETS:
        raise ValueError(
            f'未対応のデータセット: {cfg.name}'
            f'(対応: logic:<key> / {sorted(VISION_DATASETS)})'
        )
    spec = VISION_DATASETS[cfg.name]
    return ResolvedData(
        name=cfg.name, kind='classification',
        in_width=spec.in_width, out_width=spec.classes,
        train_loader=None, eval_loader=None,
    )


def load_model_from_checkpoint(
    checkpoint: str,
    *,
    config: str | None = None,
    device: str = 'cpu',
) -> tuple[nn.Sequential, TrainConfig, ResolvedData, dict]:
    """チェックポイントからモデルを再構築して重みを読み込む(griffith deploy用)。

    モデル構成は config(YAML)指定があればそれを、なければチェックポイントの
    metadata.config(run_trainingが保存)を使う。データセット本体は読まない。

    Returns:
        (eval状態のモデル, 設定, 入出力幅, torch.loadした生dict)
    """
    data = torch_load(checkpoint, map_location='cpu')
    if config is not None:
        from .config import load_config

        cfg = load_config(config)
    else:
        cfg_dict = data.get('metadata', {}).get('config')
        if cfg_dict is None:
            raise ValueError(
                f'チェックポイントに設定(metadata.config)がありません: {checkpoint}'
                '(--config で学習時の設定ファイルを指定してください)'
            )
        cfg = TrainConfig.model_validate(cfg_dict)

    shape = _resolve_data_shape(cfg.data)
    model = _build_model(cfg.model, shape, device, seed=cfg.run.seed)
    state = data.get('model_state_dict', data)
    try:
        model.load_state_dict(state)
    except RuntimeError as e:
        raise ValueError(
            f'チェックポイントとモデル構成の形状が不整合です'
            f'(--config の model/data 設定を確認): {e}'
        ) from e
    model.eval()
    return model, cfg, shape, data


def _build_optimizer(cfg, model: nn.Module) -> torch.optim.Optimizer:
    """OptimConfigからオプティマイザを構築する"""
    optims = {
        'adam': torch.optim.Adam,
        'adamw': torch.optim.AdamW,
        'sgd': torch.optim.SGD,
    }
    return optims[cfg.name](model.parameters(), lr=cfg.lr)


def _hard_eval(
    model: nn.Module, loader: DataLoader, task: TaskSpec, device: str,
) -> dict[str, float]:
    """離散化評価(eval mode=weight argmax + 入力round)。

    バッチごとのメトリクスをサンプル数で加重平均する。
    """
    was_training = model.training
    model.eval()
    sums: dict[str, float] = {}
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(torch.float32).to(device).round()
            y = y.to(device)
            metrics = task.metrics(model(x), y)
            n = y.shape[0]
            for name, value in metrics.items():
                sums[name] = sums.get(name, 0.0) + value * n
            total += n
    model.train(mode=was_training)
    return {name: value / total for name, value in sums.items()}


def _soft_eval(
    model: nn.Module, loader: DataLoader, task: TaskSpec, device: str,
) -> dict[str, float]:
    """soft(連続混合)forwardでのテスト精度(先行実験)。

    離散化ギャップ = soft_acc − hard_acc を出すための評価。LutLayer.forward は
    `self.training` と `self.selection` だけで経路が決まるので、対象層の
    selection を一時的に 'soft' にし train モードで forward する
    (hard_st / gumbel_st の層でも soft 経路を通せる)。決定的な評価にするため
    Gumbelノイズ注入も一時的に切る。**_hard_eval の挙動は一切変えない。**
    入力は hard eval と違い round しない(連続値のまま流す)。

    本関数は「追加の測定パス」なので、**グローバルRNG状態を復元**する。
    復元しないと評価パスが消費した乱数の分だけ次エポックの DataLoader の
    シャッフル順が変わり、save_eval_history を有効にしただけで学習軌道が
    変わってしまう(実測で確認)。
    """
    was_training = model.training
    saved = [
        (m, m.selection, m.gumbel_noise)
        for m in model.modules() if isinstance(m, LutLayer)
    ]
    rng_state = torch.get_rng_state()
    use_cuda = device.startswith('cuda') and torch.cuda.is_available()
    cuda_state = torch.cuda.get_rng_state_all() if use_cuda else None
    for layer, _, _ in saved:
        layer.selection = 'soft'
        layer.gumbel_noise = False
    model.train()
    sums: dict[str, float] = {}
    total = 0
    try:
        with torch.no_grad():
            for x, y in loader:
                x = x.to(torch.float32).to(device)
                y = y.to(device)
                metrics = task.metrics(model(x), y)
                n = y.shape[0]
                for name, value in metrics.items():
                    sums[name] = sums.get(name, 0.0) + value * n
                total += n
    finally:
        for layer, selection, gumbel_noise in saved:
            layer.selection = selection
            layer.gumbel_noise = gumbel_noise
        model.train(mode=was_training)
        torch.set_rng_state(rng_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)
    return {name: value / total for name, value in sums.items()}


def _op_histograms(model: nn.Module) -> list[list[int]]:
    """LutLayerごとの16ゲート出現ヒストグラム(eval の argmax 基準、先行実験)。"""
    hists: list[list[int]] = []
    with torch.no_grad():
        for layer in model.modules():
            if not isinstance(layer, LutLayer):
                continue
            if not getattr(layer, '_use_16gate', False):
                raise ValueError(
                    'op_hist は16ゲートモードのLutLayer専用です: '
                    f'n_inputs={layer.n_inputs}, '
                    f'parameterization={layer.parameterization!r}'
                )
            chosen = layer._masked_weights().argmax(dim=-1)
            hists.append(
                torch.bincount(chosen.flatten(), minlength=16).tolist()
            )
    return hists


def _netlist_dict(model: nn.Module, cfg: TrainConfig, data: ResolvedData) -> dict:
    """学習済みモデルを平坦ネットリスト(op, a, b)へ変換する(先行実験 P0形式)。

    op は IR の真理値表を `ir.weights.gate_id_from_tt_n2` で変換したもので、
    規約は op = 8·P00 + 4·P01 + 2·P10 + P11(= 社内先行実験の op ニブル)。
    最終層のゲート i はクラス i // group_size に加算される。
    """
    from ..conv import LutConv2d
    from ..export import export_ir
    from ..ir.serialize import tt_from_hex
    from ..ir.weights import gate_id_from_tt_n2

    if any(isinstance(m, LutConv2d) for m in model.modules()):
        raise ValueError(
            'export_netlist(op/a/b 平坦ネットリスト)は全結合 DLGN 専用です。'
            'conv_dlgn は export_ir(native_conv=True) の IR を使ってください'
        )
    graph = export_ir(model, input_width=data.in_width)
    layers: list[list[dict]] = []
    group: dict | None = None
    for node in graph.nodes:
        if node.type == 'lut_layer':
            if node.params['n_inputs'] != 2:
                raise ValueError(
                    'export_netlist は n_inputs=2 専用です: '
                    f'n_inputs={node.params["n_inputs"]}'
                )
            idx = np.asarray(node.params['indices'], dtype=np.int64)
            tts = tt_from_hex(node.params['truth_tables'])
            layers.append([
                {
                    'op': gate_id_from_tt_n2(int(tt)),
                    'a': int(idx[0][o]),
                    'b': int(idx[1][o]),
                }
                for o, tt in enumerate(tts)
            ])
        elif node.type == 'group_sum':
            group = node.params

    output = None
    if group is not None:
        classes = int(group['k'])
        output = {
            'classes': classes,
            'group_size': int(group['in_width']) // classes,
            'tau': float(group['tau']),
        }
    return {
        'input_bits': int(data.in_width),
        'binarization': (
            f'{cfg.data.name}; '
            f'params={json.dumps(cfg.data.params, sort_keys=True)}'
        ),
        'layers': layers,
        'output': output,
    }


def _write_json(path: Path, payload) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8'
    )
    return str(path)


def _default_run_name(cfg: TrainConfig) -> str:
    """run_name省略時の自動命名"""
    dataset = cfg.data.name.replace(':', '-')
    return f'{dataset}_{cfg.model.kind}_k{cfg.model.k}_l{cfg.model.layers}'


def _save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    best_metric: float,
    metadata: dict,
) -> None:
    """チェックポイント保存(checkpoint_utils.pyと同一dictスキーマ)"""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': None,
        'step': step,
        'best_metric': best_metric,
        'metadata': metadata,
    }, str(path))


def _apply_distill(cfg: TrainConfig, data: ResolvedData, device: str) -> TaskSpec:
    """蒸留を data(train側)に適用し、KDタスクを返す(最小diff統合)。

    副作用: data.train_loader を PackedDistillDataset ベースの新しい DataLoader へ
    差し替える(train側のみ。data.eval_loader は素のまま=評価は分類タスクを維持)。

    処理の流れ:
        resolve_teacher → teacher_transform 上書き → load_raw(train) →
        get_or_compute_logits(cache=Falseなら常時再計算) →
        train_loader.dataset を PackedDistillDataset で包む → DataLoader 再構築 →
        make_distill_task を返す

    蒸留は classification タスク限定(GroupSum終端のクラスロジットが前提)。
    それ以外(binary_vector)は ValueError。

    Args:
        cfg: 学習設定(cfg.distill は非None)。
        data: _resolve_data の結果(classification 前提)。
        device: teacher/学習デバイス。

    Returns:
        KD用 TaskSpec(make_distill_task)。学習ループの task として使う。
    """
    from torch.utils.data import Subset

    from ..convert.cache import (
        PackedDistillDataset,
        get_or_compute_logits,
        teacher_cache_key,
    )
    from ..convert.distill import make_distill_task
    from ..convert.teacher import TEACHER_TRANSFORMS, resolve_teacher
    from ..datasets.vision import load_raw

    dcfg = cfg.distill
    if data.kind != 'classification':
        raise ValueError(
            f'蒸留(distill)は classification タスク専用です。'
            f'{data.name}({data.kind})では使用できません。'
        )

    # teacher解決 + transform上書き(teacher側で正規化する規約)
    teacher = resolve_teacher(dcfg.teacher, device=device)
    if dcfg.teacher_transform not in TEACHER_TRANSFORMS:
        raise ValueError(
            f'未知の teacher_transform: {dcfg.teacher_transform!r}'
            f'(対応: {sorted(TEACHER_TRANSFORMS)})'
        )
    teacher.transform = TEACHER_TRANSFORMS[dcfg.teacher_transform]

    # student(二値化済み)train datasetを取得。Subset(limit)なら基底とインデックスを分離
    student_train = data.train_loader.dataset
    if isinstance(student_train, Subset):
        base_student = student_train.dataset
        indices = list(student_train.indices)
    else:
        base_student = student_train
        indices = None

    # 未二値化のraw train set(既存ローダと同一の基底・同一インデックス順)
    raw_params = dict(cfg.data.params)
    raw_params.pop('limit', None)
    raw_params.pop('binarize_threshold', None)
    raw_params.pop('threshold_count', None)
    download = bool(raw_params.pop('download', True))
    raw_root = raw_params.pop('root', None)
    raw_train, _ = load_raw(cfg.data.name, data_root=raw_root, download=download)

    key = teacher_cache_key(
        teacher.fingerprint,
        cfg.data.name,
        {'transform': dcfg.teacher_transform},
        split='train',
        n=len(raw_train),
    )
    if dcfg.cache:
        logits = get_or_compute_logits(
            teacher, raw_train, key=key, device=device,
        )
    else:
        # 常時再計算(キャッシュ読み書きしない)
        logits = get_or_compute_logits(
            teacher, raw_train, key=key, device=device,
            root=str(Path(tempfile.mkdtemp(prefix='md-distill-nocache-'))),
        )

    packed = PackedDistillDataset(
        base_student, logits, pseudo_labels=dcfg.pseudo_labels,
    )
    if indices is not None:
        packed = Subset(packed, indices)

    pin_memory = device == 'cuda'
    data.train_loader = DataLoader(
        packed, batch_size=cfg.data.batch_size, shuffle=True,
        drop_last=True, pin_memory=pin_memory,
    )
    return make_distill_task(
        data.out_width,
        alpha=dcfg.alpha,
        temperature=dcfg.temperature,
        standardize=dcfg.standardize,
        kind=dcfg.kind,
    )


def _wandb_tags(cfg: TrainConfig, data: ResolvedData) -> list[str]:
    """wandbのタグ(ランナー既定 + レシピの run.wandb_tags)を順序を保って重複なく並べる。

    既定タグは 'md-train' / model.kind / データ種別(+ 蒸留時 'distill')で従来どおり。
    レシピ側の追加タグは後ろへ足す(既に入っているものは足さない)。
    """
    tags = ['md-train', cfg.model.kind, data.kind]
    if cfg.distill is not None:
        tags.append('distill')
    for tag in cfg.run.wandb_tags:
        if tag not in tags:
            tags.append(tag)
    return tags


def run_training(cfg: TrainConfig, *, out_dir: str | None = None) -> dict:
    """設定に従って学習を実行し、最終メトリクスを返す。

    Args:
        cfg: 学習設定(load_configで構築)
        out_dir: 出力先ディレクトリ(省略時は outputs/<timestamp>_<run名>)

    Returns:
        最終メトリクス辞書(hard_acc / best_hard_acc / final_loss /
        steps / out_dir / checkpoint_latest / checkpoint_best 等)
    """
    if cfg.run.method != 'bp':
        raise NotImplementedError(
            f'共通ランナーは現状bpのみ対応: method={cfg.run.method}'
        )

    device = cfg.run.device or _auto_device()
    _set_seed(cfg.run.seed)

    data = _resolve_data(cfg.data, device)
    # 評価は常に素の分類/二値ベクトルタスク(packedを流さない)。
    # 蒸留時も _hard_eval は eval_task+元のeval_loader を使う(下の _apply_distill 参照)。
    eval_task = make_task(data.kind, cfg.run.loss)
    task = eval_task
    if cfg.distill is not None:
        task = _apply_distill(cfg, data, device)
    model = _build_model(cfg.model, data, device, seed=cfg.run.seed)
    optimizer = _build_optimizer(cfg.optim, model)

    run_name = cfg.run.run_name or _default_run_name(cfg)
    if out_dir is None:
        timestamp = datetime.now(JST).strftime('%Y%m%d_%H%M%S')
        out = Path('outputs') / f'{timestamp}_{run_name}'
    else:
        out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # 解決済み設定を保存(griffith train <out>/config.yaml で完全再実行できる)
    config_path = save_config(cfg, str(out / 'config.yaml'))

    latest_path = out / 'checkpoints' / 'latest.pt'
    best_path = out / 'checkpoints' / 'best.pt'
    ckpt_metadata = {'config': dump_config(cfg), 'run_name': run_name}

    # wandb(任意機能): パッケージ内の共通ヘルパーを使う
    use_wandb = False
    if cfg.run.wandb:
        use_wandb = wandb_utils.init_wandb(
            run_name=run_name,
            config_dict=dump_config(cfg),
            model=model,
            device=device,
            tags=_wandb_tags(cfg, data),
            group=cfg.run.wandb_group,
            job_type=cfg.run.wandb_job_type,
            notes=cfg.run.wandb_notes,
            hypothesis=cfg.run.wandb_hypothesis,
        )

    # 構造メトリクス(SM4): flip率・エントロピー・定数/パススルー率。
    # wandb有効時のみeval間隔で収集する(headless実験で構造推移を残す)
    struct_tracker = None
    if use_wandb:
        try:
            struct_tracker = StructMetricsTracker(model)
        except Exception as e:  # noqa: BLE001 — 計測失敗は学習を止めない
            print(f'構造メトリクス無効(初期化失敗): {e}')

    steps_per_epoch = len(data.train_loader)
    total_steps = cfg.run.epochs * steps_per_epoch

    # 正則化(先行実験)。lam_fn は step(0始まり。loss計算時点のstep) → λ。
    # ramp_frac>0 のとき [0, ramp_frac·total_steps] で 0 → λ 線形、以降 λ 固定。
    penalty_fn = None
    lam_fn = None
    if cfg.regularization is not None:
        from ..convert.distill import resolve_schedule
        from ..regularization.apply import irreducibility_penalty

        penalty_fn = irreducibility_penalty
        lam = float(cfg.regularization.lambda_)
        ramp_steps = cfg.regularization.ramp_frac * total_steps
        if ramp_steps > 0:
            lam_fn = resolve_schedule(f'linear:0:{lam}:{ramp_steps}')
        else:
            lam_fn = resolve_schedule(lam)
        print(f'正則化: {cfg.regularization.kind} λ={lam} '
              f'ramp_frac={cfg.regularization.ramp_frac} '
              f'(ramp {ramp_steps:.0f}ステップ)')

    print(f'デバイス: {device}')
    print(f'データセット: {data.name}({data.kind}, '
          f'in={data.in_width}, out={data.out_width})')
    print(f'モデル: {cfg.model.kind} k={cfg.model.k} layers={cfg.model.layers} '
          f'n_inputs={cfg.model.n_inputs}')
    print(f'学習: {cfg.run.epochs}エポック × {steps_per_epoch}ステップ '
          f'= {total_steps}ステップ, loss={task.loss_name}')
    print(f'出力先: {out}')

    step = 0
    best_hard_acc = 0.0
    last_loss = float('nan')
    last_acc = float('nan')
    reg_penalty = float('nan')
    reg_lambda = float('nan')
    soft_acc = float('nan')
    hard_metrics: dict[str, float] = {}
    eval_history: list[dict] = []
    start = time.time()
    last_print = start

    def do_eval(epoch: int) -> None:
        """hard eval → チェックポイント保存 → ログ"""
        nonlocal best_hard_acc, hard_metrics, soft_acc
        hard_metrics = _hard_eval(model, data.eval_loader, eval_task, device)
        is_best = hard_metrics['acc'] > best_hard_acc
        if is_best:
            best_hard_acc = hard_metrics['acc']
        _save_checkpoint(
            latest_path, model, optimizer, step, best_hard_acc, ckpt_metadata
        )
        if is_best:
            _save_checkpoint(
                best_path, model, optimizer, step, best_hard_acc, ckpt_metadata
            )
            # best更新のたびにArtifact化(evalステージ=WANDB_ARTIFACTS=all時のみ実送信)
            if use_wandb:
                wandb_utils.log_checkpoint_artifact(
                    str(best_path), name=run_name, stage='eval',
                    metadata={**ckpt_metadata, 'best_hard_acc': best_hard_acc},
                    model=model,
                )
        extra = ''.join(
            f'  {name}={value:.4f}'
            for name, value in hard_metrics.items() if name != 'acc'
        )
        if cfg.run.save_eval_history:
            # soft(連続混合)テスト精度 → 離散化ギャップ soft − hard
            soft_acc = _soft_eval(
                model, data.eval_loader, eval_task, device
            )['acc']
            eval_history.append({
                'step': step,
                'epoch': epoch,
                'hard_acc': hard_metrics['acc'],
                'soft_acc': soft_acc,
                'loss': last_loss,
                'reg_penalty': reg_penalty,
                'reg_lambda': reg_lambda,
            })
            extra += f'  soft_acc={soft_acc:.4f}'
        if penalty_fn is not None:
            extra += f'  reg={reg_penalty:.4f}(λ={reg_lambda:.4g})'
        print(f'[epoch {epoch}/{cfg.run.epochs} step {step}/{total_steps}] '
              f'loss={last_loss:.4f}  hard_acc={hard_metrics["acc"]:.4f}{extra}'
              f'{"  ★ BEST" if is_best else ""}')
        if use_wandb:
            payload = {
                'epoch': epoch,
                'step': step,
                'loss': last_loss,
                'train_acc_soft': last_acc,
                **{f'hard_{k}': v for k, v in hard_metrics.items()},
                'best_hard_acc': best_hard_acc,
            }
            if struct_tracker is not None:
                try:
                    payload.update(struct_tracker.collect())
                    # selection gap: soft(連続緩和)評価パスを1回追加して算出
                    soft = soft_eval_acc(
                        model, data.eval_loader, eval_task.metric_fn, device,
                    )
                    payload['struct/val_acc_soft'] = soft
                    payload['struct/selection_gap'] = (
                        soft - hard_metrics['acc']
                    )
                except Exception as e:  # noqa: BLE001 — 計測失敗は継続
                    print(f'構造メトリクス収集失敗(継続): {e}')
            if penalty_fn is not None:
                payload['reg_penalty'] = reg_penalty
                payload['reg_lambda'] = reg_lambda
            wandb_utils.log_metrics(payload)

    try:
        for epoch in range(1, cfg.run.epochs + 1):
            model.train()
            for x, y in data.train_loader:
                x = x.to(torch.float32).to(device)
                y = y.to(device)
                output = model(x)
                loss = task.loss_fn(output, y)
                if penalty_fn is not None:
                    # λ=0 のときは損失グラフに一切触らない(正則化なしと bitwise 不変)。
                    # 指標としての reg_penalty は no_grad で測って残す
                    reg_lambda = float(lam_fn(step))
                    if reg_lambda != 0.0:
                        pen = penalty_fn(model)
                        reg_penalty = float(pen.detach())
                        loss = loss + reg_lambda * pen
                    else:
                        with torch.no_grad():
                            reg_penalty = float(penalty_fn(model))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                step += 1
                last_loss = loss.item()
                last_acc = task.metric_fn(output.detach(), y)

                now = time.time()
                if now - last_print >= PROGRESS_INTERVAL_SEC:
                    elapsed = now - start
                    eta_sec = elapsed / step * (total_steps - step)
                    print(f'[step {step}/{total_steps}] loss={last_loss:.4f}  '
                          f'acc(soft)={last_acc:.4f}  '
                          f'ETA {timedelta(seconds=int(eta_sec))}')
                    last_print = now

                if cfg.run.eval_interval > 0 and step % cfg.run.eval_interval == 0:
                    do_eval(epoch)
                    model.train()

            # エポック末: eval_interval=0のとき、または区切りが合わないときに評価
            if cfg.run.eval_interval == 0 or step % cfg.run.eval_interval != 0:
                do_eval(epoch)

        total_time = timedelta(seconds=int(time.time() - start))
        print(f'完了: hard_acc={hard_metrics["acc"]:.4f} '
              f'(best={best_hard_acc:.4f}, 総時間 {total_time})')

        result = {
            'run_name': run_name,
            'out_dir': str(out),
            'config_path': config_path,
            'checkpoint_latest': str(latest_path),
            'checkpoint_best': str(best_path),
            'steps': step,
            'final_loss': last_loss,
            'train_acc_soft': last_acc,
            **{f'hard_{k}' if k != 'acc' else 'hard_acc': v
               for k, v in hard_metrics.items()},
            'best_hard_acc': best_hard_acc,
        }

        # 構造の最終レポート(SM4): IR抽出→sweep→到達コーン。
        # wandb無効でもresultに残す(失敗は通知の上で学習結果を優先)。
        # conv_dlgnは平坦入力の export_ir を前提とする構造計測の対象外
        # (analysis/structure.py は LutConv2d 未対応)なので黙って飛ばさず通知する
        if cfg.model.kind == 'conv_dlgn':
            print('構造最終レポート: conv_dlgn は対象外(LutConv2d 未対応)のためスキップ')
        else:
            try:
                struct_final = struct_final_report(model, data.in_width)
            except Exception as e:  # noqa: BLE001
                print(f'構造最終レポート失敗(継続): {e}')
            else:
                result['struct_metrics'] = struct_final
                if use_wandb:
                    wandb_utils.log_metrics(struct_final)
        if penalty_fn is not None:
            result['reg_penalty'] = reg_penalty
            result['reg_lambda'] = reg_lambda

        # 先行実験の追加成果物(既定Falseで従来と同じ出力)
        if cfg.run.save_eval_history:
            result['eval_history_path'] = _write_json(
                out / 'eval_history.json', eval_history
            )
            result['soft_acc'] = soft_acc
            result['summary_path'] = _write_json(out / 'summary.json', {
                'hard_acc_final': hard_metrics['acc'],
                'hard_acc_best': best_hard_acc,
                'soft_acc_final': soft_acc,
                'op_hist': _op_histograms(model),
                'steps': step,
                'wall_clock_s': time.time() - start,
            })
        if cfg.run.export_netlist:
            result['netlist_path'] = _write_json(
                out / 'netlist.json', _netlist_dict(model, cfg, data)
            )
    except Exception:
        # 学習失敗時も終了状態をローカルに残してから再raise
        if use_wandb:
            wandb_utils.finish_wandb(status='failed')
        raise

    # 正常終了: best.ptを最終Artifact化してからwandbを終了する
    if use_wandb:
        if best_path.exists():
            wandb_utils.log_checkpoint_artifact(
                str(best_path), name=run_name, stage='final',
                metadata={**ckpt_metadata, 'best_hard_acc': best_hard_acc},
                model=model,
            )
        wandb_utils.finish_wandb(status='finished', final_metrics=result)

    # 期待メトリクス照合。未達でも例外にはしない(--strictはCLI側で判定)
    if cfg.expected_metrics:
        verdict = check_expected_metrics(cfg.expected_metrics, result)
        result['expected_metrics'] = verdict
        print('--- expected_metrics照合 ---')
        for c in verdict['checks']:
            sym = '>=' if c['bound'] == 'min' else '<='
            actual = 'N/A' if c['actual'] is None else f'{c["actual"]:.4f}'
            mark = 'OK' if c['ok'] else '未達'
            print(f'  {c["metric"]}={actual} {sym} {c["expected"]}  … {mark}')
        print(f'照合結果: {"全件合格" if verdict["ok"] else "未達あり"}')

    return result


def evaluate_checkpoint(
    checkpoint: str,
    *,
    dataset: str | None = None,
    batch_size: int | None = None,
    device: str | None = None,
) -> dict:
    """チェックポイントをhard eval(離散化評価)する(md eval用)。

    モデル構成はチェックポイントのmetadata.config(run_trainingが保存)から
    復元する。datasetを指定すると評価データセットを差し替える。

    Returns:
        メトリクス辞書(hard_acc / step / dataset 等)
    """
    data = torch_load(checkpoint, map_location='cpu')
    cfg_dict = data.get('metadata', {}).get('config')
    if cfg_dict is None:
        raise ValueError(
            f'チェックポイントに設定(metadata.config)がありません: {checkpoint}'
            '(griffith train / run_training が保存したチェックポイントを指定してください)'
        )
    cfg = TrainConfig.model_validate(cfg_dict)
    if dataset is not None:
        cfg.data.name = dataset
    if batch_size is not None:
        cfg.data.batch_size = batch_size

    dev = device or cfg.run.device or _auto_device()
    resolved = _resolve_data(cfg.data, dev)
    task = make_task(resolved.kind, cfg.run.loss)
    model = _build_model(cfg.model, resolved, dev, seed=cfg.run.seed)
    try:
        model.load_state_dict(data['model_state_dict'])
    except RuntimeError as e:
        raise ValueError(
            f'チェックポイントとモデル構成/データセットの形状が不整合です'
            f'(--datasetの入出力幅を確認): {e}'
        ) from e

    metrics = _hard_eval(model, resolved.eval_loader, task, dev)
    return {
        'checkpoint': checkpoint,
        'dataset': cfg.data.name,
        'step': data.get('step'),
        **{f'hard_{k}' if k != 'acc' else 'hard_acc': v
           for k, v in metrics.items()},
    }
