"""学習設定スキーマ。

pydantic v2 BaseModel + YAML + ドット記法CLIオーバーライド(hydra不使用)。
ランナー(runner.py)が保存する config.yaml は同一スキーマなので、
`griffith train outputs/<run>/config.yaml` でそのまま再実行できる。

オーバーライドの型解釈はYAMLスカラー(yaml.safe_load)→pydantic検証の
二段構え。未知キーは extra='forbid' で即エラーにしてタイポを検出する。
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)


class _StrictModel(BaseModel):
    """未知キーを拒否する共通基底(ドット記法上書きのタイポ検出)"""

    model_config = ConfigDict(extra='forbid')


class DataConfig(_StrictModel):
    """データセット設定。

    name: 'mnist' 等(VISION_DATASETS)または 'logic:xor' 等(LOGIC_DATASETS)
    params: ローダへの追加引数(binarize_threshold / repeat / limit / root 等)
    """

    name: str = 'mnist'
    batch_size: int = Field(default=128, ge=1)
    params: dict[str, Any] = Field(default_factory=dict)


class LayerConnectionsConfig(_StrictModel):
    """1層分の配線指定(先行実験。`ConnectionsConfig.kind='per_layer'` の要素)。

    kind:
      rot            — (g+s_a) mod in_dim / (g+s_b) mod in_dim。**s_a / s_b 必須**
                       (層ごとの単一値。リストではない)
      stretched_rot  — 基準位置を round(g·in_dim/out_dim) に引き伸ばした rot。
                       **s_a / s_b 必須**。in_dim == out_dim では rot と一致
      perm           — 均衡全射(配線用 generator を消費)。s_a / s_b は無視
      random_seeded  — get_random_connections を配線用 generator で引く。同上
                       (**均衡する**。無制約の乱択ではない。生成器の docstring 参照)
      random_uniform — 各スロット独立一様の無制約乱択(配線用 generator)。同上

    構造化固定配線(乱数を引かない閉形式。移植元は内部調査リポジトリ
    先行実験):
      rot_n          — n 入力 rot: idx[m] = (g·step + shifts[m]) mod in_dim。
                       **shifts 必須**(長さ = model.n_inputs)
      butterfly      — 循環 radix-n バタフライ 1 段: (g + m·stride) mod in_dim。
                       **stride 必須**
      local1d        — 1D 局所窓: (c_g + m·stride) mod in_dim(c_g = 比例写像)。
                       **stride 必須**
      local2d        — 2D 局所窓: 格子 grid(None = 正方)× bits_per_cell ビットのセルを
                       offsets [(dx, dy), …] で読む。**offsets 必須**、
                       len(offsets) · bits_per_cell == model.n_inputs
    rot / rot_n は step(歩進、既定 1。先行実験 rot2 の r)を取る。dedupe(既定 false)は
    rot_n / butterfly / local1d / local2d の重複ピンを +1 ずらしで潰す
    (rot 系・乱択系には適用しない = 既存レシピは不変)。
    """

    kind: Literal['rot', 'stretched_rot', 'perm', 'random_seeded',
                  'random_uniform', 'rot_n', 'butterfly', 'local1d', 'local2d']
    s_a: int | None = None
    s_b: int | None = None
    step: int = Field(default=1, ge=1)
    shifts: list[int] | None = None
    stride: int | None = None
    offsets: list[tuple[int, int]] | None = None
    grid: tuple[int, int] | None = None
    bits_per_cell: int = Field(default=1, ge=1)
    dedupe: bool = False

    @model_validator(mode='after')
    def _check_shifts(self) -> LayerConnectionsConfig:
        if self.kind in ('rot', 'stretched_rot') and (
                self.s_a is None or self.s_b is None):
            raise ValueError(
                f"connections.layers の kind='{self.kind}' は s_a / s_b"
                f'(整数)が必要です'
            )
        if self.kind == 'rot_n' and not self.shifts:
            raise ValueError("connections.layers の kind='rot_n' は shifts(整数の"
                             'リスト)が必要です')
        if self.kind in ('butterfly', 'local1d') and self.stride is None:
            raise ValueError(
                f"connections.layers の kind='{self.kind}' は stride(整数)が必要です")
        if self.kind == 'local2d' and not self.offsets:
            raise ValueError("connections.layers の kind='local2d' は offsets"
                             '([dx, dy] のリスト)が必要です')
        return self


class ConnectionsConfig(_StrictModel):
    """規則配線の設定(先行実験)。文字列 'random' / 'unique' の代わりに指定する。

    kind:
      rot  — idx_a[g]=(g+s_a) mod in_dim, idx_b[g]=(g+s_b) mod in_dim。
             s_a / s_b は**層数と同じ長さ**のリスト(層ごとのずらし量)。
      perm — 2·out_dim スロットを前層出力が等回数使う均衡全射(層ごとに
             run.seed から導いた generator で生成)。**s_a / s_b は無視される**
             (レシピを `model.connections.kind=perm` の1行上書きで
             置換配線へ切り替えられるようにするため)。
      random_seeded  — 'random' と同じ生成器を run.seed 由来の generator で
             引く(先行実験)。s_a / s_b は無視される
      random_uniform — 各スロット独立一様の無制約乱択(先行実験)。同上
      per_layer — 層ごとに配線種を変える(先行実験)。**layers 必須**で、
             長さは model.layers と一致していなければならない。s_a / s_b
             (リスト)は使わず、各層の LayerConnectionsConfig が持つ

    構造化固定配線(全層を閉形式で決める。1 層ずつ指定したいときは
    per_layer の rot_n / butterfly / local1d / local2d を使う):
      butterfly — 基数 = model.n_inputs。層 0 = gather(idx[m] = (g·n + m) mod in_dim、
             n·k ≥ in_dim なら全入力を被覆)、層 l ≥ 1 = (g + m·stride_l) mod k。
             strides(長さ layers−1)を省略すると n^(l−1)、order='desc' で逆順。
             内部調査リポジトリの先行実験の BF6(n=6)/ BF4(n=4)
      local1d — 層ごとの strides(長さ layers、**必須**)で (c_g + m·stride_l) mod in_dim。
             先行実験候補 B。`core.connections.local1d_strides` が幾何級数列を作る
      local2d — 層 0 は画像格子 grid(H, W)× bits_per_cell(サーモメータ z)を
             layer0_offsets で読む(省略時: n_inputs == bits_per_cell なら自セル [(0,0)]、
             n_inputs == 2·bits_per_cell なら右隣・下隣 [(1,0),(0,1)])。隠れ層は
             ceil(√k) の正方トーラス格子に ring 列を assign(contig / stride)で割り当てる。
             先行実験の L2D6 / L2D4 / 候補 C
    rot は step(層ごとの歩進のリスト、省略 = 全層 1。先行実験 rot2)を取る。
    dedupe(既定 false)は構造化配線の重複ピンを +1 ずらしで潰す(先行実験の build() 相当)。
    """

    kind: Literal['rot', 'perm', 'random_seeded', 'random_uniform',
                  'per_layer', 'butterfly', 'local1d', 'local2d']
    s_a: list[int] | None = None
    s_b: list[int] | None = None
    step: list[int] | None = None
    layers: list[LayerConnectionsConfig] | None = None
    strides: list[int] | None = None
    order: Literal['asc', 'desc'] = 'asc'
    grid: tuple[int, int] | None = None
    bits_per_cell: int = Field(default=1, ge=1)
    layer0_offsets: list[tuple[int, int]] | None = None
    assign: Literal['contig', 'stride'] = 'contig'
    dedupe: bool = False


class ConvDLGNModelConfig(_StrictModel):
    """`model.kind='conv_dlgn'` の conv 部と頭部の形(内部調査リポジトリ先行実験
    ConvLogicNet / cdlgn)。ゲートのパラメータ化・推定器・RI・GroupSum の τ と割当は
    ModelConfig 側(parameterization / estimator / residual_init / ri_mu / ri_sigma /
    selection / tau / group_assign)を conv 木と頭部に共通で使う。

      in_shape                — 入力 (C, H, W)。要素数はデータの入力幅と一致すること
                                (CIFAR-10 3 閾値は閾値 major で [9, 32, 32])
      channels                — conv 木ブロックの出力チャネル列(cdlgn k=32: [32, 128, 512, 1024])
      sampler                 — 葉の読み座標。random(従来)/ random8(cdlgn 忠実)/
                                cross5 / dilated8 / checker13(座標が (チャネル, 層) の計算可能関数)
      kernel_size / tree_depth / padding — sampler=random のときだけ(他は腕ごとに固定)
      input_channels_per_tree — 木 1 本が読む入力チャネル数(cdlgn num_chn)
      downsample              — orpool(stride-1 conv + OR プール 2×2)/ stride(stride-2 conv)
      head_dims               — 頭部の隠れ幅列(入力 flat を含まない)。省略で cdlgn 頭部則
                                [10·flat, 5·flat, 2.5·flat] × head_scale
      head_wiring — rot2(歩進付き rot、先行実験 rot2_schedule)/ random / unique
    """

    in_shape: tuple[int, int, int]
    channels: list[int] = Field(default_factory=lambda: [32, 128, 512, 1024])
    sampler: Literal['random', 'random8', 'cross5', 'dilated8', 'checker13'] = 'random8'
    kernel_size: int = Field(default=3, ge=1)
    tree_depth: int = Field(default=3, ge=1)
    padding: int = Field(default=1, ge=0)
    input_channels_per_tree: int = Field(default=2, ge=1)
    downsample: Literal['orpool', 'stride'] = 'orpool'
    head_dims: list[int] | None = None
    head_wiring: Literal['rot2', 'random', 'unique'] = 'rot2'
    head_scale: float = Field(default=1.0, gt=0)


class ModelConfig(_StrictModel):
    """モデル設定。

    kind:
      dlgn      — Flatten + LutLayer×layers + GroupSum(classification用)
      logic_mlp — Flatten + LutLayer×layers(GroupSumなし、binary_vector用)
      conv_dlgn — LutConv2d(+OrPooling2d)×段 + Flatten + LutLayer×段 + GroupSum
                  (classification用。形は conv: ConvDLGNModelConfig)

    先行実験で追加(いずれも既定は従来動作):
      selection        — 訓練時のゲート選択('soft'/'hard_st'/'gumbel_st')
      init_mode        — 16ゲートモードのロジット初期化(None=従来のrandn)
      init_scale       — 残差初期化('ri'/'ri_randn'/'rri')の加算量(既定5.0。
                         社内先行実験 Phase 3 の OP 腕は 1.0)
      grad_factor      — 勾配スケーリング係数
      gate_mask_hidden — 隠れ層で**除外する**ゲートIDの集合
      gate_mask_final  — 最終LutLayerで除外するゲートIDの集合
      connections      — 'random' / 'unique' / ConnectionsConfig
                         (rot / perm / 先行実験で random_seeded / random_uniform /
                          per_layer=層ごとの指定を追加)
    """

    kind: Literal['dlgn', 'logic_mlp', 'conv_dlgn'] = 'dlgn'
    conv: ConvDLGNModelConfig | None = None   # kind='conv_dlgn' のとき必須
    k: int = Field(default=2000, ge=1)  # 層あたりニューロン数
    layers: int = Field(default=3, ge=1)  # LUT層数
    n_inputs: Literal[2, 4, 6] = 2
    parameterization: Literal['softmax', 'independent', 'iwp', 'raw', 'warp'] = 'softmax'
    connections: Literal['random', 'unique'] | ConnectionsConfig = 'random'
    tau: float = Field(default=1.0, gt=0)  # GroupSum温度(dlgnのみ使用)
    # GroupSum のクラス割当(dlgn のみ使用): contig = ニューロン i → クラス i // m
    # (既定、従来と bitwise 同一)、stride = i → i mod k(内部調査リポジトリ先行実験の既定)
    group_assign: Literal['contig', 'stride'] = 'contig'
    # エントリ確率の推定器ρ(parameterization='independent'=論文IWPのみ有効)。
    # 'sin'は arXiv:2510.03250 §C.1.1 の sin01(Ω)=0.5+0.5·sin(Ω)
    estimator: Literal['sigmoid', 'sin'] = 'sigmoid'
    # residual initialization(arXiv:2510.03250 §C.1.1)。既定Falseは従来挙動
    residual_init: bool = False
    ri_mu: float = 1.2  # residual initializationの平均μ(LutLayerの既定と一致)
    ri_sigma: float = 0.25  # residual initializationの標準偏差σ(同上)
    selection: Literal['soft', 'hard_st', 'gumbel_st'] = 'soft'
    init_mode: Literal['randn', 'ri', 'ri_randn', 'rri', 'uniform'] | None = None
    init_scale: float = Field(default=5.0, gt=0)
    grad_factor: float = Field(default=1.0, gt=0)
    gate_mask_hidden: list[int] | None = None
    gate_mask_final: list[int] | None = None

    @field_validator('gate_mask_hidden', 'gate_mask_final')
    @classmethod
    def _check_gate_mask(cls, v: list[int] | None) -> list[int] | None:
        if v is None:
            return v
        if any(not 0 <= g <= 15 for g in v):
            raise ValueError(f'ゲートマスクのIDは0〜15の範囲が必要です: {v}')
        if len(set(v)) >= 16:
            raise ValueError('16種すべてを除外することはできません')
        return v

    @model_validator(mode='after')
    def _check_conv(self) -> ModelConfig:
        if self.kind == 'conv_dlgn':
            if self.conv is None:
                raise ValueError("model.kind='conv_dlgn' は model.conv(in_shape 等)が必要です")
            if self.n_inputs != 2:
                raise ValueError(
                    f"model.kind='conv_dlgn' は n_inputs=2 専用です(指定: {self.n_inputs})")
            if self.parameterization not in ('softmax', 'independent', 'iwp'):
                raise ValueError(
                    "model.kind='conv_dlgn' の parameterization は softmax / independent / "
                    f'iwp のいずれか(指定: {self.parameterization!r})')
        elif self.conv is not None:
            raise ValueError("model.conv は model.kind='conv_dlgn' のときだけ指定できます")
        return self

    @model_validator(mode='after')
    def _check_connections(self) -> ModelConfig:
        conn = self.connections
        if not isinstance(conn, ConnectionsConfig):
            return self
        if conn.kind == 'rot':
            if conn.s_a is None or conn.s_b is None:
                raise ValueError(
                    "connections.kind='rot' は s_a / s_b(層数と同じ長さの"
                    'リスト)が必要です'
                )
            if len(conn.s_a) != self.layers or len(conn.s_b) != self.layers:
                raise ValueError(
                    f'connections.s_a / s_b の長さは layers({self.layers})と'
                    f'一致が必要です: s_a={len(conn.s_a)}, s_b={len(conn.s_b)}'
                )
            if conn.step is not None and len(conn.step) != self.layers:
                raise ValueError(
                    f'connections.step の長さは layers({self.layers})と'
                    f'一致が必要です: {len(conn.step)}'
                )
        elif conn.kind == 'per_layer':
            if conn.layers is None:
                raise ValueError(
                    "connections.kind='per_layer' は layers(層ごとの"
                    '{kind, s_a, s_b} のリスト)が必要です'
                )
            if len(conn.layers) != self.layers:
                raise ValueError(
                    f'connections.layers の長さは layers({self.layers})と'
                    f'一致が必要です: {len(conn.layers)}'
                )
            for i, spec in enumerate(conn.layers):
                if spec.kind == 'rot_n' and len(spec.shifts) != self.n_inputs:
                    raise ValueError(
                        f'connections.layers[{i}] の shifts の長さは '
                        f'n_inputs({self.n_inputs})と一致が必要です: {len(spec.shifts)}'
                    )
                if (spec.kind == 'local2d'
                        and len(spec.offsets) * spec.bits_per_cell != self.n_inputs):
                    raise ValueError(
                        f'connections.layers[{i}] の len(offsets)·bits_per_cell = '
                        f'{len(spec.offsets)}·{spec.bits_per_cell} は '
                        f'n_inputs({self.n_inputs})と一致が必要です'
                    )
        elif conn.kind == 'butterfly':
            if conn.strides is not None and len(conn.strides) != self.layers - 1:
                raise ValueError(
                    f"connections.kind='butterfly' の strides は層 1 以降の分"
                    f'(長さ layers−1 = {self.layers - 1})が必要です: {len(conn.strides)}'
                )
        elif conn.kind == 'local1d':
            if conn.strides is None or len(conn.strides) != self.layers:
                raise ValueError(
                    "connections.kind='local1d' は strides(層ごとのストライド、"
                    f'長さ layers = {self.layers})が必要です'
                )
        elif conn.kind == 'local2d':
            if conn.layer0_offsets is None:
                if self.n_inputs not in (conn.bits_per_cell, 2 * conn.bits_per_cell):
                    raise ValueError(
                        "connections.kind='local2d' で layer0_offsets を省略できるのは"
                        ' n_inputs == bits_per_cell(自セル)か 2·bits_per_cell'
                        f'(右隣・下隣)のときだけです: n_inputs={self.n_inputs}, '
                        f'bits_per_cell={conn.bits_per_cell}'
                    )
            elif len(conn.layer0_offsets) * conn.bits_per_cell != self.n_inputs:
                raise ValueError(
                    f'connections.layer0_offsets の len·bits_per_cell = '
                    f'{len(conn.layer0_offsets)}·{conn.bits_per_cell} は '
                    f'n_inputs({self.n_inputs})と一致が必要です'
                )
        return self


class OptimConfig(_StrictModel):
    """オプティマイザ設定"""

    name: Literal['adam', 'adamw', 'sgd'] = 'adam'
    lr: float = Field(default=0.01, gt=0)


class RegConfig(_StrictModel):
    """正則化設定(先行実験)。省略(None)時は正則化なし=従来動作。

    kind:
      irreducibility — 縮約されやすいゲート種(定数・pass-through)へ置かれた
        softmax確率質量のペナルティ R = (1/N_h) Σ_n Σ_i p_{n,i}·c_i。
        隠れ層は {0,15,3,5,10,12}、最終LutLayerは {3,5,10,12} に c=1。
    lambda: 正則化係数(YAMLキーは `lambda`。Python予約語のためフィールド名は
        lambda_)。0 なら損失に加算しない(**学習は正則化なしと bitwise 不変**)が、
        指標 reg_penalty は記録する。
    ramp_frac: 学習ステップ全体に対する線形ランプ区間の割合。
        step ∈ [0, ramp_frac·total_steps] で 0 → lambda に線形、以降は lambda 固定。
        0 なら最初から定数 lambda。
    """

    model_config = ConfigDict(extra='forbid', populate_by_name=True)

    kind: Literal['irreducibility'] = 'irreducibility'
    lambda_: float = Field(default=0.0, ge=0, alias='lambda')
    ramp_frac: float = Field(default=0.5, ge=0, le=1)


class RunConfig(_StrictModel):
    """実行設定。

    eval_interval: 0=エポック末のみhard eval / N>0=Nステップごと
    loss: None=タスク既定(classification→ce / binary_vector→bce)
    device: None=自動検出(cuda→mps→cpu)

    wandb_* はレシピからwandbのtaxonomy(experiments/wandb_utils.py の
    `init_wandb`)を指定するためのもの。**既定値は従来のランナーの挙動と同一**
    (group='md-train' / job_type='train' / 追加タグなし / notes・hypothesis なし)
    なので、既存レシピの記録内容は変わらない。
      wandb_group      — runのグループ名。**モデル族**を指定する(例 'dlgn-ffn')
      wandb_job_type   — ジョブ種別。本番学習と補助runを区別する
                         (語彙は training-script skill の train / ablation / sweep /
                         eval / smoke。較正アンカーの再走は 'train' + tag 'calibration')
      wandb_tags       — 横断ラベル。ランナー既定のタグへ**追加**する
                         (既定タグ: 'md-train' / model.kind / データ種別)
      wandb_notes      — この実験で何をなぜ変えたか(wandbのnotesへ)
      wandb_hypothesis — 検証したい仮説(configの experiment/hypothesis へ)
    """

    seed: int = 42
    epochs: int = Field(default=3, ge=1)
    eval_interval: int = Field(default=0, ge=0)
    wandb: bool = False
    wandb_group: str = 'md-train'
    wandb_job_type: Literal['train', 'ablation', 'sweep', 'eval', 'smoke'] = 'train'
    wandb_tags: list[str] = Field(default_factory=list)
    wandb_notes: str | None = None
    wandb_hypothesis: str | None = None
    run_name: str | None = None
    method: str = 'bp'  # 現状ランナーはbpのみ対応(ap/dfa等は将来拡張)
    loss: str | None = None
    device: str | None = None
    # 先行実験: いずれも既定Falseで従来と同じ出力(config.yaml + checkpointsのみ)
    export_netlist: bool = False    # 学習後に <out_dir>/netlist.json を保存
    save_eval_history: bool = False  # eval履歴 + summary.json を保存


class DistillConfig(_StrictModel):
    """知識蒸留(C4)設定。

    TrainConfig.distill に設定されると、runner が classification タスクの学習を
    KD(convert/distill.make_distill_task)へ差し替える。省略(None)時は従来の
    素の分類学習(後方互換)。

    teacher: resolve_teacher(convert/teacher.py)の source 文字列
        (例 'torchvision:resnet18' / 'ckpt:path.pt')
    alpha: hard(CE)項の重み(0=純KD/1=純CE)
    temperature: KD温度 T(二重温度 τ_gs×T の T 成分。τ_gs はモデル側 GroupSum.tau)
    standardize: KD項の両ロジットを Z-score 標準化するか
    kind: 'kl'(既定)または 'mse'(KL fallback)
    teacher_transform: TEACHER_TRANSFORMS のキー('none'/'mnist'/'cifar10'/'binarize05')。
        teacher に渡す入力への正規化。teacher解決後に上書き適用する
    cache: teacherロジットをキャッシュ(data-teacher-cache/)するか。
        False なら毎回再計算(fingerprint/params が同じでもロードしない)
    """

    teacher: str
    alpha: float = Field(default=0.5, ge=0, le=1)
    temperature: float = Field(default=4.0, gt=0)
    standardize: bool = False
    kind: Literal['kl', 'mse'] = 'kl'
    teacher_transform: str = 'none'
    cache: bool = True
    # 完全ラベルフリー変換: hard項のラベルを正解でなくteacher argmax(疑似
    # ラベル)にする。正解ラベルが存在しない状況(LLM変換の実態)でのα>0安定化用。
    # 真の精度は評価系(_hard_eval、正解ラベル)で別途測られる
    pseudo_labels: bool = False


class TrainConfig(_StrictModel):
    """学習設定のルート(data/model/optim/run/expected_metrics)。

    expected_metrics: レシピの期待メトリクス照合。
    キーは '<metric>_min' / '<metric>_max'(metricはrun_trainingの返り値キー。
    例 hard_acc_min / final_loss_max)。runner終了時に照合結果をprintし、
    返り値の 'expected_metrics' キーへ格納する(未達でも例外にはしない。
    `griffith train --strict` のみ未達で非0 exit)。
    """

    data: DataConfig = Field(default_factory=DataConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    optim: OptimConfig = Field(default_factory=OptimConfig)
    run: RunConfig = Field(default_factory=RunConfig)
    distill: DistillConfig | None = None
    # 先行実験: 縮約耐性正則化。省略(None)時は損失に何も足さない=従来動作
    regularization: RegConfig | None = None
    expected_metrics: dict[str, float] = Field(default_factory=dict)

    @field_validator('expected_metrics')
    @classmethod
    def _check_expected_keys(cls, v: dict[str, float]) -> dict[str, float]:
        for key in v:
            if not (key.endswith('_min') or key.endswith('_max')):
                raise ValueError(
                    f'expected_metricsのキーは <metric>_min / <metric>_max 形式が'
                    f'必要です: {key!r}(例: hard_acc_min, final_loss_max)'
                )
        return v


def _set_by_dots(tree: dict, dotted_key: str, raw_value: str) -> None:
    """ドット記法キー(a.b.c)でネスト辞書に値を書き込む。

    値はYAMLスカラーとして解釈する(42→int / 0.5→float / true→bool)。
    型の妥当性はpydanticのバリデーションに委ねる。
    """
    keys = dotted_key.split('.')
    if not dotted_key or any(not k for k in keys):
        raise ValueError(f'不正なオーバーライドキー: {dotted_key!r}')
    node = tree
    for key in keys[:-1]:
        child = node.setdefault(key, {})
        if not isinstance(child, dict):
            raise ValueError(
                f'オーバーライドキー {dotted_key!r} の途中要素 {key!r} が'
                f'マッピングではありません: {child!r}'
            )
        node = child
    node[keys[-1]] = yaml.safe_load(raw_value) if raw_value != '' else None


def load_config(
    path: str | None = None, overrides: Sequence[str] = ()
) -> TrainConfig:
    """YAML(省略可)+ドット記法オーバーライドから設定を構築・検証する。

    Args:
        path: 設定YAMLのパス(Noneなら既定値から開始)
        overrides: 'a.b=c' 形式の上書きリスト

    Raises:
        ValueError: オーバーライド書式不正・YAMLルートが非マッピング
        pydantic.ValidationError: 未知キー・型不整合
    """
    data: dict = {}
    if path is not None:
        with open(path, encoding='utf-8') as f:
            loaded = yaml.safe_load(f)
        if loaded is None:
            loaded = {}
        if not isinstance(loaded, dict):
            raise ValueError(f'設定YAMLのルートはマッピングが必要です: {path}')
        data = loaded

    for item in overrides:
        if '=' not in item:
            raise ValueError(f'オーバーライドは key=value 形式で指定してください: {item!r}')
        key, _, value = item.partition('=')
        _set_by_dots(data, key.strip(), value)

    return TrainConfig.model_validate(data)


def check_expected_metrics(expected: dict[str, float], result: dict) -> dict:
    """期待メトリクスをrun_trainingの結果と照合する。

    Args:
        expected: {'<metric>_min': 値, '<metric>_max': 値} 形式の期待値
        result: run_trainingの返り値(hard_acc / final_loss 等を含む辞書)

    Returns:
        {'ok': 全件合格か, 'checks': [{'metric','bound','expected','actual','ok'}]}
        結果に存在しないメトリクスへの期待は未達(ok=False, actual=None)扱い。
    """
    checks = []
    for key, threshold in expected.items():
        metric, bound = key[:-4], key[-3:]  # '_min'/'_max' を分離
        actual = result.get(metric)
        if actual is None:
            ok = False
        elif bound == 'min':
            ok = float(actual) >= threshold
        else:
            ok = float(actual) <= threshold
        checks.append({
            'metric': metric, 'bound': bound,
            'expected': float(threshold),
            'actual': None if actual is None else float(actual),
            'ok': ok,
        })
    return {'ok': all(c['ok'] for c in checks), 'checks': checks}


def dump_config(cfg: TrainConfig) -> dict:
    """設定をYAML/JSON安全なdictへ変換する"""
    return cfg.model_dump(mode='json')


def save_config(cfg: TrainConfig, path: str) -> str:
    """設定をYAMLで保存する(`griffith train <path>` でそのまま再実行できる形式)"""
    with open(path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(
            dump_config(cfg), f, allow_unicode=True, sort_keys=False,
            default_flow_style=False,
        )
    return path
