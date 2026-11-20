"""実行プラン — バックエンド非依存のグラフ解析。

これまで C / CUDA / Metal の3バックエンドが**同じグラフ走査・形推論・検証**を
それぞれ持っていた(計1,600行、ノード型1つ足すのに3ファイル編集)。ここに1回だけ
書き、バックエンドは「op種別 → コード生成」の表を埋めるだけにする。

    IR + バンドル
       ↓  build_plan()          ← グラフ走査・形推論・検証・op列の生成
       ↓  ExecPlan              ← 値と op の並び(バッファは未確定)
       ├→ backends/c.py
       ├→ backends/cuda.py
       └→ backends/metal.py

**最適化(メモリ計画・バッチ・命令列融合)はこの層に書く**。そうすれば
ノード型やバックエンドが増えても最適化は増殖しない。

op の共通語彙(IRノード型より粗い = バックエンドが実装する単位):

    pack / unpack     ビット列 ⇄ ビットプレーン
    preproc           画像 → ビットプレーン(記述子駆動)
    gate_program      ゲート木の命令列(いまは conv2d。将来ほかのゲート木ノードも)
    plane_op          ビットプレーン → ビットプレーン(pool/upsample/permute/concat)
    reduce_scores     ビットプレーン → 整数スコア(group_sum / SpatialGroupSum融合)
    lut_dense         非パックのLUT層
    lut_flat          縮約済み dense 平坦ネットリスト(表駆動。全バックエンド、
                      Web/GPU も対応。GPU は段分けして段ごとに並列。GPU の CLI 既定は
                      平坦化しない = opt-in)
    copy_scores       スコア列の連結・最終出力

新しいIRノード型の大半は既存 op へ写すだけで済む(バックエンド無改造)。
"""

from dataclasses import dataclass, field

from ..ir.schema import IRGraph
from ..ir.serialize import ir_from_dict
from .layout import build_conv_layout_split, words_per_row

# 値の種別
KIND_BITPLANE = 'bitplane'
KIND_FLAT = 'flat'
KIND_SCORES = 'scores'
#: op が自分の実行中だけ使う作業領域(ワード単位。`width` にワード数を持つ)。
#: 生存区間が1つの op に閉じるので、メモリ計画が全部まとめて畳んでくれる。
KIND_SCRATCH = 'scratch'


class UnsupportedNode(ValueError):
    """デプロイキットが対応していないノードの形。

    「どのノードが・なぜ駄目か」をこの1箇所の型で表す(以前は3バックエンドに
    51箇所散っていた)。バックエンド固有の制約は `backend` に入れる。
    """

    def __init__(self, message: str, *, node_id: str = '', node_type: str = '',
                 backend: str = ''):
        self.node_id = node_id
        self.node_type = node_type
        self.backend = backend
        where = f'{backend}: ' if backend else ''
        what = f'ノード {node_id}({node_type}): ' if node_id else ''
        super().__init__(f'{where}{what}{message}')


@dataclass
class PlanValue:
    """プラン上の値。バッファの実体はバックエンドが割り当てる。"""

    id: int
    kind: str
    c: int = 0
    h: int = 0
    w: int = 0
    width: int = 0
    #  由来のノードid(デバッグ表示用)
    origin: str = ''
    #:: この値が独立バッファを持たず、`alias_of`の値の内部
    #: `alias_offset`バイト目からの「親バッファ+オフセット」として存在する
    #: ことを示す(Noneなら通常どおり独立割り当て)。RDDLGNの退化ゲート除去
    #:が中間層の出力をアキュムレータへ物理コピー無しで追記するために
    #: 使う(`_plan_accumulate_concat`が設定する)。memplanはこの値を
    #: プールから除外し(独自領域を持たない)、Cバックエンドは
    #: `buf(alias_of) + alias_offset`として参照する(cbackend.py参照)。
    #: GPUバックエンド(CUDA/Metal)は現状alias値を「未対応」として明示拒否
    #: する(recurrent/accumulate系のノードがGPU側でdormantなため)。
    alias_of: int | None = None
    #: `alias_of`内でのバイトオフセット(KIND_FLAT=1バイト/要素前提)。
    alias_offset: int = 0

    @property
    def wpr(self) -> int:
        return words_per_row(self.w)

    @property
    def words(self) -> int:
        """ビットプレーンのワード数。"""
        return self.c * self.h * self.wpr

    def nbytes(self) -> int:
        if self.kind == KIND_BITPLANE:
            return self.words * 4
        if self.kind == KIND_FLAT:
            return self.width
        return self.width * 4    # scores(int32) / scratch(u32ワード)


@dataclass
class PlanOp:
    """1つの実行単位。`kind` がバックエンドの実装単位。"""

    kind: str
    inputs: list[int]
    output: int
    attrs: dict = field(default_factory=dict)
    node_id: str = ''
    node_type: str = ''
    #: 空文字列なら「主経路」の op。recurrentノードのbodyから生まれたopは
    #: 生成元のrecurrentノードidが入る(`_plan_recurrent`が設定
    #: する)。バックエンドはこれを見て主経路のop列とは別の(ステップごとに
    #: 複数回呼ばれる)関数へ切り出す。
    scope: str = ''

    def __repr__(self) -> str:      # デバッグ表示
        sub = self.attrs.get('op', '')
        label = f'{self.kind}/{sub}' if sub else self.kind
        return f'<{label} {self.inputs}->{self.output} {self.node_id}>'


@dataclass
class ExecPlan:
    """バックエンド非依存の実行プラン。"""

    values: list[PlanValue]
    ops: list[PlanOp]
    input_value: int
    output_value: int
    input_width: int
    output_width: int
    labels: list[str] = field(default_factory=list)
    #  画像前処理(無ければ None)
    preproc: dict | None = None
    #  前処理の出力先とグラフ入力のパック先が同一バッファのとき、その値id
    input_bitplane: int | None = None
    #  由来メタ(ヘッダのコメント用)
    meta: dict = field(default_factory=dict)
    #: 同時に流す枚数の上限。バッファはこの倍数で確保する。
    #: 実行時の枚数は n <= batch で可変(起動スレッド数を n で決める)。
    batch: int = 1
    #: RDDLGN自己回帰生成用の埋め込み表(無ければNone)。
    #: {'vocab', 'd_emb', 'bits'}。export/rddlgn_ir.py::extract_embed_table
    #: が作り、bundle['embed_table']からそのまま引き継ぐ(バックエンド非依存
    #: の情報なのでlabels/preprocと同じ扱いにする)。
    embed_table: dict | None = None
    #: RDDLGN語彙折り畳み表(無ければNone)。{'vocab', 'width', 'bits'}。
    #: export/rddlgn_ir.py::extract_folded_n_table/extract_folded_l_tableが
    #: 作り、bundle['folded_n_table']/['folded_l_table']からそのまま引き継ぐ。
    #: 折り畳み対象外(自動判定で見送り、または非RDDLGNバンドル)ならNone
    #: (embed_tableと同型、グラフのop列そのものには一切影響しない — 折り
    #: 畳みはruntime/c/src/mdl_rddlgn.cの生成greedyハーネスだけが使う
    #: 補助テーブルなので、_plan_recurrentのop生成は無改造)。
    folded_n_table: dict | None = None
    folded_l_table: dict | None = None

    def value(self, vid: int) -> PlanValue:
        return self.values[vid]


# ---------------------------------------------------------------------------
# ノード型レジストリ(ir/build.py::register_builder と同じ流儀)
# ---------------------------------------------------------------------------

NODE_KINDS: dict[str, callable] = {}
#  対応していないノード型と、その理由(カバレッジ行列テストが読む)
UNSUPPORTED_NODE_TYPES: dict[str, str] = {}


def register_node_kind(node_type: str):
    """IRノード型 → プラン上の op へ写す関数を登録する。

    関数のシグネチャ: `fn(builder, node, ins: list[PlanValue]) -> PlanValue`
    (`builder` は下の `_PlanBuilder`。op の追加と値の生成を行う)
    """
    def deco(fn):
        NODE_KINDS[node_type] = fn
        return fn
    return deco


def declare_unsupported(node_type: str, reason: str) -> None:
    """対応しないノード型を、理由つきで明示する。

    「未対応」を黙って例外にせず宣言しておくことで、カバレッジ行列テストが
    「全ノード型が対応か理由付き非対応のどちらかである」ことを強制できる。
    """
    UNSUPPORTED_NODE_TYPES[node_type] = reason


# ---------------------------------------------------------------------------
# 形の小道具
# ---------------------------------------------------------------------------

def pair(value) -> tuple[int, int]:
    if isinstance(value, list | tuple):
        return int(value[0]), int(value[1])
    return int(value), int(value)


def channel_order_from_permute(params: dict) -> list[int]:
    """先頭(非空間)軸だけのpermuteを、チャネル順のorder配列へ展開する。

    bitplaneGraph.ts::channelOrderFromPermute のミラー。
    """
    in_shape = [int(v) for v in params['in_shape']]
    perm = [int(v) for v in params['perm']]
    m = len(in_shape) - 2
    lead_dims = in_shape[:m]
    lead_perm = perm[:m]
    total = 1
    for d in lead_dims:
        total *= d
    in_strides = [1] * m
    for i in range(m - 2, -1, -1):
        in_strides[i] = in_strides[i + 1] * lead_dims[i + 1]
    out_dims = [lead_dims[q] for q in lead_perm]
    out_strides = [1] * m
    for i in range(m - 2, -1, -1):
        out_strides[i] = out_strides[i + 1] * out_dims[i + 1]
    order = [0] * total
    for out_idx in range(total):
        rem = out_idx
        src = 0
        for i in range(m):
            coord = rem // out_strides[i]
            rem -= coord * out_strides[i]
            src += coord * in_strides[lead_perm[i]]
        order[out_idx] = src
    return order


def spatial_group_sum_dims(params: dict):
    """SpatialGroupSum由来のpermute(in_shape=[nOut,gs,H,W], perm=[0,2,3,1])か。"""
    in_shape = [int(v) for v in params.get('in_shape', [])]
    perm = [int(v) for v in params.get('perm', [])]
    if len(in_shape) != 4 or perm != [0, 2, 3, 1]:
        return None
    return tuple(in_shape)


def keeps_trailing_spatial(params: dict) -> bool:
    perm = [int(v) for v in params['perm']]
    n = len(perm)
    return n >= 2 and perm[n - 2] == n - 2 and perm[n - 1] == n - 1


def slice_truth_tables(truth_tables, n_inputs: int) -> list[int]:
    """真理値表を最小項ごとの平面へ転置する(ビットスライス表現)。

    返り値の並びは `[2^n_inputs][ceil(len/32)]` で、
    語 `(m, j)` のビット `l` がゲート `j*32+l` の真理値表のビット `m`。

    ゲートあたり4バイト持つ代わりに 2^n ビット(2入力なら0.5バイト)で済み、
    32ゲートぶんの真理値表が数語に収まる。**種類の違う32ゲートを1組の
    ビット演算で評価する**ための下ごしらえで、ランタイム側の
    `mdl_lut_layer_slice` / `mdl_lut_layer_packed` が消費する。
    """
    tts = [int(t) for t in truth_tables]
    wpr = (len(tts) + 31) // 32
    out = [0] * ((1 << n_inputs) * wpr)
    for g, tt in enumerate(tts):
        w, bit = divmod(g, 32)
        for m in range(1 << n_inputs):
            if (tt >> m) & 1:
                out[m * wpr + w] |= 1 << bit
    return out


#: この出力幅(バイト)を超える lut_dense だけ並べ替える。
#:
#: 実測(Apple M3、3層の合成モデル、a添字の昇順に並べ替え):
#:
#:   幅 6,000(6KB)      1.00×      幅 400,000(391KB)   **1.27×**
#:   幅 32,000(31KB)    1.02×      幅 1,500,000(1.4MB) **1.30×**
#:   幅 120,000(117KB)  1.05×
#:
#: **効き始めるのは層の出力がL1を溢れたところから**。それ以下では
#: 集めに行く先が全部L1に居るので、距離を縮めても命令数が変わらない。
#: `should_fuse` の `FUSE_POOL_BYTES` と同じ「L1に収まるか」判定である。
REORDER_MIN_BYTES = 128 * 1024

#: 添字がuint16_tに収まる上限。lut_dense の添字はこの層の**入力**
#: (1つ前の層の出力幅、in_width)内の位置を指すので、判定基準は
#: out_width ではなく in_width。値域は 0..in_width-1 なので、
#: in_width <= 65536 なら最大添字 65535 が uint16_t に収まる。
#:
#:実測(paper_scale1、in_width境界の取り方で対象層が変わる):
#: 32768未満で区切ると添字の3.1%しかu16化できないが、正しい境界
#: (65536以下)だと12.9%まで伸びる。3モデル(copy_tiny/reverse_deep/
#: charlm)は全lut_denseの in_width が最大でも20,000台なので、
#: どちらの境界でも100%u16化できて差が出ない。
MDL_U16_INDEX_MAX = 65536


def lut_dense_indices_fit_u16(in_width: int) -> bool:
    """このlut_denseの添字をuint16_tで持てるか(in_width<=65536)。

    バックエンド非依存の1箇所(プラン層方針)。cbackend/gpubackendの
    双方がこの関数で経路を選ぶので、判定がずれない。
    `export/rddlgn_ir.py::_lutstack_weight_bytes`(折り畳み判定が
    参照する重み見積り)もこの関数を使う — こちらは学習側のtorchモデル
    (LutStack)から直接呼ぶので、ExecPlan/PlanOpは経由しない。
    """
    return int(in_width) <= MDL_U16_INDEX_MAX


def reorder_dense_gates(plan: 'ExecPlan') -> list[int]:
    """配線の近さで lut_dense のゲート順を並べ替える(挙動不変)。

    `mdl_lut_layer_slice` は32ゲートを1群として、群ごとに64本の入力を
    集めてくる。配線は学習が決めたものなので添字はばらばらで、実測では
    32ゲートが27本のキャッシュラインに散っていた。a側の添字の昇順に
    並べ替えると、これが2〜3本に落ちる。

    **出力番号を付け替えるので、その層を読む側の添字を必ず書き換える**。
    書き換え先が lut_dense でない層(group_sum など、順序に意味がある)
    が1つでも読んでいる層は並べ替えない。

    値は1ビットも変わらない。返り値は並べ替えた op のindexのリスト。
    """
    consumers: dict[int, list] = {}
    for op in plan.ops:
        for vid in op.inputs:
            consumers.setdefault(vid, []).append(op)

    done: list[int] = []
    for i, op in enumerate(plan.ops):
        if op.kind != 'lut_dense':
            continue
        w = int(op.attrs['out_width'])
        if w <= REORDER_MIN_BYTES:
            continue
        # 読み手が「lut_dense ただ1つ」でなければ順序を動かせない
        users = consumers.get(op.output, [])
        if len(users) != 1 or users[0].kind != 'lut_dense':
            continue
        n = int(op.attrs['n_inputs'])
        idx = list(op.attrs['indices'])
        order = sorted(range(w), key=lambda g: (idx[g], g))
        if order == list(range(w)):
            continue
        inv = [0] * w
        for slot, g in enumerate(order):
            inv[g] = slot
        new_idx = [0] * (n * w)
        for slot, g in enumerate(order):
            for j in range(n):
                new_idx[j * w + slot] = idx[j * w + g]
        tt = list(op.attrs['truth_tables'])
        op.attrs['indices'] = new_idx
        op.attrs['truth_tables'] = [tt[g] for g in order]
        # 読み手の添字を新しい番号へ写す
        nxt = users[0]
        nxt.attrs['indices'] = [inv[v] for v in nxt.attrs['indices']]
        done.append(i)
    return done


#: 融合後のping-pong活性2本(1本あたり = 鎖に登場する全ての入出力幅の最大)
#: がこのバイト数を超える鎖は融合しない(Apple GPUのthreadgroupメモリ
#: 上限32KBに収まる幅だけを対象にする)。CUDAも同じ定数を使う(shared
#: メモリの一般的な上限も48KB既定/最大228KB可変だが、Metal側の制約に
#: 揃えておけば両バックエンドで同じプランがそのまま使い回せる)。
DENSE_FUSE_MAX_SHARED_BYTES = 32 * 1024

#: 融合鎖の最大層数。由来はMetalのカーネル引数テーブル上限。
#:
#: `densefuse.build_metal_kernel` は buffer(0)=入力・buffer(1)=出力に加え、
#: i層目(0始まり)へ buffer(2+2i)=添字表・buffer(3+2i)=真理値表を割り当てる
#: (densefuse.py参照)。Metalのカーネル引数テーブルは index 0..30(31本)
#: までしか無いので、最後の層(i = n_layers-1)の buffer 番号
#: 3+2*(n_layers-1) = 2*n_layers+1 が30を超えてはならない ─
#: 2*n_layers+1 <= 30  ⇔  n_layers <= 14.5  ⇔  n_layers <= 14。
#: 15層以上だと buffer(31) 以降を要求し、Metalは実行時コンパイル
#: (`newLibraryWithSource:`)なのでこれが `mdl_metal_init` の初期化失敗
#: (戻り値-1)として実行時にしか発覚しない。
#:
#: CUDAは引数を直渡しするのでこの制約自体は無いが、融合鎖の判定は
#: バックエンド非依存の1箇所に置く方針(`lut_dense_indices_fit_u16`と
#: 同じ思想)なので、両バックエンド共通でこの上限を使う。
DENSE_FUSE_MAX_LAYERS = 14

#: dense-fuseのping-pong活性2本ぶんの上限をCUDA向けに緩めた値
#:。`DENSE_FUSE_MAX_SHARED_BYTES`(32KB)はApple GPUのthreadgroup
#: メモリ上限からの流用で、CUDAのstatic `__shared__`既定上限(48KB、
#: sm_89含めopt-in無しで確実に使える共通の下限)より小さい。CUDAだけこの
#: 値を使うことで、Metalでは融合対象外(32KB超)のモデル(例:
#: mnist_adderの層幅、ping-pong2本で約45.9KB)もCUDAでは融合できる。
#: `densefuse.build_cuda_kernel`は`__shared__ unsigned char bufA[act_bytes]`
#: という静的配列を生成するので、2*act_bytes<=48KBに収まる限り追加の
#: dynamic shared mem opt-inは不要。
#:: `resolve_auto_config`がバックエンド別のdense_fuse共有メモリ上限を
#: 選ぶのに使うため、CUDA固有の値だがここ(バックエンド非依存のplan.py)に
#: 置く(`cudabackend.py`から取り込むとplan.py→cudabackend.py→plan.pyの
#: 循環importになるため)。`cudabackend.py`はここから再exportする
#: (`from .plan import CUDA_DENSE_FUSE_MAX_SHARED_BYTES`、外部の既存import
#: `from griffith_logic.deploy.cudabackend import
#: CUDA_DENSE_FUSE_MAX_SHARED_BYTES`はそのまま動く)。
CUDA_DENSE_FUSE_MAX_SHARED_BYTES = 48 * 1024


def dense_fuse_act_bytes(layers: list[dict]) -> int:
    """融合鎖のping-pong活性バッファ1本ぶんのバイト数。

    鎖の最初の入力幅(1層目の in_width、後続層の out_width より大きい
    ことがある — 実際 har_dlgn_k600/k2400/dwn は in_width=4608 >
    out_width=600/2400)と、各層の out_width の最大を取る。
    """
    widths = [int(layers[0]['in_width'])]
    widths += [int(layer['out_width']) for layer in layers]
    return max(widths)


def fuse_dense_layers(plan: 'ExecPlan',
                      max_shared_bytes: int = DENSE_FUSE_MAX_SHARED_BYTES,
                      max_layers: int = DENSE_FUSE_MAX_LAYERS,
                      *, dry_run: bool = False
                      ) -> list[str]:
    """連続する lut_dense 列 + 末尾の group_sum を1つの `dense_fuse` op へ
    まとめる(GEM(DAC2025) Boomerang型の層間融合)。

    denseモデル(通常DLGN)は現在、層ごとに別カーネル起動
    (lut_layer→lut_layer→…→group_sum)で、層間の活性がグローバルメモリを
    往復している。全lut層+group_sumを1カーネルへ融合し、活性を
    共有メモリ(Metal: threadgroup、CUDA: shared)に置き、層間はブロック内
    バリアで区切れば、層間の往復とカーネル起動回数が減る
    (最小実験: batch=1ではむしろ悪化する
    ―1スレッドグループしか立たず占有率が下がるため―が、batch>=32では
    最良1.9〜2.5倍)。

    検出条件:
      - lut_dense が2個以上連続していて(各opの出力を読むのが次のopただ1つ)
      - 鎖の直後(次のop)が reduce_scores(op='group_sum')でただ1つ
      - 活性のping-pong2本(`dense_fuse_act_bytes`)が `max_shared_bytes`
        に収まる(収まらない鎖 ─ 例: mnist_adderの層幅45.9KB ─ は対象外
        のまま元の lut_dense 列+group_sum で生成する)
      - **鎖に含まれる全層の `n_inputs == 2` であること**(`densefuse._layer_lines` のaddr式が
        `cur[idx[o]] | cur[idx[ow+o]]<<1` という2入力決め打ちで、
        3入力目以降を一切読まない。鎖の先頭候補・延伸候補のどちらも
        n_inputs!=2なら鎖に含めない ─ `fits` と同じ「収まらなければ
        対象外」の流儀。現行実モデルは全層n_inputs=2なので影響なし、
        mnist_adder等のn_inputs=4混在モデルは既にこの層で弾かれる)
      - **鎖の長さは `max_layers`(既定 `DENSE_FUSE_MAX_LAYERS`=14)以下**
        (超える分は貪欲な鎖延伸をそこで打ち切るので、鎖の終端が
        ちょうど group_sum の直前に来る最大14層の区間だけが融合され、
        それより手前の層は元の lut_dense のまま残る)
      - **鎖・末尾のgroup_sumとも `scope == ''`(主経路)であること**
        (recurrentノードbody内lut_dense/group_sumは
        `scope=recurrent_node_id`を持つ。bodyはバックエンドがステップ
        関数へ切り出す前提の走査単位なので、主経路の融合ロジックを
        跨がせない。dense_fuseはMetal/CUDAのみ対応・recurrentはGPU側で
        現状dormantだが、将来発火したときに誤って融合されないよう
        ここで先に締め出しておく)

    値は1ビットも変わらない(整数厳密一致はテストで固定)。opt-in
    (呼び出し側が明示的に呼んだときだけ適用、`build_plan(dense_fuse=True)`)。

    Args:
        dry_run: Trueなら`plan`を一切書き換えず(`op.output`の幅も
            `plan.ops`の並びも元のまま)、融合できたであろうgroup_sumの
            ノードidだけを返す。`resolve_auto_config`が「dense_fuseを
            自動ONにしたら実際に何か融合されるか」を、生成に使う本物の
            プランを壊さずに下見するために使う(下見用にプランを複製する
            コストを避けるための最小限のフラグ)。

    Returns:
        融合したgroup_sumノードidのリスト(空なら1つも融合しなかった)
    """
    consumers: dict[int, list[int]] = {}
    for i, op in enumerate(plan.ops):
        for vid in op.inputs:
            consumers.setdefault(vid, []).append(i)

    fused_ids: list[str] = []
    new_ops: list[PlanOp] = []
    n = len(plan.ops)
    i = 0
    while i < n:
        op = plan.ops[i]
        if (op.kind != 'lut_dense' or op.scope != ''
                or int(op.attrs['n_inputs']) != 2):
            # scope!='' は recurrent body 内。
            # n_inputs!=2 は `densefuse._layer_lines` のaddr式が2入力
            # 決め打ちなので、この層を鎖の先頭候補にもしない。
            # いずれも主経路の融合対象外として素通りする。
            new_ops.append(op)
            i += 1
            continue
        # 鎖を貪欲に伸ばす: op[j]の出力を読むのがop[j+1]ただ1つで、
        # op[j+1]もscope==''・n_inputs==2のlut_denseである限り続ける。
        # `max_layers`(Metalのバッファ引数上限由来)に達したら
        # そこで打ち切る ─ 打ち切った後の残り層は次のiで改めて鎖の
        # 先頭候補として調べ直されるので、group_sumの直前に来る
        # 最大`max_layers`層の区間だけが融合対象になる。
        chain = [i]
        j = i
        while len(chain) < max_layers:
            users = consumers.get(plan.ops[j].output, [])
            if len(users) != 1 or users[0] != j + 1:
                break
            nxt = plan.ops[j + 1]
            if (nxt.kind != 'lut_dense' or nxt.scope != ''
                    or int(nxt.attrs['n_inputs']) != 2):
                break
            chain.append(j + 1)
            j += 1
        tail = plan.ops[chain[-1]]
        users = consumers.get(tail.output, [])
        gs_i = users[0] if len(users) == 1 else -1
        gs_op = plan.ops[gs_i] if 0 <= gs_i < n else None
        is_group_sum = (len(chain) >= 2 and gs_op is not None
                       and gs_i == chain[-1] + 1
                       and gs_op.kind == 'reduce_scores'
                       and gs_op.attrs.get('op') == 'group_sum'
                       and gs_op.scope == '')
        if is_group_sum:
            layers = []
            for k in chain:
                lop = plan.ops[k]
                a = lop.attrs
                layers.append({
                    'n_inputs': int(a['n_inputs']),
                    'out_width': int(a['out_width']),
                    'in_width': plan.value(lop.inputs[0]).width,
                    'indices': a['indices'], 'truth_tables': a['truth_tables'],
                })
            fits = 2 * dense_fuse_act_bytes(layers) <= max_shared_bytes
        else:
            fits = False
        if is_group_sum and fits:
            fused_ids.append(gs_op.node_id)
            if not dry_run:
                gs_a = gs_op.attrs
                new_op = PlanOp(
                    kind='dense_fuse', inputs=[plan.ops[chain[0]].inputs[0]],
                    output=gs_op.output,
                    attrs={'layers': layers, 'k': int(gs_a['k']),
                          'gsz': int(gs_a['in_width']) // int(gs_a['k'])},
                    node_id=gs_op.node_id, node_type='dense_fuse')
                new_ops.append(new_op)
                # 鎖の中間出力は孤児化する(参照するopが無くなる)。0幅に
                # してメモリ計画に無駄な領域を持たせない(生存区間
                # 解析はop列だけを見るので、参照が消えれば自然に畳まれる
                # が、念のため幅を0にしておく)。dry_runではこの書き換え
                # 自体を行わない(下見だけで実プランは変えない)。
                for k in chain:
                    plan.value(plan.ops[k].output).width = 0
            i = gs_i + 1
            continue
        new_ops.append(op)
        i += 1
    if not dry_run:
        plan.ops[:] = new_ops
    return fused_ids


#: dense_packedが要求する最小バッチ。CPU版実測(9倍)はこの
#: バッチ軸ビットパック方式そのものへの根拠だが、パック1語=32サンプルの
#: 語を作るコスト(mdl_g_pack_batch_word、32スカラー読みの償却されない
#: 1回コスト)がバッチが小さいと相対的に重くなる。batch<32では1語すら
#: 埋まらず(bwords=1、末尾31レーンがゼロ詰め)償却が全く効かないので、
#: 適格性の下限として明示的に拒否する(タスク指示のv1スコープ)。
DENSE_PACKED_MIN_BATCH = 32


def dense_packed_eligible(plan: 'ExecPlan') -> None:
    """dense のバッチ軸ビットプレーン化の適格性を判定する。

    バックエンド非依存の1箇所(`lut_dense_indices_fit_u16`/
    `fuse_dense_layers`と同じ思想)。満たさなければ理由つきの`ValueError`
    を投げる(生成後の実行時ではなく、生成時にここで落とす)。

    適格条件(v1スコープ、タスク指示):
      - `plan.batch >= DENSE_PACKED_MIN_BATCH`(既定32)
      - 画像前処理を持たない(`plan.preproc is None`。thermometer等の
        非画像bits入力のみ対象 ─ preprocのbitplane/conv経路は別物)
      - 主経路(`scope == ''`)の全opが `lut_dense` または
        `reduce_scores`(`op == 'group_sum'`)のみであること。conv系
        (`gate_program`/`plane_op`)・recurrent系(`recurrent_exec`と
        そのbody由来のscope付きop)はv1では一切対象外
      - recurrentノード自体が存在しないこと(scope付きopが1つも無い)

    `dense_fuse=True`や`n_streams>1`との併用はv1では対象外
    ─ こちらは呼び出し側(`generate_metal`/`generate_cuda`)が
    プラン構築より前の引数チェックで`ValueError`にする(このプランは
    `dense_fuse`フラグを覚えていないので、ここでは判定できない)。
    """
    if plan.batch < DENSE_PACKED_MIN_BATCH:
        raise ValueError(
            f'dense_packed には batch>={DENSE_PACKED_MIN_BATCH} が必要 '
            f'(実際は batch={plan.batch})。1語=32サンプルへ詰めるコストが '
            'バッチが小さいと償却されない(タスク指示のv1スコープ)')
    if plan.preproc is not None:
        raise ValueError(
            'dense_packed は非画像(bits入力)のdenseモデル専用。'
            'このバンドルは画像前処理(preproc)を持つ ─ '
            'conv/preproc経路はv1のdense_packed対象外')
    bad: list[str] = []
    has_recurrent = False
    for op in plan.ops:
        if op.scope != '':
            has_recurrent = True
            continue
        if op.kind == 'lut_dense':
            continue
        if op.kind == 'reduce_scores' and op.attrs.get('op') == 'group_sum':
            continue
        if op.kind == 'recurrent_exec':
            has_recurrent = True
            continue
        bad.append(f'{op.node_id}({op.kind}'
                  f'{"/" + op.attrs["op"] if op.kind == "reduce_scores" else ""})')
    if has_recurrent:
        raise ValueError(
            'dense_packed はrecurrentノードを含むモデル未対応(v1スコープ)。'
            'recurrent body内のop(scope付き)はパック経路を通らない')
    if bad:
        raise ValueError(
            'dense_packed は主経路の全opがlut_dense/group_sumのみの'
            'モデルにしか適用できない(v1スコープ、conv2d等は対象外)。'
            f'非対応のop: {", ".join(bad)}')


# ---------------------------------------------------------------------------
# 最適化の自動選択(auto-config)
#
#(conv_codegen)・(n_streams)(dense_fuse)(dense_packed)は
# いずれもopt-in(既定OFF)で導入された ─ 理由は「モデルの支配項(床/計算/
# 帯域)次第で実際に負けるケースが実測で7個中5個見つかったから」であって
# 「効果が疑わしいから」ではない。ここでは実測済みの勝敗条件を判定表として
# 実装し、**勝てる条件でだけ自動でONにする**(前例: `dense_packed_eligible`
# のbatch>=32ガード、`DENSE_FUSE_MAX_LAYERS`の由来コメントと同じ「実測→
# 定数化」の流儀)。判定表全体の解説・実測値の出典は
# `docs/guides/deploy-optimization.md` 参照。
# ---------------------------------------------------------------------------

#: dense_fuseを自動ONにする最小batch。
#:
#: `metalbackend.generate_metal`/`cudabackend.generate_cuda`の
#: `--dense-fuse`ヘルプ文字列(導入当初から記載)が実測(har_dlgn_k6000_
#: gumbel、Apple M3、最小実験)として明記している: batch=1は0.81倍
#: (悪化、1スレッドグループしか立たず占有率が下がるため)、batch=32は
#: 1.91倍、batch=128は2.51倍(最良)。「本測定」(4モデル×
#: batch{1,32,128,256}、GPU busy時間+e2e)
#: では、batch=1のe2e倍率はモデル依存で1.11〜1.34倍(全モデル非悪化)、
#: batch>=32のe2e倍率は1.20〜2.01倍で安定して勝つ。唯一
#: har_dlgn_k600がbatch=32で単発0.95倍(GPU busy時間そのものは同じ行で
#: 2.10倍の明確な勝ち)を記録しているが、この計測は3セット中央値で
#: (同キャンペーンのk6000はバイモーダルな電源状態ノイズのため7セット
#: 必要だったと開発記録に明記されている)、GPU側の実測と矛盾する
#: 単発の外れ値の疑いが強い。batch=1のみ明確な悪化域として除外し、
#: batch>=32を自動ON条件とする(この閾値は変更しない ─ もし
#: batch=32での悪化が外れ値でないと分かった場合は、この閾値の再較正が
#: 必要になる。較正は開発記録に記録の上で実測ベースのみで行うこと)。
DENSE_FUSE_AUTO_MIN_BATCH = 32

#: dense_packedを自動ONにする最小batch(CUDA/Metal共通)。
#:
#: **タスク指示の原案は「CUDA: batch>=32(全実測>=1.0倍)」だったが、
#: 実測の生データを直接検証すると誤りだったため訂正した**(負けも記録する
#: 方針、`.claude/skills/accuracy-hunt`と同じ姿勢)。
#: `results/deploy/cuda_dense_packed_k53_ab.json::part2_ab_throughput`を
#: 見ると、batch=32では5モデル中3モデルが明確に悪化している
#: (har_dlgn_k600 0.891倍・har_dlgn_k2400 0.885倍・har_dlgn_dwn 0.878倍。
#: mnist_adder 1.072倍・har_dlgn_k6000 1.002倍の2モデルのみ非悪化)。
#: batch=128まで上げると悪化はhar_dlgn_k600の-2.1%(0.979倍、この
#: キャンペーンで別途確立されたノイズ床1〜2%のすぐ外)まで縮み、他4
#: モデルは+10%以上の明確な勝ち(k2400 1.109倍・dwn 1.115倍・
#: k6000 1.603倍・mnist_adder 2.262倍)。batch=256まで上げるとk600も
#: -0.5%(0.995倍、ノイズ床相当)まで縮む。
#: Metal側は`results/deploy/metal_dense_packed_k53_ab.json`実測で
#: batch=32のk2400が0.887倍(タスク指示の「b32でk2400 0.89×の悪化」は
#: この実測と一致)、batch=128で全4モデルが1.04〜1.40倍と明確な勝ちに
#: 転じる。CUDA/Metal両方が独立にbatch>=128で「既知の悪化ケースなし」に
#: 収束するため、両バックエンド共通の1つの閾値に統一する。
#: (`DENSE_PACKED_MIN_BATCH`=32は「パック方式が機能する最小batch」という
#: 別の下限 ─ v1スコープの適格性ゲート、`dense_packed_eligible`参照。
#: こちらは「自動でONにしてよい最小batch」というより厳しい条件で、
#: 32以上128未満のbatchでも`--dense-packed`を明示指定すれば従来どおり
#: 使える。)
DENSE_PACKED_AUTO_MIN_BATCH = 128


#:: C 生成時の dense-flat(`lut_flat`)自動採用の閾値。平坦化後の実効ゲート
#: 削減率(置換対象 lut_layer 列の gates_before→gates_after)がこの値以上なら
#: 採用する。lut_flat は縮約で消えたゲートの分だけ速くなるが、1 ゲートあたりの
#: コストは従来の SIMD LUT 引きより高い(添字表を引く表駆動、SIMD 化しにくい)。
#: そのためゲートを十分減らせないと逆に遅くなる。ホスト実測
#: (`scripts/bench_dense_flat.py`、`docs/guides/deploy-optimization.md`)では
#: 損益分岐が削減率 ≈26% で、k2400(定数・パススルーが多い)や HAR k6000
#: (削減 58%)など多くの実モデルは分岐点を大きく超える。余裕を見て 25% を
#: 閾値にする(境界付近はほぼ引き分けで、悪化幅も小さい)。
DENSE_FLAT_AUTO_MIN_REDUCTION = 0.25

#:: Web(codegen-web / verify-web)での dense-flat auto の採用閾値。
#: Web の lut_layer 経路は SIMD 化されていない JS のゲート 1 本ずつの表引きで、
#: 1 ゲートあたりのコストが lut_flat(信号表を引く JS ループ)より**高い**
#: (HAR k6000・1 枚、Node 24/M3: lut_layer 223µs / 28,636 ゲート = 7.8ns、
#: lut_flat 62µs / 15,124 ゲート = 4.1ns)。C と違って「ゲートを減らせないと
#: 逆に遅くなる」領域が無いので、2 入力 lut_layer 列があれば常に平坦化する
#: (閾値 0)。実測は公開リファレンス reference/deploy-optimizations の Web 節。
DENSE_FLAT_AUTO_MIN_REDUCTION_WEB = 0.0

#:: GPU(codegen-cuda / codegen-metal / verify-metal)で `--dense-flat auto` を
#: **明示したとき**の採用閾値。GPU の CLI 既定は `off`(opt-in): Apple M3 の Metal
#: 実測では HAR k6000(削減 47%)で既存最良(dense_packed / dense_fuse)比
#: 0.47〜0.71 倍に落ちた(batch 128 以上。公開リファレンス reference/deploy-optimizations
#: の GPU 節)。GPU の 1 ゲートあたりのコストは段ごとの gather で lut_dense と同程度
#: なので、auto でも C と同じく「ゲートが十分減るとき」に限る(0.25)。
DENSE_FLAT_AUTO_MIN_REDUCTION_GPU = 0.25

#:: CLI の codegen-c が、平坦化(lut_flat)を採用した dense モデルに既定で
#: 生やすバッチ軸パック入口(`mdl_infer_batch_packed`)の最大枚数。
#: 実測(M3、`scripts/bench_flat_variants.py`、results/deploy/flat_bitpar_*):
#: 32 枚で 1 枚ずつの flat 比 HAR k600 1.7x / k2400 4.1x / k6000 5.9x /
#: MNIST 5.4x のスループット(スカラービルドでも 1.1〜5.1x)。lut_flat の表を
#: 共有するので flash の増分は関数 1 本(生成 C で約 0.7KB)だけ、作業領域は
#: 呼び出し側持ち(k6000・32 枚で 103KB)。1 枚の推論(レイテンシ)は従来の
#: `mdl_infer` が最速のまま(パック入口に 1 枚だけ渡すと 5〜12 倍遅い)。
#: 枚数を増やしたいときは `--dense-batch N` で上書きする。
DENSE_BATCH_AUTO = 32


def resolve_auto_config(plan: 'ExecPlan', *, batch: int, backend: str) -> dict:
    """実測済みの勝敗条件だけを見て、conv_engine/dense_fuse/
    dense_packedを自動選択する(判定表の実装)。

    `generate_metal`/`generate_cuda`(`auto='measured'`が既定)が、呼び出し
    側が明示的に指定しなかった引数をこの関数の判断で埋める。この関数自体は
    副作用を持たない(`plan`を書き換えない)ので、呼び出し側の都合で何度
    呼んでもよい。

    Args:
        plan: `build_plan(bundle, batch=batch)`で作った「素の」プラン
            (`dense_fuse=False`で構築したもの)。**既にdense_fuseで融合
            済みのプランを渡さないこと** ─ この関数はdense_fuseが実際に
            鎖を融合できるかを`fuse_dense_layers(..., dry_run=True)`で
            自分で試すので、素のプランを前提にしている。
        batch: 生成に使うバッチ数。`generate_metal`/`generate_cuda`の
            `batch`引数と同じ値を渡すこと(`plan.batch`と一致している
            必要がある ─ `dense_packed_eligible`が`plan.batch`を見る)。
        backend: `'metal'` または `'cuda'`。dense_fuseの共有メモリ上限
            (Metal 32KB/CUDA 48KB)の選択にだけ使う。

    Returns:
        以下のキーを持つdict:
          - `conv_engine`: `'split'`(既定)または`'combined'`
            (max_slots_per_oc超過ノードがあれば自動フォールバック)
          - `dense_fuse`: bool(batch>=32かつ融合可能な鎖がある)
          - `dense_packed`: bool(dense_fuseが自動ONにならず、かつ
            適格かつbatch>=128)。`dense_fuse`と`dense_packed`は生成側で
            併用不可なため、この関数は両方Trueを返さない
            (dense_fuseの方が実測で概ね上回るため優先する ─
            `docs/guides/deploy-optimization.md`の比較表参照)。
          - `reasons`: 上記3キーそれぞれの判定理由を人間可読の文字列で
            持つdict(生成ソースのコメント・CLIサマリに使う)。

    Raises:
        ValueError: `backend`が`'metal'`/`'cuda'`以外
    """
    if backend not in ('metal', 'cuda'):
        raise ValueError(f"backend={backend!r} は未対応('metal'/'cuda'のみ)")
    reasons: dict[str, str] = {}

    # 1) conv_engine: split(既定)でmax_slots_per_ocがCONV_SPLIT_MAX_SLOTSを
    #    超えるconv2dノードが1つでもあれば、モデル全体をcombinedへ切り替える
    #    (実測: det_multimnistがmax_slots_per_oc=9>8で生成時に即
    #    ValueErrorになっていた壁)。`_plan_conv2d`がgate_program opの
    #    attrsに'params'/'program'を積んであるので、実際にコード生成する
    #    前にここで軽く判定できる。
    conv_engine = 'split'
    conv_reason = ('split(既定)。全conv2dノードで'
                   'max_slots_per_oc<=CONV_SPLIT_MAX_SLOTS')
    for op in plan.ops:
        if op.kind != 'gate_program':
            continue
        try:
            build_conv_layout_split(op.attrs['params'], op.attrs['program'])
        except ValueError as exc:
            conv_engine = 'combined'
            conv_reason = (
                f'combinedへ自動フォールバック: ノード'
                f'{op.node_id!r}がsplitでは生成できない({exc})')
            break
    reasons['conv_engine'] = conv_reason

    # 2) dense_fuse: 実際に1つでも鎖が融合できて、かつbatchが閾値以上の
    #    ときだけ自動ON。`dry_run=True`なので`plan`自体は書き換えない。
    max_shared_bytes = (CUDA_DENSE_FUSE_MAX_SHARED_BYTES if backend == 'cuda'
                        else DENSE_FUSE_MAX_SHARED_BYTES)
    fusible = fuse_dense_layers(plan, max_shared_bytes=max_shared_bytes,
                                dry_run=True)
    if not fusible:
        fuse_ok = False
        fuse_reason = ('融合可能な鎖なし(lut_dense2層以上+群sumの並びが'
                       '無い、conv主体のモデル、または活性が共有メモリ'
                       f'上限{max_shared_bytes}Bを超える)')
    elif batch < DENSE_FUSE_AUTO_MIN_BATCH:
        fuse_ok = False
        fuse_reason = (
            f'{len(fusible)}本の鎖が融合可能だがbatch={batch} < '
            f'{DENSE_FUSE_AUTO_MIN_BATCH}(実測: batch=1は1.11〜1.34倍と'
            '安定はしているが「明確な勝ち」の閾値に届かないため自動ONに'
            'しない。DENSE_FUSE_AUTO_MIN_BATCHのコメント参照)')
    else:
        fuse_ok = True
        fuse_reason = (
            f'{len(fusible)}本の鎖が融合可能、batch={batch}>='
            f'{DENSE_FUSE_AUTO_MIN_BATCH}(実測: batch>=32のe2e倍率は'
            '1.20〜2.01倍で安定して勝つ)')
    reasons['dense_fuse'] = fuse_reason

    # 3) dense_packed: dense_fuseが自動ONになったら見送る(生成側で併用
    # 不可・実測でもdense_fuseの方が概ね上回るため優先、優先順位)。
    #    dense_fuseが自動ONにならないときだけ、適格性+batch>=128を見る。
    if fuse_ok:
        packed_ok = False
        packed_reason = 'dense_fuseが自動ONのため見送り(併用不可、dense_fuse優先)'
    else:
        try:
            dense_packed_eligible(plan)
        except ValueError as exc:
            packed_ok = False
            packed_reason = f'不適格: {exc}'
        else:
            if batch < DENSE_PACKED_AUTO_MIN_BATCH:
                packed_ok = False
                packed_reason = (
                    f'適格だがbatch={batch} < {DENSE_PACKED_AUTO_MIN_BATCH}'
                    '(実測: batch=32はCUDA/Metalとも複数モデルで'
                    '悪化するため自動ONにしない。'
                    'DENSE_PACKED_AUTO_MIN_BATCHのコメント参照)')
            else:
                packed_ok = True
                packed_reason = (
                    f'適格かつbatch={batch}>={DENSE_PACKED_AUTO_MIN_BATCH}'
                    '(実測: 両バックエンドともこのbatch以上で既知の'
                    '悪化ケースなし)')
    reasons['dense_packed'] = packed_reason

    return {'conv_engine': conv_engine, 'dense_fuse': fuse_ok,
            'dense_packed': packed_ok, 'reasons': reasons}


def node_inputs(graph: IRGraph) -> dict[str, list[str]]:
    """ノードid → 入力srcのリスト(dst_slot昇順)。"""
    acc: dict[str, list[tuple[int, str]]] = {}
    for e in graph.edges:
        acc.setdefault(e.dst, []).append((e.dst_slot, e.src))
    return {k: [s for _, s in sorted(v)] for k, v in acc.items()}


# ---------------------------------------------------------------------------
# プラン構築
# ---------------------------------------------------------------------------

class _PlanBuilder:
    """ノード型ハンドラが使う作業台。"""

    def __init__(self, bundle: dict):
        self.bundle = bundle
        self.graph: IRGraph = ir_from_dict(bundle['ir'])
        self.programs = bundle.get('programs') or {}
        self.preproc_plan = bundle.get('preproc_desc')
        self.values: list[PlanValue] = []
        self.ops: list[PlanOp] = []
        self.input_bitplane: int | None = None
        self.preproc_used = False
        #  SpatialGroupSum型 permute の遅延情報(値id → 次元)
        self.deferred: dict[int, tuple] = {}
        self._node = None      # 現在処理中のノード(エラーメッセージ用)
        #:: `accumulate_group`(id_prefix文字列)→ そのグループの
        #: 親バッファ(PlanValue)。`accumulate_group`→ そこまでに書き込んだ
        #: バイト数(次の書き込みオフセット)。recurrentのbodyごとに
        #: リセットする(`_plan_recurrent`参照 — N/K/L-layersが同じ
        #: id_prefix='lut'を使い回すため、body間で名前空間を分けないと
        #: 別々のLutStackのグループが衝突する)。
        self._acc_parents: dict[str, PlanValue] = {}
        self._acc_used: dict[str, int] = {}

    # -- 値 -----------------------------------------------------------
    def new_value(self, kind: str, *, c=0, h=0, w=0, width=0,
                  origin='') -> PlanValue:
        v = PlanValue(id=len(self.values), kind=kind, c=c, h=h, w=w,
                      width=width, origin=origin)
        self.values.append(v)
        return v

    def add_op(self, kind: str, inputs, output: PlanValue, **attrs) -> None:
        self.ops.append(PlanOp(
            kind=kind,
            inputs=[v.id for v in inputs],
            output=output.id,
            attrs=attrs,
            node_id=self._node.id if self._node is not None else '',
            node_type=self._node.type if self._node is not None else '',
        ))

    def fail(self, message: str):
        node = self._node
        return UnsupportedNode(
            message,
            node_id=node.id if node is not None else '',
            node_type=node.type if node is not None else '')

    # -- 値の変換 -----------------------------------------------------
    def as_bitplane(self, v: PlanValue, c: int, h: int, w: int) -> PlanValue:
        """FLATならパックしてBITPLANEにする。

        グラフ入力を最初にパックする箇所は前処理の出力先とまったく同じ形なので、
        バッファを共有して中間(実モデルでは数MBの0/1バイト列)を作らない。
        """
        plan = self.preproc_plan
        if (v.id == 0 and plan is not None
                and (c, h, w) == (plan['channels'], plan['height'],
                                  plan['width'])):
            if self.input_bitplane is None:
                bp = self.new_value(KIND_BITPLANE, c=c, h=h, w=w,
                                    origin='input_bitplane')
                self.input_bitplane = bp.id
                self.preproc_used = True
            return self.values[self.input_bitplane]
        if v.kind == KIND_BITPLANE:
            if (v.c, v.h, v.w) != (c, h, w):
                raise self.fail(
                    f'ビットプレーンの形が合わない: 期待 {(c, h, w)}、'
                    f'実際 {(v.c, v.h, v.w)}')
            return v
        if v.kind != KIND_FLAT:
            raise self.fail(f'{v.kind} をビットプレーンへ変換できない')
        out = self.new_value(KIND_BITPLANE, c=c, h=h, w=w, origin='pack')
        self.add_op('pack', [v], out, c=c, h=h, w=w)
        return out

    def as_flat(self, v: PlanValue) -> PlanValue:
        if v.kind == KIND_FLAT:
            return v
        if v.kind != KIND_BITPLANE:
            raise self.fail(f'{v.kind} を非パック列へ変換できない')
        width = v.c * v.h * v.w
        out = self.new_value(KIND_FLAT, width=width, origin='unpack')
        self.add_op('unpack', [v], out, c=v.c, h=v.h, w=v.w)
        return out


# ---- ノード型ハンドラ -------------------------------------------------

@register_node_kind('conv2d')
def _plan_conv2d(b: _PlanBuilder, node, ins):
    from .layout import build_conv_layout

    program = b.programs.get(node.id)
    if program is None:
        raise b.fail('programs節が無い(バンドルを作り直すこと)')
    layout = build_conv_layout(node.params, program)
    c, h, w = (int(v) for v in node.params['in_shape'])
    x = b.as_bitplane(ins[0], c, h, w)
    out = b.new_value(KIND_BITPLANE, c=layout.out_channels, h=layout.h_out,
                      w=layout.w_out, origin=node.id)
    # スクラッチ(定数平面+タップpool+スロット)は実行中しか要らない。
    # 値として宣言しておけば、メモリ計画が全conv分をまとめて1本に畳む。
    scratch = b.new_value(KIND_SCRATCH, width=layout.scratch_words,
                          origin=f'{node.id}:scratch')
    # params/program(生の入力)も持たせておく: GPUバックエンドが
    # ocごとの命令列分割レイアウト(`build_conv_layout_split`)を
    # 遅延構築するのに使う。Cバックエンドは`layout`(全oc結合)だけを見る
    # ので、この2キーの追加はCの生成結果に影響しない。
    b.add_op('gate_program', [x], out, layout=layout, scratch=[scratch.id],
             params=node.params, program=program)
    return out


@register_node_kind('or_pool2d')
def _plan_or_pool2d(b: _PlanBuilder, node, ins):
    p = node.params
    kh, kw = pair(p['kernel_size'])
    sh, sw = pair(p.get('stride', kh))
    ph, pw = pair(p.get('padding', 0))
    if (kh, kw, sh, sw, ph, pw) != (2, 2, 2, 2, 0, 0):
        raise b.fail(
            f'k={kh}x{kw} s={sh}x{sw} p={ph}x{pw} は未対応'
            '(対応するのは 2x2/stride2/pad0 のみ)')
    c, h, w = (int(v) for v in p['in_shape'])
    x = b.as_bitplane(ins[0], c, h, w)
    h_out, w_out = (h - 2) // 2 + 1, (w - 2) // 2 + 1
    out = b.new_value(KIND_BITPLANE, c=c, h=h_out, w=w_out, origin=node.id)
    b.add_op('plane_op', [x], out, op='or_pool2x2', c=c, h=h, w=w)
    return out


@register_node_kind('upsample2d')
def _plan_upsample2d(b: _PlanBuilder, node, ins):
    p = node.params
    scale = int(p['scale'])
    if scale != 2:
        raise b.fail(f'scale={scale} は未対応(対応するのは 2 のみ)')
    c, h, w = (int(v) for v in p['in_shape'])
    x = b.as_bitplane(ins[0], c, h, w)
    out = b.new_value(KIND_BITPLANE, c=c, h=2 * h, w=2 * w, origin=node.id)
    b.add_op('plane_op', [x], out, op='upsample2d_x2', c=c, h=h, w=w)
    return out


@register_node_kind('permute')
def _plan_permute(b: _PlanBuilder, node, ins):
    p = node.params
    dims = spatial_group_sum_dims(p)
    if dims is not None:
        # SpatialGroupSum型は実体化せず、後続の group_sum と融合する
        n_out, gs, h, w = dims
        x = b.as_bitplane(ins[0], n_out * gs, h, w)
        b.deferred[x.id] = ('spatial_group_sum', n_out, gs, h, w)
        return x
    if not keeps_trailing_spatial(p):
        raise b.fail(
            f'perm={p["perm"]} は未対応'
            '(対応するのはチャネル軸のみ / SpatialGroupSum型のみ)')
    in_shape = [int(v) for v in p['in_shape']]
    h, w = in_shape[-2], in_shape[-1]
    c = 1
    for d in in_shape[:-2]:
        c *= d
    x = b.as_bitplane(ins[0], c, h, w)
    order = channel_order_from_permute(p)
    out = b.new_value(KIND_BITPLANE, c=len(order), h=h, w=w, origin=node.id)
    b.add_op('plane_op', [x], out, op='permute_channels', order=order,
             c_out=len(order), h=h, wpr=x.wpr)
    return out


def _plan_accumulate_concat(b: _PlanBuilder, node, ins, group_id: str) -> PlanValue:
    """退化ゲート除去のアキュムレータ連結を物理コピー無しで行う。

    `export/rddlgn_optimize.py::reduce_lutstack_nodes`が付ける
    `accumulate_group`/`accumulate_final_width`(concat.params)を持つ
    concatノードだけがこのパスへ来る(state/step束ねや多ヘッド
    出力連結など、通常のconcat呼び出しはこのparamsを持たないため既存経路は
    完全に不変)。

    従来は層ごとに「entry ++ これまでの生存ゲート」を毎回
    `flat_concat`で実コピーしており、tick毎に評価されるLutStackで
    O(層数)回・幅が単調増加するコピーが発生していた(charlmでencode+96%
    の回帰の原因)。ここでは:

    - グループの最初のconcatでだけ`accumulate_final_width`ぶんの
      1本のバッファ(親)を確保し、`ins[0]`(entry、またはこれまでの
      アキュムレータ)を一度だけ物理コピーする(この1回だけがコスト)。
    - 2回目以降のconcat(同じgroup_id)は一切コピーせず、`ins[1]`
      (直前に`_plan_lut_layer`が独立バッファとして確保済みの新層の出力)
      を「親バッファ+オフセット」のaliasへ書き換えるだけ。lut_denseの
      opは書き換えないので、`_declare_buffers`(cbackend.py)が
      alias先マクロを親+オフセットへ差し替えれば、既存のlut_dense呼び出し
      コードは無改造のまま親バッファへ直接書き込む形になる。

    Args:
        ins: [acc_prev, new_layer_output] — rddlgn_optimize.pyが必ず
            この順(slot0=既存アキュムレータ、slot1=新層の出力)でエッジを
            張る
    """
    acc_prev, new_val = ins
    final_width = int(node.params['accumulate_final_width'])
    parent = b._acc_parents.get(group_id)
    if parent is None:
        parent = b.new_value(KIND_FLAT, width=final_width,
                             origin=f'acc:{group_id}')
        b.add_op('flat_concat', [acc_prev], parent, widths=[acc_prev.width])
        b._acc_parents[group_id] = parent
        b._acc_used[group_id] = acc_prev.width
    assert parent.width == final_width, (
        f'accumulate_group {group_id!r}: final_widthがconcatノード間で'
        f'食い違っている({parent.width} != {final_width})')
    offset = b._acc_used[group_id]
    new_val.alias_of = parent.id
    new_val.alias_offset = offset
    b._acc_used[group_id] = offset + new_val.width
    return parent


@register_node_kind('concat')
def _plan_concat(b: _PlanBuilder, node, ins):
    group_id = node.params.get('accumulate_group')
    if group_id is not None:
        return _plan_accumulate_concat(b, node, ins, group_id)
    # 検出モデルは多ヘッド出力(hm/size/off)を concat で1本に束ねるので、
    # スコア列の連結も要る(export/dag.py の output_heads)。
    if all(v.kind == KIND_SCORES for v in ins):
        total = sum(v.width for v in ins)
        out = b.new_value(KIND_SCORES, width=total, origin=node.id)
        b.add_op('copy_scores', ins, out, widths=[v.width for v in ins])
        return out
    # 非パックの平坦ビット列同士の連結(recurrentのbody内でstate/step等の
    # scan_varを1本のlut_layer入力へ束ねるのに使う)。
    if all(v.kind == KIND_FLAT for v in ins):
        total = sum(v.width for v in ins)
        out = b.new_value(KIND_FLAT, width=total, origin=node.id)
        b.add_op('flat_concat', ins, out, widths=[v.width for v in ins])
        return out
    if not all(v.kind == KIND_BITPLANE and v.id not in b.deferred
               for v in ins):
        kinds = ', '.join(v.kind for v in ins)
        raise b.fail(
            f'入力の種類が混在している({kinds})。'
            'ビットプレーン同士かスコア同士のみ対応')
    h, w = ins[0].h, ins[0].w
    if any((v.h, v.w) != (h, w) for v in ins):
        raise b.fail('入力の空間形状が揃っていない')
    total_c = sum(v.c for v in ins)
    out = b.new_value(KIND_BITPLANE, c=total_c, h=h, w=w, origin=node.id)
    b.add_op('plane_op', ins, out, op='concat_planes',
             c_in=[v.c for v in ins], h=h, wpr=ins[0].wpr)
    return out


@register_node_kind('group_sum')
def _plan_group_sum(b: _PlanBuilder, node, ins):
    k = int(node.params['k'])
    if any(int(v) != 0 for v in node.params.get('bias') or []):
        raise b.fail(
            'group_sum の判定バイアス(bias)は C ランタイム未対応。'
            'バイアス無しのスコアを出し、比較はホスト側で行うこと')
    x = ins[0]
    deferred = b.deferred.get(x.id)
    if deferred is not None and deferred[0] == 'spatial_group_sum':
        _, n_out, gs, h, w = deferred
        if k != n_out * h * w:
            raise b.fail(
                f'k={k} が SpatialGroupSum の {n_out}*{h}*{w} と一致しない')
        out = b.new_value(KIND_SCORES, width=k, origin=node.id)
        # 桁上げ加算の作業平面(gsのビット幅ぶん)。scratch として宣言すれば
        # メモリ計画が他の一時領域と一緒に畳んでくれる。
        n_planes = max(1, int(gs).bit_length())
        scratch = b.new_value(KIND_SCRATCH,
                              width=n_planes * words_per_row(w),
                              origin=f'{node.id}:scratch')
        b.add_op('reduce_scores', [x], out, op='spatial_group_sum',
                 n_out=n_out, gs=gs, h=h, w=w, scratch=[scratch.id])
        return out
    flat = b.as_flat(x)
    if flat.width % k != 0:
        raise b.fail(f'幅{flat.width}がk={k}で割り切れない')
    out = b.new_value(KIND_SCORES, width=k, origin=node.id)
    b.add_op('reduce_scores', [flat], out, op='group_sum',
             in_width=flat.width, k=k)
    return out


@register_node_kind('lut_layer')
def _plan_lut_layer(b: _PlanBuilder, node, ins):
    from ..ir.serialize import tt_from_hex

    p = node.params
    n_inputs = int(p['n_inputs'])
    if n_inputs > 5:
        raise b.fail(
            f'n_inputs={n_inputs} は未対応(真理値表が32bitに収まらない)')
    out_width = int(p['out_width'])
    flat = b.as_flat(ins[0])
    out = b.new_value(KIND_FLAT, width=out_width, origin=node.id)
    b.add_op('lut_dense', [flat], out,
             n_inputs=n_inputs, out_width=out_width,
             indices=[int(v) for row in p['indices'] for v in row],
             truth_tables=tt_from_hex(list(p['truth_tables'])))
    return out


#: lut_flat の添字表(a/b/outputs)がuint16_tに収まる上限。
#: outputs は `2*信号番号 + 極性` を持つので、最大値 2*(n_signals-1)+1 が
#: 65535 以下 ⇔ n_signals <= 32768。a/b/outputs は同じ型で揃える
#: (カーネルが `mdl_lut_flat_u16` / `_u32` の2種だけで済む)。
LUT_FLAT_U16_MAX_SIGNALS = 32768


def lut_flat_indices_fit_u16(in_width: int, n_gates: int) -> bool:
    """この lut_flat の添字表をuint16_tで持てるか(信号数 1+in_width+n_gates)。

    `lut_dense_indices_fit_u16` と同じくバックエンド非依存の1箇所に置く。
    """
    return 1 + int(in_width) + int(n_gates) <= LUT_FLAT_U16_MAX_SIGNALS


@register_node_kind('lut_flat')
def _plan_lut_flat(b: _PlanBuilder, node, ins):
    """縮約済み dense 平坦ネットリスト→ op `lut_flat`。

    出力は lut_layer と同じ0/1バイト列(KIND_FLAT、幅 out_width)なので、後段の
    group_sum / to_scores は無改造で読める。信号表(1+in_width+n_gates バイト)
    は op の実行中だけ使うので KIND_SCRATCH として宣言し、メモリ計画に
    畳んでもらう(`MDL_CTX_BYTES` に自動で算入される)。
    """
    p = node.params
    in_width = int(p['in_width'])
    out_width = int(p['out_width'])
    gates = [int(v) for g in p['gates'] for v in g]
    outputs = [int(v) for v in p['outputs']]
    flat = b.as_flat(ins[0])
    if flat.width != in_width:
        raise b.fail(f'入力幅{flat.width}が in_width={in_width} と一致しない')
    n_gates = len(gates) // 3
    n_signals = 1 + in_width + n_gates
    scratch = b.new_value(KIND_SCRATCH, width=(n_signals + 3) // 4,
                          origin=f'{node.id}:scratch')
    out = b.new_value(KIND_FLAT, width=out_width, origin=node.id)
    b.add_op('lut_flat', [flat], out, in_width=in_width, out_width=out_width,
             gates=gates, outputs=outputs, scratch=[scratch.id])
    return out


@register_node_kind('flatten')
def _plan_flatten(b: _PlanBuilder, node, ins):
    return ins[0]


@register_node_kind('recurrent')
def _plan_recurrent(b: _PlanBuilder, node, ins):
    """recurrentノード。

    bodyはunrollしない。同じ _PlanBuilder(b)を使ってbodyのノード列を1回
    plan し、生成された op すべてに `scope=node.id` を付けて主経路と区別
    する。scan_var はプラン層ハンドラを持たず、body走査時に
    state バッファ/外部入力バッファ(値としてのみ存在)を直接差し込む。

    op自体は3つ:
      - state_init(state役scan_varの初期値。steps回のループの外、1回だけ)
      - <body内のops>(scope=node.idタグ付き。バックエンドがステップ関数へ
        切り出す)
      - recurrent_exec(ホスト側のsteps回ループ+state更新+出力の取り出し)
    """
    p = node.params
    body: IRGraph = p['body']
    steps = int(p['steps'])
    scan_vars = p['scan_vars']
    ext_svs = [sv for sv in scan_vars if sv['role'] in ('step', 'invariant')]
    state_svs = [sv for sv in scan_vars if sv['role'] == 'state']
    if len(ext_svs) != len(ins):
        raise b.fail(
            f'scan_vars(step/invariant)={len(ext_svs)}本 だが '
            f'入力エッジ={len(ins)}本')

    # 1. state役scan_varの初期化(steps回のループの外、1回だけ実行)。
    #    initはrecurrentレベルのscan_vars要約には無く、body内のscan_var
    #    ノード自身が持つ(schema.py参照。単一情報源をbody側に保つ)。
    state_bufs: dict[str, PlanValue] = {}
    for sv in state_svs:
        w = int(sv['width'])
        buf = b.new_value(KIND_FLAT, width=w, origin=f'{node.id}:state:{sv["id"]}')
        init = body.node_by_id(sv['id']).params['init']
        bits = [int(x) for x in init['bits']]
        b.add_op('state_init', [], buf, bits=bits)
        state_bufs[sv['id']] = buf

    # 2. 各ステップの外部入力(step/invariant)を保持する小バッファ。
    #    steps回のループのたびにホスト側コードが該当スライスをコピーする
    #    (emit_recurrent_execのホストループ)。ext_init自体はop列上の
    #    「定義済み」を成立させるためだけの形式op(生存区間解析の前提
    #    「値はどれかのopの出力として生まれる」を満たす。state_bufsの
    #    state_initと対称。実際の値はコピーインが毎tick上書きするので、
    #    ext_initはCコード上は何も生成しなくてよい)。
    ext_bufs: dict[str, PlanValue] = {}
    ext_srcs: dict[str, int] = {}
    for sv, ext_in in zip(ext_svs, ins, strict=True):
        w = int(sv['width'])
        buf = b.new_value(KIND_FLAT, width=w, origin=f'{node.id}:ext:{sv["id"]}')
        b.add_op('ext_init', [], buf, width=w)
        ext_bufs[sv['id']] = buf
        ext_srcs[sv['id']] = ext_in.id

    # 3. bodyを同一builderで1回planする(scan_varは値を直接差し込む)。
    body_srcs = node_inputs(body)
    body_values: dict[str, PlanValue] = {}
    scope_start = len(b.ops)
    # accumulate-groupの名前空間(_acc_parents/_acc_used)をbodyごとに
    # リセットする。N/K/L-layersは同じid_prefix='lut'を使い回す(別々の
    # bodyグラフなのでIRノードidとしては衝突しない)ため、リセットせずに
    # 使い回すと別のLutStackのグループを誤って共有してしまう。
    saved_acc = (b._acc_parents, b._acc_used)
    b._acc_parents, b._acc_used = {}, {}
    for bn in body.topological_order():
        if bn.type == 'scan_var':
            role = bn.params['role']
            body_values[bn.id] = (state_bufs[bn.id] if role == 'state'
                                  else ext_bufs[bn.id])
            continue
        handler = NODE_KINDS.get(bn.type)
        if handler is None:
            reason = UNSUPPORTED_NODE_TYPES.get(bn.type)
            raise UnsupportedNode(
                reason or 'デプロイキット未対応のノード型(recurrent body内)',
                node_id=bn.id, node_type=bn.type)
        b_ins = [body_values[s] for s in body_srcs.get(bn.id, [])]
        b._node = bn
        body_values[bn.id] = handler(b, bn, b_ins)
    b._node = node
    b._acc_parents, b._acc_used = saved_acc
    for op in b.ops[scope_start:]:
        op.scope = node.id

    # 4. state_update: scan_var_id -> body内のPlanValue.id
    state_update_vids = {sv_id: body_values[bnid].id
                         for sv_id, bnid in p['state_update'].items()}

    # 5. 出力。final_stateはstate_updateで最終tickに更新されたstateバッファ
    #    をそのまま読む(output_source_idがstate_updateの対象と一致する
    #    設計のため、追加のコピー先を持たなくてよい)。stackedは
    #    steps*output_widthの専用バッファへ、tickごとに1行書き込む。
    output_source = body_values[p['output_source_id']]
    output_width = int(p['output_width'])
    output_mode = p['output_mode']
    output_state_buf = None
    if output_mode == 'final_state':
        for sv_id, bnid in p['state_update'].items():
            if body_values[bnid].id == output_source.id:
                output_state_buf = state_bufs[sv_id]
                break
        if output_state_buf is None:
            raise b.fail(
                "output_mode='final_state'だがoutput_source_idがどの"
                'state_updateの対象とも一致しない')
        total_width = output_width
    else:
        total_width = steps * output_width

    # body内でgroup_sumを経由し、output_sourceが整数スコア行(KIND_SCORES)に
    # なっているケース(P+M+GroupSumを1つのbodyへ畳んだ場合。
    # スコアはint32なので、決め打ちでKIND_FLAT(1バイト/要素)にすると
    # emit_recurrent_execのコピー(buf()がkindを見てuint8_t*/int32_t*の
    # キャストを選ぶ)で値が壊れる。output_sourceの種別をそのまま引き継ぐ
    # (body=lut_layer/concatのみのケースはKIND_FLATのままなので
    # 挙動不変)。
    out_kind = output_source.kind
    if out_kind not in (KIND_FLAT, KIND_SCORES):
        raise b.fail(
            f'recurrentのoutput_sourceの値種別{out_kind}は未対応'
            '(KIND_FLAT/KIND_SCORESのみ)')
    out = b.new_value(out_kind, width=total_width, origin=node.id)

    # 生存区間解析(memplan)がstate/ext/body内部の値を「recurrent_execまで
    # 生きている」と判定できるよう、参照するもの全部をinputsに含める
    # (state_updateの対象は複数stateがあるとき output_source と別々になり
    # うるので、両方を明示的に含める。重複はliveness計算上無害なので
    # dedupは順序保持のみ行う)。
    protected = (list(ins) + list(state_bufs.values())
                + [body_values[bnid] for bnid in p['state_update'].values()]
                + [output_source])
    seen_vids: set = set()
    exec_inputs: list[PlanValue] = []
    for v in protected:
        if v.id not in seen_vids:
            seen_vids.add(v.id)
            exec_inputs.append(v)
    b.add_op('recurrent_exec', exec_inputs, out,
             steps=steps,
             output_mode=output_mode,
             output_width=output_width,
             output_source=output_source.id,
             output_state_buf=(output_state_buf.id
                               if output_state_buf is not None else None),
             state_bufs={sv_id: buf.id for sv_id, buf in state_bufs.items()},
             state_update=state_update_vids,
             ext_bufs={sv_id: buf.id for sv_id, buf in ext_bufs.items()},
             ext_srcs=ext_srcs,
             ext_roles={sv['id']: sv['role'] for sv in ext_svs},
             ext_widths={sv['id']: int(sv['width']) for sv in ext_svs})
    return out


# 対応しないノード型は理由つきで宣言する(カバレッジ行列テストが読む)
declare_unsupported(
    'blackbox',
    'softmax/argmax など論理回路で表せない実数演算を含む(原理的に不可)。'
    'IRネイティブ化するなら比較器木+MUXでargmaxを組む別トラックが要る')
declare_unsupported(
    'add', 'ビットプレーンのOR/XOR。elementwise op として追加予定(未実装)')
declare_unsupported(
    'binarize', '閾値比較。elementwise op として追加予定(未実装)')
declare_unsupported(
    'constant', '定数平面。arity 0 なので入力判定の見直しも要る(未実装)')
declare_unsupported(
    'scan_var',
    'recurrentのbody内でのみ意味を持つプレースホルダ。トップレベルの'
    'op列には現れない(register_node_kindのハンドラ経由ではなく、'
    '_plan_recurrentがbody走査時にstate/外部入力バッファを直接差し込む'
    'ため専用ハンドラは不要)。body外に単体で置かれた場合はここに来て'
    '未対応として落ちる(それ自体が正しい挙動)')


# ---------------------------------------------------------------------------

def build_plan(bundle: dict, *, batch: int = 1,
               dense_fuse: bool = False,
               dense_fuse_max_shared_bytes: int = DENSE_FUSE_MAX_SHARED_BYTES
               ) -> ExecPlan:
    """バンドル → バックエンド非依存の実行プラン。

    Args:
        bundle: 配布バンドル
        batch: 同時に流す枚数の上限。バッファがこの倍数になる。
            **ビットプレーンの並びが [batch][C][H][wpr] なので、
            チャネル方向に独立な演算はバッチ透過**(チャネル数に枚数を
            掛けるだけ)。それ以外は各バックエンドが枚数を明示的に渡す
        dense_fuse:(既定False)。Trueなら連続するlut_dense列+末尾の
            group_sumを`fuse_dense_layers`で1つの`dense_fuse` opへまとめる
            (Metal/CUDAバックエンドだけが対応、C/Webは未対応のノード種別
            として拒否するのでこのフラグは立てないこと)。
        dense_fuse_max_shared_bytes: `fuse_dense_layers`へそのまま
            渡す共有メモリ上限(既定はApple GPU threadgroupメモリ由来の
            `DENSE_FUSE_MAX_SHARED_BYTES`=32KB、Metal版はこの既定のまま
            呼ぶ)。CUDAはstatic `__shared__`の既定上限48KBまで使えるので、
            `cudabackend.generate_cuda`が`dense_fuse=True`のときだけ
            `CUDA_DENSE_FUSE_MAX_SHARED_BYTES`(48KB)を明示的に渡す
            (バックエンド別の値はcudabackend.py側に置き、ここは
            受け取ったバイト数をそのまま`fuse_dense_layers`へ渡すだけ)。

    Raises:
        UnsupportedNode: 対応していないノードの形が含まれる場合
    """
    if batch < 1:
        raise ValueError(f'batch={batch} は1以上でなければならない')
    b = _PlanBuilder(bundle)
    graph = b.graph
    if len(graph.inputs) != 1:
        raise UnsupportedNode(
            f'入力ポートは1本のみ対応(実際は{len(graph.inputs)}本)')
    in_port = graph.inputs[0]
    in_width = int(in_port.width)

    # 値0 は必ずグラフ入力(as_bitplane の共有判定が id==0 を見る)
    b.new_value(KIND_FLAT, width=in_width, origin='input')

    srcs = node_inputs(graph)
    node_values: dict[str, PlanValue] = {in_port.id: b.values[0]}
    for node in graph.topological_order():
        handler = NODE_KINDS.get(node.type)
        if handler is None:
            reason = UNSUPPORTED_NODE_TYPES.get(node.type)
            raise UnsupportedNode(
                reason or 'デプロイキット未対応のノード型',
                node_id=node.id, node_type=node.type)
        ins = [node_values[s] for s in srcs.get(node.id, [])]
        if not ins:
            raise UnsupportedNode('入力が無い', node_id=node.id,
                                  node_type=node.type)
        b._node = node
        node_values[node.id] = handler(b, node, ins)
        b._node = None

    out_src = graph.output_source()
    if out_src is None:
        raise UnsupportedNode('出力ポートに接続が無い')
    final = node_values[out_src]
    out_width = int(graph.outputs[0].width)
    if final.kind != KIND_SCORES:
        # スコアでない出力は 0/1 をそのまま整数スコアとして書き出す。
        # op にしておくと「これができないバックエンド」を capability で
        # 申告できる(Metal はスコア出力しか扱えない)。
        flat = b.as_flat(final)
        final = b.new_value(KIND_SCORES, width=out_width, origin='to_scores')
        b.add_op('to_scores', [flat], final, width=out_width)

    plan = ExecPlan(
        values=b.values,
        ops=b.ops,
        input_value=0,
        output_value=final.id,
        input_width=in_width,
        output_width=out_width,
        labels=list(bundle.get('labels') or []),
        preproc=(b.preproc_plan if b.preproc_used else None),
        input_bitplane=b.input_bitplane,
        meta=dict(bundle.get('about') or {},
                  name=bundle.get('ir', {}).get('metadata', {}).get('name', '')),
        batch=batch,
        embed_table=bundle.get('embed_table'),
        folded_n_table=bundle.get('folded_n_table'),
        folded_l_table=bundle.get('folded_l_table'),
    )
    # 幅の広い dense だけ、配線の近さでゲート順を詰める(挙動不変)。
    # 全バックエンド共通の効果になるよう、ここ(プラン層)に置く。
    reorder_dense_gates(plan)
    if dense_fuse:
        fuse_dense_layers(plan, max_shared_bytes=dense_fuse_max_shared_bytes)
    return plan
