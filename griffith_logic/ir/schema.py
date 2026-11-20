"""magicdelta-ir: モデル中間表現のスキーマ。

PythonコアAPI・ビジュアルエディタ・HDLトランスパイラをつなぐ共通表現。

設計判断:
    - ノード粒度はレイヤ単位(ゲート単位に展開しない)。ゲートビューは
      indices + truth_tables から導出可能
    - 真理値表は「2^Nビット整数」に正規化: bit k = butterfly基底順の
      入力パターンk(bit j = 入力スロットjの値)に対する出力。
      JSON上はhex文字列(N=6は64bitでJSの安全整数を超えるため)
    - パラメータ化方式の差異はIRに持ち込まない(抽出時に真理値表へ確定)。
      train_info は参考情報でありトランスパイラは参照しない
    - meta はエディタ専用の自由namespace(座標・表示名等)。
      トランスパイラは完全に無視する
    - 非論理層(ハイブリッドモデルの実数部)は type='blackbox'

ノード型(v0.1):
    lut_layer  — params: in_width, out_width, n_inputs,
                 indices(list[list[int]], (n_inputs, out_width)),
                 truth_tables(list[hex文字列], (out_width,))
    group_sum  — params: in_width, k, tau, bias(任意。クラスごとの整数の判定バイアス、
                 長さ k。省略 = 全 0。simulate はクラス和に足す。HDL / C は未対応で
                 validate(target='hdl') が bias≠0 を拒む)
    binarize   — params: width, threshold
    flatten    — params: in_shape, width
    blackbox   — params: torch_class, config, in_width, out_width

ノード型(v0.2、多入力DAG対応):
    concat     — params: in_widths(list[int]、slot順)。出力幅 = sum(in_widths)
    add        — params: width(全slot同幅), mode('or'|'xor'|'sum')

ノード型(v0.2、定数線):
    constant   — params: width, bits(list[0|1]、長さ=width)。
                 入力なし(arity 0)のルートノード。simulate=定数行、
                 torch=バッチ方向へexpand、HDL=ハードワイヤ定数。
                 bits[j] が出力ビットj(concat規約と同じくindex 0が下位)

ノード型(v0.2、conv/poolネイティブ表現。A2):
    conv2d     — params: in_shape([C,H,W]), out_channels,
                 kernel_size([kH,kW]), stride([sH,sW]), padding([pH,pW]),
                 tree_depth, n_inputs,
                 indices_flat((out_channels, n_inputs^tree_depth)、
                 LutConv2dのindices_flatバッファ=受容野内フラット位置),
                 truth_tables(hex文字列list、長さ=out_channels*内部ノード数。
                 oc優先(oc-major)で木のレベル順=LutConv2d.weightsのdim1順)
    or_pool2d  — params: in_shape([C,H,W]), kernel_size, stride, padding(既定0)

conv/pool幅規約: 入出力とも width = C*H*W のNCHW flatten
(入力幅=prod(in_shape)、出力幅=out_channels*H_out*W_out)。
HDL生成前に export/lowering.lower_to_primitives でlut_layer列へ展開する。

ノード型(v0.2、恒等配線ネイティブ表現):
    upsample2d — params: in_shape([C,H,W]), scale(int、最近傍拡大倍率)。
                 out[c,y,x] = in[c, y//scale, x//scale] の固定配線
                 (out_shape=[C, H*scale, W*scale])。indexは持たず、
                 実行時(simulate/lower_to_primitives)にJIT計算する
    permute    — params: in_shape(list[int]、入力を見なすreshape形状。
                 prod(in_shape)=in_width), perm(list[int]、
                 range(len(in_shape))の順列)。out_shape=[in_shape[p] for p
                 in perm]、out_width=in_width(要素数不変の軸並べ替え)。
                 SpatialGroupSumのチャネル→空間並べ替えやwebbenchの
                 rgb/thr順序変換等、恒等配線を汎用的に表現する
                 (indexは持たずJIT計算。lut_layerへ全展開すると
                 1位置=1エントリで肥大するため導入)

恒等配線幅規約: upsample2d/permuteとも入出力width計算はin_shapeの積
(upsample2dは拡大後、permuteは不変)。HDL生成前は conv2d/or_pool2d同様に
export/lowering.lower_to_primitives でlut_layer列へ展開する
(後方互換: 旧export_ir(native_conv=False)は従来どおりlut_layerへ
直接展開したIRを返す)。

v0.2 スキーマ変更(v0.1完全後方互換):
    - IREdge.dst_slot: 多入力ノードの入力スロット番号(既定0)。
      v0.1 JSON(dst_slotなし)は読み込み時に0扱い。
      dst_slot=0のエッジはJSON出力時にキーを省略する(表現の安定性)
    - fanout(同一srcから複数エッジ)を許容(専用ノード不要)

concatビット順規約(simulate/torch/HDLで完全統一):
    slot 0 が出力ビット列の先頭(インデックス0側)= 整数パック時の下位ビット。
    - numpy/torch: concatenate([slot0, slot1, ...], axis=-1)
    - Verilog:     assign w = {slotN-1, ..., slot1, slot0};  ({}は左がMSB)
    - VHDL:        w <= slotN-1 & ... & slot1 & slot0;       (&は左がMSB)

addビット厳密意味論:
    or/xor はビットOR/XOR。sum は整数和で出力が{0,1}を超えるため
    論理領域の終端(group_sum直前等)専用。HDL生成は不可(validateで拒否)。
    soft(torch学習時)は or→1-(1-a)(1-b)、xor→a+b-2ab、sum→a+b(微分可能)

ノード型(v0.2、リカレント表現):
    scan_var   — params: width(int), role('state'|'step'|'invariant')。
                 recurrentのbodyグラフ内でのみ使う、入力の「穴」を表す
                 arity 0ノード。role='state' のときのみ
                 init={'kind':'const', 'bits':[0|1,...]}を持つ
                 (kind!='const'は未対応。role!='state'はinitを持たない)。
                 bodyの外(親グラフ)には現れない
    recurrent  — params: steps(int), body(IRGraph), scan_vars(list of
                 {id,role,width}), state_update(dict: scan_var_id →
                 body内ノードid), output_mode('final_state'|'stacked'),
                 output_source_id(str、body内ノードid),
                 output_width(int、1ステップぶんの出力幅)。
                 親からの入力エッジ(dst_slot昇順)は scan_vars のうち
                 role∈{'step','invariant'}の並びに1対1対応する。
                 step入力は[steps*width]の平坦ベクトル(親視点では
                 全ステップぶんを連結した1本)、invariantは[width]
                 そのまま(全ステップで共通)。
                 bodyはunrollしない(ONNX Scan型。「1回だけ記述し、
                 ランタイムがsteps回実行する」)。state 0個+
                 output_mode='stacked'で純フィードフォワードの
                 「全ステップに同じ重みを適用してstack」も表現できる
                 (weight共有・配信サイズ削減のため)。
                 body内は既存ノード(lut_layer/concat/group_sum)と
                 scan_varのみ

ノード型(v0.2、縮約済みdense平坦ネットリスト):
    lut_flat   — params: in_width(入力ビット数), out_width(出力ビット数),
                 gates(list[[tt, a, b]]), outputs(list[int]、長さ=out_width)。
                 **連続する2入力lut_layer列を、`export/netlist_reduce.py` で
                 lossless縮約(定数・パススルー・NOT・CSE・未参照除去)した
                 平坦DAG**。`deploy/flatten_dense.py` が lut_layer 列を置換して
                 作る(学習側からは作らない。torchへは戻せない)。
                 simulate時に出力ビット列(=置換前の最終lut_layerと同じ
                 out_width 本のビット)を返すので、後段のgroup_sum等は無改造。

                 信号番号の規約(a/b/outputsが参照する「信号」の添字):
                     0                       定数0
                     1 .. in_width           入力ビット k → 信号 1+k
                     in_width+1 .. +n_gates  ゲート g の出力 → 信号 in_width+1+g
                 ゲート g は [tt, a, b]: tt は 0..15 の真理値表で lut_layer と
                 同じ規約(bit index = (b<<1)|a、出力 = (tt >> idx) & 1)。
                 a, b は信号番号で、**自分より前の信号**(< in_width+1+g)
                 のみを指せる(トポロジカル順)。極性(NOT)はゲート入力側には
                 持たない(縮約器が下流ゲートの真理値表へ吸収済み)。
                 outputs[i] は出力ビット i の参照で、`2*信号番号 + 極性`
                 (極性1 = 反転)。定数1は信号0の反転 = 1。
"""

from dataclasses import dataclass, field

IR_FORMAT = 'magicdelta-ir'
IR_VERSION = '0.2.0'

NODE_TYPES = ('lut_layer', 'group_sum', 'binarize', 'flatten', 'blackbox',
              'concat', 'add', 'conv2d', 'or_pool2d', 'constant',
              'upsample2d', 'permute', 'recurrent', 'scan_var', 'lut_flat')

ADD_MODES = ('or', 'xor', 'sum')


@dataclass
class IRPort:
    """グラフの入出力ポート"""
    id: str
    width: int
    shape: list | None = None
    encoding: str | None = None    # 例: 'binary_threshold_0.5'
    semantics: str | None = None   # 例: 'class_scores'


@dataclass
class IRNode:
    id: str
    type: str
    params: dict = field(default_factory=dict)
    train_info: dict = field(default_factory=dict)  # 参考情報(HDL不使用)
    meta: dict = field(default_factory=dict)        # エディタ専用namespace


@dataclass
class IREdge:
    src: str  # ノードid or 入力ポートid
    dst: str  # ノードid or 出力ポートid
    dst_slot: int = 0  # dstの入力スロット番号(v0.2、多入力ノード用)


@dataclass
class IRGraph:
    metadata: dict = field(default_factory=dict)
    inputs: list = field(default_factory=list)    # list[IRPort]
    outputs: list = field(default_factory=list)   # list[IRPort]
    nodes: list = field(default_factory=list)     # list[IRNode]
    edges: list = field(default_factory=list)     # list[IREdge]
    format: str = IR_FORMAT
    version: str = IR_VERSION

    def node_by_id(self, node_id: str) -> IRNode:
        for n in self.nodes:
            if n.id == node_id:
                return n
        raise KeyError(f'ノードが存在しない: {node_id}')

    def output_source(self) -> str | None:
        """最初の出力ポートへ接続されたsrc idを返す(未接続ならNone)。

        simulate/build_torch/HDL生成が「最終出力」を同一規則で決めるための
        共通ヘルパー。直列グラフではトポ末尾ノードと一致する。
        """
        if not self.outputs:
            return None
        out_id = self.outputs[0].id
        for e in self.edges:
            if e.dst == out_id:
                return e.src
        return None

    def topological_order(self) -> list:
        """エッジからノードのトポロジカル順を返す(循環はvalidateで検出)"""
        input_ids = {p.id for p in self.inputs}
        output_ids = {p.id for p in self.outputs}
        indeg = {n.id: 0 for n in self.nodes}
        succ: dict[str, list[str]] = {}
        for e in self.edges:
            if e.dst in indeg and e.src not in input_ids:
                indeg[e.dst] += 1
            succ.setdefault(e.src, []).append(e.dst)
        order, queue = [], [nid for nid, d in indeg.items() if d == 0]
        while queue:
            nid = queue.pop(0)
            order.append(nid)
            for m in succ.get(nid, []):
                if m in output_ids or m not in indeg:
                    continue
                indeg[m] -= 1
                if indeg[m] == 0:
                    queue.append(m)
        return [self.node_by_id(nid) for nid in order]
