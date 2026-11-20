"""オーバーレイの設計計算・配線検証・.mem 生成(L2 hdlgen.overlay)。

オーバーレイは「固定回路 + メモリ差替」でモデルを載せる方式で、ゲート記述を
毎サイクル V 語読んで V 並列に 2 入力ゲートを評価する。配線の持ち方は 2 通り:

    computed(計算型配線)   配線 (a, b) をメモリに持たず層ごとのシフト量から
                           算術で生成する。ゲート記述は op 4bit だけになる
                           (保存型 32bit/ゲート比 8x)。配線規則は下の 2 つ
    stored(保存型・自由配線) 1 ゲートの記録に前層アドレスを 2 本持つ
                           ({op 4bit, a, b} = 32 or 40bit)。**任意の添字**
                           (置換配線・ランダム配線・学習配線)を載せられる
                           代わりにゲート ROM が一桁太る(社内先行実験の形式。DE0-CV の実測上限 54,000 ゲート)

wiring='auto' は全層が閉形式に一致すれば computed、外れれば stored を選ぶ。

対応する配線規則(computed のとき層ごとに自動判別する。判別できない IR は
computed では明示拒否、stored なら記録に落とす):
    rot            idx[g] = (g + s) mod W_in
    引き伸ばし rot  idx[g] = (base[g] + s) mod W_in,
                   base[g] = (2*g*W_in + W_out) // (2*W_out)

アドレス生成の実装(RTL hdl/overlay/lgn_overlay.v と 1 対 1):
    レーン c は語ごとに V ゲート飛びで進む。基準位置の分子を
    N = 2*g*W_in + W_out と置くと、g -> g+V で N は 2*V*W_in 増える。
    2*V*W_in = int_step * divisor + rem_step(divisor = 2*W_out、
    int_step = (V*W_in)//W_out、rem_step = (2*V*W_in) mod (2*W_out))なので、
    余りレジスタ r に rem_step を足して divisor 以上なら引く(条件減算 1 段。
    r + rem_step < 2*divisor が保証される)ことで基準位置の歩進が
    int_step + carry として得られる。**乗算器を使わない**。
    rot 層は int_step = V / rem_step = 0 で同じ回路を通る。

層の出力幅 W は全層一様(オーバーレイの活性バッファがこの前提で組まれている)。
W_in == W_out の層では base[g] == g なので引き伸ばし rot は rot と同一に
なり、両者を区別する必要があるのは層 0(W_in = 入力幅)だけである。

M10K 見積りは Quartus の実測(先行実験の op ROM 4 点 + 先行実験のゲート ROM
1 点)に較正した近似で、保証値ではない(advisor と同じ扱い)。較正点は
ESTIMATE_CALIBRATION 参照。
"""

from __future__ import annotations

import json
import math
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from griffith_logic.export.lowering import lower_to_primitives
from griffith_logic.ir.schema import IRGraph
from griffith_logic.ir.serialize import tt_from_hex
from griffith_logic.ir.weights import gate_id_from_tt_n2

# op ROM 1 本の語幅(bit)。V=8 ゲート分の op(4bit)がちょうど 1 語に入る
ROM_WORD_BITS = 32
# M10K 1 個のビット数(Cyclone V)
M10K_BITS = 10240
# 配線の持ち方
WIRING_MODES = ('computed', 'stored', 'auto')
# 計算型配線のゲート記述幅(op のみ)
COMPUTED_GATE_BITS = 4
# 保存型の記録幅と a/b フィールド幅。32bit は先行実験と同一形式
# ({op[31:28], a[27:14], b[13:0]})。層幅が 16,384 を超えるときだけ
# 2*AW + 4 を M10K の最も広いモードの自然幅 40bit へ丸める
STORED_RECORD_FORMATS = ((32, 14), (40, 18))

ESTIMATE_CALIBRATION = """\
先行実験(DE0-CV, Quartus 24.1std)の fit 実測を較正点に持つ近似式:
  幅 32bit: 6000 語 -> 32 blk / 12000 語 -> 48 blk / 3000 語 -> 16 blk /
            1500 語 -> 6 blk(先行実験の op ROM)
  幅 30bit: 6000 語 -> 30 blk(先行実験のレーン別ゲート ROM。240 blk / 8 レーン。
            a/b は 14bit フィールドのうち 13bit しか使わないので、定数 0 の
            2 bit-plane を Quartus が刈り込んだ結果)
  活性バッファ・画像 ROM は ±10% 程度の誤差を持つ(Quartus が
  一部を MLAB へ振るため厳密には予測できない)"""


class OverlayUnsupported(Exception):
    """オーバーレイ化できない IR / 構成(原因と次の一手を含む)。"""


def clog2(n: int) -> int:
    """Verilog $clog2 と同一(n>=1、$clog2(1)=0)。"""
    assert n >= 1
    return int(math.ceil(math.log2(n))) if n > 1 else 0


def bits_for(value: int) -> int:
    """value を表現できる符号なしビット数(value=0 でも 1)。"""
    return max(1, int(value).bit_length())


def stretched_base(g: int, w_in: int, w_out: int) -> int:
    """引き伸ばし rot の基準位置 base[g] = (2*g*W_in + W_out) // (2*W_out)。"""
    return (2 * g * w_in + w_out) // (2 * w_out)


@dataclass(frozen=True)
class LayerPlan:
    """1 層の配線計画(RTL の定数と初期表の元)。

    保存型(kind='stored')では配線が記録に入るので、歩幅・初期値は使わない
    (0 のまま)。層の幅だけが意味を持つ。
    """
    index: int
    kind: str            # 'rot' | 'stretched_rot' | 'stored'
    w_in: int
    w_out: int
    s_a: int
    s_b: int
    int_step: int        # 1 語(V ゲート)進むときの基準位置の整数歩幅
    rem_step: int        # 同、余りの歩幅(rot は 0)
    init_a: tuple[int, ...] = ()   # レーンごとの a 初期値
    init_b: tuple[int, ...] = ()
    init_r: tuple[int, ...] = ()   # レーンごとの余り初期値

    def to_dict(self) -> dict:
        return {'index': self.index, 'kind': self.kind, 'w_in': self.w_in,
                'w_out': self.w_out, 's_a': self.s_a, 's_b': self.s_b,
                'int_step': self.int_step, 'rem_step': self.rem_step}


@dataclass
class OverlayPlan:
    """オーバーレイ 1 構成の設計計算結果(生成器・見積り・検証の単一情報源)。"""
    v: int
    w: int
    depth: int
    in_bits: int
    classes: int
    group: int
    layers: list[LayerPlan]
    ops: np.ndarray = field(repr=False, default=None)   # (depth*w,) uint8
    warnings: list[str] = field(default_factory=list)
    # ---- 配線の持ち方(computed = 計算型 / stored = 保存型・自由配線)----
    wiring: str = 'computed'
    # 保存型のときだけ埋まる: 発行順(層 -> 語 -> レーン)の全添字と記録幅
    idx_a: np.ndarray = field(repr=False, default=None)   # (depth*w,) int64
    idx_b: np.ndarray = field(repr=False, default=None)
    record_bits: int = COMPUTED_GATE_BITS   # 1 ゲートの記録幅(bit)
    addr_field_bits: int = 0                # 記録内の a / b フィールド幅

    # ---- 導出定数(RTL の localparam と 1 対 1)----
    @property
    def stored(self) -> bool:
        return self.wiring == 'stored'
    @property
    def logv(self) -> int:
        return clog2(self.v)

    @property
    def act_words(self) -> int:
        return self.w // self.v

    @property
    def act_addr(self) -> int:
        return clog2(self.act_words)

    @property
    def layer_words(self) -> int:
        return self.w // self.v

    @property
    def gate_words(self) -> int:
        return self.depth * self.w // self.v

    @property
    def gate_addr(self) -> int:
        return clog2(self.gate_words)

    @property
    def nrom(self) -> int:
        """ゲート ROM の本数。計算型は op を詰めた 32bit ROM を ceil(4V/32) 本、
        保存型はレーンごとに 1 本(= V 本。先行実験のレーン分割)。"""
        if self.stored:
            return self.v
        return (4 * self.v + ROM_WORD_BITS - 1) // ROM_WORD_BITS

    @property
    def gate_rom_depth(self) -> int:
        """ゲート ROM 1 本の深さ(語)。"""
        return self.gate_words

    @property
    def gate_rom_blocks(self) -> int:
        """ゲート ROM の M10K ブロック数(見積り。estimate_m10k と同値)。"""
        if self.stored:
            return self.nrom * rom_blocks(self.record_planes,
                                          self.gate_rom_depth)
        return self.nrom * rom32_blocks(self.gate_rom_depth)

    @property
    def record_planes(self) -> int:
        """保存型の記録のうち実際に使う bit 数(定数 0 の面は Quartus が刈る)。

        4(op)+ a の最大値の bit 数 + b の最大値の bit 数。先行実験の 48K
        (W=8000)では 4 + 13 + 13 = 30 で、実測の 30 blk/レーンと一致する。
        """
        if not self.stored:
            return ROM_WORD_BITS
        return (COMPUTED_GATE_BITS + bits_for(int(self.idx_a.max()))
                + bits_for(int(self.idx_b.max())))

    @property
    def in_pad(self) -> int:
        return ((self.in_bits + self.v - 1) // self.v) * self.v

    @property
    def in_words(self) -> int:
        return self.in_pad // self.v

    @property
    def aw(self) -> int:
        """配線アドレス幅。**W をちょうど表現できる幅**(W が 2 の冪でも可)。"""
        return bits_for(self.w)

    @property
    def rem_div(self) -> int:
        """余りアキュムレータの除数 2*W_out(全層一様幅なので単一定数)。"""
        return 2 * self.w

    @property
    def rem_w(self) -> int:
        return bits_for(self.rem_div)

    @property
    def cnt_w(self) -> int:
        """クラス popcount カウンタ幅 = ceil(log2(group+1))。"""
        return bits_for(self.group)

    @property
    def pred_w(self) -> int:
        return bits_for(self.classes - 1)

    @property
    def total_gates(self) -> int:
        return self.depth * self.w

    @property
    def words_per_class(self) -> int:
        return self.group // self.v

    @property
    def cycles_per_image(self) -> int:
        """ceil(IN/V) + DEPTH*(W/V) + 4*DEPTH + 1 + 6(先行実験と同一)。"""
        return (self.in_words + self.depth * self.layer_words
                + 4 * self.depth + 1 + 6)

    @property
    def gate_bits(self) -> int:
        """ゲート記述のビット数/ゲート(計算型配線では op のみの 4bit)。"""
        return self.record_bits

    def to_dict(self) -> dict:
        return {
            'WIRING': self.wiring, 'RECORD_BITS': self.record_bits,
            'ADDR_F': self.addr_field_bits, 'GATE_ROMS': self.nrom,
            'GATE_ROM_DEPTH': self.gate_rom_depth,
            'GATE_ROM_BLOCKS': self.gate_rom_blocks,
            'V': self.v, 'W': self.w, 'DEPTH': self.depth,
            'IN_BITS': self.in_bits, 'IN_PAD': self.in_pad,
            'CLASSES': self.classes, 'GS': self.group,
            'NROM': self.nrom, 'TOTAL_GATES': self.total_gates,
            'GATE_WORDS': self.gate_words, 'LAYER_WORDS': self.layer_words,
            'IN_WORDS': self.in_words, 'ACT_WORDS': self.act_words,
            'AW': self.aw, 'ACT_ADDR': self.act_addr,
            'GATE_ADDR': self.gate_addr, 'REM_DIV': self.rem_div,
            'REM_W': self.rem_w, 'CNT_W': self.cnt_w, 'PRED_W': self.pred_w,
            'WORDS_PER_CLASS': self.words_per_class,
            'CYCLES_PER_IMAGE': self.cycles_per_image,
            'GATE_BITS': self.gate_bits,
            'layers': [ly.to_dict() for ly in self.layers],
            'warnings': list(self.warnings),
        }


# ---------------------------------------------------------------- 配線判別


def detect_layer(indices, w_in: int, w_out: int,
                 layer_index: int = 0) -> tuple[str, int, int]:
    """1 層の (kind, s_a, s_b) を indices から判別する(全ゲート照合)。

    indices は (2, out_width)。閉形式に全数一致しなければ拒否する。
    """
    g = np.arange(w_out, dtype=np.int64)
    base_rot = g % w_in
    base_str = (2 * g * w_in + w_out) // (2 * w_out) % w_in
    kinds: list[str] = []
    shifts: list[int] = []
    for j, name in ((0, 'a'), (1, 'b')):
        row = np.asarray(indices[j], dtype=np.int64)
        if row.min() < 0 or row.max() >= w_in:
            raise OverlayUnsupported(
                f'層 {layer_index} の operand {name} に範囲外の index '
                f'({int(row.min())}..{int(row.max())}, 入力幅 {w_in})。'
                'IR が壊れている可能性がある')
        s = int(row[0])
        if np.array_equal(row, (base_rot + s) % w_in):
            kinds.append('rot')
        elif np.array_equal(row, (base_str + s) % w_in):
            kinds.append('stretched_rot')
        else:
            raise OverlayUnsupported(
                f'層 {layer_index} の operand {name} は rot / 引き伸ばし rot '
                'の配線ではないので overlay 化できない'
                f'(入力幅 {w_in} 出力幅 {w_out}、先頭 index {s} から導いた'
                '閉形式と一致しない)。次の一手: 自由配線のまま載せるなら'
                '保存型(--overlay-wiring stored / auto。記録 32bit/ゲート)、'
                '記述を 4bit に落としたいなら計算型配線(rot 系)で学習した'
                'モデルを使う。空間焼込み(--mode spatial)も選択肢')
        shifts.append(s)
    # a と b で kind が違う場合は「一般形」= 引き伸ばし rot に寄せる
    # (W_in == W_out では両者が同一なので、食い違うのは層 0 だけ)
    kind = 'rot' if kinds[0] == kinds[1] == 'rot' else 'stretched_rot'
    if kind == 'stretched_rot':
        for j in (0, 1):
            row = np.asarray(indices[j], dtype=np.int64)
            if not np.array_equal(row, (base_str + shifts[j]) % w_in):
                raise OverlayUnsupported(
                    f'層 {layer_index}: operand a と b で配線規則が異なる'
                    f'(a={kinds[0]}, b={kinds[1]})。同一層は同じ規則である'
                    '必要がある(アドレス生成器が層単位で 1 つのため)')
    return kind, shifts[0], shifts[1]


def layer_constants(kind: str, w_in: int, w_out: int, v: int) -> tuple[int, int]:
    """(int_step, rem_step) を返す。rot は (V, 0)。"""
    if kind == 'rot':
        return v, 0
    int_step = (v * w_in) // w_out
    rem_step = (2 * v * w_in) % (2 * w_out)
    return int_step, rem_step


def layer_inits(kind: str, w_in: int, w_out: int, v: int,
                s_a: int, s_b: int) -> tuple[list[int], list[int], list[int]]:
    """レーンごとの (a, b, 余り) 初期値。"""
    init_a, init_b, init_r = [], [], []
    div = 2 * w_out
    for c in range(v):
        if kind == 'rot':
            q, r = c, 0
        else:
            n_c = 2 * c * w_in + w_out
            q, r = n_c // div, n_c % div
        init_a.append((q + s_a) % w_in)
        init_b.append((q + s_b) % w_in)
        init_r.append(r)
    return init_a, init_b, init_r


def plan_layer(indices, w_in: int, w_out: int, v: int,
               index: int = 0) -> LayerPlan:
    """1 層の配線を判別して LayerPlan(定数と初期表)を作る。

    RTL 実装可能性(整数歩幅が条件減算 1 段に収まるか等)の検査は
    plan_overlay 側で行う。ここは配線規則の判別と定数計算だけを担う。
    """
    kind, s_a, s_b = detect_layer(indices, w_in, w_out, index)
    int_step, rem_step = layer_constants(kind, w_in, w_out, v)
    init_a, init_b, init_r = layer_inits(kind, w_in, w_out, v, s_a, s_b)
    return LayerPlan(index=index, kind=kind, w_in=w_in, w_out=w_out,
                     s_a=s_a, s_b=s_b, int_step=int_step, rem_step=rem_step,
                     init_a=tuple(init_a), init_b=tuple(init_b),
                     init_r=tuple(init_r))


def simulate_layer(layer: LayerPlan, v: int) -> tuple[np.ndarray, np.ndarray]:
    """1 層ぶんのアキュムレータを回して発行順アドレス列 (a, b) を返す。"""
    words = layer.w_out // v
    a_all = np.empty(layer.w_out, dtype=np.int64)
    b_all = np.empty(layer.w_out, dtype=np.int64)
    acc_a = np.array(layer.init_a, dtype=np.int64)
    acc_b = np.array(layer.init_b, dtype=np.int64)
    rem = np.array(layer.init_r, dtype=np.int64)
    div = 2 * layer.w_out
    w_in = layer.w_in
    for wi in range(words):
        a_all[wi * v:(wi + 1) * v] = acc_a
        b_all[wi * v:(wi + 1) * v] = acc_b
        rem_sum = rem + layer.rem_step
        carry = (rem_sum >= div).astype(np.int64)
        rem = rem_sum - carry * div
        inc = layer.int_step + carry
        acc_a = acc_a + inc
        acc_a -= (acc_a >= w_in) * w_in
        acc_b = acc_b + inc
        acc_b -= (acc_b >= w_in) * w_in
    return a_all, b_all


# ---------------------------------------------------------------- 計画


# 入力側に置かれても overlay の意味論を変えない素通しノード
#   flatten  : 幅を変えない整形(オーバーレイの入力は平坦なビット列)
#   binarize : 0/1 の入力に対しては恒等(閾値が (0,1) のとき)。
#              実数入力の二値化は **呼び出し側が済ませて** images.mem に入れる
_PASSTHROUGH_TYPES = ('flatten', 'binarize')


def _chain_nodes(graph: IRGraph) -> tuple[list, object, list]:
    """入力 -> (素通し) -> lut_layer x D -> group_sum の一本鎖か確かめる。"""
    order = graph.topological_order()
    luts = [n for n in order if n.type == 'lut_layer']
    gs = [n for n in order if n.type == 'group_sum']
    first_lut = next((i for i, n in enumerate(order)
                      if n.type == 'lut_layer'), len(order))
    passthrough = [n for n in order[:first_lut]
                   if n.type in _PASSTHROUGH_TYPES]
    known = set(_PASSTHROUGH_TYPES) | {'lut_layer', 'group_sum'}
    others = [n for n in order if n.type not in known]
    late = [n for n in order[first_lut:] if n.type in _PASSTHROUGH_TYPES]
    if others or late:
        bad = others + late
        raise OverlayUnsupported(
            'オーバーレイ対象外のノードがある: '
            f'{", ".join(f"{n.id}({n.type})" for n in bad[:5])}。'
            'v1 のオーバーレイは (flatten / binarize) + lut_layer 列 + '
            'group_sum の FFN のみ対応')
    if len(gs) != 1 or order[-1].type != 'group_sum':
        raise OverlayUnsupported(
            'group_sum で終端する分類モデルのみ対応(終端ノードが '
            f'{order[-1].type})')
    if len(luts) + len(passthrough) != len(order) - 1:
        raise OverlayUnsupported('lut_layer 列 + group_sum の一本鎖ではない')
    for n in passthrough:
        if n.type == 'binarize' and not (0.0 <= float(n.params['threshold']) < 1.0):
            raise OverlayUnsupported(
                f'{n.id}: binarize の閾値 {n.params["threshold"]} は 0/1 入力に'
                '対して恒等でない。オーバーレイの入力は二値化済みビット列')
    return luts, gs[0], passthrough


def record_format(max_index: int) -> tuple[int, int]:
    """保存型の (記録幅, a/b フィールド幅) を最大添字から決める。"""
    for bits, field_bits in STORED_RECORD_FORMATS:
        if max_index < (1 << field_bits):
            return bits, field_bits
    raise OverlayUnsupported(
        f'前層アドレスの最大値 {max_index} が保存型の記録幅に収まらない'
        f'(a/b フィールドは最大 {STORED_RECORD_FORMATS[-1][1]}bit = '
        f'{1 << STORED_RECORD_FORMATS[-1][1]} 未満)。層幅を下げるか'
        '計算型配線(rot 系)で学習する')


def plan_overlay(graph: IRGraph, v: int = 8,
                 wiring: str = 'computed') -> OverlayPlan:
    """IR からオーバーレイ構成を計画する(配線は全ゲート照合で検証する)。

    Args:
        graph: 分類 IR(lut_layer x D + group_sum)。conv/pool は展開される
        v: レーン数(2 の冪、V % 8 == 0)
        wiring: 'computed'(計算型配線。閉形式に一致しなければ拒否)/
            'stored'(保存型。任意の添字を記録に落とす)/
            'auto'(全層が閉形式なら computed、外れれば stored。判断理由を
            plan.warnings に残す)。**この関数の既定は computed**(「計算型
            として生成できるか」を問う純関数として使う advisor 等のため)。
            build / check / manifest の既定は auto

    Raises:
        OverlayUnsupported: 構造がオーバーレイの前提(一様幅・2 入力ゲート等)
            を満たさない場合、wiring='computed' で配線が rot / 引き伸ばし rot
            でない場合、保存型の記録幅に収まらない場合
    """
    if wiring not in WIRING_MODES:
        raise OverlayUnsupported(
            f'wiring={wiring!r} は非対応({" | ".join(WIRING_MODES)})')
    graph = lower_to_primitives(graph)
    luts, gs, passthrough = _chain_nodes(graph)

    widths = {n.params['out_width'] for n in luts}
    if len(widths) != 1:
        raise OverlayUnsupported(
            f'層幅が一様でない({sorted(widths)})。オーバーレイの活性バッファは'
            '全層同一幅 W を前提にしている')
    w = luts[0].params['out_width']
    depth = len(luts)
    in_bits = luts[0].params['in_width']
    classes = int(gs.params['k'])
    if int(gs.params['in_width']) != w:
        raise OverlayUnsupported(
            f'group_sum の入力幅 {gs.params["in_width"]} が層幅 {w} と異なる')
    if w % classes:
        raise OverlayUnsupported(f'W={w} がクラス数 {classes} で割り切れない')
    group = w // classes

    for n in luts:
        if int(n.params['n_inputs']) != 2:
            raise OverlayUnsupported(
                f'{n.id}: n_inputs={n.params["n_inputs"]}。'
                'オーバーレイの評価器は 2 入力ゲート専用')

    warnings: list[str] = []
    if v < 8 or (v & (v - 1)) or v % 8:
        raise OverlayUnsupported(
            f'V={v} は非対応(2 の冪かつ 8 の倍数。op ROM の 32bit 語詰めの前提)')
    if w % v:
        raise OverlayUnsupported(f'W={w} が V={v} で割り切れない')
    if group % v:
        raise OverlayUnsupported(
            f'group_size={group} が V={v} で割り切れない(集約融合の前提)')
    if w < in_bits:
        raise OverlayUnsupported(
            f'W={w} < 入力幅 {in_bits}(活性バッファが画像を保持できない)')

    # ---- 層ごとの構造検査(配線の持ち方によらない)----
    shapes: list[tuple[int, int, np.ndarray]] = []
    for i, node in enumerate(luts):
        w_in = int(node.params['in_width'])
        w_out = int(node.params['out_width'])
        if i > 0 and w_in != w:
            raise OverlayUnsupported(
                f'層 {i} の入力幅 {w_in} が層幅 {w} と異なる(一様幅が前提)')
        indices = np.asarray(node.params['indices'], dtype=np.int64)
        if indices.shape != (2, w_out):
            raise OverlayUnsupported(
                f'層 {i}: indices 形状 {indices.shape} が (2, {w_out}) でない')
        shapes.append((w_in, w_out, indices))

    def _computed_layers() -> list[LayerPlan]:
        out: list[LayerPlan] = []
        for i, (w_in, w_out, indices) in enumerate(shapes):
            layer = plan_layer(indices, w_in, w_out, v, i)
            if v > w_in:
                raise OverlayUnsupported(
                    f'層 {i}: V={v} > 入力幅 {w_in}'
                    '(条件減算 1 段の mod が壊れる)')
            if layer.int_step + 1 > w_in:
                raise OverlayUnsupported(
                    f'層 {i}: 整数歩幅 {layer.int_step}+1 > 入力幅 {w_in}'
                    '(条件減算 1 段でアドレスが閉じない)')
            out.append(layer)
        return out

    mode = wiring
    layers: list[LayerPlan] = []
    if wiring in ('computed', 'auto'):
        try:
            layers = _computed_layers()
            mode = 'computed'
        except OverlayUnsupported as exc:
            if wiring == 'computed':
                raise
            mode = 'stored'
            warnings.append(
                '配線が閉形式(rot / 引き伸ばし rot)に一致しないので保存型'
                f'(自由配線)で生成する: {str(exc).split("。")[0]}')
    idx_a = idx_b = None
    record_bits, field_bits = COMPUTED_GATE_BITS, 0
    if mode == 'stored':
        layers = []
        for i, (w_in, w_out, indices) in enumerate(shapes):
            if indices.min() < 0 or indices.max() >= w_in:
                raise OverlayUnsupported(
                    f'層 {i} に範囲外の index '
                    f'({int(indices.min())}..{int(indices.max())}、'
                    f'入力幅 {w_in})。IR が壊れている可能性がある')
            layers.append(LayerPlan(index=i, kind='stored', w_in=w_in,
                                    w_out=w_out, s_a=0, s_b=0,
                                    int_step=0, rem_step=0))
        idx_a = np.concatenate([ix[0] for _, _, ix in shapes]).astype(np.int64)
        idx_b = np.concatenate([ix[1] for _, _, ix in shapes]).astype(np.int64)
        # RTL は a[LOGV +: ACT_ADDR] で活性語アドレスを取り出す(合計
        # ceil(log2(W)) bit)。実際の最大添字が小さくてもこの幅は要るので、
        # 記録幅は「最大添字」と「活性語アドレス幅」の両方で決める
        need = clog2(v) + clog2(w // v)
        record_bits, field_bits = record_format(
            max(int(idx_a.max()), int(idx_b.max()), (1 << need) - 1))

    ops = np.empty(depth * w, dtype=np.uint8)
    for i, node in enumerate(luts):
        tts = tt_from_hex(node.params['truth_tables'])
        if len(tts) != w:
            raise OverlayUnsupported(
                f'層 {i}: truth_tables 数 {len(tts)} != W {w}')
        ops[i * w:(i + 1) * w] = [gate_id_from_tt_n2(int(t)) for t in tts]

    if passthrough:
        warnings.append(
            '入力側の素通しノード ' +
            ', '.join(f'{n.id}({n.type})' for n in passthrough) +
            ' は overlay では回路にしない(images.mem に入れるベクトルが'
            '二値化済みであることが前提)')
    plan = OverlayPlan(v=v, w=w, depth=depth, in_bits=in_bits,
                       classes=classes, group=group, layers=layers, ops=ops,
                       warnings=warnings, wiring=mode,
                       idx_a=idx_a, idx_b=idx_b, record_bits=record_bits,
                       addr_field_bits=field_bits)
    if plan.in_pad > w:
        raise OverlayUnsupported(
            f'IN_PAD={plan.in_pad} > W={w}(パディング後の画像が活性バッファに'
            '収まらない)')
    aw_words = plan.act_words
    alloc = 1 << plan.act_addr
    if alloc > aw_words * 1.1:
        warnings.append(
            f'W/V = {aw_words} に対し活性バッファは 2^{plan.act_addr} = {alloc} 語'
            f'ぶんの M10K を確保する(無駄 {100 * (alloc / aw_words - 1):.0f}%)。'
            'W/V を 2 の冪に近づけると複製が減る'
            '(先行実験の設計則「W/V は 2 の冪以下に保つ」)')
    verify_wiring(plan, graph)
    return plan


# ---------------------------------------------------------------- 配線検証


def simulate_addresses(plan: OverlayPlan) -> tuple[np.ndarray, np.ndarray]:
    """RTL のアキュムレータを Python で 1 語ずつ回し、発行順のアドレス列を返す。

    返り値は (a, b) それぞれ (depth*W,) の int64。順序は
    層 major -> 層内語 -> レーン(= RTL の発行順 = addrref の並び)。
    保存型では発行順のアドレス = 記録の添字そのもの。
    """
    if plan.stored:
        return plan.idx_a, plan.idx_b
    a_all = np.empty(plan.total_gates, dtype=np.int64)
    b_all = np.empty(plan.total_gates, dtype=np.int64)
    for i, ly in enumerate(plan.layers):
        a_seg, b_seg = simulate_layer(ly, plan.v)
        a_all[i * plan.w:(i + 1) * plan.w] = a_seg
        b_all[i * plan.w:(i + 1) * plan.w] = b_seg
    return a_all, b_all


def verify_wiring(plan: OverlayPlan, graph: IRGraph) -> dict:
    """発行順アドレス(計算型はアキュムレータ、保存型は記録の添字)を
    IR の indices と全数照合する。

    生成時のセルフチェック(先行実験 gen_opmem のチェック ii と同じ役割)。
    不一致があれば OverlayUnsupported を投げ、.mem を書かない。
    """
    graph = lower_to_primitives(graph)
    luts = [n for n in graph.topological_order() if n.type == 'lut_layer']
    a_all, b_all = simulate_addresses(plan)
    checked = 0
    for i, node in enumerate(luts):
        ly = plan.layers[i]
        idx = np.asarray(node.params['indices'], dtype=np.int64)
        # 発行順(層 -> 語 -> レーン)の位置 w*V + c は ゲート番号 g と一致する
        seg_a = a_all[i * plan.w:(i + 1) * plan.w]
        seg_b = b_all[i * plan.w:(i + 1) * plan.w]
        if not np.array_equal(seg_a, idx[0]) or not np.array_equal(seg_b, idx[1]):
            bad = int(np.argmax((seg_a != idx[0]) | (seg_b != idx[1])))
            raise OverlayUnsupported(
                f'層 {i} ゲート {bad}: アドレス生成器の再現値 '
                f'(a={int(seg_a[bad])}, b={int(seg_b[bad])}) が IR の indices '
                f'(a={int(idx[0][bad])}, b={int(idx[1][bad])}) と一致しない。'
                'overlay 化を中止する(生成器の不具合か、対応外の配線規則)')
        if seg_a.min() < 0 or seg_a.max() >= ly.w_in or \
                seg_b.min() < 0 or seg_b.max() >= ly.w_in:
            raise OverlayUnsupported(
                f'層 {i}: 生成アドレスが入力幅 {ly.w_in} の範囲外')
        checked += 2 * plan.w
    return {'addr_entries_checked': checked}


# ---------------------------------------------------------------- .mem 生成


def emit_opmem(plan: OverlayPlan) -> list[list[str]]:
    """op ROM の行(32bit hex)を NROM 本ぶん返す。

    語 w = ゲート {w*V .. w*V+V-1} の op。ROM r はレーン 8r..8r+7 を担当し、
    語内では レーン 8r+j を bits[4j+3:4j] に置く(lane0 = LSB)。
    """
    ops = plan.ops.astype(np.int64)
    v, nrom = plan.v, plan.nrom
    words = ops.reshape(-1, v)                       # (gate_words, V)
    roms: list[list[str]] = []
    for r in range(nrom):
        lanes = words[:, 8 * r:8 * r + 8]            # (gate_words, 8)
        shifts = (4 * np.arange(8, dtype=np.int64))[None, :]
        val = (lanes << shifts).sum(axis=1)
        roms.append([format(int(x), '08x') for x in val])
    return roms


def check_opmem(plan: OverlayPlan, roms: list[list[str]]) -> int:
    """opmem の全ニブルを plan.ops と突合する(生成時セルフチェック i)。"""
    ops = plan.ops.astype(np.int64)
    checked = 0
    for wi in range(plan.gate_words):
        val = 0
        for r in range(plan.nrom):
            val |= int(roms[r][wi], 16) << (ROM_WORD_BITS * r)
        for k in range(plan.v):
            got = (val >> (4 * k)) & 0xF
            exp = int(ops[wi * plan.v + k])
            if got != exp:
                raise OverlayUnsupported(
                    f'opmem 語 {wi} レーン {k}: {got} != {exp}(生成器の不具合)')
            checked += 1
    return checked


def _hex_lines(values, digits: int) -> str:
    return '\n'.join(format(int(x), f'0{digits}x') for x in values) + '\n'


def emit_addrinit(plan: OverlayPlan) -> dict[str, str]:
    """層頭でロードするアドレス/余りの初期表(3 ファイル分の本文)。"""
    ad = (plan.aw + 3) // 4
    rd = (plan.rem_w + 3) // 4
    init_a = [x for ly in plan.layers for x in ly.init_a]
    init_b = [x for ly in plan.layers for x in ly.init_b]
    init_r = [x for ly in plan.layers for x in ly.init_r]
    return {
        'addrinit_a.mem': _hex_lines(init_a, ad),
        'addrinit_b.mem': _hex_lines(init_b, ad),
        'reminit.mem': _hex_lines(init_r, rd),
    }


def pack_record(op: int, a: int, b: int, field_bits: int) -> int:
    """保存型のゲート記録 = {op 4bit, a, b}(MSB 側から op, a, b)。

    field_bits=14 なら {op[31:28], a[27:14], b[13:0]} = 先行実験と同一形式。
    """
    if not 0 <= op < 16:
        raise OverlayUnsupported(f'op={op} が 4bit に収まらない')
    limit = 1 << field_bits
    if not (0 <= a < limit and 0 <= b < limit):
        raise OverlayUnsupported(
            f'アドレス (a={a}, b={b}) が {field_bits}bit に収まらない')
    return (op << (2 * field_bits)) | (a << field_bits) | b


def unpack_record(word: int, field_bits: int) -> tuple[int, int, int]:
    """記録 1 語を (op, a, b) に戻す(pack_record の逆)。"""
    mask = (1 << field_bits) - 1
    return (word >> (2 * field_bits)) & 0xF, \
        (word >> field_bits) & mask, word & mask


def emit_gatemem(plan: OverlayPlan) -> list[list[str]]:
    """保存型のゲート記録 ROM(レーン別)の行を V 本ぶん返す。

    レーン k のファイルの語 w = ゲート w*V + k の記録(先行実験と同じ割付)。
    1 行 = 1 語の hex(記録幅 / 4 桁)。
    """
    if not plan.stored:
        raise OverlayUnsupported('emit_gatemem は保存型の plan にだけ使える')
    f = plan.addr_field_bits
    digits = plan.record_bits // 4
    ops = plan.ops.astype(np.int64).reshape(-1, plan.v)
    a = plan.idx_a.reshape(-1, plan.v)
    b = plan.idx_b.reshape(-1, plan.v)
    if a.max() >= (1 << f) or b.max() >= (1 << f):
        raise OverlayUnsupported(
            f'アドレスが {f}bit フィールドに収まらない(生成器の不具合)')
    roms: list[list[str]] = []
    for k in range(plan.v):
        vals = (ops[:, k] << (2 * f)) | (a[:, k] << f) | b[:, k]
        roms.append([format(int(x), f'0{digits}x') for x in vals])
    return roms


def check_gatemem(plan: OverlayPlan, roms: list[list[str]]) -> int:
    """記録 ROM の全語を復号して IR の (op, a, b) と突合する。

    生成時セルフチェック(計算型の check_opmem + verify_wiring に相当)。
    照合したゲート数を返す。
    """
    f = plan.addr_field_bits
    if len(roms) != plan.v:
        raise OverlayUnsupported(
            f'記録 ROM が {len(roms)} 本(V={plan.v} 本であるべき)')
    checked = 0
    for k, lines in enumerate(roms):
        if len(lines) != plan.gate_words:
            raise OverlayUnsupported(
                f'レーン {k} の語数 {len(lines)} != {plan.gate_words}')
        for wi, line in enumerate(lines):
            g = wi * plan.v + k
            op, a, b = unpack_record(int(line, 16), f)
            exp = (int(plan.ops[g]), int(plan.idx_a[g]), int(plan.idx_b[g]))
            if (op, a, b) != exp:
                raise OverlayUnsupported(
                    f'記録 レーン {k} 語 {wi}(ゲート {g}): '
                    f'(op={op}, a={a}, b={b}) が IR の '
                    f'(op={exp[0]}, a={exp[1]}, b={exp[2]}) と一致しない'
                    '(生成器の不具合か .mem の破損)')
            checked += 1
    return checked


def emit_addrref(plan: OverlayPlan) -> dict[str, str]:
    """アドレス生成 TB 用の参照列(発行順の全アドレス)。"""
    a_all, b_all = simulate_addresses(plan)
    ad = (plan.aw + 3) // 4
    return {'addrref_a.mem': _hex_lines(a_all, ad),
            'addrref_b.mem': _hex_lines(b_all, ad)}


# ---------------------------------------------------------------- 資源見積り


def rom32_blocks(depth_words: int) -> int:
    """32bit 幅 ROM(depth_words 語)の M10K ブロック数(先行実験較正の近似)。

    Quartus は深さ 2048 語を超えると 4096 語(=16 blk)単位で確保し、
    それ以下は 256 語(=1 blk)単位で刈り込む、という挙動に一致する。
    """
    if depth_words <= 0:
        return 0
    if depth_words > 2048:
        return 16 * math.ceil(depth_words / 4096)
    return math.ceil(depth_words / 256)


def rom_blocks(width_bits: int, depth_words: int) -> int:
    """任意幅の同期読み ROM の M10K ブロック数(**近似**。保証値ではない)。

    rom32_blocks(幅 32bit)を幅方向へ一般化したもので、Quartus の M10K 構成
    モード(深いときは 4096 語 x 2bit、浅いときは 256 語 x 40bit)を
    実測から当てはめた式:

        depth > 2048 : ceil(width/2)  * ceil(depth/4096)
        それ以外      : ceil(width/40) * ceil(depth/256)

    較正点(いずれも DE0-CV / Quartus 24.1std の fit 実測):
      幅 32bit: 6000 語 32 / 12000 語 48 / 3000 語 16 / 1500 語 6(先行実験)
      幅 30bit: 6000 語 30(先行実験のゲート ROM 240 blk / 8 レーン)
    保存型の記録は 32bit 宣言でも a/b の上位が定数 0 になるため、Quartus が
    刈り込んだ後の実効 bit-plane 数(OverlayPlan.record_planes)を幅に渡す。
    深さが 8192 語を超える領域は較正点が無いので外挿である。
    """
    if width_bits <= 0 or depth_words <= 0:
        return 0
    if depth_words > 2048:
        return math.ceil(width_bits / 2) * math.ceil(depth_words / 4096)
    return math.ceil(width_bits / 40) * math.ceil(depth_words / 256)


# Cyclone V M10K の構成モード(深さ, 幅)。幅広・浅い ROM では Quartus が
# この中から最も少ないブロック数になるモードを選ぶ(実測: 784bit x 256 枚 =
# 20 blk = x40 モード / x512 枚 = 40 blk = x20 モード / x1000 枚 = 78〜80 blk
# = x10 モード。先行実験 §4・先行実験 fit)
_M10K_MODES = ((8192, 1), (4096, 2), (2048, 5), (1024, 10), (512, 20),
               (256, 40))


def _wide_rom_blocks(width_bits: int, depth_words: int) -> int:
    """幅広 ROM(画像 ROM 等)の M10K ブロック数(近似)。

    モードごとの ceil(幅/モード幅) x ceil(深さ/モード深さ) の**最小**を採る。
    深く狭い ROM(ゲート ROM)はこの最小化に従わない(Quartus が浅広モードを
    選ばない)ので rom_blocks / rom32_blocks を使うこと。
    """
    if width_bits <= 0 or depth_words <= 0:
        return 0
    return min(math.ceil(width_bits / w) * math.ceil(depth_words / d)
               for d, w in _M10K_MODES)


def estimate_m10k(plan: OverlayPlan, n_images: int = 0) -> dict:
    """M10K 使用ブロック数の内訳見積り(ゲート ROM / 活性 / 画像 ROM)。

    ゲート ROM は構造で決まるため実測と一致する(計算型 48K の 32 blk /
    96K の 48 blk、保存型 48K V=8 の 240 blk)。活性・画像は Quartus の
    割付に依存するため近似(±10% 程度)。'op_rom' と 'gate_rom' は同値
    (前者は計算型時代からの名前、後者は fit 報告の内訳と同じ名前)。
    """
    op_rom = plan.gate_rom_blocks
    act_words_alloc = 1 << plan.act_addr
    per_buf = math.ceil(act_words_alloc * plan.v / M10K_BITS)
    activation = 2 * plan.v * per_buf
    image_rom = _wide_rom_blocks(plan.in_bits, n_images)
    label_rom = _wide_rom_blocks(plan.pred_w, n_images)
    total = op_rom + activation + image_rom + label_rom
    return {'op_rom': op_rom, 'gate_rom': op_rom, 'activation': activation,
            'image_rom': image_rom, 'label_rom': label_rom, 'total': total,
            'note': '見積り(ゲート ROM は構造値、活性・画像は近似)'}


def estimate_gate_rom_bits(plan: OverlayPlan) -> dict:
    """ゲート記述のビット数(計算型配線 4bit/ゲート vs 保存型)。

    保存型は 1 ゲート = op 4bit + 前層アドレス 2 本(各 ceil(log2(W)) bit)。
    実装では語幅を 2 の冪へ切り上げてメモリに置く(先行実験の実装は
    {op[31:28], a[27:14], b[13:0]} = 32bit/ゲート)ため、比はその語幅で出す。
    plan の wiring によらず**両方式の比**を返す(比較用の純関数)。
    """
    computed = plan.total_gates * COMPUTED_GATE_BITS
    addr_bits = clog2(plan.w)
    minimal_word = 4 + 2 * addr_bits
    stored_word = 1 << clog2(minimal_word)
    stored = plan.total_gates * stored_word
    return {'computed_bits': computed, 'stored_bits': stored,
            'stored_word_bits': stored_word,
            'stored_word_bits_minimal': minimal_word,
            'addr_bits': addr_bits,
            'ratio': stored / computed if computed else 0.0}


# ---------------------------------------------------------------- 生成

STATIC_SOURCES = ('lgn_overlay.v', 'rom32.v', 'tdp_ram.v', 'gate_eval.v',
                  'overlay_harness.v')
# 保存型(自由配線)のテンプレート。rom32 / tdp_ram / gate_eval は共用
STORED_SOURCES = ('lgn_overlay_stored.v', 'gate_rom.v', 'rom32.v', 'rom40.v',
                  'tdp_ram.v', 'gate_eval.v', 'overlay_harness_stored.v')
# シム専用(Quartus のソースには入れない): ISSP のダミー
SIM_ONLY_SOURCES = ('altsource_probe_stub.v',)
HDL_DIR = Path(__file__).resolve().parent.parent / 'hdl' / 'overlay'


def static_sources(wiring: str) -> tuple[str, ...]:
    """配線の持ち方に応じた静的 RTL テンプレートの一覧。"""
    return STORED_SOURCES if wiring == 'stored' else STATIC_SOURCES


def wiring_label(plan: OverlayPlan) -> str:
    """人向けの 1 行表記(CLI / 報告で共通に使う)。"""
    kinds = '/'.join(sorted({ly.kind for ly in plan.layers}))
    if plan.stored:
        return f'stored(自由配線、記録 {plan.record_bits}bit/ゲート)'
    return f'computed({kinds}、記述 {plan.gate_bits}bit/ゲート)'


def generate_argmax(plan: OverlayPlan, module_name: str = 'overlay_argmax') -> str:
    """クラススコアの argmax 比較木モジュール(同点は小 index 勝ち)。

    比較木は hdlgen.system と同一の生成器を使い、emit 前に
    argmax_tree_self_test(乱数 + 同点多発)で np.argmax 一致を確かめる。
    """
    from .system import _emit_argmax_tree, argmax_tree_self_test

    k, sw, pw = plan.classes, plan.cnt_w, plan.pred_w
    body: list[str]
    if k < 2:
        body = [f"    wire [{pw - 1}:0] pred = {pw}'d0;", '    wire _unused = |scores;']
    else:
        argmax_tree_self_test(k, plan.group)
        body = _emit_argmax_tree(k, sw, pw, 'scores')
    return (
        f'// mdl-fpga 自動生成 argmax 比較木(CLASSES={k}, スコア幅={sw}bit)\n'
        '// 同点は小さい index が勝つ(= numpy.argmax)。生成時に自己テスト済み。\n'
        '`timescale 1ns/1ps\n'
        f'module {module_name} (\n'
        f'    input  wire [{k * sw - 1}:0] scores,\n'
        f'    output wire [{pw - 1}:0] pred_index\n'
        ');\n'
        + '\n'.join(body) + '\n'
        '    assign pred_index = pred;\n'
        'endmodule\n')


@dataclass
class OverlayGen:
    """生成物一式(RTL テキストと .mem の本文)。"""
    plan: OverlayPlan
    argmax_verilog: str
    mems: dict[str, str] = field(default_factory=dict)
    sources: list[str] = field(default_factory=list)


def generate_overlay(plan: OverlayPlan, *, with_addrref: bool = False) -> OverlayGen:
    """計画から RTL(生成部)と .mem の本文を作る(ファイル書き出しは別)。

    .mem は書き出す前に必ず復号して IR と全数照合する(計算型は
    check_opmem + verify_wiring、保存型は check_gatemem)。
    """
    if plan.stored:
        roms = emit_gatemem(plan)
        check_gatemem(plan, roms)
        mems = {f'gatemem_lane{k}.mem': '\n'.join(lines) + '\n'
                for k, lines in enumerate(roms)}
    else:
        roms = emit_opmem(plan)
        check_opmem(plan, roms)
        mems = {f'opmem{r}.mem': '\n'.join(lines) + '\n'
                for r, lines in enumerate(roms)}
        mems.update(emit_addrinit(plan))
        if with_addrref:
            mems.update(emit_addrref(plan))
    return OverlayGen(
        plan=plan,
        argmax_verilog=generate_argmax(plan),
        mems=mems,
        sources=['overlay_argmax.v', *static_sources(plan.wiring)],
    )


def copy_static_overlay_hdl(dest_dir: str | Path,
                            include_sim_only: bool = False,
                            wiring: str = 'computed') -> list[str]:
    """静的 RTL テンプレートを作業ディレクトリへ複製する。"""
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    names = list(static_sources(wiring))
    if include_sim_only:
        names += list(SIM_ONLY_SOURCES)
    for name in names:
        shutil.copyfile(HDL_DIR / name, dest / name)
    return names


def write_overlay_files(gen: OverlayGen, outdir: str | Path, *,
                        include_sim_only: bool = False) -> list[str]:
    """.mem・生成 RTL・静的 RTL・params.json を outdir へ書き出す。"""
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    for name, text in gen.mems.items():
        (out / name).write_text(text, encoding='utf-8')
    (out / 'overlay_argmax.v').write_text(gen.argmax_verilog, encoding='utf-8')
    copy_static_overlay_hdl(out, include_sim_only=include_sim_only,
                            wiring=gen.plan.wiring)
    (out / 'overlay_params.json').write_text(
        json.dumps(gen.plan.to_dict(), ensure_ascii=False, indent=2),
        encoding='utf-8')
    return sorted(gen.mems) + list(gen.sources)
