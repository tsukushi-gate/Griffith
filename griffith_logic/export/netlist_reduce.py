"""dense DLGN 平坦ネットリストの論理縮約器(実効ゲート数 U の測定装置)。

社内先行実験パイロット `pilot/experiments/irreducible-dlgn-p0/reduce_netlist.py`
(以下 P0)の忠実な移植。`export/optimize.py` の縮約は **conv2d 重み共有木専用**
(`optimize.py:725` で `node.type != 'conv2d'` を弾く)で dense の lut_layer 列には
掛からないため、別モジュールとして置く。真理値表の代数だけ optimize.py から
import して再利用する(あちらは TS `convProgram.ts` のミラーという契約があるので
本モジュールの規則をあちらへ足さない)。

入力形式(P0 の netlist.json と同一。JSON そのまま)::

    {"input_bits": int,
     "binarization": str,                      # 任意。二値化の説明文字列
     "layers": [[{"op": int, "a": int, "b": int}, ...], ...],
     "output": {"classes": int, "group_size": int, "tau": float}}

- 層 l のゲートの `a`/`b` は層 l-1 の出力添字(層 0 は入力ビット添字)
- `op` は 16 種のゲート ID: ``op = 8·f(0,0) + 4·f(0,1) + 2·f(1,0) + f(1,1)``
  (`a` が第 1 引数)。IR の真理値表(bit index = ``(b<<1)|a``)とは
  ``ir/weights.py:gate_id_from_tt_n2`` で相互変換できる(bit0 と bit3 の
  入替なので**対合**。16 通りの一致は L1 テストで担保)
- 最終層のゲート i はクラス ``i // group_size`` のスコアに加算される
  (最終層幅 = classes × group_size)

縮約規則(入力が解決済みの状態で 1 ゲートを作る局所規則なので、トポロジカル順の
1 パスがそのまま不動点。到達性の刈り込みだけ後ろ向きの 1 パス):

  R1 pass-through 畳み込み — 出力が上流信号そのもの(tt 0xa/0xc = op 3/5)または
     その否定(tt 0x3/0x5 = op 10/12)、および畳み込みで 1 入力関数に退化した
     ゲートを配線に置換する。**否定は参照の極性ビットにして下流ゲートの
     真理値表へ吸収する(NOT は 0 ゲート)**
  R2 定数畳み込み — tt 0x0/0xf(op 0/15)、および入力が定数に解決したゲート。
     片側定数 → 1 入力関数 → R1 へ連鎖。両側定数 → 定数
  R3 同一入力畳み込み — 畳み込み後の実配線で a == b なら f(a,a) の 1 入力関数
  R4 CSE — 可換正規化(参照の順序で整列。非可換 tt は A/B 入替で吸収)した
     (tt, a, b) が同一のゲートを統合する。**dict はネット全体で 1 つ**(層を
     またいで統合する。conv 側の CSE がノード内で閉じるのとは異なる)
  R5 未参照除去 — 2 指標を分ける:
     - ``structurally_unreferenced``: **縮約前**の原ネットで出力から到達しない
       ゲート数(配線方式が決める量。P0 の主要な発見を測る指標)
     - ``unreachable_after_fold``: 縮約で作ったゲートのうち、畳み込みの結果
       出力から到達しなくなったもの

出力(クラス別 group-sum)の扱い:
    最終層のゲート i はクラス c = i // group_size のスコアへ加算される。
    - 定数に解決 → クラス和へのバイアス(ゲートではない。``output_bias_consts``)
    - pass-through に解決 → 上流信号を直接参照(ゲート数から除く)
    - 極性反転付きの参照 → 「バイアス +1、係数 −1」で厳密に表現
      (``output_inverted_refs``)
    したがって score[c] = bias[c] + Σ sign·value(ref) が原ネットの popcount と
    **整数で完全一致**する(``verify_equivalence`` で機械的に確認する)。

ゲート数の単位(報告時に必ず明示すること): 本モジュールは「任意の 2 入力関数
= 1 ゲート、入力否定は無料」で数える。ハードウェアでは出力段の反転に実体の
インバータが要る(P0 群 E の ABC 対比を参照)。

公開 API:
    flat_netlist_from_ir  — IR(lut_layer 列 + group_sum)→ 平坦ネットリスト
    reduce_netlist        — 縮約(ReduceResult を返す)
    ReduceResult          — 統計・縮約後 DAG・その評価器
    class_scores          — 原ネットのクラス別スコア(int64)
    verify_equivalence    — 縮約前後のクラス別スコアが整数完全一致か
    analyze               — 縮約 + 等価性検証をまとめた dict を返す
"""

import time
from dataclasses import dataclass, field

import numpy as np

from ..ir.serialize import tt_from_hex
from ..ir.weights import gate_id_from_tt_n2
from .optimize import (
    CONST_FOLD_A,
    CONST_FOLD_B,
    RED_BINARY,
    RED_CONST0,
    RED_CONST1,
    RED_ID,
    SAME_FOLD,
    TT_DEGEN,
    TT_DEGEN_ON_B,
    TT_NOT_A,
    TT_NOT_B,
    TT_SWAP,
    tt_eval,
)

__all__ = [
    'CONST_TTS',
    'PASSTHROUGH_TTS',
    'ReduceResult',
    'analyze',
    'class_scores',
    'flat_netlist_from_ir',
    'op_from_tt',
    'reduce_netlist',
    'tt_from_op',
    'verify_equivalence',
]


# ============================================================
# op(P0 のゲート ID) ⇔ tt(IR/optimize の真理値表整数)
# ============================================================

def op_from_tt(tt: int) -> int:
    """IR 真理値表(bit index = (b<<1)|a) → P0 の op(8·f00+4·f01+2·f10+f11)。"""
    return gate_id_from_tt_n2(int(tt) & 0xf)


def tt_from_op(op: int) -> int:
    """P0 の op → IR 真理値表。bit0↔bit3 の入替なので op_from_tt と同一写像。"""
    return gate_id_from_tt_n2(int(op) & 0xf)


# pass-through(a / b / ¬b / ¬a)= op 3/5/10/12 に対応する tt
PASSTHROUGH_TTS = (0xa, 0xc, 0x3, 0x5)
# 定数(FALSE / TRUE)= op 0/15 に対応する tt
CONST_TTS = (0x0, 0xf)

# LUT[tt, addr] -> bit。addr = (b<<1)|a(IR/optimize と同じビット規約)
_LUT = np.array([[tt_eval(tt, addr & 1, (addr >> 1) & 1) for addr in range(4)]
                 for tt in range(16)], dtype=np.uint8)


# ============================================================
# 参照(ref)表現
#   ('c', v, 0)  定数 v(0/1)。極性は値に正規化済み
#   ('i', k, p)  入力ビット k、p=1 なら否定
#   ('g', n, p)  縮約後ゲート n、p=1 なら否定
# ============================================================

_C0 = ('c', 0, 0)
_C1 = ('c', 1, 0)
_KIND_ORDER = {'i': 0, 'g': 1}


def _flip(ref):
    return (ref[0], ref[1], 1 - ref[2])


def _apply_unary(code: int, ref):
    """単項簡約コード(optimize.RED_*)を極性 0 の非定数 ref に適用する。

    Returns:
        (結果 ref, outcome) outcome ∈ {'const', 'wire'}
    """
    if code == RED_CONST0:
        return _C0, 'const'
    if code == RED_CONST1:
        return _C1, 'const'
    if code == RED_ID:
        return ref, 'wire'
    return _flip(ref), 'wire'   # RED_NOT


# ============================================================
# 縮約結果
# ============================================================

@dataclass
class ReduceResult:
    """1 ネットの縮約結果(統計 + 縮約後 DAG + 出力段の表現)。

    Attributes:
        stats: 内訳・実効ゲート数・構造統計(キー名は P0 `p0_results.json`
            の `nets` エントリと同一)
        gates: 縮約後ゲート列 (tt, ref_a, ref_b, origin_layer)。作成順 =
            トポロジカル順
        reach: 各ゲートが出力から到達可能か (n_gates,) bool
        bias: クラス別バイアス(定数項 + 反転参照の +1)
        terms: (class, (kind, idx), sign) のリスト。極性は sign に吸収済み
        final_refs: 最終層ゲートの解決済み参照(HDL 化などに使う)
    """

    stats: dict
    gates: list
    reach: np.ndarray
    bias: list
    terms: list
    input_bits: int
    classes: int
    group_size: int
    tau: float = 0.0
    final_refs: list = field(default_factory=list)
    _plan: list | None = field(default=None, repr=False, compare=False)
    _cls_refs: list | None = field(default=None, repr=False, compare=False)
    _cls_sgn: list | None = field(default=None, repr=False, compare=False)

    # ---------------- 縮約後 DAG の評価 ----------------
    def _ref_index(self, r) -> int:
        """統合配列 ALL の行番号: 入力は k、ゲートは input_bits + n。"""
        return r[1] if r[0] == 'i' else self.input_bits + r[1]

    def _build_plan(self) -> None:
        """origin_layer ごとの連続区間へ分けたベクトル化計画を作る。

        同一 origin_layer のゲートは互いを参照しない(参照元は必ず前層の
        解決済み ref か入力)ので、区間単位で一括評価できる。
        """
        plan = []
        cur_li = None
        buf_tt: list[int] = []
        buf_a: list[int] = []
        buf_b: list[int] = []
        buf_n: list[int] = []
        for n, (tt, ra, rb, li) in enumerate(self.gates):
            if cur_li is not None and li != cur_li:
                plan.append((np.array(buf_tt, dtype=np.int64),
                             np.array(buf_a, dtype=np.int64),
                             np.array(buf_b, dtype=np.int64),
                             np.array(buf_n, dtype=np.int64)))
                buf_tt, buf_a, buf_b, buf_n = [], [], [], []
            cur_li = li
            buf_tt.append(tt)
            buf_a.append(self._ref_index(ra))
            buf_b.append(self._ref_index(rb))
            buf_n.append(self.input_bits + n)
        if buf_tt:
            plan.append((np.array(buf_tt, dtype=np.int64),
                         np.array(buf_a, dtype=np.int64),
                         np.array(buf_b, dtype=np.int64),
                         np.array(buf_n, dtype=np.int64)))
        self._plan = plan

        cls_refs: list[list[int]] = [[] for _ in range(self.classes)]
        cls_sgn: list[list[int]] = [[] for _ in range(self.classes)]
        for (c, r, s) in self.terms:
            cls_refs[c].append(self._ref_index(r))
            cls_sgn[c].append(s)
        self._cls_refs = [np.array(v, dtype=np.int64) for v in cls_refs]
        self._cls_sgn = [np.array(v, dtype=np.int64) for v in cls_sgn]

    def class_scores(self, X: np.ndarray, batch: int = 1000) -> np.ndarray:
        """縮約後ネットのクラス別スコア (N, classes) int64。

        Args:
            X: (N, input_bits) の 0/1 uint8 配列
            batch: 一度に評価する行数(メモリと速度のトレードオフ)
        """
        X = np.asarray(X)
        assert X.ndim == 2 and X.shape[1] == self.input_bits, (
            f'X の形状 {X.shape} が input_bits={self.input_bits} と合わない')
        if self._plan is None:
            self._build_plan()
        n_new = len(self.gates)
        n = X.shape[0]
        scores = np.zeros((n, self.classes), dtype=np.int64)
        for i0 in range(0, n, batch):
            xb = X[i0:i0 + batch]
            nb = xb.shape[0]
            all_bits = np.zeros((self.input_bits + n_new, nb), dtype=np.uint8)
            all_bits[:self.input_bits] = xb.T
            for (tts, ia, ib, gn) in self._plan:
                a = all_bits[ia]
                b = all_bits[ib]
                addr = (b << 1) | a
                all_bits[gn] = _LUT[tts[:, None], addr]
            for c in range(self.classes):
                refs = self._cls_refs[c]
                if refs.size == 0:
                    s = np.zeros(nb, dtype=np.int64)
                else:
                    v = all_bits[refs].astype(np.int64)
                    s = (v * self._cls_sgn[c][:, None]).sum(axis=0)
                scores[i0:i0 + nb, c] = s + self.bias[c]
        return scores

    # ---------------- 便宜 ----------------
    @property
    def effective_gates(self) -> int:
        return int(self.stats['effective_gates'])

    @property
    def total_gates(self) -> int:
        return int(self.stats['total_gates'])

    @property
    def effective_ratio(self) -> float:
        return float(self.stats['effective_ratio'])


# ============================================================
# 縮約本体
# ============================================================

class _Reducer:
    """1 ネットの縮約器(P0 `Reducer` の移植)。"""

    def __init__(self, netlist: dict):
        self.netlist = netlist
        self.input_bits = int(netlist['input_bits'])
        self.layers = netlist['layers']
        self.classes = int(netlist['output']['classes'])
        self.group_size = int(netlist['output']['group_size'])
        self.tau = float(netlist['output'].get('tau', 0.0))
        assert len(self.layers) >= 1, 'layers が空'
        assert len(self.layers[-1]) == self.classes * self.group_size, (
            f'最終層幅 {len(self.layers[-1])} != '
            f'classes*group_size {self.classes * self.group_size}')

    # ---------------- 原ネットの構造統計 ----------------
    def structural_stats(self) -> dict:
        """縮約前の原ネットで、出力から後ろ向きに到達しないゲートを数える。

        - reachable_orig: 出力(最終層は group-sum が全ゲートを読む)から
          到達するゲート数
        - structurally_unreferenced: total - reachable_orig(推移的な未参照)
        - structural_fanout0: 次層のどのゲートからも a/b で参照されない
          ゲート数(直接 fan-out 0。最終層は group-sum が読むので常に参照済み)
        """
        n_layers = len(self.layers)
        used = [np.zeros(len(layer), dtype=bool) for layer in self.layers]
        used[-1][:] = True
        fanout = [np.zeros(len(layer), dtype=np.int64) for layer in self.layers]
        for li in range(n_layers - 1, 0, -1):
            a = np.array([g['a'] for g in self.layers[li]], dtype=np.int64)
            b = np.array([g['b'] for g in self.layers[li]], dtype=np.int64)
            np.add.at(fanout[li - 1], a, 1)
            np.add.at(fanout[li - 1], b, 1)
            sel = used[li]
            used[li - 1][a[sel]] = True
            used[li - 1][b[sel]] = True
        total = sum(len(layer) for layer in self.layers)
        reach = int(sum(int(u.sum()) for u in used))
        fanout0 = 0
        for li in range(n_layers - 1):
            fanout0 += int((fanout[li] == 0).sum())
        per_layer_unref = [int(len(self.layers[li]) - used[li].sum())
                           for li in range(n_layers)]
        per_layer_fanout0 = ([int((fanout[li] == 0).sum())
                              for li in range(n_layers - 1)] + [0])
        return {
            'total_gates': total,
            'reachable_orig': reach,
            'structurally_unreferenced': total - reach,
            'structural_fanout0': fanout0,
            'per_layer_structurally_unreferenced': per_layer_unref,
            'per_layer_structural_fanout0': per_layer_fanout0,
        }

    # ---------------- 縮約 ----------------
    def reduce(self) -> ReduceResult:
        n_layers = len(self.layers)
        # 縮約後ゲート: (tt, ra, rb, origin_layer)。ra/rb は極性 0 の ('i'|'g')
        gates: list = []
        cse: dict = {}
        prev = [('i', k, 0) for k in range(self.input_bits)]

        cnt = dict(
            passthrough_folded=0, passthrough_folded_native=0,
            passthrough_folded_induced=0,
            negation_absorbed=0,
            const_folded=0, const_folded_native=0, const_folded_induced=0,
            same_input_folded=0, const_input_gates=0,
            cse_merged=0, new_gates=0,
            native_passthrough_ops=0, native_const_ops=0,
        )
        per_layer = []

        for li in range(n_layers):
            layer = self.layers[li]
            lc = dict(n_gates=len(layer), passthrough_folded=0, const_folded=0,
                      same_input_folded=0, negation_absorbed=0, cse_merged=0,
                      new_gates=0, const_input_gates=0)
            cur = []
            for g in layer:
                tt = tt_from_op(int(g['op']))
                native_pt = tt in PASSTHROUGH_TTS
                native_const = tt in CONST_TTS
                if native_pt:
                    cnt['native_passthrough_ops'] += 1
                if native_const:
                    cnt['native_const_ops'] += 1
                ra = prev[g['a']]
                rb = prev[g['b']]

                # --- R1 後半: 上流 pass-through の否定を真理値表へ吸収 ---
                if ra[2]:
                    tt = TT_NOT_A[tt]
                    ra = (ra[0], ra[1], 0)
                    cnt['negation_absorbed'] += 1
                    lc['negation_absorbed'] += 1
                if rb[2]:
                    tt = TT_NOT_B[tt]
                    rb = (rb[0], rb[1], 0)
                    cnt['negation_absorbed'] += 1
                    lc['negation_absorbed'] += 1

                a_const = (ra[0] == 'c')
                b_const = (rb[0] == 'c')
                if a_const or b_const:
                    cnt['const_input_gates'] += 1
                    lc['const_input_gates'] += 1

                if a_const and b_const:
                    # --- R2 両側定数 ---
                    res = ('c', tt_eval(tt, ra[1], rb[1]), 0)
                    outcome = 'const'
                elif a_const:
                    # --- R2 片側定数 → 1 入力関数 → R1 ---
                    res, outcome = _apply_unary(CONST_FOLD_A[tt * 2 + ra[1]], rb)
                elif b_const:
                    res, outcome = _apply_unary(CONST_FOLD_B[tt * 2 + rb[1]], ra)
                elif ra == rb:
                    # --- R3 同一入力 ---
                    cnt['same_input_folded'] += 1
                    lc['same_input_folded'] += 1
                    res, outcome = _apply_unary(SAME_FOLD[tt], ra)
                else:
                    code = TT_DEGEN[tt]
                    if code != RED_BINARY:
                        # tt 自体が退化(定数 / 片方の入力に依存しない)
                        target = rb if TT_DEGEN_ON_B[tt] else ra
                        res, outcome = _apply_unary(code, target)
                    else:
                        # --- 真の 2 入力ゲート: R4 CSE(可換正規化つき) ---
                        ka = (_KIND_ORDER[ra[0]], ra[1])
                        kb = (_KIND_ORDER[rb[0]], rb[1])
                        if kb < ka:
                            tt = TT_SWAP[tt]
                            ra, rb = rb, ra
                        key = (tt, ra[0], ra[1], rb[0], rb[1])
                        hit = cse.get(key)
                        if hit is not None:
                            res = ('g', hit, 0)
                            outcome = 'cse'
                        else:
                            idx = len(gates)
                            gates.append((tt, ra, rb, li))
                            cse[key] = idx
                            res = ('g', idx, 0)
                            outcome = 'new'

                if outcome == 'const':
                    cnt['const_folded'] += 1
                    lc['const_folded'] += 1
                    if native_const:
                        cnt['const_folded_native'] += 1
                    else:
                        cnt['const_folded_induced'] += 1
                elif outcome == 'wire':
                    cnt['passthrough_folded'] += 1
                    lc['passthrough_folded'] += 1
                    if native_pt:
                        cnt['passthrough_folded_native'] += 1
                    else:
                        cnt['passthrough_folded_induced'] += 1
                elif outcome == 'cse':
                    cnt['cse_merged'] += 1
                    lc['cse_merged'] += 1
                else:
                    cnt['new_gates'] += 1
                    lc['new_gates'] += 1
                cur.append(res)
            per_layer.append(lc)
            prev = cur

        # ---------------- 出力(クラス別 group-sum)の構築 ----------------
        bias = [0] * self.classes
        terms = []   # (class, (kind, idx), sign)
        out_stats = dict(output_bias_consts=0, output_bias_ones=0,
                         output_wire_refs=0, output_inverted_refs=0,
                         output_input_refs=0)
        for i, ref in enumerate(prev):
            c = i // self.group_size
            if ref[0] == 'c':
                out_stats['output_bias_consts'] += 1
                if ref[1] == 1:
                    bias[c] += 1
                    out_stats['output_bias_ones'] += 1
            else:
                if ref[0] == 'i':
                    out_stats['output_input_refs'] += 1
                if ref[2] == 1:
                    bias[c] += 1
                    terms.append((c, (ref[0], ref[1]), -1))
                    out_stats['output_inverted_refs'] += 1
                else:
                    terms.append((c, (ref[0], ref[1]), +1))
        out_stats['output_wire_refs'] = len(terms)

        # ---------------- R5 到達性(縮約後) ----------------
        n_new = len(gates)
        reach = np.zeros(n_new, dtype=bool)
        for (_c, r, _s) in terms:
            if r[0] == 'g':
                reach[r[1]] = True
        # gates は作成順 = トポロジカル順なので降順 1 パスで推移閉包が取れる
        for n in range(n_new - 1, -1, -1):
            if not reach[n]:
                continue
            _tt, ra, rb, _li = gates[n]
            if ra[0] == 'g':
                reach[ra[1]] = True
            if rb[0] == 'g':
                reach[rb[1]] = True
        effective = int(reach.sum())
        unreachable_after_fold = n_new - effective

        eff_per_layer = [0] * n_layers
        for n in range(n_new):
            if reach[n]:
                eff_per_layer[gates[n][3]] += 1
        for li in range(n_layers):
            per_layer[li]['effective_gates'] = eff_per_layer[li]
            per_layer[li]['unreachable_after_fold'] = (
                per_layer[li]['new_gates'] - eff_per_layer[li])

        total = sum(len(layer) for layer in self.layers)
        assert (cnt['const_folded'] + cnt['passthrough_folded']
                + cnt['cse_merged'] + cnt['new_gates']) == total, (
            '内訳の合計が総ゲート数に一致しない')

        stats = dict(cnt)
        stats.update(out_stats)
        stats['total_gates'] = total
        stats['effective_gates'] = effective
        stats['unreachable_after_fold'] = unreachable_after_fold
        stats['effective_ratio'] = effective / total
        stats['reduction_ratio'] = 1.0 - effective / total
        stats['per_layer'] = per_layer
        stats['class_bias'] = bias
        stats['widths'] = [len(layer) for layer in self.layers]
        stats['input_bits'] = self.input_bits
        stats['classes'] = self.classes
        stats['group_size'] = self.group_size
        stats.update(self.structural_stats())

        return ReduceResult(
            stats=stats, gates=gates, reach=reach, bias=bias, terms=terms,
            input_bits=self.input_bits, classes=self.classes,
            group_size=self.group_size, tau=self.tau, final_refs=prev,
        )


def reduce_netlist(netlist: dict) -> ReduceResult:
    """平坦ネットリストを縮約し、内訳統計と縮約後 DAG を返す。

    Args:
        netlist: モジュール docstring の平坦形式(P0 netlist.json と同一)

    Returns:
        ReduceResult。``stats`` のキー名・意味は P0 `p0_results.json` の
        `nets` エントリと同一
    """
    return _Reducer(netlist).reduce()


# ============================================================
# 原ネットの評価と等価性検証
# ============================================================

def class_scores(netlist: dict, X: np.ndarray, batch: int = 2000) -> np.ndarray:
    """原ネットのクラス別 group-sum (N, classes) int64。

    Args:
        netlist: 平坦ネットリスト
        X: (N, input_bits) の 0/1 uint8 配列
        batch: 一度に評価する行数
    """
    layers = netlist['layers']
    classes = int(netlist['output']['classes'])
    gs = int(netlist['output']['group_size'])
    X = np.asarray(X)
    assert X.ndim == 2 and X.shape[1] == int(netlist['input_bits']), (
        f'X の形状 {X.shape} が input_bits={netlist["input_bits"]} と合わない')
    prepped = []
    for layer in layers:
        prepped.append((
            np.array([tt_from_op(int(g['op'])) for g in layer], dtype=np.int64),
            np.array([g['a'] for g in layer], dtype=np.int64),
            np.array([g['b'] for g in layer], dtype=np.int64)))
    n = X.shape[0]
    out = np.zeros((n, classes), dtype=np.int64)
    for i0 in range(0, n, batch):
        x = X[i0:i0 + batch]
        for (tts, ia, ib) in prepped:
            a = x[:, ia]
            b = x[:, ib]
            addr = (b << 1) | a
            x = _LUT[tts[None, :], addr]
        out[i0:i0 + x.shape[0]] = x.reshape(x.shape[0], classes, gs).sum(axis=2)
    return out


def verify_equivalence(netlist: dict, result: ReduceResult,
                       X: np.ndarray) -> bool:
    """縮約前後のクラス別スコアが整数で完全一致するか。

    バイアス・係数 −1 込みで比較するので、pass-through / 定数 / 反転参照の
    扱いを含めた厳密な等価性検証になる。
    """
    return bool(np.array_equal(class_scores(netlist, X),
                               result.class_scores(X)))


def analyze(netlist: dict, X_data: np.ndarray | None = None,
            n_random: int = 2000, seed: int = 0) -> dict:
    """1 ネットを縮約し、内訳統計と等価性検証の結果を dict で返す。

    Args:
        netlist: 平坦ネットリスト
        X_data: 実データのビット列 (N, input_bits) uint8。None なら
            ランダム入力のみで検証する
        n_random: 一様ランダム入力の本数(0 なら省略)
        seed: ランダム入力の seed

    Returns:
        ``reduce_netlist`` の stats に ``reduce_time_s`` と ``equivalence``
        を足した dict。``equivalence`` は
        ``{'data','data_n','data_mismatch_rows','random','random_n',
        'random_mismatch_rows','overall'}``
        (P0 は MNIST 固定だったので 'mnist' というキー名だった)
    """
    t0 = time.time()
    result = reduce_netlist(netlist)
    res = dict(result.stats)
    res['reduce_time_s'] = time.time() - t0

    ib = int(netlist['input_bits'])
    equivalence: dict = {}

    if X_data is not None and np.asarray(X_data).shape[1] == ib \
            and np.asarray(X_data).shape[0] > 0:
        xd = np.asarray(X_data)
        so = class_scores(netlist, xd)
        sr = result.class_scores(xd)
        equivalence['data'] = 'pass' if np.array_equal(so, sr) else 'fail'
        equivalence['data_n'] = int(xd.shape[0])
        equivalence['data_mismatch_rows'] = int((so != sr).any(axis=1).sum())
    else:
        equivalence['data'] = 'skip'
        equivalence['data_n'] = 0
        equivalence['data_mismatch_rows'] = 0

    if n_random > 0:
        rng = np.random.default_rng(seed)
        xr = rng.integers(0, 2, size=(n_random, ib), dtype=np.uint8)
        so = class_scores(netlist, xr)
        sr = result.class_scores(xr)
        equivalence['random'] = 'pass' if np.array_equal(so, sr) else 'fail'
        equivalence['random_n'] = int(n_random)
        equivalence['random_mismatch_rows'] = int((so != sr).any(axis=1).sum())
    else:
        equivalence['random'] = 'skip'
        equivalence['random_n'] = 0
        equivalence['random_mismatch_rows'] = 0

    equivalence['overall'] = (
        'pass' if all(equivalence[k] in ('pass', 'skip')
                      for k in ('data', 'random')) else 'fail')
    res['equivalence'] = equivalence
    return res


# ============================================================
# IR → 平坦ネットリスト
# ============================================================

_SKIP_NODE_TYPES = ('flatten',)


def flat_netlist_from_ir(ir, *, tau: float | None = None,
                         binarization: str = '') -> dict:
    """IR(直列 lut_layer 列 + group_sum 終端)→ P0 形式の平坦ネットリスト。

    対応範囲は **n_inputs=2 の lut_layer が直列に並び group_sum で終わる**
    グラフに限る(P0 の縮約器の前提と同じ)。それ以外は NotImplementedError。

    Args:
        ir: `export.export_ir` が返す IRGraph
        tau: 出力段の tau。None なら group_sum ノードの params から取る
            (P0 形式の JSON に載せるだけで縮約には使わない)
        binarization: 二値化の説明文字列(任意のメタ情報)

    Returns:
        {"input_bits", "binarization", "layers", "output"} の dict
    """
    if len(ir.inputs) != 1:
        raise NotImplementedError(
            f'入力ポートが 1 本でない(={len(ir.inputs)})。'
            '直列 lut_layer + group_sum のグラフのみ対応')
    input_width = int(ir.inputs[0].width)

    # 入力エッジの本数を数え、直列であることを確認する
    in_srcs: dict[str, list[str]] = {}
    for e in ir.edges:
        in_srcs.setdefault(e.dst, []).append(e.src)

    order = ir.topological_order()
    if not order:
        raise NotImplementedError('ノードが 1 つも無い')

    expected_src = ir.inputs[0].id
    layers = []
    width = input_width
    group_sum = None
    for node in order:
        srcs = in_srcs.get(node.id, [])
        if len(srcs) != 1 or srcs[0] != expected_src:
            raise NotImplementedError(
                f'{node.id}: 入力元が {srcs}(期待 [{expected_src}])。'
                '直列グラフのみ対応')
        if group_sum is not None:
            raise NotImplementedError(
                f'{node.id}: group_sum の後ろにノードがある。'
                'group_sum 終端のグラフのみ対応')
        if node.type in _SKIP_NODE_TYPES:
            pass          # flatten は幅不変の恒等ノード
        elif node.type == 'lut_layer':
            p = node.params
            n_inputs = int(p['n_inputs'])
            if n_inputs != 2:
                raise NotImplementedError(
                    f'{node.id}: n_inputs={n_inputs}。2 のみ対応'
                    '(n>2 の分解は未実装)')
            in_width = int(p['in_width'])
            if in_width != width:
                raise NotImplementedError(
                    f'{node.id}: in_width={in_width} が前段幅 {width} と不一致')
            indices = np.asarray(p['indices'], dtype=np.int64)
            tts = tt_from_hex(p['truth_tables'])
            out_width = int(p['out_width'])
            if indices.shape != (2, out_width) or len(tts) != out_width:
                raise NotImplementedError(
                    f'{node.id}: indices{indices.shape} / '
                    f'truth_tables{len(tts)} が out_width={out_width} と不整合')
            a_idx = indices[0]
            b_idx = indices[1]
            layers.append([
                {'op': op_from_tt(int(tts[o])),
                 'a': int(a_idx[o]), 'b': int(b_idx[o])}
                for o in range(out_width)])
            width = out_width
        elif node.type == 'group_sum':
            group_sum = node
        else:
            raise NotImplementedError(
                f'{node.id}: ノード型 {node.type} は未対応'
                '(lut_layer / group_sum / flatten のみ)')
        expected_src = node.id

    if group_sum is None:
        raise NotImplementedError('group_sum 終端でない。終端は group_sum のみ対応')
    if not layers:
        raise NotImplementedError('lut_layer ノードが 1 つも無い')

    classes = int(group_sum.params['k'])
    gs_in = int(group_sum.params.get('in_width', width))
    if gs_in != width:
        raise NotImplementedError(
            f'group_sum の in_width={gs_in} が前段幅 {width} と不一致')
    if width % classes != 0:
        raise NotImplementedError(
            f'最終層幅 {width} が classes={classes} で割り切れない')
    group_size = width // classes
    out_tau = float(group_sum.params.get('tau', 0.0)) if tau is None else float(tau)

    return {
        'input_bits': input_width,
        'binarization': binarization,
        'layers': layers,
        'output': {'classes': classes, 'group_size': group_size,
                   'tau': out_tau},
    }
