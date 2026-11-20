"""conv2d(ocごとの命令列分割)のモデル特化コード生成。

`mdl_k_conv_exec_split`(CUDA/Metal共通のインタプリタ、
`runtime/gpu/mdl_gpu_kernels.h`)は命令列 `[tt,a,b,slot]` を実行時に
バッファから読む。SASS解析(進捗記録節
§5、`results/deploy/sass/analysis_memo.md`)で、この構造には2つの
「実行時に確定するが最適化を妨げる」ムダがあると分かった:

    1. tt が実行時値のため、4bitマスク抽出(LOP3×4)が毎ゲート必要
       → ttをコンパイル時定数にすれば `deploy/cbackend.py::GATE_EXPR`
         (CPU直線化テーブル)と同じ最小項展開に畳める(最大3命令/ゲート)
    2. `slots[MDL_CONV_SPLIT_MAX_SLOTS]` が動的添字のためレジスタ昇格
       されず local memory(LDL/STL)落ちする
       → ocごとに個別のスカラーローカル変数(`s0, s1, ...`)へ展開すれば
         添字が消える

このモジュールはCUDA/Metal共通のロジック(GATE_EXPR適用・ref解決・
switch(oc)ディスパッチ本体)を1本にまとめる。ホスト側の呼び出し規約
(バッファ確保・パイプライン管理)は言語ごとに違うので、それは
`cudabackend.py`/`metalbackend.py` 側に残す。

最小実験(Metal、seg_convdlgn_k32の実ノード3本、5セット中央値)で
interp比2.1〜2.6倍(batch大)・1.39〜1.44倍(batch=1、Metalの床支配領域)、
出力bit厳密一致を確認済み(1.3倍ゲート通過)。
"""

from __future__ import annotations

from .cbackend import GATE_EXPR
from .layout import ConvLayoutSplit

__all__ = ['GATE_EXPR', 'ref_expr', 'oc_function_body', 'build_switch_body',
          'CODEGEN_KERNEL_PREFIX']

#: 生成カーネル名の接頭辞。Metalは `mdl_k_` で始まる関数を自動でパイプライン
#: 登録する(`runtime/metal/mdl_metal.mm::mdl_metal_init`)ので、これに合わせる
#: (CUDA側は接頭辞に意味は無いが、SASS解析でノードを識別しやすいよう揃える)。
CODEGEN_KERNEL_PREFIX = 'mdl_k_conv_codegen_'


def ref_expr(packed: int, trw: int) -> str:
    """パック済みref((value<<1)|is_reg) → C/MSL共通の式文字列。

    bit0=1: このocのローカルレジスタ変数 `sK`。
    bit0=0: 共有スクラッチの絶対ワードオフセット。ただし
      `mdl_k_conv_const`(CUDA/Metal共通)が保証する不変条件
      ([0,trw)=CONST-0平面, [trw,2*trw)=CONST-1平面)を使って
      即値へ畳み込む(`cbackend.py::_emit_gate_program_fused` の
      `ref()` と同じ最適化)。それ以外(タップ平面プール)は `sc[off+i]`。
    """
    if packed & 1:
        return f's{packed >> 1}'
    off = packed >> 1
    if off == 0:
        return '0u'
    if off == trw:
        return '0xffffffffu'
    return f'sc[{off} + i]'


def oc_function_body(layout: ConvLayoutSplit, oc: int) -> tuple[str, str]:
    """1ocぶんの直線コード本体(ローカル変数宣言+代入列)と最終式を返す。

    戻り値はCUDA/Metalどちらでも使える生の文(アドレス空間修飾子を含まない)。
    呼び出し側が `device const uint *sc` / `const unsigned int *sc` 等の
    シグネチャで包む。
    """
    trw = layout.tap_rows_words
    lo = layout.oc_instr_start[oc] * 4
    hi = layout.oc_instr_start[oc + 1] * 4
    instrs4 = layout.instr_flat[lo:hi]
    final_ref = layout.oc_final_ref[oc]

    lines: list[str] = []
    slots_used: set[int] = set()
    n_local = len(instrs4) // 4
    for k in range(n_local):
        tt, ap, bp, dst = instrs4[4 * k:4 * k + 4]
        a = ref_expr(ap, trw)
        b = ref_expr(bp, trw)
        expr = GATE_EXPR[tt & 0xf].format(a=a, b=b)
        lines.append(f'  s{dst} = {expr};')
        slots_used.add(dst)
    final_expr = ref_expr(final_ref, trw)
    # unsigned int(uint32_tではない): CUDA側は<cstdint>を生成物へ足していない
    # ので、環境によってuint32_tが未定義になりうる(実測: リモートnvccで
    # "identifier uint32_t is undefined")。runtime/gpu/mdl_gpu_kernels.hも
    # 一貫して unsigned int を使っているので、そちらに合わせる。
    decl = ('  unsigned int ' + ', '.join(f's{k}' for k in sorted(slots_used))
            + ';\n') if slots_used else ''
    body = decl + ('\n'.join(lines) + '\n' if lines else '')
    return body, final_expr


def build_switch_body(layout: ConvLayoutSplit, oc_fn_name,
                      buf_arg: str = 'scratch') -> str:
    """`switch (oc) { case K: v = <oc_fn_name(K)>(<buf_arg>, i); break; ... }`。

    同一SIMDグループ/warp内は `i` が最内(≡tap_rows_words本ぶん連続)
    なので oc は事実上warp一様 — 分岐はコンパイラにとって「一様分岐
    (uniform branch)」に近く、最小実験で分岐コストの悪化は
    観測されなかった(oc数32〜128で確認)。

    Args:
        oc_fn_name: `int -> str`(oc番号 → 生成関数名)
        buf_arg: 呼び出し元カーネルの読み取り専用スクラッチ引数名
            (Metal/CUDAどちらも先頭引数をこの名前にすれば別名変数は不要)
    """
    cases = '\n'.join(
        f'    case {oc}: v = {oc_fn_name(oc)}({buf_arg}, i); break;'
        for oc in range(layout.out_channels))
    return f'  switch (oc) {{\n{cases}\n    default: v = 0u; break;\n  }}'
