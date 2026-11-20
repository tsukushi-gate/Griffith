"""実行プラン → 自己完結したCソース(プラン層へ分離)。

**このファイルには op 種別ごとのコード生成しか無い**。グラフ走査・形推論・検証は
`deploy/plan.py` にあり、CUDA/Metal と共有している(以前は3バックエンドが同じ
走査を持っていた)。新しいノード型の大半は plan 側の登録だけで済み、ここは
本当に新しい演算が来たときだけ1行増える。

生成物:
    <name>.h  公開API(mdl_infer / サイズ定数 / メタデータ)
    <name>.c  static const のテーブル群 + トポロジ順の直線ドライバ

モデル固有の判断はすべて生成時に済ませる。実行時のC側にはグラフ探索も
種別分岐もメモリ割り付けも残らない。
"""

import re

from .memplan import check_no_overlap, plan_memory
from .plan import (
    KIND_BITPLANE,
    KIND_FLAT,
    KIND_SCORES,
    KIND_SCRATCH,
    ExecPlan,
    UnsupportedNode,
    build_plan,
    lut_dense_indices_fit_u16,
    lut_flat_indices_fit_u16,
    slice_truth_tables,
)


def _sanitize_c_ident(node_id: str) -> str:
    """ノードidをC識別子として使える文字列へ変換する(ir/build.py::_sanitize_idと同趣旨)。"""
    return re.sub(r'\W', '_', node_id)

__all__ = ['UnsupportedNode', 'generate_c', 'generate_rddlgn_c', 'CBackend',
           'KIND_BITPLANE', 'KIND_FLAT', 'KIND_SCORES',
           'RDDLGN_RECURRENT_NODES', 'RDDLGN_BUFFER_NODES']

#: 真理値表(下位4bit)→ C の式。**生成時に確定するので実行時の分岐が消える**。
#: `mdl_gate2_range` の switch と同一(runtime/c/src/mdl_kernels.c)。
GATE_EXPR = {
    0x0: '0u',
    0x1: '~({a} | {b})',
    0x2: '{a} & ~{b}',
    0x3: '~{b}',
    0x4: '~{a} & {b}',
    0x5: '~{a}',
    0x6: '{a} ^ {b}',
    0x7: '~({a} & {b})',
    0x8: '{a} & {b}',
    0x9: '~({a} ^ {b})',
    0xa: '{a}',
    0xb: '{a} | ~{b}',
    0xc: '{b}',
    0xd: '~{a} | {b}',
    0xe: '{a} | {b}',
    0xf: '0xffffffffu',
}

class CBackend:
    """プランからCソースを組み立てる。"""

    name = 'c'

    #: plane_op / reduce_scores の下位種別(ORTの capability に倣った申告)。
    #: op種別のほうは `EMITTERS` から導くので、別に書かない。
    SUB_OPS = frozenset({
        'or_pool2x2', 'upsample2d_x2', 'permute_channels', 'concat_planes',
        'spatial_group_sum', 'group_sum',
    })

    def __init__(self, plan: ExecPlan, model_name: str = 'model', *,
                 fuse: bool = False, dense_batch: int = 0,
                 expose_recurrent: frozenset[str] = frozenset(),
                 expose_buffers: frozenset[str] = frozenset(),
                 packed_teacher_forcing: str | None = None,
                 drop_folded_recurrents: frozenset[str] = frozenset(),
                 dense_flat_decision: dict | None = None):
        self.plan = plan
        self.model_name = model_name
        #:: `resolve_dense_flat` の判定結果。与えられたときだけ model.h の
        #: 先頭コメントに採否と削減率を書き、`MDL_DENSE_FLAT 1/0` を定義する。
        #: None(既定)なら出力は1バイトも変わらない。
        self.dense_flat_decision = dense_flat_decision
        #:: 語彙折り畳みが完全に効いている(N/L両方foldした)ときだけ
        #: 呼び出し側が渡すノードid集合(通常 {'enc_n_rec', 'dec_l_rec'})。
        #: このidのrecurrentは重み(indices/truth_tables)・step関数・
        #: mdl_infer/mdl_run本体からの呼び出し(host_loop_block)を
        #: **一切emitしない**(配信サイズの実削減が目的)。出力バッファの
        #: 取得関数(mdl_out_<id>、expose_buffers経由)だけは残す —
        #: 折り畳みハーネス(mdl_rddlgn_generate_greedy_folded)がそこへ
        #: 表引き結果を直接書き込むため。この結果、`mdl_infer`/`mdl_run`
        #: (教師強制)と`mdl_rddlgn_generate_greedy`(bits版、非折り畳み)は
        #: この生成物では動かなくなる(前者はN/L出力が埋まらず壊れた値を
        #: 返す、後者はシンボル自体が無くなりリンクできない — ヘッダ側で
        #: `MDL_RDDLGN_WEIGHTS_DROPPED`マクロを見て宣言ごと落とすため)。
        #: 既定は空集合で、既存の全テストは無変更(生成コードが1バイトも
        #: 変わらない)。
        self.drop_folded_recurrents = frozenset(drop_folded_recurrents)
        #: 命令列を直線コードへ展開する。既定は False で、
        #: インタプリタ経路(`mdl_conv_exec`)は常に残してある。
        self.fuse = fuse
        self.fused_fns: list[str] = []
        #: recurrentノードがbodyから切り出す静的ステップ関数
        #: (`mdl_step_<id>`)。fused_fnsと同じく mdl_run より前に出す。
        self.fns: list[str] = []
        #: scope(recurrentノードid)ごとのop列。build()の先頭で1回だけ
        #: 分類し、emit_recurrent_execが自分のscopeぶんを取り出す。
        self._scoped: dict[str, list] = {}
        #: この集合に入っているrecurrentノードidは、step関数を非static
        #: 公開し、実行ラッパ(mdl_run_recurrent_<id>)・state取得/初期化
        #: 関数(mdl_state_<id>_<sv>/mdl_reset_state_<id>)・外部入力取得
        #: 関数(mdl_ext_<id>_<sv>)を追加で生成する(RDDLGN自己回帰
        #: 生成ハーネスruntime/c/src/mdl_rddlgn.c向け)。既定は空集合で、
        #: 既存の全テストは無変更(生成コードが1バイトも変わらない)。
        self.expose_recurrent = frozenset(expose_recurrent)
        #: この集合に入っているノード(recurrentに限らない)の出力バッファへ
        #: 取得関数(mdl_out_<id>)を追加で生成する。
        self.expose_buffers = frozenset(expose_buffers)
        #: 追加した公開API(ヘッダ宣言/マクロ・定義)。空なら
        #: 出力に何も追加されない(既存生成物への影響ゼロ)。
        self.public_decls: list[str] = []
        self.public_defs: list[str] = []
        #: state_initオペレータの出力バッファid → 初期化ビット列。
        #: build()の先頭で1回だけ集め、_emit_recurrent_public_apiが
        #: mdl_reset_state_<id>を組み立てるのに使う。
        self._state_bits_by_buf: dict[int, list[int]] = {}
        #: dense のバッチ軸パック経路を生やす枚数(0 で生やさない)
        self.dense_batch = dense_batch
        self.flat_tables: dict = {}
        self.dense_work_words = 0
        #: 教師強制の時間バッチ化。この node_id の recurrent が
        #: 「state 1個 + output_mode='stacked' + KIND_SCORES出力、
        #: tailがflat_concat→lut_dense*→group_sumで閉じている」形なら、
        #: state更新(head、順に steps 回)と M+GroupSum 相当(tail、
        #: steps ぶんまとめて1回のpacked評価)へ分割した追加entrypoint
        #: `mdl_infer_packed_tf_ctx` を生やす。既定は None(何も生成しない、
        #: 生成物は完全に既存互換)。パターン不一致なら ValueError で
        #: 明示的に落ちる(黙って無視しない — 呼び出し側が明示的に
        #: opt-inしている機能のため)。
        self.packed_tf_target = packed_teacher_forcing
        self.packed_tf_entry = ''
        self.packed_tf_work_words = 0
        #:: 未対応の組み合わせを明示的に拒否する。
        #: `mdl_infer_packed_tf_ctx`はrecurrentノード直前までの
        #: `self.body`(prefix_lines、他recurrentのhost_loop_block込み)を
        #: 再利用して教師強制のprefixを組み立てる。`drop_folded_recurrents`
        #:はそのnode_idのhost_loop_block自体を生成しない(重み無しで
        #: 出力バッファが埋まらない)ため、packed_teacher_forcing側の
        #: prefixがそのノードの出力を欠いたまま`mdl_infer_packed_tf_ctx`に
        #: 焼き込まれ、黙って壊れた値を返す事故になる。どちらも
        #: 呼び出し側が明示的にopt-inする機能なので、黙ってどちらかを
        #: 無視せずValueErrorで落とす(節「両方foldしたときだけ有効」の
        #: 安全側の既定と同じ方針)。
        if self.packed_tf_target is not None and self.drop_folded_recurrents:
            raise ValueError(
                'packed_teacher_forcing と drop_folded_recurrents の同時指定は'
                '未対応: drop_folded_recurrentsは対象recurrentの出力バッファを'
                '教師強制経路で埋めなくなるため、packed_teacher_forcingが'
                '再利用するprefixが壊れる。折り畳み後の生成物では'
                'packed_teacher_forcingを指定しないこと')
        #: lut_dense の出力値id → 定数表の接頭辞(パック経路が再利用する)
        self.lut_tables: dict[int, str] = {}
        #: lut_dense の出力値id → 真理値表(パック経路だけが元の並びを使う)
        self.lut_raw_tt: dict[int, list] = {}
        #: lut_dense の出力値id → 添字がuint16_tか。`{tag}_idx` は
        #: 単枚経路(emit_lut_dense)とパック経路(_emit_dense_packed /
        #: _emit_packed_teacher_forcing)が**同じ配列を共有する**ので、
        #: 型の選択もop単位でここに1回だけ記録し、両経路が読む
        #: (別々に判定すると型が食い違って未定義動作になりうる)。
        self.lut_idx_u16: dict[int, bool] = {}
        self.data_lines: list[str] = []   # static const テーブル
        self.buf_lines: list[str] = []    # static 作業バッファ
        self.body: list[str] = []         # 推論本体
        self.prepare: list[str] = []      # 1回だけ実行する初期化
        self._n = 0
        self.scratch_words = 0            # conv スクラッチの合計(報告用)
        # 入力は呼び出し側のポインタ、出力は呼び出し側の配列なので、
        # どちらもこちらのアリーナには載せない。
        self.memory = plan_memory(
            plan, exclude={plan.input_value, plan.output_value})
        check_no_overlap(plan, self.memory)

    # -- 小道具 -------------------------------------------------------
    def _uid(self, prefix: str) -> str:
        self._n += 1
        return f'{prefix}{self._n}'

    def _const_i32(self, name: str, values) -> str:
        items = ','.join(str(int(v)) for v in values) or '0'
        self.data_lines.append(f'static const int32_t {name}[] = {{{items}}};')
        return name

    def _const_u32(self, name: str, values) -> str:
        items = ','.join(f'{int(v) & 0xffffffff}u' for v in values) or '0u'
        self.data_lines.append(f'static const uint32_t {name}[] = {{{items}}};')
        return name

    def _const_u8(self, name: str, values) -> str:
        items = ','.join(str(int(v) & 0xff) for v in values) or '0'
        self.data_lines.append(f'static const uint8_t {name}[] = {{{items}}};')
        return name

    def _const_u16(self, name: str, values) -> str:
        items = ','.join(str(int(v) & 0xffff) for v in values) or '0'
        self.data_lines.append(f'static const uint16_t {name}[] = {{{items}}};')
        return name

    def buf(self, vid: int) -> str:
        """値id → バッファ名。"""
        if vid == self.plan.input_value:
            return 'mdl_input'
        if vid == self.plan.output_value:
            # 最終スコアは呼び出し側の配列へ直接書く(コピーもバッファも不要)
            return 'scores'
        return f'mdl_v{vid}'

    def _declare_buffers(self) -> None:
        """作業領域を1本のアリーナにまとめ、値をその中のオフセットへ写す。

        生存区間が重ならない値は同じ領域を共有する。名前をマクロに
        するのは、concat のソース配列が `static const` でバッファアドレスを
        並べる都合(アドレス定数式でなければならない)。
        """
        mem = self.memory
        self.buf_lines.append(
            f'/* 作業領域はコンテキストが持つ({mem.naive_words * 4:,}B ぶんの'
            f'値を生存区間解析で {mem.arena_words * 4:,}B に畳んである)。 */')
        for v in self.plan.values:
            if v.alias_of is not None:
                # 退化ゲート除去のアキュムレータ追記先。独自領域を
                # 持たない(memplanのpooledから除外済み)ので、親バッファの
                # マクロ(必ず先に列挙される保証は無いが、#defineはC側で
                # 遅延評価されるので順不同でよい)へのポインタ演算として
                # 定義する。alias_ofはKIND_FLAT(uint8_t*)前提。
                assert v.kind == KIND_FLAT, (
                    f'v{v.id}: alias_ofはKIND_FLATのみ対応(kind={v.kind})')
                parent_expr = self.buf(v.alias_of)
                expr = f'((uint8_t *)({parent_expr}) + {v.alias_offset})'
                self.buf_lines.append(f'#define {self.buf(v.id)} {expr}')
                continue
            if v.id not in mem.offsets:
                continue
            off = mem.offsets[v.id]
            base = f'(ctx->work + {off})'
            if v.kind in (KIND_BITPLANE, KIND_SCRATCH):
                expr = base                       # どちらも uint32_t のワード列
            elif v.kind == KIND_FLAT:
                expr = f'((uint8_t *){base})'
            else:
                expr = f'((int32_t *){base})'
            self.buf_lines.append(f'#define {self.buf(v.id)} {expr}')

    # -- op ごとのコード生成 -------------------------------------------
    def emit_pack(self, op) -> None:
        a = op.attrs
        self.body.append(
            f'  mdl_pack_planes({self.buf(op.inputs[0])}, '
            f'{self.buf(op.output)}, {a["c"]}, {a["h"]}, {a["w"]});')

    def emit_unpack(self, op) -> None:
        a = op.attrs
        self.body.append(
            f'  mdl_unpack_planes({self.buf(op.inputs[0])}, '
            f'{self.buf(op.output)}, {a["c"]}, {a["h"]}, {a["w"]});')

    #: 融合するタップpoolの上限(バイト)。**融合が効くかはpoolがL1に
    #: 収まるかで決まる**、というのが実測から出た判定条件。収まらないconvは
    #: インタプリタ経路(2〜3ストリームの素直な流し込み)のほうが速い。
    FUSE_POOL_BYTES = 96 * 1024

    def should_fuse(self, layout) -> bool:
        """この conv を融合するか。"""
        if self.fuse in (False, None):
            return False
        if self.fuse == 'always':
            return True
        pool_bytes = (layout.reg_base - layout.pool_base) * 4
        return pool_bytes <= self.FUSE_POOL_BYTES

    def emit_gate_program(self, op) -> None:
        if self.should_fuse(op.attrs['layout']):
            self._emit_gate_program_fused(op)
        else:
            self._emit_gate_program_interp(op)

    def _emit_gate_program_fused(self, op) -> None:
        """命令列を直線コードへ展開する(レジスタタイリング)。

        インタプリタ経路は「命令が外・ワードが内」で、命令ごとに
        ループを立ち上げ、中間値を毎回スクラッチへ書き戻していた。
        ここでは逆に**ワードが外・命令が内**にして、中間値(スロットは
        実測で2〜6本)をローカル変数に保つ。効くのは3点:

        1. **真理値表が生成時に確定する** — 実行時の16分岐が式に化ける
        2. **中間値のロード/ストアが消える**(スロットがレジスタに載る)
        3. 命令ごとのループ立ち上げが消える(trwは16〜512語と短い)

        外側ループは i について完全に要素独立なので、コンパイラは
        そのままベクタ化できる(スロットはベクタレジスタになる)。
        """
        layout = op.attrs['layout']
        tag = self._uid('conv')
        scratch = self.buf(op.attrs['scratch'][0])
        self._conv_tables(tag, layout)
        trw = layout.tap_rows_words
        reg_base = layout.reg_base
        plane_words = layout.h_out * layout.wpr_out
        chunks = self._chunk_instrs(layout)

        def is_slot(off: int) -> bool:
            return off >= reg_base

        def slot_of(off: int) -> int:
            return (off - reg_base) // trw

        def ref(off: int, live: set) -> str:
            """スクラッチのワードオフセット → 式。スロットならレジスタ変数。"""
            if is_slot(off):
                return f'r{slot_of(off)}'
            if off == 0:
                return '0u'                       # 定数0平面
            if off == trw:
                return '0xffffffffu'              # 定数1平面
            return f'sc[{off} + i]'

        # 各命令が読むスロット / 書くスロットを先に洗い出す
        reads: list[set] = []
        writes: list[int] = []
        for j in range(layout.n_instrs):
            _, a_off, b_off, dst_off = layout.instr_flat[4 * j:4 * j + 4]
            reads.append({slot_of(o) for o in (a_off, b_off) if is_slot(o)})
            writes.append(slot_of(dst_off))
            for k in range(layout.write_start[j + 1],
                           layout.write_start[j + 2]):
                src = layout.oc_src_off[layout.write_oc[k]]
                if is_slot(src):
                    reads[j].add(slot_of(src))

        n_slots = (max(writes) + 1) if writes else 0
        blocks: list[str] = []
        # **チャンクが外、ワードが内**。この順序でないと同時に触るタップ列が
        # 減らない(内側に入れて測ったら効果ゼロだった)。
        for ci, (lo, hi) in enumerate(chunks):
            written_here: set = set()
            need_load: set = set()
            for j in range(lo, hi):
                need_load |= (reads[j] - written_here)
                written_here.add(writes[j])
            later = set()
            for j in range(hi, layout.n_instrs):
                later |= reads[j]
            for k in range(layout.write_start[hi + 1] if hi < layout.n_instrs
                           else layout.write_start[layout.n_instrs + 1],
                           layout.write_start[layout.n_instrs + 1]):
                src = layout.oc_src_off[layout.write_oc[k]]
                if is_slot(src):
                    later.add(slot_of(src))
            spill = written_here & later
            lines: list[str] = []
            for s in sorted(need_load):
                lines.append(f'      r{s} = sc[{reg_base + s * trw} + i];')
            if ci == 0:
                # バケット0(命令に依存しない出力)は先頭で書く
                for k in range(layout.write_start[0], layout.write_start[1]):
                    oc = layout.write_oc[k]
                    lines.append(f'      out[{oc * plane_words} + i] = '
                                 f'{ref(layout.oc_src_off[oc], set())};')
            for j in range(lo, hi):
                tt, a_off, b_off, _ = layout.instr_flat[4 * j:4 * j + 4]
                expr = GATE_EXPR[tt & 0xf].format(a=ref(a_off, None),
                                                  b=ref(b_off, None))
                lines.append(f'      r{writes[j]} = {expr};')
                for k in range(layout.write_start[j + 1],
                               layout.write_start[j + 2]):
                    oc = layout.write_oc[k]
                    lines.append(f'      out[{oc * plane_words} + i] = '
                                 f'{ref(layout.oc_src_off[oc], None)};')
            for s in sorted(spill):
                lines.append(f'      sc[{reg_base + s * trw} + i] = r{s};')
            used = sorted(need_load | written_here)
            regs = ('' if not used else
                    '      uint32_t ' + ', '.join(f'r{k}' for k in used) + ';')
            blocks.append(
                f'  for (i = 0; i < {trw}; i++) {{\n'
                + (regs + '\n' if regs else '')
                + chr(10).join(lines) + '\n  }')
        n_streams = self._stream_count(layout)
        self.fused_fns.append(f'''
/* {op.node_id}: 命令列 {layout.n_instrs} 本を直線展開したもの。
   タップ列 {n_streams} 本を {len(chunks)} チャンクへ切り、
   **チャンク外側 × ワード内側**で回す(1チャンクが同時に触るのは
   {self.STREAM_BUDGET} 本まで)。中間スロット {n_slots} 本はレジスタに載る。 */
static void {tag}_fused(uint32_t *sc, uint32_t *out) {{
  int32_t i;
{chr(10).join(blocks)}
}}''')
        self.body.append(
            f'  mdl_conv_init_scratch(&{tag}_layout, {scratch});')
        self.body.append(
            f'  mdl_conv_build_planes(&{tag}_layout, '
            f'{self.buf(op.inputs[0])}, {scratch});')
        self.body.append(f'  {tag}_fused({scratch}, {self.buf(op.output)});')
        # 内側ループを分岐無しに保つため、行末語のマスクは最後にまとめる
        self.body.append(
            f'  mdl_conv_mask_tails(&{tag}_layout, {self.buf(op.output)});')

    #: 1チャンクが同時に触ってよいタップ列の本数。ハードウェアプリフェッチャが
    #: 追跡できるストリーム数は十数本程度で、それを超えると**プリフェッチが
    #: 効かなくなり全アクセスが需要ミスになる**。実測で detface n1 は
    #: 284本のストリームを同時に触っていて、融合したのに31%遅くなっていた。
    STREAM_BUDGET = 24

    @staticmethod
    def _stream_count(layout) -> int:
        """命令列が参照する相異なるタップ/定数オフセットの本数。"""
        offs = set()
        for j in range(layout.n_instrs):
            for o in layout.instr_flat[4 * j + 1:4 * j + 3]:
                if o < layout.reg_base:
                    offs.add(o)
        return len(offs)

    def _chunk_instrs(self, layout) -> list:
        """命令列を (lo, hi) のチャンクへ切る。

        チャンク内で同時に触るタップ列を `STREAM_BUDGET` 本以下に保つ。
        境界では生きているスロットをスクラッチへ退避して読み直す
        (スロットは2〜6本なので退避の費用は小さい)。
        """
        if layout.n_instrs == 0:
            return []
        chunks = []
        lo = 0
        seen: set = set()
        for j in range(layout.n_instrs):
            new = {o for o in layout.instr_flat[4 * j + 1:4 * j + 3]
                   if o < layout.reg_base} - seen
            if seen and len(seen) + len(new) > self.STREAM_BUDGET:
                chunks.append((lo, j))
                lo = j
                seen = set()
                new = {o for o in layout.instr_flat[4 * j + 1:4 * j + 3]
                       if o < layout.reg_base}
            seen |= new
        chunks.append((lo, layout.n_instrs))
        return chunks

    def _conv_tables(self, tag: str, layout) -> None:
        """conv のレイアウト構造体と定数配列(両経路で共通)。"""
        self._const_i32(f'{tag}_plane_c', layout.plane_c)
        self._const_i32(f'{tag}_plane_dy', layout.plane_dy)
        self._const_i32(f'{tag}_plane_dx', layout.plane_dx)
        self._const_i32(f'{tag}_instr', layout.instr_flat)
        self._const_i32(f'{tag}_oc_src', layout.oc_src_off)
        self._const_i32(f'{tag}_wstart', layout.write_start)
        self._const_i32(f'{tag}_woc', layout.write_oc)
        self.data_lines.append(f'''static const mdl_conv_layout {tag}_layout = {{
  {layout.h}, {layout.w}, {layout.h_out}, {layout.w_out},
  {layout.s_h}, {layout.s_w}, {layout.p_h}, {layout.p_w},
  {layout.out_channels},
  {layout.wpr_in}, {layout.wpr_out}, {layout.tap_wpr}, {layout.tap_rows_words},
  {layout.mask_out}u, {1 if layout.use_fast_path else 0},
  {tag}_plane_c, {tag}_plane_dy, {tag}_plane_dx,
  {layout.n_planes}, {layout.plane_stride}, {layout.plane_rows},
  {layout.in_plane_words}, {layout.out_plane_words},
  {tag}_instr, {layout.n_instrs},
  {tag}_oc_src, {tag}_wstart, {tag}_woc,
  {layout.pool_base}, {layout.reg_base}, {layout.scratch_words}
}};''')

    def _emit_gate_program_interp(self, op) -> None:
        layout = op.attrs['layout']
        tag = self._uid('conv')
        scratch = self.buf(op.attrs['scratch'][0])
        self._conv_tables(tag, layout)
        # 定数平面(0 と 全1)を埋めてから実行する。スクラッチは全convで
        # 共有しているので、1回だけでは足りない(埋めるのは 2*tap_rows_words
        # ワードだけなので、共有で浮くRAMのほうが遥かに大きい)。
        self.body.append(
            f'  mdl_conv_init_scratch(&{tag}_layout, {scratch});')
        self.body.append(
            f'  mdl_conv_exec(&{tag}_layout, {self.buf(op.inputs[0])}, '
            f'{scratch}, {self.buf(op.output)});')

    def emit_plane_op(self, op) -> None:
        a = op.attrs
        kind = a['op']
        src = self.buf(op.inputs[0])
        dst = self.buf(op.output)
        if kind == 'or_pool2x2':
            self.body.append(
                f'  mdl_or_pool2x2({src}, {dst}, {a["c"]}, {a["h"]}, {a["w"]});')
        elif kind == 'upsample2d_x2':
            self.body.append(
                f'  mdl_upsample2d_x2({src}, {dst}, '
                f'{a["c"]}, {a["h"]}, {a["w"]});')
        elif kind == 'permute_channels':
            tag = self._uid('perm')
            self._const_i32(f'{tag}_order', a['order'])
            self.body.append(
                f'  mdl_permute_channels({src}, {dst}, {tag}_order, '
                f'{a["c_out"]}, {a["h"]}, {a["wpr"]});')
        elif kind == 'concat_planes':
            tag = self._uid('cat')
            self._const_i32(f'{tag}_cin', a['c_in'])
            # ソース配列だけは実行時に組む。中身は ctx->work + 定数オフセット
            # なので、コンパイル時定数にはできない。要素数は2〜3なので
            # costは無視できる。
            names = ','.join(self.buf(i) for i in op.inputs)
            self.body.append(
                f'  {{ const uint32_t *{tag}_srcs[] = {{{names}}};\n'
                f'    mdl_concat_planes({tag}_srcs, {tag}_cin, '
                f'{len(op.inputs)}, {dst}, {a["h"]}, {a["wpr"]}); }}')
        else:
            raise UnsupportedNode(f'plane_op {kind} は未対応',
                                  node_id=op.node_id, node_type=op.node_type,
                                  backend=self.name)

    def emit_reduce_scores(self, op) -> None:
        a = op.attrs
        src = self.buf(op.inputs[0])
        dst = self.buf(op.output)
        if a['op'] == 'spatial_group_sum':
            planes = self.buf(op.attrs['scratch'][0])
            self.body.append(
                f'  mdl_spatial_group_sum({src}, {dst}, {a["n_out"]}, '
                f'{a["gs"]}, {a["h"]}, {a["w"]}, {planes});')
        elif a['op'] == 'group_sum':
            self.body.append(
                f'  mdl_group_sum({src}, {dst}, {a["in_width"]}, {a["k"]});')
        else:
            raise UnsupportedNode(f'reduce_scores {a["op"]} は未対応',
                                  node_id=op.node_id, node_type=op.node_type,
                                  backend=self.name)

    def emit_lut_dense(self, op) -> None:
        a = op.attrs
        tag = self._uid('lut')
        # 添字はこの層の**入力**(1つ前の層の出力)内の位置を指すので、
        # in_width<=65536ならuint16_tへ半減できる。実測で添字が
        # 添字+真理値表の94%を占めていたので、真理値表に次いで
        # ここが支配項だった(デプロイキット設計メモ節)。
        in_width = self.plan.value(op.inputs[0]).width
        u16 = lut_dense_indices_fit_u16(in_width)
        if u16:
            self._const_u16(f'{tag}_idx', a['indices'])
        else:
            self._const_i32(f'{tag}_idx', a['indices'])
        self._const_u32(f'{tag}_ts',
                        slice_truth_tables(a['truth_tables'], a['n_inputs']))
        # 添字表はパック経路と共有する(実測で937KB→504KBに減った)。
        # 真理値表は経路ごとに向く並びが違うので、パック経路を生やすときだけ
        # 元の並びも別に置く(下の `_emit_dense_packed`)。
        self.lut_tables[op.output] = tag
        self.lut_raw_tt[op.output] = list(a['truth_tables'])
        self.lut_idx_u16[op.output] = u16
        fn = 'mdl_lut_layer_slice_u16' if u16 else 'mdl_lut_layer_slice'
        self.body.append(
            f'  {fn}({self.buf(op.inputs[0])}, '
            f'{self.buf(op.output)}, '
            f'{a["n_inputs"]}, {a["out_width"]}, {tag}_idx, {tag}_ts);')

    def emit_lut_flat(self, op) -> None:
        """縮約済み dense 平坦ネットリストの表駆動評価。

        テーブル(`static const`)は flash に置く: ゲートごとに tt(uint8_t)、
        a/b/outputs(信号数 <= 32768 なら uint16_t、超えたら uint32_t)。
        信号表(1+in_width+n_gates バイト)は op のスクラッチ(コンテキストの
        作業領域)を使う。1ゲート = 1ループ反復の評価本体は
        runtime/c/src/mdl_kernels.c の `mdl_lut_flat_u16/u32`。
        """
        a = op.attrs
        tag = self._uid('flat')
        gates = a['gates']
        n_gates = len(gates) // 3
        u16 = lut_flat_indices_fit_u16(a['in_width'], n_gates)
        const_idx = self._const_u16 if u16 else self._const_u32
        self._const_u8(f'{tag}_tt', gates[0::3])
        const_idx(f'{tag}_a', gates[1::3])
        const_idx(f'{tag}_b', gates[2::3])
        const_idx(f'{tag}_o', a['outputs'])
        fn = 'mdl_lut_flat_u16' if u16 else 'mdl_lut_flat_u32'
        # バッチ軸パック経路が同じ表を共有できるよう覚えておく
        self.flat_tables[op.output] = (tag, u16, n_gates)
        self.body.append(
            f'  {fn}({self.buf(op.inputs[0])}, {self.buf(op.output)}, '
            f'(uint8_t *){self.buf(a["scratch"][0])}, '
            f'{a["in_width"]}, {n_gates}, {a["out_width"]}, '
            f'{tag}_tt, {tag}_a, {tag}_b, {tag}_o);')

    def emit_copy_scores(self, op) -> None:
        offset = 0
        dst = self.buf(op.output)
        for vid, width in zip(op.inputs, op.attrs['widths'], strict=True):
            self.body.append(
                f'  for (i = 0; i < {width}; i++) '
                f'{dst}[{offset} + i] = {self.buf(vid)}[i];')
            offset += width

    def emit_to_scores(self, op) -> None:
        self.body.append(
            f'  for (i = 0; i < {op.attrs["width"]}; i++) '
            f'{self.buf(op.output)}[i] = (int32_t){self.buf(op.inputs[0])}[i];')

    # -- recurrent --------------------------------------------
    def emit_flat_concat(self, op) -> None:
        """非パックの平坦ビット列同士の連結(recurrentのbody内、state+step等)。

        `deploy/plan.py::_plan_accumulate_concat`が発行する
        「アキュムレータのシード」(1入力のflat_concat。退化ゲート除去
       のグループの最初のconcatが、entryをアキュムレータの
        先頭へ1回だけ物理コピーする)は`memcpy`で書く。スカラーループ
        (下の一般形)は幅が小さい(state/step束ねは数十〜数百
        バイトが典型)なら差が出ないが、シードはentry幅
        (charlmのK-layersで2,200バイト)になりうるため、コンパイラの
        イディオム認識に頼らず明示的にmemcpyへ落とす(1入力=IRの
        concatノード(arity>=2)からは絶対に来ない、
        `_plan_accumulate_concat`だけが作る形なので安全に判別できる)。
        """
        if len(op.inputs) == 1:
            (width,) = op.attrs['widths']
            self.body.append(
                f'  mdl_memcpy({self.buf(op.output)}, {self.buf(op.inputs[0])}, '
                f'{width});')
            return
        dst = self.buf(op.output)
        offset = 0
        for vid, width in zip(op.inputs, op.attrs['widths'], strict=True):
            self.body.append(
                f'  for (i = 0; i < {width}; i++) '
                f'{dst}[{offset} + i] = {self.buf(vid)}[i];')
            offset += width

    def emit_ext_init(self, op) -> None:
        """recurrentの外部入力バッファの形式的な「定義」。

        実際の値はemit_recurrent_execのホストループが毎tick上書きする
        (tickごとに違う値なので、ここで書く値に意味は無い)。
        存在理由はop列のSSA前提(値はどれかのopの出力として生まれる)を
        満たし、生存区間解析(memplan)がこのバッファを「recurrentの
        直前あたりから生きている」と正しく判定できるようにするためだけ
        (この宣言が無いと、値の生存開始がop列の先頭扱いになり、
        メモリ計画が必要以上に長く領域を予約してしまう)。
        Cコードは何も出さない(次に読まれる前に必ず上書きされるため)。
        """

    def emit_state_init(self, op) -> None:
        """recurrentのstate役scan_varの初期値(steps回のループの外、1回だけ)。"""
        dst = self.buf(op.output)
        bits = op.attrs['bits']
        tag = self._uid('sinit')
        self._const_u8(f'{tag}_bits', bits)
        self.body.append(
            f'  for (i = 0; i < {len(bits)}; i++) '
            f'{dst}[i] = {tag}_bits[i];')

    def emit_recurrent_exec(self, op) -> None:
        """recurrentノードの本体(scores直接出力に対応)。

        bodyから切り出したop列(scope=op.node_idでタグ付けされている)を
        static な `mdl_step_<id>(mdl_ctx *ctx, int32_t t)` へ切り出し、
        メイン関数(mdl_run)にはホスト側のsteps回ループを出す。

        ステップ関数はctxだけを受け取り、body内部の値もexternal入力
        バッファ(ext_bufs)もすべて既存の `mdl_vN`(ctx->work基準の
        アドレスマクロ)経由で参照する — recurrent専用の第二の
        アドレシング手段は導入しない(既存のbuf()を素直に再利用する
        ほうが、メモリ計画・命令列不変条件との整合を壊しにくいため)。
        外部入力の該当ステップぶんのコピーインと、stacked/final_stateの
        出力の取り出しだけ、明示的なforループとして焼く。

        例外は1つ: このrecurrentの出力がグラフの最終出力そのもの
        (`op.output == self.plan.output_value`)で、かつstacked
        (body内でgroup_sumを経由してint32スコア行を直接stackする、
        デコーダのケース)のとき。`self.buf(op.output)`は
        呼び出し側配列そのものを指す特別扱いの識別子`scores`を返すが、
        それは`mdl_run`のパラメータであってstep関数のスコープには無い。
        このケースだけstep関数に`int32_t *scores`引数を追加する
        (`mdl_vN`はfile-scopeの`#define`マクロなのでstep関数内でも
        素通しで見えるが、`scores`はそうではないため必要な例外)。

        `node_id in self.expose_recurrent`のときだけ、追加で
        `_emit_recurrent_public_api`を呼ぶ。それ以外は生成物が1バイトも
        変わらない(既存動作を保つため、host_loop_block/final_state_block
        を変数へ切り出しただけで元のself.bodyへの追記は変えていない)。

        `node_id in self.drop_folded_recurrents`のときは上記を
        すべて飛ばし、折り畳みハーネスがループ境界に使う2つのマクロ
        (`MDL_STEPS_<ID>`/`MDL_OUT_WIDTH_<ID>`)だけを出して即returnする —
        重み(indices/truth_tables)・step関数・mdl_run/mdl_inferからの
        呼び出し(host_loop_block)が一切生成されないため、この
        recurrentぶんの配信サイズが丸ごと消える。出力バッファの取得関数
        (`mdl_out_<id>`)はこのメソッドの外(build()のexpose_buffers処理)
        で別途出るので、ここでは関与しない。
        """
        a = op.attrs
        node_id = op.node_id
        safe_id = _sanitize_c_ident(node_id)
        if node_id in self.drop_folded_recurrents:
            upper = safe_id.upper()
            self.public_decls.append(f'#define MDL_STEPS_{upper} {a["steps"]}')
            self.public_decls.append(
                f'#define MDL_OUT_WIDTH_{upper} {a["output_width"]}')
            return
        scope_ops = self._scoped.get(node_id, [])
        tag = f'mdl_step_{safe_id}'
        needs_scores_param = (a['output_mode'] == 'stacked'
                              and op.output == self.plan.output_value)
        expose = node_id in self.expose_recurrent

        saved_body, self.body = self.body, []
        #: op境界。scope_ops[i] が step_lines のどの範囲を書いたかを
        #: 記録するだけの副作用の無い bookkeeping で、既存の生成物は
        #: 1バイトも変えない。`self.packed_tf_target == node_id` のときだけ
        #: `_emit_packed_teacher_forcing` がこれを使って body を
        #: head(state更新まで)/tail(M+GroupSum相当)へ分割する。
        op_bounds: list[tuple] = []
        for sub in scope_ops:
            method = self.EMITTERS.get(sub.kind)
            if method is None:
                raise UnsupportedNode(
                    f'op種別 {sub.kind} は未対応(recurrent body内)',
                    node_id=sub.node_id, node_type=sub.node_type,
                    backend=self.name)
            lo = len(self.body)
            getattr(self, method)(sub)
            op_bounds.append((sub, lo, len(self.body)))
        # state_update: bodyが計算した新しいstate候補をstateバッファへ
        # コピーする(次tickの入力になる。最終tickの後もコピーするが、
        # 副作用が無いだけで害はない)。
        for sv_id, buf_id in a['state_bufs'].items():
            src_vid = a['state_update'][sv_id]
            w = self.plan.value(buf_id).width
            self.body.append(
                f'  for (i = 0; i < {w}; i++) '
                f'{self.buf(buf_id)}[i] = {self.buf(src_vid)}[i];')
        if a['output_mode'] == 'stacked':
            w = a['output_width']
            self.body.append(
                f'  for (i = 0; i < {w}; i++) '
                f'{self.buf(op.output)}[(size_t)t * {w} + i] = '
                f'{self.buf(a["output_source"])}[i];')
        step_lines = self.body
        self.body = saved_body
        #: このrecurrentノードの直前まで(前処理/split/N/K/L等、unscopedな
        #: 主経路op)に生成されたCの行。packed teacher forcing entry
        #: はこの続きに「packed decode」を足すだけで、mdl_run本体は
        #: 一切変更しない(既存経路を触らない)。
        prefix_lines = list(self.body)
        step_params = 'mdl_ctx *ctx, int32_t t'
        if needs_scores_param:
            step_params += ', int32_t *scores'
        qualifier = '' if expose else 'static '
        self.fns.append(
            f'{qualifier}void {tag}({step_params}) {{\n'
            f'  int32_t i;\n  (void)i;\n  (void)t;\n'
            + '\n'.join(step_lines) + '\n}')
        if expose:
            self.public_decls.append(f'void {tag}({step_params});')

        # ホスト側: steps回、外部入力の該当スライスをコピーしてから
        # ステップ関数を呼ぶ。host_loop_blockはmdl_runへ埋め込むのと
        # 同時に、expose時は_emit_recurrent_public_apiがラッパ関数へも
        # そのまま転用する(2箇所で同じ文字列を使うだけで、ロジックの
        # 二重化はしない)。
        lines: list[str] = []
        for sv_id, ext_id in a['ext_bufs'].items():
            w = a['ext_widths'][sv_id]
            src_id = a['ext_srcs'][sv_id]
            role = a['ext_roles'][sv_id]
            src = self.buf(src_id)
            dst = self.buf(ext_id)
            if role == 'step':
                lines.append(
                    f'    for (i = 0; i < {w}; i++) '
                    f'{dst}[i] = {src}[(size_t)t * {w} + i];')
            else:  # invariant: 全ステップ共通(毎tick同じ値を詰め直すだけ)
                lines.append(
                    f'    for (i = 0; i < {w}; i++) {dst}[i] = {src}[i];')
        call_args = 'ctx, t' + (', scores' if needs_scores_param else '')
        lines.append(f'    {tag}({call_args});')
        host_loop_block = (
            f'  {{ int32_t t; for (t = 0; t < {a["steps"]}; t++) {{\n'
            + '\n'.join(lines) + '\n  } }')
        self.body.append(host_loop_block)

        final_state_block = None
        if a['output_mode'] == 'final_state':
            buf_id = a['output_state_buf']
            w = a['output_width']
            final_state_block = (
                f'  for (i = 0; i < {w}; i++) '
                f'{self.buf(op.output)}[i] = {self.buf(buf_id)}[i];')
            self.body.append(final_state_block)

        if expose:
            self._emit_recurrent_public_api(
                op, a, safe_id, tag, needs_scores_param, host_loop_block,
                final_state_block)

        if node_id == self.packed_tf_target:
            self._emit_packed_teacher_forcing(
                op, a, safe_id, scope_ops, op_bounds, step_lines,
                prefix_lines)

    def _emit_recurrent_public_api(self, op, a, safe_id, tag,
                                   needs_scores_param, host_loop_block,
                                   final_state_block) -> None:
        """recurrentノード1つぶんの公開API。

        自己回帰生成ハーネス(runtime/c/src/mdl_rddlgn.c)がRDDLGN固有の
        ノードid(enc_n_rec/enc_k_rec/dec_l_rec/dec_pm_rec、export/
        rddlgn_ir.pyが常に同じ名前で生成する)を直接呼ぶ前提の、小さな
        公開関数群を追加する。このメソッド自体はノード型に依存しない
        汎用ロジックで、どのrecurrentノードを公開するかは
        `self.expose_recurrent`(呼び出し側 = generate_rddlgn_c)が決める。

        追加するもの:
            mdl_run_recurrent_<id>(ctx[, scores])
                state再初期化(あれば) + host_loop_block + 最終state
                コピー(あれば)。教師強制のmdl_run内で起きるのと同じ
                処理を単独で呼べる形にしただけ(挙動は完全に同一)。
            mdl_ext_<id>_<sv>(ctx) -> uint8_t*
                外部入力(step/invariant)バッファの直接書き込み用アドレス。
            mdl_state_<id>_<sv>(ctx) -> uint8_t*
                stateバッファの直接読み書き用アドレス。生成ハーネスは
                他recurrentのstep関数を挟んでもこの値が生存していると
                仮定しない(アリーナ共有の詳細に依存しないよう、呼ぶたびに
                取得して手動でsave/restoreする設計。進捗記録
                deploykit-plan.md節参照)。
            mdl_reset_state_<id>(ctx)
                state初期値の再書き込み(bundle生成時の定数)。
            MDL_STEPS_<ID> / MDL_EXT_WIDTH_<ID>_<SV>
                コンパイル時定数(ループ上限・バッファ幅)。
        """
        upper = safe_id.upper()
        self.public_decls.append(f'#define MDL_STEPS_{upper} {a["steps"]}')
        self.public_decls.append(
            f'#define MDL_OUT_WIDTH_{upper} {a["output_width"]}')

        for sv_id, buf_id in a['ext_bufs'].items():
            safe_sv = _sanitize_c_ident(sv_id)
            name = f'mdl_ext_{safe_id}_{safe_sv}'
            self.public_defs.append(
                f'uint8_t *{name}(mdl_ctx *ctx) {{ return {self.buf(buf_id)}; }}')
            self.public_decls.append(f'uint8_t *{name}(mdl_ctx *ctx);')
            w = a['ext_widths'][sv_id]
            self.public_decls.append(
                f'#define MDL_EXT_WIDTH_{upper}_{safe_sv.upper()} {w}')

        reset_fn = None
        reset_lines: list[str] = []
        for sv_id, buf_id in a['state_bufs'].items():
            safe_sv = _sanitize_c_ident(sv_id)
            state_name = f'mdl_state_{safe_id}_{safe_sv}'
            w = self.plan.value(buf_id).width
            self.public_defs.append(
                f'uint8_t *{state_name}(mdl_ctx *ctx) '
                f'{{ return {self.buf(buf_id)}; }}')
            self.public_decls.append(f'uint8_t *{state_name}(mdl_ctx *ctx);')
            self.public_decls.append(
                f'#define MDL_STATE_WIDTH_{upper}_{safe_sv.upper()} {w}')

            bits = self._state_bits_by_buf[buf_id]
            btag = self._uid('rsinit')
            self._const_u8(f'{btag}_bits', bits)
            dst = self.buf(buf_id)
            reset_lines.append(f'  for (i = 0; i < {len(bits)}; i++) '
                               f'{dst}[i] = {btag}_bits[i];')
        if reset_lines:
            reset_fn = f'mdl_reset_state_{safe_id}'
            self.public_defs.append(
                f'void {reset_fn}(mdl_ctx *ctx) {{\n  int32_t i;\n  (void)i;\n'
                + '\n'.join(reset_lines) + '\n}')
            self.public_decls.append(f'void {reset_fn}(mdl_ctx *ctx);')

        run_params = 'mdl_ctx *ctx' + (
            ', int32_t *scores' if needs_scores_param else '')
        run_name = f'mdl_run_recurrent_{safe_id}'
        run_lines = []
        if reset_fn:
            run_lines.append(f'  {reset_fn}(ctx);')
        run_lines.append(host_loop_block)
        if final_state_block:
            run_lines.append(final_state_block)
        self.public_defs.append(
            f'void {run_name}({run_params}) {{\n  int32_t i;\n  (void)i;\n'
            + '\n'.join(run_lines) + '\n}')
        self.public_decls.append(f'void {run_name}({run_params});')

    # -- 教師強制の時間バッチ化 ------------------------------------
    def _emit_packed_teacher_forcing(self, op, a, safe_id, scope_ops,
                                     op_bounds, step_lines,
                                     prefix_lines) -> None:
        """recurrent bodyを head(state更新まで)/tail(M+GroupSum相当)へ
        分割し、tailをsteps回まとめて1回のpacked評価にする。

        効く理由: RDDLGNのデコーダは `state_t = P(state_{t-1}, ext_t)`
        (真のリカレント、順に計算するしかない)の直後に
        `out_t = M(state_t, ext_t)` (state_tだけの純関数、tick間の依存が
        無い)が続く1本のrecurrent bodyに畳まれている。
        stateだけをsteps回順に計算しながら`state_t`をバッファへ書き貯め、
        tail(M-layers+GroupSum)はsteps本まとめて`mdl_lut_layer_packed`
        (n_inputs=2をbutterfly-select特化、実測9.3〜14.7倍)で
        1回評価する。既存の`mdl_step_<id>`/`mdl_run_recurrent_<id>`や
        `mdl_run`本体は一切変更しない — このメソッドは新しい関数
        `mdl_infer_packed_tf_ctx`を**追加**するだけ(既存経路を残して選択制、
        perf-huntスキル・タスクの方針通り)。

        前提を満たさなければ ValueError で落ちる(呼び出し側が明示的に
        `packed_teacher_forcing=node_id` を指定しているので、黙ってスキップ
        すると「機能が効いていないことに気づかない」事故になるため)。
        """
        def fail(msg: str) -> None:
            raise ValueError(
                f'packed_teacher_forcing={op.node_id!r}: {msg}')

        if a['output_mode'] != 'stacked':
            fail("output_mode='stacked'のrecurrentにしか使えない")
        if self.plan.value(op.output).kind != KIND_SCORES:
            fail('出力がKIND_SCORES(body内でgroup_sum経由)である必要がある')
        if len(a['state_bufs']) != 1:
            fail(f'state変数が{len(a["state_bufs"])}個(1個のみ対応)')
        if op.node_id not in self.expose_recurrent:
            fail('expose_recurrentにも含める必要がある(state初期化を'
                'mdl_reset_state_<id>に委ねるため)')

        (state_sv_id,) = a['state_bufs'].keys()
        cut_vid = a['state_update'][state_sv_id]
        cut_idx = next((i for i, (sub, _, _) in enumerate(op_bounds)
                        if sub.output == cut_vid), None)
        if cut_idx is None:
            fail('state_updateの対象がbody内のどのopの出力とも一致しない')

        tail_ops = scope_ops[cut_idx + 1:]
        if not tail_ops:
            fail('state更新がbodyの最後のopで、tail(M相当)が空')
        if tail_ops[0].kind != 'flat_concat':
            fail(f'tailの先頭が{tail_ops[0].kind}(flat_concatのみ対応)')
        if (tail_ops[-1].kind != 'reduce_scores'
                or tail_ops[-1].attrs.get('op') != 'group_sum'):
            fail(f'tailの末尾が{tail_ops[-1].kind}(group_sumのみ対応)')
        if tail_ops[-1].output != a['output_source']:
            fail('tail末尾のgroup_sum出力がoutput_sourceと一致しない')
        for sub in tail_ops[1:-1]:
            if sub.kind == 'flat_concat':
                # 退化ゲート除去(export/rddlgn_optimize.py)が
                # アキュムレータグループの最初のconcatにだけ挿む「entryを
                # 1回だけ物理コピーするシード」op(1入力のflat_concat —
                # 通常のconcat(2入力以上)とはここで判別できる)。tailの
                # packed経路では`mdl_pack_batch(pm_rows, pm_x0, ...)`が
                # 既にpm_x0[0:row_width]へ同じ範囲を詰めているので、この
                # opはpacked側では読み飛ばしてよい(下のレイアウト構築が
                # `val.alias_of`を見て自動的に扱う)。
                if len(sub.inputs) != 1:
                    fail(f'tail中間のflat_concatが{len(sub.inputs)}入力'
                        '(accumulateシード(1入力)のみ許容)')
                continue
            if sub.kind != 'lut_dense':
                fail(f'tail中間に{sub.kind}'
                    '(lut_dense連鎖のaccumulateシードのみ対応)')
            if sub.attrs['n_inputs'] != 2:
                fail(f'lut_dense n_inputs={sub.attrs["n_inputs"]}'
                    '(2のみ対応、mdl_lut_layer_packedのn2特化が前提)')

        # 依存関係チェック: tailはcut_vid / ext_bufs以外の外部値(=headだけの
        # 中間値やstateバッファそのもの)を読んでいないこと。
        ext_vids = set(a['ext_bufs'].values())
        allowed = {cut_vid} | ext_vids
        tail_internal: set = set()
        for sub in tail_ops:
            for iv in sub.inputs:
                if iv not in tail_internal and iv not in allowed:
                    fail(f'tailのop({sub.kind})がhead専用の中間値'
                        f'(value id={iv})に依存している(未対応の形)')
            tail_internal.add(sub.output)

        # ---- head-only step関数(既存のmdl_step_<id>から重み表を再生成
        #      せず、step_linesのうちheadぶんの行をそのまま再利用する。
        #      これで定数表(_const_*)が二重に出ない)。 ----
        head_end = op_bounds[cut_idx][2]
        cut_width = self.plan.value(cut_vid).width
        state_buf = self.buf(a['state_bufs'][state_sv_id])
        # state_update(次tickのp_prevへの書き戻し)をhead自身に含める。
        # 元のstep_lines(emit_recurrent_execの本体)ではscope_opsループの
        # **外**(op_boundsの対象外)にあるため、ここで明示的に1行足す。
        # head_tagを「Pだけを1tickぶん進め、次に読めるようstateも
        # 確定させる」自己完結な関数にしておくことで、(greedy生成の
        # 系列バッチ化)がcut_vid自体のバッファ名(mdl_vNの実装詳細)を
        # 知らなくても`mdl_state_<id>_<sv>`経由でp_tを読めるようになる。
        head_lines = list(step_lines[:head_end]) + [
            f'  for (i = 0; i < {cut_width}; i++) '
            f'{state_buf}[i] = {self.buf(cut_vid)}[i];']
        head_tag = f'mdl_step_{safe_id}_head'
        # 非staticで公開する(greedy生成の系列バッチ化が
        # 32本の独立系列それぞれのPだけを単発評価するのに再利用する。
        # ctxのアリーナは1個ぶんしか無いので、複数系列を同じctxで回す
        # 呼び出し側はmdl_rddlgn.cのsave/restore規律に従うこと)。
        self.fns.append(
            f'void {head_tag}(mdl_ctx *ctx) {{\n'
            f'  int32_t i;\n  (void)i;\n'
            + '\n'.join(head_lines) + '\n}')
        self.public_decls.append(f'void {head_tag}(mdl_ctx *ctx);')

        steps = int(a['steps'])
        bwords = (steps + 31) // 32

        ext_meta: dict[int, tuple[str, int, str]] = {}
        for sv_id, buf_id in a['ext_bufs'].items():
            role = a['ext_roles'][sv_id]
            w = a['ext_widths'][sv_id]
            src_expr = self.buf(a['ext_srcs'][sv_id])
            ext_meta[buf_id] = (role, w, src_expr)

        # concatの並び通りに「行(1tickぶんのM入力)」のレイアウトを決める。
        concat_op = tail_ops[0]
        row_fields = []   # (kind, offset, width, src_expr|None)
        row_off = 0
        for vid, w in zip(concat_op.inputs, concat_op.attrs['widths'],
                          strict=True):
            if vid == cut_vid:
                row_fields.append(('state', row_off, w, None))
            elif vid in ext_meta:
                role, ew, src_expr = ext_meta[vid]
                if ew != w:
                    fail('concat入力幅とext幅が食い違う')
                row_fields.append((role, row_off, w, src_expr))
            else:
                fail('concat入力がcut_value/ext_bufsのどちらでもない')
            row_off += w
        row_width = row_off

        # ---- work領域のレイアウト(呼び出し側持ち。_emit_dense_packed
        #      と同じ規律 — staticにすると並行呼び出しでスレッドセーフで
        #      なくなる)。 ----
        # tail中間に(退化ゲート除去、export/rddlgn_optimize.py)の
        # accumulate-groupがあれば(alias_of!=Noneの出力を持つlut_denseが
        # あれば)、pm_x0自体をそのグループの最終幅ぶんへ拡張する。
        # `mdl_pack_batch(pm_rows, pm_x0, row_width, steps)`(下)が既に
        # pm_x0[0:row_width]をentry(state+ext)で埋めるので、
        # 「シードコピー」(通常経路でのflat_concat)はpacked経路では
        # 何もしなくてよい — グループの各層は「pm_x0内の絶対オフセット」を
        # 直接読み書きする(deploy/plan.py::_plan_accumulate_concatの
        # 通常経路(uint8_t*アリーナ)と同じ考え方をpacked
        # (bwords幅ビットプレーン)のpm_x0に当てはめている)。無効の
        # tailは全opがalias_of=Noneの独立バッファ連鎖のままなので、
        # final_width=row_widthで従来通り(生成コードは1バイトも変わらない)。
        mid_ops = tail_ops[1:-1]
        entry_vid = concat_op.output
        final_width = row_width
        parent_vid: int | None = None
        for sub in mid_ops:
            if sub.kind != 'lut_dense':
                continue
            val = self.plan.value(sub.output)
            if val.alias_of is not None:
                parent_vid = val.alias_of
                final_width = self.plan.value(parent_vid).width
                break   # 1スタックにaccumulate-groupは高々1つ(設計上一意)

        off = 0
        rows_off = off
        rows_words = (steps * row_width + 3) // 4
        off += rows_words
        defs = [f'#define pm_rows ((uint8_t *)(work + {rows_off}))']
        x_off = off
        off += final_width * bwords
        defs.append(f'#define pm_x0 (work + {x_off})')

        tail_lines = []
        # で再利用: (tag, out_width, alias_offset|None, src_is_base,
        # idx_u16)。alias_offsetがNoneでなければaccumulate先
        # (pm_x0内オフセット)、src_is_baseはこの層の入力がpm_x0(entryまたは
        # グループのアキュムレータ)そのものかどうか。idx_u16は添字表の型
        # (単枚経路emit_lut_dense/self.lut_idx_u16と共有)。
        tail_layer_tags: list[tuple[str, int, int | None, bool, bool]] = []
        buf_names: dict[int, str] = {}
        prev = 'pm_x0'
        n = 0
        for sub in mid_ops:
            if sub.kind == 'flat_concat':
                continue   # のシード。packed経路ではmdl_pack_batchが代役
            aa = sub.attrs
            val = self.plan.value(sub.output)
            src_vid = sub.inputs[0]
            src_is_base = src_vid in (entry_vid, parent_vid)
            src = 'pm_x0' if src_is_base else buf_names[src_vid]
            if val.alias_of is not None:
                out = f'(pm_x0 + {val.alias_offset} * {bwords})'
            else:
                n += 1
                out = f'pm_v{n}'
                defs.append(f'#define {out} (work + {off})')
                off += aa['out_width'] * bwords
            tag = self.lut_tables[sub.output]
            self._const_u32(f'{tag}_ptt', self.lut_raw_tt[sub.output])
            # 添字表は単枚経路(emit_lut_dense)と共有。型もそちらの
            # 判定をそのまま使う(charlmのM-layersが主要経路)。
            # (下のgreedy32経路)がtail_layer_tagsを再利用するので、
            # u16フラグもalias情報と一緒にここで運ぶ。
            u16 = self.lut_idx_u16[sub.output]
            tail_layer_tags.append(
                (tag, int(aa['out_width']),
                 val.alias_offset if val.alias_of is not None else None,
                 src_is_base, u16))
            pfn = 'mdl_lut_layer_packed_u16' if u16 else 'mdl_lut_layer_packed'
            tail_lines.append(
                f'  {pfn}({src}, {out}, 2, '
                f'{aa["out_width"]}, {tag}_idx, {tag}_ptt, {bwords});')
            buf_names[sub.output] = out
            prev = out
        gs = tail_ops[-1].attrs
        tail_lines.append(
            f'  mdl_group_sum_packed({prev}, scores, {gs["in_width"]}, '
            f'{gs["k"]}, {steps}, {bwords});')
        work_words = off
        self.packed_tf_work_words = work_words

        # ---- head loop: stateをsteps回進めながらpm_rowsへ書き貯める ----
        loop_lines = []
        for sv_id, buf_id in a['ext_bufs'].items():
            role = a['ext_roles'][sv_id]
            w = a['ext_widths'][sv_id]
            src = self.buf(a['ext_srcs'][sv_id])
            dst = self.buf(buf_id)
            if role == 'step':
                loop_lines.append(
                    f'      for (i = 0; i < {w}; i++) '
                    f'{dst}[i] = {src}[(size_t)t * {w} + i];')
            else:
                loop_lines.append(
                    f'      for (i = 0; i < {w}; i++) {dst}[i] = {src}[i];')
        # head_tag自身がstate_update(次tickのp_prevへの書き戻し)を
        # 済ませているので、ここでは呼ぶだけでよい(次tickのhead呼び出しが
        # 更新後のstate_bufを正しく読める)。
        loop_lines.append(f'      {head_tag}(ctx);')
        for kind, roff, w, src_expr in row_fields:
            dstexpr = f'pm_rows[(size_t)t * {row_width} + {roff} + i]'
            if kind == 'state':
                loop_lines.append(
                    f'      for (i = 0; i < {w}; i++) '
                    f'{dstexpr} = {state_buf}[i];')
            elif kind == 'invariant':
                loop_lines.append(
                    f'      for (i = 0; i < {w}; i++) '
                    f'{dstexpr} = {src_expr}[i];')
            else:  # step
                loop_lines.append(
                    f'      for (i = 0; i < {w}; i++) '
                    f'{dstexpr} = {src_expr}[(size_t)t * {w} + i];')

        upper = safe_id.upper()
        entry_name = 'mdl_infer_packed_tf_ctx'
        self.packed_tf_entry = f'''
/* 教師強制の時間バッチ化。{op.node_id} の state は steps回順に
   計算するしかないが(真のリカレント)、tail(M-layers+GroupSum、
   state_tだけの純関数)はtick間に依存が無いので、steps本まとめて
   mdl_lut_layer_packed(n_inputs=2特化)で1回評価する。値は既存の
   `mdl_infer_ctx`/`mdl_run_recurrent_{safe_id}`と完全に一致する
   (tests/deploy/test_c_rddlgn_packed_tf.pyで固定)。
   work は呼び出し側が用意する(dense-batchと同じ規律)。*/
#define MDL_{upper}_PACKED_WORK_BYTES {work_words * 4}
{chr(10).join(defs)}
int {entry_name}(mdl_ctx *ctx, const uint8_t *mdl_input,
                 uint32_t *work, size_t work_bytes, int32_t *scores) {{
  int32_t i, t;
  (void)i;
  if (ctx == NULL || ctx->work == NULL || mdl_input == NULL) return -1;
  if (work == NULL || work_bytes < MDL_{upper}_PACKED_WORK_BYTES) return -1;
{chr(10).join(prefix_lines)}
  mdl_reset_state_{safe_id}(ctx);
  for (t = 0; t < {steps}; t++) {{
{chr(10).join(loop_lines)}
  }}
  mdl_pack_batch(pm_rows, pm_x0, {row_width}, {steps});
{chr(10).join(tail_lines)}
  return 0;
}}
'''
        self.public_decls.append(
            f'#define MDL_{upper}_PACKED_WORK_BYTES {work_words * 4}')
        self.public_decls.append(
            f'int {entry_name}(mdl_ctx *ctx, const uint8_t *mdl_input, '
            f'uint32_t *work, size_t work_bytes, int32_t *scores);')

        # ----: tail(M+GroupSum)を「32本の独立系列」向けにも公開する
        #      (greedy生成の系列バッチ化)。教師強制はsteps本(tick軸)を
        #      詰めたが、こちらはlanes=32固定(1語ぶん)で、呼び出し側が
        #      「何を32本詰めたか」を自由に決められる(rowsを渡すだけの
        #      関数なので、教師強制のtickでも独立系列でも同じtail計算に
        #      写せる)。_ptt/_idx定数表は上ですでに出ているので、
        #      ここでは名前を再利用するだけ(二重定義しない)。 */
        lanes = 32
        lbwords = 1
        loff = 0
        ldefs = [f'#define pg_x0 (work + {loff})']
        # 上のpm_x0と同じ理由でfinal_widthぶん確保する
        # (accumulate-groupが無ければfinal_width==row_widthで従来通り)。
        loff += final_width * lbwords
        ltail_lines = []
        lprev = 'pg_x0'
        ln = 0
        for tag, out_w, alias_off, src_is_base, u16 in tail_layer_tags:
            lsrc = 'pg_x0' if src_is_base else lprev
            if alias_off is not None:
                lout = f'(pg_x0 + {alias_off} * {lbwords})'
            else:
                ln += 1
                lout = f'pg_v{ln}'
                ldefs.append(f'#define {lout} (work + {loff})')
                loff += out_w * lbwords
            lpfn = 'mdl_lut_layer_packed_u16' if u16 else 'mdl_lut_layer_packed'
            ltail_lines.append(
                f'  {lpfn}({lsrc}, {lout}, 2, '
                f'{out_w}, {tag}_idx, {tag}_ptt, {lbwords});')
            lprev = lout
        ltail_lines.append(
            f'  mdl_group_sum_packed({lprev}, scores, {gs["in_width"]}, '
            f'{gs["k"]}, {lanes}, {lbwords});')
        lwork_words = loff
        ltag_name = f'mdl_{safe_id}_tail_packed32'
        self.packed_tf_entry += f'''
/* {op.node_id} のtail(M-layers+GroupSum)を32本ぶんまとめて
   1回のpacked評価にする、教師強制と独立に再利用できる形。rowsは
   呼び出し側が[32][{row_width}]バイトで用意する
   (レイアウトはMDL_{upper}_ROW_WIDTHの節を参照 — state/{{context/l}}を
   concatの並び通りに詰める)。work は呼び出し側が用意する。 */
#define MDL_{upper}_TAIL_PACKED32_WORK_BYTES {lwork_words * 4}
#define MDL_{upper}_ROW_WIDTH {row_width}
{chr(10).join(ldefs)}
int {ltag_name}(uint32_t *work, size_t work_bytes,
                const uint8_t *rows, int32_t *scores) {{
  if (work == NULL
      || work_bytes < MDL_{upper}_TAIL_PACKED32_WORK_BYTES)
    return -1;
  mdl_pack_batch(rows, pg_x0, {row_width}, {lanes});
{chr(10).join(ltail_lines)}
  return 0;
}}
'''
        self.public_decls.append(
            f'#define MDL_{upper}_TAIL_PACKED32_WORK_BYTES {lwork_words * 4}')
        self.public_decls.append(f'#define MDL_{upper}_ROW_WIDTH {row_width}')
        self.public_decls.append(
            f'int {ltag_name}(uint32_t *work, size_t work_bytes, '
            f'const uint8_t *rows, int32_t *scores);')

    def _emit_public_buffer(self, op) -> None:
        """任意opの出力バッファへ直接アクセスする取得関数。

        `self.expose_buffers`に入っているノードidのopに対して、build()の
        主ループから呼ばれる(recurrent/非recurrent問わず同じ扱い)。
        マクロ名は`MDL_BUF_WIDTH_<ID>`とし、`_emit_recurrent_public_api`の
        `MDL_OUT_WIDTH_<ID>`(recurrentの論理出力幅=1tickぶん)とは
        意図的に別名にしてある — stacked出力ノード(例: dec_l_rec)では
        実バッファ幅(steps*output_width)と論理出力幅(output_width)が
        異なり、同名マクロを2箇所で定義すると値が食い違って再定義エラーに
        なるため。
        """
        safe_id = _sanitize_c_ident(op.node_id)
        val = self.plan.value(op.output)
        ctype = 'uint8_t' if val.kind == KIND_FLAT else 'int32_t'
        name = f'mdl_out_{safe_id}'
        self.public_defs.append(
            f'{ctype} *{name}(mdl_ctx *ctx) {{ return {self.buf(op.output)}; }}')
        self.public_decls.append(f'{ctype} *{name}(mdl_ctx *ctx);')
        self.public_decls.append(
            f'#define MDL_BUF_WIDTH_{safe_id.upper()} {val.width}')

    def _emit_embed_table(self) -> None:
        """RDDLGN埋め込み表(bundle['embed_table'])をmodel.cへ焼き込む。

        export/rddlgn_ir.py::extract_embed_table が作った、行(トークン)
        ごとにceil(d_emb/32)語・LSB-firstでパックした表をそのまま
        `static const`配列へ落とし、生成ハーネスから使う
        `mdl_embed_lookup(token, out_bits)`(トークンID → d_emb個の0/1
        バイト列)を1つだけ生成する。パック解除はここに1回書くだけで、
        コアmdl_kernels.cは触らない。
        """
        et = self.plan.embed_table
        vocab = int(et['vocab'])
        d_emb = int(et['d_emb'])
        wpr = (d_emb + 31) // 32
        self._const_u32('mdl_embed_table', et['bits'])
        self.public_decls.append(f'#define MDL_VOCAB_SIZE {vocab}')
        self.public_decls.append(f'#define MDL_D_EMB {d_emb}')
        self.public_decls.append(
            'void mdl_embed_lookup(uint32_t token, uint8_t *out_bits);')
        self.public_defs.append(f'''void mdl_embed_lookup(uint32_t token, uint8_t *out_bits) {{
  int32_t j;
  const uint32_t *row = mdl_embed_table + (size_t)token * {wpr}u;
  for (j = 0; j < {d_emb}; j++) {{
    out_bits[j] = (uint8_t)((row[j >> 5] >> (j & 31)) & 1u);
  }}
}}''')

    #: (bundleセクション名, Cシンボル接尾辞, 公開マクロ名) の3つ組。
    #: emit_folded_tablesがこのペアぶんだけ呼ばれる。
    FOLDED_TABLE_SECTIONS = (
        ('folded_n_table', 'folded_n', 'FOLDED_N_TABLE'),
        ('folded_l_table', 'folded_l', 'FOLDED_L_TABLE'),
    )

    def _emit_folded_table(self, section: str, sym: str, macro: str) -> None:
        """RDDLGN語彙折り畳み表(bundle['folded_n_table']/['folded_l_table'])
        をmodel.cへ焼き込む。

        export/rddlgn_ir.py::extract_folded_n_table/extract_folded_l_table
        が作った、行(トークン)ごとにceil(width/32)語・LSB-firstでパックした
        表を`static const`配列へ落とし、`mdl_folded_n_lookup`/
        `mdl_folded_l_lookup`(トークンID → width個の0/1バイト列)を生成する。
        `_emit_embed_table`と同型 — 表の中身が「埋め込みビットそのもの」か
        「埋め込み+N/L-layers通過後のビット」かが違うだけ。

        あわせて`#define MDL_HAS_FOLDED_N_TABLE 1`(存在フラグ)を出す。
        runtime/c/src/mdl_rddlgn.cの`mdl_rddlgn_generate_greedy_folded`は
        このマクロの有無で「N/L-layersをスキップして表引きする経路」と
        「従来通りN/L-layersを評価する経路」をコンパイル時に切り替える
        (折り畳み非対象のモデルでも同じ関数が両方の意味で動く)。
        """
        ft = getattr(self.plan, section)
        vocab = int(ft['vocab'])
        width = int(ft['width'])
        wpr = (width + 31) // 32
        table_name = f'mdl_{sym}_table'
        self._const_u32(table_name, ft['bits'])
        self.public_decls.append(f'#define MDL_HAS_{macro} 1')
        self.public_decls.append(f'#define MDL_{macro}_VOCAB {vocab}')
        self.public_decls.append(f'#define MDL_{macro}_WIDTH {width}')
        self.public_decls.append(
            f'void mdl_{sym}_lookup(uint32_t token, uint8_t *out_bits);')
        self.public_defs.append(f'''void mdl_{sym}_lookup(uint32_t token, uint8_t *out_bits) {{
  int32_t j;
  const uint32_t *row = {table_name} + (size_t)token * {wpr}u;
  for (j = 0; j < {width}; j++) {{
    out_bits[j] = (uint8_t)((row[j >> 5] >> (j & 31)) & 1u);
  }}
}}''')

    # -- dense のバッチ軸パック経路 ------------------------------
    #: この経路で扱える op(全部これなら packed 版を追加で生成できる)
    DENSE_PACKED_OPS = frozenset({'lut_dense', 'lut_flat', 'reduce_scores',
                                  'to_scores'})

    def dense_packed_eligible(self) -> bool:
        """グラフ全体が非パックの dense 経路だけでできているか。

        conv は「1ワード=空間32画素」で既にパックされているが、dense には
        空間が無いので同じ手が使えず、1ビットを1バイトで持っていた。
        バッチ軸へ詰めれば32画像を1命令で進められる(difflogic 本家と同じ形)。
        """
        if self.plan.preproc is not None:
            return False
        for op in self.plan.ops:
            if op.kind not in self.DENSE_PACKED_OPS:
                return False
            if op.kind == 'reduce_scores' and op.attrs.get('op') != 'group_sum':
                return False
        return bool(self.plan.ops)

    def _emit_dense_packed(self) -> str:
        """バッチ軸パック版の入口を組み立てる(通常の単枚経路とは別に生やす)。"""
        plan = self.plan
        b = self.dense_batch
        bw = (b + 31) // 32
        lines: list[str] = []
        names: dict[int, str] = {plan.input_value: 'dp_x'}
        # 作業領域は呼び出し側が用意する(と同じ規律。static にすると
        # せっかくバッチで並列化したのにスレッドセーフでなくなる)。
        # 入力を直接読む lut_flat があれば、その信号表の中(定数0の
        # 1 信号ぶん後ろ)へ入力を詰めて、入力のコピーを省く。
        first_flat = next((op for op in plan.ops if op.kind == 'lut_flat'
                           and op.inputs[0] == plan.input_value), None)
        if first_flat is not None:
            fa = first_flat.attrs
            n_sig = 1 + fa['in_width'] + len(fa['gates']) // 3
            defs = ['#define dp_s1 (work)', f'#define dp_x (work + {bw})']
            off = n_sig * bw
        else:
            off = plan.input_width * bw
            defs = ["#define dp_x (work)"]
        n = 0
        n_flat = 0
        for op in plan.ops:
            a = op.attrs
            if op.kind == 'lut_flat':
                n_flat += 1
                tag, u16, n_gates = self.flat_tables[op.output]
                if op is first_flat:
                    sig = 'dp_s1'
                else:
                    sig = f'dp_s{n_flat}'
                    defs.append(f'#define {sig} (work + {off})')
                    off += (1 + a['in_width'] + n_gates) * bw
                out = f'dp_f{n_flat}'
                defs.append(f'#define {out} (work + {off})')
                off += a['out_width'] * bw
                names[op.output] = out
                pfn = ('mdl_lut_flat_packed_u16' if u16
                       else 'mdl_lut_flat_packed_u32')
                lines.append(
                    f'  {pfn}({names[op.inputs[0]]}, {out}, {sig}, '
                    f'{a["in_width"]}, {n_gates}, {a["out_width"]}, '
                    f'{tag}_tt, {tag}_a, {tag}_b, {tag}_o, bw);')
            elif op.kind == 'lut_dense':
                n += 1
                out = f'dp_v{n}'
                defs.append(f'#define {out} (work + {off})')
                off += a['out_width'] * bw
                names[op.output] = out
                tag = self.lut_tables[op.output]
                # パック経路は「1ゲートの表 × 語ループ」なので、スライス表現から
                # 毎回組み直すと7%遅い(実測 49,900 → 46,700 枚/秒)。この経路を
                # 生やすときだけ元の並びも置く。単枚経路のサイズ削減は損なわない。
                self._const_u32(f'{tag}_tt', self.lut_raw_tt[op.output])
                # 添字表は単枚経路(emit_lut_dense)と共有するので、型も
                # そちらの判定をそのまま使う(ここで独自に判定すると
                # 食い違って未定義動作になりうる)。
                pfn = ('mdl_lut_layer_packed_u16' if self.lut_idx_u16[op.output]
                      else 'mdl_lut_layer_packed')
                lines.append(
                    f'  {pfn}({names[op.inputs[0]]}, {out}, '
                    f'{a["n_inputs"]}, {a["out_width"]}, {tag}_idx, '
                    f'{tag}_tt, bw);')
            elif op.kind == 'reduce_scores':
                names[op.output] = 'scores'
                lines.append(
                    f'  mdl_group_sum_packed({names[op.inputs[0]]}, scores, '
                    f'{a["in_width"]}, {a["k"]}, n, bw);')
            else:      # to_scores: 0/1 をそのままスコアへ(グループ幅1)
                names[op.output] = 'scores'
                lines.append(
                    f'  mdl_group_sum_packed({names[op.inputs[0]]}, scores, '
                    f'{a["width"]}, {a["width"]}, n, bw);')
        self.dense_work_words = off
        return f'''
/* バッチ軸パック経路。1ワード=32画像で、**重みと添字表が32画像で
   1回しか読まれない**のが効く(k6000 は表が422KBあり、1枚ずつ回すと
   毎回L1から溢れる)。値は単枚経路と完全に一致する。
   作業領域は呼び出し側持ちなので、別々の work を渡せば並行に呼べる。 */
{chr(10).join(defs)}
int mdl_infer_batch_packed(uint32_t *work, size_t bytes,
                           const uint8_t *inputs, int32_t *scores, int n) {{
  const int32_t bw = (n + 31) / 32;
  if (n < 1 || n > MDL_DENSE_BATCH) return -1;
  if (work == NULL || bytes < MDL_DENSE_WORK_BYTES) return -1;
  mdl_pack_batch(inputs, dp_x, {plan.input_width}, n);
{chr(10).join(lines)}
  return 0;
}}
'''

    #: op種別 → メソッド名(サブクラスが個別に上書きできるよう名前で持つ)
    EMITTERS = {
        'pack': 'emit_pack',
        'unpack': 'emit_unpack',
        'gate_program': 'emit_gate_program',
        'plane_op': 'emit_plane_op',
        'reduce_scores': 'emit_reduce_scores',
        'lut_dense': 'emit_lut_dense',
        'lut_flat': 'emit_lut_flat',
        'copy_scores': 'emit_copy_scores',
        'to_scores': 'emit_to_scores',
        'flat_concat': 'emit_flat_concat',
        'ext_init': 'emit_ext_init',
        'state_init': 'emit_state_init',
        'recurrent_exec': 'emit_recurrent_exec',
    }

    # -- 全体 ---------------------------------------------------------
    def build(self) -> dict[str, str]:
        plan = self.plan
        self._declare_buffers()
        if plan.preproc is not None:
            self._emit_preproc_tables()
        if plan.embed_table is not None:
            self._emit_embed_table()
        for section, sym, macro in self.FOLDED_TABLE_SECTIONS:
            if getattr(plan, section) is not None:
                self._emit_folded_table(section, sym, macro)
        if self.drop_folded_recurrents:
            # mdl_rddlgn.cの旧関数(mdl_rddlgn_generate_greedy、bits版)は
            # このマクロを見て宣言/定義ごと自分を無効化する — 対応する
            # weights/step関数がこの生成物に無いのでリンクできないため。
            self.public_decls.append('#define MDL_RDDLGN_WEIGHTS_DROPPED 1')
        # state_initの出力バッファ→初期化ビット列(# _emit_recurrent_public_apiのmdl_reset_state_<id>が使う)。
        # 常に集めておく(exposeが空なら誰も読まないので無害)。
        self._state_bits_by_buf = {
            op.output: op.attrs['bits'] for op in plan.ops
            if op.kind == 'state_init'
        }
        # scopeが付いたop(recurrentのbody由来)は主経路には出さず、
        # emit_recurrent_execがステップ関数として切り出す。
        for op in plan.ops:
            if op.scope:
                self._scoped.setdefault(op.scope, []).append(op)
        for op in plan.ops:
            if op.scope:
                continue
            method = self.EMITTERS.get(op.kind)
            if method is None:
                raise UnsupportedNode(
                    f'op種別 {op.kind} は未対応', node_id=op.node_id,
                    node_type=op.node_type, backend=self.name)
            getattr(self, method)(op)
            # state_init/ext_init はrecurrentの補助バッファ(state/ext)で、
            # 同じnode_idを共有するが「そのノードの意味論的な出力」では
            # ない(recurrent_execの出力とは別バッファ・別サイズ)。
            # expose_buffersはrecurrent_exec(または非recurrentノードの
            # 唯一のop)にだけ適用する。
            if (op.node_id in self.expose_buffers
                    and op.kind not in ('state_init', 'ext_init')
                    and op.output != plan.output_value):
                self._emit_public_buffer(op)
        self.packed_entry = ''
        if self.dense_batch and self.dense_packed_eligible():
            self.packed_entry = self._emit_dense_packed()
        return {
            f'{self.model_name}.h': self._header(),
            f'{self.model_name}.c': self._source(),
        }

    def _emit_preproc_tables(self) -> None:
        p = self.plan.preproc
        self._const_i32('mdl_preproc_desc', p['desc'])
        self._const_u32('mdl_preproc_qval', p['qval'])

    def _header(self) -> str:
        plan = self.plan
        meta = plan.meta
        dense_flat_note = ''
        dense_flat_define = ''
        if self.dense_flat_decision is not None:
            from .flatten_dense import describe_dense_flat
            applied = bool(self.dense_flat_decision['applied'])
            dense_flat_note = '\n * ' + describe_dense_flat(
                self.dense_flat_decision).replace('*/', '* /')
            dense_flat_define = (
                '\n/** 1 = dense の lut_layer 列を縮約済みの平坦ネットリスト'
                '(lut_flat)で実行する。\n'
                ' *  0 = 従来の bit-slice カーネル(codegen-c の '
                '--dense-flat 判定結果)。 */\n'
                f'#define MDL_DENSE_FLAT {1 if applied else 0}')
        public_decl = ''
        if self.public_decls:
            public_decl = (
                '\n/* ----: RDDLGN自己回帰生成向け公開API ---- */\n'
                + '\n'.join(self.public_decls) + '\n')
        packed_decl = ''
        if getattr(self, 'packed_entry', ''):
            packed_decl = f'''
/** dense のバッチ軸パック経路。n 枚をまとめて処理する。
 *  inputs は n*MDL_INPUT_WIDTH、scores は n*MDL_OUTPUT_WIDTH。
 *  **1ワード=32画像**で進むので、重みと添字表が32枚で1回しか読まれない。
 *  値は単枚経路と完全に一致する(テストで固定)。 */
#define MDL_DENSE_BATCH {self.dense_batch}
/** packed 経路が使う作業領域(呼び出し側が用意する。4バイト整列)。 */
#define MDL_DENSE_WORK_BYTES {self.dense_work_words * 4}
int mdl_infer_batch_packed(uint32_t *work, size_t bytes,
                           const uint8_t *inputs, int32_t *scores, int n);
'''
        labels = plan.labels
        guard = f'MDL_MODEL_{self.model_name.upper()}_H'
        label_decl = ''
        if labels:
            label_decl = (
                f'\n/** クラス名({len(labels)}個)。 */\n'
                f'extern const char *const mdl_labels[{len(labels)}];\n')
        p = plan.preproc
        c_in = p['channels'] if p else 0
        h_in = p['height'] if p else 0
        w_in = p['width'] if p else 0
        image_entry = ''
        if p is not None and plan.input_bitplane is not None:
            image_entry = f'''
/** 画像を直接受ける推論(前処理込み)。image は
 *  [MDL_IMAGE_CHANNELS][MDL_INPUT_HEIGHT][MDL_INPUT_WIDTH_PX] の u8(NCHW)。
 *  前処理は浮動小数点演算を一切使わない(整数比較とLUT参照のみ)。 */
int mdl_infer_image_ctx(mdl_ctx *ctx, const uint8_t *image, int32_t *scores);
/** `mdl_threads.h` のワーカへ渡すための void* 版。 */
int mdl_infer_image_ctx_v(void *ctx, const uint8_t *image, int32_t *scores);

/** 前処理が作る二値チャネル数(並列化の分割数の上限)。 */
#define MDL_PREPROC_CHANNELS {p['channels']}

/** 前処理を**チャネル範囲だけ**実行する。範囲が重ならなければ別スレッドから
 *  同時に呼んでよい。`mdl_threads.h` の `mdl_preproc_parallel` に
 *  そのまま渡せる形にしてある。
 *
 *  画像が大きいモデルでは推論時間の大半が前処理なので(detface は実測で
 *  76%)、**1枚のレイテンシを詰めたいならここを割るのが一番効く**:
 *
 *      mdl_preproc_parallel(mdl_preproc_range_v, &ctx, image,
 *                           MDL_PREPROC_CHANNELS, mdl_hw_threads());
 *      mdl_run_ctx(&ctx, scores);
 */
int mdl_preproc_range_v(void *ctx, const uint8_t *image, int start,
                        int stride);

/** 前処理済みのコンテキストからグラフだけを走らせる。 */
int mdl_run_ctx(mdl_ctx *ctx, int32_t *scores);
#ifndef MDL_NO_DEFAULT_CTX
int mdl_infer_image(const uint8_t *image, int32_t *scores);
#endif

/** 前処理が読む画像チャネル数。 */
#define MDL_IMAGE_CHANNELS {p['image_channels']}
'''
        return f'''/* 自動生成 — griffith-logic デプロイキット(md deploy codegen-c)
 * モデル: {meta.get('name', self.model_name)}
 * タスク: {meta.get('task_label') or meta.get('task') or '-'}
 * 手で編集しないこと。バンドルを更新して再生成する。{dense_flat_note}
 */
#ifndef {guard}
#define {guard}

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {{
#endif

/** 入力ビット数(前処理後。NCHW flatten で bit = ((ch*H)+y)*W+x)。 */
#define MDL_INPUT_WIDTH {plan.input_width}
/** 出力スコア数。 */
#define MDL_OUTPUT_WIDTH {plan.output_width}{dense_flat_define}
/** 前処理が作る二値チャネル数と画像サイズ(前処理記述子が無ければ0)。 */
#define MDL_INPUT_CHANNELS {c_in}
#define MDL_INPUT_HEIGHT {h_in}
#define MDL_INPUT_WIDTH_PX {w_in}
/** クラス数(ラベルが無ければ0)。 */
#define MDL_NUM_LABELS {len(labels)}
/** 1コンテキストが要る作業RAM。生存区間解析でバッファを再利用した後の値
 *  ({self.memory.naive_words * 4:,}B ぶんの値をここへ畳んである)。
 *  重み・命令列はここに含まれない — `static const` として .rodata に載り
 *  (MCUではフラッシュ)、**全コンテキストで共有**される。 */
#define MDL_CTX_BYTES {self.memory.arena_words * 4}
{label_decl}
/** 推論コンテキスト。作業領域は**呼び出し側が用意する**(ヒープ確保も
 *  スレッドライブラリも持ち込まないため)。コンテキストを複数作れば、
 *  同じモデルを複数スレッドで同時に走らせられる。 */
typedef struct {{
  uint32_t *work;
}} mdl_ctx;

/** コンテキストを初期化する。work は MDL_CTX_BYTES 以上、4バイト整列。
 *  戻り値 0 で成功、-1 で引数不正(サイズ不足・整列違反・NULL)。 */
int mdl_ctx_init(mdl_ctx *ctx, void *work, size_t bytes);

/** 推論(コンテキスト版)。input_bits は MDL_INPUT_WIDTH 個の 0/1、
 *  scores は MDL_OUTPUT_WIDTH 個の整数スコア。戻り値 0 で成功。
 *  **別々のコンテキストなら並行に呼んでよい**(共有するのは読み取り専用の
 *  重みだけ)。同じコンテキストを同時に使ってはいけない。 */
int mdl_infer_ctx(mdl_ctx *ctx, const uint8_t *input_bits, int32_t *scores);

/** `mdl_threads.h` のワーカへ渡すための void* 版(関数ポインタのキャストを
 *  避けるための薄い実体)。 */
int mdl_infer_ctx_v(void *ctx, const uint8_t *input_bits, int32_t *scores);

/** n 枚を続けて推論する。inputs は n*MDL_INPUT_WIDTH、scores は
 *  n*MDL_OUTPUT_WIDTH。
 *
 *  **これは逐次ループで、単体では速くならない**。CPU の1推論は既に
 *  「1命令で32画素」まで詰めてあり、枚数をまとめても命令数は変わらないため。
 *  CPUでスループットを上げたいときは `mdl_threads.h` の `mdl_run_pool` で
 *  コンテキストをコアぶん並べること(実測でPコア数までほぼ線形)。
 *  GPU側(`mdl_cuda_infer_batch` / `mdl_metal_infer_batch`)は事情が逆で、
 *  バッチがそのまま並列度になるので**まとめるほど速くなる**。 */
int mdl_infer_batch(mdl_ctx *ctx, const uint8_t *inputs, int32_t *scores,
                    int n);

{packed_decl}
#ifndef MDL_NO_DEFAULT_CTX
/** 推論(既定コンテキスト版)。単一スレッドならこれで足りる。
 *  作業領域を static で持つので `.bss` が MDL_CTX_BYTES 増える。
 *  自分でメモリを用意する場合は `-DMDL_NO_DEFAULT_CTX` で丸ごと省ける。
 *  **スレッドセーフではない**(既定コンテキストを共有するため)。 */
int mdl_infer(const uint8_t *input_bits, int32_t *scores);
#endif
{image_entry}
{public_decl}
#ifdef __cplusplus
}}
#endif

#endif /* {guard} */
'''

    def _source(self) -> str:
        plan = self.plan
        labels = plan.labels
        label_def = ''
        if labels:
            items = ','.join(
                '"' + str(v).replace('\\', '\\\\').replace('"', '\\"') + '"'
                for v in labels)
            label_def = (f'\nconst char *const mdl_labels[{len(labels)}] '
                         f'= {{{items}}};\n')
        p = plan.preproc
        shares = plan.input_bitplane is not None
        default_ctx = f'''
#ifndef MDL_NO_DEFAULT_CTX
/* 既定コンテキスト。単一スレッド用の便利口で、要らなければ
   -DMDL_NO_DEFAULT_CTX でこの .bss ごと消える。 */
static uint32_t mdl_default_work[{max(1, self.memory.arena_words)}];
static mdl_ctx mdl_default_ctx = {{ mdl_default_work }};

int mdl_infer(const uint8_t *input_bits, int32_t *scores) {{
  return mdl_infer_ctx(&mdl_default_ctx, input_bits, scores);
}}
'''
        if shares and p is not None:
            bp = self.buf(plan.input_bitplane)
            entry = f'''int mdl_infer_ctx(mdl_ctx *ctx, const uint8_t *input_bits,
                  int32_t *scores) {{
  if (ctx == NULL || ctx->work == NULL) return -1;
  /* 前処理の出力先と同じバッファへ直接パックする(中間の0/1列を作らない)。 */
  mdl_pack_planes(input_bits, {bp}, {p['channels']}, {p['height']},
                  {p['width']});
  return mdl_run(ctx, NULL, scores);
}}

int mdl_infer_image_ctx(mdl_ctx *ctx, const uint8_t *image, int32_t *scores) {{
  if (ctx == NULL || ctx->work == NULL) return -1;
  mdl_preproc_exec(image, {p['image_channels']}, {p['height']}, {p['width']},
                   mdl_preproc_desc, {p['channels']}, mdl_preproc_qval,
                   {bp});
  return mdl_run(ctx, NULL, scores);
}}

int mdl_infer_image_ctx_v(void *ctx, const uint8_t *image, int32_t *scores) {{
  return mdl_infer_image_ctx((mdl_ctx *)ctx, image, scores);
}}

int mdl_preproc_range_v(void *vctx, const uint8_t *image, int start,
                        int stride) {{
  mdl_ctx *ctx = (mdl_ctx *)vctx;
  if (ctx == NULL || ctx->work == NULL) return -1;
  mdl_preproc_exec_range(image, {p['image_channels']}, {p['height']},
                         {p['width']}, mdl_preproc_desc, start, stride,
                         {p['channels']}, mdl_preproc_qval, {bp});
  return 0;
}}

int mdl_run_ctx(mdl_ctx *ctx, int32_t *scores) {{
  if (ctx == NULL || ctx->work == NULL) return -1;
  return mdl_run(ctx, NULL, scores);
}}
{default_ctx}
int mdl_infer_image(const uint8_t *image, int32_t *scores) {{
  return mdl_infer_image_ctx(&mdl_default_ctx, image, scores);
}}
#endif
'''
        else:
            entry = f'''int mdl_infer_ctx(mdl_ctx *ctx, const uint8_t *input_bits,
                  int32_t *scores) {{
  if (ctx == NULL || ctx->work == NULL) return -1;
  return mdl_run(ctx, input_bits, scores);
}}
{default_ctx}#endif
'''
        entry += f'''
int mdl_infer_ctx_v(void *ctx, const uint8_t *input_bits, int32_t *scores) {{
  return mdl_infer_ctx((mdl_ctx *)ctx, input_bits, scores);
}}

int mdl_infer_batch(mdl_ctx *ctx, const uint8_t *inputs, int32_t *scores,
                    int n) {{
  int32_t k;
  for (k = 0; k < n; k++) {{
    const int rc = mdl_infer_ctx(ctx, inputs + (size_t)k * {plan.input_width},
                                 scores + (size_t)k * {plan.output_width});
    if (rc != 0) return rc;
  }}
  return 0;
}}
'''
        input_decl = ''
        # 1回だけの初期化は、いまは無い(conv の定数平面はスクラッチを
        # 共有した結果、毎回埋め直すほうが安いため)。将来必要になったときに
        # 備えて仕組みだけ残し、空なら関数ごと出さない。
        prepare_def = prepare_call = ''
        if self.prepare:
            prepare_def = f'''
/* 書き換わらないテーブルを1回だけ用意する。 */
static int mdl_prepared = 0;
static void mdl_prepare(void) {{
  if (mdl_prepared) return;
{chr(10).join(self.prepare)}
  mdl_prepared = 1;
}}
'''
            prepare_call = '  mdl_prepare();\n'
        # バッファを先に出す: concat のソース配列(const)がバッファを参照する。
        return f'''/* 自動生成 — griffith-logic デプロイキット。手で編集しないこと。 */
#include "{self.model_name}.h"
#include "mdl_kernels.h"

/* ---- 作業バッファ(実行時のヒープ確保はゼロ) ---- */
{input_decl}{chr(10).join(self.buf_lines)}

/* ---- 重み・命令列(コンパイル時定数) ---- */
{chr(10).join(self.data_lines)}
{label_def}{chr(10).join(self.fused_fns)}
{chr(10).join(self.fns)}
{chr(10).join(self.public_defs)}
{prepare_def}
int mdl_ctx_init(mdl_ctx *ctx, void *work, size_t bytes) {{
  if (ctx == NULL || work == NULL || bytes < MDL_CTX_BYTES) return -1;
  if (((uintptr_t)work & 3u) != 0u) return -1;   /* u32 のワード列として使う */
  ctx->work = (uint32_t *)work;
  return 0;
}}

static int mdl_run(mdl_ctx *ctx, const uint8_t *mdl_input, int32_t *scores) {{
  int32_t i;
  (void)i;
  (void)mdl_input;
{prepare_call}{chr(10).join(self.body)}
  return 0;
}}

{entry}{self.packed_entry}{self.packed_tf_entry}'''


def generate_c(bundle: dict, name: str = 'model', *,
               validate: bool = True, fuse: bool = False,
               dense_batch: int = 0,
               expose_recurrent: frozenset[str] | None = None,
               expose_buffers: frozenset[str] | None = None,
               packed_teacher_forcing: str | None = None,
               drop_folded_recurrents: frozenset[str] | None = None,
               dense_flat: str | None = None,
               dense_flat_min_reduction: float | None = None,
               dense_flat_decision: dict | None = None
               ) -> dict[str, str]:
    """バンドル → {ファイル名: 中身} のCソース。

    Args:
        dense_flat: ``'auto'|'on'|'off'`` を渡すと、C 生成の直前に
            `flatten_dense.resolve_dense_flat` で lut_flat 化の採否を決め
            (バンドル自体は変えず、メモリ上の複製だけを平坦化)、採否と削減率を
            model.h の先頭コメントと ``MDL_DENSE_FLAT`` マクロに残す。
            既定 None は従来どおり何もしない(バンドルのIRをそのまま生成。
            ライブラリ呼び出しの既定は不変。CLI の codegen-c/verify-c の
            既定が auto)。``dense_batch`` 指定時もパック経路が lut_flat を
            32 枚/語で評価する
        dense_flat_min_reduction: ``'auto'`` の採用閾値(実効ゲート削減率)。
            None なら `plan.DENSE_FLAT_AUTO_MIN_REDUCTION`
        dense_flat_decision: 呼び出し側が既に `resolve_dense_flat` を実行し、
            その結果のバンドルを ``bundle`` に渡している場合に、その decision
            を渡す(二重に平坦化せず model.h への記録だけを行う)。
            ``dense_flat`` と同時指定は不可
        fuse: ゲート命令列を直線コードへ展開する。真理値表が
            生成時に確定し中間値がレジスタに載るぶん速いが、生成コードは
            命令数に比例して大きくなる。インタプリタ経路は常に残してある
        expose_recurrent: このノードidのrecurrentは、step関数を非static
            公開し、`mdl_run_recurrent_<id>`/`mdl_ext_<id>_<sv>`/
            `mdl_state_<id>_<sv>`/`mdl_reset_state_<id>`を追加生成する。既定はNone(=何も公開しない、生成物は完全に既存互換)
        expose_buffers: このノードidの出力バッファへ`mdl_out_<id>`
            (直接アクセス用の取得関数)を追加生成する
        packed_teacher_forcing: この node_id の recurrent が「state 1個 +
            output_mode='stacked' + KIND_SCORES出力、tailがflat_concat→
            lut_dense(n_inputs=2)*→group_sumで閉じている」形なら、
            教師強制の時間バッチ化を追加entrypoint
            `mdl_infer_packed_tf_ctx`として生やす。既定はNone(既存互換)。
            `expose_recurrent`にも同じnode_idを含める必要がある。
            `drop_folded_recurrents`と同時指定するとValueError(下記)
        drop_folded_recurrents: このノードidのrecurrentは重み・step関数・
            mdl_run/mdl_inferからの呼び出しを一切生成しない(語彙
            折り畳みが完全に効いているときの配信サイズ削減用)。既定は
            None(=何も削らない、生成物は完全に既存互換)。**このオプションを
            使うと`mdl_infer`/`mdl_run`(教師強制)はもう正しく動かない**
            (該当recurrentの出力が埋まらないため) — 折り畳みハーネス
            (mdl_rddlgn_generate_greedy_folded)専用のビルド

    Raises:
        UnsupportedNode: 対応していないノードの形が含まれる場合
        ValueError: `packed_teacher_forcing`と`drop_folded_recurrents`を
            同時指定した場合(未対応の組み合わせ。前者が
            再利用するprefixを後者が欠落させるため)
    """
    if validate:
        from .bundle import validate_bundle
        for warning in validate_bundle(bundle):
            print(f'警告: {warning}')
    if dense_flat is not None and dense_flat_decision is not None:
        raise ValueError(
            'generate_c: dense_flat と dense_flat_decision は同時に指定できない')
    if dense_flat is not None:
        from .flatten_dense import resolve_dense_flat
        bundle, dense_flat_decision = resolve_dense_flat(
            bundle, mode=dense_flat, min_reduction=dense_flat_min_reduction,
            dense_batch=dense_batch)
    return CBackend(build_plan(bundle), model_name=name, fuse=fuse,
                    dense_batch=dense_batch,
                    dense_flat_decision=dense_flat_decision,
                    expose_recurrent=frozenset(expose_recurrent or ()),
                    expose_buffers=frozenset(expose_buffers or ()),
                    packed_teacher_forcing=packed_teacher_forcing,
                    drop_folded_recurrents=frozenset(
                        drop_folded_recurrents or ())).build()


#: RDDLGN(export/rddlgn_ir.py::export_rddlgn_ir)のIRが常に使う固定
#: ノードid(id_prefix='enc_'/'dec_'は定数)。自己回帰生成ハーネス
#: (runtime/c/src/mdl_rddlgn.c、モデル非依存の手書きコード)がこれらの
#: 名前のCシンボルを直接呼ぶ前提なので、コード生成側もこの名前で公開する。
RDDLGN_RECURRENT_NODES = frozenset({
    'enc_n_rec', 'enc_k_rec', 'dec_l_rec', 'dec_pm_rec'})
#: 'dec_pm_rec' はグラフの最終出力そのもの(呼び出し側の scores 配列)で
#: ctx内部バッファを持たないため、出力バッファ公開の対象には含めない。
#: 'enc_n_rec'(追加)はmdl_out_enc_n_rec/MDL_BUF_WIDTH_ENC_N_RECを
#: 公開する — 語彙折り畳み時、N-layers自体を評価せずこのバッファへ
#: `mdl_folded_n_lookup`の結果を直接書き込み、K-layers(enc_k_rec、
#: 無改造)にそのまま読ませるための書き込み先として使う
#: (`mdl_rddlgn_generate_greedy_folded`参照)。
RDDLGN_BUFFER_NODES = frozenset({'x_src', 'enc_n_rec', 'enc_k_rec', 'dec_l_rec'})


#: 完全に折り畳める2つのrecurrent(N-layers/L-layers)。
#: drop_folded_weights=Trueのときに落とす候補(両方のfolded_*_tableが
#: bundleに揃っているときだけ実際に落とす — 半端な削除はしない)。
RDDLGN_FOLDABLE_RECURRENT_NODES = frozenset({'enc_n_rec', 'dec_l_rec'})


def generate_rddlgn_c(bundle: dict, name: str = 'model', *,
                      drop_folded_weights: bool = False,
                      **kwargs) -> dict[str, str]:
    """RDDLGN(export/rddlgn_ir.py::export_rddlgn_ir)向けにgenerate_c()を
    呼ぶ薄いラッパ(drop_folded_weights追加)。

    runtime/c/src/mdl_rddlgn.c が要求する公開API(RDDLGN_RECURRENT_NODES/
    RDDLGN_BUFFER_NODESの節参照)を、generate_c()の汎用パラメータ
    (expose_recurrent/expose_buffers)へRDDLGN固有のノードid集合を渡すだけ
    で有効化する。generate_c自体はRDDLGNを一切知らない(汎用のまま)。

    Args:
        drop_folded_weights: True かつ bundle に folded_n_table と
            folded_l_table が**両方**あるときだけ、N-layers/L-layers の
            重み・step関数・mdl_infer/mdl_runからの呼び出しを生成物から
            落とす(配信サイズの実削減)。片方だけ折り畳まれている
            場合や、対象外(語彙が大きく自動判定で見送られた)場合は
            黙って通常ビルドにフォールバックする(教師強制やbits版
            greedy関数を壊さない安全側の既定)。**Trueにすると生成された
            model.c/hでは`mdl_infer`/`mdl_run`(教師強制)と
            `mdl_rddlgn_generate_greedy`(bits版)が使えなくなる**
            (後者はMDL_RDDLGN_WEIGHTS_DROPPEDマクロを見て宣言ごと自分を
            無効化するので、リンクエラーとして即座に分かる)。
    """
    kwargs.setdefault('expose_recurrent', RDDLGN_RECURRENT_NODES)
    kwargs.setdefault('expose_buffers', RDDLGN_BUFFER_NODES)
    if drop_folded_weights and bundle.get('folded_n_table') is not None \
            and bundle.get('folded_l_table') is not None:
        kwargs.setdefault('drop_folded_recurrents',
                          RDDLGN_FOLDABLE_RECURRENT_NODES)
    return generate_c(bundle, name, **kwargs)
