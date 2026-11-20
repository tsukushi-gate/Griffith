"""実行プラン → Metal(Apple GPU)ソース(プラン層へ分離)。

CUDAバックエンドと同じレイアウト・同じ命令列を使い、ホスト側APIだけ
Metal(`mdl_metal_*`)へ差し替える。op → 呼び出し文の対応表は
`deploy/gpubackend.py` と共有していて、ここに残るのは Metal 固有の4点だけ
(バッファ確保・アップロード・blitコピー・conv のパラメータ渡し)。

カーネルの式は runtime/gpu/mdl_gpu_kernels.h(CUDAと共有)、入口は
runtime/metal/mdl_metal_entry.metal。

シェーダは**実行時にソースからコンパイル**する(newLibraryWithSource:)。
オフラインの metal ツールチェーンを要求しないので、Xcodeのコンポーネントを
追加しなくてもビルド・実行できる。そのため生成した .mm には MSL ソースを
文字列として埋め込む。ソースは生成時に共有ヘッダと入口ファイルを読んで
連結するので、**チェックインされた重複は無い**(ずれようがない)。
"""

from pathlib import Path

from .gpubackend import (
    GpuBackend,
    auto_config_header_comment,
    resolve_generate_config,
)
from .plan import UnsupportedNode, build_plan  # noqa: F401

RUNTIME = Path(__file__).resolve().parents[2] / 'runtime'

#: runtime/metal/mdl_metal.h の同名マクロと同じ値。生成時にここで
#: 弾いておけば、ビルドして初めて範囲外に気づく事態を避けられる。
MDL_METAL_MAX_STREAMS = 8

MSL_PRELUDE = '''#include <metal_stdlib>
using namespace metal;

/* 共有カーネル(runtime/gpu/mdl_gpu_kernels.h)を Metal の
   アドレス空間修飾子に合わせて取り込むためのマクロ。 */
#define MDL_DEV
#define MDL_GLOBAL device
#define MDL_CONST constant
#define MDL_LOCAL thread
typedef unsigned char uint8_t;
typedef unsigned int uint32_t;
'''


def build_shader_source() -> str:
    """MSLソース = プレリュード + 共有カーネル + Metal用入口。"""
    shared = (RUNTIME / 'gpu' / 'mdl_gpu_kernels.h').read_text(encoding='utf-8')
    entry = (RUNTIME / 'metal' / 'mdl_metal_entry.metal').read_text(
        encoding='utf-8')
    # インクルードガードはMSLでは不要(1本に連結するため)
    shared = shared.replace('#ifndef MDL_GPU_KERNELS_H', '') \
                   .replace('#define MDL_GPU_KERNELS_H', '') \
                   .replace('#endif /* MDL_GPU_KERNELS_H */', '')
    return MSL_PRELUDE + shared + '\n' + entry


def _c_string_literal(text: str) -> str:
    """MSLソースをCの文字列リテラルへ(行ごとに分けて読みやすく保つ)。"""
    out = []
    for line in text.split('\n'):
        escaped = line.replace('\\', '\\\\').replace('"', '\\"')
        out.append(f'  "{escaped}\\n"')
    return '\n'.join(out)


class MetalBackend(GpuBackend):
    """プランからMetalソース(.mm)を組み立てる。"""

    name = 'metal'
    prefix = 'metal'
    #:: lut_flat 融合版の信号表(1バイト/信号)の上限 =
    #: Apple GPU の threadgroup メモリ上限(32KB)
    LUT_FLAT_FUSED_MAX_BYTES = 32768

    def __init__(self, plan, model_name: str = 'model_metal',
                conv_engine: str = 'split', n_streams: int = 1,
                conv_codegen: bool = False, dense_packed: bool = False):
        """Args:
            n_streams: 1(既定)なら従来どおり単一バッファ集合。2以上なら
                ストリーム・パイプライン化向けにバッファ集合を
                n_streams本複製し、`mdl_metal_infer_batch_submit`/`_collect`
                (画像入力があれば `mdl_metal_infer_image_batch_submit`/
                `_collect` も)を追加で生成する。既存の
                `mdl_metal_infer`/`mdl_metal_infer_batch`等は
                n_streams>1でも**そのまま同じ形**で残り、常にslot0を使う
                (追加のみ、既存経路は不変)。
                MDL_METAL_MAX_STREAMS(runtime/metal/mdl_metal.h)を超えては
                いけない。dense_packed=Trueとの併用はv1では未対応
                (下でValueError)。
            conv_codegen: 既定False(インタプリタ
                `mdl_k_conv_exec_split` を使う従来経路)。Trueにすると
                conv2dノードごとにtt即値+スロット静的展開のMSLを生成し、
                `mdl_metal_conv_exec_named` で起動する。
            dense_packed: 既定False(層ごとに1スレッド1ゲート、
                バッチはただの平坦展開のまま)。Trueで主経路
                (lut_dense列+末尾group_sum)を32サンプル/語のバッチ軸
                ビットプレーンへ詰め替えて評価する(算術強度をバッチ方向
                に約32倍に上げる)。適格性(batch>=32・全主経路opが
                lut_dense/group_sumのみ・conv/recurrent非混在)は
                `plan.py::dense_packed_eligible`がコンストラクタ内
                (`GpuBackend.__init__`)で即座に検証し、満たさなければ
                `ValueError`。n_streams>1との併用はv1では未対応。
        """
        if n_streams < 1:
            raise ValueError(f'n_streams={n_streams} は1以上でなければならない')
        if n_streams > MDL_METAL_MAX_STREAMS:
            raise ValueError(
                f'n_streams={n_streams} は MDL_METAL_MAX_STREAMS='
                f'{MDL_METAL_MAX_STREAMS}(runtime/metal/mdl_metal.h)を超える')
        if dense_packed and n_streams > 1:
            raise ValueError(
                'dense_packed と n_streams>1 の併用はv1では未対応'
                '(タスク指示のスコープ外。併用は後続タスク)')
        super().__init__(plan, model_name, conv_engine, conv_codegen,
                         dense_packed)
        self.n_streams = n_streams
        self.decl: list[str] = []
        self.init: list[str] = []
        self.free: list[str] = []
        #: モデル特化MSL(conv2dノードごとのデバイス関数+
        #: `mdl_k_conv_codegen_*` カーネル)。`_source()` がシェーダ本文の
        #: 末尾へ連結する。
        self.custom_kernels: list[str] = []

    # -- 原始操作 -----------------------------------------------------
    def _upload(self, name: str, values, ctype: str, fmt) -> str:
        items = ','.join(fmt(v) for v in values) or fmt(0)
        self.decl.append(f'static const {ctype} h_{name}[] = {{{items}}};')
        self.decl.append(f'static mdl_metal_buf d_{name} = NULL;')
        self.init.append(
            f'  d_{name} = mdl_metal_alloc(sizeof(h_{name}));\n'
            f'  if (!d_{name}) return -1;\n'
            f'  mdl_metal_upload(d_{name}, h_{name}, sizeof(h_{name}));')
        self.free.append(f'  mdl_metal_free(d_{name}); d_{name} = NULL;')
        return f'd_{name}'

    def up_i32(self, name, values):
        return self._upload(name, values, 'int', lambda v: str(int(v)))

    def up_u32(self, name, values):
        return self._upload(name, values, 'unsigned int',
                            lambda v: f'{int(v) & 0xffffffff}u')

    def up_i16(self, name, values):
        return self._upload(name, values, 'short', lambda v: str(int(v)))

    def up_u16(self, name, values):
        return self._upload(name, values, 'unsigned short',
                            lambda v: str(int(v) & 0xffff))

    def up_u8(self, name, values):
        return self._upload(name, values, 'unsigned char',
                            lambda v: str(int(v) & 0xff))

    def alloc(self, name: str, n_bytes: int, elem: str) -> str:
        if self.n_streams <= 1:
            self.decl.append(f'static mdl_metal_buf {name} = NULL;')
            self.init.append(
                f'  {name} = mdl_metal_alloc({max(4, n_bytes)});\n'
                f'  if (!{name}) return -1;')
            self.free.append(f'  mdl_metal_free({name}); {name} = NULL;')
            return name
        # ストリーム・パイプライン化: n_streams本ぶんの独立コピーを
        # 配列で確保する。呼び出し側は `{name}[slot]` で参照する
        # (buf()のオーバーライドが自動で `[slot]` を足す)。
        n = self.n_streams
        self.decl.append(f'static mdl_metal_buf {name}[{n}] = {{NULL}};')
        self.init.append(
            f'  for (int s_ = 0; s_ < {n}; s_++) {{\n'
            f'    {name}[s_] = mdl_metal_alloc({max(4, n_bytes)});\n'
            f'    if (!{name}[s_]) return -1;\n'
            f'  }}')
        self.free.append(
            f'  for (int s_ = 0; s_ < {n}; s_++) {{\n'
            f'    mdl_metal_free({name}[s_]); {name}[s_] = NULL;\n'
            f'  }}')
        return name

    def view(self, name: str, arena: str, off_words: int, elem: str) -> str:
        # Metal はバッファがハンドルなので、オフセット付きビューを作る。
        # ランタイム側で (バッファ, オフセット) の組にしてあるので、
        # 生成コードの呼び出しは1つも変わらない。
        if self.n_streams <= 1:
            self.decl.append(f'static mdl_metal_buf {name} = NULL;')
            self.init.append(
                f'  {name} = mdl_metal_view({arena}, {off_words * 4});\n'
                f'  if (!{name}) return -1;')
            # ビューはベースより先に解放する(所有していないので順序が要る)。
            self.free.insert(0, f'  mdl_metal_free({name}); {name} = NULL;')
            return name
        # streaming: `arena` もn_streams本の配列(`alloc`経由で確保済み)
        # なので、スロットごとに対応するアリーナへビューを張る。値ごとの
        # オフセット(off_words)はスロット間で共通(各スロットのアリーナは
        # 同じメモリ計画=同じレイアウト)。
        n = self.n_streams
        self.decl.append(f'static mdl_metal_buf {name}[{n}] = {{NULL}};')
        self.init.append(
            f'  for (int s_ = 0; s_ < {n}; s_++) {{\n'
            f'    {name}[s_] = mdl_metal_view({arena}[s_], {off_words * 4});\n'
            f'    if (!{name}[s_]) return -1;\n'
            f'  }}')
        self.free.insert(0,
            f'  for (int s_ = 0; s_ < {n}; s_++) {{\n'
            f'    mdl_metal_free({name}[s_]); {name}[s_] = NULL;\n'
            f'  }}')
        return name

    def buf(self, vid: int) -> str:
        base = super().buf(vid)
        if self.n_streams <= 1:
            return base
        # streaming: 呼び出し文の中では実行時スロットで引く。
        # (宣言名としてこのメソッドの戻り値を使う場所は `_raw_buf` 経由で
        # この分岐を迂回する — `declare_buffers` のオーバーライド参照)
        return f'{base}[slot]'

    def _raw_buf(self, vid: int) -> str:
        """buf()のstreamingサフィックス([slot])を付けない素の名前。
        宣言(alloc/viewへ渡す名前)を組み立てるときだけに使う
        (buf()自体は呼び出し文の中で参照として使われるので[slot]が要る)。
        """
        return GpuBackend.buf(self, vid)

    def declare_buffers(self) -> None:
        if self.n_streams <= 1:
            super().declare_buffers()
            return
        # GpuBackend.declare_buffers と同じロジックだが、宣言名には
        # `_raw_buf`(素の名前)を使う(buf()は[slot]付きの参照形を返すので
        # 宣言名には使えない)。
        mem = self.memory
        self.alloc('d_mdl_arena', max(4, mem.arena_words * 4), 'u32')
        for v in self.plan.values:
            name = self._raw_buf(v.id)
            if v.id not in mem.offsets:
                self.alloc(name, max(4, v.nbytes()), self.ELEM_OF.get(v.kind, 'i32'))
                continue
            self.view(name, 'd_mdl_arena', mem.offsets[v.id],
                     self.ELEM_OF.get(v.kind, 'i32'))

    def guard(self, call: str) -> str:
        return f'  MDL_TRY({call});'

    def copy_bytes_expr(self, dst, dst_off, src, src_off, n_bytes) -> None:
        self.body.append(
            f'  MDL_TRY(mdl_metal_copy({dst}, ({dst_off}), {src}, '
            f'({src_off}), {n_bytes}));')

    def copy_strided(self, dst, dst_off_expr, dst_stride, src, src_off_expr,
                     src_stride, n_bytes) -> None:
        self.body.append(
            f'  MDL_TRY(mdl_metal_copy_strided({dst}, ({dst_off_expr}), '
            f'{dst_stride}, {src}, ({src_off_expr}), {src_stride}, '
            f'{n_bytes}, {self.N}));')

    def emit_gate_program(self, op) -> None:
        if self.conv_engine == 'split':
            if self.conv_codegen:
                self._emit_gate_program_split_codegen(op)
            else:
                self._emit_gate_program_split(op)
        else:
            self._emit_gate_program_combined(op)

    def _emit_gate_program_combined(self, op) -> None:
        layout = op.attrs['layout']
        tag = self.uid('conv')
        t = self.conv_tables(layout, tag)
        scratch = self.scratch_of(op)
        # Metalの構造体にはデバイスポインタを置けないので、スカラーだけを
        # 構造体に入れ、配列は呼び出し引数で渡す。
        self.decl.append(
            f'static const mdl_metal_conv_params {tag}_p = {{'
            f'{layout.h}, {layout.w}, {layout.h_out}, {layout.w_out}, '
            f'{layout.s_h}, {layout.s_w}, {layout.p_h}, {layout.p_w}, '
            f'{layout.out_channels}, {layout.wpr_in}, {layout.wpr_out}, '
            f'{layout.tap_wpr}, {layout.tap_rows_words}, {layout.mask_out}u, '
            f'{1 if layout.use_fast_path else 0}, {layout.n_planes}, '
            f'{layout.plane_stride}, {layout.plane_rows}, {layout.pool_base}, '
            f'{layout.n_instrs}, 0, '
            f'{layout.in_plane_words}, {layout.out_plane_words}, '
            f'{layout.scratch_words}}};')
        # スクラッチは全convで1本を共有しているので、定数平面は毎回埋め直す。
        self.call(f'mdl_metal_conv_init(&{tag}_p, {scratch}, {self.N})')
        self.call(
            f'mdl_metal_conv(&{tag}_p, {self.buf(op.inputs[0])}, {scratch}, '
            f'{self.buf(op.output)}, {t["plane_c"]}, {t["plane_dy"]}, '
            f'{t["plane_dx"]}, {t["instr"]}, {t["oc_src"]}, {t["wstart"]}, '
            f'{t["woc"]}, {self.N})')

    def _emit_gate_program_split(self, op) -> None:
        """ocごとの命令列分割版(既定)。conv並列度を(batch,ワード位置)
        から(batch,oc,ワード位置)へ拡張する。バンドルの programs節(全oc
        結合)は変更せず、`build_conv_layout_split` がここで
        `op.attrs['params']`/`op.attrs['program']`(plan.py::_plan_conv2d が
        持たせた生の入力)から遅延構築する。"""
        from .layout import build_conv_layout_split

        layout = build_conv_layout_split(op.attrs['params'], op.attrs['program'])
        tag = self.uid('convs')
        t = self.conv_tables_split(layout, tag)
        scratch = self.scratch_of(op)
        # n_instrs/n_write0(combinedだけが使うフィールド)は分割版の
        # mdl_k_conv_exec_split が読まないので0のままでよい。
        self.decl.append(
            f'static const mdl_metal_conv_params {tag}_p = {{'
            f'{layout.h}, {layout.w}, {layout.h_out}, {layout.w_out}, '
            f'{layout.s_h}, {layout.s_w}, {layout.p_h}, {layout.p_w}, '
            f'{layout.out_channels}, {layout.wpr_in}, {layout.wpr_out}, '
            f'{layout.tap_wpr}, {layout.tap_rows_words}, {layout.mask_out}u, '
            f'{1 if layout.use_fast_path else 0}, {layout.n_planes}, '
            f'{layout.plane_stride}, {layout.plane_rows}, {layout.pool_base}, '
            f'0, 0, '
            f'{layout.in_plane_words}, {layout.out_plane_words}, '
            f'{layout.scratch_words}}};')
        self.call(f'mdl_metal_conv_init(&{tag}_p, {scratch}, {self.N})')
        self.call(
            f'mdl_metal_conv_split(&{tag}_p, {self.buf(op.inputs[0])}, '
            f'{scratch}, {self.buf(op.output)}, {t["plane_c"]}, '
            f'{t["plane_dy"]}, {t["plane_dx"]}, {t["oc_instr_start"]}, '
            f'{t["instr"]}, {t["oc_final"]}, {self.N})')

    def _emit_gate_program_split_codegen(self, op) -> None:
        """モデル特化MSL生成(`--conv-codegen`)。

        `_emit_gate_program_split` と同じ `ConvLayoutSplit` を使うが、
        命令列(instr_flat/oc_instr_start/oc_final_ref)を実行時バッファに
        せず、ocごとにtt即値+スロット静的展開の直線コードへ生成時に
        焼き込む(`convcodegen.py`)。タップ平面プールの構築
        (`mdl_metal_conv_planes`)だけは共通のインタプリタ経路と同じ関数を
        使い回す(幾何・タップはコード生成の影響を受けない共有データ)。

        最小実験(scripts/gpu_calib/gen_conv_codegen_ab.py、seg_convdlgn_k32
        の実ノード3本、5セット中央値、出力bit厳密一致確認済み): カーネル
        時間でinterp比1.39〜1.44倍(batch=1)、2.11〜2.56倍(batch大)。
        """
        from . import convcodegen as cg
        from .layout import build_conv_layout_split

        layout = build_conv_layout_split(op.attrs['params'], op.attrs['program'])
        tag = self.uid('convcg')
        t = self.conv_tables_planes(layout, tag)
        scratch = self.scratch_of(op)
        kernel_name = f'{cg.CODEGEN_KERNEL_PREFIX}{tag}'

        fn_defs = []
        for oc in range(layout.out_channels):
            body, final_expr = cg.oc_function_body(layout, oc)
            fn_defs.append(
                f'static inline unsigned int {kernel_name}_oc{oc}'
                f'(device const uint *sc, int i) {{\n'
                f'{body}  return {final_expr};\n}}')
        switch_body = cg.build_switch_body(
            layout, lambda oc: f'{kernel_name}_oc{oc}')
        self.custom_kernels.append('\n\n'.join(fn_defs) + f'''

kernel void {kernel_name}(
    device const uint *scratch [[buffer(0)]],
    device uint *out [[buffer(1)]],
    constant int &total [[buffer(2)]],
    uint gid [[thread_position_in_grid]]) {{
  if ((int)gid >= total) return;
  const int w = {layout.tap_rows_words};
  const int per_n = {layout.out_channels} * w;
  const int n = (int)gid / per_n;
  const int rem = (int)gid - n * per_n;
  const int oc = rem / w;
  const int i = rem - oc * w;
  const uint tail = ((i % {layout.wpr_out}) == {layout.wpr_out} - 1)
      ? {layout.mask_out}u : 0xffffffffu;
  unsigned int v = 0u;
{switch_body}
  out[(uint)n * (uint)per_n + (uint)oc * (uint)w + (uint)i] = v & tail;
}}
''')

        self.decl.append(
            f'static const mdl_metal_conv_params {tag}_p = {{'
            f'{layout.h}, {layout.w}, {layout.h_out}, {layout.w_out}, '
            f'{layout.s_h}, {layout.s_w}, {layout.p_h}, {layout.p_w}, '
            f'{layout.out_channels}, {layout.wpr_in}, {layout.wpr_out}, '
            f'{layout.tap_wpr}, {layout.tap_rows_words}, {layout.mask_out}u, '
            f'{1 if layout.use_fast_path else 0}, {layout.n_planes}, '
            f'{layout.plane_stride}, {layout.plane_rows}, {layout.pool_base}, '
            f'0, 0, '
            f'{layout.in_plane_words}, {layout.out_plane_words}, '
            f'{layout.scratch_words}}};')
        self.call(f'mdl_metal_conv_init(&{tag}_p, {scratch}, {self.N})')
        self.call(
            f'mdl_metal_conv_planes(&{tag}_p, {self.buf(op.inputs[0])}, '
            f'{scratch}, {t["plane_c"]}, {t["plane_dy"]}, {t["plane_dx"]}, '
            f'{self.N})')
        self.call(
            f'mdl_metal_conv_exec_named("{kernel_name}", {scratch}, '
            f'{self.buf(op.output)}, '
            f'{layout.out_channels} * {layout.tap_rows_words} * {self.N})')

    def emit_dense_fuse(self, op) -> None:
        """dense多層融合カーネル(全lut層+group_sumを1dispatchへ)。

        真理値表・添字表は従来どおりconstantバッファのままアップロードし、
        `densefuse.build_metal_kernel` が生成したMSLを`custom_kernels`
        (同じ仕組み)へ追加、`mdl_metal_dense_fuse_exec`
        (runtime/metal/mdl_metal.mm、新規追加)で名前引き起動する。
        """
        from . import densefuse as df

        a = op.attrs
        layers = a['layers']
        tag = self.uid('densefuse')
        kernel_name = f'{df.CODEGEN_KERNEL_PREFIX}{tag}'
        self.custom_kernels.append(df.build_metal_kernel(a, kernel_name))

        idx_names = []
        tt_names = []
        for i, layer in enumerate(layers):
            ln = i + 1
            if df.idx_ctype_metal(layer) == 'ushort':
                idx = self.up_u16(f'{tag}_idx{ln}', layer['indices'])
            else:
                idx = self.up_i32(f'{tag}_idx{ln}', layer['indices'])
            tt = self.up_u32(f'{tag}_tt{ln}', layer['truth_tables'])
            idx_names.append(idx)
            tt_names.append(tt)
        n_layers = len(layers)
        # ホスト側の mdl_metal_buf 配列(実体はmdl_model_init内で埋める。
        # 個々の d_..._idxN/d_..._ttN は上のup_u16/up_i32/up_u32が既に
        # self.initへalloc+upload文を積んでいるので、その直後に配列への
        # 代入文を積めば初期化順序は保証される)。
        idx_arr = f'{tag}_idx_arr'
        tt_arr = f'{tag}_tt_arr'
        self.decl.append(f'static mdl_metal_buf {idx_arr}[{n_layers}];')
        self.decl.append(f'static mdl_metal_buf {tt_arr}[{n_layers}];')
        for i, (idx, tt) in enumerate(zip(idx_names, tt_names, strict=True)):
            self.init.append(f'  {idx_arr}[{i}] = {idx};')
            self.init.append(f'  {tt_arr}[{i}] = {tt};')
        act_bytes = df.act_buf_bytes(layers)
        self.call(
            f'mdl_metal_dense_fuse_exec("{kernel_name}", '
            f'{self.buf(op.inputs[0])}, {self.buf(op.output)}, '
            f'{idx_arr}, {tt_arr}, {n_layers}, {act_bytes}, {self.N})')

    # -- 全体 ---------------------------------------------------------
    def build(self) -> dict[str, str]:
        self.declare_buffers()
        self.declare_preproc()
        if self.dense_packed:
            self.emit_dense_packed_body()
        else:
            self.emit_body()
        return {f'{self.model_name}.mm': self._source()}

    def _slotted(self, name: str, slot_expr: str) -> str:
        """n_streams>1なら `name[slot_expr]`、そうでなければ `name` のまま。
        `name` は素の(streaming時は配列の)識別子(`_raw_buf`/'d_mdl_image'/
        'd_mdl_input'相当)であること。"""
        if self.n_streams <= 1:
            return name
        return f'{name}[{slot_expr}]'

    def _source(self) -> str:
        plan = self.plan
        in_width, out_width = plan.input_width, plan.output_width
        p = plan.preproc
        streaming = self.n_streams > 1
        out_raw = self._raw_buf(plan.output_value) if streaming else \
            self.buf(plan.output_value)
        out_s = self._slotted(out_raw, 'slot')
        # mdl_model_run内(streaming時はslotが実引数)ではout_sを使う
        # (常に0決め打ちにすると、将来mdl_model_runへ非0スロットを渡した
        # ときに黙って間違ったスロットを読む落とし穴になる)。
        fetch = (f'  MDL_TRY(mdl_metal_sync());\n'
                 f'  mdl_metal_download(scores, {out_s if streaming else out_raw}, '
                 f'(size_t)n * {out_width} * sizeof(int));')
        stream_entry = ''
        if p is not None:
            n_img = p['image_channels'] * p['height'] * p['width']
            bp_raw = self._raw_buf(plan.input_bitplane) if streaming else \
                self.buf(plan.input_bitplane)
            bp0 = self._slotted(bp_raw, '0')
            bp_s = self._slotted(bp_raw, 'slot')
            img0 = self._slotted('d_mdl_image', '0')
            img_s = self._slotted('d_mdl_image', 'slot')
            in0 = self._slotted('d_mdl_input', '0')
            in_s = self._slotted('d_mdl_input', 'slot')
            image_entry = f'''
int mdl_metal_infer_image_batch(const unsigned char *image, int *scores,
                                int n) {{
  if (n < 1 || n > MDL_MAX_BATCH) return -1;
  if (mdl_model_init() != 0) return -1;
  mdl_metal_upload({img0}, image, (size_t)n * {n_img});
  MDL_TRY(mdl_metal_preproc({img0}, {p['image_channels']},
                            {p['height']}, {p['width']}, d_mdl_preproc_desc,
                            {p['channels']}, d_mdl_preproc_thr,
                            d_mdl_preproc_qval, {bp0}, n));
  return mdl_model_run(scores, n{', 0' if streaming else ''});
}}

int mdl_metal_infer_image(const unsigned char *image, int *scores) {{
  return mdl_metal_infer_image_batch(image, scores, 1);
}}
'''
            bits_entry = f'''
int mdl_metal_infer_batch(const unsigned char *input_bits, int *scores,
                          int n) {{
  if (n < 1 || n > MDL_MAX_BATCH) return -1;
  if (mdl_model_init() != 0) return -1;
  mdl_metal_upload({in0}, input_bits, (size_t)n * {in_width});
  MDL_TRY(mdl_metal_pack({in0}, {bp0}, {p['channels']} * n,
                         {p['height']}, {p['width']}));
  return mdl_model_run(scores, n{', 0' if streaming else ''});
}}
'''
            # streaming版のビット列入力submit(下のstream_entry追記部)は
            # 非streaming版が呼んでいる mdl_metal_pack({in0}, {bp0}, ...) を
            # mdl_model_encode の前に同様に呼ぶ必要がある(呼ばないと
            # bitplaneバッファが前回のスロット内容のまま=壊れる。
            # 潜在バグ、CUDA側に見つかった同型バグのMetal側修正)。
            pack_call = (
                f'  MDL_TRY(mdl_metal_pack({in_s}, {bp_s}, '
                f'{p["channels"]} * n, {p["height"]}, {p["width"]}));\n')
            if streaming:
                stream_entry += f'''
/* ---- ストリーム・パイプライン化(画像入力経路) ---- */
int mdl_metal_infer_image_batch_submit(const unsigned char *image, int n,
                                       int slot) {{
  if (n < 1 || n > MDL_MAX_BATCH) return -1;
  if (slot < 0 || slot >= {self.n_streams}) return -1;
  if (mdl_model_init() != 0) return -1;
  mdl_metal_upload({img_s}, image, (size_t)n * {n_img});
  MDL_TRY(mdl_metal_preproc({img_s}, {p['image_channels']},
                            {p['height']}, {p['width']}, d_mdl_preproc_desc,
                            {p['channels']}, d_mdl_preproc_thr,
                            d_mdl_preproc_qval, {bp_s}, n));
  MDL_TRY(mdl_model_encode(slot, n));
  return mdl_metal_commit_async(slot);
}}

int mdl_metal_infer_image_batch_collect(int *scores, int n, int slot) {{
  if (n < 1 || n > MDL_MAX_BATCH) return -1;
  if (slot < 0 || slot >= {self.n_streams}) return -1;
  MDL_TRY(mdl_metal_wait_slot(slot));
  mdl_metal_download(scores, {out_s}, (size_t)n * {out_width} * sizeof(int));
  return 0;
}}
'''
        else:
            in0 = self._slotted('d_mdl_input', '0')
            in_s = self._slotted('d_mdl_input', 'slot')
            bits_entry = f'''
int mdl_metal_infer_batch(const unsigned char *input_bits, int *scores,
                          int n) {{
  if (n < 1 || n > MDL_MAX_BATCH) return -1;
  if (mdl_model_init() != 0) return -1;
  mdl_metal_upload({in0}, input_bits, (size_t)n * {in_width});
  return mdl_model_run(scores, n{', 0' if streaming else ''});
}}
'''
            image_entry = ''
            # 前処理なしモデルはビットプレーンへのpack段そのものが無い
            # (グラフの入力バッファが直接ゲート木の入力になる)。
            pack_call = ''
        if streaming:
            stream_entry += f'''
/* ---- ストリーム・パイプライン化(ビット列入力経路) ---- */
int mdl_metal_infer_batch_submit(const unsigned char *input_bits, int n,
                                 int slot) {{
  if (n < 1 || n > MDL_MAX_BATCH) return -1;
  if (slot < 0 || slot >= {self.n_streams}) return -1;
  if (mdl_model_init() != 0) return -1;
  mdl_metal_upload({in_s}, input_bits, (size_t)n * {in_width});
{pack_call}  MDL_TRY(mdl_model_encode(slot, n));
  return mdl_metal_commit_async(slot);
}}

int mdl_metal_infer_batch_collect(int *scores, int n, int slot) {{
  if (n < 1 || n > MDL_MAX_BATCH) return -1;
  if (slot < 0 || slot >= {self.n_streams}) return -1;
  MDL_TRY(mdl_metal_wait_slot(slot));
  mdl_metal_download(scores, {out_s}, (size_t)n * {out_width} * sizeof(int));
  return 0;
}}
'''
        bits_entry += '''
int mdl_metal_infer(const unsigned char *input_bits, int *scores) {
  return mdl_metal_infer_batch(input_bits, scores, 1);
}
'''
        if streaming:
            run_fn = f'''static int mdl_model_encode(int slot, int n) {{
{chr(10).join(self.body)}
  return 0;
}}

static int mdl_model_run(int *scores, int n, int slot) {{
  MDL_TRY(mdl_model_encode(slot, n));
{fetch}
  return 0;
}}
'''
        else:
            run_fn = f'''static int mdl_model_run(int *scores, int n) {{
{chr(10).join(self.body)}
{fetch}
  return 0;
}}
'''
        shader_text = build_shader_source()
        if self.custom_kernels:
            # モデル特化カーネル(mdl_k_conv_codegen_*)を末尾へ連結。
            # "mdl_k_"接頭辞なので mdl_metal_init が自動でパイプライン登録する。
            shader_text += '\n' + '\n\n'.join(self.custom_kernels)
        shader = _c_string_literal(shader_text)
        return f'''/* 自動生成 — griffith-logic デプロイキット(md deploy codegen-metal)
 * 手で編集しないこと。バンドルを更新して再生成する。 */
#include "mdl_metal.h"
#include <stdio.h>

#define MDL_INPUT_WIDTH {in_width}
#define MDL_OUTPUT_WIDTH {out_width}
/* 同時に流せる最大枚数(バッファはこのぶん確保済み)。実行時の枚数は
   1..MDL_MAX_BATCH で可変。conv_exec のスレッド数は「画素数/32」しか
   ないので、深い層ではバッチがそのまま並列度になる。 */
#define MDL_MAX_BATCH {plan.batch}
#define MDL_TRY(expr) do {{ if ((expr) != 0) return -1; }} while (0)

/* ---- シェーダ(実行時にコンパイルする。生成時に
       runtime/gpu/mdl_gpu_kernels.h と runtime/metal/mdl_metal_entry.metal を
       連結したもの) ---- */
static const char *MDL_SHADER_SRC =
{shader};

/* ---- 定数テーブルとバッファ ---- */
{chr(10).join(self.decl)}

static int mdl_model_ready = 0;

extern "C" int mdl_model_init(void) {{
  if (mdl_model_ready) return 0;
  if (mdl_metal_init(MDL_SHADER_SRC) != 0) return -1;
{chr(10).join(self.init)}
  mdl_model_ready = 1;
  return 0;
}}

extern "C" void mdl_model_shutdown(void) {{
  if (!mdl_model_ready) return;
{chr(10).join(self.free)}
  mdl_metal_shutdown();
  mdl_model_ready = 0;
}}

{run_fn}
extern "C" {{
{bits_entry}{image_entry}{stream_entry}
int mdl_metal_input_width(void) {{ return MDL_INPUT_WIDTH; }}
int mdl_metal_output_width(void) {{ return MDL_OUTPUT_WIDTH; }}
int mdl_metal_max_batch(void) {{ return MDL_MAX_BATCH; }}
{f'int mdl_metal_n_streams(void) {{ return {self.n_streams}; }}' if streaming else ''}
}}
'''


#: 旧名(スクリプト互換)
MetalModelBuilder = MetalBackend


def generate_metal(bundle: dict, name: str = 'model_metal', *,
                   batch: int = 1, conv_engine: str | None = None,
                   n_streams: int = 1,
                   conv_codegen: bool = False,
                   dense_fuse: bool | None = None,
                   dense_packed: bool | None = None,
                   auto: str = 'measured',
                   lut_flat_mode: str = 'auto') -> dict[str, str]:
    """バンドル → {ファイル名: 中身} のMetalソース。

    Args:
        batch: 同時に流せる最大枚数。デバイスメモリはこの倍数になる
        conv_engine: 'split'または'combined'(旧経路)。既定`None`は
            `auto`に選択を委ねる(下記`auto`参照)。新旧の出力一致
            テスト以外で明示指定する必要は通常ない。
        n_streams: 1(既定)なら従来どおり。2以上でストリーム・
            パイプライン化向けの `mdl_metal_infer_batch_submit`/
            `_collect`(画像入力があれば image 版も)を追加生成する。
            デバイスメモリは `batch * n_streams` 倍になる(バッファ集合を
            n_streams本複製するため)。dense_packed=Trueとの併用は
            v1では未対応(下でValueError)。も自動ONにしない
            (モデル依存で悪化する実測があるためopt-inのまま、
            `docs/guides/deploy-optimization.md`参照)。
        conv_codegen: 既定False(インタプリタのまま)。Trueで
            conv2dノードごとにtt即値+スロット静的展開のモデル特化MSLを
            生成する(`conv_engine='split'` 前提)。も自動ONにしない
            (e2e改善がほぼ無い割にコードサイズが1.78〜6.65倍になる実測の
            ため)。
        dense_fuse: Trueで連続するlut_dense列+末尾group_sumを1つの
            融合カーネルへまとめる(`plan.py::fuse_dense_layers`。活性が
            threadgroupメモリの32KBに収まる鎖だけが対象、収まらない鎖は
            自動的に従来経路のまま生成される)。既定`None`は`auto`に
            選択を委ねる(`batch>=32`かつ融合可能な鎖があれば自動
            ON、`densefuse.py`のdocstring参照)。dense_packed=Trueとの
            併用はv1では未対応(下でValueError)。
        dense_packed: Trueで主経路(lut_dense列+末尾group_sum)を32
            サンプル/語のバッチ軸ビットプレーンへ詰め替えて評価する
            (算術強度をバッチ方向に約32倍に上げる、CPU側9倍実測と
            同じ変換)。適格性(batch>=32・全主経路opがlut_dense/
            group_sumのみ・非画像入力・conv/recurrent非混在)を満たさ
            なければ`ValueError`(明確な理由つき)。既定`None`は`auto`に
            選択を委ねる(dense_fuseが自動ONにならず・適格・
            `batch>=128`なら自動ON)。`dense_fuse=True`/`n_streams>1`との
            併用はv1では未対応(併用は後続タスク)。
        auto: `'measured'`(既定)なら`conv_engine`/`dense_fuse`/
            `dense_packed`のうち`None`のまま(=未指定)の引数を
            `plan.py::resolve_auto_config`の実測済み判定表で自動選択する
            (詳細・実測値は`docs/guides/deploy-optimization.md`)。
            `'off'`なら自動選択を無効化し、未指定の引数は旧来の既定値
            (`conv_engine='split'`/`dense_fuse=False`/
            `dense_packed=False`、旧版の挙動)のまま生成する。
            明示的に値を指定した引数は`auto`の値によらず常にそのまま
            使われる(明示が勝つ)。選択結果と理由は生成ソース冒頭の
            コメントと標準出力の1行サマリに出る。
    """
    plan, conv_engine, dense_fuse, dense_packed, decision = resolve_generate_config(
        bundle, batch=batch, backend='metal', conv_engine=conv_engine,
        dense_fuse=dense_fuse, dense_packed=dense_packed, auto=auto)
    if auto == 'measured':
        print(f'md deploy codegen-metal: auto-config conv_engine={conv_engine!r} '
             f'dense_fuse={dense_fuse} dense_packed={dense_packed}')
    backend = MetalBackend(
        plan, model_name=name, conv_engine=conv_engine, conv_codegen=conv_codegen,
        n_streams=n_streams, dense_packed=dense_packed)
    if lut_flat_mode not in backend.LUT_FLAT_MODES:
        raise ValueError(f'lut_flat_mode={lut_flat_mode!r} は未対応'
                         f'(選べるのは {sorted(backend.LUT_FLAT_MODES)})')
    backend.lut_flat_mode = lut_flat_mode
    files = backend.build()
    key = f'{name}.mm'
    files[key] = auto_config_header_comment(
        'metal', auto, conv_engine, dense_fuse, dense_packed, decision) + files[key]
    return files
