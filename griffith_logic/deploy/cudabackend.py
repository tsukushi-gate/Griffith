"""実行プラン → CUDAソース(プラン層へ分離)。

C バックエンドとまったく同じレイアウト・同じ命令列を使い、実行だけを GPU
カーネルへ振り替える。グラフの解析・縮約・メモリ割り付けは `deploy/plan.py`
で終わっているので、ここがやるのは

    ・定数テーブルをデバイスメモリへ載せるコードを吐く
    ・op 順に mdl_cuda_* を呼ぶドライバを吐く

だけ。カーネルの式は runtime/gpu/mdl_gpu_kernels.h に1本化してあり、
CPU/CUDA/Metal で整数出力が一致する。

前処理の EDGE/CHROMA/SAT は、CPU 版が実行時に作っているしきい値表を
**生成時に焼き込む**(デバイス側から64bit演算を消すため)。
"""

from .gpubackend import (
    GpuBackend,
    auto_config_header_comment,
    resolve_generate_config,
)
from .plan import (  # noqa: F401
    CUDA_DENSE_FUSE_MAX_SHARED_BYTES,
    DENSE_FUSE_MAX_SHARED_BYTES,
    UnsupportedNode,
    build_plan,
    resolve_auto_config,
)
from .preproc_desc import threshold_tables

#: 旧名(外部スクリプト互換)。実体は preproc_desc.py へ移した。
_threshold_tables = threshold_tables

_CTYPE = {'u32': 'unsigned int', 'u8': 'unsigned char', 'i32': 'int'}

#: runtime/cuda/mdl_cuda.h の同名マクロと同じ値(Metal版
#: MDL_METAL_MAX_STREAMS のミラー)。生成時にここで弾いておけば、ビルドして
#: 初めて範囲外に気づく事態を避けられる。
MDL_CUDA_MAX_STREAMS = 8

#: CUDA_DENSE_FUSE_MAX_SHARED_BYTES の定義本体は `plan.py` に移した
#: (`resolve_auto_config`がバックエンド非依存のplan.pyからこの値を
#: 参照する必要があり、ここからimportすると循環importになるため)。
#: ここでは外部の既存import(`from griffith_logic.deploy.cudabackend
#: import CUDA_DENSE_FUSE_MAX_SHARED_BYTES`、scripts/verify_cuda_stream_
#: remote.py等)を壊さないよう上のimport文で再exportしている。


class CudaBackend(GpuBackend):
    """プランからCUDAソースを組み立てる。"""

    name = 'cuda'
    prefix = 'cuda'
    #:: lut_flat 融合版の信号表(1バイト/信号)の上限 =
    #: CUDA の動的共有メモリの既定上限(48KB、opt-in 無し)
    LUT_FLAT_FUSED_MAX_BYTES = 49152

    def __init__(self, plan, model_name: str = 'model_cuda',
                conv_engine: str = 'split', conv_codegen: bool = False,
                dense_packed: bool = False, n_streams: int = 1):
        """Args:
            n_streams: 1(既定)なら従来どおり単一バッファ集合。2以上なら
                ストリーム・パイプライン化(Metal版ミラー)向けに
                バッファ集合をn_streams本複製し、
                `mdl_cuda_infer_batch_submit`/`_collect`(画像入力があれば
                `mdl_cuda_infer_image_batch_submit`/`_collect` も)を追加
                生成する。既存の `mdl_cuda_infer`/`mdl_cuda_infer_batch` 等は
                n_streams>1でも**そのまま同じ形**で残り、常にslot0を使う
                (追加のみ、既存経路は不変)。MDL_CUDA_MAX_STREAMS
                (runtime/cuda/mdl_cuda.h)を超えてはいけない。
                dense_packed=Trueとの併用はv1では未対応(Metal版と同じ
                制約、下でValueError)。
        """
        if n_streams < 1:
            raise ValueError(f'n_streams={n_streams} は1以上でなければならない')
        if n_streams > MDL_CUDA_MAX_STREAMS:
            raise ValueError(
                f'n_streams={n_streams} は MDL_CUDA_MAX_STREAMS='
                f'{MDL_CUDA_MAX_STREAMS}(runtime/cuda/mdl_cuda.h)を超える')
        if dense_packed and n_streams > 1:
            raise ValueError(
                'dense_packed と n_streams>1 の併用はv1では未対応'
                '(Metal版同じスコープ外。併用は後続タスク)')
        super().__init__(plan, model_name, conv_engine, conv_codegen,
                         dense_packed)
        self.n_streams = n_streams
        self.host_data: list[str] = []   # ホスト側 static const テーブル
        self.dev_decl: list[str] = []    # デバイスポインタの宣言
        self.dev_init: list[str] = []    # 確保 + 転送
        self.dev_free: list[str] = []
        #: モデル特化 `__global__` カーネル(conv2dノードごとの
        #: tt即値+スロット静的展開)。CUDAはAOTコンパイルなので、Metalと
        #: 違い生成した.cu内に直接定義して `<<<>>>` で起動すればよい
        #: (名前引きの実行時ディスパッチは不要)。
        self.custom_kernels: list[str] = []

    # -- 原始操作 -----------------------------------------------------
    def _upload(self, name: str, values, ctype: str, fmt) -> str:
        items = ','.join(fmt(v) for v in values) or fmt(0)
        self.host_data.append(f'static const {ctype} h_{name}[] = {{{items}}};')
        self.dev_decl.append(f'static {ctype} *d_{name} = 0;')
        self.dev_init.append(
            f'  MDL_TRY(cudaMalloc((void**)&d_{name}, sizeof(h_{name})));\n'
            f'  MDL_TRY(cudaMemcpy(d_{name}, h_{name}, sizeof(h_{name}), '
            f'cudaMemcpyHostToDevice));')
        self.dev_free.append(f'  cudaFree(d_{name}); d_{name} = 0;')
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
        ctype = _CTYPE[elem]
        if self.n_streams <= 1:
            self.dev_decl.append(f'static {ctype} *{name} = 0;')
            self.dev_init.append(
                f'  MDL_TRY(cudaMalloc((void**)&{name}, {max(1, n_bytes)}));')
            self.dev_free.append(f'  cudaFree({name}); {name} = 0;')
            return name
        # ストリーム・パイプライン化(Metal版alloc()のミラー):
        # n_streams本ぶんの独立コピーを配列で確保する。呼び出し側は
        # `{name}[slot]` で参照する(buf()のオーバーライドが自動で
        # `[slot]` を足す)。
        n = self.n_streams
        self.dev_decl.append(f'static {ctype} *{name}[{n}] = {{0}};')
        self.dev_init.append(
            f'  for (int s_ = 0; s_ < {n}; s_++) {{\n'
            f'    MDL_TRY(cudaMalloc((void**)&{name}[s_], '
            f'{max(1, n_bytes)}));\n'
            f'  }}')
        self.dev_free.append(
            f'  for (int s_ = 0; s_ < {n}; s_++) {{\n'
            f'    cudaFree({name}[s_]); {name}[s_] = 0;\n'
            f'  }}')
        return name

    def view(self, name: str, arena: str, off_words: int, elem: str) -> str:
        # CUDA はただのポインタなので、アリーナ先頭からのオフセットで足りる。
        cast = '' if elem == 'u32' else f'({_CTYPE[elem]} *)'
        if self.n_streams <= 1:
            self.dev_decl.append(
                f'#define {name} ({cast}({arena} + {off_words}))')
            return name
        # streaming: `#define` の文字列置換ではスロットごとの配列を表せない
        # ので、Metal版view()(mdl_metal_viewで実ハンドルを作る)のミラーと
        # して、`arena`(streaming時は配列)の各スロットから実ポインタを
        # 計算してname[n_streams]へ格納する(値そのものは同じアリーナ+
        # オフセットの単純なポインタ演算、mdl_cuda_init内で1回だけ実行)。
        ctype = _CTYPE[elem]
        n = self.n_streams
        self.dev_decl.append(f'static {ctype} *{name}[{n}] = {{0}};')
        self.dev_init.append(
            f'  for (int s_ = 0; s_ < {n}; s_++) {{\n'
            f'    {name}[s_] = {cast}({arena}[s_] + {off_words});\n'
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
                self.alloc(name, max(4, v.nbytes()),
                          self.ELEM_OF.get(v.kind, 'i32'))
                continue
            self.view(name, 'd_mdl_arena', mem.offsets[v.id],
                     self.ELEM_OF.get(v.kind, 'i32'))

    def guard(self, call: str) -> str:
        return f'  MDL_TRY_INT({call});'

    def copy_bytes_expr(self, dst, dst_off, src, src_off, n_bytes) -> None:
        # 従来の同期cudaMemcpy(D2D)から、現在選択中のストリームへ
        # 積むcudaMemcpyAsyncへ変更。streams_init未実行(=n_streams<=1、
        # 従来経路)では mdl_cuda_current_stream() は常に0(レガシー
        # デフォルトストリーム)を返すので、非ストリーミング経路の挙動は
        # 変わらない(cudaMemcpyAsyncはstream=0でもホスト同期の意味論が
        # cudaMemcpyと実質同じ ─ 後続のmdl_cuda_sync/同期cudaMemcpyが
        # 完了を保証する)。
        self.body.append(
            f'  MDL_TRY(cudaMemcpyAsync((char*){dst} + ({dst_off}), '
            f'(const char*){src} + ({src_off}), {n_bytes}, '
            f'cudaMemcpyDeviceToDevice, mdl_cuda_current_stream()));')

    def copy_strided(self, dst, dst_off_expr, dst_stride, src, src_off_expr,
                     src_stride, n_bytes) -> None:
        self.body.append(
            f'  MDL_TRY_INT(mdl_cuda_copy_strided('
            f'(const unsigned char*){src}, (unsigned char*){dst}, '
            f'({dst_off_expr}), {dst_stride}, ({src_off_expr}), '
            f'{src_stride}, {n_bytes}, {self.N}));')

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
        self.dev_decl.append(f'static mdl_cuda_conv_layout {tag}_layout;')
        self.dev_init.append(f'''  {tag}_layout.h = {layout.h};
  {tag}_layout.w = {layout.w};
  {tag}_layout.h_out = {layout.h_out};
  {tag}_layout.w_out = {layout.w_out};
  {tag}_layout.s_h = {layout.s_h}; {tag}_layout.s_w = {layout.s_w};
  {tag}_layout.p_h = {layout.p_h}; {tag}_layout.p_w = {layout.p_w};
  {tag}_layout.out_channels = {layout.out_channels};
  {tag}_layout.wpr_in = {layout.wpr_in};
  {tag}_layout.wpr_out = {layout.wpr_out};
  {tag}_layout.tap_wpr = {layout.tap_wpr};
  {tag}_layout.tap_rows_words = {layout.tap_rows_words};
  {tag}_layout.mask_out = {layout.mask_out}u;
  {tag}_layout.use_fast_path = {1 if layout.use_fast_path else 0};
  {tag}_layout.plane_c = {t['plane_c']};
  {tag}_layout.plane_dy = {t['plane_dy']};
  {tag}_layout.plane_dx = {t['plane_dx']};
  {tag}_layout.n_planes = {layout.n_planes};
  {tag}_layout.plane_stride = {layout.plane_stride};
  {tag}_layout.plane_rows = {layout.plane_rows};
  {tag}_layout.in_plane_words = {layout.in_plane_words};
  {tag}_layout.out_plane_words = {layout.out_plane_words};
  {tag}_layout.instr_flat = {t['instr']};
  {tag}_layout.n_instrs = {layout.n_instrs};
  {tag}_layout.oc_src_off = {t['oc_src']};
  {tag}_layout.write_start = {t['wstart']};
  {tag}_layout.write_oc = {t['woc']};
  {tag}_layout.pool_base = {layout.pool_base};
  {tag}_layout.reg_base = {layout.reg_base};
  {tag}_layout.scratch_words = {layout.scratch_words};''')
        # スクラッチは全convで1本を共有しているので、定数平面は毎回
        # 埋め直す(埋めるのは 2*tap_rows_words ワードだけ)。
        self.call(f'mdl_cuda_conv_init(&{tag}_layout, {scratch}, {self.N})')
        self.call(f'mdl_cuda_conv(&{tag}_layout, {self.buf(op.inputs[0])}, '
                  f'{scratch}, {self.buf(op.output)}, {self.N})')

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
        self.dev_decl.append(f'static mdl_cuda_conv_layout {tag}_layout;')
        # instr_flat/oc_src_off/write_start/write_oc/reg_base/n_instrsは
        # 分割版の mdl_k_conv_exec_split が読まないフィールド(NULL/0でよい)。
        self.dev_init.append(f'''  {tag}_layout.h = {layout.h};
  {tag}_layout.w = {layout.w};
  {tag}_layout.h_out = {layout.h_out};
  {tag}_layout.w_out = {layout.w_out};
  {tag}_layout.s_h = {layout.s_h}; {tag}_layout.s_w = {layout.s_w};
  {tag}_layout.p_h = {layout.p_h}; {tag}_layout.p_w = {layout.p_w};
  {tag}_layout.out_channels = {layout.out_channels};
  {tag}_layout.wpr_in = {layout.wpr_in};
  {tag}_layout.wpr_out = {layout.wpr_out};
  {tag}_layout.tap_wpr = {layout.tap_wpr};
  {tag}_layout.tap_rows_words = {layout.tap_rows_words};
  {tag}_layout.mask_out = {layout.mask_out}u;
  {tag}_layout.use_fast_path = {1 if layout.use_fast_path else 0};
  {tag}_layout.plane_c = {t['plane_c']};
  {tag}_layout.plane_dy = {t['plane_dy']};
  {tag}_layout.plane_dx = {t['plane_dx']};
  {tag}_layout.n_planes = {layout.n_planes};
  {tag}_layout.plane_stride = {layout.plane_stride};
  {tag}_layout.plane_rows = {layout.plane_rows};
  {tag}_layout.in_plane_words = {layout.in_plane_words};
  {tag}_layout.out_plane_words = {layout.out_plane_words};
  {tag}_layout.instr_flat = 0;
  {tag}_layout.n_instrs = 0;
  {tag}_layout.oc_src_off = 0;
  {tag}_layout.write_start = 0;
  {tag}_layout.write_oc = 0;
  {tag}_layout.pool_base = {layout.pool_base};
  {tag}_layout.reg_base = 0;
  {tag}_layout.scratch_words = {layout.scratch_words};''')
        self.call(f'mdl_cuda_conv_init(&{tag}_layout, {scratch}, {self.N})')
        self.call(
            f'mdl_cuda_conv_split(&{tag}_layout, {self.buf(op.inputs[0])}, '
            f'{scratch}, {self.buf(op.output)}, {t["oc_instr_start"]}, '
            f'{t["instr"]}, {t["oc_final"]}, {self.N})')

    def _emit_gate_program_split_codegen(self, op) -> None:
        """モデル特化CUDAカーネル生成(`--conv-codegen`)。

        Metal版(`metalbackend.py::_emit_gate_program_split_codegen`)と
        同じ `ConvLayoutSplit`/`convcodegen.py` を使う対称実装。CUDAは
        AOTコンパイルなので、生成した `__global__` カーネルを直接
        `<<<>>>` で起動できる(Metalの `mdl_metal_conv_exec_named` に
        相当する名前引き実行時ディスパッチは不要)。タップ平面プールの
        構築だけは共有ランタイムの `mdl_cuda_conv_planes` を使い回す。
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
                f'__device__ __forceinline__ unsigned int {kernel_name}_oc{oc}'
                f'(const unsigned int *sc, int i) {{\n'
                f'{body}  return {final_expr};\n}}')
        switch_body = cg.build_switch_body(
            layout, lambda oc: f'{kernel_name}_oc{oc}')
        self.custom_kernels.append('\n\n'.join(fn_defs) + f'''

__global__ void {kernel_name}(
    const unsigned int *scratch, unsigned int *out, int total) {{
  const int gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid >= total) return;
  const int w = {layout.tap_rows_words};
  const int per_n = {layout.out_channels} * w;
  const int n = gid / per_n;
  const int rem = gid - n * per_n;
  const int oc = rem / w;
  const int i = rem - oc * w;
  const unsigned int tail = ((i % {layout.wpr_out}) == {layout.wpr_out} - 1)
      ? {layout.mask_out}u : 0xffffffffu;
  unsigned int v = 0u;
{switch_body}
  out[(unsigned)n * (unsigned)per_n + (unsigned)oc * (unsigned)w
      + (unsigned)i] = v & tail;
}}
''')

        self.dev_decl.append(f'static mdl_cuda_conv_layout {tag}_layout;')
        self.dev_init.append(f'''  {tag}_layout.h = {layout.h};
  {tag}_layout.w = {layout.w};
  {tag}_layout.h_out = {layout.h_out};
  {tag}_layout.w_out = {layout.w_out};
  {tag}_layout.s_h = {layout.s_h}; {tag}_layout.s_w = {layout.s_w};
  {tag}_layout.p_h = {layout.p_h}; {tag}_layout.p_w = {layout.p_w};
  {tag}_layout.out_channels = {layout.out_channels};
  {tag}_layout.wpr_in = {layout.wpr_in};
  {tag}_layout.wpr_out = {layout.wpr_out};
  {tag}_layout.tap_wpr = {layout.tap_wpr};
  {tag}_layout.tap_rows_words = {layout.tap_rows_words};
  {tag}_layout.mask_out = {layout.mask_out}u;
  {tag}_layout.use_fast_path = {1 if layout.use_fast_path else 0};
  {tag}_layout.plane_c = {t['plane_c']};
  {tag}_layout.plane_dy = {t['plane_dy']};
  {tag}_layout.plane_dx = {t['plane_dx']};
  {tag}_layout.n_planes = {layout.n_planes};
  {tag}_layout.plane_stride = {layout.plane_stride};
  {tag}_layout.plane_rows = {layout.plane_rows};
  {tag}_layout.in_plane_words = {layout.in_plane_words};
  {tag}_layout.out_plane_words = {layout.out_plane_words};
  {tag}_layout.instr_flat = 0;
  {tag}_layout.n_instrs = 0;
  {tag}_layout.oc_src_off = 0;
  {tag}_layout.write_start = 0;
  {tag}_layout.write_oc = 0;
  {tag}_layout.pool_base = {layout.pool_base};
  {tag}_layout.reg_base = 0;
  {tag}_layout.scratch_words = {layout.scratch_words};''')
        self.call(f'mdl_cuda_conv_init(&{tag}_layout, {scratch}, {self.N})')
        self.call(f'mdl_cuda_conv_planes(&{tag}_layout, '
                  f'{self.buf(op.inputs[0])}, {scratch}, {self.N})')
        exec_total = (f'{layout.out_channels} * {layout.tap_rows_words} * '
                     f'{self.N}')
        self.body.append(
            f'  {{\n'
            f'    const int total_ = {exec_total};\n'
            f'    {kernel_name}<<<mdl_grid_(total_), MDL_BLOCK_, 0, '
            f'mdl_cuda_current_stream()>>>(\n'
            f'        {scratch}, {self.buf(op.output)}, total_);\n'
            f'    MDL_TRY(cudaGetLastError());\n'
            f'  }}')

    def emit_dense_fuse(self, op) -> None:
        """dense多層融合カーネル(全lut層+group_sumを1起動へ)。

        Metal版と対称の実装(`densefuse.build_cuda_kernel`が同じ
        `_layer_lines`共有本体からCUDAカーネルを生成する)。CUDAは
        AOTコンパイルなので名前引きディスパッチは要らず、生成した
        `__global__`をそのまま`<<<batch, CUDA_BLOCK_THREADS>>>`で起動する
        (conv codegenと同じ設計)。

        **実行未検証**(このMacにCUDA実機/nvccが無いため。タスク指示
        どおり「ソース対称実装、実行検証は後日」)。
        """
        from . import densefuse as df

        a = op.attrs
        layers = a['layers']
        tag = self.uid('densefuse')
        kernel_name = f'{df.CODEGEN_KERNEL_PREFIX}{tag}'
        self.custom_kernels.append(df.build_cuda_kernel(a, kernel_name))

        idx_names = []
        tt_names = []
        for i, layer in enumerate(layers):
            ln = i + 1
            if df.idx_ctype_cuda(layer) == 'unsigned short':
                idx = self.up_u16(f'{tag}_idx{ln}', layer['indices'])
            else:
                idx = self.up_i32(f'{tag}_idx{ln}', layer['indices'])
            tt = self.up_u32(f'{tag}_tt{ln}', layer['truth_tables'])
            idx_names.append(idx)
            tt_names.append(tt)
        args = ', '.join(
            f'{idx}, {tt}' for idx, tt in zip(idx_names, tt_names,
                                              strict=True))
        self.body.append(
            f'  {kernel_name}<<<{self.N}, {df.CUDA_BLOCK_THREADS}, 0, '
            f'mdl_cuda_current_stream()>>>(\n'
            f'      {self.buf(op.inputs[0])}, {self.buf(op.output)}, '
            f'{args});\n'
            f'  MDL_TRY(cudaGetLastError());')

    # -- 全体 ---------------------------------------------------------
    def build(self) -> dict[str, str]:
        self.declare_buffers()
        self.declare_preproc()
        if self.dense_packed:
            self.emit_dense_packed_body()
        else:
            self.emit_body()
        return {f'{self.model_name}.cu': self._source()}

    def _slotted(self, name: str, slot_expr: str) -> str:
        """n_streams>1なら `name[slot_expr]`、そうでなければ `name` のまま。
        `name` は素の(streaming時は配列の)識別子(`_raw_buf`/'d_mdl_image'/
        'd_mdl_input'相当)であること(Metal版MetalBackend._slottedの
        ミラー)。"""
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
        # mdl_cuda_run内(streaming時はslotが実引数)ではout_sを使う
        # (常に0決め打ちにすると、将来mdl_cuda_runへ非0スロットを渡した
        # ときに黙って間違ったスロットを読む落とし穴になる)。
        fetch = (f'  MDL_TRY(cudaMemcpy(scores, '
                 f'{out_s if streaming else out_raw}, '
                 f'(size_t)n * {out_width} * sizeof(int), '
                 f'cudaMemcpyDeviceToHost));')
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
            # ビット列入力ストリーミング経路(下のstream_entry追記部)は
            # p is not Noneのときも生成される(モデルの持つ「前処理付き
            # image入力」と「既パック済みbits入力」の2経路は独立に選べる
            # ─ 非streamingのentry関数(下のmdl_cuda_infer_batch)と対称)。
            # そのため、非streaming版が明示的に呼んでいる
            # mdl_cuda_pack({in0}, {bp0}, ...) を、streaming版でも
            # mdl_model_encode の前に同様に呼ぶ必要がある
            # (呼ばないとbitplaneバッファが前回のまま=壊れる)。
            pack_call = (
                f'  MDL_TRY_INT(mdl_cuda_pack({in_s}, {bp_s}, '
                f'{p["channels"]} * n, {p["height"]}, {p["width"]}));\n')
            image_entry = f'''
int mdl_cuda_infer_image_batch(const unsigned char *image, int *scores,
                               int n) {{
  if (n < 1 || n > MDL_MAX_BATCH) return -1;
  if (mdl_cuda_init() != 0) return -1;
  MDL_TRY(cudaMemcpy({img0}, image, (size_t)n * {n_img},
                     cudaMemcpyHostToDevice));
  MDL_TRY_INT(mdl_cuda_preproc({img0}, {p['image_channels']},
                               {p['height']}, {p['width']},
                               d_mdl_preproc_desc, {p['channels']},
                               d_mdl_preproc_thr, d_mdl_preproc_qval,
                               {bp0}, n));
  return mdl_cuda_run(scores, n{', 0' if streaming else ''});
}}

int mdl_cuda_infer_image(const unsigned char *image, int *scores) {{
  return mdl_cuda_infer_image_batch(image, scores, 1);
}}
'''
            entry = f'''
int mdl_cuda_infer_batch(const unsigned char *input_bits, int *scores, int n) {{
  if (n < 1 || n > MDL_MAX_BATCH) return -1;
  if (mdl_cuda_init() != 0) return -1;
  MDL_TRY(cudaMemcpy({in0}, input_bits, (size_t)n * {in_width},
                     cudaMemcpyHostToDevice));
  MDL_TRY_INT(mdl_cuda_pack({in0}, {bp0}, {p['channels']} * n,
                            {p['height']}, {p['width']}));
  return mdl_cuda_run(scores, n{', 0' if streaming else ''});
}}
'''
            if streaming:
                stream_entry += f'''
/* ---- ストリーム・パイプライン化(画像入力経路) ---- */
int mdl_cuda_infer_image_batch_submit(const unsigned char *image, int n,
                                      int slot) {{
  if (n < 1 || n > MDL_MAX_BATCH) return -1;
  if (slot < 0 || slot >= {self.n_streams}) return -1;
  if (mdl_cuda_init() != 0) return -1;
  MDL_TRY_INT(mdl_cuda_select_stream(slot));
  MDL_TRY_INT(mdl_cuda_upload_async({img_s}, image, (size_t)n * {n_img}));
  MDL_TRY_INT(mdl_cuda_preproc({img_s}, {p['image_channels']},
                               {p['height']}, {p['width']},
                               d_mdl_preproc_desc, {p['channels']},
                               d_mdl_preproc_thr, d_mdl_preproc_qval,
                               {bp_s}, n));
  MDL_TRY_INT(mdl_model_encode(slot, n));
  return mdl_cuda_commit_async(slot);
}}

int mdl_cuda_infer_image_batch_collect(int *scores, int n, int slot) {{
  if (n < 1 || n > MDL_MAX_BATCH) return -1;
  if (slot < 0 || slot >= {self.n_streams}) return -1;
  MDL_TRY_INT(mdl_cuda_wait_slot(slot));
  MDL_TRY(cudaMemcpy(scores, {out_s}, (size_t)n * {out_width} * sizeof(int),
                     cudaMemcpyDeviceToHost));
  return 0;
}}
'''
        else:
            in0 = self._slotted('d_mdl_input', '0')
            in_s = self._slotted('d_mdl_input', 'slot')
            pack_call = ''
            entry = f'''
int mdl_cuda_infer_batch(const unsigned char *input_bits, int *scores, int n) {{
  if (n < 1 || n > MDL_MAX_BATCH) return -1;
  if (mdl_cuda_init() != 0) return -1;
  MDL_TRY(cudaMemcpy({in0}, input_bits, (size_t)n * {in_width},
                     cudaMemcpyHostToDevice));
  return mdl_cuda_run(scores, n{', 0' if streaming else ''});
}}
'''
            image_entry = ''
        if streaming:
            in_s = self._slotted('d_mdl_input', 'slot')
            stream_entry += f'''
/* ---- ストリーム・パイプライン化(ビット列入力経路) ---- */
int mdl_cuda_infer_batch_submit(const unsigned char *input_bits, int n,
                                int slot) {{
  if (n < 1 || n > MDL_MAX_BATCH) return -1;
  if (slot < 0 || slot >= {self.n_streams}) return -1;
  if (mdl_cuda_init() != 0) return -1;
  MDL_TRY_INT(mdl_cuda_select_stream(slot));
  MDL_TRY_INT(mdl_cuda_upload_async({in_s}, input_bits, (size_t)n * {in_width}));
{pack_call}  MDL_TRY_INT(mdl_model_encode(slot, n));
  return mdl_cuda_commit_async(slot);
}}

int mdl_cuda_infer_batch_collect(int *scores, int n, int slot) {{
  if (n < 1 || n > MDL_MAX_BATCH) return -1;
  if (slot < 0 || slot >= {self.n_streams}) return -1;
  MDL_TRY_INT(mdl_cuda_wait_slot(slot));
  MDL_TRY(cudaMemcpy(scores, {out_s}, (size_t)n * {out_width} * sizeof(int),
                     cudaMemcpyDeviceToHost));
  return 0;
}}
'''
        entry += '''
int mdl_cuda_infer(const unsigned char *input_bits, int *scores) {
  return mdl_cuda_infer_batch(input_bits, scores, 1);
}
'''
        custom_kernels_block = ''
        if self.custom_kernels:
            # モデル特化カーネル(mdl_k_conv_codegen_*)。CUDAはAOT
            # コンパイルなので生成した.cu内に直接__global__を定義し、
            # 起動用のグリッド算出だけ小さなヘルパを添える
            # (mdl_cuda.cu内部のmdl_grid/MDL_BLOCKとは別TUなので衝突しない
            # が、名前を意図的にずらして紛れないようにしてある)。
            custom_kernels_block = f'''/* ---- モデル特化conv2dカーネル ---- */
#define MDL_BLOCK_ 256
static inline int mdl_grid_(int n) {{ return (n + MDL_BLOCK_ - 1) / MDL_BLOCK_; }}

{chr(10).join(self.custom_kernels)}

'''
        if streaming:
            run_fn = f'''static int mdl_model_encode(int slot, int n) {{
{chr(10).join(self.body)}
  return 0;
}}

static int mdl_cuda_run(int *scores, int n, int slot) {{
  /* 既存の非スロット経路(mdl_cuda_infer_batch等、常にslot=0)が、直前の
     ストリーミングsubmitが残した選択(g_mdl_cur_stream)を引き継がない
     ようにする(教訓と同種の「暗黙の状態共有」を潰す)。*/
  mdl_cuda_select_default_stream();
  MDL_TRY_INT(mdl_model_encode(slot, n));
  MDL_TRY_INT(mdl_cuda_sync());
{fetch}
  return 0;
}}
'''
        else:
            run_fn = f'''static int mdl_cuda_run(int *scores, int n) {{
{chr(10).join(self.body)}
  MDL_TRY_INT(mdl_cuda_sync());
{fetch}
  return 0;
}}
'''
        return f'''/* 自動生成 — griffith-logic デプロイキット(griffith deploy codegen-cuda)
 * 手で編集しないこと。バンドルを更新して再生成する。 */
#include "mdl_cuda.h"
#include <cuda_runtime.h>
#include <stdio.h>

#define MDL_INPUT_WIDTH {in_width}
#define MDL_OUTPUT_WIDTH {out_width}
/* 同時に流せる最大枚数(バッファはこのぶん確保済み)。実行時の枚数は
   1..MDL_MAX_BATCH で可変で、起動スレッド数もそのぶんしか使わない。
   conv_exec のスレッド数は「画素数/32」しかないので、深い層では
   バッチがそのまま並列度になる。 */
#define MDL_MAX_BATCH {plan.batch}

#define MDL_TRY(expr)                                                         \\
  do {{                                                                        \\
    cudaError_t e_ = (expr);                                                  \\
    if (e_ != cudaSuccess) {{                                                  \\
      fprintf(stderr, "CUDA error %s at %s:%d\\n", cudaGetErrorString(e_),     \\
              __FILE__, __LINE__);                                            \\
      return -1;                                                              \\
    }}                                                                         \\
  }} while (0)

#define MDL_TRY_INT(expr) do {{ if ((expr) != 0) return -1; }} while (0)

{custom_kernels_block}
/* ---- ホスト側の定数テーブル ---- */
{chr(10).join(self.host_data)}

/* ---- デバイス側のバッファ・テーブル ---- */
{chr(10).join(self.dev_decl)}

static int mdl_cuda_ready = 0;

extern "C" int mdl_cuda_init(void) {{
  if (mdl_cuda_ready) return 0;
{f'  MDL_TRY_INT(mdl_cuda_streams_init({self.n_streams}));' if streaming else ''}
{chr(10).join(self.dev_init)}
  mdl_cuda_ready = 1;
  return 0;
}}

extern "C" void mdl_cuda_shutdown(void) {{
  if (!mdl_cuda_ready) return;
{chr(10).join(self.dev_free)}
{'  mdl_cuda_streams_shutdown();' if streaming else ''}
  mdl_cuda_ready = 0;
}}

{run_fn}
extern "C" {{
{entry}{image_entry}{stream_entry}
int mdl_cuda_input_width(void) {{ return MDL_INPUT_WIDTH; }}
int mdl_cuda_output_width(void) {{ return MDL_OUTPUT_WIDTH; }}
int mdl_cuda_max_batch(void) {{ return MDL_MAX_BATCH; }}
{f'int mdl_cuda_n_streams(void) {{ return {self.n_streams}; }}' if streaming else ''}
}}
'''


#: 旧名(スクリプト互換)
CudaModelBuilder = CudaBackend


def generate_cuda(bundle: dict, name: str = 'model_cuda', *,
                  batch: int = 1, conv_engine: str | None = None,
                  n_streams: int = 1,
                  conv_codegen: bool = False,
                  dense_fuse: bool | None = None,
                  dense_packed: bool | None = None,
                  auto: str = 'measured',
                  lut_flat_mode: str = 'auto') -> dict[str, str]:
    """バンドル → {ファイル名: 中身} のCUDAソース。

    Args:
        batch: 同時に流せる最大枚数。デバイスメモリはこの倍数になる
        conv_engine: 'split'または'combined'(旧経路)。既定
            `None`は`auto`に選択を委ねる(下記`auto`参照)。新旧の
            出力一致テスト以外で明示指定する必要は通常ない。
        n_streams: 1(既定)なら従来どおり。2以上でストリーム・
            パイプライン化(Metal版ミラー)向けの
            `mdl_cuda_infer_batch_submit`/`_collect`(画像入力があれば
            image 版も)を追加生成する。デバイスメモリは
            `batch * n_streams` 倍になる。dense_packed=Trueとの併用は
            v1では未対応(下でValueError)。も自動ONにしない
            (モデル依存で悪化する実測があるためopt-inのまま、
            `docs/guides/deploy-optimization.md`参照)。
        conv_codegen: 既定False(インタプリタのまま)。Trueで
            conv2dノードごとにtt即値+スロット静的展開のモデル特化CUDA
            カーネルを生成する(`conv_engine='split'` 前提)。も
            自動ONにしない(e2e改善がほぼ無い割にコードサイズが
            1.78〜6.65倍になる実測のため)。
        dense_fuse: TrueでMetal版と対称のdense多層融合カーネルを
            生成する(`plan.py::fuse_dense_layers`)。CUDA実機による
            実行検証・整数一致を確認済み(リモートGPU)。共有メモリ上限は
            `CUDA_DENSE_FUSE_MAX_SHARED_BYTES`(48KB、CUDAのstatic
            `__shared__`既定上限)をMetal版の32KBの代わりに使う ─
            Metalでは32KB超で融合対象外になるモデル(層幅の合計が
            32KB〜48KBの範囲)もCUDAでは融合できる。既定`None`は`auto`に
            選択を委ねる(`batch>=32`かつ融合可能な鎖があれば自動
            ON)。dense_packed=Trueとの併用はv1では未対応(下でValueError)。
        dense_packed: TrueでMetal版と対称のdenseバッチ軸
            ビットプレーン化カーネル(`mdl_cuda_pack_batch`/
            `mdl_cuda_lut_layer_packed[_u16]`/`mdl_cuda_group_sum_packed`)
            を使う経路を生成する。適格性は`plan.py::dense_packed_eligible`
            (Metal版と共有)。既定`None`は`auto`に選択を委ねる(dense_fuseが自動ONにならず・適格・`batch>=128`なら自動ON)。
            `dense_fuse=True`との併用はv1では未対応。
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
        bundle, batch=batch, backend='cuda', conv_engine=conv_engine,
        dense_fuse=dense_fuse, dense_packed=dense_packed, auto=auto)
    if auto == 'measured':
        print(f'griffith deploy codegen-cuda: auto-config conv_engine={conv_engine!r} '
             f'dense_fuse={dense_fuse} dense_packed={dense_packed}')
    backend = CudaBackend(
        plan, model_name=name, conv_engine=conv_engine, n_streams=n_streams,
        conv_codegen=conv_codegen, dense_packed=dense_packed)
    if lut_flat_mode not in backend.LUT_FLAT_MODES:
        raise ValueError(f'lut_flat_mode={lut_flat_mode!r} は未対応'
                         f'(選べるのは {sorted(backend.LUT_FLAT_MODES)})')
    backend.lut_flat_mode = lut_flat_mode
    files = backend.build()
    key = f'{name}.cu'
    files[key] = auto_config_header_comment(
        'cuda', auto, conv_engine, dense_fuse, dense_packed, decision) + files[key]
    return files
