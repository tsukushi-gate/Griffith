"""RDDLGN自己回帰decodeのmegakernel化(opt-in)。

`rddlgn_decode_gpu.py`は「1tickぶんの処理」をPython側で1回だけ生成し、
生成コード側の`for (t_ ...)` ループで`steps`回実行する——ただし**1tickあたり
複数のカーネルdispatch+D2Dコピー5本**をホストが順にissueする形は変わって
いない(embed_lookup 1 + dec_l_rec層数分 + D2Dコピー5本 + dec_pm_rec層数分 +
argmax 1)。MetalのD2Dコピー(`mdl_metal_copy`)は毎回compute encoderを
endEncodingしてblitを張り直すため、1tickあたりエンコーダ切替が複数回
発生し、CUDA実測(レーン32→1024でtok/sが6%しか伸びない)からも「ホスト側
発行回数」が律速している強い兆候があった(タスク指示の背景)。

設計(v1): **1スレッドグループ(Metal)/1スレッドブロック(CUDA) = 1レーン**。
decodeの`steps`ぶんのtickループ全体をカーネル内へ持ち込み(threadgroup/
sharedメモリのバリアで層間同期)、ホスト側は**1回のdispatch+1回の同期**で
decode全体が終わる形にする。

    - レーンローカルな中間値(embed出力・dec_l_rec/dec_pm_recの各層出力・
      context・state・argmaxのスコア)はすべてthreadgroup/sharedメモリの
      **単一の平坦バッファ**(`tgbuf`)へ、値ごとに4バイト境界のオフセットで
      詰める(`_threadgroup_layout`、アリーナ方式のthreadgroup版。
      ただしメモリ計画のような生存区間解析はせず、値ごとに専用の
      オフセットを割り当てるだけ——decode1回あたりのthreadgroupメモリ
      所要が32KBに収まる小〜中規模モデルがv1のスコープなので、
      複雑な再利用は複雑さに見合わない)。
    - 重み(lut_denseの添字/真理値表・埋め込み表)はdeviceメモリのまま
      (constant/const、読み取り専用)。
    - **自動フォールバック**: threadgroup/sharedメモリ所要が32KBを超える、
      Metalのバッファ引数上限(31本、`RDDLGN_MEGAKERNEL_MAX_LAYERS`)を
      超える、またはdec_l_rec/dec_pm_recのbodyにlut_dense/flat_concat/
      reduce_scores(group_sum)以外のop種別が現れる場合、従来経路
      (`rddlgn_decode_gpu._emit_decode_tick_loop`をそのまま再利用)を
      生成する。**どちらを生成したかは生成ソース中のコメントマーカー**
      (`/* megakernel: enabled (...) */` または
      `/* megakernel: fallback (reason=...) */`)で明示する
      (ValueErrorで落とすのではなく、常に動く生成物を返す——
      「収まらなければ対象外」の流儀と同じ)。

適格性(v1スコープ、RDDLGNの実モデル構成で常に満たされる想定):
    - dec_l_rec body: lut_denseの連鎖のみ(L-layers=LutStackは他のop種別を
      生成しない)
    - dec_pm_rec body: lut_dense/flat_concat/reduce_scores(group_sum)のみ
      (P-layers/M-layers=LutStack、`[p_prev; context; l_t]`等のconcatが
      flat_concat、末尾のGroupSumがreduce_scores)
    - lut_denseのn_inputsは2に限定しない(dense_fuseと違い、汎用の
      最小項展開アドレス式を使うため任意のn_inputsで動く)

既存の経路(`generate_rddlgn_decode_metal/_cuda`の`megakernel=False`、
既定)は無改変で残る(A/B用)。CUDA側は**実行未検証**(このMacにnvcc/CUDA
実機が無いため。ソース対称実装+ローカル構文検査のみ、同じ限界)。
"""

from __future__ import annotations

from dataclasses import dataclass

from .cudabackend import CudaBackend
from .metalbackend import MetalBackend
from .plan import KIND_SCORES, build_plan, lut_dense_indices_fit_u16
from .rddlgn_decode_gpu import (  # noqa: SLF001 -- 同一サブシステム内の共有
    DecodeBuffers,
    DecodeInfo,
    _cuda_decode_entry,
    _DecodeSpec,
    _discover_decode_spec,
    _emit_decode_preamble,
    _emit_decode_tick_loop,
    _metal_decode_entry,
)

#: threadgroup(Metal)/shared(CUDA)メモリの上限(DENSE_FUSE_MAX_SHARED_
#: BYTESと同じ物理制約=Apple GPUのthreadgroupメモリ32KB。CUDAのshared
#: メモリ上限は既定48KB/最大228KB可変だが、両バックエンドで同じプランが
#: 使い回せるようMetal側に揃える、同じ判断)。
RDDLGN_MEGAKERNEL_MAX_SHARED_BYTES = 32 * 1024

#: 融合可能なlut_dense層数の上限(DENSE_FUSE_MAX_LAYERSと同じ理由)。
#: Metalのカーネル引数テーブルはindex 0..30(31本)まで。megakernelの
#: 固定引数は5本(tok_init/table/ctx_init/pstate_init/out、buffer(0..4))、
#: 層iはbuffer(5+2i)=idx/buffer(6+2i)=ttを使うので、最後の層
#: (i=n_layers-1)のtt引数番号 6+2*(n_layers-1) = 4+2*n_layers が30を
#: 超えてはならない ⇔ n_layers <= 13。CUDAはこの制約自体は無いが、
#: 両バックエンド共通の1箇所で判定する方針(同じ)。
RDDLGN_MEGAKERNEL_MAX_LAYERS = 13

#: CUDA側のスレッドブロック幅(生成時の固定値。`densefuse.
#: CUDA_BLOCK_THREADS`と同じ考え方 ─ 1レーンぶんの並列度で、body側は
#: すべてtid_/tgs_のstride loopなのでどの値でも正しく動く)。
RDDLGN_MEGAKERNEL_CUDA_THREADS = 1024

#: v1が対応するdec_pm_rec body内のop種別(reduce_scoresはop='group_sum'に
#: 限る)。これ以外が現れたら自動フォールバックする。
_SUPPORTED_PM_OPS = frozenset({'lut_dense', 'flat_concat', 'reduce_scores'})


@dataclass
class ThreadgroupLayout:
    """threadgroup/sharedメモリ上の単一平坦バッファ`tgbuf`内で、各値
    (PlanValueのid、またはトークンスカラーを表す特殊キー`'tok'`)が
    どこに(バイトオフセット、要素型)置かれるかの対応表。"""

    offsets: dict[int | str, tuple[int, str]]   # id -> (byte_off, 'u8'|'i32')
    total_bytes: int


def _lut_dense_ops(spec: _DecodeSpec) -> list:
    """dec_l_rec→dec_pm_recの順で、lut_denseのopだけを並べたもの
    (megakernelのidx/tt表・カーネル引数の並び順、`idx1`が1番目のlut_dense
    層に対応する)。"""
    return ([op for op in spec.l_scope_ops if op.kind == 'lut_dense']
           + [op for op in spec.pm_scope_ops if op.kind == 'lut_dense'])


def _fits(spec: _DecodeSpec) -> tuple[bool, str]:
    """megakernel経路を生成してよいか(32KB以内・op種別が対応範囲内・
    Metalのバッファ引数上限以内)を判定する。理由つきで返す
    (ValueErrorではなく、呼び出し側が自動フォールバックの判断に使う)。"""
    for op in spec.l_scope_ops:
        if op.kind != 'lut_dense':
            return False, (
                f'dec_l_recのbodyにmegakernel未対応のop種別{op.kind!r}'
                '(lut_denseのみ対応)')
    for op in spec.pm_scope_ops:
        if op.kind not in _SUPPORTED_PM_OPS:
            return False, (
                f'dec_pm_recのbodyにmegakernel未対応のop種別{op.kind!r}'
                "(lut_dense/flat_concat/reduce_scoresのみ対応)")
        if op.kind == 'reduce_scores' and op.attrs.get('op') != 'group_sum':
            return False, (
                "dec_pm_recのbodyにmegakernel未対応のreduce_scores種別"
                f"{op.attrs.get('op')!r}(group_sumのみ対応)")
    n_layers = len(_lut_dense_ops(spec))
    if n_layers > RDDLGN_MEGAKERNEL_MAX_LAYERS:
        return False, (
            f'lut_dense層数{n_layers}がMetalのバッファ引数上限'
            f'(={RDDLGN_MEGAKERNEL_MAX_LAYERS}層)を超える')
    layout = _threadgroup_layout(spec)
    if layout.total_bytes > RDDLGN_MEGAKERNEL_MAX_SHARED_BYTES:
        return False, (
            f'threadgroup/sharedメモリ所要{layout.total_bytes}Bが上限'
            f'{RDDLGN_MEGAKERNEL_MAX_SHARED_BYTES}B(32KB)を超える')
    return True, ''


def _threadgroup_layout(spec: _DecodeSpec) -> ThreadgroupLayout:
    """decodeのtickループ全体をカーネル内で回すのに要る全レーンローカル値
    (アリーナ方式のthreadgroup版、値ごとに専用オフセット・4バイト
    境界に整列)。"""
    plan = spec.plan
    ids: list[int] = []
    seen: set[int] = set()

    def add(vid: int) -> None:
        if vid not in seen:
            seen.add(vid)
            ids.append(vid)

    add(spec.l_ext_id)
    for op in spec.l_scope_ops:
        add(op.output)
    add(spec.ctx_ext_id)
    add(spec.l_in_ext_id)
    add(spec.p_state_id)
    for op in spec.pm_scope_ops:
        add(op.output)

    offsets: dict[int | str, tuple[int, str]] = {}
    off = 0
    for vid in ids:
        val = plan.value(vid)
        kind = 'i32' if val.kind == KIND_SCORES else 'u8'
        nbytes = val.width * (4 if kind == 'i32' else 1)
        off = (off + 3) & ~3
        offsets[vid] = (off, kind)
        off += nbytes
    # トークン(現在の生成token、全スレッド共有の1スカラー)。
    off = (off + 3) & ~3
    offsets['tok'] = (off, 'i32')
    off += 4
    total = (off + 15) & ~15   # 16バイト境界へ余裕を持たせる
    return ThreadgroupLayout(offsets=offsets, total_bytes=total)


def _var(vid: int | str) -> str:
    return 'tok_shared' if vid == 'tok' else f'v{vid}'


def _decl_lines(layout: ThreadgroupLayout, addr_space: str) -> list[str]:
    """`tgbuf`(単一平坦バッファ)から各値への型付きローカルポインタ変数を
    宣言する行。`addr_space`はMetalなら`'threadgroup '`、CUDAなら`''`。"""
    lines = []
    for vid, (off, kind) in layout.offsets.items():
        name = _var(vid)
        if kind == 'i32':
            lines.append(
                f'{addr_space}int *{name} = ({addr_space}int*)(tgbuf + {off});')
        else:
            lines.append(f'{addr_space}unsigned char *{name} = tgbuf + {off};')
    return lines


def _preload_lines(spec: _DecodeSpec) -> list[str]:
    """カーネル起動直後(tickループの外)で1回だけ、レーン初期値をdevice→
    threadgroup/sharedへ読み込む(context=invariant・state=初期state・
    token=bos)。"""
    return [
        f'for (int i = tid_; i < {spec.ctx_width}; i += tgs_) '
        f'{_var(spec.ctx_ext_id)}[i] = ctx_init[(long)bid_ * {spec.ctx_width} + i];',
        f'for (int i = tid_; i < {spec.p_width}; i += tgs_) '
        f'{_var(spec.p_state_id)}[i] = pstate_init[(long)bid_ * {spec.p_width} + i];',
        f'if (tid_ == 0) {_var("tok")}[0] = tok_init[(int)bid_];',
        '{BARRIER}',
    ]


def _lut_dense_lines(op, idx_tt_names: dict[int, tuple[str, str]]) -> list[str]:
    """1つのlut_dense opの本体(汎用の最小項展開アドレス式。n_inputsは
    任意——densefuseと違いn_inputs==2に限定しない)。"""
    a = op.attrs
    n_inputs = int(a['n_inputs'])
    out_width = int(a['out_width'])
    src_vid = op.inputs[0]
    dst_vid = op.output
    idx_name, tt_name = idx_tt_names[dst_vid]
    lines = [f'for (int o = tid_; o < {out_width}; o += tgs_) {{',
            '  unsigned int addr = 0u;']
    for j in range(n_inputs):
        lines.append(
            f'  addr |= ((unsigned int){_var(src_vid)}'
            f'[{idx_name}[{j} * {out_width} + o]]) << {j}u;')
    lines.append(
        f'  {_var(dst_vid)}[o] = (unsigned char)(({tt_name}[o] >> addr) & 1u);')
    lines.append('}')
    lines.append('{BARRIER}')
    return lines


def _flat_concat_lines(op) -> list[str]:
    a = op.attrs
    dst_vid = op.output
    lines = []
    offset = 0
    for vid, width in zip(op.inputs, a['widths'], strict=True):
        lines.append(
            f'for (int i = tid_; i < {width}; i += tgs_) '
            f'{_var(dst_vid)}[{offset} + i] = {_var(vid)}[i];')
        offset += width
    lines.append('{BARRIER}')
    return lines


def _group_sum_lines(op, vocab: int) -> list[str]:
    a = op.attrs
    src_vid = op.inputs[0]
    dst_vid = op.output
    k = int(a['k'])
    gsz = int(a['in_width']) // k
    if k != vocab:
        raise ValueError(f'group_sum.k({k}) != dec_pm_rec.vocab({vocab})')
    return [
        f'if (tid_ < {k}) {{',
        '  int s = 0;',
        f'  for (int i = 0; i < {gsz}; i++) s += (int){_var(src_vid)}[tid_ * {gsz} + i];',
        f'  {_var(dst_vid)}[tid_] = s;',
        '}',
        '{BARRIER}',
    ]


def _tick_lines(spec: _DecodeSpec,
                idx_tt_names: dict[int, tuple[str, str]]) -> list[str]:
    """1tickぶんの本体(embed_lookup→L body→copy→PM body→state保存→
    argmax→出力)。`_emit_decode_tick_loop`と論理的に同じ手順を
    threadgroup/sharedメモリのstride loop+バリアで書き直したもの。"""
    lines = [
        f'for (int j = tid_; j < {spec.d_emb}; j += tgs_) '
        f'{_var(spec.l_ext_id)}[j] = table[(long){_var("tok")}[0] * '
        f'{spec.d_emb} + j];',
        '{BARRIER}',
    ]
    for op in spec.l_scope_ops:
        lines += _lut_dense_lines(op, idx_tt_names)

    lines.append(
        f'for (int i = tid_; i < {spec.l_out_width}; i += tgs_) '
        f'{_var(spec.l_in_ext_id)}[i] = {_var(spec.l_out_id)}[i];')
    lines.append('{BARRIER}')

    for op in spec.pm_scope_ops:
        if op.kind == 'lut_dense':
            lines += _lut_dense_lines(op, idx_tt_names)
        elif op.kind == 'flat_concat':
            lines += _flat_concat_lines(op)
        else:
            lines += _group_sum_lines(op, spec.vocab)

    lines.append(
        f'for (int i = tid_; i < {spec.p_width}; i += tgs_) '
        f'{_var(spec.p_state_id)}[i] = {_var(spec.p_new_id)}[i];')
    lines.append('{BARRIER}')

    lines.append('if (tid_ == 0) {')
    lines.append('  int best = 0;')
    lines.append(
        f'  for (int i = 1; i < {spec.vocab}; i++) {{ '
        f'if ({_var(spec.scores_id)}[i] > {_var(spec.scores_id)}[best]) '
        'best = i; }')
    lines.append(f'  {_var("tok")}[0] = best;')
    lines.append(f'  out[(long)t_ * {spec.batch} + (int)bid_] = best;')
    lines.append('}')
    lines.append('{BARRIER}')
    return lines


def _prepare_layers(backend, spec: _DecodeSpec):
    """全lut_dense層(dec_l_rec→dec_pm_rec順)のidx/tt表をアップロードする。

    `backend.up_u16`/`up_i32`/`up_u32`を直接呼ばず
    `backend._cached_upload`経由にする——dec_l_rec/dec_pm_rec自身の各層は
    教師強制本体(`emit_body()`)が既に一度処理済み(同じop=同じ添字/
    真理値表)なので、直呼びのままだと全layerぶんの重み表がここでもう1組
    (megakernel専用のidx1/tt1...という別名で)焼かれてしまう
    (`_emit_decode_preamble`が呼び出し側で`backend._dedup_uploads`を
    既にTrueにしている前提——本関数は`_emit_metal`/`_emit_cuda`から
    `_emit_decode_preamble`より後にしか呼ばれない)。

    戻り値: (idx_tt_names, layer_meta)。
      idx_tt_names: dst_vid -> (カーネル内パラメータ名'idxN', 'ttN')
      layer_meta: [{'u16': bool, 'host_idx': ホスト側バッファ名,
                    'host_tt': ホスト側バッファ名}, ...](層順)
    """
    idx_tt_names: dict[int, tuple[str, str]] = {}
    layer_meta = []
    for i, op in enumerate(_lut_dense_ops(spec)):
        a = op.attrs
        ln = i + 1
        tag = backend.uid('rddlgnmega')
        in_width = backend.plan.value(op.inputs[0]).width
        u16 = lut_dense_indices_fit_u16(in_width)
        if u16:
            host_idx = backend._cached_upload(  # noqa: SLF001 -- 共有機構
                'up_u16', f'{tag}_idx', a['indices'])
        else:
            host_idx = backend._cached_upload(  # noqa: SLF001 -- 共有機構
                'up_i32', f'{tag}_idx', a['indices'])
        host_tt = backend._cached_upload(  # noqa: SLF001 -- 共有機構
            'up_u32', f'{tag}_tt', a['truth_tables'])
        idx_tt_names[op.output] = (f'idx{ln}', f'tt{ln}')
        layer_meta.append({'u16': u16, 'host_idx': host_idx,
                           'host_tt': host_tt})
    return idx_tt_names, layer_meta


def _metal_kernel_source(spec: _DecodeSpec, layout: ThreadgroupLayout,
                         idx_tt_names, layer_meta, kernel_name: str) -> str:
    args = [
        'device const int *tok_init [[buffer(0)]]',
        'device const uchar *table [[buffer(1)]]',
        'device const uchar *ctx_init [[buffer(2)]]',
        'device const uchar *pstate_init [[buffer(3)]]',
        'device int *out [[buffer(4)]]',
    ]
    for i, lm in enumerate(layer_meta):
        b = 5 + 2 * i
        ct = 'ushort' if lm['u16'] else 'int'
        args.append(f'constant {ct} *idx{i + 1} [[buffer({b})]]')
        args.append(f'constant uint *tt{i + 1} [[buffer({b + 1})]]')
    args += [
        'threadgroup uchar *tgbuf [[threadgroup(0)]]',
        'uint tid_ [[thread_position_in_threadgroup]]',
        'uint tgs_ [[threads_per_threadgroup]]',
        'uint bid_ [[threadgroup_position_in_grid]]',
    ]
    sig = ',\n    '.join(args)
    body_lines = (_decl_lines(layout, 'threadgroup ')
                 + _preload_lines(spec)
                 + [f'for (int t_ = 0; t_ < {spec.steps}; t_++) {{']
                 + [f'  {tl}' for tl in _tick_lines(spec, idx_tt_names)]
                 + ['}'])
    body = '\n  '.join(body_lines).replace(
        '{BARRIER}', 'threadgroup_barrier(mem_flags::mem_threadgroup);')
    return f'kernel void {kernel_name}(\n    {sig}) {{\n  {body}\n}}\n'


def _cuda_kernel_source(spec: _DecodeSpec, layout: ThreadgroupLayout,
                        idx_tt_names, layer_meta, kernel_name: str) -> str:
    params = [
        'const int *tok_init', 'const unsigned char *table',
        'const unsigned char *ctx_init', 'const unsigned char *pstate_init',
        'int *out',
    ]
    for i, lm in enumerate(layer_meta):
        ct = 'unsigned short' if lm['u16'] else 'int'
        params.append(f'const {ct} *idx{i + 1}')
        params.append(f'const unsigned int *tt{i + 1}')
    sig = ',\n    '.join(params)
    body_lines = (_decl_lines(layout, '')
                 + _preload_lines(spec)
                 + [f'for (int t_ = 0; t_ < {spec.steps}; t_++) {{']
                 + [f'  {tl}' for tl in _tick_lines(spec, idx_tt_names)]
                 + ['}'])
    body = '\n  '.join(body_lines).replace('{BARRIER}', '__syncthreads();')
    #(実機検証で踏んだバグ): threads=1024(RDDLGN_MEGAKERNEL_
    # CUDA_THREADS)だと、tgbuf内の値ごとの専用ポインタ変数(v0,v1,...)が
    # スレッドあたりレジスタを多く食い(tiny実測128本)、1024スレッド×
    # 128レジスタ=131072 > sm_89のブロックあたりレジスタ上限65536で
    # "too many resources requested for launch"が実機で発生した(構文検査
    # だけのローカル検証では検出不能——ptxasの実際のレジスタ割付が要る)。
    # __launch_bounds__でブロックあたりレジスタ予算をthreads数から
    # 逆算させ、必要ならコンパイラに待避(local memoryへのspill、正しさは
    # 不変)させることで確実に起動可能にする(Metal側はthreadgroupメモリが
    # 主体でこの種のレジスタ制約が無いため、CUDA固有の最小修正)。
    return f'''__global__ void __launch_bounds__({RDDLGN_MEGAKERNEL_CUDA_THREADS})
{kernel_name}(
    {sig}) {{
  /* __align__(4): tgbuf内のint32アクセス(scores等、_decl_linesが
     (int*)キャストする箇所)がアラインメント要件を満たすように明示する
     (unsigned char配列自体はC++の保証としては1バイト整列で足りるため)。 */
  __shared__ __align__(4) unsigned char tgbuf[{layout.total_bytes}];
  const int tid_ = (int)threadIdx.x, tgs_ = (int)blockDim.x;
  const int bid_ = (int)blockIdx.x;
  {body}
}}
'''


def _emit_metal(backend: MetalBackend, spec: _DecodeSpec,
                buffers: DecodeBuffers) -> str:
    """megakernelが適格ならbackend.bodyへdispatchを1回積み、そうでなければ
    従来tickループへフォールバックする。戻り値はどちらを使ったかを
    示すコメントマーカー(生成ソースへそのまま埋め込む)。"""
    fits, reason = _fits(spec)
    if not fits:
        _emit_decode_tick_loop(backend, spec, buffers)
        return f'/* megakernel: fallback (reason={reason}) */'

    layout = _threadgroup_layout(spec)
    idx_tt_names, layer_meta = _prepare_layers(backend, spec)
    tag = backend.uid('rddlgnmega')
    kernel_name = f'mdl_k_rddlgn_decode_mega_{tag}'
    backend.custom_kernels.append(
        _metal_kernel_source(spec, layout, idx_tt_names, layer_meta,
                             kernel_name))

    idx_arr = f'{tag}_idx_arr'
    tt_arr = f'{tag}_tt_arr'
    n_layers = len(layer_meta)
    backend.decl.append(f'static mdl_metal_buf {idx_arr}[{n_layers}];')
    backend.decl.append(f'static mdl_metal_buf {tt_arr}[{n_layers}];')
    for i, lm in enumerate(layer_meta):
        backend.init.append(f'  {idx_arr}[{i}] = {lm["host_idx"]};')
        backend.init.append(f'  {tt_arr}[{i}] = {lm["host_tt"]};')

    backend.call(
        f'mdl_metal_rddlgn_decode_megakernel_exec("{kernel_name}", '
        f'{buffers.d_token}, {buffers.d_table}, {buffers.d_context}, '
        f'{buffers.d_pstate}, {buffers.d_out}, {idx_arr}, {tt_arr}, '
        f'{n_layers}, {layout.total_bytes}, {backend.N})')
    return (f'/* megakernel: enabled (threadgroup_bytes='
           f'{layout.total_bytes}, layers={n_layers}) */')


def _emit_cuda(backend: CudaBackend, spec: _DecodeSpec,
              buffers: DecodeBuffers) -> str:
    fits, reason = _fits(spec)
    if not fits:
        _emit_decode_tick_loop(backend, spec, buffers)
        return f'/* megakernel: fallback (reason={reason}) */'

    layout = _threadgroup_layout(spec)
    idx_tt_names, layer_meta = _prepare_layers(backend, spec)
    tag = backend.uid('rddlgnmega')
    kernel_name = f'mdl_k_rddlgn_decode_mega_{tag}'
    backend.custom_kernels.append(
        _cuda_kernel_source(spec, layout, idx_tt_names, layer_meta,
                            kernel_name))

    args = [buffers.d_token, buffers.d_table, buffers.d_context,
           buffers.d_pstate, buffers.d_out]
    for lm in layer_meta:
        args.append(lm['host_idx'])
        args.append(lm['host_tt'])
    args_text = ',\n      '.join(args)
    backend.body.append(
        f'  {kernel_name}<<<{backend.N}, {RDDLGN_MEGAKERNEL_CUDA_THREADS}>>>(\n'
        f'      {args_text});\n'
        f'  MDL_TRY(cudaGetLastError());')
    return (f'/* megakernel: enabled (shared_bytes={layout.total_bytes}, '
           f'layers={len(layer_meta)}) */')


def build_decode_metal_megakernel(bundle: dict, name: str, batch: int,
                                  embed_table: dict) -> dict[str, str]:
    """`rddlgn_decode_gpu.generate_rddlgn_decode_metal(megakernel=True)`の
    実体。呼び出し側で`embed_table`存在チェック済みであること。"""
    plan = build_plan(bundle, batch=batch)
    backend = MetalBackend(plan, model_name=name)
    backend.declare_buffers()
    backend.declare_preproc()
    backend.emit_body()
    saved, backend.body = backend.body, []
    spec = _discover_decode_spec(plan, embed_table)
    buffers = _emit_decode_preamble(backend, spec)
    marker = _emit_metal(backend, spec, buffers)
    decode_lines = backend.body
    backend.body = saved
    text = backend._source()  # noqa: SLF001 -- 同一パッケージ内、既存パターン踏襲
    info = DecodeInfo(steps=spec.steps, src_width=spec.src_width,
                      x_src_buf=backend.buf(spec.x_src_id),
                      d_bos=buffers.d_bos, d_out=buffers.d_out,
                      batch=spec.batch)
    text += f'\n{marker}\n'
    text += _metal_decode_entry(info, decode_lines)
    return {f'{name}.mm': text}


def build_decode_cuda_megakernel(bundle: dict, name: str, batch: int,
                                 embed_table: dict) -> dict[str, str]:
    """`rddlgn_decode_gpu.generate_rddlgn_decode_cuda(megakernel=True)`の
    実体。CUDA対称版(実行未検証、構文検査のみ——このモジュールdocstring
    参照)。"""
    plan = build_plan(bundle, batch=batch)
    backend = CudaBackend(plan, model_name=name)
    backend.declare_buffers()
    backend.declare_preproc()
    backend.emit_body()
    saved, backend.body = backend.body, []
    spec = _discover_decode_spec(plan, embed_table)
    buffers = _emit_decode_preamble(backend, spec)
    marker = _emit_cuda(backend, spec, buffers)
    decode_lines = backend.body
    backend.body = saved
    text = backend._source()  # noqa: SLF001 -- 同一パッケージ内、既存パターン踏襲
    info = DecodeInfo(steps=spec.steps, src_width=spec.src_width,
                      x_src_buf=backend.buf(spec.x_src_id),
                      d_bos=buffers.d_bos, d_out=buffers.d_out,
                      batch=spec.batch)
    text += f'\n{marker}\n'
    text += _cuda_decode_entry(info, decode_lines)
    return {f'{name}.cu': text}
