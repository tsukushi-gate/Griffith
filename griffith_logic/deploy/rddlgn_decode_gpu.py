"""RDDLGN自己回帰生成(decode)のGPU(CUDA/Metal)ドライバ生成。

背景: `deploy/gpubackend.py::emit_recurrent_exec` は教師強制(teacher
forcing)しか対応していない——外部入力(ext_bufs)を「あらかじめ用意された
配列からtickごとのスライスを読む」前提でコピーインするだけで、
「前tickの出力をargmaxで選び埋め込み直して次tickの入力にする」という
自己回帰生成のデータ依存は表現できない(移植時点で開発記録に
「GPUドライバは未実装」と明記されていた領域)。

設計(CPU版 `runtime/c/src/mdl_rddlgn.c::mdl_rddlgn_generate_greedy` が
リファレンス。1トークンぶんの処理をtickごとに手書きする点は同じで、
それをGPUカーネル列へ焼き直す):

    1. src_bitsをx_srcバッファへ直接書き、エンコーダ(enc_n_rec/enc_k_rec)を
       既存の`emit_recurrent_exec`でそのまま1回実行する(教師強制と
       完全に同じ経路 — decodeで変わるのはデコーダ側だけ)。
    2. dec_pm_recのstate(sv_p)をアリーナ外の専用バッファ
       (`d_dec_pstate`)へ初期化する。context(encoder出力)も専用バッファ
       (`d_dec_context`)へコピーしておく。**これらをアリーナ内のstate/ext
       バッファに直接持たせず一度シャドーへ逃がすのは、CPU版ハーネスの
       ファイル先頭コメントと同じ理由**: メモリ計画(deploy/memplan.py)は
       「dec_l_recの全tickが終わってからdec_pm_recが始まる」という
       教師強制の実行順序を前提にアリーナを詰めており(l_recの内部
       スクラッチとpm_recのstate/ext/内部値の生存区間は教師強制の実行順では
       重ならないため、同じアドレスへ詰められていておかしくない)、
       decodeハーネスはl_recとpm_recのstep相当を**tickごとに交互に**呼ぶため
       その前提が崩れる。l_rec側の(次tickでの)再実行がpm_recの永続値の
       アリーナ領域を踏む可能性があるので、真の値はシャドーに置き、
       pm_recのstep相当を呼ぶ**直前に毎回書き戻す**(CPU版と同じ
       save/restore規律)。
    3. 固定max_len(=IRのtgt_len)ぶんのtickをGPU側で**1回のコマンド
       バッファ/ストリームに全部エンコードし、ホスト同期は1回だけ**
       (確立済みの「発行順=実行順、追加同期不要」の性質を
       decodeのtickループにもそのまま適用する)。各tick:
         (a) embed_lookup: token_buf(前tickのargmax結果、tick0はbos)を
             埋め込み表で引きl_extへ書く(mdl_k_embed_lookup、新設)
         (b) dec_l_recのbody(scope ops)をそのまま1回評価 → l_out
         (c) l_out → pm_l_ext(直後にコピー。他の書き込みが割り込まない
             うちに、CPU版のmemcpy(pm_l_ext, l_out, ...)と同じタイミング)
         (d) シャドーのcontext/p_state → pm_ctx_ext/pm_state(使う直前に
             毎回書き戻す。上記2.の理由)
         (e) dec_pm_recのbody(scope ops)をそのまま1回評価 →
             新state候補 + scores(group_sum出力)
         (f) 新state候補 → シャドーp_state(直後に保存)
         (g) argmax: scores→token_buf(mdl_k_argmax、新設)
         (h) token_buf → out_tokens[t](tickごとの行に書き出す)
       EOSは**デバイス側で判定しない**(レーンごとに打ち切りタイミングが
       違うと分岐/マスクが要り、1コマンドバッファ化の単純さを壊す)。
       全レーンが常にmax_lenぶん走り切り、EOS判定は呼び出し側
       (テストのゴールデン比較・実運用のポスト処理)がCPU版と同じ規約
       (最初にeos_tokenが現れた位置で打ち切り)で行う
       —「そのまま走らせ後処理で切る」設計方針(タスク指示の設計指針2)。

制約: 整数厳密一致(decode出力=CPU golden `mdl_rddlgn_generate_greedy`/
`export/rddlgn_ir.py::simulate_rddlgn_greedy_bits` と、EOSより前の
プレフィックスで全トークン一致)。既定のteacher forcing生成物
(`generate_metal`/`generate_cuda`)は一切変更しない——本モジュールは
独立したopt-in経路(`generate_rddlgn_decode_metal`/`_cuda`)。

新カーネル(`mdl_k_embed_lookup`/`mdl_k_argmax`、runtime/gpu/
mdl_gpu_kernels.h + runtime/{metal,cuda}側の入口)はIRのop語彙に無い
(argmax/表引きは論理回路で表せないblackbox相当、CPU版のmdl_rddlgn.cが
cbackend.pyのEMITTERSの外に手書きされているのと対称の位置づけ)ので、
`deploy/plan.py`のregister_node_kind/EMITTERSには登録しない。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .cudabackend import CudaBackend
from .metalbackend import MetalBackend
from .plan import ExecPlan, PlanOp, build_plan

#: RDDLGN(export/rddlgn_ir.py::export_rddlgn_ir)が常に使う固定ノードid
#: (cbackend.pyのRDDLGN_RECURRENT_NODES/RDDLGN_BUFFER_NODESと同じ前提)。
_X_SRC = 'x_src'
_ENC_N = 'enc_n_rec'
_ENC_K = 'enc_k_rec'
_DEC_L = 'dec_l_rec'
_DEC_PM = 'dec_pm_rec'


def _find_op(plan: ExecPlan, kind: str, node_id: str) -> PlanOp:
    for op in plan.ops:
        if op.kind == kind and op.node_id == node_id:
            return op
    raise ValueError(f'{kind}(node_id={node_id!r}) がplan.opsに無い'
                     '(RDDLGN以外のバンドルを渡していないか確認)')


def _find_output(plan: ExecPlan, node_id: str) -> int:
    for op in plan.ops:
        if op.node_id == node_id:
            return op.output
    raise ValueError(f'node_id={node_id!r} のopがplan.opsに無い')


def _reset_recurrent_state(backend, op: PlanOp) -> None:
    """opのstate_bufs(あれば)をすべて初期値へ書き戻す。

    cbackend.py::_emit_recurrent_public_apiが生成するCPU版
    `mdl_run_recurrent_<id>`が`mdl_reset_state_<id>`を毎回呼ぶのと同じ
    契機・同じ理由: K-layers(enc_k_rec)は真のリカレント
    (`sv_state`を持つ)なので、decode呼び出しのたびに(同じctx/バッファを
    使い回しても前回の生成の残滓が乗らないよう)明示的に初期値へ戻す
    必要がある。N-layers(enc_n_rec)/L-layers(dec_l_rec)はstate 0個なので
    このループは空になり何もしない(該当なしなら安全に無視)。
    """
    plan = backend.plan
    for _sv_id, buf_id in op.attrs.get('state_bufs', {}).items():
        init_op = next(o for o in plan.ops
                       if o.kind == 'state_init' and o.output == buf_id)
        backend.emit_state_init(init_op)


def unpack_embed_table(embed_table: dict) -> np.ndarray:
    """bundle['embed_table']['bits'](u32 LSB-firstパック)を
    (vocab, d_emb)の0/1(uint8)配列へ展開する(export/rddlgn_ir.py::
    _pack_bits_lsbの逆変換)。

    GPU側は表引きのたびに64bit演算・追加のアンパックを持ち込みたくない
    ので、C側(u32パック+ビット展開)と違い「1バイト=1ビット」の
    フラットな表としてデバイスへ常駐させる(値はCPU版と同じ埋め込み表)。
    """
    vocab = int(embed_table['vocab'])
    d_emb = int(embed_table['d_emb'])
    wpr = (d_emb + 31) // 32
    words = np.asarray(embed_table['bits'], dtype=np.uint64).reshape(vocab, wpr)
    total_bits = wpr * 32
    bits = np.zeros((vocab, total_bits), dtype=np.uint8)
    shifts = np.arange(32, dtype=np.uint64)
    for w in range(wpr):
        col = words[:, w]
        bits[:, w * 32:(w + 1) * 32] = (
            (col[:, None] >> shifts[None, :]) & np.uint64(1)).astype(np.uint8)
    return np.ascontiguousarray(bits[:, :d_emb])


class DecodeInfo:
    """decode生成の結果、エントリ関数を書くのに要る値をまとめて返す。"""

    def __init__(self, steps: int, src_width: int, x_src_buf: str,
                d_bos: str, d_out: str, batch: int):
        self.steps = steps
        self.src_width = src_width
        self.x_src_buf = x_src_buf
        self.d_bos = d_bos
        self.d_out = d_out
        self.batch = batch


@dataclass
class _DecodeSpec:
    """RDDLGNデコーダのIR構造(dec_l_rec/dec_pm_rec/enc_n_rec/enc_k_rec)を
    調べて得られる値・opをまとめたもの(バックエンド生成から分離)。

    `_discover_decode_spec`が作る。**コード生成を一切行わない純粋な調査**
    なので、従来のtickループ経路(`_emit_decode_tick_loop`)と
    megakernel経路(`deploy/rddlgn_decode_megakernel.py`)の両方がこれを
    共有できる——調査ロジックが2箇所に分岐してズレる事故を防ぐのが狙い。
    """

    plan: ExecPlan
    l_op: PlanOp
    pm_op: PlanOp
    steps: int
    l_ext_id: int
    l_ext_width: int
    l_out_id: int
    l_out_width: int
    p_state_id: int
    p_width: int
    p_new_id: int
    ctx_ext_id: int
    ctx_width: int
    l_in_ext_id: int
    l_in_width: int
    scores_id: int
    vocab: int
    enc_n_op: PlanOp
    enc_k_op: PlanOp
    context_src_id: int
    x_src_id: int
    src_width: int
    state_init_op: PlanOp
    d_emb: int
    table_bits: np.ndarray
    batch: int
    #: dec_l_rec/dec_pm_recのbody(scope付きop列、plan.ops内での出現順)。
    #: `backend._scoped[node_id]`(emit_body()が同じ走査で作るもの)と
    #: 常に同じ中身になる(megakernelはbackend状態に依存せずこちらを
    #: 直接使う)。
    l_scope_ops: list[PlanOp] = field(default_factory=list)
    pm_scope_ops: list[PlanOp] = field(default_factory=list)


@dataclass
class DecodeBuffers:
    """`_emit_decode_preamble`が確保する、アリーナ外の専用バッファ名
    (CPU版のcontext_local/p_prev_localに相当)。"""

    d_table: str
    d_context: str
    d_pstate: str
    d_token: str
    d_bos: str
    d_out: str


def _discover_decode_spec(plan: ExecPlan, embed_table: dict) -> _DecodeSpec:
    """decode生成(従来のtickループ・ megakernelどちらも)が必要とする
    値・opを`plan`から調べる。バックエンド非依存・副作用無し。"""
    batch = plan.batch

    l_op = _find_op(plan, 'recurrent_exec', _DEC_L)
    pm_op = _find_op(plan, 'recurrent_exec', _DEC_PM)
    steps = int(pm_op.attrs['steps'])
    if int(l_op.attrs['steps']) != steps:
        raise ValueError(
            f'dec_l_rec.steps({l_op.attrs["steps"]}) != '
            f'dec_pm_rec.steps({steps})')

    # dec_l_rec: ext_bufs = {'sv_step': ext_id}(state無し、幅=d_emb=埋め込み
    # 表引きの出力幅)。output_width(L-layers自体の出力幅、l_outの幅)とは
    # 別物なので取り違えないこと。
    (l_sv_id, l_ext_id), = l_op.attrs['ext_bufs'].items()
    l_ext_width = int(l_op.attrs['ext_widths'][l_sv_id])   # = d_emb
    l_out_id = l_op.attrs['output_source']
    l_out_width = int(l_op.attrs['output_width'])          # L-layers出力幅

    # dec_pm_rec: state_bufs={'sv_p': buf_id}、
    # ext_bufs={'sv_context': invariant, 'sv_l': step}。
    (p_sv_id, p_state_id), = pm_op.attrs['state_bufs'].items()
    p_width = int(plan.value(p_state_id).width)
    p_new_id = pm_op.attrs['state_update'][p_sv_id]

    ctx_sv_id = next(sv for sv, role in pm_op.attrs['ext_roles'].items()
                     if role == 'invariant')
    l_in_sv_id = next(sv for sv, role in pm_op.attrs['ext_roles'].items()
                      if role == 'step')
    ctx_ext_id = pm_op.attrs['ext_bufs'][ctx_sv_id]
    ctx_width = int(pm_op.attrs['ext_widths'][ctx_sv_id])
    l_in_ext_id = pm_op.attrs['ext_bufs'][l_in_sv_id]
    l_in_width = int(pm_op.attrs['ext_widths'][l_in_sv_id])
    if l_in_width != l_out_width:
        raise ValueError(
            f'dec_l_recの出力幅({l_out_width})とdec_pm_recのsv_l入力幅'
            f'({l_in_width})が不一致')

    scores_id = pm_op.attrs['output_source']
    vocab = int(pm_op.attrs['output_width'])

    enc_n_op = _find_op(plan, 'recurrent_exec', _ENC_N)
    enc_k_op = _find_op(plan, 'recurrent_exec', _ENC_K)
    # 注意: node_id='enc_k_rec'を持つopはrecurrent_exec以外にもある
    # (state_initがrecurrentノード自身のnode_idを引き継ぐため、_find_output
    # のような「最初に見つかったop」だとstate_initの出力(=stateバッファ、
    # まだ最終tick分の値が反映される前)を掴んでしまう——実際に
    # 踏んだバグ)。必ずrecurrent_exec op自身の.output
    # (=final_stateコピー後の値)を使う。
    context_src_id = enc_k_op.output
    if int(plan.value(context_src_id).width) != ctx_width:
        raise ValueError('encoder出力幅とdec_pm_recのsv_context幅が不一致')
    x_src_id = _find_output(plan, _X_SRC)
    src_width = int(plan.value(x_src_id).width)

    state_init_op = next(
        op for op in plan.ops
        if op.kind == 'state_init' and op.output == p_state_id)

    d_emb = int(embed_table['d_emb'])
    if d_emb != l_ext_width:
        raise ValueError(
            f'embed_table.d_emb({d_emb})とdec_l_recのsv_step幅'
            f'({l_ext_width})が不一致')
    table_bits = unpack_embed_table(embed_table)

    # emit_body()が`backend._scoped[node_id]`を作るのと同じ走査
    # (plan.ops出現順)。backend状態に依存せずここで独立に再現しておく
    # ことで、megakernel経路もbackend.emit_body()の内部実装に依存しない。
    l_scope_ops = [op for op in plan.ops if op.scope == _DEC_L]
    pm_scope_ops = [op for op in plan.ops if op.scope == _DEC_PM]

    return _DecodeSpec(
        plan=plan, l_op=l_op, pm_op=pm_op, steps=steps, l_ext_id=l_ext_id,
        l_ext_width=l_ext_width, l_out_id=l_out_id, l_out_width=l_out_width,
        p_state_id=p_state_id, p_width=p_width, p_new_id=p_new_id,
        ctx_ext_id=ctx_ext_id, ctx_width=ctx_width, l_in_ext_id=l_in_ext_id,
        l_in_width=l_in_width, scores_id=scores_id, vocab=vocab,
        enc_n_op=enc_n_op, enc_k_op=enc_k_op, context_src_id=context_src_id,
        x_src_id=x_src_id, src_width=src_width, state_init_op=state_init_op,
        d_emb=d_emb, table_bits=table_bits, batch=batch,
        l_scope_ops=l_scope_ops, pm_scope_ops=pm_scope_ops)


def _emit_decode_preamble(backend, spec: _DecodeSpec) -> DecodeBuffers:
    """アリーナ外の専用バッファ(CPU版のcontext_local/p_prev_localに相当)を
    確保し、`backend.body`へ「state初期化+bos複製+エンコーダ+context退避」
    (steps 1-4)を追記する(末尾追記、既存bodyは変更しない)。

    従来のtickループ経路・ megakernel経路のどちらもこの関数で作った
    `DecodeBuffers`(初期値の置き場)を読む——エンコーダ呼び出しは1回だけ、
    という前提はどちらの経路でも同じため。

    ここから先、`backend`の生成コード発行はすべてdecode専用
    (呼び出し側は既に`backend.emit_body()`で教師強制本体を発行し終えて
    いる)。`backend._dedup_uploads`をここでTrueへ切り替えることで、
    以降の`emit_recurrent_exec`(enc_n_rec/enc_k_rec、そして呼び出し側の
    `_emit_decode_tick_loop`/`rddlgn_decode_megakernel._prepare_layers`が
    処理するdec_l_rec/dec_pm_rec自身の層)が教師強制本体で発行済みの
    op(=同一の添字/真理値表/state初期値)を再度処理しても、
    `GpuBackend._cached_upload`が新規アップロードをスキップし既存の
    シンボルを再利用する(RDDLGN decode生成物がencoder/decoderの重み表を
    2組焼き込んでいた問題、本体)。フラグは`emit_body()`実行後にしか
    立たないため、**教師強制単体の生成(`generate_metal`/`generate_cuda`,
    decode無し)の出力は一切変わらない**。
    """
    backend._dedup_uploads = True  # noqa: SLF001 -- 同一サブシステム内の共有
    batch = spec.batch
    d_table = backend.up_u8('mdl_dec_embed_table',
                            spec.table_bits.reshape(-1).tolist())
    d_context = backend.alloc('d_dec_context',
                              max(4, spec.ctx_width * batch), 'u8')
    d_pstate = backend.alloc('d_dec_pstate', max(4, spec.p_width * batch),
                             'u8')
    d_token = backend.alloc('d_dec_token', max(4, 4 * batch), 'i32')
    d_bos = backend.alloc('d_dec_bos', 4, 'i32')
    d_out = backend.alloc('d_dec_out', max(4, 4 * batch * spec.steps), 'i32')

    # 1. state初期化(呼び出しごとに毎回。CPU版のmdl_reset_state_dec_pm_recと
    #    同じ——同じctx/バッファを使い回しても前回の生成が残らない)。
    # このbits配列は教師強制本体のdec_pm_rec state_init
    # (`emit_body()`が既に`emit_state_init`経由でアップロード済み)と
    # 中身が同一なので、`_cached_upload`経由にして重複アップロードを避ける
    # (コピー先はこの関数専用のシャドーバッファ`d_pstate`なので、
    # コピー呼び出し自体は従来どおり必要——重複排除の対象は定数配列のみ)。
    state_bits = [int(x) for x in spec.state_init_op.attrs['bits']]
    tag = backend.uid('decpinit')
    src = backend._cached_upload(  # noqa: SLF001 -- 同一サブシステム内の共有
        'up_u8', f'{tag}_bits', state_bits)
    backend.copy_batched_ex(d_pstate, '0', spec.p_width, src, '0', 0,
                            spec.p_width)

    # 2. bos_tokenをtoken_bufへ全レーン分ブロードキャスト
    #    (d_bosはエントリ関数がbos_token(実行時引数)を書き込み済み)。
    backend.copy_batched_ex(d_token, '0', 4, d_bos, '0', 0, 4)

    # 3. エンコーダ(x_srcはエントリ関数が直接アップロード済み)。
    #    教師強制と完全に同一の経路(emit_recurrent_execをそのまま再利用)
    #    —decodeで変わるのはデコーダ側だけ、という設計どおり。
    # K-layers(enc_k_rec)は真のリカレント(sv_state持ち)なので、教師強制の
    # mdl_model_run/mdl_cuda_run本体(呼び出し前のbackend.body、1プロセスで
    # 1回しか状態初期化されない)とは独立に、decode呼び出しのたびに明示的に
    # 初期状態へ戻す(_reset_recurrent_state参照。CPU版のmdl_run_recurrent_
    # enc_k_recが毎回mdl_reset_state_enc_k_recを呼ぶのと同じ理由)。
    #
    # **reset→execの対を1ノードぶんずつ交互に発行すること**(2ノード分の
    # resetをまとめて先出ししない)。メモリ計画(deploy/memplan.py)は
    # 教師強制の生成順(enc_n_recの全tick→enc_k_recのstate_init→
    # enc_k_recの全tick)を前提にアリーナを詰めており、enc_k_recのstate
    # バッファ(sv_state)はenc_n_recの内部スクラッチのアリーナ領域を
    # 再利用している可能性がある(生存区間がenc_n_recの実行位置より後で
    # 始まるため)。resetを先にまとめて出すと、その直後のenc_n_recの
    # tick実行がenc_k_recのreset結果を上書きしてしまう
    # ——実際に踏んだバグ(このファイル先頭のl_rec/pm_recの
    # save/restore規律と同種の理由)。
    _reset_recurrent_state(backend, spec.enc_n_op)
    backend.emit_recurrent_exec(spec.enc_n_op)
    _reset_recurrent_state(backend, spec.enc_k_op)
    backend.emit_recurrent_exec(spec.enc_k_op)

    # 4. contextをシャドーへコピー(以降dec_l_rec/dec_pm_recのtick評価が
    #    encoderの出力バッファを再利用してもcontextの値は失われない)。
    backend.copy_bytes(d_context, 0, backend.buf(spec.context_src_id),
                       backend.times_n(spec.ctx_width))

    return DecodeBuffers(d_table=d_table, d_context=d_context,
                         d_pstate=d_pstate, d_token=d_token, d_bos=d_bos,
                         d_out=d_out)


def _emit_decode_tick_loop(backend, spec: _DecodeSpec,
                           buffers: DecodeBuffers) -> None:
    """1tickぶんの本体(embed_lookup→L body→copy→PM body→copy→argmax→出力)を
    `for (t_...)` ループへ包んで`backend.body`の末尾へ追記する(従来
    経路)。megakernelが32KB超等でフォールバックするときも、この関数を
    そのまま再利用する(フォールバック時の生成物が1バイトも違わない
    ことを保証するため)。
    """
    call, fn, N = backend.call, backend.fn, backend.N

    # tickループ本体(1回だけPythonで発行し、生成コード側のfor(t_)ループで
    # steps回実行する——教師強制のemit_recurrent_execと同じ「bodyは
    # unrollしない」原則)。
    saved_body, backend.body = backend.body, []

    call(f'{fn("embed_lookup")}({buffers.d_token}, {buffers.d_table}, '
        f'{backend.buf(spec.l_ext_id)}, {spec.d_emb}, {N})')
    for sub in spec.l_scope_ops:
        getattr(backend, backend.EMITTERS[sub.kind])(sub)

    # l_out → pm_l_ext を最優先(他の書き込みが割り込む前に。CPU版の
    # step3と同じ順序 — pm_ctx_ext/pm_stateの書き戻し(6.)より先に行うのは、
    # それらがl_outのアリーナ領域をエイリアスしている可能性があるため
    # (このファイル先頭のコメント参照)。
    backend.copy_bytes(backend.buf(spec.l_in_ext_id), 0,
                       backend.buf(spec.l_out_id),
                       backend.times_n(spec.l_out_width))
    # 6. 永続入力(context/state)を使う直前に必ず書き戻す。
    backend.copy_bytes(backend.buf(spec.ctx_ext_id), 0, buffers.d_context,
                       backend.times_n(spec.ctx_width))
    backend.copy_bytes(backend.buf(spec.p_state_id), 0, buffers.d_pstate,
                       backend.times_n(spec.p_width))

    for sub in spec.pm_scope_ops:
        getattr(backend, backend.EMITTERS[sub.kind])(sub)

    # 7. 新state候補をシャドーへ即座に保存(次tickのl_rec再実行がこの
    #    アリーナ領域を踏んでも真の値はシャドーに残る)。
    backend.copy_bytes(buffers.d_pstate, 0, backend.buf(spec.p_new_id),
                       backend.times_n(spec.p_width))

    call(f'{fn("argmax")}({backend.buf(spec.scores_id)}, {buffers.d_token}, '
        f'{spec.vocab}, {N})')

    # 8. token_buf → out_tokens[t](tickごとの行。行幅はコンパイル時容量
    #    plan.batchで固定 — 呼び出し時のnがbatchより小さくても、次回
    #    別のnで呼んだときに行アドレスがずれない)。tick本体の一部として
    #    (backend.bodyがまだtick_linesを指している間に)追加する。
    backend.copy_bytes_expr(buffers.d_out, f't_ * {spec.batch} * 4',
                            buffers.d_token, '0', backend.times_n(4))

    tick_lines = backend.body
    backend.body = saved_body
    # emit_recurrent_execと同じ書き方(`for (int t_ = ...)`はC99のfor文
    # スコープなので、直前のエンコーダ用ループの `t_` とは独立)。
    backend.body.append(f'  for (int t_ = 0; t_ < {spec.steps}; t_++) {{')
    backend.body.extend(tick_lines)
    backend.body.append('  }')


def _build_decode_body(backend, embed_table: dict) -> DecodeInfo:
    """backend(既にdeclare_buffers/declare_preproc/emit_body済み)へ
    decode専用の定数バッファ・カーネル呼び出しを追加し、`backend.body`の
    末尾へ「1回の呼び出しぶん(state初期化+エンコーダ+tickループ)」の
    コードを積む。呼び出し前の`backend.body`(教師強制のmdl_model_run/
    mdl_cuda_run本体)は変更しない——この関数は`backend.body`を一時退避
    せず**末尾に追記する**ので、呼び出し側が事前に
    `backend.body = []`等で退避してから呼ぶこと(下のgenerate_*参照)。

    調査(`_discover_decode_spec`)・専用バッファ+エンコーダ
    (`_emit_decode_preamble`)・tickループ本体(`_emit_decode_tick_loop`)の
    3段に分離した(megakernel経路が同じ調査結果・同じ専用バッファ・
    同じtickループ生成コードを再利用できるようにするため)。生成される
    コードはリファクタ前と1バイトも変わらない。
    """
    spec = _discover_decode_spec(backend.plan, embed_table)
    buffers = _emit_decode_preamble(backend, spec)
    _emit_decode_tick_loop(backend, spec, buffers)

    return DecodeInfo(steps=spec.steps, src_width=spec.src_width,
                      x_src_buf=backend.buf(spec.x_src_id),
                      d_bos=buffers.d_bos, d_out=buffers.d_out,
                      batch=spec.batch)


def _metal_decode_entry(info: DecodeInfo, decode_lines: list[str]) -> str:
    body = '\n'.join(decode_lines)
    return f'''
/* ---- RDDLGN自己回帰生成(decode、opt-in) ---- */
#define MDL_STEPS_DECODE {info.steps}

extern "C" int mdl_metal_rddlgn_decode(const unsigned char *src_bits,
                                       int bos_token, int *out_tokens,
                                       int n) {{
  if (n < 1 || n > MDL_MAX_BATCH) return -1;
  if (mdl_model_init() != 0) return -1;
  mdl_metal_upload({info.x_src_buf}, src_bits, (size_t)n * {info.src_width});
  mdl_metal_upload({info.d_bos}, &bos_token, sizeof(int));
{body}
  MDL_TRY(mdl_metal_sync());
  mdl_metal_download(out_tokens, {info.d_out},
                     (size_t)MDL_STEPS_DECODE * MDL_MAX_BATCH * sizeof(int));
  return 0;
}}

extern "C" int mdl_metal_rddlgn_decode_steps(void) {{ return MDL_STEPS_DECODE; }}
'''


def _cuda_decode_entry(info: DecodeInfo, decode_lines: list[str]) -> str:
    body = '\n'.join(decode_lines)
    return f'''
/* ---- RDDLGN自己回帰生成(decode、opt-in) ---- */
#define MDL_STEPS_DECODE {info.steps}

extern "C" int mdl_cuda_rddlgn_decode(const unsigned char *src_bits,
                                      int bos_token, int *out_tokens, int n) {{
  if (n < 1 || n > MDL_MAX_BATCH) return -1;
  if (mdl_cuda_init() != 0) return -1;
  MDL_TRY(cudaMemcpy({info.x_src_buf}, src_bits, (size_t)n * {info.src_width},
                     cudaMemcpyHostToDevice));
  MDL_TRY(cudaMemcpy({info.d_bos}, &bos_token, sizeof(int),
                     cudaMemcpyHostToDevice));
{body}
  MDL_TRY_INT(mdl_cuda_sync());
  MDL_TRY(cudaMemcpy(out_tokens, {info.d_out},
                     (size_t)MDL_STEPS_DECODE * MDL_MAX_BATCH * sizeof(int),
                     cudaMemcpyDeviceToHost));
  return 0;
}}

extern "C" int mdl_cuda_rddlgn_decode_steps(void) {{ return MDL_STEPS_DECODE; }}
'''


def generate_rddlgn_decode_metal(bundle: dict, name: str = 'model_metal', *,
                                 batch: int = 1,
                                 megakernel: bool = True) -> dict[str, str]:
    """RDDLGNバンドル → decode込みのMetalソース(opt-in)。

    既存の`generate_metal`(教師強制、`mdl_metal_infer_batch`等)が生成する
    ものと**完全に同じ内容**へ、追加で`mdl_metal_rddlgn_decode`
    (自己回帰貪欲デコード)を1個生やす。既定の`generate_metal`は一切
    呼ばない・変更しない(このモジュール内で独立にビルドする)。

    Args:
        bundle: `deploy/bundle.py::build_bundle` の出力。
            `bundle['embed_table']`(`extract_embed_table`の出力)が
            必須——教師強制用バンドルには無いことがあるので、呼び出し側が
            `doc['embed_table'] = extract_embed_table(model.src_embedding)`
            してから`build_bundle`すること(tests/deploy/
            test_c_rddlgn_greedy.pyのgreedy_bundleフィクスチャと同じ)。
        batch: 同時に処理するレーン数の上限(「B系列レーン並列」)。
            実行時のnはn<=batchで可変。
        megakernel:、**既定True(既定化)**。decodeの`steps`
            ループ全体を1スレッドグループ=1レーンの単一カーネルへ畳む
            (`deploy/rddlgn_decode_megakernel.py`)。実機A/B(Metal
            1.61〜2.10倍・CUDA 1.71〜2.34倍、進捗記録
           参照)で**全実測が正**(悪化ケース無し)なため、
            判定表では「実測で常に勝つ」側として既定ONにした
            (dense_fuse/dense_packedのようなbatch閾値は不要)。
            threadgroupメモリ32KB・Metalバッファ引数31本を超えるモデルは
            生成時に自動的に従来経路へフォールバックする安全機構がある
            ため(ValueErrorにはならない——生成ソース中の
            `/* megakernel: ... */`コメントでどちらを使ったか分かる)、
            既定ONにしても壊れたコードが出ることはない。Falseで従来の
            tickループ経路(無改変)へ戻せる。`mdl_metal_rddlgn_decode`
            という外部シンボル名自体は`megakernel`の値によらず同じ
            (呼び出し側/ベンチはこの引数だけを切り替えればよい)。
    """
    embed_table = bundle.get('embed_table')
    if embed_table is None:
        raise ValueError(
            "bundle['embed_table']が無い(extract_embed_tableで作った"
            'embed_tableをdocへ追加してからbuild_bundleすること)')
    if megakernel:
        from .rddlgn_decode_megakernel import build_decode_metal_megakernel
        return build_decode_metal_megakernel(bundle, name, batch, embed_table)
    plan = build_plan(bundle, batch=batch)
    backend = MetalBackend(plan, model_name=name)
    backend.declare_buffers()
    backend.declare_preproc()
    backend.emit_body()
    saved, backend.body = backend.body, []
    info = _build_decode_body(backend, embed_table)
    decode_lines = backend.body
    backend.body = saved
    text = backend._source()  # noqa: SLF001 -- 同一パッケージ内、既存パターン踏襲
    text += _metal_decode_entry(info, decode_lines)
    return {f'{name}.mm': text}


def generate_rddlgn_decode_cuda(bundle: dict, name: str = 'model_cuda', *,
                                batch: int = 1,
                                megakernel: bool = True) -> dict[str, str]:
    """RDDLGNバンドル → decode込みのCUDAソース(opt-in)。

    `generate_rddlgn_decode_metal`のCUDA対称版。引数・制約は同じ
    (megakernel既定は**Trueへ変更**——実機A/Bで悪化ケース無し
    を確認済み、`generate_rddlgn_decode_metal`のdocstring参照)。
    """
    embed_table = bundle.get('embed_table')
    if embed_table is None:
        raise ValueError(
            "bundle['embed_table']が無い(extract_embed_tableで作った"
            'embed_tableをdocへ追加してからbuild_bundleすること)')
    if megakernel:
        from .rddlgn_decode_megakernel import build_decode_cuda_megakernel
        return build_decode_cuda_megakernel(bundle, name, batch, embed_table)
    plan = build_plan(bundle, batch=batch)
    backend = CudaBackend(plan, model_name=name)
    backend.declare_buffers()
    backend.declare_preproc()
    backend.emit_body()
    saved, backend.body = backend.body, []
    info = _build_decode_body(backend, embed_table)
    decode_lines = backend.body
    backend.body = saved
    text = backend._source()  # noqa: SLF001 -- 同一パッケージ内、既存パターン踏襲
    text += _cuda_decode_entry(info, decode_lines)
    return {f'{name}.cu': text}
