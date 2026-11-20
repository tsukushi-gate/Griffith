"""CUDA と Metal に共通するコード生成。

両者のホストAPIは `mdl_cuda_pack` / `mdl_metal_pack` のように**接頭辞だけが
違う対称な形**で意図的に揃えてある。そのため op → 呼び出し文の対応表は1本で
足り、サブクラスが埋めるのは本当に違う4点だけ:

    ・定数テーブルのアップロード(cudaMalloc+Memcpy / mdl_metal_alloc+upload)
    ・作業バッファの確保
    ・デバイス内コピー(cudaMemcpy D2D / blit encoder)
    ・conv のパラメータ渡し(CUDA は構造体に配列ポインタを持てる、Metal は持てない)

新しいGPUバックエンドを足すときも、埋めるのはこの4点になる。

**バッチ**: 起動スレッド数はすべて実行時の枚数 `n` から決める
(バッファは `plan.batch` 枚ぶん確保してあり、`n <= plan.batch`)。
ビットプレーンの並びが `[batch][C][H][wpr]` なので、**チャネル方向に独立な
演算はバッチ透過**で、チャネル数に `n` を掛けるだけでよい —
pack / unpack / or_pool / upsample / spatial_group_sum / group_sum が該当する。
枚ごとに記述子や順列表を引き直す必要がある preproc / permute / lut / conv
だけが `n` を明示的に受け取る。
"""

from .layout import ConvLayout, ConvLayoutSplit
from .memplan import check_no_overlap, plan_memory
from .plan import (
    CUDA_DENSE_FUSE_MAX_SHARED_BYTES,
    DENSE_FUSE_MAX_SHARED_BYTES,
    KIND_BITPLANE,
    KIND_FLAT,
    KIND_SCORES,
    KIND_SCRATCH,
    ExecPlan,
    UnsupportedNode,
    build_plan,
    dense_packed_eligible,
    lut_dense_indices_fit_u16,
    resolve_auto_config,
)
from .preproc_desc import threshold_tables


def resolve_generate_config(bundle: dict, *, batch: int, backend: str,
                            conv_engine: str | None, dense_fuse: bool | None,
                            dense_packed: bool | None, auto: str):
    """`generate_metal`/`generate_cuda`が共有する「明示引数+自動選択の
    合成」ロジック。

    `conv_engine`/`dense_fuse`/`dense_packed`は`None`なら「呼び出し側が
    明示的に指定しなかった」ことを表す(=`auto`の判断に委ねる)。
    `None`以外の値(`True`/`False`/`'split'`/`'combined'`)は常にそのまま
    尊重する(明示が勝つ、タスク指示)。

    `dense_fuse`と`dense_packed`はどちらか一方だけが有効になりうる
    (生成側で併用不可)。呼び出し側がどちらか一方だけを明示的にTrueへ
    したときは、**もう一方は自動ONの対象から外す**(auto側が勝手に
    もう一方までTrueにして併用エラーを起こさないようにするため)。

    Args:
        bundle: 配布バンドル。
        batch: `build_plan`/`resolve_auto_config`へそのまま渡すバッチ数。
        backend: `'metal'` または `'cuda'`。
        conv_engine, dense_fuse, dense_packed: 呼び出し側の生引数
            (`None`=未指定)。
        auto: `'measured'`(自動選択ON)または`'off'`(未指定引数は旧来の
            既定値 `conv_engine='split'`/`dense_fuse=False`/
            `dense_packed=False` のまま=旧版の挙動)。

    Returns:
        `(plan, conv_engine, dense_fuse, dense_packed, decision)`。
        `decision`は`resolve_auto_config`の戻り値(呼ばなかった場合は
        `None` ─ `auto='off'`のとき、または3引数とも明示済みのとき)。

    Raises:
        ValueError: `auto`が`'measured'`/`'off'`以外、または
            `dense_packed`と`dense_fuse`がどちらも明示的に`True`
    """
    if auto not in ('measured', 'off'):
        raise ValueError(f"auto={auto!r} は未対応('measured'/'off'のみ)")
    if dense_packed is True and dense_fuse is True:
        raise ValueError(
            'dense_packed と dense_fuse の併用はv1では未対応'
            '(タスク指示のスコープ外。併用は後続タスク)')

    probe_plan = build_plan(bundle, batch=batch)
    decision = None
    need_auto = auto == 'measured' and (
        conv_engine is None or dense_fuse is None or dense_packed is None)
    if need_auto:
        decision = resolve_auto_config(probe_plan, batch=batch, backend=backend)

    resolved_conv_engine = (
        conv_engine if conv_engine is not None
        else (decision['conv_engine'] if decision else 'split'))

    if dense_packed is True or dense_fuse is True:
        # 明示的にどちらか一方をONにしたら、もう一方は未指定でも自動ONの
        # 対象から外す(併用不可のため)。
        resolved_dense_fuse = bool(dense_fuse)
        resolved_dense_packed = bool(dense_packed)
        if decision is not None:
            # decision['reasons']はauto単独の判断(=もう一方を考慮しない
            # 仮定の理由)を持ったままなので、実際に強制されたことが
            # コメントから読み取れるよう上書きする。
            if dense_packed is True:
                decision = {**decision, 'reasons': {
                    **decision['reasons'],
                    'dense_fuse': ('dense_packedが明示的にTrueのため見送り'
                                   '(併用不可)'),
                }}
            else:
                decision = {**decision, 'reasons': {
                    **decision['reasons'],
                    'dense_packed': ('dense_fuseが明示的にTrueのため見送り'
                                     '(併用不可)'),
                }}
    else:
        resolved_dense_fuse = (
            dense_fuse if dense_fuse is not None
            else (decision['dense_fuse'] if decision else False))
        resolved_dense_packed = (
            dense_packed if dense_packed is not None
            else (decision['dense_packed'] if decision else False))

    if resolved_dense_fuse:
        max_shared_bytes = (CUDA_DENSE_FUSE_MAX_SHARED_BYTES
                            if backend == 'cuda' else DENSE_FUSE_MAX_SHARED_BYTES)
        plan = build_plan(bundle, batch=batch, dense_fuse=True,
                          dense_fuse_max_shared_bytes=max_shared_bytes)
    else:
        plan = probe_plan

    return plan, resolved_conv_engine, resolved_dense_fuse, resolved_dense_packed, decision


def auto_config_header_comment(backend: str, auto: str, conv_engine: str,
                               dense_fuse: bool, dense_packed: bool,
                               decision: dict | None) -> str:
    """生成ソース冒頭へ埋め込む、自動選択の結果+理由のコメント。

    `md deploy info`相当の情報を生成物自体からも読めるようにする
    (タスク指示: 「選択結果と理由を生成ソース冒頭コメント...で確認
    できる形に」)。
    """
    lines = [f'/* auto-config({backend}, auto={auto!r}):',
            f'     conv_engine={conv_engine!r} dense_fuse={dense_fuse} '
            f'dense_packed={dense_packed}']
    if decision is not None:
        for key in ('conv_engine', 'dense_fuse', 'dense_packed'):
            lines.append(f'     - {key}: {decision["reasons"][key]}')
    elif auto == 'off':
        lines.append('     - auto=off のため判定表は評価していない'
                     '(未指定引数は旧来の既定値のまま)')
    else:
        lines.append('     - conv_engine/dense_fuse/dense_packedを全て明示'
                     '指定したため判定表は評価していない(明示が勝つ)')
    lines.append('*/\n')
    return '\n'.join(lines)


class GpuBackend:
    """プラン → GPUソース。サブクラスは接頭辞と4つの原始操作を与える。"""

    name = ''
    prefix = ''          # 'cuda' | 'metal' — 呼び出し名 mdl_<prefix>_*

    #: plane_op / reduce_scores の下位種別(op種別は `EMITTERS` から導く)
    SUB_OPS = frozenset({
        'or_pool2x2', 'upsample2d_x2', 'permute_channels', 'concat_planes',
        'spatial_group_sum', 'group_sum',
    })

    #: conv2dの実行方式。'split'(既定): ocごとの命令列分割
    #: (mdl_*_conv_exec_split、並列度=(batch,oc,ワード位置))。
    #: 'combined': 旧来の全oc結合(mdl_*_conv_exec、並列度=(batch,ワード位置))。
    #: どちらを使うかは生成時に確定させ、ランタイム分岐(環境変数等)は
    #: 作らない方針(タスク指示)。新旧の出力一致テスト用に選べるように
    #: してあるだけで、通常のデプロイは既定の'split'のままでよい。
    CONV_ENGINES = frozenset({'split', 'combined'})

    def __init__(self, plan: ExecPlan, model_name: str,
                conv_engine: str = 'split', conv_codegen: bool = False,
                dense_packed: bool = False):
        if conv_engine not in self.CONV_ENGINES:
            raise ValueError(
                f'conv_engine={conv_engine!r} は未対応(選べるのは '
                f'{sorted(self.CONV_ENGINES)})')
        if conv_codegen and conv_engine != 'split':
            raise ValueError(
                "conv_codegen=Trueは conv_engine='split' のときだけ選べる"
                f'(conv_engine={conv_engine!r} が指定されている)。'
                'モデル特化コード生成はoc分割レイアウト前提のため')
        self.plan = plan
        self.model_name = model_name
        self.conv_engine = conv_engine
        #: モデル特化MSL/CUDA生成。既定False=インタプリタ
        #: (`mdl_*_conv_exec_split`)のまま。Trueならttをコンパイル時定数、
        #: スロットをスカラーレジスタへ展開した直線コードを生成する
        #: (`convcodegen.py`)。既定動作は変えない(opt-in)。
        self.conv_codegen = conv_codegen
        #: dense のバッチ軸ビットプレーン化。既定False=層ごとに
        #: 1スレッド1ゲート(バッチはただの平坦展開)のまま。Trueなら
        #: `emit_dense_packed_body`が主経路(lut_dense/group_sumのみ)を
        #: 32サンプル/語のパック表現で評価する`emit_body`の代替経路へ
        #: 差し替える(`build()`側の分岐、MetalBackend/CudaBackend共通)。
        #: 適格性(batch>=32・全主経路opがlut_dense/group_sumのみ・
        #: conv/recurrent非混在)はここで即座に検証する(生成時に落とす)。
        self.dense_packed = dense_packed
        if dense_packed:
            dense_packed_eligible(plan)
        #: `buf_packed`が作るパック済みu32エイリアスのキャッシュ
        #: (値id → 呼び出し名)。同じ値を複数回パック参照しても
        #: view/allocを重複発行しない。
        self._packed_view_cache: dict[int, str] = {}
        #:: `_cached_upload`が使う「アップロード済み定数テーブル」の
        #: 記録(キー=(up_*のメソッド名, 値のtuple) → 返ってきたシンボル名)。
        #: `_dedup_uploads`の値に関わらず**常に記録**する(記録自体は生成物へ
        #: 一切影響しない副作用フリーな操作なので、既定の教師強制生成でも
        #: 安全に動かしておける)。
        self._upload_cache: dict[tuple, str] = {}
        #:: `_cached_upload`が既存シンボルの再利用を試みるかどうか。
        #: 既定False——`emit_body()`(教師強制の主経路)はこのフラグに一切
        #: 触れないので、既定の`generate_metal`/`generate_cuda`(decode無し)
        #: の生成物は本フラグの存在下でも従来と1バイトも変わらない。
        #: RDDLGN decode生成(`rddlgn_decode_gpu.py`)が、教師強制の
        #: `emit_body()`実行後・decode専用コード生成の直前でTrueへ切り替える
        #: ——decodeはencoder(enc_n_rec/enc_k_rec)や自身のL/PM層のbodyを
        #: `emit_recurrent_exec`経由でもう一度発行するため(教師強制の
        #: 本体で1回・decode本体で1回、計2回同じop=同じ重みが処理される)、
        #: 2回目はここでキャッシュヒットさせて新規アップロードをスキップする
        #: (「アップロード表の重複排除」タスク実体)。
        self._dedup_uploads = False
        self.body: list[str] = []
        self._n = 0
        self.image_buffer: str | None = None
        # 退化ゲート除去のアキュムレータalias(PlanValue.alias_of)
        # はCバックエンドの `_declare_buffers` でのみ「親バッファ+オフセット」
        # へ変換される(runtime/c/mdl_kernels.cはuint8_t*ポインタ演算で
        # 素直に表せるため)。GPUの `declare_buffers` はこれを解釈せず、
        # `mem.offsets` に無い値をこれまで通り**独立**バッファとして
        # 確保してしまう — その独立バッファはlut_denseの書き込み先には
        # なるが、後続層/境界層が読む親バッファには反映されず、黙って
        # 誤った値になる(perf-huntスキルの「同じバグでもバックエンドで
        # 出方が違う」の実例)。対応するまでは明示的に拒否する。
        aliased = [v.id for v in plan.values if v.alias_of is not None]
        if aliased:
            raise UnsupportedNode(
                '(退化ゲート除去)のaccumulate-group'
                f'(alias化された値id={aliased})はCUDA/Metalバックエンド'
                '未対応です。plan.py::_plan_accumulate_concatが作る'
                'alias_of付きPlanValueはCバックエンドの_declare_buffersだけが'
                '解釈します。optimize_degenerate=Trueで生成したRDDLGNバンドル'
                'をGPU向けにビルドする場合は無効化するか、GPU側の'
                'alias対応(declare_buffersでの親+オフセット参照)を'
                '別途追加してください。',
                backend=self.name)
        # GPU側は入力も出力もこちらのデバイスバッファなので、全部アリーナに
        # 載せられる(CPU側は呼び出し側のメモリなので載せられない)。
        # バッチ倍になる領域なので、ここを畳んでおく価値が大きい。
        self.memory = plan_memory(plan)
        check_no_overlap(plan, self.memory)
        #: scope(recurrentノードid)ごとのop列。emit_body()の先頭で
        #: 1回だけ分類し、emit_recurrent_execが自分のscopeぶんを取り出す
        #: (cbackend.CBackend._scopedと同じ役割)。
        self._scoped: dict[str, list] = {}

    # -- サブクラスが実装する原始操作 ---------------------------------
    def up_i32(self, name: str, values) -> str:
        raise NotImplementedError

    def up_u32(self, name: str, values) -> str:
        raise NotImplementedError

    def up_i16(self, name: str, values) -> str:
        raise NotImplementedError

    def up_u16(self, name: str, values) -> str:
        """u16添字テーブルのアップロード。lut_dense専用。"""
        raise NotImplementedError

    def up_u8(self, name: str, values) -> str:
        raise NotImplementedError

    def alloc(self, name: str, n_bytes: int, elem: str) -> str:
        """デバイスバッファを確保する。elem は 'u32' | 'u8' | 'i32'。"""
        raise NotImplementedError

    def view(self, name: str, arena: str, off_words: int, elem: str) -> str:
        """アリーナの `off_words` ワード目を指す名前を作る。"""
        raise NotImplementedError

    def guard(self, call: str) -> str:
        """呼び出し文をエラーチェックで包む。"""
        raise NotImplementedError

    def emit_gate_program(self, op) -> None:
        """conv の呼び出し。パラメータの渡し方だけが両者で違う。"""
        raise NotImplementedError

    def emit_dense_fuse(self, op) -> None:
        """dense多層融合カーネルの呼び出し。カーネル生成そのものは
        `densefuse.py` に共有ロジックがあるが、バッファの確保・アップロード・
        起動方法(Metalは名前引きディスパッチ、CUDAは直接`<<<>>>`)が
        バックエンドごとに違うのでサブクラスが実装する。"""
        raise NotImplementedError

    def scratch_of(self, op) -> str:
        return self.buf(op.attrs['scratch'][0])

    # -- 共通 ---------------------------------------------------------
    def uid(self, prefix: str) -> str:
        self._n += 1
        return f'{prefix}{self._n}'

    def _cached_upload(self, method_name: str, name: str, values) -> str:
        """`up_{i32,u32,u16,u8}`の重複排除版。

        `self._dedup_uploads`が真かつ`(method_name, 値の中身)`が過去に
        アップロード済みなら、新規デバイスバッファ・静的配列を発行せず
        既存のシンボル名をそのまま返す。`_dedup_uploads`が偽の間(既定、
        教師強制の`emit_body()`が動いている間)は常にキャッシュ照会をせず
        実際にアップロードする——**したがって既定の`generate_metal`/
        `generate_cuda`(decode無し)の出力は本メソッドの存在下でも1バイトも
        変わらない**。アップロード結果は`_dedup_uploads`の値に関わらず毎回
        記録するので、後段(decode専用コード、`_dedup_uploads=True`)から
        教師強制本体で発行済みの同一内容テーブルを見つけて再利用できる。

        呼び出し側の`name`(=`uid()`で採番したタグ込みの名前)はキャッシュ
        ミス時のみ実際に使われる——ヒット時は無駄になる(採番だけ消費して
        シンボルは作られない)が、`uid()`の連番が飛ぶだけで実害は無い。
        """
        key = (method_name, tuple(int(v) for v in values))
        if self._dedup_uploads:
            cached = self._upload_cache.get(key)
            if cached is not None:
                return cached
        symbol = getattr(self, method_name)(name, values)
        self._upload_cache[key] = symbol
        return symbol

    def call(self, text: str) -> None:
        self.body.append(self.guard(text))

    #: 生成コード中で「この呼び出しで流す枚数」を指す実行時変数名。
    #: 定数ではなく変数にしてあるので、確保した上限より少ない枚数でも
    #: 起動スレッド数がそのぶん減る。
    N = 'n'

    def fn(self, op_name: str) -> str:
        return f'mdl_{self.prefix}_{op_name}'

    def times_n(self, value) -> str:
        """バッチ透過な演算のカウントに枚数を掛ける。"""
        return f'{value} * {self.N}'

    def buf(self, vid: int) -> str:
        if vid == self.plan.input_value:
            return 'd_mdl_input'
        if vid == self.plan.input_bitplane:
            return 'd_mdl_preproc_out'
        return f'd_v{vid}'

    ELEM_OF = {
        KIND_BITPLANE: 'u32', KIND_SCRATCH: 'u32', KIND_FLAT: 'u8',
        KIND_SCORES: 'i32',
    }

    #: 値種別ごとの1要素バイト数(recurrentのstacked/final_state出力コピーが
    #: KIND_FLAT(1バイト)とKIND_SCORES(int32=4バイト)の両方を要素単位で
    #: 正しく扱うために使う。、cbackendの`buf()`が返す型付きポインタの
    #: 添字アクセスに相当する明示版)。
    BYTES_OF = {KIND_FLAT: 1, KIND_SCORES: 4}

    def elem_bytes(self, vid: int) -> int:
        kind = self.plan.value(vid).kind
        n = self.BYTES_OF.get(kind)
        if n is None:
            raise UnsupportedNode(
                f'recurrent出力の値種別{kind}は未対応(KIND_FLAT/KIND_SCORESのみ)',
                backend=self.name)
        return n

    def declare_buffers(self) -> None:
        """1本のアリーナを確保し、値をその中のオフセットへ写す。"""
        mem = self.memory
        self.alloc('d_mdl_arena', max(4, mem.arena_words * 4), 'u32')
        for v in self.plan.values:
            if v.id not in mem.offsets:      # 念のため(通常は全部入る)
                self.alloc(self.buf(v.id), max(4, v.nbytes()),
                           self.ELEM_OF.get(v.kind, 'i32'))
                continue
            self.view(self.buf(v.id), 'd_mdl_arena', mem.offsets[v.id],
                      self.ELEM_OF.get(v.kind, 'i32'))

    def declare_preproc(self) -> None:
        """前処理の定数表。CPU版が実行時に作るしきい値表は生成時に焼き込む
        (デバイス側から64bit演算を消すため)。"""
        p = self.plan.preproc
        if p is None:
            return
        self.up_i32('mdl_preproc_desc', p['desc'])
        self.up_u32('mdl_preproc_qval', p['qval'])
        self.up_i16('mdl_preproc_thr', threshold_tables(p))
        # 画像バッファは値グラフの外(入口が書き込む)なので、
        # アリーナではなく個別に確保する。ここもバッチ倍が要る。
        n_img = p['image_channels'] * p['height'] * p['width']
        self.image_buffer = self.alloc('d_mdl_image',
                                       n_img * self.plan.batch, 'u8')

    # -- op ごとのコード生成 -------------------------------------------
    def emit_pack(self, op) -> None:
        a = op.attrs
        self.call(f'{self.fn("pack")}({self.buf(op.inputs[0])}, '
                  f'{self.buf(op.output)}, {self.times_n(a["c"])}, '
                  f'{a["h"]}, {a["w"]})')

    def emit_unpack(self, op) -> None:
        a = op.attrs
        self.call(f'{self.fn("unpack")}({self.buf(op.inputs[0])}, '
                  f'{self.buf(op.output)}, {self.times_n(a["c"])}, '
                  f'{a["h"]}, {a["w"]})')

    def emit_plane_op(self, op) -> None:
        a = op.attrs
        kind = a['op']
        src = self.buf(op.inputs[0])
        dst = self.buf(op.output)
        if kind == 'or_pool2x2':
            self.call(f'{self.fn("or_pool")}({src}, {dst}, '
                      f'{self.times_n(a["c"])}, {a["h"]}, {a["w"]})')
        elif kind == 'upsample2d_x2':
            self.call(f'{self.fn("upsample")}({src}, {dst}, '
                      f'{self.times_n(a["c"])}, {a["h"]}, {a["w"]})')
        elif kind == 'permute_channels':
            tag = self.uid('perm')
            order = self.up_i32(f'{tag}_order', a['order'])
            # 順列表は1枚ぶんなので、枚数はカーネルへ明示的に渡す
            self.call(f'{self.fn("permute")}({src}, {dst}, {order}, '
                      f'{a["c_out"]}, {a["h"]}, {a["wpr"]}, {self.N})')
        elif kind == 'concat_planes':
            # チャネル方向の連結は連続領域への追記なので、カーネルではなく
            # デバイス内コピーで足りる。バッチがあると1枚ごとに宛先が
            # 飛ぶので、枚数ぶんループを生成する。
            total = sum(c * a['h'] * a['wpr'] for c in a['c_in'])
            offset = 0
            for vid, c_in in zip(op.inputs, a['c_in'], strict=True):
                words = c_in * a['h'] * a['wpr']
                self.copy_batched(self.buf(op.output), offset * 4, total * 4,
                                  self.buf(vid), words * 4)
                offset += words
        else:
            raise UnsupportedNode(f'plane_op {kind} は未対応',
                                  node_id=op.node_id, node_type=op.node_type,
                                  backend=self.name)

    def emit_reduce_scores(self, op) -> None:
        a = op.attrs
        src = self.buf(op.inputs[0])
        dst = self.buf(op.output)
        if a['op'] == 'spatial_group_sum':
            self.call(f'{self.fn("spatial_group_sum")}({src}, {dst}, '
                      f'{self.times_n(a["n_out"])}, {a["gs"]}, '
                      f'{a["h"]}, {a["w"]})')
        elif a['op'] == 'group_sum':
            self.call(f'{self.fn("group_sum")}({src}, {dst}, '
                      f'{self.times_n(a["in_width"])}, '
                      f'{self.times_n(a["k"])})')
        else:
            raise UnsupportedNode(f'reduce_scores {a["op"]} は未対応',
                                  node_id=op.node_id, node_type=op.node_type,
                                  backend=self.name)

    def emit_lut_dense(self, op) -> None:
        a = op.attrs
        tag = self.uid('lut')
        # 添字表は1枚ぶんなので、1枚あたりの入力幅と枚数を渡す
        in_width = self.plan.value(op.inputs[0]).width
        # 添字はこの層の入力(in_width)内の位置を指すので、in_width<=65536
        # ならuint16_tへ半減できる(cbackend.emit_lut_denseと同じ判定)。
        u16 = lut_dense_indices_fit_u16(in_width)
        # `_cached_upload`経由(dedup無効時は従来のup_*直呼びと完全に
        # 同じ)。同一opの添字/真理値表が二重に発行される事故(RDDLGN
        # decode生成、rddlgn_decode_gpu.py参照)を1箇所で塞ぐ。
        if u16:
            idx = self._cached_upload('up_u16', f'{tag}_idx', a['indices'])
        else:
            idx = self._cached_upload('up_i32', f'{tag}_idx', a['indices'])
        tt = self._cached_upload('up_u32', f'{tag}_tt', a['truth_tables'])
        kernel = 'lut_layer_u16' if u16 else 'lut_layer'
        self.call(f'{self.fn(kernel)}({self.buf(op.inputs[0])}, '
                  f'{self.buf(op.output)}, {a["n_inputs"]}, '
                  f'{a["out_width"]}, {idx}, {tt}, {in_width}, {self.N})')

    #:: lut_flat の実行方式。'fused'=1スレッドグループ(ブロック)=1枚で
    #: 信号表を共有メモリに置く単一起動、'levels'=段ごとに1起動(信号表は
    #: デバイスメモリのスクラッチ)、'auto'=信号表が共有メモリ上限に収まれば
    #: fused。上限は `LUT_FLAT_FUSED_MAX_BYTES`(サブクラスが与える)。
    #: 'packed'=段ごと起動のバッチ軸パック版(信号表を32枚/語のu32で持つ。
    #: plan.batch>=32 のときだけ。スクラッチに収まることを生成時に確かめる)。
    LUT_FLAT_MODES = frozenset({'auto', 'fused', 'levels', 'packed'})
    LUT_FLAT_FUSED_MAX_BYTES = 0
    lut_flat_mode = 'auto'

    def emit_lut_flat(self, op) -> None:
        """縮約済み dense 平坦ネットリスト。

        生成時に `flatlevels.levelize_lut_flat` で ASAP の段ごとにゲートを
        並べ替え(同じ段のゲートは互いに依存しない)、参照される入力だけを
        信号表へ詰め直す。融合版は起動1回(段の間はブロック内バリア)、
        段ごと版は init → 段の数だけ level → out。
        """
        from .flatlevels import levelize_lut_flat

        a = op.attrs
        lf = levelize_lut_flat({'in_width': a['in_width'],
                                'out_width': a['out_width'],
                                'gates': [a['gates'][i:i + 3] for i in
                                          range(0, len(a['gates']), 3)],
                                'outputs': a['outputs']})
        tag = self.uid('flat')
        tt = self._cached_upload('up_u8', f'{tag}_tt', lf.tt)
        ga = self._cached_upload('up_i32', f'{tag}_a', lf.a)
        gb = self._cached_upload('up_i32', f'{tag}_b', lf.b)
        go = self._cached_upload('up_i32', f'{tag}_o', lf.outputs)
        gi = self._cached_upload('up_i32', f'{tag}_in', lf.in_idx)
        src = self.buf(op.inputs[0])
        dst = self.buf(op.output)
        mode = self.lut_flat_mode
        if mode == 'auto':
            mode = ('fused' if lf.n_signals <= self.LUT_FLAT_FUSED_MAX_BYTES
                    else 'levels')
        if mode == 'fused':
            if lf.n_signals > self.LUT_FLAT_FUSED_MAX_BYTES:
                raise UnsupportedNode(
                    f'lut_flat の信号表 {lf.n_signals}B が共有メモリ上限 '
                    f'{self.LUT_FLAT_FUSED_MAX_BYTES}B を超えるので fused は'
                    '使えない(levels を使う)',
                    node_id=op.node_id, node_type=op.node_type,
                    backend=self.name)
            lv = self._cached_upload('up_i32', f'{tag}_lv', lf.level_off)
            self.call(f'{self.fn("lut_flat_fused")}({src}, {dst}, {tt}, {ga}, '
                      f'{gb}, {go}, {gi}, {lv}, {lf.n_in}, {a["in_width"]}, '
                      f'{a["out_width"]}, {lf.n_levels}, {lf.n_signals}, '
                      f'{self.N})')
            return
        # 段ごと版: 信号表はプランが確保したスクラッチ(1枚 4*words バイト)
        sig = self.scratch_of(op)
        stride = self.plan.value(op.attrs['scratch'][0]).width * 4
        if mode == 'packed':
            bw_max = (self.plan.batch + 31) // 32
            if self.plan.batch < 32 or lf.n_signals * bw_max * 4 > \
                    stride * self.plan.batch:
                raise UnsupportedNode(
                    'lut_flat の packed は batch>=32 が必要(信号表 '
                    f'{lf.n_signals}×{bw_max}語 がスクラッチに収まらない)',
                    node_id=op.node_id, node_type=op.node_type,
                    backend=self.name)
            self.call(f'{self.fn("lut_flat_pk_init")}({src}, {sig}, {gi}, '
                      f'{lf.n_in}, {a["in_width"]}, {self.N})')
            for g0, w in zip(lf.level_off[:-1], lf.level_widths, strict=True):
                self.call(f'{self.fn("lut_flat_pk_level")}({sig}, {tt}, {ga}, '
                          f'{gb}, {lf.n_in}, {g0}, {w}, {self.N})')
            self.call(f'{self.fn("lut_flat_pk_out")}({sig}, {dst}, {go}, '
                      f'{a["out_width"]}, {self.N})')
            return
        if lf.n_signals > stride:     # levelize は信号数を増やさない(念のため)
            raise UnsupportedNode('lut_flat のスクラッチが信号表より小さい',
                                  node_id=op.node_id, backend=self.name)
        self.call(f'{self.fn("lut_flat_init")}({src}, {sig}, {gi}, '
                  f'{lf.n_in}, {a["in_width"]}, {stride}, {self.N})')
        for g0, w in zip(lf.level_off[:-1], lf.level_widths, strict=True):
            self.call(f'{self.fn("lut_flat_level")}({sig}, {tt}, {ga}, {gb}, '
                      f'{lf.n_in}, {stride}, {g0}, {w}, {self.N})')
        self.call(f'{self.fn("lut_flat_out")}({sig}, {dst}, {go}, '
                  f'{a["out_width"]}, {stride}, {self.N})')

    def emit_copy_scores(self, op) -> None:
        total = sum(op.attrs['widths'])
        offset = 0
        for vid, width in zip(op.inputs, op.attrs['widths'], strict=True):
            self.copy_batched(self.buf(op.output), offset * 4, total * 4,
                              self.buf(vid), width * 4)
            offset += width

    def emit_to_scores(self, op) -> None:
        # 0/1 をそのままスコアにするのは「グループ幅1の group_sum」と同じ。
        # 専用カーネルを増やさずに済む。
        width = op.attrs['width']
        self.call(f'{self.fn("group_sum")}({self.buf(op.inputs[0])}, '
                  f'{self.buf(op.output)}, {self.times_n(width)}, '
                  f'{self.times_n(width)})')

    def copy_batched(self, dst: str, dst_off: int, dst_stride: int,
                     src: str, n_bytes: int) -> None:
        """枚ごとに宛先が飛ぶコピー(concat)。枚数ぶんループを生成する。"""
        self.copy_batched_ex(dst, str(dst_off), dst_stride, src, '0',
                             n_bytes, n_bytes)

    def copy_batched_ex(self, dst: str, dst_off_expr: str, dst_stride: int,
                        src: str, src_off_expr: str, src_stride: int,
                        n_bytes: int) -> None:
        """src/dstでストライドが異なる版(recurrentのext入力
        コピーイン・stacked出力書き出しに使う)。

        オフセットは実行時の式(tを含んでよい)を文字列で渡す。`copy_batched`
        はこれの特殊形(src_off_expr='0', src_stride=n_bytes)。
        `src_stride=0` を渡すと全枚へ同じ範囲を複製する(recurrentの
        state_initが定数配列をbatch枚ぶんへ複製するのに使う)。

        batch>1のときは`copy_strided`(1回のカーネル起動でバッチ全枚を
        運ぶ)を使う。以前はホスト側`for(bi_)`ループでバッチ枚数ぶん
        `copy_bytes_expr`(=1回のD2Dコピー)をissueしていたが、Metalは
        D2Dコピー1回ごとにcompute encoderをendEncodingしてblitを張り直す
        ため、batchが大きい(RDDLGN decodeのmegakernel検討で判明した
        lanes=1024等)とエンコーダ切替がそのままbatch倍のオーバーヘッドに
        なっていた。教師強制経路(emit_recurrent_execのext/stacked出力
        コピー、state_initの複製)も含め`copy_batched_ex`を使う全呼び出しに
        共通で効く(タスク指示: 既定経路の改善)。
        """
        if self.plan.batch == 1:
            self.copy_bytes_expr(dst, dst_off_expr, src, src_off_expr,
                                 n_bytes)
            return
        self.copy_strided(dst, dst_off_expr, dst_stride, src, src_off_expr,
                          src_stride, n_bytes)

    def copy_bytes(self, dst: str, dst_off: int, src: str,
                   n_bytes: int) -> None:
        self.copy_bytes_expr(dst, str(dst_off), src, '0', n_bytes)

    def copy_bytes_expr(self, dst, dst_off, src, src_off, n_bytes) -> None:
        """デバイス内コピー。オフセットは実行時変数を含む式でよい
        (**サブクラスが実装する4点のうちの1つ**)。"""
        raise NotImplementedError

    def copy_strided(self, dst: str, dst_off_expr: str, dst_stride: int,
                     src: str, src_off_expr: str, src_stride: int,
                     n_bytes: int) -> None:
        """`copy_batched_ex`のbatch>1経路が使うバッチ一括コピー
        (1回のカーネル起動でバッチ全枚を運ぶ。ホスト側`for`ループでの
        N回issueをやめる)。オフセットは実行時の式(tを含んでよい)。
        サブクラス(Metal/CUDA)が実装する。"""
        raise NotImplementedError

    # -- recurrent --------------------------------------------
    def emit_flat_concat(self, op) -> None:
        """非パックの平坦ビット列同士の連結(recurrentのbody内、state+step等)。

        concat_planes / copy_scores と同じく**バイト範囲の連結でしかない**
        (KIND_FLATは1要素=1バイト)。両者ともGPU側は新しい計算式のカーネルを
        持たずデバイス内コピーで足りているので、flat_concatも同じ流儀に倣う
        (計算式を持たないぶん、CUDA/Metalに書き分ける余地自体が無い)。
        """
        a = op.attrs
        total = sum(a['widths'])
        offset = 0
        for vid, width in zip(op.inputs, a['widths'], strict=True):
            self.copy_batched(self.buf(op.output), offset, total,
                              self.buf(vid), width)
            offset += width

    def emit_ext_init(self, op) -> None:
        """recurrentの外部入力バッファの形式的な「定義」(CPU版と同じ
        理由。plan.py::_plan_recurrent参照)。実際の値はemit_recurrent_execの
        ホストループが毎tick上書きするので、ここでは何も発行しない。"""

    def emit_state_init(self, op) -> None:
        """recurrentのstate役scan_varの初期値(steps回のループの外、
        mdl_*_run呼び出し=1推論ごとに1回)。

        定数ビット列は他の定数表と同じ手口(up_u8)で1回だけデバイスへ
        アップロードし(既存の定数テーブルアップロード機構を流用)、
        batch枚ぶんへ複製するのは既存の「デバイス内コピー」原始操作
        (copy_batched_ex、src_stride=0で全枚に同じ範囲を複製)に任せる。
        """
        bits = op.attrs['bits']
        w = len(bits)
        tag = self.uid('sinit')
        # dedup有効時(RDDLGN decodeがstate_initを再発行するケース、
        # rddlgn_decode_gpu.py::_reset_recurrent_state)は同一opの初期値
        # 定数テーブルを使い回す。
        src = self._cached_upload('up_u8', f'{tag}_bits', bits)
        self.copy_batched_ex(self.buf(op.output), '0', w, src, '0', 0, w)

    def emit_recurrent_exec(self, op) -> None:
        """recurrentノードの本体。

        時間ループはホスト側(生成コードのドライバ層)でsteps回、scope内
        opのカーネル/コピー列を逐次発行する(カーネル内時間ループは作らない
        設計方針)。CUDAは同一デフォルトストリーム、Metalは単一command
        buffer系列に積むだけで発行順が実行順を保証するので(cudaMemcpyは
        デフォルトストリームに対して暗黙に同期し、Metalのblit/compute
        encoderは同じcommand bufferへ順に積まれる)、追加の同期は要らない。

        Cバックエンドはbodyを`mdl_step_<id>(ctx, t)`というstatic関数へ
        切り出すが、GPU側はレジスタ割付やタイル化のような最適化の余地が
        無く(呼び出し列をただ順に発行するだけ)、関数へ切り出す実益が
        無いので直接その場に(ホストのfor(t)ループ内へ)展開する
        (挙動はCと同一、コードがより単純という逸脱)。
        """
        a = op.attrs
        node_id = op.node_id
        steps = int(a['steps'])
        scope_ops = self._scoped.get(node_id, [])

        saved_body, self.body = self.body, []
        # 1. 外部入力の該当ステップぶんのコピーイン(tを含む式)。
        #    role='step' は自分の入力元から t*w オフセットのスライスを、
        #    role='invariant' は毎tick同じ範囲を詰め直す(CPU版と同じ
        #    冗長さをそのまま踏襲。挙動を合わせることを優先する)。
        #    ext_bufsはplan.py::_plan_recurrentが決め打ちでKIND_FLAT
        #    (scan_varのstep/invariantは埋め込みビット列のみで、int32の
        #    スコアが来ることはない)なので要素サイズ対応は不要
        #    (w=バイト数=要素数のまま、stacked出力(4のみ)とは異なる)。
        for sv_id, ext_id in a['ext_bufs'].items():
            w = int(a['ext_widths'][sv_id])
            src_id = a['ext_srcs'][sv_id]
            role = a['ext_roles'][sv_id]
            dst = self.buf(ext_id)
            src = self.buf(src_id)
            if role == 'step':
                full_w = steps * w
                self.copy_batched_ex(dst, '0', w, src, f't_ * {w}', full_w, w)
            else:  # invariant
                self.copy_batched_ex(dst, '0', w, src, '0', w, w)
        # 2. body(scope付きop列)を発行する。
        for sub in scope_ops:
            method = self.EMITTERS.get(sub.kind)
            if method is None:
                raise UnsupportedNode(
                    f'op種別 {sub.kind} は未対応(recurrent body内)',
                    node_id=sub.node_id, node_type=sub.node_type,
                    backend=self.name)
            getattr(self, method)(sub)
        # 3. state_update: bodyが計算した新しいstate候補をstateバッファへ
        #    コピーする。state/次候補はどちらもbatch枚ぶん同じ幅で並んで
        #    いるので、bi_ループ無しの1回のD2Dコピーで足りる。
        for sv_id, buf_id in a['state_bufs'].items():
            src_vid = a['state_update'][sv_id]
            w = self.plan.value(buf_id).width
            self.copy_bytes(self.buf(buf_id), 0, self.buf(src_vid),
                            w * self.plan.batch)
        # 4. stacked出力: このtickぶんをstackedバッファの該当行へ。
        # 要素サイズ対応: output_sourceがbody内でgroup_sumを
        #    経由するとKIND_SCORES(int32=4バイト/要素)になる
        #    (body=lut_layer/concatのみのケースはKIND_FLAT=1バイト
        #    のままなので eb=1 で従来どおり)。dst_stride/src_stride/n_bytesは
        #    すべて「バイト」なので要素幅wへ要素バイト数ebを掛けてから渡す
        #    (cbackendのbuf()が返す型付きポインタの添字演算に相当)。
        if a['output_mode'] == 'stacked':
            w = int(a['output_width'])
            eb = self.elem_bytes(op.output)
            row_bytes = w * eb
            total_bytes = steps * row_bytes
            self.copy_batched_ex(self.buf(op.output), f't_ * {row_bytes}',
                                 total_bytes, self.buf(a['output_source']),
                                 '0', row_bytes, row_bytes)
        step_lines = self.body
        self.body = saved_body
        self.body.append(f'  for (int t_ = 0; t_ < {steps}; t_++) {{')
        self.body.extend(step_lines)
        self.body.append('  }')

        # final_state出力: 最終tickのstateバッファをそのまま1回でコピーする
        # (bi_ループ無しの1回のD2Dコピーで足りる。理由は3と同じ)。stateは
        # RDDLGN構成上つねにKIND_FLATだが(plan.py::_plan_recurrentのstate_bufs
        # は決め打ちでKIND_FLAT)、cbackendの型付きポインタ演算と意味的に
        # 揃えるためelem_bytes経由にしておく(eb=1なら従来どおり)。
        if a['output_mode'] == 'final_state':
            buf_id = a['output_state_buf']
            w = int(a['output_width'])
            eb = self.elem_bytes(op.output)
            self.copy_bytes(self.buf(op.output), 0, self.buf(buf_id),
                            w * eb * self.plan.batch)

    #: op種別 → メソッド名。**関数オブジェクトではなく名前**で持つことで、
    #: サブクラスが個別の op だけを上書きできる(Metal の conv など)。
    EMITTERS = {
        'pack': 'emit_pack',
        'unpack': 'emit_unpack',
        'gate_program': 'emit_gate_program',
        'plane_op': 'emit_plane_op',
        'reduce_scores': 'emit_reduce_scores',
        'lut_dense': 'emit_lut_dense',
        'lut_flat': 'emit_lut_flat',
        'dense_fuse': 'emit_dense_fuse',
        'copy_scores': 'emit_copy_scores',
        'to_scores': 'emit_to_scores',
        'flat_concat': 'emit_flat_concat',
        'ext_init': 'emit_ext_init',
        'state_init': 'emit_state_init',
        'recurrent_exec': 'emit_recurrent_exec',
    }

    def emit_body(self) -> None:
        plan = self.plan
        # scopeが付いたop(recurrentのbody由来)は主経路には出さず、
        # emit_recurrent_execがそのtickぶんの呼び出し列として発行する
        # (cbackend.CBackend.build()と同じ2段構え)。
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

    # -- dense のバッチ軸ビットプレーン化 ------------------------
    def buf_packed(self, vid: int) -> str:
        """値`vid`(lut_denseの中間出力、KIND_FLAT u8で既にアリーナ確保
        済み)を指す**u32パック平面としてのエイリアス**を作る(メモ化)。

        `declare_buffers()`が作る非パック(u8)ビューとは別名の、同じ
        アリーナ・同じバイトオフセットを指す第2のビュー/ポインタを
        `view()`(サブクラスが実装する4原始操作の1つ)経由で発行する
        だけなので、追加のデバイスメモリ確保は発生しない。batch>=32なら
        パック所要バイト数(width*ceil(batch/32)*4)は必ず元のu8確保サイズ
        (width*batch)以下に収まる(`DENSE_PACKED_MIN_BATCH`のとき
        ceil(32/32)*4=4<=32が最小ケース、batchが増えるほど余裕が増す)。
        """
        if vid not in self._packed_view_cache:
            base = self.buf(vid)
            name = f'{base}_p32'
            off_words = self.memory.offsets[vid]
            self._packed_view_cache[vid] = self.view(
                name, 'd_mdl_arena', off_words, 'u32')
        return self._packed_view_cache[vid]

    def emit_dense_packed_body(self) -> None:
        """主経路(lut_dense列+末尾group_sum、`dense_packed_eligible`が
        検証済み)を32サンプル/語のバッチ軸パック表現で評価する
        `emit_body`の代替経路(opt-in、`build()`側の分岐)。

        流れ:
          1. 生の入力(`d_mdl_input`、u8、[batch][in_width])を専用の
             パック済み入力バッファ(u32、アリーナ外の独立確保、
             `mdl_pack_batch`)へ1回変換する。`d_mdl_input`自身を
             書き換えない(パック元とパック先が同一メモリだと、あるレーンの
             読み取りが終わる前に別スレッドが書き込む競合になるため
             `mdl_g_pack_batch_word`は別バッファを要求する)。
          2. 各lut_dense層を`lut_layer_packed[_u16]`で評価する。層間の
             中間値は`buf_packed()`(既存アリーナのu32エイリアス)を使う
             ─ 新規確保はしない。
          3. 末尾のgroup_sumを`group_sum_packed`で評価する。出力は
             非パック経路と同じint32レイアウト(`scores[sample*k+g]`、
             `plan.output_value`のKIND_SCORESバッファ)に直接書くので、
             `_source()`側の`fetch`(ホストへのdownload)は無改造で使える。

        値は非パック経路・`ir/simulate.py`と1ビットも変わらない
        (tests/deploy/test_metal_dense_packed.pyで整数厳密一致を固定)。
        """
        plan = self.plan
        in_width = plan.input_width
        bwords_max = (plan.batch + 31) >> 5
        packed_in = self.alloc(
            'd_mdl_dense_packed_in', max(4, in_width * bwords_max * 4), 'u32')
        self.call(f'{self.fn("pack_batch")}({self.buf(plan.input_value)}, '
                  f'{packed_in}, {in_width}, {self.N})')

        def packed_src(vid: int) -> str:
            return packed_in if vid == plan.input_value else self.buf_packed(vid)

        for op in plan.ops:
            a = op.attrs
            if op.kind == 'lut_dense':
                tag = self.uid('densepk')
                layer_in_width = plan.value(op.inputs[0]).width
                u16 = lut_dense_indices_fit_u16(layer_in_width)
                if u16:
                    idx = self.up_u16(f'{tag}_idx', a['indices'])
                    kernel = 'lut_layer_packed_u16'
                else:
                    idx = self.up_i32(f'{tag}_idx', a['indices'])
                    kernel = 'lut_layer_packed'
                tt = self.up_u32(f'{tag}_tt', a['truth_tables'])
                self.call(
                    f'{self.fn(kernel)}({packed_src(op.inputs[0])}, '
                    f'{self.buf_packed(op.output)}, {a["n_inputs"]}, '
                    f'{a["out_width"]}, {idx}, {tt}, {self.N})')
            elif op.kind == 'reduce_scores' and a.get('op') == 'group_sum':
                self.call(
                    f'{self.fn("group_sum_packed")}({packed_src(op.inputs[0])}, '
                    f'{self.buf(op.output)}, {a["in_width"]}, {a["k"]}, '
                    f'{self.N})')
            else:  # pragma: no cover - dense_packed_eligibleが事前に弾く
                raise UnsupportedNode(
                    f'dense_packed: op種別 {op.kind} は未対応'
                    '(dense_packed_eligibleの判定漏れ)',
                    node_id=op.node_id, node_type=op.node_type,
                    backend=self.name)

    def conv_tables(self, layout: ConvLayout, tag: str) -> dict:
        """conv(全oc結合)の定数配列をアップロードして名前を返す(両者共通)。"""
        return {
            'plane_c': self.up_i32(f'{tag}_plane_c', layout.plane_c),
            'plane_dy': self.up_i32(f'{tag}_plane_dy', layout.plane_dy),
            'plane_dx': self.up_i32(f'{tag}_plane_dx', layout.plane_dx),
            'instr': self.up_i32(f'{tag}_instr', layout.instr_flat),
            'oc_src': self.up_i32(f'{tag}_oc_src', layout.oc_src_off),
            'wstart': self.up_i32(f'{tag}_wstart', layout.write_start),
            'woc': self.up_i32(f'{tag}_woc', layout.write_oc),
        }

    def conv_tables_split(self, layout: ConvLayoutSplit, tag: str) -> dict:
        """conv(ocごとの命令列分割)の定数配列をアップロードして名前を返す。

        `instr_flat` のオペランドは既に `(value<<1)|is_reg` へパック済み
        (`build_conv_layout_split` が生成時に済ませてある)ので、ここでは
        combinedと同じくただの `int` 配列としてアップロードするだけでよい。
        """
        return {
            'plane_c': self.up_i32(f'{tag}_plane_c', layout.plane_c),
            'plane_dy': self.up_i32(f'{tag}_plane_dy', layout.plane_dy),
            'plane_dx': self.up_i32(f'{tag}_plane_dx', layout.plane_dx),
            'oc_instr_start': self.up_i32(f'{tag}_oc_instr_start',
                                          layout.oc_instr_start),
            'instr': self.up_i32(f'{tag}_instr', layout.instr_flat),
            'oc_final': self.up_i32(f'{tag}_oc_final', layout.oc_final_ref),
        }

    def conv_tables_planes(self, layout: ConvLayoutSplit, tag: str) -> dict:
        """conv(モデル特化コード生成)向け: タップ平面テーブルだけを
        アップロードする。命令列(instr_flat/oc_instr_start/oc_final_ref)は
        生成時に直線コードへ焼き込み済みで実行時に読まないので、
        `conv_tables_split` と違いここではアップロードしない
        (デバイスメモリ・転送量の両方を無駄にしないため)。
        """
        return {
            'plane_c': self.up_i32(f'{tag}_plane_c', layout.plane_c),
            'plane_dy': self.up_i32(f'{tag}_plane_dy', layout.plane_dy),
            'plane_dx': self.up_i32(f'{tag}_plane_dx', layout.plane_dx),
        }
