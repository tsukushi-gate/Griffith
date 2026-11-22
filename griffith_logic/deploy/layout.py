"""conv2d命令列の実行レイアウト。

TS側の真実源: tools/webbench/src/engines/bitplane.ts::buildConvProgramLayout

命令列(programs節)は「どのゲートをどの順で評価するか」しか持たない。
それを実際に走らせるには、タップ平面と作業平面をメモリのどこに置くかを
決める必要がある。その割り付けを**エクスポータ側で済ませて**、ターゲット
ランタイムには「絶対ワードオフセットが解決済みの命令列」だけを渡す。
おかげでC側は kind 分岐も生存区間解析も持たない。

スクラッチは1本の連続バッファ(単位はワード=32bit):

    [0,             tapRowsWords)   定数0平面
    [tapRowsWords,  2*tapRowsWords) 定数1平面(全ビット1)
    [poolBase,      regBase)        タップ平面プール(n_planes * plane_stride)
    [regBase,       scratch_words)  命令の作業平面(n_slots * tapRowsWords)

ビットプレーンのレイアウト契約(bitplane.ts と同一):
    words は [batch][C][H][wpr] の row-major。ビット順は列 x → ワード x>>5 の
    ビット x&31 の LSB-first。行頭は必ずワード境界。端数ビット(x>=W)は
    常に0を不変条件として全プリミティブが維持する。
"""

from dataclasses import dataclass, field

from ..export.optimize import (
    REF_KIND_CONST,
    REF_KIND_INSTR,
    REF_KIND_TAP,
    ConvProgram,
    make_ref,
    ref_idx,
    ref_kind,
)
from .programs import program_from_dict


def words_per_row(width: int) -> int:
    """1行に必要なワード数(= ceil(W/32))。"""
    return (int(width) + 31) >> 5


def tail_mask(width: int) -> int:
    """行末尾ワードの有効ビットマスク(W%32==0なら全ビット有効)。

    NOT系ゲートが端数ビットに1を立てるので、行を書き終える箇所で必ず
    これをANDする(呼び出し側の責務)。返り値は符号なし32bit。
    """
    r = int(width) & 31
    return 0xffffffff if r == 0 else (1 << r) - 1


def _pair(value) -> tuple[int, int]:
    if isinstance(value, list | tuple):
        return int(value[0]), int(value[1])
    return int(value), int(value)


@dataclass
class ConvLayout:
    """1つのconv2dノードの実行レイアウト(バッチ非依存)。"""

    h: int
    w: int
    h_out: int
    w_out: int
    s_h: int
    s_w: int
    p_h: int
    p_w: int
    out_channels: int
    wpr_in: int
    wpr_out: int
    tap_wpr: int
    tap_rows_words: int
    mask_out: int
    use_fast_path: bool
    plane_c: list[int] = field(default_factory=list)
    plane_dy: list[int] = field(default_factory=list)
    plane_dx: list[int] = field(default_factory=list)
    n_planes: int = 0
    plane_stride: int = 0
    plane_rows: int = 0
    in_plane_words: int = 0
    out_plane_words: int = 0
    # 命令ごとに4語 [tt, a_off, b_off, dst_off](絶対ワードオフセット)
    instr_flat: list[int] = field(default_factory=list)
    oc_src_off: list[int] = field(default_factory=list)
    write_start: list[int] = field(default_factory=list)
    write_oc: list[int] = field(default_factory=list)
    pool_base: int = 0
    reg_base: int = 0
    scratch_words: int = 0

    @property
    def n_instrs(self) -> int:
        return len(self.instr_flat) // 4


@dataclass
class _ConvGeom:
    """conv2dの幾何(バッチ非依存)+タップ平面プールのアドレス解決。

    `build_conv_layout`(全oc結合)と `build_conv_layout_split`
    (ocごとの命令列分割)が共有する部分をここへ1回だけ書く。
    """

    h: int
    w: int
    h_out: int
    w_out: int
    s_h: int
    s_w: int
    p_h: int
    p_w: int
    out_channels: int
    wpr_in: int
    wpr_out: int
    tap_wpr: int
    tap_rows_words: int
    mask_out: int
    use_fast_path: bool
    plane_c: list[int]
    plane_dy: list[int]
    plane_dx: list[int]
    n_planes: int
    plane_stride: int
    plane_rows: int
    in_plane_words: int
    out_plane_words: int
    pool_base: int
    #: タップ(prog.taps の添字)→ タップ平面プール内の絶対ワードオフセット。
    tap_word_base: list[int]

    def mem_ref_off(self, r: int) -> int:
        """CONST/TAPな ref → 共有スクラッチの絶対ワードオフセット。

        INSTR kind はここでは解決しない(全oc結合と oc分割でアドレス空間の
        意味が違う — 結合は「reg平面」、分割は「スレッドローカルレジスタ」
        なので、呼び出し側がそれぞれの流儀で解決する)。
        """
        kind = ref_kind(r)
        if kind == REF_KIND_CONST:
            return ref_idx(r) * self.tap_rows_words
        if kind == REF_KIND_TAP:
            return self.tap_word_base[ref_idx(r)]
        raise ValueError(f'mem_ref_off: INSTR kindのrefは解決できない({r})')


def _conv_geometry(params: dict, prog: ConvProgram) -> _ConvGeom:
    """IR conv2dノードparams + ConvProgram → 幾何+タップ平面プールの解決。

    `build_conv_layout` と `build_conv_layout_split` の共通部分。
    """
    _, h, w = (int(v) for v in params['in_shape'])
    kh, kw = _pair(params['kernel_size'])
    s_h, s_w = _pair(params.get('stride', 1))
    p_h, p_w = _pair(params.get('padding', 0))
    h_out = (h + 2 * p_h - kh) // s_h + 1
    w_out = (w + 2 * p_w - kw) // s_w + 1
    oc_count = int(params['out_channels'])
    if len(prog.oc_out) != oc_count:
        raise ValueError(
            f'oc_out長 {len(prog.oc_out)} が out_channels={oc_count} と違う')
    khw = kh * kw

    wpr_in = words_per_row(w)
    wpr_out = words_per_row(w_out)
    use_fast_path = s_h == 1 and s_w == 1 and w_out == w
    tap_wpr = wpr_in if use_fast_path else wpr_out
    tap_rows_words = h_out * tap_wpr
    plane_rows = (h + 2 * p_h) if use_fast_path else h_out
    plane_stride = plane_rows * tap_wpr

    # タップ(受容野内フラット位置)→ 平面へdedup。キーは bitplane.ts と同一。
    key_to_plane: dict[int, int] = {}
    plane_c: list[int] = []
    plane_dy: list[int] = []
    plane_dx: list[int] = []
    tap_plane: list[int] = []
    tap_row_off: list[int] = []
    for pp in prog.taps:
        rem = pp % khw
        c = pp // khw
        dy = rem // kw
        dx = rem % kw
        key = c * kw + dx if use_fast_path else c * khw + dy * kw + dx
        pid = key_to_plane.get(key)
        if pid is None:
            pid = len(plane_c)
            plane_c.append(c)
            plane_dy.append(dy)
            plane_dx.append(dx)
            key_to_plane[key] = pid
        tap_plane.append(pid)
        tap_row_off.append(dy * tap_wpr if use_fast_path else 0)

    n_planes = len(plane_c)
    pool_base = 2 * tap_rows_words
    tap_word_base = [
        pool_base + tap_plane[t] * plane_stride + tap_row_off[t]
        for t in range(len(prog.taps))
    ]

    return _ConvGeom(
        h=h, w=w, h_out=h_out, w_out=w_out,
        s_h=s_h, s_w=s_w, p_h=p_h, p_w=p_w,
        out_channels=oc_count,
        wpr_in=wpr_in, wpr_out=wpr_out, tap_wpr=tap_wpr,
        tap_rows_words=tap_rows_words,
        mask_out=tail_mask(w_out), use_fast_path=use_fast_path,
        plane_c=plane_c, plane_dy=plane_dy, plane_dx=plane_dx,
        n_planes=n_planes, plane_stride=plane_stride, plane_rows=plane_rows,
        in_plane_words=int(params['in_shape'][0]) * h * wpr_in,
        out_plane_words=oc_count * h_out * wpr_out,
        pool_base=pool_base, tap_word_base=tap_word_base,
    )


def build_conv_layout(params: dict, program: dict) -> ConvLayout:
    """IRのconv2dノードparams + programs節エントリ → 実行レイアウト。

    Args:
        params: IRNode(type='conv2d').params
        program: バンドルの programs[node_id]

    Raises:
        ValueError: 幾何が命令列と食い違う場合
    """
    prog = program_from_dict(program)
    g = _conv_geometry(params, prog)
    reg_base = g.pool_base + g.n_planes * g.plane_stride
    scratch_words = reg_base + max(1, prog.n_slots) * g.tap_rows_words

    def ref_off(r: int) -> int:
        if ref_kind(r) == REF_KIND_INSTR:
            return reg_base + prog.instrs[ref_idx(r)].slot * g.tap_rows_words
        return g.mem_ref_off(r)

    instr_flat: list[int] = []
    for ins in prog.instrs:
        instr_flat.extend((
            ins.tt,
            ref_off(ins.a),
            ref_off(ins.b),
            reg_base + ins.slot * g.tap_rows_words,
        ))

    return ConvLayout(
        h=g.h, w=g.w, h_out=g.h_out, w_out=g.w_out,
        s_h=g.s_h, s_w=g.s_w, p_h=g.p_h, p_w=g.p_w,
        out_channels=g.out_channels,
        wpr_in=g.wpr_in, wpr_out=g.wpr_out, tap_wpr=g.tap_wpr,
        tap_rows_words=g.tap_rows_words,
        mask_out=g.mask_out, use_fast_path=g.use_fast_path,
        plane_c=g.plane_c, plane_dy=g.plane_dy, plane_dx=g.plane_dx,
        n_planes=g.n_planes, plane_stride=g.plane_stride,
        plane_rows=g.plane_rows,
        in_plane_words=g.in_plane_words,
        out_plane_words=g.out_plane_words,
        instr_flat=instr_flat,
        oc_src_off=[ref_off(r) for r in prog.oc_out],
        write_start=list(prog.write_start),
        write_oc=list(prog.write_oc),
        pool_base=g.pool_base, reg_base=reg_base, scratch_words=scratch_words,
    )


# ============================================================
# ocごとの命令列分割(WebGPU版oc分割をCUDA/Metalへ移植)
# ============================================================
#
# 現行の mdl_k_conv_exec は「1スレッド=1ワード列が全oc共通の命令列全体を
# 直列実行」する形で、深い層ではスレッド数が tap_rows_words(=h_out*wpr_out)
# しか立たない(実測ALU使用率0.41%)。ocを並列度に足すには、
# oc単位で「そのocの出力へ到達する命令の依存閉包」を後方スライスし、
# スロットをoc内ローカルに振り直した命令列を作ればよい
# (バンドルのprograms節=全oc結合の命令列を**真実源のまま**使い、
# 木からの再導出・再縮約はしない)。
#
# 各ocの依存閉包は元の命令列(トポロジカル順)の部分集合なので、
# 昇順に並べればそのままローカルな実行順として使える。さらに、
# 「そのocの出力ref」は定義よりその閉包内で最大のグローバル命令番号を
# 持つ(閉包はその祖先だけで構成されるため)ので、**ローカル命令列の
# 最後の命令が必ずそのocの最終値**になる。これは全oc結合の
# write_start/write_oc(命令jの直後にどのocへ書き出すか、というCSR)が
# 分割後は不要になることを意味する: 実行順=ソート順の一石三鳥設計
# (dpu-v2-sort-streamedの知見と同じ形)。


@dataclass
class OcInstr:
    """oc分割後の1命令(ref参照レベル。アドレス解決前)。

    `a`/`b` は元の ConvInstr と同じ `(idx<<2)|kind` エンコード。
    kind=INSTR の idx は**このocのローカル命令列内の位置**
    (0-based、`OcProgramSplit.instrs` の添字)であり、グローバル命令番号
    ではない。kind=CONST/TAP の idx は元のグローバルな番号のまま
    (定数は0/1固定、タップは共有プールを指すので renumber不要)。
    """

    tt: int
    a: int
    b: int
    slot: int = -1


@dataclass
class OcProgramSplit:
    """1つの出力チャネルぶんのローカル命令列。"""

    instrs: list[OcInstr]
    #: このocの最終値へのref。instrsが空なら CONST/TAP、非空なら
    #: 必ず `make_ref(REF_KIND_INSTR, len(instrs) - 1)`(上のコメント参照)。
    out_ref: int
    #: ローカルスロット数(free-listで使い回した後の最大同時生存数)。
    n_slots: int


def split_program_by_oc(prog: ConvProgram) -> list[OcProgramSplit]:
    """全oc結合の ConvProgram を、ocごとの依存閉包+ローカルスロットへ分割する。

    アドレス解決前のref参照レベルの変換(このモジュールの他の関数から
    独立にテストできるようにするため、わざと純粋なグラフ変換に留めている)。
    """
    out: list[OcProgramSplit] = []
    for out_ref in prog.oc_out:
        if ref_kind(out_ref) != REF_KIND_INSTR:
            out.append(OcProgramSplit(instrs=[], out_ref=out_ref, n_slots=0))
            continue

        # --- 後方到達可能な命令(依存閉包)をDFSで集める ---
        root = ref_idx(out_ref)
        in_cone = {root}
        stack = [root]
        while stack:
            j = stack.pop()
            ins = prog.instrs[j]
            for r in (ins.a, ins.b):
                if ref_kind(r) == REF_KIND_INSTR and ref_idx(r) not in in_cone:
                    in_cone.add(ref_idx(r))
                    stack.append(ref_idx(r))
        # 元の命令列はトポロジカル順(生成順)なので、昇順ソートがそのまま
        # このocに限定した有効な実行順になる。root(=out_ref)はその祖先
        # だけで閉包が構成されるため、閉包内で最大のグローバル番号を持つ
        # → 昇順ソートの末尾に来る。
        cone = sorted(in_cone)
        local_of = {g_idx: i for i, g_idx in enumerate(cone)}

        def remap(r: int, _local_of=local_of) -> int:
            if ref_kind(r) == REF_KIND_INSTR:
                return make_ref(REF_KIND_INSTR, _local_of[ref_idx(r)])
            return r

        local_instrs = [
            OcInstr(tt=prog.instrs[g_idx].tt,
                    a=remap(prog.instrs[g_idx].a),
                    b=remap(prog.instrs[g_idx].b))
            for g_idx in cone
        ]

        # --- ローカル生存区間解析(build_conv_program と同じfree-listアルゴリズム) ---
        n = len(local_instrs)
        last_use = list(range(n))
        for j, ins in enumerate(local_instrs):
            if ref_kind(ins.a) == REF_KIND_INSTR:
                last_use[ref_idx(ins.a)] = j
            if ref_kind(ins.b) == REF_KIND_INSTR:
                last_use[ref_idx(ins.b)] = j
        free: list[int] = []
        n_slots = 0
        for j, ins in enumerate(local_instrs):
            if free:
                slot = free.pop()
            else:
                slot = n_slots
                n_slots += 1
            ins.slot = slot
            ia, ib = ins.a, ins.b
            if ref_kind(ia) == REF_KIND_INSTR and last_use[ref_idx(ia)] == j:
                free.append(local_instrs[ref_idx(ia)].slot)
            if (ref_kind(ib) == REF_KIND_INSTR and ib != ia
                    and last_use[ref_idx(ib)] == j):
                free.append(local_instrs[ref_idx(ib)].slot)
            if last_use[j] == j:
                free.append(slot)

        local_out_ref = make_ref(REF_KIND_INSTR, local_of[root])
        assert local_of[root] == n - 1, (
            'root(out_ref)はローカル命令列の末尾に来るはず'
            f'(local_of[root]={local_of[root]}, n-1={n - 1})')
        out.append(OcProgramSplit(instrs=local_instrs, out_ref=local_out_ref,
                                  n_slots=n_slots))
    return out


#: `runtime/gpu/mdl_gpu_kernels.h` の `MDL_CONV_SPLIT_MAX_SLOTS` と同じ値
#: に**手で**同期させること(コンパイル時定数なのでここから直接読めない)。
#: 実測: 全oc結合のn_slotsは1〜6。oc分割はそれぞれの部分グラフに
#: 限定されるぶん通常はこれ以下になるが、保証はしていないので余裕を見て
#: 8にしてある。これを超えるモデルは `build_conv_layout_split` が
#: `ValueError` で落とす(未検証のレジスタ数を黙って使わない)。
CONV_SPLIT_MAX_SLOTS = 8


def _pack_mem(addr: int) -> int:
    """MEM(共有スクラッチのアドレス)参照のパック。下位1bit=0。"""
    return addr << 1


def _pack_reg(slot: int) -> int:
    """REG(スレッドローカルレジスタ)参照のパック。下位1bit=1。"""
    return (slot << 1) | 1


@dataclass
class ConvLayoutSplit:
    """conv2dノードの「ocごとの命令列分割」実行レイアウト。

    `ConvLayout`(全oc結合)と同じ幾何・タップ平面プールを使うが、命令の
    実行単位が「1スレッド=1ワード列が全ocを直列実行」から
    「1スレッド=(oc, ワード列)」に変わる。オペランド参照は
    `(value<<1)|is_reg` にパックしてあり、is_reg=0 なら共有スクラッチの
    絶対ワードオフセット(`sc[value + i]`、CONST/TAPが両方ここに来る —
    定数0/1平面もタップ平面プールも「全スレッド共通の読み取り専用データ」
    という点で同じ扱いにできる)、is_reg=1 なら**スレッドローカルレジスタ
    配列の添字**(`slots[value]`、命令オペランドのINSTR参照はここに来る)。

    命令に依存しないoc(命令ゼロ、出力がCONST/TAP直結)は `oc_final_ref`
    が直接MEM参照になり、`oc_instr_start[oc] == oc_instr_start[oc+1]`
    (このocの命令区間が空)になる。
    """

    h: int
    w: int
    h_out: int
    w_out: int
    s_h: int
    s_w: int
    p_h: int
    p_w: int
    out_channels: int
    wpr_in: int
    wpr_out: int
    tap_wpr: int
    tap_rows_words: int
    mask_out: int
    use_fast_path: bool
    plane_c: list[int] = field(default_factory=list)
    plane_dy: list[int] = field(default_factory=list)
    plane_dx: list[int] = field(default_factory=list)
    n_planes: int = 0
    plane_stride: int = 0
    plane_rows: int = 0
    in_plane_words: int = 0
    out_plane_words: int = 0
    pool_base: int = 0
    #: タップ平面プールまでで打ち止め(全oc結合と違い、reg平面が要らない
    #: ぶん小さい: `pool_base + n_planes*plane_stride`)。
    scratch_words: int = 0
    #: CSR(長さ out_channels+1、単位=命令)。oc の命令区間は
    #: `instr_flat[4*oc_instr_start[oc] : 4*oc_instr_start[oc+1]]`。
    oc_instr_start: list[int] = field(default_factory=list)
    #: ocごとに連結した命令列。4語/命令 [tt, a_packed, b_packed, dst_slot]
    #: (dst_slotはそのoc内ローカルなレジスタ番号、パック不要)。
    instr_flat: list[int] = field(default_factory=list)
    #: oc → 最終値への参照(パック済み)。長さ out_channels。
    oc_final_ref: list[int] = field(default_factory=list)
    #: 全ocの中での最大ローカルスロット数(スレッドローカル配列のサイズに使う)。
    max_slots_per_oc: int = 1

    @property
    def n_instrs_total(self) -> int:
        """全oc合計の命令数(oc間で重複した命令は複数回数える)。"""
        return len(self.instr_flat) // 4


def build_conv_layout_split(params: dict, program: dict) -> ConvLayoutSplit:
    """IRのconv2dノードparams + programs節エントリ → oc分割レイアウト。

    `build_conv_layout` と同じ入力(バンドルのprograms節=全oc結合の命令列)
    から作る。バンドルフォーマット自体は変更しない
    (このレイアウトはCUDA/Metalコード生成時にだけ使う中間表現)。

    Raises:
        ValueError: 幾何が命令列と食い違う、または
            `max_slots_per_oc > CONV_SPLIT_MAX_SLOTS`(コンパイル時定数の
            スレッドローカル配列に収まらない、未検証のサイズなので拒否する)
    """
    prog = program_from_dict(program)
    g = _conv_geometry(params, prog)
    splits = split_program_by_oc(prog)

    def pack_operand(r: int, slot_of_local: list[int]) -> int:
        if ref_kind(r) == REF_KIND_INSTR:
            return _pack_reg(slot_of_local[ref_idx(r)])
        return _pack_mem(g.mem_ref_off(r))

    oc_instr_start = [0]
    instr_flat: list[int] = []
    oc_final_ref: list[int] = []
    max_slots = 1
    for sp in splits:
        slot_of_local = [ins.slot for ins in sp.instrs]
        for ins in sp.instrs:
            instr_flat.extend((
                ins.tt,
                pack_operand(ins.a, slot_of_local),
                pack_operand(ins.b, slot_of_local),
                ins.slot,
            ))
        oc_instr_start.append(oc_instr_start[-1] + len(sp.instrs))
        oc_final_ref.append(pack_operand(sp.out_ref, slot_of_local))
        if sp.n_slots > max_slots:
            max_slots = sp.n_slots

    if max_slots > CONV_SPLIT_MAX_SLOTS:
        raise ValueError(
            f'build_conv_layout_split: max_slots_per_oc={max_slots} が '
            f'CONV_SPLIT_MAX_SLOTS={CONV_SPLIT_MAX_SLOTS} を超えている'
            '(スレッドローカル配列のコンパイル時上限。このモデルはoc分割'
            '経路では未対応 — CONV_SPLIT_MAX_SLOTSを実測に基づいて見直す'
            '必要がある)')

    return ConvLayoutSplit(
        h=g.h, w=g.w, h_out=g.h_out, w_out=g.w_out,
        s_h=g.s_h, s_w=g.s_w, p_h=g.p_h, p_w=g.p_w,
        out_channels=g.out_channels,
        wpr_in=g.wpr_in, wpr_out=g.wpr_out, tap_wpr=g.tap_wpr,
        tap_rows_words=g.tap_rows_words,
        mask_out=g.mask_out, use_fast_path=g.use_fast_path,
        plane_c=g.plane_c, plane_dy=g.plane_dy, plane_dx=g.plane_dx,
        n_planes=g.n_planes, plane_stride=g.plane_stride,
        plane_rows=g.plane_rows,
        in_plane_words=g.in_plane_words,
        out_plane_words=g.out_plane_words,
        pool_base=g.pool_base,
        scratch_words=g.pool_base + g.n_planes * g.plane_stride,
        oc_instr_start=oc_instr_start,
        instr_flat=instr_flat,
        oc_final_ref=oc_final_ref,
        max_slots_per_oc=max_slots,
    )


def conv_layout_split_stats(prog_dict: dict, layout_split: ConvLayoutSplit) -> dict:
    """oc分割の効果測定用の統計(命令重複率・ローカルスロット数)。

    Args:
        prog_dict: バンドルの programs[node_id](全oc結合、分割前)
        layout_split: 同じノードから作った `ConvLayoutSplit`

    Returns:
        {'n_instrs_combined', 'n_instrs_split_total', 'duplication_pct',
         'max_slots_per_oc', 'out_channels', 'zero_instr_oc_count'}
    """
    n_combined = int(prog_dict['n_instrs'])
    n_split = layout_split.n_instrs_total
    dup_pct = (
        (n_split - n_combined) / n_combined * 100.0 if n_combined > 0
        else (0.0 if n_split == 0 else float('inf'))
    )
    zero_instr = sum(
        1 for i in range(layout_split.out_channels)
        if layout_split.oc_instr_start[i] == layout_split.oc_instr_start[i + 1]
    )
    return {
        'n_instrs_combined': n_combined,
        'n_instrs_split_total': n_split,
        'duplication_pct': dup_pct,
        'max_slots_per_oc': layout_split.max_slots_per_oc,
        'out_channels': layout_split.out_channels,
        'zero_instr_oc_count': zero_instr,
    }
