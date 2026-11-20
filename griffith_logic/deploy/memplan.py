"""メモリ計画 — 生存区間解析によるバッファ再利用。

ノード出力ごとに別バッファを取ると、**作業RAMが重みの13倍**になる
(seg 実測: `__const` 37KB に対し `__bss` 488KB)。数十KB RAM級のMCUでは
ここが載るか載らないかを決める。

ONNX Runtime の **memory pattern** と同じ考え方: 値がいつ生まれていつ最後に
読まれるかを op 列から求め、生存区間が重ならない値に同じ領域を割り当てる。
**プラン層に書くので C / CUDA / Metal に同時に効く**。ノード型の知識は一切
使わない — op 列の (inputs, output) だけを見る。

割り当ての単位は32bitワード。u8列もi32列も同じアリーナに載せるが、
オフセットをワード単位に揃えてあるので**どの型でも4バイト整列が保証される**。

安全性の要:
    ある op の出力に、その op の入力の領域を割り当ててはいけない
    (`mdl_or_pool2x2(src, dst, ...)` は src を読みながら dst を書く)。
    下の実装は「最終使用が現在の op **より前**」のものだけを解放するので、
    入力は解放対象にならない。この性質はテストで固定してある。
"""

from dataclasses import dataclass, field

from .plan import ExecPlan


@dataclass
class MemoryPlan:
    """値id → アリーナ内のワードオフセット。"""

    offsets: dict[int, int] = field(default_factory=dict)
    sizes: dict[int, int] = field(default_factory=dict)
    arena_words: int = 0
    #: 再利用しない場合の合計(効果を記録するため)
    naive_words: int = 0
    #: プールに入れなかった値(呼び出し側のメモリなど)
    excluded: frozenset = frozenset()

    def saved_bytes(self) -> int:
        return (self.naive_words - self.arena_words) * 4

    def ratio(self) -> float:
        if not self.naive_words:
            return 1.0
        return self.arena_words / self.naive_words


def liveness(plan: ExecPlan) -> tuple[dict[int, int], dict[int, int]]:
    """値id → (定義されたop番号, 最後に読まれたop番号)。

    実行前から生きている値(グラフ入力・前処理の出力先)は -1、
    最後まで生きている値(出力)は len(ops) を返す。
    """
    first: dict[int, int] = {}
    last: dict[int, int] = {}
    #: recurrentノードid → そのbody(scope)が最初に現れるop番号。
    scope_start: dict[str, int] = {}
    for i, op in enumerate(plan.ops):
        first[op.output] = i
        last[op.output] = max(last.get(op.output, i), i)
        for vid in op.inputs:
            last[vid] = max(last.get(vid, i), i)
        # スクラッチはその op の実行中しか生きない。だから conv が何本あっても
        # 1本ぶんの領域に畳める(実測: seg で 188KB → 85KB)。
        for vid in op.attrs.get('scratch', ()):
            first[vid] = i
            last[vid] = i
        if op.scope and op.scope not in scope_start:
            scope_start[op.scope] = i
    # recurrentの output_mode='stacked' 出力は、bodyがsteps回再実行される
    # あいだtickごとに該当行へ書き足される「持続する蓄積バッファ」であり、
    # 「recurrent_exec opの実行時に初めて書かれる」単発の値ではない
    # (gpubackend.py::emit_recurrent_exec / cbackend.pyの
    # emit_recurrent_execが実際にtickループの中で毎回書き込む)。
    # first[out]をrecurrent_exec自身のop番号のままにすると、生存区間解析が
    # 「bodyのscope内op(scratch含む)はstacked出力より前に死んでいる」と
    # 誤判定し(scope内opはbody一巡ぶんの「テンプレート」としてop列に
    # 1回しか現れないため、op番号上はstacked出力の定義=recurrent_exec自身
    # より前で完結して見える)、同じメモリ領域を割り当ててしまう
    # (次tickの再実行でbodyのscratchが1つ前のtickで書いたstacked出力の
    # 行を上書きする実バグになる。C backendは`plan.output_value`を
    # プールから除外するため、recurrentの出力がグラフ最終出力のときだけ
    # 偶然この経路を踏まないが、GPU backendは出力もプールするため
    # 一般に踏みうる)。stacked出力の生存開始をbodyの先頭(scope_start)
    # まで遡らせ、body全体(=steps回の再実行区間)と同時に生きている
    # ことにする。
    for op in plan.ops:
        if op.kind == 'recurrent_exec' and op.attrs.get('output_mode') == 'stacked':
            s = scope_start.get(op.node_id)
            if s is not None:
                first[op.output] = min(first.get(op.output, s), s)
    first[plan.input_value] = -1
    last.setdefault(plan.input_value, -1)
    if plan.input_bitplane is not None:
        first[plan.input_bitplane] = -1
        last.setdefault(plan.input_bitplane, -1)
    last[plan.output_value] = len(plan.ops)
    return first, last


def _words(plan: ExecPlan, vid: int) -> int:
    """バッチ込みのワード数。全ての値が枚数ぶん並ぶ。"""
    return max(1, (plan.value(vid).nbytes() * plan.batch + 3) // 4)


def plan_memory(plan: ExecPlan, *, exclude=()) -> MemoryPlan:
    """生存区間が重ならない値に同じ領域を割り当てる。

    Args:
        plan: 実行プラン
        exclude: プールに入れない値id(呼び出し側が持つメモリなど)

    Returns:
        MemoryPlan。`offsets` に載っていない値はバックエンドが個別に確保する
    """
    exclude = frozenset(exclude)
    first, last = liveness(plan)
    # alias_of!=None の値(退化ゲート除去のアキュムレータ追記先)は
    # 独自の領域を持たない(親バッファの一部を指すだけ)ので、プールから
    # 除外する。naive_words(効果測定用の「畳まなかった場合の合計」)からも
    # 除く — 元々このバイト数は「親バッファに間借りする」設計であり、
    # 独立確保していた過去は無いため、素朴合計の基準に含めると水増しになる。
    pooled = [v.id for v in plan.values
             if v.id not in exclude and v.alias_of is None]
    naive = sum(_words(plan, vid) for vid in pooled)

    # 割り当ての順序: 実行前に生きている値 → op順
    order = sorted(pooled, key=lambda vid: (first.get(vid, 0), vid))

    offsets: dict[int, int] = {}
    sizes = {vid: _words(plan, vid) for vid in pooled}
    free: list[list[int]] = []        # [offset, size] を offset 昇順で保つ
    busy: list[tuple[int, int, int]] = []   # (last_use, offset, size)
    arena = 0

    def release(before: int) -> None:
        """最終使用が `before` より前の領域を空きへ戻す(隣接は併合)。"""
        nonlocal busy
        keep = []
        for end, off, size in busy:
            if end < before:
                free.append([off, size])
            else:
                keep.append((end, off, size))
        busy = keep
        free.sort()
        merged: list[list[int]] = []
        for off, size in free:
            if merged and merged[-1][0] + merged[-1][1] == off:
                merged[-1][1] += size
            else:
                merged.append([off, size])
        free[:] = merged

    for vid in order:
        need = sizes[vid]
        # 0 で下限を切らないこと。実行前に生きている値(グラフ入力と前処理の
        # 出力先)は時刻 -1 で同時に生きていて、しかも入口関数が
        # 「入力を読んで出力先へパックする」ので、重ねると壊れる。
        release(first.get(vid, 0))
        # best-fit: 入る中で最小の空き。断片化を抑えつつ探索は線形で足りる
        # (実モデルの値は10〜30本で、ここが律速になることはない)
        best = -1
        for i, (_, size) in enumerate(free):
            if size >= need and (best < 0 or size < free[best][1]):
                best = i
        if best >= 0:
            off = free[best][0]
            if free[best][1] == need:
                free.pop(best)
            else:
                free[best][0] += need
                free[best][1] -= need
        else:
            off = arena
            arena += need
        offsets[vid] = off
        busy.append((last.get(vid, first.get(vid, 0)), off, need))

    return MemoryPlan(offsets=offsets, sizes=sizes, arena_words=arena,
                      naive_words=naive, excluded=exclude)


def check_no_overlap(plan: ExecPlan, mem: MemoryPlan) -> None:
    """同時に生きている値が領域を共有していないことを確認する。

    メモリ計画の誤りは「たまたま動く」形で出るので(上書きされた値が
    まだ読まれていないだけ)、生成前にここで落とす。
    """
    first, last = liveness(plan)
    items = [(vid, mem.offsets[vid], mem.sizes[vid]) for vid in mem.offsets]
    for i, (a, off_a, size_a) in enumerate(items):
        for b, off_b, size_b in items[i + 1:]:
            overlap_mem = off_a < off_b + size_b and off_b < off_a + size_a
            if not overlap_mem:
                continue
            live_a = (first.get(a, 0), last.get(a, 0))
            live_b = (first.get(b, 0), last.get(b, 0))
            if live_a[0] <= live_b[1] and live_b[0] <= live_a[1]:
                raise AssertionError(
                    f'メモリ計画の誤り: v{a}{live_a} と v{b}{live_b} が'
                    f'同時に生きているのに領域が重なっている '
                    f'(v{a}@{off_a}+{size_a}, v{b}@{off_b}+{size_b})')

    # alias_of!=None の値(親バッファ+オフセットで存在する、独自の
    # プール割り当てを持たない値)は上のループでは検査されない
    # (mem.offsetsに載らないため)。ここで別途、(1) 親バッファの範囲内に
    # 収まっているか、(2) 同じ親を指す値同士が互いに重ならないか、の2点を
    # 検査する。親自身の生存区間は通常の値と同じ liveness() ロジックで
    # 扱われる(親はどれかのopのinputs/outputに現れる普通の値なので、
    # ここでの追加検証は不要)。
    by_parent: dict[int, list[tuple[int, int, int]]] = {}
    for v in plan.values:
        if v.alias_of is None:
            continue
        parent = plan.value(v.alias_of)
        parent_bytes = parent.nbytes()
        v_bytes = v.nbytes()
        if v.alias_offset < 0 or v.alias_offset + v_bytes > parent_bytes:
            raise AssertionError(
                f'メモリ計画の誤り: v{v.id}(alias of v{v.alias_of})の範囲 '
                f'[{v.alias_offset}, {v.alias_offset + v_bytes}) が'
                f'親バッファ({parent_bytes}B)からはみ出している')
        by_parent.setdefault(v.alias_of, []).append(
            (v.alias_offset, v_bytes, v.id))
    for parent_id, ranges in by_parent.items():
        ranges.sort()
        for (off_a, size_a, id_a), (off_b, size_b, id_b) in zip(
                ranges, ranges[1:], strict=False):
            if off_a + size_a > off_b:
                raise AssertionError(
                    f'メモリ計画の誤り: v{id_a}とv{id_b}(どちらもv{parent_id}'
                    'のalias)の範囲が重なっている '
                    f'(v{id_a}@{off_a}+{size_a}, v{id_b}@{off_b}+{size_b})')
