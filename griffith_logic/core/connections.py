import math

import torch


def get_rot_connections(
    in_dim: int, out_dim: int, s_a: int, s_b: int, device: str = 'cpu',
    step: int = 1,
) -> torch.Tensor:
    """rot 規則の規則配線(2入力固定)。

    社内先行実験 `wiring.py` の `rot_layer` と同一定義:

        idx_a[g] = (g · step + s_a) mod in_dim
        idx_b[g] = (g · step + s_b) mod in_dim      (g = 0 .. out_dim-1)

    「ずらし量 s」が入力の平坦化順の中で幾何的な意味を持つ配線
    (例: 先行実験 CIFAR サーモメータ順 ((c*32+row)*32+col)*3+t では
    s=3 が右1画素、s=96 が1行下)。乱数を使わないので完全に決定的。

    歩進 `step`(既定 1 = 従来の rot と bitwise 同一)は内部調査リポジトリ先行実験の
    rot2(`(g·r + s) mod W_in`。conv 頭部で前層の幅 > 出力幅のとき、隣接ゲートを
    r ずつ離して前層の出力をデッドにしない)。計算型 overlay では歩進定数を
    V から r·V に替えるだけなので、IR・HDL・overlay の形は変わらない。

    Args:
        in_dim: 入力次元数
        out_dim: 出力ニューロン数
        s_a: 第1入力のずらし量(負値・in_dim超も可。剰余で丸める)
        s_b: 第2入力のずらし量
        device: 出力テンソルのデバイス
        step: 歩進 r(ゲート g の基準位置 g·r。既定 1)

    Returns:
        indices: (2, out_dim) int64 テンソル
    """
    g = torch.arange(out_dim, dtype=torch.int64) * int(step)
    idx_a = (g + int(s_a)) % in_dim
    idx_b = (g + int(s_b)) % in_dim
    return torch.stack([idx_a, idx_b], dim=0).to(device).contiguous()


def get_stretched_rot_connections(
    in_dim: int, out_dim: int, s_a: int, s_b: int, device: str = 'cpu'
) -> torch.Tensor:
    """引き伸ばし rot 配線(2入力固定、先行実験 SR 腕)。

    rot は基準位置が g(1 ずつ)なので out_dim < in_dim では入力の先頭側
    out_dim + max(s_a, s_b) 本しか読まない。本配線は**基準位置だけを
    in_dim / out_dim 刻みに引き伸ばし**、対の作り方(ずらし量 s_a / s_b)は
    rot のまま残す:

        base[g] = round(g · in_dim / out_dim)
        idx_a[g] = (base[g] + s_a) mod in_dim
        idx_b[g] = (base[g] + s_b) mod in_dim      (g = 0 .. out_dim-1)

    丸めは**四捨五入(半整数は切り上げ)を整数演算で**行う:

        base[g] = (2·g·in_dim + out_dim) // (2·out_dim)   ( = floor(g·in/out + 1/2) )

    Python 組み込みの `round` は偶数丸めなので半整数でずれる(例 in=4, out=8 の
    g=1 は本定義が 1、`round(0.5)` は 0)。浮動小数を使わないので
    デバイス・環境によらず完全に決定的。

    性質:
    - `in_dim == out_dim` では base[g] = g となり `get_rot_connections` と
      **bitwise 一致**する。
    - 対の差は rot と同じく一定: `(idx_b - idx_a) ≡ (s_b - s_a) (mod in_dim)`。
      平坦化順が幾何を持つ符号化(先行実験 CIFAR サーモメータ順)では、
      s_b - s_a = 93 のような「右1画素と1行下」の局所性が保たれる。
    - 先行実験層0(in_dim=9216, out_dim=8000, s_a=3, s_b=96)の被覆率は
      **9216/9216 = 100.0%**(実測値。tests/l1_math/test_connections_generators.py
      の `test_exp100_layer0_covers_every_input` が assert する)。
      同条件の rot は 8093/9216 = 87.81% しか読まない。
      使用回数は 1 回が 2432 本・2 回が 6784 本で、未参照の入力は無い。
    - **被覆率 100% は一般には保証しない**。基準位置は in_dim / out_dim 刻みで
      飛ぶので、比が整数のときは s_b - s_a の剰余次第で穴が残る
      (例 in=16, out=8 は base が偶数のみ。s_a=3, s_b=5 だと奇数しか読まず 50%、
      s_a=3, s_b=6 なら 100%)。in_dim / out_dim が整数でない実験条件
      (9216/8000 = 1.152)では基準位置の増分が 1 と 2 の混在になり穴が埋まる。
      使う前に `connection_stats` で被覆率を確認すること。

    Args:
        in_dim: 入力次元数
        out_dim: 出力ニューロン数
        s_a: 第1入力のずらし量(負値・in_dim超も可。剰余で丸める)
        s_b: 第2入力のずらし量
        device: 出力テンソルのデバイス

    Returns:
        indices: (2, out_dim) int64 テンソル
    """
    g = torch.arange(out_dim, dtype=torch.int64)
    base = (2 * g * int(in_dim) + int(out_dim)) // (2 * int(out_dim))
    idx_a = (base + int(s_a)) % in_dim
    idx_b = (base + int(s_b)) % in_dim
    return torch.stack([idx_a, idx_b], dim=0).to(device).contiguous()


def get_permutation_connections(
    in_dim: int,
    out_dim: int,
    generator: torch.Generator | None = None,
    device: str = 'cpu',
) -> torch.Tensor:
    """置換(均衡全射)配線(2入力固定)。

    社内先行実験 `wiring.py` の `perm_index` / `perm_layer` と同じ意味の
    配線で、**2·out_dim 個の入力スロットを in_dim 個のソースが等回数ずつ埋める**。
    各ソースの使用回数(= fan-out)は floor(2·out_dim / in_dim) と
    ceil(2·out_dim / in_dim) の 2 値しか取らないので、
    「構造的に未参照なソース」が原理的に出ない(2·out_dim >= in_dim のとき)。

    実装: q, r = divmod(2·out_dim, in_dim) として、arange(in_dim) を q 回
    タイルしたものに「**ランダムに選んだ** r 個のソース」を足してプールを作り、
    シャッフルしてから前半を idx_a、後半を idx_b にする。
    → 使用回数はちょうど q 回か q+1 回で、+1 になるソースの選び方に偏りが無い。

    先行実験の原実装は idx_a / idx_b を**独立に**均衡全射として引き、
    余りをタイルの切り詰めで決める。それは (a) in_dim <= out_dim を前提にしており
    (`wiring.py` の docstring に「w_in > w_out は本実験では発生しない…
    その場合は画素 0..w_out-1 しか使われない。要再設計」と明記)、
    先行実験の層0(9216 -> 8000)では条件を満たさない。(b) 切り詰めなので
    余りが常に先頭のソース(画像の左上側)に偏る。本実装は
    **2·out_dim スロットをまとめて均衡させ、余りを乱択する**ことでどちらも避ける。
    in_dim == out_dim(r = 0)では原実装と同じく各ソースがちょうど 2 回使われる。

    idx_a[g] == idx_b[g](同一入力ゲート)は原実装と同じく**排除しない**
    (縮約器が「同一入力」規則で数える対象として残す)。

    注意: 2·out_dim < in_dim のときは鳩の巣原理で全ソースは使えない
    (q=0, r=2·out_dim なので「ランダムに選んだ 2·out_dim 個のソースを 1 回ずつ」)。

    Args:
        in_dim: 入力次元数
        out_dim: 出力ニューロン数
        generator: 乱数生成器(Noneならグローバル RNG)
        device: 出力テンソルのデバイス

    Returns:
        indices: (2, out_dim) int64 テンソル
    """
    total = 2 * out_dim
    q, r = divmod(total, in_dim)
    parts = []
    if q:
        parts.append(torch.arange(in_dim, dtype=torch.int64).repeat(q))
    if r:
        parts.append(torch.randperm(in_dim, generator=generator)[:r])
    pool = (
        torch.cat(parts) if parts else torch.zeros(0, dtype=torch.int64)
    )
    perm = torch.randperm(total, generator=generator)
    return pool[perm].reshape(2, out_dim).to(device).contiguous()


def get_random_connections(
    in_dim: int, out_dim: int, n_inputs: int, device: str = 'cpu',
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """
    ランダムにN個の入力を各出力ニューロンに割り当てる。
    全入力が少なくとも1回使われることは保証しない。

    **注意(先行実験 T3 で確認した実測の性質)**: 本実装は
    `randperm(n_inputs·out_dim) % in_dim` を `randperm(in_dim)` で貼り替えるので、
    各入力の使用回数は floor / ceil(n_inputs·out_dim / in_dim) の 2 値しか取らない
    (= **均衡している**)。したがって `n_inputs·out_dim >= in_dim` のときは
    被覆率が常に 100% で、`get_permutation_connections` と同じ分布になる
    (in=9216 / out=8000 / n=2 で被覆 9216/9216、使用回数 1 回が 2432・2 回が 6784)。
    「各スロットを独立一様に引く」無制約の乱択が要るときは
    `get_uniform_random_connections` を使う(同条件で被覆率 約 82%)。

    Args:
        in_dim: 入力次元数
        out_dim: 出力ニューロン数
        n_inputs: 1ゲートあたりの入力数
        device: 出力テンソルのデバイス
        generator: 乱数生成器(None ならグローバル RNG = 従来動作と bitwise 不変)

    Returns:
        indices: (n_inputs, out_dim) int64 テンソル
    """
    total = n_inputs * out_dim
    c = torch.randperm(total, generator=generator) % in_dim
    c = torch.randperm(in_dim, generator=generator)[c]
    indices = c.reshape(n_inputs, out_dim).to(torch.int64).to(device)
    return indices


def get_uniform_random_connections(
    in_dim: int, out_dim: int, n_inputs: int, device: str = 'cpu',
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """無制約の一様乱択配線(各スロットを独立に一様サンプル、先行実験 RAND 腕)。

        idx[j, g] ~ Uniform{0, ..., in_dim-1}   (独立同分布)

    `get_random_connections` / `get_permutation_connections` と違い
    **使用回数を均衡させない**ので、未参照の入力が確率的に生じる。
    未参照率の期待値は (1 - 1/in_dim)^(n_inputs·out_dim) で、
    in_dim=9216 / out_dim=8000 / n_inputs=2 では exp(-16000/9216) ≈ 0.176
    (被覆率 約 82.4%。実測 1 サンプルで 0.8244)。

    Args:
        in_dim: 入力次元数
        out_dim: 出力ニューロン数
        n_inputs: 1ゲートあたりの入力数
        device: 出力テンソルのデバイス
        generator: 乱数生成器(Noneならグローバル RNG)

    Returns:
        indices: (n_inputs, out_dim) int64 テンソル
    """
    idx = torch.randint(
        0, in_dim, (n_inputs, out_dim), dtype=torch.int64, generator=generator,
    )
    return idx.to(device).contiguous()


def connection_stats(connections: torch.Tensor, in_dim: int) -> dict:
    """配線テーブルの被覆率・使用回数分布を要約する(先行実験の配線検証)。

    純粋関数(乱数もデバイス依存も無い)。腕ごとの配線が仕様どおりかを
    起動時に記録するために使う。

    Args:
        connections: (n_inputs, out_dim) の接続テーブル(int)
        in_dim: 入力次元数

    Returns:
        {
          'coverage':   読まれる入力の種類数 / in_dim (0.0〜1.0),
          'usage_hist': {'0','1','2','3+'} → その回数だけ読まれる入力の本数
                        (4 つの和は必ず in_dim),
          'same_input': 同じ入力を2回以上読むゲート数(n_inputs=2 では a == b),
          'n_gates':    ゲート数(= out_dim),
        }

    Raises:
        ValueError: 形状が2次元でない / 添字が [0, in_dim) の範囲外
    """
    if connections.dim() != 2:
        raise ValueError(
            f'connections は (n_inputs, out_dim) の2次元が必要です: '
            f'{tuple(connections.shape)}'
        )
    idx = connections.detach().to('cpu').to(torch.int64)
    if idx.numel() and (int(idx.min()) < 0 or int(idx.max()) >= in_dim):
        raise ValueError(
            f'添字が [0, {in_dim}) の範囲外です: '
            f'min={int(idx.min())}, max={int(idx.max())}'
        )
    counts = torch.bincount(idx.reshape(-1), minlength=in_dim)
    sorted_idx, _ = idx.sort(dim=0)
    dup = (sorted_idx[1:] == sorted_idx[:-1]).any(dim=0)
    return {
        'coverage': int((counts > 0).sum()) / in_dim,
        'usage_hist': {
            '0': int((counts == 0).sum()),
            '1': int((counts == 1).sum()),
            '2': int((counts == 2).sum()),
            '3+': int((counts >= 3).sum()),
        },
        'same_input': int(dup.sum()),
        'n_gates': int(idx.shape[1]),
    }


def get_unique_connections(
    in_dim: int, out_dim: int, n_inputs: int, device: str = 'cpu'
) -> torch.Tensor:
    """
    すべての入力が少なくとも1回は使われることを保証する接続。
    difflogicのオフセットペアリングアルゴリズムをN入力に一般化。

    N=2の場合はdifflogicと同一のアルゴリズム:
    - offset=0: (0,1), (2,3), (4,5), ...
    - offset=1: (1,2), (3,4), (5,6), ...
    - offset=2: (0,2), (1,3), (2,4), ...
    - out_dim個のユニークなペアが揃うまで継続

    N>2の場合: 各入力スロットに対して独立にオフセットペアリングを適用。

    Returns:
        indices: (n_inputs, out_dim) int64 テンソル
    """
    assert out_dim * n_inputs >= in_dim, (
        f'ニューロン数({out_dim}) × 入力数({n_inputs}) = {out_dim * n_inputs} が '
        f'入力次元({in_dim})以上である必要があります'
    )

    if n_inputs == 2:
        a, b = _unique_connections_n2(in_dim, out_dim)
        indices = torch.stack([a, b], dim=0)
    else:
        indices = _unique_connections_general(in_dim, out_dim, n_inputs)

    return indices.to(torch.int64).to(device).contiguous()


def _unique_connections_n2(in_dim: int, out_dim: int):
    """
    N=2用: difflogicと同一のオフセットペアリング。
    すべてのニューロンがユニークな(a,b)ペアを持つ。
    """
    x = torch.arange(in_dim).long().unsqueeze(0)

    # offset=0: 隣接ペア (0,1), (2,3), (4,5), ...
    a, b = x[..., ::2], x[..., 1::2]
    if a.shape[-1] != b.shape[-1]:
        m = min(a.shape[-1], b.shape[-1])
        a = a[..., :m]
        b = b[..., :m]

    # offset=1: 1つずらしたペア (1,2), (3,4), (5,6), ...
    if a.shape[-1] < out_dim:
        a_, b_ = x[..., 1::2], x[..., 2::2]
        a = torch.cat([a, a_], dim=-1)
        b = torch.cat([b, b_], dim=-1)
        if a.shape[-1] != b.shape[-1]:
            m = min(a.shape[-1], b.shape[-1])
            a = a[..., :m]
            b = b[..., :m]

    # offset>=2: さらに遠いペアを追加
    offset = 2
    while out_dim > a.shape[-1] > offset:
        a_, b_ = x[..., :-offset], x[..., offset:]
        a = torch.cat([a, a_], dim=-1)
        b = torch.cat([b, b_], dim=-1)
        offset += 1
        assert a.shape[-1] == b.shape[-1], (a.shape[-1], b.shape[-1])

    assert a.shape[-1] >= out_dim, (
        f'ユニークペアが不足: {a.shape[-1]} < {out_dim}'
    )
    a = a[..., :out_dim]
    b = b[..., :out_dim]

    perm = torch.randperm(out_dim)
    a = a[:, perm].squeeze(0)
    b = b[:, perm].squeeze(0)

    return a, b


def _unique_connections_general(in_dim: int, out_dim: int, n_inputs: int):
    """
    N>2用: オフセットペアリングの一般化。
    各入力スロットに対して異なるオフセットで接続を生成し、
    タプル全体の多様性を最大化する。
    """
    slots = []
    for slot in range(n_inputs):
        indices: list[torch.Tensor] = []
        # 各スロットは異なる開始オフセットでストライド
        base_offset = slot
        stride_offset = 0
        while len(indices) < out_dim:
            start = (base_offset + stride_offset) % in_dim
            # startから1つずつ取る
            chunk = torch.arange(start, in_dim)
            if stride_offset > 0:
                # 既存と重複しにくいようにシャッフル
                chunk = chunk[torch.randperm(len(chunk))]
            indices.append(chunk)
            stride_offset += 1
        slot_indices = torch.cat(indices)[:out_dim]
        slots.append(slot_indices)

    perm = torch.randperm(out_dim)
    return torch.stack([s[perm] for s in slots], dim=0)


def build_reverse_index(indices: torch.Tensor, in_dim: int):
    """
    逆引きインデックスを構築: 各入力xがどの出力yに接続されているか。
    カスタムautograd backward_x最適化用。

    Args:
        indices: (n_inputs, out_dim) 接続テーブル
        in_dim: 入力次元数

    Returns:
        (starts, flat_y, flat_slot):
        starts[x] ~ starts[x+1] が入力xに接続された出力yのインデックス範囲
        flat_y[i]: 出力ニューロンのインデックス
        flat_slot[i]: 入力スロット番号 (0..n_inputs-1)
    """
    n_inputs, out_dim = indices.shape
    device = indices.device

    given_x_y: list[list[int]] = [[] for _ in range(in_dim)]
    given_x_slot: list[list[int]] = [[] for _ in range(in_dim)]
    indices_np = indices.cpu().numpy()

    for input_slot in range(n_inputs):
        for y in range(out_dim):
            x_idx = indices_np[input_slot, y]
            given_x_y[x_idx].append(y)
            given_x_slot[x_idx].append(input_slot)

    lengths = [len(g) for g in given_x_y]
    starts = torch.tensor(
        [0] + list(torch.tensor(lengths).cumsum(0).numpy()),
        device=device, dtype=torch.int64
    )
    flat_y = torch.tensor(
        [item for sublist in given_x_y for item in sublist],
        dtype=torch.int64, device=device
    )
    flat_slot = torch.tensor(
        [item for sublist in given_x_slot for item in sublist],
        dtype=torch.int64, device=device
    )
    return starts, flat_y, flat_slot


# ---------------------------------------------------------------------------
# 構造化固定配線。内部調査リポジトリ先行実験の
# 生成器を数式で写したもの。いずれも乱数を使わない閉形式で、戻り値は
# (n_inputs, out_dim) の int64 テンソル(LutLayer の connections にそのまま渡せる)。
# 配線は IR の indices 表にしか残らないので、新しい配線種は IR・HDL・overlay に
# 変更を要求しない。
# ---------------------------------------------------------------------------


def prop_map(out_dim: int, in_dim: int) -> torch.Tensor:
    """ゲート番号 → 入力座標の比例写像 c_g = round(g · (in_dim − 1) / (out_dim − 1))。

    内部調査リポジトリ先行実験の `prop_map` と同一式(丸めは偶数丸め =
    numpy / torch の round。float64 で同じ順に演算するので numpy 版と bitwise 一致)。
    in_dim == out_dim では恒等写像、out_dim == 1 では 0。局所窓系(local1d / local2d)の
    「ゲートの中心」を決めるのに使う。`get_stretched_rot_connections` の基準位置
    floor(g·in/out + 1/2) とは式が違う(あちらは半整数を切り上げ、両端を in/out 刻み)。

    Returns:
        (out_dim,) int64 テンソル
    """
    if out_dim <= 1:
        return torch.zeros(max(int(out_dim), 0), dtype=torch.int64)
    g = torch.arange(out_dim, dtype=torch.float64)
    return torch.round(g * (in_dim - 1) / (out_dim - 1)).to(torch.int64)


def get_rot_n_connections(
    in_dim: int, out_dim: int, shifts, device: str = 'cpu', step: int = 1,
) -> torch.Tensor:
    """n 入力 rot: idx[m, g] = (g · step + shifts[m]) mod in_dim(m = 0 .. n−1)。

    `get_rot_connections` の shifts=[s_a, s_b] への一般化。歩進 step は先行実験の
    rot2(`(g·r + s) mod W_in`)。butterfly の gather 段(内部調査リポジトリ先行実験
    `bf6_gather`: `(n·i + m) mod D`)は shifts = 0..n−1、step = n に当たる
    (n·out_dim ≥ in_dim なら全入力を構成的に被覆する)。

    Args:
        in_dim: 入力次元数
        out_dim: 出力ニューロン数
        shifts: スロットごとのずらし量(長さ n_inputs)
        device: 出力テンソルのデバイス
        step: 歩進(既定 1)

    Returns:
        indices: (len(shifts), out_dim) int64 テンソル
    """
    shifts = [int(s) for s in shifts]
    if not shifts:
        raise ValueError('shifts は 1 要素以上必要です')
    g = torch.arange(out_dim, dtype=torch.int64) * int(step)
    rows = [(g + s) % in_dim for s in shifts]
    return torch.stack(rows, dim=0).to(device).contiguous()


def get_butterfly_connections(
    in_dim: int, out_dim: int, n_inputs: int, stride: int, device: str = 'cpu',
) -> torch.Tensor:
    """循環 radix-n バタフライ 1 段(内部調査リポジトリの先行実験の BF6 / BF4 の
    `bf6_stage`)。

        idx[m, g] = (g + m · stride) mod in_dim       (m = 0 .. n_inputs−1)

    Cooley-Tukey の radix-n 段を、幅が n の冪でないときは mod で巻き戻した形。
    層 s(1 始まり)で stride = n^(s−1) を使うと、k 段で各出力ノード j が
    { j + t (mod W) : 0 ≤ t < n^k } の連続区間に到達する(n 進数の全表現)。
    n^k ≥ W なら全ノードに到達し、gather 段(`get_rot_n_connections` の
    shifts 0..n−1、step n)の完全被覆と合わせて全出力が全入力に到達する。
    in_dim == out_dim のとき各入力はちょうど n 回読まれる(fan-out 均衡)。

    stride · m が mod in_dim で衝突する組合せ(例 in_dim=12, n=6, stride=6)は
    同じ入力を複数スロットで読む。必要なら `dedupe_connections` で潰す
    (先行実験の build() は常に潰していた)。

    Returns:
        indices: (n_inputs, out_dim) int64 テンソル
    """
    return get_rot_n_connections(
        in_dim, out_dim, [m * int(stride) for m in range(int(n_inputs))], device)


def butterfly_strides(n_stages: int, radix: int, order: str = 'asc') -> list[int]:
    """バタフライ段の stride 列 [radix^0, radix^1, …, radix^(n_stages−1)]
    (内部調査リポジトリの先行実験の `bf6_plan` の strides)。

    order='desc' は粗→細の逆順(先行実験の "desc" 段割当: 早い層の長距離ジャンプを
    後段の和集合伝播で複利にする)。
    """
    if order not in ('asc', 'desc'):
        raise ValueError(f"order は 'asc' / 'desc' のどちらか: {order!r}")
    strides = [int(radix) ** i for i in range(int(n_stages))]
    return strides[::-1] if order == 'desc' else strides


def get_local1d_connections(
    in_dim: int, out_dim: int, n_inputs: int, stride: int, device: str = 'cpu',
) -> torch.Tensor:
    """1D 局所窓(内部調査リポジトリの先行実験の候補 B `arm_local1d` の n 入力版)。

        c_g       = prop_map(out_dim, in_dim)
        idx[m, g] = (c_g + m · stride) mod in_dim      (m = 0 .. n_inputs−1)

    n_inputs=2 では a = c_g、b = (c_g + stride) mod in_dim で先行実験と同じ。
    in_dim == out_dim では rot(s_a=0, s_b=stride, …)に一致する。

    Returns:
        indices: (n_inputs, out_dim) int64 テンソル
    """
    c = prop_map(out_dim, in_dim)
    rows = [(c + m * int(stride)) % in_dim for m in range(int(n_inputs))]
    return torch.stack(rows, dim=0).to(device).contiguous()


def local1d_strides(depth: int, width: int, frac: float = 0.13) -> list[int]:
    """内部調査リポジトリの先行実験の `local1d_stride` の幾何級数ストライド(層 0 .. depth−1)。

        S_max = max(1, floor(frac · width))
        s_l   = clip(round(S_max^(l / (depth−1))), 1, S_max)

    frac=0.13 は先行実験の既定(幅の 13% = 先行実験@w2000 と同等の窓)。
    """
    smax = max(1, int(math.floor(frac * width)))
    if depth <= 1:
        return [1] * max(int(depth), 0)
    out = []
    for layer in range(int(depth)):
        s = int(round(smax ** (layer / (depth - 1))))
        out.append(int(min(max(s, 1), smax)))
    return out


def ring_offsets(n: int) -> list[tuple[int, int]]:
    """半径 s のリング 4 方向 [(s,0), (0,s), (s,s), (s,−s)] を s = 1, 2, … と並べた列の
    先頭 n 個(内部調査リポジトリ先行実験の `ring_seq`)。要素は (dx, dy)。

    先頭 6 個は先行実験の C2D_OFFSETS [(1,0),(0,1),(1,1),(1,−1),(2,0),(0,2)] に一致する。
    """
    out = []
    for k in range(int(n)):
        s = k // 4 + 1
        out.append([(s, 0), (0, s), (s, s), (s, -s)][k % 4])
    return out


def local2d_offset_schedule(
    n_layers: int, n_inputs: int, assign: str = 'contig',
) -> list[list[tuple[int, int]]]:
    """隠れ層 n_layers 本それぞれの n_inputs 個のオフセット表
    (内部調査リポジトリの先行実験の `l2d6_offsets` の contig / stride 割当)。

    contig: 層 l(0 始まり)は ring 列の [n·l, n·(l+1))。n_inputs=4 では層 l が
            半径 l+1 のリング 1 周になる(先行実験 L2D4 の設計)。
    stride: 層 l は ring 列の l から n_layers 飛びで n 個(各層が多スケール混合)。
    """
    n_layers, n_inputs = int(n_layers), int(n_inputs)
    seq = ring_offsets(n_inputs * n_layers)
    if assign == 'contig':
        return [seq[n_inputs * l:n_inputs * (l + 1)] for l in range(n_layers)]
    if assign == 'stride':
        return [[seq[l + n_layers * j] for j in range(n_inputs)]
                for l in range(n_layers)]
    raise ValueError(f"assign は 'contig' / 'stride' のどちらか: {assign!r}")


def get_local2d_connections(
    in_dim: int, out_dim: int, offsets, grid=None, bits_per_cell: int = 1,
    device: str = 'cpu',
) -> torch.Tensor:
    """2D 局所窓(内部調査リポジトリの先行実験の候補 C / L2D6 / L2D4)。

    入力 in_dim ビットを「(H, W) 格子のセル × bits_per_cell ビット」と見なす
    (bit = cell · z + t の feature-major。サーモメータ z 本の画像なら z = bits_per_cell)。
    ゲート g の中心セル c_g = prop_map(out_dim, n_cells)(n_cells = in_dim / z)、
    (y, x) = divmod(c_g, W)。オフセット k = (dx, dy) ごとにトーラスで隣接セルを取り、
    格子がセル数より大きい分(正方格子の末尾)は mod n_cells で巻き戻す:

        cell_k(g)         = (((y + dy) mod H) · W + ((x + dx) mod W)) mod n_cells
        idx[k · z + t, g] = cell_k(g) · z + t        (t = 0 .. z−1)

    grid=None は G = ceil(√n_cells) の正方格子(隠れ層の既定。in_dim == out_dim なら
    c_g = g で先行実験の隠れ層式に一致)。画像の層 0 は grid=(H, W) を渡す:
    先行実験 L2D6 層 0 = grid (28, 28)、z=3、offsets [(1,0),(0,1)](右隣・下隣の
    2 画素 × 3 bit)、先行実験 L2D4 層 0 = z=4、offsets [(0,0)](自画素の 4 bit)。

    Args:
        in_dim: 入力次元数(bits_per_cell の倍数)
        out_dim: 出力ニューロン数
        offsets: (dx, dy) の列。`ring_offsets` / `local2d_offset_schedule` が作る
        grid: (H, W)。None なら正方格子
        bits_per_cell: セル当たりのビット数 z
        device: 出力テンソルのデバイス

    Returns:
        indices: (len(offsets) · bits_per_cell, out_dim) int64 テンソル
    """
    z = int(bits_per_cell)
    if z < 1 or in_dim % z != 0:
        raise ValueError(f'in_dim({in_dim}) は bits_per_cell({z}) の倍数が必要です')
    n_cells = in_dim // z
    if grid is None:
        side = int(math.ceil(math.sqrt(n_cells)))
        grid = (side, side)
    h, w = int(grid[0]), int(grid[1])
    if h * w < n_cells:
        raise ValueError(f'格子 {(h, w)} はセル数 {n_cells} を収められません')
    offsets = [(int(dx), int(dy)) for dx, dy in offsets]
    if not offsets:
        raise ValueError('offsets は 1 要素以上必要です')
    c = prop_map(out_dim, n_cells)
    y, x = c // w, c % w
    t = torch.arange(z, dtype=torch.int64)
    rows = []
    for dx, dy in offsets:
        cell = (((y + dy) % h) * w + ((x + dx) % w)) % n_cells
        rows.append(cell.unsqueeze(0) * z + t.unsqueeze(1))   # (z, out_dim)
    return torch.cat(rows, dim=0).to(device).contiguous()


def dedupe_connections(indices: torch.Tensor, in_dim: int) -> torch.Tensor:
    """同じ入力を複数スロットで読むゲートの重複を決定的に潰す(+1 ずらし)。

    内部調査リポジトリの先行実験の `_fix_degenerate`(2 入力: a == b なら b ← (b+1) mod in_dim)
    と `dedupe_ins`(n 入力: スロット順に見て既出の添字なら +1 ずつ進めて
    未使用の添字にする)の共通化。重複の無いゲートは変えない。

    Args:
        indices: (n_inputs, out_dim) の接続テーブル(int)
        in_dim: 入力次元数

    Returns:
        (n_inputs, out_dim) int64 テンソル(新しいテンソル。入力は変更しない)

    Raises:
        ValueError: 形状が 2 次元でない / n_inputs > in_dim で潰しきれない
    """
    if indices.dim() != 2:
        raise ValueError(
            f'indices は (n_inputs, out_dim) の2次元が必要です: {tuple(indices.shape)}')
    n, _ = indices.shape
    if n > in_dim:
        raise ValueError(f'n_inputs({n}) > in_dim({in_dim}) では重複を潰しきれません')
    idx = indices.detach().to('cpu').to(torch.int64).clone()
    sorted_idx, _ = idx.sort(dim=0)
    dup_cols = (sorted_idx[1:] == sorted_idx[:-1]).any(dim=0).nonzero().flatten().tolist()
    for g in dup_cols:
        seen: set[int] = set()
        for m in range(n):
            v = int(idx[m, g])
            while v in seen:
                v = (v + 1) % in_dim
            idx[m, g] = v
            seen.add(v)
    return idx.to(indices.device).contiguous()


# --- 配線種の単一の入口 ---

# 配線種ごとに受け付ける引数名(make_connections の spec)。エディタの配線生成 API と
# ir/build.py(draft ノードの train_info.wiring)の検証にも使う
WIRING_SPEC_KEYS: dict[str, tuple[str, ...]] = {
    'random': ('seed',),
    'random_seeded': ('seed',),
    'random_uniform': ('seed',),
    'unique': (),
    'perm': ('seed',),
    'rot': ('s_a', 's_b', 'step'),
    'stretched_rot': ('s_a', 's_b'),
    'rot_n': ('shifts', 'step'),
    'butterfly': ('stride',),
    'local1d': ('stride',),
    'local2d': ('offsets', 'grid', 'bits_per_cell'),
}
WIRING_KINDS = tuple(WIRING_SPEC_KEYS)
# 乱数を引かず引数だけで決まる種(エディタのインスペクタに出す候補)
STRUCTURED_WIRING_KINDS = ('rot', 'stretched_rot', 'rot_n', 'butterfly', 'local1d', 'local2d')


def make_connections(
    kind: str, in_dim: int, out_dim: int, n_inputs: int, *,
    device: str = 'cpu', generator=None, dedupe: bool = False, **spec,
) -> torch.Tensor:
    """配線種と引数から 1 層分の接続テーブル (n_inputs, out_dim) を作る。

    レシピ(training/runner.py)、IR の draft ノード(ir/build.py の
    train_info.connections + wiring)、エディタの配線生成 API が同じ関数を呼ぶ
    単一の入口。種ごとの引数は WIRING_SPEC_KEYS のとおり:

        random / random_seeded / random_uniform / perm : seed(任意。generator 未指定時)
        unique        : なし
        rot           : s_a, s_b, step(既定 1。2 入力固定)
        stretched_rot : s_a, s_b
        rot_n         : shifts(長さ n_inputs), step(既定 1)
        butterfly     : stride
        local1d       : stride
        local2d       : offsets, grid(任意), bits_per_cell(既定 1)

    Args:
        kind: 配線種(WIRING_KINDS のいずれか)
        in_dim / out_dim / n_inputs: 層の形
        device: 返すテンソルの配置先
        generator: 乱数を引く種で使う torch.Generator(省略時は seed から作る)
        dedupe: 真なら重複ピンを dedupe_connections で潰す
        **spec: 種ごとの引数

    Returns:
        (n_inputs, out_dim) の int64 テンソル

    Raises:
        ValueError: 未知の種、必須引数の欠落、受け付けない引数、入力数の不一致
    """
    if kind not in WIRING_SPEC_KEYS:
        raise ValueError(f'未知の配線種: {kind!r}(対応: {", ".join(WIRING_KINDS)})')
    allowed = WIRING_SPEC_KEYS[kind]
    spec = {k: v for k, v in spec.items() if v is not None}
    unknown = sorted(set(spec) - set(allowed))
    if unknown:
        raise ValueError(
            f"配線種 '{kind}' は引数 {unknown} を受け付けません(対応: {list(allowed)})")

    def need(name: str):
        if name not in spec:
            raise ValueError(f"配線種 '{kind}' には {name} が必要です")
        return spec[name]

    if generator is None and spec.get('seed') is not None:
        generator = torch.Generator().manual_seed(int(spec['seed']))

    if kind in ('random', 'random_seeded'):
        idx = get_random_connections(in_dim, out_dim, n_inputs, device, generator)
    elif kind == 'random_uniform':
        idx = get_uniform_random_connections(in_dim, out_dim, n_inputs, device, generator)
    elif kind == 'unique':
        idx = get_unique_connections(in_dim, out_dim, n_inputs, device)
    elif kind == 'perm':
        idx = get_permutation_connections(in_dim, out_dim, generator, device)
    elif kind == 'rot':
        idx = get_rot_connections(
            in_dim, out_dim, int(need('s_a')), int(need('s_b')), device,
            step=int(spec.get('step', 1)))
    elif kind == 'stretched_rot':
        idx = get_stretched_rot_connections(
            in_dim, out_dim, int(need('s_a')), int(need('s_b')), device)
    elif kind == 'rot_n':
        idx = get_rot_n_connections(
            in_dim, out_dim, [int(s) for s in need('shifts')], device,
            step=int(spec.get('step', 1)))
    elif kind == 'butterfly':
        idx = get_butterfly_connections(in_dim, out_dim, n_inputs, int(need('stride')), device)
    elif kind == 'local1d':
        idx = get_local1d_connections(in_dim, out_dim, n_inputs, int(need('stride')), device)
    else:  # local2d
        grid = spec.get('grid')
        idx = get_local2d_connections(
            in_dim, out_dim, [tuple(int(v) for v in o) for o in need('offsets')],
            tuple(int(v) for v in grid) if grid is not None else None,
            int(spec.get('bits_per_cell', 1)), device)
    if idx.shape[0] != n_inputs:
        raise ValueError(
            f"配線種 '{kind}' の入力数 {idx.shape[0]} が n_inputs({n_inputs})と一致しません")
    if dedupe:
        idx = dedupe_connections(idx, in_dim)
    return idx
