"""conv-DLGN の木カーネルの「葉の読み座標」サンプラ。

内部調査リポジトリの先行実験の `samplers.py` の数式を写したもの。木の形(深さ d の完全
二分木・葉 2^d・ゲート 2^d − 1)は腕によらず同一で、**葉がどの (入力チャネル, 窓内 dy,
窓内 dx) を読むか** だけを差し替える。戻り値は常に (C_out, L) の int64 テンソル 3 つ
(sel_ch / sel_dy / sel_dx)。dy, dx は窓内座標 [0, K) で、実入力座標は
iy = oy·stride + dy − pad になる(LutConv2d の unfold と同じ)。

腕:
  random8   — 3×3 窓 × num_chn ch の候補から 8 葉を乱択(cdlgn `generate_selection` 忠実。
              対の相手と同一読みなら引き直す)。座標は乱数 = HW ではカーネルごとの座標表が要る
  cross5    — 5×5 窓の中心 + 距離 2 の十字 4 点 = 5 点 → 8 葉(位相 25 通り × 5 回転)
  dilated8  — 5×5 窓の「2 マス空け」9 点から 1 点を落とした 8 点(位相 4 × drop 9 × 8 回転)
  checker13 — 5×5 市松 13 点 → 16 葉(深さ 4。位相 2 × 13 回転)
cross5 / dilated8 / checker13 は座標が (出力チャネル, 層) の計算可能関数 = 座標ストアゼロ。

読み点数 × 木サイズ(先行実験の表):
  | 腕        | 窓  | distinct 読み点 | 葉 | 深さ | ゲート/木 |
  | cross5    | 5×5 |  5              |  8 | 3    | 7         |
  | random8   | 3×3 |  8(乱数)       |  8 | 3    | 7         |
  | dilated8  | 5×5 |  8              |  8 | 3    | 7         |
  | checker13 | 5×5 | 13              | 16 | 4    | 15        |

対(ペアリング)の規約: 先行実験は各段で前半 / 後半に半割して対にする(葉 j と j + L/2)。
ライブラリの LutConv2d は隣接対(葉 2i と 2i+1、`_tree_forward_*` の reshape)。両者は
葉番号のビット反転で写り合う同型の完全二分木なので、`leaf_indices_flat` が先行実験の
葉 j を LutConv2d の葉 bitrev(j) に置く。これで「対の相手と同一読みを避ける」性質
(random8 の衝突回避、パターン腕の退化ゲート無し)がそのまま保たれる。
"""

from __future__ import annotations

import torch

# 腕 → 木と窓の形。padding は空間サイズを保つ (K−1)//2
SAMPLER_SPECS: dict[str, dict] = {
    'random8':   {'kernel': 3, 'depth': 3, 'padding': 1, 'computable': False, 'n_points': None},
    'cross5':    {'kernel': 5, 'depth': 3, 'padding': 2, 'computable': True, 'n_points': 5},
    'dilated8':  {'kernel': 5, 'depth': 3, 'padding': 2, 'computable': True, 'n_points': 8},
    'checker13': {'kernel': 5, 'depth': 4, 'padding': 2, 'computable': True, 'n_points': 13},
}


# ---------------------------------------------------------------- チャネル規則

def channel_rule_computable(c_out: int, c_in: int, n_leaves: int, layer_idx: int,
                            num_chn: int = 2) -> torch.Tensor:
    """計算可能なチャネル選択(パターン腕)。

    出力チャネル c は入力チャネル対 base = (c + layer) mod C_in、base + t·(C_in // 2)
    (t < num_chn、mod C_in)を見る。葉 j はそのうち (j + c) mod num_chn 番目を読む
    (葉ごとに交互 = num_chn 制限を維持)。HW では c のカウンタと定数加算だけで生成できる。
    """
    c = torch.arange(c_out, dtype=torch.int64)
    j = torch.arange(n_leaves, dtype=torch.int64)
    half = max(1, c_in // 2)
    base = (c + layer_idx) % c_in                                            # (C,)
    cand = torch.stack([(base + t * half) % c_in for t in range(num_chn)], dim=1)   # (C, num_chn)
    pick = (j.view(1, -1) + c.view(-1, 1)) % num_chn                         # (C, L)
    return torch.gather(cand, 1, pick)


def channel_rule_random(c_out: int, c_in: int, n_leaves: int, num_chn: int = 2,
                        generator: torch.Generator | None = None) -> torch.Tensor:
    """cdlgn 忠実: 出力チャネルごとに num_chn 本を復元抽出し、各葉がそのどれかを引く。"""
    n = min(num_chn, c_in)
    cand = torch.stack([torch.randint(0, c_in, (c_out,), generator=generator)
                        for _ in range(n)], dim=1)                            # (C, n)
    pick = torch.randint(0, n, (c_out, n_leaves), generator=generator)
    return torch.gather(cand, 1, pick)


# ---------------------------------------------------------------- 腕 a: random8

def sampler_random8(c_out: int, c_in: int, layer_idx: int = 0, kernel: int = 3,
                    n_leaves: int = 8, num_chn: int = 2,
                    generator: torch.Generator | None = None):
    """cdlgn `generate_selection` の忠実移植(対の定義は半割ペアリング: 葉 j と j + L/2)。

    対の相手と同一読み(同チャネル・同座標)なら相手側を引き直す。同じ位置でも
    チャネルが違えば可(原実装と同条件)。
    """
    sel_ch = channel_rule_random(c_out, c_in, n_leaves, num_chn, generator)
    dy = torch.randint(0, kernel, (c_out, n_leaves), generator=generator)
    dx = torch.randint(0, kernel, (c_out, n_leaves), generator=generator)
    half = n_leaves // 2
    for j in range(half):
        k = j + half
        for _ in range(64):                     # 実測 1〜2 回で解消。無限ループ防止の上限
            same = ((sel_ch[:, j] == sel_ch[:, k]) & (dy[:, j] == dy[:, k])
                    & (dx[:, j] == dx[:, k]))
            if not bool(same.any()):
                break
            n = int(same.sum())
            dy[same, k] = torch.randint(0, kernel, (n,), generator=generator)
            dx[same, k] = torch.randint(0, kernel, (n,), generator=generator)
        else:
            raise RuntimeError('random8 の衝突回避が収束しなかった')
    return sel_ch, dy, dx


# ---------------------------------------------------------------- 腕 b: パターン

def dilated8_points(phase_y: int, phase_x: int, drop: int,
                    kernel: int = 5) -> list[tuple[int, int]]:
    """2 マス空け 3×3 格子(トーラス位相 shift)から 1 点を落とした 8 点、raster 順。

    点 (i, j) → ((2i + phase_y) mod K, (2j + phase_x) mod K)、raster 順の `drop` 番目を除外。
    drop を固定(例: 中心)にすると位相 (0,0) の中心など 4 点が層全体で誰にも読まれなくなる
    (先行実験 selfcheck T7c が 21/25 で検出)。drop も (チャネル, 層) の関数にして被覆 25/25。
    """
    pts = [((2 * i + phase_y) % kernel, (2 * j + phase_x) % kernel)
           for i in range(3) for j in range(3)]
    del pts[drop % 9]
    return pts


def cross5_points(phase_y: int, phase_x: int, kernel: int = 5) -> list[tuple[int, int]]:
    """中心 + 距離 arm(= K // 2)の十字 4 点 = 5 点、raster 順。トーラス位相 shift 付き。

    位相 (py, px) が 5×5 = 25 通りを巡れば中心点だけで窓の全 25 画素を覆う。
    """
    arm = kernel // 2
    base = [(0, arm), (arm, 0), (arm, arm), (arm, kernel - 1), (kernel - 1, arm)]
    pts = [((dy + phase_y) % kernel, (dx + phase_x) % kernel) for dy, dx in base]
    assert len(set(pts)) == 5, pts
    return pts


def checker13_points(phase: int, kernel: int = 5) -> list[tuple[int, int]]:
    """5×5 市松 13 点、raster 順。

    phase=0: (dy + dx) が偶数の 13 点 / phase=1: 奇数の 12 点 + 中心(13 点に揃える規則)。
    2 位相の和集合が 5×5 全域。
    """
    pts = [(dy, dx) for dy in range(kernel) for dx in range(kernel)
           if (dy + dx) % 2 == phase]
    if phase == 1:
        pts.append((kernel // 2, kernel // 2))
    assert len(pts) == 13, len(pts)
    return pts


def _pattern_coords(c_out: int, layer_idx: int, n_leaves: int, point_fn, rot_mult: int,
                    kernel: int):
    """(出力チャネル, 層) の計算可能関数で葉座標を作る共通ルーチン。

    point_fn(c, layer) → (キャッシュキー, 点リスト)。回転 rot = (c·rot_mult + layer) mod P
    (P = 点数)で「葉 j → 点 (j + rot) mod P」。P < L の場合(checker13: 13 点 → 16 葉、
    cross5: 5 点 → 8 葉)はこの式が自動的に重複読みのパディング規則になり、半割ペアリング
    下で「同一点同士の対」を作らない(葉 j と j + L/2 は点 j + rot と j + L/2 + rot で常に別)。
    """
    sel_dy = torch.zeros(c_out, n_leaves, dtype=torch.int64)
    sel_dx = torch.zeros(c_out, n_leaves, dtype=torch.int64)
    cache: dict = {}
    for c in range(c_out):
        key, pts = point_fn(c, layer_idx)
        pts = cache.setdefault(key, pts)
        p = len(pts)
        rot = (c * rot_mult + layer_idx) % p
        for j in range(n_leaves):
            dy, dx = pts[(j + rot) % p]
            sel_dy[c, j], sel_dx[c, j] = dy, dx
    assert int(sel_dy.max()) < kernel and int(sel_dx.max()) < kernel
    return sel_dy, sel_dx


def sampler_dilated8(c_out, c_in, layer_idx=0, kernel=5, n_leaves=8, num_chn=2,
                     generator=None):
    """位相 4 通り(2 軸 × 偶奇)× 落とす点 9 通り × 8 回転。すべて (c, 層) の計算可能関数。

    位相 ph = (c + layer) mod 4 → (py, px) = (ph mod 2, ph // 2)、drop = (c + layer) mod 9。
    gcd(4, 9) = 1 なので同一位相のチャネル群の中で drop は 9 通り全部を巡る。
    """
    def point_fn(c, layer):
        ph = (c + layer) % 4
        drop = (c + layer) % 9
        return (ph, drop), dilated8_points(ph % 2, ph // 2, drop, kernel)
    sel_dy, sel_dx = _pattern_coords(c_out, layer_idx, n_leaves, point_fn, 1, kernel)
    sel_ch = channel_rule_computable(c_out, c_in, n_leaves, layer_idx, num_chn)
    return sel_ch, sel_dy, sel_dx


def sampler_cross5(c_out, c_in, layer_idx=0, kernel=5, n_leaves=8, num_chn=2,
                   generator=None):
    """5 点 → 8 葉。位相 25 通り × 5 回転(rot = (3c + layer) mod 5。3 は 5 と互いに素)。

    葉 5, 6, 7 は葉 0, 1, 2 と同じ点の重複読みになるが、チャネル規則 (j + c) mod 2 が
    5 が奇数ゆえ必ず別チャネルになるので完全な冗長ではない。
    """
    def point_fn(c, layer):
        ph = (c + layer) % 25
        return ph, cross5_points(ph % 5, ph // 5, kernel)
    sel_dy, sel_dx = _pattern_coords(c_out, layer_idx, n_leaves, point_fn, 3, kernel)
    sel_ch = channel_rule_computable(c_out, c_in, n_leaves, layer_idx, num_chn)
    return sel_ch, sel_dy, sel_dx


def sampler_checker13(c_out, c_in, layer_idx=0, kernel=5, n_leaves=16, num_chn=2,
                      generator=None):
    """市松 2 位相 × 13 回転(乗数 7 は 13 と互いに素なのでチャネル方向に一巡する)。"""
    def point_fn(c, layer):
        ph = (c + layer) % 2
        return ph, checker13_points(ph, kernel)
    sel_dy, sel_dx = _pattern_coords(c_out, layer_idx, n_leaves, point_fn, 7, kernel)
    sel_ch = channel_rule_computable(c_out, c_in, n_leaves, layer_idx, num_chn)
    return sel_ch, sel_dy, sel_dx


SAMPLER_FNS = {
    'random8': sampler_random8,
    'cross5': sampler_cross5,
    'dilated8': sampler_dilated8,
    'checker13': sampler_checker13,
}


def build_leaf_selection(name: str, c_out: int, c_in: int, layer_idx: int = 0,
                         num_chn: int = 2, generator: torch.Generator | None = None):
    """腕名から (sel_ch, sel_dy, sel_dx) を返す(窓・木の形は SAMPLER_SPECS[name])。"""
    if name not in SAMPLER_SPECS:
        raise ValueError(f'未知のサンプラ: {name!r}(対応: {sorted(SAMPLER_SPECS)})')
    spec = SAMPLER_SPECS[name]
    n_leaves = 2 ** spec['depth']
    return SAMPLER_FNS[name](c_out, c_in, layer_idx=layer_idx, kernel=spec['kernel'],
                             n_leaves=n_leaves, num_chn=num_chn, generator=generator)


# ---------------------------------------------------------------- LutConv2d への写像

def bit_reversal_permutation(n_leaves: int) -> torch.Tensor:
    """葉番号のビット反転置換(n_leaves = 2^d)。perm[j] = bitrev_d(j)。

    半割ペアリング(先行実験: 葉 j と j + L/2 が対)の木を、隣接ペアリング
    (LutConv2d: 葉 2i と 2i+1 が対)の木へ写す葉の並べ替え。
    """
    d = n_leaves.bit_length() - 1
    if 1 << d != n_leaves:
        raise ValueError(f'n_leaves は 2 の冪が必要です: {n_leaves}')
    perm = torch.zeros(n_leaves, dtype=torch.int64)
    for j in range(n_leaves):
        r = 0
        for b in range(d):
            r |= ((j >> b) & 1) << (d - 1 - b)
        perm[j] = r
    return perm


def leaf_indices_flat(sel_ch: torch.Tensor, sel_dy: torch.Tensor, sel_dx: torch.Tensor,
                      kernel_h: int, kernel_w: int) -> torch.Tensor:
    """先行実験と同じ形式の葉座標を LutConv2d の `indices_flat`(unfold 後の平坦添字)に変換する。

    flat = ch · kH · kW + dy · kW + dx(unfold の次元 1 の並び)。先行実験の葉 j を
    LutConv2d の葉 bitrev(j) に置く(半割ペアリング → 隣接ペアリング)。
    """
    flat = (sel_ch.to(torch.int64) * (kernel_h * kernel_w)
            + sel_dy.to(torch.int64) * kernel_w + sel_dx.to(torch.int64))
    n_leaves = flat.shape[1]
    perm = bit_reversal_permutation(n_leaves)
    out = torch.empty_like(flat)
    out[:, perm] = flat
    return out.contiguous()
