"""`lut_flat`の段(レベル)分け — GPU 実行用の並べ替え。

`lut_flat` のゲートはトポロジカル順に並んでいるだけで、隣り合うゲートが
依存しうるので、そのままでは GPU の 1 スレッド = 1 ゲートで並列に回せない。
ここでは各ゲートに **ASAP の段番号**(入力 = 段 0、ゲート = 1 + 入力の段の最大)
を振り、段ごとに連続するようゲートを並べ替える。同じ段のゲートは互いに
依存しないので、段ごとに 1 回の起動(または 1 回のバリア区間)で全ゲートを
同時に評価できる。

実モデルの段数は置換前の lut_layer の層数を超えない(パススルーを配線へ
畳んでも、ゲート g の入力は「元の 1 つ前の層以前」の信号なので)。HAR は
k600/k2400 が 4 段、k6000 が 6 段で、段の幅は置換前の層幅の 47〜58% 減。

あわせて **参照される入力ビットだけ** を信号表へ詰め直す(Web 版
`tools/webbench/src/engines/lutFlat.ts::compileLutFlat` と同じ考え方。
HAR k600 は入力 4,608 本のうち 668 本しか読まない)。

並べ替え後の信号番号:
    0                      定数0
    1 .. n_in              入力ビット in_idx[u](元の入力番号)→ 信号 1+u
    n_in+1 ..              ゲート(段の昇順、段内は元の順)
出力 `outputs[i]` は `2*信号番号 + 極性`(元の規約のまま番号だけ付け替え)。
"""

from dataclasses import dataclass

import numpy as np

__all__ = ['LevelizedFlat', 'levelize_lut_flat', 'eval_levelized']


@dataclass(frozen=True)
class LevelizedFlat:
    """段分け済み `lut_flat`。GPU バックエンドがそのままアップロードする。"""

    in_width: int
    in_idx: list[int]          # 信号 1+u ← 入力 in_idx[u]
    tt: list[int]              # 並べ替え後のゲート順
    a: list[int]
    b: list[int]
    outputs: list[int]
    level_off: list[int]       # 段 L のゲートは [level_off[L], level_off[L+1])

    @property
    def n_in(self) -> int:
        return len(self.in_idx)

    @property
    def n_gates(self) -> int:
        return len(self.tt)

    @property
    def n_levels(self) -> int:
        return len(self.level_off) - 1

    @property
    def n_signals(self) -> int:
        return 1 + self.n_in + self.n_gates

    @property
    def level_widths(self) -> list[int]:
        return [self.level_off[i + 1] - self.level_off[i]
                for i in range(self.n_levels)]


def levelize_lut_flat(params: dict) -> LevelizedFlat:
    """`lut_flat` の params → 段分け済みネットリスト(意味は元と同一)。"""
    in_w = int(params['in_width'])
    gates = [(int(t), int(x), int(y)) for t, x, y in params['gates']]
    outputs = [int(v) for v in params['outputs']]
    n_g = len(gates)
    n_sig = 1 + in_w + n_g

    level = np.zeros(n_sig, dtype=np.int64)
    for g, (_, x, y) in enumerate(gates):
        s = 1 + in_w + g
        if not (0 <= x < s and 0 <= y < s):
            raise ValueError(f'lut_flat: ゲート{g}が自分以降の信号を参照している')
        level[s] = 1 + max(level[x], level[y])

    # 参照される入力(昇順)
    used = set()
    for _, x, y in gates:
        for s in (x, y):
            if 1 <= s <= in_w:
                used.add(s)
    for r in outputs:
        s = r >> 1
        if not 0 <= s < n_sig:
            raise ValueError(f'lut_flat: 出力参照 {r} が信号数 {n_sig} を超える')
        if 1 <= s <= in_w:
            used.add(s)
    in_sig = sorted(used)
    n_in = len(in_sig)

    remap = np.full(n_sig, -1, dtype=np.int64)
    remap[0] = 0
    for u, s in enumerate(in_sig):
        remap[s] = 1 + u
    glevel = level[1 + in_w:]
    order = sorted(range(n_g), key=lambda g: (int(glevel[g]), g))   # 安定
    for new, g in enumerate(order):
        remap[1 + in_w + g] = 1 + n_in + new

    n_levels = int(glevel.max()) if n_g else 0
    level_off = [0] * (n_levels + 1)
    counts = np.bincount(glevel, minlength=n_levels + 1)
    acc = 0
    for lv in range(1, n_levels + 1):
        level_off[lv - 1] = acc
        acc += int(counts[lv])
    level_off[n_levels] = acc

    tt, a, b = [], [], []
    for g in order:
        t, x, y = gates[g]
        tt.append(t & 15)
        a.append(int(remap[x]))
        b.append(int(remap[y]))
    outs = [int(remap[r >> 1]) * 2 + (r & 1) for r in outputs]
    return LevelizedFlat(in_width=in_w, in_idx=[s - 1 for s in in_sig],
                         tt=tt, a=a, b=b, outputs=outs, level_off=level_off)


def eval_levelized(lf: LevelizedFlat, x: np.ndarray) -> np.ndarray:
    """段分け済みネットリストを numpy で評価する(テスト・検算用の素朴な実装)。

    段ごとに全ゲートを一括で評価するので、「同じ段のゲートは互いに依存しない」
    ことの検査にもなる(依存していれば結果が元と食い違う)。

    Args:
        x: (N, in_width) の 0/1
    Returns:
        (N, out_width) の 0/1(uint8)
    """
    x = np.asarray(x, dtype=np.uint8)
    n = x.shape[0]
    sig = np.zeros((n, lf.n_signals), dtype=np.uint8)
    sig[:, 1:1 + lf.n_in] = x[:, lf.in_idx]
    tt = np.asarray(lf.tt, dtype=np.uint8)
    a = np.asarray(lf.a, dtype=np.int64)
    b = np.asarray(lf.b, dtype=np.int64)
    base = 1 + lf.n_in
    for lv in range(lf.n_levels):
        g0, g1 = lf.level_off[lv], lf.level_off[lv + 1]
        addr = (sig[:, b[g0:g1]] << 1) | sig[:, a[g0:g1]]
        sig[:, base + g0:base + g1] = (tt[g0:g1][None, :] >> addr) & 1
    o = np.asarray(lf.outputs, dtype=np.int64)
    return sig[:, o >> 1] ^ (o & 1).astype(np.uint8)[None, :]
