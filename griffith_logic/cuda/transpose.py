"""
2次元テンソルの高速転置(`x.t().contiguous()` 相当)。

LUT層のCUDAカーネルは (dim, batch) レイアウトを要求するため、境界(非LUT層由来の
入力・損失からの勾配)で巨大な転置コピーが発生する。素の `.t().contiguous()` は
ストライドアクセスで非コアレストなため帯域を活かせない。ここではタイル化した
Triton転置カーネルを用意し、利用可能な最速経路を自動選択する。

重要: 前段がLUT層の場合 `x.t()` は既にcontiguousで、その場合は**コピーを行わない**
(連鎖内側のno-opを壊すとコピーが激増して逆に遅くなる)。

経路の強制切替: 環境変数 `MD_TRANSPOSE=triton|torch`(既定=auto)。
"""

import os

_backend = None          # 'triton' | 'torch'
_logged = False
_triton_kernel = None

# Tritonタイルサイズ(32の倍数。境界はマスクで処理するため任意形状に対応)
_BLOCK_M = 64
_BLOCK_N = 64

# Triton経路を使う最小要素数。
#
# 【転置単体の実測】A10G(float32, ms / experiments/bench_transpose.py):
#   (128,1024)=131k    torch 0.012 / triton 0.026  ← Triton起動オーバーヘッド負け
#   (512,1024)=524k    torch 0.014 / triton 0.025  ← 負け
#   (1024,1024)=1.05M  torch 0.026 / triton 0.026  ← 同等
#   (1024,4096)=4.2M   torch 0.092 / triton 0.075  ← 勝ち
#   (1024,65536)=67M   torch 3.988 / triton 1.124  ← 3.5x
#   (1024,400000)=410M torch 36.93 / triton 6.848  ← 5.4x (89→479 GB/s)
#
# 【重要】転置単体のクロスオーバー(約1M)を閾値にすると**実経路で退行する**。
# 転置は単独で走るのではなくLUTカーネルの直前後に挟まるため、Triton経路が背負う
# ホスト側コスト(torch.empty + grid計算 + Tritonランチャ)が、隣接する小さな
# LUTカーネルの実行時間に隠れずそのまま加算されるからである。
#
# そこで**実経路(単層LutLayer n=2/softmax の fwd+bwd)** で、変更前
# (`x.t().contiguous()`)と既定経路を同一プロセス内で交互に実測した
# (A10G, 中央値 s/it, experiments/bench_kernel_gate.py --only sweep)。
# speedup = 変更前 / 既定経路。1.0未満が退行:
#
#   width    batch    numel    変更前     既定経路   speedup
#    1024       64      65k   0.000499   0.000505   0.989x
#    1024      128     131k   0.000505   0.000510   0.991x
#    1024      512     524k   0.000499   0.000498   1.002x
#    1024     1024    1.05M   0.000515   0.000593   0.870x  ← 退行
#    4096       64     262k   0.000524   0.000518   1.013x
#    4096      128     524k   0.000519   0.000522   0.995x
#    4096      512    2.10M   0.000578   0.000655   0.882x  ← 退行
#    4096     1024    4.19M   0.000973   0.000954   1.019x
#   16384       64    1.05M   0.000523   0.000593   0.883x  ← 退行
#   16384      128    2.10M   0.000608   0.000678   0.897x  ← 退行
#   16384      512    8.39M   0.001884   0.001829   1.030x
#   16384     1024   16.8M    0.003741   0.003441   1.087x
#   65536       64    4.19M   0.001463   0.001513   0.967x
#   65536      128    8.39M   0.002980   0.002521   1.182x
#   65536      512   33.6M    0.012752   0.009666   1.319x
#   65536     1024   67.1M    0.026484   0.019739   1.342x
#  400000       64   25.6M    0.009908   0.009206   1.076x
#  400000      128   51.2M    0.021907   0.015944   1.374x
#  400000      512  204.8M    0.099865   0.065170   1.532x
#  400000     1024  409.6M    0.213960   0.145244   1.473x
#
# 退行は 1.05M〜4.19M に集中し、8.39M 以上は全点で 1.03x 以上の改善になる
# (4.19M は 0.967x / 1.019x で有意差なし)。よって閾値は **8M(1<<23)** とする。
# これより下は `x.t().contiguous()` = 最適化前と同一の1行に落ちるので、
# 退行は原理的に起きない(fast_transpose の「退行ゲート」を参照)。
_MIN_NUMEL_FOR_TRITON = 1 << 23


def _log_once(msg: str) -> None:
    global _logged
    if not _logged:
        _logged = True
        print(msg)


def _build_triton_kernel():
    """Tritonのタイル化転置カーネルをコンパイル(importのみ、JITは初回呼び出し時)。"""
    import triton
    import triton.language as tl

    @triton.jit
    def _transpose_kernel(
        in_ptr, out_ptr, M, N,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    ):
        # 入力 (M, N) row-major → 出力 (N, M) row-major
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

        mask_m = offs_m < M
        mask_n = offs_n < N

        # 読み: 行方向(N)に連続 = コアレスト
        in_offs = offs_m[:, None] * N + offs_n[None, :]
        tile = tl.load(
            in_ptr + in_offs,
            mask=mask_m[:, None] & mask_n[None, :],
            other=0,
        )
        # 書き: タイルをレジスタ内で転置してから行方向(M)に連続に書く = コアレスト
        out_offs = offs_n[:, None] * M + offs_m[None, :]
        tl.store(
            out_ptr + out_offs,
            tl.trans(tile),
            mask=mask_n[:, None] & mask_m[None, :],
        )

    return _transpose_kernel


def _init_backend() -> str:
    """利用可能な最速経路を決定する(1度だけ)。"""
    global _backend, _triton_kernel
    if _backend is not None:
        return _backend

    forced = os.environ.get('MD_TRANSPOSE', 'auto').strip().lower()

    if forced == 'torch':
        _backend = 'torch'
        _log_once('[griffith transpose] 経路=torch (MD_TRANSPOSE=torch で強制)')
        return _backend

    try:
        import torch
        if not torch.cuda.is_available():
            _backend = 'torch'
            _log_once('[griffith transpose] 経路=torch device=cpu (CUDA未対応)')
            return _backend

        _triton_kernel = _build_triton_kernel()
        _backend = 'triton'
        _log_once(
            f'[griffith transpose] 経路=triton device={torch.cuda.get_device_name(0)}'
        )
    except Exception as e:  # pragma: no cover - triton非搭載環境
        _backend = 'torch'
        if forced == 'triton':
            _log_once(
                f'[griffith transpose] MD_TRANSPOSE=triton だがTriton利用不可 — '
                f'torch経路へフォールバック: {e}'
            )
        else:
            _log_once(
                f'[griffith transpose] 経路=torch (Triton利用不可のためフォールバック): {e}'
            )
    return _backend


def transpose_backend() -> str:
    """現在選択されている転置経路名を返す('triton' | 'torch')。"""
    return _init_backend()


def _triton_transpose(x):
    import torch
    m, n = x.shape
    out = torch.empty((n, m), dtype=x.dtype, device=x.device)
    grid = (
        (m + _BLOCK_M - 1) // _BLOCK_M,
        (n + _BLOCK_N - 1) // _BLOCK_N,
    )
    _triton_kernel[grid](x, out, m, n, BLOCK_M=_BLOCK_M, BLOCK_N=_BLOCK_N)
    return out


def fast_transpose(x):
    """`x.t().contiguous()` と厳密に等価な結果を返す(2次元テンソル)。

    `x.t()` が既にcontiguous(= xが (1, k) 相当か列優先)な場合はコピーせず
    ビューをそのまま返す。LUT層の連鎖内側ではこの経路になり、コピーは発生しない。
    """
    xt = x.t()
    if xt.is_contiguous():
        return xt

    # ---- 退行ゲート ----------------------------------------------
    # 閾値未満は **最適化前と同一の1行**(`x.t().contiguous()`)に落とす。
    # ゲート外の挙動が定義上baselineと同一になるため、Triton経路が勝たない領域で
    # 退行が起きえない。判定順もこの目的に合わせてある:
    #   - `numel()` は TensorImpl の即値読み出しで、`_init_backend()` の
    #     Python関数呼び出し+グローバル参照より安い。小テンソル(既定経路)では
    #     baselineに対する追加コストが numel 比較1回だけになる。
    #   - `_init_backend()` は閾値を超えた場合にのみ呼ぶ。
    if x.numel() < _MIN_NUMEL_FOR_TRITON:
        return xt.contiguous()

    if (
        not x.is_cuda
        or x.dim() != 2
        or not x.is_contiguous()
        or _init_backend() != 'triton'
    ):
        return xt.contiguous()

    try:
        return _triton_transpose(x)
    except Exception as e:  # pragma: no cover - 実行時失敗は1度だけ報告して縮退
        global _backend
        _backend = 'torch'
        print(f'[griffith transpose] Triton実行に失敗 — torch経路へ縮退: {e}')
        return xt.contiguous()
