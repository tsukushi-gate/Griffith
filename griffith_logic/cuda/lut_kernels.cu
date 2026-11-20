// SPDX-License-Identifier: Apache-2.0 AND MIT
//
// Portions of this file are derived from difflogic
// (https://github.com/Felix-Petersen/difflogic, revision 469702c0):
// the CUDA argument-checking macros (CHECK_CUDA, CHECK_CONTIGUOUS, CHECK_INPUT),
// the error-handling helpers (gpuErrchk, gpuAssert), the ceil_div helper,
// and the overall kernel structure. Those portions remain under the MIT License
// below; see THIRD_PARTY_NOTICES.md (section 1) and LICENSES/MIT.txt.
//
// MIT License
//
// Copyright (c) 2021-2023 Dr. Felix Petersen
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
//
// Upstream note: difflogic states that its gpuErrchk/gpuAssert idiom was adapted
// from https://stackoverflow.com/questions/14038589/ .
//
// All other code in this file, and Griffith's modifications to the derived
// portions, are licensed under the Apache License, Version 2.0 (see LICENSE).

#include <torch/extension.h>
#include <ATen/AccumulateType.h>
#include <c10/cuda/CUDAStream.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <cstring>

#define BACKWARD_W_BATCH_THREADS 32

#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(const cudaError_t code, const char *const file, const int line) {
    // exit()せずPython例外(RuntimeError)として送出する
    TORCH_CHECK(code == cudaSuccess,
                "CUDA error: ", cudaGetErrorString(code), " at ", file, ":", line);
}

template <typename T> T ceil_div(const T x, const T y) { return x / y + !!(x % y); }

/**********************************************************************************************************************/
/**  起動設定(ブロック形状)の決定                                                                                        */
/**********************************************************************************************************************/
//
// 設計方針(なぜこうしたか):
//
//  1. difflogic は dim3(32, 32) = 1024スレッド の**コンパイル時定数**を使う。
//     本カーネルは N入力LUTに一般化しており、1スレッドが 2^N 個の基底/部分和を
//     レジスタ配列で保持する。N=6 かつ double では 64要素 × 8byte がレジスタを圧迫し、
//     1024はもちろん512スレッドでも cudaErrorLaunchOutOfResources になる。
//     そのため「常に1024」は採用できない。
//
//  2. **ブロックを大きくすると大きい層では逆に遅くなる**(A10G実測。3条件を
//     experiments/bench_lut_kernel.py により比較。fwd+bwd の中央値 s/it):
//       ブロックを difflogic 同等の 1024スレッドまで広げた版は
//         - 小さい層(in=out=1024): 例 n=4 softmax b=512 が 0.000532 → 0.000386
//         - **大きい層(in=out=400k /実スケール): 悪化**
//           single softmax b=1024  0.207292 → 0.212006 (+2.3%)
//           stack   softmax b=1024 0.316820 → 0.332029 (+4.8%)
//           stack   indep   b=64   0.018757 → 0.019729 (+5.2%)
//     大きい層はメモリ帯域律速で、1024スレッドに広げるとスレッドあたりのレジスタ
//     (basis[2^N] / partial[2^N])のせいでSMあたりの同時実行ブロック数が減り占有率が
// 落ちるため。実訓練は後者のレジームなので**大スケールを優先**する。
//     → ブロックサイズは **cudaOccupancyMaxPotentialBlockSize**(占有率が最大になる
//        ブロックサイズ)を主軸にし、さらにカーネルごとの実測較正キャップを掛ける。
//        cudaFuncGetAttributes の maxThreadsPerBlock(レジスタ制約から決まる起動可能な
//        最大ブロックサイズ)は「絶対に超えてはならないハード上限」としてのみ使う。
//        これで N=6/double でも cudaErrorLaunchOutOfResources は原理的に起きない。
//
//  3. 小さい層での高速化の主因は**ブロックサイズではなくホストAPIの削減**だった
//     (旧実装は起動のたびに cudaOccupancyMaxPotentialBlockSize と
//      cudaDeviceSynchronize を呼んでいた)。小さい層は1カーネルが数マイクロ秒しかなく
//     ホスト側のレイテンシが支配的なため。
//     分離実験(ブロック形状は旧来のまま + キャッシュ + 同期削除)の実測:
//       micro n=4 softmax b=512   0.000532 → 0.000413 (1.29x)
//       micro n=6 indep   b=128   0.000608 → 0.000426 (1.43x)
//       large stack softmax b=1024 0.316820 → 0.316863 (同等)
//     = 小スケールの改善(1.06〜1.43x)の大半はホストAPI削減だけで得られ、
//       ブロック拡大による追加分は数%にすぎない。よってブロック拡大は採らない。
//     占有率クエリの結果は (カーネル種別KIND, scalar_t, N_INPUTS, デバイス) ごとに
//     キャッシュし、2回目以降の起動ではホストAPIを一切叩かない。
//     キャッシュは関数ローカル static の atomic 配列。複数スレッドが同時に初期化しても
//     書き込まれる値は同一なので benign race であり、ロックは不要。
//     デバイスごとに分けるのは、異種GPU混在時にレジスタ制約が変わりうるため。

#define MD_WARP_SIZE 32
static constexpr int MD_MAX_BLOCK_THREADS = 1024;  // CUDAのブロック上限
static constexpr int MD_MAX_BLOCK_Y = MD_MAX_BLOCK_THREADS / MD_WARP_SIZE;  // = 32
static constexpr int MD_MAX_CACHED_DEVICES = 16;

// カーネルごとのブロックy上限(実測較正値)。
// forward / backward_x = 16 (=512スレッド)、backward_w = 8 (=256スレッド)。
// difflogic の (32,32)=1024 より小さいが、A10G の実測(上記2.)で
// 大スケール(out_dim=400k)ではこの値の方が 2〜5% 速いことを確認している。
// 小スケールでの改善はブロック拡大ではなくホストAPI削減側(上記3.)で取る。
static constexpr int MD_CAP_Y_FORWARD = 16;
static constexpr int MD_CAP_Y_BACKWARD_W = 256 / BACKWARD_W_BATCH_THREADS;  // = 8
static constexpr int MD_CAP_Y_BACKWARD_X = 16;

// KIND: 同一シグネチャの関数ポインタ型を持つ別カーネル/別Nを型レベルで区別するためのタグ
// (lut_forward_kernel<float,2> と <float,4> は同じ関数ポインタ型になるため、
//  KernelT だけではキャッシュが衝突する)
// A/B計測用の退行ゲート(既定=無効)。環境変数 MD_LUT_LEGACY_LAUNCH=1 で
// 最適化前の起動設定に戻す:
//   - ブロック形状を毎起動 cudaOccupancyMaxPotentialBlockSize で問い合わせる(キャッシュ無効)
//   - clamp_block_y を適用しない
//   - 毎起動 cudaDeviceSynchronize する
// 目的は「同一コンテナ・同一GPU・同一プロセス構成で新旧を交互に測り、退行の有無を
// 実行間ノイズと分離する」こと(experiments/bench_kernel_gate.py)。
// 既定経路(未設定)には一切影響しない。ブロック形状の計算式自体は新旧で同一なので、
// このフラグで変わるのは「ホストAPIを毎起動叩くか」「同期するか」「y方向を行数で
// 頭打ちにするか」の3点だけである。
inline bool md_legacy_launch_enabled() {
    static const bool enabled = [] {
        const char* v = std::getenv("MD_LUT_LEGACY_LAUNCH");
        return v != nullptr && v[0] == '1' && v[1] == '\0';
    }();
    return enabled;
}

template <int KIND, typename scalar_t, int N_INPUTS, int CAP_Y, typename KernelT>
dim3 cached_block_2d(KernelT kernel) {
    static std::atomic<int> cache[MD_MAX_CACHED_DEVICES];  // 0 = 未計算

    int device = 0;
    gpuErrchk(cudaGetDevice(&device));
    const bool cacheable =
        (device >= 0 && device < MD_MAX_CACHED_DEVICES) && !md_legacy_launch_enabled();

    int y = cacheable ? cache[device].load(std::memory_order_relaxed) : 0;
    if (y == 0) {
        // 占有率が最大になるブロックサイズ(主軸)
        int min_grid = 0;
        int occupancy_block = 0;
        gpuErrchk(cudaOccupancyMaxPotentialBlockSize(
            &min_grid, &occupancy_block, kernel, 0, 0));
        // レジスタ制約から決まるハード上限(これを超えると起動エラー)
        cudaFuncAttributes attr{};
        gpuErrchk(cudaFuncGetAttributes(&attr, kernel));

        const int limit = std::min({occupancy_block, attr.maxThreadsPerBlock,
                                    MD_MAX_BLOCK_THREADS});
        y = std::max(1, std::min({CAP_Y, MD_MAX_BLOCK_Y, limit / MD_WARP_SIZE}));
        if (cacheable) cache[device].store(y, std::memory_order_relaxed);
    }
    return dim3(MD_WARP_SIZE, static_cast<unsigned>(y));
}

// ブロックのy方向スレッド数を「実際に処理する行数」で頭打ちにする。
// 例: out_dim=8 のとき y=32 では 24/32 のスレッドが空回りするだけなので無駄。
// 正しさには影響しない(カーネルはgrid-strideループで範囲外を弾く)純粋な効率調整。
inline dim3 clamp_block_y(const dim3 block, const int64_t rows) {
    if (md_legacy_launch_enabled()) return block;  // A/B計測用(上記フラグ参照)
    const unsigned y = static_cast<unsigned>(
        std::max<int64_t>(1, std::min<int64_t>(block.y, rows)));
    return dim3(block.x, y);
}

// 起動後のエラーチェック。
// difflogic 由来の cudaDeviceSynchronize() は PyTorch のストリーム意味論では不要で、
// 毎起動でホストをブロックしてCPU/GPUのオーバーラップを潰すため既定では行わない。
// 代償としてカーネル内で起きた実行時エラーの検出は**遅延**する(次の同期点で
// 別のAPI呼び出しの例外として現れる)。調査時は環境変数 MD_CUDA_SYNC=1 を立てると
// 起動ごとの同期が復活し、エラー地点をこのファイル/行で特定できる。
inline bool md_cuda_sync_enabled() {
    static const bool enabled = [] {
        const char* v = std::getenv("MD_CUDA_SYNC");
        return v != nullptr && v[0] == '1' && v[1] == '\0';
    }();
    return enabled;
}

#define MD_POST_LAUNCH_CHECK()                              \
    do {                                                    \
        gpuErrchk(cudaPeekAtLastError());                   \
        if (md_cuda_sync_enabled() || md_legacy_launch_enabled()) { \
            gpuErrchk(cudaDeviceSynchronize());             \
        }                                                   \
    } while (0)

/**********************************************************************************************************************/
/**  BACKWARD_W の行方向並列度(gridDim.x)の決定 */
/**********************************************************************************************************************/
//
// 問題(実測から):
//   旧実装は gridDim.x = 1 固定で、カーネル側も BACKWARD_W_BATCH_THREADS(=32) で
//   全行をストライドしていた。したがって**行方向の並列度は常に 32 スレッド**。
//   総スレッド数 ≈ 32 × out_dim になるため、out_dim が小さい層では GPU が飢餓状態になる。
//   ToeplitzTokenMix は (B,T,Cg) を (B*T*Cg, K) へ潰すので
//   **行数 3.3M・out_dim 16** という形状になり、512 スレッドで 330 万行を縮約していた。
//   これが 1 step の 76.5% を占める `lut_backward_w_kernel` の正体である。
//   一方 head(out_dim=400k)は 32 × 400k スレッドが立つので飢餓が起きない。
//
// 対策:
//   grid.x を R(row_blocks)に一般化し、grad_w_partial を (out_dim, R*32, 2^n) に広げる。
//   各スレッドは row_ = blockIdx.x*32 + threadIdx.x から R*32 ストライドで行を走る。
//   2段目のリダクションは既存の Python 側 `.sum(1)` がそのまま担う(数式は不変)。
//
// 損益分岐(閾値ゲート):
//   R>1 は「grad_w_partial の中間次元が R 倍になる」ぶん、2段目 `.sum(1)` の縮約コストが
//   増える。行数が少ないと1段目の削減より2段目の増加が勝ち、**新経路が負ける**。
//   A10G 実測(同一コンテナ内・子プロセス交互A/B、experiments/bench_backward_w.py
//   --only breakeven。out_dim × 行数の格子で legacy/new の中央値比):
//       行数    out_dim=16   64      256     1024    4096
//        1024     0.826x   0.897x  0.762x  0.826x  1.025x   ← 全面的に負け
//        2048     1.036x   1.025x  0.915x  1.043x  1.059x   ← out_dim=256 でまだ負け
//        4096     1.406x   1.396x  1.008x  1.083x  1.117x   ← 全 out_dim で勝ち(境界)
//        8192     2.150x   2.162x  1.807x  1.149x  1.156x
//       65536    16.178x   7.999x  3.229x  1.285x  1.230x
//   → **行数 4096 未満では分割しない**(R=1 = 旧実装へフォールスルー)。
//      4096 は「全 out_dim で新経路が負けない最小の行数」であり、実測から直接決めた値。
static constexpr int64_t MD_BW_MIN_ROWS_FOR_SPLIT = 4096;
//
// R の決め方(3つの上限の最小値。R=1 は旧実装と厳密に同一挙動):
//   1. 飽和条件: 総スレッド数 ≈ R × 32 × out_dim が GPU を埋めるだけあればよい。
//      R_sat = ceil(SM数 × MD_BW_THREADS_PER_SM / (32 × out_dim))
//   2. 仕事量条件: 行数より多くブロックを立てても空回りするだけ。
//      R_work = ceil(n_rows / 32)
//   3. メモリ条件: 中間バッファは out_dim × R×32 × 2^n × sizeof(scalar) バイト。
//      out_dim=400k, R=1, n=2, fp32 で既に 204MB あるので、R を無条件に増やすと大モデルが
//      OOM する。バッファ総量を MD_BW_PARTIAL_BUDGET_BYTES 以下に収める R しか採らない。
//      (この条件で R が 0 以下になる = 旧実装のバッファが既に予算超過の場合は R=1、
//       つまり**旧実装と完全に同一**にフォールスルーする。定義上、退行が起きえない構造)
//
// 環境変数 MD_BACKWARD_W:
//   "legacy" → R=1 固定(最適化前と厳密に同一)。A/B計測とデバッグ用。
//   "new"    → メモリ条件のみ適用し飽和条件の上限を外す(スイープ用)。
//   未設定   → 上記の適応則(既定)。
//
// 定数の根拠は experiments/bench_backward_w.py の A10G 実測(同一コンテナ内交互A/B)。

static constexpr int64_t MD_BW_PARTIAL_BUDGET_BYTES = 256LL << 20;  // 256MiB
// SMあたりの目標スレッド数。A10G(72SM)で 72×2048 = 147456 スレッドが目標になる。
// これは「常駐可能スレッド数を1波ぶん埋める」水準で、実測スイープの最良点に一致する。
static constexpr int64_t MD_BW_THREADS_PER_SM = 2048;
static constexpr int64_t MD_BW_MAX_ROW_BLOCKS = 65535;

enum class MdBackwardWMode { Adaptive, Legacy, Forced };

// 他のフラグ(MD_CUDA_SYNC 等)と違い**毎回 getenv する**。
// R の決定はホスト側の分岐でしかなく、テストから os.environ を切り替えて
// 同一プロセス内で新旧を照合できることの価値が、getenv 1回(~100ns、
// カーネル起動コストに対して無視できる)より大きいため。
inline MdBackwardWMode md_backward_w_mode() {
    const char* v = std::getenv("MD_BACKWARD_W");
    if (v == nullptr) return MdBackwardWMode::Adaptive;
    if (std::strcmp(v, "legacy") == 0) return MdBackwardWMode::Legacy;
    if (std::strcmp(v, "new") == 0) return MdBackwardWMode::Forced;
    return MdBackwardWMode::Adaptive;
}

inline int64_t md_sm_count() {
    static thread_local int cached_device = -1;
    static thread_local int cached_sm = 0;
    int device = 0;
    gpuErrchk(cudaGetDevice(&device));
    if (device != cached_device) {
        cudaDeviceProp prop{};
        gpuErrchk(cudaGetDeviceProperties(&prop, device));
        cached_device = device;
        cached_sm = prop.multiProcessorCount;
    }
    return static_cast<int64_t>(cached_sm);
}

inline int64_t backward_w_row_blocks(
    const int64_t out_size, const int64_t n_rows,
    const int64_t n_entries, const int64_t elem_size
) {
    const MdBackwardWMode mode = md_backward_w_mode();
    if (mode == MdBackwardWMode::Legacy) return 1;
    if (out_size <= 0 || n_rows <= 0) return 1;

    // 閾値ゲート: 損益分岐を下回る行数では分割せず旧経路にフォールスルーする
    // (MD_BACKWARD_W=new のときだけスイープ用に外す)
    if (mode != MdBackwardWMode::Forced && n_rows < MD_BW_MIN_ROWS_FOR_SPLIT) return 1;

    // 3. メモリ条件(最優先。ここを破ると OOM = 明確な退行)
    const int64_t bytes_per_row_block =
        out_size * BACKWARD_W_BATCH_THREADS * n_entries * elem_size;
    const int64_t r_mem = bytes_per_row_block > 0
        ? MD_BW_PARTIAL_BUDGET_BYTES / bytes_per_row_block
        : MD_BW_MAX_ROW_BLOCKS;

    // 2. 仕事量条件
    const int64_t r_work = ceil_div(n_rows, static_cast<int64_t>(BACKWARD_W_BATCH_THREADS));

    // 1. 飽和条件(MD_BACKWARD_W=new のときだけ外す)
    const int64_t r_sat = (mode == MdBackwardWMode::Forced)
        ? MD_BW_MAX_ROW_BLOCKS
        : ceil_div(md_sm_count() * MD_BW_THREADS_PER_SM,
                   static_cast<int64_t>(BACKWARD_W_BATCH_THREADS) * out_size);

    const int64_t r = std::min({r_mem, r_work, r_sat, MD_BW_MAX_ROW_BLOCKS});
    return std::max<int64_t>(1, r);  // R=1 は旧実装と厳密に同一(フォールスルー)
}


// キャッシュ衝突を避けるためのカーネル種別タグ
static constexpr int MD_KIND_FORWARD = 0;
static constexpr int MD_KIND_BACKWARD_W = 1;
static constexpr int MD_KIND_BACKWARD_X = 2;
static constexpr int MD_KIND_IWP_FORWARD = 3;
static constexpr int MD_KIND_IWP_BACKWARD_W = 4;
static constexpr int MD_KIND_IWP_BACKWARD_X = 5;
static constexpr int MD_KIND_WARP_FORWARD = 6;
static constexpr int MD_KIND_WARP_BACKWARD_W = 7;
static constexpr int MD_KIND_WARP_BACKWARD_X = 8;


/**********************************************************************************************************************/
/**  FORWARD カーネル                                                                                                  */
/**  difflogic の logic_layer_cuda_forward_kernel を N入力LUTに一般化。                                                   */
/**  1スレッド = 1つの (batch, neuron) ペア。butterfly基底をレジスタ上で構築し、重み付き和を計算。                                   */
/**********************************************************************************************************************/

template <typename scalar_t, int N_INPUTS>
__global__ void lut_forward_kernel(
    torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> x,       // (in_dim, batch)
    torch::PackedTensorAccessor64<int64_t, 2, torch::RestrictPtrTraits> indices,  // (n_inputs, out_dim)
    torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> w,       // (out_dim, 2^n_inputs)
    torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> y        // (out_dim, batch)
) {
    // 累積・中間計算は at::acc_type(half/bf16→float、float/doubleはそのまま)で行う。
    // float/double では acc_t == scalar_t なので従来と同一のコードになる。
    using acc_t = at::acc_type<scalar_t, true>;
    constexpr int N_ENTRIES = 1 << N_INPUTS;

    for (
        auto row = blockIdx.x * blockDim.x + threadIdx.x;
        row < y.size(1);  // batch dim
        row += blockDim.x * gridDim.x
    ) {
        for (
            auto col = blockIdx.y * blockDim.y + threadIdx.y;
            col < y.size(0);  // neuron dim
            col += blockDim.y * gridDim.y
        ) {
            // 1. N個の入力値をレジスタにgather
            acc_t xvals[N_INPUTS];
            for (int j = 0; j < N_INPUTS; j++) {
                xvals[j] = x[indices[j][col]][row];
            }

            // 2. Butterfly基底をレジスタ上で構築（Python cat([lo, hi])と同一順序）
            //    basis[k] = Π_{j} (bit(k,j) ? xvals[j] : (1 - xvals[j]))
            //    bit j of k = 1 → x_j present
            acc_t basis[N_ENTRIES];
            basis[0] = static_cast<acc_t>(1);
            int size = 1;
            for (int j = 0; j < N_INPUTS; j++) {
                // 逆順にたどって in-place で拡張
                // 前半 [0..size-1] = x_j absent, 後半 [size..2*size-1] = x_j present
                for (int k = size - 1; k >= 0; k--) {
                    basis[k + size] = basis[k] * xvals[j];
                    basis[k]        = basis[k] * (static_cast<acc_t>(1) - xvals[j]);
                }
                size *= 2;
            }

            // 3. 重み付き和: output = Σ_k basis[k] * w[col][k]
            acc_t result = static_cast<acc_t>(0);
            for (int k = 0; k < N_ENTRIES; k++) {
                result += basis[k] * w[col][k];
            }

            y[col][row] = static_cast<scalar_t>(result);
        }
    }
}


/**********************************************************************************************************************/
/**  BACKWARD_W カーネル                                                                                               */
/**  difflogic の logic_layer_cuda_backward_w_kernel を N入力LUTに一般化。                                                */
/**  バッチ次元で部分リダクション。各スレッドがbutterfly基底を再計算し grad_output * basis[k] を累積。                              */
/**********************************************************************************************************************/

template <typename scalar_t, int N_INPUTS>
__global__ void lut_backward_w_kernel(
    torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> x,        // (in_dim, batch)
    torch::PackedTensorAccessor64<int64_t, 2, torch::RestrictPtrTraits> indices,   // (n_inputs, out_dim)
    torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> grad_y,   // (out_dim, batch)
    torch::PackedTensorAccessor64<at::acc_type<scalar_t, true>, 3, torch::RestrictPtrTraits> grad_w_partial  // (out_dim, BATCH_THREADS, 2^n_inputs)
) {
    // 累積・中間計算は at::acc_type(half/bf16→float、float/doubleはそのまま)で行う。
    // float/double では acc_t == scalar_t なので従来と同一のコードになる。
    using acc_t = at::acc_type<scalar_t, true>;
    constexpr int N_ENTRIES = 1 << N_INPUTS;

    // 行(バッチ)方向のスレッド番号。grid.x を 1 より大きくできるよう一般化した。
    // row_ ∈ [0, blockDim.x * gridDim.x) が grad_w_partial の中間次元に 1:1 対応する。
    const auto row_ = blockIdx.x * blockDim.x + threadIdx.x;  // batch thread index
    const auto row_stride = blockDim.x * gridDim.x;

    for (
        auto col = blockIdx.y * blockDim.y + threadIdx.y;
        col < grad_y.size(0);  // neuron dim
        col += blockDim.y * gridDim.y
    ) {
        // 各エントリの部分和を累積
        acc_t partial[N_ENTRIES];
        for (int k = 0; k < N_ENTRIES; k++) partial[k] = static_cast<acc_t>(0);

        // N個の入力インデックスを取得（全バッチで共通）
        int64_t idx[N_INPUTS];
        for (int j = 0; j < N_INPUTS; j++) idx[j] = indices[j][col];

        for (int64_t row = row_; row < grad_y.size(1); row += row_stride) {
            // 入力値をgather
            acc_t xvals[N_INPUTS];
            for (int j = 0; j < N_INPUTS; j++) xvals[j] = x[idx[j]][row];

            // Butterfly基底を再計算（Python cat([lo, hi])と同一順序）
            acc_t basis[N_ENTRIES];
            basis[0] = static_cast<acc_t>(1);
            int size = 1;
            for (int j = 0; j < N_INPUTS; j++) {
                for (int k = size - 1; k >= 0; k--) {
                    basis[k + size] = basis[k] * xvals[j];
                    basis[k]        = basis[k] * (static_cast<acc_t>(1) - xvals[j]);
                }
                size *= 2;
            }

            const acc_t gy = grad_y[col][row];
            for (int k = 0; k < N_ENTRIES; k++) {
                partial[k] += basis[k] * gy;
            }
        }

        // 部分和をグローバルメモリに書き出し
        for (int k = 0; k < N_ENTRIES; k++) {
            grad_w_partial[col][row_][k] = partial[k];
        }
    }
}


/**********************************************************************************************************************/
/**  BACKWARD_X カーネル                                                                                               */
/**  difflogic の logic_layer_cuda_backward_x_kernel を N入力LUTに一般化。                                                */
/**  逆引きインデックス (rev_starts, rev_flat_y, rev_flat_slot) を使用。                                                    */
/**  各入力xについて、接続先の全ニューロンをたどり、解析的に dy/dx を計算。                                                        */
/**********************************************************************************************************************/

template <typename scalar_t, int N_INPUTS>
__global__ void lut_backward_x_kernel(
    torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> x,        // (in_dim, batch)
    torch::PackedTensorAccessor64<int64_t, 2, torch::RestrictPtrTraits> indices,   // (n_inputs, out_dim)
    torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> w,        // (out_dim, 2^n_inputs)
    torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> grad_y,   // (out_dim, batch)
    torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> grad_x,   // (in_dim, batch)
    torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> rev_starts,    // (in_dim + 1,)
    torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> rev_flat_y,    // (total_connections,)
    torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> rev_flat_slot  // (total_connections,)
) {
    // 累積・中間計算は at::acc_type(half/bf16→float、float/doubleはそのまま)で行う。
    // float/double では acc_t == scalar_t なので従来と同一のコードになる。
    using acc_t = at::acc_type<scalar_t, true>;
    constexpr int N_ENTRIES = 1 << N_INPUTS;

    for (
        auto row = blockIdx.x * blockDim.x + threadIdx.x;
        row < grad_x.size(1);  // batch dim
        row += blockDim.x * gridDim.x
    ) {
        for (
            auto col = blockIdx.y * blockDim.y + threadIdx.y;
            col < grad_x.size(0);  // in_dim
            col += blockDim.y * gridDim.y
        ) {
            acc_t grad_x_accum = static_cast<acc_t>(0);

            const auto start = rev_starts[col];
            const auto end = rev_starts[col + 1];

            for (int cur = start; cur < end; cur++) {
                const auto neuron_idx = rev_flat_y[cur];
                const auto slot = rev_flat_slot[cur];

                // この neuron の N個の入力値をgather
                acc_t xvals[N_INPUTS];
                for (int j = 0; j < N_INPUTS; j++) {
                    xvals[j] = x[indices[j][neuron_idx]][row];
                }

                // dy/dx_slot = Σ_k w[k] * sign(k, slot) * Π_{i≠slot} factor(k, i)
                // sign(k, slot) = bit(k, slot) ? +1 : -1
                // factor(k, i) = bit(k, i) ? xvals[i] : (1 - xvals[i])
                acc_t dy_dx = static_cast<acc_t>(0);
                for (int k = 0; k < N_ENTRIES; k++) {
                    acc_t product = static_cast<acc_t>(1);
                    for (int j = 0; j < N_INPUTS; j++) {
                        if (j == slot) continue;
                        if ((k >> j) & 1) {
                            product *= xvals[j];
                        } else {
                            product *= (static_cast<acc_t>(1) - xvals[j]);
                        }
                    }
                    // bit(k, slot) == 1 → +product, == 0 → -product
                    if ((k >> slot) & 1) {
                        dy_dx += w[neuron_idx][k] * product;
                    } else {
                        dy_dx -= w[neuron_idx][k] * product;
                    }
                }

                grad_x_accum += grad_y[neuron_idx][row] * dy_dx;
            }

            grad_x[col][row] = static_cast<scalar_t>(grad_x_accum);
        }
    }
}


/**********************************************************************************************************************/
/**  IWP (Input-Wise Parameterization / LDDLGN方式) カーネル                                                            */
/**                                                                                                                    */
/**  output = Π_{j=0}^{N-1} [p_j x_j + (1-p_j)(1-x_j)],  p_j = sigmoid(α_j)                                          */
/**                                                                                                                    */
/**  softmax/independent系カーネルとの設計上の違い:                                                                     */
/**    - sigmoid変換は他カーネルの慣習(softmax/sigmoidをPython側で事前適用してから                                        */
/**      w を渡す)に合わせ、ここでも p = sigmoid(weights) をPython側(functions.py)で                                    */
/**      計算してから渡す。カーネル自体はsigmoidを知らず p をそのまま使う                                                  */
/**    - レジスタ使用量が2^N_INPUTSではなくN_INPUTSに比例する(積の形なのでbutterfly                                       */
/**      基底2^N個を持つ必要がない)。N=6でも xvals[6]+p[6]程度で、既存softmax/                                          */
/**      independentカーネルのbasis[64](N=6, double 512B)のような                                                     */
/**      cudaErrorLaunchOutOfResources制約が構造的に発生しない                                                          */
/**********************************************************************************************************************/

template <typename scalar_t, int N_INPUTS>
__global__ void iwp_forward_kernel(
    torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> x,       // (in_dim, batch)
    torch::PackedTensorAccessor64<int64_t, 2, torch::RestrictPtrTraits> indices,  // (n_inputs, out_dim)
    torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> p,       // (out_dim, n_inputs) = sigmoid(weights)
    torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> y        // (out_dim, batch)
) {
    // 累積・中間計算は at::acc_type(half/bf16→float、float/doubleはそのまま)で行う。
    // float/double では acc_t == scalar_t なので従来と同一のコードになる。
    using acc_t = at::acc_type<scalar_t, true>;
    for (
        auto row = blockIdx.x * blockDim.x + threadIdx.x;
        row < y.size(1);  // batch dim
        row += blockDim.x * gridDim.x
    ) {
        for (
            auto col = blockIdx.y * blockDim.y + threadIdx.y;
            col < y.size(0);  // neuron dim
            col += blockDim.y * gridDim.y
        ) {
            // Π_j [p_j x_j + (1-p_j)(1-x_j)] をレジスタ上の走行積で計算(2^N配列不要)
            acc_t result = static_cast<acc_t>(1);
            for (int j = 0; j < N_INPUTS; j++) {
                const acc_t xj = x[indices[j][col]][row];
                const acc_t pj = p[col][j];
                result *= pj * xj + (static_cast<acc_t>(1) - pj) * (static_cast<acc_t>(1) - xj);
            }
            y[col][row] = static_cast<scalar_t>(result);
        }
    }
}


/**********************************************************************************************************************/
/**  IWP BACKWARD_W カーネル                                                                                           */
/**  Python版(LutFunctionIWP.backward)と同じprefix/suffix積(exclusive cumprod)で                                       */
/**  除算なしに partial_j = Π_{i≠j} factor_i を計算する。                                                              */
/**  sigmoid Jacobian(p*(1-p))はここでは掛けず、Σ_batch grad_y·partial_j·(2x_j-1) の                                    */
/**  「生の」勾配のみ返す(softmax/independentカーネルのbackward_wがsoftmax/sigmoid                                       */
/**  Jacobianの前段(grad_eff)のみ返すのと同じ流儀。p*(1-p)はPython側で乗じる)                                            */
/**********************************************************************************************************************/

template <typename scalar_t, int N_INPUTS>
__global__ void iwp_backward_w_kernel(
    torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> x,        // (in_dim, batch)
    torch::PackedTensorAccessor64<int64_t, 2, torch::RestrictPtrTraits> indices,   // (n_inputs, out_dim)
    torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> p,        // (out_dim, n_inputs)
    torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> grad_y,   // (out_dim, batch)
    torch::PackedTensorAccessor64<at::acc_type<scalar_t, true>, 3, torch::RestrictPtrTraits> grad_w_partial  // (out_dim, BATCH_THREADS, n_inputs)
) {
    // 累積・中間計算は at::acc_type(half/bf16→float、float/doubleはそのまま)で行う。
    // float/double では acc_t == scalar_t なので従来と同一のコードになる。
    using acc_t = at::acc_type<scalar_t, true>;
    // 行(バッチ)方向のスレッド番号。lut_backward_w_kernel と同じく grid.x = row_blocks で
    // 行方向を分割できる(row_ ∈ [0, blockDim.x * gridDim.x) が grad_w_partial の中間次元に対応)。
    const auto row_ = blockIdx.x * blockDim.x + threadIdx.x;  // batch thread index
    const auto row_stride = blockDim.x * gridDim.x;

    for (
        auto col = blockIdx.y * blockDim.y + threadIdx.y;
        col < grad_y.size(0);  // neuron dim
        col += blockDim.y * gridDim.y
    ) {
        int64_t idx[N_INPUTS];
        acc_t pvals[N_INPUTS];
        for (int j = 0; j < N_INPUTS; j++) {
            idx[j] = indices[j][col];
            pvals[j] = p[col][j];
        }

        acc_t partial_sum[N_INPUTS];
        for (int j = 0; j < N_INPUTS; j++) partial_sum[j] = static_cast<acc_t>(0);

        for (int64_t row = row_; row < grad_y.size(1); row += row_stride) {
            acc_t xvals[N_INPUTS];
            for (int j = 0; j < N_INPUTS; j++) xvals[j] = x[idx[j]][row];

            acc_t factor[N_INPUTS];
            for (int j = 0; j < N_INPUTS; j++) {
                factor[j] = pvals[j] * xvals[j] + (static_cast<acc_t>(1) - pvals[j]) * (static_cast<acc_t>(1) - xvals[j]);
            }

            // exclusive prefix/suffix積(除算なし)
            acc_t prefix[N_INPUTS];
            acc_t suffix[N_INPUTS];
            prefix[0] = static_cast<acc_t>(1);
            suffix[N_INPUTS - 1] = static_cast<acc_t>(1);
            for (int j = 1; j < N_INPUTS; j++) {
                prefix[j] = prefix[j - 1] * factor[j - 1];
                suffix[N_INPUTS - 1 - j] = suffix[N_INPUTS - j] * factor[N_INPUTS - j];
            }

            const acc_t gy = grad_y[col][row];
            for (int j = 0; j < N_INPUTS; j++) {
                const acc_t partial_j = prefix[j] * suffix[j];
                const acc_t d_factor_dp = static_cast<acc_t>(2) * xvals[j] - static_cast<acc_t>(1);
                partial_sum[j] += gy * partial_j * d_factor_dp;
            }
        }

        for (int j = 0; j < N_INPUTS; j++) {
            grad_w_partial[col][row_][j] = partial_sum[j];
        }
    }
}


/**********************************************************************************************************************/
/**  IWP BACKWARD_X カーネル                                                                                           */
/**  逆引きインデックス方式(softmax/independent版と同じ)。各接続についてprefix/suffix積で                                  */
/**  partial_slot = Π_{i≠slot} factor_i を除算なしに計算し、dy/dx_slot = partial_slot·(2p_slot-1)。                     */
/**********************************************************************************************************************/

template <typename scalar_t, int N_INPUTS>
__global__ void iwp_backward_x_kernel(
    torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> x,        // (in_dim, batch)
    torch::PackedTensorAccessor64<int64_t, 2, torch::RestrictPtrTraits> indices,   // (n_inputs, out_dim)
    torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> p,        // (out_dim, n_inputs)
    torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> grad_y,   // (out_dim, batch)
    torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> grad_x,   // (in_dim, batch)
    torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> rev_starts,    // (in_dim + 1,)
    torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> rev_flat_y,    // (total_connections,)
    torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> rev_flat_slot  // (total_connections,)
) {
    // 累積・中間計算は at::acc_type(half/bf16→float、float/doubleはそのまま)で行う。
    // float/double では acc_t == scalar_t なので従来と同一のコードになる。
    using acc_t = at::acc_type<scalar_t, true>;
    for (
        auto row = blockIdx.x * blockDim.x + threadIdx.x;
        row < grad_x.size(1);  // batch dim
        row += blockDim.x * gridDim.x
    ) {
        for (
            auto col = blockIdx.y * blockDim.y + threadIdx.y;
            col < grad_x.size(0);  // in_dim
            col += blockDim.y * gridDim.y
        ) {
            acc_t grad_x_accum = static_cast<acc_t>(0);

            const auto start = rev_starts[col];
            const auto end = rev_starts[col + 1];

            for (int cur = start; cur < end; cur++) {
                const auto neuron_idx = rev_flat_y[cur];
                const auto slot = rev_flat_slot[cur];

                acc_t xvals[N_INPUTS];
                acc_t pvals[N_INPUTS];
                for (int j = 0; j < N_INPUTS; j++) {
                    xvals[j] = x[indices[j][neuron_idx]][row];
                    pvals[j] = p[neuron_idx][j];
                }

                acc_t factor[N_INPUTS];
                for (int j = 0; j < N_INPUTS; j++) {
                    factor[j] = pvals[j] * xvals[j] + (static_cast<acc_t>(1) - pvals[j]) * (static_cast<acc_t>(1) - xvals[j]);
                }

                // exclusive prefix/suffix積(除算なし)。slot番目だけ必要だが
                // 配列は再利用のため全体を構築する(N<=6で軽量)
                acc_t prefix[N_INPUTS];
                acc_t suffix[N_INPUTS];
                prefix[0] = static_cast<acc_t>(1);
                suffix[N_INPUTS - 1] = static_cast<acc_t>(1);
                for (int j = 1; j < N_INPUTS; j++) {
                    prefix[j] = prefix[j - 1] * factor[j - 1];
                    suffix[N_INPUTS - 1 - j] = suffix[N_INPUTS - j] * factor[N_INPUTS - j];
                }
                const acc_t partial_slot = prefix[slot] * suffix[slot];

                const acc_t dy_dx = partial_slot * (static_cast<acc_t>(2) * pvals[slot] - static_cast<acc_t>(1));

                grad_x_accum += grad_y[neuron_idx][row] * dy_dx;
            }

            grad_x[col][row] = static_cast<scalar_t>(grad_x_accum);
        }
    }
}


/**********************************************************************************************************************/
/**  WARP (Walsh-Hadamard係数パラメータ化) カーネル                                                                     */
/**                                                                                                                    */
/**  z = Σ_e θ_e φ_e^W(1-2x) / τ  (+ noise),  y = σ(z)                                                               */
/**  φ^W はWH基底(butterfly): φ[0]=1、以降 φ = cat([φ, φ·s_j]) （s_j = 1-2x_j）。                                       */
/**  softmax/independentカーネルのbutterfly基底(φ=cat([φ·(1-x_j), φ·x_j]))と違い、                                       */
/**  「不在」側が(1-x)ではなく定数1なので基底構築・逆伝播の式がやや簡素になる。                                                 */
/**                                                                                                                    */
/**  対象は selection='soft' の学習時経路のみ(Python側 layers/lut.py で保証)。                                          */
/**  hard_st/gumbel_st(LutFunctionWARPHardST、エントリロジット閾値化)は数理が                                           */
/**  異なるため本カーネルの対象外。                                                                                     */
/**                                                                                                                    */
/**  レジスタ使用量はsoftmax系と同じくbasis[2^N_INPUTS]がボトルネック(N=6で64要素)。                                     */
/**  既存softmax/independentカーネルと同じ cudaErrorLaunchOutOfResources 制約を受けるため                              */
/**  cached_block_2d による自動ブロック形状決定を同様に適用する。                                                          */
/**                                                                                                                    */
/**  Gumbelノイズはカーネル外(Python側、torch.randベース、LutFunctionWARPと同一のRNG消費順)                              */
/**  で生成し、noiseテンソル(またはNone)としてこのカーネルに渡す。カーネルは                                              */
/**  「与えられたら z に加算するだけ」の役割に限定し、乱数生成そのものには一切関与しない。                                    */
/**  has_noise=false時、noiseアクセサは実際には参照されない(呼び出し側からダミーテンソルが渡される)。                       */
/**                                                                                                                    */
/**  τスケーリングの扱い: backward_w/backward_xはWH-basis逆伝播が θ・grad_logit に対して                                */
/**  線形であることを利用し、カーネル自体はτを知らない「生の」勾配(Σ basis·grad_logit、                                   */
/**  および θ をそのまま使った ∂z/∂x)を返す。Python側(functions.py)で最後に /τ するだけで                              */
/**  数学的に同値(softmax/independentカーネルがsoftmax/sigmoid Jacobianを事後適用するのと同じ流儀)。                     */
/**********************************************************************************************************************/

template <typename scalar_t, int N_INPUTS>
__global__ void warp_forward_kernel(
    torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> x,       // (in_dim, batch)
    torch::PackedTensorAccessor64<int64_t, 2, torch::RestrictPtrTraits> indices,  // (n_inputs, out_dim)
    torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> weights, // (out_dim, 2^n_inputs) = θ（度数キャップ済み）
    torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> y,       // (out_dim, batch)
    at::acc_type<scalar_t, true> inv_tau,
    bool has_noise,
    torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> noise    // (out_dim, batch)。has_noise=falseなら未参照
) {
    // 累積・中間計算は at::acc_type(half/bf16→float、float/doubleはそのまま)で行う。
    // float/double では acc_t == scalar_t なので従来と同一のコードになる。
    using acc_t = at::acc_type<scalar_t, true>;
    constexpr int N_ENTRIES = 1 << N_INPUTS;

    for (
        auto row = blockIdx.x * blockDim.x + threadIdx.x;
        row < y.size(1);  // batch dim
        row += blockDim.x * gridDim.x
    ) {
        for (
            auto col = blockIdx.y * blockDim.y + threadIdx.y;
            col < y.size(0);  // neuron dim
            col += blockDim.y * gridDim.y
        ) {
            // 1. N個の入力値を符号化 s_j = 1 - 2*x_j
            acc_t s[N_INPUTS];
            for (int j = 0; j < N_INPUTS; j++) {
                const acc_t xj = x[indices[j][col]][row];
                s[j] = static_cast<acc_t>(1) - static_cast<acc_t>(2) * xj;
            }

            // 2. WH基底をレジスタ上で構築（Python cat([basis, basis*sj])と同一順序）
            //    basis[k] = Π_{j: bit(k,j)=1} s_j（不在側は1のまま）
            acc_t basis[N_ENTRIES];
            basis[0] = static_cast<acc_t>(1);
            int size = 1;
            for (int j = 0; j < N_INPUTS; j++) {
                for (int k = size - 1; k >= 0; k--) {
                    basis[k + size] = basis[k] * s[j];
                    // basis[k] はそのまま（不在 → 係数1）
                }
                size *= 2;
            }

            // 3. z = (θ・basis) / τ (+ noise)
            acc_t z = static_cast<acc_t>(0);
            for (int k = 0; k < N_ENTRIES; k++) {
                z += basis[k] * weights[col][k];
            }
            z *= inv_tau;
            if (has_noise) {
                z += noise[col][row];
            }

            // 4. y = σ(z)。half/floatでも指数関数はdoubleで計算し精度・コンパイル可搬性を確保
            const double zd = static_cast<double>(z);
            const double sig = 1.0 / (1.0 + ::exp(-zd));
            y[col][row] = static_cast<scalar_t>(sig);
        }
    }
}


/**********************************************************************************************************************/
/**  WARP BACKWARD_W カーネル                                                                                          */
/**  lut_backward_w_kernel と同型のバッチ部分縮約。基底構築のみWH方式（不在側=1）。                                        */
/**  返り値は Σ_batch grad_logit · basis[k] の「生の」勾配（/τ未適用、Python側で除算）。                                  */
/**********************************************************************************************************************/

template <typename scalar_t, int N_INPUTS>
__global__ void warp_backward_w_kernel(
    torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> x,           // (in_dim, batch)
    torch::PackedTensorAccessor64<int64_t, 2, torch::RestrictPtrTraits> indices,      // (n_inputs, out_dim)
    torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> grad_logit,  // (out_dim, batch) = dL/dz（sigmoid微分適用済み、/τ未適用）
    torch::PackedTensorAccessor64<at::acc_type<scalar_t, true>, 3, torch::RestrictPtrTraits> grad_w_partial  // (out_dim, BATCH_THREADS, 2^n_inputs)
) {
    // 累積・中間計算は at::acc_type(half/bf16→float、float/doubleはそのまま)で行う。
    // float/double では acc_t == scalar_t なので従来と同一のコードになる。
    using acc_t = at::acc_type<scalar_t, true>;
    constexpr int N_ENTRIES = 1 << N_INPUTS;

    // 行(バッチ)方向のスレッド番号。lut_backward_w_kernel と同じく grid.x = row_blocks で
    // 行方向を分割できる(row_ ∈ [0, blockDim.x * gridDim.x) が grad_w_partial の中間次元に対応)。
    const auto row_ = blockIdx.x * blockDim.x + threadIdx.x;  // batch thread index
    const auto row_stride = blockDim.x * gridDim.x;

    for (
        auto col = blockIdx.y * blockDim.y + threadIdx.y;
        col < grad_logit.size(0);  // neuron dim
        col += blockDim.y * gridDim.y
    ) {
        acc_t partial[N_ENTRIES];
        for (int k = 0; k < N_ENTRIES; k++) partial[k] = static_cast<acc_t>(0);

        int64_t idx[N_INPUTS];
        for (int j = 0; j < N_INPUTS; j++) idx[j] = indices[j][col];

        for (int64_t row = row_; row < grad_logit.size(1); row += row_stride) {
            acc_t s[N_INPUTS];
            for (int j = 0; j < N_INPUTS; j++) {
                const acc_t xj = x[idx[j]][row];
                s[j] = static_cast<acc_t>(1) - static_cast<acc_t>(2) * xj;
            }

            acc_t basis[N_ENTRIES];
            basis[0] = static_cast<acc_t>(1);
            int size = 1;
            for (int j = 0; j < N_INPUTS; j++) {
                for (int k = size - 1; k >= 0; k--) {
                    basis[k + size] = basis[k] * s[j];
                }
                size *= 2;
            }

            const acc_t gl = grad_logit[col][row];
            for (int k = 0; k < N_ENTRIES; k++) {
                partial[k] += basis[k] * gl;
            }
        }

        for (int k = 0; k < N_ENTRIES; k++) {
            grad_w_partial[col][row_][k] = partial[k];
        }
    }
}


/**********************************************************************************************************************/
/**  WARP BACKWARD_X カーネル                                                                                          */
/**  逆引きインデックス方式(softmax/IWP版と同じ)。z = Σ_k θ_k Π_{j: bit(k,j)=1} s_j の                                  */
/**  s_slot に関する偏微分を直接計算する:                                                                               */
/**      ∂z/∂s_slot = Σ_{k: bit(k,slot)=1} θ_k · Π_{j≠slot, bit(k,j)=1} s_j                                          */
/**  softmax版と異なり「不在」側は係数1固定（-の項が出ない）ため bit(k,slot)=1 の k のみ和を取る。                          */
/**  s_slot = 1 - 2 x_slot より ∂s_slot/∂x_slot = -2。                                                                */
/**  返り値は τ 未適用の「生の」∂z/∂x（Python側で /τ）。                                                                */
/**********************************************************************************************************************/

template <typename scalar_t, int N_INPUTS>
__global__ void warp_backward_x_kernel(
    torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> x,        // (in_dim, batch)
    torch::PackedTensorAccessor64<int64_t, 2, torch::RestrictPtrTraits> indices,   // (n_inputs, out_dim)
    torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> weights,  // (out_dim, 2^n_inputs) = θ（度数キャップ済み、/τ未適用）
    torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> grad_logit,  // (out_dim, batch)
    torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> grad_x,   // (in_dim, batch)
    torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> rev_starts,    // (in_dim + 1,)
    torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> rev_flat_y,    // (total_connections,)
    torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> rev_flat_slot  // (total_connections,)
) {
    // 累積・中間計算は at::acc_type(half/bf16→float、float/doubleはそのまま)で行う。
    // float/double では acc_t == scalar_t なので従来と同一のコードになる。
    using acc_t = at::acc_type<scalar_t, true>;
    constexpr int N_ENTRIES = 1 << N_INPUTS;

    for (
        auto row = blockIdx.x * blockDim.x + threadIdx.x;
        row < grad_x.size(1);  // batch dim
        row += blockDim.x * gridDim.x
    ) {
        for (
            auto col = blockIdx.y * blockDim.y + threadIdx.y;
            col < grad_x.size(0);  // in_dim
            col += blockDim.y * gridDim.y
        ) {
            acc_t grad_x_accum = static_cast<acc_t>(0);

            const auto start = rev_starts[col];
            const auto end = rev_starts[col + 1];

            for (int cur = start; cur < end; cur++) {
                const auto neuron_idx = rev_flat_y[cur];
                const auto slot = rev_flat_slot[cur];

                acc_t s[N_INPUTS];
                for (int j = 0; j < N_INPUTS; j++) {
                    const acc_t xj = x[indices[j][neuron_idx]][row];
                    s[j] = static_cast<acc_t>(1) - static_cast<acc_t>(2) * xj;
                }

                acc_t dz_ds = static_cast<acc_t>(0);
                for (int k = 0; k < N_ENTRIES; k++) {
                    if (!((k >> slot) & 1)) continue;  // bit(k,slot)=1 のみ寄与
                    acc_t product = static_cast<acc_t>(1);
                    for (int j = 0; j < N_INPUTS; j++) {
                        if (j == slot) continue;
                        if ((k >> j) & 1) {
                            product *= s[j];
                        }
                        // 不在(bit=0)は係数1のまま(何も乗じない)
                    }
                    dz_ds += weights[neuron_idx][k] * product;
                }

                const acc_t dy_dx = dz_ds * static_cast<acc_t>(-2);
                grad_x_accum += grad_logit[neuron_idx][row] * dy_dx;
            }

            grad_x[col][row] = static_cast<scalar_t>(grad_x_accum);
        }
    }
}


/**********************************************************************************************************************/
/**  ホスト関数                                                                                                         */
/**********************************************************************************************************************/

// N_INPUTS に応じてカーネルをディスパッチするマクロ
// 可変長引数を使用: テンプレート引数のカンマがマクロ引数区切りと誤解釈されるのを回避
#define DISPATCH_N_INPUTS(N_INPUTS, ...)   \
    switch (N_INPUTS) {                     \
        case 2: { constexpr int N = 2; __VA_ARGS__; break; }  \
        case 4: { constexpr int N = 4; __VA_ARGS__; break; }  \
        case 6: { constexpr int N = 6; __VA_ARGS__; break; }  \
        default: TORCH_CHECK(false, "n_inputs は 2, 4, 6 のいずれか: ", N_INPUTS); \
    }


torch::Tensor lut_forward(
    torch::Tensor x,        // (in_dim, batch)
    torch::Tensor indices,   // (n_inputs, out_dim)
    torch::Tensor w,         // (out_dim, 2^n_inputs)
    int n_inputs
) {
    CHECK_INPUT(x);
    CHECK_INPUT(indices);
    CHECK_INPUT(w);

    const auto batch_size = x.size(1);
    const auto out_size = w.size(0);

    auto y = torch::empty({out_size, batch_size}, torch::dtype(x.dtype()).device(x.device()));

    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, x.scalar_type(), "lut_forward", ([&] {
        DISPATCH_N_INPUTS(n_inputs,
            auto kernel = lut_forward_kernel<scalar_t, N>;
            const dim3 threads_per_block = clamp_block_y(
                cached_block_2d<MD_KIND_FORWARD, scalar_t, N, MD_CAP_Y_FORWARD>(kernel), out_size);
            const dim3 blocks_per_grid(
                min(static_cast<int64_t>(65535), ceil_div(batch_size, static_cast<int64_t>(threads_per_block.x))),
                min(static_cast<int64_t>(65535), ceil_div(out_size, static_cast<int64_t>(threads_per_block.y)))
            );
            kernel<<<blocks_per_grid, threads_per_block, 0,
                    c10::cuda::getCurrentCUDAStream()>>>(
                x.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
                indices.packed_accessor64<int64_t, 2, torch::RestrictPtrTraits>(),
                w.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
                y.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>()
            )
        );
    }));

    MD_POST_LAUNCH_CHECK();

    return y;
}


torch::Tensor lut_backward_w(
    torch::Tensor x,         // (in_dim, batch)
    torch::Tensor indices,    // (n_inputs, out_dim)
    torch::Tensor grad_y,     // (out_dim, batch)
    int n_inputs
) {
    CHECK_INPUT(x);
    CHECK_INPUT(indices);
    CHECK_INPUT(grad_y);

    const auto out_size = grad_y.size(0);
    const auto n_rows = grad_y.size(1);
    const int n_entries = 1 << n_inputs;

    // 部分和は累積型(half/bf16→float)で持つ。float/double では x と同じdtype(従来どおり)
    const auto acc_dtype = at::toAccumulateType(x.scalar_type(), /*is_cuda=*/true);
    const int64_t row_blocks = backward_w_row_blocks(
        out_size, n_rows, n_entries, static_cast<int64_t>(c10::elementSize(acc_dtype)));

    auto grad_w_partial = torch::empty(
        {out_size, row_blocks * BACKWARD_W_BATCH_THREADS, n_entries},
        torch::dtype(acc_dtype).device(x.device())
    );

    // difflogic と同じブロック構成: バッチスレッド × ニューロン
    // (blockDim.x はカーネルのストライド前提により BACKWARD_W_BATCH_THREADS 固定。
    //  行方向の並列度は gridDim.x = row_blocks で拡張する)
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, x.scalar_type(), "lut_backward_w", ([&] {
        DISPATCH_N_INPUTS(n_inputs,
            auto kernel = lut_backward_w_kernel<scalar_t, N>;
            const dim3 threads_per_block = clamp_block_y(
                cached_block_2d<MD_KIND_BACKWARD_W, scalar_t, N, MD_CAP_Y_BACKWARD_W>(kernel), out_size);
            const dim3 blocks_per_grid(
                static_cast<unsigned>(row_blocks),
                min(static_cast<int64_t>(65535), ceil_div(out_size, static_cast<int64_t>(threads_per_block.y)))
            );
            kernel<<<blocks_per_grid, threads_per_block, 0,
                    c10::cuda::getCurrentCUDAStream()>>>(
                x.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
                indices.packed_accessor64<int64_t, 2, torch::RestrictPtrTraits>(),
                grad_y.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
                grad_w_partial.packed_accessor64<at::acc_type<scalar_t, true>, 3, torch::RestrictPtrTraits>()
            )
        );
    }));

    MD_POST_LAUNCH_CHECK();

    // 部分和をバッチスレッド次元でリダクション → (out_dim, 2^n_inputs)。
    // 累積型から入力dtypeへ戻す(float/double では同一dtypeなのでno-op)
    return grad_w_partial.sum(1).to(x.scalar_type());
}


torch::Tensor lut_backward_x(
    torch::Tensor x,                      // (in_dim, batch)
    torch::Tensor indices,                 // (n_inputs, out_dim)
    torch::Tensor w,                       // (out_dim, 2^n_inputs)
    torch::Tensor grad_y,                  // (out_dim, batch)
    torch::Tensor rev_starts,              // (in_dim + 1,)
    torch::Tensor rev_flat_y,              // (total_connections,)
    torch::Tensor rev_flat_slot,           // (total_connections,)
    int n_inputs
) {
    CHECK_INPUT(x);
    CHECK_INPUT(indices);
    CHECK_INPUT(w);
    CHECK_INPUT(grad_y);
    CHECK_INPUT(rev_starts);
    CHECK_INPUT(rev_flat_y);
    CHECK_INPUT(rev_flat_slot);

    auto grad_x = torch::empty_like(x);

    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, x.scalar_type(), "lut_backward_x", ([&] {
        DISPATCH_N_INPUTS(n_inputs,
            auto kernel = lut_backward_x_kernel<scalar_t, N>;
            const dim3 threads_per_block = clamp_block_y(
                cached_block_2d<MD_KIND_BACKWARD_X, scalar_t, N, MD_CAP_Y_BACKWARD_X>(kernel), x.size(0));
            const dim3 blocks_per_grid(
                min(static_cast<int64_t>(65535), ceil_div(x.size(1), static_cast<int64_t>(threads_per_block.x))),
                min(static_cast<int64_t>(65535), ceil_div(x.size(0), static_cast<int64_t>(threads_per_block.y)))
            );
            kernel<<<blocks_per_grid, threads_per_block, 0,
                    c10::cuda::getCurrentCUDAStream()>>>(
                x.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
                indices.packed_accessor64<int64_t, 2, torch::RestrictPtrTraits>(),
                w.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
                grad_y.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
                grad_x.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
                rev_starts.packed_accessor64<int64_t, 1, torch::RestrictPtrTraits>(),
                rev_flat_y.packed_accessor64<int64_t, 1, torch::RestrictPtrTraits>(),
                rev_flat_slot.packed_accessor64<int64_t, 1, torch::RestrictPtrTraits>()
            )
        );
    }));

    MD_POST_LAUNCH_CHECK();

    return grad_x;
}


/**********************************************************************************************************************/
/**  IWP ホスト関数                                                                                                     */
/**  n_inputs は既存カーネルと同じ 2/4/6 のみサポート(DISPATCH_N_INPUTSマクロ共用)。                                       */
/**********************************************************************************************************************/

torch::Tensor iwp_forward(
    torch::Tensor x,        // (in_dim, batch)
    torch::Tensor indices,   // (n_inputs, out_dim)
    torch::Tensor p,         // (out_dim, n_inputs) = sigmoid(weights)
    int n_inputs
) {
    CHECK_INPUT(x);
    CHECK_INPUT(indices);
    CHECK_INPUT(p);

    const auto batch_size = x.size(1);
    const auto out_size = p.size(0);

    auto y = torch::empty({out_size, batch_size}, torch::dtype(x.dtype()).device(x.device()));

    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, x.scalar_type(), "iwp_forward", ([&] {
        DISPATCH_N_INPUTS(n_inputs,
            auto kernel = iwp_forward_kernel<scalar_t, N>;
            const dim3 threads_per_block = clamp_block_y(
                cached_block_2d<MD_KIND_IWP_FORWARD, scalar_t, N, MD_CAP_Y_FORWARD>(kernel), out_size);
            const dim3 blocks_per_grid(
                min(static_cast<int64_t>(65535), ceil_div(batch_size, static_cast<int64_t>(threads_per_block.x))),
                min(static_cast<int64_t>(65535), ceil_div(out_size, static_cast<int64_t>(threads_per_block.y)))
            );
            kernel<<<blocks_per_grid, threads_per_block, 0,
                    c10::cuda::getCurrentCUDAStream()>>>(
                x.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
                indices.packed_accessor64<int64_t, 2, torch::RestrictPtrTraits>(),
                p.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
                y.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>()
            )
        );
    }));

    MD_POST_LAUNCH_CHECK();

    return y;
}


torch::Tensor iwp_backward_w(
    torch::Tensor x,         // (in_dim, batch)
    torch::Tensor indices,    // (n_inputs, out_dim)
    torch::Tensor p,          // (out_dim, n_inputs)
    torch::Tensor grad_y,     // (out_dim, batch)
    int n_inputs
) {
    CHECK_INPUT(x);
    CHECK_INPUT(indices);
    CHECK_INPUT(p);
    CHECK_INPUT(grad_y);

    const auto out_size = grad_y.size(0);
    const auto n_rows = grad_y.size(1);

    // 部分和は累積型(half/bf16→float)で持つ。行方向の分割数は lut_backward_w と同じ規則で決める
    const auto acc_dtype = at::toAccumulateType(x.scalar_type(), /*is_cuda=*/true);
    const int64_t row_blocks = backward_w_row_blocks(
        out_size, n_rows, n_inputs, static_cast<int64_t>(c10::elementSize(acc_dtype)));

    auto grad_w_partial = torch::empty(
        {out_size, row_blocks * BACKWARD_W_BATCH_THREADS, n_inputs},
        torch::dtype(acc_dtype).device(x.device())
    );

    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, x.scalar_type(), "iwp_backward_w", ([&] {
        DISPATCH_N_INPUTS(n_inputs,
            auto kernel = iwp_backward_w_kernel<scalar_t, N>;
            const dim3 threads_per_block = clamp_block_y(
                cached_block_2d<MD_KIND_IWP_BACKWARD_W, scalar_t, N, MD_CAP_Y_BACKWARD_W>(kernel), out_size);
            const dim3 blocks_per_grid(
                static_cast<unsigned>(row_blocks),
                min(static_cast<int64_t>(65535), ceil_div(out_size, static_cast<int64_t>(threads_per_block.y)))
            );
            kernel<<<blocks_per_grid, threads_per_block, 0,
                    c10::cuda::getCurrentCUDAStream()>>>(
                x.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
                indices.packed_accessor64<int64_t, 2, torch::RestrictPtrTraits>(),
                p.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
                grad_y.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
                grad_w_partial.packed_accessor64<at::acc_type<scalar_t, true>, 3, torch::RestrictPtrTraits>()
            )
        );
    }));

    MD_POST_LAUNCH_CHECK();

    // 部分和をバッチスレッド次元でリダクション → (out_dim, n_inputs)
    // sigmoid Jacobian(p*(1-p))はPython側(functions.py)で乗じる
    return grad_w_partial.sum(1).to(x.scalar_type());
}


torch::Tensor iwp_backward_x(
    torch::Tensor x,                      // (in_dim, batch)
    torch::Tensor indices,                 // (n_inputs, out_dim)
    torch::Tensor p,                       // (out_dim, n_inputs)
    torch::Tensor grad_y,                  // (out_dim, batch)
    torch::Tensor rev_starts,              // (in_dim + 1,)
    torch::Tensor rev_flat_y,              // (total_connections,)
    torch::Tensor rev_flat_slot,           // (total_connections,)
    int n_inputs
) {
    CHECK_INPUT(x);
    CHECK_INPUT(indices);
    CHECK_INPUT(p);
    CHECK_INPUT(grad_y);
    CHECK_INPUT(rev_starts);
    CHECK_INPUT(rev_flat_y);
    CHECK_INPUT(rev_flat_slot);

    auto grad_x = torch::empty_like(x);

    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, x.scalar_type(), "iwp_backward_x", ([&] {
        DISPATCH_N_INPUTS(n_inputs,
            auto kernel = iwp_backward_x_kernel<scalar_t, N>;
            const dim3 threads_per_block = clamp_block_y(
                cached_block_2d<MD_KIND_IWP_BACKWARD_X, scalar_t, N, MD_CAP_Y_BACKWARD_X>(kernel), x.size(0));
            const dim3 blocks_per_grid(
                min(static_cast<int64_t>(65535), ceil_div(x.size(1), static_cast<int64_t>(threads_per_block.x))),
                min(static_cast<int64_t>(65535), ceil_div(x.size(0), static_cast<int64_t>(threads_per_block.y)))
            );
            kernel<<<blocks_per_grid, threads_per_block, 0,
                    c10::cuda::getCurrentCUDAStream()>>>(
                x.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
                indices.packed_accessor64<int64_t, 2, torch::RestrictPtrTraits>(),
                p.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
                grad_y.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
                grad_x.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
                rev_starts.packed_accessor64<int64_t, 1, torch::RestrictPtrTraits>(),
                rev_flat_y.packed_accessor64<int64_t, 1, torch::RestrictPtrTraits>(),
                rev_flat_slot.packed_accessor64<int64_t, 1, torch::RestrictPtrTraits>()
            )
        );
    }));

    MD_POST_LAUNCH_CHECK();

    return grad_x;
}


/**********************************************************************************************************************/
/**  WARP ホスト関数                                                                                                    */
/**  n_inputs は既存カーネルと同じ 2/4/6 のみサポート(DISPATCH_N_INPUTSマクロ共用)。                                       */
/**  weights(θ)はPython側で度数キャップ(wh_degree_cap)マスク済みのものを渡す設計とし、                                    */
/**  カーネル自体はマスクを一切知らない(layers/lut.py の _warp_weights() が担当)。                                        */
/**********************************************************************************************************************/

torch::Tensor warp_forward(
    torch::Tensor x,                       // (in_dim, batch)
    torch::Tensor indices,                  // (n_inputs, out_dim)
    torch::Tensor weights,                  // (out_dim, 2^n_inputs) = θ（度数キャップ済み）
    double inv_tau,
    c10::optional<torch::Tensor> noise,     // (out_dim, batch) or None（Python側で転置済みのものを渡す）
    int n_inputs
) {
    CHECK_INPUT(x);
    CHECK_INPUT(indices);
    CHECK_INPUT(weights);

    const auto batch_size = x.size(1);
    const auto out_size = weights.size(0);

    auto y = torch::empty({out_size, batch_size}, torch::dtype(x.dtype()).device(x.device()));

    const bool has_noise = noise.has_value();
    torch::Tensor noise_t;
    if (has_noise) {
        noise_t = noise.value();
        CHECK_INPUT(noise_t);
    } else {
        // has_noise=falseの場合カーネル内で参照されないダミー(xを流用、追加確保なし)
        noise_t = x;
    }

    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, x.scalar_type(), "warp_forward", ([&] {
        // 1/τ は bf16/half に丸めず累積型(float)のまま渡す
        using acc_t = at::acc_type<scalar_t, true>;
        const acc_t inv_tau_s = static_cast<acc_t>(inv_tau);
        DISPATCH_N_INPUTS(n_inputs,
            auto kernel = warp_forward_kernel<scalar_t, N>;
            const dim3 threads_per_block = clamp_block_y(
                cached_block_2d<MD_KIND_WARP_FORWARD, scalar_t, N, MD_CAP_Y_FORWARD>(kernel), out_size);
            const dim3 blocks_per_grid(
                min(static_cast<int64_t>(65535), ceil_div(batch_size, static_cast<int64_t>(threads_per_block.x))),
                min(static_cast<int64_t>(65535), ceil_div(out_size, static_cast<int64_t>(threads_per_block.y)))
            );
            kernel<<<blocks_per_grid, threads_per_block, 0,
                    c10::cuda::getCurrentCUDAStream()>>>(
                x.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
                indices.packed_accessor64<int64_t, 2, torch::RestrictPtrTraits>(),
                weights.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
                y.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
                inv_tau_s,
                has_noise,
                noise_t.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>()
            )
        );
    }));

    MD_POST_LAUNCH_CHECK();

    return y;
}


torch::Tensor warp_backward_w(
    torch::Tensor x,           // (in_dim, batch)
    torch::Tensor indices,      // (n_inputs, out_dim)
    torch::Tensor grad_logit,   // (out_dim, batch) = dL/dz（/τ未適用）
    int n_inputs
) {
    CHECK_INPUT(x);
    CHECK_INPUT(indices);
    CHECK_INPUT(grad_logit);

    const auto out_size = grad_logit.size(0);
    const auto n_rows = grad_logit.size(1);
    const int n_entries = 1 << n_inputs;

    // 部分和は累積型(half/bf16→float)で持つ。行方向の分割数は lut_backward_w と同じ規則で決める
    const auto acc_dtype = at::toAccumulateType(x.scalar_type(), /*is_cuda=*/true);
    const int64_t row_blocks = backward_w_row_blocks(
        out_size, n_rows, n_entries, static_cast<int64_t>(c10::elementSize(acc_dtype)));

    auto grad_w_partial = torch::empty(
        {out_size, row_blocks * BACKWARD_W_BATCH_THREADS, n_entries},
        torch::dtype(acc_dtype).device(x.device())
    );

    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, x.scalar_type(), "warp_backward_w", ([&] {
        DISPATCH_N_INPUTS(n_inputs,
            auto kernel = warp_backward_w_kernel<scalar_t, N>;
            const dim3 threads_per_block = clamp_block_y(
                cached_block_2d<MD_KIND_WARP_BACKWARD_W, scalar_t, N, MD_CAP_Y_BACKWARD_W>(kernel), out_size);
            const dim3 blocks_per_grid(
                static_cast<unsigned>(row_blocks),
                min(static_cast<int64_t>(65535), ceil_div(out_size, static_cast<int64_t>(threads_per_block.y)))
            );
            kernel<<<blocks_per_grid, threads_per_block, 0,
                    c10::cuda::getCurrentCUDAStream()>>>(
                x.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
                indices.packed_accessor64<int64_t, 2, torch::RestrictPtrTraits>(),
                grad_logit.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
                grad_w_partial.packed_accessor64<at::acc_type<scalar_t, true>, 3, torch::RestrictPtrTraits>()
            )
        );
    }));

    MD_POST_LAUNCH_CHECK();

    // 部分和をバッチスレッド次元でリダクション → (out_dim, 2^n_inputs)。/τはPython側で適用
    return grad_w_partial.sum(1).to(x.scalar_type());
}


torch::Tensor warp_backward_x(
    torch::Tensor x,                      // (in_dim, batch)
    torch::Tensor indices,                 // (n_inputs, out_dim)
    torch::Tensor weights,                 // (out_dim, 2^n_inputs) = θ（度数キャップ済み、/τ未適用）
    torch::Tensor grad_logit,              // (out_dim, batch)
    torch::Tensor rev_starts,              // (in_dim + 1,)
    torch::Tensor rev_flat_y,              // (total_connections,)
    torch::Tensor rev_flat_slot,           // (total_connections,)
    int n_inputs
) {
    CHECK_INPUT(x);
    CHECK_INPUT(indices);
    CHECK_INPUT(weights);
    CHECK_INPUT(grad_logit);
    CHECK_INPUT(rev_starts);
    CHECK_INPUT(rev_flat_y);
    CHECK_INPUT(rev_flat_slot);

    auto grad_x = torch::empty_like(x);

    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, x.scalar_type(), "warp_backward_x", ([&] {
        DISPATCH_N_INPUTS(n_inputs,
            auto kernel = warp_backward_x_kernel<scalar_t, N>;
            const dim3 threads_per_block = clamp_block_y(
                cached_block_2d<MD_KIND_WARP_BACKWARD_X, scalar_t, N, MD_CAP_Y_BACKWARD_X>(kernel), x.size(0));
            const dim3 blocks_per_grid(
                min(static_cast<int64_t>(65535), ceil_div(x.size(1), static_cast<int64_t>(threads_per_block.x))),
                min(static_cast<int64_t>(65535), ceil_div(x.size(0), static_cast<int64_t>(threads_per_block.y)))
            );
            kernel<<<blocks_per_grid, threads_per_block, 0,
                    c10::cuda::getCurrentCUDAStream()>>>(
                x.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
                indices.packed_accessor64<int64_t, 2, torch::RestrictPtrTraits>(),
                weights.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
                grad_logit.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
                grad_x.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
                rev_starts.packed_accessor64<int64_t, 1, torch::RestrictPtrTraits>(),
                rev_flat_y.packed_accessor64<int64_t, 1, torch::RestrictPtrTraits>(),
                rev_flat_slot.packed_accessor64<int64_t, 1, torch::RestrictPtrTraits>()
            )
        );
    }));

    MD_POST_LAUNCH_CHECK();

    return grad_x;
}
