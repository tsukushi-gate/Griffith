/* griffith-logic — gate2のA/Bマイクロベンチ。
 *
 * runtime/gpu/mdl_gpu_kernels.h の実物 mdl_g_gate2(switch(tt&0xf)の16分岐)と、
 * 無分岐マスク式 mdl_g_gate2_mask を、mdl_k_conv_exec と同型の
 * 「命令列(tt,a_off,b_off,out_off)を読んでスクラッチへ read-modify-write」
 * ループで比較する。runtime/ 本体には入れない独立ベンチ。
 *
 * マスク式: r = (m0 & ~a & ~b) | (m1 & a & ~b) | (m2 & ~a & b) | (m3 & a & b)
 *   m_i = 0u - ((tt>>i)&1u)  (ttのbit iが1なら全ビット1、0なら全ビット0)
 * これは2入力LUTの標準的な最小項展開そのものなので、switch版と論理的に
 * 等価なはず — が、**出力の一致検査を必ず入れる**(ベンチに一致検査を
 * 入れ忘れて嘘の結論が出た前例がプロジェクトにある。「測ってから最適化する」の徹底)。
 *
 * 実カーネルのアクセスパターン(deploy/layout.py::_build_conv_layout):
 *   instr_flat は (tt, a_off, b_off, out_off) の4整数組の平坦配列。
 *   off は「スロット番号 × tap_rows_words」で**あらかじめthreads倍済み**。
 *   mdl_k_conv_exec は thread i(0..tap_rows_words-1)ごとに
 *     sc[out_off + i] = gate2(tt, sc[a_off + i], sc[b_off + i])
 *   を命令の並び順に直列実行する。このベンチもそれをそのまま再現する
 *   (スロット番号→ワードオフセットへの倍率だけがthreads=スレッド数)。
 *
 * ビルド:
 *   clang++ -std=c++17 -O2 -framework Metal -framework Foundation \
 *     scripts/gpu_calib/gate2_ab.mm -o /tmp/gate2_ab
 * 実行(リポジトリルートから。共有ヘッダの相対パスを解決するため):
 *   /tmp/gate2_ab [n_instr] [n_slots] [trials]
 *   (省略時 n_instr=2000, n_slots=32, trials=8)
 */
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

namespace {

std::string ReadFile(const std::string &path) {
  std::ifstream f(path);
  if (!f) {
    std::fprintf(stderr, "共有ヘッダを読めない: %s\n"
                "(リポジトリルートから実行しているか確認すること)\n",
                path.c_str());
    std::exit(1);
  }
  std::ostringstream ss;
  ss << f.rdbuf();
  return ss.str();
}

std::string StripIncludeGuard(std::string src) {
  auto erase_all = [&](const std::string &needle) {
    size_t pos;
    while ((pos = src.find(needle)) != std::string::npos) {
      src.erase(pos, needle.size());
    }
  };
  erase_all("#ifndef MDL_GPU_KERNELS_H");
  erase_all("#define MDL_GPU_KERNELS_H");
  erase_all("#endif /* MDL_GPU_KERNELS_H */");
  return src;
}

constexpr const char *kMaskFnAndEntries = R"MSL(
/* 無分岐マスク式(A/B比較対象)。標準的な2入力LUTの最小項展開。*/
MDL_DEV inline unsigned int mdl_g_gate2_mask(unsigned int tt, unsigned int a,
                                             unsigned int b) {
  const unsigned int m0 = 0u - ((tt >> 0) & 1u);
  const unsigned int m1 = 0u - ((tt >> 1) & 1u);
  const unsigned int m2 = 0u - ((tt >> 2) & 1u);
  const unsigned int m3 = 0u - ((tt >> 3) & 1u);
  const unsigned int na = ~a, nb = ~b;
  return (m0 & na & nb) | (m1 & a & nb) | (m2 & na & b) | (m3 & a & b);
}

/* 両カーネルとも初期化式は完全に同一(同じ入力から始めることで、命令列の
   実行結果だけを比較できるようにする)。 */
MDL_DEV inline uint mdl_calib_init(uint tid, int s) {
  uint v = tid ^ ((uint)s * 2654435761u) ^ 0x9e3779b9u;
  v ^= v << 13; v ^= v >> 17; v ^= v << 5;
  return v;
}

kernel void mdl_calib_gate2_switch(device uint *scratch [[buffer(0)]],
                                   constant int *instr [[buffer(1)]],
                                   constant int &n_instr [[buffer(2)]],
                                   constant int &threads [[buffer(3)]],
                                   constant int &n_slots [[buffer(4)]],
                                   uint tid [[thread_position_in_grid]]) {
  if ((int)tid >= threads) return;
  for (int s = 0; s < n_slots; s++) {
    scratch[(uint)s * (uint)threads + tid] = mdl_calib_init(tid, s);
  }
  for (int j = 0; j < n_instr; j++) {
    constant int *ins = instr + j * 4;
    const uint a = scratch[(uint)ins[1] + tid];
    const uint b = scratch[(uint)ins[2] + tid];
    scratch[(uint)ins[3] + tid] = mdl_g_gate2((uint)ins[0], a, b);
  }
}

kernel void mdl_calib_gate2_mask(device uint *scratch [[buffer(0)]],
                                 constant int *instr [[buffer(1)]],
                                 constant int &n_instr [[buffer(2)]],
                                 constant int &threads [[buffer(3)]],
                                 constant int &n_slots [[buffer(4)]],
                                 uint tid [[thread_position_in_grid]]) {
  if ((int)tid >= threads) return;
  for (int s = 0; s < n_slots; s++) {
    scratch[(uint)s * (uint)threads + tid] = mdl_calib_init(tid, s);
  }
  for (int j = 0; j < n_instr; j++) {
    constant int *ins = instr + j * 4;
    const uint a = scratch[(uint)ins[1] + tid];
    const uint b = scratch[(uint)ins[2] + tid];
    scratch[(uint)ins[3] + tid] = mdl_g_gate2_mask((uint)ins[0], a, b);
  }
}
)MSL";

struct Instr { int32_t tt, a_slot, b_slot, out_slot; };

std::vector<Instr> GenProgram(int n_instr, int n_slots, uint32_t seed) {
  std::mt19937 rng(seed);
  std::uniform_int_distribution<int> tt_dist(0, 15);
  std::uniform_int_distribution<int> slot_dist(0, n_slots - 1);
  std::vector<Instr> prog(n_instr);
  for (auto &ins : prog) {
    ins.tt = tt_dist(rng);
    ins.a_slot = slot_dist(rng);
    ins.b_slot = slot_dist(rng);
    ins.out_slot = slot_dist(rng);
  }
  return prog;
}

struct RunResult {
  std::vector<double> elapsed_ms;
  std::vector<uint32_t> final_scratch;
};

RunResult RunKernel(id<MTLDevice> device, id<MTLCommandQueue> queue,
                    id<MTLComputePipelineState> pso, uint32_t threads,
                    int n_slots, const std::vector<Instr> &prog, int trials,
                    int warmups) {
  const int n_instr = (int)prog.size();
  std::vector<int32_t> instr_scaled(prog.size() * 4);
  for (size_t j = 0; j < prog.size(); j++) {
    instr_scaled[j * 4 + 0] = prog[j].tt;
    instr_scaled[j * 4 + 1] = prog[j].a_slot * (int32_t)threads;
    instr_scaled[j * 4 + 2] = prog[j].b_slot * (int32_t)threads;
    instr_scaled[j * 4 + 3] = prog[j].out_slot * (int32_t)threads;
  }
  id<MTLBuffer> scratchBuf =
      [device newBufferWithLength:(size_t)n_slots * threads * 4
                           options:MTLResourceStorageModeShared];
  id<MTLBuffer> instrBuf =
      [device newBufferWithBytes:instr_scaled.data()
                           length:instr_scaled.size() * 4
                          options:MTLResourceStorageModeShared];
  const int32_t threads_i = (int32_t)threads;

  auto dispatch_once = [&]() {
    id<MTLCommandBuffer> cmd = [queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:pso];
    [enc setBuffer:scratchBuf offset:0 atIndex:0];
    [enc setBuffer:instrBuf offset:0 atIndex:1];
    [enc setBytes:&n_instr length:sizeof(n_instr) atIndex:2];
    [enc setBytes:&threads_i length:sizeof(threads_i) atIndex:3];
    [enc setBytes:&n_slots length:sizeof(n_slots) atIndex:4];
    NSUInteger tg = pso.maxTotalThreadsPerThreadgroup;
    if (tg > (NSUInteger)threads) tg = threads;
    if (tg > 256) tg = 256;
    if (tg < 1) tg = 1;
    [enc dispatchThreads:MTLSizeMake(threads, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];
    return cmd;
  };

  for (int w = 0; w < warmups; w++) dispatch_once();

  RunResult result;
  for (int t = 0; t < trials; t++) {
    id<MTLCommandBuffer> cmd = dispatch_once();
    result.elapsed_ms.push_back(
        (double)(cmd.GPUEndTime - cmd.GPUStartTime) * 1000.0);
  }
  const uint32_t *data = (const uint32_t *)[scratchBuf contents];
  result.final_scratch.assign(data, data + (size_t)n_slots * threads);
  return result;
}

double Median(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  return v[v.size() / 2];
}

}  // namespace

int main(int argc, char **argv) {
  const int n_instr = argc > 1 ? std::atoi(argv[1]) : 2000;
  const int n_slots = argc > 2 ? std::atoi(argv[2]) : 32;
  const int trials = argc > 3 ? std::atoi(argv[3]) : 8;
  const int warmups = 4;
  const std::string header_path = argc > 4 ? argv[4]
                                            : "runtime/gpu/mdl_gpu_kernels.h";

  @autoreleasepool {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (device == nil) {
      std::fprintf(stderr, "Metalデバイスが見つからない\n");
      return 1;
    }
    std::printf("device = %s\n", [[device name] UTF8String]);
    std::printf("n_instr=%d n_slots=%d trials=%d(+warmup%d)\n", n_instr,
               n_slots, trials, warmups);

    const std::string shared = StripIncludeGuard(ReadFile(header_path));
    const std::string src =
        std::string("#include <metal_stdlib>\nusing namespace metal;\n"
                    "#define MDL_DEV\n#define MDL_GLOBAL device\n"
                    "#define MDL_CONST constant\n"
                    "typedef unsigned int uint32_t;\n") +
        shared + "\n" + kMaskFnAndEntries;

    NSError *err = nil;
    id<MTLLibrary> lib =
        [device newLibraryWithSource:[NSString stringWithUTF8String:src.c_str()]
                              options:nil error:&err];
    if (lib == nil) {
      std::fprintf(stderr, "シェーダコンパイル失敗: %s\n",
                   [[err localizedDescription] UTF8String]);
      return 1;
    }
    id<MTLFunction> fnSwitch = [lib newFunctionWithName:@"mdl_calib_gate2_switch"];
    id<MTLFunction> fnMask = [lib newFunctionWithName:@"mdl_calib_gate2_mask"];
    id<MTLComputePipelineState> psoSwitch =
        [device newComputePipelineStateWithFunction:fnSwitch error:&err];
    id<MTLComputePipelineState> psoMask =
        [device newComputePipelineStateWithFunction:fnMask error:&err];
    if (psoSwitch == nil || psoMask == nil) {
      std::fprintf(stderr, "パイプライン作成失敗: %s\n",
                   [[err localizedDescription] UTF8String]);
      return 1;
    }
    id<MTLCommandQueue> queue = [device newCommandQueue];

    const std::vector<Instr> prog = GenProgram(n_instr, n_slots, 0xC0FFEEu);
    const std::vector<uint32_t> parallelism_configs = {32u, 1u << 20};

    bool all_match = true;
    std::printf("\n%-10s %-8s %14s %14s %10s %s\n", "threads", "variant",
               "median(ms)", "mean(ms)", "vs switch", "判定");
    for (uint32_t threads : parallelism_configs) {
      RunResult rs = RunKernel(device, queue, psoSwitch, threads, n_slots,
                               prog, trials, warmups);
      RunResult rm = RunKernel(device, queue, psoMask, threads, n_slots, prog,
                               trials, warmups);

      const bool match = rs.final_scratch == rm.final_scratch;
      all_match = all_match && match;
      if (!match) {
        size_t n_diff = 0;
        for (size_t i = 0; i < rs.final_scratch.size(); i++) {
          if (rs.final_scratch[i] != rm.final_scratch[i]) n_diff++;
        }
        std::fprintf(stderr,
                    "threads=%u: switch/mask 出力不一致! %zu/%zu 要素\n",
                    threads, n_diff, rs.final_scratch.size());
      }

      const double med_s = Median(rs.elapsed_ms);
      const double med_m = Median(rm.elapsed_ms);
      double mean_s = 0, mean_m = 0;
      for (double v : rs.elapsed_ms) mean_s += v;
      for (double v : rm.elapsed_ms) mean_m += v;
      mean_s /= rs.elapsed_ms.size();
      mean_m /= rm.elapsed_ms.size();

      const double speedup = med_s / med_m;   /* >1: maskの方が速い */
      std::string verdict;
      if (!match) {
        verdict = "不一致(判定不能)";
      } else if (speedup > 1.2) {
        verdict = "mask有意に速い(置換候補)";
      } else if (speedup < 1.0 / 1.2) {
        verdict = "switch有意に速い";
      } else {
        verdict = "誤差内(switchで良い)";
      }
      std::printf("%-10u %-8s %14.6f %14.6f %10s %s\n", threads, "switch",
                 med_s, mean_s, "-", "");
      std::printf("%-10u %-8s %14.6f %14.6f %9.3fx %s\n", threads, "mask",
                 med_m, mean_m, speedup, verdict.c_str());
      std::printf("  出力一致: %s\n", match ? "OK(bit-exact)" : "NG");
    }
    std::printf("\n総合: 一致検査 %s\n", all_match ? "全パターンOK" : "NG(要調査)");
    return all_match ? 0 : 1;
  }
}
