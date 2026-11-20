/* griffith-logic — Metal(Apple GPU)の自己較正ルーフライン計測。
 *
 * runtime/ 本体には入れない、独立したスタンドアロンのベンチ。
 * 「メモリアクセスなし・分岐なし・依存連鎖なし(独立な演算列)」の
 * 32bit XOR/AND/OR を大量に並べたカーネルを1M要素で走らせ、
 *     (総命令数 ÷ GPU時間) = このM3の実測ALUピーク(instr/s)
 * を出す。これが以後「ALU使用率 = 対象カーネルの実効instr/s ÷
 * この較正値」の分母になる(deploykit-integrationキャンペーン)。
 *
 * 依存連鎖を避けるため、8本の独立アキュムレータ(a0..a7)を持ち、
 * 各アンロール反復ごとに**異なる**マスク値(ループ不変の実行時値cと
 * コンパイル時定数の反復番号をXORしたもの)を適用する。同じ値を
 * 繰り返し XOR/AND/OR すると(a^=c を偶数回 → a、a&=c を複数回 → 1回と
 * 等価、など)LLVMベースのMetalコンパイラが代数的に畳んでしまい、
 * 「実行したつもり」が実際には実行されない罠があるため
 * (perf-huntスキルの「測ってから最適化する」を計測基盤自身にも適用)。
 *
 * ビルド:
 *   clang++ -std=c++17 -O2 -framework Metal -framework Foundation \
 *     scripts/gpu_calib/metal_calib.mm -o /tmp/metal_calib
 * 実行:
 *   /tmp/metal_calib [threads] [unroll] [trials]
 *   (省略時 threads=1048576, unroll=256, trials=3)
 */
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>

namespace {

/* 1反復あたりの「命令」内訳: 反復ごとに変わるマスクmを作るXOR1回 +
 * それを8本のアキュムレータへ XOR/AND/OR で適用する8回 = 9命令。
 * すべて32bit整数演算のみ(浮動小数点・分岐・メモリロードは無い)。 */
constexpr int kOpsPerIter = 9;

std::string BuildKernelSource(int unroll) {
  std::string body;
  body.reserve((size_t)unroll * 96);
  char line[192];
  for (int i = 0; i < unroll; i++) {
    const uint32_t k = (uint32_t)(i * 0x9E3779B1u);
    std::snprintf(line, sizeof(line),
        "  m = c ^ %uu; a0^=m; a1&=m; a2|=m; a3^=m; a4&=m; a5|=m; a6^=m; a7&=m;\n",
        k);
    body += line;
  }
  return
      "#include <metal_stdlib>\n"
      "using namespace metal;\n"
      "kernel void mdl_calib_alu(device uint *out [[buffer(0)]],\n"
      "                          constant uint &seed [[buffer(1)]],\n"
      "                          uint idx [[thread_position_in_grid]]) {\n"
      "  uint c = seed ^ idx;\n"
      "  uint a0=idx,a1=idx+1u,a2=idx+2u,a3=idx+3u,a4=idx+5u,a5=idx+7u,"
      "a6=idx+11u,a7=idx+13u;\n"
      "  uint m;\n" + body +
      "  out[idx] = a0^a1^a2^a3^a4^a5^a6^a7;\n"
      "}\n";
}

}  // namespace

int main(int argc, char **argv) {
  const uint32_t threads = argc > 1 ? (uint32_t)std::atoll(argv[1]) : (1u << 20);
  const int unroll = argc > 2 ? std::atoi(argv[2]) : 256;
  const int trials = argc > 3 ? std::atoi(argv[3]) : 3;
  const int warmups = argc > 4 ? std::atoi(argv[4]) : 8;

  @autoreleasepool {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (device == nil) {
      std::fprintf(stderr, "Metalデバイスが見つからない\n");
      return 1;
    }
    std::printf("device = %s\n", [[device name] UTF8String]);
    std::printf("threads=%u unroll=%d trials=%d ops_per_thread=%lld "
               "(=unroll*%d)\n",
               threads, unroll, trials,
               (long long)unroll * kOpsPerIter, kOpsPerIter);

    NSError *err = nil;
    const std::string src = BuildKernelSource(unroll);
    id<MTLLibrary> lib =
        [device newLibraryWithSource:[NSString stringWithUTF8String:src.c_str()]
                              options:nil error:&err];
    if (lib == nil) {
      std::fprintf(stderr, "シェーダコンパイル失敗: %s\n",
                   [[err localizedDescription] UTF8String]);
      return 1;
    }
    id<MTLFunction> fn = [lib newFunctionWithName:@"mdl_calib_alu"];
    id<MTLComputePipelineState> pso =
        [device newComputePipelineStateWithFunction:fn error:&err];
    if (pso == nil) {
      std::fprintf(stderr, "パイプライン作成失敗: %s\n",
                   [[err localizedDescription] UTF8String]);
      return 1;
    }

    id<MTLBuffer> outBuf =
        [device newBufferWithLength:(size_t)threads * 4
                             options:MTLResourceStorageModeShared];
    id<MTLCommandQueue> queue = [device newCommandQueue];
    const uint32_t seed = 0x1234abcdu;

    const long long total_instr = (long long)threads * unroll * kOpsPerIter;
    double best = 0.0, worst = 1e300, sum = 0.0;
    double sum_sq = 0.0;

    /* ウォームアップ(計測に含めない)。実測: プロセス起動直後のGPUは
     * 低電力クロック状態にあり、連続dispatchで動的にクロックが上がって
     * いく(初回と定常状態で2〜3倍instr/sが変わることがあった)。
     * 報告する3回は「立ち上がった後」の定常値にする。 */
    for (int w = 0; w < warmups; w++) {
      id<MTLCommandBuffer> cmd = [queue commandBuffer];
      id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
      [enc setComputePipelineState:pso];
      [enc setBuffer:outBuf offset:0 atIndex:0];
      [enc setBytes:&seed length:sizeof(seed) atIndex:1];
      NSUInteger tg0 = pso.maxTotalThreadsPerThreadgroup;
      if (tg0 > 256) tg0 = 256;
      [enc dispatchThreads:MTLSizeMake(threads, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(tg0, 1, 1)];
      [enc endEncoding];
      [cmd commit];
      [cmd waitUntilCompleted];
    }

    for (int t = 0; t < trials; t++) {
      id<MTLCommandBuffer> cmd = [queue commandBuffer];
      id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
      [enc setComputePipelineState:pso];
      [enc setBuffer:outBuf offset:0 atIndex:0];
      [enc setBytes:&seed length:sizeof(seed) atIndex:1];
      NSUInteger tg = pso.maxTotalThreadsPerThreadgroup;
      if (tg > 256) tg = 256;
      [enc dispatchThreads:MTLSizeMake(threads, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
      [enc endEncoding];
      [cmd commit];
      [cmd waitUntilCompleted];

      const double elapsed_s = (double)(cmd.GPUEndTime - cmd.GPUStartTime);
      const double instr_per_s = (double)total_instr / elapsed_s;
      std::printf("  trial %d: gpu_elapsed=%.6fms  instr/s=%.4e\n", t,
                 elapsed_s * 1000.0, instr_per_s);
      sum += instr_per_s;
      sum_sq += instr_per_s * instr_per_s;
      if (instr_per_s > best) best = instr_per_s;
      if (instr_per_s < worst) worst = instr_per_s;
    }

    const double mean = sum / trials;
    const double variance = sum_sq / trials - mean * mean;
    const double stddev = variance > 0 ? std::sqrt(variance) : 0.0;
    std::printf("---\n");
    std::printf("mean instr/s = %.4e  (best=%.4e worst=%.4e stddev/mean=%.2f%%)\n",
               mean, best, worst, 100.0 * stddev / mean);

    // out を1個だけ読んで、最適化で消されていないことを目視確認する
    // (全部同じ値になっていたら怪しい=定数畳み込みされている可能性)。
    const uint32_t *out = (const uint32_t *)[outBuf contents];
    std::printf("sanity: out[0]=%u out[1]=%u out[threads/2]=%u out[threads-1]=%u\n",
               out[0], out[1], out[threads / 2], out[threads - 1]);
  }
  return 0;
}
