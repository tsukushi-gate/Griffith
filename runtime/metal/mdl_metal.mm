/* griffith-logic — Metal(Apple GPU)ランタイム
 *
 * カーネルの計算式は runtime/gpu/mdl_gpu_kernels.h(CUDAと共有)、
 * 入口は runtime/metal/mdl_metal_entry.metal。どちらも生成コードが
 * 1本のMSLソース文字列に連結して mdl_metal_init へ渡す。
 *
 * 設計:
 *   - **1フレーム = 1コマンドバッファ**。ノードごとにエンコーダを開き、
 *     最後に mdl_metal_sync でまとめて流す(W10 で WebGPU が
 *     「層ごとに submit + 読み戻し」から「単一 submit」へ移して30倍に
 *     なったのと同じ理由。GPUは往復同期がすぐ支配項になる)
 *   - バッファは MTLResourceStorageModeShared(Apple Silicon は
 *     ユニファイドメモリなので、これで転送コピー自体が消える)
 */
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "mdl_metal.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <string>

/* `mdl_metal_buf` の実体。MTLBuffer そのものではなく (バッファ, オフセット)
 * の組にしてある。メモリ計画が1本の大きなアリーナを切り出して使う
 * ため、**呼び出し側の関数シグネチャを1つも変えずに**部分領域を渡せる。
 * `owns` はアリーナ本体だけが1で、ビューは参照しか持たない。 */
typedef struct {
  void *buf;            /* __bridge_retained id<MTLBuffer>(ownsのときだけ所有) */
  unsigned long off;
  int owns;
} mdl_view;

#define MDL_BUF(h) ((__bridge id<MTLBuffer>)(((mdl_view *)(h))->buf))
#define MDL_OFF(h) ((NSUInteger)(((mdl_view *)(h))->off))

namespace {

struct MdlMetalState {
  id<MTLDevice> device = nil;
  id<MTLCommandQueue> queue = nil;
  id<MTLLibrary> library = nil;
  id<MTLCommandBuffer> cmd = nil;
  id<MTLComputeCommandEncoder> enc = nil;
  NSMutableDictionary<NSString *, id<MTLComputePipelineState>> *pso = nil;
  /* ストリーム・パイプライン化。commit_asyncでコミットしたが
   * まだwait_slotで待っていないコマンドバッファ(スロットごと)。
   * ARCが保持するので、コミット後にgが参照を持ち続ければ解放されない。 */
  id<MTLCommandBuffer> pending[MDL_METAL_MAX_STREAMS] = {nil};
  char name[128] = {0};
  bool ready = false;
  /* 計測用(既定では数えるだけで、経路には影響しない)。 */
  long n_dispatch = 0;
  double encode_ms = 0.0;
  double wait_ms = 0.0;

  /* GPU時間の直接計測。既定OFF、encode_ms/wait_msとは独立に持つ。
   * mdl_metal.h のコメント参照(counterSets探索結果)。 */
  bool gpu_timing_enabled = false;
  bool gpu_counters_supported = false;   /* per-dispatchサンプリング対応 */
  id<MTLCounterSampleBuffer> gpu_counter_buf = nil;
  NSMutableArray<NSString *> *gpu_counter_names = nil;
  int gpu_counter_cap = 0;      /* 確保済みdispatchスロット数(サンプルは2倍) */
  int gpu_counter_used = 0;     /* 次に使うスロット */
  bool gpu_counter_overflow_warned = false;
  double gpu_ns_total = 0.0;
  NSMutableDictionary<NSString *, NSNumber *> *gpu_ns_by_kernel = nil;
  NSMutableDictionary<NSString *, NSNumber *> *gpu_count_by_kernel = nil;
  std::string gpu_breakdown_buf;
};

/* GPUカウンタサンプルバッファの容量(dispatch呼び出し単位。サンプルは
 * 開始/終了の2個ずつ使うので実サンプル数はこの2倍)。
 *
 * 実測(Apple M3): newCounterSampleBufferWithDescriptor: は
 * "Invalid sample buffer length: 65536 B. Expected range: 8 -> 32768"
 * で失敗する — サンプルバッファ長は **32768バイト(=4096サンプル、
 * 1サンプル8バイト)が上限**。dispatch1回に2サンプル使うので、
 * 確保できる上限は2048dispatchぶん。実モデルのdispatch回数は数百〜
 * 数千のオーダーなので、2048あれば1推論ぶん大抵は足りる
 * (超えた分はカウンタ計測をスキップして通常経路へフォールバックする=
 * 推論自体は壊れない)。 */
static const int kMdlGpuCounterCap = 2048;
/* MTLCounterErrorValue相当。SDKのシンボルに依存せず値で持つ
 * (Appleのドキュメント上、無効サンプルは UINT64_MAX で返る)。 */
static const uint64_t kMdlCounterErrorValue = UINT64_MAX;

double mdl_now_ms() {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return (double)t.tv_sec * 1000.0 + (double)t.tv_nsec / 1.0e6;
}

MdlMetalState g;

/* 積みっぱなしのエンコーダを取り出す(無ければ開く)。
 * ノードごとに commit しないのが肝。 */
id<MTLComputeCommandEncoder> encoder() {
  if (g.enc != nil) return g.enc;
  /* blit を積んだ直後は enc==nil だが cmd は生きている。ここで
   * commandBuffer を作り直すと、積んだ blit ごと捨ててしまう
   * (concat がグラフ途中にある seg で実際に結果が壊れた)。 */
  if (g.cmd == nil) g.cmd = [g.queue commandBuffer];
  g.enc = [g.cmd computeCommandEncoder];
  return g.enc;
}

/* GPUカウンタサンプルバッファを(必要なら)確保する。失敗したら
 * gpu_counters_supported を落として以後はコマンドバッファ粒度へ
 * フォールバックする(推論そのものは壊れない)。 */
void ensure_gpu_counter_buffer() {
  if (g.gpu_counter_buf != nil || !g.gpu_counters_supported || !g.ready) return;
  @autoreleasepool {
    id<MTLCounterSet> ts_set = nil;
    for (id<MTLCounterSet> cs in g.device.counterSets) {
      if ([cs.name isEqualToString:MTLCommonCounterSetTimestamp]) {
        ts_set = cs;
        break;
      }
    }
    if (ts_set == nil) {
      g.gpu_counters_supported = false;
      return;
    }
    MTLCounterSampleBufferDescriptor *desc =
        [MTLCounterSampleBufferDescriptor new];
    desc.counterSet = ts_set;
    desc.storageMode = MTLStorageModeShared;
    desc.sampleCount = (NSUInteger)kMdlGpuCounterCap * 2;
    NSError *err = nil;
    id<MTLCounterSampleBuffer> buf =
        [g.device newCounterSampleBufferWithDescriptor:desc error:&err];
    if (buf == nil) {
      NSLog(@"mdl_metal: GPUカウンタサンプルバッファ確保に失敗、"
            "コマンドバッファ粒度の計測へフォールバックする: %@",
            [err localizedDescription]);
      g.gpu_counters_supported = false;
      return;
    }
    g.gpu_counter_buf = buf;
    g.gpu_counter_cap = kMdlGpuCounterCap;
    g.gpu_counter_used = 0;
    g.gpu_counter_names =
        [NSMutableArray arrayWithCapacity:(NSUInteger)kMdlGpuCounterCap];
    for (int i = 0; i < kMdlGpuCounterCap; i++) {
      [g.gpu_counter_names addObject:@""];
    }
    g.gpu_ns_by_kernel = [NSMutableDictionary dictionary];
    g.gpu_count_by_kernel = [NSMutableDictionary dictionary];
  }
}

/* サンプルバッファに積んだ分を解決してカーネル別に累積する。
 * mdl_metal_sync が「積んだコマンドバッファをwaitUntilCompletedした直後」
 * にのみ呼ぶ(GPU側の書き込みが終わっている必要がある)。 */
void resolve_gpu_counters() {
  @autoreleasepool {
    const int used = g.gpu_counter_used;
    if (used <= 0) return;
    NSData *data =
        [g.gpu_counter_buf resolveCounterRange:NSMakeRange(0, (NSUInteger)used * 2)];
    if (data == nil ||
        data.length < (NSUInteger)used * 2 * sizeof(MTLCounterResultTimestamp)) {
      NSLog(@"mdl_metal: GPUカウンタの解決に失敗(resolveCounterRange)");
      g.gpu_counter_used = 0;
      return;
    }
    const MTLCounterResultTimestamp *results =
        (const MTLCounterResultTimestamp *)data.bytes;
    for (int i = 0; i < used; i++) {
      const uint64_t t0 = results[i * 2].timestamp;
      const uint64_t t1 = results[i * 2 + 1].timestamp;
      if (t0 == kMdlCounterErrorValue || t1 == kMdlCounterErrorValue || t1 < t0) {
        continue;   /* 無効サンプル。この呼び出しぶんは計測から除く */
      }
      const double ns = (double)(t1 - t0);
      NSString *name = g.gpu_counter_names[i];
      g.gpu_ns_total += ns;
      NSNumber *prev_ns = g.gpu_ns_by_kernel[name];
      g.gpu_ns_by_kernel[name] = @(ns + (prev_ns != nil ? prev_ns.doubleValue : 0.0));
      NSNumber *prev_cnt = g.gpu_count_by_kernel[name];
      g.gpu_count_by_kernel[name] =
          @((prev_cnt != nil ? prev_cnt.intValue : 0) + 1);
    }
    g.gpu_counter_used = 0;
  }
}

/* `explicit_tg`/`tg_mem_bytes`/`by_groups` は既定(0/0/false)なら
 * 元の挙動と完全に同じ(全14既存呼び出しは無変更のまま)。並列縮約カーネル
 * (mdl_k_sgs_r / mdl_k_group_sum_r)だけがこれを使う:
 *   explicit_tg   スレッドグループ幅を明示指定(host側が選んだR)。
 *                 0なら従来通り maxTotalThreadsPerThreadgroup を256で
 *                 キャップした値を使う。
 *   tg_mem_bytes  [[threadgroup(0)]] のバッファ長(共有メモリ、バイト)。
 *                 0なら setThreadgroupMemoryLength を呼ばない。
 *   by_groups     true なら `total` を「スレッドグループ数」として
 *                 dispatchThreadgroups(各グループはちょうど tg 幅)で
 *                 起動する。並列縮約カーネルは「1グループ=1出力(ワード)」
 *                 という前提でレーンのg_lo/g_hiを計算しているため、
 *                 dispatchThreads(非一様グループ分割)ではなく明示的に
 *                 グループ境界を揃える必要がある。 */
int dispatch(NSString *name, int total,
             void (^bind)(id<MTLComputeCommandEncoder>),
             int explicit_tg = 0, int tg_mem_bytes = 0,
             bool by_groups = false, int tg_mem_bytes2 = 0) {
  if (!g.ready || total <= 0) return g.ready ? 0 : -1;
  {
    id<MTLComputePipelineState> pso = g.pso[name];
    if (pso == nil) {
      NSLog(@"mdl_metal: パイプラインが無い: %@", name);
      return -1;
    }
    if (g.gpu_timing_enabled && g.gpu_counters_supported) {
      ensure_gpu_counter_buffer();
    }
    const bool use_counter = g.gpu_timing_enabled && g.gpu_counters_supported &&
                             g.gpu_counter_buf != nil &&
                             g.gpu_counter_used < g.gpu_counter_cap;
    if (g.gpu_timing_enabled && g.gpu_counters_supported &&
        g.gpu_counter_buf != nil && !use_counter &&
        !g.gpu_counter_overflow_warned) {
      NSLog(@"mdl_metal: GPUカウンタスロット上限(%d)超過。"
            "以後このフレームのdispatchはGPU計測をスキップする"
            "(推論結果には影響しない)。", kMdlGpuCounterCap);
      g.gpu_counter_overflow_warned = true;
    }
    const double t_enc = mdl_now_ms();
    id<MTLComputeCommandEncoder> enc;
    if (use_counter) {
      /* このデバイスは AtDispatchBoundary 非対応(実測: Apple M3は
       * AtStageBoundary=trueのみ)なので、dispatch粒度の時間を取るには
       * 「1dispatch = 1エンコーダ」に分割してエンコーダ開始/終了で
       * サンプルするしかない。既定OFFのときはこの分岐に入らないので、
       * 通常経路の「複数dispatchを1エンコーダにまとめる」最適化は
       * 一切影響を受けない。 */
      if (g.enc != nil) {
        [g.enc endEncoding];
        g.enc = nil;
      }
      if (g.cmd == nil) g.cmd = [g.queue commandBuffer];
      MTLComputePassDescriptor *cpd = [MTLComputePassDescriptor computePassDescriptor];
      cpd.sampleBufferAttachments[0].sampleBuffer = g.gpu_counter_buf;
      cpd.sampleBufferAttachments[0].startOfEncoderSampleIndex =
          (NSUInteger)(g.gpu_counter_used * 2);
      cpd.sampleBufferAttachments[0].endOfEncoderSampleIndex =
          (NSUInteger)(g.gpu_counter_used * 2 + 1);
      enc = [g.cmd computeCommandEncoderWithDescriptor:cpd];
      g.gpu_counter_names[g.gpu_counter_used] = name;
      g.gpu_counter_used++;
    } else {
      enc = encoder();
    }
    [enc setComputePipelineState:pso];
    bind(enc);
    g.n_dispatch++;
    {
      NSUInteger tg = pso.maxTotalThreadsPerThreadgroup;
      if (tg > 256) tg = 256;
      if (explicit_tg > 0) {
        if ((NSUInteger)explicit_tg > pso.maxTotalThreadsPerThreadgroup) {
          NSLog(@"mdl_metal: %@ の要求スレッドグループ幅%dがGPU上限%luを"
                "超える(結果が壊れるので実行しない)", name, explicit_tg,
                (unsigned long)pso.maxTotalThreadsPerThreadgroup);
          if (use_counter) [enc endEncoding];
          return -1;
        }
        tg = (NSUInteger)explicit_tg;
      }
      if (tg_mem_bytes > 0) {
        [enc setThreadgroupMemoryLength:(NSUInteger)tg_mem_bytes atIndex:0];
      }
      /* dense多層融合カーネルはping-pong活性のthreadgroupメモリを
       * 2本([[threadgroup(0)]]/[[threadgroup(1)]])使う。既存の14呼び出しは
       * tg_mem_bytes2を渡さない(既定0)ので、この分岐に入らず無変更。 */
      if (tg_mem_bytes2 > 0) {
        [enc setThreadgroupMemoryLength:(NSUInteger)tg_mem_bytes2 atIndex:1];
      }
      if (by_groups) {
        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)total, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
      } else {
        [enc dispatchThreads:MTLSizeMake((NSUInteger)total, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
      }
    }
    if (use_counter) {
      [enc endEncoding];   /* g.enc はこの経路では常にnilのまま */
    }
    g.encode_ms += mdl_now_ms() - t_enc;
    return 0;
  }
}

/* 小さなスカラー構造体(MdlGeom)をそのままエンコーダへ載せる。 */
struct Geom {
  int c, h, w, wpr, h_out, w_out, wpr_out, extra;
  int batch, in_stride;      /* mdl_metal_entry.metal の MdlGeom と同一 */
};

int words_per_row(int w) { return (w + 31) >> 5; }

/* group_sum / spatial_group_sum の並列度不足対策で使う縮約幅R。
 * mdl_cuda.cu の同名関数と全く同じロジック(ホスト専用の純粋な計算なので
 * mdl_gpu_kernels.h には置かず、CUDA/Metal 双方の入口ファイルに小さく
 * 複製している — 詳しい理由はそちら側のコメント参照)。 */
int mdl_gpu_pick_r(int work_items, int elems_per_item, int target_threads,
                   int max_r) {
  int r = 1;
  if (work_items <= 0 || elems_per_item <= 1) return 1;
  while (r * 2 <= elems_per_item && r * 2 <= max_r &&
        (long long)work_items * (r * 2) <= (long long)target_threads) {
    r *= 2;
  }
  return r;
}

static const int kMdlGsTargetThreads = 1024;
static const int kMdlGsMaxR = 64;

}  // namespace

extern "C" {

int mdl_metal_init(const char *shader_source) {
  return mdl_metal_init_ex(NULL, shader_source);
}

int mdl_metal_init_ex(const char *metallib_path, const char *shader_source) {
  if (g.ready) return 0;
  @autoreleasepool {
    g.device = MTLCreateSystemDefaultDevice();
    if (g.device == nil) {
      NSLog(@"mdl_metal: Metalデバイスが見つからない");
      return -1;
    }
    snprintf(g.name, sizeof(g.name), "%s", [[g.device name] UTF8String]);

    /* GPU時間の直接計測の対応可否をここで一度だけ判定する。
     * 実測(Apple M3): counterSets に timestamp(MTLCommonCounterSetTimestamp)
     * は1本あるが、supportsCounterSampling は AtStageBoundary のみ true
     * (AtDispatchBoundary/AtDrawBoundary/AtBlitBoundary/
     * AtTileDispatchBoundary は false)。dispatch()側は
     * gpu_counters_supported を見て「1dispatch=1エンコーダ」に分割するか
     * どうかを決める。 */
    g.gpu_counters_supported = false;
    for (id<MTLCounterSet> cs in g.device.counterSets) {
      if ([cs.name isEqualToString:MTLCommonCounterSetTimestamp]) {
        g.gpu_counters_supported =
            [g.device supportsCounterSampling:MTLCounterSamplingPointAtStageBoundary];
        break;
      }
    }
    {
      const char *env = getenv("MDL_METAL_GPU_TIMING");
      if (env != NULL && env[0] == '1') g.gpu_timing_enabled = true;
    }

    NSError *err = nil;

    /* 事前コンパイル済み .metallib があればそれを使う。
     * 実測: 実行時コンパイルは「Metalコンパイラの起動 約51ms」+
     * 「このシェーダのコンパイル 約73ms」= 約124ms の**起動時のみ**の費用。
     * 推論の定常性能(約0.9ms)には影響しない。短命プロセスで効く。 */
    if (metallib_path != NULL && metallib_path[0] != '\0') {
      NSString *path = [NSString stringWithUTF8String:metallib_path];
      if ([[NSFileManager defaultManager] fileExistsAtPath:path]) {
        g.library = [g.device newLibraryWithURL:[NSURL fileURLWithPath:path]
                                          error:&err];
        if (g.library == nil) {
          NSLog(@"mdl_metal: %@ を読めないのでソースからコンパイルする: %@",
                path, [err localizedDescription]);
          err = nil;
        }
      }
    }

    if (g.library == nil) {
      if (shader_source == NULL) {
        NSLog(@"mdl_metal: .metallib も シェーダソースも無い");
        return -1;
      }
      MTLCompileOptions *opt = [MTLCompileOptions new];
      g.library = [g.device
          newLibraryWithSource:[NSString stringWithUTF8String:shader_source]
                       options:opt
                         error:&err];
    }
    if (g.library == nil) {
      NSLog(@"mdl_metal: シェーダのコンパイルに失敗: %@",
            [err localizedDescription]);
      return -1;
    }
    g.queue = [g.device newCommandQueue];
    g.pso = [NSMutableDictionary dictionary];
    /* パイプラインはライブラリに実在するカーネルから作る。名前を手で
     * 並べると、カーネルを1つ足したときに登録を忘れて「実行時に黙って
     * 動かない」が起きる。`functionNames` から取れば、その取りこぼしが
     * 原理的に無くなる。 */
    for (NSString *name in [g.library functionNames]) {
      if (![name hasPrefix:@"mdl_k_"]) continue;
      id<MTLFunction> fn = [g.library newFunctionWithName:name];
      if (fn == nil) continue;
      id<MTLComputePipelineState> pso =
          [g.device newComputePipelineStateWithFunction:fn error:&err];
      if (pso == nil) {
        NSLog(@"mdl_metal: パイプライン生成に失敗 %@: %@", name,
              [err localizedDescription]);
        return -1;
      }
      g.pso[name] = pso;
    }
    g.ready = true;
    return 0;
  }
}

void mdl_metal_shutdown(void) {
  /* ストリーム・パイプライン化: 積みっぱなし(commit_async済み・
   * wait_slot未実施)のコマンドバッファが残っていれば待ってから捨てる
   * (待たずに参照を外してもGPU側は完走するが、shutdown直後にプロセスが
   * 終了する用途を考えると行儀が悪い。ここでの待ちはshutdown時だけの
   * コストなので推論の定常性能には影響しない)。 */
  for (int i = 0; i < MDL_METAL_MAX_STREAMS; i++) {
    if (g.pending[i] != nil) {
      [g.pending[i] waitUntilCompleted];
      g.pending[i] = nil;
    }
  }
  g.enc = nil;
  g.cmd = nil;
  g.pso = nil;
  g.library = nil;
  g.queue = nil;
  g.device = nil;
  g.ready = false;
  /* GPU計測。gpu_timing_enabled(呼び出し側の設定)自体は次のinitでも
   * 引き継ぐ。対応可否は次のinitで検出し直すのでリセットする。 */
  g.gpu_counters_supported = false;
  g.gpu_counter_buf = nil;
  g.gpu_counter_names = nil;
  g.gpu_counter_cap = 0;
  g.gpu_counter_used = 0;
  g.gpu_counter_overflow_warned = false;
  g.gpu_ns_by_kernel = nil;
  g.gpu_count_by_kernel = nil;
}

const char *mdl_metal_device_name(void) { return g.ready ? g.name : NULL; }

mdl_metal_buf mdl_metal_alloc(size_t bytes) {
  if (!g.ready) return NULL;
  if (bytes == 0) bytes = 4;
  {
    id<MTLBuffer> b = [g.device newBufferWithLength:bytes
                                            options:MTLResourceStorageModeShared];
    mdl_view *v = (mdl_view *)malloc(sizeof(mdl_view));
    if (v == NULL) return NULL;
    v->buf = (__bridge_retained void *)b;
    v->off = 0;
    v->owns = 1;
    return (mdl_metal_buf)v;
  }
}

mdl_metal_buf mdl_metal_view(mdl_metal_buf base, size_t offset) {
  if (base == NULL) return NULL;
  {
    mdl_view *b = (mdl_view *)base;
    mdl_view *v = (mdl_view *)malloc(sizeof(mdl_view));
    if (v == NULL) return NULL;
    v->buf = b->buf;              /* 所有はベース側。ビューは参照だけ持つ */
    v->off = b->off + offset;
    v->owns = 0;
    return (mdl_metal_buf)v;
  }
}

void mdl_metal_free(mdl_metal_buf buf) {
  if (buf == NULL) return;
  {
    mdl_view *v = (mdl_view *)buf;
    if (v->owns) {
      id<MTLBuffer> b = (__bridge_transfer id<MTLBuffer>)v->buf;
      b = nil;
    }
    v->buf = NULL;
    free(v);
  }
}

int mdl_metal_upload(mdl_metal_buf dst, const void *src, size_t bytes) {
  if (dst == NULL) return -1;
  memcpy((char *)[MDL_BUF(dst) contents] + MDL_OFF(dst), src, bytes);
  return 0;   /* ユニファイドメモリなので直接書ける */
}

int mdl_metal_download(void *dst, mdl_metal_buf src, size_t bytes) {
  if (src == NULL) return -1;
  memcpy(dst, (const char *)[MDL_BUF(src) contents] + MDL_OFF(src), bytes);
  return 0;
}

int mdl_metal_copy(mdl_metal_buf dst, size_t dst_off, mdl_metal_buf src,
                   size_t src_off, size_t bytes) {
  if (!g.ready || dst == NULL || src == NULL) return -1;
  /* concat は blit エンコーダでGPUのタイムライン上に積む。
   * ここでCPU側 memcpy をやると直前のカーネル結果を待つために
   * コマンドバッファを流し切ることになり、グラフの途中でGPU往復が入る
   * (W10 が WebGPU で潰したのとまったく同じ落とし穴)。 */
  if (g.enc != nil) {
    [g.enc endEncoding];
    g.enc = nil;
  }
  if (g.cmd == nil) g.cmd = [g.queue commandBuffer];
  {
    id<MTLBlitCommandEncoder> blit = [g.cmd blitCommandEncoder];
    [blit copyFromBuffer:MDL_BUF(src)
            sourceOffset:MDL_OFF(src) + src_off
                toBuffer:MDL_BUF(dst)
       destinationOffset:MDL_OFF(dst) + dst_off
                    size:bytes];
    [blit endEncoding];
  }
  return 0;
}

/* copy_batched_exのバッチ一括版。ホスト側でN回`mdl_metal_copy`
 * (blitエンコーダ)を発行する代わりに、1回のcomputeカーネル起動で
 * バッチ全レーンぶんを運ぶ(mdl_gpu_kernels.h::mdl_g_copy_strided_byte
 * 参照)。MdlCopyStridedParams(runtime/metal/mdl_metal_entry.metal)と
 * 同一レイアウトのローカル構造体を使う(MdlGeomと同じ約束)。 */
int mdl_metal_copy_strided(mdl_metal_buf dst, int dst_off, int dst_stride,
                           mdl_metal_buf src, int src_off, int src_stride,
                           int n_bytes, int batch) {
  struct {
    int dst_off, dst_stride, src_off, src_stride, n_bytes, batch;
  } p = {dst_off, dst_stride, src_off, src_stride, n_bytes, batch};
  return dispatch(@"mdl_k_copy_strided", n_bytes * batch,
                  ^(id<MTLComputeCommandEncoder> e) {
    [e setBuffer:MDL_BUF(src) offset:MDL_OFF(src) atIndex:0];
    [e setBuffer:MDL_BUF(dst) offset:MDL_OFF(dst) atIndex:1];
    [e setBytes:&p length:sizeof(p) atIndex:2];
  });
}

/* ---- RDDLGN自己回帰生成(decode)ハーネス専用 ---- */
int mdl_metal_embed_lookup(mdl_metal_buf tokens, mdl_metal_buf table,
                           mdl_metal_buf out, int d_emb, int batch) {
  Geom geom = {d_emb, 0, 0, 0, 0, 0, 0, 0, batch, 0};
  return dispatch(@"mdl_k_embed_lookup", d_emb * batch,
                  ^(id<MTLComputeCommandEncoder> e) {
    [e setBuffer:MDL_BUF(tokens) offset:MDL_OFF(tokens) atIndex:0];
    [e setBuffer:MDL_BUF(table) offset:MDL_OFF(table) atIndex:1];
    [e setBuffer:MDL_BUF(out) offset:MDL_OFF(out) atIndex:2];
    [e setBytes:&geom length:sizeof(geom) atIndex:3];
  });
}

int mdl_metal_argmax(mdl_metal_buf scores, mdl_metal_buf tokens, int width,
                     int batch) {
  Geom geom = {width, 0, 0, 0, 0, 0, 0, 0, batch, 0};
  return dispatch(@"mdl_k_argmax", batch, ^(id<MTLComputeCommandEncoder> e) {
    [e setBuffer:MDL_BUF(scores) offset:MDL_OFF(scores) atIndex:0];
    [e setBuffer:MDL_BUF(tokens) offset:MDL_OFF(tokens) atIndex:1];
    [e setBytes:&geom length:sizeof(geom) atIndex:2];
  });
}

/* RDDLGN decodeのmegakernel経路(1コマンドバッファ+1同期だった
 * 「1tickあたり複数dispatch+D2Dコピー5本」を、1スレッドグループ=
 * 1レーンの単一カーネルへ畳む。tickループ自体をカーネル内(threadgroup
 * バリアで層間同期)へ持ち込むので、ホスト側は1回のdispatchで済む。
 * `idx_bufs`/`tt_bufs`はdec_l_rec+dec_pm_recの全lut_dense層ぶん(層順、
 * `mdl_metal_dense_fuse_exec`と同じ可変長バインディング)。tg_bytesは
 * 生成時に計算したthreadgroupバッファ1本ぶんの総バイト数
 * (`deploy/rddlgn_decode_megakernel.py::_threadgroup_layout`が計算し、
 * 32KB以内でなければこの関数を使う経路自体を生成しない=収まらない
 * モデルは自動的に従来経路へフォールバックする)。*/
int mdl_metal_rddlgn_decode_megakernel_exec(
    const char *kernel_name, mdl_metal_buf d_token, mdl_metal_buf d_table,
    mdl_metal_buf d_context, mdl_metal_buf d_pstate, mdl_metal_buf d_out,
    mdl_metal_buf *idx_bufs, mdl_metal_buf *tt_bufs, int n_layers,
    int tg_bytes, int n) {
  NSString *name = [NSString stringWithUTF8String:kernel_name];
  id<MTLComputePipelineState> pso = g.pso[name];
  if (pso == nil) {
    NSLog(@"mdl_metal: パイプラインが無い: %@", name);
    return -1;
  }
  {
    NSUInteger tg = pso.maxTotalThreadsPerThreadgroup;
    if (tg > 1024) tg = 1024;
    return dispatch(name, n, ^(id<MTLComputeCommandEncoder> e) {
      [e setBuffer:MDL_BUF(d_token) offset:MDL_OFF(d_token) atIndex:0];
      [e setBuffer:MDL_BUF(d_table) offset:MDL_OFF(d_table) atIndex:1];
      [e setBuffer:MDL_BUF(d_context) offset:MDL_OFF(d_context) atIndex:2];
      [e setBuffer:MDL_BUF(d_pstate) offset:MDL_OFF(d_pstate) atIndex:3];
      [e setBuffer:MDL_BUF(d_out) offset:MDL_OFF(d_out) atIndex:4];
      for (int L = 0; L < n_layers; L++) {
        [e setBuffer:MDL_BUF(idx_bufs[L]) offset:MDL_OFF(idx_bufs[L])
               atIndex:5 + 2 * L];
        [e setBuffer:MDL_BUF(tt_bufs[L]) offset:MDL_OFF(tt_bufs[L])
               atIndex:6 + 2 * L];
      }
    }, /*explicit_tg=*/(int)tg, /*tg_mem_bytes=*/tg_bytes, /*by_groups=*/true);
  }
}

int mdl_metal_sync(void) {
  if (!g.ready) return -1;
  if (g.enc != nil) {
    [g.enc endEncoding];
    g.enc = nil;
  }
  /* エンコーダを閉じただけでは何も走らない。blit だけを積んだ場合も
   * コマンドバッファが残っているので、cmd の有無で判断する。 */
  if (g.cmd != nil) {
    id<MTLCommandBuffer> cmd = g.cmd;
    const double t0 = mdl_now_ms();
    [cmd commit];
    [cmd waitUntilCompleted];
    g.cmd = nil;
    g.wait_ms += mdl_now_ms() - t0;
    if (g.gpu_timing_enabled) {
      if (g.gpu_counters_supported && g.gpu_counter_buf != nil &&
          g.gpu_counter_used > 0) {
        resolve_gpu_counters();
      } else if (!g.gpu_counters_supported) {
        /* フォールバック: コマンドバッファ粒度(カーネル別内訳は取れない)。
         * 実測でGPUStartTime/GPUEndTimeは秒単位。 */
        g.gpu_ns_total += (double)(cmd.GPUEndTime - cmd.GPUStartTime) * 1.0e9;
      }
    }
  }
  return 0;
}

/* ---- ストリーム・パイプライン化 ---- */

int mdl_metal_commit_async(int slot) {
  if (!g.ready) return -1;
  if (slot < 0 || slot >= MDL_METAL_MAX_STREAMS) return -1;
  if (g.pending[slot] != nil) {
    /* 前回ぶんをまだ mdl_metal_wait_slot していない。ここで黙って
     * 上書きすると、GPUがまだ読み書きしているかもしれないバッファへ
     * 次の入力をエンコードすることになり結果が壊れる(教訓)。
     * 安全側に倒して拒否する。 */
    NSLog(@"mdl_metal: スロット%d がまだpending(mdl_metal_wait_slotを"
          "呼ばずに次のcommit_asyncを呼んだ)", slot);
    return -1;
  }
  if (g.enc != nil) {
    [g.enc endEncoding];
    g.enc = nil;
  }
  if (g.cmd == nil) return -1;   /* 何も積んでいない */
  {
    id<MTLCommandBuffer> cmd = g.cmd;
    [cmd commit];   /* waitしない — GPUはキュー上で前の分の後に走る */
    g.pending[slot] = cmd;
    g.cmd = nil;
  }
  return 0;
}

int mdl_metal_wait_slot(int slot) {
  if (!g.ready) return -1;
  if (slot < 0 || slot >= MDL_METAL_MAX_STREAMS) return -1;
  if (g.pending[slot] == nil) return 0;   /* 何も積んでいなければ冪等に戻る */
  {
    id<MTLCommandBuffer> cmd = g.pending[slot];
    const double t0 = mdl_now_ms();
    [cmd waitUntilCompleted];
    g.pending[slot] = nil;
    g.wait_ms += mdl_now_ms() - t0;
    if (g.gpu_timing_enabled) {
      if (g.gpu_counters_supported && g.gpu_counter_buf != nil &&
          g.gpu_counter_used > 0) {
        resolve_gpu_counters();
      } else if (!g.gpu_counters_supported) {
        g.gpu_ns_total += (double)(cmd.GPUEndTime - cmd.GPUStartTime) * 1.0e9;
      }
    }
  }
  return 0;
}

long mdl_metal_dispatch_count(void) { return g.n_dispatch; }
double mdl_metal_encode_ms(void) { return g.encode_ms; }
double mdl_metal_wait_ms(void) { return g.wait_ms; }
void mdl_metal_reset_stats(void) {
  g.n_dispatch = 0; g.encode_ms = 0.0; g.wait_ms = 0.0;
}

/* ---- GPU時間の直接計測 ---- */

void mdl_metal_set_gpu_timing(int enable) {
  g.gpu_timing_enabled = (enable != 0);
}

int mdl_metal_gpu_timing_enabled(void) { return g.gpu_timing_enabled ? 1 : 0; }

int mdl_metal_gpu_timing_supported(void) {
  return (g.ready && g.gpu_counters_supported) ? 1 : 0;
}

double mdl_metal_gpu_ms(void) { return g.gpu_ns_total / 1.0e6; }

const char *mdl_metal_gpu_breakdown(void) {
  std::string &out = g.gpu_breakdown_buf;
  out.clear();
  if (!g.gpu_timing_enabled) {
    out = "(GPU計測はOFF。mdl_metal_set_gpu_timing(1)で有効化すること)";
    return out.c_str();
  }
  if (!g.gpu_counters_supported || g.gpu_ns_by_kernel == nil) {
    char buf[192];
    snprintf(buf, sizeof(buf),
            "(per-dispatchカウンタ非対応: コマンドバッファ粒度のみ。"
            "累積 %.4fms)", g.gpu_ns_total / 1.0e6);
    out = buf;
    return out.c_str();
  }
  @autoreleasepool {
    NSDictionary<NSString *, NSNumber *> *ns_by_kernel = g.gpu_ns_by_kernel;
    NSArray<NSString *> *names = [ns_by_kernel.allKeys
        sortedArrayUsingComparator:^NSComparisonResult(NSString *a, NSString *b) {
      const double na = ns_by_kernel[a].doubleValue;
      const double nb = ns_by_kernel[b].doubleValue;
      if (na < nb) return NSOrderedDescending;
      if (na > nb) return NSOrderedAscending;
      return NSOrderedSame;
    }];
    for (NSString *name in names) {
      const double ns = ns_by_kernel[name].doubleValue;
      const int cnt = g.gpu_count_by_kernel[name].intValue;
      char line[256];
      snprintf(line, sizeof(line), "%-28s %10.4fms  (%d回)\n",
              name.UTF8String, ns / 1.0e6, cnt);
      out += line;
    }
  }
  if (out.empty()) out = "(記録なし)";
  return out.c_str();
}

void mdl_metal_reset_gpu_stats(void) {
  g.gpu_ns_total = 0.0;
  g.gpu_counter_used = 0;
  g.gpu_counter_overflow_warned = false;
  if (g.gpu_ns_by_kernel != nil) [g.gpu_ns_by_kernel removeAllObjects];
  if (g.gpu_count_by_kernel != nil) [g.gpu_count_by_kernel removeAllObjects];
}

int mdl_metal_pack(mdl_metal_buf src, mdl_metal_buf dst, int c, int h, int w) {
  Geom geom = {c, h, w, words_per_row(w), 0, 0, 0, 0, 1, 0};
  return dispatch(@"mdl_k_pack", c * h * geom.wpr,
                  ^(id<MTLComputeCommandEncoder> e) {
    [e setBuffer:MDL_BUF(src) offset:MDL_OFF(src) atIndex:0];
    [e setBuffer:MDL_BUF(dst) offset:MDL_OFF(dst) atIndex:1];
    [e setBytes:&geom length:sizeof(geom) atIndex:2];
  });
}

int mdl_metal_unpack(mdl_metal_buf src, mdl_metal_buf dst, int c, int h,
                     int w) {
  Geom geom = {c, h, w, words_per_row(w), 0, 0, 0, 0, 1, 0};
  return dispatch(@"mdl_k_unpack", c * h * w,
                  ^(id<MTLComputeCommandEncoder> e) {
    [e setBuffer:MDL_BUF(src) offset:MDL_OFF(src) atIndex:0];
    [e setBuffer:MDL_BUF(dst) offset:MDL_OFF(dst) atIndex:1];
    [e setBytes:&geom length:sizeof(geom) atIndex:2];
  });
}

int mdl_metal_preproc(mdl_metal_buf image, int c_img, int h, int w,
                      mdl_metal_buf desc, int channels, mdl_metal_buf thr,
                      mdl_metal_buf qval, mdl_metal_buf out, int batch) {
  Geom geom = {c_img, h, w, words_per_row(w), 0, 0, 0, channels, batch, 0};
  return dispatch(@"mdl_k_preproc", channels * h * geom.wpr * batch,
                  ^(id<MTLComputeCommandEncoder> e) {
    [e setBuffer:MDL_BUF(image) offset:MDL_OFF(image) atIndex:0];
    [e setBuffer:MDL_BUF(out) offset:MDL_OFF(out) atIndex:1];
    [e setBuffer:MDL_BUF(desc) offset:MDL_OFF(desc) atIndex:2];
    [e setBuffer:MDL_BUF(thr) offset:MDL_OFF(thr) atIndex:3];
    [e setBuffer:MDL_BUF(qval) offset:MDL_OFF(qval) atIndex:4];
    [e setBytes:&geom length:sizeof(geom) atIndex:5];
  });
}

int mdl_metal_conv_init(const mdl_metal_conv_params *p,
                        mdl_metal_buf scratch, int batch) {
  /* 定数平面(0 と 全1)を batch 枚ぶん埋める。 */
  mdl_metal_conv_params params = *p;
  const int total = params.tap_rows_words * batch;
  return dispatch(@"mdl_k_conv_const", total,
                  ^(id<MTLComputeCommandEncoder> e) {
    [e setBuffer:MDL_BUF(scratch) offset:MDL_OFF(scratch) atIndex:0];
    [e setBytes:&params length:sizeof(params) atIndex:1];
    [e setBytes:&total length:sizeof(total) atIndex:2];
  });
}

int mdl_metal_conv(const mdl_metal_conv_params *p, mdl_metal_buf in,
                   mdl_metal_buf scratch, mdl_metal_buf out,
                   mdl_metal_buf plane_c, mdl_metal_buf plane_dy,
                   mdl_metal_buf plane_dx, mdl_metal_buf instr,
                   mdl_metal_buf oc_src, mdl_metal_buf wstart,
                   mdl_metal_buf woc, int batch) {
  /* バッチはここで並列度そのものになる。conv_exec のスレッド数は
     画素数/32 しかなく(深い層で16〜32本)、単枚ではGPUが空回りしていた。 */
  mdl_metal_conv_params params = *p;
  const int planes_total = params.n_planes * params.plane_stride * batch;
  const int exec_total = params.tap_rows_words * batch;
  if (params.n_planes > 0 &&
      dispatch(@"mdl_k_conv_planes", planes_total,
               ^(id<MTLComputeCommandEncoder> e) {
        [e setBuffer:MDL_BUF(in) offset:MDL_OFF(in) atIndex:0];
        [e setBuffer:MDL_BUF(scratch) offset:MDL_OFF(scratch) atIndex:1];
        [e setBytes:&params length:sizeof(params) atIndex:2];
        [e setBuffer:MDL_BUF(plane_c) offset:MDL_OFF(plane_c) atIndex:3];
        [e setBuffer:MDL_BUF(plane_dy) offset:MDL_OFF(plane_dy) atIndex:4];
        [e setBuffer:MDL_BUF(plane_dx) offset:MDL_OFF(plane_dx) atIndex:5];
        [e setBytes:&planes_total length:sizeof(planes_total) atIndex:6];
      }) != 0) {
    return -1;
  }
  return dispatch(@"mdl_k_conv_exec", exec_total,
                  ^(id<MTLComputeCommandEncoder> e) {
    [e setBuffer:MDL_BUF(scratch) offset:MDL_OFF(scratch) atIndex:0];
    [e setBuffer:MDL_BUF(out) offset:MDL_OFF(out) atIndex:1];
    [e setBytes:&params length:sizeof(params) atIndex:2];
    [e setBuffer:MDL_BUF(instr) offset:MDL_OFF(instr) atIndex:3];
    [e setBuffer:MDL_BUF(oc_src) offset:MDL_OFF(oc_src) atIndex:4];
    [e setBuffer:MDL_BUF(wstart) offset:MDL_OFF(wstart) atIndex:5];
    [e setBuffer:MDL_BUF(woc) offset:MDL_OFF(woc) atIndex:6];
    [e setBytes:&exec_total length:sizeof(exec_total) atIndex:7];
  });
}

/* ocごとの命令列分割版。exec_total は combined 版の
 * `tap_rows_words * batch` から `out_channels` 倍に増える
 * (= 並列度そのもの。深い層でALU使用率0.41%だった主因が、
 * 律速段のスレッド数不足だったため)。タップ平面プールの構築
 * (mdl_k_conv_planes)は combined 版と完全に同じ関数を使い回す
 * (幾何・タップは分割の影響を受けない共有データ)。 */
int mdl_metal_conv_planes(const mdl_metal_conv_params *p, mdl_metal_buf in,
                          mdl_metal_buf scratch, mdl_metal_buf plane_c,
                          mdl_metal_buf plane_dy, mdl_metal_buf plane_dx,
                          int batch) {
  mdl_metal_conv_params params = *p;
  const int planes_total = params.n_planes * params.plane_stride * batch;
  if (params.n_planes <= 0) return 0;
  return dispatch(@"mdl_k_conv_planes", planes_total,
                  ^(id<MTLComputeCommandEncoder> e) {
    [e setBuffer:MDL_BUF(in) offset:MDL_OFF(in) atIndex:0];
    [e setBuffer:MDL_BUF(scratch) offset:MDL_OFF(scratch) atIndex:1];
    [e setBytes:&params length:sizeof(params) atIndex:2];
    [e setBuffer:MDL_BUF(plane_c) offset:MDL_OFF(plane_c) atIndex:3];
    [e setBuffer:MDL_BUF(plane_dy) offset:MDL_OFF(plane_dy) atIndex:4];
    [e setBuffer:MDL_BUF(plane_dx) offset:MDL_OFF(plane_dx) atIndex:5];
    [e setBytes:&planes_total length:sizeof(planes_total) atIndex:6];
  });
}

int mdl_metal_conv_split(const mdl_metal_conv_params *p, mdl_metal_buf in,
                         mdl_metal_buf scratch, mdl_metal_buf out,
                         mdl_metal_buf plane_c, mdl_metal_buf plane_dy,
                         mdl_metal_buf plane_dx, mdl_metal_buf oc_instr_start,
                         mdl_metal_buf instr, mdl_metal_buf oc_final_ref,
                         int batch) {
  mdl_metal_conv_params params = *p;
  const int exec_total = params.out_channels * params.tap_rows_words * batch;
  if (mdl_metal_conv_planes(p, in, scratch, plane_c, plane_dy, plane_dx,
                            batch) != 0) {
    return -1;
  }
  return dispatch(@"mdl_k_conv_exec_split", exec_total,
                  ^(id<MTLComputeCommandEncoder> e) {
    [e setBuffer:MDL_BUF(scratch) offset:MDL_OFF(scratch) atIndex:0];
    [e setBuffer:MDL_BUF(out) offset:MDL_OFF(out) atIndex:1];
    [e setBytes:&params length:sizeof(params) atIndex:2];
    [e setBuffer:MDL_BUF(oc_instr_start) offset:MDL_OFF(oc_instr_start)
           atIndex:3];
    [e setBuffer:MDL_BUF(instr) offset:MDL_OFF(instr) atIndex:4];
    [e setBuffer:MDL_BUF(oc_final_ref) offset:MDL_OFF(oc_final_ref) atIndex:5];
    [e setBytes:&exec_total length:sizeof(exec_total) atIndex:6];
  });
}

/* モデル特化コード生成。名前で指定したカーネルを
 * (scratch, out, total) の3引数だけで起動する。ttとoc構造は生成時に
 * 直線コードへ焼き込み済みなので、命令列バッファもconv_paramsも要らない。 */
int mdl_metal_conv_exec_named(const char *kernel_name, mdl_metal_buf scratch,
                              mdl_metal_buf out, int total) {
  NSString *name = [NSString stringWithUTF8String:kernel_name];
  const int32_t total_i = total;
  return dispatch(name, total, ^(id<MTLComputeCommandEncoder> e) {
    [e setBuffer:MDL_BUF(scratch) offset:MDL_OFF(scratch) atIndex:0];
    [e setBuffer:MDL_BUF(out) offset:MDL_OFF(out) atIndex:1];
    [e setBytes:&total_i length:sizeof(total_i) atIndex:2];
  });
}

int mdl_metal_or_pool(mdl_metal_buf in, mdl_metal_buf out, int c, int h,
                      int w) {
  const int h_out = ((h - 2) >> 1) + 1;
  const int w_out = ((w - 2) >> 1) + 1;
  Geom geom = {c, h, w, words_per_row(w), h_out, w_out, words_per_row(w_out), 0, 1, 0};
  return dispatch(@"mdl_k_or_pool", c * h_out * geom.wpr_out,
                  ^(id<MTLComputeCommandEncoder> e) {
    [e setBuffer:MDL_BUF(in) offset:MDL_OFF(in) atIndex:0];
    [e setBuffer:MDL_BUF(out) offset:MDL_OFF(out) atIndex:1];
    [e setBytes:&geom length:sizeof(geom) atIndex:2];
  });
}

int mdl_metal_upsample(mdl_metal_buf in, mdl_metal_buf out, int c, int h,
                       int w) {
  Geom geom = {c, h, w, words_per_row(w), 0, 0, words_per_row(2 * w), 0, 1, 0};
  return dispatch(@"mdl_k_upsample", c * (2 * h) * geom.wpr_out,
                  ^(id<MTLComputeCommandEncoder> e) {
    [e setBuffer:MDL_BUF(in) offset:MDL_OFF(in) atIndex:0];
    [e setBuffer:MDL_BUF(out) offset:MDL_OFF(out) atIndex:1];
    [e setBytes:&geom length:sizeof(geom) atIndex:2];
  });
}

int mdl_metal_permute(mdl_metal_buf in, mdl_metal_buf out, mdl_metal_buf order,
                      int c_out, int h, int wpr, int batch) {
  Geom geom = {c_out, h, 0, wpr, 0, 0, 0, 0, batch, 0};
  return dispatch(@"mdl_k_permute", c_out * batch * h * wpr,
                  ^(id<MTLComputeCommandEncoder> e) {
    [e setBuffer:MDL_BUF(in) offset:MDL_OFF(in) atIndex:0];
    [e setBuffer:MDL_BUF(out) offset:MDL_OFF(out) atIndex:1];
    [e setBuffer:MDL_BUF(order) offset:MDL_OFF(order) atIndex:2];
    [e setBytes:&geom length:sizeof(geom) atIndex:3];
  });
}

int mdl_metal_spatial_group_sum(mdl_metal_buf in, mdl_metal_buf scores,
                                int n_out, int gs, int h, int w) {
  int n_planes = 0, v = gs;
  const int wpr = words_per_row(w);
  const int total = n_out * h * wpr;
  while (v > 0) { n_planes++; v >>= 1; }
  {
    const int r = mdl_gpu_pick_r(total, gs, kMdlGsTargetThreads, kMdlGsMaxR);
    if (r <= 1) {
      Geom geom = {n_out, h, w, wpr, n_planes, 0, 0, gs, 1, 0};
      return dispatch(@"mdl_k_sgs", total,
                      ^(id<MTLComputeCommandEncoder> e) {
        [e setBuffer:MDL_BUF(in) offset:MDL_OFF(in) atIndex:0];
        [e setBuffer:MDL_BUF(scores) offset:MDL_OFF(scores) atIndex:1];
        [e setBytes:&geom length:sizeof(geom) atIndex:2];
      });
    }
    /* 並列縮約版。g.c=gs, g.h_out=n_planes(チャンク側),
     * g.w_out=chunk, g.wpr_out=R(mdl_k_sgs_rのコメント参照)。
     * dispatch1回=1出力ワードのスレッドグループなので by_groups で
     * 明示的にグループ境界を揃える。 */
    {
      const int chunk = (gs + r - 1) / r;
      int n_planes_chunk = 0;
      v = chunk;
      while (v > 0) { n_planes_chunk++; v >>= 1; }
      {
        Geom geom = {gs, h, w, wpr, n_planes_chunk, chunk, r, 0, 1, 0};
        return dispatch(@"mdl_k_sgs_r", total,
                        ^(id<MTLComputeCommandEncoder> e) {
          [e setBuffer:MDL_BUF(in) offset:MDL_OFF(in) atIndex:0];
          [e setBuffer:MDL_BUF(scores) offset:MDL_OFF(scores) atIndex:1];
          [e setBytes:&geom length:sizeof(geom) atIndex:2];
        }, /*explicit_tg=*/r, /*tg_mem_bytes=*/r * 32 * (int)sizeof(int32_t),
           /*by_groups=*/true);
      }
    }
  }
}

int mdl_metal_lut_layer(mdl_metal_buf x, mdl_metal_buf out, int n_inputs,
                        int out_width, mdl_metal_buf indices,
                        mdl_metal_buf tt, int in_width, int batch) {
  Geom geom = {n_inputs, 0, out_width, 0, 0, 0, 0, 0, batch, in_width};
  return dispatch(@"mdl_k_lut_layer", out_width * batch,
                  ^(id<MTLComputeCommandEncoder> e) {
    [e setBuffer:MDL_BUF(x) offset:MDL_OFF(x) atIndex:0];
    [e setBuffer:MDL_BUF(out) offset:MDL_OFF(out) atIndex:1];
    [e setBuffer:MDL_BUF(indices) offset:MDL_OFF(indices) atIndex:2];
    [e setBuffer:MDL_BUF(tt) offset:MDL_OFF(tt) atIndex:3];
    [e setBytes:&geom length:sizeof(geom) atIndex:4];
  });
}

/* 添字uint16_t版。ホスト側の呼び出し文はindices型がバッファの
   中身(ushort)に変わるだけで、mdl_metal_buf自体は不透明ハンドルなので
   関数の形は同一。 */
int mdl_metal_lut_layer_u16(mdl_metal_buf x, mdl_metal_buf out, int n_inputs,
                            int out_width, mdl_metal_buf indices,
                            mdl_metal_buf tt, int in_width, int batch) {
  Geom geom = {n_inputs, 0, out_width, 0, 0, 0, 0, 0, batch, in_width};
  return dispatch(@"mdl_k_lut_layer_u16", out_width * batch,
                  ^(id<MTLComputeCommandEncoder> e) {
    [e setBuffer:MDL_BUF(x) offset:MDL_OFF(x) atIndex:0];
    [e setBuffer:MDL_BUF(out) offset:MDL_OFF(out) atIndex:1];
    [e setBuffer:MDL_BUF(indices) offset:MDL_OFF(indices) atIndex:2];
    [e setBuffer:MDL_BUF(tt) offset:MDL_OFF(tt) atIndex:3];
    [e setBytes:&geom length:sizeof(geom) atIndex:4];
  });
}

/* dense多層融合カーネル(GEM(DAC2025) Boomerang型)。名前で指定した
 * カーネル(`deploy/densefuse.py`が生成、`mdl_k49_fused_*`)を、
 * 1スレッドグループ=1サンプル(グループ数=batch)で起動する。
 * ttと添字表は従来どおりconstantバッファ(idx_bufs/tt_bufs、n_layers本ずつ)
 * のまま、活性だけが[[threadgroup(0)]]/[[threadgroup(1)]]のping-pongに
 * 乗る(2本ともact_bytesぶん)。
 *
 * 最小実験(har_dlgn_k6000_gumbel、5セット中央値、GPU busy時間、
 * ir.simulateとの整数厳密一致を確認済み): batch=1で0.81x(1グループしか
 * 立たず占有率が下がるため悪化)、batch=32で1.91x、batch=128で2.51x
 * (最良)、batch=256で2.30x。`densefuse.py`のdocstringに詳細。 */
int mdl_metal_dense_fuse_exec(const char *kernel_name, mdl_metal_buf x,
                              mdl_metal_buf out, mdl_metal_buf *idx_bufs,
                              mdl_metal_buf *tt_bufs, int n_layers,
                              int act_bytes, int batch) {
  NSString *name = [NSString stringWithUTF8String:kernel_name];
  id<MTLComputePipelineState> pso = g.pso[name];
  if (pso == nil) {
    NSLog(@"mdl_metal: パイプラインが無い: %@", name);
    return -1;
  }
  {
    NSUInteger tg = pso.maxTotalThreadsPerThreadgroup;
    if (tg > 1024) tg = 1024;
    return dispatch(name, batch, ^(id<MTLComputeCommandEncoder> e) {
      [e setBuffer:MDL_BUF(x) offset:MDL_OFF(x) atIndex:0];
      [e setBuffer:MDL_BUF(out) offset:MDL_OFF(out) atIndex:1];
      for (int L = 0; L < n_layers; L++) {
        [e setBuffer:MDL_BUF(idx_bufs[L]) offset:MDL_OFF(idx_bufs[L])
               atIndex:2 + 2 * L];
        [e setBuffer:MDL_BUF(tt_bufs[L]) offset:MDL_OFF(tt_bufs[L])
               atIndex:3 + 2 * L];
      }
    }, /*explicit_tg=*/(int)tg, /*tg_mem_bytes=*/act_bytes,
       /*by_groups=*/true, /*tg_mem_bytes2=*/act_bytes);
  }
}

/* ---- lut_flat ----
 * カーネル本体は mdl_metal_entry.metal(mdl_k_lut_flat_*)。式は
 * mdl_gpu_kernels.h の mdl_g_lut_flat_*。表(tt/a/b/o/in_idx/level_off)は
 * `deploy/flatlevels.py` が段ごとに並べ替えたもの。 */
struct FlatParams {
  int n_in, in_width, sig_stride, g0, n, out_width, n_levels, batch;
};  /* mdl_metal_entry.metal の MdlFlatParams と同一 */

int mdl_metal_lut_flat_init(mdl_metal_buf x, mdl_metal_buf sig,
                            mdl_metal_buf in_idx, int n_in, int in_width,
                            int sig_stride, int batch) {
  FlatParams p = {n_in, in_width, sig_stride, 0, 0, 0, 0, batch};
  return dispatch(@"mdl_k_lut_flat_init", (n_in + 1) * batch,
                  ^(id<MTLComputeCommandEncoder> e) {
    [e setBuffer:MDL_BUF(x) offset:MDL_OFF(x) atIndex:0];
    [e setBuffer:MDL_BUF(sig) offset:MDL_OFF(sig) atIndex:1];
    [e setBuffer:MDL_BUF(in_idx) offset:MDL_OFF(in_idx) atIndex:2];
    [e setBytes:&p length:sizeof(p) atIndex:3];
  });
}

int mdl_metal_lut_flat_level(mdl_metal_buf sig, mdl_metal_buf tt,
                             mdl_metal_buf a, mdl_metal_buf b, int n_in,
                             int sig_stride, int g0, int n, int batch) {
  FlatParams p = {n_in, 0, sig_stride, g0, n, 0, 0, batch};
  return dispatch(@"mdl_k_lut_flat_level", n * batch,
                  ^(id<MTLComputeCommandEncoder> e) {
    [e setBuffer:MDL_BUF(sig) offset:MDL_OFF(sig) atIndex:0];
    [e setBuffer:MDL_BUF(tt) offset:MDL_OFF(tt) atIndex:1];
    [e setBuffer:MDL_BUF(a) offset:MDL_OFF(a) atIndex:2];
    [e setBuffer:MDL_BUF(b) offset:MDL_OFF(b) atIndex:3];
    [e setBytes:&p length:sizeof(p) atIndex:4];
  });
}

int mdl_metal_lut_flat_out(mdl_metal_buf sig, mdl_metal_buf out,
                           mdl_metal_buf o, int out_width, int sig_stride,
                           int batch) {
  FlatParams p = {0, 0, sig_stride, 0, 0, out_width, 0, batch};
  return dispatch(@"mdl_k_lut_flat_out", out_width * batch,
                  ^(id<MTLComputeCommandEncoder> e) {
    [e setBuffer:MDL_BUF(sig) offset:MDL_OFF(sig) atIndex:0];
    [e setBuffer:MDL_BUF(out) offset:MDL_OFF(out) atIndex:1];
    [e setBuffer:MDL_BUF(o) offset:MDL_OFF(o) atIndex:2];
    [e setBytes:&p length:sizeof(p) atIndex:3];
  });
}

/* バッチ軸パック版。bwords=ceil(batch/32) を sig_stride に載せて渡す。*/
int mdl_metal_lut_flat_pk_init(mdl_metal_buf x, mdl_metal_buf sig,
                               mdl_metal_buf in_idx, int n_in, int in_width,
                               int batch) {
  const int bw = (batch + 31) >> 5;
  FlatParams p = {n_in, in_width, bw, 0, 0, 0, 0, batch};
  return dispatch(@"mdl_k_lut_flat_pk_init", (n_in + 1) * bw,
                  ^(id<MTLComputeCommandEncoder> e) {
    [e setBuffer:MDL_BUF(x) offset:MDL_OFF(x) atIndex:0];
    [e setBuffer:MDL_BUF(sig) offset:MDL_OFF(sig) atIndex:1];
    [e setBuffer:MDL_BUF(in_idx) offset:MDL_OFF(in_idx) atIndex:2];
    [e setBytes:&p length:sizeof(p) atIndex:3];
  });
}

int mdl_metal_lut_flat_pk_level(mdl_metal_buf sig, mdl_metal_buf tt,
                                mdl_metal_buf a, mdl_metal_buf b, int n_in,
                                int g0, int n, int batch) {
  const int bw = (batch + 31) >> 5;
  FlatParams p = {n_in, 0, bw, g0, n, 0, 0, batch};
  return dispatch(@"mdl_k_lut_flat_pk_level", n * bw,
                  ^(id<MTLComputeCommandEncoder> e) {
    [e setBuffer:MDL_BUF(sig) offset:MDL_OFF(sig) atIndex:0];
    [e setBuffer:MDL_BUF(tt) offset:MDL_OFF(tt) atIndex:1];
    [e setBuffer:MDL_BUF(a) offset:MDL_OFF(a) atIndex:2];
    [e setBuffer:MDL_BUF(b) offset:MDL_OFF(b) atIndex:3];
    [e setBytes:&p length:sizeof(p) atIndex:4];
  });
}

int mdl_metal_lut_flat_pk_out(mdl_metal_buf sig, mdl_metal_buf out,
                              mdl_metal_buf o, int out_width, int batch) {
  const int bw = (batch + 31) >> 5;
  FlatParams p = {0, 0, bw, 0, 0, out_width, 0, batch};
  return dispatch(@"mdl_k_lut_flat_pk_out", out_width * batch,
                  ^(id<MTLComputeCommandEncoder> e) {
    [e setBuffer:MDL_BUF(sig) offset:MDL_OFF(sig) atIndex:0];
    [e setBuffer:MDL_BUF(out) offset:MDL_OFF(out) atIndex:1];
    [e setBuffer:MDL_BUF(o) offset:MDL_OFF(o) atIndex:2];
    [e setBytes:&p length:sizeof(p) atIndex:3];
  });
}

int mdl_metal_lut_flat_fused(mdl_metal_buf x, mdl_metal_buf out,
                             mdl_metal_buf tt, mdl_metal_buf a,
                             mdl_metal_buf b, mdl_metal_buf o,
                             mdl_metal_buf in_idx, mdl_metal_buf level_off,
                             int n_in, int in_width, int out_width,
                             int n_levels, int sig_bytes, int batch) {
  FlatParams p = {n_in, in_width, 0, 0, 0, out_width, n_levels, batch};
  id<MTLComputePipelineState> pso = g.pso[@"mdl_k_lut_flat_fused"];
  if (pso == nil) {
    NSLog(@"mdl_metal: パイプラインが無い: mdl_k_lut_flat_fused");
    return -1;
  }
  {
    NSUInteger tg = pso.maxTotalThreadsPerThreadgroup;
    if (tg > 1024) tg = 1024;
    /* threadgroupメモリ長は16バイト単位に切り上げる(Metalの要件) */
    const int tg_bytes = (sig_bytes + 15) & ~15;
    return dispatch(@"mdl_k_lut_flat_fused", batch,
                    ^(id<MTLComputeCommandEncoder> e) {
      [e setBuffer:MDL_BUF(x) offset:MDL_OFF(x) atIndex:0];
      [e setBuffer:MDL_BUF(out) offset:MDL_OFF(out) atIndex:1];
      [e setBuffer:MDL_BUF(tt) offset:MDL_OFF(tt) atIndex:2];
      [e setBuffer:MDL_BUF(a) offset:MDL_OFF(a) atIndex:3];
      [e setBuffer:MDL_BUF(b) offset:MDL_OFF(b) atIndex:4];
      [e setBuffer:MDL_BUF(o) offset:MDL_OFF(o) atIndex:5];
      [e setBuffer:MDL_BUF(in_idx) offset:MDL_OFF(in_idx) atIndex:6];
      [e setBuffer:MDL_BUF(level_off) offset:MDL_OFF(level_off) atIndex:7];
      [e setBytes:&p length:sizeof(p) atIndex:8];
    }, /*explicit_tg=*/(int)tg, /*tg_mem_bytes=*/tg_bytes, /*by_groups=*/true);
  }
}

int mdl_metal_group_sum(mdl_metal_buf x, mdl_metal_buf out, int in_width,
                        int k) {
  const int gsz = in_width / k;
  const int r = mdl_gpu_pick_r(k, gsz, kMdlGsTargetThreads, kMdlGsMaxR);
  if (r <= 1) {
    Geom geom = {k, 0, gsz, 0, 0, 0, 0, 0, 1, 0};
    return dispatch(@"mdl_k_group_sum", k, ^(id<MTLComputeCommandEncoder> e) {
      [e setBuffer:MDL_BUF(x) offset:MDL_OFF(x) atIndex:0];
      [e setBuffer:MDL_BUF(out) offset:MDL_OFF(out) atIndex:1];
      [e setBytes:&geom length:sizeof(geom) atIndex:2];
    });
  }
  /* 並列縮約版。g.c=k(=グリッド側、グループ数), g.w=gsz, g.h=R
   * (mdl_k_group_sum_rのコメント参照)。dispatch1回=1出力グループの
   * スレッドグループなので by_groups で明示的にグループ境界を揃える。 */
  {
    Geom geom = {k, r, gsz, 0, 0, 0, 0, 0, 1, 0};
    return dispatch(@"mdl_k_group_sum_r", k,
                    ^(id<MTLComputeCommandEncoder> e) {
      [e setBuffer:MDL_BUF(x) offset:MDL_OFF(x) atIndex:0];
      [e setBuffer:MDL_BUF(out) offset:MDL_OFF(out) atIndex:1];
      [e setBytes:&geom length:sizeof(geom) atIndex:2];
    }, /*explicit_tg=*/r, /*tg_mem_bytes=*/r * (int)sizeof(int32_t),
       /*by_groups=*/true);
  }
}

/* ---- dense のバッチ軸ビットプレーン化 ----
 * カーネル本体は mdl_metal_entry.metal(mdl_k_pack_batch/
 * mdl_k_lut_layer_packed[_u16]/mdl_k_group_sum_packed)。ここは
 * 既存のlut_layer/group_sumと同じ「Geomを詰めてdispatch」の薄いラッパ。
 * `batch` はこの呼び出しでの実行時サンプル数(既存カーネルの`batch`引数と
 * 同じ意味、確保上限ではなく実行時 n)。 */

int mdl_metal_pack_batch(mdl_metal_buf x, mdl_metal_buf out, int width,
                         int batch) {
  const int bwords = (batch + 31) >> 5;
  Geom geom = {0, 0, width, bwords, 0, 0, 0, batch, 1, 0};
  return dispatch(@"mdl_k_pack_batch", width * bwords,
                  ^(id<MTLComputeCommandEncoder> e) {
    [e setBuffer:MDL_BUF(x) offset:MDL_OFF(x) atIndex:0];
    [e setBuffer:MDL_BUF(out) offset:MDL_OFF(out) atIndex:1];
    [e setBytes:&geom length:sizeof(geom) atIndex:2];
  });
}

int mdl_metal_lut_layer_packed(mdl_metal_buf x, mdl_metal_buf out,
                               int n_inputs, int out_width,
                               mdl_metal_buf indices, mdl_metal_buf tt,
                               int batch) {
  const int bwords = (batch + 31) >> 5;
  Geom geom = {n_inputs, 0, out_width, 0, 0, 0, 0, bwords, 1, 0};
  return dispatch(@"mdl_k_lut_layer_packed", out_width * bwords,
                  ^(id<MTLComputeCommandEncoder> e) {
    [e setBuffer:MDL_BUF(x) offset:MDL_OFF(x) atIndex:0];
    [e setBuffer:MDL_BUF(out) offset:MDL_OFF(out) atIndex:1];
    [e setBuffer:MDL_BUF(indices) offset:MDL_OFF(indices) atIndex:2];
    [e setBuffer:MDL_BUF(tt) offset:MDL_OFF(tt) atIndex:3];
    [e setBytes:&geom length:sizeof(geom) atIndex:4];
  });
}

/* 添字uint16_t版(同じ理由)。*/
int mdl_metal_lut_layer_packed_u16(mdl_metal_buf x, mdl_metal_buf out,
                                   int n_inputs, int out_width,
                                   mdl_metal_buf indices, mdl_metal_buf tt,
                                   int batch) {
  const int bwords = (batch + 31) >> 5;
  Geom geom = {n_inputs, 0, out_width, 0, 0, 0, 0, bwords, 1, 0};
  return dispatch(@"mdl_k_lut_layer_packed_u16", out_width * bwords,
                  ^(id<MTLComputeCommandEncoder> e) {
    [e setBuffer:MDL_BUF(x) offset:MDL_OFF(x) atIndex:0];
    [e setBuffer:MDL_BUF(out) offset:MDL_OFF(out) atIndex:1];
    [e setBuffer:MDL_BUF(indices) offset:MDL_OFF(indices) atIndex:2];
    [e setBuffer:MDL_BUF(tt) offset:MDL_OFF(tt) atIndex:3];
    [e setBytes:&geom length:sizeof(geom) atIndex:4];
  });
}

/* group_sum のバッチ軸パック版。出力は非パック版と同じint32レイアウト
 * (scores[sample*k+g])に直接書くので、この後段にアンパックは要らない。
 *
 * 並列度不足対策(同じ`mdl_gpu_pick_r`判定): work_items=
 * k*bwordsが小さいモデル(har_dlgn_k600、batch=128でk*bwords=24)では
 * 実測でこのカーネルがGPU時間の78%を占めていた。既存のgroup_sum/
 * spatial_group_sumと同じ閾値判定で、並列度が既に十分なら(r<=1)
 * 従来のmdl_k_group_sum_packedのまま(変更ゼロ)。 */
int mdl_metal_group_sum_packed(mdl_metal_buf x, mdl_metal_buf out,
                               int in_width, int k, int batch) {
  const int gsz = in_width / k;
  const int bwords = (batch + 31) >> 5;
  int n_planes = 0, v = gsz;
  while (v > 0) { n_planes++; v >>= 1; }
  if (n_planes == 0) n_planes = 1;
  {
    const int r = mdl_gpu_pick_r(k * bwords, gsz, kMdlGsTargetThreads,
                                 kMdlGsMaxR);
    if (r <= 1) {
      Geom geom = {k, 0, gsz, 0, n_planes, 0, 0, bwords, batch, 0};
      return dispatch(@"mdl_k_group_sum_packed", k * bwords,
                      ^(id<MTLComputeCommandEncoder> e) {
        [e setBuffer:MDL_BUF(x) offset:MDL_OFF(x) atIndex:0];
        [e setBuffer:MDL_BUF(out) offset:MDL_OFF(out) atIndex:1];
        [e setBytes:&geom length:sizeof(geom) atIndex:2];
      });
    }
    /* 並列縮約版。g.c=gsz, g.h=k, g.h_out=n_planes_chunk,
     * g.w_out=chunk, g.wpr_out=R(mdl_k_group_sum_packed_rのコメント参照)。
     * dispatch1回=1出力ワードのスレッドグループなのでby_groupsで
     * 明示的にグループ境界を揃える。 */
    {
      const int chunk = (gsz + r - 1) / r;
      int n_planes_chunk = 0;
      v = chunk;
      while (v > 0) { n_planes_chunk++; v >>= 1; }
      if (n_planes_chunk == 0) n_planes_chunk = 1;
      {
        Geom geom = {gsz, k, 0, 0, n_planes_chunk, chunk, r, bwords, batch, 0};
        return dispatch(@"mdl_k_group_sum_packed_r", k * bwords,
                        ^(id<MTLComputeCommandEncoder> e) {
          [e setBuffer:MDL_BUF(x) offset:MDL_OFF(x) atIndex:0];
          [e setBuffer:MDL_BUF(out) offset:MDL_OFF(out) atIndex:1];
          [e setBytes:&geom length:sizeof(geom) atIndex:2];
        }, /*explicit_tg=*/r, /*tg_mem_bytes=*/r * 32 * (int)sizeof(int32_t),
           /*by_groups=*/true);
      }
    }
  }
}

}  /* extern "C" */
