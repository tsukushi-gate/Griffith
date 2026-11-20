#!/usr/bin/env python
"""最小実験: conv_exec_split の(a)現行インタプリタ vs (b)tt即値+
slots静的展開の直線コード、をMetalマイクロベンチで比較する .mm を生成する。

対象は実モデル(既定 seg_convdlgn_k32)の実際の conv2d ノード1つぶんの
命令列(`deploy/layout.py::build_conv_layout_split` の出力そのもの)。
tt即値化の効果はモデルに依存しない普遍的な最適化なので、シミュレート値
ではなく実命令列を使う(教訓: 展開の善し悪しはノードの命令数分布に
依存するため、実測でしか判断できない)。

(b)側は `deploy/cbackend.py::GATE_EXPR`(CPU直線化テーブル)をそのまま
流用し、ocごとに:
  - tt をコンパイル時定数として式へ畳み込む(9→最大3 LOP3系/ゲート)
  - スロットをスレッドローカル配列でなく個別のレジスタ変数にする
    (動的添字が消える → local memory落ちが消える)
ocの選択は switch(oc) 一発(同一SIMDグループ内はiが最内なのでocが揃う
= 分岐は事実上warp一様)。

出力bit一致検査は本体(main)で必須(switch/mask前例に倣う)。

ビルド・実行はこのスクリプトが自動で行う(clang++が必要、macOSのみ)。
    uv run python scripts/gpu_calib/gen_conv_codegen_ab.py
    uv run python scripts/gpu_calib/gen_conv_codegen_ab.py \
        --model seg_convdlgn_k32 --node n1_lutconv2d --batch 512
"""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from griffith_logic.deploy import build_bundle  # noqa: E402
from griffith_logic.deploy.cbackend import GATE_EXPR  # noqa: E402
from griffith_logic.deploy.layout import (  # noqa: E402
    build_conv_layout_split,
    conv_layout_split_stats,
)
from griffith_logic.ir.serialize import ir_from_dict  # noqa: E402

MODELS_DIR = REPO / 'tools' / 'webbench' / 'public' / 'models'


def load_node_layout(model: str, node_id: str | None):
    path = MODELS_DIR / f'{model}.json'
    doc = json.loads(path.read_text(encoding='utf-8'))
    bundle = build_bundle(doc)
    graph = ir_from_dict(bundle['ir'])
    conv_nodes = [n for n in graph.nodes if n.type == 'conv2d']
    if not conv_nodes:
        raise SystemExit(f'{model}: conv2dノードが無い')
    if node_id is None:
        node = conv_nodes[0]
    else:
        matches = [n for n in conv_nodes if n.id == node_id]
        if not matches:
            ids = ', '.join(n.id for n in conv_nodes)
            raise SystemExit(f'{model}: node_id={node_id} が無い(候補: {ids})')
        node = matches[0]
    prog_dict = bundle['programs'][node.id]
    layout = build_conv_layout_split(node.params, prog_dict)
    stats = conv_layout_split_stats(prog_dict, layout)
    return node.id, layout, stats


def ref_expr(packed: int, trw: int) -> str:
    if packed & 1:
        return f's{packed >> 1}'
    off = packed >> 1
    if off == 0:
        return '0u'
    if off == trw:
        return '0xffffffffu'
    return f'sc[{off} + i]'


def gen_oc_function(oc: int, instrs4: list[int], final_ref: int, trw: int) -> str:
    """1ocぶんの直線コード関数を生成する。instrs4は [tt,a,b,dst]*n_local。"""
    lines = []
    slots_used: set[int] = set()
    n_local = len(instrs4) // 4
    for k in range(n_local):
        tt, ap, bp, dst = instrs4[4 * k:4 * k + 4]
        a = ref_expr(ap, trw)
        b = ref_expr(bp, trw)
        expr = GATE_EXPR[tt & 0xf].format(a=a, b=b)
        lines.append(f'  s{dst} = {expr};')
        slots_used.add(dst)
    final_expr = ref_expr(final_ref, trw)
    decl = ('  uint32_t ' + ', '.join(f's{k}' for k in sorted(slots_used)) + ';\n'
            if slots_used else '')
    body = '\n'.join(lines)
    return (f'static inline uint32_t conv_oc_codegen_{oc}'
            f'(device const uint *sc, int i) {{\n'
            f'{decl}{body}\n'
            f'  return {final_expr};\n'
            f'}}')


def build_codegen_source(layout, tag: str = 'n0') -> tuple[str, int]:
    """codegenカーネル一式のMSLソースと生成行数を返す。"""
    trw = layout.tap_rows_words
    oc_fns = []
    for oc in range(layout.out_channels):
        lo = layout.oc_instr_start[oc] * 4
        hi = layout.oc_instr_start[oc + 1] * 4
        instrs4 = layout.instr_flat[lo:hi]
        final_ref = layout.oc_final_ref[oc]
        oc_fns.append(gen_oc_function(oc, instrs4, final_ref, trw))
    cases = '\n'.join(
        f'    case {oc}: v = conv_oc_codegen_{oc}(scratch, i); break;'
        for oc in range(layout.out_channels))
    src = '\n\n'.join(oc_fns) + f'''

kernel void mdl_calib_conv_codegen(
    device const uint *scratch [[buffer(0)]],
    device uint *out [[buffer(1)]],
    constant int &tap_rows_words [[buffer(2)]],
    constant int &out_channels [[buffer(3)]],
    constant uint &mask_out [[buffer(4)]],
    constant int &wpr_out [[buffer(5)]],
    constant int &total [[buffer(6)]],
    uint gid [[thread_position_in_grid]]) {{
  if ((int)gid >= total) return;
  const int w = tap_rows_words;
  const int per_n = out_channels * w;
  const int n = (int)gid / per_n;
  const int rem = (int)gid - n * per_n;
  const int oc = rem / w;
  const int i = rem - oc * w;
  const uint tail = ((i % wpr_out) == wpr_out - 1) ? mask_out : 0xffffffffu;
  uint32_t v = 0u;
  switch (oc) {{
{cases}
    default: v = 0u; break;
  }}
  out[(uint)n * (uint)per_n + (uint)oc * (uint)w + (uint)i] = v & tail;
}}
'''
    n_lines = src.count('\n')
    return src, n_lines


INTERP_KERNEL_TMPL = '''
#define MDL_CALIB_MAX_SLOTS 8

kernel void mdl_calib_conv_interp(
    device const uint *scratch [[buffer(0)]],
    device uint *out [[buffer(1)]],
    constant int *oc_instr_start [[buffer(2)]],
    constant int *instr [[buffer(3)]],
    constant int *oc_final_ref [[buffer(4)]],
    constant int &tap_rows_words [[buffer(5)]],
    constant int &out_channels [[buffer(6)]],
    constant uint &mask_out [[buffer(7)]],
    constant int &wpr_out [[buffer(8)]],
    constant int &total [[buffer(9)]],
    uint gid [[thread_position_in_grid]]) {
  if ((int)gid >= total) return;
  const int w = tap_rows_words;
  const int per_n = out_channels * w;
  const int n = (int)gid / per_n;
  const int rem = (int)gid - n * per_n;
  const int oc = rem / w;
  const int i = rem - oc * w;
  const uint tail = ((i % wpr_out) == wpr_out - 1) ? mask_out : 0xffffffffu;
  thread uint slots[MDL_CALIB_MAX_SLOTS];
  const int start = oc_instr_start[oc];
  const int end = oc_instr_start[oc + 1];
  for (int k = start; k < end; k++) {
    constant int *ins = instr + k * 4;
    const int ap = ins[1];
    const int bp = ins[2];
    const uint av = (ap & 1) ? slots[ap >> 1] : scratch[(ap >> 1) + i];
    const uint bv = (bp & 1) ? slots[bp >> 1] : scratch[(bp >> 1) + i];
    slots[ins[3]] = mdl_g_gate2((uint)ins[0], av, bv);
  }
  {
    const int fr = oc_final_ref[oc];
    const uint v = (fr & 1) ? slots[fr >> 1] : scratch[(fr >> 1) + i];
    out[(uint)n * (uint)per_n + (uint)oc * (uint)w + (uint)i] = v & tail;
  }
}
'''

MM_TMPL = r'''/* 最小実験(自動生成、コミットしない) — conv_exec_split の
 * (a)現行インタプリタ vs (b)tt即値+slots静的展開のMetal A/Bベンチ。
 * 対象: {model} / node={node_id} (out_channels={out_channels},
 * tap_rows_words={trw}, n_instr_total={n_instr}, max_slots={max_slots})
 *
 * ビルド:
 *   clang++ -std=c++17 -O2 -framework Metal -framework Foundation \
 *     {mm_path} -o {bin_path}
 * 実行(リポジトリルートから):
 *   {bin_path} [batch] [trials] [sets]
 */
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace {{

std::string ReadFile(const std::string &path) {{
  std::ifstream f(path);
  if (!f) {{
    std::fprintf(stderr, "共有ヘッダを読めない: %s\n"
                "(リポジトリルートから実行しているか確認すること)\n",
                path.c_str());
    std::exit(1);
  }}
  std::ostringstream ss;
  ss << f.rdbuf();
  return ss.str();
}}

std::string StripIncludeGuard(std::string src) {{
  auto erase_all = [&](const std::string &needle) {{
    size_t pos;
    while ((pos = src.find(needle)) != std::string::npos) {{
      src.erase(pos, needle.size());
    }}
  }};
  erase_all("#ifndef MDL_GPU_KERNELS_H");
  erase_all("#define MDL_GPU_KERNELS_H");
  erase_all("#endif /* MDL_GPU_KERNELS_H */");
  return src;
}}

/* 実ノードのCSRデータ(oc_instr_start/instr_flat/oc_final_ref)。
   これはインタプリタ側だけが実行時に読む(codegen側はPython生成時に
   直線コードへ焼き込み済み)。 */
static const int32_t kOcInstrStart[] = {{{oc_instr_start}}};
static const int32_t kInstrFlat[] = {{{instr_flat}}};
static const int32_t kOcFinalRef[] = {{{oc_final_ref}}};

constexpr int kOutChannels = {out_channels};
constexpr int kTapRowsWords = {trw};
constexpr int kWprOut = {wpr_out};
constexpr uint32_t kMaskOut = {mask_out}u;
constexpr int kScratchWords = {scratch_words};

/* scratch(読み取り専用、全batch共通の1テンプレート)を決定的に初期化する。
   実画像ではなく合成値だが、両カーネルとも**同一の値**を読むので
   命令列の実行結果だけを比較できる(gate2_ab.mm と同じ方針)。
   ただし [0,trw) はCONST-0平面(全0)、[trw,2*trw)はCONST-1平面(全1)
   という実カーネルの不変条件(`mdl_k_conv_const` が保証する)だけは
   合成データでも守る必要がある — codegen側は `cbackend.py` 同様この
   不変条件を前提に off==0/off==trw を即値0u/0xffffffffuへ畳み込む
   最適化をしているため、破ると偽の不一致が出る(実際に一度出た)。 */
std::vector<uint32_t> BuildScratch() {{
  std::vector<uint32_t> sc(kScratchWords);
  for (size_t i = 0; i < sc.size(); i++) {{
    uint32_t v = (uint32_t)i ^ 0x9e3779b9u;
    v ^= v << 13; v ^= v >> 17; v ^= v << 5;
    sc[i] = v;
  }}
  for (int i = 0; i < kTapRowsWords; i++) sc[i] = 0u;
  for (int i = 0; i < kTapRowsWords; i++) sc[kTapRowsWords + i] = 0xffffffffu;
  return sc;
}}

struct RunResult {{
  std::vector<double> elapsed_ms;
  std::vector<uint32_t> out;
}};

RunResult RunKernel(id<MTLDevice> device, id<MTLCommandQueue> queue,
                    id<MTLComputePipelineState> pso, bool is_interp,
                    id<MTLBuffer> scratchBuf, int batch, int trials,
                    int warmups) {{
  const int total = kOutChannels * kTapRowsWords * batch;
  id<MTLBuffer> outBuf =
      [device newBufferWithLength:(size_t)total * 4
                           options:MTLResourceStorageModeShared];
  id<MTLBuffer> ocInstrStartBuf = nil, instrBuf = nil, ocFinalRefBuf = nil;
  if (is_interp) {{
    ocInstrStartBuf = [device newBufferWithBytes:kOcInstrStart
                                          length:sizeof(kOcInstrStart)
                                         options:MTLResourceStorageModeShared];
    instrBuf = [device newBufferWithBytes:kInstrFlat
                                    length:sizeof(kInstrFlat)
                                   options:MTLResourceStorageModeShared];
    ocFinalRefBuf = [device newBufferWithBytes:kOcFinalRef
                                        length:sizeof(kOcFinalRef)
                                       options:MTLResourceStorageModeShared];
  }}
  const int32_t trw_i = kTapRowsWords, oc_i = kOutChannels, wpr_i = kWprOut;
  const uint32_t mask_i = kMaskOut;
  const int32_t total_i = total;

  auto dispatch_once = [&]() {{
    id<MTLCommandBuffer> cmd = [queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:pso];
    [enc setBuffer:scratchBuf offset:0 atIndex:0];
    [enc setBuffer:outBuf offset:0 atIndex:1];
    if (is_interp) {{
      [enc setBuffer:ocInstrStartBuf offset:0 atIndex:2];
      [enc setBuffer:instrBuf offset:0 atIndex:3];
      [enc setBuffer:ocFinalRefBuf offset:0 atIndex:4];
      [enc setBytes:&trw_i length:sizeof(trw_i) atIndex:5];
      [enc setBytes:&oc_i length:sizeof(oc_i) atIndex:6];
      [enc setBytes:&mask_i length:sizeof(mask_i) atIndex:7];
      [enc setBytes:&wpr_i length:sizeof(wpr_i) atIndex:8];
      [enc setBytes:&total_i length:sizeof(total_i) atIndex:9];
    }} else {{
      [enc setBytes:&trw_i length:sizeof(trw_i) atIndex:2];
      [enc setBytes:&oc_i length:sizeof(oc_i) atIndex:3];
      [enc setBytes:&mask_i length:sizeof(mask_i) atIndex:4];
      [enc setBytes:&wpr_i length:sizeof(wpr_i) atIndex:5];
      [enc setBytes:&total_i length:sizeof(total_i) atIndex:6];
    }}
    NSUInteger tg = pso.maxTotalThreadsPerThreadgroup;
    if (tg > (NSUInteger)total) tg = (NSUInteger)total;
    if (tg > 256) tg = 256;
    if (tg < 1) tg = 1;
    [enc dispatchThreads:MTLSizeMake((NSUInteger)total, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];
    return cmd;
  }};

  for (int w = 0; w < warmups; w++) dispatch_once();

  RunResult result;
  for (int t = 0; t < trials; t++) {{
    id<MTLCommandBuffer> cmd = dispatch_once();
    result.elapsed_ms.push_back(
        (double)(cmd.GPUEndTime - cmd.GPUStartTime) * 1000.0);
  }}
  const uint32_t *data = (const uint32_t *)[outBuf contents];
  result.out.assign(data, data + (size_t)total);
  return result;
}}

double Median(std::vector<double> v) {{
  std::sort(v.begin(), v.end());
  return v[v.size() / 2];
}}

}}  // namespace

int main(int argc, char **argv) {{
  const int batch = argc > 1 ? std::atoi(argv[1]) : {default_batch};
  const int trials = argc > 2 ? std::atoi(argv[2]) : 30;
  const int sets = argc > 3 ? std::atoi(argv[3]) : 5;
  const int warmups = 8;
  const std::string header_path = argc > 4 ? argv[4]
                                            : "runtime/gpu/mdl_gpu_kernels.h";

  @autoreleasepool {{
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (device == nil) {{
      std::fprintf(stderr, "Metalデバイスが見つからない\n");
      return 1;
    }}
    std::printf("device = %s\n", [[device name] UTF8String]);
    std::printf("model={model} node={node_id} out_channels=%d "
               "tap_rows_words=%d n_instr_total={n_instr} "
               "max_slots={max_slots}\n", kOutChannels, kTapRowsWords);
    std::printf("batch=%d trials=%d sets=%d(+warmup%d) "
               "total_threads=%lld\n", batch, trials, sets, warmups,
               (long long)kOutChannels * kTapRowsWords * batch);

    const std::string shared = StripIncludeGuard(ReadFile(header_path));
    const std::string src =
        std::string("#include <metal_stdlib>\nusing namespace metal;\n"
                    "#define MDL_DEV\n#define MDL_GLOBAL device\n"
                    "#define MDL_CONST constant\n#define MDL_LOCAL thread\n"
                    "typedef unsigned int uint32_t;\n") +
        shared + "\n" + kInterpSrc + "\n" + kCodegenSrc;

    NSError *err = nil;
    id<MTLLibrary> lib =
        [device newLibraryWithSource:[NSString stringWithUTF8String:src.c_str()]
                              options:nil error:&err];
    if (lib == nil) {{
      std::fprintf(stderr, "シェーダコンパイル失敗: %s\n",
                   [[err localizedDescription] UTF8String]);
      return 1;
    }}
    id<MTLFunction> fnInterp = [lib newFunctionWithName:@"mdl_calib_conv_interp"];
    id<MTLFunction> fnCodegen = [lib newFunctionWithName:@"mdl_calib_conv_codegen"];
    id<MTLComputePipelineState> psoInterp =
        [device newComputePipelineStateWithFunction:fnInterp error:&err];
    id<MTLComputePipelineState> psoCodegen =
        [device newComputePipelineStateWithFunction:fnCodegen error:&err];
    if (psoInterp == nil || psoCodegen == nil) {{
      std::fprintf(stderr, "パイプライン作成失敗: %s\n",
                   [[err localizedDescription] UTF8String]);
      return 1;
    }}
    std::printf("registers: interp maxThreadsPerTG=%lu  "
               "codegen maxThreadsPerTG=%lu\n",
               (unsigned long)psoInterp.maxTotalThreadsPerThreadgroup,
               (unsigned long)psoCodegen.maxTotalThreadsPerThreadgroup);
    id<MTLCommandQueue> queue = [device newCommandQueue];
    std::vector<uint32_t> scratch_h = BuildScratch();
    id<MTLBuffer> scratchBuf =
        [device newBufferWithBytes:scratch_h.data()
                             length:scratch_h.size() * 4
                            options:MTLResourceStorageModeShared];

    bool all_match = true;
    std::vector<double> meds_i, meds_c;
    for (int s = 0; s < sets; s++) {{
      RunResult ri = RunKernel(device, queue, psoInterp, true, scratchBuf,
                               batch, trials, warmups);
      RunResult rc = RunKernel(device, queue, psoCodegen, false, scratchBuf,
                               batch, trials, warmups);
      const bool match = ri.out == rc.out;
      all_match = all_match && match;
      if (!match) {{
        size_t n_diff = 0;
        for (size_t i = 0; i < ri.out.size(); i++)
          if (ri.out[i] != rc.out[i]) n_diff++;
        std::fprintf(stderr, "set %d: 不一致! %zu/%zu 要素\n", s, n_diff,
                    ri.out.size());
      }}
      const double mi = Median(ri.elapsed_ms), mc = Median(rc.elapsed_ms);
      meds_i.push_back(mi);
      meds_c.push_back(mc);
      std::printf("  set %d: interp=%.6fms codegen=%.6fms speedup=%.3fx "
                 "match=%s\n", s, mi, mc, mi / mc, match ? "OK" : "NG");
    }}
    const double mmi = Median(meds_i), mmc = Median(meds_c);
    std::printf("\n=== 総合(setsの中央値) ===\n");
    std::printf("interp  median-of-medians = %.6f ms\n", mmi);
    std::printf("codegen median-of-medians = %.6f ms\n", mmc);
    std::printf("speedup(interp/codegen)   = %.3fx\n", mmi / mmc);
    std::printf("出力一致: %s\n", all_match ? "全setでOK(bit-exact)" : "NG(要調査)");
    return all_match ? 0 : 1;
  }}
}}
'''


def format_int_array(values: list[int]) -> str:
    return ','.join(str(int(v)) for v in values) or '0'


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='seg_convdlgn_k32')
    ap.add_argument('--node', default=None,
                    help='conv2dノードid(省略時は最初のノード)')
    ap.add_argument('--batch', type=int, default=512)
    ap.add_argument('--out', default=None, help='.mm出力先(省略時は自動)')
    args = ap.parse_args()

    node_id, layout, stats = load_node_layout(args.model, args.node)
    print(f'ノード選択: {node_id}  stats={stats}')

    codegen_src, n_lines = build_codegen_source(layout)
    print(f'codegen生成行数(関数本体+switch): {n_lines}')

    interp_src = INTERP_KERNEL_TMPL

    mm_text = MM_TMPL.format(
        model=args.model, node_id=node_id,
        out_channels=layout.out_channels, trw=layout.tap_rows_words,
        n_instr=layout.n_instrs_total, max_slots=layout.max_slots_per_oc,
        mm_path='(生成先)', bin_path='(生成先)',
        oc_instr_start=format_int_array(layout.oc_instr_start),
        instr_flat=format_int_array(layout.instr_flat),
        oc_final_ref=format_int_array(layout.oc_final_ref),
        wpr_out=layout.wpr_out, mask_out=layout.mask_out,
        scratch_words=layout.scratch_words,
        default_batch=args.batch,
    )
    # kInterpSrc/kCodegenSrc をC++側の文字列定数として埋め込む
    # (MSLソース自体は実行時に newLibraryWithSource: で連結するので、
    #  ここではC++の `static const char*` へエスケープして入れる)。
    def cxx_string_literal(text: str) -> str:
        out = []
        for line in text.split('\n'):
            escaped = line.replace('\\', '\\\\').replace('"', '\\"')
            out.append(f'  "{escaped}\\n"')
        return '\n'.join(out)

    prelude = (
        f'static const char *kInterpSrc =\n{cxx_string_literal(interp_src)};\n'
        f'static const char *kCodegenSrc =\n{cxx_string_literal(codegen_src)};\n'
    )
    mm_text = mm_text.replace('namespace {{\n'.replace('{{', '{'),
                              'namespace {\n' + prelude, 1)

    out_path = Path(args.out) if args.out else (
        REPO / 'scripts' / 'gpu_calib' / f'_conv_codegen_ab_{node_id}.mm')
    out_path.write_text(mm_text, encoding='utf-8')
    print(f'.mm出力: {out_path}')

    clangxx = shutil.which('clang++')
    if clangxx is None:
        print('clang++が見つからないためビルドはスキップ')
        return 0
    bin_path = out_path.with_suffix('')
    build = subprocess.run(
        [clangxx, '-std=c++17', '-O2', '-framework', 'Metal',
         '-framework', 'Foundation', str(out_path), '-o', str(bin_path)],
        capture_output=True, text=True)
    if build.returncode != 0:
        print('ビルド失敗:')
        print(build.stderr)
        return 1
    print(f'ビルド成功: {bin_path}')
    print(f'実行: {bin_path} [batch] [trials] [sets]  (リポジトリルートから)')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
