/* 最小実験(自動生成、コミットしない) — conv_exec_split の
 * (a)現行インタプリタ vs (b)tt即値+slots静的展開のMetal A/Bベンチ。
 * 対象: seg_convdlgn_k32 / node=n1_lutconv2d (out_channels=32,
 * tap_rows_words=128, n_instr_total=132, max_slots=5)
 *
 * ビルド:
 *   clang++ -std=c++17 -O2 -framework Metal -framework Foundation \
 *     (生成先) -o (生成先)
 * 実行(リポジトリルートから):
 *   (生成先) [batch] [trials] [sets]
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

namespace {
static const char *kInterpSrc =
  "\n"
  "#define MDL_CALIB_MAX_SLOTS 8\n"
  "\n"
  "kernel void mdl_calib_conv_interp(\n"
  "    device const uint *scratch [[buffer(0)]],\n"
  "    device uint *out [[buffer(1)]],\n"
  "    constant int *oc_instr_start [[buffer(2)]],\n"
  "    constant int *instr [[buffer(3)]],\n"
  "    constant int *oc_final_ref [[buffer(4)]],\n"
  "    constant int &tap_rows_words [[buffer(5)]],\n"
  "    constant int &out_channels [[buffer(6)]],\n"
  "    constant uint &mask_out [[buffer(7)]],\n"
  "    constant int &wpr_out [[buffer(8)]],\n"
  "    constant int &total [[buffer(9)]],\n"
  "    uint gid [[thread_position_in_grid]]) {\n"
  "  if ((int)gid >= total) return;\n"
  "  const int w = tap_rows_words;\n"
  "  const int per_n = out_channels * w;\n"
  "  const int n = (int)gid / per_n;\n"
  "  const int rem = (int)gid - n * per_n;\n"
  "  const int oc = rem / w;\n"
  "  const int i = rem - oc * w;\n"
  "  const uint tail = ((i % wpr_out) == wpr_out - 1) ? mask_out : 0xffffffffu;\n"
  "  thread uint slots[MDL_CALIB_MAX_SLOTS];\n"
  "  const int start = oc_instr_start[oc];\n"
  "  const int end = oc_instr_start[oc + 1];\n"
  "  for (int k = start; k < end; k++) {\n"
  "    constant int *ins = instr + k * 4;\n"
  "    const int ap = ins[1];\n"
  "    const int bp = ins[2];\n"
  "    const uint av = (ap & 1) ? slots[ap >> 1] : scratch[(ap >> 1) + i];\n"
  "    const uint bv = (bp & 1) ? slots[bp >> 1] : scratch[(bp >> 1) + i];\n"
  "    slots[ins[3]] = mdl_g_gate2((uint)ins[0], av, bv);\n"
  "  }\n"
  "  {\n"
  "    const int fr = oc_final_ref[oc];\n"
  "    const uint v = (fr & 1) ? slots[fr >> 1] : scratch[(fr >> 1) + i];\n"
  "    out[(uint)n * (uint)per_n + (uint)oc * (uint)w + (uint)i] = v & tail;\n"
  "  }\n"
  "}\n"
  "\n";
static const char *kCodegenSrc =
  "static inline uint32_t conv_oc_codegen_0(device const uint *sc, int i) {\n"
  "  uint32_t s0, s1, s2;\n"
  "  s0 = sc[260 + i] | ~sc[392 + i];\n"
  "  s1 = sc[520 + i] | sc[654 + i];\n"
  "  s2 = s0 | ~s1;\n"
  "  s1 = sc[784 + i] ^ sc[918 + i];\n"
  "  s0 = s2 & s1;\n"
  "  return s0;\n"
  "}\n"
  "\n"
  "static inline uint32_t conv_oc_codegen_1(device const uint *sc, int i) {\n"
  "  uint32_t s0, s1, s2, s3;\n"
  "  s0 = sc[1052 + i] & sc[1048 + i];\n"
  "  s1 = sc[1182 + i] & sc[1314 + i];\n"
  "  s2 = sc[1448 + i] | ~sc[1578 + i];\n"
  "  s3 = s0 | s1;\n"
  "  s1 = s2 & sc[1444 + i];\n"
  "  s2 = s3 | s1;\n"
  "  return s2;\n"
  "}\n"
  "\n"
  "static inline uint32_t conv_oc_codegen_2(device const uint *sc, int i) {\n"
  "  uint32_t s0, s1;\n"
  "  s0 = sc[260 + i] | sc[256 + i];\n"
  "  s1 = s0 | sc[1712 + i];\n"
  "  return s1;\n"
  "}\n"
  "\n"
  "static inline uint32_t conv_oc_codegen_3(device const uint *sc, int i) {\n"
  "  uint32_t s0, s1, s2, s3;\n"
  "  s0 = ~(sc[1974 + i] ^ sc[2106 + i]);\n"
  "  s1 = ~(sc[2236 + i] ^ sc[2372 + i]);\n"
  "  s2 = sc[2372 + i] | ~sc[2504 + i];\n"
  "  s3 = s0 & s1;\n"
  "  s1 = s2 | sc[2632 + i];\n"
  "  s2 = s3 & s1;\n"
  "  return s2;\n"
  "}\n"
  "\n"
  "static inline uint32_t conv_oc_codegen_4(device const uint *sc, int i) {\n"
  "  uint32_t s0, s1, s2, s3, s4;\n"
  "  s0 = sc[2766 + i] & sc[2898 + i];\n"
  "  s1 = sc[3028 + i] & sc[3032 + i];\n"
  "  s2 = ~(sc[3160 + i] ^ sc[3296 + i]);\n"
  "  s3 = sc[3424 + i] & sc[3292 + i];\n"
  "  s4 = s0 | ~s1;\n"
  "  s1 = s2 & s3;\n"
  "  s3 = s4 | ~s1;\n"
  "  return s3;\n"
  "}\n"
  "\n"
  "static inline uint32_t conv_oc_codegen_5(device const uint *sc, int i) {\n"
  "  uint32_t s0, s1, s2, s3;\n"
  "  s0 = sc[3560 + i] & sc[3688 + i];\n"
  "  s1 = sc[3824 + i] ^ sc[3954 + i];\n"
  "  s2 = sc[4086 + i] | ~sc[4220 + i];\n"
  "  s3 = s0 | ~s1;\n"
  "  s1 = ~(s2 ^ s3);\n"
  "  return s1;\n"
  "}\n"
  "\n"
  "static inline uint32_t conv_oc_codegen_6(device const uint *sc, int i) {\n"
  "  uint32_t s0, s1, s2, s3;\n"
  "  s0 = sc[2764 + i] & sc[4350 + i];\n"
  "  s1 = sc[4484 + i] & sc[4480 + i];\n"
  "  s2 = sc[4350 + i] & sc[4348 + i];\n"
  "  s3 = s0 & sc[4352 + i];\n"
  "  s0 = s1 & s2;\n"
  "  s2 = s3 & s0;\n"
  "  return s2;\n"
  "}\n"
  "\n"
  "static inline uint32_t conv_oc_codegen_7(device const uint *sc, int i) {\n"
  "  uint32_t s0, s1, s2;\n"
  "  s0 = sc[4612 + i] & ~sc[4746 + i];\n"
  "  s1 = sc[1840 + i] | sc[4876 + i];\n"
  "  s2 = sc[3160 + i] ^ s0;\n"
  "  s0 = sc[256 + i] ^ s1;\n"
  "  s1 = s2 ^ s0;\n"
  "  return s1;\n"
  "}\n"
  "\n"
  "static inline uint32_t conv_oc_codegen_8(device const uint *sc, int i) {\n"
  "  uint32_t s0, s1, s2, s3;\n"
  "  s0 = sc[5012 + i] & ~sc[5140 + i];\n"
  "  s1 = sc[5276 + i] | sc[5408 + i];\n"
  "  s2 = sc[5540 + i] | sc[5668 + i];\n"
  "  s3 = ~(s0 ^ s1);\n"
  "  s1 = ~(s2 ^ s3);\n"
  "  return s1;\n"
  "}\n"
  "\n"
  "static inline uint32_t conv_oc_codegen_9(device const uint *sc, int i) {\n"
  "  uint32_t s0;\n"
  "  s0 = sc[5800 + i] & sc[5932 + i];\n"
  "  return s0;\n"
  "}\n"
  "\n"
  "static inline uint32_t conv_oc_codegen_10(device const uint *sc, int i) {\n"
  "  uint32_t s0, s1, s2, s3;\n"
  "  s0 = ~(sc[4220 + i] ^ sc[6068 + i]);\n"
  "  s1 = sc[6064 + i] | sc[6196 + i];\n"
  "  s2 = sc[4216 + i] | sc[6328 + i];\n"
  "  s3 = s0 & s1;\n"
  "  s1 = s2 & s3;\n"
  "  return s1;\n"
  "}\n"
  "\n"
  "static inline uint32_t conv_oc_codegen_11(device const uint *sc, int i) {\n"
  "  uint32_t s0, s1, s2;\n"
  "  s0 = sc[6596 + i] & sc[6724 + i];\n"
  "  s1 = sc[6858 + i] | sc[6594 + i];\n"
  "  s2 = s0 & sc[6460 + i];\n"
  "  s0 = s1 & s2;\n"
  "  return s0;\n"
  "}\n"
  "\n"
  "static inline uint32_t conv_oc_codegen_12(device const uint *sc, int i) {\n"
  "  uint32_t s0;\n"
  "  s0 = ~(sc[6990 + i] ^ sc[7124 + i]);\n"
  "  return s0;\n"
  "}\n"
  "\n"
  "static inline uint32_t conv_oc_codegen_13(device const uint *sc, int i) {\n"
  "  uint32_t s0, s1, s2, s3;\n"
  "  s0 = sc[3556 + i] & ~sc[7254 + i];\n"
  "  s1 = ~(sc[3820 + i] | sc[7388 + i]);\n"
  "  s2 = sc[2634 + i] & sc[2636 + i];\n"
  "  s3 = s0 ^ s1;\n"
  "  s1 = s2 & ~sc[6462 + i];\n"
  "  s2 = s3 ^ s1;\n"
  "  return s2;\n"
  "}\n"
  "\n"
  "static inline uint32_t conv_oc_codegen_14(device const uint *sc, int i) {\n"
  "  uint32_t s0, s1, s2, s3, s4;\n"
  "  s0 = sc[7516 + i] & sc[7650 + i];\n"
  "  s1 = sc[7780 + i] | sc[7912 + i];\n"
  "  s2 = sc[8048 + i] & ~sc[5404 + i];\n"
  "  s3 = sc[8180 + i] | ~sc[8308 + i];\n"
  "  s4 = ~(s0 ^ s1);\n"
  "  s1 = s2 & ~s3;\n"
  "  s3 = s4 | s1;\n"
  "  return s3;\n"
  "}\n"
  "\n"
  "static inline uint32_t conv_oc_codegen_15(device const uint *sc, int i) {\n"
  "  uint32_t s0, s1, s2;\n"
  "  s0 = ~(sc[2504 + i] ^ sc[8442 + i]);\n"
  "  s1 = sc[8704 + i] | ~sc[8840 + i];\n"
  "  s2 = s0 & ~sc[8572 + i];\n"
  "  s0 = ~s1 & s2;\n"
  "  return s0;\n"
  "}\n"
  "\n"
  "static inline uint32_t conv_oc_codegen_16(device const uint *sc, int i) {\n"
  "  uint32_t s0, s1;\n"
  "  s0 = sc[8968 + i] & sc[9104 + i];\n"
  "  s1 = s0 ^ sc[786 + i];\n"
  "  return s1;\n"
  "}\n"
  "\n"
  "static inline uint32_t conv_oc_codegen_17(device const uint *sc, int i) {\n"
  "\n"
  "  return sc[9232 + i];\n"
  "}\n"
  "\n"
  "static inline uint32_t conv_oc_codegen_18(device const uint *sc, int i) {\n"
  "  uint32_t s0, s1;\n"
  "  s0 = ~(sc[9500 + i] | sc[5272 + i]);\n"
  "  s1 = ~s0 | sc[9364 + i];\n"
  "  s0 = s1 | ~sc[7914 + i];\n"
  "  return s0;\n"
  "}\n"
  "\n"
  "static inline uint32_t conv_oc_codegen_19(device const uint *sc, int i) {\n"
  "  uint32_t s0, s1, s2, s3;\n"
  "  s0 = sc[9632 + i] | ~sc[9764 + i];\n"
  "  s1 = sc[9762 + i] & sc[9896 + i];\n"
  "  s2 = sc[10288 + i] & ~sc[10420 + i];\n"
  "  s3 = s0 | ~s1;\n"
  "  s1 = s2 | sc[10024 + i];\n"
  "  s2 = s3 | ~s1;\n"
  "  return s2;\n"
  "}\n"
  "\n"
  "static inline uint32_t conv_oc_codegen_20(device const uint *sc, int i) {\n"
  "  uint32_t s0, s1, s2, s3;\n"
  "  s0 = sc[6728 + i] | sc[10556 + i];\n"
  "  s1 = sc[10688 + i] & sc[10684 + i];\n"
  "  s2 = sc[10820 + i] & ~sc[10950 + i];\n"
  "  s3 = ~(s0 ^ sc[5672 + i]);\n"
  "  s0 = s1 & s2;\n"
  "  s2 = s3 & ~s0;\n"
  "  return s2;\n"
  "}\n"
  "\n"
  "static inline uint32_t conv_oc_codegen_21(device const uint *sc, int i) {\n"
  "  uint32_t s0, s1;\n"
  "  s0 = sc[11084 + i] & ~sc[11216 + i];\n"
  "  s1 = s0 & ~sc[3956 + i];\n"
  "  return s1;\n"
  "}\n"
  "\n"
  "static inline uint32_t conv_oc_codegen_22(device const uint *sc, int i) {\n"
  "  uint32_t s0, s1, s2, s3;\n"
  "  s0 = sc[10160 + i] | sc[11480 + i];\n"
  "  s1 = sc[11612 + i] & ~sc[11742 + i];\n"
  "  s2 = sc[2368 + i] ^ sc[12004 + i];\n"
  "  s3 = s0 | s1;\n"
  "  s1 = s2 ^ sc[11872 + i];\n"
  "  s2 = s3 | s1;\n"
  "  return s2;\n"
  "}\n"
  "\n"
  "static inline uint32_t conv_oc_codegen_23(device const uint *sc, int i) {\n"
  "  uint32_t s0, s1;\n"
  "  s0 = sc[3428 + i] & ~sc[3690 + i];\n"
  "  s1 = s0 & ~sc[3690 + i];\n"
  "  s0 = ~s1;\n"
  "  return s0;\n"
  "}\n"
  "\n"
  "static inline uint32_t conv_oc_codegen_24(device const uint *sc, int i) {\n"
  "  uint32_t s0, s1, s2, s3;\n"
  "  s0 = sc[12138 + i] & sc[12270 + i];\n"
  "  s1 = sc[12400 + i] | ~sc[12534 + i];\n"
  "  s2 = sc[12664 + i] | sc[12796 + i];\n"
  "  s3 = s1 | s2;\n"
  "  s2 = s0 | ~s3;\n"
  "  return s2;\n"
  "}\n"
  "\n"
  "static inline uint32_t conv_oc_codegen_25(device const uint *sc, int i) {\n"
  "  uint32_t s0, s1, s2, s3;\n"
  "  s0 = ~sc[6728 + i] & sc[11872 + i];\n"
  "  s1 = sc[7782 + i] & ~sc[12932 + i];\n"
  "  s2 = sc[10556 + i] & sc[8176 + i];\n"
  "  s3 = s0 | ~s1;\n"
  "  s1 = s2 & ~sc[7784 + i];\n"
  "  s2 = s3 & ~s1;\n"
  "  return s2;\n"
  "}\n"
  "\n"
  "static inline uint32_t conv_oc_codegen_26(device const uint *sc, int i) {\n"
  "  uint32_t s0, s1, s2;\n"
  "  s0 = sc[13064 + i] & sc[10424 + i];\n"
  "  s1 = sc[1974 + i] & sc[11876 + i];\n"
  "  s2 = s0 | ~s1;\n"
  "  return s2;\n"
  "}\n"
  "\n"
  "static inline uint32_t conv_oc_codegen_27(device const uint *sc, int i) {\n"
  "  uint32_t s0, s1;\n"
  "  s0 = sc[13194 + i] & sc[6332 + i];\n"
  "  s1 = s0 | sc[4614 + i];\n"
  "  return s1;\n"
  "}\n"
  "\n"
  "static inline uint32_t conv_oc_codegen_28(device const uint *sc, int i) {\n"
  "  uint32_t s0;\n"
  "  s0 = sc[13324 + i] & sc[12536 + i];\n"
  "  return s0;\n"
  "}\n"
  "\n"
  "static inline uint32_t conv_oc_codegen_29(device const uint *sc, int i) {\n"
  "  uint32_t s0, s1, s2;\n"
  "  s0 = sc[13590 + i] ^ sc[13720 + i];\n"
  "  s1 = sc[13852 + i] & sc[13984 + i];\n"
  "  s2 = s0 | ~s1;\n"
  "  s1 = ~(sc[11348 + i] & s2);\n"
  "  return s1;\n"
  "}\n"
  "\n"
  "static inline uint32_t conv_oc_codegen_30(device const uint *sc, int i) {\n"
  "  uint32_t s0, s1, s2, s3;\n"
  "  s0 = sc[14120 + i] | ~sc[14252 + i];\n"
  "  s1 = ~(sc[14384 + i] ^ sc[14512 + i]);\n"
  "  s2 = ~sc[13456 + i] & sc[14116 + i];\n"
  "  s3 = s0 & s1;\n"
  "  s1 = s2 & s3;\n"
  "  return s1;\n"
  "}\n"
  "\n"
  "static inline uint32_t conv_oc_codegen_31(device const uint *sc, int i) {\n"
  "  uint32_t s0, s1, s2;\n"
  "  s0 = sc[5272 + i] & ~sc[14910 + i];\n"
  "  s1 = sc[14648 + i] & sc[14780 + i];\n"
  "  s2 = ~s0 | s1;\n"
  "  return s2;\n"
  "}\n"
  "\n"
  "kernel void mdl_calib_conv_codegen(\n"
  "    device const uint *scratch [[buffer(0)]],\n"
  "    device uint *out [[buffer(1)]],\n"
  "    constant int &tap_rows_words [[buffer(2)]],\n"
  "    constant int &out_channels [[buffer(3)]],\n"
  "    constant uint &mask_out [[buffer(4)]],\n"
  "    constant int &wpr_out [[buffer(5)]],\n"
  "    constant int &total [[buffer(6)]],\n"
  "    uint gid [[thread_position_in_grid]]) {\n"
  "  if ((int)gid >= total) return;\n"
  "  const int w = tap_rows_words;\n"
  "  const int per_n = out_channels * w;\n"
  "  const int n = (int)gid / per_n;\n"
  "  const int rem = (int)gid - n * per_n;\n"
  "  const int oc = rem / w;\n"
  "  const int i = rem - oc * w;\n"
  "  const uint tail = ((i % wpr_out) == wpr_out - 1) ? mask_out : 0xffffffffu;\n"
  "  uint32_t v = 0u;\n"
  "  switch (oc) {\n"
  "    case 0: v = conv_oc_codegen_0(scratch, i); break;\n"
  "    case 1: v = conv_oc_codegen_1(scratch, i); break;\n"
  "    case 2: v = conv_oc_codegen_2(scratch, i); break;\n"
  "    case 3: v = conv_oc_codegen_3(scratch, i); break;\n"
  "    case 4: v = conv_oc_codegen_4(scratch, i); break;\n"
  "    case 5: v = conv_oc_codegen_5(scratch, i); break;\n"
  "    case 6: v = conv_oc_codegen_6(scratch, i); break;\n"
  "    case 7: v = conv_oc_codegen_7(scratch, i); break;\n"
  "    case 8: v = conv_oc_codegen_8(scratch, i); break;\n"
  "    case 9: v = conv_oc_codegen_9(scratch, i); break;\n"
  "    case 10: v = conv_oc_codegen_10(scratch, i); break;\n"
  "    case 11: v = conv_oc_codegen_11(scratch, i); break;\n"
  "    case 12: v = conv_oc_codegen_12(scratch, i); break;\n"
  "    case 13: v = conv_oc_codegen_13(scratch, i); break;\n"
  "    case 14: v = conv_oc_codegen_14(scratch, i); break;\n"
  "    case 15: v = conv_oc_codegen_15(scratch, i); break;\n"
  "    case 16: v = conv_oc_codegen_16(scratch, i); break;\n"
  "    case 17: v = conv_oc_codegen_17(scratch, i); break;\n"
  "    case 18: v = conv_oc_codegen_18(scratch, i); break;\n"
  "    case 19: v = conv_oc_codegen_19(scratch, i); break;\n"
  "    case 20: v = conv_oc_codegen_20(scratch, i); break;\n"
  "    case 21: v = conv_oc_codegen_21(scratch, i); break;\n"
  "    case 22: v = conv_oc_codegen_22(scratch, i); break;\n"
  "    case 23: v = conv_oc_codegen_23(scratch, i); break;\n"
  "    case 24: v = conv_oc_codegen_24(scratch, i); break;\n"
  "    case 25: v = conv_oc_codegen_25(scratch, i); break;\n"
  "    case 26: v = conv_oc_codegen_26(scratch, i); break;\n"
  "    case 27: v = conv_oc_codegen_27(scratch, i); break;\n"
  "    case 28: v = conv_oc_codegen_28(scratch, i); break;\n"
  "    case 29: v = conv_oc_codegen_29(scratch, i); break;\n"
  "    case 30: v = conv_oc_codegen_30(scratch, i); break;\n"
  "    case 31: v = conv_oc_codegen_31(scratch, i); break;\n"
  "    default: v = 0u; break;\n"
  "  }\n"
  "  out[(uint)n * (uint)per_n + (uint)oc * (uint)w + (uint)i] = v & tail;\n"
  "}\n"
  "\n";

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

/* 実ノードのCSRデータ(oc_instr_start/instr_flat/oc_final_ref)。
   これはインタプリタ側だけが実行時に読む(codegen側はPython生成時に
   直線コードへ焼き込み済み)。 */
static const int32_t kOcInstrStart[] = {0,5,11,13,19,26,31,37,42,47,48,53,57,58,64,71,75,77,77,80,86,92,94,100,103,108,114,117,119,120,124,129,132};
static const int32_t kInstrFlat[] = {11,520,784,0,14,1040,1308,1,11,1,3,2,6,1568,1836,1,8,5,3,0,8,2104,2096,0,8,2364,2628,1,11,2896,3156,2,14,1,3,3,8,5,2888,1,14,7,3,2,14,520,512,0,14,1,3424,1,9,3948,4212,0,9,4472,4744,1,11,4744,5008,2,8,1,3,3,14,5,5264,1,8,7,3,2,8,5532,5796,0,8,6056,6064,1,9,6320,6592,2,8,6848,6584,3,11,1,3,4,8,5,7,1,11,9,3,3,8,7120,7376,0,6,7648,7908,1,11,8172,8440,2,11,1,3,3,9,5,7,1,8,5528,8700,0,8,8968,8960,1,8,8700,8696,2,8,1,8704,3,8,3,5,0,8,7,1,2,2,9224,9492,0,14,3680,9752,1,6,6320,1,2,6,512,3,0,6,5,1,1,2,10024,10280,0,14,10552,10816,1,14,11080,11336,2,9,1,3,3,9,5,7,1,8,11600,11864,0,9,8440,12136,0,14,12128,12392,1,14,8432,12656,2,8,1,3,3,8,5,7,1,8,13192,13448,0,14,13716,13188,1,8,1,12920,2,8,3,5,0,9,13980,14248,0,2,7112,14508,0,1,7640,14776,1,8,5268,5272,2,6,1,3,3,2,5,12924,1,6,7,3,2,8,15032,15300,0,14,15560,15824,1,2,16096,10808,2,11,16360,16616,3,9,1,3,4,2,5,7,1,14,9,3,3,9,5008,16884,0,11,17408,17680,1,2,1,17144,2,4,3,5,0,8,17936,18208,0,6,1,1572,1,1,19000,10544,0,13,1,18728,1,11,3,15828,0,11,19264,19528,0,8,19524,19792,1,2,20576,20840,2,11,1,3,3,14,5,20048,1,11,7,3,2,14,13456,21112,0,8,21376,21368,1,2,21640,21900,2,9,1,11344,3,8,3,5,0,2,7,1,2,2,22168,22432,0,2,1,7912,1,14,20320,22960,0,2,23224,23484,1,6,4736,24008,2,14,1,3,3,6,5,23744,1,14,7,3,2,2,6856,7380,0,2,1,7380,1,5,3,3,0,8,24276,24540,0,11,24800,25068,1,14,25328,25592,2,14,3,5,3,11,1,7,2,4,13456,23744,0,2,15564,25864,1,8,21112,16352,2,11,1,3,3,2,5,15568,1,2,7,3,2,8,26128,20848,0,8,3948,23752,1,11,1,3,2,8,26388,12664,0,14,1,9228,1,8,26648,25072,0,6,27180,27440,0,8,27704,27968,1,11,1,3,2,7,22696,5,1,11,28240,28504,0,9,28768,29024,1,4,26912,28232,2,8,1,3,3,8,5,7,1,2,10544,29820,0,8,29296,29560,1,13,1,3,2};
static const int32_t kOcFinalRef[] = {1,5,3,5,7,3,5,3,3,1,3,1,1,5,7,1,3,18464,1,5,5,3,5,1,5,5,5,3,1,3,3,5};

constexpr int kOutChannels = 32;
constexpr int kTapRowsWords = 128;
constexpr int kWprOut = 2;
constexpr uint32_t kMaskOut = 4294967295u;
constexpr int kScratchWords = 15040;

/* scratch(読み取り専用、全batch共通の1テンプレート)を決定的に初期化する。
   実画像ではなく合成値だが、両カーネルとも**同一の値**を読むので
   命令列の実行結果だけを比較できる(gate2_ab.mm と同じ方針)。
   ただし [0,trw) はCONST-0平面(全0)、[trw,2*trw)はCONST-1平面(全1)
   という実カーネルの不変条件(`mdl_k_conv_const` が保証する)だけは
   合成データでも守る必要がある — codegen側は `cbackend.py` 同様この
   不変条件を前提に off==0/off==trw を即値0u/0xffffffffuへ畳み込む
   最適化をしているため、破ると偽の不一致が出る(実際に一度出た)。 */
std::vector<uint32_t> BuildScratch() {
  std::vector<uint32_t> sc(kScratchWords);
  for (size_t i = 0; i < sc.size(); i++) {
    uint32_t v = (uint32_t)i ^ 0x9e3779b9u;
    v ^= v << 13; v ^= v >> 17; v ^= v << 5;
    sc[i] = v;
  }
  for (int i = 0; i < kTapRowsWords; i++) sc[i] = 0u;
  for (int i = 0; i < kTapRowsWords; i++) sc[kTapRowsWords + i] = 0xffffffffu;
  return sc;
}

struct RunResult {
  std::vector<double> elapsed_ms;
  std::vector<uint32_t> out;
};

RunResult RunKernel(id<MTLDevice> device, id<MTLCommandQueue> queue,
                    id<MTLComputePipelineState> pso, bool is_interp,
                    id<MTLBuffer> scratchBuf, int batch, int trials,
                    int warmups) {
  const int total = kOutChannels * kTapRowsWords * batch;
  id<MTLBuffer> outBuf =
      [device newBufferWithLength:(size_t)total * 4
                           options:MTLResourceStorageModeShared];
  id<MTLBuffer> ocInstrStartBuf = nil, instrBuf = nil, ocFinalRefBuf = nil;
  if (is_interp) {
    ocInstrStartBuf = [device newBufferWithBytes:kOcInstrStart
                                          length:sizeof(kOcInstrStart)
                                         options:MTLResourceStorageModeShared];
    instrBuf = [device newBufferWithBytes:kInstrFlat
                                    length:sizeof(kInstrFlat)
                                   options:MTLResourceStorageModeShared];
    ocFinalRefBuf = [device newBufferWithBytes:kOcFinalRef
                                        length:sizeof(kOcFinalRef)
                                       options:MTLResourceStorageModeShared];
  }
  const int32_t trw_i = kTapRowsWords, oc_i = kOutChannels, wpr_i = kWprOut;
  const uint32_t mask_i = kMaskOut;
  const int32_t total_i = total;

  auto dispatch_once = [&]() {
    id<MTLCommandBuffer> cmd = [queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:pso];
    [enc setBuffer:scratchBuf offset:0 atIndex:0];
    [enc setBuffer:outBuf offset:0 atIndex:1];
    if (is_interp) {
      [enc setBuffer:ocInstrStartBuf offset:0 atIndex:2];
      [enc setBuffer:instrBuf offset:0 atIndex:3];
      [enc setBuffer:ocFinalRefBuf offset:0 atIndex:4];
      [enc setBytes:&trw_i length:sizeof(trw_i) atIndex:5];
      [enc setBytes:&oc_i length:sizeof(oc_i) atIndex:6];
      [enc setBytes:&mask_i length:sizeof(mask_i) atIndex:7];
      [enc setBytes:&wpr_i length:sizeof(wpr_i) atIndex:8];
      [enc setBytes:&total_i length:sizeof(total_i) atIndex:9];
    } else {
      [enc setBytes:&trw_i length:sizeof(trw_i) atIndex:2];
      [enc setBytes:&oc_i length:sizeof(oc_i) atIndex:3];
      [enc setBytes:&mask_i length:sizeof(mask_i) atIndex:4];
      [enc setBytes:&wpr_i length:sizeof(wpr_i) atIndex:5];
      [enc setBytes:&total_i length:sizeof(total_i) atIndex:6];
    }
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
  };

  for (int w = 0; w < warmups; w++) dispatch_once();

  RunResult result;
  for (int t = 0; t < trials; t++) {
    id<MTLCommandBuffer> cmd = dispatch_once();
    result.elapsed_ms.push_back(
        (double)(cmd.GPUEndTime - cmd.GPUStartTime) * 1000.0);
  }
  const uint32_t *data = (const uint32_t *)[outBuf contents];
  result.out.assign(data, data + (size_t)total);
  return result;
}

double Median(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  return v[v.size() / 2];
}

}  // namespace

int main(int argc, char **argv) {
  const int batch = argc > 1 ? std::atoi(argv[1]) : 512;
  const int trials = argc > 2 ? std::atoi(argv[2]) : 30;
  const int sets = argc > 3 ? std::atoi(argv[3]) : 5;
  const int warmups = 8;
  const std::string header_path = argc > 4 ? argv[4]
                                            : "runtime/gpu/mdl_gpu_kernels.h";

  @autoreleasepool {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (device == nil) {
      std::fprintf(stderr, "Metalデバイスが見つからない\n");
      return 1;
    }
    std::printf("device = %s\n", [[device name] UTF8String]);
    std::printf("model=seg_convdlgn_k32 node=n1_lutconv2d out_channels=%d "
               "tap_rows_words=%d n_instr_total=132 "
               "max_slots=5\n", kOutChannels, kTapRowsWords);
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
    if (lib == nil) {
      std::fprintf(stderr, "シェーダコンパイル失敗: %s\n",
                   [[err localizedDescription] UTF8String]);
      return 1;
    }
    id<MTLFunction> fnInterp = [lib newFunctionWithName:@"mdl_calib_conv_interp"];
    id<MTLFunction> fnCodegen = [lib newFunctionWithName:@"mdl_calib_conv_codegen"];
    id<MTLComputePipelineState> psoInterp =
        [device newComputePipelineStateWithFunction:fnInterp error:&err];
    id<MTLComputePipelineState> psoCodegen =
        [device newComputePipelineStateWithFunction:fnCodegen error:&err];
    if (psoInterp == nil || psoCodegen == nil) {
      std::fprintf(stderr, "パイプライン作成失敗: %s\n",
                   [[err localizedDescription] UTF8String]);
      return 1;
    }
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
    for (int s = 0; s < sets; s++) {
      RunResult ri = RunKernel(device, queue, psoInterp, true, scratchBuf,
                               batch, trials, warmups);
      RunResult rc = RunKernel(device, queue, psoCodegen, false, scratchBuf,
                               batch, trials, warmups);
      const bool match = ri.out == rc.out;
      all_match = all_match && match;
      if (!match) {
        size_t n_diff = 0;
        for (size_t i = 0; i < ri.out.size(); i++)
          if (ri.out[i] != rc.out[i]) n_diff++;
        std::fprintf(stderr, "set %d: 不一致! %zu/%zu 要素\n", s, n_diff,
                    ri.out.size());
      }
      const double mi = Median(ri.elapsed_ms), mc = Median(rc.elapsed_ms);
      meds_i.push_back(mi);
      meds_c.push_back(mc);
      std::printf("  set %d: interp=%.6fms codegen=%.6fms speedup=%.3fx "
                 "match=%s\n", s, mi, mc, mi / mc, match ? "OK" : "NG");
    }
    const double mmi = Median(meds_i), mmc = Median(meds_c);
    std::printf("\n=== 総合(setsの中央値) ===\n");
    std::printf("interp  median-of-medians = %.6f ms\n", mmi);
    std::printf("codegen median-of-medians = %.6f ms\n", mmc);
    std::printf("speedup(interp/codegen)   = %.3fx\n", mmi / mmc);
    std::printf("出力一致: %s\n", all_match ? "全setでOK(bit-exact)" : "NG(要調査)");
    return all_match ? 0 : 1;
  }
}
