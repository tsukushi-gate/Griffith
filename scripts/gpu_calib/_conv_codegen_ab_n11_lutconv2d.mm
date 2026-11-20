/* 最小実験(自動生成、コミットしない) — conv_exec_split の
 * (a)現行インタプリタ vs (b)tt即値+slots静的展開のMetal A/Bベンチ。
 * 対象: seg_convdlgn_k32 / node=n11_lutconv2d (out_channels=64,
 * tap_rows_words=128, n_instr_total=207, max_slots=5)
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
  "  s0 = ~(sc[258 + i] ^ sc[390 + i]);\n"
  "  s1 = sc[522 + i] | sc[654 + i];\n"
  "  s2 = s0 | ~s1;\n"
  "  s1 = ~s2;\n"
  "  return s1;\n"
  "}\n"
  "\n"
  "static inline uint32_t conv_oc_codegen_1(device const uint *sc, int i) {\n"
  "  uint32_t s0, s1, s2;\n"
  "  s0 = sc[920 + i] & sc[1050 + i];\n"
  "  s1 = sc[1182 + i] | sc[1312 + i];\n"
  "  s2 = s0 | ~s1;\n"
  "  return s2;\n"
  "}\n"
  "\n"
  "static inline uint32_t conv_oc_codegen_2(device const uint *sc, int i) {\n"
  "  uint32_t s0, s1;\n"
  "  s0 = sc[1578 + i] | ~sc[1710 + i];\n"
  "  s1 = ~(s0 ^ sc[1840 + i]);\n"
  "  return s1;\n"
  "}\n"
  "\n"
  "static inline uint32_t conv_oc_codegen_3(device const uint *sc, int i) {\n"
  "  uint32_t s0, s1;\n"
  "  s0 = sc[2504 + i] | ~sc[2768 + i];\n"
  "  s1 = s0 | sc[2636 + i];\n"
  "  s0 = ~(s1 ^ sc[2504 + i]);\n"
  "  return s0;\n"
  "}\n"
  "\n"
  "static inline uint32_t conv_oc_codegen_4(device const uint *sc, int i) {\n"
  "  uint32_t s0, s1, s2;\n"
  "  s0 = sc[2896 + i] | ~sc[3030 + i];\n"
  "  s1 = sc[3296 + i] | sc[3424 + i];\n"
  "  s2 = s0 & sc[1972 + i];\n"
  "  s0 = s1 & sc[3160 + i];\n"
  "  s1 = ~(s2 ^ s0);\n"
  "  return s1;\n"
  "}\n"
  "\n"
  "static inline uint32_t conv_oc_codegen_5(device const uint *sc, int i) {\n"
  "  uint32_t s0, s1;\n"
  "  s0 = ~(sc[3560 + i] ^ sc[3556 + i]);\n"
  "  s1 = s0 & sc[3688 + i];\n"
  "  return s1;\n"
  "}\n"
  "\n"
  "static inline uint32_t conv_oc_codegen_6(device const uint *sc, int i) {\n"
  "  uint32_t s0, s1;\n"
  "  s0 = sc[4216 + i] & sc[1444 + i];\n"
  "  s1 = ~s0;\n"
  "  return s1;\n"
  "}\n"
  "\n"
  "static inline uint32_t conv_oc_codegen_7(device const uint *sc, int i) {\n"
  "  uint32_t s0, s1, s2;\n"
  "  s0 = sc[4348 + i] | ~sc[4484 + i];\n"
  "  s1 = sc[4612 + i] | sc[4748 + i];\n"
  "  s2 = s0 | ~s1;\n"
  "  return s2;\n"
  "}\n"
  "\n"
  "static inline uint32_t conv_oc_codegen_8(device const uint *sc, int i) {\n"
  "  uint32_t s0, s1, s2;\n"
  "  s0 = sc[5140 + i] & sc[5274 + i];\n"
  "  s1 = sc[5538 + i] & sc[1448 + i];\n"
  "  s2 = ~(s0 | s1);\n"
  "  return s2;\n"
  "}\n"
  "\n"
  "static inline uint32_t conv_oc_codegen_9(device const uint *sc, int i) {\n"
  "  uint32_t s0, s1;\n"
  "  s0 = sc[6064 + i] | sc[6196 + i];\n"
  "  s1 = s0 & sc[5932 + i];\n"
  "  s0 = ~s1;\n"
  "  return s0;\n"
  "}\n"
  "\n"
  "static inline uint32_t conv_oc_codegen_10(device const uint *sc, int i) {\n"
  "  uint32_t s0, s1, s2;\n"
  "  s0 = sc[2104 + i] & sc[6592 + i];\n"
  "  s1 = sc[2632 + i] | sc[2500 + i];\n"
  "  s2 = ~s0 & s1;\n"
  "  s1 = ~s2;\n"
  "  return s1;\n"
  "}\n"
  "\n"
  "static inline uint32_t conv_oc_codegen_11(device const uint *sc, int i) {\n"
  "  uint32_t s0, s1, s2;\n"
  "  s0 = sc[1578 + i] & sc[5800 + i];\n"
  "  s1 = sc[1708 + i] & sc[6860 + i];\n"
  "  s2 = s1 | sc[6988 + i];\n"
  "  s1 = s0 | ~s2;\n"
  "  return s1;\n"
  "}\n"
  "\n"
  "static inline uint32_t conv_oc_codegen_12(device const uint *sc, int i) {\n"
  "  uint32_t s0, s1, s2;\n"
  "  s0 = sc[3164 + i] | ~sc[7120 + i];\n"
  "  s1 = ~(sc[7252 + i] ^ sc[7384 + i]);\n"
  "  s2 = ~(sc[1976 + i] ^ s0);\n"
  "  s0 = s1 & s2;\n"
  "  return s0;\n"
  "}\n"
  "\n"
  "static inline uint32_t conv_oc_codegen_13(device const uint *sc, int i) {\n"
  "  uint32_t s0, s1, s2;\n"
  "  s0 = sc[7518 + i] & sc[7648 + i];\n"
  "  s1 = sc[7652 + i] & sc[7784 + i];\n"
  "  s2 = s0 | s1;\n"
  "  s1 = ~s2;\n"
  "  return s1;\n"
  "}\n"
  "\n"
  "static inline uint32_t conv_oc_codegen_14(device const uint *sc, int i) {\n"
  "  uint32_t s0, s1;\n"
  "  s0 = sc[3952 + i] | sc[7388 + i];\n"
  "  s1 = s0 & sc[7256 + i];\n"
  "  s0 = ~s1 | sc[7252 + i];\n"
  "  return s0;\n"
  "}\n"
  "\n"
  "static inline uint32_t conv_oc_codegen_15(device const uint *sc, int i) {\n"
  "  uint32_t s0;\n"
  "  s0 = ~(sc[7916 + i] ^ sc[8176 + i]);\n"
  "  return s0;\n"
  "}\n"
  "\n"
  "static inline uint32_t conv_oc_codegen_16(device const uint *sc, int i) {\n"
  "  uint32_t s0, s1, s2, s3, s4;\n"
  "  s0 = sc[5406 + i] | sc[8308 + i];\n"
  "  s1 = sc[8444 + i] | sc[8572 + i];\n"
  "  s2 = sc[8706 + i] | sc[8836 + i];\n"
  "  s3 = sc[7916 + i] | sc[7914 + i];\n"
  "  s4 = s0 & s1;\n"
  "  s1 = s2 & s3;\n"
  "  s3 = ~s4 | s1;\n"
  "  return s3;\n"
  "}\n"
  "\n"
  "static inline uint32_t conv_oc_codegen_17(device const uint *sc, int i) {\n"
  "  uint32_t s0, s1;\n"
  "  s0 = sc[9104 + i] & sc[9236 + i];\n"
  "  s1 = ~s0 | sc[8972 + i];\n"
  "  return s1;\n"
  "}\n"
  "\n"
  "static inline uint32_t conv_oc_codegen_18(device const uint *sc, int i) {\n"
  "  uint32_t s0, s1;\n"
  "  s0 = sc[1444 + i] | sc[9368 + i];\n"
  "  s1 = s0 | sc[9364 + i];\n"
  "  s0 = ~s1;\n"
  "  return s0;\n"
  "}\n"
  "\n"
  "static inline uint32_t conv_oc_codegen_19(device const uint *sc, int i) {\n"
  "  uint32_t s0, s1, s2;\n"
  "  s0 = sc[9500 + i] & sc[9496 + i];\n"
  "  s1 = sc[9628 + i] | sc[9630 + i];\n"
  "  s2 = s0 | ~s1;\n"
  "  s1 = sc[3820 + i] & s2;\n"
  "  return s1;\n"
  "}\n"
  "\n"
  "static inline uint32_t conv_oc_codegen_20(device const uint *sc, int i) {\n"
  "  uint32_t s0, s1, s2;\n"
  "  s0 = sc[9762 + i] | ~sc[1052 + i];\n"
  "  s1 = sc[9894 + i] | ~sc[7122 + i];\n"
  "  s2 = ~s0 | s1;\n"
  "  return s2;\n"
  "}\n"
  "\n"
  "static inline uint32_t conv_oc_codegen_21(device const uint *sc, int i) {\n"
  "  uint32_t s0, s1, s2, s3;\n"
  "  s0 = sc[10552 + i] & sc[10684 + i];\n"
  "  s1 = ~(sc[10024 + i] ^ sc[10816 + i]);\n"
  "  s2 = sc[10024 + i] & sc[10422 + i];\n"
  "  s3 = s0 & s1;\n"
  "  s1 = ~(s2 ^ s3);\n"
  "  return s1;\n"
  "}\n"
  "\n"
  "static inline uint32_t conv_oc_codegen_22(device const uint *sc, int i) {\n"
  "  uint32_t s0, s1, s2;\n"
  "  s0 = sc[10952 + i] & sc[3292 + i];\n"
  "  s1 = sc[11082 + i] & sc[11214 + i];\n"
  "  s2 = ~(s0 & sc[10948 + i]);\n"
  "  s0 = s1 & sc[11346 + i];\n"
  "  s1 = ~s2 | s0;\n"
  "  return s1;\n"
  "}\n"
  "\n"
  "static inline uint32_t conv_oc_codegen_23(device const uint *sc, int i) {\n"
  "  uint32_t s0, s1;\n"
  "  s0 = sc[5800 + i] | sc[11608 + i];\n"
  "  s1 = s0 & sc[10156 + i];\n"
  "  return s1;\n"
  "}\n"
  "\n"
  "static inline uint32_t conv_oc_codegen_24(device const uint *sc, int i) {\n"
  "  uint32_t s0, s1, s2;\n"
  "  s0 = sc[11742 + i] | ~sc[11872 + i];\n"
  "  s1 = sc[2240 + i] | ~sc[5140 + i];\n"
  "  s2 = s0 & s1;\n"
  "  return s2;\n"
  "}\n"
  "\n"
  "static inline uint32_t conv_oc_codegen_25(device const uint *sc, int i) {\n"
  "  uint32_t s0, s1, s2;\n"
  "  s0 = sc[5672 + i] & sc[5668 + i];\n"
  "  s1 = sc[12270 + i] ^ sc[12402 + i];\n"
  "  s2 = s0 | ~s1;\n"
  "  return s2;\n"
  "}\n"
  "\n"
  "static inline uint32_t conv_oc_codegen_26(device const uint *sc, int i) {\n"
  "  uint32_t s0, s1;\n"
  "  s0 = sc[8704 + i] | sc[12536 + i];\n"
  "  s1 = s0 & sc[12668 + i];\n"
  "  return s1;\n"
  "}\n"
  "\n"
  "static inline uint32_t conv_oc_codegen_27(device const uint *sc, int i) {\n"
  "  uint32_t s0, s1, s2, s3, s4;\n"
  "  s0 = ~sc[5408 + i] | sc[11612 + i];\n"
  "  s1 = sc[6460 + i] | sc[12800 + i];\n"
  "  s2 = sc[12932 + i] | sc[11480 + i];\n"
  "  s3 = sc[5008 + i] | sc[5408 + i];\n"
  "  s4 = s0 | ~s1;\n"
  "  s1 = s2 & s3;\n"
  "  s3 = s4 & ~s1;\n"
  "  return s3;\n"
  "}\n"
  "\n"
  "static inline uint32_t conv_oc_codegen_28(device const uint *sc, int i) {\n"
  "  uint32_t s0, s1;\n"
  "  s0 = sc[13192 + i] | sc[13196 + i];\n"
  "  s1 = ~(s0 & sc[13064 + i]);\n"
  "  return s1;\n"
  "}\n"
  "\n"
  "static inline uint32_t conv_oc_codegen_29(device const uint *sc, int i) {\n"
  "  uint32_t s0, s1, s2, s3;\n"
  "  s0 = sc[13324 + i] | ~sc[13458 + i];\n"
  "  s1 = sc[4480 + i] & sc[13588 + i];\n"
  "  s2 = sc[8572 + i] & sc[4616 + i];\n"
  "  s3 = s1 & s2;\n"
  "  s2 = ~(s0 ^ s3);\n"
  "  return s2;\n"
  "}\n"
  "\n"
  "static inline uint32_t conv_oc_codegen_30(device const uint *sc, int i) {\n"
  "  uint32_t s0, s1;\n"
  "  s0 = sc[6464 + i] & sc[13856 + i];\n"
  "  s1 = ~s0;\n"
  "  return s1;\n"
  "}\n"
  "\n"
  "static inline uint32_t conv_oc_codegen_31(device const uint *sc, int i) {\n"
  "  uint32_t s0;\n"
  "  s0 = ~sc[10024 + i] | sc[11212 + i];\n"
  "  return s0;\n"
  "}\n"
  "\n"
  "static inline uint32_t conv_oc_codegen_32(device const uint *sc, int i) {\n"
  "  uint32_t s0, s1, s2;\n"
  "  s0 = sc[14120 + i] & ~sc[14250 + i];\n"
  "  s1 = ~sc[11214 + i] & sc[14380 + i];\n"
  "  s2 = s0 | s1;\n"
  "  s1 = s2 & sc[14646 + i];\n"
  "  return s1;\n"
  "}\n"
  "\n"
  "static inline uint32_t conv_oc_codegen_33(device const uint *sc, int i) {\n"
  "  uint32_t s0, s1;\n"
  "  s0 = sc[14910 + i] & sc[15040 + i];\n"
  "  s1 = s0 | sc[6330 + i];\n"
  "  return s1;\n"
  "}\n"
  "\n"
  "static inline uint32_t conv_oc_codegen_34(device const uint *sc, int i) {\n"
  "\n"
  "  return sc[15172 + i];\n"
  "}\n"
  "\n"
  "static inline uint32_t conv_oc_codegen_35(device const uint *sc, int i) {\n"
  "\n"
  "  return sc[15306 + i];\n"
  "}\n"
  "\n"
  "static inline uint32_t conv_oc_codegen_36(device const uint *sc, int i) {\n"
  "  uint32_t s0, s1, s2;\n"
  "  s0 = sc[4084 + i] | sc[15436 + i];\n"
  "  s1 = sc[6332 + i] | sc[4878 + i];\n"
  "  s2 = ~s0 & s1;\n"
  "  return s2;\n"
  "}\n"
  "\n"
  "static inline uint32_t conv_oc_codegen_37(device const uint *sc, int i) {\n"
  "  uint32_t s0, s1, s2, s3;\n"
  "  s0 = sc[15568 + i] | sc[15700 + i];\n"
  "  s1 = sc[15834 + i] & ~sc[15964 + i];\n"
  "  s2 = sc[13724 + i] & sc[11744 + i];\n"
  "  s3 = s0 | s1;\n"
  "  s1 = ~s2 & sc[15700 + i];\n"
  "  s2 = s3 & s1;\n"
  "  return s2;\n"
  "}\n"
  "\n"
  "static inline uint32_t conv_oc_codegen_38(device const uint *sc, int i) {\n"
  "  uint32_t s0, s1;\n"
  "  s0 = sc[4084 + i] & sc[4088 + i];\n"
  "  s1 = ~s0 & sc[16098 + i];\n"
  "  s0 = s1 & sc[14776 + i];\n"
  "  return s0;\n"
  "}\n"
  "\n"
  "static inline uint32_t conv_oc_codegen_39(device const uint *sc, int i) {\n"
  "  uint32_t s0;\n"
  "  s0 = sc[10288 + i] & sc[16360 + i];\n"
  "  return s0;\n"
  "}\n"
  "\n"
  "static inline uint32_t conv_oc_codegen_40(device const uint *sc, int i) {\n"
  "  uint32_t s0, s1, s2;\n"
  "  s0 = sc[652 + i] | sc[11608 + i];\n"
  "  s1 = sc[2372 + i] | sc[16230 + i];\n"
  "  s2 = s0 & ~sc[16496 + i];\n"
  "  s0 = ~(s1 ^ s2);\n"
  "  return s0;\n"
  "}\n"
  "\n"
  "static inline uint32_t conv_oc_codegen_41(device const uint *sc, int i) {\n"
  "  uint32_t s0;\n"
  "  s0 = sc[16628 + i] & sc[16760 + i];\n"
  "  return s0;\n"
  "}\n"
  "\n"
  "static inline uint32_t conv_oc_codegen_42(device const uint *sc, int i) {\n"
  "  uint32_t s0, s1, s2, s3;\n"
  "  s0 = sc[7652 + i] & sc[17022 + i];\n"
  "  s1 = sc[3556 + i] & sc[17156 + i];\n"
  "  s2 = ~(sc[1974 + i] ^ sc[3162 + i]);\n"
  "  s3 = ~s0 & s1;\n"
  "  s1 = sc[3560 + i] & s2;\n"
  "  s2 = ~(s3 & s1);\n"
  "  return s2;\n"
  "}\n"
  "\n"
  "static inline uint32_t conv_oc_codegen_43(device const uint *sc, int i) {\n"
  "  uint32_t s0, s1;\n"
  "  s0 = sc[11480 + i] & sc[13196 + i];\n"
  "  s1 = s0 | sc[17288 + i];\n"
  "  return s1;\n"
  "}\n"
  "\n"
  "static inline uint32_t conv_oc_codegen_44(device const uint *sc, int i) {\n"
  "  uint32_t s0, s1;\n"
  "  s0 = sc[784 + i] | sc[4352 + i];\n"
  "  s1 = ~s0 & sc[17420 + i];\n"
  "  return s1;\n"
  "}\n"
  "\n"
  "static inline uint32_t conv_oc_codegen_45(device const uint *sc, int i) {\n"
  "  uint32_t s0, s1;\n"
  "  s0 = sc[1446 + i] | sc[4612 + i];\n"
  "  s1 = s0 | sc[15836 + i];\n"
  "  return s1;\n"
  "}\n"
  "\n"
  "static inline uint32_t conv_oc_codegen_46(device const uint *sc, int i) {\n"
  "  uint32_t s0, s1, s2;\n"
  "  s0 = sc[17684 + i] & sc[16756 + i];\n"
  "  s1 = sc[13458 + i] & sc[17814 + i];\n"
  "  s2 = s0 | ~s1;\n"
  "  return s2;\n"
  "}\n"
  "\n"
  "static inline uint32_t conv_oc_codegen_47(device const uint *sc, int i) {\n"
  "  uint32_t s0, s1, s2, s3;\n"
  "  s0 = sc[12932 + i] | sc[12930 + i];\n"
  "  s1 = ~sc[5276 + i] & sc[11476 + i];\n"
  "  s2 = sc[11476 + i] & sc[17946 + i];\n"
  "  s3 = s0 ^ s1;\n"
  "  s1 = s2 | s3;\n"
  "  return s1;\n"
  "}\n"
  "\n"
  "static inline uint32_t conv_oc_codegen_48(device const uint *sc, int i) {\n"
  "  uint32_t s0, s1, s2, s3;\n"
  "  s0 = sc[4876 + i] & sc[18076 + i];\n"
  "  s1 = ~sc[4084 + i] & sc[18080 + i];\n"
  "  s2 = sc[6724 + i] & sc[1048 + i];\n"
  "  s3 = s0 | s1;\n"
  "  s1 = sc[5672 + i] | s2;\n"
  "  s2 = ~(s3 ^ s1);\n"
  "  return s2;\n"
  "}\n"
  "\n"
  "static inline uint32_t conv_oc_codegen_49(device const uint *sc, int i) {\n"
  "  uint32_t s0, s1;\n"
  "  s0 = sc[4616 + i] | sc[18208 + i];\n"
  "  s1 = s0 | sc[15832 + i];\n"
  "  return s1;\n"
  "}\n"
  "\n"
  "static inline uint32_t conv_oc_codegen_50(device const uint *sc, int i) {\n"
  "  uint32_t s0, s1;\n"
  "  s0 = sc[3556 + i] & sc[3952 + i];\n"
  "  s1 = sc[15436 + i] | s0;\n"
  "  s0 = ~s1;\n"
  "  return s0;\n"
  "}\n"
  "\n"
  "static inline uint32_t conv_oc_codegen_51(device const uint *sc, int i) {\n"
  "  uint32_t s0;\n"
  "  s0 = sc[18340 + i] | sc[18476 + i];\n"
  "  return s0;\n"
  "}\n"
  "\n"
  "static inline uint32_t conv_oc_codegen_52(device const uint *sc, int i) {\n"
  "  uint32_t s0, s1, s2;\n"
  "  s0 = sc[11872 + i] | sc[16624 + i];\n"
  "  s1 = sc[9628 + i] & ~sc[12668 + i];\n"
  "  s2 = sc[15700 + i] | s0;\n"
  "  s0 = s1 | s2;\n"
  "  return s0;\n"
  "}\n"
  "\n"
  "static inline uint32_t conv_oc_codegen_53(device const uint *sc, int i) {\n"
  "  uint32_t s0, s1, s2;\n"
  "  s0 = sc[2104 + i] | sc[15964 + i];\n"
  "  s1 = sc[1048 + i] | sc[18608 + i];\n"
  "  s2 = ~(s0 | s1);\n"
  "  return s2;\n"
  "}\n"
  "\n"
  "static inline uint32_t conv_oc_codegen_54(device const uint *sc, int i) {\n"
  "  uint32_t s0, s1, s2;\n"
  "  s0 = ~sc[14516 + i] & sc[18740 + i];\n"
  "  s1 = sc[14780 + i] & sc[9102 + i];\n"
  "  s2 = s0 | s1;\n"
  "  return s2;\n"
  "}\n"
  "\n"
  "static inline uint32_t conv_oc_codegen_55(device const uint *sc, int i) {\n"
  "  uint32_t s0;\n"
  "  s0 = sc[18872 + i] | sc[18870 + i];\n"
  "  return s0;\n"
  "}\n"
  "\n"
  "static inline uint32_t conv_oc_codegen_56(device const uint *sc, int i) {\n"
  "  uint32_t s0, s1, s2;\n"
  "  s0 = sc[11480 + i] | sc[19000 + i];\n"
  "  s1 = sc[19132 + i] & sc[10290 + i];\n"
  "  s2 = sc[11478 + i] & s0;\n"
  "  s0 = s1 | sc[11476 + i];\n"
  "  s1 = s2 | s0;\n"
  "  return s1;\n"
  "}\n"
  "\n"
  "static inline uint32_t conv_oc_codegen_57(device const uint *sc, int i) {\n"
  "  uint32_t s0, s1, s2, s3, s4;\n"
  "  s0 = sc[15832 + i] & sc[17020 + i];\n"
  "  s1 = sc[9100 + i] | sc[19264 + i];\n"
  "  s2 = ~sc[3956 + i] & sc[17286 + i];\n"
  "  s3 = sc[3556 + i] | sc[11740 + i];\n"
  "  s4 = s0 & s1;\n"
  "  s1 = s2 | ~s3;\n"
  "  s3 = s4 ^ s1;\n"
  "  return s3;\n"
  "}\n"
  "\n"
  "static inline uint32_t conv_oc_codegen_58(device const uint *sc, int i) {\n"
  "  uint32_t s0, s1;\n"
  "  s0 = sc[14382 + i] & sc[7780 + i];\n"
  "  s1 = ~s0 & sc[19398 + i];\n"
  "  return s1;\n"
  "}\n"
  "\n"
  "static inline uint32_t conv_oc_codegen_59(device const uint *sc, int i) {\n"
  "  uint32_t s0, s1, s2;\n"
  "  s0 = ~sc[17552 + i] | sc[19532 + i];\n"
  "  s1 = sc[6332 + i] | sc[15572 + i];\n"
  "  s2 = ~(s0 & s1);\n"
  "  s1 = ~s2;\n"
  "  return s1;\n"
  "}\n"
  "\n"
  "static inline uint32_t conv_oc_codegen_60(device const uint *sc, int i) {\n"
  "  uint32_t s0, s1, s2, s3, s4;\n"
  "  s0 = sc[15572 + i] & sc[19660 + i];\n"
  "  s1 = sc[8048 + i] | sc[13988 + i];\n"
  "  s2 = ~(sc[12004 + i] ^ sc[12136 + i]);\n"
  "  s3 = ~(sc[19792 + i] ^ sc[19928 + i]);\n"
  "  s4 = s0 | ~s1;\n"
  "  s1 = s2 & s3;\n"
  "  s3 = ~(s4 ^ s1);\n"
  "  return s3;\n"
  "}\n"
  "\n"
  "static inline uint32_t conv_oc_codegen_61(device const uint *sc, int i) {\n"
  "  uint32_t s0, s1;\n"
  "  s0 = sc[5274 + i] & sc[20058 + i];\n"
  "  s1 = s0 | sc[16096 + i];\n"
  "  return s1;\n"
  "}\n"
  "\n"
  "static inline uint32_t conv_oc_codegen_62(device const uint *sc, int i) {\n"
  "  uint32_t s0, s1, s2;\n"
  "  s0 = sc[1708 + i] & ~sc[20192 + i];\n"
  "  s1 = sc[20320 + i] | sc[20324 + i];\n"
  "  s2 = ~s0 & s1;\n"
  "  return s2;\n"
  "}\n"
  "\n"
  "static inline uint32_t conv_oc_codegen_63(device const uint *sc, int i) {\n"
  "  uint32_t s0, s1, s2;\n"
  "  s0 = sc[16888 + i] & sc[20452 + i];\n"
  "  s1 = sc[9628 + i] & sc[15308 + i];\n"
  "  s2 = s0 | sc[20452 + i];\n"
  "  s0 = s1 | s2;\n"
  "  return s0;\n"
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
  "    case 32: v = conv_oc_codegen_32(scratch, i); break;\n"
  "    case 33: v = conv_oc_codegen_33(scratch, i); break;\n"
  "    case 34: v = conv_oc_codegen_34(scratch, i); break;\n"
  "    case 35: v = conv_oc_codegen_35(scratch, i); break;\n"
  "    case 36: v = conv_oc_codegen_36(scratch, i); break;\n"
  "    case 37: v = conv_oc_codegen_37(scratch, i); break;\n"
  "    case 38: v = conv_oc_codegen_38(scratch, i); break;\n"
  "    case 39: v = conv_oc_codegen_39(scratch, i); break;\n"
  "    case 40: v = conv_oc_codegen_40(scratch, i); break;\n"
  "    case 41: v = conv_oc_codegen_41(scratch, i); break;\n"
  "    case 42: v = conv_oc_codegen_42(scratch, i); break;\n"
  "    case 43: v = conv_oc_codegen_43(scratch, i); break;\n"
  "    case 44: v = conv_oc_codegen_44(scratch, i); break;\n"
  "    case 45: v = conv_oc_codegen_45(scratch, i); break;\n"
  "    case 46: v = conv_oc_codegen_46(scratch, i); break;\n"
  "    case 47: v = conv_oc_codegen_47(scratch, i); break;\n"
  "    case 48: v = conv_oc_codegen_48(scratch, i); break;\n"
  "    case 49: v = conv_oc_codegen_49(scratch, i); break;\n"
  "    case 50: v = conv_oc_codegen_50(scratch, i); break;\n"
  "    case 51: v = conv_oc_codegen_51(scratch, i); break;\n"
  "    case 52: v = conv_oc_codegen_52(scratch, i); break;\n"
  "    case 53: v = conv_oc_codegen_53(scratch, i); break;\n"
  "    case 54: v = conv_oc_codegen_54(scratch, i); break;\n"
  "    case 55: v = conv_oc_codegen_55(scratch, i); break;\n"
  "    case 56: v = conv_oc_codegen_56(scratch, i); break;\n"
  "    case 57: v = conv_oc_codegen_57(scratch, i); break;\n"
  "    case 58: v = conv_oc_codegen_58(scratch, i); break;\n"
  "    case 59: v = conv_oc_codegen_59(scratch, i); break;\n"
  "    case 60: v = conv_oc_codegen_60(scratch, i); break;\n"
  "    case 61: v = conv_oc_codegen_61(scratch, i); break;\n"
  "    case 62: v = conv_oc_codegen_62(scratch, i); break;\n"
  "    case 63: v = conv_oc_codegen_63(scratch, i); break;\n"
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
static const int32_t kOcInstrStart[] = {0,4,7,9,12,17,19,21,24,27,30,34,38,42,46,49,50,57,59,62,66,69,74,79,81,84,87,89,96,98,103,105,106,110,112,112,112,115,121,124,125,129,130,136,138,140,142,145,150,156,158,161,162,166,169,172,173,178,185,187,191,198,200,203,207};
static const int32_t kInstrFlat[] = {9,516,780,0,14,1044,1308,1,11,1,3,2,5,5,5,1,8,1840,2100,0,14,2364,2624,1,11,1,3,2,11,3156,3420,0,9,1,3680,1,11,5008,5536,0,14,1,5272,1,9,3,5008,0,11,5792,6060,0,14,6592,6848,1,8,1,3944,2,8,3,6320,0,9,5,1,1,9,7120,7112,0,8,1,7376,1,8,8432,2888,0,5,1,1,1,11,8696,8968,0,14,9224,9496,1,11,1,3,2,8,10280,10548,0,8,11076,2896,1,1,1,3,2,14,12128,12392,0,8,1,11864,1,5,3,3,0,8,4208,13184,0,14,5264,5000,1,4,1,3,2,5,5,5,1,8,3156,11600,0,8,3416,13720,1,14,3,13976,2,11,1,5,1,11,6328,14240,0,9,14504,14768,1,9,3952,1,2,8,3,5,0,8,15036,15296,0,8,15304,15568,1,14,1,3,2,5,5,5,1,14,7904,14776,0,8,1,14512,1,13,3,14504,0,9,15832,16352,0,14,10812,16616,0,14,16888,17144,1,14,17412,17672,2,14,15832,15828,3,8,1,3,4,8,5,7,1,13,9,3,3,8,18208,18472,0,13,1,17944,1,14,2888,18736,0,14,1,18728,1,5,3,3,0,8,19000,18992,0,14,19256,19260,1,11,1,3,2,8,7640,5,1,11,19524,2104,0,11,19788,14244,1,13,1,3,2,8,21104,21368,0,9,20048,21632,1,8,20048,20844,2,8,1,3,3,9,5,7,1,8,21904,6584,0,8,22164,22428,1,7,1,21896,2,8,3,22692,0,13,5,1,1,14,11600,23216,0,8,1,20312,1,11,23484,23744,0,11,4480,10280,1,8,1,3,2,8,11344,11336,0,6,24540,24804,1,11,1,3,2,14,17408,25072,0,8,1,25336,1,13,10816,23224,0,14,12920,25600,1,14,25864,22960,2,14,10016,10816,3,11,1,3,4,8,5,7,1,2,9,3,3,14,26384,26392,0,7,1,26128,1,11,26648,26916,0,8,8960,27176,1,8,17144,9232,2,8,3,5,3,9,1,7,2,8,12928,27712,0,5,1,1,1,13,20048,22424,0,2,28240,28500,0,4,22428,28760,1,14,1,3,2,8,5,29292,1,8,29820,30080,0,14,1,12660,1,14,8168,30872,0,14,12664,9756,1,4,1,3,2,14,31136,31400,0,2,31668,31928,1,8,27448,23488,2,14,1,3,3,4,5,31400,1,8,7,3,2,8,8168,8176,0,4,1,32196,1,8,3,29552,0,8,20576,32720,0,14,1304,23216,0,14,4744,32460,1,2,1,32992,2,9,3,5,0,8,33256,33520,0,8,15304,34044,0,8,7112,34312,1,9,3948,6324,2,4,1,3,3,8,7120,5,1,7,7,3,2,8,22960,26392,0,14,1,34576,1,14,1568,8704,0,4,1,34840,1,14,2892,9224,0,14,1,31672,1,8,35368,33512,0,8,26916,35628,1,11,1,3,2,14,25864,25860,0,4,10552,22952,1,8,22952,35892,2,6,1,3,3,14,5,7,1,8,9752,36152,0,4,8168,36160,1,8,13448,2096,2,14,1,3,3,14,11344,5,1,9,7,3,2,14,9232,36416,0,14,1,31664,1,8,7112,7904,0,14,30872,1,1,5,3,3,0,14,36680,36952,0,14,23744,33248,0,2,19256,25336,1,14,31400,1,2,14,3,5,0,14,4208,31928,0,14,2096,37216,1,1,1,3,2,4,29032,37480,0,8,29560,18204,1,14,1,3,2,14,37744,37740,0,14,22960,38000,0,8,38264,20580,1,8,22956,1,2,14,3,22952,0,14,5,1,1,8,31664,34040,0,14,18200,38528,1,4,7912,34572,2,14,7112,23480,3,8,1,3,4,11,5,7,1,6,9,3,3,8,28764,15560,0,4,1,38796,1,13,35104,39064,0,14,12664,31144,1,7,1,3,2,5,5,5,1,8,31144,39320,0,14,16096,27976,1,9,24008,24272,2,9,39584,39856,3,11,1,3,4,8,5,7,1,9,9,3,3,8,10548,40116,0,14,1,32192,1,2,3416,40384,0,14,40640,40648,1,4,1,3,2,8,33776,40904,0,8,19256,30616,1,14,1,40904,2,14,3,5,0};
static const int32_t kOcFinalRef[] = {3,5,3,1,3,3,3,5,5,1,3,3,1,3,1,1,7,3,1,3,5,3,3,3,5,5,3,7,3,5,3,1,3,3,30344,30612,5,5,1,1,1,1,5,3,3,3,5,3,5,3,1,1,1,5,5,1,3,7,3,3,7,3,5,1};

constexpr int kOutChannels = 64;
constexpr int kTapRowsWords = 128;
constexpr int kWprOut = 2;
constexpr uint32_t kMaskOut = 4294967295u;
constexpr int kScratchWords = 20584;

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
    std::printf("model=seg_convdlgn_k32 node=n11_lutconv2d out_channels=%d "
               "tap_rows_words=%d n_instr_total=207 "
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
