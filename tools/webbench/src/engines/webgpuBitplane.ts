// DLGN WebGPU「ワードパック(ビットプレーン)常駐」executor。
//
// webgpuGraph.ts(W10)は 1bit を 1個の i32 に入れて持ち(帯域32倍浪費)、
// 1スレッド=1出力ビットで動いていた。timestamp-query分解で
// det/detmono/detface は「GPUカーネル律速(ΣGPUの8割が最初の2〜3個のconv2d)」、
// detreal/seg は「床律速(submit→mapAsyncの固定床、パス数13〜16)」と判明した。
// このファイルはその両方を同時に潰す:
//
//   (a) ワードパック化: CPU側 bitplane エンジン(bitplane.ts / bitplaneGraph.ts)の
//       レイアウト契約([batch][C][H][wpr] の32bitワード列、行頭ワード境界、
//       wpr=(W+31)>>5、LSB-first、端数ビット(x>=W)は常に0が不変条件)を
//       GPUバッファへそのまま採用し、1スレッド=1出力ワード(=32空間位置)にする。
//       conv2dのリーフ取得は行ワードのシフト合成
//       ((row[j]>>>s)|(row[j+1]<<(32-s)))でWGSL内に閉じる(タップ平面の
//       実体化は不要=bitplane.ts の buildSharedShiftPlane と同じ値をレジスタ上で
//       直接作る)。木の縮約は u32 レジスタ上の gate2 連鎖。
//
//   (b) 単一 compute pass 化 + 配線ノードの融合: 全ノードのdispatchを1つの
//       beginComputePass にまとめる(WebGPUはpass内のdispatch間に実装が
//       メモリバリアを挿入する契約なので、storage buffer経由の依存は保たれる)。
//       さらに「配線しかしないノード」を隣接カーネルへ融合してパス数を減らす:
//         - 入力の pack(CPUではなくGPUで実行)+ 先頭のチャネルpermute
//         - permute([0,2,3,1]) + group_sum(SpatialGroupSum)
//         - concat: 生産者ノードが最初から連結先バッファのチャネル/要素
//           オフセットへ直接書く(=dispatch 0個。View{bufId,cOff}抽象)
//
// 正しさの錨: CPU側 bitplane エンジン(bitplaneGraph.ts::evalIrBitplane、
// これ自体が conv_golden/seg_e2e_golden/det_e2e_golden と整数一致済み)と
// 全画素・全チャネルの生スコアで整数完全一致すること。node環境ではWebGPUを
// 実行できないため、実行時一致検証は wasmgpu_verify.html(Playwright)と
// scratchpadの一致確認スクリプトが正式な経路で、vitestは「プラン構築ロジック
// (幾何・パス数・融合判定・タップ表の詰め方)」を検証する。
//
// 未対応構成(lut_layer / add / binarize / stride>1のconv2d / scale!=2の
// upsample2d / 2x2以外のor_pool2d / 出力がビットプレーンのまま等)は黙って
// 誤った結果を返さず必ず throw する。呼び出し側(conv.ts)は従来の
// packed-webgpu へフォールバックする。
import { ttParse } from '../shared';
import type {
  Conv2dParams,
  GroupSumParams,
  IRGraphJson,
  IRNodeJson,
  OrPool2dParams,
  PermuteParams,
  UpSample2dParams,
} from '../shared';
import {
  REF_KIND_INSTR, buildConvProgram, makeRef, refIdx, refKind, verifyConvProgram,
} from './convProgram';
import { collectInputSources, topologicalOrder } from './irflat';
import type { NodeValue } from './irflat';
import { buildPreprocDescriptors, buildPreprocPackShader } from './gpuPreproc';
import type { ImagePreprocDoc } from '../convpreproc';
import { GATE2_WGSL, dispatch2d } from './webgpuGraph';
import type { WebgpuBreakdown, WebgpuCpuBreakdown, WebgpuNodeTiming } from './webgpuGraph';

// ============================================================
// 0. レイアウト契約のヘルパー(bitplane.ts と同一定義。あちらは別エージェントが
//    改修中のため import せず、契約だけを写して同値性をテストで固定する)
// ============================================================

/** 1行に必要なワード数(=ceil(W/32))。bitplane.ts::wordsPerRow と同一。 */
export function wordsPerRow(W: number): number {
  return (W + 31) >>> 5;
}

/** 行末尾ワードの有効ビットマスク(W%32===0なら全ビット有効)。u32として返す
    (bitplane.ts::tailMask は i32 の -1 を返すのでそこだけ表現が違う)。 */
export function tailMaskU32(W: number): number {
  const r = W & 31;
  return r === 0 ? 0xffffffff : ((1 << r) - 1) >>> 0;
}

function pair(v: number | number[]): [number, number] {
  return Array.isArray(v) ? [v[0], v[1]] : [v, v];
}

/** conv2dリーフ1個ぶんの「どのワードをどうシフトして読むか」を1つのu32へ詰める。
    bit 0..15 : 入力チャネル c
    bit 16..20: bitShift bs(0..31)
    bit 21..26: dyRel+32(dyRel = dy - pH、出力行 oy に対する入力行の相対)
    bit 27..31: wordShift+16(ws = floor((dx-pW)/32))
    範囲外は黙って切り詰めず throw する(この関数はテスト対象)。 */
export function packTap(c: number, dyRel: number, ws: number, bs: number): number {
  if (!Number.isInteger(c) || c < 0 || c > 0xffff) {
    throw new Error(`packTap: チャネル c=${c} が16bitに収まらない`);
  }
  if (!Number.isInteger(bs) || bs < 0 || bs > 31) {
    throw new Error(`packTap: bitShift bs=${bs} が[0,31]外`);
  }
  if (!Number.isInteger(dyRel) || dyRel < -32 || dyRel > 31) {
    throw new Error(`packTap: dyRel=${dyRel} が[-32,31]外`);
  }
  if (!Number.isInteger(ws) || ws < -16 || ws > 15) {
    throw new Error(`packTap: wordShift ws=${ws} が[-16,15]外`);
  }
  return (
    ((c & 0xffff) | (bs << 16) | ((dyRel + 32) << 21) | ((ws + 16) << 27)) >>> 0
  );
}

/** packTap の逆(テスト用。WGSL側のデコードと同じ式)。 */
export function unpackTap(t: number): { c: number; bs: number; dyRel: number; ws: number } {
  return {
    c: t & 0xffff,
    bs: (t >>> 16) & 31,
    dyRel: ((t >>> 21) & 63) - 32,
    ws: ((t >>> 27) & 31) - 16,
  };
}

/** 水平シフト量 s を wordShift/bitShift へ分解する(bitplane.ts::shiftRowInto と
    同じ floor 分解: wordShift*32 + bitShift === s、bitShift ∈ [0,32))。 */
export function splitShift(s: number): { ws: number; bs: number } {
  const ws = Math.floor(s / 32);
  return { ws, bs: s - ws * 32 };
}

// ============================================================
// 1. WGSL 共通断片
// ============================================================

const PARAMS_WGSL = `
struct Params { total: u32, xthreads: u32, _p1: u32, _p2: u32 };
@group(0) @binding(0) var<uniform> params: Params;
`;

const GIDX_WGSL = `  let gidx = gid.y * params.xthreads + gid.x;
  if (gidx >= params.total) { return; }`;

/** 偶数位置ビット(0,2,...,30)を下位16bitへ圧縮(bitplane.ts::compressEven32Shift)。 */
const COMPRESS_EVEN_WGSL = `
fn compressEven(v: u32) -> u32 {
  var x = v & 0x55555555u;
  x = (x | (x >> 1u)) & 0x33333333u;
  x = (x | (x >> 2u)) & 0x0f0f0f0fu;
  x = (x | (x >> 4u)) & 0x00ff00ffu;
  x = (x | (x >> 8u)) & 0x0000ffffu;
  return x;
}
`;

/** 下位16bitを偶数位置へ拡散(bitplane.ts::spreadEven16)。 */
const SPREAD_EVEN_WGSL = `
fn spreadEven(v: u32) -> u32 {
  var x = v & 0xffffu;
  x = (x | (x << 8u)) & 0x00ff00ffu;
  x = (x | (x << 4u)) & 0x0f0f0f0fu;
  x = (x | (x << 2u)) & 0x33333333u;
  x = (x | (x << 1u)) & 0x55555555u;
  return x;
}
`;

// ============================================================
// 2. プラン(GPUデバイス非依存の純データ)
// ============================================================

/** GPUバッファ1本ぶんの割り当て情報。bytesPerSample はバッチ1要素あたり。 */
export interface PlanBuffer {
  label: string;
  bytesPerSample: number;
}

/** dispatch 1個ぶん。bindingは 0=Params(uniform)、1..=入力バッファ(read)、
    その後=静的データ(read)、最後=出力バッファ(read_write)で固定。 */
export interface PlanCompute {
  id: string;
  nodeType: string;
  code: string;
  staticArrays: Uint32Array[];
  inputBufs: number[];
  outputBuf: number;
  /** バッチ1要素あたりのスレッド数(総スレッド数 = batch * これ)。 */
  threadsPerSample: number;
}

/** ビットプレーン値のビュー(バッファ内のチャネル範囲を指す)。concat融合で
    cOff!=0 になりうる。sampleWords はバッファ側(PlanBuffer)から引く。 */
interface BpView {
  kind: 'bp';
  bufId: number;
  cOff: number;
  C: number;
  H: number;
  W: number;
  wpr: number;
}

/** 整数(非パック)値のビュー。group_sum以降で使う。 */
interface DenseView {
  kind: 'dense';
  bufId: number;
  off: number;
  width: number;
}

/** permute([0,2,3,1]) を group_sum と融合するための保留状態(dispatchを持たない)。 */
interface PendingSgsView {
  kind: 'pending-sgs';
  src: BpView;
  n: number;
  g: number;
}

type View = BpView | DenseView | PendingSgsView;

export interface BitplanePlan {
  buffers: PlanBuffer[];
  computes: PlanCompute[];
  /** 生ビット列(0/1のバイト列)をそのまま載せる入力バッファ。 */
  inputBufId: number;
  inputWidth: number;
  inputBytesPerSample: number;
  output: { bufId: number; off: number; width: number; total: number };
  /** 何をどう融合したか(報告・テスト用の可読メモ)。 */
  notes: string[];
  /** computes 内で入力packを行う dispatch の位置(常に0)。
      GPU前処理モードではこの1個だけが preproc に差し替わる。 */
  packComputeIndex: number;
  /** RGBAを直接受けてビットプレーンを作るGPU前処理カーネル
      (computes[packComputeIndex] の代替)。preproc未指定/未対応なら null。 */
  preproc: PlanCompute | null;
  /** RGBA(u8×4/画素)入力バッファ。preproc===null なら -1。*/
  preprocInputBufId: number;
  /** バッチ1要素あたりのRGBAバイト数(=H*W*4)。*/
  preprocBytesPerSample: number;
  /** GPU前処理を組めなかった理由(黙殺しない)。組めたときは null。*/
  gpuPreprocNote: string | null;
}

// ============================================================
// 3. ノード型別のWGSL生成
// ============================================================

interface BpBind {
  cOff: number;
  C: number;
  H: number;
  W: number;
  wpr: number;
  sampleWords: number;
}

/** 入力ビット列(0/1バイト列)→ ビットプレーン。CPUでpackPlanes相当をやると
    detfaceで2.8M要素のJSループになるため、GPUで実行する(転送も1バイト/ビットの
    まま=従来のi32アップロード比1/4、writeBuffer 1.01ms対策)。
    先頭がチャネルpermuteのモデル(detreal/seg の n0_chanperm)は
    channelOrder を渡してここへ融合する(dispatch 1個ぶんの削減)。 */
function buildPackShader(
  inWidth: number, inStride: number, out: BpBind, channelOrder: Int32Array | null,
): { code: string; staticArrays: Uint32Array[]; threadsPerSample: number } {
  const { C, H, W, wpr } = out;
  const hasOrder = channelOrder !== null;
  const orderDecl = hasOrder
    ? '@group(0) @binding(2) var<storage, read> chanOrder: array<u32>;'
    : '';
  const outBinding = hasOrder ? 3 : 2;
  const srcChan = hasOrder ? 'chanOrder[o]' : 'o';
  const code = `${PARAMS_WGSL}
@group(0) @binding(1) var<storage, read> rawBits: array<u32>;
${orderDecl}
@group(0) @binding(${outBinding}) var<storage, read_write> outBuf: array<u32>;

const C: u32 = ${C}u;
const H: u32 = ${H}u;
const W: u32 = ${W}u;
const WPR: u32 = ${wpr}u;
const IN_WIDTH: u32 = ${inWidth}u;
const IN_STRIDE: u32 = ${inStride}u;
const OUT_C_OFF: u32 = ${out.cOff}u;
const OUT_SAMPLE: u32 = ${out.sampleWords}u;
const TPS: u32 = ${C * H * wpr}u;
const TAIL: u32 = ${tailMaskU32(W)}u;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
${GIDX_WGSL}
  let b = gidx / TPS;
  let rem = gidx % TPS;
  let o = rem / (H * WPR);
  let r2 = rem % (H * WPR);
  let y = r2 / WPR;
  let jw = r2 % WPR;

  let rowBase = b * IN_STRIDE + (${srcChan}) * H * W + y * W;
  let x0 = jw * 32u;
  var v: u32 = 0u;
  for (var i: u32 = 0u; i < 32u; i = i + 1u) {
    let xx = x0 + i;
    if (xx < W) {
      let fi = rowBase + xx;
      let byte = (rawBits[fi >> 2u] >> ((fi & 3u) * 8u)) & 0xffu;
      if (byte != 0u) { v = v | (1u << i); }
    }
  }
  if (jw == WPR - 1u) { v = v & TAIL; }
  outBuf[b * OUT_SAMPLE + (OUT_C_OFF + o) * H * WPR + y * WPR + jw] = v;
}
`;
  const statics: Uint32Array[] = hasOrder ? [Uint32Array.from(channelOrder!)] : [];
  return { code, staticArrays: statics, threadsPerSample: C * H * wpr };
}

/** conv2d(n_inputs=2、stride=1、WOut===W の高速路のみ)。1スレッド=1出力ワード。
    リーフ取得は bitplane.ts::buildSharedShiftPlane + shiftRowInto と同一の値を
    レジスタ上で直接合成する(タップ平面を実体化しない)。木の縮約は
    packedConv.ts / bitplane.ts と同じ gate2 の16式(GATE2_WGSLを共有)。
    出力行末尾のみ tailMask を適用する(端数ビット=0の不変条件の維持。途中level
    の端数ビットが汚れても、ゲートはビット位置をまたがないので実位置に影響しない)。 */
function buildConvShader(
  p: Conv2dParams, inb: BpBind, out: BpBind,
): { code: string; staticArrays: Uint32Array[]; threadsPerSample: number } {
  const [C, H, W] = p.in_shape;
  const [kH, kW] = pair(p.kernel_size);
  const [sH, sW] = pair(p.stride);
  const [pH, pW] = pair(p.padding);
  const HOut = Math.floor((H + 2 * pH - kH) / sH) + 1;
  const WOut = Math.floor((W + 2 * pW - kW) / sW) + 1;
  if (p.n_inputs !== 2) {
    throw new Error(`bitplane-webgpu conv2d は n_inputs=2 のみ対応(実際: ${p.n_inputs})`);
  }
  if (sH !== 1 || sW !== 1 || WOut !== W) {
    throw new Error(
      `bitplane-webgpu conv2d は stride=1 かつ WOut===W のみ対応`
      + `(stride=${sH},${sW} W=${W} WOut=${WOut})`,
    );
  }
  if (inb.C !== C || inb.H !== H || inb.W !== W) {
    throw new Error(
      `bitplane-webgpu conv2d: 入力形状不一致(期待 C=${C} H=${H} W=${W}, `
      + `実際 C=${inb.C} H=${inb.H} W=${inb.W})`,
    );
  }
  const OC = p.out_channels;
  const depth = p.tree_depth;
  const nLeaves = 2 ** depth;
  const nInternal = nLeaves - 1;
  const khw = kH * kW;
  const idxf = p.indices_flat;

  // --- タップ表(oc-major、リーフ順)。値は (c, dyRel, ws, bs) の詰め物。 ---
  const tap = new Uint32Array(OC * nLeaves);
  for (let oc = 0; oc < OC; oc++) {
    const row = idxf[oc];
    for (let k = 0; k < nLeaves; k++) {
      const ppos = row[k];
      const rem = ppos % khw;
      const c = Math.floor(ppos / khw);
      const dy = Math.floor(rem / kW);
      const dx = rem % kW;
      const { ws, bs } = splitShift(dx - pW);
      tap[oc * nLeaves + k] = packTap(c, dy - pH, ws, bs);
    }
  }
  const ttNum = new Uint32Array(OC * nInternal);
  for (let i = 0; i < ttNum.length; i++) {
    ttNum[i] = Number(ttParse(p.truth_tables[i])) & 0xf;
  }

  // --- 深い木は「ocごとの簡約済み命令列」で回す ---
  //
  // 背景: 下の完全展開は nLeaves 個の葉を全部読み、nInternal 個のゲートを
  // 全部評価する。深さ3(8葉/7ゲート)なら全部レジスタに載って速いが、
  // 6入力LUTをShannon分解した深さ9(512葉/511ゲート)では、実質6本しか
  // 入力が無いのにパディングまで評価してしまう。実測でカラー検出の webgpu が
  // wasm の7倍遅くなっていた原因がこれ(GPUが 1,602,496 ゲート、CPUは簡約後
  // 60,853 命令を実行していた)。
  //
  // そこで CPU 側(bitplane.ts::buildConvProgramLayout)と同じ buildConvProgram を
  // ocごとに掛け、簡約後の命令列だけをGPUに渡す。ocごとに分けるのは、
  // 全oc共通の命令列にするとスレッド=(oy,jw) の288本まで並列度が落ちるため。
  // oc単位なら (oc,oy,jw) の並列度を保ったまま命令数だけ減らせる
  // (oc間CSEは失うが、実測でヒットは全体の0.02%以下なので損失は無視できる)。
  if (nLeaves > PROGRAM_MIN_LEAVES) {
    return buildConv2dProgramShader(p, inb, out, {
      OC, depth, nInternal, kH, kW, pH, pW, H, HOut, WOut, idxf,
    });
  }

  // --- 木を構築時に完全展開する(array<u32,N>の動的添字を避けレジスタに載せる) ---
  const lines: string[] = [];
  for (let k = 0; k < nLeaves; k++) {
    lines.push(`  var v${k} = leafWord(xb, tap[lb + ${k}u], oy, jw);`);
  }
  let cur = Array.from({ length: nLeaves }, (_, k) => `v${k}`);
  let gateOffset = 0;
  let level = 0;
  while (cur.length > 1) {
    const next: string[] = [];
    for (let g = 0; g < cur.length / 2; g++) {
      const name = `t${level}_${g}`;
      lines.push(
        `  let ${name} = gate2(ttNum[ocBase + ${gateOffset + g}u], ${cur[2 * g]}, ${cur[2 * g + 1]});`,
      );
      next.push(name);
    }
    gateOffset += cur.length / 2;
    cur = next;
    level++;
  }
  const resultVar = cur[0];

  const code = `${PARAMS_WGSL}
@group(0) @binding(1) var<storage, read> xBuf: array<u32>;
@group(0) @binding(2) var<storage, read> tap: array<u32>;
@group(0) @binding(3) var<storage, read> ttNum: array<u32>;
@group(0) @binding(4) var<storage, read_write> outBuf: array<u32>;

${GATE2_WGSL}

const H_I: i32 = ${H};
const WPR_IN: u32 = ${inb.wpr}u;
const WPR_IN_I: i32 = ${inb.wpr};
const IN_PLANE: u32 = ${H * inb.wpr}u;
const IN_C_OFF: u32 = ${inb.cOff}u;
const IN_SAMPLE: u32 = ${inb.sampleWords}u;

const HOUT: u32 = ${HOut}u;
const WPR: u32 = ${out.wpr}u;
const OUT_C_OFF: u32 = ${out.cOff}u;
const OUT_SAMPLE: u32 = ${out.sampleWords}u;
const N_LEAVES: u32 = ${nLeaves}u;
const N_INTERNAL: u32 = ${nInternal}u;
const TPS: u32 = ${OC * HOut * out.wpr}u;
const TAIL: u32 = ${tailMaskU32(WOut)}u;

/** タップ1個ぶんの入力ワード。bitplane.ts::shiftRowInto の
    dst[j] = (at(j)>>>bs) | (at(j+1)<<(32-bs)) を、行外ワード=0・範囲外行=0の
    境界処理込みで再現する。bs===0 のとき (bb<<31)<<1 == 0 になるので
    「32ビットシフト(WGSLでは未定義)」を踏まずに1本の式で書ける。 */
fn leafWord(xb: u32, t: u32, oy: u32, jw: u32) -> u32 {
  let c = t & 0xffffu;
  let bs = (t >> 16u) & 31u;
  let dyRel = i32((t >> 21u) & 63u) - 32;
  let ws = i32((t >> 27u) & 31u) - 16;
  let y = i32(oy) + dyRel;
  if (y < 0 || y >= H_I) { return 0u; }
  let rowBase = xb + (IN_C_OFF + c) * IN_PLANE + u32(y) * WPR_IN;
  let j0 = i32(jw) + ws;
  var a: u32 = 0u;
  if (j0 >= 0 && j0 < WPR_IN_I) { a = xBuf[rowBase + u32(j0)]; }
  var bb: u32 = 0u;
  let j1 = j0 + 1;
  if (j1 >= 0 && j1 < WPR_IN_I) { bb = xBuf[rowBase + u32(j1)]; }
  return (a >> bs) | ((bb << (31u - bs)) << 1u);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
${GIDX_WGSL}
  let b = gidx / TPS;
  let rem = gidx % TPS;
  let oc = rem / (HOUT * WPR);
  let r2 = rem % (HOUT * WPR);
  let oy = r2 / WPR;
  let jw = r2 % WPR;
  let xb = b * IN_SAMPLE;
  let lb = oc * N_LEAVES;
  let ocBase = oc * N_INTERNAL;

${lines.join('\n')}

  var outv = ${resultVar};
  if (jw == WPR - 1u) { outv = outv & TAIL; }
  outBuf[b * OUT_SAMPLE + (OUT_C_OFF + oc) * (HOUT * WPR) + oy * WPR + jw] = outv;
}
`;
  return { code, staticArrays: [tap, ttNum], threadsPerSample: OC * HOut * out.wpr };
}

/** 完全展開をやめて命令列で回す閾値(葉の数)。深さ4(16葉)までは
    展開してレジスタに載せた方が速い。深さ5以上(32葉〜)は命令列にする。
    6入力LUTのShannon分解は深さ9(512葉)なのでここに入る。 */
const PROGRAM_MIN_LEAVES = 16;

interface ProgShaderCtx {
  OC: number; depth: number; nInternal: number;
  kH: number; kW: number; pH: number; pW: number;
  H: number; HOut: number; WOut: number;
  idxf: ArrayLike<ArrayLike<number>>;
}

/** ocごとに buildConvProgram を掛け、簡約後の命令列を実行するWGSLを組む。
 *
 * バッファ構成(すべて u32):
 *   tapVals      : ocごとのユニークタップを連結(値は packTap 済み)
 *   ocTapStart   : oc → tapVals の開始位置
 *   instrBuf     : 命令1個 = 4語 [tt, aRef, bRef, dstSlot]
 *   ocInstr      : oc → [開始命令index, 命令数, 出力Ref] の3語
 *
 * Ref の規約は convProgram.ts と同一(下位2bit=kind、上位=idx)。
 * kind 0=定数(idx 0/1)、1=タップ(oc内index)、2=命令の出力(dstSlotへ解決済み)。
 */
function buildConv2dProgramShader(
  p: Conv2dParams, inb: BpBind, out: BpBind, cx: ProgShaderCtx,
): { code: string; staticArrays: Uint32Array[]; threadsPerSample: number } {
  const { OC, depth, nInternal, kH, kW, pH, pW, H, HOut, WOut, idxf } = cx;
  const khw = kH * kW;

  const ttAll = new Array<number>(OC * nInternal);
  for (let i = 0; i < ttAll.length; i++) {
    ttAll[i] = Number(ttParse(p.truth_tables[i])) & 0xf;
  }

  const tapVals: number[] = [];
  const ocTapStart = new Uint32Array(OC);
  const instrFlat: number[] = [];
  const ocInstr = new Uint32Array(OC * 3);
  let maxSlots = 1;

  for (let oc = 0; oc < OC; oc++) {
    const prog = buildConvProgram({
      outChannels: 1,
      treeDepth: depth,
      indicesFlat: [idxf[oc]],
      truthTables: ttAll.slice(oc * nInternal, (oc + 1) * nInternal),
    });
    verifyConvProgram(
      {
        outChannels: 1,
        treeDepth: depth,
        indicesFlat: [idxf[oc]],
        truthTables: ttAll.slice(oc * nInternal, (oc + 1) * nInternal),
      },
      prog,
    );

    ocTapStart[oc] = tapVals.length;
    for (const ppos of prog.taps) {
      const rem = ppos % khw;
      const c = Math.floor(ppos / khw);
      const dy = Math.floor(rem / kW);
      const dx = rem % kW;
      const { ws, bs } = splitShift(dx - pW);
      tapVals.push(packTap(c, dy - pH, ws, bs));
    }

    // 命令オペランドの INSTR 参照は「命令index」なので、実行時に読める
    // 「スロット番号」へ解決しておく(CPU側がオフセットへ解決するのと同じ)。
    const toSlotRef = (r: number): number => (
      refKind(r) === REF_KIND_INSTR
        ? makeRef(REF_KIND_INSTR, prog.instrs[refIdx(r)].slot)
        : r
    );
    ocInstr[oc * 3] = instrFlat.length / 4;
    ocInstr[oc * 3 + 1] = prog.instrs.length;
    ocInstr[oc * 3 + 2] = toSlotRef(prog.ocOut[0]);
    for (const ins of prog.instrs) {
      instrFlat.push(ins.tt, toSlotRef(ins.a), toSlotRef(ins.b), ins.slot);
    }
    if (prog.nSlots > maxSlots) maxSlots = prog.nSlots;
  }

  const resolve = (ref: string) => `
    {
      let k = ${ref} & 3u;
      let ix = ${ref} >> 2u;
      if (k == 2u) { v = slots[ix]; }
      else if (k == 1u) { v = leafWord(xb, tapVals[tapBase + ix], oy, jw); }
      else if (ix == 1u) { v = 0xffffffffu; }
      else { v = 0u; }
    }`;

  const code = `${PARAMS_WGSL}
@group(0) @binding(1) var<storage, read> xBuf: array<u32>;
@group(0) @binding(2) var<storage, read> tapVals: array<u32>;
@group(0) @binding(3) var<storage, read> ocTapStart: array<u32>;
@group(0) @binding(4) var<storage, read> instrBuf: array<u32>;
@group(0) @binding(5) var<storage, read> ocInstr: array<u32>;
@group(0) @binding(6) var<storage, read_write> outBuf: array<u32>;

${GATE2_WGSL}

const H_I: i32 = ${H};
const WPR_IN: u32 = ${inb.wpr}u;
const WPR_IN_I: i32 = ${inb.wpr};
const IN_PLANE: u32 = ${H * inb.wpr}u;
const IN_C_OFF: u32 = ${inb.cOff}u;
const IN_SAMPLE: u32 = ${inb.sampleWords}u;

const HOUT: u32 = ${HOut}u;
const WPR: u32 = ${out.wpr}u;
const OUT_C_OFF: u32 = ${out.cOff}u;
const OUT_SAMPLE: u32 = ${out.sampleWords}u;
const TPS: u32 = ${OC * HOut * out.wpr}u;
const TAIL: u32 = ${tailMaskU32(WOut)}u;
const MAX_SLOTS: u32 = ${maxSlots}u;

fn leafWord(xb: u32, t: u32, oy: u32, jw: u32) -> u32 {
  let c = t & 0xffffu;
  let bs = (t >> 16u) & 31u;
  let dyRel = i32((t >> 21u) & 63u) - 32;
  let ws = i32((t >> 27u) & 31u) - 16;
  let y = i32(oy) + dyRel;
  if (y < 0 || y >= H_I) { return 0u; }
  let rowBase = xb + (IN_C_OFF + c) * IN_PLANE + u32(y) * WPR_IN;
  let j0 = i32(jw) + ws;
  var a: u32 = 0u;
  if (j0 >= 0 && j0 < WPR_IN_I) { a = xBuf[rowBase + u32(j0)]; }
  var bb: u32 = 0u;
  let j1 = j0 + 1;
  if (j1 >= 0 && j1 < WPR_IN_I) { bb = xBuf[rowBase + u32(j1)]; }
  return (a >> bs) | ((bb << (31u - bs)) << 1u);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
${GIDX_WGSL}
  let b = gidx / TPS;
  let rem = gidx % TPS;
  let oc = rem / (HOUT * WPR);
  let r2 = rem % (HOUT * WPR);
  let oy = r2 / WPR;
  let jw = r2 % WPR;
  let xb = b * IN_SAMPLE;

  var slots: array<u32, ${maxSlots}>;
  let tapBase = ocTapStart[oc];
  let iStart = ocInstr[oc * 3u];
  let iCount = ocInstr[oc * 3u + 1u];
  let oRef = ocInstr[oc * 3u + 2u];

  for (var i: u32 = 0u; i < iCount; i = i + 1u) {
    let o = (iStart + i) * 4u;
    let tt = instrBuf[o];
    let ar = instrBuf[o + 1u];
    let br = instrBuf[o + 2u];
    let ds = instrBuf[o + 3u];
    var v: u32 = 0u;${resolve('ar')}
    let av = v;${resolve('br')}
    let bv = v;
    slots[ds] = gate2(tt, av, bv);
  }

  var v: u32 = 0u;${resolve('oRef')}
  var outv = v;
  if (jw == WPR - 1u) { outv = outv & TAIL; }
  outBuf[b * OUT_SAMPLE + (OUT_C_OFF + oc) * (HOUT * WPR) + oy * WPR + jw] = outv;
}
`;
  return {
    code,
    staticArrays: [
      Uint32Array.from(tapVals), ocTapStart,
      Uint32Array.from(instrFlat), ocInstr,
    ],
    threadsPerSample: OC * HOut * out.wpr,
  };
}

/** or_pool2d(kernel=2,stride=2,padding=0)。bitplane.ts::orPool2x2 の
    「縦OR → 水平隣接OR → 偶数ビット圧縮」カスケードをそのまま移植する。
    1スレッド=1出力ワード(=32出力位置)、必要な入力ワードは t[2w],t[2w+1],t[2w+2]。 */
function buildOrPoolShader(
  inb: BpBind, out: BpBind,
): { code: string; staticArrays: Uint32Array[]; threadsPerSample: number } {
  const code = `${PARAMS_WGSL}
@group(0) @binding(1) var<storage, read> xBuf: array<u32>;
@group(0) @binding(2) var<storage, read_write> outBuf: array<u32>;

${COMPRESS_EVEN_WGSL}

const WPR_IN: u32 = ${inb.wpr}u;
const IN_PLANE: u32 = ${inb.H * inb.wpr}u;
const IN_C_OFF: u32 = ${inb.cOff}u;
const IN_SAMPLE: u32 = ${inb.sampleWords}u;

const HOUT: u32 = ${out.H}u;
const WPR: u32 = ${out.wpr}u;
const OUT_C_OFF: u32 = ${out.cOff}u;
const OUT_SAMPLE: u32 = ${out.sampleWords}u;
const TPS: u32 = ${out.C * out.H * out.wpr}u;
const TAIL: u32 = ${tailMaskU32(out.W)}u;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
${GIDX_WGSL}
  let b = gidx / TPS;
  let rem = gidx % TPS;
  let c = rem / (HOUT * WPR);
  let r2 = rem % (HOUT * WPR);
  let oy = r2 / WPR;
  let wo = r2 % WPR;

  let rowA = b * IN_SAMPLE + (IN_C_OFF + c) * IN_PLANE + (2u * oy) * WPR_IN;
  let rowB = rowA + WPR_IN;
  let j0 = 2u * wo;
  var t0: u32 = 0u;
  var t1: u32 = 0u;
  var t2: u32 = 0u;
  if (j0 < WPR_IN) { t0 = xBuf[rowA + j0] | xBuf[rowB + j0]; }
  if (j0 + 1u < WPR_IN) { t1 = xBuf[rowA + j0 + 1u] | xBuf[rowB + j0 + 1u]; }
  if (j0 + 2u < WPR_IN) { t2 = xBuf[rowA + j0 + 2u] | xBuf[rowB + j0 + 2u]; }
  // u[j] = t[j] | (t[j]>>>1) | (t[j+1]<<31)  (bitplane.ts::orPoolRow と同一)
  let u0 = t0 | (t0 >> 1u) | (t1 << 31u);
  var u1: u32 = 0u;
  if (j0 + 1u < WPR_IN) { u1 = t1 | (t1 >> 1u) | (t2 << 31u); }
  var v = compressEven(u0) | (compressEven(u1) << 16u);
  if (wo == WPR - 1u) { v = v & TAIL; }
  outBuf[b * OUT_SAMPLE + (OUT_C_OFF + c) * (HOUT * WPR) + oy * WPR + wo] = v;
}
`;
  return { code, staticArrays: [], threadsPerSample: out.C * out.H * out.wpr };
}

/** upsample2d(scale=2, nearest)。bitplane.ts::dilateX2 の移植
    (16bit半語 → spreadEven16 → 2ビット複製、同じ内容を2行へ)。 */
function buildUpsampleShader(
  inb: BpBind, out: BpBind,
): { code: string; staticArrays: Uint32Array[]; threadsPerSample: number } {
  const code = `${PARAMS_WGSL}
@group(0) @binding(1) var<storage, read> xBuf: array<u32>;
@group(0) @binding(2) var<storage, read_write> outBuf: array<u32>;

${SPREAD_EVEN_WGSL}

const WPR_IN: u32 = ${inb.wpr}u;
const IN_PLANE: u32 = ${inb.H * inb.wpr}u;
const IN_C_OFF: u32 = ${inb.cOff}u;
const IN_SAMPLE: u32 = ${inb.sampleWords}u;

const HOUT: u32 = ${out.H}u;
const WPR: u32 = ${out.wpr}u;
const OUT_C_OFF: u32 = ${out.cOff}u;
const OUT_SAMPLE: u32 = ${out.sampleWords}u;
const TPS: u32 = ${out.C * out.H * out.wpr}u;
const TAIL: u32 = ${tailMaskU32(out.W)}u;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
${GIDX_WGSL}
  let b = gidx / TPS;
  let rem = gidx % TPS;
  let c = rem / (HOUT * WPR);
  let r2 = rem % (HOUT * WPR);
  let oy = r2 / WPR;
  let ow = r2 % WPR;

  let srcIdx = ow >> 1u;
  var srcWord: u32 = 0u;
  if (srcIdx < WPR_IN) {
    srcWord = xBuf[b * IN_SAMPLE + (IN_C_OFF + c) * IN_PLANE + (oy >> 1u) * WPR_IN + srcIdx];
  }
  var half: u32 = (srcWord >> 16u) & 0xffffu;
  if ((ow & 1u) == 0u) { half = srcWord & 0xffffu; }
  let even = spreadEven(half);
  var v = even | (even << 1u);
  if (ow == WPR - 1u) { v = v & TAIL; }
  outBuf[b * OUT_SAMPLE + (OUT_C_OFF + c) * (HOUT * WPR) + oy * WPR + ow] = v;
}
`;
  return { code, staticArrays: [], threadsPerSample: out.C * out.H * out.wpr };
}

/** チャネル並べ替えのみの permute(bitplane.ts::copyPlanes)。1スレッド=1ワード。 */
function buildCopyPlanesShader(
  inb: BpBind, out: BpBind, order: Int32Array,
): { code: string; staticArrays: Uint32Array[]; threadsPerSample: number } {
  const code = `${PARAMS_WGSL}
@group(0) @binding(1) var<storage, read> xBuf: array<u32>;
@group(0) @binding(2) var<storage, read> chanOrder: array<u32>;
@group(0) @binding(3) var<storage, read_write> outBuf: array<u32>;

const PLANE: u32 = ${out.H * out.wpr}u;
const IN_C_OFF: u32 = ${inb.cOff}u;
const IN_SAMPLE: u32 = ${inb.sampleWords}u;
const OUT_C_OFF: u32 = ${out.cOff}u;
const OUT_SAMPLE: u32 = ${out.sampleWords}u;
const TPS: u32 = ${out.C * out.H * out.wpr}u;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
${GIDX_WGSL}
  let b = gidx / TPS;
  let rem = gidx % TPS;
  let o = rem / PLANE;
  let r = rem % PLANE;
  outBuf[b * OUT_SAMPLE + (OUT_C_OFF + o) * PLANE + r] =
    xBuf[b * IN_SAMPLE + (IN_C_OFF + chanOrder[o]) * PLANE + r];
}
`;
  return {
    code,
    staticArrays: [Uint32Array.from(order)],
    threadsPerSample: out.C * out.H * out.wpr,
  };
}

/** SpatialGroupSum(permute[0,2,3,1] + group_sum の融合)。
    入力ビットプレーン [n*g][H][W] から、出力 [(n*H+y)*W+x] = Σ_g bit(n*g+gi, y, x)。
    bitplaneGraph.ts が「permuteでアンパック → irflat::evalGroupSum」で得る値と
    同一(permute は out[n,y,x,gi]=in[n,gi,y,x]、group_sumは gsz=g の連続和)。 */
function buildSpatialGroupSumShader(
  inb: BpBind, n: number, g: number, out: DenseView, outTotal: number,
): { code: string; staticArrays: Uint32Array[]; threadsPerSample: number } {
  const { H, W, wpr } = inb;
  const code = `${PARAMS_WGSL}
@group(0) @binding(1) var<storage, read> xBuf: array<u32>;
@group(0) @binding(2) var<storage, read_write> outBuf: array<i32>;

const H: u32 = ${H}u;
const W: u32 = ${W}u;
const WPR: u32 = ${wpr}u;
const PLANE: u32 = ${H * wpr}u;
const G: u32 = ${g}u;
const IN_C_OFF: u32 = ${inb.cOff}u;
const IN_SAMPLE: u32 = ${inb.sampleWords}u;
const OUT_W: u32 = ${n * H * W}u;
const OUT_OFF: u32 = ${out.off}u;
const OUT_TOTAL: u32 = ${outTotal}u;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
${GIDX_WGSL}
  let b = gidx / OUT_W;
  let rem = gidx % OUT_W;
  let ni = rem / (H * W);
  let p = rem % (H * W);
  let y = p / W;
  let x = p % W;
  let bit = x & 31u;
  var base = b * IN_SAMPLE + (IN_C_OFF + ni * G) * PLANE + y * WPR + (x >> 5u);
  var s: i32 = 0;
  for (var gi: u32 = 0u; gi < G; gi = gi + 1u) {
    s = s + i32((xBuf[base] >> bit) & 1u);
    base = base + PLANE;
  }
  outBuf[b * OUT_TOTAL + OUT_OFF + rem] = s;
}
`;
  return { code, staticArrays: [], threadsPerSample: n * H * W };
}

/** 非パック(整数)入力に対する素朴な group_sum(irflat.ts::evalGroupSum と同一)。
    SpatialGroupSum融合に載らない group_sum が来た場合の正しさの保険。 */
function buildDenseGroupSumShader(
  inv: DenseView, inTotal: number, k: number, out: DenseView, outTotal: number,
): { code: string; staticArrays: Uint32Array[]; threadsPerSample: number } {
  if (inv.width % k !== 0) throw new Error(`group_sum: 幅${inv.width}がk=${k}で割り切れない`);
  const gsz = inv.width / k;
  const code = `${PARAMS_WGSL}
@group(0) @binding(1) var<storage, read> xBuf: array<i32>;
@group(0) @binding(2) var<storage, read_write> outBuf: array<i32>;

const K: u32 = ${k}u;
const GSZ: u32 = ${gsz}u;
const IN_OFF: u32 = ${inv.off}u;
const IN_TOTAL: u32 = ${inTotal}u;
const OUT_OFF: u32 = ${out.off}u;
const OUT_TOTAL: u32 = ${outTotal}u;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
${GIDX_WGSL}
  let b = gidx / K;
  let g = gidx % K;
  let base = b * IN_TOTAL + IN_OFF + g * GSZ;
  var s: i32 = 0;
  for (var i: u32 = 0u; i < GSZ; i = i + 1u) { s = s + xBuf[base + i]; }
  outBuf[b * OUT_TOTAL + OUT_OFF + g] = s;
}
`;
  return { code, staticArrays: [], threadsPerSample: k };
}

/** concat をエイリアス(生産者が直接連結先へ書く)にできないときのコピーカーネル。
    ビットプレーン版: チャネル方向の連結なので、出力ワード索引から
    「どの入力のどのワードか」を段階的に引き算して決める。 */
function buildBpConcatShader(
  ins: BpBind[], out: BpBind,
): { code: string; staticArrays: Uint32Array[]; threadsPerSample: number } {
  const plane = out.H * out.wpr;
  const bindings = ins
    .map((_, i) => `@group(0) @binding(${i + 1}) var<storage, read> in${i}: array<u32>;`)
    .join('\n');
  const outBinding =
    `@group(0) @binding(${ins.length + 1}) var<storage, read_write> outBuf: array<u32>;`;
  const consts = ins
    .map((v, i) =>
      `const N${i}: u32 = ${v.C * plane}u;\n`
      + `const B${i}: u32 = ${v.cOff * plane}u;\n`
      + `const S${i}: u32 = ${v.sampleWords}u;`)
    .join('\n');
  const branches: string[] = ['  var pos = rem;'];
  ins.forEach((_, i) => {
    if (i < ins.length - 1) {
      branches.push(`  if (pos < N${i}) { outBuf[dst] = in${i}[b * S${i} + B${i} + pos]; return; }`);
      branches.push(`  pos = pos - N${i};`);
    } else {
      branches.push(`  outBuf[dst] = in${i}[b * S${i} + B${i} + pos];`);
    }
  });
  const code = `${PARAMS_WGSL}
${bindings}
${outBinding}

${consts}
const PLANE: u32 = ${plane}u;
const OUT_C_OFF: u32 = ${out.cOff}u;
const OUT_SAMPLE: u32 = ${out.sampleWords}u;
const TPS: u32 = ${out.C * plane}u;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
${GIDX_WGSL}
  let b = gidx / TPS;
  let rem = gidx % TPS;
  let dst = b * OUT_SAMPLE + OUT_C_OFF * PLANE + rem;
${branches.join('\n')}
}
`;
  return { code, staticArrays: [], threadsPerSample: out.C * plane };
}

/** 整数値のconcat(irflat.ts::evalConcat と同一のスロット順連結)のコピーカーネル。 */
function buildDenseConcatShader(
  ins: { view: DenseView; total: number }[], out: DenseView, outTotal: number,
): { code: string; staticArrays: Uint32Array[]; threadsPerSample: number } {
  const bindings = ins
    .map((_, i) => `@group(0) @binding(${i + 1}) var<storage, read> in${i}: array<i32>;`)
    .join('\n');
  const outBinding =
    `@group(0) @binding(${ins.length + 1}) var<storage, read_write> outBuf: array<i32>;`;
  const consts = ins
    .map((v, i) =>
      `const N${i}: u32 = ${v.view.width}u;\n`
      + `const O${i}: u32 = ${v.view.off}u;\n`
      + `const S${i}: u32 = ${v.total}u;`)
    .join('\n');
  const branches: string[] = ['  var pos = rem;'];
  ins.forEach((_, i) => {
    if (i < ins.length - 1) {
      branches.push(`  if (pos < N${i}) { outBuf[dst] = in${i}[b * S${i} + O${i} + pos]; return; }`);
      branches.push(`  pos = pos - N${i};`);
    } else {
      branches.push(`  outBuf[dst] = in${i}[b * S${i} + O${i} + pos];`);
    }
  });
  const code = `${PARAMS_WGSL}
${bindings}
${outBinding}

${consts}
const OUT_OFF: u32 = ${out.off}u;
const OUT_TOTAL: u32 = ${outTotal}u;
const TPS: u32 = ${out.width}u;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
${GIDX_WGSL}
  let b = gidx / TPS;
  let rem = gidx % TPS;
  let dst = b * OUT_TOTAL + OUT_OFF + rem;
${branches.join('\n')}
}
`;
  return { code, staticArrays: [], threadsPerSample: out.width };
}

// ============================================================
// 4. プラン構築(純関数、GPU不要)
// ============================================================

/** permuteが「末尾2軸(H,W)不変」の純チャネル並べ替えか
    (bitplaneGraph.ts::permuteKeepsTrailingSpatial と同一判定)。 */
function permuteKeepsTrailingSpatial(p: PermuteParams): boolean {
  const n = p.in_shape.length;
  if (n < 2) return false;
  return p.perm[n - 2] === n - 2 && p.perm[n - 1] === n - 1;
}

/** チャネル軸だけのindex変換をorder配列へ展開する
    (bitplaneGraph.ts::channelOrderFromPermute と同一の式)。 */
function channelOrderFromPermute(p: PermuteParams): Int32Array {
  const n = p.in_shape.length;
  const m = n - 2;
  const leadDims = p.in_shape.slice(0, m);
  const leadPerm = p.perm.slice(0, m);
  const total = leadDims.reduce((a, b) => a * b, 1);
  const inStrides = new Array<number>(m).fill(1);
  for (let i = m - 2; i >= 0; i--) inStrides[i] = inStrides[i + 1] * leadDims[i + 1];
  const outDims = leadPerm.map((q) => leadDims[q]);
  const outStrides = new Array<number>(m).fill(1);
  for (let i = m - 2; i >= 0; i--) outStrides[i] = outStrides[i + 1] * outDims[i + 1];
  const order = new Int32Array(total);
  for (let outIdx = 0; outIdx < total; outIdx++) {
    let rem = outIdx;
    let src = 0;
    for (let i = 0; i < m; i++) {
      const coord = Math.floor(rem / outStrides[i]);
      rem -= coord * outStrides[i];
      src += coord * inStrides[leadPerm[i]];
    }
    order[outIdx] = src;
  }
  return order;
}

/** 入力ポートの (C,H,W) を最初の消費ノードのin_shapeから決める。 */
function inferInputGeom(node: IRNodeJson, inputWidth: number): { C: number; H: number; W: number } {
  const t = node.type as string;
  let C: number;
  let H: number;
  let W: number;
  if (t === 'conv2d' || t === 'or_pool2d' || t === 'upsample2d') {
    const shape = (node.params as unknown as { in_shape: number[] }).in_shape;
    [C, H, W] = shape;
  } else if (t === 'permute') {
    const dims = (node.params as unknown as PermuteParams).in_shape;
    const n = dims.length;
    H = dims[n - 2];
    W = dims[n - 1];
    C = dims.slice(0, n - 2).reduce((a, b) => a * b, 1);
  } else {
    throw new Error(`bitplane-webgpu: 入力ポートの直後が空間ノードでない(${node.id}: ${t})`);
  }
  if (C * H * W !== inputWidth) {
    throw new Error(
      `bitplane-webgpu: 入力幅${inputWidth} != C*H*W=${C}*${H}*${W}(${node.id})`,
    );
  }
  return { C, H, W };
}

/** IR(lowered、nativeConv=true / nativeOrPool=true)を「単一compute passで
    流せるdispatch列」へ落とす。GPUデバイスに一切触らない純関数なので、
    node環境のvitestで幾何・パス数・融合判定を検証できる。 */
export function planBitplaneGraph(
  ir: IRGraphJson, preproc?: ImagePreprocDoc | null,
): BitplanePlan {
  if (ir.inputs.length !== 1) {
    throw new Error(`bitplane-webgpu は単一入力ポートのみ対応(実際 ${ir.inputs.length})`);
  }
  const inputId = ir.inputs[0].id;
  const inputWidth = ir.inputs[0].width;

  const inSrcs = collectInputSources(ir);
  const topo = topologicalOrder(ir, inSrcs);

  const buffers: PlanBuffer[] = [];
  const bufferUsed: boolean[] = [];
  const notes: string[] = [];
  const newBuf = (label: string, bytesPerSample: number): number => {
    buffers.push({ label, bytesPerSample });
    bufferUsed.push(true);
    return buffers.length - 1;
  };
  const bpBytes = (cTotal: number, H: number, wpr: number): number => cTotal * H * wpr * 4;

  // 入力ポートの生ビット列(0/1バイト列)。4バイト境界へ切り上げる。
  const inputBytesPerSample = Math.ceil(inputWidth / 4) * 4;
  const inputBufId = newBuf('input:rawBits', inputBytesPerSample);

  // 消費者一覧(concatエイリアス可否と入力ポート融合の判定に使う)。
  const consumers = new Map<string, string[]>();
  for (const [dst, srcs] of inSrcs) {
    for (const s of srcs) {
      const list = consumers.get(s) ?? [];
      list.push(dst);
      consumers.set(s, list);
    }
  }

  const firstConsumerId = (consumers.get(inputId) ?? [])[0];
  if (firstConsumerId === undefined) throw new Error('bitplane-webgpu: 入力ポートの消費者が無い');
  const firstConsumer = topo.find((n) => n.id === firstConsumerId)!;
  const geom = inferInputGeom(firstConsumer, inputWidth);
  const inWpr = wordsPerRow(geom.W);

  // --- 入力の pack(+先頭チャネルpermuteの融合)---
  let fusedPermuteId: string | null = null;
  let packOrder: Int32Array | null = null;
  if (
    (firstConsumer.type as string) === 'permute'
    && (consumers.get(inputId) ?? []).length === 1
    && permuteKeepsTrailingSpatial(firstConsumer.params as unknown as PermuteParams)
  ) {
    fusedPermuteId = firstConsumer.id;
    packOrder = channelOrderFromPermute(firstConsumer.params as unknown as PermuteParams);
    notes.push(`pack融合: 先頭チャネルpermute ${fusedPermuteId} を pack カーネルへ取り込み`);
  }

  const views = new Map<string, View>();
  // プランは「①ノード記述の収集 → ②concatエイリアス確定 → ③WGSL生成」の3段。
  // WGSLにはビュー(cOff/sampleWords)が定数として焼き込まれるため、エイリアスで
  // 確定するまでコード生成を遅らせる必要がある。
  interface PendingNode {
    id: string;
    nodeType: string;
    inputViewIds: string[]; // 実際に読むバッファを持つビューのノードid(入力ポート含む)
    emit: () => { code: string; staticArrays: Uint32Array[]; threadsPerSample: number };
    outBufId: number;
  }
  const pending: PendingNode[] = [];
  /** 出力ビューが「自前の専用バッファを丸ごと使っている(cOff=0かつ全チャネル)」
      ノードだけがconcatエイリアスの対象になる。1度エイリアスしたら再利用しない。 */
  const retargeted = new Set<string>();

  const packOutView: BpView = {
    kind: 'bp',
    bufId: newBuf('pack', bpBytes(geom.C, geom.H, inWpr)),
    cOff: 0,
    C: geom.C,
    H: geom.H,
    W: geom.W,
    wpr: inWpr,
  };
  const packOwner = fusedPermuteId ?? `${inputId}__pack`;
  views.set(packOwner, packOutView);
  if (fusedPermuteId === null) views.set(inputId, packOutView);
  pending.push({
    id: packOwner,
    nodeType: 'pack',
    inputViewIds: [],
    outBufId: packOutView.bufId,
    emit: () =>
      buildPackShader(inputWidth, inputBytesPerSample, {
        cOff: packOutView.cOff,
        C: packOutView.C,
        H: packOutView.H,
        W: packOutView.W,
        wpr: packOutView.wpr,
        sampleWords: buffers[packOutView.bufId].bytesPerSample / 4,
      }, packOrder),
  });

  const bpOf = (id: string): BpView => {
    const v = views.get(id);
    if (!v || v.kind !== 'bp') {
      throw new Error(`bitplane-webgpu: ${id} はビットプレーン値ではない(kind=${v?.kind})`);
    }
    return v;
  };
  const denseOf = (id: string): DenseView => {
    const v = views.get(id);
    if (!v || v.kind !== 'dense') {
      throw new Error(`bitplane-webgpu: ${id} は整数値ではない(kind=${v?.kind})`);
    }
    return v;
  };
  const bindOf = (v: BpView): BpBind => ({
    cOff: v.cOff,
    C: v.C,
    H: v.H,
    W: v.W,
    wpr: v.wpr,
    sampleWords: buffers[v.bufId].bytesPerSample / 4,
  });

  /** concatノードを解決する。全ソースが「専用バッファを丸ごと持つ生産ノード」
      なら、ソースの出力ビューを連結先バッファのオフセットへ差し替えて
      dispatchを0個にする(エイリアス。WebGPUはpass内dispatch間にバリアを
      入れるので、複数の生産者が同じバッファの別領域へ書いてから読んでも
      正しい)。条件を満たさない場合はコピーカーネルを1個作る
      (正しさ優先のフォールバック)。 */
  const resolveConcat = (node: IRNodeJson, srcs: string[]): void => {
    if (srcs.length < 2) throw new Error(`bitplane-webgpu concat は2入力以上(${node.id})`);
    const svs = srcs.map((s) => views.get(s));
    if (svs.some((v) => v === undefined)) {
      throw new Error(`bitplane-webgpu concat: 入力未解決(${node.id})`);
    }
    const allBp = svs.every((v) => v!.kind === 'bp');
    const allDense = svs.every((v) => v!.kind === 'dense');

    /** ソースがエイリアス可能か: 専用バッファを丸ごと使っており(cOff/off=0かつ
        バッファ全体がそのノードの出力)、まだ他のconcatへ差し替えられていない、
        かつ「pendingに居る生産ノード」であること(alias転送等は除外)。 */
    const aliasable = (id: string, v: BpView | DenseView): boolean => {
      if (retargeted.has(id)) return false;
      const producer = pending.find((q) => q.id === id);
      if (!producer || producer.outBufId !== v.bufId) return false;
      if (v.kind === 'bp') {
        return v.cOff === 0 && buffers[v.bufId].bytesPerSample === v.C * v.H * v.wpr * 4;
      }
      return v.off === 0 && buffers[v.bufId].bytesPerSample === v.width * 4;
    };

    if (allBp) {
      const bps = svs as BpView[];
      const H = bps[0].H;
      const W = bps[0].W;
      const wpr = bps[0].wpr;
      if (bps.some((v) => v.H !== H || v.W !== W || v.wpr !== wpr)) {
        throw new Error(`bitplane-webgpu concat: 空間形状が揃っていない(${node.id})`);
      }
      const cTotal = bps.reduce((a, v) => a + v.C, 0);
      const canAlias = srcs.every((s, i) => aliasable(s, bps[i]));
      const outBufId = newBuf(node.id, bpBytes(cTotal, H, wpr));
      const out: BpView = { kind: 'bp', bufId: outBufId, cOff: 0, C: cTotal, H, W, wpr };
      views.set(node.id, out);
      if (canAlias) {
        let cOff = 0;
        srcs.forEach((s, i) => {
          const v = bps[i];
          bufferUsed[v.bufId] = false;
          pending.find((q) => q.id === s)!.outBufId = outBufId;
          v.bufId = outBufId;
          v.cOff = cOff;
          retargeted.add(s);
          cOff += v.C;
        });
        notes.push(`concat融合: ${node.id}(${srcs.join(',')})を生産者の直接書き込みへ(dispatch 0)`);
        return;
      }
      pending.push({
        id: node.id,
        nodeType: 'concat',
        inputViewIds: srcs,
        outBufId,
        emit: () => buildBpConcatShader(bps.map((v) => bindOf(v)), bindOf(out)),
      });
      return;
    }

    if (!allDense) {
      throw new Error(`bitplane-webgpu concat: ビットプレーンと整数値の混在(${node.id})`);
    }
    const dvs = svs as DenseView[];
    const total = dvs.reduce((a, v) => a + v.width, 0);
    const canAlias = srcs.every((s, i) => aliasable(s, dvs[i]));
    const outBufId = newBuf(node.id, total * 4);
    const out: DenseView = { kind: 'dense', bufId: outBufId, off: 0, width: total };
    views.set(node.id, out);
    if (canAlias) {
      let off = 0;
      srcs.forEach((s, i) => {
        const v = dvs[i];
        bufferUsed[v.bufId] = false;
        pending.find((q) => q.id === s)!.outBufId = outBufId;
        v.bufId = outBufId;
        v.off = off;
        retargeted.add(s);
        off += v.width;
      });
      notes.push(`concat融合: ${node.id}(${srcs.join(',')})を生産者の直接書き込みへ(dispatch 0)`);
      return;
    }
    pending.push({
      id: node.id,
      nodeType: 'concat',
      inputViewIds: srcs,
      outBufId,
      emit: () =>
        buildDenseConcatShader(
          dvs.map((v) => ({ view: v, total: buffers[v.bufId].bytesPerSample / 4 })),
          out, total,
        ),
    });
  };

  for (const node of topo) {
    if (node.id === fusedPermuteId) continue; // packへ融合済み
    const srcs = inSrcs.get(node.id) ?? [];
    const type = node.type as string;

    switch (type) {
      case 'conv2d': {
        const p = node.params as unknown as Conv2dParams;
        const inv = bpOf(srcs[0]);
        const [, H, W] = p.in_shape;
        const [kH, kW] = pair(p.kernel_size);
        const [pH, pW] = pair(p.padding);
        const HOut = H + 2 * pH - kH + 1;
        const WOut = W + 2 * pW - kW + 1;
        const wprOut = wordsPerRow(WOut);
        const out: BpView = {
          kind: 'bp',
          bufId: newBuf(node.id, bpBytes(p.out_channels, HOut, wprOut)),
          cOff: 0,
          C: p.out_channels,
          H: HOut,
          W: WOut,
          wpr: wprOut,
        };
        views.set(node.id, out);
        pending.push({
          id: node.id,
          nodeType: type,
          inputViewIds: [srcs[0]],
          outBufId: out.bufId,
          emit: () => buildConvShader(p, bindOf(inv), bindOf(out)),
        });
        break;
      }
      case 'or_pool2d': {
        const p = node.params as unknown as OrPool2dParams;
        const pad = p.padding ?? 0;
        if (p.kernel_size !== 2 || p.stride !== 2 || pad !== 0) {
          throw new Error(
            `bitplane-webgpu or_pool2d は kernel=2,stride=2,padding=0 のみ対応`
            + `(${node.id}: k=${p.kernel_size} s=${p.stride} pad=${pad})`,
          );
        }
        const inv = bpOf(srcs[0]);
        if (inv.H < 2 || inv.W < 2) {
          throw new Error(`bitplane-webgpu or_pool2d: H/Wが2未満(${node.id})`);
        }
        const HOut = ((inv.H - 2) >> 1) + 1;
        const WOut = ((inv.W - 2) >> 1) + 1;
        const wprOut = wordsPerRow(WOut);
        const out: BpView = {
          kind: 'bp',
          bufId: newBuf(node.id, bpBytes(inv.C, HOut, wprOut)),
          cOff: 0,
          C: inv.C,
          H: HOut,
          W: WOut,
          wpr: wprOut,
        };
        views.set(node.id, out);
        pending.push({
          id: node.id,
          nodeType: type,
          inputViewIds: [srcs[0]],
          outBufId: out.bufId,
          emit: () => buildOrPoolShader(bindOf(inv), bindOf(out)),
        });
        break;
      }
      case 'upsample2d': {
        const p = node.params as unknown as UpSample2dParams;
        if (p.scale !== 2) {
          throw new Error(`bitplane-webgpu upsample2d は scale=2 のみ対応(${node.id}: ${p.scale})`);
        }
        const inv = bpOf(srcs[0]);
        const HOut = inv.H * 2;
        const WOut = inv.W * 2;
        const wprOut = wordsPerRow(WOut);
        const out: BpView = {
          kind: 'bp',
          bufId: newBuf(node.id, bpBytes(inv.C, HOut, wprOut)),
          cOff: 0,
          C: inv.C,
          H: HOut,
          W: WOut,
          wpr: wprOut,
        };
        views.set(node.id, out);
        pending.push({
          id: node.id,
          nodeType: type,
          inputViewIds: [srcs[0]],
          outBufId: out.bufId,
          emit: () => buildUpsampleShader(bindOf(inv), bindOf(out)),
        });
        break;
      }
      case 'permute': {
        const p = node.params as unknown as PermuteParams;
        const inv = bpOf(srcs[0]);
        if (permuteKeepsTrailingSpatial(p)) {
          const n = p.in_shape.length;
          const H = p.in_shape[n - 2];
          const W = p.in_shape[n - 1];
          const C = p.in_shape.slice(0, n - 2).reduce((a, b) => a * b, 1);
          if (inv.H !== H || inv.W !== W || inv.C !== C) {
            throw new Error(`bitplane-webgpu permute: 形状不一致(${node.id})`);
          }
          const order = channelOrderFromPermute(p);
          const out: BpView = {
            kind: 'bp',
            bufId: newBuf(node.id, bpBytes(C, H, inv.wpr)),
            cOff: 0,
            C,
            H,
            W,
            wpr: inv.wpr,
          };
          views.set(node.id, out);
          pending.push({
            id: node.id,
            nodeType: type,
            inputViewIds: [srcs[0]],
            outBufId: out.bufId,
            emit: () => buildCopyPlanesShader(bindOf(inv), bindOf(out), order),
          });
          break;
        }
        // SpatialGroupSum型 permute([n,g,H,W] → [n,H,W,g])。後続 group_sum と融合
        // するためここではdispatchを作らず保留する。
        const dims = p.in_shape;
        const okShape =
          dims.length === 4
          && p.perm.length === 4
          && p.perm[0] === 0 && p.perm[1] === 2 && p.perm[2] === 3 && p.perm[3] === 1;
        if (!okShape) {
          throw new Error(
            `bitplane-webgpu permute: 未対応の並べ替え(${node.id}: perm=${JSON.stringify(p.perm)})`,
          );
        }
        const [nOut, gsz, H, W] = dims;
        if (inv.C !== nOut * gsz || inv.H !== H || inv.W !== W) {
          throw new Error(`bitplane-webgpu SpatialGroupSum permute: 形状不一致(${node.id})`);
        }
        views.set(node.id, { kind: 'pending-sgs', src: inv, n: nOut, g: gsz });
        notes.push(`group_sum融合: ${node.id}(permute[0,2,3,1])を後続group_sumへ取り込み`);
        break;
      }
      case 'group_sum': {
        const p = node.params as unknown as GroupSumParams;
        const sv = views.get(srcs[0]);
        if (sv && sv.kind === 'pending-sgs') {
          const { src, n, g } = sv;
          if (p.k !== n * src.H * src.W) {
            throw new Error(
              `bitplane-webgpu SpatialGroupSum: k=${p.k} != n*H*W=${n * src.H * src.W}`
              + `(${node.id})`,
            );
          }
          const out: DenseView = {
            kind: 'dense',
            bufId: newBuf(node.id, p.k * 4),
            off: 0,
            width: p.k,
          };
          views.set(node.id, out);
          pending.push({
            id: node.id,
            nodeType: 'spatial_group_sum',
            inputViewIds: [srcs[0]],
            outBufId: out.bufId,
            emit: () =>
              buildSpatialGroupSumShader(
                bindOf(src), n, g, out, buffers[out.bufId].bytesPerSample / 4,
              ),
          });
          break;
        }
        const inv = denseOf(srcs[0]);
        const out: DenseView = {
          kind: 'dense',
          bufId: newBuf(node.id, p.k * 4),
          off: 0,
          width: p.k,
        };
        views.set(node.id, out);
        pending.push({
          id: node.id,
          nodeType: type,
          inputViewIds: [srcs[0]],
          outBufId: out.bufId,
          emit: () =>
            buildDenseGroupSumShader(
              inv, buffers[inv.bufId].bytesPerSample / 4, p.k, out,
              buffers[out.bufId].bytesPerSample / 4,
            ),
        });
        break;
      }
      case 'flatten': {
        const v = views.get(srcs[0]);
        if (!v) throw new Error(`bitplane-webgpu flatten: 入力未解決(${node.id})`);
        views.set(node.id, v);
        break;
      }
      case 'concat': {
        resolveConcat(node, srcs);
        break;
      }
      default:
        throw new Error(`bitplane-webgpu 未対応ノード型: ${node.id}(${node.type})`);
    }
  }

  // --- 出力ポート ---
  const outPort = ir.outputs[0];
  const outSrc = inSrcs.get(outPort.id)?.[0] ?? ir.nodes[ir.nodes.length - 1].id;
  const outView = views.get(outSrc);
  if (!outView || outView.kind !== 'dense') {
    throw new Error(
      `bitplane-webgpu: 出力(${outSrc})が整数値でない(kind=${outView?.kind})。`
      + 'ビットプレーンのまま終わるIRは未対応(アンパックカーネル未実装)',
    );
  }

  // --- WGSL生成(ここで初めてビューが確定している) ---
  const computes: PlanCompute[] = pending.map((q) => {
    const built = q.emit();
    const inputBufs = q.inputViewIds.map((id) => {
      const v = views.get(id);
      if (v && (v.kind === 'bp' || v.kind === 'dense')) return v.bufId;
      if (v && v.kind === 'pending-sgs') return v.src.bufId;
      throw new Error(`bitplane-webgpu: 入力ビュー未解決(${q.id} <- ${id})`);
    });
    if (q.nodeType === 'pack') inputBufs.push(inputBufId);
    return {
      id: q.id,
      nodeType: q.nodeType,
      code: built.code,
      staticArrays: built.staticArrays,
      inputBufs,
      outputBuf: q.outBufId,
      threadsPerSample: built.threadsPerSample,
    };
  });

  // --- GPU前処理カーネル(RGBA → ビットプレーン)---
  // computes[0](=入力pack)とまったく同じ出力ビュー・同じ channelOrder 融合を
  // 持つ**代替**dispatchとして作る。実行時に「どちらか一方」を発行するので、
  // 逐次のCPU前処理経路(evalAll(bits))は無改修のまま残る。
  if (computes[0]?.nodeType !== 'pack') {
    throw new Error('bitplane-webgpu: computes[0] が入力packでない(プラン構成の前提崩れ)');
  }
  let preprocCompute: PlanCompute | null = null;
  let preprocInputBufId = -1;
  let preprocBytesPerSample = 0;
  let gpuPreprocNote: string | null = null;
  if (preproc) {
    try {
      const descs = buildPreprocDescriptors(preproc);
      const built = buildPreprocPackShader(descs, {
        cOff: packOutView.cOff,
        C: packOutView.C,
        H: packOutView.H,
        W: packOutView.W,
        wpr: packOutView.wpr,
        sampleWords: buffers[packOutView.bufId].bytesPerSample / 4,
        tailMask: tailMaskU32(packOutView.W),
      }, packOrder);
      if (packOutView.C * packOutView.H * packOutView.W !== inputWidth) {
        throw new Error(
          `入力幅不一致(IR ${inputWidth} vs 幾何 ${packOutView.C}x${packOutView.H}x${packOutView.W})`,
        );
      }
      preprocBytesPerSample = geom.H * geom.W * 4;
      preprocInputBufId = newBuf('input:rgba', preprocBytesPerSample);
      preprocCompute = {
        id: `${packOwner}__gpupreproc`,
        nodeType: 'preproc',
        code: built.code,
        staticArrays: built.staticArrays,
        inputBufs: [preprocInputBufId],
        outputBuf: packOutView.bufId,
        threadsPerSample: built.threadsPerSample,
      };
      notes.push(`GPU前処理: RGBA(${geom.H}x${geom.W}x4=${preprocBytesPerSample}B)直投入`
        + `(CPU前処理 ${inputWidth}B の転送・スカラーループを削除)`);
    } catch (e) {
      // 未対応 kind / 幾何不一致は黙殺せず記録し、CPU前処理経路のまま動かす。
      gpuPreprocNote = `GPU前処理を構築できずCPU前処理を使用: ${String(e)}`;
    }
  }

  // --- 安全性検査: 1つのdispatchが同じバッファを読み書きしていないこと。
  // (WebGPUのusage scope規則違反=validationエラーになる。concatエイリアスで
  //  同一バッファに複数の生産者が書く構成を許すので明示的に検査する。) ---
  for (const c of preprocCompute ? [...computes, preprocCompute] : computes) {
    if (c.inputBufs.includes(c.outputBuf)) {
      throw new Error(
        `bitplane-webgpu: ${c.id} が同一バッファ(${buffers[c.outputBuf].label})を`
        + '読み書きしている(concatエイリアスの適用不可構成)',
      );
    }
  }

  // 未使用になったバッファ(concatエイリアスで吸収された生産者の元バッファ)は
  // 確保しない。索引はそのまま保つため bytesPerSample を 0 にする。
  buffers.forEach((buf, i) => {
    if (!bufferUsed[i]) buf.bytesPerSample = 0;
  });

  return {
    buffers,
    computes,
    inputBufId,
    inputWidth,
    inputBytesPerSample,
    output: {
      bufId: outView.bufId,
      off: outView.off,
      width: outView.width,
      total: buffers[outView.bufId].bytesPerSample / 4,
    },
    notes,
    packComputeIndex: 0,
    preproc: preprocCompute,
    preprocInputBufId,
    preprocBytesPerSample,
    gpuPreprocNote,
  };
}

// ============================================================
// 5. ランタイム
// ============================================================

interface ComputeRes {
  plan: PlanCompute;
  pipeline: GPUComputePipeline;
  uniformBuf: GPUBuffer;
  staticBufs: GPUBuffer[];
}

/** パイプラインモードで発行済み・未回収のフレーム1本ぶん。*/
interface InflightFrame {
  /** readback(mapAsync解決 → コピー → unmap)まで含めた結果。 */
  result: Promise<NodeValue>;
}

/** 同時にin-flightにできるフレーム数(=readbackリングの本数)。
    2 = 「フレームNのmapAsync待ちの裏でフレームN+1のwriteBuffer+encode+submitを
    済ませておく」ための最小値。3以上にしても隠せる床は増えず(1フレームぶんの
    往復を隠せば十分)、表示レイテンシだけ伸びるので2で固定する。 */
export const PIPELINE_DEPTH = 2;

export interface WebgpuBitplaneOptions {
  /** doc.preproc。渡すとRGBA直投入のGPU前処理カーネルを組む
      (組めなければ plan.gpuPreprocNote に理由が残り、CPU前処理経路のみになる)。 */
  preproc?: ImagePreprocDoc | null;
}

/** 1フレームの入力。'bits' はCPU前処理済みの0/1バイト列(従来経路)、
    'rgba' はモデル解像度のRGBA(GPU前処理経路)。 */
export type BitplaneFrameInput =
  | { kind: 'bits'; bits: ArrayLike<number> }
  | { kind: 'rgba'; rgba: Uint8ClampedArray | Uint8Array };

/** ビットプレーン常駐 WebGPU executor。
    1フレーム = writeBuffer(入力)1回 + beginComputePass 1回 +
    dispatch N回(全ノード) + copyBufferToBuffer 1回 + submit 1回 + mapAsync 1回。 */
export class WebgpuBitplaneRuntime {
  private readonly device: GPUDevice;
  readonly plan: BitplanePlan;
  private readonly computes: ComputeRes[] = [];
  /** RGBA直投入の前処理dispatch(computes[packComputeIndex]の代替)。*/
  private readonly preprocRes: ComputeRes | null = null;
  private preprocBindGroup: GPUBindGroup | null = null;

  private batch = -1;
  private gpuBuffers: (GPUBuffer | null)[] = [];
  private bindGroups: GPUBindGroup[] = [];
  private readBuf: GPUBuffer | null = null;
  private inputStaging: Uint8Array | null = null;
  private validated = false;

  // --- プロファイリング(既定オフ)。単一passのbegin/endタイムスタンプで
  // ΣGPU(=融合パス全体のGPU時間)を、performance.nowでCPU側の分解を取る。
  // window.__DLGN_GPU_PROFILE_DETAIL__ === true のときだけノード別内訳のために
  // passをノード単位へ分割する(調査用フック。既定の計測経路は単一pass)。 ---
  private profiling = false;
  private profilingInited = false;
  private profilingSupported = false;
  private detail = false;
  private querySet: GPUQuerySet | null = null;
  private queryResolveBuf: GPUBuffer | null = null;
  private queryReadBuf: GPUBuffer | null = null;
  private readonly nodeGpuAccumNs = new Map<string, bigint>();
  private readonly nodeGpuCount = new Map<string, number>();
  private cpuAccum: WebgpuCpuBreakdown = {
    encodeMs: 0, writeBufferMs: 0, submitToMapMs: 0, readbackCopyMs: 0, totalMs: 0,
  };
  private cpuFrameCount = 0;

  // --- フレームパイプライン(2フレームin-flight)用の状態。
  // 逐次API(evalAll)とは完全に独立した資源しか持たない(evalAllは無改修)。
  /** readback(MAP_READ)バッファのリング。唯一の二重化対象。 */
  private ring: GPUBuffer[] = [];
  private ringBatch = -1;
  private ringNext = 0;
  /** 発行済み・未回収のフレーム(古い順)。 */
  private readonly inflight: InflightFrame[] = [];
  /** バックプレッシャで発行を待たされているフレーム数(0なら同期発行できる)。 */
  private queuedSubmits = 0;
  /** 遅延発行どうしの順序を保つチェーン(解決=「そのフレームのsubmitが済んだ」)。 */
  private submitTail: Promise<void> = Promise.resolve();

  constructor(
    device: GPUDevice, ir: IRGraphJson, opts: WebgpuBitplaneOptions = {},
  ) {
    this.device = device;
    this.plan = planBitplaneGraph(ir, opts.preproc ?? null);
    for (const c of this.plan.computes) this.computes.push(this.buildCompute(c));
    if (this.plan.preproc) this.preprocRes = this.buildCompute(this.plan.preproc);
  }

  private buildCompute(c: PlanCompute): ComputeRes {
    const device = this.device;
    const module = device.createShaderModule({ code: c.code });
    const pipeline = device.createComputePipeline({
      layout: 'auto',
      compute: { module, entryPoint: 'main' },
    });
    const uniformBuf = device.createBuffer({
      size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    const staticBufs = c.staticArrays.map((arr) => {
      const buf = device.createBuffer({
        size: Math.max(4, arr.byteLength),
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });
      device.queue.writeBuffer(buf, 0, arr as unknown as ArrayBufferView<ArrayBuffer>);
      return buf;
    });
    return { plan: c, pipeline, uniformBuf, staticBufs };
  }

  /** dispatch数(=単一pass内のdispatch回数)。融合効果の報告・テスト用。
      GPU前処理モードでも pack が preproc に置き換わるだけなので同数。 */
  get dispatchCount(): number {
    return this.computes.length;
  }

  /** RGBAを直接投入できるか(=GPU前処理カーネルを組めたか)。*/
  get supportsGpuPreproc(): boolean {
    return this.preprocRes !== null;
  }

  /** GPU前処理を組めなかった理由(組めたときは null)。*/
  get gpuPreprocNote(): string | null {
    return this.plan.gpuPreprocNote;
  }

  setProfiling(enabled: boolean): void {
    this.profiling = enabled;
    if (!enabled || this.profilingInited) return;
    this.profilingInited = true;
    const g = globalThis as unknown as Record<string, unknown>;
    this.detail = g.__DLGN_GPU_PROFILE_DETAIL__ === true;
    this.profilingSupported = this.device.features.has('timestamp-query')
      && this.computes.length > 0;
    if (!this.profilingSupported) return;
    const count = this.detail ? this.computes.length * 2 : 2;
    this.querySet = this.device.createQuerySet({ type: 'timestamp', count });
    const byteLength = count * 8;
    this.queryResolveBuf = this.device.createBuffer({
      size: byteLength,
      usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
    });
    this.queryReadBuf = this.device.createBuffer({
      size: byteLength,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
  }

  resetProfilingStats(): void {
    this.nodeGpuAccumNs.clear();
    this.nodeGpuCount.clear();
    this.cpuAccum = {
      encodeMs: 0, writeBufferMs: 0, submitToMapMs: 0, readbackCopyMs: 0, totalMs: 0,
    };
    this.cpuFrameCount = 0;
  }

  getBreakdown(): WebgpuBreakdown | null {
    if (this.cpuFrameCount === 0) return null;
    const n = this.cpuFrameCount;
    const cpu: WebgpuCpuBreakdown = {
      encodeMs: this.cpuAccum.encodeMs / n,
      writeBufferMs: this.cpuAccum.writeBufferMs / n,
      submitToMapMs: this.cpuAccum.submitToMapMs / n,
      readbackCopyMs: this.cpuAccum.readbackCopyMs / n,
      totalMs: this.cpuAccum.totalMs / n,
    };
    const nodeAvgs: WebgpuNodeTiming[] = [];
    for (const [id, sumNs] of this.nodeGpuAccumNs) {
      const cnt = this.nodeGpuCount.get(id) ?? 1;
      nodeAvgs.push({
        nodeId: id,
        nodeType: id === '(fused-pass)' ? `pass x${this.computes.length}dispatch` : '?',
        gpuMs: Number(sumNs) / 1e6 / cnt,
      });
    }
    for (const t of nodeAvgs) {
      if (t.nodeId !== '(fused-pass)') {
        const hit = this.plan.preproc?.id === t.nodeId
          ? this.plan.preproc
          : this.computes.find((c) => c.plan.id === t.nodeId)?.plan;
        t.nodeType = hit?.nodeType ?? '?';
      }
    }
    nodeAvgs.sort((a, b) => b.gpuMs - a.gpuMs);
    return {
      timestampSupported: this.profilingSupported,
      frameCount: n,
      topNodesGpu: nodeAvgs.slice(0, 12),
      cpu,
    };
  }

  private ensureBatch(batch: number): void {
    if (batch === this.batch) return;
    const device = this.device;
    const isInput = (i: number): boolean =>
      i === this.plan.inputBufId || i === this.plan.preprocInputBufId;
    this.gpuBuffers = this.plan.buffers.map((buf, i) => {
      if (buf.bytesPerSample === 0) return null;
      const usage = isInput(i)
        ? GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
        : GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
      return device.createBuffer({
        size: Math.max(4, batch * buf.bytesPerSample),
        usage,
      });
    });

    this.bindGroups = this.computes.map((c) => this.buildBindGroup(c, batch));
    this.preprocBindGroup = this.preprocRes
      ? this.buildBindGroup(this.preprocRes, batch)
      : null;

    this.readBuf = device.createBuffer({
      size: Math.max(4, batch * this.plan.output.width * 4),
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
    this.inputStaging = null;
    this.batch = batch;
  }

  /** binding規約: 0=Params(uniform)、1..=入力バッファ、その後=静的データ、
      最後=出力バッファ(read_write)。 */
  private buildBindGroup(c: ComputeRes, batch: number): GPUBindGroup {
    const device = this.device;
    const total = batch * c.plan.threadsPerSample;
    const d2 = dispatch2d(total, device.limits.maxComputeWorkgroupsPerDimension);
    device.queue.writeBuffer(
      c.uniformBuf, 0, new Uint32Array([total, d2.xThreads, 0, 0]),
    );
    const entries: GPUBindGroupEntry[] = [
      { binding: 0, resource: { buffer: c.uniformBuf } },
    ];
    c.plan.inputBufs.forEach((bufId, i) => {
      entries.push({ binding: 1 + i, resource: { buffer: this.gpuBuffers[bufId]! } });
    });
    const staticBase = 1 + c.plan.inputBufs.length;
    c.staticBufs.forEach((buf, i) => {
      entries.push({ binding: staticBase + i, resource: { buffer: buf } });
    });
    entries.push({
      binding: staticBase + c.staticBufs.length,
      resource: { buffer: this.gpuBuffers[c.plan.outputBuf]! },
    });
    return device.createBindGroup({
      layout: c.pipeline.getBindGroupLayout(0),
      entries,
    });
  }

  /** 入力ビット列(0/1)をGPUの生バイトバッファへ載せる。encodeImageAuto の
      既定出力は Uint8Array なので、4の倍数長ならゼロコピーでそのまま
      writeBuffer する(detfaceで11.3MB→2.8MBの転送削減)。それ以外の型・長さは
      再利用ステージングへ詰め替える(正しさ優先の一般路)。 */
  private uploadInput(bits: ArrayLike<number>, batch: number): void {
    const device = this.device;
    const buf = this.gpuBuffers[this.plan.inputBufId]!;
    const need = batch * this.plan.inputWidth;
    if (bits.length !== need) {
      throw new Error(`bitplane-webgpu 入力: 期待${need}要素, 実際${bits.length}`);
    }
    const padded = batch * this.plan.inputBytesPerSample;
    if (bits instanceof Uint8Array && this.plan.inputWidth % 4 === 0) {
      device.queue.writeBuffer(buf, 0, bits as unknown as ArrayBufferView<ArrayBuffer>, 0, need);
      return;
    }
    if (!this.inputStaging || this.inputStaging.length !== padded) {
      this.inputStaging = new Uint8Array(padded);
    }
    // 端数パディングがある場合はサンプル間ストライドが inputWidth と食い違う
    // (WGSL側は IN_STRIDE=inputBytesPerSample で読む)ので、必ずサンプル単位で置く。
    const st = this.inputStaging;
    const w = this.plan.inputWidth;
    const stride = this.plan.inputBytesPerSample;
    for (let b = 0; b < batch; b++) {
      const dst = b * stride;
      if (bits instanceof Uint8Array) {
        st.set(bits.subarray(b * w, (b + 1) * w), dst);
      } else {
        for (let i = 0; i < w; i++) st[dst + i] = bits[b * w + i] !== 0 ? 1 : 0;
      }
    }
    device.queue.writeBuffer(buf, 0, st as unknown as ArrayBufferView<ArrayBuffer>, 0, padded);
  }

  /** RGBA(u8×4/画素)をそのままGPUへ載せる。CPU前処理経路が
      2.8MBの0/1バイト列を送っていたのに対し、こちらは 64KB(128x128)で済む。 */
  private uploadRgba(rgba: Uint8ClampedArray | Uint8Array, batch: number): void {
    if (!this.preprocRes) {
      throw new Error('bitplane-webgpu: GPU前処理が無効なのにRGBAが渡された'
        + `(${this.plan.gpuPreprocNote ?? 'preproc未指定'})`);
    }
    const need = batch * this.plan.preprocBytesPerSample;
    if (rgba.length !== need) {
      throw new Error(`bitplane-webgpu RGBA入力: 期待${need}バイト, 実際${rgba.length}`);
    }
    const buf = this.gpuBuffers[this.plan.preprocInputBufId]!;
    // Uint8ClampedArray も ArrayBufferView なのでゼロコピーで載せられる。
    this.device.queue.writeBuffer(
      buf, 0, rgba as unknown as ArrayBufferView<ArrayBuffer>, 0, need,
    );
  }

  private uploadFrame(input: BitplaneFrameInput, batch: number): void {
    if (input.kind === 'rgba') this.uploadRgba(input.rgba, batch);
    else this.uploadInput(input.bits, batch);
  }

  /** そのフレームで発行するdispatch列(GPU前処理モードでは先頭のpackだけ
      preprocへ差し替える)。順序も個数も両モードで同一。 */
  private frameDispatches(useGpuPreproc: boolean): { res: ComputeRes; bg: GPUBindGroup }[] {
    const list = this.computes.map((res, i) => ({ res, bg: this.bindGroups[i] }));
    if (useGpuPreproc) {
      list[this.plan.packComputeIndex] = {
        res: this.preprocRes!,
        bg: this.preprocBindGroup!,
      };
    }
    return list;
  }

  /** [batch*inWidth] の 0/1 ビット列 → [batch*outWidth] の整数スコア。 */
  async evalAll(bits: ArrayLike<number>, batch: number): Promise<NodeValue> {
    return this.evalFrame({ kind: 'bits', bits }, batch);
  }

  /** evalAll のRGBA版(前処理をGPUで行う)。返り値は完全に同一。*/
  async evalAllRgba(
    rgba: Uint8ClampedArray | Uint8Array, batch: number,
  ): Promise<NodeValue> {
    return this.evalFrame({ kind: 'rgba', rgba }, batch);
  }

  async evalFrame(input: BitplaneFrameInput, batch: number): Promise<NodeValue> {
    this.ensureBatch(batch);
    const device = this.device;
    const profiling = this.profiling;
    const tFrameStart = profiling ? performance.now() : 0;
    const firstEval = !this.validated;
    if (firstEval) device.pushErrorScope('validation');

    const tWriteStart = profiling ? performance.now() : 0;
    this.uploadFrame(input, batch);
    const tWriteEnd = profiling ? performance.now() : 0;

    const dispatches = this.frameDispatches(input.kind === 'rgba');
    const encoder = device.createCommandEncoder();
    const useTs = profiling && this.profilingSupported;
    if (useTs && this.detail) {
      // 調査用: ノード別GPU時間を取るためだけにpassを分割する(既定経路ではない)。
      dispatches.forEach((c, i) => {
        const pass = encoder.beginComputePass({
          timestampWrites: {
            querySet: this.querySet!,
            beginningOfPassWriteIndex: i * 2,
            endOfPassWriteIndex: i * 2 + 1,
          },
        });
        this.encodeDispatch(pass, c.res, c.bg, batch);
        pass.end();
      });
    } else {
      const desc: GPUComputePassDescriptor = {};
      if (useTs) {
        desc.timestampWrites = {
          querySet: this.querySet!,
          beginningOfPassWriteIndex: 0,
          endOfPassWriteIndex: 1,
        };
      }
      const pass = encoder.beginComputePass(desc);
      for (const c of dispatches) this.encodeDispatch(pass, c.res, c.bg, batch);
      pass.end();
    }

    const out = this.plan.output;
    const srcBuf = this.gpuBuffers[out.bufId]!;
    const readBuf = this.readBuf!;
    for (let b = 0; b < batch; b++) {
      encoder.copyBufferToBuffer(
        srcBuf, (b * out.total + out.off) * 4,
        readBuf, b * out.width * 4,
        out.width * 4,
      );
    }
    if (useTs) {
      const count = this.detail ? this.computes.length * 2 : 2;
      encoder.resolveQuerySet(this.querySet!, 0, count, this.queryResolveBuf!, 0);
      encoder.copyBufferToBuffer(this.queryResolveBuf!, 0, this.queryReadBuf!, 0, count * 8);
    }
    const tEncodeEnd = profiling ? performance.now() : 0;

    device.queue.submit([encoder.finish()]);
    const tSubmitStart = profiling ? performance.now() : 0;

    if (firstEval) {
      const err = await device.popErrorScope();
      if (err) {
        throw new Error(`bitplane-webgpu validationエラー(黙殺せずthrow): ${err.message}`);
      }
      this.validated = true;
    }

    if (useTs) {
      await Promise.all([
        readBuf.mapAsync(GPUMapMode.READ),
        this.queryReadBuf!.mapAsync(GPUMapMode.READ),
      ]);
    } else {
      await readBuf.mapAsync(GPUMapMode.READ);
    }
    const tMapEnd = profiling ? performance.now() : 0;

    const mapped = new Int32Array(readBuf.getMappedRange().slice(0, batch * out.width * 4));
    readBuf.unmap();
    const tReadEnd = profiling ? performance.now() : 0;

    if (profiling) {
      this.cpuAccum.writeBufferMs += tWriteEnd - tWriteStart;
      this.cpuAccum.encodeMs += tEncodeEnd - tWriteEnd;
      this.cpuAccum.submitToMapMs += tMapEnd - tSubmitStart;
      this.cpuAccum.readbackCopyMs += tReadEnd - tMapEnd;
      this.cpuAccum.totalMs += tReadEnd - tFrameStart;
      this.cpuFrameCount++;
      if (this.profilingSupported) {
        const raw = new BigUint64Array(
          this.queryReadBuf!.getMappedRange().slice(0) as ArrayBuffer,
        );
        const acc = (id: string, i: number): void => {
          const beginNs = raw[i * 2];
          const endNs = raw[i * 2 + 1];
          const diff = endNs >= beginNs ? endNs - beginNs : 0n;
          this.nodeGpuAccumNs.set(id, (this.nodeGpuAccumNs.get(id) ?? 0n) + diff);
          this.nodeGpuCount.set(id, (this.nodeGpuCount.get(id) ?? 0) + 1);
        };
        if (this.detail) {
          dispatches.forEach((c, i) => acc(c.res.plan.id, i));
        } else {
          acc('(fused-pass)', 0);
        }
        this.queryReadBuf!.unmap();
      }
    }

    return { data: mapped, width: out.width };
  }

  private encodeDispatch(
    pass: GPUComputePassEncoder, c: ComputeRes, bg: GPUBindGroup, batch: number,
  ): void {
    pass.setPipeline(c.pipeline);
    pass.setBindGroup(0, bg);
    const total = batch * c.plan.threadsPerSample;
    const d2 = dispatch2d(total, this.device.limits.maxComputeWorkgroupsPerDimension);
    pass.dispatchWorkgroups(d2.x, d2.y);
  }

  // ============================================================
  // フレームパイプライン(2フレームin-flight)
  //
  // 動機: ΣGPUは0.25〜0.51msまで落ちたが、フレーム時間の大半は
  // 「submit → mapAsync解決」の往復(Dawn/Metal IPC×2 + イベントループ)で、
  // 1.2〜2.2msの床として残っていた。この床は1フレームぶんの結果を毎回awaitする
  // 限り原理的に消せない。そこでフレームNのmapAsync待ちの裏でフレームN+1の
  // writeBuffer+encode+submitを走らせ、床をスループットから隠す。
  //
  // ■ 二重化の範囲と根拠(重要)
  //   - 中間バッファ(全storageバッファ)= 共有でよい。
  //     根拠: 同一GPUQueueへ submit されたコマンドバッファは**発行順に**実行され
  //     (WebGPU仕様: queueのoperationは順序付き)、フレームNの単一compute passと
  //     それに続く copyBufferToBuffer が完了してからフレームN+1のdispatchが走る。
  //     つまりGPU上では依然として逐次であり、中間バッファの上書き競合は起きない。
  //     (「並列に走らせる」のではなく「CPU側の待ちをGPU実行に重ねる」のが狙い。)
  //   - 入力バッファ = 共有でよい。
  //     根拠: queue.writeBuffer もキュー操作であり、直前の submit との相対順序が
  //     保たれる。フレームN+1のwriteBufferはフレームNのdispatchより後に適用される。
  //     さらに writeBuffer は呼び出し時点でソースの内容をコピーする契約なので、
  //     呼び出し側のUint8Arrayを即座に再利用しても安全。
  //   - readback(MAP_READ)バッファ = **二重化が必須**。
  //     根拠: mapAsync中/マップ済みのバッファはコピー先に使えない(validationエラー)。
  //     フレームNの結果をmapしている間にフレームN+1のcopyBufferToBufferを積むには
  //     別のバッファが要る。よってリング長=PIPELINE_DEPTH=2。
  //   → 結論: 二重化するのはreadbackバッファのみ。入力・中間は共有。
  //     (この設計の正しさは wasmgpu_verify.html の「逐次 vs パイプライン
  //      50フレーム整数一致ストレス」で実測確認する。)
  //
  // ■ 計測(プロファイリング)について
  //   パイプラインモードではtimestamp-queryを使わない。querySet/resolve用バッファも
  //   フレーム毎に二重化が要るうえ、2本目のmapAsyncが床を復活させて計測対象を
  //   変えてしまうため。パイプラインの指標は「実効フレーム時間=全体時間/N」で取る。
  // ============================================================

  /** パイプラインモードで同時発行できるフレーム数。 */
  get pipelineDepth(): number {
    return PIPELINE_DEPTH;
  }

  /** 発行済み・未回収のフレーム数(テスト/バックプレッシャ判定用)。 */
  get pipelineInFlight(): number {
    return this.inflight.length;
  }

  /** フレームNの結果を待たずに次フレームを発行できる非同期API。
      返り値のPromiseはそのフレームの結果(逐次APIの evalAll と同一の値)。

      呼び出し規約: in-flightが上限(PIPELINE_DEPTH)未満のときに呼ぶと、
      encode+submitまでを**同期に**済ませてから未解決のPromiseを返す(入力の
      コピーも発生しない)。上限に達している状態で呼ばれた場合も壊れずに動くよう、
      入力をスナップショットして最古フレームの完了後に発行する(順序は保たれる)。
      推奨パターン:
        const pend = [];
        for (...) { if (pend.length >= depth) await pend.shift(); pend.push(rt.submitFrame(bits, 1)); }
        await Promise.all(pend);
   */
  submitFrame(bits: ArrayLike<number>, batch: number): Promise<NodeValue> {
    return this.submitFrameInput({ kind: 'bits', bits }, batch);
  }

  /** submitFrame のRGBA版(前処理をGPUで行う)。*/
  submitFrameRgba(
    rgba: Uint8ClampedArray | Uint8Array, batch: number,
  ): Promise<NodeValue> {
    return this.submitFrameInput({ kind: 'rgba', rgba }, batch);
  }

  submitFrameInput(input: BitplaneFrameInput, batch: number): Promise<NodeValue> {
    this.ensureBatch(batch);
    this.ensurePipelineRing(batch);
    if (this.queuedSubmits === 0 && this.inflight.length < PIPELINE_DEPTH) {
      return this.beginPipelinedFrame(input, batch);
    }
    // 発行を遅らせる必要がある = 呼び出し側のバッファが書き換わり得るのでコピー。
    const snap = snapshotInput(input);
    this.queuedSubmits++;
    let released!: () => void;
    const gate = new Promise<void>((r) => { released = r; });
    const prev = this.submitTail;
    this.submitTail = gate;
    return (async () => {
      let frame: Promise<NodeValue>;
      try {
        await prev;
        while (this.inflight.length >= PIPELINE_DEPTH) {
          await this.inflight[0].result.catch(() => undefined);
        }
        this.queuedSubmits--;
        frame = this.beginPipelinedFrame(snap, batch);
      } finally {
        // submitが済んだ(あるいは失敗した)時点で次の遅延発行を解放する。
        // フレームの完了(readback)を待たないのがパイプラインの肝。
        released();
      }
      return frame;
    })();
  }

  /** in-flightフレームを全て回収する(バッチ変更前・計測終了時・破棄前に呼ぶ)。 */
  async flushPipeline(): Promise<void> {
    await this.submitTail.catch(() => undefined);
    while (this.inflight.length > 0) {
      await this.inflight[0].result.catch(() => undefined);
    }
  }

  /** readbackリング(唯一の二重化資源)を用意する。 */
  private ensurePipelineRing(batch: number): void {
    if (this.ringBatch === batch && this.ring.length === PIPELINE_DEPTH) return;
    if (this.inflight.length > 0) {
      throw new Error('bitplane-webgpu: in-flightフレームがある状態でbatchを変更できない'
        + '(flushPipeline()を先に呼ぶこと)');
    }
    for (const b of this.ring) b.destroy();
    const size = Math.max(4, batch * this.plan.output.width * 4);
    this.ring = [];
    for (let i = 0; i < PIPELINE_DEPTH; i++) {
      this.ring.push(this.device.createBuffer({
        size,
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
      }));
    }
    this.ringBatch = batch;
    this.ringNext = 0;
  }

  /** encode+submitまでを同期に行い、readback完了のPromiseを返す。
      呼び出し前に in-flight < PIPELINE_DEPTH であることが保証されていること。 */
  private beginPipelinedFrame(input: BitplaneFrameInput, batch: number): Promise<NodeValue> {
    const device = this.device;
    // 初回のみvalidationスコープを張る(evalAllと同じ「黙殺せずthrow」方針)。
    // validatedを先に立てるので、スコープのpush/popが入れ子で食い違うことはない。
    const firstEval = !this.validated;
    if (firstEval) {
      this.validated = true;
      device.pushErrorScope('validation');
    }

    this.uploadFrame(input, batch);

    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    for (const c of this.frameDispatches(input.kind === 'rgba')) {
      this.encodeDispatch(pass, c.res, c.bg, batch);
    }
    pass.end();

    const out = this.plan.output;
    const srcBuf = this.gpuBuffers[out.bufId]!;
    const slot = this.ringNext;
    this.ringNext = (this.ringNext + 1) % this.ring.length;
    const readBuf = this.ring[slot];
    for (let b = 0; b < batch; b++) {
      encoder.copyBufferToBuffer(
        srcBuf, (b * out.total + out.off) * 4,
        readBuf, b * out.width * 4,
        out.width * 4,
      );
    }
    device.queue.submit([encoder.finish()]);

    const entry: InflightFrame = { result: undefined as unknown as Promise<NodeValue> };
    entry.result = (async () => {
      try {
        if (firstEval) {
          const err = await device.popErrorScope();
          if (err) {
            throw new Error(`bitplane-webgpu validationエラー(黙殺せずthrow): ${err.message}`);
          }
        }
        await readBuf.mapAsync(GPUMapMode.READ);
        const mapped = new Int32Array(readBuf.getMappedRange().slice(0, batch * out.width * 4));
        readBuf.unmap();
        return { data: mapped, width: out.width };
      } finally {
        // unmap済みになってからリングの席を空ける(次の周回で同じslotへ
        // copyBufferToBufferを積めるのはこの後)。
        const i = this.inflight.indexOf(entry);
        if (i >= 0) this.inflight.splice(i, 1);
      }
    })();
    this.inflight.push(entry);
    return entry.result;
  }

  /** [検証用] 入力pack(またはGPU前処理)の dispatch **1個だけ**を実行し、
      ビットプレーンバッファの全ワードを読み戻す。
      CPU前処理経路とGPU前処理経路が「最終スコア」ではなく「packed planes」の
      段階で1ワードも違わないことの直接証拠を取るために使う(どのチャネル・
      どの行で食い違ったかがそのまま分かる)。
      同一ランタイム上で2回呼んで比較する運用を想定しているので、pack以外の
      領域(concatエイリアスの相手側)は両者とも未書き込みのまま一致する。 */
  async debugPackedPlanes(
    input: BitplaneFrameInput, batch: number,
  ): Promise<Uint32Array> {
    await this.flushPipeline();
    this.ensureBatch(batch);
    const device = this.device;
    this.uploadFrame(input, batch);
    const d = this.frameDispatches(input.kind === 'rgba')[this.plan.packComputeIndex];
    const packBufId = d.res.plan.outputBuf;
    const bytes = batch * this.plan.buffers[packBufId].bytesPerSample;
    const staging = device.createBuffer({
      size: Math.max(4, bytes),
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    this.encodeDispatch(pass, d.res, d.bg, batch);
    pass.end();
    encoder.copyBufferToBuffer(this.gpuBuffers[packBufId]!, 0, staging, 0, bytes);
    device.queue.submit([encoder.finish()]);
    await staging.mapAsync(GPUMapMode.READ);
    const out = new Uint32Array(staging.getMappedRange().slice(0, bytes));
    staging.unmap();
    staging.destroy();
    return out;
  }
}

/** 遅延発行時のみ使う入力スナップショット。uploadInput の型分岐
    (Uint8Arrayはゼロコピー経路 / それ以外はステージング経路)を変えないよう、
    元の型カテゴリを保ったままコピーする。 */
function snapshotBits(bits: ArrayLike<number>): ArrayLike<number> {
  if (bits instanceof Uint8Array) return bits.slice();
  return Int32Array.from(bits as ArrayLike<number>);
}

/** snapshotBits のフレーム入力版(RGBAはそのままsliceでよい)。*/
function snapshotInput(input: BitplaneFrameInput): BitplaneFrameInput {
  if (input.kind === 'rgba') return { kind: 'rgba', rgba: input.rgba.slice() };
  return { kind: 'bits', bits: snapshotBits(input.bits) };
}
