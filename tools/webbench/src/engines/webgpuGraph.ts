// DLGN WebGPU「グラフ常駐」executor。
//
// 以前の実装(webgpuConv.ts)は conv2d ノードだけを WGSL で実行し、他ノード
// (upsample2d/permute/group_sum/concat/add/binarize/lut_layer)は純JSグルーで
// 実行していたため、conv2dノードの数(seg=5, det=6)だけ「dispatch+mapAsync
// 読み戻し」が1フレームに発生していた(CPU↔GPU同期がボトルネック)。
//
// このファイルは lowered IR(ConvRuntime.lowered、nativeConv=true。conv2d/
// upsample2d/permuteはネイティブのまま、or_pool2dはlut_layerへ展開済み)を
// 丸ごとGPUへ載せる: ノードごとに1つのWGSL computeシェーダ+専用
// GPUComputePipelineを構築し、各ノードの出力を「GPU常駐のstorage buffer」に
// 割り当てる。全ノードのcompute passを1つのcommandEncoderへエンコードして
// 1回queue.submitし、最終出力バッファだけmapAsyncで読み戻す
// (per-frameのCPU同期を1回に削減する)。
//
// 正しさの錨: irflat.ts の evalIrFlat/evalIrFlatAsync(CPU参照実装、
// ir/simulate.py 移植)と全ノード型・全アドレッシングをビット単位で一致させる。
// 各ノードの gather/index 計算は packedConv.ts / lowerConv.ts の同一ロジックを
// そのままWGSLへ移植したもの(乖離防止のためコメントで対応元を明記する)。
import { ttParse } from '../shared';
import type {
  AddParams,
  Conv2dParams,
  GroupSumParams,
  IRGraphJson,
  IRNodeJson,
  LutLayerParams,
  PermuteParams,
  UpSample2dParams,
} from '../shared';
import { collectInputSources, topologicalOrder } from './irflat';
import type { NodeValue } from './irflat';

// ============================================================
// フレーム時間内訳計測(TimingHelperパターン)。
// 計測はオプトイン(setProfiling(true))。無効時は下記の型・状態を一切
// 参照せず、ホットパス(evalAll)にコストを足さない。
// ============================================================

/** ノード1つぶんのGPU実行時間(プロファイリング期間の平均)。 */
export interface WebgpuNodeTiming {
  nodeId: string;
  nodeType: string;
  gpuMs: number;
}

/** 1フレームのCPU側分解(プロファイリング期間の平均)。performance.now
    ベースの実測区間: encodeMs(コマンドエンコード) / writeBufferMs /
    submitToMapMs(submit→mapAsync解決まで) / readbackCopyMs / totalMs。 */
export interface WebgpuCpuBreakdown {
  encodeMs: number;
  writeBufferMs: number;
  submitToMapMs: number;
  readbackCopyMs: number;
  totalMs: number;
}

/** setProfiling(true)後に蓄積した統計のスナップショット。timestampSupported
    はデバイスが timestamp-query feature を実際に取得できたか(非対応環境では
    false。黙殺はせず、topNodesGpuは空のままcpu側の分解は継続して返す)。 */
export interface WebgpuBreakdown {
  timestampSupported: boolean;
  frameCount: number;
  topNodesGpu: WebgpuNodeTiming[];
  cpu: WebgpuCpuBreakdown;
}

function pair(v: number | number[]): [number, number] {
  return Array.isArray(v) ? [v[0], v[1]] : [v, v];
}

/** 2入力ゲート(tt: 0..15)。webgpuConv.ts / packedConv.ts の gate2 と同一16式。
    webgpuBitplane.ts(ワードパック版executor)も同じ16式を使うため
    export する(定義を2箇所に持つと乖離するリスクがあるので必ず共有すること)。
    無分岐マスク式(runtime/gpu/mdl_gpu_kernels.h::mdl_g_gate2と同型)。
    WGSL A/B実測(gpuAluCalib.ts)で低並列1.32×/高並列1.11×とマスク式が
    速かったため switch から置換。標準的な2入力LUTの最小項展開で、出力は
    switch版とbit-exactに一致する(gpuAluCalib.tsのA/Bハーネスで検査済み)。 */
export const GATE2_WGSL = `
fn gate2(tt: u32, a: u32, b: u32) -> u32 {
  let m0 = 0u - ((tt >> 0u) & 1u);
  let m1 = 0u - ((tt >> 1u) & 1u);
  let m2 = 0u - ((tt >> 2u) & 1u);
  let m3 = 0u - ((tt >> 3u) & 1u);
  let na = ~a;
  let nb = ~b;
  return (m0 & na & nb) | (m1 & a & nb) | (m2 & na & b) | (m3 & a & b);
}
`;

/** 全ノード共通のuniform(バッチ変化時のみ書き換える。16byte確保でuniform最小
    バインディングサイズの実装差を回避する)。
    _p0 = 2Dディスパッチのx方向スレッド数(xGroups*64)。: 1次元dispatchは
    maxComputeWorkgroupsPerDimension(既定65,535)を大モデルで超える
    (k640のstage1=5.9M要素→92,160 workgroups)ため、超過分をy次元へ分割し
    WGSL側で gidx = gid.y*_p0 + gid.x に復元する。 */
const PARAMS_STRUCT_WGSL = `
struct Params { total: u32, _p0: u32, _p1: u32, _p2: u32 };
@group(0) @binding(0) var<uniform> params: Params;
`;

/** total要素/wgSizeスレッドを2Dディスパッチへ分割する。
    x=min(必要WG数, maxPerDim)、y=残りを繰り上げ。xThreads(=x*wgSize)は
    Params._p0 としてWGSLのインデックス復元に使う。純関数(テスト対象)。 */
export function dispatch2d(
  total: number, maxPerDim: number, wgSize = 64,
): { x: number; y: number; xThreads: number } {
  const nWg = Math.max(1, Math.ceil(total / wgSize));
  const x = Math.min(nWg, maxPerDim);
  const y = Math.ceil(nWg / x);
  return { x, y, xThreads: x * wgSize };
}

/** 1ノードぶんのWGSLビルド結果。バインディング規約は全ノード型で共通:
    0=Params(uniform), 1..nInputs=先行ノード出力(read), nInputs+1..+nStatic=
    静的データ(read), 最後=このノードの出力(read_write)。 */
interface ShaderBuild {
  code: string;
  nInputs: number;
  staticArrays: (Int32Array | Uint32Array)[];
  width: number; // 1サンプルあたりの出力幅
}

// ============================================================
// ノード型別のWGSL生成(packedConv.ts / lowerConv.ts のアドレッシングを移植)
// ============================================================

/** conv2d(n_inputs=2固定): webgpuConv.ts の元シェーダをバッチ常駐アドレッシング
    (xBuf/outBufとも[batch,width]フラット)へ書き換えたもの。タップ分解+木の
    ping-pong評価は同一。inWidth は先行ノードの実測幅(x.width相当)を使う
    (packedConv.evalConv2dPackedのxoff=b*inWと同じ契約)。 */
function buildConv2d(p: Conv2dParams, inWidth: number): ShaderBuild {
  const [, H, W] = p.in_shape;
  const [kH, kW] = pair(p.kernel_size);
  const [sH, sW] = pair(p.stride);
  const [pH, pW] = pair(p.padding);
  const HOut = Math.floor((H + 2 * pH - kH) / sH) + 1;
  const WOut = Math.floor((W + 2 * pW - kW) / sW) + 1;
  const OC = p.out_channels;
  const N = p.n_inputs;
  if (N !== 2) throw new Error(`webgpuGraph conv2d は n_inputs=2 のみ対応(実際: ${N})`);
  const depth = p.tree_depth;
  const nLeaves = N ** depth;
  const nInternal = (nLeaves - 1) / (N - 1);
  const Lout = HOut * WOut;
  const khw = kH * kW;
  const idxf = p.indices_flat;
  const outWidth = OC * Lout;

  const leafC = new Int32Array(OC * nLeaves);
  const leafDy = new Int32Array(OC * nLeaves);
  const leafDx = new Int32Array(OC * nLeaves);
  for (let oc = 0; oc < OC; oc++) {
    const row = idxf[oc];
    for (let k = 0; k < nLeaves; k++) {
      const ppos = row[k];
      const rem = ppos % khw;
      leafC[oc * nLeaves + k] = Math.floor(ppos / khw);
      leafDy[oc * nLeaves + k] = Math.floor(rem / kW);
      leafDx[oc * nLeaves + k] = rem % kW;
    }
  }
  const ttNum = new Uint32Array(OC * nInternal);
  for (let i = 0; i < ttNum.length; i++) {
    ttNum[i] = Number(ttParse(p.truth_tables[i])) & 0xf;
  }

  const code = `
${PARAMS_STRUCT_WGSL}
@group(0) @binding(1) var<storage, read> xBuf: array<i32>;
@group(0) @binding(2) var<storage, read> leafC: array<i32>;
@group(0) @binding(3) var<storage, read> leafDy: array<i32>;
@group(0) @binding(4) var<storage, read> leafDx: array<i32>;
@group(0) @binding(5) var<storage, read> ttNum: array<u32>;
@group(0) @binding(6) var<storage, read_write> outBuf: array<i32>;

${GATE2_WGSL}

const N_LEAVES: u32 = ${nLeaves}u;
const H: i32 = ${H};
const W: i32 = ${W};
const WOut: i32 = ${WOut};
const S_H: i32 = ${sH};
const S_W: i32 = ${sW};
const P_H: i32 = ${pH};
const P_W: i32 = ${pW};
const N_LEAVES_I: i32 = ${nLeaves};
const N_INTERNAL: i32 = ${nInternal};
const L_OUT: i32 = ${Lout};
const IN_WIDTH: i32 = ${inWidth};
const OUT_WIDTH: i32 = ${outWidth};

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let gidx = gid.y * params._p0 + gid.x;
  if (gidx >= params.total) { return; }
  let idx = i32(gidx);
  let b = idx / OUT_WIDTH;
  let rem = idx % OUT_WIDTH;
  let oc = rem / L_OUT;
  let po = rem % L_OUT;
  let oy = po / WOut;
  let ox = po % WOut;
  let xBase = b * IN_WIDTH;

  var leaf: array<u32, N_LEAVES>;
  let lb = oc * N_LEAVES_I;
  for (var k: i32 = 0; k < N_LEAVES_I; k = k + 1) {
    let c = leafC[lb + k];
    let dy = leafDy[lb + k];
    let dx = leafDx[lb + k];
    let y = oy * S_H - P_H + dy;
    let x = ox * S_W - P_W + dx;
    var v: u32 = 0u;
    if (y >= 0 && y < H && x >= 0 && x < W) {
      v = u32(xBuf[xBase + c * H * W + y * W + x]);
    }
    leaf[k] = v;
  }

  var nGates = N_LEAVES_I;
  var offset: i32 = 0;
  loop {
    if (nGates <= 1) { break; }
    let gOut = nGates / 2;
    for (var g: i32 = 0; g < gOut; g = g + 1) {
      let tt = ttNum[oc * N_INTERNAL + offset + g];
      leaf[g] = gate2(tt, leaf[2 * g], leaf[2 * g + 1]);
    }
    offset = offset + gOut;
    nGates = gOut;
  }

  outBuf[idx] = i32(leaf[0] & 1u);
}
`;
  return { code, nInputs: 1, staticArrays: [leafC, leafDy, leafDx, ttNum], width: outWidth };
}

/** lut_layer(n_inputs<=5、真理値表32bitに収まる高速経路のみ。irflat.ts
    evalLutLayer の n<=5 分岐と同一契約。or_pool2d展開はnIn=k*k<=6だが実モデル
    はk=2=>nIn=4のみでこの経路に収まる。n>5はBigInt必須でWGSL側は未対応な
    ため明示的にthrowする)。 */
function buildLutLayer(p: LutLayerParams, inWidth: number): ShaderBuild {
  const n = p.n_inputs;
  if (n > 5) {
    throw new Error(`webgpuGraph lut_layer は n_inputs<=5 のみ対応(実際: ${n})`);
  }
  const k = p.out_width;
  const idxBuf = new Int32Array(n * k);
  for (let j = 0; j < n; j++) {
    for (let o = 0; o < k; o++) idxBuf[j * k + o] = p.indices[j][o];
  }
  const ttBuf = new Uint32Array(k);
  for (let o = 0; o < k; o++) ttBuf[o] = Number(ttParse(p.truth_tables[o]));

  const code = `
${PARAMS_STRUCT_WGSL}
@group(0) @binding(1) var<storage, read> xBuf: array<i32>;
@group(0) @binding(2) var<storage, read> idxBuf: array<i32>;
@group(0) @binding(3) var<storage, read> ttBuf: array<u32>;
@group(0) @binding(4) var<storage, read_write> outBuf: array<i32>;

const IN_WIDTH: i32 = ${inWidth};
const OUT_WIDTH: i32 = ${k};
const N_INPUTS: i32 = ${n};

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let gidx = gid.y * params._p0 + gid.x;
  if (gidx >= params.total) { return; }
  let idx = i32(gidx);
  let b = idx / OUT_WIDTH;
  let o = idx % OUT_WIDTH;
  let xBase = b * IN_WIDTH;
  var addr: u32 = 0u;
  for (var j: i32 = 0; j < N_INPUTS; j = j + 1) {
    let srcPos = idxBuf[j * OUT_WIDTH + o];
    let bit = u32(xBuf[xBase + srcPos]);
    addr = addr | (bit << u32(j));
  }
  outBuf[idx] = i32((ttBuf[o] >> addr) & 1u);
}
`;
  return { code, nInputs: 1, staticArrays: [idxBuf, ttBuf], width: k };
}

/** upsample2d(nearest) / permute 共通の gather シェーダ。恒等コピーの
    index配列(srcOf)だけが異なる(index計算はJIT時=構築時に1回だけ行う。
    packedConv.ts evalUpsample2dPacked/evalPermutePacked の逐次計算と違い、
    ここではbatch非依存の静的テーブルとして事前確定する)。 */
function buildGather(srcOf: Int32Array, inWidth: number, outWidth: number): ShaderBuild {
  const code = `
${PARAMS_STRUCT_WGSL}
@group(0) @binding(1) var<storage, read> xBuf: array<i32>;
@group(0) @binding(2) var<storage, read> srcOf: array<i32>;
@group(0) @binding(3) var<storage, read_write> outBuf: array<i32>;

const IN_WIDTH: i32 = ${inWidth};
const OUT_WIDTH: i32 = ${outWidth};

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let gidx = gid.y * params._p0 + gid.x;
  if (gidx >= params.total) { return; }
  let idx = i32(gidx);
  let b = idx / OUT_WIDTH;
  let o = idx % OUT_WIDTH;
  outBuf[idx] = xBuf[b * IN_WIDTH + srcOf[o]];
}
`;
  return { code, nInputs: 1, staticArrays: [srcOf], width: outWidth };
}

/** upsample2d の srcOf 事前計算。lowerConv.ts upsampleLutNode / packedConv.ts
    evalUpsample2dPacked と同一の index 式(out[c,y,x]=in[c,y//s,x//s])。 */
function computeUpsampleSrcOf(p: UpSample2dParams): { srcOf: Int32Array; outWidth: number } {
  const [C, H, W] = p.in_shape;
  const s = p.scale;
  const HOut = H * s;
  const WOut = W * s;
  const outWidth = C * HOut * WOut;
  const srcOf = new Int32Array(outWidth);
  for (let c = 0; c < C; c++) {
    for (let y = 0; y < HOut; y++) {
      for (let x = 0; x < WOut; x++) {
        const outIdx = (c * HOut + y) * WOut + x;
        srcOf[outIdx] = (c * H + Math.floor(y / s)) * W + Math.floor(x / s);
      }
    }
  }
  return { srcOf, outWidth };
}

/** permute の srcOf 事前計算。lowerConv.ts permuteLutNode / packedConv.ts
    evalPermutePacked と同一のstrides式。 */
function computePermuteSrcOf(p: PermuteParams): { srcOf: Int32Array; outWidth: number } {
  const dims = p.in_shape;
  const perm = p.perm;
  const n = dims.length;
  const width = dims.reduce((a, b) => a * b, 1);

  const inStrides = new Array<number>(n).fill(1);
  for (let i = n - 2; i >= 0; i--) inStrides[i] = inStrides[i + 1] * dims[i + 1];
  const outDims = perm.map((q) => dims[q]);
  const outStrides = new Array<number>(n).fill(1);
  for (let i = n - 2; i >= 0; i--) outStrides[i] = outStrides[i + 1] * outDims[i + 1];

  const srcOf = new Int32Array(width);
  for (let outIdx = 0; outIdx < width; outIdx++) {
    let rem = outIdx;
    let src = 0;
    for (let i = 0; i < n; i++) {
      const coord = Math.floor(rem / outStrides[i]);
      rem -= coord * outStrides[i];
      src += coord * inStrides[perm[i]];
    }
    srcOf[outIdx] = src;
  }
  return { srcOf, outWidth: width };
}

/** group_sum: irflat.ts evalGroupSum と同一(グループ内総和)。 */
function buildGroupSum(p: GroupSumParams, inWidth: number): ShaderBuild {
  const k = p.k;
  if (inWidth % k !== 0) throw new Error(`group_sum: 幅${inWidth}がk=${k}で割り切れない`);
  const gsz = inWidth / k;
  const code = `
${PARAMS_STRUCT_WGSL}
@group(0) @binding(1) var<storage, read> xBuf: array<i32>;
@group(0) @binding(2) var<storage, read_write> outBuf: array<i32>;

const IN_WIDTH: i32 = ${inWidth};
const K: i32 = ${k};
const GSZ: i32 = ${gsz};

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let gidx = gid.y * params._p0 + gid.x;
  if (gidx >= params.total) { return; }
  let idx = i32(gidx);
  let b = idx / K;
  let g = idx % K;
  let base = b * IN_WIDTH + g * GSZ;
  var s: i32 = 0;
  for (var i: i32 = 0; i < GSZ; i = i + 1) {
    s = s + xBuf[base + i];
  }
  outBuf[idx] = s;
}
`;
  return { code, nInputs: 1, staticArrays: [], width: k };
}

/** concat: irflat.ts evalConcat と同一(slot順にオフセットコピー)。 */
function buildConcat(srcWidths: number[]): ShaderBuild {
  const n = srcWidths.length;
  if (n < 2) throw new Error(`webgpuGraph concat は2入力以上が必要(実際: ${n})`);
  const outWidth = srcWidths.reduce((a, b) => a + b, 0);
  const bindings = Array.from(
    { length: n },
    (_, i) => `@group(0) @binding(${i + 1}) var<storage, read> in${i}: array<i32>;`,
  ).join('\n');
  const outBinding = `@group(0) @binding(${n + 1}) var<storage, read_write> outBuf: array<i32>;`;
  const widthConsts = srcWidths.map((w, i) => `const W${i}: i32 = ${w};`).join('\n');
  const lines: string[] = ['  var pos = idx % OUT_WIDTH;'];
  for (let i = 0; i < n; i++) {
    if (i < n - 1) {
      lines.push(`  if (pos < W${i}) { outBuf[idx] = in${i}[b * W${i} + pos]; return; }`);
      lines.push(`  pos = pos - W${i};`);
    } else {
      lines.push(`  outBuf[idx] = in${i}[b * W${i} + pos];`);
    }
  }
  const code = `
${PARAMS_STRUCT_WGSL}
${bindings}
${outBinding}

const OUT_WIDTH: i32 = ${outWidth};
${widthConsts}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let gidx = gid.y * params._p0 + gid.x;
  if (gidx >= params.total) { return; }
  let idx = i32(gidx);
  let b = idx / OUT_WIDTH;
${lines.join('\n')}
}
`;
  return { code, nInputs: n, staticArrays: [], width: outWidth };
}

/** add: irflat.ts evalAdd と同一(mode: or=ビットOR/xor=ビットXOR/sum=整数和)。
    全slot同幅(width)前提はIR契約どおり(呼び出し側=先行ノードの実測幅で
    そろえる)。 */
function buildAdd(mode: string, width: number, nInputs: number): ShaderBuild {
  if (nInputs < 2) throw new Error(`webgpuGraph add は2入力以上が必要(実際: ${nInputs})`);
  const op = mode === 'or' ? '|' : mode === 'xor' ? '^' : '+';
  const bindings = Array.from(
    { length: nInputs },
    (_, i) => `@group(0) @binding(${i + 1}) var<storage, read> in${i}: array<i32>;`,
  ).join('\n');
  const outBinding = `@group(0) @binding(${nInputs + 1}) var<storage, read_write> outBuf: array<i32>;`;
  const combine = Array.from({ length: nInputs - 1 }, (_, i) => `  v = v ${op} in${i + 1}[idx];`).join('\n');
  const code = `
${PARAMS_STRUCT_WGSL}
${bindings}
${outBinding}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let gidx = gid.y * params._p0 + gid.x;
  if (gidx >= params.total) { return; }
  let idx = i32(gidx);
  var v: i32 = in0[idx];
${combine}
  outBuf[idx] = v;
}
`;
  return { code, nInputs, staticArrays: [], width };
}

/** binarize: irflat.ts の `x.data[i] > thr ? 1 : 0` と同一(f32比較。
    thrは通常0.5でgroup_sum等の小整数出力に対しては f32 丸めの影響を受けない)。 */
function buildBinarize(threshold: number, width: number): ShaderBuild {
  const code = `
${PARAMS_STRUCT_WGSL}
@group(0) @binding(1) var<storage, read> xBuf: array<i32>;
@group(0) @binding(2) var<storage, read_write> outBuf: array<i32>;

const THR: f32 = ${threshold}f;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let gidx = gid.y * params._p0 + gid.x;
  if (gidx >= params.total) { return; }
  let idx = i32(gidx);
  outBuf[idx] = select(0, 1, f32(xBuf[idx]) > THR);
}
`;
  return { code, nInputs: 1, staticArrays: [], width };
}

// ============================================================
// グラフ常駐executor本体
// ============================================================

interface ComputeMeta {
  kind: 'compute';
  width: number;
  inputIds: string[];
  pipeline: GPUComputePipeline;
  totalBuf: GPUBuffer;
  staticBuffers: GPUBuffer[];
}
interface AliasMeta {
  kind: 'alias';
  width: number;
  aliasOf: string;
}
interface LiteralMeta {
  kind: 'literal';
  width: number;
  bits: Int32Array;
}
interface InputMeta {
  kind: 'input';
  width: number;
}
type NodeMeta = ComputeMeta | AliasMeta | LiteralMeta | InputMeta;

/** IR(ConvRuntime.lowered、nativeConv=true)をGPU常駐グラフとして構築・実行する。
    構築(コンストラクタ)は全ノードのシェーダmodule/pipeline/静的バッファを
    1回だけ用意する(cold load費用)。evalAll()は「入力書き込み→全ノードの
    compute passを1つのcommandEncoderへ→1回submit→最終出力だけmapAsync」を
    行う(per-frameのCPU↔GPU同期は1回のみ)。 */
export class WebgpuGraphRuntime {
  private readonly device: GPUDevice;
  private readonly nodeMeta = new Map<string, NodeMeta>();
  private readonly allocOrder: string[]; // input port群 → topoノード群(alias含む)
  private readonly dispatchOrder: string[]; // computeノードのみ、topo順
  private readonly inputId: string;
  private readonly inputWidth: number;
  private readonly outputId: string;

  private batch = -1;
  private readonly valueBuffers = new Map<string, GPUBuffer>();
  private readonly bindGroups = new Map<string, GPUBindGroup>();
  private readBuf: GPUBuffer | null = null;

  // --- プロファイリング状態(既定オフ、setProfiling(true)まで一切参照しない) ---
  private readonly nodeTypeById = new Map<string, string>();
  private profiling = false;
  private profilingInited = false;
  private profilingSupported = false;
  private querySet: GPUQuerySet | null = null;
  private queryResolveBuf: GPUBuffer | null = null;
  private queryReadBuf: GPUBuffer | null = null;
  private readonly nodeGpuAccumNs = new Map<string, bigint>();
  private readonly nodeGpuCount = new Map<string, number>();
  private cpuAccum: WebgpuCpuBreakdown = {
    encodeMs: 0, writeBufferMs: 0, submitToMapMs: 0, readbackCopyMs: 0, totalMs: 0,
  };
  private cpuFrameCount = 0;

  constructor(device: GPUDevice, ir: IRGraphJson) {
    this.device = device;
    if (ir.inputs.length !== 1) {
      throw new Error(`webgpuGraph は単一入力ポートのみ対応(実際 ${ir.inputs.length})`);
    }
    this.inputId = ir.inputs[0].id;
    this.inputWidth = ir.inputs[0].width;
    this.nodeMeta.set(this.inputId, { kind: 'input', width: this.inputWidth });

    const inSrcs = collectInputSources(ir);
    const topo = topologicalOrder(ir, inSrcs);
    const allocOrder: string[] = [this.inputId];
    const dispatchOrder: string[] = [];

    for (const node of topo) {
      const srcs = inSrcs.get(node.id) ?? [];
      const meta = this.buildNode(node, srcs);
      this.nodeMeta.set(node.id, meta);
      this.nodeTypeById.set(node.id, node.type as string);
      allocOrder.push(node.id);
      if (meta.kind === 'compute') dispatchOrder.push(node.id);
    }
    this.allocOrder = allocOrder;
    this.dispatchOrder = dispatchOrder;

    const outPort = ir.outputs[0];
    const outSrc = inSrcs.get(outPort.id)?.[0];
    this.outputId = outSrc !== undefined ? outSrc : ir.nodes[ir.nodes.length - 1].id;
  }

  /** プロファイリングの有効/無効を切り替える(オプトイン)。初回有効化時
      のみ timestamp-query 用の QuerySet/読み戻しバッファを確保する(以後は
      使い回す。batch変化では作り直さない=ensureBatchの対象外)。デバイスが
      timestamp-query featureを持たない場合はGPUノード別計測を黙殺せず諦める
      (profilingSupported=false)。CPU側分解の収集は継続する。 */
  setProfiling(enabled: boolean): void {
    this.profiling = enabled;
    if (!enabled || this.profilingInited) return;
    this.profilingInited = true;
    this.profilingSupported = this.device.features.has('timestamp-query')
      && this.dispatchOrder.length > 0;
    if (!this.profilingSupported) return;
    const count = this.dispatchOrder.length * 2;
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

  /** 蓄積した統計をクリアする(ウォームアップ後・本計測開始前に
      呼ぶことを想定)。 */
  resetProfilingStats(): void {
    this.nodeGpuAccumNs.clear();
    this.nodeGpuCount.clear();
    this.cpuAccum = {
      encodeMs: 0, writeBufferMs: 0, submitToMapMs: 0, readbackCopyMs: 0, totalMs: 0,
    };
    this.cpuFrameCount = 0;
  }

  /** 蓄積した統計のスナップショットを返す(未計測ならnull)。
      ノード別GPU時間は平均・降順で上位10件のみ。 */
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
        nodeType: this.nodeTypeById.get(id) ?? '?',
        gpuMs: Number(sumNs) / 1e6 / cnt,
      });
    }
    nodeAvgs.sort((a, b) => b.gpuMs - a.gpuMs);
    return {
      timestampSupported: this.profilingSupported,
      frameCount: n,
      topNodesGpu: nodeAvgs.slice(0, 10),
      cpu,
    };
  }

  private srcWidth(srcId: string): number {
    return this.nodeMeta.get(srcId)!.width;
  }

  private makeComputeMeta(build: ShaderBuild, inputIds: string[]): ComputeMeta {
    const device = this.device;
    const module = device.createShaderModule({ code: build.code });
    const pipeline = device.createComputePipeline({
      layout: 'auto',
      compute: { module, entryPoint: 'main' },
    });
    const totalBuf = device.createBuffer({
      size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    const staticBuffers = build.staticArrays.map((arr) => {
      const buf = device.createBuffer({
        size: Math.max(4, arr.byteLength),
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });
      device.queue.writeBuffer(buf, 0, arr as unknown as ArrayBufferView<ArrayBuffer>);
      return buf;
    });
    return {
      kind: 'compute', width: build.width, inputIds, pipeline, totalBuf, staticBuffers,
    };
  }

  /** ノード1つぶんのメタを構築する(WGSL生成+静的データ確定)。evalIrFlat/
      evalIrFlatAsync のswitchと同じノード型集合を扱う(型を追加した場合は
      両方に追加すること)。 */
  private buildNode(node: IRNodeJson, srcs: string[]): NodeMeta {
    switch (node.type as string) {
      case 'conv2d': {
        if (srcs.length !== 1) throw new Error(`${node.id}: conv2dは単一入力`);
        const inWidth = this.srcWidth(srcs[0]);
        const build = buildConv2d(node.params as unknown as Conv2dParams, inWidth);
        return this.makeComputeMeta(build, srcs);
      }
      case 'lut_layer': {
        if (srcs.length !== 1) throw new Error(`${node.id}: lut_layerは単一入力`);
        const inWidth = this.srcWidth(srcs[0]);
        const build = buildLutLayer(node.params as unknown as LutLayerParams, inWidth);
        return this.makeComputeMeta(build, srcs);
      }
      case 'upsample2d': {
        if (srcs.length !== 1) throw new Error(`${node.id}: upsample2dは単一入力`);
        const inWidth = this.srcWidth(srcs[0]);
        const { srcOf, outWidth } = computeUpsampleSrcOf(node.params as unknown as UpSample2dParams);
        const build = buildGather(srcOf, inWidth, outWidth);
        return this.makeComputeMeta(build, srcs);
      }
      case 'permute': {
        if (srcs.length !== 1) throw new Error(`${node.id}: permuteは単一入力`);
        const inWidth = this.srcWidth(srcs[0]);
        const { srcOf, outWidth } = computePermuteSrcOf(node.params as unknown as PermuteParams);
        const build = buildGather(srcOf, inWidth, outWidth);
        return this.makeComputeMeta(build, srcs);
      }
      case 'group_sum': {
        if (srcs.length !== 1) throw new Error(`${node.id}: group_sumは単一入力`);
        const inWidth = this.srcWidth(srcs[0]);
        const build = buildGroupSum(node.params as unknown as GroupSumParams, inWidth);
        return this.makeComputeMeta(build, srcs);
      }
      case 'flatten': {
        if (srcs.length !== 1) throw new Error(`${node.id}: flattenは単一入力`);
        return { kind: 'alias', width: this.srcWidth(srcs[0]), aliasOf: srcs[0] };
      }
      case 'concat': {
        const widths = srcs.map((s) => this.srcWidth(s));
        const build = buildConcat(widths);
        return this.makeComputeMeta(build, srcs);
      }
      case 'add': {
        const mode = (node.params as unknown as AddParams).mode as string;
        const width = this.srcWidth(srcs[0]);
        const build = buildAdd(mode, width, srcs.length);
        return this.makeComputeMeta(build, srcs);
      }
      case 'binarize': {
        if (srcs.length !== 1) throw new Error(`${node.id}: binarizeは単一入力`);
        const width = this.srcWidth(srcs[0]);
        const thr = (node.params.threshold as number | undefined) ?? 0.5;
        const build = buildBinarize(thr, width);
        return this.makeComputeMeta(build, srcs);
      }
      case 'constant': {
        const bits = node.params.bits as number[];
        return { kind: 'literal', width: bits.length, bits: Int32Array.from(bits) };
      }
      default:
        throw new Error(`webgpuGraph 未対応ノード型: ${node.id}(${node.type})`);
    }
  }

  private buildBindGroup(id: string, meta: ComputeMeta): GPUBindGroup {
    const entries: GPUBindGroupEntry[] = [{ binding: 0, resource: { buffer: meta.totalBuf } }];
    meta.inputIds.forEach((srcId, i) => {
      entries.push({ binding: 1 + i, resource: { buffer: this.valueBuffers.get(srcId)! } });
    });
    const staticBase = 1 + meta.inputIds.length;
    meta.staticBuffers.forEach((buf, i) => {
      entries.push({ binding: staticBase + i, resource: { buffer: buf } });
    });
    entries.push({
      binding: staticBase + meta.staticBuffers.length,
      resource: { buffer: this.valueBuffers.get(id)! },
    });
    return this.device.createBindGroup({ layout: meta.pipeline.getBindGroupLayout(0), entries });
  }

  /** バッチサイズが変わった時だけ、全ノード値バッファ/bindGroup/読み戻し
      バッファを作り直す(通常runは同一batchを繰り返すため初回のみ発生)。
      truth_table/wiring等の「静的データ」バッファ(makeComputeMetaで構築済み)
      はここでは触らない=frame間/batch間で再利用される。 */
  private ensureBatch(batch: number): void {
    if (batch === this.batch) return;
    const device = this.device;

    for (const id of this.allocOrder) {
      const meta = this.nodeMeta.get(id)!;
      if (meta.kind === 'alias') continue;
      const buf = device.createBuffer({
        size: Math.max(4, batch * meta.width * 4),
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
      });
      this.valueBuffers.set(id, buf);
      if (meta.kind === 'literal') {
        const tiled = new Int32Array(batch * meta.width);
        for (let b = 0; b < batch; b++) tiled.set(meta.bits, b * meta.width);
        device.queue.writeBuffer(buf, 0, tiled);
      }
    }
    // alias(flatten)は解決済みバッファを参照する(alias連鎖にも対応)。
    for (const id of this.allocOrder) {
      const meta = this.nodeMeta.get(id)!;
      if (meta.kind !== 'alias') continue;
      let target = meta.aliasOf;
      let targetMeta = this.nodeMeta.get(target)!;
      while (targetMeta.kind === 'alias') {
        target = targetMeta.aliasOf;
        targetMeta = this.nodeMeta.get(target)!;
      }
      this.valueBuffers.set(id, this.valueBuffers.get(target)!);
    }
    // computeノード: totalユニフォーム更新 + bindGroup再構築。
    for (const id of this.dispatchOrder) {
      const meta = this.nodeMeta.get(id)! as ComputeMeta;
      const total = batch * meta.width;
      // xThreadsを_p0へ(WGSL側の2Dインデックス復元用)
      const d2 = dispatch2d(
        total, device.limits.maxComputeWorkgroupsPerDimension);
      device.queue.writeBuffer(
        meta.totalBuf, 0, new Uint32Array([total, d2.xThreads, 0, 0]));
      this.bindGroups.set(id, this.buildBindGroup(id, meta));
    }

    const outWidth = this.nodeMeta.get(this.outputId)!.width;
    this.readBuf = device.createBuffer({
      size: Math.max(4, batch * outWidth * 4),
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
    this.batch = batch;
  }

  /** [batch*inWidth]の0/1入力→[batch*outWidth]の整数スコア。ConvRuntime.
      scoresAsync('packed-webgpu')の唯一の実行経路。1フレームにつき
      queue.writeBuffer(入力)1回 + queue.submit 1回 + mapAsync 1回のみ
      (中間ノードはGPU常駐バッファのまま連鎖し、CPUへ読み戻さない)。 */
  /** 初回evalAllのみvalidationエラースコープを張ったか。*/
  private validated = false;

  async evalAll(x: NodeValue, batch: number): Promise<NodeValue> {
    if (x.data.length !== batch * this.inputWidth) {
      throw new Error(
        `webgpuGraph 入力: 期待${batch * this.inputWidth}要素, 実際${x.data.length}`,
      );
    }
    this.ensureBatch(batch);
    const device = this.device;
    // プロファイリング無効時はperformance.now()呼び出し自体を挟まない
    // (ホットパスにコストを足さない、というオプトイン契約を型でも徹底する)。
    const profiling = this.profiling;
    const tFrameStart = profiling ? performance.now() : 0;
    // 黙殺根絶。従来はvalidationエラー(例: dispatch1次元上限65,535超)で
    // command bufferごと破棄され「出力ゼロ=box 0/0」が黙って出ていた。初回だけ
    // エラースコープで捕捉してthrowし、上位(DetDlgnRuntime等)のフォールバック
    // 経路に理由を載せる。
    const firstEval = !this.validated;
    if (firstEval) device.pushErrorScope('validation');

    const inputBuf = this.valueBuffers.get(this.inputId)!;
    const tWriteStart = profiling ? performance.now() : 0;
    device.queue.writeBuffer(inputBuf, 0, x.data as unknown as ArrayBufferView<ArrayBuffer>);
    const tWriteEnd = profiling ? performance.now() : 0;

    const encoder = device.createCommandEncoder();
    let passIndex = 0;
    for (const id of this.dispatchOrder) {
      const meta = this.nodeMeta.get(id)! as ComputeMeta;
      const bindGroup = this.bindGroups.get(id)!;
      const total = batch * meta.width;
      const passDesc: GPUComputePassDescriptor = {};
      if (profiling && this.profilingSupported) {
        passDesc.timestampWrites = {
          querySet: this.querySet!,
          beginningOfPassWriteIndex: passIndex * 2,
          endOfPassWriteIndex: passIndex * 2 + 1,
        };
      }
      const pass = encoder.beginComputePass(passDesc);
      pass.setPipeline(meta.pipeline);
      pass.setBindGroup(0, bindGroup);
      const d2 = dispatch2d(
        total, this.device.limits.maxComputeWorkgroupsPerDimension);
      pass.dispatchWorkgroups(d2.x, d2.y);
      pass.end();
      passIndex++;
    }

    const outWidth = this.nodeMeta.get(this.outputId)!.width;
    const outBuf = this.valueBuffers.get(this.outputId)!;
    const readBuf = this.readBuf!;
    encoder.copyBufferToBuffer(outBuf, 0, readBuf, 0, Math.max(4, batch * outWidth * 4));
    if (profiling && this.profilingSupported) {
      const nPasses = this.dispatchOrder.length;
      encoder.resolveQuerySet(this.querySet!, 0, nPasses * 2, this.queryResolveBuf!, 0);
      encoder.copyBufferToBuffer(this.queryResolveBuf!, 0, this.queryReadBuf!, 0, nPasses * 2 * 8);
    }
    const tEncodeEnd = profiling ? performance.now() : 0;

    device.queue.submit([encoder.finish()]);
    const tSubmitStart = profiling ? performance.now() : 0;

    if (firstEval) {
      const err = await device.popErrorScope();
      if (err) {
        throw new Error(`webgpuGraph validationエラー(黙殺せずthrow): ${err.message}`);
      }
      this.validated = true;
    }

    if (profiling && this.profilingSupported) {
      await Promise.all([
        readBuf.mapAsync(GPUMapMode.READ),
        this.queryReadBuf!.mapAsync(GPUMapMode.READ),
      ]);
    } else {
      await readBuf.mapAsync(GPUMapMode.READ);
    }
    const tMapEnd = profiling ? performance.now() : 0;

    const mapped = new Int32Array(readBuf.getMappedRange().slice(0, batch * outWidth * 4));
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
        for (let i = 0; i < this.dispatchOrder.length; i++) {
          const id = this.dispatchOrder[i];
          const beginNs = raw[i * 2];
          const endNs = raw[i * 2 + 1];
          const diff = endNs >= beginNs ? endNs - beginNs : 0n;
          this.nodeGpuAccumNs.set(id, (this.nodeGpuAccumNs.get(id) ?? 0n) + diff);
          this.nodeGpuCount.set(id, (this.nodeGpuCount.get(id) ?? 0) + 1);
        }
        this.queryReadBuf!.unmap();
      }
    }

    return { data: mapped, width: outWidth };
  }
}
