// WebGPU側の自己較正ルーフライン計測(deploykit-integrationキャンペーン)。
//
// scripts/gpu_calib/metal_calib.mm / gate2_ab.mm(Metal側、M3直計測)と
// **同一のカーネル設計**をWGSLへ移植する(比較列として並べるため、命令内訳・
// アキュムレータ本数・アクセスパターンを可能な限り一致させる。乱数列そのものは
// C++のstd::mt19937と一致させていないため、命令列は構造的に同型だが
// ビット単位で同一ではない点に注意)。
//
//   (a) 純ALU較正カーネル(buildAluCalibWgsl): 「メモリアクセスなし・分岐なし・
//       依存連鎖なし」の32bit XOR/AND/OR を実行し、
//       (総命令数 ÷ GPU時間) = このマシンのWebGPU実測ALUピーク(instr/s) を出す。
//       依存連鎖を避けるため8本の独立アキュムレータ(a0..a7)を持ち、反復ごとに
//       異なるマスク値(実行時値cとコンパイル時定数のXOR)を適用する
//       (同一値の繰り返しXOR/ANDは代数的に畳まれてしまう罠を避ける。
//       metal_calib.mmコメント参照)。
//
//   (b) gate2 A/Bマイクロベンチ(buildGate2AbWgsl): 実カーネルと同じ
//       「命令列(tt,a_off,b_off,out_off)をストレージバッファから読み、
//       スクラッチ配列へread-modify-write」ループで、switch(tt&0xf)の16分岐
//       (webgpuGraph.ts::GATE2_WGSLをそのまま再利用=乖離防止)と、
//       無分岐マスク式 gate2Mask を比較する。
// 上記A/Bで無分岐マスク式が有意に速いと実測できたため、本番の
//       webgpuGraph.ts::GATE2_WGSL 自体をswitchからmask式へ置換済み。よって
//       main_switch はもう「switch実装」ではなく「gate2(=mask式)」を呼ぶ
//       (=main_maskと退化的に同一)。過去の実測値の再現性確認・将来別の
//       候補式が出た時の比較列として残す(意図的、削除不要)。
//
// このファイルはGPUデバイスに一切触れない純関数のみを持つ(WGSL生成・
// 命令列生成はnode/vitestで検証できる。実行はgpualubench_main.tsが担う)。
import { GATE2_WGSL } from './webgpuGraph';

/** 1反復あたりの命令数: マスクmを作るXOR1回 + 8本のアキュムレータへの適用8回。
    metal_calib.mm::kOpsPerIter と同一(比較列を保つため値を合わせること)。 */
export const ALU_OPS_PER_ITER = 9;

/** 32bit符号なし乗算(JSのNumberは倍精度なので、桁があふれる前に一度
    >>> 0 へ落として metal_calib.mm の `(uint32_t)(i * 0x9E3779B1u)` と
    同じラップアラウンドを再現する。i<=4096程度なら乗算自体は
    Number.MAX_SAFE_INTEGER 内に収まるため精度損失は起きない)。 */
function wrappingMulU32(i: number, k: number): number {
  return (i * k) >>> 0;
}

/** 純ALU較正カーネルのWGSLを生成する(unroll回ぶん完全展開)。
    メモリアクセスは末尾の1回のストア(outBuf[idx]=...)のみ、分岐は
    グリッド超過ガード1個のみ(total を workgroup_size の倍数にすれば
    実質発火しない)、依存連鎖は8本の独立アキュムレータに分散させて
    避ける(metal_calib.mm::BuildKernelSource と同一設計)。 */
export function buildAluCalibWgsl(unroll: number): string {
  if (!Number.isInteger(unroll) || unroll < 1) {
    throw new Error(`buildAluCalibWgsl: unroll=${unroll} は1以上の整数のみ`);
  }
  const lines: string[] = [];
  for (let i = 0; i < unroll; i++) {
    const k = wrappingMulU32(i, 0x9e3779b1);
    lines.push(
      `  m = c ^ ${k}u; a0 ^= m; a1 &= m; a2 |= m; a3 ^= m;`
      + ` a4 &= m; a5 |= m; a6 ^= m; a7 &= m;`,
    );
  }
  return `struct AluParams { seed: u32, total: u32, _p2: u32, _p3: u32 };
@group(0) @binding(0) var<storage, read_write> outBuf: array<u32>;
@group(0) @binding(1) var<uniform> params: AluParams;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= params.total) { return; }
  let c = params.seed ^ idx;
  var a0 = idx; var a1 = idx + 1u; var a2 = idx + 2u; var a3 = idx + 3u;
  var a4 = idx + 5u; var a5 = idx + 7u; var a6 = idx + 11u; var a7 = idx + 13u;
  var m: u32 = 0u;
${lines.join('\n')}
  outBuf[idx] = a0 ^ a1 ^ a2 ^ a3 ^ a4 ^ a5 ^ a6 ^ a7;
}
`;
}

/** gate2 A/BマイクロベンチのWGSL(switch/mask 両エントリポイントを1モジュールに
    持つ。gate2 は webgpuGraph.ts::GATE2_WGSL の共有定義そのもの)。
    scratchレイアウトは gate2_ab.mm と同一: scratch[slot*threads + tid]
    (スロット番号 → ワードオフセットは呼び出し側が threads 倍済みで instrBuf に
    詰める。scaleProgramForThreads 参照)。 */
export function buildGate2AbWgsl(): string {
  return `struct Gate2Params { nInstr: u32, threads: u32, nSlots: u32, _p3: u32 };
@group(0) @binding(0) var<storage, read_write> scratch: array<u32>;
@group(0) @binding(1) var<storage, read> instrBuf: array<u32>;
@group(0) @binding(2) var<uniform> params: Gate2Params;

${GATE2_WGSL}

/** 無分岐マスク式(A/B比較対象)。標準的な2入力LUTの最小項展開。
    gate2_ab.mm::mdl_g_gate2_mask と同一の式。 */
fn gate2Mask(tt: u32, a: u32, b: u32) -> u32 {
  let m0 = 0u - (tt & 1u);
  let m1 = 0u - ((tt >> 1u) & 1u);
  let m2 = 0u - ((tt >> 2u) & 1u);
  let m3 = 0u - ((tt >> 3u) & 1u);
  let na = ~a;
  let nb = ~b;
  return (m0 & na & nb) | (m1 & a & nb) | (m2 & na & b) | (m3 & a & b);
}

/** 両カーネルとも初期化式は完全に同一(同じ入力から始めることで、命令列の
    実行結果だけを比較できるようにする)。gate2_ab.mm::mdl_calib_init と同一。 */
fn calibInit(tid: u32, s: u32) -> u32 {
  var v = tid ^ (s * 2654435761u) ^ 0x9e3779b9u;
  v ^= v << 13u;
  v ^= v >> 17u;
  v ^= v << 5u;
  return v;
}

@compute @workgroup_size(64)
fn main_switch(@builtin(global_invocation_id) gid: vec3<u32>) {
  let tid = gid.x;
  if (tid >= params.threads) { return; }
  for (var s: u32 = 0u; s < params.nSlots; s = s + 1u) {
    scratch[s * params.threads + tid] = calibInit(tid, s);
  }
  for (var j: u32 = 0u; j < params.nInstr; j = j + 1u) {
    let o = j * 4u;
    let tt = instrBuf[o];
    let a = scratch[instrBuf[o + 1u] + tid];
    let b = scratch[instrBuf[o + 2u] + tid];
    scratch[instrBuf[o + 3u] + tid] = gate2(tt, a, b);
  }
}

@compute @workgroup_size(64)
fn main_mask(@builtin(global_invocation_id) gid: vec3<u32>) {
  let tid = gid.x;
  if (tid >= params.threads) { return; }
  for (var s: u32 = 0u; s < params.nSlots; s = s + 1u) {
    scratch[s * params.threads + tid] = calibInit(tid, s);
  }
  for (var j: u32 = 0u; j < params.nInstr; j = j + 1u) {
    let o = j * 4u;
    let tt = instrBuf[o];
    let a = scratch[instrBuf[o + 1u] + tid];
    let b = scratch[instrBuf[o + 2u] + tid];
    scratch[instrBuf[o + 3u] + tid] = gate2Mask(tt, a, b);
  }
}
`;
}

/** xorshift32(決定的PRNG。C++のstd::mt19937とビット互換ではないが、
    再現性のある命令列生成には十分)。seed=0だと縮退するので強制的に1へ補正。 */
function makeXorshift32(seed: number): () => number {
  let s = (seed >>> 0) || 1;
  return () => {
    s ^= s << 13; s >>>= 0;
    s ^= s >>> 17;
    s ^= s << 5; s >>>= 0;
    return s;
  };
}

export interface Gate2Instr {
  tt: number;
  aSlot: number;
  bSlot: number;
  outSlot: number;
}

/** gate2 A/Bベンチの命令列を生成する(gate2_ab.mm::GenProgram と構造は同一:
    tt∈[0,16)一様、a/b/out slot∈[0,nSlots)一様。RNG実装はxorshift32で
    mt19937とは異なるため、Metal側と命令列がビット一致するわけではないが、
    構造(命令数・スロット幅・分布)は揃っている)。 */
export function genGate2Program(nInstr: number, nSlots: number, seed: number): Gate2Instr[] {
  if (!Number.isInteger(nInstr) || nInstr < 1) {
    throw new Error(`genGate2Program: nInstr=${nInstr} は1以上の整数のみ`);
  }
  if (!Number.isInteger(nSlots) || nSlots < 2) {
    throw new Error(`genGate2Program: nSlots=${nSlots} は2以上の整数のみ`);
  }
  const rnd = makeXorshift32(seed);
  const prog: Gate2Instr[] = [];
  for (let i = 0; i < nInstr; i++) {
    prog.push({
      tt: rnd() % 16,
      aSlot: rnd() % nSlots,
      bSlot: rnd() % nSlots,
      outSlot: rnd() % nSlots,
    });
  }
  return prog;
}

/** スロット番号(0..nSlots)を実行時オフセット(スロット*threads)へ展開し、
    WGSL側がそのまま `scratch[instrBuf[o+k] + tid]` で読めるu32配列にする
    (gate2_ab.mm::RunKernel の instr_scaled と同一の詰め方)。 */
export function scaleProgramForThreads(prog: Gate2Instr[], threads: number): Uint32Array {
  if (!Number.isInteger(threads) || threads < 1) {
    throw new Error(`scaleProgramForThreads: threads=${threads} は1以上の整数のみ`);
  }
  const out = new Uint32Array(prog.length * 4);
  for (let i = 0; i < prog.length; i++) {
    const ins = prog[i];
    out[i * 4] = ins.tt;
    out[i * 4 + 1] = ins.aSlot * threads;
    out[i * 4 + 2] = ins.bSlot * threads;
    out[i * 4 + 3] = ins.outSlot * threads;
  }
  return out;
}
