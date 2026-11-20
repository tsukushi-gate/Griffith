// 画像前処理(RGBA → ビット符号化)のGPU化。
//
// 背景: 推論の実効フレーム時間が det 0.88 / detmono 0.59 / detface 1.08 /
// detreal 0.82 / seg 0.29 ms まで落ちた結果、e2e の支配項は **CPU前処理**
// (convpreproc.ts::encodeImageAuto のスカラーJSループ。detface は 128x128x173ch
// = 2.8M ビットを1本ずつ計算するので 6.6ms)になった。ここではその前処理を
// WGSL カーネルへ移し、GPUへ載せるのを「RGBA 64KB」だけにする
// (従来は 2.8MB の 0/1 バイト列を writeBuffer していた)。
//
// ============================================================
// 数値経路(なぜビット完全一致できるのか)
// ============================================================
// CPU側(convpreproc.ts)は Float32Array に入れた画素値を **f64 で** 演算して
// 閾値比較する。WGSL は f32 しか無いので、同じ式を f32 で書くと閾値の境界で
// 1ビット差が出る(例: edge の d = a - b は f32 減算だと丸めが起きるが、f64
// では厳密)。そこで **f32演算を一切使わず厳密整数比較に置き換える**。
//
// 鍵は「GPU前処理の入力は必ず RGBA(u8)である」こと。rgbaToNchw が作る画素値は
// f32(k/255)(k∈[0,255])の256通りしかない。
//
//   (1) f32(k/255) は 2^-31 の整数倍である。
//       k>=1 なら値の指数は [-8, 0] なので ulp = 2^(e-23) >= 2^-31。
//       よって I(k) := f32(k/255) * 2^31 は **厳密な整数**で 0 <= I(k) <= 2^31
//       (u32 に収まる)。k=0 は 0。
//   (2) CPU が計算する量はすべて I の整数結合である。
//       edge : d = x[p+off] - x[p]            → D = I(a) - I(b)          (|D| <= 2^31)
//       curv : d = x[p-off] - 2x[p] + x[p+off] → D = I(a) - 2I(m) + I(c)  (|D| <= 2^32)
//       chroma: d = x[ci] - x[cj]              → D = I(a) - I(b)
//       彩度  : d = max - min                  → D = I(kmax) - I(kmin)
//       (f64 はこれらを丸めなしで表現できる(|D| <= 2^33 << 2^53)ので、
//        CPU の d はちょうど D * 2^-31 に等しい。)
//   (3) 比較も厳密化できる。t は f64、2^31 は2冪なので t*2^31 は厳密。
//       d >  t  ⟺ D >  t*2^31    ⟺ D >= floor(t*2^31) + 1
//       d < -t  ⟺ D < -t*2^31    ⟺ D <= ceil(-t*2^31) - 1
//       (D は整数なので実数比較と整数比較が一致する。)
//   → WGSL 側は u32 二語による 64bit 整数(vec2<u32>、|D| が 33bit まで伸びる
//     curv のため)で D を組み立て、事前計算した cut と比較するだけでよい。
//     浮動小数点は一切現れない。
//
// thermometer だけは z = (x - mean) * (1/std) > th という「任意の f64 係数を
// 含む式」なので整数化できない。しかし入力が 256 通りしかないので、
// **256ビットのルックアップ表**(CPU側とまったく同じJS式で埋める)にすれば
// 定義上ビット一致する。表は記述子に埋め込む(8ワード/チャネル)。
//
// この設計の網羅検証は gpuPreproc.test.ts:
//   - edge/chroma: 全 (k1,k2) 256^2 ペア × 実閾値で CPU式と一致
//   - curv       : 全 (k1,k2,k3) 256^3 三つ組で CPU式と一致
//   - 実モデル5本の preproc 設定 × ランダム画像で encodeImageAuto と全ビット一致
// 実機GPUでの一致は wasmgpu_verify.html(packed planes の全ワード一致)で取る。
import type { ImagePreprocDoc } from '../convpreproc';

/** 固定小数のスケール(2^31)。I(k) = f32(k/255) * 2^31 は厳密整数。 */
const SCALE = 2 ** 31;

/** 記述子1チャネルぶんのワード数(w0..w7 = 演算指定、w8..w15 = thermometer LUT)。 */
export const PREPROC_DESC_STRIDE = 16;

export const PREPROC_OP_THERMO = 0;
export const PREPROC_OP_EDGE = 1;
export const PREPROC_OP_CURV = 2;
export const PREPROC_OP_CHROMA = 3;
export const PREPROC_OP_SAT = 4;

/** 比較の向き。0: D >= cut(d > t 相当) / 1: D <= cut(d < -t 相当)。 */
const DIR_GE = 0;
const DIR_LE = 1;

export interface PreprocDescriptors {
  /** [channels * PREPROC_DESC_STRIDE] の記述子表(エンコーダ出力チャネル順)。 */
  desc: Uint32Array;
  /** I(k) = f32(k/255) * 2^31 の256要素表。 */
  qval: Uint32Array;
  /** エンコーダ出力チャネル数(= 入力ビット幅 / (H*W))。 */
  channels: number;
  /** 画像チャネル数 C(mean.length。彩度の max/min の走査幅)。 */
  imageChannels: number;
  height: number;
  width: number;
}

/** I(k) = f32(k/255) * 2^31。整数でなければ設計前提が崩れているので throw。 */
function quantTable(): Uint32Array {
  const q = new Uint32Array(256);
  for (let k = 0; k < 256; k++) {
    const s = Math.fround(k / 255) * SCALE;
    if (!Number.isInteger(s) || s < 0 || s > 0xffffffff) {
      throw new Error(`gpuPreproc: I(${k})=${s} が u32 の整数でない(固定小数の前提崩れ)`);
    }
    q[k] = s;
  }
  return q;
}

/** 安全整数を 64bit 二の補数の (lo, hi) u32 ペアへ。BigInt を使わず厳密に行う。 */
function i64Words(n: number): [number, number] {
  if (!Number.isSafeInteger(n)) {
    throw new Error(`gpuPreproc: cut=${n} が安全整数でない`);
  }
  const two32 = 4294967296;
  const lo = ((n % two32) + two32) % two32;
  const hi = Math.floor(n / two32);
  return [lo, ((hi % two32) + two32) % two32];
}

/** d > t ⟺ D >= cut。t*2^31 は2冪倍なので厳密。 */
function cutGreater(t: number): number {
  const s = t * SCALE;
  if (!Number.isFinite(s) || Math.abs(s) > 2 ** 52) {
    throw new Error(`gpuPreproc: 閾値 ${t} が固定小数の表現範囲外`);
  }
  return Math.floor(s) + 1;
}

/** d < -t ⟺ D <= cut。 */
function cutLess(t: number): number {
  const s = -t * SCALE;
  if (!Number.isFinite(s) || Math.abs(s) > 2 ** 52) {
    throw new Error(`gpuPreproc: 閾値 ${t} が固定小数の表現範囲外`);
  }
  return Math.ceil(s) - 1;
}

/** 記述子を1チャネルぶん書き込むビルダ。 */
class DescWriter {
  private readonly rows: number[][] = [];

  push(words: number[]): void {
    if (words.length !== PREPROC_DESC_STRIDE) {
      throw new Error(`gpuPreproc: 記述子長 ${words.length} != ${PREPROC_DESC_STRIDE}`);
    }
    this.rows.push(words);
  }

  /** thermometer: 256ビットLUT(CPU側 convpreproc.ts と同一のJS式で埋める)。 */
  pushThermo(c: number, mean: number, std: number, th: number): void {
    const invStd = 1 / std;
    const w = new Array<number>(PREPROC_DESC_STRIDE).fill(0);
    w[0] = PREPROC_OP_THERMO;
    w[1] = c;
    for (let k = 0; k < 256; k++) {
      // convpreproc.ts::encodeImageThermometer / encodeImageLowLevel と同じ式。
      const z = (Math.fround(k / 255) - mean) * invStd;
      if (z > th) w[8 + (k >>> 5)] |= 1 << (k & 31);
    }
    this.rows.push(w.map((v) => v >>> 0));
  }

  pushCmp(op: number, a: number, b: number, dy: number, dx: number, cut: number, dir: number): void {
    const [lo, hi] = i64Words(cut);
    const w = new Array<number>(PREPROC_DESC_STRIDE).fill(0);
    w[0] = op;
    w[1] = a;
    w[2] = b;
    w[3] = dy >>> 0; // i32 の二の補数ビット列として渡す
    w[4] = dx >>> 0;
    w[5] = lo;
    w[6] = hi;
    w[7] = dir;
    this.rows.push(w);
  }

  get length(): number {
    return this.rows.length;
  }

  toArray(): Uint32Array {
    const out = new Uint32Array(this.rows.length * PREPROC_DESC_STRIDE);
    this.rows.forEach((r, i) => out.set(r, i * PREPROC_DESC_STRIDE));
    return out;
  }
}

/** doc.preproc → GPU前処理カーネルの記述子表。
 *
 * チャネル順は convpreproc.ts::encodeImageAuto と**厳密に**同じでなければ
 * ならない(image_thermometer は rgb-major、image_lowlevel は
 * [thermo(thr-major)][edge][curv][chroma pair][saturation])。
 * 未対応の kind は黙って別物を返さず throw する(呼び出し側はCPU前処理へ落ちる)。
 */
export function buildPreprocDescriptors(p: ImagePreprocDoc): PreprocDescriptors {
  const C = p.mean.length;
  if (C < 1 || p.std.length !== C) {
    throw new Error(`gpuPreproc: mean/std の長さ不一致(mean=${C}, std=${p.std.length})`);
  }
  if (C > 4) throw new Error(`gpuPreproc: 画像チャネル数 ${C} は未対応(RGBAの4まで)`);
  for (const s of p.std) {
    if (!(s > 0)) throw new Error(`gpuPreproc: std=${s} が正でない(単調性の前提崩れ)`);
  }
  const T = p.thresholds[0]?.length ?? 0;
  const w = new DescWriter();

  if (p.kind === 'image_thermometer') {
    // rgb-major: ch = c*T + k
    for (let c = 0; c < C; c++) {
      if (p.thresholds[c]?.length !== T) {
        throw new Error(`gpuPreproc: thresholds[${c}] の長さが不揃い`);
      }
      for (let k = 0; k < T; k++) w.pushThermo(c, p.mean[c], p.std[c], p.thresholds[c][k]);
    }
  } else if (p.kind === 'image_lowlevel') {
    const edgeThr = p.edge_thresholds ?? [];
    const curvThr = p.curv_thresholds ?? [];
    const chromaThr = p.chroma_thresholds ?? [];
    const edgeOff = p.edge_offsets ?? [];
    const curvOff = p.curv_offsets ?? [];
    const chromaPairs = p.chroma_pairs ?? [];
    // thermo は thr-major(k が外、c が内)
    for (let k = 0; k < T; k++) {
      for (let c = 0; c < C; c++) w.pushThermo(c, p.mean[c], p.std[c], p.thresholds[c][k]);
    }
    for (const [dy, dx] of edgeOff) {
      for (const t of edgeThr) {
        for (let c = 0; c < C; c++) {
          w.pushCmp(PREPROC_OP_EDGE, c, 0, dy, dx, cutGreater(t), DIR_GE);
        }
        for (let c = 0; c < C; c++) {
          w.pushCmp(PREPROC_OP_EDGE, c, 0, dy, dx, cutLess(t), DIR_LE);
        }
      }
    }
    for (const [dy, dx] of curvOff) {
      for (const t of curvThr) {
        for (let c = 0; c < C; c++) {
          w.pushCmp(PREPROC_OP_CURV, c, 0, dy, dx, cutGreater(t), DIR_GE);
        }
        for (let c = 0; c < C; c++) {
          w.pushCmp(PREPROC_OP_CURV, c, 0, dy, dx, cutLess(t), DIR_LE);
        }
      }
    }
    for (const [ci, cj] of chromaPairs) {
      for (const t of chromaThr) {
        w.pushCmp(PREPROC_OP_CHROMA, ci, cj, 0, 0, cutGreater(t), DIR_GE);
        w.pushCmp(PREPROC_OP_CHROMA, ci, cj, 0, 0, cutLess(t), DIR_LE);
      }
    }
    // 彩度(max-min)は全 chroma ペアの後に閾値順でまとめて置かれる(Python同順)
    for (const t of chromaThr) {
      w.pushCmp(PREPROC_OP_SAT, 0, 0, 0, 0, cutGreater(t), DIR_GE);
    }
  } else {
    throw new Error(`gpuPreproc: 未対応の preproc.kind=${p.kind}`);
  }

  return {
    desc: w.toArray(),
    qval: quantTable(),
    channels: w.length,
    imageChannels: C,
    height: p.height,
    width: p.width,
  };
}

// ============================================================
// WGSL
// ============================================================

/** 64bit 符号付き整数(vec2<u32> = (lo, hi)、二の補数)。curv の D は値域が
    [-2^32, 2^32] で、二の補数では 34bit 必要(33bitは +2^32 を表せない)。
    u32/i32 では表せないためここで64bitをエミュレートする。
    使う演算は加算・符号反転・比較だけ。 */
const I64_WGSL = `
fn i64Add(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
  let lo = a.x + b.x;
  var carry: u32 = 0u;
  if (lo < a.x) { carry = 1u; }
  return vec2<u32>(lo, a.y + b.y + carry);
}
fn i64Neg(a: vec2<u32>) -> vec2<u32> {
  let lo = ~a.x + 1u;
  var carry: u32 = 0u;
  if (lo == 0u) { carry = 1u; }
  return vec2<u32>(lo, ~a.y + carry);
}
fn i64Sub(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> { return i64Add(a, i64Neg(b)); }
fn i64Ge(a: vec2<u32>, b: vec2<u32>) -> bool {
  let ah = bitcast<i32>(a.y);
  let bh = bitcast<i32>(b.y);
  if (ah != bh) { return ah > bh; }
  return a.x >= b.x;
}
`;

export interface PreprocBind {
  cOff: number;
  C: number;
  H: number;
  W: number;
  wpr: number;
  sampleWords: number;
  /** 行末尾ワードの有効ビットマスク(webgpuBitplane.ts::tailMaskU32 の値。
      レイアウト契約の単一情報源をあちらに保つため呼び出し側から渡す)。 */
  tailMask: number;
}

/** RGBA(u8×4/画素)→ ビットプレーン(ワードパック)を1カーネルで行う。
    出力レイアウト・端数ビットの扱い・先頭チャネルpermuteの融合は
    webgpuBitplane.ts::buildPackShader と同一契約(あちらの rawBits 読み出しが
    「記述子に従って画素から符号化する」に変わっただけ)。
    1スレッド=1出力ワード(=32空間位置)。 */
export function buildPreprocPackShader(
  d: PreprocDescriptors,
  out: PreprocBind,
  channelOrder: Int32Array | null,
): { code: string; staticArrays: Uint32Array[]; threadsPerSample: number } {
  const { C, H, W, wpr } = out;
  if (H !== d.height || W !== d.width) {
    throw new Error(
      `gpuPreproc: 幾何不一致(IR ${H}x${W} vs preproc ${d.height}x${d.width})`,
    );
  }
  if (C !== d.channels) {
    throw new Error(`gpuPreproc: チャネル数不一致(IR ${C} vs 記述子 ${d.channels})`);
  }
  const hasOrder = channelOrder !== null;
  const orderDecl = hasOrder
    ? '@group(0) @binding(4) var<storage, read> chanOrder: array<u32>;'
    : '';
  const outBinding = hasOrder ? 5 : 4;
  const srcChan = hasOrder ? 'chanOrder[o]' : 'o';
  const code = `struct Params { total: u32, xthreads: u32, _p1: u32, _p2: u32 };
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> rgba: array<u32>;
@group(0) @binding(2) var<storage, read> desc: array<u32>;
@group(0) @binding(3) var<storage, read> qval: array<u32>;
${orderDecl}
@group(0) @binding(${outBinding}) var<storage, read_write> outBuf: array<u32>;

const C: u32 = ${C}u;
const H: u32 = ${H}u;
const W: u32 = ${W}u;
const WPR: u32 = ${wpr}u;
const CIMG: u32 = ${d.imageChannels}u;
const IN_STRIDE: u32 = ${H * W}u;
const OUT_C_OFF: u32 = ${out.cOff}u;
const OUT_SAMPLE: u32 = ${out.sampleWords}u;
const TPS: u32 = ${C * H * wpr}u;
const TAIL: u32 = ${out.tailMask >>> 0}u;
const STRIDE: u32 = ${PREPROC_DESC_STRIDE}u;
${I64_WGSL}
/** 画素 (y,x) のチャネル c の u8 値。RGBA はリトルエンディアンの u32 1個。 */
fn u8At(b: u32, c: u32, y: u32, x: u32) -> u32 {
  return (rgba[b * IN_STRIDE + y * W + x] >> (c * 8u)) & 0xffu;
}
/** replicate padding つきの I(k) 取得(CPU側 shiftAt と同じ端の畳み込み)。 */
fn qAt(b: u32, c: u32, y: i32, x: i32) -> vec2<u32> {
  let yy = u32(clamp(y, 0, i32(H) - 1));
  let xx = u32(clamp(x, 0, i32(W) - 1));
  return vec2<u32>(qval[u8At(b, c, yy, xx)], 0u);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let gidx = gid.y * params.xthreads + gid.x;
  if (gidx >= params.total) { return; }
  let b = gidx / TPS;
  let rem = gidx % TPS;
  let o = rem / (H * WPR);
  let r2 = rem % (H * WPR);
  let y = r2 / WPR;
  let jw = r2 % WPR;

  // 記述子は1スレッド内で不変なのでループ外で読む。
  let base = (${srcChan}) * STRIDE;
  let op = desc[base];
  let ca = desc[base + 1u];
  let cb = desc[base + 2u];
  let dy = bitcast<i32>(desc[base + 3u]);
  let dx = bitcast<i32>(desc[base + 4u]);
  let cut = vec2<u32>(desc[base + 5u], desc[base + 6u]);
  let dirLe = desc[base + 7u] != 0u;
  let iy = i32(y);

  let x0 = jw * 32u;
  var v: u32 = 0u;
  for (var i: u32 = 0u; i < 32u; i = i + 1u) {
    let xx = x0 + i;
    if (xx < W) {
      var bit: u32 = 0u;
      if (op == ${PREPROC_OP_THERMO}u) {
        let k = u8At(b, ca, y, xx);
        bit = (desc[base + 8u + (k >> 5u)] >> (k & 31u)) & 1u;
      } else {
        let ix = i32(xx);
        var d: vec2<u32>;
        if (op == ${PREPROC_OP_EDGE}u) {
          d = i64Sub(qAt(b, ca, iy + dy, ix + dx), qAt(b, ca, iy, ix));
        } else if (op == ${PREPROC_OP_CURV}u) {
          let m = qAt(b, ca, iy, ix);
          d = i64Add(
            i64Sub(qAt(b, ca, iy - dy, ix - dx), i64Add(m, m)),
            qAt(b, ca, iy + dy, ix + dx),
          );
        } else if (op == ${PREPROC_OP_CHROMA}u) {
          d = i64Sub(qAt(b, ca, iy, ix), qAt(b, cb, iy, ix));
        } else {
          var mx: u32 = 0u;
          var mn: u32 = 255u;
          for (var c: u32 = 0u; c < CIMG; c = c + 1u) {
            let q = u8At(b, c, y, xx);
            mx = max(mx, q);
            mn = min(mn, q);
          }
          d = i64Sub(vec2<u32>(qval[mx], 0u), vec2<u32>(qval[mn], 0u));
        }
        if (dirLe) {
          if (i64Ge(cut, d)) { bit = 1u; }
        } else {
          if (i64Ge(d, cut)) { bit = 1u; }
        }
      }
      if (bit != 0u) { v = v | (1u << i); }
    }
  }
  if (jw == WPR - 1u) { v = v & TAIL; }
  outBuf[b * OUT_SAMPLE + (OUT_C_OFF + o) * H * WPR + y * WPR + jw] = v;
}
`;
  const statics: Uint32Array[] = [d.desc, d.qval];
  if (hasOrder) statics.push(Uint32Array.from(channelOrder));
  return { code, staticArrays: statics, threadsPerSample: C * H * wpr };
}

// ============================================================
// CPU シミュレータ(vitest 用。WGSL と同じ記述子を同じ手順で解釈する)
// ============================================================

/** 記述子1チャネルぶんを1画素に適用して 0/1 を返す(WGSL の本体と同じ手順)。
    GPU を持たない node 環境で「記述子表そのものの正しさ」を検証するために使う。 */
export function simulatePreprocBit(
  d: PreprocDescriptors,
  rgba: ArrayLike<number>,
  ch: number,
  y: number,
  x: number,
): number {
  const { desc, qval, height: H, width: W, imageChannels } = d;
  const u8At = (c: number, yy: number, xx: number): number =>
    rgba[(yy * W + xx) * 4 + c];
  const qAt = (c: number, yy: number, xx: number): number => {
    const cy = yy < 0 ? 0 : yy >= H ? H - 1 : yy;
    const cx = xx < 0 ? 0 : xx >= W ? W - 1 : xx;
    return qval[u8At(c, cy, cx)];
  };
  const base = ch * PREPROC_DESC_STRIDE;
  const op = desc[base];
  const ca = desc[base + 1];
  const cb = desc[base + 2];
  const dy = desc[base + 3] | 0;
  const dx = desc[base + 4] | 0;
  if (op === PREPROC_OP_THERMO) {
    const k = u8At(ca, y, x);
    return (desc[base + 8 + (k >>> 5)] >>> (k & 31)) & 1;
  }
  // JS の number は |D| <= 2^33 を厳密に表せるので、ここは素直な整数演算でよい
  // (WGSL 側だけが 64bit を2語で持つ必要がある)。
  let dv: number;
  if (op === PREPROC_OP_EDGE) {
    dv = qAt(ca, y + dy, x + dx) - qAt(ca, y, x);
  } else if (op === PREPROC_OP_CURV) {
    dv = qAt(ca, y - dy, x - dx) - 2 * qAt(ca, y, x) + qAt(ca, y + dy, x + dx);
  } else if (op === PREPROC_OP_CHROMA) {
    dv = qAt(ca, y, x) - qAt(cb, y, x);
  } else {
    let mx = 0;
    let mn = 255;
    for (let c = 0; c < imageChannels; c++) {
      const q = u8At(c, y, x);
      if (q > mx) mx = q;
      if (q < mn) mn = q;
    }
    dv = qval[mx] - qval[mn];
  }
  // 記述子の cut は 64bit 二の補数。JS 側では符号付き値へ戻して比較する。
  const lo = desc[base + 5];
  const hi = desc[base + 6];
  const cut = (hi >= 0x80000000 ? hi - 4294967296 : hi) * 4294967296 + lo;
  return desc[base + 7] !== 0 ? (dv <= cut ? 1 : 0) : (dv >= cut ? 1 : 0);
}

/** RGBA → エンコーダ出力の 0/1 ビット列([channels*H*W]、encodeImageAuto と同順)。 */
export function simulatePreprocBits(
  d: PreprocDescriptors,
  rgba: ArrayLike<number>,
): Uint8Array {
  const { channels, height: H, width: W } = d;
  const out = new Uint8Array(channels * H * W);
  for (let ch = 0; ch < channels; ch++) {
    const dst = ch * H * W;
    for (let y = 0; y < H; y++) {
      for (let x = 0; x < W; x++) {
        out[dst + y * W + x] = simulatePreprocBit(d, rgba, ch, y, x);
      }
    }
  }
  return out;
}
