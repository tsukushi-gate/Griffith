// 画像前処理(RGBA→ビット符号化)と入力パックの融合(CPU側)。
//
// 背景: 統一計測で、CPUバックエンド(js/wasm)のフレームを支配して
// いたのは conv2d でも pack でもなく **CPU前処理** だった。wasm推論が
// det 1.14 / detface 1.08ms まで縮んだのに、e2e は det 5.0 / detface 7.8ms。
// 実体は convpreproc.ts::encodeImageAuto:
//   RGBA → rgbaToNchw で float NCHW → 最大173ch×128×128 = 2.8M ビットを
//   1本ずつ f64 演算して 0/1 バイト列へ書き、それを packPlanes が読み直す。
// つまり「2.8MB を書いて 2.8MB を読む」中間表現がまるごと無駄になっていた。
//
// ここでは(gpuPreproc.ts)がGPU向けに確立した**厳密整数化 + 記述子表**を
// そのままCPUで再利用し、RGBA から**ビットプレーンのワードを直接**生成する。
// 記述子ビルダ(buildPreprocDescriptors)はGPUと共有するので、チャネル順・
// 閾値・比較の向きの単一情報源は gpuPreproc.ts のまま1つに保たれる。
//
// ============================================================
// 数値経路(なぜビット完全一致できるのか)
// ============================================================
// 詳細は gpuPreproc.ts 冒頭を参照(単一情報源)。要点だけ:
//   - 入力は u8 なので画素値は f32(k/255) の256通りしかなく、
//     I(k) := f32(k/255)*2^31 は厳密な整数(0..2^31、u32に収まる)。
//   - edge/curv/chroma/彩度は I の整数結合。比較も
//     d > t ⟺ D >= floor(t*2^31)+1、d < -t ⟺ D <= ceil(-t*2^31)-1 と厳密化できる。
//   - thermometer だけは (x-mean)*(1/std) > th に任意の f64 係数が入るので
//     整数化できないが、入力が256通りなので **256ビットLUT**(CPU側と同一の
//     JS式で生成)にすれば定義上ビット一致する。
//
// [の追加] thermometer は std>0 のもとで k について単調なので、LUT は
// ほぼ必ず「k >= k0 の1閾値」に畳める(THERMO_STEP)。畳めたかどうかは
// **LUTを総なめして実際に階段関数か検査**してから決める(畳めなければ
// LUT参照のまま)。これで thermometer チャネル(実モデルでは全チャネルの
// 過半数)が u8 の符号なし比較1個になり、WASM SIMD では16画素→16ビットを
// i8x16.ge_u + i8x16.bitmask の2命令で作れる。
//
// 正しさの錨: cpuPreproc.test.ts が
//   encodeImageAuto + packPlanes(既存のゴールデン固定済み経路)と
//   このモジュール(JS)/ WASM(scalar/simd)の**全ワード一致**を、
//   配信中の5モデル + ゴールデン由来パラメータ + 端値飽和入力で確認する。
import { tailMask, wordsPerRow } from './bitplane';
import type { BitplaneValue } from './bitplane';
import {
  PREPROC_DESC_STRIDE, PREPROC_OP_CHROMA, PREPROC_OP_CURV, PREPROC_OP_EDGE,
  PREPROC_OP_THERMO, buildPreprocDescriptors,
} from './gpuPreproc';
import type { ImagePreprocDoc } from '../convpreproc';

/** 単調な thermometer を「k >= k0」の1閾値へ畳んだ演算種別
    (0..4 は gpuPreproc.ts の PREPROC_OP_* と共通、5だけがCPU専用)。
    k0 は記述子の w2(chroma の cb と同じ枠。thermo では未使用だった)。 */
export const PREPROC_OP_THERMO_STEP = 5;

/** 融合前処理カーネルの実行計画(記述子 + 平面レイアウト)。 */
export interface CpuPreprocPlan {
  /** [channels * PREPROC_DESC_STRIDE]。gpuPreproc の記述子をそのまま i32 として
      持ち、thermo のうち単調なものだけ op/w2 を THERMO_STEP/k0 へ書き換えた版。 */
  desc: Int32Array;
  /** I(k) = f32(k/255) * 2^31 の256要素表(u32のビット列を i32 として保持)。 */
  qval: Int32Array;
  /** エンコーダ出力チャネル数(= 入力ビット幅 / (H*W))。 */
  channels: number;
  /** 画像チャネル数 C(彩度の max/min 走査幅)。 */
  imageChannels: number;
  height: number;
  width: number;
  /** wordsPerRow(width) / 行末尾ワードの有効ビットマスク。 */
  wpr: number;
  tail: number;
  /** 固定小数平面の行両端に置く replicate padding の要素数
      (|dx| + ベクタ読み越し分。これがあるので水平シフトは範囲検査不要)。 */
  rowPad: number;
  /** 固定小数平面の行ストライド(= width + 2*rowPad)。 */
  rowStride: number;
  /** 内訳(黙って遅い経路に落ちていないかの可視化用)。 */
  stepChannels: number;
  lutChannels: number;
  cmpChannels: number;
}

/** 記述子の cut(64bit二の補数の2語)を安全整数へ戻す
    (gpuPreproc.ts::simulatePreprocBit と同じ式)。 */
export function cutOf(desc: Int32Array, base: number): number {
  const lo = desc[base + 5] >>> 0;
  const hi = desc[base + 6];
  return hi * 4294967296 + lo;
}

/** thermometer の256ビットLUTが「k >= k0 の階段関数」かどうかを総なめで
    検査し、そうなら k0(全ビット0なら256)を返す。違えば null(LUT参照のまま)。 */
function thermoStepIndex(desc: Int32Array, base: number): number | null {
  const bitAt = (k: number): number => (desc[base + 8 + (k >>> 5)] >>> (k & 31)) & 1;
  let first = 256;
  for (let k = 0; k < 256; k++) {
    if (bitAt(k) === 1) {
      first = k;
      break;
    }
  }
  for (let k = 0; k < 256; k++) {
    if (bitAt(k) !== (k >= first ? 1 : 0)) return null;
  }
  return first;
}

/** doc.preproc → 融合前処理の実行計画。未対応 kind では
    buildPreprocDescriptors が throw する(呼び出し側は従来のCPU前処理へ落ちる)。 */
export function buildCpuPreprocPlan(p: ImagePreprocDoc): CpuPreprocPlan {
  const d = buildPreprocDescriptors(p);
  const desc = new Int32Array(d.desc.length);
  for (let i = 0; i < d.desc.length; i++) desc[i] = d.desc[i] | 0; // u32 → i32 ビット再解釈
  const qval = new Int32Array(256);
  for (let i = 0; i < 256; i++) qval[i] = d.qval[i] | 0;

  let stepChannels = 0;
  let lutChannels = 0;
  let cmpChannels = 0;
  let maxAbsDx = 0;
  for (let ch = 0; ch < d.channels; ch++) {
    const base = ch * PREPROC_DESC_STRIDE;
    if (desc[base] === PREPROC_OP_THERMO) {
      const k0 = thermoStepIndex(desc, base);
      if (k0 === null) {
        lutChannels += 1;
      } else {
        desc[base] = PREPROC_OP_THERMO_STEP;
        desc[base + 2] = k0;
        stepChannels += 1;
      }
      continue;
    }
    cmpChannels += 1;
    const dx = desc[base + 4];
    if (Math.abs(dx) > maxAbsDx) maxAbsDx = Math.abs(dx);
  }
  // ベクタ経路は4画素まとめて読むので、|dx| に加えて3要素ぶんの読み越しを見込む。
  const rowPad = Math.max(8, (maxAbsDx + 4 + 3) & ~3);
  return {
    desc,
    qval,
    channels: d.channels,
    imageChannels: d.imageChannels,
    height: d.height,
    width: d.width,
    wpr: wordsPerRow(d.width),
    tail: tailMask(d.width),
    rowPad,
    rowStride: d.width + 2 * rowPad,
    stepChannels,
    lutChannels,
    cmpChannels,
  };
}

/** 計画がこのIR入力ポート(幅 inputWidth)に本当に対応しているか。
    合わない計画で走らせると「動くが別物を食わせている」壊れ方をするので、
    構築時にここで弾いてCPU前処理(従来経路)へ落とす。 */
export function planMatchesInputWidth(plan: CpuPreprocPlan, inputWidth: number): boolean {
  return plan.channels * plan.height * plan.width === inputWidth;
}

/** JS版の融合前処理パッカー(wasm を持たない 'bitplane' エンジン用、
    かつ WASM 実装の参照実装)。scratch(固定小数平面/バイト平面)をインスタンスに
    抱えるので、フレームごとのアロケーションは出力ワードだけになる。

    固定小数平面は Float64Array に I(k) をそのまま入れる(JSの number は
    |D| <= 2^33 を厳密に表せるので、edge/curv/chroma/彩度の整数結合と閾値比較は
    すべて丸めなしで行える)。行の両端には replicate padding を置くので、
    水平シフト x+dx は範囲検査なしで読める(CPU側 shiftAt の x クランプと同値)。 */
export class JsPreprocPacker {
  private readonly q: Float64Array;
  private readonly bytes: Uint8Array;
  /** バイト平面の u32 ビュー(thermometer の4バイト同時比較用。
      行頭が4バイト境界に載る W%4===0 のときだけ使う)。 */
  private readonly bytes32: Uint32Array | null;

  constructor(readonly plan: CpuPreprocPlan) {
    const { imageChannels, height, rowStride, width } = plan;
    this.q = new Float64Array(imageChannels * height * rowStride);
    const n = imageChannels * height * width;
    this.bytes = new Uint8Array((n + 3) & ~3);
    this.bytes32 = (width & 3) === 0
      ? new Uint32Array(this.bytes.buffer, 0, this.bytes.length >>> 2)
      : null;
  }

  /** RGBA(batch枚ぶんの連結)→ ビットプレーン([batch][C][H][wpr])。 */
  pack(rgba: ArrayLike<number>, batch: number): BitplaneValue {
    const { channels, height: H, width: W, wpr } = this.plan;
    const sampleWords = channels * H * wpr;
    const words = new Int32Array(batch * sampleWords);
    for (let b = 0; b < batch; b++) {
      this.quantize(rgba, b * H * W * 4);
      this.packSample(words, b * sampleWords);
    }
    return { words, C: channels, H, W, wpr };
  }

  /** RGBA → バイト平面 + 固定小数平面(行両端 replicate padding つき)。 */
  private quantize(rgba: ArrayLike<number>, off: number): void {
    const { imageChannels, height: H, width: W, rowPad, rowStride, qval } = this.plan;
    const q = this.q;
    const bytes = this.bytes;
    for (let c = 0; c < imageChannels; c++) {
      for (let y = 0; y < H; y++) {
        const src = off + y * W * 4 + c;
        const bRow = (c * H + y) * W;
        const qRow = (c * H + y) * rowStride + rowPad;
        for (let x = 0; x < W; x++) {
          const k = rgba[src + x * 4];
          bytes[bRow + x] = k;
          q[qRow + x] = qval[k] >>> 0;
        }
        const q0 = q[qRow];
        const q1 = q[qRow + W - 1];
        for (let x = 1; x <= rowPad; x++) {
          q[qRow - x] = q0;
          q[qRow + W - 1 + x] = q1;
        }
      }
    }
  }

  /** 固定小数平面の (c, clamp(y)) 行の x=0 の要素番号。 */
  private qRow(c: number, y: number): number {
    const { height: H, rowPad, rowStride } = this.plan;
    const cy = y < 0 ? 0 : y >= H ? H - 1 : y;
    return (c * H + cy) * rowStride + rowPad;
  }

  private packSample(words: Int32Array, outBase: number): void {
    const { channels, desc, height: H, wpr } = this.plan;
    for (let ch = 0; ch < channels; ch++) {
      const base = ch * PREPROC_DESC_STRIDE;
      const op = desc[base];
      const ca = desc[base + 1];
      const cb = desc[base + 2];
      const dy = desc[base + 3];
      const dx = desc[base + 4];
      const cut = cutOf(desc, base);
      const le = desc[base + 7] !== 0;
      const outCh = outBase + ch * H * wpr;
      if (op === PREPROC_OP_THERMO_STEP || op === PREPROC_OP_THERMO) {
        this.packThermo(words, outCh, op, ca, cb, base);
        continue;
      }
      for (let y = 0; y < H; y++) {
        const dst = outCh + y * wpr;
        if (op === PREPROC_OP_EDGE) {
          this.packDiff2(words, dst, this.qRow(ca, y + dy) + dx, this.qRow(ca, y), cut, le);
        } else if (op === PREPROC_OP_CURV) {
          this.packCurv(
            words, dst,
            this.qRow(ca, y + dy) + dx, this.qRow(ca, y), this.qRow(ca, y - dy) - dx,
            cut, le,
          );
        } else if (op === PREPROC_OP_CHROMA) {
          this.packDiff2(words, dst, this.qRow(ca, y), this.qRow(cb, y), cut, le);
        } else {
          this.packSat(words, dst, y, cut, le);
        }
      }
    }
  }

  private packThermo(
    words: Int32Array, outCh: number, op: number, ca: number, k0: number, base: number,
  ): void {
    const { desc, height: H, width: W, wpr, tail } = this.plan;
    const bytes = this.bytes;
    for (let y = 0; y < H; y++) {
      const row = (ca * H + y) * W;
      const dst = outCh + y * wpr;
      for (let j = 0; j < wpr; j++) {
        const x0 = j << 5;
        const n = Math.min(32, W - x0);
        let acc = 0;
        if (op === PREPROC_OP_THERMO_STEP) {
          const u32 = k0 <= 255 ? this.bytes32 : null; // k0=256(常に0)は素直な経路へ
          if (u32 !== null) {
            // 4バイト同時の符号なし比較(SWAR)。1個の u32 ロードから
            // 「各バイトが k0 以上か」の4ビットを取り出して acc へ載せる。
            //   t = (x|H) - (k0 & ~H) の bit7 は「下位7bitの比較 xl >= k0l」、
            //   最上位ビットが違えばそちらが勝つので
            //   res = (x & ~k0) | (~(x ^ k0) & t) の bit7 が x >= k0(全256値で総当り検証済み)。
            // 取り出しは bitplane.ts::packRowU8 と同じ imul の桁寄せ。
            const y = Math.imul(k0, 0x01010101);
            const yl = y & 0x7f7f7f7f;
            const q = (row + x0) >> 2;
            for (let i = 0; i < n; i += 4) {
              const x = u32[q + (i >> 2)];
              const t = ((x | 0x80808080) - yl) | 0;
              const ge = ((x & ~y) | (~(x ^ y) & t)) & 0x80808080;
              const nz = (ge >>> 7) & 0x01010101;
              acc |= (Math.imul(nz, 0x01020408) >>> 24) << i;
            }
          } else {
            for (let i = 0; i < n; i++) if (bytes[row + x0 + i] >= k0) acc |= 1 << i;
          }
        } else {
          for (let i = 0; i < n; i++) {
            const k = bytes[row + x0 + i];
            acc |= ((desc[base + 8 + (k >>> 5)] >>> (k & 31)) & 1) << i;
          }
        }
        words[dst + j] = j === wpr - 1 ? acc & tail : acc;
      }
    }
  }

  private packDiff2(
    words: Int32Array, dst: number, pa: number, pb: number, cut: number, le: boolean,
  ): void {
    const { width: W, wpr, tail } = this.plan;
    const q = this.q;
    for (let j = 0; j < wpr; j++) {
      const x0 = j << 5;
      const n = Math.min(32, W - x0);
      let acc = 0;
      if (le) {
        for (let i = 0; i < n; i++) {
          if (q[pa + x0 + i] - q[pb + x0 + i] <= cut) acc |= 1 << i;
        }
      } else {
        for (let i = 0; i < n; i++) {
          if (q[pa + x0 + i] - q[pb + x0 + i] >= cut) acc |= 1 << i;
        }
      }
      words[dst + j] = j === wpr - 1 ? acc & tail : acc;
    }
  }

  private packCurv(
    words: Int32Array, dst: number, pp: number, pm: number, pn: number,
    cut: number, le: boolean,
  ): void {
    const { width: W, wpr, tail } = this.plan;
    const q = this.q;
    for (let j = 0; j < wpr; j++) {
      const x0 = j << 5;
      const n = Math.min(32, W - x0);
      let acc = 0;
      for (let i = 0; i < n; i++) {
        const x = x0 + i;
        const d = q[pn + x] - 2 * q[pm + x] + q[pp + x];
        if (le ? d <= cut : d >= cut) acc |= 1 << i;
      }
      words[dst + j] = j === wpr - 1 ? acc & tail : acc;
    }
  }

  private packSat(
    words: Int32Array, dst: number, y: number, cut: number, le: boolean,
  ): void {
    const { imageChannels, height: H, width: W, wpr, tail, rowStride } = this.plan;
    const q = this.q;
    const row0 = this.qRow(0, y);
    const planeStride = H * rowStride;
    for (let j = 0; j < wpr; j++) {
      const x0 = j << 5;
      const n = Math.min(32, W - x0);
      let acc = 0;
      for (let i = 0; i < n; i++) {
        const o = row0 + x0 + i;
        let mx = q[o];
        let mn = mx;
        for (let c = 1; c < imageChannels; c++) {
          const v = q[o + c * planeStride];
          if (v > mx) mx = v;
          if (v < mn) mn = v;
        }
        const d = mx - mn;
        if (le ? d <= cut : d >= cut) acc |= 1 << i;
      }
      words[dst + j] = j === wpr - 1 ? acc & tail : acc;
    }
  }
}
