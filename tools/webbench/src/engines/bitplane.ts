// ビットスライス・パイプラインの核心プリミティブ。
//
// packedConv.ts / assembly/index.ts は「木の縮約だけワード並列(32空間位置/
// ワード、フラットな Lout=HOut*WOut ビット列)」で、タップ集め
// (buildTapPlane)は1ビットずつのスカラー散布だった(主コスト)。
// ここでは入力を「行頭ワード境界」のビットプレーンへパックし、タップ
// (dy,dx)平面の構築を「行ワード列のシフト合成」に置き換える(1行=W位置が
// wpr個のワード演算になり、W/32倍のワード数削減)。
//
// レイアウト契約(確定): words は [batch][C][H][wpr] の row-major
// Int32Array。ビット順は列 x → ワード j=x>>5 のビット i=x&31 の
// LSB-first(既存 buildTapPlane / Python x_bits_b64 と同一)。行頭は必ず
// ワード境界(水平シフトが行内で完結するための必須条件)。端数ビット
// (x>=W)は「常に0」を不変条件として全プリミティブが維持する
// (下流のシフト/OR-pool/upsampleがこの不変条件に依存して1ビットずつの
// 範囲チェックを省略できる)。
//
// 正しさの錨: bitplane.test.ts が各プリミティブをスカラー参照実装
// (もしくは packedConv.ts の既存実装)と全域照合する。
import { ttParse } from '../shared';
import type { Conv2dParams } from '../shared';
import {
  REF_KIND_CONST, REF_KIND_TAP, buildConvProgram, refIdx, refKind, verifyConvProgram,
} from './convProgram';
import type { ConvProgram, ConvRef, ConvTreeSpec } from './convProgram';
import { compileConvProgram } from './convProgramCodegen';
import type { CompiledConvProgram, ConvCodegenOptions } from './convProgramCodegen';
import type { NodeValue } from './irflat';

export interface BitplaneValue {
  words: Int32Array; // [batch][C][H][wpr] row-major
  C: number;
  H: number;
  W: number;
  wpr: number; // wordsPerRow(W) = ceil(W/32)
}

/** 1行に必要なワード数(=ceil(W/32))。 */
export function wordsPerRow(W: number): number {
  return (W + 31) >>> 5;
}

/** 行末尾ワードの有効ビットマスク(W%32===0なら全ビット有効=-1)。
    NOT系ゲート(tt奇数系)が端数ビットに1を立てるため、行を書き終える
    箇所で必ずこれをANDする契約(呼び出し側の責務)。 */
export function tailMask(W: number): number {
  const r = W & 31;
  return r === 0 ? -1 : (1 << r) - 1;
}

// ============================================================
// 1. pack/unpack: NodeValue(非パック 0/1 flat) <-> BitplaneValue
// ============================================================

/** packPlanes の入力。NodeValue(data: Int32Array)はそのまま渡せるが、
    グラフ入力ポートの生ビット列(Uint8Array等)を Int32Array へ
    詰め替えずに直接パックできるよう ArrayLike で受ける
    (入力は数百万要素あり、詰め替えだけで det/detface で1.9〜3.3ms/frameを
    消費していた)。 */
export interface PackSource {
  data: ArrayLike<number>;
  width: number;
}

/** Uint8Array 入力の4バイト同時パック。1個の u32 ロードから
    「各バイトが非ゼロか」の4ビットを取り出して acc の i..i+3 へ載せる。
    - `nz`: 各バイトについて非ゼロなら bit0 を立てる
      (`w | ((w&0x7f7f7f7f)+0x7f7f7f7f)` の bit7 が「非ゼロ」を表す。
       0x80以上の値も w 自身の bit7 で拾えるので全256値で正しい)。
    - `Math.imul(nz, 0x01020408) >>> 24`: バイト0..3のbit0を
      結果の bit0..3 へ集める(桁上がりは起きない: 最大 0x0F0E0C08)。
    要素ごとの分岐と `ArrayLike<number>` 経由の多相アクセスが消え、実測で
    現行比 2.3〜2.5倍(0.5ns/要素)。値の意味論は素朴ループと完全同一
    (bitplane.test.ts が 0/1 以外のバイト値も含めて両経路を照合する)。 */
function packRowU8(d: Uint8Array, u32: Uint32Array, base: number, n: number): number {
  let acc = 0;
  let i = 0;
  const q = base >> 2;
  for (; i + 4 <= n; i += 4) {
    const w = u32[q + (i >> 2)];
    const nz = ((w | ((w & 0x7f7f7f7f) + 0x7f7f7f7f)) >>> 7) & 0x01010101;
    acc |= (Math.imul(nz, 0x01020408) >>> 24) << i;
  }
  for (; i < n; i++) {
    const v = d[base + i];
    acc |= ((v | (-v)) >>> 31) << i;
  }
  return acc;
}

/** 0/1 flat([batch*(C*H*W)], NCHW)をビットプレーンへパックする。
    行頭ワード境界を守るため、W%32!=0のときは各行末尾に無駄ビットが残る
    (常に0、上でいう不変条件)。 */
export function packPlanes(
  x: PackSource, batch: number, C: number, H: number, W: number,
): BitplaneValue {
  const wpr = wordsPerRow(W);
  const inW = x.width;
  if (inW !== C * H * W) {
    throw new Error(`packPlanes: 幅不一致 (期待 ${C * H * W}, 実際 ${inW})`);
  }
  const words = new Int32Array(batch * C * H * wpr);
  const data = x.data;
  // Uint8Array かつ全ての行頭が4バイト境界に載る場合だけ高速路。
  // (W%4===0 なら inW=C*H*W も4の倍数なので、b/c/y のどの組み合わせでも
  //  rowIn は4の倍数。j<<5 も4の倍数なので base も必ず4の倍数。)
  const fastU8 = data instanceof Uint8Array
    && (W & 3) === 0 && (data.byteOffset & 3) === 0 && (data.length & 3) === 0;
  const u32 = fastU8
    ? new Uint32Array((data as Uint8Array).buffer, (data as Uint8Array).byteOffset,
      (data as Uint8Array).length >>> 2)
    : null;
  // 1要素ずつ words[...] |= ... と読み書きするのではなく、
  // ワードごとにローカルへ32ビット積んでから1回だけ書く(入力ビット列は
  // 数百万要素になるため、出力側の read-modify-write を消すだけで効く)。
  for (let b = 0; b < batch; b++) {
    const xo = b * inW;
    const oo = b * C * H * wpr;
    for (let c = 0; c < C; c++) {
      for (let y = 0; y < H; y++) {
        const rowIn = xo + (c * H + y) * W;
        const rowOut = oo + (c * H + y) * wpr;
        for (let j = 0; j < wpr; j++) {
          const base = rowIn + (j << 5);
          const n = Math.min(32, W - (j << 5));
          if (u32 !== null) {
            words[rowOut + j] = packRowU8(data as Uint8Array, u32, base, n);
            continue;
          }
          let acc = 0;
          for (let i = 0; i < n; i++) if (data[base + i]) acc |= 1 << i;
          words[rowOut + j] = acc;
        }
      }
    }
  }
  return { words, C, H, W, wpr };
}

/** ビットプレーン → NodeValue(0/1 flat, NCHW)。端数ビット(x>=W)は
    読まないため、その値がどうであっても結果に影響しない。 */
export function unpackPlanes(bp: BitplaneValue, batch: number): NodeValue {
  const { C, H, W, wpr } = bp;
  const width = C * H * W;
  const data = new Int32Array(batch * width);
  const planeWords = C * H * wpr;
  for (let b = 0; b < batch; b++) {
    const oo = b * width;
    const bo = b * planeWords;
    for (let c = 0; c < C; c++) {
      for (let y = 0; y < H; y++) {
        const rowOut = oo + (c * H + y) * W;
        const rowIn = bo + (c * H + y) * wpr;
        for (let xx = 0; xx < W; xx++) {
          data[rowOut + xx] = (bp.words[rowIn + (xx >>> 5)] >>> (xx & 31)) & 1;
        }
      }
    }
  }
  return { data, width };
}

// ============================================================
// 2. shiftRowInto: 行内ワード列シフト(タップ集め・or_pool・upsampleの基礎)
// ============================================================

/** dst[0..wpr) の行に「src の行を s ビットシフトした結果」を書く
    (dst bit p = src bit (p+s)、範囲外は0)。s は wpr*32 を大きく超えない
    想定だが任意の整数を許容する(word境界を跨ぐ分は wordShift として処理)。
    s>0: (src[j]>>>s)|(src[j+1]<<(32-s))。s<0: 対称の逆シフト。s==0: コピー。
    dstOff/srcOff は「行の先頭ワード」への基底オフセット(大きな共有バッファ
    の中の1行を、subarrayを作らず直接指すためのゼロコピー経路)。 */
/** 実装メモ: 以前はワード境界チェックをクロージャ `at(j)` に
    切り出していたが、この関数は conv の共有シフト平面構築(全タップの基礎)
    から呼ばれる最ホットパスであり、ワードごとのクロージャ呼び出しが単価を
    押し上げていた(C1計測)。意味論を変えずにインライン展開する。 */
export function shiftRowInto(
  dst: Int32Array, src: Int32Array, wpr: number, s: number,
  dstOff = 0, srcOff = 0,
): void {
  // s を [0,32) の bitShift と word 単位の wordShift に分解する
  // (floor除算: 負のsでも wordShift*32+bitShift===s を保つ)。
  const wordShift = Math.floor(s / 32);
  const bitShift = s - wordShift * 32; // [0,32)
  if (bitShift === 0) {
    for (let j = 0; j < wpr; j++) {
      const k = j + wordShift;
      dst[dstOff + j] = k >= 0 && k < wpr ? src[srcOff + k] : 0;
    }
    return;
  }
  const inv = 32 - bitShift;
  for (let j = 0; j < wpr; j++) {
    const k = j + wordShift;
    const lo = k >= 0 && k < wpr ? src[srcOff + k] : 0;
    const k1 = k + 1;
    const hi = k1 >= 0 && k1 < wpr ? src[srcOff + k1] : 0;
    dst[dstOff + j] = (lo >>> bitShift) | (hi << inv);
  }
}

// ============================================================
// 3/4. タップ平面構築: stride=1高速路(行シフト) / 汎用の遅い正確パス
// ============================================================

/** stride=1・同幅(WOut===W)専用の高速タップ構築。1つの (channel, dy, dx)
    に対する HOut 行ぶんのタップ平面を、行ごとの shiftRowInto で作る
    (窓外の行は0埋め)。呼び出し側は WOut===W を保証すること(そうでない
    場合は wpr が入出力で食い違いうるため buildTapPlaneStrided を使う)。
    xWords: 入力平面全体、srcBase: このチャネル・このバッチ要素の行0の
    ワード基底オフセット(= b*C*H*wpr + c*H*wpr)。dst/dstBase も同様の
    ゼロコピー基底オフセット規約。 */
export function buildTapPlaneBitplane(
  xWords: Int32Array, srcBase: number,
  H: number, wpr: number, HOut: number,
  dy: number, dx: number, pH: number, pW: number,
  dst: Int32Array, dstBase: number,
): void {
  const s = dx - pW;
  for (let oy = 0; oy < HOut; oy++) {
    const y = oy - pH + dy; // sH=1 前提(呼び出し側で保証)
    const outRowOff = dstBase + oy * wpr;
    if (y < 0 || y >= H) {
      for (let j = 0; j < wpr; j++) dst[outRowOff + j] = 0;
    } else {
      shiftRowInto(dst, xWords, wpr, s, outRowOff, srcBase + y * wpr);
    }
  }
}

/** stride=1・同幅(WOut===W)専用の「共有シフト平面」構築。
    水平シフト量 s = dx - pW は dy に依存しない。したがって (channel, dx) ごとに
    「上下 pH 行のゼロパディングを含む H+2*pH 行ぶんの水平シフト済み平面」を
    1回だけ実体化すれば、dy 違いの全タップは同じ平面を行オフセット dy から
    HOut 行読むだけで得られる(XNNPACK の indirection buffer /
    xnn_indirection_init_dwconv2d_compressed と同型の設計)。
    共有平面の行 r の内容は「入力行 y=r-pH を s ビットシフトしたもの」
    (y が範囲外ならゼロ行)。タップ (dy,dx) の出力行 oy が要求する入力行は
    y = oy - pH + dy なので、共有平面の行 r = oy + dy と厳密に一致する
    (buildTapPlaneBitplane とビット単位で同一)。行数の十分性:
    dy <= kH-1、HOut = H+2*pH-kH+1 より dy + HOut <= H+2*pH で常に範囲内。
    端数ビット(x>=W)は入力の不変条件(常に0)とシフト合成の境界処理
    (行外ワードは0)により buildTapPlaneBitplane と同じ値になる。

    実装: 「行ごとに shiftRowInto を呼ぶ」形をやめ、内部H行を
    **1本の連続ワード列**として処理する。共有平面の本体([pH, pH+H) 行)は
    入力チャネル平面([srcBase, srcBase+H*wpr))と同じ連続領域・同じ行ストライド
    なので、行境界を無視した長さ H*wpr の直線ループでシフト合成でき、
    「行をまたいで混入した語」だけを行末(または行頭)1語ずつ潰せばよい。
    実モデルは wpr=1〜4(=1行あたり1〜4語)しかないため、行ごとのループ/
    呼び出し立ち上げが語あたりコストを支配していた(C5計測: タップ平面構築が
    conv時間の45〜80%を占め、その2.3〜2.5倍が本改修で消える)。
    どちらの分岐も shiftRowInto と語単位で同一の値を書く(整数出力不変)。
    注意: dst と xWords は別バッファであること(エイリアス不可)。 */
export function buildSharedShiftPlane(
  xWords: Int32Array, srcBase: number,
  H: number, wpr: number, pH: number, s: number,
  dst: Int32Array, dstBase: number,
): void {
  const n = H * wpr;
  const body = dstBase + pH * wpr;
  // 上下 pH 行のゼロパディング。
  dst.fill(0, dstBase, body);
  dst.fill(0, body + n, body + n + pH * wpr);
  if (s === 0) {
    dst.set(xWords.subarray(srcBase, srcBase + n), body);
    return;
  }
  if (s <= -32 || s >= 32) {
    // |s|>=32(kW>33 等の想定外ジオメトリ)は wordShift が -1/0 に収まらない。
    // 正しさ優先で従来どおり行ごとの汎用路へ落とす。
    for (let y = 0; y < H; y++) {
      shiftRowInto(dst, xWords, wpr, s, body + y * wpr, srcBase + y * wpr);
    }
    return;
  }
  // |s|<32 なので wordShift は 0(s>0)か -1(s<0)のみ、bitShift は (0,32)。
  const wordShift = s >> 5;
  const bitShift = s - wordShift * 32;
  const inv = 32 - bitShift;
  if (wpr === 1) {
    // 1行=1語: 全ての語が行境界なので隣接語は常に0 = 混入自体が起きない。
    if (wordShift === 0) {
      for (let i = 0; i < n; i++) dst[body + i] = xWords[srcBase + i] >>> bitShift;
    } else {
      for (let i = 0; i < n; i++) dst[body + i] = xWords[srcBase + i] << inv;
    }
    return;
  }
  if (wordShift === 0) {
    // dst[i] = (src[i]>>>bitShift) | (src[i+1]<<inv)。i が行末語のとき src[i+1] は
    // 次行の先頭語=行外なので0でなければならない → 行末語だけ後で上書きする。
    const last = n - 1;
    for (let i = 0; i < last; i++) {
      dst[body + i] = (xWords[srcBase + i] >>> bitShift) | (xWords[srcBase + i + 1] << inv);
    }
    dst[body + last] = xWords[srcBase + last] >>> bitShift;
    for (let y = 0; y < H - 1; y++) {
      const i = y * wpr + (wpr - 1);
      dst[body + i] = xWords[srcBase + i] >>> bitShift;
    }
  } else {
    // dst[i] = (src[i-1]>>>bitShift) | (src[i]<<inv)。i が行頭語のとき src[i-1] は
    // 前行の末尾語=行外なので0 → 行頭語だけ後で上書きする。
    dst[body] = xWords[srcBase] << inv;
    for (let i = 1; i < n; i++) {
      dst[body + i] = (xWords[srcBase + i - 1] >>> bitShift) | (xWords[srcBase + i] << inv);
    }
    for (let y = 1; y < H; y++) {
      const i = y * wpr;
      dst[body + i] = xWords[srcBase + i] << inv;
    }
  }
}

/** 汎用(stride任意・pad任意・WOut!==W可)の正確なタップ構築。1ビットずつの
    gather(packedConv.ts::buildTapPlane と同じ幾何)だが、出力を
    行ワード境界レイアウト(wprOut)へ書く点だけが異なる。stride>1、または
    同幅前提が崩れるケースの正しさの錨として常に残す(遅いが常に正しい)。 */
export function buildTapPlaneStrided(
  xWords: Int32Array, srcBase: number,
  H: number, W: number, wprIn: number,
  dy: number, dx: number,
  HOut: number, WOut: number, wprOut: number,
  sH: number, sW: number, pH: number, pW: number,
  dst: Int32Array, dstBase: number,
): void {
  for (let i = 0; i < HOut * wprOut; i++) dst[dstBase + i] = 0;
  for (let oy = 0; oy < HOut; oy++) {
    const y = oy * sH - pH + dy;
    if (y < 0 || y >= H) continue;
    const rowBase = srcBase + y * wprIn;
    const outRowBase = dstBase + oy * wprOut;
    for (let ox = 0; ox < WOut; ox++) {
      const xx = ox * sW - pW + dx;
      if (xx < 0 || xx >= W) continue;
      const bit = (xWords[rowBase + (xx >>> 5)] >>> (xx & 31)) & 1;
      if (bit) dst[outRowBase + (ox >>> 5)] |= 1 << (ox & 31);
    }
  }
}

// ============================================================
// 5. orPool2x2: 縦OR→横OR→偶数ビット圧縮(k=2,s=2,pad=0専用のネイティブ路)
// ============================================================

/** 8bit→4bit(偶数ビット b0,b2,b4,b6 を連続ニブルへ)LUT。 */
const EVEN_NIBBLE_LUT = (() => {
  const t = new Uint8Array(256);
  for (let b = 0; b < 256; b++) {
    const b0 = b & 1;
    const b2 = (b >>> 2) & 1;
    const b4 = (b >>> 4) & 1;
    const b6 = (b >>> 6) & 1;
    t[b] = b0 | (b2 << 1) | (b4 << 2) | (b6 << 3);
  }
  return t;
})();

/** 32bit中の偶数位置ビット(0,2,4,...,30)を連続する下位16bitへ圧縮する
    (8bit×4のLUT版)。奇数位置ビットの値は結果に影響しない。 */
export function compressEven32LUT(v: number): number {
  const b0 = v & 0xff;
  const b1 = (v >>> 8) & 0xff;
  const b2 = (v >>> 16) & 0xff;
  const b3 = (v >>> 24) & 0xff;
  return EVEN_NIBBLE_LUT[b0] | (EVEN_NIBBLE_LUT[b1] << 4)
    | (EVEN_NIBBLE_LUT[b2] << 8) | (EVEN_NIBBLE_LUT[b3] << 12);
}

/** 同じ偶数ビット圧縮を、テーブル無しの O(log n) シフト&マスク段階
    (Hacker's Delight型の "compress" カスケード)で計算する第2実装。
    仕様は「8bit LUT×4 と magic-multiply の2実装」を求めているが、この
    パターン(0x55555555固定マスク・4要素)に対して桁上がり衝突なしに
    成立する単一の乗数を手計算/総当たりで安全に導出できなかったため
    (総当たり探索(M<2^24, S<32)でも解なし = 単純な乗算1発では原理的に
    成立しない可能性が高いパターン)、「テーブル版と独立に正しさを検証
    できるもう1つの手法」としてこのシフトカスケード版を採用する。
    正しさ最優先(このファイルの設計方針)の判断として明示する。 */
export function compressEven32Shift(v: number): number {
  let x = v & 0x55555555;
  x = (x | (x >>> 1)) & 0x33333333;
  x = (x | (x >>> 2)) & 0x0f0f0f0f;
  x = (x | (x >>> 4)) & 0x00ff00ff;
  x = (x | (x >>> 8)) & 0x0000ffff;
  return x;
}

export type CompressFn = (v: number) => number;

/** 1行ぶんの2x2 OR-pool(縦OR→水平隣接OR→偶数ビット圧縮)。srcRowA/B は
    入力の2行(隣接、常に範囲内: pad=0保証は呼び出し側)。出力は wprOut
    ワード(tailMask適用済み)。
    scratch(長さ>=wpr)は呼び出し側が使い回す作業領域。以前は
    この関数の中で行ごとに Int32Array を3本 new していたが、行数 =
    C*HOut(実モデルで1万行超)ぶんの短命アロケーションになり GC 負荷が
    支配的だった。水平シフトは s=1 固定なので shiftRowInto を呼ばずに
    その場で合成し、中間配列も1本(t)だけにする。 */
function orPoolRow(
  words: Int32Array, rowABase: number, rowBBase: number, wpr: number,
  dst: Int32Array, dstBase: number, wprOut: number, maskOut: number,
  compress: CompressFn, scratch: Int32Array,
): void {
  // t = rowA | rowB (縦OR)。u(j) = t[j] | (t[j]>>>1 | t[j+1]<<31) で
  // 水平隣接ペア(2i,2i+1)のORを偶数位置へ集約する
  // (shiftRowInto(...,s=1) と同一の合成式を s=1 に特殊化したもの)。
  for (let j = 0; j < wpr; j++) scratch[j] = words[rowABase + j] | words[rowBBase + j];
  for (let wo = 0; wo < wprOut; wo++) {
    const jl = 2 * wo;
    const tl = jl < wpr ? scratch[jl] : 0;
    const tl1 = jl + 1 < wpr ? scratch[jl + 1] : 0;
    const lo = tl | (tl >>> 1) | (tl1 << 31);
    const jh = jl + 1;
    const th = jh < wpr ? scratch[jh] : 0;
    const th1 = jh + 1 < wpr ? scratch[jh + 1] : 0;
    const hi = jh < wpr ? (th | (th >>> 1) | (th1 << 31)) : 0;
    dst[dstBase + wo] = compress(lo) | (compress(hi) << 16);
  }
  dst[dstBase + wprOut - 1] &= maskOut;
}

/** or_pool2d(kernel=2,stride=2,padding=0)をビットプレーンのままネイティブ
    実行する。それ以外の窓/ストライド/パディングは呼び出し側
    (bitplaneGraph.ts)がアンパックへフォールバックさせること。 */
/** マイクロベンチ実測(node vite-node、20M回、V8 JITウォームアップ後3回平均):
    LUT版 ~164ms vs Shift版 ~159ms(ほぼ互角、Shift版がわずかに速い)。
    Shift版はテーブル参照(L1キャッシュ競合の可能性、特に大きな木の
    ワードバッファと同時アクセスされる実ワークロードでLUTより不利になり
    うる)が無く、将来のWASM SIMD移植(v128にはgather命令が無くLUT化が
    そのまま素直に移植できない)にも直結するため既定とする。 */
export function orPool2x2(
  bp: BitplaneValue, batch: number, compress: CompressFn = compressEven32Shift,
): BitplaneValue {
  const { C, H, W, wpr } = bp;
  if (H < 2 || W < 2) throw new Error(`orPool2x2: 窓2x2に対しH/Wが小さすぎる (H=${H}, W=${W})`);
  const HOut = ((H - 2) >> 1) + 1;
  const WOut = ((W - 2) >> 1) + 1;
  const wprOut = wordsPerRow(WOut);
  const maskOut = tailMask(WOut);
  const words = new Int32Array(batch * C * HOut * wprOut);
  const inPlaneWords = C * H * wpr;
  const outPlaneWords = C * HOut * wprOut;
  const scratch = new Int32Array(wpr); // 全行で使い回す作業領域
  for (let b = 0; b < batch; b++) {
    const ib = b * inPlaneWords;
    const ob = b * outPlaneWords;
    for (let c = 0; c < C; c++) {
      const icb = ib + c * H * wpr;
      const ocb = ob + c * HOut * wprOut;
      for (let oy = 0; oy < HOut; oy++) {
        const rowA = icb + (2 * oy) * wpr;
        const rowB = icb + (2 * oy + 1) * wpr;
        orPoolRow(
          bp.words, rowA, rowB, wpr, words, ocb + oy * wprOut, wprOut, maskOut,
          compress, scratch,
        );
      }
    }
  }
  return { words, C, H: HOut, W: WOut, wpr: wprOut };
}

// ============================================================
// 6. dilateX2: nearest upsample x2(圧縮の逆、ビット拡散)
// ============================================================

/** 16bit(下位16bitのみ使用)を偶数位置(0,2,...,30)へ拡散する
    (compressEven32LUTの逆演算)。 */
export function spreadEven16(v: number): number {
  let x = v & 0xffff;
  x = (x | (x << 8)) & 0x00ff00ff;
  x = (x | (x << 4)) & 0x0f0f0f0f;
  x = (x | (x << 2)) & 0x33333333;
  x = (x | (x << 1)) & 0x55555555;
  return x;
}

/** 1入力ビットを出力2ビット(2i,2i+1)へ複製する(nearest upsample x2の核)。 */
function dilate16To32(v: number): number {
  const even = spreadEven16(v);
  return even | (even << 1);
}

/** upsample2d(scale=2, nearest)をビットプレーンのままネイティブ実行する。 */
export function dilateX2(bp: BitplaneValue, batch: number): BitplaneValue {
  const { C, H, W, wpr } = bp;
  const HOut = H * 2;
  const WOut = W * 2;
  const wprOut = wordsPerRow(WOut);
  const maskOut = tailMask(WOut);
  const words = new Int32Array(batch * C * HOut * wprOut);
  const inPlaneWords = C * H * wpr;
  const outPlaneWords = C * HOut * wprOut;
  for (let b = 0; b < batch; b++) {
    const ib = b * inPlaneWords;
    const ob = b * outPlaneWords;
    for (let c = 0; c < C; c++) {
      const icb = ib + c * H * wpr;
      const ocb = ob + c * HOut * wprOut;
      for (let y = 0; y < H; y++) {
        const rowIn = icb + y * wpr;
        const rowOut0 = ocb + (2 * y) * wprOut;
        const rowOut1 = ocb + (2 * y + 1) * wprOut;
        for (let ow = 0; ow < wprOut; ow++) {
          const srcIdx = ow >>> 1;
          const srcWord = srcIdx < wpr ? bp.words[rowIn + srcIdx] : 0;
          const half = (ow & 1) === 0 ? (srcWord & 0xffff) : ((srcWord >>> 16) & 0xffff);
          const v = dilate16To32(half);
          words[rowOut0 + ow] = v;
          words[rowOut1 + ow] = v;
        }
        words[rowOut0 + wprOut - 1] &= maskOut;
        words[rowOut1 + wprOut - 1] &= maskOut;
      }
    }
  }
  return { words, C, H: HOut, W: WOut, wpr: wprOut };
}

// ============================================================
// 7. copyPlanes: チャネル選択コピー(permuteのチャネルのみ並べ替え用)
// ============================================================

/** チャネル軸だけを並べ替える(H,W,wprは不変)。出力チャネル o の内容は
    src のチャネル channelOrder[o] のコピー。permute のうち「末尾2軸
    (H,W)が不変なチャネル並べ替えのみ」に限定して呼び出すこと
    (spatialgroupsum型 permute のような空間軸を跨ぐ並べ替えは対象外)。 */
export function copyPlanes(
  src: BitplaneValue, batch: number, channelOrder: number[],
): BitplaneValue {
  const { H, W, wpr } = src;
  const COut = channelOrder.length;
  const planeWordsIn = src.C * H * wpr;
  const planeWordsOut = COut * H * wpr;
  const chWords = H * wpr;
  const words = new Int32Array(batch * planeWordsOut);
  for (let b = 0; b < batch; b++) {
    const ib = b * planeWordsIn;
    const ob = b * planeWordsOut;
    for (let o = 0; o < COut; o++) {
      const srcC = channelOrder[o];
      words.set(
        src.words.subarray(ib + srcC * chWords, ib + srcC * chWords + chWords),
        ob + o * chWords,
      );
    }
  }
  return { words, C: COut, H, W, wpr };
}

// ============================================================
// 8. concatPlanes: チャネル軸concat(空間形状が揃った入力のみ)
// ============================================================

/** 空間形状(H,W)が全入力で一致する concat を、ビットプレーンの
    まま平面連結する。NodeValue の concat は「バッチ要素ごとに [ins0, ins1,
    ...] を連結」だが、全入力が同じ (H,W) の NCHW なら、これは
    「チャネル平面の連結(C_out = ΣC_i、平面の並びは入力順)」と厳密に同一
    (どちらも flat index = ((c_global*H)+y)*W+x の順)。
    UNet型のskip接続(seg: n7/n10_concat)がこれに当たり、従来は両入力を
    アンパック→int32連結→次のconv2dで再パックしていた往復が消える。
    空間形状が食い違う入力(実モデルには無いが概念上ありうる)は呼び出し側
    (bitplaneGraph.ts)が従来のアンパック経路へ落とすこと。 */
export function concatPlanes(ins: BitplaneValue[], batch: number): BitplaneValue {
  const { H, W, wpr } = ins[0];
  const COut = ins.reduce((a, v) => a + v.C, 0);
  const outPlaneWords = COut * H * wpr;
  const words = new Int32Array(batch * outPlaneWords);
  for (let b = 0; b < batch; b++) {
    let off = b * outPlaneWords;
    for (const v of ins) {
      const n = v.C * H * wpr;
      words.set(v.words.subarray(b * n, b * n + n), off);
      off += n;
    }
  }
  return { words, C: COut, H, W, wpr };
}

// ============================================================
// 9. spatialGroupSumBitplane: 読み出しヘッド(permute+group_sum)のパックネイティブ評価
// ============================================================

/** SpatialGroupSum(lowering.py::_extract_spatial_group_sum_native が
    permute(in_shape=[nOut,gs,H,W], perm=[0,2,3,1]) + group_sum(k=nOut*H*W) の
    2ノードへ落とすパターン)を、ビットプレーンをアンパックせずに
    整数スコア列へ直接畳み込む。
    意味論(irflat.ts::evalPermutePacked + evalGroupSum と厳密に同一):
      permute後のflat index = ((o*H + y)*W + x)*gs + g
      → group_sum の groupインデックス = floor(idx/gs) = o*H*W + y*W + x
      ∴ score[o*H*W + y*W + x] = Σ_{g<gs} bit(channel o*gs+g, y, x)
    ビットプレーン([C][H][wpr])ではこれは「同一ワード位置における gs 枚の
    チャネル平面のビット縦和」なので、packed.ts::groupSumScores32 と同じ
    ビットスライス carry-save 加算(垂直カウンタ: planes[p] ^= carry;
    carry = old & carry)で1ワード=32空間位置ぶんを並列に足せる。
    平面数 nPlanes = ceil(log2(gs+1)) = 32 - clz32(gs)(総和は高々 gs なので
    最上位平面からの繰り上がりは常に0)。
    最後にビットスライス平面 → Int32 スコアへ転置する。転置は「立っている
    ビットだけを走査する」(v &= v-1 のポップループ)ので、コストは
    32*nPlanes ではなく実際の総ポップカウントに比例する(スコアが小さい
    実データほど安くなる。全ゼロワードは自動的にスキップされる)。
    行末尾の端数ビット(x>=W)はビットプレーンの不変条件により常に0だが、
    転置時に tailMask を明示的に掛けて出力幅の外へ書かないことを構造的に
    保証する(不変条件への依存を減らす)。 */
export function spatialGroupSumBitplane(
  bp: BitplaneValue, batch: number, nOut: number, gs: number,
): NodeValue {
  const { C, H, W, wpr } = bp;
  if (C !== nOut * gs) {
    throw new Error(`spatialGroupSumBitplane: チャネル数不一致 (C=${C}, nOut*gs=${nOut * gs})`);
  }
  const width = nOut * H * W;
  const data = new Int32Array(batch * width);
  const nPlanes = 32 - Math.clz32(gs); // ceil(log2(gs+1))
  const planes = new Int32Array(nPlanes * wpr); // [p][j] レイアウト
  const chWords = H * wpr;
  const inPlaneWords = C * chWords;
  const lastMask = tailMask(W);
  for (let b = 0; b < batch; b++) {
    const ib = b * inPlaneWords;
    const ob = b * width;
    for (let o = 0; o < nOut; o++) {
      const oBase = ib + o * gs * chWords;
      for (let y = 0; y < H; y++) {
        planes.fill(0);
        // --- 垂直カウンタ加算(g方向) ---
        for (let g = 0; g < gs; g++) {
          const rowBase = oBase + g * chWords + y * wpr;
          for (let j = 0; j < wpr; j++) {
            let carry = bp.words[rowBase + j];
            for (let p = 0; carry !== 0 && p < nPlanes; p++) {
              const idx = p * wpr + j;
              const t = planes[idx] & carry;
              planes[idx] ^= carry;
              carry = t;
            }
          }
        }
        // --- ビットスライス平面 → Int32 スコアへ転置 ---
        const outRow = ob + (o * H + y) * W;
        for (let j = 0; j < wpr; j++) {
          const xBase = j << 5;
          const mask = j === wpr - 1 ? lastMask : -1;
          for (let p = 0; p < nPlanes; p++) {
            let v = planes[p * wpr + j] & mask;
            if (v === 0) continue;
            const bit = 1 << p;
            while (v !== 0) {
              const low = v & -v;
              data[outRow + xBase + (31 - Math.clz32(low))] |= bit;
              v ^= low;
            }
          }
        }
      }
    }
  }
  return { data, width };
}

// ============================================================
// conv2d をビットプレーンのままネイティブ実行
// ============================================================

function pair(v: number | number[]): [number, number] {
  return Array.isArray(v) ? [v[0], v[1]] : [v, v];
}

/** gate2(packedConv.ts / assembly/index.ts と同一の16式)。 */
function gate2(tt: number, a: number, b: number): number {
  switch (tt & 0xf) {
    case 0x0: return 0;
    case 0x1: return ~(a | b);
    case 0x2: return a & ~b;
    case 0x3: return ~b;
    case 0x4: return ~a & b;
    case 0x5: return ~a;
    case 0x6: return a ^ b;
    case 0x7: return ~(a & b);
    case 0x8: return a & b;
    case 0x9: return ~(a ^ b);
    case 0xa: return a;
    case 0xb: return a | ~b;
    case 0xc: return b;
    case 0xd: return ~a | b;
    case 0xe: return a | b;
    case 0xf: return -1;
    default: return 0;
  }
}

/** conv2d をビットプレーン入出力のままネイティブ実行する
    (evalConv2dPacked のビットプレーン版)。高速路(stride=1 かつ
    WOut===W)のタップ収集は「共有シフト平面」方式:
    dedupキーを (c,dy,dx) から (c,dx) へ粗くし、(c,dx) ごとに H+2*pH 行の
    シフト済み平面を1回だけ構築、各リーフは「平面ID + dy 由来の定数行
    オフセット」で参照する(木縮約の最下層だけがプールを間接参照し、
    第1レベル出力以降は従来どおり整列済みバンクの ping-pong)。
    strided路(stride>1 / WOut!==W)は従来どおり (c,dy,dx) ごとに HOut 行の
    タップ平面を buildTapPlaneStrided で作る(正確性のアンカー、変更なし)。
    どちらの経路も「平面プール(ワード連続) + リーフごとのワード基点」という
    同一の表現に落とすので、木の評価コードは1本で済む。木のping-pong評価は
    従来と同型。
    出力の行末尾マスクは「木の最終出力の各行」にのみ適用すればよい
    (ゲート演算はワード内のビット位置をまたがない純粋な位置ごとの論理
    演算なので、途中levelのタップ/中間ノードの端数ビットに何が入って
    いても、最終的に読み出す実位置(x<WOut)のビットには一切影響しない。
    レベルごとのマスクは冗長なため省略している)。 */
export function evalConv2dBitplane(
  p: Conv2dParams, x: BitplaneValue, batch: number,
): BitplaneValue {
  const [, H, W] = p.in_shape;
  if (x.H !== H || x.W !== W) {
    throw new Error(`evalConv2dBitplane: 入力形状不一致 (期待 H=${H} W=${W}, 実際 H=${x.H} W=${x.W})`);
  }
  const [kH, kW] = pair(p.kernel_size);
  const [sH, sW] = pair(p.stride);
  const [pH, pW] = pair(p.padding);
  const HOut = Math.floor((H + 2 * pH - kH) / sH) + 1;
  const WOut = Math.floor((W + 2 * pW - kW) / sW) + 1;
  const OC = p.out_channels;
  const N = p.n_inputs;
  const depth = p.tree_depth;
  if (N !== 2) throw new Error(`bitplane conv は n_inputs=2 のみ対応(実際: ${N})`);
  const nLeaves = N ** depth;
  const nInternal = (nLeaves - 1) / (N - 1);
  const khw = kH * kW;
  const idxf = p.indices_flat;

  const ttNum = new Uint8Array(OC * nInternal);
  for (let i = 0; i < ttNum.length; i++) {
    ttNum[i] = Number(ttParse(p.truth_tables[i])) & 0xf;
  }

  const wprIn = x.wpr;
  const wprOut = wordsPerRow(WOut);
  const maskOut = tailMask(WOut);
  const useFastPath = sH === 1 && sW === 1 && WOut === W;
  // 高速路では入出力のwprが一致する前提(shiftRowInto)なのでタップ平面の
  // wprはwprIn、遅いパスではwprOutをそのまま使う。
  const tapWpr = useFastPath ? wprIn : wprOut;
  const tapRowsWords = HOut * tapWpr;
  // 平面プールの1平面あたり行数: 高速路は共有シフト平面(上下pH行のゼロ
  // パディング込みでH+2*pH行)、strided路は従来どおり1タップ=HOut行。
  const planeRows = useFastPath ? H + 2 * pH : HOut;
  const planeStride = planeRows * tapWpr;

  // --- 平面のdedup(高速路:(c,dx) / strided路:(c,dy,dx))とリーフのワード基点。
  // バッチ非依存の静的ジオメトリなのでバッチループの外で1回だけ計算する。 ---
  const keyToPlane = new Map<number, number>();
  const planeC: number[] = [];
  const planeDy: number[] = [];
  const planeDx: number[] = [];
  const leafWordBase = new Int32Array(OC * nLeaves);
  for (let oc = 0; oc < OC; oc++) {
    const row = idxf[oc];
    for (let k = 0; k < nLeaves; k++) {
      const pp = row[k];
      const rem = pp % khw;
      const c = Math.floor(pp / khw);
      const dy = Math.floor(rem / kW);
      const dx = rem % kW;
      const key = useFastPath ? c * kW + dx : c * khw + dy * kW + dx;
      let pid = keyToPlane.get(key);
      if (pid === undefined) {
        pid = planeC.length;
        planeC.push(c); planeDy.push(dy); planeDx.push(dx);
        keyToPlane.set(key, pid);
      }
      // 高速路のみ dy が行オフセットになる(strided路は1タップ1平面)。
      leafWordBase[oc * nLeaves + k] = pid * planeStride + (useFastPath ? dy * tapWpr : 0);
    }
  }
  const nPlanes = planeC.length;

  const inPlaneWords = x.C * x.H * wprIn;
  const outPlaneWords = OC * HOut * wprOut;
  const words = new Int32Array(batch * outPlaneWords);

  const maxBank = Math.max(1, nLeaves >> 1);
  const bankA = Array.from({ length: maxBank }, () => new Int32Array(tapRowsWords));
  const bankB = Array.from({ length: maxBank }, () => new Int32Array(tapRowsWords));
  // 平面プール(全平面が連続、バッチ要素ごとに全域を書き直すので使い回す)。
  const pool = new Int32Array(nPlanes * planeStride);

  for (let b = 0; b < batch; b++) {
    const xoff = b * inPlaneWords;
    const ooff = b * outPlaneWords;

    for (let u = 0; u < nPlanes; u++) {
      const cBase = xoff + planeC[u] * x.H * wprIn;
      const dstBase = u * planeStride;
      if (useFastPath) {
        buildSharedShiftPlane(x.words, cBase, H, wprIn, pH, planeDx[u] - pW, pool, dstBase);
      } else {
        buildTapPlaneStrided(
          x.words, cBase, H, W, wprIn, planeDy[u], planeDx[u],
          HOut, WOut, wprOut, sH, sW, pH, pW, pool, dstBase,
        );
      }
    }

    for (let oc = 0; oc < OC; oc++) {
      const lb = oc * nLeaves;
      // 木の最下層(リーフ→第1レベル)だけがプールを間接参照する。以降は
      // 整列済みバンク(基点0)のping-pongで従来と全く同じ。
      let outPlane: Int32Array;
      let outBase: number;
      if (nLeaves === 1) {
        // depth=0相当(木自体が無い): リーフ0がそのまま出力。
        outPlane = pool;
        outBase = leafWordBase[lb];
      } else {
        const gOut0 = nLeaves >> 1;
        for (let g = 0; g < gOut0; g++) {
          const tt = ttNum[oc * nInternal + g];
          const aB = leafWordBase[lb + (g << 1)];
          const bB = leafWordBase[lb + (g << 1) + 1];
          const dst = bankA[g];
          for (let wi = 0; wi < tapRowsWords; wi++) {
            dst[wi] = gate2(tt, pool[aB + wi], pool[bB + wi]);
          }
        }
        let src = bankA;
        let srcLen = gOut0;
        let dstBank = bankB;
        let gOffset = gOut0;
        while (srcLen > 1) {
          const gOut = srcLen >> 1;
          for (let g = 0; g < gOut; g++) {
            const tt = ttNum[oc * nInternal + gOffset + g];
            const a = src[g << 1];
            const bp2 = src[(g << 1) + 1];
            const dst = dstBank[g];
            for (let wi = 0; wi < tapRowsWords; wi++) dst[wi] = gate2(tt, a[wi], bp2[wi]);
          }
          gOffset += gOut;
          src = dstBank;
          srcLen = gOut;
          dstBank = dstBank === bankA ? bankB : bankA;
        }
        outPlane = src[0];
        outBase = 0;
      }

      // tapWpr===wprOut(高速路ではWOut===W前提よりwprIn===wprOut、遅い
      // パスではtapWpr自体がwprOut)なので経路によらず同一の行コピー+
      // 行末尾マスクでよい。
      const ocBase = ooff + oc * HOut * wprOut;
      for (let oy = 0; oy < HOut; oy++) {
        const srcRow = outBase + oy * tapWpr;
        const dstRow = ocBase + oy * wprOut;
        for (let j = 0; j < wprOut; j++) words[dstRow + j] = outPlane[srcRow + j];
        words[dstRow + wprOut - 1] &= maskOut;
      }
    }
  }
  return { words, C: OC, H: HOut, W: WOut, wpr: wprOut };
}

// ============================================================
// conv2d 命令列(convProgram.ts)版のレイアウトと実行
// ============================================================

/** Conv2dParams から命令列化の入力仕様(ビットプレーン非依存)を作る。 */
export function convTreeSpecOf(p: Conv2dParams): ConvTreeSpec {
  if (p.n_inputs !== 2) {
    throw new Error(`conv命令列化は n_inputs=2 のみ対応(実際: ${p.n_inputs})`);
  }
  const nInternal = (1 << p.tree_depth) - 1;
  const tt = new Uint8Array(p.out_channels * nInternal);
  for (let i = 0; i < tt.length; i++) tt[i] = Number(ttParse(p.truth_tables[i])) & 0xf;
  return {
    outChannels: p.out_channels,
    treeDepth: p.tree_depth,
    indicesFlat: p.indices_flat,
    truthTables: tt,
  };
}

/** 命令列版カーネルの静的レイアウト(JS版/WASM版で完全に共有する)。
    スクラッチは1本の連続バッファ(ワード単位オフセットで全参照を解決済み):
      [0, tapRowsWords)                : 定数0平面
      [tapRowsWords, 2*tapRowsWords)   : 定数1平面(全ビット1)
      [poolBase, regBase)              : タップ平面プール(nPlanes*planeStride)
      [regBase, scratchWords)          : 命令の作業平面(nSlots*tapRowsWords)
    こうすると命令のオペランド/デスティネーションは全て「このバッファ内の
    絶対ワードオフセット」1個で表せる(kind分岐が実行時から消える)。
    平面プールの構築(共有シフト平面/strided)と幾何は evalConv2dBitplane と
    完全に同一。 */
export interface ConvProgramLayout {
  H: number; W: number; HOut: number; WOut: number;
  sH: number; sW: number; pH: number; pW: number;
  OC: number;
  wprIn: number; wprOut: number; tapWpr: number; tapRowsWords: number;
  maskOut: number; useFastPath: boolean;
  planeC: Int32Array; planeDy: Int32Array; planeDx: Int32Array;
  nPlanes: number; planeStride: number;
  inPlaneWords: number; outPlaneWords: number;
  program: ConvProgram;
  /** 命令ごとに4語 [tt, aOff, bOff, dstOff](オフセットは絶対ワード)。 */
  instrFlat: Int32Array;
  /** 各ocの出力元となる絶対ワードオフセット。 */
  ocSrcOff: Int32Array;
  /** 各タップの絶対ワードオフセット(codegenが直接埋め込む)。*/
  tapWordBase: Int32Array;
  poolBase: number; regBase: number; scratchWords: number;
}

/** conv2dノードの命令列版レイアウトを構築する(バッチ非依存・構築時1回)。
    `prebuilt` が渡されればそれをそのまま使い、buildConvProgram/
    verifyConvProgram を再実行しない(呼び出し側
    (bundleProgram.ts::resolveConvProgram)が既にデシリアライズ+構造検証+
    全数真理値表照合を済ませている前提)。`prebuilt` 省略時は従来どおり
    ここで構築し、最後に verifyConvProgram で「元のバランス木 vs 縮約後
    命令列」を全数真理値表照合する(不一致なら例外 → 呼び出し側は旧経路へ
    フォールバック)。 */
export function buildConvProgramLayout(p: Conv2dParams, prebuilt?: ConvProgram): ConvProgramLayout {
  const [, H, W] = p.in_shape;
  const [kH, kW] = pair(p.kernel_size);
  const [sH, sW] = pair(p.stride);
  const [pH, pW] = pair(p.padding);
  const HOut = Math.floor((H + 2 * pH - kH) / sH) + 1;
  const WOut = Math.floor((W + 2 * pW - kW) / sW) + 1;
  const OC = p.out_channels;
  const khw = kH * kW;

  let program: ConvProgram;
  if (prebuilt) {
    program = prebuilt;
  } else {
    const spec = convTreeSpecOf(p);
    program = buildConvProgram(spec);
    verifyConvProgram(spec, program);
  }

  const wprIn = wordsPerRow(W);
  const wprOut = wordsPerRow(WOut);
  const useFastPath = sH === 1 && sW === 1 && WOut === W;
  const tapWpr = useFastPath ? wprIn : wprOut;
  const tapRowsWords = HOut * tapWpr;
  const planeRows = useFastPath ? H + 2 * pH : HOut;
  const planeStride = planeRows * tapWpr;

  // タップ(ユニークな indices_flat 値)→ 平面dedup。キーは evalConv2dBitplane
  // と同一(高速路 (c,dx) / strided路 (c,dy,dx))。
  const keyToPlane = new Map<number, number>();
  const planeCArr: number[] = [];
  const planeDyArr: number[] = [];
  const planeDxArr: number[] = [];
  const nTaps = program.taps.length;
  const tapWordBase = new Int32Array(nTaps);
  const tapRowOff = new Int32Array(nTaps);
  const tapPlane = new Int32Array(nTaps);
  for (let t = 0; t < nTaps; t++) {
    const pp = program.taps[t];
    const rem = pp % khw;
    const c = Math.floor(pp / khw);
    const dy = Math.floor(rem / kW);
    const dx = rem % kW;
    const key = useFastPath ? c * kW + dx : c * khw + dy * kW + dx;
    let pid = keyToPlane.get(key);
    if (pid === undefined) {
      pid = planeCArr.length;
      planeCArr.push(c); planeDyArr.push(dy); planeDxArr.push(dx);
      keyToPlane.set(key, pid);
    }
    tapPlane[t] = pid;
    tapRowOff[t] = useFastPath ? dy * tapWpr : 0;
  }
  const nPlanes = planeCArr.length;

  const poolBase = 2 * tapRowsWords;
  const regBase = poolBase + nPlanes * planeStride;
  const scratchWords = regBase + Math.max(1, program.nSlots) * tapRowsWords;
  for (let t = 0; t < nTaps; t++) {
    tapWordBase[t] = poolBase + tapPlane[t] * planeStride + tapRowOff[t];
  }

  const refOff = (r: ConvRef): number => {
    const kind = refKind(r);
    if (kind === REF_KIND_CONST) return refIdx(r) * tapRowsWords;
    if (kind === REF_KIND_TAP) return tapWordBase[refIdx(r)];
    return regBase + program.instrs[refIdx(r)].slot * tapRowsWords;
  };

  const nInstr = program.instrs.length;
  const instrFlat = new Int32Array(nInstr * 4);
  for (let j = 0; j < nInstr; j++) {
    const ins = program.instrs[j];
    instrFlat[j * 4] = ins.tt;
    instrFlat[j * 4 + 1] = refOff(ins.a);
    instrFlat[j * 4 + 2] = refOff(ins.b);
    instrFlat[j * 4 + 3] = regBase + ins.slot * tapRowsWords;
  }
  const ocSrcOff = new Int32Array(OC);
  for (let oc = 0; oc < OC; oc++) ocSrcOff[oc] = refOff(program.ocOut[oc]);

  return {
    H, W, HOut, WOut, sH, sW, pH, pW, OC,
    wprIn, wprOut, tapWpr, tapRowsWords,
    maskOut: tailMask(WOut), useFastPath,
    planeC: Int32Array.from(planeCArr),
    planeDy: Int32Array.from(planeDyArr),
    planeDx: Int32Array.from(planeDxArr),
    nPlanes, planeStride,
    inPlaneWords: p.in_shape[0] * H * wprIn,
    outPlaneWords: OC * HOut * wprOut,
    program, instrFlat, ocSrcOff, tapWordBase,
    poolBase, regBase, scratchWords,
  };
}

/** 1命令ぶんの区間演算(gate2の16式、switchをワードループの外に出した版)。
    buf は単一スクラッチ(定数平面/タップ平面プール/命令の作業平面が同居)。
    dstOff はスロット割当により常にオペランドと異なる領域なのでエイリアスは
    起きない(convProgram.ts の生存区間解析がそれを保証する)。 */
export function gate2RangeInto(
  buf: Int32Array, dstOff: number, aOff: number, bOff: number, tt: number, n: number,
): void {
  switch (tt & 0xf) {
    case 0x0: for (let i = 0; i < n; i++) buf[dstOff + i] = 0; break;
    case 0x1: for (let i = 0; i < n; i++) buf[dstOff + i] = ~(buf[aOff + i] | buf[bOff + i]); break;
    case 0x2: for (let i = 0; i < n; i++) buf[dstOff + i] = buf[aOff + i] & ~buf[bOff + i]; break;
    case 0x3: for (let i = 0; i < n; i++) buf[dstOff + i] = ~buf[bOff + i]; break;
    case 0x4: for (let i = 0; i < n; i++) buf[dstOff + i] = ~buf[aOff + i] & buf[bOff + i]; break;
    case 0x5: for (let i = 0; i < n; i++) buf[dstOff + i] = ~buf[aOff + i]; break;
    case 0x6: for (let i = 0; i < n; i++) buf[dstOff + i] = buf[aOff + i] ^ buf[bOff + i]; break;
    case 0x7: for (let i = 0; i < n; i++) buf[dstOff + i] = ~(buf[aOff + i] & buf[bOff + i]); break;
    case 0x8: for (let i = 0; i < n; i++) buf[dstOff + i] = buf[aOff + i] & buf[bOff + i]; break;
    case 0x9: for (let i = 0; i < n; i++) buf[dstOff + i] = ~(buf[aOff + i] ^ buf[bOff + i]); break;
    case 0xa: for (let i = 0; i < n; i++) buf[dstOff + i] = buf[aOff + i]; break;
    case 0xb: for (let i = 0; i < n; i++) buf[dstOff + i] = buf[aOff + i] | ~buf[bOff + i]; break;
    case 0xc: for (let i = 0; i < n; i++) buf[dstOff + i] = buf[bOff + i]; break;
    case 0xd: for (let i = 0; i < n; i++) buf[dstOff + i] = ~buf[aOff + i] | buf[bOff + i]; break;
    case 0xe: for (let i = 0; i < n; i++) buf[dstOff + i] = buf[aOff + i] | buf[bOff + i]; break;
    default: for (let i = 0; i < n; i++) buf[dstOff + i] = -1; break;
  }
}

/** BitplaneProgramConvKernel の生成オプション。codegen=false(既定)なら
    からの命令列インタプリタで、整数出力は両者で完全に一致する。*/
export interface BitplaneProgramConvOptions extends ConvCodegenOptions {
  /** 命令列を直線JSコードへ生成して実行する(既定 false)。 */
  codegen?: boolean;
  /** 事前構築済みの命令列(配布バンドルの programs 節をデシリアライズ・
      検証済みのもの、bundleProgram.ts::resolveConvProgram の結果)。渡された
      場合は buildConvProgramLayout がこれをそのまま使い、buildConvProgram/
      verifyConvProgram を再実行しない。 */
  program?: ConvProgram;
}

/** conv2dノードの命令列版カーネル(純JS)。evalConv2dBitplane と
    整数完全一致する(縮約は losslessなブール恒等式のみ、出力行末尾は同じ
    tailMask を適用する)。レイアウトは構築時1回・以後はスクラッチ使い回し。 */
export class BitplaneProgramConvKernel {
  readonly layout: ConvProgramLayout;
  private readonly scratch: Int32Array;
  /** 直線JSコード生成が有効なら生成済みチャンク関数(null=インタプリタ)。*/
  readonly compiled: CompiledConvProgram | null = null;
  /** codegenを要求されたが使えなかった理由(黙殺しない)。*/
  readonly codegenNote: string | null = null;

  constructor(p: Conv2dParams, opts: BitplaneProgramConvOptions = {}) {
    this.layout = buildConvProgramLayout(p, opts.program);
    const L = this.layout;
    this.scratch = new Int32Array(L.scratchWords);
    this.scratch.fill(-1, L.tapRowsWords, 2 * L.tapRowsWords); // 定数1平面
    if (opts.codegen) {
      try {
        this.compiled = compileConvProgram(L, opts);
        // 生成直後に1回空打ちしてバイトコード生成(遅延パース)をロード時へ寄せる。
        // 書き先はスクラッチの作業平面領域と使い捨て出力だけなので副作用は無い。
        const dummy = new Int32Array(L.outPlaneWords);
        for (const fn of this.compiled.fns) fn(this.scratch, dummy, 0);
      } catch (e) {
        this.compiled = null;
        this.codegenNote = `conv2d codegen不可→命令列インタプリタで実行: ${String(e)}`;
      }
    }
  }

  evalAll(x: BitplaneValue, batch: number): BitplaneValue {
    const L = this.layout;
    if (x.H !== L.H || x.W !== L.W) {
      throw new Error(`BitplaneProgramConvKernel: 入力形状不一致 (期待 H=${L.H} W=${L.W}, 実際 H=${x.H} W=${x.W})`);
    }
    const s = this.scratch;
    const instr = L.instrFlat;
    const nInstr = instr.length >> 2;
    const { writeStart } = L.program;
    const words = new Int32Array(batch * L.outPlaneWords);

    for (let b = 0; b < batch; b++) {
      const xoff = b * L.inPlaneWords;
      const ooff = b * L.outPlaneWords;

      for (let u = 0; u < L.nPlanes; u++) {
        const cBase = xoff + L.planeC[u] * L.H * L.wprIn;
        const dstBase = L.poolBase + u * L.planeStride;
        if (L.useFastPath) {
          buildSharedShiftPlane(x.words, cBase, L.H, L.wprIn, L.pH, L.planeDx[u] - L.pW, s, dstBase);
        } else {
          buildTapPlaneStrided(
            x.words, cBase, L.H, L.W, L.wprIn, L.planeDy[u], L.planeDx[u],
            L.HOut, L.WOut, L.wprOut, L.sH, L.sW, L.pH, L.pW, s, dstBase,
          );
        }
      }

      if (this.compiled) {
        // 直線コード版: 全命令をワードループ内へインラインした
        // チャンク関数を順に呼ぶ(oc書き出しも生成コードに埋まっている)。
        for (const fn of this.compiled.fns) fn(s, words, ooff);
        continue;
      }

      // バケット0: 出力が定数/タップ直結の oc(命令の実行前に書ける)。
      this.writeBucket(0, ooff, words);
      for (let j = 0; j < nInstr; j++) {
        const o = j << 2;
        gate2RangeInto(s, instr[o + 3], instr[o + 1], instr[o + 2], instr[o], L.tapRowsWords);
        // 命令jの直後に「出力がjの oc」を書き出す(生存区間はここで閉じる)。
        if (writeStart[j + 2] > writeStart[j + 1]) this.writeBucket(j + 1, ooff, words);
      }
    }
    return { words, C: L.OC, H: L.HOut, W: L.WOut, wpr: L.wprOut };
  }

  /** tapWpr === wprOut は経路によらず成り立つ(高速路は WOut===W より
      wprIn===wprOut、strided路は tapWpr がそのまま wprOut)。したがって1つの oc の
      出力平面は「作業平面からの HOut*wprOut 語の連続コピー + 各行末語への
      tailMask」で書ける。行ごとに wprOut(実モデルで1〜4)語だけの内側ループを
      回す旧形は、行あたりのループ立ち上げが語あたりコストを支配していた。
      書き込む語とマスク結果は旧形と完全に同一(整数出力不変)。 */
  private writeBucket(bucket: number, ooff: number, words: Int32Array): void {
    const L = this.layout;
    const { writeStart, writeOc } = L.program;
    const s = this.scratch;
    const { HOut, wprOut, maskOut } = L;
    const planeWords = HOut * wprOut;
    for (let i = writeStart[bucket]; i < writeStart[bucket + 1]; i++) {
      const oc = writeOc[i];
      const srcOff = L.ocSrcOff[oc];
      const ocBase = ooff + oc * planeWords;
      // subarray を作らない素の連続コピー(oc ごとの短命ビューを避ける)。
      for (let k = 0; k < planeWords; k++) words[ocBase + k] = s[srcOff + k];
      for (let oy = 0; oy < HOut; oy++) words[ocBase + oy * wprOut + wprOut - 1] &= maskOut;
    }
  }
}
