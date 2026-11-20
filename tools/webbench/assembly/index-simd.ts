// assembly/index.ts の bitplane/dense conv2d カーネルを WASM v128 SIMD
// (4ワード=128位置同時)で評価する独立エントリファイル。AssemblyScriptは
// v128型の使用に `--enable simd` を要求し、それを無効化した「release」
// (index.ts、スカラーu32版)側では型検査自体が通らないため、機能フラグの
// 異なる2つの出力(build/release.wasm と build/release-simd.wasm)を
// 別々のエントリファイルとして完全に独立させている(共有すると
// asconfig.json のターゲットごとに feature flag を跨いだimportになり
// コンパイルが破綻するため、意図的な自己完結コピー)。
//
// index.tsとの唯一の意味的な違い: evalConv2dBatch/evalConv2dBitplaneBatch
// 内の「木をボトムアップ評価」ループ(gOut回のgate2適用)を、位置(wi)方向に
// 4ワード(128ビット)ずつv128.and/or/xor/notでまとめて処理する。ゲート演算は
// ワード内のビット位置をまたがない純粋な位置ごとの論理演算なので、
// 4ワード単位にチャンク化しても各ワードの値は完全に不変(端数は
// スカラーgate2にフォールバックするのでtapRowsWords/wordsOutが4の倍数
// でなくても常に正しい)。leaf/タップ平面構築(shiftRowIntoAsm等)は
// ワードあたりのコストが小さいため意図的にスカラーのまま(計画書の
// 「タップ平面構築のshiftRowはwpr≦3なのでスカラーのままで良い」を踏襲)。
//
// 正しさの錨: index.ts(スカラー)とのビット完全一致(bitplaneSimd.test.ts
// でconv_golden.json/実モデルgoldenを両ビルドで実行し突き合わせる)。

// ---- 単純バンプアロケータ(index.tsと同一実装) ----
let heapPtr: i32 = 8; // 0番地は「ヌル」として避ける

export function alloc(bytes: i32): i32 {
  const ptr = heapPtr;
  let end = ptr + bytes;
  end = (end + 7) & ~7; // 8バイト境界に揃える
  const neededPages = (end + 0xffff) >>> 16;
  const curPages = memory.size();
  if (neededPages > curPages) {
    memory.grow(neededPages - curPages);
  }
  heapPtr = end;
  return ptr;
}

/** ヒープを先頭(8)へ巻き戻す(ConvRuntime再構築時の再利用用、任意)。 */
export function resetHeap(): void {
  heapPtr = 8;
}

// codegen.ts gateExpr / packedConv.ts gate2 と同一の16式(u32ビット並列、
// 端数ワード用のスカラー経路)。
function gate2(tt: u32, a: u32, b: u32): u32 {
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
    default: return 0xffffffff; // 0xf
  }
}

// gate2のv128版(4x u32レーン=128位置を同時評価)。各レーンは他レーンから
// 完全に独立(ワード内ビット位置をまたがない純粋な位置ごとの論理演算)なので、
// gate2をレーン毎に適用したのと逐語的に同じ結果になる。
function gate2v128(tt: u32, a: v128, b: v128): v128 {
  switch (tt & 0xf) {
    case 0x0: return i32x4.splat(0);
    case 0x1: return v128.not(v128.or(a, b));
    case 0x2: return v128.andnot(a, b);
    case 0x3: return v128.not(b);
    case 0x4: return v128.andnot(b, a);
    case 0x5: return v128.not(a);
    case 0x6: return v128.xor(a, b);
    case 0x7: return v128.not(v128.and(a, b));
    case 0x8: return v128.and(a, b);
    case 0x9: return v128.not(v128.xor(a, b));
    case 0xa: return a;
    case 0xb: return v128.or(a, v128.not(b));
    case 0xc: return b;
    case 0xd: return v128.or(v128.not(a), b);
    case 0xe: return v128.or(a, b);
    default: return i32x4.splat(-1); // 0xf
  }
}

/** dst[dstBase..)の[wi0,wiEnd)区間へ src[a]とsrc[b]のgate2(tt,a,b)を書く。
    4ワード単位はv128、端数(<4)はスカラーgate2。ptrはワードオフセット
    (バイトアドレスへは呼び出し側で<<2しない、この関数の中で<<2する)。 */
function gate2RangeInto(
  dstPtr: i32, dstBase: i32,
  aPtr: i32, aBase: i32,
  bPtr: i32, bBase: i32,
  tt: u32, n: i32,
): void {
  const vecEnd = n & ~3;
  let wi = 0;
  for (; wi < vecEnd; wi += 4) {
    const av: v128 = v128.load(aPtr + ((aBase + wi) << 2), 0, 4);
    const bv: v128 = v128.load(bPtr + ((bBase + wi) << 2), 0, 4);
    v128.store(dstPtr + ((dstBase + wi) << 2), gate2v128(tt, av, bv), 0, 4);
  }
  for (; wi < n; wi++) {
    const a: u32 = load<i32>(aPtr + ((aBase + wi) << 2));
    const b: u32 = load<i32>(bPtr + ((bBase + wi) << 2));
    store<i32>(dstPtr + ((dstBase + wi) << 2), gate2(tt, a, b));
  }
}

/** conv2d 木プログラムをバッチ1要素ぶん評価する(全oc)。index.tsの
    evalConv2dBatchと幾何・入出力契約は完全同一、木縮約ループのみ
    gate2RangeIntoでv128チャンク化。 */
export function evalConv2dBatch(
  xPtr: i32,
  leafCPtr: i32, leafDyPtr: i32, leafDxPtr: i32, ttNumPtr: i32,
  leafPlanesPtr: i32, bankAPtr: i32, bankBPtr: i32,
  outPtr: i32,
  H: i32, W: i32, HOut: i32, WOut: i32,
  sH: i32, sW: i32, pH: i32, pW: i32,
  OC: i32, nLeaves: i32, nInternal: i32, wordsOut: i32,
): void {
  const Lout = HOut * WOut;

  for (let oc = 0; oc < OC; oc++) {
    const lb = oc * nLeaves;

    // --- leaf(タップ)平面を構築(ゲート木の葉、空間位置パック、スカラー) ---
    for (let k = 0; k < nLeaves; k++) {
      const c = load<i32>(leafCPtr + ((lb + k) << 2));
      const dy = load<i32>(leafDyPtr + ((lb + k) << 2));
      const dx = load<i32>(leafDxPtr + ((lb + k) << 2));
      const planeBase = leafPlanesPtr + (k * wordsOut << 2);
      for (let wi = 0; wi < wordsOut; wi++) {
        store<i32>(planeBase + (wi << 2), 0);
      }
      let po: i32 = 0;
      for (let oy = 0; oy < HOut; oy++) {
        const y = oy * sH - pH + dy;
        if (y < 0 || y >= H) {
          po += WOut;
          continue;
        }
        const rowBase = c * H * W + y * W;
        for (let ox = 0; ox < WOut; ox++, po++) {
          const xx = ox * sW - pW + dx;
          if (xx < 0 || xx >= W) continue;
          const xv = load<i32>(xPtr + ((rowBase + xx) << 2));
          if (xv != 0) {
            const wi = po >> 5;
            const addr = planeBase + (wi << 2);
            store<i32>(addr, load<i32>(addr) | (1 << (po & 31)));
          }
        }
      }
    }

    // --- 木をボトムアップ評価(ping-pongバンク、割当なし、木縮約はv128) ---
    let srcPtr = leafPlanesPtr;
    let srcLen = nLeaves;
    let dstPtr = bankAPtr;
    let gOffset = 0;
    while (srcLen > 1) {
      const gOut = srcLen >> 1;
      for (let g = 0; g < gOut; g++) {
        const tt: u32 = load<u8>(ttNumPtr + (oc * nInternal + gOffset + g));
        const aBase = (g << 1) * wordsOut;
        const bBase = ((g << 1) + 1) * wordsOut;
        const dBase = g * wordsOut;
        gate2RangeInto(dstPtr, dBase, srcPtr, aBase, srcPtr, bBase, tt, wordsOut);
      }
      gOffset += gOut;
      srcPtr = dstPtr;
      srcLen = gOut;
      dstPtr = srcPtr == bankAPtr ? bankBPtr : bankAPtr;
    }

    // --- 出力ビットを 0/1 の i32 列として書き出す ---
    const outBase = outPtr + (oc * Lout << 2);
    for (let po = 0; po < Lout; po++) {
      const wi = po >> 5;
      const v = (load<i32>(srcPtr + (wi << 2)) >>> (po & 31)) & 1;
      store<i32>(outBase + (po << 2), v);
    }
  }
}

// ============================================================
// bitplane conv2d カーネル(index.ts::evalConv2dBitplaneBatch の
// v128版)。幾何・入出力契約は完全同一、木縮約ループのみgate2RangeIntoで
// v128チャンク化。
// ============================================================

/** dst[dstBase..dstBase+wpr) へ src[srcBase..) を s ビットシフトして書く
    (index.ts::shiftRowIntoAsm と同一実装、スカラーのまま)。 */
function shiftRowIntoAsm(
  dstPtr: i32, dstBase: i32, srcPtr: i32, srcBase: i32, wpr: i32, s: i32,
): void {
  const wordShift = s >> 5;
  const bitShift = s & 31;
  if (bitShift == 0) {
    for (let j = 0; j < wpr; j++) {
      const k = j + wordShift;
      const v: i32 = (k >= 0 && k < wpr) ? load<i32>(srcPtr + ((srcBase + k) << 2)) : 0;
      store<i32>(dstPtr + ((dstBase + j) << 2), v);
    }
    return;
  }
  for (let j = 0; j < wpr; j++) {
    const k0 = j + wordShift;
    const k1 = k0 + 1;
    const a: u32 = (k0 >= 0 && k0 < wpr) ? load<i32>(srcPtr + ((srcBase + k0) << 2)) : 0;
    const b: u32 = (k1 >= 0 && k1 < wpr) ? load<i32>(srcPtr + ((srcBase + k1) << 2)) : 0;
    const v: i32 = (a >>> bitShift) | (b << (32 - bitShift));
    store<i32>(dstPtr + ((dstBase + j) << 2), v);
  }
}

/** stride=1・WOut===W専用の共有シフト平面構築
    (index.ts::buildSharedShiftAsm と幾何・出力語ともに完全同一)。
    本体(内部H行)を「行境界を無視した長さ H*wpr の直線ループ」で
    シフト合成し、行をまたいで混入した語だけを1語ずつ潰す形へ変更した。
    直線ループになったことで、ここが v128 化できるようになった(4語=128空間
    位置を同時にシフト合成する。各レーンは独立なのでスカラーと逐語同一)。
    行境界フィックスアップはスカラーのまま(1行あたり1語)。 */
function buildSharedShiftAsm(
  dstPtr: i32, dstBase: i32, srcPtr: i32, srcBase: i32,
  H: i32, wpr: i32, pH: i32, s: i32,
): void {
  const n = H * wpr;
  const body = dstBase + pH * wpr;
  for (let i = dstBase; i < body; i++) store<i32>(dstPtr + (i << 2), 0);
  const tail = body + n;
  for (let i = tail; i < tail + pH * wpr; i++) store<i32>(dstPtr + (i << 2), 0);
  if (s == 0) {
    let i = 0;
    for (; i + 4 <= n; i += 4) {
      v128.store(
        dstPtr + ((body + i) << 2), v128.load(srcPtr + ((srcBase + i) << 2), 0, 4), 0, 4,
      );
    }
    for (; i < n; i++) {
      store<i32>(dstPtr + ((body + i) << 2), load<i32>(srcPtr + ((srcBase + i) << 2)));
    }
    return;
  }
  if (s <= -32 || s >= 32) {
    for (let y = 0; y < H; y++) {
      shiftRowIntoAsm(dstPtr, body + y * wpr, srcPtr, srcBase + y * wpr, wpr, s);
    }
    return;
  }
  const wordShift = s >> 5;            // 0(s>0)か -1(s<0)
  const bitShift = s - wordShift * 32; // (0,32)
  const inv = 32 - bitShift;
  if (wpr == 1) {
    // 1行=1語: 隣接語は常に行外(=0)なので混入が起きない純粋な要素ごとシフト。
    let i = 0;
    if (wordShift == 0) {
      for (; i + 4 <= n; i += 4) {
        const av: v128 = v128.load(srcPtr + ((srcBase + i) << 2), 0, 4);
        v128.store(dstPtr + ((body + i) << 2), i32x4.shr_u(av, bitShift), 0, 4);
      }
      for (; i < n; i++) {
        const a: u32 = load<i32>(srcPtr + ((srcBase + i) << 2));
        store<i32>(dstPtr + ((body + i) << 2), a >>> bitShift);
      }
    } else {
      for (; i + 4 <= n; i += 4) {
        const av: v128 = v128.load(srcPtr + ((srcBase + i) << 2), 0, 4);
        v128.store(dstPtr + ((body + i) << 2), i32x4.shl(av, inv), 0, 4);
      }
      for (; i < n; i++) {
        const a: u32 = load<i32>(srcPtr + ((srcBase + i) << 2));
        store<i32>(dstPtr + ((body + i) << 2), a << inv);
      }
    }
    return;
  }
  if (wordShift == 0) {
    const last = n - 1;
    let i = 0;
    for (; i + 4 <= last; i += 4) {
      const av: v128 = v128.load(srcPtr + ((srcBase + i) << 2), 0, 4);
      const bv: v128 = v128.load(srcPtr + ((srcBase + i + 1) << 2), 0, 4);
      v128.store(
        dstPtr + ((body + i) << 2),
        v128.or(i32x4.shr_u(av, bitShift), i32x4.shl(bv, inv)), 0, 4,
      );
    }
    for (; i < last; i++) {
      const a: u32 = load<i32>(srcPtr + ((srcBase + i) << 2));
      const b: u32 = load<i32>(srcPtr + ((srcBase + i + 1) << 2));
      store<i32>(dstPtr + ((body + i) << 2), (a >>> bitShift) | (b << inv));
    }
    const al: u32 = load<i32>(srcPtr + ((srcBase + last) << 2));
    store<i32>(dstPtr + ((body + last) << 2), al >>> bitShift);
    for (let y = 0; y < H - 1; y++) {
      const k = y * wpr + (wpr - 1);
      const a: u32 = load<i32>(srcPtr + ((srcBase + k) << 2));
      store<i32>(dstPtr + ((body + k) << 2), a >>> bitShift);
    }
  } else {
    const a0: u32 = load<i32>(srcPtr + (srcBase << 2));
    store<i32>(dstPtr + (body << 2), a0 << inv);
    let i = 1;
    for (; i + 4 <= n; i += 4) {
      const av: v128 = v128.load(srcPtr + ((srcBase + i - 1) << 2), 0, 4);
      const bv: v128 = v128.load(srcPtr + ((srcBase + i) << 2), 0, 4);
      v128.store(
        dstPtr + ((body + i) << 2),
        v128.or(i32x4.shr_u(av, bitShift), i32x4.shl(bv, inv)), 0, 4,
      );
    }
    for (; i < n; i++) {
      const a: u32 = load<i32>(srcPtr + ((srcBase + i - 1) << 2));
      const b: u32 = load<i32>(srcPtr + ((srcBase + i) << 2));
      store<i32>(dstPtr + ((body + i) << 2), (a >>> bitShift) | (b << inv));
    }
    for (let y = 1; y < H; y++) {
      const k = y * wpr;
      const a: u32 = load<i32>(srcPtr + ((srcBase + k) << 2));
      store<i32>(dstPtr + ((body + k) << 2), a << inv);
    }
  }
}

/** 汎用(stride任意)の正確なタップ構築(index.ts::buildTapStridedAsm と
    同一実装、スカラーのまま)。 */
function buildTapStridedAsm(
  dstPtr: i32, dstBase: i32, srcPtr: i32, srcBase: i32,
  H: i32, W: i32, wprIn: i32, dy: i32, dx: i32,
  HOut: i32, WOut: i32, wprOut: i32, sH: i32, sW: i32, pH: i32, pW: i32,
): void {
  for (let i = 0; i < HOut * wprOut; i++) store<i32>(dstPtr + ((dstBase + i) << 2), 0);
  for (let oy = 0; oy < HOut; oy++) {
    const y = oy * sH - pH + dy;
    if (y < 0 || y >= H) continue;
    const rowBase = srcBase + y * wprIn;
    const outRowBase = dstBase + oy * wprOut;
    for (let ox = 0; ox < WOut; ox++) {
      const xx = ox * sW - pW + dx;
      if (xx < 0 || xx >= W) continue;
      const bit = (load<i32>(srcPtr + ((rowBase + (xx >> 5)) << 2)) >>> (xx & 31)) & 1;
      if (bit != 0) {
        const addr = dstPtr + ((outRowBase + (ox >> 5)) << 2);
        store<i32>(addr, load<i32>(addr) | (1 << (ox & 31)));
      }
    }
  }
}

/** bitplane conv2d をバッチ1要素ぶん評価する(全oc)。index.ts::
    evalConv2dBitplaneBatch と幾何・入出力契約は完全同一、木縮約ループのみ
    gate2RangeIntoでv128チャンク化(全長=tapRowsWordsぶんを位置独立に
    4ワード単位で処理できる、という計画書の指摘どおり)。*/
export function evalConv2dBitplaneBatch(
  xPtr: i32,
  planeCPtr: i32, planeDyPtr: i32, planeDxPtr: i32, nPlanes: i32,
  leafWordBasePtr: i32, ttNumPtr: i32,
  tapPoolPtr: i32, bankAPtr: i32, bankBPtr: i32,
  outPtr: i32,
  H: i32, W: i32, HOut: i32, WOut: i32,
  sH: i32, sW: i32, pH: i32, pW: i32,
  OC: i32, nLeaves: i32, nInternal: i32,
  wprIn: i32, wprOut: i32, tapWpr: i32, tailMaskOut: i32, useFastPath: i32,
): void {
  const tapRowsWords = HOut * tapWpr;
  // 平面1枚の行数/ワード数(index.ts / wasmBitplaneConv.ts と同一の導出式)。
  const planeRows = useFastPath != 0 ? H + 2 * pH : HOut;
  const planeStride = planeRows * tapWpr;

  // --- 一意平面をバッチ要素ぶん1回だけ構築(oc横断dedup、スカラー) ---
  for (let u = 0; u < nPlanes; u++) {
    const c = load<i32>(planeCPtr + (u << 2));
    const dy = load<i32>(planeDyPtr + (u << 2));
    const dx = load<i32>(planeDxPtr + (u << 2));
    const cBase = c * H * wprIn;
    const dstBase = u * planeStride;
    if (useFastPath != 0) {
      buildSharedShiftAsm(tapPoolPtr, dstBase, xPtr, cBase, H, wprIn, pH, dx - pW);
    } else {
      buildTapStridedAsm(
        tapPoolPtr, dstBase, xPtr, cBase, H, W, wprIn, dy, dx,
        HOut, WOut, wprOut, sH, sW, pH, pW,
      );
    }
  }

  for (let oc = 0; oc < OC; oc++) {
    const lb = oc * nLeaves;

    // --- 木をボトムアップ評価(ping-pongバンク、割当なし、木縮約はv128) ---
    // レベル0の「src」は平面プール(leafWordBaseで間接参照)なので、
    // 最初の1レベルだけ特別扱いしてbankAへ書き、以降は
    // evalConv2dBatchと同型のping-pong(整列済みバンク)。
    const gOut0 = nLeaves >> 1;
    for (let g = 0; g < gOut0; g++) {
      const tt: u32 = load<u8>(ttNumPtr + (oc * nInternal + g));
      const aBase = load<i32>(leafWordBasePtr + ((lb + (g << 1)) << 2));
      const bBase = load<i32>(leafWordBasePtr + ((lb + (g << 1) + 1) << 2));
      const dBase = g * tapRowsWords;
      gate2RangeInto(bankAPtr, dBase, tapPoolPtr, aBase, tapPoolPtr, bBase, tt, tapRowsWords);
    }

    let srcPtr = bankAPtr;
    let srcLen = gOut0;
    let dstPtr = bankBPtr;
    let gOffset = gOut0;
    while (srcLen > 1) {
      const gOut = srcLen >> 1;
      for (let g = 0; g < gOut; g++) {
        const tt: u32 = load<u8>(ttNumPtr + (oc * nInternal + gOffset + g));
        const aBase = (g << 1) * tapRowsWords;
        const bBase = ((g << 1) + 1) * tapRowsWords;
        const dBase = g * tapRowsWords;
        gate2RangeInto(dstPtr, dBase, srcPtr, aBase, srcPtr, bBase, tt, tapRowsWords);
      }
      gOffset += gOut;
      srcPtr = dstPtr;
      srcLen = gOut;
      dstPtr = srcPtr == bankAPtr ? bankBPtr : bankAPtr;
    }

    // nLeaves==1(depth=0相当)は木自体が無く、leaf 0 がそのまま出力
    // (evalConv2dPacked/evalConv2dBitplaneと同じ扱い: 通常は起きないが
    // 安全のため対応しておく)。
    const finalPtr = nLeaves > 1 ? srcPtr : tapPoolPtr
      + (load<i32>(leafWordBasePtr + (lb << 2)) << 2);

    // --- 出力の各行を書き出す(行末尾にtailMaskOutをAND、スカラー) ---
    const ocBase = outPtr + (oc * HOut * wprOut << 2);
    for (let oy = 0; oy < HOut; oy++) {
      const srcRow = finalPtr + (oy * tapWpr << 2);
      const dstRow = ocBase + (oy * wprOut << 2);
      for (let j = 0; j < wprOut; j++) {
        store<i32>(dstRow + (j << 2), load<i32>(srcRow + (j << 2)));
      }
      const lastAddr = dstRow + ((wprOut - 1) << 2);
      store<i32>(lastAddr, load<i32>(lastAddr) & tailMaskOut);
    }
  }
}

// ============================================================
// conv2d 命令列カーネル(index.ts::evalConvProgBatch の v128 版)。
// 幾何・入出力契約・スクラッチレイアウト・命令エンコードは index.ts と
// 完全同一(2ファイル自己完結コピーの契約)。違いは gate2RangeProg の
// ワードループを4ワード(128位置)単位の v128 演算にした点だけ
// (ゲート演算はワード内のビット位置をまたがないので、チャンク化しても
// 各ワードの値は完全に不変。端数はスカラーへフォールバック)。
// 命令列化そのものの説明は index.ts の同名セクションを参照。
// ============================================================

/** 1命令ぶんの区間演算(gate2の16式、switchをワードループの外に出した版)。
    4ワード単位は v128、端数(<4)はスカラー。全オペランドが同一バッファ
    (scratch)内の絶対ワードオフセットで来る。 */
function gate2RangeProg(
  bufPtr: i32, dstOff: i32, aOff: i32, bOff: i32, tt: u32, n: i32,
): void {
  const d = bufPtr + (dstOff << 2);
  const ap = bufPtr + (aOff << 2);
  const bp = bufPtr + (bOff << 2);
  const vecEnd = n & ~3;
  switch (tt & 0xf) {
    case 0x0: {
      const zv: v128 = i32x4.splat(0);
      let i = 0;
      for (; i < vecEnd; i += 4) v128.store(d + (i << 2), zv, 0, 4);
      for (; i < n; i++) store<i32>(d + (i << 2), 0);
      break;
    }
    case 0x1: {
      let i = 0;
      for (; i < vecEnd; i += 4) {
        const av: v128 = v128.load(ap + (i << 2), 0, 4);
        const bv: v128 = v128.load(bp + (i << 2), 0, 4);
        v128.store(d + (i << 2), v128.not(v128.or(av, bv)), 0, 4);
      }
      for (; i < n; i++) {
        const a: u32 = load<i32>(ap + (i << 2));
        const b: u32 = load<i32>(bp + (i << 2));
        store<i32>(d + (i << 2), ~(a | b));
      }
      break;
    }
    case 0x2: {
      let i = 0;
      for (; i < vecEnd; i += 4) {
        const av: v128 = v128.load(ap + (i << 2), 0, 4);
        const bv: v128 = v128.load(bp + (i << 2), 0, 4);
        v128.store(d + (i << 2), v128.andnot(av, bv), 0, 4);
      }
      for (; i < n; i++) {
        const a: u32 = load<i32>(ap + (i << 2));
        const b: u32 = load<i32>(bp + (i << 2));
        store<i32>(d + (i << 2), a & ~b);
      }
      break;
    }
    case 0x3: {
      let i = 0;
      for (; i < vecEnd; i += 4) {
        const av: v128 = v128.load(ap + (i << 2), 0, 4);
        const bv: v128 = v128.load(bp + (i << 2), 0, 4);
        v128.store(d + (i << 2), v128.not(bv), 0, 4);
      }
      for (; i < n; i++) {
        const a: u32 = load<i32>(ap + (i << 2));
        const b: u32 = load<i32>(bp + (i << 2));
        store<i32>(d + (i << 2), ~b);
      }
      break;
    }
    case 0x4: {
      let i = 0;
      for (; i < vecEnd; i += 4) {
        const av: v128 = v128.load(ap + (i << 2), 0, 4);
        const bv: v128 = v128.load(bp + (i << 2), 0, 4);
        v128.store(d + (i << 2), v128.andnot(bv, av), 0, 4);
      }
      for (; i < n; i++) {
        const a: u32 = load<i32>(ap + (i << 2));
        const b: u32 = load<i32>(bp + (i << 2));
        store<i32>(d + (i << 2), ~a & b);
      }
      break;
    }
    case 0x5: {
      let i = 0;
      for (; i < vecEnd; i += 4) {
        const av: v128 = v128.load(ap + (i << 2), 0, 4);
        const bv: v128 = v128.load(bp + (i << 2), 0, 4);
        v128.store(d + (i << 2), v128.not(av), 0, 4);
      }
      for (; i < n; i++) {
        const a: u32 = load<i32>(ap + (i << 2));
        const b: u32 = load<i32>(bp + (i << 2));
        store<i32>(d + (i << 2), ~a);
      }
      break;
    }
    case 0x6: {
      let i = 0;
      for (; i < vecEnd; i += 4) {
        const av: v128 = v128.load(ap + (i << 2), 0, 4);
        const bv: v128 = v128.load(bp + (i << 2), 0, 4);
        v128.store(d + (i << 2), v128.xor(av, bv), 0, 4);
      }
      for (; i < n; i++) {
        const a: u32 = load<i32>(ap + (i << 2));
        const b: u32 = load<i32>(bp + (i << 2));
        store<i32>(d + (i << 2), a ^ b);
      }
      break;
    }
    case 0x7: {
      let i = 0;
      for (; i < vecEnd; i += 4) {
        const av: v128 = v128.load(ap + (i << 2), 0, 4);
        const bv: v128 = v128.load(bp + (i << 2), 0, 4);
        v128.store(d + (i << 2), v128.not(v128.and(av, bv)), 0, 4);
      }
      for (; i < n; i++) {
        const a: u32 = load<i32>(ap + (i << 2));
        const b: u32 = load<i32>(bp + (i << 2));
        store<i32>(d + (i << 2), ~(a & b));
      }
      break;
    }
    case 0x8: {
      let i = 0;
      for (; i < vecEnd; i += 4) {
        const av: v128 = v128.load(ap + (i << 2), 0, 4);
        const bv: v128 = v128.load(bp + (i << 2), 0, 4);
        v128.store(d + (i << 2), v128.and(av, bv), 0, 4);
      }
      for (; i < n; i++) {
        const a: u32 = load<i32>(ap + (i << 2));
        const b: u32 = load<i32>(bp + (i << 2));
        store<i32>(d + (i << 2), a & b);
      }
      break;
    }
    case 0x9: {
      let i = 0;
      for (; i < vecEnd; i += 4) {
        const av: v128 = v128.load(ap + (i << 2), 0, 4);
        const bv: v128 = v128.load(bp + (i << 2), 0, 4);
        v128.store(d + (i << 2), v128.not(v128.xor(av, bv)), 0, 4);
      }
      for (; i < n; i++) {
        const a: u32 = load<i32>(ap + (i << 2));
        const b: u32 = load<i32>(bp + (i << 2));
        store<i32>(d + (i << 2), ~(a ^ b));
      }
      break;
    }
    case 0xa: {
      let i = 0;
      for (; i < vecEnd; i += 4) {
        const av: v128 = v128.load(ap + (i << 2), 0, 4);
        const bv: v128 = v128.load(bp + (i << 2), 0, 4);
        v128.store(d + (i << 2), av, 0, 4);
      }
      for (; i < n; i++) {
        const a: u32 = load<i32>(ap + (i << 2));
        const b: u32 = load<i32>(bp + (i << 2));
        store<i32>(d + (i << 2), a);
      }
      break;
    }
    case 0xb: {
      let i = 0;
      for (; i < vecEnd; i += 4) {
        const av: v128 = v128.load(ap + (i << 2), 0, 4);
        const bv: v128 = v128.load(bp + (i << 2), 0, 4);
        v128.store(d + (i << 2), v128.or(av, v128.not(bv)), 0, 4);
      }
      for (; i < n; i++) {
        const a: u32 = load<i32>(ap + (i << 2));
        const b: u32 = load<i32>(bp + (i << 2));
        store<i32>(d + (i << 2), a | ~b);
      }
      break;
    }
    case 0xc: {
      let i = 0;
      for (; i < vecEnd; i += 4) {
        const av: v128 = v128.load(ap + (i << 2), 0, 4);
        const bv: v128 = v128.load(bp + (i << 2), 0, 4);
        v128.store(d + (i << 2), bv, 0, 4);
      }
      for (; i < n; i++) {
        const a: u32 = load<i32>(ap + (i << 2));
        const b: u32 = load<i32>(bp + (i << 2));
        store<i32>(d + (i << 2), b);
      }
      break;
    }
    case 0xd: {
      let i = 0;
      for (; i < vecEnd; i += 4) {
        const av: v128 = v128.load(ap + (i << 2), 0, 4);
        const bv: v128 = v128.load(bp + (i << 2), 0, 4);
        v128.store(d + (i << 2), v128.or(v128.not(av), bv), 0, 4);
      }
      for (; i < n; i++) {
        const a: u32 = load<i32>(ap + (i << 2));
        const b: u32 = load<i32>(bp + (i << 2));
        store<i32>(d + (i << 2), ~a | b);
      }
      break;
    }
    case 0xe: {
      let i = 0;
      for (; i < vecEnd; i += 4) {
        const av: v128 = v128.load(ap + (i << 2), 0, 4);
        const bv: v128 = v128.load(bp + (i << 2), 0, 4);
        v128.store(d + (i << 2), v128.or(av, bv), 0, 4);
      }
      for (; i < n; i++) {
        const a: u32 = load<i32>(ap + (i << 2));
        const b: u32 = load<i32>(bp + (i << 2));
        store<i32>(d + (i << 2), a | b);
      }
      break;
    }
    default: {
      const zv: v128 = i32x4.splat(-1);
      let i = 0;
      for (; i < vecEnd; i += 4) v128.store(d + (i << 2), zv, 0, 4);
      for (; i < n; i++) store<i32>(d + (i << 2), 0xffffffff);
      break;
    }
  }
}

/** 出力書き出しスケジュールの1バケット分(命令 bucket-1 の直後、bucket=0 は
    プログラム実行前)を出力平面へ書く。行末尾に tailMaskOut を AND する
    (bitplane.ts と同じ契約: 中間の端数ビットは実位置の値に影響しない)。 */
function writeProgBucket(
  bucket: i32, writeStartPtr: i32, writeOcPtr: i32, ocSrcOffPtr: i32,
  scratchPtr: i32, outPtr: i32,
  HOut: i32, wprOut: i32, tapWpr: i32, tailMaskOut: i32,
): void {
  // tapWpr === wprOut(高速路は WOut===W より wprIn===wprOut、strided路は
  // tapWpr がそのまま wprOut)なので、oc の出力平面は HOut*wprOut 語の連続コピー
  // 1本 + 各行末語へのマスクで書ける。連続化により本体を v128 化できる。
  // tapWpr は ABI 互換のため引数に残す(値は使わない)。
  const s0 = load<i32>(writeStartPtr + (bucket << 2));
  const s1 = load<i32>(writeStartPtr + ((bucket + 1) << 2));
  const planeWords = HOut * wprOut;
  for (let i = s0; i < s1; i++) {
    const oc = load<i32>(writeOcPtr + (i << 2));
    const srcOff = load<i32>(ocSrcOffPtr + (oc << 2));
    const ocBase = outPtr + (oc * planeWords << 2);
    const srcBase = scratchPtr + (srcOff << 2);
    let k = 0;
    for (; k + 4 <= planeWords; k += 4) {
      v128.store(ocBase + (k << 2), v128.load(srcBase + (k << 2), 0, 4), 0, 4);
    }
    for (; k < planeWords; k++) {
      store<i32>(ocBase + (k << 2), load<i32>(srcBase + (k << 2)));
    }
    for (let oy = 0; oy < HOut; oy++) {
      const lastAddr = ocBase + ((oy * wprOut + wprOut - 1) << 2);
      store<i32>(lastAddr, load<i32>(lastAddr) & tailMaskOut);
    }
  }
}

/** conv2d 命令列をバッチ1要素ぶん評価する(全oc)。幾何(タップ平面プールの
    構築)は evalConv2dBitplaneBatch と完全同一で、木のping-pong評価だけが
    命令列の逐次実行に置き換わる。 */
export function evalConvProgBatch(
  xPtr: i32,
  planeCPtr: i32, planeDyPtr: i32, planeDxPtr: i32, nPlanes: i32,
  instrPtr: i32, nInstr: i32,
  writeStartPtr: i32, writeOcPtr: i32, ocSrcOffPtr: i32,
  scratchPtr: i32, outPtr: i32,
  H: i32, W: i32, HOut: i32, WOut: i32,
  sH: i32, sW: i32, pH: i32, pW: i32,
  wprIn: i32, wprOut: i32, tapWpr: i32, tailMaskOut: i32, useFastPath: i32,
  poolBase: i32, planeStride: i32,
): void {
  const tapRowsWords = HOut * tapWpr;

  // --- 一意平面をバッチ要素ぶん1回だけ構築(oc横断dedup) ---
  for (let u = 0; u < nPlanes; u++) {
    const c = load<i32>(planeCPtr + (u << 2));
    const dy = load<i32>(planeDyPtr + (u << 2));
    const dx = load<i32>(planeDxPtr + (u << 2));
    const cBase = c * H * wprIn;
    const dstBase = poolBase + u * planeStride;
    if (useFastPath != 0) {
      buildSharedShiftAsm(scratchPtr, dstBase, xPtr, cBase, H, wprIn, pH, dx - pW);
    } else {
      buildTapStridedAsm(
        scratchPtr, dstBase, xPtr, cBase, H, W, wprIn, dy, dx,
        HOut, WOut, wprOut, sH, sW, pH, pW,
      );
    }
  }

  // --- 命令列を逐次実行し、各命令の直後に「出力がその命令のoc」を書き出す ---
  writeProgBucket(
    0, writeStartPtr, writeOcPtr, ocSrcOffPtr, scratchPtr, outPtr,
    HOut, wprOut, tapWpr, tailMaskOut,
  );
  for (let j = 0; j < nInstr; j++) {
    const o = instrPtr + (j << 4); // 4語/命令
    const tt: u32 = load<i32>(o);
    gate2RangeProg(
      scratchPtr, load<i32>(o + 12), load<i32>(o + 4), load<i32>(o + 8), tt, tapRowsWords,
    );
    if (load<i32>(writeStartPtr + ((j + 2) << 2)) > load<i32>(writeStartPtr + ((j + 1) << 2))) {
      writeProgBucket(
        j + 1, writeStartPtr, writeOcPtr, ocSrcOffPtr, scratchPtr, outPtr,
        HOut, wprOut, tapWpr, tailMaskOut,
      );
    }
  }
}

// ============================================================
// グラフ入力パック / or_pool2d のWASMカーネル(index.ts の同名関数の
// v128 版)。設計・シャード契約・意味論の説明は index.ts 側を参照
// (2ファイル自己完結コピーの契約)。
//
// 違いは packPlanesAsm のみ: 32バイト(=出力1ワード)を
//   i8x16.ne(v, 0) → i8x16.bitmask  (16バイト→16ビット)
// の2組で取り出す。bitmask はレーン0が bit0 に対応するため、bitplane.ts の
// LSB-first 契約とそのまま一致する。端数(n<32)はスカラーへフォールバック。
// orPool2x2Asm は compressEven32 のシフトカスケードが v128 化しても語数
// (wprOut は実モデルで1〜2)に対して割に合わないためスカラーのまま。
// ============================================================

export function packPlanesAsm(
  inPtr: i32, outPtr: i32, rowFirst: i32, rowCount: i32, W: i32, wpr: i32,
): void {
  const zero: v128 = i8x16.splat(0);
  const rEnd = rowFirst + rowCount;
  for (let r = rowFirst; r < rEnd; r++) {
    const src = inPtr + r * W;
    const dst = outPtr + ((r * wpr) << 2);
    for (let j = 0; j < wpr; j++) {
      const base = src + (j << 5);
      let n = W - (j << 5);
      if (n > 32) n = 32;
      let acc: u32 = 0;
      if (n == 32) {
        const a: v128 = v128.load(base, 0, 1);
        const b: v128 = v128.load(base + 16, 0, 1);
        acc = (<u32>i8x16.bitmask(i8x16.ne(a, zero)))
          | ((<u32>i8x16.bitmask(i8x16.ne(b, zero))) << 16);
      } else if (n >= 16) {
        const a: v128 = v128.load(base, 0, 1);
        acc = <u32>i8x16.bitmask(i8x16.ne(a, zero));
        for (let i = 16; i < n; i++) {
          if (load<u8>(base + i) != 0) acc |= (<u32>1) << i;
        }
      } else {
        for (let i = 0; i < n; i++) {
          if (load<u8>(base + i) != 0) acc |= (<u32>1) << i;
        }
      }
      store<i32>(dst + (j << 2), acc);
    }
  }
}

/** 32bit中の偶数位置ビット(0,2,...,30)を下位16bitへ圧縮する
    (bitplane.ts::compressEven32Shift / index.ts::compressEven32 と同一)。 */
function compressEven32(v: u32): u32 {
  let x = v & 0x55555555;
  x = (x | (x >>> 1)) & 0x33333333;
  x = (x | (x >>> 2)) & 0x0f0f0f0f;
  x = (x | (x >>> 4)) & 0x00ff00ff;
  x = (x | (x >>> 8)) & 0x0000ffff;
  return x;
}

export function orPool2x2Asm(
  inPtr: i32, outPtr: i32, planeFirst: i32, planeCount: i32,
  H: i32, wpr: i32, HOut: i32, wprOut: i32, maskOut: i32,
): void {
  const inPlane = H * wpr;
  const outPlane = HOut * wprOut;
  const pEnd = planeFirst + planeCount;
  for (let p = planeFirst; p < pEnd; p++) {
    const ib = inPtr + ((p * inPlane) << 2);
    const ob = outPtr + ((p * outPlane) << 2);
    for (let oy = 0; oy < HOut; oy++) {
      const rowA = ib + (((oy << 1) * wpr) << 2);
      const rowB = ib + ((((oy << 1) + 1) * wpr) << 2);
      const dstRow = ob + ((oy * wprOut) << 2);
      for (let wo = 0; wo < wprOut; wo++) {
        const jl = wo << 1;
        const jh = jl + 1;
        const tl: u32 = jl < wpr
          ? <u32>(load<i32>(rowA + (jl << 2)) | load<i32>(rowB + (jl << 2))) : 0;
        const th: u32 = jh < wpr
          ? <u32>(load<i32>(rowA + (jh << 2)) | load<i32>(rowB + (jh << 2))) : 0;
        const th1: u32 = jh + 1 < wpr
          ? <u32>(load<i32>(rowA + ((jh + 1) << 2)) | load<i32>(rowB + ((jh + 1) << 2))) : 0;
        const lo: u32 = tl | (tl >>> 1) | (th << 31);
        const hi: u32 = jh < wpr ? (th | (th >>> 1) | (th1 << 31)) : 0;
        store<i32>(dstRow + (wo << 2), compressEven32(lo) | (compressEven32(hi) << 16));
      }
      const lastAddr = dstRow + ((wprOut - 1) << 2);
      store<i32>(lastAddr, load<i32>(lastAddr) & maskOut);
    }
  }
}

// ============================================================
// 画像前処理 + 入力パックの融合カーネル(index.ts の同名関数の v128 版)。
// 設計・記述子の意味・シャード契約・数値経路の説明は index.ts 側を参照
// (2ファイル自己完結コピーの契約。ABI・出力ビットは完全同一)。
//
// v128 化の要点:
//   - thermometer(単調 → k >= k0 の1閾値へ畳んである)は **バイトのまま**
//     i8x16.ge_u + i8x16.bitmask で16画素→16ビットを一撃で作る(全チャネルの
//     過半数がこれ)。非単調(LUT参照)だけスカラーのまま。
//   - edge/curv/chroma/彩度は D = Σ w_i I(p_i) の符号付き整数比較だが、
//     I(k) は最大 2^31、curv の D は 34bit まで伸びるので i32x4 では表せない。
//     そこで I = 4Q + r(Q = I>>>2 ∈ [0,2^29], r = I&3)と分解し
//         D = 4G + ρ,  G = Σ w_i Q_i,  ρ = Σ w_i r_i
//     と書き換える。G は edge で |G| <= 2^29、curv でも |G| <= 2^30 なので
//     i32 に収まり、比較は
//         D >= cut ⟺ G >= ceil((cut-ρ)/4) = (cut-ρ+3) >> 2   (算術シフト)
//         D <= cut ⟺ G <= floor((cut-ρ)/4) = (cut-ρ) >> 2
//     と **i32 だけで厳密に**表せる(ρ は edge で ±3、curv で ±6)。
//     cut-ρ+3 が i32 に収まらない極端な閾値(|t| がほぼ1以上)のときだけ
//     チャネル単位でスカラー(i64)経路へ落ちる(出力は同一)。
//   - 端数ワード(n<32)は常にスカラー。
// ============================================================

/** cut-ρ+3 が i32 に収まる安全域(ρ は最大±6)。 */
const PREPROC_CUT_SIMD_LIMIT: i64 = 2147483632;

/** 記述子の cut(64bit二の補数)の下位語をゼロ拡張するためのマスク。 */
const PREPROC_LO_MASK: i64 = 4294967295;

function clampIdx(v: i32, hi: i32): i32 {
  if (v < 0) return 0;
  return v > hi ? hi : v;
}

function qRowPtr(
  qPtr: i32, c: i32, y: i32, H: i32, rowStride: i32, rowPad: i32,
): i32 {
  return qPtr + (((c * H + clampIdx(y, H - 1)) * rowStride + rowPad) << 2);
}

export function preprocQuantAsm(
  rgbaPtr: i32, qvalPtr: i32, qPtr: i32, bytePtr: i32,
  H: i32, W: i32, rowStride: i32, rowPad: i32, byteStride: i32, cimg: i32,
): void {
  for (let c = 0; c < cimg; c++) {
    const bBase = bytePtr + c * byteStride;
    for (let y = 0; y < H; y++) {
      const src = rgbaPtr + ((y * W) << 2) + c;
      const bRow = bBase + y * W;
      const qRow = qPtr + (((c * H + y) * rowStride + rowPad) << 2);
      for (let x = 0; x < W; x++) {
        const k = <i32>load<u8>(src + (x << 2));
        store<u8>(bRow + x, <u8>k);
        store<i32>(qRow + (x << 2), load<i32>(qvalPtr + (k << 2)));
      }
      const q0 = load<i32>(qRow);
      const q1 = load<i32>(qRow + ((W - 1) << 2));
      for (let x = 1; x <= rowPad; x++) {
        store<i32>(qRow - (x << 2), q0);
        store<i32>(qRow + ((W - 1 + x) << 2), q1);
      }
    }
  }
}

function packThermoStepRows(
  srcPtr: i32, outPtr: i32, H: i32, W: i32, wpr: i32, tail: i32, k0: i32,
): void {
  const kv: v128 = i8x16.splat(<i8>k0);
  const useSimd = k0 >= 0 && k0 <= 255;
  for (let y = 0; y < H; y++) {
    const row = srcPtr + y * W;
    const dst = outPtr + ((y * wpr) << 2);
    for (let j = 0; j < wpr; j++) {
      const base = row + (j << 5);
      let n = W - (j << 5);
      if (n > 32) n = 32;
      let acc: u32 = 0;
      if (useSimd && n == 32) {
        const a: v128 = v128.load(base, 0, 1);
        const b: v128 = v128.load(base + 16, 0, 1);
        acc = (<u32>i8x16.bitmask(i8x16.ge_u(a, kv)))
          | ((<u32>i8x16.bitmask(i8x16.ge_u(b, kv))) << 16);
      } else {
        for (let i = 0; i < n; i++) {
          if (<i32>load<u8>(base + i) >= k0) acc |= (<u32>1) << i;
        }
      }
      if (j == wpr - 1) acc &= <u32>tail;
      store<i32>(dst + (j << 2), acc);
    }
  }
}

function packThermoLutRows(
  srcPtr: i32, outPtr: i32, H: i32, W: i32, wpr: i32, tail: i32, lutPtr: i32,
): void {
  for (let y = 0; y < H; y++) {
    const row = srcPtr + y * W;
    const dst = outPtr + ((y * wpr) << 2);
    for (let j = 0; j < wpr; j++) {
      const base = row + (j << 5);
      let n = W - (j << 5);
      if (n > 32) n = 32;
      let acc: u32 = 0;
      for (let i = 0; i < n; i++) {
        const k = <i32>load<u8>(base + i);
        const w: u32 = <u32>load<i32>(lutPtr + ((k >>> 5) << 2));
        acc |= ((w >>> (k & 31)) & 1) << i;
      }
      if (j == wpr - 1) acc &= <u32>tail;
      store<i32>(dst + (j << 2), acc);
    }
  }
}

/** 端数ワード/範囲外 cut 用のスカラー2項差(index.ts と逐語同一の式)。 */
function diff2Word(
  paRow: i32, pbRow: i32, x0: i32, n: i32, cut: i64, dirLe: bool,
): u32 {
  let acc: u32 = 0;
  for (let i = 0; i < n; i++) {
    const o = (x0 + i) << 2;
    const d: i64 = <i64>load<u32>(paRow + o) - <i64>load<u32>(pbRow + o);
    const bit = dirLe ? (d <= cut) : (d >= cut);
    if (bit) acc |= (<u32>1) << i;
  }
  return acc;
}

function packDiff2Rows(
  paRow: i32, pbRow: i32, dst: i32, W: i32, wpr: i32, tail: i32,
  cut: i64, dirLe: bool,
): void {
  const useSimd = cut >= -PREPROC_CUT_SIMD_LIMIT && cut <= PREPROC_CUT_SIMD_LIMIT;
  const cv: v128 = i32x4.splat(<i32>cut);
  const m3: v128 = i32x4.splat(3);
  for (let j = 0; j < wpr; j++) {
    const x0 = j << 5;
    let n = W - x0;
    if (n > 32) n = 32;
    let acc: u32 = 0;
    if (useSimd && n == 32) {
      for (let s = 0; s < 32; s += 4) {
        const o = (x0 + s) << 2;
        const va: v128 = v128.load(paRow + o, 0, 1);
        const vb: v128 = v128.load(pbRow + o, 0, 1);
        const g: v128 = i32x4.sub(i32x4.shr_u(va, 2), i32x4.shr_u(vb, 2));
        const r: v128 = i32x4.sub(v128.and(va, m3), v128.and(vb, m3));
        const t: v128 = i32x4.sub(cv, r);
        const m: v128 = dirLe
          ? i32x4.le_s(g, i32x4.shr_s(t, 2))
          : i32x4.ge_s(g, i32x4.shr_s(i32x4.add(t, m3), 2));
        acc |= (<u32>i32x4.bitmask(m)) << s;
      }
    } else {
      acc = diff2Word(paRow, pbRow, x0, n, cut, dirLe);
    }
    if (j == wpr - 1) acc &= <u32>tail;
    store<i32>(dst + (j << 2), acc);
  }
}

function curvWord(
  ppRow: i32, pmRow: i32, pnRow: i32, x0: i32, n: i32, cut: i64, dirLe: bool,
): u32 {
  let acc: u32 = 0;
  for (let i = 0; i < n; i++) {
    const o = (x0 + i) << 2;
    const d: i64 = <i64>load<u32>(pnRow + o)
      - (<i64>load<u32>(pmRow + o) << 1)
      + <i64>load<u32>(ppRow + o);
    const bit = dirLe ? (d <= cut) : (d >= cut);
    if (bit) acc |= (<u32>1) << i;
  }
  return acc;
}

function packCurvRows(
  ppRow: i32, pmRow: i32, pnRow: i32, dst: i32, W: i32, wpr: i32, tail: i32,
  cut: i64, dirLe: bool,
): void {
  const useSimd = cut >= -PREPROC_CUT_SIMD_LIMIT && cut <= PREPROC_CUT_SIMD_LIMIT;
  const cv: v128 = i32x4.splat(<i32>cut);
  const m3: v128 = i32x4.splat(3);
  for (let j = 0; j < wpr; j++) {
    const x0 = j << 5;
    let n = W - x0;
    if (n > 32) n = 32;
    let acc: u32 = 0;
    if (useSimd && n == 32) {
      for (let s = 0; s < 32; s += 4) {
        const o = (x0 + s) << 2;
        const vp: v128 = v128.load(ppRow + o, 0, 1);
        const vm: v128 = v128.load(pmRow + o, 0, 1);
        const vn: v128 = v128.load(pnRow + o, 0, 1);
        const g: v128 = i32x4.add(
          i32x4.sub(i32x4.shr_u(vn, 2), i32x4.shl(i32x4.shr_u(vm, 2), 1)),
          i32x4.shr_u(vp, 2),
        );
        const r: v128 = i32x4.add(
          i32x4.sub(v128.and(vn, m3), i32x4.shl(v128.and(vm, m3), 1)),
          v128.and(vp, m3),
        );
        const t: v128 = i32x4.sub(cv, r);
        const m: v128 = dirLe
          ? i32x4.le_s(g, i32x4.shr_s(t, 2))
          : i32x4.ge_s(g, i32x4.shr_s(i32x4.add(t, m3), 2));
        acc |= (<u32>i32x4.bitmask(m)) << s;
      }
    } else {
      acc = curvWord(ppRow, pmRow, pnRow, x0, n, cut, dirLe);
    }
    if (j == wpr - 1) acc &= <u32>tail;
    store<i32>(dst + (j << 2), acc);
  }
}

function packSatRows(
  qPtr: i32, dst: i32, y: i32, H: i32, W: i32, wpr: i32, tail: i32,
  rowStride: i32, rowPad: i32, cimg: i32, cut: i64, dirLe: bool,
): void {
  const row0 = qRowPtr(qPtr, 0, y, H, rowStride, rowPad);
  const planeBytes = (H * rowStride) << 2;
  const useSimd = cut >= -PREPROC_CUT_SIMD_LIMIT && cut <= PREPROC_CUT_SIMD_LIMIT;
  const cv: v128 = i32x4.splat(<i32>cut);
  const m3: v128 = i32x4.splat(3);
  for (let j = 0; j < wpr; j++) {
    const x0 = j << 5;
    let n = W - x0;
    if (n > 32) n = 32;
    let acc: u32 = 0;
    if (useSimd && n == 32) {
      for (let s = 0; s < 32; s += 4) {
        const o = (x0 + s) << 2;
        let vmax: v128 = v128.load(row0 + o, 0, 1);
        let vmin: v128 = vmax;
        for (let c = 1; c < cimg; c++) {
          const v: v128 = v128.load(row0 + c * planeBytes + o, 0, 1);
          vmax = i32x4.max_u(vmax, v);
          vmin = i32x4.min_u(vmin, v);
        }
        const g: v128 = i32x4.sub(i32x4.shr_u(vmax, 2), i32x4.shr_u(vmin, 2));
        const r: v128 = i32x4.sub(v128.and(vmax, m3), v128.and(vmin, m3));
        const t: v128 = i32x4.sub(cv, r);
        const m: v128 = dirLe
          ? i32x4.le_s(g, i32x4.shr_s(t, 2))
          : i32x4.ge_s(g, i32x4.shr_s(i32x4.add(t, m3), 2));
        acc |= (<u32>i32x4.bitmask(m)) << s;
      }
    } else {
      for (let i = 0; i < n; i++) {
        const o = (x0 + i) << 2;
        let mx: u32 = 0;
        let mn: u32 = 0xffffffff;
        for (let c = 0; c < cimg; c++) {
          const v: u32 = load<u32>(row0 + c * planeBytes + o);
          if (v > mx) mx = v;
          if (v < mn) mn = v;
        }
        const d: i64 = <i64>mx - <i64>mn;
        const bit = dirLe ? (d <= cut) : (d >= cut);
        if (bit) acc |= (<u32>1) << i;
      }
    }
    if (j == wpr - 1) acc &= <u32>tail;
    store<i32>(dst + (j << 2), acc);
  }
}

export function preprocPackAsm(
  qPtr: i32, bytePtr: i32, descPtr: i32, outPtr: i32,
  chFirst: i32, chCount: i32,
  H: i32, W: i32, wpr: i32, tail: i32,
  rowStride: i32, rowPad: i32, byteStride: i32, cimg: i32,
): void {
  const chEnd = chFirst + chCount;
  for (let ch = chFirst; ch < chEnd; ch++) {
    const d = descPtr + (ch << 6);
    const op = load<i32>(d);
    const ca = load<i32>(d + 4);
    const cb = load<i32>(d + 8);
    const dy = load<i32>(d + 12);
    const dx = load<i32>(d + 16);
    // 64bit二の補数の2語から cut を復元する。下位語は **必ずゼロ拡張**すること
    // (符号拡張してしまうと cut >= 2^31、すなわち |閾値| >= 1 のときだけ
    //  負の巨大値になり、比較が全画素で反転する。実モデルの閾値は 0.4 以下で
    //  この分岐に入らないため、cpuPreproc.test.ts の人工ケースで固定している)。
    const cut: i64 = ((<i64>load<i32>(d + 24)) << 32)
      | ((<i64>load<i32>(d + 20)) & PREPROC_LO_MASK);
    const dirLe: bool = load<i32>(d + 28) != 0;
    const outCh = outPtr + (((ch * H) * wpr) << 2);
    if (op == 5) {
      packThermoStepRows(bytePtr + ca * byteStride, outCh, H, W, wpr, tail, cb);
      continue;
    }
    if (op == 0) {
      packThermoLutRows(bytePtr + ca * byteStride, outCh, H, W, wpr, tail, d + 32);
      continue;
    }
    for (let y = 0; y < H; y++) {
      const dst = outCh + ((y * wpr) << 2);
      if (op == 1) {
        packDiff2Rows(
          qRowPtr(qPtr, ca, y + dy, H, rowStride, rowPad) + (dx << 2),
          qRowPtr(qPtr, ca, y, H, rowStride, rowPad),
          dst, W, wpr, tail, cut, dirLe,
        );
      } else if (op == 2) {
        packCurvRows(
          qRowPtr(qPtr, ca, y + dy, H, rowStride, rowPad) + (dx << 2),
          qRowPtr(qPtr, ca, y, H, rowStride, rowPad),
          qRowPtr(qPtr, ca, y - dy, H, rowStride, rowPad) - (dx << 2),
          dst, W, wpr, tail, cut, dirLe,
        );
      } else if (op == 3) {
        packDiff2Rows(
          qRowPtr(qPtr, ca, y, H, rowStride, rowPad),
          qRowPtr(qPtr, cb, y, H, rowStride, rowPad),
          dst, W, wpr, tail, cut, dirLe,
        );
      } else {
        packSatRows(
          qPtr, dst, y, H, W, wpr, tail, rowStride, rowPad, cimg, cut, dirLe,
        );
      }
    }
  }
}

// ---------------------------------------------------------------------------
// lut_flat(縮約済み dense 平坦ネットリスト)→ group_sum の融合カーネル。
// src/engines/lutFlat.ts::evalLutFlatGroupSum(JS)とビット完全一致
// (runtime/c/src/mdl_kernels.c の MDL_LUT_FLAT_BODY / _PACKED_BODY の写し)。
// 表(tt/a/b/o/inIdx)は wasmLutFlat.ts が params ごとに1度だけヒープへ載せる。
// 信号番号は lutFlat.ts::compile が「参照される入力だけ」に詰め直した後のもの
// (信号 1+u ← 入力 inIdx[u])。index.ts と index-simd.ts は同一コード。
//   tt: u8[nG] / a,b: i32[nG] / o: i32[outW](2*信号+極性)/ inIdx: i32[nIn]
//   x: u8[batch*inW](0/1)/ scores: i32[batch*k]

/** 1枚ずつ(sig: u8[1+nIn+nG])。 */
export function lutFlatGsAsm(
  tt: i32, a: i32, b: i32, o: i32, inIdx: i32, nIn: i32, nG: i32, outW: i32,
  k: i32, x: i32, inW: i32, batch: i32, sig: i32, scores: i32,
): void {
  const base = sig + 1 + nIn;
  const gsz = outW / k;
  store<u8>(sig, 0);
  for (let s = 0; s < batch; s++) {
    const xo = x + s * inW;
    for (let u = 0; u < nIn; u++) {
      store<u8>(sig + 1 + u, load<u8>(xo + load<i32>(inIdx + (u << 2))));
    }
    for (let g = 0; g < nG; g++) {
      const addr = (u32(load<u8>(sig + load<i32>(b + (g << 2)))) << 1)
        | u32(load<u8>(sig + load<i32>(a + (g << 2))));
      store<u8>(base + g, u8((u32(load<u8>(tt + g)) >>> addr) & 1));
    }
    for (let gi = 0; gi < k; gi++) {
      let cnt = 0;
      const i0 = gi * gsz;
      for (let i = i0; i < i0 + gsz; i++) {
        const r = load<i32>(o + (i << 2));
        cnt += i32(load<u8>(sig + (r >> 1))) ^ (r & 1);
      }
      store<i32>(scores + ((s * k + gi) << 2), cnt);
    }
  }
}

/** バッチ軸パック(sig: u32[(1+nIn+nG)*W]、W=ceil(batch/32)、
    planes: u32[ceil(log2(gsz+1))] の作業域)。group_sum は語ごとの垂直カウンタ。 */
export function lutFlatPackedGsAsm(
  tt: i32, a: i32, b: i32, o: i32, inIdx: i32, nIn: i32, nG: i32, outW: i32,
  k: i32, x: i32, inW: i32, batch: i32, sig: i32, planes: i32, scores: i32,
): void {
  const W = (batch + 31) >>> 5;
  memory.fill(sig, 0, ((1 + nIn) * W) << 2);
  for (let s = 0; s < batch; s++) {
    const xo = x + s * inW;
    const bit = s & 31;
    let p = sig + ((W + (s >>> 5)) << 2);
    for (let u = 0; u < nIn; u++, p += W << 2) {
      store<u32>(p, load<u32>(p) | (u32(load<u8>(xo + load<i32>(inIdx + (u << 2)))) << bit));
    }
  }
  const gbase = sig + (((nIn + 1) * W) << 2);
  for (let g = 0; g < nG; g++) {
    const t = u32(load<u8>(tt + g));
    const t0 = 0 - (t & 1), t1 = 0 - ((t >> 1) & 1);
    const t2 = 0 - ((t >> 2) & 1), t3 = 0 - ((t >> 3) & 1);
    const pa = sig + ((load<i32>(a + (g << 2)) * W) << 2);
    const pb = sig + ((load<i32>(b + (g << 2)) * W) << 2);
    const po = gbase + ((g * W) << 2);
    for (let w = 0; w < W; w++) {
      const av = load<u32>(pa + (w << 2)), bv = load<u32>(pb + (w << 2));
      const uu = t0 ^ (av & (t0 ^ t1)), vv = t2 ^ (av & (t2 ^ t3));
      store<u32>(po + (w << 2), uu ^ (bv & (uu ^ vv)));
    }
  }
  const gsz = outW / k;
  const nPlanes = 32 - clz<i32>(gsz);
  for (let w = 0; w < W; w++) {
    const s0 = w << 5;
    const n = min(32, batch - s0);
    for (let gi = 0; gi < k; gi++) {
      memory.fill(planes, 0, nPlanes << 2);
      const i0 = gi * gsz;
      for (let i = i0; i < i0 + gsz; i++) {
        const r = load<i32>(o + (i << 2));
        let carry = load<u32>(sig + (((r >> 1) * W + w) << 2)) ^ (0 - u32(r & 1));
        for (let q = 0; carry != 0 && q < nPlanes; q++) {
          const pq = planes + (q << 2);
          const pv = load<u32>(pq);
          store<u32>(pq, pv ^ carry);
          carry = pv & carry;
        }
      }
      for (let lane = 0; lane < n; lane++) {
        let cnt = 0;
        for (let q = 0; q < nPlanes; q++) {
          cnt |= i32((load<u32>(planes + (q << 2)) >>> lane) & 1) << q;
        }
        store<i32>(scores + (((s0 + lane) * k + gi) << 2), cnt);
      }
    }
  }
}
