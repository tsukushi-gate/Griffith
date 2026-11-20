// DLGN conv2d ゲート木カーネルの WASM(AssemblyScript)実装。
// tools/webbench/src/engines/packedConv.ts の evalConv2dPacked と
// ビット単位で同一の計算をする(重み共有木プログラム × 空間位置32ビット
// パック)。呼び出し側(src/engines/wasmConv.ts)は生のWASM線形メモリへ
// 直接 read/write する低レベル契約: バッファは全て呼び出し側が alloc() で
// 確保し、ポインタ(バイトオフセット)を渡す。GCランタイムは使わない
// (--runtime stub 相当、alloc は単純バンプアロケータ)。
//
// 契約(evalConv2dPacked との対応):
//   - leafC/leafDy/leafDx: Int32Array[OC*nLeaves] (バイト列、4バイト/要素)
//   - ttNum: Uint8Array[OC*nInternal] (1バイト/要素、値0..15)
//   - x: Int32Array[inW] (このバッチ要素の入力。0/1)
//   - leafPlanes scratch: Int32Array[nLeaves*wordsOut]
//   - bankA/bankB scratch: Int32Array[max(1,nLeaves>>1)*wordsOut]
//   - out: Int32Array[OC*Lout] (0/1、evalConv2dPacked の data と同一並び)

// ---- 単純バンプアロケータ(このモジュールは解放しない。呼び出し側が
// ConvRuntime生存期間だけ持つ固定バッファをここに積む想定) ----
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

// codegen.ts gateExpr / packedConv.ts gate2 と同一の16式(u32ビット並列)。
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

/** conv2d 木プログラムをバッチ1要素ぶん評価する(全oc)。
    packedConv.ts evalConv2dPacked の per-batch本体の忠実移植。 */
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

    // --- leaf(タップ)平面を構築(ゲート木の葉、空間位置パック) ---
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

    // --- 木をボトムアップ評価(ping-pongバンク、割当なし) ---
    let srcPtr = leafPlanesPtr;
    let srcLen = nLeaves;
    let dstPtr = bankAPtr;
    let gOffset = 0;
    while (srcLen > 1) {
      const gOut = srcLen >> 1;
      for (let g = 0; g < gOut; g++) {
        const tt: u32 = load<u8>(ttNumPtr + (oc * nInternal + gOffset + g));
        const aBase = srcPtr + ((g << 1) * wordsOut << 2);
        const bBase = srcPtr + (((g << 1) + 1) * wordsOut << 2);
        const dBase = dstPtr + (g * wordsOut << 2);
        for (let wi = 0; wi < wordsOut; wi++) {
          const a: u32 = load<i32>(aBase + (wi << 2));
          const b: u32 = load<i32>(bBase + (wi << 2));
          store<i32>(dBase + (wi << 2), gate2(tt, a, b));
        }
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
// bitplane conv2d カーネル(bitplane.ts::evalConv2dBitplane の
// WASM移植)。行頭ワード境界のビットプレーン([C][H][wpr]、LSB-first)を
// そのまま入出力する。JS版とのレイアウト上の唯一の違い: 平面の重複排除
// (高速路は (channel,dx)、strided路は (channel,dy,dx) の oc横断dedup)は、
// ジオメトリがバッチ非依存で静的なため呼び出し側(wasmBitplaneConv.ts)が
// 構築時に1回計算し、leafWordBase(各leafが指すプール内のワード基点)として
// 渡す。WASM側は一意平面ぶんだけプールを構築し(Mapを実装しない)、木を
// ping-pongバンクで評価する。行末尾マスクは最終出力の各行にのみ適用する
// (bitplane.tsのコメント参照: ゲート演算はワード内のビット位置を
// またがない純粋な位置ごとの論理演算なので、中間levelの端数ビットは
// 最終的な実位置(x<WOut)の値に一切影響しない)。
// ============================================================

/** dst[dstBase..dstBase+wpr) へ src[srcBase..) を s ビットシフトして書く
    (bitplane.ts::shiftRowInto と同一のワード演算、範囲外は0)。全て
    「ワード単位オフセット」を受け取り、ロード/ストア時にのみ<<2でバイト
    アドレスへ変換する(evalConv2dBatchの既存流儀に合わせる)。 */
function shiftRowIntoAsm(
  dstPtr: i32, dstBase: i32, srcPtr: i32, srcBase: i32, wpr: i32, s: i32,
): void {
  // JSのMath.floor(s/32)と同じfloor除算をi32算術シフトで実現
  // (負数でも x>>5 === floor(x/32)、x&31 === floor-mod)。
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
    (bitplane.ts::buildSharedShiftPlane のWASM移植)。水平シフト量
    s = dx - pW は dy に依存しないので、(channel,dx) ごとに「上下 pH 行の
    ゼロパディングを含む H+2*pH 行ぶんの水平シフト済み平面」を1回だけ作る。
    行 r の内容は入力行 y=r-pH を s ビットシフトしたもの(範囲外はゼロ行)。
    タップ (dy,dx) の出力行 oy が要求する入力行 y=oy-pH+dy は共有平面の
    行 r=oy+dy と一致するため、リーフ参照は「平面先頭 + dy*wpr」で済む。
    本体(内部H行)は入力チャネル平面と同じ連続領域・同じ行ストライド
    なので、行境界を無視した長さ H*wpr の直線ループでシフト合成し、行をまたいで
    混入した語(wordShift=0 なら行末語、-1 なら行頭語)だけを1語ずつ潰す。
    実モデルは wpr=1〜4 しかなく、行ごとのループ立ち上げが語あたりコストを
    支配していたため。bitplane.ts::buildSharedShiftPlane と語単位で同一。 */
function buildSharedShiftAsm(
  dstPtr: i32, dstBase: i32, srcPtr: i32, srcBase: i32,
  H: i32, wpr: i32, pH: i32, s: i32,
): void {
  const n = H * wpr;
  const body = dstBase + pH * wpr;
  // 上下 pH 行のゼロパディング。
  for (let i = dstBase; i < body; i++) store<i32>(dstPtr + (i << 2), 0);
  const tail = body + n;
  for (let i = tail; i < tail + pH * wpr; i++) store<i32>(dstPtr + (i << 2), 0);
  if (s == 0) {
    for (let i = 0; i < n; i++) {
      store<i32>(dstPtr + ((body + i) << 2), load<i32>(srcPtr + ((srcBase + i) << 2)));
    }
    return;
  }
  if (s <= -32 || s >= 32) {
    // 想定外の大シフト(wordShift が -1/0 に収まらない)は行ごとの汎用路へ。
    for (let y = 0; y < H; y++) {
      shiftRowIntoAsm(dstPtr, body + y * wpr, srcPtr, srcBase + y * wpr, wpr, s);
    }
    return;
  }
  const wordShift = s >> 5;            // 0(s>0)か -1(s<0)
  const bitShift = s - wordShift * 32; // (0,32)
  const inv = 32 - bitShift;
  if (wpr == 1) {
    // 1行=1語: 隣接語は常に行外(=0)なので混入自体が起きない。
    if (wordShift == 0) {
      for (let i = 0; i < n; i++) {
        const a: u32 = load<i32>(srcPtr + ((srcBase + i) << 2));
        store<i32>(dstPtr + ((body + i) << 2), a >>> bitShift);
      }
    } else {
      for (let i = 0; i < n; i++) {
        const a: u32 = load<i32>(srcPtr + ((srcBase + i) << 2));
        store<i32>(dstPtr + ((body + i) << 2), a << inv);
      }
    }
    return;
  }
  if (wordShift == 0) {
    const last = n - 1;
    for (let i = 0; i < last; i++) {
      const a: u32 = load<i32>(srcPtr + ((srcBase + i) << 2));
      const b: u32 = load<i32>(srcPtr + ((srcBase + i + 1) << 2));
      store<i32>(dstPtr + ((body + i) << 2), (a >>> bitShift) | (b << inv));
    }
    const al: u32 = load<i32>(srcPtr + ((srcBase + last) << 2));
    store<i32>(dstPtr + ((body + last) << 2), al >>> bitShift);
    for (let y = 0; y < H - 1; y++) {
      const i = y * wpr + (wpr - 1);
      const a: u32 = load<i32>(srcPtr + ((srcBase + i) << 2));
      store<i32>(dstPtr + ((body + i) << 2), a >>> bitShift);
    }
  } else {
    const a0: u32 = load<i32>(srcPtr + (srcBase << 2));
    store<i32>(dstPtr + (body << 2), a0 << inv);
    for (let i = 1; i < n; i++) {
      const a: u32 = load<i32>(srcPtr + ((srcBase + i - 1) << 2));
      const b: u32 = load<i32>(srcPtr + ((srcBase + i) << 2));
      store<i32>(dstPtr + ((body + i) << 2), (a >>> bitShift) | (b << inv));
    }
    for (let y = 1; y < H; y++) {
      const i = y * wpr;
      const a: u32 = load<i32>(srcPtr + ((srcBase + i) << 2));
      store<i32>(dstPtr + ((body + i) << 2), a << inv);
    }
  }
}

/** 汎用(stride任意)の正確なタップ構築(bitplane.ts::buildTapPlaneStrided
    のWASM移植)。1ビットずつのgatherだが常に正しい(遅いパス)。 */
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

/** bitplane conv2d をバッチ1要素ぶん評価する(全oc)。呼び出し側
    (wasmBitplaneConv.ts)が構築時に1回だけ計算する静的ジオメトリ:
      - planeC/planeDy/planeDx[nPlanes]: 一意平面の (c,dy,dx)
        (useFastPath!=0 では (c,dx) で dedup し dy は未使用。
         useFastPath==0 では従来どおり (c,dy,dx) で dedup)
      - leafWordBase[OC*nLeaves]: 各leafが指すプール内の**ワード基点**
        (= 平面ID*planeStride + 行オフセット*tapWpr。行オフセットは
         高速路の dy 由来定数、strided路では常に0)
      - ttNum[OC*nInternal]: 木ゲートの数値化truth table(0..15)
    x: 入力平面(このバッチ要素、[C*H*wprIn]ワード、バイトポインタ)。
    out: 出力平面([OC*HOut*wprOut]ワード、バイトポインタ、行末尾マスク済み)。
    tapPool/bankA/bankB: スクラッチ(呼び出し側が確保、このノードの生存
    期間中固定・再利用)。tapPool は nPlanes*planeStride ワード必要
    (planeStride は下で H,pH,HOut,tapWpr から導出、TS側と同一式)。
    useFastPath!=0 なら共有シフト平面方式、0ならstrided方式。 */
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
  // 平面1枚の行数/ワード数(TS側 wasmBitplaneConv.ts と同一の導出式。
  // 引数を増やさずに済むよう wasm 側でも同じ式で計算する)。
  const planeRows = useFastPath != 0 ? H + 2 * pH : HOut;
  const planeStride = planeRows * tapWpr;

  // --- 一意平面をバッチ要素ぶん1回だけ構築(oc横断dedup) ---
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

    // --- 木をボトムアップ評価(ping-pongバンク、割当なし) ---
    // レベル0の「src」は平面プール(leafWordBaseで間接参照)なので、
    // 最初の1レベルだけ特別扱いしてbankAへ書き、以降は
    // evalConv2dBatchと同型のping-pong(整列済みバンク)。
    const gOut0 = nLeaves >> 1;
    for (let g = 0; g < gOut0; g++) {
      const tt: u32 = load<u8>(ttNumPtr + (oc * nInternal + g));
      const aBase = tapPoolPtr
        + (load<i32>(leafWordBasePtr + ((lb + (g << 1)) << 2)) << 2);
      const bBase = tapPoolPtr
        + (load<i32>(leafWordBasePtr + ((lb + (g << 1) + 1) << 2)) << 2);
      const dBase = bankAPtr + (g * tapRowsWords << 2);
      for (let wi = 0; wi < tapRowsWords; wi++) {
        const a: u32 = load<i32>(aBase + (wi << 2));
        const b: u32 = load<i32>(bBase + (wi << 2));
        store<i32>(dBase + (wi << 2), gate2(tt, a, b));
      }
    }

    let srcPtr = bankAPtr;
    let srcLen = gOut0;
    let dstPtr = bankBPtr;
    let gOffset = gOut0;
    while (srcLen > 1) {
      const gOut = srcLen >> 1;
      for (let g = 0; g < gOut; g++) {
        const tt: u32 = load<u8>(ttNumPtr + (oc * nInternal + gOffset + g));
        const aBase = srcPtr + ((g << 1) * tapRowsWords << 2);
        const bBase = srcPtr + (((g << 1) + 1) * tapRowsWords << 2);
        const dBase = dstPtr + (g * tapRowsWords << 2);
        for (let wi = 0; wi < tapRowsWords; wi++) {
          const a: u32 = load<i32>(aBase + (wi << 2));
          const b: u32 = load<i32>(bBase + (wi << 2));
          store<i32>(dBase + (wi << 2), gate2(tt, a, b));
        }
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

    // --- 出力の各行を書き出す(行末尾にtailMaskOutをAND) ---
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
// conv2d 命令列カーネル(convProgram.ts の縮約結果を実行する)。
//
// evalConv2dBitplaneBatch は「全ocを同じ深さのバランス木として全ゲート評価」
// するが、学習済みの木はパススルー('A'=0xa)・定数ゲート・重複サブ木だらけで、
// lossless なブール恒等式だけで命令数を1/2〜1/15に落とせる(実測 det 85.5%減、
// detface 93.8%減、seg 59.7%減)。TS側(convProgram.ts + bitplane.ts::
// buildConvProgramLayout)が構築時に木→命令列へ縮約し、全参照を
// 「単一スクラッチバッファ内の絶対ワードオフセット」へ解決してから、
// 平坦な Int32 配列としてこのカーネルへ渡す。したがってここには
// 参照種別(定数/タップ/命令)の実行時分岐が一切無い。
//
// スクラッチのレイアウト(ワード単位、TS側 ConvProgramLayout と同一契約):
//   [0, tapRowsWords)              定数0平面
//   [tapRowsWords, 2*tapRowsWords) 定数1平面(全ビット1)
//   [poolBase, regBase)            タップ平面プール(nPlanes*planeStride)
//   [regBase, ...)                 命令の作業平面(nSlots*tapRowsWords)
// 定数平面はTS側が構築時に1回だけ書き込む(このカーネルは読むだけ)。
//
// instr: 命令ごとに4語 [tt, aOff, bOff, dstOff](全て絶対ワードオフセット)。
// writeStart/writeOc: 出力書き出しスケジュール(CSR)。バケット0は命令実行前
// (出力が定数/タップ直結のoc)、バケット j+1 は命令 j の実行直後。命令直後に
// 書き出すことで作業平面の生存区間が閉じ、スロット数が木の作業集合程度に収まる。
// ocSrcOff: oc ごとの出力元絶対ワードオフセット。
//
// 正しさの錨: bitplane.ts::evalConv2dBitplane との整数完全一致
// (conv.test.ts / seg_e2e.test.ts / detdecode.test.ts の ENGINES 行列)と、
// TS側ロード時の全数真理値表セルフチェック(verifyConvProgram)。
// ============================================================

/** 1命令ぶんの区間演算(gate2の16式、switchをワードループの外に出した版)。
    全オペランドが同一バッファ(scratch)内の絶対ワードオフセットで来る。
    dstOff はスロット割当により常にオペランドと別領域(エイリアス無し)。 */
function gate2RangeProg(
  bufPtr: i32, dstOff: i32, aOff: i32, bOff: i32, tt: u32, n: i32,
): void {
  const d = bufPtr + (dstOff << 2);
  const ap = bufPtr + (aOff << 2);
  const bp = bufPtr + (bOff << 2);
  switch (tt & 0xf) {
    case 0x0: {
      for (let i = 0; i < n; i++) store<i32>(d + (i << 2), 0);
      break;
    }
    case 0x1: {
      for (let i = 0; i < n; i++) {
        const a: u32 = load<i32>(ap + (i << 2));
        const b: u32 = load<i32>(bp + (i << 2));
        store<i32>(d + (i << 2), ~(a | b));
      }
      break;
    }
    case 0x2: {
      for (let i = 0; i < n; i++) {
        const a: u32 = load<i32>(ap + (i << 2));
        const b: u32 = load<i32>(bp + (i << 2));
        store<i32>(d + (i << 2), a & ~b);
      }
      break;
    }
    case 0x3: {
      for (let i = 0; i < n; i++) {
        const b: u32 = load<i32>(bp + (i << 2));
        store<i32>(d + (i << 2), ~b);
      }
      break;
    }
    case 0x4: {
      for (let i = 0; i < n; i++) {
        const a: u32 = load<i32>(ap + (i << 2));
        const b: u32 = load<i32>(bp + (i << 2));
        store<i32>(d + (i << 2), ~a & b);
      }
      break;
    }
    case 0x5: {
      for (let i = 0; i < n; i++) {
        const a: u32 = load<i32>(ap + (i << 2));
        store<i32>(d + (i << 2), ~a);
      }
      break;
    }
    case 0x6: {
      for (let i = 0; i < n; i++) {
        const a: u32 = load<i32>(ap + (i << 2));
        const b: u32 = load<i32>(bp + (i << 2));
        store<i32>(d + (i << 2), a ^ b);
      }
      break;
    }
    case 0x7: {
      for (let i = 0; i < n; i++) {
        const a: u32 = load<i32>(ap + (i << 2));
        const b: u32 = load<i32>(bp + (i << 2));
        store<i32>(d + (i << 2), ~(a & b));
      }
      break;
    }
    case 0x8: {
      for (let i = 0; i < n; i++) {
        const a: u32 = load<i32>(ap + (i << 2));
        const b: u32 = load<i32>(bp + (i << 2));
        store<i32>(d + (i << 2), a & b);
      }
      break;
    }
    case 0x9: {
      for (let i = 0; i < n; i++) {
        const a: u32 = load<i32>(ap + (i << 2));
        const b: u32 = load<i32>(bp + (i << 2));
        store<i32>(d + (i << 2), ~(a ^ b));
      }
      break;
    }
    case 0xa: {
      for (let i = 0; i < n; i++) store<i32>(d + (i << 2), load<i32>(ap + (i << 2)));
      break;
    }
    case 0xb: {
      for (let i = 0; i < n; i++) {
        const a: u32 = load<i32>(ap + (i << 2));
        const b: u32 = load<i32>(bp + (i << 2));
        store<i32>(d + (i << 2), a | ~b);
      }
      break;
    }
    case 0xc: {
      for (let i = 0; i < n; i++) store<i32>(d + (i << 2), load<i32>(bp + (i << 2)));
      break;
    }
    case 0xd: {
      for (let i = 0; i < n; i++) {
        const a: u32 = load<i32>(ap + (i << 2));
        const b: u32 = load<i32>(bp + (i << 2));
        store<i32>(d + (i << 2), ~a | b);
      }
      break;
    }
    case 0xe: {
      for (let i = 0; i < n; i++) {
        const a: u32 = load<i32>(ap + (i << 2));
        const b: u32 = load<i32>(bp + (i << 2));
        store<i32>(d + (i << 2), a | b);
      }
      break;
    }
    default: {
      for (let i = 0; i < n; i++) store<i32>(d + (i << 2), 0xffffffff);
      break;
    }
  }
}

/** 出力書き出しスケジュールの1バケット分(命令 bucket-1 の直後、bucket=0 は
    プログラム実行前)を出力平面へ書く。行末尾に tailMaskOut を AND する
    (bitplane.ts と同じ契約: 中間の端数ビットは実位置の値に影響しない)。
    tapWpr === wprOut は経路によらず成り立つ(高速路は WOut===W より
    wprIn===wprOut、strided路は tapWpr がそのまま wprOut)ので、1つの oc の
    出力平面は HOut*wprOut 語の連続コピー1本 + 各行末語へのマスクで書ける。
    tapWpr は ABI 互換のため引数に残す(値は使わない)。 */
function writeProgBucket(
  bucket: i32, writeStartPtr: i32, writeOcPtr: i32, ocSrcOffPtr: i32,
  scratchPtr: i32, outPtr: i32,
  HOut: i32, wprOut: i32, tapWpr: i32, tailMaskOut: i32,
): void {
  const s0 = load<i32>(writeStartPtr + (bucket << 2));
  const s1 = load<i32>(writeStartPtr + ((bucket + 1) << 2));
  const planeWords = HOut * wprOut;
  for (let i = s0; i < s1; i++) {
    const oc = load<i32>(writeOcPtr + (i << 2));
    const srcOff = load<i32>(ocSrcOffPtr + (oc << 2));
    const ocBase = outPtr + (oc * planeWords << 2);
    const srcBase = scratchPtr + (srcOff << 2);
    for (let k = 0; k < planeWords; k++) {
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
// グラフ入力パック / or_pool2d のWASMカーネル。
//
// C5完了時点のフレーム内訳計測(scripts/profile_cpu.mjs)で、bitplane-wasm
// エンジンのフレームを支配していたのは conv2d ではなく
//   (1) グラフ入力ポートの packPlanes(det 55.8% / detface 63.4% / detreal 69.6%)
//   (2) JS の orPool2x2  (det 22.6% / detmono 42.6% / detface 21.1%)
// だった。どちらも「行 or 平面ごとに完全独立」なので、ここでは
//   - 単スレッドでも JS より速いWASMカーネルにする
//   - かつ [first, first+count) のシャード引数を持たせ、そのまま
//     ワーカープール(src/engines/wasmThreadPool.ts)から分割呼び出しできる
// という2つを同時に満たす形で実装する。シャードは出力側で完全に
// 排他(行/平面が重ならない)なので、スレッド数によらず出力ビットは不変。
//
// 正しさの錨: bitplane.ts::packPlanes / orPool2x2 との整数完全一致
// (conv.test.ts の ENGINES 行列 + wasmBitplaneOps のラウンドトリップ検証)。
// ============================================================

/** 0/1(非ゼロ→1)バイト列 [rows][W] を行頭ワード境界のビットプレーン
    [rows][wpr] へパックする。rows は [batch][C][H] を平坦化した行番号で、
    [rowFirst, rowFirst+rowCount) のみを処理する(シャード単位)。
    ビット順は LSB-first(bitplane.ts のレイアウト契約と同一)。 */
export function packPlanesAsm(
  inPtr: i32, outPtr: i32, rowFirst: i32, rowCount: i32, W: i32, wpr: i32,
): void {
  const rEnd = rowFirst + rowCount;
  for (let r = rowFirst; r < rEnd; r++) {
    const src = inPtr + r * W;
    const dst = outPtr + ((r * wpr) << 2);
    for (let j = 0; j < wpr; j++) {
      const base = src + (j << 5);
      let n = W - (j << 5);
      if (n > 32) n = 32;
      let acc: u32 = 0;
      for (let i = 0; i < n; i++) {
        if (load<u8>(base + i) != 0) acc |= (<u32>1) << i;
      }
      store<i32>(dst + (j << 2), acc);
    }
  }
}

/** 32bit中の偶数位置ビット(0,2,...,30)を下位16bitへ圧縮する
    (bitplane.ts::compressEven32Shift と同一のシフトカスケード)。 */
function compressEven32(v: u32): u32 {
  let x = v & 0x55555555;
  x = (x | (x >>> 1)) & 0x33333333;
  x = (x | (x >>> 2)) & 0x0f0f0f0f;
  x = (x | (x >>> 4)) & 0x00ff00ff;
  x = (x | (x >>> 8)) & 0x0000ffff;
  return x;
}

/** or_pool2d(kernel=2, stride=2, padding=0)をビットプレーンのまま評価する。
    平面(=batch*C のうちの1枚)単位でシャードし、[planeFirst,
    planeFirst+planeCount) を処理する。演算は bitplane.ts::orPoolRow と
    逐語的に同一(縦OR → 水平隣接OR → 偶数ビット圧縮 → 行末尾maskOut)。 */
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
        // th(=t[jh])と tl1(=t[jl+1])は同じ語。JS版の scratch[] と同一の値。
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
// 画像前処理(RGBA→ビット符号化)+ 入力パックの融合カーネル。
//
// 統一計測で、CPUバックエンドのフレームを支配していたのは conv2d でも
// pack でもなく **CPU前処理** だった(wasm推論 det 1.14 / detface 1.08ms に対し
// e2e は det 5.0 / detface 7.8ms)。実体は convpreproc.ts::encodeImageAuto で、
// RGBA→NCHW float 化ののち最大173ch×128×128=2.8M ビットを1本ずつ float 演算で
// 求めて 0/1 バイト列に書き、それを packPlanes が読み直していた。ここでは
//   (1) 前処理と packPlanes を1パスに融合(中間の 0/1 バイト列を作らない)
//   (2) 浮動小数を一切使わない**厳密整数化**(の gpuPreproc.ts と同一の
//       数値経路)で、CPUエンコーダとビット完全一致を保つ
// の2つを同時に行う。
//
// 数値経路(gpuPreproc.ts の冒頭コメントが単一情報源):
//   入力は u8 なので画素値は f32(k/255) の256通りしかなく、
//   I(k) := f32(k/255)*2^31 は厳密な整数(u32に収まる)。edge/curv/chroma/彩度は
//   I の整数結合で、比較は d > t ⟺ D >= floor(t*2^31)+1、d < -t ⟺ D <= ceil(-t*2^31)-1
//   と厳密整数化できる。thermometer だけは任意の f64 係数を含むので 256ビットLUT
//   (CPU側と同一のJS式で生成)を参照する。単調(std>0)なら「k >= k0」の1個の
//   閾値へ畳める(記述子側で判定済み。THERMO_STEP)。
//
// 入力の持ち方:
//   - バイト平面 bytePtr[c][y*W + x]: RGBA を画像チャネルごとにデインタリーブ
//     したもの(thermometer 用)。
//   - 固定小数平面 qPtr[(c*H + y)*rowStride + rowPad + x] = I(k) (u32):
//     **行の両端に rowPad 個の replicate padding** を持たせてあるので、
//     edge/curv の水平シフト x+dx は範囲検査なしで読める(CPU側 shiftAt の
//     x クランプと定義上同一)。垂直クランプは行ポインタ選択で行う。
//
// シャード契約: 出力チャネル [chFirst, chFirst+chCount) だけを書く。チャネル平面は
// 互いに素なのでスレッド数によらず出力ビットは不変(packPlanesAsm と同じ流儀)。
//
// 正しさの錨: convpreproc.ts::encodeImageAuto + bitplane.ts::packPlanes との
// 全ワード一致(cpuPreproc.test.ts / wasmPreproc.test.ts、scalar⇔simd⇔JS の三者)。
// ============================================================

/** 記述子の演算種別(src/engines/cpuPreproc.ts と同一の値。0..4 は
    gpuPreproc.ts の PREPROC_OP_* と共通)。 */
// 0 THERMO_LUT / 1 EDGE / 2 CURV / 3 CHROMA / 4 SAT / 5 THERMO_STEP

/** 記述子の cut(64bit二の補数)の下位語をゼロ拡張するためのマスク。 */
const PREPROC_LO_MASK: i64 = 4294967295;

function clampIdx(v: i32, hi: i32): i32 {
  if (v < 0) return 0;
  return v > hi ? hi : v;
}

/** 固定小数平面の (c, clamp(y)) 行の x=0 のアドレス。 */
function qRowPtr(
  qPtr: i32, c: i32, y: i32, H: i32, rowStride: i32, rowPad: i32,
): i32 {
  return qPtr + (((c * H + clampIdx(y, H - 1)) * rowStride + rowPad) << 2);
}

/** RGBA(u8×4/画素)→ バイト平面 + 固定小数平面(行両端 replicate padding)。 */
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
      // replicate padding(CPU側 shiftAt の x クランプと同一)
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
  // 4バイト同時の符号なし比較(SWAR)。1個の u32 ロードから
  // 「各バイトが k0 以上か」の4ビットを取り出す:
  //   t = (x | H) - (k0 & ~H) の bit7 は「下位7bitの比較 xl >= k0l」を表し、
  //   最上位ビットが違えばそちらが勝つので
  //   res = (x & ~k0) | (~(x ^ k0) & t) の bit7 が x >= k0 になる(全256値総当り検証)。
  // 取り出しは bitplane.ts::packRowU8 と同じ imul の桁寄せ。
  // k0=256(常に0)は乗算で表せないので素直なループへ。
  const useSwar = k0 >= 0 && k0 <= 255;
  const yb: u32 = (<u32>k0) * 0x01010101;
  const yl: u32 = yb & 0x7f7f7f7f;
  for (let y = 0; y < H; y++) {
    const row = srcPtr + y * W;
    const dst = outPtr + ((y * wpr) << 2);
    for (let j = 0; j < wpr; j++) {
      const base = row + (j << 5);
      let n = W - (j << 5);
      if (n > 32) n = 32;
      let acc: u32 = 0;
      let i = 0;
      if (useSwar) {
        for (; i + 4 <= n; i += 4) {
          const x: u32 = load<u32>(base + i);
          const t: u32 = (x | 0x80808080) - yl;
          const ge: u32 = ((x & ~yb) | (~(x ^ yb) & t)) & 0x80808080;
          const nz: u32 = (ge >>> 7) & 0x01010101;
          acc |= ((nz * 0x01020408) >>> 24) << i;
        }
      }
      for (; i < n; i++) {
        if (<i32>load<u8>(base + i) >= k0) acc |= (<u32>1) << i;
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

/** 2項差 D = q[pa + x] - q[pb + x](edge/chroma 共通)。 */
function packDiff2Rows(
  paRow: i32, pbRow: i32, dst: i32, W: i32, wpr: i32, tail: i32,
  cut: i64, dirLe: bool,
): void {
  for (let j = 0; j < wpr; j++) {
    const x0 = j << 5;
    let n = W - x0;
    if (n > 32) n = 32;
    let acc: u32 = 0;
    for (let i = 0; i < n; i++) {
      const o = (x0 + i) << 2;
      const d: i64 = <i64>load<u32>(paRow + o) - <i64>load<u32>(pbRow + o);
      const bit = dirLe ? (d <= cut) : (d >= cut);
      if (bit) acc |= (<u32>1) << i;
    }
    if (j == wpr - 1) acc &= <u32>tail;
    store<i32>(dst + (j << 2), acc);
  }
}

/** 2階差 D = q[pn + x] - 2*q[pm + x] + q[pp + x](curv)。 */
function packCurvRows(
  ppRow: i32, pmRow: i32, pnRow: i32, dst: i32, W: i32, wpr: i32, tail: i32,
  cut: i64, dirLe: bool,
): void {
  for (let j = 0; j < wpr; j++) {
    const x0 = j << 5;
    let n = W - x0;
    if (n > 32) n = 32;
    let acc: u32 = 0;
    for (let i = 0; i < n; i++) {
      const o = (x0 + i) << 2;
      const d: i64 = <i64>load<u32>(pnRow + o)
        - (<i64>load<u32>(pmRow + o) << 1)
        + <i64>load<u32>(ppRow + o);
      const bit = dirLe ? (d <= cut) : (d >= cut);
      if (bit) acc |= (<u32>1) << i;
    }
    if (j == wpr - 1) acc &= <u32>tail;
    store<i32>(dst + (j << 2), acc);
  }
}

/** 彩度 D = max_c I(k_c) - min_c I(k_c)。I は k について狭義単調増加なので
    「バイトの max/min を取ってから I を引く」CPU/GPU側と同値。 */
function packSatRows(
  qPtr: i32, dst: i32, y: i32, H: i32, W: i32, wpr: i32, tail: i32,
  rowStride: i32, rowPad: i32, cimg: i32, cut: i64, dirLe: bool,
): void {
  const row0 = qRowPtr(qPtr, 0, y, H, rowStride, rowPad);
  const planeWords = H * rowStride;
  for (let j = 0; j < wpr; j++) {
    const x0 = j << 5;
    let n = W - x0;
    if (n > 32) n = 32;
    let acc: u32 = 0;
    for (let i = 0; i < n; i++) {
      const o = (x0 + i) << 2;
      let mx: u32 = 0;
      let mn: u32 = 0xffffffff;
      for (let c = 0; c < cimg; c++) {
        const v: u32 = load<u32>(row0 + ((c * planeWords) << 2) + o);
        if (v > mx) mx = v;
        if (v < mn) mn = v;
      }
      const d: i64 = <i64>mx - <i64>mn;
      const bit = dirLe ? (d <= cut) : (d >= cut);
      if (bit) acc |= (<u32>1) << i;
    }
    if (j == wpr - 1) acc &= <u32>tail;
    store<i32>(dst + (j << 2), acc);
  }
}

/** 記述子に従って出力チャネル [chFirst, chFirst+chCount) のビットプレーンを
    直接生成する(0/1バイト列を経由しない融合パック)。 */
export function preprocPackAsm(
  qPtr: i32, bytePtr: i32, descPtr: i32, outPtr: i32,
  chFirst: i32, chCount: i32,
  H: i32, W: i32, wpr: i32, tail: i32,
  rowStride: i32, rowPad: i32, byteStride: i32, cimg: i32,
): void {
  const chEnd = chFirst + chCount;
  for (let ch = chFirst; ch < chEnd; ch++) {
    const d = descPtr + (ch << 6); // 16語/チャネル
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
