// conv2d ゲート木の WASM(AssemblyScript)実行ラッパー(packed-wasm
// エンジン)。assembly/index.ts の evalConv2dBatch を、evalConv2dPacked
// (packedConv.ts)と全く同じ幾何計算(タップ分解・木のping-pong評価)で
// 呼び出す。正しさの錨は conv_golden.json / seg_e2e_golden.json との
// 整数完全一致(wasm.test.ts)。
//
// メモリ契約: WasmModule はGCランタイムを持たない単純バンプアロケータ
// (alloc)。ConvRuntime生存期間中、conv2dノード毎に1回だけ alloc し
// 使い回す(毎フレームの再アロケートなし)。memory.grow() 後は
// `mod.memory.buffer` の参照が変わる(古いArrayBufferはdetachされる)ため、
// 全アロケート完了後にのみ書き込み用ビューを作る(このファイルの各所で
// 徹底: alloc() 呼び出しの後に fresh view を作る)。
import { ttParse } from '../shared';
import type { Conv2dParams } from '../shared';
import type { NodeValue } from './irflat';

export interface WasmModule {
  memory: WebAssembly.Memory;
  alloc(bytes: number): number;
  resetHeap(): void;
  evalConv2dBatch(
    xPtr: number,
    leafCPtr: number, leafDyPtr: number, leafDxPtr: number, ttNumPtr: number,
    leafPlanesPtr: number, bankAPtr: number, bankBPtr: number,
    outPtr: number,
    H: number, W: number, HOut: number, WOut: number,
    sH: number, sW: number, pH: number, pW: number,
    OC: number, nLeaves: number, nInternal: number, wordsOut: number,
  ): void;
  /** bitplane conv2d(assembly/index.ts::evalConv2dBitplaneBatch)。
      wasmBitplaneConv.ts::WasmBitplaneConvKernel が使う。 */
  evalConv2dBitplaneBatch(
    xPtr: number,
    planeCPtr: number, planeDyPtr: number, planeDxPtr: number, nPlanes: number,
    leafWordBasePtr: number, ttNumPtr: number,
    tapPoolPtr: number, bankAPtr: number, bankBPtr: number,
    outPtr: number,
    H: number, W: number, HOut: number, WOut: number,
    sH: number, sW: number, pH: number, pW: number,
    OC: number, nLeaves: number, nInternal: number,
    wprIn: number, wprOut: number, tapWpr: number, tailMaskOut: number, useFastPath: number,
  ): void;
  /** conv2d 命令列カーネル(assembly/index.ts::evalConvProgBatch)。
      wasmBitplaneConv.ts::WasmBitplaneConvKernel が使う。命令列・スケジュール・
      全参照のワードオフセットは bitplane.ts::buildConvProgramLayout が構築時に
      解決し、平坦なInt32配列としてヒープに載せる(このABIには参照種別の分岐が
      無い)。古い .wasm を読んだ場合に備えて optional(未定義なら旧カーネルへ
      フォールバックする)。 */
  evalConvProgBatch?(
    xPtr: number,
    planeCPtr: number, planeDyPtr: number, planeDxPtr: number, nPlanes: number,
    instrPtr: number, nInstr: number,
    writeStartPtr: number, writeOcPtr: number, ocSrcOffPtr: number,
    scratchPtr: number, outPtr: number,
    H: number, W: number, HOut: number, WOut: number,
    sH: number, sW: number, pH: number, pW: number,
    wprIn: number, wprOut: number, tapWpr: number, tailMaskOut: number, useFastPath: number,
    poolBase: number, planeStride: number,
  ): void;
  /** グラフ入力パック(assembly/index*.ts::packPlanesAsm)。
      行 [rowFirst, rowFirst+rowCount) だけを処理するシャード契約なので、
      wasmThreadPool.ts からそのまま分割呼び出しできる。古い .wasm を読んだ
      場合に備えて optional(未定義なら JS の packPlanes が使われる)。 */
  packPlanesAsm?(
    inPtr: number, outPtr: number, rowFirst: number, rowCount: number,
    W: number, wpr: number,
  ): void;
  /** or_pool2d(2x2/stride2/pad0)。平面 [planeFirst, +planeCount) の
      シャード契約。同上 optional。 */
  orPool2x2Asm?(
    inPtr: number, outPtr: number, planeFirst: number, planeCount: number,
    H: number, wpr: number, HOut: number, wprOut: number, maskOut: number,
  ): void;
  /** 融合前処理の第1段: RGBA → バイト平面 + 固定小数平面
      (assembly/index*.ts::preprocQuantAsm)。同上 optional
      (未定義なら JS の融合前処理、さらに無効なら従来のCPU前処理へ落ちる)。 */
  preprocQuantAsm?(
    rgbaPtr: number, qvalPtr: number, qPtr: number, bytePtr: number,
    H: number, W: number, rowStride: number, rowPad: number,
    byteStride: number, cimg: number,
  ): void;
  /** 融合前処理の第2段: 記述子に従って出力チャネル
      [chFirst, chFirst+chCount) のビットプレーンを直接生成する
      (前処理とpackPlanesの融合)。チャネル単位のシャード契約。 */
  preprocPackAsm?(
    qPtr: number, bytePtr: number, descPtr: number, outPtr: number,
    chFirst: number, chCount: number,
    H: number, W: number, wpr: number, tail: number,
    rowStride: number, rowPad: number, byteStride: number, cimg: number,
  ): void;
  /** lut_flat → group_sum 融合(1枚ずつ、assembly/index*.ts::lutFlatGsAsm)。
      wasmLutFlat.ts が使う。古い .wasm に備えて optional(未定義なら JS の
      lutFlat.ts::evalLutFlatGroupSum)。 */
  lutFlatGsAsm?(
    tt: number, a: number, b: number, o: number, inIdx: number,
    nIn: number, nG: number, outW: number, k: number,
    x: number, inW: number, batch: number, sig: number, scores: number,
  ): void;
  /** 同・バッチ軸パック版(lutFlatPackedGsAsm)。*/
  lutFlatPackedGsAsm?(
    tt: number, a: number, b: number, o: number, inIdx: number,
    nIn: number, nG: number, outW: number, k: number,
    x: number, inW: number, batch: number, sig: number, planes: number, scores: number,
  ): void;
}

/** 生の.wasmバイト列からインスタンス化する(環境非依存の純関数)。
    ブラウザ側は fetch() で、テスト(vitest/node)は node:fs で読んだ
    バイト列をそれぞれこの関数に渡す(呼び出し側で環境分岐)。 */
export async function instantiateWasmModule(bytes: ArrayBuffer | Uint8Array): Promise<WasmModule> {
  const source = bytes instanceof Uint8Array ? bytes.slice().buffer : bytes;
  const result = await WebAssembly.instantiate(source, {});
  return result.instance.exports as unknown as WasmModule;
}

function pair(v: number | number[]): [number, number] {
  return Array.isArray(v) ? [v[0], v[1]] : [v, v];
}

/** 1つの conv2d ノードぶんの WASM 実行カーネル。geometry/leaf/tt は
    ノード構築時に1回だけ WASM メモリへ書き込み、以後 evalBatch では
    x(入力)の書き込みと out(出力)の読み出しだけを行う。 */
export class WasmConvKernel {
  private readonly H: number; private readonly W: number;
  private readonly HOut: number; private readonly WOut: number;
  private readonly sH: number; private readonly sW: number;
  private readonly pH: number; private readonly pW: number;
  private readonly OC: number; private readonly nLeaves: number;
  private readonly nInternal: number; private readonly wordsOut: number;
  private readonly inW: number; private readonly outW: number;

  private readonly xPtr: number;
  private readonly leafCPtr: number; private readonly leafDyPtr: number;
  private readonly leafDxPtr: number; private readonly ttNumPtr: number;
  private readonly leafPlanesPtr: number;
  private readonly bankAPtr: number; private readonly bankBPtr: number;
  private readonly outPtr: number;

  constructor(private readonly mod: WasmModule, p: Conv2dParams) {
    const [C, H, W] = p.in_shape;
    const [kH, kW] = pair(p.kernel_size);
    const [sH, sW] = pair(p.stride);
    const [pH, pW] = pair(p.padding);
    const HOut = Math.floor((H + 2 * pH - kH) / sH) + 1;
    const WOut = Math.floor((W + 2 * pW - kW) / sW) + 1;
    const OC = p.out_channels;
    const N = p.n_inputs;
    if (N !== 2) throw new Error(`packed-wasm conv は n_inputs=2 のみ対応(実際: ${N})`);
    const depth = p.tree_depth;
    const nLeaves = N ** depth;
    const nInternal = (nLeaves - 1) / (N - 1);
    const Lout = HOut * WOut;
    const wordsOut = (Lout + 31) >> 5;
    const khw = kH * kW;
    const idxf = p.indices_flat;

    this.H = H; this.W = W; this.HOut = HOut; this.WOut = WOut;
    this.sH = sH; this.sW = sW; this.pH = pH; this.pW = pW;
    this.OC = OC; this.nLeaves = nLeaves; this.nInternal = nInternal;
    this.wordsOut = wordsOut;
    this.inW = C * H * W;
    this.outW = OC * Lout;

    // evalConv2dPacked と同一のタップ分解(pp -> c,dy,dx)。
    const leafC = new Int32Array(OC * nLeaves);
    const leafDy = new Int32Array(OC * nLeaves);
    const leafDx = new Int32Array(OC * nLeaves);
    for (let oc = 0; oc < OC; oc++) {
      const row = idxf[oc];
      for (let k = 0; k < nLeaves; k++) {
        const pp = row[k];
        const rem = pp % khw;
        leafC[oc * nLeaves + k] = Math.floor(pp / khw);
        leafDy[oc * nLeaves + k] = Math.floor(rem / kW);
        leafDx[oc * nLeaves + k] = rem % kW;
      }
    }
    const ttNum = new Uint8Array(OC * nInternal);
    for (let i = 0; i < ttNum.length; i++) {
      ttNum[i] = Number(ttParse(p.truth_tables[i])) & 0xf;
    }

    // --- WASM メモリを確保(このノードの生存期間中固定) ---
    const maxBank = Math.max(1, nLeaves >> 1);
    this.leafCPtr = mod.alloc(leafC.length * 4);
    this.leafDyPtr = mod.alloc(leafDy.length * 4);
    this.leafDxPtr = mod.alloc(leafDx.length * 4);
    this.ttNumPtr = mod.alloc(ttNum.length);
    this.leafPlanesPtr = mod.alloc(nLeaves * wordsOut * 4);
    this.bankAPtr = mod.alloc(maxBank * wordsOut * 4);
    this.bankBPtr = mod.alloc(maxBank * wordsOut * 4);
    this.xPtr = mod.alloc(this.inW * 4);
    this.outPtr = mod.alloc(this.outW * 4);

    // アロケート完了後にfreshビューで書き込む(memory.grow後のdetach対策)。
    new Int32Array(mod.memory.buffer, this.leafCPtr, leafC.length).set(leafC);
    new Int32Array(mod.memory.buffer, this.leafDyPtr, leafDy.length).set(leafDy);
    new Int32Array(mod.memory.buffer, this.leafDxPtr, leafDx.length).set(leafDx);
    new Uint8Array(mod.memory.buffer, this.ttNumPtr, ttNum.length).set(ttNum);
  }

  /** [batch*(C*H*W)]のxからこのノードの出力[batch*(OC*Lout)]を計算する
      (evalConv2dPacked と同一の出力並び)。 */
  evalAll(x: NodeValue, batch: number): NodeValue {
    const data = new Int32Array(batch * this.outW);
    for (let b = 0; b < batch; b++) {
      const xoff = b * x.width;
      // evalConv2dPackedはx全幅(inWidth)を前提にoffsetでスライスするが、
      // packed-wasmではノード毎に固定確保したxPtrバッファへ都度書き込む。
      new Int32Array(this.mod.memory.buffer, this.xPtr, this.inW)
        .set(x.data.subarray(xoff, xoff + this.inW));
      this.mod.evalConv2dBatch(
        this.xPtr,
        this.leafCPtr, this.leafDyPtr, this.leafDxPtr, this.ttNumPtr,
        this.leafPlanesPtr, this.bankAPtr, this.bankBPtr,
        this.outPtr,
        this.H, this.W, this.HOut, this.WOut,
        this.sH, this.sW, this.pH, this.pW,
        this.OC, this.nLeaves, this.nInternal, this.wordsOut,
      );
      data.set(
        new Int32Array(this.mod.memory.buffer, this.outPtr, this.outW),
        b * this.outW,
      );
    }
    return { data, width: this.outW };
  }
}
