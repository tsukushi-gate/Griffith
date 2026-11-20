// conv2d ビットプレーンカーネルの WASM 実行ラッパー(bitplane-wasm
// エンジン)。assembly/index.ts の2つのカーネルのどちらかを呼ぶ:
//   - evalConvProgBatch(既定): conv2d の重み共有木を convProgram.ts で
//     lossless に命令列へ縮約してから実行する(パススルー/定数/重複サブ木の
//     除去でゲート数が1/2〜1/15になる)。命令列・スケジュール・全参照の
//     ワードオフセット解決は bitplane.ts::buildConvProgramLayout が構築時に
//     1回だけ行い、平坦なInt32配列としてWASMヒープへ書く。
//   - evalConv2dBitplaneBatch(旧経路): 全ocを同じ深さのバランス木として
//     全ゲート評価する。命令列化の構築/セルフチェックに失敗した場合、または
//     ConvRuntimeOptions.convProgram=false のときに使う。
// どちらも bitplane.ts::evalConv2dBitplane と全く同じ幾何計算(タップ分解・
// 平面プール構築)を使い、整数出力は完全に一致する。
//
// 旧経路の唯一の設計上の注意: 平面の oc横断dedup は、バッチ非依存の
// 静的ジオメトリなのでこのファイルが構築時に1回だけ計算し
// (evalConv2dBitplane の中の keyToPlane Map と同じロジック)、
// leafWordBase(各leafが指すプール内のワード基点)としてWASM側へ渡す
// (WASM側にMap実装を持たせない設計)。
// dedupキーは高速路(stride=1 かつ WOut===W)では (c,dx)、strided路
// では従来どおり (c,dy,dx)。高速路の平面は「上下pH行のゼロパディング込みで
// H+2*pH 行の水平シフト済み共有平面」で、リーフは平面ID+dy行オフセットで
// 参照する(bitplane.ts::buildSharedShiftPlane 参照)。
//
// メモリ契約は wasmConv.ts::WasmConvKernel と同じ(単純バンプアロケータ、
// ノード生存期間中固定のバッファをノード構築時に1回だけ確保)。alloc() は
// memory.grow を起こしうるので、書き込み用ビューは全alloc完了後に作る。
//
// 正しさの錨: bitplane.ts::evalConv2dBitplane との整数完全一致
// (conv.test.ts / seg_e2e.test.ts / detdecode.test.ts の ENGINES 行列に
// 'bitplane-wasm' を追加して検証)+ 命令列版はロード時の全数真理値表
// セルフチェック(convProgram.ts::verifyConvProgram)。
import { ttParse } from '../shared';
import type { Conv2dParams } from '../shared';
import { buildConvProgramLayout, tailMask, wordsPerRow } from './bitplane';
import type { BitplaneValue, ConvProgramLayout } from './bitplane';
import type { ConvProgram } from './convProgram';
import type { WasmModule } from './wasmConv';
import { addCpuSample, cpuProfilingEnabled } from './cpuProfile';

function pair(v: number | number[]): [number, number] {
  return Array.isArray(v) ? [v[0], v[1]] : [v, v];
}

interface KernelImpl {
  evalAll(x: BitplaneValue, batch: number): BitplaneValue;
}

/** 旧経路: 全ocをバランス木として全ゲート評価する。*/
class LegacyWasmBitplaneKernel implements KernelImpl {
  private readonly H: number; private readonly W: number;
  private readonly HOut: number; private readonly WOut: number;
  private readonly sH: number; private readonly sW: number;
  private readonly pH: number; private readonly pW: number;
  private readonly OC: number; private readonly nLeaves: number;
  private readonly nInternal: number;
  private readonly wprIn: number; private readonly wprOut: number;
  private readonly tapWpr: number; private readonly tailMaskOut: number;
  private readonly useFastPath: number;
  private readonly nPlanes: number;
  private readonly inPlaneWords: number; private readonly outPlaneWords: number;

  private readonly xPtr: number;
  private readonly planeCPtr: number; private readonly planeDyPtr: number; private readonly planeDxPtr: number;
  private readonly leafWordBasePtr: number; private readonly ttNumPtr: number;
  private readonly tapPoolPtr: number;
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
    if (N !== 2) throw new Error(`bitplane-wasm conv は n_inputs=2 のみ対応(実際: ${N})`);
    const depth = p.tree_depth;
    const nLeaves = N ** depth;
    const nInternal = (nLeaves - 1) / (N - 1);
    const khw = kH * kW;
    const idxf = p.indices_flat;

    this.H = H; this.W = W; this.HOut = HOut; this.WOut = WOut;
    this.sH = sH; this.sW = sW; this.pH = pH; this.pW = pW;
    this.OC = OC; this.nLeaves = nLeaves; this.nInternal = nInternal;

    const wprIn = wordsPerRow(W);
    const wprOut = wordsPerRow(WOut);
    const useFastPath = sH === 1 && sW === 1 && WOut === W;
    const tapWpr = useFastPath ? wprIn : wprOut;
    this.wprIn = wprIn; this.wprOut = wprOut; this.tapWpr = tapWpr;
    this.tailMaskOut = tailMask(WOut);
    this.useFastPath = useFastPath ? 1 : 0;
    this.inPlaneWords = C * H * wprIn;
    this.outPlaneWords = OC * HOut * wprOut;
    const tapRowsWords = HOut * tapWpr;
    // 平面1枚の行数/ワード数。WASM側(assembly/index*.ts)が同じ式で
    // 再計算するので、変更する場合は3箇所を同時に直すこと。
    const planeRows = useFastPath ? H + 2 * pH : HOut;
    const planeStride = planeRows * tapWpr;

    // leafの平面dedup(evalConv2dBitplaneのkeyToPlaneと同一ロジックだが、
    // バッチ非依存なのでノード構築時に1回だけ計算する)。
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
    this.nPlanes = nPlanes;

    const ttNum = new Uint8Array(OC * nInternal);
    for (let i = 0; i < ttNum.length; i++) {
      ttNum[i] = Number(ttParse(p.truth_tables[i])) & 0xf;
    }

    // --- WASM メモリを確保(このノードの生存期間中固定) ---
    const maxBank = Math.max(1, nLeaves >> 1);
    this.planeCPtr = mod.alloc(nPlanes * 4);
    this.planeDyPtr = mod.alloc(nPlanes * 4);
    this.planeDxPtr = mod.alloc(nPlanes * 4);
    this.leafWordBasePtr = mod.alloc(leafWordBase.length * 4);
    this.ttNumPtr = mod.alloc(ttNum.length);
    this.tapPoolPtr = mod.alloc(Math.max(1, nPlanes) * planeStride * 4);
    this.bankAPtr = mod.alloc(maxBank * tapRowsWords * 4);
    this.bankBPtr = mod.alloc(maxBank * tapRowsWords * 4);
    this.xPtr = mod.alloc(this.inPlaneWords * 4);
    this.outPtr = mod.alloc(this.outPlaneWords * 4);

    new Int32Array(mod.memory.buffer, this.planeCPtr, nPlanes).set(planeC);
    new Int32Array(mod.memory.buffer, this.planeDyPtr, nPlanes).set(planeDy);
    new Int32Array(mod.memory.buffer, this.planeDxPtr, nPlanes).set(planeDx);
    new Int32Array(mod.memory.buffer, this.leafWordBasePtr, leafWordBase.length).set(leafWordBase);
    new Uint8Array(mod.memory.buffer, this.ttNumPtr, ttNum.length).set(ttNum);
  }

  evalAll(x: BitplaneValue, batch: number): BitplaneValue {
    const words = new Int32Array(batch * this.outPlaneWords);
    for (let b = 0; b < batch; b++) {
      const xoff = b * this.inPlaneWords;
      new Int32Array(this.mod.memory.buffer, this.xPtr, this.inPlaneWords)
        .set(x.words.subarray(xoff, xoff + this.inPlaneWords));
      this.mod.evalConv2dBitplaneBatch(
        this.xPtr,
        this.planeCPtr, this.planeDyPtr, this.planeDxPtr, this.nPlanes,
        this.leafWordBasePtr, this.ttNumPtr,
        this.tapPoolPtr, this.bankAPtr, this.bankBPtr,
        this.outPtr,
        this.H, this.W, this.HOut, this.WOut,
        this.sH, this.sW, this.pH, this.pW,
        this.OC, this.nLeaves, this.nInternal,
        this.wprIn, this.wprOut, this.tapWpr, this.tailMaskOut, this.useFastPath,
      );
      words.set(
        new Int32Array(this.mod.memory.buffer, this.outPtr, this.outPlaneWords),
        b * this.outPlaneWords,
      );
    }
    return { words, C: this.OC, H: this.HOut, W: this.WOut, wpr: this.wprOut };
  }
}

/** 命令列経路: buildConvProgramLayout の結果をWASMヒープへ平坦化して
    evalConvProgBatch を呼ぶ。JS版(BitplaneProgramConvKernel)と同一レイアウト
    ・同一スケジュールなので、両者は整数完全一致する。 */
class ProgWasmBitplaneKernel implements KernelImpl {
  private readonly L: ConvProgramLayout;
  private readonly nInstr: number;
  private readonly xPtr: number;
  private readonly planeCPtr: number; private readonly planeDyPtr: number;
  private readonly planeDxPtr: number;
  private readonly instrPtr: number;
  private readonly writeStartPtr: number; private readonly writeOcPtr: number;
  private readonly ocSrcOffPtr: number;
  private readonly scratchPtr: number; private readonly outPtr: number;

  constructor(private readonly mod: WasmModule, layout: ConvProgramLayout) {
    this.L = layout;
    const L = layout;
    this.nInstr = L.instrFlat.length >> 2;
    const { writeStart, writeOc } = L.program;

    this.planeCPtr = mod.alloc(Math.max(1, L.nPlanes) * 4);
    this.planeDyPtr = mod.alloc(Math.max(1, L.nPlanes) * 4);
    this.planeDxPtr = mod.alloc(Math.max(1, L.nPlanes) * 4);
    this.instrPtr = mod.alloc(Math.max(4, L.instrFlat.length) * 4);
    this.writeStartPtr = mod.alloc(writeStart.length * 4);
    this.writeOcPtr = mod.alloc(Math.max(1, writeOc.length) * 4);
    this.ocSrcOffPtr = mod.alloc(Math.max(1, L.ocSrcOff.length) * 4);
    this.scratchPtr = mod.alloc(L.scratchWords * 4);
    this.xPtr = mod.alloc(L.inPlaneWords * 4);
    this.outPtr = mod.alloc(L.outPlaneWords * 4);

    // 全alloc完了後にfreshビューで書き込む(memory.grow後のdetach対策)。
    const buf = mod.memory.buffer;
    new Int32Array(buf, this.planeCPtr, L.nPlanes).set(L.planeC);
    new Int32Array(buf, this.planeDyPtr, L.nPlanes).set(L.planeDy);
    new Int32Array(buf, this.planeDxPtr, L.nPlanes).set(L.planeDx);
    new Int32Array(buf, this.instrPtr, L.instrFlat.length).set(L.instrFlat);
    new Int32Array(buf, this.writeStartPtr, writeStart.length).set(writeStart);
    new Int32Array(buf, this.writeOcPtr, writeOc.length).set(writeOc);
    new Int32Array(buf, this.ocSrcOffPtr, L.ocSrcOff.length).set(L.ocSrcOff);
    // 定数平面: [0,tapRowsWords)=全0、[tapRowsWords,2*tapRowsWords)=全1。
    // (resetHeap()でヒープを巻き戻した場合は前の内容が残りうるので明示する)
    const consts = new Int32Array(buf, this.scratchPtr, 2 * L.tapRowsWords);
    consts.fill(0, 0, L.tapRowsWords);
    consts.fill(-1, L.tapRowsWords, 2 * L.tapRowsWords);
  }

  evalAll(x: BitplaneValue, batch: number): BitplaneValue {
    const L = this.L;
    // `?cpuprofile=1` のときだけ JS↔wasmヒープ往復と純カーネル時間を分ける。
    const prof = cpuProfilingEnabled();
    const words = new Int32Array(batch * L.outPlaneWords);
    for (let b = 0; b < batch; b++) {
      const xoff = b * L.inPlaneWords;
      const tIn = prof ? performance.now() : 0;
      new Int32Array(this.mod.memory.buffer, this.xPtr, L.inPlaneWords)
        .set(x.words.subarray(xoff, xoff + L.inPlaneWords));
      if (prof) addCpuSample('conv.copyIn', performance.now() - tIn);
      const tK = prof ? performance.now() : 0;
      this.mod.evalConvProgBatch!(
        this.xPtr,
        this.planeCPtr, this.planeDyPtr, this.planeDxPtr, L.nPlanes,
        this.instrPtr, this.nInstr,
        this.writeStartPtr, this.writeOcPtr, this.ocSrcOffPtr,
        this.scratchPtr, this.outPtr,
        L.H, L.W, L.HOut, L.WOut,
        L.sH, L.sW, L.pH, L.pW,
        L.wprIn, L.wprOut, L.tapWpr, L.maskOut, L.useFastPath ? 1 : 0,
        L.poolBase, L.planeStride,
      );
      if (prof) addCpuSample('conv.wasm', performance.now() - tK);
      const tOut = prof ? performance.now() : 0;
      words.set(
        new Int32Array(this.mod.memory.buffer, this.outPtr, L.outPlaneWords),
        b * L.outPlaneWords,
      );
      if (prof) addCpuSample('conv.copyOut', performance.now() - tOut);
    }
    return { words, C: L.OC, H: L.HOut, W: L.WOut, wpr: L.wprOut };
  }
}

/** 1つの conv2d ノードぶんの WASM 実行カーネル(bitplane-wasm)。
    既定は命令列経路。命令列の構築/セルフチェックに失敗した場合は黙殺せず
    console.error に残した上で旧経路(バランス木全ゲート評価)へ落ちる。 */
export class WasmBitplaneConvKernel {
  private readonly impl: KernelImpl;
  /** 命令列経路で動いているか(テスト/計測用)。 */
  readonly usesProgram: boolean;
  /** 旧経路へ落ちた理由(落ちていなければ null)。 */
  readonly fallbackNote: string | null = null;

  /** `prebuilt`(bundleProgram.ts::resolveConvProgram が検証済みの
      命令列)が渡されれば buildConvProgramLayout はそれを使い、
      buildConvProgram/verifyConvProgram を再実行しない。 */
  constructor(mod: WasmModule, p: Conv2dParams, useProgram = true, prebuilt?: ConvProgram) {
    let impl: KernelImpl | null = null;
    if (useProgram && typeof mod.evalConvProgBatch === 'function') {
      try {
        impl = new ProgWasmBitplaneKernel(mod, buildConvProgramLayout(p, prebuilt));
      } catch (e) {
        this.fallbackNote = `conv2d命令列化に失敗→旧バランス木経路へフォールバック: ${String(e)}`;
        console.error(`[bitplane-wasm] ${this.fallbackNote}`);
        impl = null;
      }
    }
    this.usesProgram = impl !== null;
    this.impl = impl ?? new LegacyWasmBitplaneKernel(mod, p);
  }

  /** BitplaneValue([batch][C][H][wprIn])からこのノードの出力
      BitplaneValue([batch][OC][HOut][wprOut])を計算する(evalConv2dBitplane
      と同一の出力並び)。 */
  evalAll(x: BitplaneValue, batch: number): BitplaneValue {
    return this.impl.evalAll(x, batch);
  }
}
