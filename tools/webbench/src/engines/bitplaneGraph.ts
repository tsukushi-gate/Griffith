// IR全体をビットプレーンのまま実行するグラフ実行器。
//
// irflat.ts::evalIrFlat はノード単位で NodeValue(非パック 0/1 flat)を
// やり取りする。bitplaneエンジンは conv2d/or_pool2d(2x2)/upsample2d(x2)/
// permute(チャネルのみ)を BitplaneValue(bitplane.ts、行頭ワード境界の
// ビットプレーン)のまま連結し、パック↔アンパックの往復を最小化する。
// それ以外のノード型(lut_layer/group_sum/concat/add/binarize/flatten/
// constant)は irflat.ts の実装へ委譲する(境界でアンパック、正しさの
// 実装重複を避けるため irflat.ts から直接 export された関数を使う)。
//
// 読み出しヘッドのパックネイティブ化。C1完了後の実測で、
// conv2dカーネルは wasm エンジンでも全体の1〜4割しか占めず、残りが
// 「非convのグラフ処理」だった。その支配項を順にネイティブ化した:
//   - SpatialGroupSum(permute[0,2,3,1] + group_sum): アンパックせず
//     ビットスライス縦和で直接スコア整数へ(spatialGroupSumBitplane)
//   - 空間形状が揃った concat: チャネル平面連結(concatPlanes)
//   - グラフ入力ポート: Int32Array へ詰め替えずそのまま packPlanes へ
// いずれもフォールバック(アンパック→irflat)を残しており、条件に合わない
// グラフでは従来経路がそのまま使われる。
//
// 正しさの錨: conv_golden.json / seg_e2e_golden.json / det_e2e_golden.json
// を 'bitplane' エンジンでも他エンジンと整数完全一致で通す
// (conv.test.ts / seg_e2e.test.ts / detdecode.test.ts の ENGINES 行列)。
import type {
  AddParams,
  Conv2dParams,
  GroupSumParams,
  IRGraphJson,
  IRNodeJson,
  LutLayerParams,
  OrPool2dParams,
  PermuteParams,
  UpSample2dParams,
} from '../shared';
import type { BitplaneValue, PackSource } from './bitplane';
import {
  concatPlanes, copyPlanes, dilateX2, evalConv2dBitplane, orPool2x2, packPlanes,
  spatialGroupSumBitplane, unpackPlanes,
} from './bitplane';
import {
  collectInputSources, evalAdd, evalConcat, evalGroupSum, evalLutLayer, topologicalOrder,
} from './irflat';
import type { NodeValue } from './irflat';
import { evalPermutePacked, evalUpsample2dPacked } from './packedConv';
import { evalLutFlat, evalLutFlatGroupSum } from './lutFlat';
import type { LutFlatParams } from './lutFlat';
import { addCpuSample, beginCpuFrame, cpuProfilingEnabled, endCpuFrame, timed } from './cpuProfile';

/** conv2d ノード評価の差し替え口(irflat.ts::ConvNodeEval と同じ設計)。
    既定は evalConv2dBitplane(純JS)。'bitplane-wasm' エンジンは
    ノードidごとのWASMカーネルを引き当てる関数をここへ注入する
    (bitplaneGraph.ts自体のグラフ走査・or_pool2d/upsample2d/permute等の
    扱いは完全に共通、conv2dの計算だけがJS/WASMで切り替わる)。 */
export type BitplaneConvEval = (node: IRNodeJson, x: BitplaneValue, batch: number) => BitplaneValue;

const defaultBitplaneConvEval: BitplaneConvEval = (node, x, batch) =>
  evalConv2dBitplane(node.params as unknown as Conv2dParams, x, batch);

/** conv2d 以外の「ビットプレーンのまま計算できるノード/配線」の
    差し替え口。'bitplane-wasm' エンジンは WasmBitplaneOps(必要ならワーカー
    プール付き)を注入する。いずれも **null を返したら JS 実装へ落ちる**
    契約なので、扱えない形状・古い .wasm・SAB非対応環境のいずれでも
    グラフ走査のコードは1本のまま変わらない。 */
export interface BitplaneNativeOps {
  pack?(
    x: PackSource, batch: number, C: number, H: number, W: number,
  ): BitplaneValue | null;
  orPool?(bp: BitplaneValue, batch: number): BitplaneValue | null;
  /** lut_flat → group_sum 融合(wasm エンジンは wasmLutFlat.ts)。*/
  lutFlatGroupSum?(
    p: LutFlatParams, x: { data: ArrayLike<number>; width: number }, batch: number, k: number,
  ): NodeValue | null;
}

/** グラフ入力ポートの生ビット列。conv2d/permute/or_pool2d/
    upsample2d のいずれかが最初に消費する限り、Int32Array へ詰め替えずに
    packPlanes へ直接渡せる(数百万要素の詰め替えを1回まるごと省ける)。
    非ネイティブノードが消費した場合のみ toNodeValue が Int32Array を
    実体化する(従来と同じ値)。 */
interface RawInputValue {
  rawInput: true;
  data: ArrayLike<number>;
  width: number;
}

/** SpatialGroupSum型 permute(空間軸を跨ぐ [0,2,3,1])の遅延表現。
    この permute 単体はビットプレーンで表現できない(空間軸とチャネル軸が
    混ざる)が、直後の group_sum と融合すれば「チャネル平面のビット縦和」
    へ還元できる(bitplane.ts::spatialGroupSumBitplane)。そこで permute
    ノードでは計算せず入力平面をそのまま持ち回り、group_sum ノードが
    融合形を検出したときだけネイティブ評価する。
    group_sum 以外の消費者(概念上ありうるが実モデルには存在しない)へ
    渡った場合は toNodeValue が従来どおり「アンパック + evalPermutePacked」
    を実行するので、正しさは経路によらず保たれる。 */
interface DeferredSpatialPermute {
  spatialPermute: true;
  bp: BitplaneValue;
  params: PermuteParams;
  nOut: number;
  gs: number;
}

/** lut_flat の遅延表現。直後の group_sum が融合形
    (lutFlat.ts::evalLutFlatGroupSum)で消費すれば、out_width 本の出力ビット列を
    Int32 で実体化せずに済む(HAR k6000・1枚で出力の書き出しと group_sum の
    読み直しが網の評価の約 8 割の重さだった、実測)。group_sum 以外の
    消費者・出力ポートへ渡ったときは toNodeValue が evalLutFlat で実体化する
    (結果は1度だけ作って持ち回る)ので、正しさは経路によらず保たれる。 */
interface DeferredLutFlat {
  lutFlat: true;
  params: LutFlatParams;
  x: { data: ArrayLike<number>; width: number };
  width: number;
  materialized?: NodeValue;
}

/** グラフ内のノード値: ネイティブ路にいる間はBitplaneValue、SpatialGroupSum型
    permuteの直後だけ DeferredSpatialPermute、lut_flat の直後は DeferredLutFlat、
    それ以外は従来通りの NodeValue(非パック 0/1 flat)。 */
export type GraphValue =
  NodeValue | BitplaneValue | DeferredSpatialPermute | RawInputValue | DeferredLutFlat;

function isDeferredLutFlat(v: GraphValue): v is DeferredLutFlat {
  return (v as DeferredLutFlat).lutFlat === true;
}

function isDeferredSpatialPermute(v: GraphValue): v is DeferredSpatialPermute {
  return (v as DeferredSpatialPermute).spatialPermute === true;
}

function isRawInput(v: GraphValue): v is RawInputValue {
  return (v as RawInputValue).rawInput === true;
}

function isBitplane(v: GraphValue): v is BitplaneValue {
  return (v as BitplaneValue).wpr !== undefined;
}

function toNodeValue(v: GraphValue, batch: number): NodeValue {
  if (isDeferredSpatialPermute(v)) {
    return timed('unpack', () => evalPermutePacked(v.params, unpackPlanes(v.bp, batch), batch));
  }
  if (isBitplane(v)) return timed('unpack', () => unpackPlanes(v, batch));
  if (isDeferredLutFlat(v)) {
    v.materialized ??= evalLutFlat(v.params, v.x, batch);
    return v.materialized;
  }
  if (isRawInput(v)) {
    const data = new Int32Array(v.data.length);
    data.set(v.data);
    return { data, width: v.width };
  }
  return v;
}

/** ネイティブ路の入口: どの表現で来ても BitplaneValue へ揃える
    (既にビットプレーンならゼロコピー、生入力なら詰め替えなしで直接パック)。 */
function toBitplane(
  v: GraphValue, batch: number, C: number, H: number, W: number,
  ops?: BitplaneNativeOps,
): BitplaneValue {
  if (isBitplane(v)) return v;
  const pack = (src: PackSource, label: string): BitplaneValue => timed(
    label, () => ops?.pack?.(src, batch, C, H, W) ?? packPlanes(src, batch, C, H, W),
  );
  if (isRawInput(v)) return pack(v, 'pack.input');
  return pack(toNodeValue(v, batch), 'pack.boundary');
}

/** 表現によらないバッチ1要素あたりの要素幅。 */
function graphWidth(v: GraphValue): number {
  if (isDeferredSpatialPermute(v)) return v.bp.C * v.bp.H * v.bp.W;
  return isBitplane(v) ? v.C * v.H * v.W : v.width;
}

/** or_pool2d の一般形(kernel/stride/padding任意)を非パックのまま素朴に
    評価する安全策。bitplaneGraphは通常 kernel=2,stride=2,padding=0の
    or_pool2dしかネイティブ路で受け取らない設計(lowerConv.tsのnativeOrPool
    がそれ以外を全てlut_layer展開する)だが、万一その前提が崩れた場合でも
    黙って誤った結果を返さないための正しさの最終防衛線として実装する
    (conv.test.tsのproperty testが使うnaiveOrPoolと同じ幾何)。 */
function evalOrPool2dGeneric(p: OrPool2dParams, x: NodeValue, batch: number): NodeValue {
  const [C, H, W] = p.in_shape;
  const k = p.kernel_size;
  const s = p.stride;
  const pad = p.padding ?? 0;
  const HOut = Math.floor((H + 2 * pad - k) / s) + 1;
  const WOut = Math.floor((W + 2 * pad - k) / s) + 1;
  const inW = x.width;
  const outW = C * HOut * WOut;
  const data = new Int32Array(batch * outW);
  for (let b = 0; b < batch; b++) {
    const xo = b * inW;
    const oo = b * outW;
    for (let c = 0; c < C; c++) {
      for (let oy = 0; oy < HOut; oy++) {
        for (let ox = 0; ox < WOut; ox++) {
          let v = 0;
          for (let dy = 0; dy < k; dy++) {
            for (let dx = 0; dx < k; dx++) {
              const y = oy * s - pad + dy;
              const xx = ox * s - pad + dx;
              if (y >= 0 && y < H && xx >= 0 && xx < W) v |= x.data[xo + c * H * W + y * W + xx];
            }
          }
          data[oo + c * HOut * WOut + oy * WOut + ox] = v;
        }
      }
    }
  }
  return { data, width: outW };
}

/** permuteが「末尾2軸(H,W)がそのまま(perm[n-2]===n-2 && perm[n-1]===n-1)」
    の純チャネル並べ替えかどうか。SpatialGroupSum型permute([0,2,3,1]等、
    空間軸が別位置へ動く)はこれを満たさないため必ずfalseになる
    (lowering.pyの_extract_spatial_group_sum_native: in_shape=[nOut,gs,H,W],
    perm=[0,2,3,1] → perm[2]=3≠2で不適格。thermo channel permuteの
    in_shape=[n_rgb,n_thr,H,W], perm=[1,0,2,3] → perm[2]=2,perm[3]=3で適格)。 */
function permuteKeepsTrailingSpatial(p: PermuteParams): boolean {
  const n = p.in_shape.length;
  if (n < 2) return false;
  return p.perm[n - 2] === n - 2 && p.perm[n - 1] === n - 1;
}

/** permuteが SpatialGroupSum 由来の形
    (in_shape=[nOut,gs,H,W], perm=[0,2,3,1])かどうか。合致すれば次元を返す。
    lowering.py::_extract_spatial_group_sum_native が出す唯一の形なので
    完全一致でよい(他の空間跨ぎpermuteは従来どおりアンパック経路)。 */
function spatialGroupSumPermuteDims(
  p: PermuteParams,
): { nOut: number; gs: number; H: number; W: number } | null {
  if (p.in_shape.length !== 4 || p.perm.length !== 4) return null;
  if (p.perm[0] !== 0 || p.perm[1] !== 2 || p.perm[2] !== 3 || p.perm[3] !== 1) return null;
  const [nOut, gs, H, W] = p.in_shape;
  return { nOut, gs, H, W };
}

/** 適格なpermuteの「先頭(チャネル)軸だけのindex変換」を、チャネル本数
    ぶんの具体的なorder配列へ展開する(copyPlanesへ渡す形)。
    evalPermutePacked(packedConv.ts)の一般index計算を先頭軸だけに
    限定した特殊化(数式は同一)。 */
function channelOrderFromPermute(p: PermuteParams): number[] {
  const n = p.in_shape.length;
  const m = n - 2; // 先頭(非空間)軸数
  const leadDims = p.in_shape.slice(0, m);
  const leadPerm = p.perm.slice(0, m);
  const total = leadDims.reduce((a, b) => a * b, 1);
  const inStrides = new Array<number>(m).fill(1);
  for (let i = m - 2; i >= 0; i--) inStrides[i] = inStrides[i + 1] * leadDims[i + 1];
  const outDims = leadPerm.map((q) => leadDims[q]);
  const outStrides = new Array<number>(m).fill(1);
  for (let i = m - 2; i >= 0; i--) outStrides[i] = outStrides[i + 1] * outDims[i + 1];
  const order = new Array<number>(total).fill(0);
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

/** ノード単位の実行時間フック(オプトイン、既定undefined=無効)。
    webgpuGraph.ts の setProfiling/getBreakdownに対応するCPU側の
    計測口。渡された場合のみ performance.now() を2回呼ぶので、未指定時の
    コストはノードあたり1分岐(ノード数は十数個なので無視できる)。 */
export type BitplaneNodeTimeHook = (node: IRNodeJson, ms: number) => void;

/** IR全体をビットプレーン優先で実行する(evalIrFlatのbitplane版)。
    'flatten' は入力がBitplaneValueならこの時点でアンパックする(以降の
    全ノード型はNodeValue前提のirflat実装へ委譲するための境界)。 */
export function evalIrBitplane(
  ir: IRGraphJson,
  inputs: Record<string, ArrayLike<number>>,
  batch: number,
  convEval: BitplaneConvEval = defaultBitplaneConvEval,
  onNodeTime?: BitplaneNodeTimeHook,
  ops?: BitplaneNativeOps,
  /** 既にビットプレーンへパック済みの入力ポート(融合前処理の出力)。
      与えられたポートは inputs 側を参照せず、そのままグラフの値として使う
      (下流のノードは表現でしか分岐しないので、経路は完全に共通のまま)。 */
  packedInputs?: Record<string, BitplaneValue>,
): { values: Record<string, GraphValue>; output: NodeValue } {
  // `?cpuprofile=1` のときだけフレーム内訳を記録する(既定は分岐1回)。
  const prof = cpuProfilingEnabled();
  const tFrame = prof ? performance.now() : 0;
  if (prof) beginCpuFrame();
  const values: Record<string, GraphValue> = {};
  for (const port of ir.inputs) {
    const packed = packedInputs?.[port.id];
    if (packed) {
      if (packed.C * packed.H * packed.W !== port.width) {
        throw new Error(
          `入力ポート${port.id}(pack済): 期待幅${port.width}, `
          + `実際${packed.C * packed.H * packed.W}`,
        );
      }
      values[port.id] = packed;
      continue;
    }
    const bits = inputs[port.id];
    if (!bits || bits.length !== batch * port.width) {
      throw new Error(
        `入力ポート${port.id}: 期待${batch * port.width}要素, 実際${bits?.length ?? 0}`,
      );
    }
    values[port.id] = { rawInput: true, data: bits, width: port.width };
  }

  const inSrcs = collectInputSources(ir);
  for (const node of topologicalOrder(ir, inSrcs)) {
    const srcs = inSrcs.get(node.id) ?? [];
    const ins = srcs.map((s) => values[s]);
    const x = ins[0];
    let out: GraphValue;
    const t0 = (onNodeTime || prof) ? performance.now() : 0;
    switch (node.type as string) {
      case 'conv2d': {
        const p = node.params as unknown as Conv2dParams;
        const [C, H, W] = p.in_shape;
        out = convEval(node, toBitplane(x, batch, C, H, W, ops), batch);
        break;
      }
      case 'or_pool2d': {
        const p = node.params as unknown as OrPool2dParams;
        const [C, H, W] = p.in_shape;
        const pad = p.padding ?? 0;
        if (p.kernel_size === 2 && p.stride === 2 && pad === 0) {
          const bpIn = toBitplane(x, batch, C, H, W, ops);
          out = ops?.orPool?.(bpIn, batch) ?? orPool2x2(bpIn, batch);
        } else {
          // 正しさ優先のフォールバック(通常はlowerConv.nativeOrPoolが
          // この形のor_pool2dを常にlut_layer展開するため到達しない)。
          out = evalOrPool2dGeneric(p, toNodeValue(x, batch), batch);
        }
        break;
      }
      case 'upsample2d': {
        const p = node.params as unknown as UpSample2dParams;
        const [C, H, W] = p.in_shape;
        if (p.scale === 2) {
          out = dilateX2(toBitplane(x, batch, C, H, W, ops), batch);
        } else {
          out = evalUpsample2dPacked(p, toNodeValue(x, batch), batch);
        }
        break;
      }
      case 'permute': {
        const p = node.params as unknown as PermuteParams;
        if (permuteKeepsTrailingSpatial(p)) {
          const n = p.in_shape.length;
          const H = p.in_shape[n - 2];
          const W = p.in_shape[n - 1];
          const C = p.in_shape.slice(0, n - 2).reduce((a, b) => a * b, 1);
          const width = C * H * W;
          const xWidth = graphWidth(x);
          if (width === xWidth && (!isBitplane(x) || (x.H === H && x.W === W && x.C === C))) {
            out = copyPlanes(
              toBitplane(x, batch, C, H, W, ops), batch, channelOrderFromPermute(p),
            );
            break;
          }
        }
        // SpatialGroupSum型(perm=[0,2,3,1])は評価を後段の
        // group_sum まで遅延させ、融合してパックのまま総和する。
        const sg = spatialGroupSumPermuteDims(p);
        if (sg && isBitplane(x) && x.C === sg.nOut * sg.gs && x.H === sg.H && x.W === sg.W) {
          out = { spatialPermute: true, bp: x, params: p, nOut: sg.nOut, gs: sg.gs };
          break;
        }
        // 上記以外の空間軸を跨ぐpermute、または形状不一致は必ずアンパックして
        // 既存の汎用実装へ(正しさ優先のアンカー)。
        out = evalPermutePacked(p, toNodeValue(x, batch), batch);
        break;
      }
      case 'lut_layer':
        out = evalLutLayer(node.params as unknown as LutLayerParams, toNodeValue(x, batch), batch);
        break;
      case 'lut_flat': {
        // 評価は消費者まで遅らせる(DeferredLutFlat)。生入力はコピーせず
        // 直接読む(toNodeValue の Int32 化は幅 4,608 の HAR で網の評価より
        // 重かった、実測)
        const p = node.params as unknown as LutFlatParams;
        out = {
          lutFlat: true, params: p, width: p.out_width,
          x: isRawInput(x) ? x : toNodeValue(x, batch),
        };
        break;
      }
      case 'group_sum': {
        const p = node.params as unknown as GroupSumParams;
        // 遅延された SpatialGroupSum型 permute との融合ネイティブ路。
        // k が (nOut*H*W) と一致することまで確認してから使う(一致しない
        // group_sum が同じ permute を消費することは実モデルには無いが、
        // 意味論が変わるので必ず検査してからネイティブ路へ入る)。
        if (isDeferredLutFlat(x) && !x.materialized) {
          out = ops?.lutFlatGroupSum?.(x.params, x.x, batch, p.k)
            ?? evalLutFlatGroupSum(x.params, x.x, batch, p.k);
          break;
        }
        if (isDeferredSpatialPermute(x) && p.k === x.nOut * x.bp.H * x.bp.W) {
          out = spatialGroupSumBitplane(x.bp, batch, x.nOut, x.gs);
          break;
        }
        out = evalGroupSum(p, toNodeValue(x, batch), batch);
        break;
      }
      case 'flatten':
        out = toNodeValue(x, batch);
        break;
      case 'concat': {
        // 全入力がビットプレーンかつ空間形状(H,W)が揃っていれば
        // 平面連結で済ませる(UNetのskip接続。従来はアンパック→int32連結→
        // 次のconv2dで再パック、という往復が発生していた)。
        const first = ins[0];
        if (
          ins.every(isBitplane)
          && isBitplane(first)
          && ins.every((v) => isBitplane(v) && v.H === first.H && v.W === first.W)
        ) {
          out = concatPlanes(ins as BitplaneValue[], batch);
          break;
        }
        out = evalConcat(ins.map((v) => toNodeValue(v, batch)), batch);
        break;
      }
      case 'add':
        out = evalAdd(
          (node.params as unknown as AddParams).mode,
          ins.map((v) => toNodeValue(v, batch)),
        );
        break;
      case 'binarize': {
        const nv = toNodeValue(x, batch);
        const thr = (node.params.threshold as number | undefined) ?? 0.5;
        const data = new Int32Array(nv.data.length);
        for (let i = 0; i < data.length; i++) data[i] = nv.data[i] > thr ? 1 : 0;
        out = { data, width: nv.width };
        break;
      }
      case 'constant': {
        const bits = node.params.bits as number[];
        const w = bits.length;
        const data = new Int32Array(batch * w);
        for (let b = 0; b < batch; b++) data.set(bits, b * w);
        out = { data, width: w };
        break;
      }
      default:
        throw new Error(`未対応ノード型: ${node.id}(${node.type})`);
    }
    if (onNodeTime || prof) {
      const dt = performance.now() - t0;
      if (onNodeTime) onNodeTime(node, dt);
      if (prof) addCpuSample(`node.${node.type}`, dt);
    }
    values[node.id] = out;
  }

  const outPort = ir.outputs[0];
  const outSrc = inSrcs.get(outPort.id)?.[0];
  const finalValue = outSrc !== undefined
    ? values[outSrc]
    : values[ir.nodes[ir.nodes.length - 1].id];
  const output = timed('out.toNodeValue', () => toNodeValue(finalValue, batch));
  if (prof) endCpuFrame(performance.now() - tFrame);
  return { values, output };
}
