// フラットIR(lut_layer/lut_flat/group_sum/binarize/flatten/concat/add/constant)の
// バッチ実行器。griffith_logic/ir/simulate.py の simulate_trace 移植。
// conv2d/upsample2d/permute はネイティブ実行(packedConv.ts)、or_pool2d は
// lowerConv.lowerToPrimitives で lut_layer 化してから渡す(upsample2d/
// permuteを追加。packedエンジンはnativeConv=trueで未展開のまま渡ってくる)。
// 値は各ノード [batch, width] を row-major の Int32Array で保持(整数厳密)。
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
import { evalConv2dPacked, evalPermutePacked, evalUpsample2dPacked } from './packedConv';
import { evalLutFlat } from './lutFlat';
import type { LutFlatParams } from './lutFlat';

export interface NodeValue {
  data: Int32Array; // [batch * width]
  width: number;
}

/** dst_slot昇順の入力元マップ(欠落slot=0)。webgpuGraph.tsのグラフ常駐
    executorも同一のグラフ構造把握が要るためexportし、実装の重複(=乖離
    リスク)を避ける。 */
export function collectInputSources(ir: IRGraphJson): Map<string, string[]> {
  const grouped = new Map<string, { src: string; slot: number }[]>();
  for (const e of ir.edges) {
    const slot = e.dst_slot ?? 0;
    const list = grouped.get(e.dst) ?? [];
    list.push({ src: e.src, slot });
    grouped.set(e.dst, list);
  }
  const out = new Map<string, string[]>();
  for (const [dst, list] of grouped) {
    out.set(dst, [...list].sort((a, b) => a.slot - b.slot).map((x) => x.src));
  }
  return out;
}

/** エッジ基準のトポロジカル順(nodes配列順に依存しない)。export(理由は
    collectInputSourcesと同じ)。 */
export function topologicalOrder(
  ir: IRGraphJson,
  inSrcs: Map<string, string[]>,
): IRNodeJson[] {
  const ready = new Set<string>(ir.inputs.map((p) => p.id));
  const pending = new Map(ir.nodes.map((n) => [n.id, n]));
  const order: IRNodeJson[] = [];
  let progressed = true;
  while (progressed && pending.size > 0) {
    progressed = false;
    for (const [id, node] of pending) {
      const srcs = inSrcs.get(id) ?? [];
      if (!srcs.every((s) => ready.has(s))) continue;
      order.push(node);
      ready.add(id);
      pending.delete(id);
      progressed = true;
    }
  }
  if (pending.size > 0) {
    throw new Error(`評価不能(未接続または循環): ${[...pending.keys()].join(', ')}`);
  }
  return order;
}

// bitplaneGraph.tsがアンパック境界での「既存irflat実装への委譲」を
// 実装の重複(=乖離リスク)なく行うため export する(collectInputSources /
// topologicalOrder と同じ理由、W10のコメント参照)。
/** n<=5 の真理値表を Uint32Array へ一度だけ変換して params ごとに持つ
    (以前は呼ぶたびに ttParse(BigInt)し直しており、HAR k6000 の
    1枚推論の大半がこの変換だった)。 */
const tts32Cache = new WeakMap<object, Uint32Array>();
function lutTts32(p: LutLayerParams): Uint32Array {
  let tts = tts32Cache.get(p);
  if (!tts) {
    const k = p.out_width;
    tts = new Uint32Array(k);
    for (let o = 0; o < k; o++) tts[o] = Number(ttParse(p.truth_tables[o]));
    tts32Cache.set(p, tts);
  }
  return tts;
}

export function evalLutLayer(p: LutLayerParams, x: NodeValue, batch: number): NodeValue {
  const n = p.n_inputs;
  const k = p.out_width;
  const inW = x.width;
  const idx = p.indices; // [n][k]
  const data = new Int32Array(batch * k);
  // n<=5 は真理値表が2^n<=32エントリなので uint32 に収まり、BigInt を回避できる
  // (chanperm/upsample/permute=1入力, or_pool=4入力 など実モデルは全て n<=5)。
  if (n <= 5) {
    const tts = lutTts32(p);
    for (let b = 0; b < batch; b++) {
      const xo = b * inW;
      const oo = b * k;
      for (let o = 0; o < k; o++) {
        let addr = 0;
        for (let j = 0; j < n; j++) addr |= x.data[xo + idx[j][o]] << j;
        data[oo + o] = (tts[o] >>> addr) & 1;
      }
    }
    return { data, width: k };
  }
  const tts = p.truth_tables.map((h) => ttParse(h));
  for (let b = 0; b < batch; b++) {
    const xo = b * inW;
    const oo = b * k;
    for (let o = 0; o < k; o++) {
      let addr = 0;
      for (let j = 0; j < n; j++) addr |= x.data[xo + idx[j][o]] << j;
      data[oo + o] = Number((tts[o] >> BigInt(addr)) & 1n);
    }
  }
  return { data, width: k };
}

export function evalGroupSum(p: GroupSumParams, x: NodeValue, batch: number): NodeValue {
  const k = p.k;
  const inW = x.width;
  if (inW % k !== 0) throw new Error(`group_sum: 幅${inW}がk=${k}で割り切れない`);
  const gsz = inW / k;
  const data = new Int32Array(batch * k);
  for (let b = 0; b < batch; b++) {
    const xo = b * inW;
    const oo = b * k;
    for (let g = 0; g < k; g++) {
      let s = 0;
      const base = xo + g * gsz;
      for (let i = 0; i < gsz; i++) s += x.data[base + i];
      data[oo + g] = s;
    }
  }
  return { data, width: k };
}

export function evalConcat(ins: NodeValue[], batch: number): NodeValue {
  const totalW = ins.reduce((a, v) => a + v.width, 0);
  const data = new Int32Array(batch * totalW);
  for (let b = 0; b < batch; b++) {
    let off = b * totalW;
    for (const v of ins) {
      data.set(v.data.subarray(b * v.width, (b + 1) * v.width), off);
      off += v.width;
    }
  }
  return { data, width: totalW };
}

export function evalAdd(mode: string, ins: NodeValue[]): NodeValue {
  const w = ins[0].width;
  const data = new Int32Array(ins[0].data); // copy slot0
  for (let t = 1; t < ins.length; t++) {
    const src = ins[t].data;
    for (let i = 0; i < data.length; i++) {
      if (mode === 'or') data[i] |= src[i];
      else if (mode === 'xor') data[i] ^= src[i];
      else data[i] += src[i];
    }
  }
  return { data, width: w };
}

/** conv2d ノード評価の差し替え口。既定は evalConv2dPacked(純JS)。
    packed-wasm/packed-webgpuエンジンは ConvRuntime がノードidごとの
    カーネルを引き当てる関数をここへ注入する。 */
export type ConvNodeEval = (
  node: IRNodeJson,
  x: NodeValue,
  batch: number,
) => NodeValue;

const defaultConvEval: ConvNodeEval = (node, x, batch) =>
  evalConv2dPacked(node.params as unknown as Conv2dParams, x, batch);

/** IRを実行し全ノードの値を返す。inputs は portId → [batch*width] の 0/1。 */
export function evalIrFlat(
  ir: IRGraphJson,
  inputs: Record<string, ArrayLike<number>>,
  batch: number,
  convEval: ConvNodeEval = defaultConvEval,
): { values: Record<string, NodeValue>; output: NodeValue } {
  const values: Record<string, NodeValue> = {};
  for (const port of ir.inputs) {
    const bits = inputs[port.id];
    if (!bits || bits.length !== batch * port.width) {
      throw new Error(
        `入力ポート${port.id}: 期待${batch * port.width}要素, 実際${bits?.length ?? 0}`,
      );
    }
    values[port.id] = { data: Int32Array.from(bits), width: port.width };
  }

  const inSrcs = collectInputSources(ir);
  for (const node of topologicalOrder(ir, inSrcs)) {
    const srcs = inSrcs.get(node.id) ?? [];
    const ins = srcs.map((s) => values[s]);
    const x = ins[0];
    let out: NodeValue;
    // 'constant' はeditor側schema(IRNodeType)に未定義のためstringで分岐
    switch (node.type as string) {
      case 'lut_layer':
        out = evalLutLayer(node.params as unknown as LutLayerParams, x, batch);
        break;
      case 'lut_flat':
        out = evalLutFlat(node.params as unknown as LutFlatParams, x, batch);
        break;
      case 'conv2d':
        out = convEval(node, x, batch);
        break;
      case 'upsample2d':
        out = evalUpsample2dPacked(node.params as unknown as UpSample2dParams, x, batch);
        break;
      case 'permute':
        out = evalPermutePacked(node.params as unknown as PermuteParams, x, batch);
        break;
      case 'group_sum':
        out = evalGroupSum(node.params as unknown as GroupSumParams, x, batch);
        break;
      case 'flatten':
        out = { data: x.data, width: x.width };
        break;
      case 'concat':
        out = evalConcat(ins, batch);
        break;
      case 'add':
        out = evalAdd((node.params as unknown as AddParams).mode, ins);
        break;
      case 'binarize': {
        const thr = (node.params.threshold as number | undefined) ?? 0.5;
        const data = new Int32Array(x.data.length);
        for (let i = 0; i < data.length; i++) data[i] = x.data[i] > thr ? 1 : 0;
        out = { data, width: x.width };
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
    values[node.id] = out;
  }

  // 出力ポート: 最初の出力の接続元
  const outPort = ir.outputs[0];
  const outSrc = inSrcs.get(outPort.id)?.[0];
  const output = outSrc !== undefined
    ? values[outSrc]
    : values[ir.nodes[ir.nodes.length - 1].id];
  return { values, output };
}

/** conv2d ノード評価の非同期差し替え口(packed-webgpu用)。
    WebGPU の readback(mapAsync)は本質的に非同期なため、conv2dノードだけ
    await する。他ノードの評価ロジックは evalIrFlat と完全に同一
    (このファイル内の非exportヘルパーを直接呼ぶことで一致を保証)。
    evalIrFlat にノード型を追加した場合はこちらにも追加すること。 */
export type ConvNodeEvalAsync = (
  node: IRNodeJson,
  x: NodeValue,
  batch: number,
) => Promise<NodeValue>;

export async function evalIrFlatAsync(
  ir: IRGraphJson,
  inputs: Record<string, ArrayLike<number>>,
  batch: number,
  convEval: ConvNodeEvalAsync,
): Promise<{ values: Record<string, NodeValue>; output: NodeValue }> {
  const values: Record<string, NodeValue> = {};
  for (const port of ir.inputs) {
    const bits = inputs[port.id];
    if (!bits || bits.length !== batch * port.width) {
      throw new Error(
        `入力ポート${port.id}: 期待${batch * port.width}要素, 実際${bits?.length ?? 0}`,
      );
    }
    values[port.id] = { data: Int32Array.from(bits), width: port.width };
  }

  const inSrcs = collectInputSources(ir);
  for (const node of topologicalOrder(ir, inSrcs)) {
    const srcs = inSrcs.get(node.id) ?? [];
    const ins = srcs.map((s) => values[s]);
    const x = ins[0];
    let out: NodeValue;
    switch (node.type as string) {
      case 'lut_layer':
        out = evalLutLayer(node.params as unknown as LutLayerParams, x, batch);
        break;
      case 'lut_flat':
        out = evalLutFlat(node.params as unknown as LutFlatParams, x, batch);
        break;
      case 'conv2d':
        out = await convEval(node, x, batch);
        break;
      case 'upsample2d':
        out = evalUpsample2dPacked(node.params as unknown as UpSample2dParams, x, batch);
        break;
      case 'permute':
        out = evalPermutePacked(node.params as unknown as PermuteParams, x, batch);
        break;
      case 'group_sum':
        out = evalGroupSum(node.params as unknown as GroupSumParams, x, batch);
        break;
      case 'flatten':
        out = { data: x.data, width: x.width };
        break;
      case 'concat':
        out = evalConcat(ins, batch);
        break;
      case 'add':
        out = evalAdd((node.params as unknown as AddParams).mode, ins);
        break;
      case 'binarize': {
        const thr = (node.params.threshold as number | undefined) ?? 0.5;
        const data = new Int32Array(x.data.length);
        for (let i = 0; i < data.length; i++) data[i] = x.data[i] > thr ? 1 : 0;
        out = { data, width: x.width };
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
    values[node.id] = out;
  }

  const outPort = ir.outputs[0];
  const outSrc = inSrcs.get(outPort.id)?.[0];
  const output = outSrc !== undefined
    ? values[outSrc]
    : values[ir.nodes[ir.nodes.length - 1].id];
  return { values, output };
}
