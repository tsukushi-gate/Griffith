// フォワード実行プレイグラウンドのクライアント評価エンジン(純関数)。
// griffith_logic/ir/simulate.py(golden model)のTS移植+soft(確率伝播)拡張。
//
// 規約:
//   - lut_layer(hard): addr = Σ_j x[indices[j][o]] << j、出力 = (tt >> addr) & 1
//   - lut_layer(soft): butterfly基底の確率伝播。b←[b*(1-x), b*x] を n_inputs回
//     繰り返し(基底インデックスkのbit j = スロットjの値)、出力 = Σ_k basis_k · w_k
//   - group_sum: k分割和(hardは整数和、softは/τ)
//   - binarize: しきい値超えで1 / flatten: 恒等
//   - 多入力ノード(将来のIR v0.2)はdst_slot昇順に入力を集める

import { ttBit } from '../ir/hexTT';
import type {
  AddParams,
  ConcatParams,
  ConstantParams,
  GroupSumParams,
  IRGraphJson,
  IRNodeJson,
  LutLayerParams,
  PermuteParams,
} from '../ir/schema';

export type EvalMode = 'hard' | 'soft';

/** レイヤ別のゲート確率(row-major out_width×bins)。
    bins=16はN=2の16ゲート分布(GATE_TABLE_N2で真理値表重みへ合成)、
    bins=2^n_inputsは真理値表ビット確率としてそのまま使う。
    それ以外の形状はIRのtruth_tablesへフォールバックする。 */
export interface LayerProbs {
  data: Float32Array;
  bins: number;
}

export interface EvalResult {
  /** 全ノードの出力値(入力ポートid・出力ポートidも含む)。
      hardは0/1(group_sum以降は整数スコア)、softは[0,1](group_sumは/τ後のスコア) */
  values: Record<string, Float32Array>;
  /** 出力ポート値(複数ポートはIR宣言順に連結) */
  output: Float32Array;
}

const N2_GATE_COUNT = 16;
const N2_PATTERN_COUNT = 4;
const BIT_ON_THRESHOLD = 0.5;

/** N=2の16ゲート真理値表(butterfly基底順)。GATE_TABLE_N2[g*4+k] = ゲートgのパターンk出力。
    core/basis.py の _build_gate_table_n2 と同一規約:
      k: bit j of k = スロットjの値(k=0→(0,0), k=1→(1,0), k=2→(0,1), k=3→(1,1))
      g: 真理値表を(0,0),(0,1),(1,0),(1,1)順にMSBから読んだ2進数(ir/gates.ts ttToGateIdN2の逆) */
export const GATE_TABLE_N2: Float32Array = (() => {
  const t = new Float32Array(N2_GATE_COUNT * N2_PATTERN_COUNT);
  for (let g = 0; g < N2_GATE_COUNT; g++) {
    t[g * N2_PATTERN_COUNT + 0] = (g >> 3) & 1; // (0,0)
    t[g * N2_PATTERN_COUNT + 1] = (g >> 1) & 1; // (1,0)
    t[g * N2_PATTERN_COUNT + 2] = (g >> 2) & 1; // (0,1)
    t[g * N2_PATTERN_COUNT + 3] = g & 1; //        (1,1)
  }
  return t;
})();

function clamp01(v: number): number {
  return Math.min(Math.max(v, 0), 1);
}

/** dst_slot昇順の入力元マップ(v0.1エッジはslot省略=0、同slotは宣言順で安定) */
function collectInputSources(ir: IRGraphJson): Map<string, string[]> {
  const grouped = new Map<string, { src: string; slot: number }[]>();
  for (const e of ir.edges) {
    const slot = (e as { dst_slot?: number }).dst_slot ?? 0;
    const list = grouped.get(e.dst) ?? [];
    list.push({ src: e.src, slot });
    grouped.set(e.dst, list);
  }
  const inSrcs = new Map<string, string[]>();
  for (const [dst, list] of grouped) {
    inSrcs.set(
      dst,
      [...list].sort((a, b) => a.slot - b.slot).map((x) => x.src),
    );
  }
  return inSrcs;
}

/** エッジ基準のトポロジカル順(nodes配列の並び順に依存しない) */
function topologicalOrder(
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

/** ニューロン別の真理値表重みw(out_width×2^n、row-major)。
    hardは確定化(16ゲートargmax / ビットしきい値0.5)、softは[0,1]の期待値。 */
export function layerWeights(
  p: LutLayerParams,
  probs: LayerProbs | undefined,
  mode: EvalMode,
): Float32Array {
  const size = 1 << p.n_inputs;
  const w = new Float32Array(p.out_width * size);

  // (a) 16ゲートモード(N=2): w = p @ GATE_TABLE_N2(hardはargmaxゲートの真理値表)
  if (
    probs &&
    probs.bins === N2_GATE_COUNT &&
    p.n_inputs === 2 &&
    probs.data.length === p.out_width * N2_GATE_COUNT
  ) {
    for (let o = 0; o < p.out_width; o++) {
      if (mode === 'hard') {
        let best = 0;
        for (let g = 1; g < N2_GATE_COUNT; g++) {
          if (probs.data[o * N2_GATE_COUNT + g] > probs.data[o * N2_GATE_COUNT + best]) {
            best = g;
          }
        }
        for (let k = 0; k < size; k++) {
          w[o * size + k] = GATE_TABLE_N2[best * N2_PATTERN_COUNT + k];
        }
      } else {
        for (let g = 0; g < N2_GATE_COUNT; g++) {
          const pg = probs.data[o * N2_GATE_COUNT + g];
          if (pg === 0) continue;
          for (let k = 0; k < size; k++) {
            w[o * size + k] += pg * GATE_TABLE_N2[g * N2_PATTERN_COUNT + k];
          }
        }
      }
    }
    return w;
  }

  // (b) 真理値表ビット確率(bins=2^n): そのままw(hardはしきい値0.5で確定化)
  if (probs && probs.bins === size && probs.data.length === p.out_width * size) {
    for (let i = 0; i < w.length; i++) {
      w[i] =
        mode === 'hard'
          ? probs.data[i] >= BIT_ON_THRESHOLD
            ? 1
            : 0
          : clamp01(probs.data[i]);
    }
    return w;
  }

  // (c) フォールバック: IRのtruth_tables(確定的)
  for (let o = 0; o < p.out_width; o++) {
    const tt = p.truth_tables[o] ?? '0x0';
    for (let k = 0; k < size; k++) {
      w[o * size + k] = ttBit(tt, k);
    }
  }
  return w;
}

function evalLutLayer(
  p: LutLayerParams,
  x: Float32Array,
  mode: EvalMode,
  probs: LayerProbs | undefined,
): Float32Array {
  const n = p.n_inputs;
  const size = 1 << n;
  const w = layerWeights(p, probs, mode);
  const out = new Float32Array(p.out_width);

  if (mode === 'hard') {
    for (let o = 0; o < p.out_width; o++) {
      let addr = 0;
      for (let j = 0; j < n; j++) {
        const src = p.indices[j]?.[o] ?? 0;
        if (x[src] >= BIT_ON_THRESHOLD) addr |= 1 << j;
      }
      out[o] = w[o * size + addr];
    }
    return out;
  }

  // soft: butterfly基底の確率伝播
  const basis = new Float32Array(size);
  for (let o = 0; o < p.out_width; o++) {
    basis[0] = 1;
    let len = 1;
    for (let j = 0; j < n; j++) {
      const xv = clamp01(x[p.indices[j]?.[o] ?? 0]);
      for (let k = 0; k < len; k++) {
        basis[k + len] = basis[k] * xv;
        basis[k] = basis[k] * (1 - xv);
      }
      len *= 2;
    }
    let acc = 0;
    for (let k = 0; k < size; k++) acc += basis[k] * w[o * size + k];
    out[o] = acc;
  }
  return out;
}

function evalGroupSum(p: GroupSumParams, x: Float32Array, mode: EvalMode): Float32Array {
  if (p.k <= 0 || x.length % p.k !== 0) {
    throw new Error(`group_sum: 入力幅${x.length}がk=${p.k}で割り切れません`);
  }
  const groupSize = x.length / p.k;
  const tau = p.tau || 1;
  const out = new Float32Array(p.k);
  for (let g = 0; g < p.k; g++) {
    // 判定バイアスは τ で割る前の整数和に足す(GroupSum.forward と同じ)
    let sum = p.bias?.[g] ?? 0;
    for (let i = 0; i < groupSize; i++) sum += x[g * groupSize + i];
    out[g] = mode === 'hard' ? sum : sum / tau;
  }
  return out;
}

/** permute: x を in_shape に見なし perm の軸順に並べ替えた恒等配線(hard/soft 共通) */
function evalPermute(p: PermuteParams, x: Float32Array): Float32Array {
  const inShape = p.in_shape.map(Number);
  const perm = p.perm.map(Number);
  const nd = inShape.length;
  const total = inShape.reduce((a, v) => a * v, 1);
  if (x.length !== total) {
    throw new Error(`permute: 入力幅${x.length}がin_shapeの積${total}と一致しません`);
  }
  // 入力の各軸のストライド(row-major)
  const inStride = new Array<number>(nd).fill(1);
  for (let a = nd - 2; a >= 0; a--) inStride[a] = inStride[a + 1] * inShape[a + 1];
  const outShape = perm.map((a) => inShape[a]);
  const out = new Float32Array(total);
  const idx = new Array<number>(nd).fill(0); // 出力側の多次元添字
  for (let o = 0; o < total; o++) {
    let src = 0;
    for (let a = 0; a < nd; a++) src += idx[a] * inStride[perm[a]];
    out[o] = x[src];
    // 出力添字をインクリメント(最後の軸が最も速く回る)
    for (let a = nd - 1; a >= 0; a--) {
      idx[a] += 1;
      if (idx[a] < outShape[a]) break;
      idx[a] = 0;
    }
  }
  return out;
}

function evalBinarize(node: IRNodeJson, x: Float32Array): Float32Array {
  const threshold = (node.params.threshold as number | undefined) ?? BIT_ON_THRESHOLD;
  const out = new Float32Array(x.length);
  for (let i = 0; i < x.length; i++) out[i] = x[i] > threshold ? 1 : 0;
  return out;
}

function evalConcat(p: ConcatParams, inputs: Float32Array[]): Float32Array {
  if (inputs.length !== p.in_widths.length) {
    throw new Error(`concat: 入力数${inputs.length}がin_widths数${p.in_widths.length}と一致しません`);
  }
  const total = inputs.reduce((acc, v) => acc + v.length, 0);
  const out = new Float32Array(total);
  let offset = 0;
  for (const v of inputs) {
    out.set(v, offset);
    offset += v.length;
  }
  return out;
}

function evalAdd(p: AddParams, inputs: Float32Array[], mode: EvalMode): Float32Array {
  if (inputs.length < 2) {
    throw new Error(`add: 入力は2本以上必要です`);
  }
  const width = p.width;
  if (inputs.some((v) => v.length !== width)) {
    throw new Error(`add: 全入力幅はwidth=${width}に一致する必要があります`);
  }
  const out = new Float32Array(width);
  out.set(inputs[0]);
  for (const x of inputs.slice(1)) {
    for (let i = 0; i < width; i++) {
      if (p.mode === 'or') {
        out[i] = mode === 'hard'
          ? (out[i] >= BIT_ON_THRESHOLD || x[i] >= BIT_ON_THRESHOLD ? 1 : 0)
          : 1 - (1 - clamp01(out[i])) * (1 - clamp01(x[i]));
      } else if (p.mode === 'xor') {
        out[i] = mode === 'hard'
          ? ((out[i] >= BIT_ON_THRESHOLD) !== (x[i] >= BIT_ON_THRESHOLD) ? 1 : 0)
          : clamp01(out[i]) + clamp01(x[i]) - 2 * clamp01(out[i]) * clamp01(x[i]);
      } else {
        out[i] += x[i];
      }
    }
  }
  return out;
}

function evalConstant(p: ConstantParams): Float32Array {
  const out = new Float32Array(p.width);
  for (let i = 0; i < p.width; i++) out[i] = p.bits[i] ? 1 : 0;
  return out;
}

/**
 * IRを評価し、全ノードの出力値と最終出力を返す。
 *
 * @param ir 対象IR(lut_layer/group_sum/binarize/flattenの直列構成)
 * @param inputBits 入力ポートid→ビット値(hardは0/1へ丸め、softは[0,1]クランプ)
 * @param mode hard=真理値表評価 / soft=butterfly基底の確率伝播
 * @param layerProbs lut_layer id→ゲート確率(省略層はIRのtruth_tablesから導出)
 */
export function evalCircuit(
  ir: IRGraphJson,
  inputBits: Record<string, ArrayLike<number>>,
  mode: EvalMode,
  layerProbs?: Record<string, LayerProbs>,
): EvalResult {
  const values: Record<string, Float32Array> = {};

  for (const port of ir.inputs) {
    const bits = inputBits[port.id];
    if (!bits || bits.length !== port.width) {
      throw new Error(
        `入力ポート${port.id}のビット数不一致(期待${port.width}, 実際${bits?.length ?? 0})`,
      );
    }
    const v = new Float32Array(port.width);
    for (let i = 0; i < port.width; i++) {
      v[i] = mode === 'hard' ? (bits[i] >= BIT_ON_THRESHOLD ? 1 : 0) : clamp01(bits[i]);
    }
    values[port.id] = v;
  }

  const inSrcs = collectInputSources(ir);
  for (const node of topologicalOrder(ir, inSrcs)) {
    const srcs = inSrcs.get(node.id) ?? [];
    if (srcs.length === 0 && node.type !== 'constant') {
      throw new Error(`評価不能(入力エッジなし): ${node.id}`);
    }
    const inputs = srcs.map((src) => {
      const value = values[src];
      if (!value) throw new Error(`評価不能(未解決入力): ${src} -> ${node.id}`);
      return value;
    });
    const primaryInput = () => {
      const value = inputs[0];
      if (!value) throw new Error(`評価不能(入力エッジなし): ${node.id}`);
      return value;
    };

    switch (node.type) {
      case 'lut_layer':
        values[node.id] = evalLutLayer(
          node.params as unknown as LutLayerParams,
          primaryInput(),
          mode,
          layerProbs?.[node.id],
        );
        break;
      case 'group_sum':
        values[node.id] = evalGroupSum(
          node.params as unknown as GroupSumParams,
          primaryInput(),
          mode,
        );
        break;
      case 'binarize':
        values[node.id] = evalBinarize(node, primaryInput());
        break;
      case 'flatten':
        values[node.id] = primaryInput();
        break;
      case 'concat':
        values[node.id] = evalConcat(
          node.params as unknown as ConcatParams,
          inputs,
        );
        break;
      case 'add':
        values[node.id] = evalAdd(
          node.params as unknown as AddParams,
          inputs,
          mode,
        );
        break;
      case 'constant':
        values[node.id] = evalConstant(node.params as unknown as ConstantParams);
        break;
      case 'permute':
        values[node.id] = evalPermute(
          node.params as unknown as PermuteParams,
          primaryInput(),
        );
        break;
      default:
        throw new Error(`未対応ノード型: ${node.id}(${node.type})`);
    }
  }

  // 出力ポート: 接続元の値をポートidにも載せ、宣言順に連結してoutputへ
  const outputs: Float32Array[] = [];
  for (const port of ir.outputs) {
    const src = inSrcs.get(port.id)?.[0];
    const v = src !== undefined ? values[src] : undefined;
    if (!v) throw new Error(`出力ポート${port.id}が未接続です`);
    values[port.id] = v;
    outputs.push(v);
  }
  if (outputs.length === 1) {
    return { values, output: outputs[0] };
  }
  const total = outputs.reduce((acc, v) => acc + v.length, 0);
  const output = new Float32Array(total);
  let offset = 0;
  for (const v of outputs) {
    output.set(v, offset);
    offset += v.length;
  }
  return { values, output };
}
