// evalCircuitの単体テスト。
// hard評価は griffith_logic/ir/simulate.py(golden model)と同じ結果になること、
// soft評価は飽和確率(one-hot)でhardと一致することを全サンプル全列挙で検証する。

import { describe, expect, it } from 'vitest';
import { ttToGateIdN2 } from '../ir/gates';
import { ttBit } from '../ir/hexTT';
import type { IRGraphJson, IRNodeType, LutLayerParams } from '../ir/schema';
import {
  SAMPLE_FULL_ADDER,
  SAMPLE_HALF_ADDER,
  SAMPLE_KV_ATTENTION2_REF,
  SAMPLE_MAJORITY3,
  SAMPLE_MUX21,
  SAMPLE_PARITY3,
  SAMPLE_PARITY4,
  SAMPLE_SEQ101_RNN_REF,
  SAMPLE_XOR,
} from '../samples/sampleIr';
import {
  evalCircuit,
  GATE_TABLE_N2,
  type LayerProbs,
} from './evalCircuit';

/** パターンkのビット列(bit j = (k >> j) & 1) */
function bitsOf(k: number, width: number): number[] {
  return Array.from({ length: width }, (_, j) => (k >> j) & 1);
}

/** 単一ノードのIRを生成する(group_sum/binarize/flatten等のテスト用) */
function singleNodeIr(
  type: IRNodeType,
  params: Record<string, unknown>,
  inWidth: number,
  outWidth: number,
): IRGraphJson {
  return {
    format: 'magicdelta-ir',
    version: '0.1.0',
    metadata: {},
    inputs: [{ id: 'x', width: inWidth, shape: null, encoding: null, semantics: null }],
    outputs: [{ id: 'y', width: outWidth, shape: null, encoding: null, semantics: null }],
    nodes: [{ id: 'n1', type, params, train_info: {}, meta: {} }],
    edges: [
      { src: 'x', dst: 'n1' },
      { src: 'n1', dst: 'y' },
    ],
  };
}

describe('GATE_TABLE_N2', () => {
  it('ttToGateIdN2の逆変換になっている(全16ゲート)', () => {
    for (let g = 0; g < 16; g++) {
      let tt = 0;
      for (let k = 0; k < 4; k++) {
        tt |= GATE_TABLE_N2[g * 4 + k] << k;
      }
      expect(ttToGateIdN2(tt)).toBe(g);
    }
  });

  it('既知ゲートの真理値表と一致する(AND=g1/XOR=g6/OR=g7)', () => {
    // butterfly基底順: k=0→(0,0), k=1→(1,0), k=2→(0,1), k=3→(1,1)
    expect(Array.from(GATE_TABLE_N2.subarray(1 * 4, 2 * 4))).toEqual([0, 0, 0, 1]); // AND
    expect(Array.from(GATE_TABLE_N2.subarray(6 * 4, 7 * 4))).toEqual([0, 1, 1, 0]); // XOR
    expect(Array.from(GATE_TABLE_N2.subarray(7 * 4, 8 * 4))).toEqual([0, 1, 1, 1]); // OR
  });
});

// 各サンプルの既知真理値(入力bits→出力bits)
const SAMPLE_CASES: {
  name: string;
  ir: IRGraphJson;
  width: number;
  expected: (b: number[]) => number[];
}[] = [
  { name: 'XOR', ir: SAMPLE_XOR, width: 2, expected: (b) => [b[0] ^ b[1]] },
  {
    name: '半加算器',
    ir: SAMPLE_HALF_ADDER,
    width: 2,
    expected: (b) => [b[0] ^ b[1], b[0] & b[1]],
  },
  {
    name: '全加算器',
    ir: SAMPLE_FULL_ADDER,
    width: 3,
    expected: (b) => [b[0] ^ b[1] ^ b[2], (b[0] & b[1]) | ((b[0] ^ b[1]) & b[2])],
  },
  {
    name: '2:1 MUX',
    ir: SAMPLE_MUX21,
    width: 3,
    expected: (b) => [b[2] ? b[1] : b[0]],
  },
  {
    name: '多数決3',
    ir: SAMPLE_MAJORITY3,
    width: 3,
    expected: (b) => [Number(b[0] + b[1] + b[2] >= 2)],
  },
  {
    name: 'KV Attention 2',
    ir: SAMPLE_KV_ATTENTION2_REF,
    width: 5,
    expected: (b) => [((b[0] === b[1] ? b[2] : 0) | (b[0] === b[3] ? b[4] : 0))],
  },
  {
    name: 'RNN風 101',
    ir: SAMPLE_SEQ101_RNN_REF,
    width: 3,
    expected: (b) => [Number(b[0] === 1 && b[1] === 0 && b[2] === 1)],
  },
  { name: 'パリティ3', ir: SAMPLE_PARITY3, width: 3, expected: (b) => [b[0] ^ b[1] ^ b[2]] },
  {
    name: 'パリティ4',
    ir: SAMPLE_PARITY4,
    width: 4,
    expected: (b) => [b[0] ^ b[1] ^ b[2] ^ b[3]],
  },
];

describe('hard評価(全サンプル全列挙)', () => {
  for (const c of SAMPLE_CASES) {
    it(`${c.name}: 2^${c.width}パターンが既知の真理値と一致する`, () => {
      for (let k = 0; k < 1 << c.width; k++) {
        const bits = bitsOf(k, c.width);
        const { output, values } = evalCircuit(c.ir, { x: bits }, 'hard');
        expect(Array.from(output), `k=${k}`).toEqual(c.expected(bits));
        // 入力ポート/出力ポートidもvaluesに含まれる
        expect(Array.from(values.x)).toEqual(bits);
        expect(values.y).toBe(output);
      }
    });
  }
});

/** IRのtruth_tablesから飽和確率(one-hot)のLayerProbsを合成する */
function saturatedProbs(
  ir: IRGraphJson,
  kind: '16gate' | 'ttbits',
): Record<string, LayerProbs> {
  const result: Record<string, LayerProbs> = {};
  for (const node of ir.nodes) {
    if (node.type !== 'lut_layer') continue;
    const p = node.params as unknown as LutLayerParams;
    const size = 1 << p.n_inputs;
    if (kind === '16gate') {
      const data = new Float32Array(p.out_width * 16);
      for (let o = 0; o < p.out_width; o++) {
        let tt = 0;
        for (let k = 0; k < size; k++) tt |= ttBit(p.truth_tables[o], k) << k;
        data[o * 16 + ttToGateIdN2(tt)] = 1;
      }
      result[node.id] = { data, bins: 16 };
    } else {
      const data = new Float32Array(p.out_width * size);
      for (let o = 0; o < p.out_width; o++) {
        for (let k = 0; k < size; k++) {
          data[o * size + k] = ttBit(p.truth_tables[o], k);
        }
      }
      result[node.id] = { data, bins: size };
    }
  }
  return result;
}

describe('soft評価', () => {
  it('飽和確率(16ゲートone-hot)でhardと一致する(全サンプル全列挙)', () => {
    for (const c of SAMPLE_CASES) {
      const probs = saturatedProbs(c.ir, '16gate');
      for (let k = 0; k < 1 << c.width; k++) {
        const bits = bitsOf(k, c.width);
        const hard = evalCircuit(c.ir, { x: bits }, 'hard').output;
        const soft = evalCircuit(c.ir, { x: bits }, 'soft', probs).output;
        for (let j = 0; j < hard.length; j++) {
          expect(soft[j], `${c.name} k=${k} bit=${j}`).toBeCloseTo(hard[j], 6);
        }
      }
    }
  });

  it('飽和確率(真理値表ビット0/1)でhardと一致する(全サンプル全列挙)', () => {
    for (const c of SAMPLE_CASES) {
      const probs = saturatedProbs(c.ir, 'ttbits');
      for (let k = 0; k < 1 << c.width; k++) {
        const bits = bitsOf(k, c.width);
        const hard = evalCircuit(c.ir, { x: bits }, 'hard').output;
        const soft = evalCircuit(c.ir, { x: bits }, 'soft', probs).output;
        for (let j = 0; j < hard.length; j++) {
          expect(soft[j], `${c.name} k=${k} bit=${j}`).toBeCloseTo(hard[j], 6);
        }
      }
    }
  });

  it('16ゲート混合分布は期待値になる(XOR 0.6 + AND 0.4)', () => {
    const data = new Float32Array(16);
    data[6] = 0.6; // XOR
    data[1] = 0.4; // AND
    const probs = { n1_lutlayer: { data, bins: 16 } };
    // (1,0): XOR=1, AND=0 → 0.6
    expect(
      evalCircuit(SAMPLE_XOR, { x: [1, 0] }, 'soft', probs).output[0],
    ).toBeCloseTo(0.6, 6);
    // (1,1): XOR=0, AND=1 → 0.4
    expect(
      evalCircuit(SAMPLE_XOR, { x: [1, 1] }, 'soft', probs).output[0],
    ).toBeCloseTo(0.4, 6);
  });

  it('中間の入力確率をbutterfly基底で伝播する(XORに0.5入力→0.5)', () => {
    const { output } = evalCircuit(SAMPLE_XOR, { x: [0.5, 0.5] }, 'soft');
    expect(output[0]).toBeCloseTo(0.5, 6);
  });
});

describe('確率のhard化(argmax真理値表化)', () => {
  it('16ゲート分布のargmaxで評価する(XOR層にOR優勢分布→OR動作)', () => {
    const data = new Float32Array(16);
    data[7] = 0.7; // OR
    data[6] = 0.3; // XOR
    const probs = { n1_lutlayer: { data, bins: 16 } };
    const results = [0, 1, 2, 3].map(
      (k) => evalCircuit(SAMPLE_XOR, { x: bitsOf(k, 2) }, 'hard', probs).output[0],
    );
    expect(results).toEqual([0, 1, 1, 1]); // OR真理値
  });

  it('真理値表ビット確率はしきい値0.5で確定化する', () => {
    // w=[0.1, 0.9, 0.4, 0.6] → hard tt = [0,1,0,1](=Bのパススルー…ではなくk=1,3で1)
    const probs = {
      n1_lutlayer: { data: new Float32Array([0.1, 0.9, 0.4, 0.6]), bins: 4 },
    };
    const results = [0, 1, 2, 3].map(
      (k) => evalCircuit(SAMPLE_XOR, { x: bitsOf(k, 2) }, 'hard', probs).output[0],
    );
    expect(results).toEqual([0, 1, 0, 1]);
  });

  it('binsが想定外の形状ならIRのtruth_tablesへフォールバックする', () => {
    const probs = { n1_lutlayer: { data: new Float32Array(8), bins: 8 } };
    const { output } = evalCircuit(SAMPLE_XOR, { x: [1, 0] }, 'hard', probs);
    expect(output[0]).toBe(1); // XOR(IRのtt)のまま
  });
});

describe('lut_layer以外のノード型', () => {
  it('group_sum: hardは整数和、softは/τ', () => {
    const ir = singleNodeIr('group_sum', { in_width: 4, k: 2, tau: 2.0 }, 4, 2);
    const hard = evalCircuit(ir, { x: [1, 0, 1, 1] }, 'hard');
    expect(Array.from(hard.output)).toEqual([1, 2]);
    const soft = evalCircuit(ir, { x: [1, 0, 1, 1] }, 'soft');
    expect(Array.from(soft.output)).toEqual([0.5, 1]);
  });

  it('group_sum: bias は τ で割る前の整数和に足す', () => {
    const ir = singleNodeIr('group_sum', { in_width: 4, k: 2, tau: 2.0, bias: [0, 1] }, 4, 2);
    const hard = evalCircuit(ir, { x: [1, 0, 1, 1] }, 'hard');
    expect(Array.from(hard.output)).toEqual([1, 3]);
    const soft = evalCircuit(ir, { x: [1, 0, 1, 1] }, 'soft');
    expect(Array.from(soft.output)).toEqual([0.5, 1.5]);
  });

  it('permute: in_shape の軸を perm の順に並べ替える恒等配線', () => {
    // (2, 3) → (3, 2): out[j, i] = x[i, j]
    const ir = singleNodeIr('permute', { in_shape: [2, 3], perm: [1, 0] }, 6, 6);
    const hard = evalCircuit(ir, { x: [0, 1, 0, 1, 1, 0] }, 'hard');
    expect(Array.from(hard.output)).toEqual([0, 1, 1, 1, 0, 0]);
    const soft = evalCircuit(ir, { x: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6] }, 'soft');
    expect(Array.from(soft.output).map((v) => Math.round(v * 10) / 10)).toEqual([
      0.1, 0.4, 0.2, 0.5, 0.3, 0.6,
    ]);
  });

  it('group_sum: 入力幅がkで割り切れない場合はエラー', () => {
    const ir = singleNodeIr('group_sum', { in_width: 3, k: 2, tau: 1.0 }, 3, 2);
    expect(() => evalCircuit(ir, { x: [1, 0, 1] }, 'hard')).toThrow(/割り切れません/);
  });

  it('binarize: しきい値超えで1', () => {
    const ir = singleNodeIr('binarize', { width: 2, threshold: 0.5 }, 2, 2);
    const { output } = evalCircuit(ir, { x: [0.4, 0.6] }, 'soft');
    expect(Array.from(output)).toEqual([0, 1]);
  });

  it('flatten: 恒等', () => {
    const ir = singleNodeIr('flatten', { width: 3 }, 3, 3);
    const { output } = evalCircuit(ir, { x: [1, 0, 1] }, 'hard');
    expect(Array.from(output)).toEqual([1, 0, 1]);
  });

  it('constant: 入力なしで固定ビット列を出力する', () => {
    const ir: IRGraphJson = {
      format: 'magicdelta-ir',
      version: '0.2.0',
      metadata: {},
      inputs: [{ id: 'x', width: 1, shape: null, encoding: null, semantics: null }],
      outputs: [{ id: 'y', width: 3, shape: null, encoding: null, semantics: null }],
      nodes: [
        {
          id: 'c1',
          type: 'constant',
          params: { width: 3, bits: [1, 0, 1] },
          train_info: {},
          meta: {},
        },
      ],
      edges: [{ src: 'c1', dst: 'y' }],
    };
    const { output } = evalCircuit(ir, { x: [0] }, 'hard');
    expect(Array.from(output)).toEqual([1, 0, 1]);
  });

  it('concat/add: slot順に結合し、同幅入力をxorで合成する', () => {
    const ir: IRGraphJson = {
      format: 'magicdelta-ir',
      version: '0.2.0',
      metadata: {},
      inputs: [{ id: 'x', width: 2, shape: null, encoding: null, semantics: null }],
      outputs: [{ id: 'y', width: 4, shape: null, encoding: null, semantics: null }],
      nodes: [
        {
          id: 'c1',
          type: 'constant',
          params: { width: 2, bits: [1, 0] },
          train_info: {},
          meta: {},
        },
        {
          id: 'a1',
          type: 'add',
          params: { width: 2, mode: 'xor' },
          train_info: {},
          meta: {},
        },
        {
          id: 'cat1',
          type: 'concat',
          params: { in_widths: [2, 2] },
          train_info: {},
          meta: {},
        },
      ],
      edges: [
        { src: 'x', dst: 'a1' },
        { src: 'c1', dst: 'a1', dst_slot: 1 },
        { src: 'a1', dst: 'cat1' },
        { src: 'c1', dst: 'cat1', dst_slot: 1 },
        { src: 'cat1', dst: 'y' },
      ],
    };
    const { output } = evalCircuit(ir, { x: [1, 1] }, 'hard');
    expect(Array.from(output)).toEqual([0, 1, 1, 0]);
  });

  it('blackboxは明示エラー', () => {
    const ir = singleNodeIr('blackbox', {}, 2, 2);
    expect(() => evalCircuit(ir, { x: [0, 0] }, 'hard')).toThrow(/未対応ノード型/);
  });
});

describe('グラフ構造の異常系', () => {
  it('入力ビット数の不一致はエラー', () => {
    expect(() => evalCircuit(SAMPLE_XOR, { x: [0] }, 'hard')).toThrow(/ビット数不一致/);
  });

  it('nodes配列の順序に依存しない(逆順でもトポロジカル順で評価)', () => {
    const reversed: IRGraphJson = {
      ...SAMPLE_PARITY3,
      nodes: [...SAMPLE_PARITY3.nodes].reverse(),
    };
    for (let k = 0; k < 8; k++) {
      const bits = bitsOf(k, 3);
      expect(Array.from(evalCircuit(reversed, { x: bits }, 'hard').output)).toEqual([
        bits[0] ^ bits[1] ^ bits[2],
      ]);
    }
  });

  it('入力エッジのないノードはエラー', () => {
    const ir: IRGraphJson = {
      ...SAMPLE_XOR,
      edges: [{ src: 'n1_lutlayer', dst: 'y' }], // xとの接続を欠落させる
    };
    expect(() => evalCircuit(ir, { x: [0, 0] }, 'hard')).toThrow(/入力エッジなし/);
  });

  it('上流未解決(存在しないソース)のノードはエラー', () => {
    const ir: IRGraphJson = {
      ...SAMPLE_XOR,
      edges: [
        { src: 'ghost', dst: 'n1_lutlayer' }, // 存在しない上流
        { src: 'n1_lutlayer', dst: 'y' },
      ],
    };
    expect(() => evalCircuit(ir, { x: [0, 0] }, 'hard')).toThrow(/未接続または循環/);
  });
});
