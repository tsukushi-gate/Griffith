// 埋め込みサンプルIR(オンボーディング /拡充)。
// PUT /api/models/current/ir → build の既存契約のみで投入できる構成。
// 小規模なので詳細ゲートビューのショーケースを兼ねる。
//
// 真理値表の規約: butterfly基底順、bit k の k = s0 + 2·s1(bit j of k = スロットjの値)
//   XOR=0x6 / AND=0x8 / OR=0xe / A(s0パススルー)=0xa / A∧¬B=0x2 / FALSE=0x0

import {
  Binary,
  BrainCircuit,
  CircuitBoard,
  Cpu,
  GitMerge,
  Grid3X3,
  Network,
  Plus,
  RotateCw,
  Split,
  Zap,
} from 'lucide-react';
import type { LucideIcon } from 'lucide-react';
import type { IRGraphJson, IRNodeJson } from '../ir/schema';
import type { TrainingPreset } from '../store/trainingStore';
import wiringSamples from './wiringSamples.json';

/** lut_layerノードの生成ヘルパ(サンプル定義の重複を減らす) */
function lut(
  id: string,
  inWidth: number,
  indices: number[][],
  tts: string[],
  opts?: { initialized?: boolean },
): IRNodeJson {
  return {
    id,
    type: 'lut_layer',
    params: {
      in_width: inWidth,
      out_width: tts.length,
      n_inputs: 2,
      indices,
      truth_tables: tts,
    },
    train_info: {
      parameterization: 'softmax',
      selection: 'soft',
      ...(opts?.initialized === false ? { initialized: false } : {}),
    },
    meta: {},
  };
}

function draftLut(id: string, inWidth: number, outWidth: number): IRNodeJson {
  return lut(
    id,
    inWidth,
    [new Array<number>(outWidth).fill(0), new Array<number>(outWidth).fill(0)],
    new Array<string>(outWidth).fill('0x0'),
    { initialized: false },
  );
}

function graph(
  name: string,
  inWidth: number,
  outWidth: number,
  nodes: IRNodeJson[],
  opts?: { inputEncoding?: string; outputSemantics?: string },
): IRGraphJson {
  const ids = nodes.map((n) => n.id);
  return {
    format: 'magicdelta-ir',
    version: '0.1.0',
    metadata: { name },
    inputs: [
      {
        id: 'x',
        width: inWidth,
        shape: null,
        encoding: opts?.inputEncoding ?? null,
        semantics: null,
      },
    ],
    outputs: [
      {
        id: 'y',
        width: outWidth,
        shape: null,
        encoding: null,
        semantics: opts?.outputSemantics ?? null,
      },
    ],
    nodes,
    edges: [
      { src: 'x', dst: ids[0] },
      ...ids.slice(1).map((dst, i) => ({ src: ids[i], dst })),
      { src: ids[ids.length - 1], dst: 'y' },
    ],
  };
}

/** 構造化配線サンプルの仕様。`wiringSamples.json` をフロントと
    `tests/test_editor_samples_wiring.py` が同じ規則で draft の IR に展開する。
    layers は入力側から順に(in_width は前層の out_width)、group_sum は任意。 */
export interface WiredLayerSpec {
  out_width: number;
  n_inputs: number;
  /** 構造化配線の種(rot / rot_n / butterfly / local1d / local2d) */
  connections: string;
  /** 種ごとの引数(core.connections.WIRING_SPEC_KEYS) */
  wiring: Record<string, unknown>;
}

export interface WiredSampleSpec {
  dataset: string;
  in_width: number;
  out_width: number;
  input_encoding: string | null;
  output_semantics: string | null;
  /** 同じ配線のレシピ(configs/recipes/、無ければ null) */
  recipe: string | null;
  layers: WiredLayerSpec[];
  group_sum: { k: number; tau: number } | null;
}

export const WIRING_SAMPLE_SPECS: Record<string, WiredSampleSpec> = wiringSamples;

/** 構造化配線の draft lut_layer。indices / truth_tables は placeholder で、build 時に
    train_info.connections + wiring から make_connections で再生成される。 */
function draftWiredLut(id: string, inWidth: number, layer: WiredLayerSpec): IRNodeJson {
  return {
    id,
    type: 'lut_layer',
    params: {
      in_width: inWidth,
      out_width: layer.out_width,
      n_inputs: layer.n_inputs,
      indices: Array.from({ length: layer.n_inputs }, () =>
        new Array<number>(layer.out_width).fill(0),
      ),
      truth_tables: new Array<string>(layer.out_width).fill('0x0'),
    },
    train_info: {
      parameterization: 'softmax',
      selection: 'soft',
      initialized: false,
      connections: layer.connections,
      wiring: { ...layer.wiring },
    },
    meta: {},
  };
}

/** 仕様 → 直列 IR(lut_layer 列 + 任意の group_sum)。 */
export function wiredSampleGraph(name: string, spec: WiredSampleSpec): IRGraphJson {
  const nodes: IRNodeJson[] = [];
  let width = spec.in_width;
  spec.layers.forEach((layer, i) => {
    nodes.push(draftWiredLut(`l${i}_lut`, width, layer));
    width = layer.out_width;
  });
  if (spec.group_sum) {
    nodes.push({
      id: 'group_sum',
      type: 'group_sum',
      params: { in_width: width, k: spec.group_sum.k, tau: spec.group_sum.tau },
      train_info: {},
      meta: {},
    });
  }
  return graph(name, spec.in_width, spec.out_width, nodes, {
    inputEncoding: spec.input_encoding ?? undefined,
    outputSemantics: spec.output_semantics ?? undefined,
  });
}

/** 2bit 加算器 rot 配線((a)): 4 → rot(0,2) → 16 → rot(0,1)/(0,2)/(0,4)/(0,8) → rot(0,8, step 5) → 3 */
export const SAMPLE_ADDER2_ROT16: IRGraphJson = wiredSampleGraph(
  'sample_adder2_rot16',
  WIRING_SAMPLE_SPECS.adder2_rot16,
);
/** MNIST rot 1000((b)): 事前実験 fmc-wiring-w1000 の rot 腕(configs/recipes/fmc_w1000_rot.yaml)と同じ配線 */
export const SAMPLE_MNIST_ROT1000: IRGraphJson = wiredSampleGraph(
  'sample_mnist_rot1000',
  WIRING_SAMPLE_SPECS.mnist_rot1000,
);
/** MNIST butterfly 4 入力 1000((c)): 同 bf4 腕(configs/recipes/fmc_w1000_bf4.yaml)と同じ配線 */
export const SAMPLE_MNIST_BF4_1000: IRGraphJson = wiredSampleGraph(
  'sample_mnist_bf4_1000',
  WIRING_SAMPLE_SPECS.mnist_bf4_1000,
);

const LOGIC_TRAINING_PRESET: Omit<TrainingPreset, 'dataset'> = {
  lr: 0.08,
  batchSize: 16,
  epochs: 60,
  method: 'bp',
  loss: 'bce',
  optimizer: 'adam',
};

function logicPreset(dataset: string): TrainingPreset {
  return { ...LOGIC_TRAINING_PRESET, dataset };
}

const MNIST_TRAINING_PRESET: TrainingPreset = {
  dataset: 'mnist',
  lr: 0.01,
  batchSize: 128,
  epochs: 3,
  method: 'bp',
  loss: 'ce',
  optimizer: 'adam',
};

/** 幅 1000 の構造化配線サンプル(rot / butterfly)用。事前実験 fmc-wiring-w1000 の
    学習レジーム(lr 0.01 / batch 100 / ce / adam)。epoch は CPU の 1 epoch 所要時間から
    決めた既定。レシピ(8 epoch、入力を 0.5 で二値化)の参考値は
    rot 92.3% / butterfly 89.2%。 */
const MNIST_W1000_TRAINING_PRESET: TrainingPreset = {
  dataset: 'mnist',
  lr: 0.01,
  batchSize: 100,
  epochs: 2,
  method: 'bp',
  loss: 'ce',
  optimizer: 'adam',
};

const LLVIT_DEBUG_TRAINING_PRESET: TrainingPreset = {
  dataset: 'llvit_synthetic_debug',
  lr: 0.001,
  logicLr: 0.02,
  annLr: 0.001,
  batchSize: 32,
  epochs: 8,
  method: 'bp',
  loss: 'ce',
  optimizer: 'adam',
};

/** XORミニ: x(2bit) → XORゲート1個 → y */
export const SAMPLE_XOR: IRGraphJson = graph('sample_xor', 2, 1, [
  lut('n1_lutlayer', 2, [[0], [1]], ['0x6']),
]);

/** パリティ3: x(3bit) → [XOR(x0,x1), A(x2)] → XOR → y */
export const SAMPLE_PARITY3: IRGraphJson = graph('sample_parity3', 3, 1, [
  lut('n1_lutlayer', 3, [[0, 2], [1, 2]], ['0x6', '0xa']),
  lut('n2_lutlayer', 2, [[0], [1]], ['0x6']),
]);

/** 半加算器: x(2bit: a,b) → [sum=XOR, carry=AND] → y(2bit) */
export const SAMPLE_HALF_ADDER: IRGraphJson = graph('sample_half_adder', 2, 2, [
  lut('n1_lutlayer', 2, [[0, 0], [1, 1]], ['0x6', '0x8']),
]);

/** 半加算器 学習用: x(2bit) → draft(2→8) → draft(8→2) → y */
export const SAMPLE_HALF_ADDER_TRAIN: IRGraphJson = graph(
  'sample_half_adder_train',
  2,
  2,
  [draftLut('n1_lutlayer', 2, 8), draftLut('n2_lutlayer', 8, 2)],
);

/** 全加算器(3層): x(3bit: a,b,cin) → y(2bit: sum,carry)
    L1=[p=XOR(a,b), q=AND(a,b), cin] / L2=[sum=XOR(p,cin), r=AND(p,cin), q] /
    L3=[sum, carry=OR(r,q)] */
export const SAMPLE_FULL_ADDER: IRGraphJson = graph('sample_full_adder', 3, 2, [
  lut('n1_lutlayer', 3, [[0, 0, 2], [1, 1, 2]], ['0x6', '0x8', '0xa']),
  lut('n2_lutlayer', 3, [[0, 0, 1], [2, 2, 1]], ['0x6', '0x8', '0xa']),
  lut('n3_lutlayer', 3, [[0, 1], [0, 2]], ['0xa', '0xe']),
]);

/** 全加算器 学習用: x(3bit) → draft(3→12) → draft(12→8) → draft(8→2) → y */
export const SAMPLE_FULL_ADDER_TRAIN: IRGraphJson = graph(
  'sample_full_adder_train',
  3,
  2,
  [
    draftLut('n1_lutlayer', 3, 12),
    draftLut('n2_lutlayer', 12, 8),
    draftLut('n3_lutlayer', 8, 2),
  ],
);

/** 多数決3: x(3bit) → (ab/ac/bc) → ORツリー → y */
export const SAMPLE_MAJORITY3: IRGraphJson = graph('sample_majority3', 3, 1, [
  lut('n1_lutlayer', 3, [[0, 0, 1], [1, 2, 2]], ['0x8', '0x8', '0x8']),
  lut('n2_lutlayer', 3, [[0, 2], [1, 2]], ['0xe', '0xa']),
  lut('n3_lutlayer', 2, [[0], [1]], ['0xe']),
]);

/** 2:1マルチプレクサ: x(3bit: a,b,sel) → out = sel ? b : a
    L1=[a∧¬sel, b∧sel] / L2=OR */
export const SAMPLE_MUX21: IRGraphJson = graph('sample_mux21', 3, 1, [
  lut('n1_lutlayer', 3, [[0, 1], [2, 2]], ['0x2', '0x8']),
  lut('n2_lutlayer', 2, [[0], [1]], ['0xe']),
]);

/** パリティ4(XORツリー): x(4bit) → [XOR(x0,x1), XOR(x2,x3)] → XOR → y */
export const SAMPLE_PARITY4: IRGraphJson = graph('sample_parity4', 4, 1, [
  lut('n1_lutlayer', 4, [[0, 2], [1, 3]], ['0x6', '0x6']),
  lut('n2_lutlayer', 2, [[0], [1]], ['0x6']),
]);

/** MNISTミニ(学習デモ): x(784) → LutLayer draft(784→60) → GroupSum(k=10) → y(10)
    draft(initialized=false)なのでbuild時にランダム初期化される。
    out=60≤64なので詳細ゲートビューが自動展開され、full capture+確率分布
    ストリーミング+タイムライン巻き戻しの全部入りショーケースになる。 */
export const SAMPLE_MNIST_MINI: IRGraphJson = {
  format: 'magicdelta-ir',
  version: '0.1.0',
  metadata: { name: 'sample_mnist_mini' },
  inputs: [
    {
      id: 'x',
      width: 784,
      shape: null,
      encoding: 'binary_threshold_0.5',
      semantics: null,
    },
  ],
  outputs: [
    { id: 'y', width: 10, shape: null, encoding: null, semantics: 'class_scores' },
  ],
  nodes: [
    lut(
      'n1_lutlayer',
      784,
      // draftのplaceholder(build時にconnections設定でランダム化される)
      [new Array<number>(60).fill(0), new Array<number>(60).fill(0)],
      new Array<string>(60).fill('0x0'),
      { initialized: false },
    ),
    {
      id: 'n2_groupsum',
      type: 'group_sum',
      params: { in_width: 60, k: 10, tau: 1.0 },
      train_info: {},
      meta: {},
    },
  ],
  edges: [
    { src: 'x', dst: 'n1_lutlayer' },
    { src: 'n1_lutlayer', dst: 'n2_groupsum' },
    { src: 'n2_groupsum', dst: 'y' },
  ],
};

/** 残差XORデモ(v0.2 DAG): x(2bit) →fanout→ [XOR,AND] → add(xor) → y(2bit)
    skip接続はadd slot1(dst_slot=1、残差エッジ描画対象)。
    入出力2bit/2bitなので logic:half_adder と幅一致で学習可能な構成。 */
export const SAMPLE_RESIDUAL_XOR: IRGraphJson = {
  format: 'magicdelta-ir',
  version: '0.2.0',
  metadata: { name: 'sample_residual_xor' },
  inputs: [{ id: 'x', width: 2, shape: null, encoding: null, semantics: null }],
  outputs: [{ id: 'y', width: 2, shape: null, encoding: null, semantics: null }],
  nodes: [
    lut('n1_lutlayer', 2, [[0, 0], [1, 1]], ['0x6', '0x8']),
    {
      id: 'n2_add',
      type: 'add',
      params: { width: 2, mode: 'xor' },
      train_info: {},
      meta: {},
    },
  ],
  edges: [
    { src: 'x', dst: 'n1_lutlayer' },
    { src: 'n1_lutlayer', dst: 'n2_add' },
    { src: 'x', dst: 'n2_add', dst_slot: 1 },
    { src: 'n2_add', dst: 'y' },
  ],
};

/** KV Lookup Toy 参照: qが一致したkeyのvalueを返す2要素のlookup教材 */
export const SAMPLE_KV_ATTENTION2_REF: IRGraphJson = graph(
  'sample_kv_attention2_ref',
  5,
  1,
  [
    lut(
      'n1_match',
      5,
      [
        [0, 2, 0, 4],
        [1, 2, 3, 4],
      ],
      ['0x9', '0xa', '0x9', '0xa'],
    ),
    lut('n2_mask_value', 4, [[0, 2], [1, 3]], ['0x8', '0x8']),
    lut('n3_reduce', 2, [[0], [1]], ['0xe']),
  ],
  {
    inputEncoding: 'bits: q,k0,v0,k1,v1',
    outputSemantics: 'selected_value',
  },
);

/** KV Lookup Toy 学習用: 参照版と同じI/Oをdraft層から学習する */
export const SAMPLE_KV_ATTENTION2_TRAIN: IRGraphJson = graph(
  'sample_kv_attention2_train',
  5,
  1,
  [
    draftLut('n1_lutlayer', 5, 6),
    draftLut('n2_lutlayer', 6, 3),
    draftLut('n3_lutlayer', 3, 1),
  ],
  {
    inputEncoding: 'bits: q,k0,v0,k1,v1',
    outputSemantics: 'selected_value',
  },
);

/** RNN風 101 検出 参照: 3step系列をx0,x1,x2として受け取り101だけを検出 */
export const SAMPLE_SEQ101_RNN_REF: IRGraphJson = graph(
  'sample_seq101_rnn_ref',
  3,
  1,
  [
    lut('n1_state_update', 3, [[0, 2], [1, 2]], ['0x2', '0xa']),
    lut('n2_accept', 2, [[0], [1]], ['0x8']),
  ],
  {
    inputEncoding: 'bits: t0,t1,t2',
    outputSemantics: 'sequence_accept',
  },
);

/** RNN風 101 検出 学習用: 小さなdraft状態層から検出器を学習する */
export const SAMPLE_SEQ101_RNN_TRAIN: IRGraphJson = graph(
  'sample_seq101_rnn_train',
  3,
  1,
  [draftLut('n1_state', 3, 6), draftLut('n2_accept', 6, 1)],
  {
    inputEncoding: 'bits: t0,t1,t2',
    outputSemantics: 'sequence_accept',
  },
);

export interface SampleDef {
  key: string;
  label: string;
  desc: string;
  icon: LucideIcon;
  category: 'logic' | 'learning' | 'sequence' | 'vision' | 'hybrid' | 'dag';
  /** 学習進捗観察用か、正解構造の参照用か */
  variant: 'train' | 'reference' | 'demo';
  recommendedDataset?: string;
  trainingPreset?: TrainingPreset;
  /** 学習可能なサンプル(学習デモバッジ表示) */
  trainable?: boolean;
  source?: 'ir' | 'python';
  pythonImport?: { path: string; symbol?: string | null; mode?: 'auto' | 'display' };
  ir?: IRGraphJson;
}

export const SAMPLES: SampleDef[] = [
  {
    key: 'llvit_ivitt_cifar10',
    label: 'LL-ViT I-ViT-T CIFAR-10 Preview',
    desc: '構造表示専用: 再現学習はModal/実験スクリプト',
    icon: Cpu,
    category: 'hybrid',
    variant: 'demo',
    recommendedDataset: 'llvit:cifar10',
    trainable: false,
    source: 'python',
    pythonImport: { path: 'tools/editor/samples/llvit_ivitt_cifar10.py' },
  },
  {
    key: 'llvit_synthetic_debug',
    label: 'LL-ViT Synthetic Debug',
    desc: '検証用toy: 4token合成分類',
    icon: Cpu,
    category: 'hybrid',
    variant: 'demo',
    recommendedDataset: 'llvit_synthetic_debug',
    trainingPreset: LLVIT_DEBUG_TRAINING_PRESET,
    trainable: true,
    source: 'python',
    pythonImport: { path: 'tools/editor/samples/llvit_tiny.py' },
  },
  {
    key: 'half_adder_train',
    label: '半加算器 学習',
    desc: '2bit加算をdraft層から学習',
    icon: Plus,
    category: 'learning',
    variant: 'train',
    recommendedDataset: 'logic:half_adder',
    trainingPreset: logicPreset('logic:half_adder'),
    trainable: true,
    ir: SAMPLE_HALF_ADDER_TRAIN,
  },
  {
    key: 'full_adder_train',
    label: '全加算器 学習',
    desc: '3入力→2出力を60epochで観察',
    icon: CircuitBoard,
    category: 'learning',
    variant: 'train',
    recommendedDataset: 'logic:full_adder',
    trainingPreset: logicPreset('logic:full_adder'),
    trainable: true,
    ir: SAMPLE_FULL_ADDER_TRAIN,
  },
  {
    key: 'majority3',
    label: '多数決3',
    desc: '3入力中2つ以上が1なら発火',
    icon: Grid3X3,
    category: 'learning',
    variant: 'train',
    recommendedDataset: 'logic:majority3',
    trainingPreset: logicPreset('logic:majority3'),
    trainable: true,
    ir: SAMPLE_MAJORITY3,
  },
  {
    key: 'adder2_rot16',
    label: '2bit 加算器 rot 配線 学習',
    desc: '層 0 で a_i,b_i を対にし、隠れ層のずらし量を 1,2,4,8 と倍々にする段別 rot',
    icon: RotateCw,
    category: 'learning',
    variant: 'train',
    recommendedDataset: 'logic:adder2',
    trainingPreset: logicPreset('logic:adder2'),
    trainable: true,
    ir: SAMPLE_ADDER2_ROT16,
  },
  {
    key: 'kv_attention2_train',
    label: 'KV Lookup Toy 学習',
    desc: '1bit key一致でvalueを読む教材',
    icon: GitMerge,
    category: 'learning',
    variant: 'train',
    recommendedDataset: 'logic:kv_attention2',
    trainingPreset: logicPreset('logic:kv_attention2'),
    trainable: true,
    ir: SAMPLE_KV_ATTENTION2_TRAIN,
  },
  {
    key: 'seq101_rnn_train',
    label: 'RNN風 101 学習',
    desc: '3step系列101を検出',
    icon: BrainCircuit,
    category: 'sequence',
    variant: 'train',
    recommendedDataset: 'logic:seq101',
    trainingPreset: logicPreset('logic:seq101'),
    trainable: true,
    ir: SAMPLE_SEQ101_RNN_TRAIN,
  },
  {
    key: 'xor',
    label: 'XORミニ',
    desc: '2入力XOR 1ゲート(最小例)',
    icon: Zap,
    category: 'logic',
    variant: 'reference',
    recommendedDataset: 'logic:xor',
    trainingPreset: logicPreset('logic:xor'),
    ir: SAMPLE_XOR,
  },
  {
    key: 'half_adder',
    label: '半加算器 参照',
    desc: 'sum=XOR / carry=AND',
    icon: Plus,
    category: 'logic',
    variant: 'reference',
    recommendedDataset: 'logic:half_adder',
    trainingPreset: logicPreset('logic:half_adder'),
    ir: SAMPLE_HALF_ADDER,
  },
  {
    key: 'full_adder',
    label: '全加算器 参照',
    desc: 'a+b+cin(3層カスケード)',
    icon: CircuitBoard,
    category: 'logic',
    variant: 'reference',
    recommendedDataset: 'logic:full_adder',
    trainingPreset: logicPreset('logic:full_adder'),
    ir: SAMPLE_FULL_ADDER,
  },
  {
    key: 'mux21',
    label: '2:1 MUX',
    desc: 'sel ? b : a(選択回路)',
    icon: Split,
    category: 'logic',
    variant: 'reference',
    recommendedDataset: 'logic:mux2',
    trainingPreset: logicPreset('logic:mux2'),
    ir: SAMPLE_MUX21,
  },
  {
    key: 'parity3',
    label: 'パリティ3',
    desc: '3入力パリティ(XOR×2層)',
    icon: Binary,
    category: 'logic',
    variant: 'reference',
    recommendedDataset: 'logic:parity3',
    trainingPreset: logicPreset('logic:parity3'),
    ir: SAMPLE_PARITY3,
  },
  {
    key: 'parity4',
    label: 'パリティ4',
    desc: 'XORツリー(2+1ゲート)',
    icon: Grid3X3,
    category: 'logic',
    variant: 'reference',
    recommendedDataset: 'logic:parity4',
    trainingPreset: logicPreset('logic:parity4'),
    ir: SAMPLE_PARITY4,
  },
  {
    key: 'kv_attention2_ref',
    label: 'KV Lookup Toy 参照',
    desc: 'Attentionではない小さな照合回路',
    icon: GitMerge,
    category: 'learning',
    variant: 'reference',
    recommendedDataset: 'logic:kv_attention2',
    trainingPreset: logicPreset('logic:kv_attention2'),
    ir: SAMPLE_KV_ATTENTION2_REF,
  },
  {
    key: 'seq101_rnn_ref',
    label: 'RNN風 101 参照',
    desc: '状態更新→受理判定の小回路',
    icon: BrainCircuit,
    category: 'sequence',
    variant: 'reference',
    recommendedDataset: 'logic:seq101',
    trainingPreset: logicPreset('logic:seq101'),
    ir: SAMPLE_SEQ101_RNN_REF,
  },
  {
    key: 'residual_xor',
    label: '残差XORデモ',
    desc: 'fanout+add(xor)のskip接続(v0.2 DAG)',
    icon: GitMerge,
    category: 'dag',
    variant: 'demo',
    recommendedDataset: 'logic:half_adder',
    trainingPreset: logicPreset('logic:half_adder'),
    ir: SAMPLE_RESIDUAL_XOR,
  },
  {
    key: 'mnist_mini',
    label: 'MNISTミニ',
    desc: '学習デモ: 60ゲート×1層(詳細ビュー+全監視機能対応)',
    icon: BrainCircuit,
    category: 'vision',
    variant: 'train',
    recommendedDataset: 'mnist',
    trainingPreset: MNIST_TRAINING_PRESET,
    trainable: true,
    ir: SAMPLE_MNIST_MINI,
  },
  {
    key: 'mnist_rot1000',
    label: 'MNIST rot 1000 学習',
    desc: '6 層 × 1000。層 0 は 2D 近傍 rot(1,28)、隠れ層は受容野 100% のずらし量 6,30,150,750,500',
    icon: RotateCw,
    category: 'vision',
    variant: 'train',
    recommendedDataset: 'mnist',
    trainingPreset: MNIST_W1000_TRAINING_PRESET,
    trainable: true,
    ir: SAMPLE_MNIST_ROT1000,
  },
  {
    key: 'mnist_bf4_1000',
    label: 'MNIST butterfly(4 入力)1000 学習',
    desc: '層 0 は rot_n で 4bit ずつ gather、隠れ層は stride 1,4,16,64,256 の循環 radix-4 バタフライ',
    icon: Network,
    category: 'vision',
    variant: 'train',
    recommendedDataset: 'mnist',
    trainingPreset: MNIST_W1000_TRAINING_PRESET,
    trainable: true,
    ir: SAMPLE_MNIST_BF4_1000,
  },
];
