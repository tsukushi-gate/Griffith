// 複合ブロックテンプレート: パラメトリックなIR部分グラフ生成器(純関数)。
// 各生成器は {nodes, edges, entryId, exitId} を返す。idは`u{seq}_`draft規約の
// 採番コールバック(graphStore.createDraftIdAllocator)を受け、既存ノードと
// 衝突しない。座標はmeta.editor.positionへ相対配置で埋め、addSubgraphが
// ドロップ位置オフセットを適用する。

import {
  CirclePlus,
  Cpu,
  GitMerge,
  Grid3X3,
  Hash,
  Layers,
  Maximize2,
  MapPin,
  Merge,
  Network,
  Repeat,
  Shrink,
  Sigma,
} from 'lucide-react';
import type { LucideIcon } from 'lucide-react';
import { unrollRecurrent, type RecurrentCellType } from './recurrentUnit';
import { passThroughTT, placeholderIndices } from '../ir/graphToIr';
import {
  conv2dNode,
  lutInit,
  mulberry32,
  orPool2dNode,
  type Conv2dInitSpec,
  type InitPreset,
} from './nodeInit';
import { reductionTreeLuts, type ReductionTreeOptions } from './reductionTree';
import { conv2dOutShape, orPool2dOutShape } from '../ir/shapeInference';
import type { IREdgeJson, IRGraphJson, IRNodeJson } from '../ir/schema';

/** id採番コールバック(suffix→'u{seq}_{suffix}') */
export type IdAllocator = (suffix: string) => string;

export interface SubgraphTemplate {
  nodes: IRNodeJson[];
  edges: IREdgeJson[];
  entryId: string; // 外部入力を接続するノード
  exitId: string; // 外部出力へ接続するノード
}

// 相対配置の間隔(autoLayoutを使わずドロップ位置基準で読める配置にする)
const COL_X = 220;
const ROW_Y = 140;

const pos = (col: number, row = 0) => ({ x: col * COL_X, y: row * ROW_Y });

/** draft規約のlut_layerノード(placeholder indices/tt、build時にランダム初期化) */
function draftLut(
  id: string,
  inWidth: number,
  outWidth: number,
  position: { x: number; y: number },
): IRNodeJson {
  return {
    id,
    type: 'lut_layer',
    params: {
      in_width: inWidth,
      out_width: outWidth,
      n_inputs: 2,
      indices: placeholderIndices(inWidth, outWidth, 2),
      truth_tables: new Array<string>(outWidth).fill(passThroughTT(2)),
    },
    train_info: {
      parameterization: 'softmax',
      connections: 'random',
      selection: 'soft',
      initialized: false,
    },
    meta: { editor: { position } },
  };
}

/** 恒等lut_layer(幅維持・initialized)。残差ブロックのfanout元(entry)に使う */
function identityLut(
  id: string,
  width: number,
  position: { x: number; y: number },
): IRNodeJson {
  const idx = Array.from({ length: width }, (_, o) => o);
  return {
    id,
    type: 'lut_layer',
    params: {
      in_width: width,
      out_width: width,
      n_inputs: 2,
      indices: [idx, [...idx]],
      truth_tables: new Array<string>(width).fill('0xa'), // s0パススルー
    },
    train_info: {
      parameterization: 'softmax',
      connections: 'random',
      selection: 'soft',
    },
    meta: { editor: { position } },
  };
}

// 初期化プリセットを反映した lut_layer ノードを生成する。
// 'random' は従来どおり draft(placeholder + initialized:false → build時ランダム)。
// 'pass-through'/'residual-init' は具体的な indices/truth_tables を確定して置く。
let presetSeq = 0;
function presetLut(
  id: string,
  inWidth: number,
  outWidth: number,
  position: { x: number; y: number },
  preset: InitPreset,
): IRNodeJson {
  if (preset === 'random') return draftLut(id, inWidth, outWidth, position);
  const nInputs = 2;
  const { indices, truth_tables } = lutInit(
    inWidth,
    outWidth,
    nInputs,
    preset,
    mulberry32(0x9e3779b9 ^ (presetSeq++ & 0xffff)),
  );
  return {
    id,
    type: 'lut_layer',
    params: { in_width: inWidth, out_width: outWidth, n_inputs: nInputs, indices, truth_tables },
    train_info: {
      parameterization: 'softmax',
      connections: 'identity',
      selection: 'soft',
      initialized: true,
    },
    meta: { editor: { position } },
  };
}

/**
 * MLPブロック: LutLayer連鎖draft。幅列 = inWidth → hidden[] → outWidth。
 */
export function mlpBlock(
  inWidth: number,
  hidden: number[],
  outWidth: number,
  alloc: IdAllocator,
  preset: InitPreset = 'random',
): SubgraphTemplate {
  const widths = [inWidth, ...hidden, outWidth];
  const nodes: IRNodeJson[] = [];
  const edges: IREdgeJson[] = [];
  for (let i = 0; i < widths.length - 1; i++) {
    const id = alloc('lutlayer');
    nodes.push(presetLut(id, widths[i], widths[i + 1], pos(i), preset));
    if (i > 0) edges.push({ src: nodes[i - 1].id, dst: id });
  }
  return {
    nodes,
    edges,
    entryId: nodes[0].id,
    exitId: nodes[nodes.length - 1].id,
  };
}

/**
 * 残差ブロック: 恒等entry → fanout → [LutLayer×innerLayers] → add(merge)。
 * skip接続はadd slot 1(残差エッジ描画対象)。
 */
export function residualBlock(
  width: number,
  innerLayers: number,
  merge: 'or' | 'xor',
  alloc: IdAllocator,
  preset: InitPreset = 'random',
): SubgraphTemplate {
  const entry = identityLut(alloc('res_in'), width, pos(0));
  const nodes: IRNodeJson[] = [entry];
  const edges: IREdgeJson[] = [];

  let prev = entry.id;
  for (let i = 0; i < innerLayers; i++) {
    const id = alloc('lutlayer');
    nodes.push(presetLut(id, width, width, pos(i + 1), preset));
    edges.push({ src: prev, dst: id });
    prev = id;
  }

  const addId = alloc('add');
  nodes.push({
    id: addId,
    type: 'add',
    params: { width, mode: merge },
    train_info: {},
    meta: { editor: { position: pos(innerLayers + 1) } },
  });
  edges.push({ src: prev, dst: addId }); // slot 0(本流)
  edges.push({ src: entry.id, dst: addId, dst_slot: 1 }); // slot 1(skip)

  return { nodes, edges, entryId: entry.id, exitId: addId };
}

/**
 * 分類ヘッド: LutLayer(inWidth→k*neuronsPerClass draft) + GroupSum(k, tau)。
 */
export function classifierHead(
  inWidth: number,
  k: number,
  tau: number,
  alloc: IdAllocator,
  neuronsPerClass = 6,
  preset: InitPreset = 'random',
): SubgraphTemplate {
  const outWidth = k * neuronsPerClass;
  const lut = presetLut(alloc('lutlayer'), inWidth, outWidth, pos(0), preset);
  const gsId = alloc('groupsum');
  const gs: IRNodeJson = {
    id: gsId,
    type: 'group_sum',
    params: { in_width: outWidth, k, tau },
    train_info: {},
    meta: { editor: { position: pos(1) } },
  };
  return {
    nodes: [lut, gs],
    edges: [{ src: lut.id, dst: gsId }],
    entryId: lut.id,
    exitId: gsId,
  };
}

// --- conv2d / or_pool2d 単体ノード生成 ---

/** conv2d 単体ノードを SubgraphTemplate(1ノード)として生成する。 */
export function conv2dGenBlock(
  params: Conv2dParams,
  alloc: IdAllocator,
  preset: InitPreset = 'random',
): SubgraphTemplate {
  const spec: Conv2dInitSpec = {
    inShape: [params.inChannels, params.inH, params.inW],
    outChannels: params.outChannels,
    kernelSize: params.kernel,
    stride: params.stride,
    padding: params.padding,
    treeDepth: params.treeDepth,
    nInputs: params.nInputs,
  };
  const id = alloc('conv2d');
  const node = conv2dNode(id, spec, preset, mulberry32(params.seed ?? 0x51ed270b), pos(0));
  return { nodes: [node], edges: [], entryId: id, exitId: id };
}

/** or_pool2d 単体ノードを SubgraphTemplate(1ノード)として生成する。 */
export function orPool2dGenBlock(
  params: OrPool2dBlockParams,
  alloc: IdAllocator,
): SubgraphTemplate {
  const id = alloc('or_pool2d');
  const node = orPool2dNode(
    id,
    [params.inChannels, params.inH, params.inW],
    params.kernel,
    params.stride,
    params.padding,
    pos(0),
  );
  return { nodes: [node], edges: [], entryId: id, exitId: id };
}

// --- モデル部品(Attention/Transformer/PE/Embedding/DWN/UpSample) ---
// D系列(blocks/*.py)で実装済みの部品を「GUIで組む」ためのフロント純関数マクロ。
// 新しいモデル数理はここで実装しない(本体 griffith_logic/blocks/ の規約を写すだけ)。

/** 位置エンコーディングのコード種別(griffith_logic/blocks/positional.py と一致)。 */
export type PositionalCode = 'binary' | 'gray' | 'thermometer';

export const POSITIONAL_CODES: PositionalCode[] = ['binary', 'gray', 'thermometer'];

/** attention/transformer共通のモード・正規化選択肢(blocks/attention.py・transformer.py準拠)。 */
export const ATTENTION_MODES = ['soft', 'hard'] as const;
export const TRANSFORMER_NORMS = ['layernorm', 'batchnorm', 'none'] as const;
export const TRANSFORMER_MERGES = ['or', 'xor'] as const;

/**
 * 位置→ビット列テーブルを生成する(positional.py の position_codes と同一規約)。
 * - binary/gray: bit j = ((p or gray(p)) >> j) & 1(seq_len ≦ 2^width が必要)
 * - thermometer: bit j = 1 if p > j(width ≧ seq_len-1 が必要)
 * @returns (seqLen, width) の {0,1} テーブル
 */
export function positionCodes(
  seqLen: number,
  width: number,
  code: PositionalCode,
): number[][] {
  const table: number[][] = [];
  for (let p = 0; p < seqLen; p++) {
    const row: number[] = [];
    if (code === 'thermometer') {
      for (let j = 0; j < width; j++) row.push(p > j ? 1 : 0);
    } else {
      const g = code === 'gray' ? p ^ (p >> 1) : p;
      for (let j = 0; j < width; j++) row.push((g >> j) & 1);
    }
    table.push(row);
  }
  return table;
}

/** 位置テーブルの位置優先flatten(constantノードのbits規約: index p*width+j)。 */
export function positionBitsFlat(
  seqLen: number,
  width: number,
  code: PositionalCode,
): number[] {
  return positionCodes(seqLen, width, code).flat();
}

/** 位置コードのパラメータ制約(binary/gray: seq_len≦2^width、thermometer: width≧seq_len-1)。 */
export function positionalCodeValid(
  seqLen: number,
  width: number,
  code: PositionalCode,
): boolean {
  if (seqLen < 1 || width < 1) return false;
  if (code === 'thermometer') return width >= seqLen - 1;
  return seqLen <= 1 << width;
}

/**
 * Attention/Transformer共通の contract 文字列(to_ir.py の _emit_attention_blackbox と一致)。
 */
function attentionContract(dModel: number, inWidth: number, outWidth: number): Record<string, string> {
  const s = Math.floor(inWidth / dModel);
  return {
    input: `(B, S, ${dModel})のNCHW flatten(S=${s})`,
    output: `(B, S, ${Math.floor(outWidth / Math.max(s, 1))})のflatten`,
    hdl: 'softmax/argmaxを含むためHDL生成不可',
  };
}

/**
 * LogicAttention → blackbox+contractノード。
 * IR規約は to_ir.py の _emit_attention_blackbox と同一(torch_class/config/in_width/out_width)。
 * self-attention契約(d_v=d_model)なので out_width = in_width。
 */
export function attentionBlackboxBlock(
  params: AttentionParams,
  alloc: IdAllocator,
): SubgraphTemplate {
  const inWidth = params.seqLen * params.dModel;
  const outWidth = inWidth; // d_v = d_model(既定 v_sizes=[d_model])
  const id = alloc('logicattention');
  const node: IRNodeJson = {
    id,
    type: 'blackbox',
    params: {
      torch_class: 'LogicAttention',
      config: {
        d_model: params.dModel,
        n_heads: params.nHeads,
        mode: params.mode,
        similarity: 'xnor',
      },
      in_width: inWidth,
      out_width: outWidth,
    },
    train_info: { contract: attentionContract(params.dModel, inWidth, outWidth) },
    meta: {
      editor: {
        position: pos(0),
        detail: `LogicAttention d=${params.dModel} h=${params.nHeads} ${params.mode}`,
      },
    },
  };
  return { nodes: [node], edges: [], entryId: id, exitId: id };
}

/**
 * LogicTransformerBlock → blackbox+contractノード。幅不変(B,S,D)→(B,S,D)。
 */
export function transformerBlackboxBlock(
  params: TransformerParams,
  alloc: IdAllocator,
): SubgraphTemplate {
  const inWidth = params.seqLen * params.dModel;
  const outWidth = inWidth;
  const id = alloc('logictransformerblock');
  const node: IRNodeJson = {
    id,
    type: 'blackbox',
    params: {
      torch_class: 'LogicTransformerBlock',
      config: {
        d_model: params.dModel,
        n_heads: params.nHeads,
        norm: params.norm,
        merge: params.merge,
      },
      in_width: inWidth,
      out_width: outWidth,
    },
    train_info: { contract: attentionContract(params.dModel, inWidth, outWidth) },
    meta: {
      editor: {
        position: pos(0),
        detail: `LogicTransformerBlock d=${params.dModel} h=${params.nHeads} ${params.norm}`,
      },
    },
  };
  return { nodes: [node], edges: [], entryId: id, exitId: id };
}

/**
 * PositionalEncoding → constantノード + concatノード(to_ir.py _emit_positional と一致)。
 * 外部入力は concat の slot 0 に接続する(定数線は slot 1)。
 */
export function positionalEncodingBlock(
  params: PositionalEncodingParams,
  alloc: IdAllocator,
): SubgraphTemplate {
  const constWidth = params.seqLen * params.codeWidth;
  const bits = positionBitsFlat(params.seqLen, params.codeWidth, params.code);
  const constId = alloc('constant');
  const catId = alloc('concat');
  const constNode: IRNodeJson = {
    id: constId,
    type: 'constant',
    params: { width: constWidth, bits },
    train_info: {
      from: 'LogicPositionalEncoding',
      code: params.code,
      seq_len: params.seqLen,
    },
    meta: { editor: { position: pos(0, 1) } },
  };
  const catNode: IRNodeJson = {
    id: catId,
    type: 'concat',
    params: { in_widths: [params.inWidth, constWidth] },
    train_info: { from: 'LogicPositionalEncoding' },
    meta: { editor: { position: pos(1) } },
  };
  return {
    nodes: [constNode, catNode],
    edges: [{ src: constId, dst: catId, dst_slot: 1 }],
    entryId: catId,
    exitId: catId,
  };
}

/**
 * Embedding(簡易) → 単一 lut_layer。
 * vocab×dim の完全LUTテーブルは過大なため、one-hot入力前提の lut_layer 初期化で表す
 * (in_width=vocab の one-hot → out_width=dim)。判断は報告参照。
 */
export function embeddingBlock(
  params: EmbeddingParams,
  alloc: IdAllocator,
  preset: InitPreset = 'random',
): SubgraphTemplate {
  const id = alloc('embedding');
  const node = presetLut(id, params.vocab, params.dim, pos(0), preset);
  node.train_info = {
    ...node.train_info,
    kind: 'embedding',
    note: 'one-hot入力前提の簡易Embedding',
  };
  return { nodes: [node], edges: [], entryId: id, exitId: id };
}

/**
 * DWN層 → lut_layer + train_info.kind='dwn'(抽出器の逆)。
 * DWNのeval意味論は二値入力では標準lut_layer意味論と厳密一致(extract.py参照)なので、
 * draft lut_layer(build時ランダム初期化)に kind='dwn' を付す。
 */
export function dwnBlock(params: DwnParams, alloc: IdAllocator): SubgraphTemplate {
  const { inWidth, outWidth, nInputs, mapping } = params;
  const id = alloc('dwn');
  const node: IRNodeJson = {
    id,
    type: 'lut_layer',
    params: {
      in_width: inWidth,
      out_width: outWidth,
      n_inputs: nInputs,
      indices: placeholderIndices(inWidth, outWidth, nInputs),
      truth_tables: new Array<string>(outWidth).fill(passThroughTT(nInputs)),
    },
    train_info: {
      kind: 'dwn',
      mapping,
      efd: 'none',
      parameterization: 'softmax',
      connections: 'random',
      selection: 'soft',
      initialized: false,
    },
    meta: { editor: { position: pos(0), detail: `DWN ${mapping} n=${nInputs}` } },
  };
  return { nodes: [node], edges: [], entryId: id, exitId: id };
}

/**
 * UpSample2d(nearest) → 恒等配線 lut_layer(_extract_upsample2d の意味論)。
 * out[c,y,x] = in[c, y//s, x//s] の複製配線。抽出器は n_inputs=1(tt=0x2)で表すが、
 * エディタの学習可能build経路(LutLayerは n_inputs∈{2,4,6})に載せるため、
 * 両slotとも source pixel を指す n_inputs=2 の slot0 パススルー(tt=0xa)で等価表現する
 * (出力=slot0=source pixel でnearest複製と一致し、build/学習も可能)。
 */
export function upsample2dBlock(
  params: UpSample2dParams,
  alloc: IdAllocator,
): SubgraphTemplate {
  const { inC, inH, inW, scale } = params;
  const hOut = inH * scale;
  const wOut = inW * scale;
  const widthOut = inC * hOut * wOut;
  const inWidth = inC * inH * inW;
  const row = new Array<number>(widthOut);
  for (let c = 0; c < inC; c++) {
    for (let y = 0; y < hOut; y++) {
      for (let x = 0; x < wOut; x++) {
        const outIdx = (c * hOut + y) * wOut + x;
        row[outIdx] = (c * inH + Math.floor(y / scale)) * inW + Math.floor(x / scale);
      }
    }
  }
  const id = alloc('upsample2d');
  const node: IRNodeJson = {
    id,
    type: 'lut_layer',
    params: {
      in_width: inWidth,
      out_width: widthOut,
      n_inputs: 2,
      indices: [row, [...row]],
      truth_tables: new Array<string>(widthOut).fill(passThroughTT(2)),
    },
    train_info: {
      kind: 'binarybrain',
      from: 'UpSample2d',
      parameterization: 'softmax',
      connections: 'identity',
      selection: 'soft',
      initialized: true,
    },
    meta: { editor: { position: pos(0), detail: `UpSample2d x${scale}` } },
  };
  return { nodes: [node], edges: [], entryId: id, exitId: id };
}

/**
 * LL-ViT複合ブロック: パッチ埋込 → PositionalEncoding → Transformer×N → 分類ヘッド。
 * backend templates.py の `_build_llvit` と同型の SubgraphTemplate 版。
 *
 * IR表現の判断(パッチ埋め込みの前段):
 *   models/llvit.py の PatchEmbed は画像→パッチへの nn.Conv2d 射影(連続値)で、二値論理IRでは
 *   表現できない ANN 前段。よって入力は「ビット化済みパッチ列」(幅=seqLen×patchBits)とし、
 *   画像→パッチ分割→ビット化の前処理は学習スクリプト(experiments/train_llvit.py)側の責務とする。
 *   IR上のパッチ埋込は draft lut_layer(patchBits/トークン → dEmbed/トークン)で表す。
 *
 * blackbox(LogicTransformerBlock)を含むため HDL/エディタbuild不可(validateのみ)。
 * 学習は Python import 経路で行う。
 */
export function llvitBlock(
  params: LLViTParams,
  alloc: IdAllocator,
  preset: InitPreset = 'random',
): SubgraphTemplate {
  const { seqLen, dEmbed, posBits, depth, nHeads, patchBits, numClasses } = params;
  const code = params.code ?? 'binary';
  const neuronsPerClass = params.neuronsPerClass ?? 6;
  const dModel = dEmbed + posBits;
  const inWidth = seqLen * patchBits;
  const embedWidth = seqLen * dEmbed;
  const posWidth = seqLen * posBits;
  const tokenWidth = seqLen * dModel;

  const nodes: IRNodeJson[] = [];
  const edges: IREdgeJson[] = [];
  let col = 0;

  // パッチ埋め込み: ビット化パッチ → トークン埋め込み(draft、学習可能)
  const patchId = alloc('patch_embed');
  const patchNode = presetLut(patchId, inWidth, embedWidth, pos(col), preset);
  patchNode.train_info = {
    ...patchNode.train_info,
    kind: 'patch_embed',
    note: 'ビット化済みパッチ→トークン埋め込み(画像→パッチ化はscript責務)',
  };
  nodes.push(patchNode);
  col += 1;

  // PositionalEncoding: 位置コード定数線 + 連結(positional.py 規約)
  const constId = alloc('constant');
  const catId = alloc('concat');
  nodes.push({
    id: constId,
    type: 'constant',
    params: { width: posWidth, bits: positionBitsFlat(seqLen, posBits, code) },
    train_info: { from: 'LogicPositionalEncoding', code, seq_len: seqLen },
    meta: { editor: { position: pos(col, 1) } },
  });
  nodes.push({
    id: catId,
    type: 'concat',
    params: { in_widths: [embedWidth, posWidth] },
    train_info: { from: 'LogicPositionalEncoding' },
    meta: { editor: { position: pos(col) } },
  });
  edges.push({ src: patchId, dst: catId, dst_slot: 0 });
  edges.push({ src: constId, dst: catId, dst_slot: 1 });
  col += 1;

  // Transformerブロック×N(blackbox、幅不変 tokenWidth→tokenWidth)
  let prev = catId;
  for (let i = 0; i < depth; i++) {
    const tid = alloc('logictransformerblock');
    nodes.push({
      id: tid,
      type: 'blackbox',
      params: {
        torch_class: 'LogicTransformerBlock',
        config: { d_model: dModel, n_heads: nHeads, norm: 'layernorm', merge: 'or' },
        in_width: tokenWidth,
        out_width: tokenWidth,
      },
      train_info: { contract: attentionContract(dModel, tokenWidth, tokenWidth) },
      meta: {
        editor: { position: pos(col), detail: `LogicTransformerBlock d=${dModel} h=${nHeads} #${i}` },
      },
    });
    edges.push({ src: prev, dst: tid });
    prev = tid;
    col += 1;
  }

  // 分類ヘッド: lut_layer(tokenWidth → numClasses×neuronsPerClass) + GroupSum
  const outWidth = numClasses * neuronsPerClass;
  const headId = alloc('lutlayer');
  nodes.push(presetLut(headId, tokenWidth, outWidth, pos(col), preset));
  col += 1;
  const gsId = alloc('groupsum');
  nodes.push({
    id: gsId,
    type: 'group_sum',
    params: { in_width: outWidth, k: numClasses, tau: 1.0 },
    train_info: {},
    meta: { editor: { position: pos(col) } },
  });
  edges.push({ src: prev, dst: headId });
  edges.push({ src: headId, dst: gsId });

  return { nodes, edges, entryId: patchId, exitId: gsId };
}

// --- パレット/ダイアログ用のブロック定義とパラメータ ---

export type BlockKind =
  | 'mlp'
  | 'residual'
  | 'classifier'
  | 'conv2d'
  | 'or_pool2d'
  | 'reduction_tree'
  // モデル部品
  | 'attention'
  | 'transformer'
  | 'positional_encoding'
  | 'embedding'
  | 'dwn'
  | 'upsample2d'
  // LL-ViT複合(パッチ埋込+PE+Transformer×N+分類ヘッド)
  | 'llvit'
  // Recurrent時間展開(LSTM/GRUのunroll)
  | 'recurrent';

/** LSTM/GRUセルを unroll 回展開するパラメータ。*/
export interface RecurrentParams {
  cellType: RecurrentCellType;
  unroll: number;
  inWidth: number;
  hidden: number;
  depth: number;
}

export interface AttentionParams {
  dModel: number;
  seqLen: number;
  nHeads: number;
  mode: (typeof ATTENTION_MODES)[number];
}

export interface TransformerParams {
  dModel: number;
  seqLen: number;
  nHeads: number;
  norm: (typeof TRANSFORMER_NORMS)[number];
  merge: (typeof TRANSFORMER_MERGES)[number];
}

export interface PositionalEncodingParams {
  inWidth: number;
  seqLen: number;
  codeWidth: number;
  code: PositionalCode;
}

export interface EmbeddingParams {
  vocab: number;
  dim: number;
}

export interface DwnParams {
  inWidth: number;
  outWidth: number;
  nInputs: number;
  mapping: 'random' | 'learnable';
}

export interface UpSample2dParams {
  inC: number;
  inH: number;
  inW: number;
  scale: number;
}

/** LL-ViT複合ブロックのパラメータ。dModel = dEmbed + posBits(PE連結後のトークン幅)。*/
export interface LLViTParams {
  seqLen: number; // トークン数(実寸は patch_tokens+1、テンプレは小型固定)
  dEmbed: number; // パッチ埋め込みビット幅/トークン
  posBits: number; // 位置コード幅(binary: seqLen ≦ 2^posBits が必要)
  depth: number; // Transformerブロック段数
  nHeads: number;
  patchBits: number; // 入力の1トークンあたりビット幅(ビット化済みパッチ)
  numClasses: number;
  neuronsPerClass?: number; // 既定6
  code?: PositionalCode; // 既定binary
}

export interface MlpParams {
  inWidth: number;
  hidden: number[];
  outWidth: number;
}

export interface ResidualParams {
  width: number;
  innerLayers: number;
  merge: 'or' | 'xor';
}

export interface ClassifierParams {
  inWidth: number;
  k: number;
  tau: number;
}

export interface Conv2dParams {
  inChannels: number;
  inH: number;
  inW: number;
  outChannels: number;
  kernel: number;
  stride: number;
  padding: number;
  treeDepth: number;
  nInputs: number;
  seed?: number;
}

export interface OrPool2dBlockParams {
  inChannels: number;
  inH: number;
  inW: number;
  kernel: number;
  stride: number;
  padding: number;
}

export interface ReductionTreeParams {
  inWidth: number;
  fanin: number;
  minWidth: number;
}

export type BlockParams =
  | { kind: 'mlp'; params: MlpParams }
  | { kind: 'residual'; params: ResidualParams }
  | { kind: 'classifier'; params: ClassifierParams }
  | { kind: 'conv2d'; params: Conv2dParams }
  | { kind: 'or_pool2d'; params: OrPool2dBlockParams }
  | { kind: 'reduction_tree'; params: ReductionTreeParams }
  | { kind: 'attention'; params: AttentionParams }
  | { kind: 'transformer'; params: TransformerParams }
  | { kind: 'positional_encoding'; params: PositionalEncodingParams }
  | { kind: 'embedding'; params: EmbeddingParams }
  | { kind: 'dwn'; params: DwnParams }
  | { kind: 'upsample2d'; params: UpSample2dParams }
  | { kind: 'llvit'; params: LLViTParams }
  | { kind: 'recurrent'; params: RecurrentParams };

export interface BlockDef {
  kind: BlockKind;
  label: string;
  desc: string;
  icon: LucideIcon;
  color: string;
}

export const BLOCK_DEFS: BlockDef[] = [
  {
    kind: 'mlp',
    label: 'MLPブロック',
    desc: 'LutLayer連鎖(幅列を指定)',
    icon: Layers,
    color: 'var(--slot-0)',
  },
  {
    kind: 'residual',
    label: '残差ブロック',
    desc: 'fanout + add(or/xor)のskip接続',
    icon: GitMerge,
    color: 'var(--slot-2)',
  },
  {
    kind: 'classifier',
    label: '分類ヘッド',
    desc: 'LutLayer + GroupSum(k分類)',
    icon: Sigma,
    color: 'var(--slot-1)',
  },
];

// conv2d/or_pool2d のパレットカード定義(Palette.tsx が参照)。
export const CONV_DEFS: BlockDef[] = [
  {
    kind: 'conv2d',
    label: 'Conv2d',
    desc: '論理畳み込み(indices自動生成)',
    icon: Grid3X3,
    color: 'var(--slot-1)',
  },
  {
    kind: 'or_pool2d',
    label: 'OrPool2d',
    desc: 'ORプーリング(窓≦6入力)',
    icon: Shrink,
    color: 'var(--slot-3)',
  },
  {
    kind: 'reduction_tree',
    label: '縮約ツリー',
    desc: 'BinaryBrain流 fanin縮約LutLayer列',
    icon: Shrink,
    color: 'var(--slot-4)',
  },
];

// モデル部品のパレットカード定義(Palette.tsx が「モデル部品」節で参照)。
// Attention/TransformerはIR上blackbox(HDL不可・エディタbuild学習不可=Python import経路で学習)。
export const PARTS_DEFS: BlockDef[] = [
  {
    kind: 'attention',
    label: 'Attention',
    desc: 'LogicAttention(blackbox+contract)',
    icon: Network,
    color: 'var(--slot-5)',
  },
  {
    kind: 'transformer',
    label: 'TransformerBlock',
    desc: 'LogicTransformerBlock(blackbox)',
    icon: Network,
    color: 'var(--slot-5)',
  },
  {
    kind: 'positional_encoding',
    label: 'PositionalEncoding',
    desc: '位置コード定数+連結(binary/gray/thermometer)',
    icon: MapPin,
    color: 'var(--slot-5)',
  },
  {
    kind: 'embedding',
    label: 'Embedding',
    desc: 'one-hot→dim の簡易埋め込み(lut_layer)',
    icon: Hash,
    color: 'var(--slot-0)',
  },
  {
    kind: 'dwn',
    label: 'DWN層',
    desc: 'lut_layer(kind=dwn)。DWN学習則を推奨',
    icon: Cpu,
    color: 'var(--slot-0)',
  },
  {
    kind: 'upsample2d',
    label: 'UpSample2d',
    desc: 'nearest拡大=恒等配線(BinaryBrain系)',
    icon: Maximize2,
    color: 'var(--slot-3)',
  },
  // RNN/LSTM の時間展開(unroll)。セルを既存ノード型で展開する。
  {
    kind: 'recurrent',
    label: 'RNN/LSTM(unroll)',
    desc: 'LSTM/GRUセルをT回時間展開(lut/concat/add合成)',
    icon: Repeat,
    color: 'var(--slot-2)',
  },
];

// パレット単体ノード追加用のアイコン(Palette.tsxから参照)
export const CONCAT_ICON = Merge;
export const ADD_ICON = CirclePlus;

/** kind+paramsから部分グラフを生成する(ダイアログ/ギャラリー共用ディスパッチャ) */
export function generateBlock(
  spec: BlockParams,
  alloc: IdAllocator,
  preset: InitPreset = 'random',
): SubgraphTemplate {
  switch (spec.kind) {
    case 'mlp':
      return mlpBlock(spec.params.inWidth, spec.params.hidden, spec.params.outWidth, alloc, preset);
    case 'residual':
      return residualBlock(
        spec.params.width,
        spec.params.innerLayers,
        spec.params.merge,
        alloc,
        preset,
      );
    case 'classifier':
      return classifierHead(spec.params.inWidth, spec.params.k, spec.params.tau, alloc, 6, preset);
    case 'conv2d':
      return conv2dGenBlock(spec.params, alloc, preset);
    case 'or_pool2d':
      return orPool2dGenBlock(spec.params, alloc);
    case 'reduction_tree': {
      const opts: ReductionTreeOptions = {
        fanin: spec.params.fanin,
        minWidth: spec.params.minWidth,
        preset,
        seed: 0x2545f491,
      };
      return reductionTreeLuts(spec.params.inWidth, alloc, opts);
    }
    case 'attention':
      return attentionBlackboxBlock(spec.params, alloc);
    case 'transformer':
      return transformerBlackboxBlock(spec.params, alloc);
    case 'positional_encoding':
      return positionalEncodingBlock(spec.params, alloc);
    case 'embedding':
      return embeddingBlock(spec.params, alloc, preset);
    case 'dwn':
      return dwnBlock(spec.params, alloc);
    case 'upsample2d':
      return upsample2dBlock(spec.params, alloc);
    case 'llvit':
      return llvitBlock(spec.params, alloc, preset);
    case 'recurrent':
      return unrollRecurrent(spec.params, alloc);
  }
}

/** ブロックの外部入出力幅(standalone IR化と接続検証に使う) */
export function blockIoWidths(spec: BlockParams): { in: number; out: number } {
  switch (spec.kind) {
    case 'mlp':
      return { in: spec.params.inWidth, out: spec.params.outWidth };
    case 'residual':
      return { in: spec.params.width, out: spec.params.width };
    case 'classifier':
      return { in: spec.params.inWidth, out: spec.params.k };
    case 'conv2d': {
      const p = spec.params;
      const shape = conv2dOutShape({
        in_shape: [p.inChannels, p.inH, p.inW],
        out_channels: p.outChannels,
        kernel_size: p.kernel,
        stride: p.stride,
        padding: p.padding,
      });
      return {
        in: p.inChannels * p.inH * p.inW,
        out: shape ? shape[0] * shape[1] * shape[2] : 0,
      };
    }
    case 'or_pool2d': {
      const p = spec.params;
      const shape = orPool2dOutShape({
        in_shape: [p.inChannels, p.inH, p.inW],
        kernel_size: p.kernel,
        stride: p.stride,
        padding: p.padding,
      });
      return {
        in: p.inChannels * p.inH * p.inW,
        out: shape ? shape[0] * shape[1] * shape[2] : 0,
      };
    }
    case 'reduction_tree': {
      const p = spec.params;
      const w = reductionTreeWidthsForIo(p.inWidth, p.fanin, p.minWidth);
      return { in: p.inWidth, out: w };
    }
    case 'attention':
    case 'transformer': {
      const w = spec.params.seqLen * spec.params.dModel;
      return { in: w, out: w };
    }
    case 'positional_encoding': {
      const p = spec.params;
      return { in: p.inWidth, out: p.inWidth + p.seqLen * p.codeWidth };
    }
    case 'embedding':
      return { in: spec.params.vocab, out: spec.params.dim };
    case 'dwn':
      return { in: spec.params.inWidth, out: spec.params.outWidth };
    case 'upsample2d': {
      const p = spec.params;
      return {
        in: p.inC * p.inH * p.inW,
        out: p.inC * p.inH * p.scale * p.inW * p.scale,
      };
    }
    case 'llvit': {
      const p = spec.params;
      return { in: p.seqLen * p.patchBits, out: p.numClasses };
    }
    case 'recurrent': {
      const p = spec.params;
      return { in: p.unroll * p.inWidth, out: p.hidden };
    }
  }
}

/** reduction_tree の最終出力幅を求める(blockIoWidths用の薄いラッパ)。 */
function reductionTreeWidthsForIo(inWidth: number, fanin: number, minWidth: number): number {
  const f = Math.max(2, fanin);
  const m = Math.max(1, minWidth);
  let w = inWidth;
  while (w > m) {
    const next = Math.max(m, Math.ceil(w / f));
    if (next >= w) break;
    w = next;
  }
  return w;
}

/**
 * テンプレートを単体モデルIRへ包む(ギャラリー用):
 * x(inWidth) → entry … exit → y(outWidth)。
 */
export function templateToGraph(
  name: string,
  tpl: SubgraphTemplate,
  inWidth: number,
  outWidth: number,
): IRGraphJson {
  return {
    format: 'magicdelta-ir',
    version: '0.2.0',
    metadata: { name },
    inputs: [{ id: 'x', width: inWidth, shape: null, encoding: null, semantics: null }],
    outputs: [
      { id: 'y', width: outWidth, shape: null, encoding: null, semantics: null },
    ],
    nodes: tpl.nodes,
    edges: [
      { src: 'x', dst: tpl.entryId },
      ...tpl.edges,
      { src: tpl.exitId, dst: 'y' },
    ],
  };
}
