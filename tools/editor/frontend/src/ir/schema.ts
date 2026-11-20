// IR JSON (magicdelta-ir v0.2) のTS型定義。griffith_logic/ir/schema.py と対応する。

export type IRNodeType =
  | 'lut_layer'
  | 'group_sum'
  | 'binarize'
  | 'flatten'
  | 'blackbox'
  | 'concat'
  | 'add'
  | 'conv2d'
  | 'or_pool2d'
  | 'constant'
  | 'upsample2d'
  | 'permute';

export interface IRPortJson {
  id: string;
  width: number;
  shape?: number[] | null;
  encoding?: string | null;
  semantics?: string | null;
}

export interface IRNodeJson {
  id: string;
  type: IRNodeType;
  params: Record<string, unknown>;
  train_info: Record<string, unknown>;
  meta: Record<string, unknown>; // エディタ専用namespace(座標等)
}

export interface NodeCapabilities {
  python_executable?: boolean;
  trainable?: boolean;
  ir_buildable?: boolean;
  gate_expandable?: boolean;
  hdl_exportable?: boolean;
}

export function nodeCapabilities(node: IRNodeJson): NodeCapabilities {
  const caps = (node.meta?.capabilities ?? {}) as NodeCapabilities;
  return caps;
}

// dst_slot: 多入力ノードの入力スロット番号(v0.2)。
// バックエンドと同一規約: =0はJSONキー省略、読み手は欠落=0として扱う。
export interface IREdgeJson {
  src: string;
  dst: string;
  dst_slot?: number;
}

export interface IRGraphJson {
  format: string;
  version: string;
  metadata: Record<string, unknown>;
  inputs: IRPortJson[];
  outputs: IRPortJson[];
  nodes: IRNodeJson[];
  edges: IREdgeJson[];
}

// ノード型別のparams(アクセス時にキャストして使う)
export interface LutLayerParams {
  in_width: number;
  out_width: number;
  n_inputs: number;
  indices: number[][]; // (n_inputs, out_width)
  truth_tables: string[]; // hex文字列 (out_width,)
}

export interface GroupSumParams {
  in_width: number;
  k: number;
  tau: number;
  /** クラスごとの整数の判定バイアス(長さ k、省略 = 全 0) */
  bias?: number[];
}

export type AddMode = 'or' | 'xor' | 'sum';

export const ADD_MODES: AddMode[] = ['or', 'xor', 'sum'];

/** concat: 出力幅 = sum(in_widths)。slot 0が出力ビット列の先頭 */
export interface ConcatParams {
  in_widths: number[]; // slot順の入力幅(2要素以上)
}

/** add: 全slot同幅。sumは論理領域の終端専用(HDL不可) */
export interface AddParams {
  width: number;
  mode: AddMode;
}

/** conv2d: 幅規約は入出力とも C*H*W のNCHW flatten */
export interface Conv2dParams {
  in_shape: number[]; // [C, H, W]
  out_channels: number;
  kernel_size: number | number[];
  stride: number | number[];
  padding: number | number[];
  tree_depth: number;
  n_inputs: number;
  indices_flat: number[][]; // (out_channels, n_inputs^tree_depth)
  truth_tables: string[];
}

export interface OrPool2dParams {
  in_shape: number[]; // [C, H, W]
  kernel_size: number;
  stride: number;
  padding?: number;
}

/** constant: 入力なし。bits[j] が出力ビットj */
export interface ConstantParams {
  width: number;
  bits: number[];
}

/** upsample2d: nearest拡大の固定配線。out[c,y,x]=in[c,y//scale,x//scale]。
    indexは持たず実行時にJIT計算する(griffith_logic/ir/schema.py参照) */
export interface UpSample2dParams {
  in_shape: number[]; // [C, H, W]
  scale: number;
}

/** permute: 入力をin_shape形状のreshape viewとみなし、permの軸順で
    並べ替えた恒等配線(要素数不変)。out_width=in_width。
    SpatialGroupSumのチャネル/空間並べ替えや画像thermometerの
    rgb/thr順序変換等を、全展開せず表現する */
export interface PermuteParams {
  in_shape: number[]; // reshape view dims(積 = in_width)
  perm: number[]; // range(in_shape.length)の順列
}

/** エッジのdst_slot読取り(キー欠落=0のv0.2規約) */
export function edgeSlot(e: IREdgeJson): number {
  return e.dst_slot ?? 0;
}

/** 多入力ノード型(2本以上のslot連番入力を取る) */
export function isMultiInputType(type: IRNodeType): boolean {
  return type === 'concat' || type === 'add';
}
