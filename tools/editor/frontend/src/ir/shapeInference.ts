// IRノード列に対する形状推論と診断(v0.2多入力DAG対応)。
// バックエンドvalidate(ir/validate.py)のフロント版サブセット:
// エッジ幅整合(slot別)・arity表・slot重複/欠番/範囲外・group_sum割り切り・
// conv/poolのshape伝播を検査する。fanout(同一srcから複数エッジ)は合法。
// 接続操作自体はブロックせず、エラー存在中は訓練開始/コード生成を無効化する。

import type { IREdgeJson, IRNodeJson, IRNodeType, IRPortJson } from './schema';
import { edgeSlot } from './schema';

export interface ShapeDiagnostic {
  level: 'error' | 'warning';
  message: string;
  nodeId?: string;
  edgeId?: string; // `${src}->${dst}` 形式(FlowCanvasの赤表示用)
}

export interface NodeWidths {
  in: number | null; // slot共通の入力幅(concatはslot別のためnull)
  out: number | null;
}

export interface ShapeResult {
  widths: Record<string, NodeWidths>;
  diagnostics: ShapeDiagnostic[];
}

/** ノード型別の入力エッジ本数(min, max)。maxのnullは上限なし。validate.pyのNODE_ARITYと同一 */
export const NODE_ARITY: Record<IRNodeType, [number, number | null]> = {
  lut_layer: [1, 1],
  group_sum: [1, 1],
  binarize: [1, 1],
  flatten: [1, 1],
  blackbox: [1, 1],
  concat: [2, null],
  add: [2, null],
  conv2d: [1, 1],
  or_pool2d: [1, 1],
  constant: [0, 0],
  upsample2d: [1, 1],
  permute: [1, 1],
};

/** int または [h, w] を (h, w) に正規化する(validate.pyの_pairと同一) */
function pair(v: unknown): [number, number] | null {
  if (Array.isArray(v) && v.length >= 2) return [Number(v[0]), Number(v[1])];
  if (typeof v === 'number') return [v, v];
  return null;
}

function inShape(p: Record<string, unknown>): [number, number, number] | null {
  const s = p.in_shape;
  if (!Array.isArray(s) || s.length !== 3) return null;
  const [c, h, w] = s.map(Number);
  if (![c, h, w].every((v) => Number.isInteger(v) && v > 0)) return null;
  return [c, h, w];
}

/** conv2dの出力形状 [out_channels, H_out, W_out](in_shape等が不正ならnull) */
export function conv2dOutShape(
  p: Record<string, unknown>,
): [number, number, number] | null {
  const shape = inShape(p);
  const k = pair(p.kernel_size);
  const s = pair(p.stride);
  const pd = pair(p.padding);
  const oc = Number(p.out_channels);
  if (!shape || !k || !s || !pd || !Number.isInteger(oc) || oc <= 0) return null;
  const [, h, w] = shape;
  const hOut = Math.floor((h + 2 * pd[0] - k[0]) / s[0]) + 1;
  const wOut = Math.floor((w + 2 * pd[1] - k[1]) / s[1]) + 1;
  return [oc, hOut, wOut];
}

/** or_pool2dの出力形状 [C, H_out, W_out](in_shape等が不正ならnull) */
export function orPool2dOutShape(
  p: Record<string, unknown>,
): [number, number, number] | null {
  const shape = inShape(p);
  const k = Number(p.kernel_size);
  const s = Number(p.stride);
  const pd = Number(p.padding ?? 0);
  if (!shape || !Number.isInteger(k) || k <= 0 || !Number.isInteger(s) || s <= 0) {
    return null;
  }
  const [c, h, w] = shape;
  const hOut = Math.floor((h + 2 * pd - k) / s) + 1;
  const wOut = Math.floor((w + 2 * pd - k) / s) + 1;
  return [c, hOut, wOut];
}

/** upsample2d(nearest)の出力形状 [C, H_out, W_out](in_shape/scaleが不正ならnull) */
export function upsample2dOutShape(
  p: Record<string, unknown>,
): [number, number, number] | null {
  const shape = inShape(p);
  const s = Number(p.scale);
  if (!shape || !Number.isInteger(s) || s <= 0) return null;
  const [c, h, w] = shape;
  return [c, h * s, w * s];
}

/** permuteの入力幅(in_shapeの積)。in_shapeが不正ならnull */
function permuteWidth(p: Record<string, unknown>): number | null {
  const dims = p.in_shape;
  if (!Array.isArray(dims) || dims.length === 0) return null;
  if (!dims.every((v) => Number.isInteger(v) && Number(v) > 0)) return null;
  return dims.reduce((a: number, v) => a * Number(v), 1);
}

/** permuteのpermが有効な順列か(in_shapeと同じ要素数の0..n-1並べ替え) */
function isValidPerm(p: Record<string, unknown>): boolean {
  const dims = p.in_shape;
  const perm = p.perm;
  if (!Array.isArray(dims) || !Array.isArray(perm)) return false;
  const n = dims.length;
  if (perm.length !== n) return false;
  const sorted = [...perm].map(Number).sort((a, b) => a - b);
  return sorted.every((v, i) => v === i);
}

/** ノード型ごとの(in, out)幅。validate.py の node_in_width/node_out_width と同一規約。 */
export function nodeWidths(n: IRNodeJson): NodeWidths {
  const p = n.params as Record<string, number | undefined>;
  switch (n.type) {
    case 'lut_layer':
      return { in: p.in_width ?? null, out: p.out_width ?? null };
    case 'group_sum':
      return { in: p.in_width ?? null, out: p.k ?? null };
    case 'binarize':
    case 'flatten':
      return { in: p.width ?? null, out: p.width ?? null };
    case 'blackbox':
      return { in: p.in_width ?? null, out: p.out_width ?? null };
    case 'concat': {
      // 入力幅はslot別(nodeSlotInWidthで引く)
      const widths = n.params.in_widths;
      if (!Array.isArray(widths) || widths.length === 0) return { in: null, out: null };
      return { in: null, out: widths.reduce((a: number, w) => a + Number(w), 0) };
    }
    case 'add': {
      const w = typeof n.params.width === 'number' ? n.params.width : null;
      return { in: w, out: w };
    }
    case 'conv2d': {
      const shape = inShape(n.params);
      const out = conv2dOutShape(n.params);
      return {
        in: shape ? shape[0] * shape[1] * shape[2] : null,
        out: out ? out[0] * out[1] * out[2] : null,
      };
    }
    case 'or_pool2d': {
      const shape = inShape(n.params);
      const out = orPool2dOutShape(n.params);
      return {
        in: shape ? shape[0] * shape[1] * shape[2] : null,
        out: out ? out[0] * out[1] * out[2] : null,
      };
    }
    case 'constant': {
      const w = typeof n.params.width === 'number' ? n.params.width : null;
      return { in: null, out: w };
    }
    case 'upsample2d': {
      const shape = inShape(n.params);
      const out = upsample2dOutShape(n.params);
      return {
        in: shape ? shape[0] * shape[1] * shape[2] : null,
        out: out ? out[0] * out[1] * out[2] : null,
      };
    }
    case 'permute': {
      // 要素数不変の軸並べ替え(in==out)
      const w = permuteWidth(n.params);
      return { in: w, out: w };
    }
    default:
      return { in: null, out: null };
  }
}

/** slot別の期待入力幅(concatのみslotで変わる。範囲外slotはnull) */
export function nodeSlotInWidth(n: IRNodeJson, slot: number): number | null {
  if (n.type === 'concat') {
    const widths = n.params.in_widths;
    if (!Array.isArray(widths)) return null;
    const w = widths[slot];
    return typeof w === 'number' ? w : null;
  }
  return nodeWidths(n).in;
}

/** ノード固有paramsの診断(group_sum割り切り/concat・addのparams整合/conv形状) */
function nodeParamDiagnostics(n: IRNodeJson, out: ShapeDiagnostic[]): void {
  const p = n.params as Record<string, unknown>;
  if (n.type === 'group_sum') {
    const inW = p.in_width;
    const k = p.k;
    if (
      typeof inW === 'number' &&
      typeof k === 'number' &&
      k > 0 &&
      inW % k !== 0
    ) {
      out.push({
        level: 'error',
        message: `${n.id}: in_width(${inW}) % k(${k}) != 0`,
        nodeId: n.id,
      });
    }
    return;
  }
  if (n.type === 'concat') {
    const widths = p.in_widths;
    if (
      !Array.isArray(widths) ||
      widths.length < 2 ||
      widths.some((w) => !Number.isInteger(w) || Number(w) <= 0)
    ) {
      out.push({
        level: 'error',
        message: `${n.id}: in_widthsは2要素以上の正整数リストが必要`,
        nodeId: n.id,
      });
    }
    return;
  }
  if (n.type === 'add') {
    if (typeof p.width !== 'number' || p.width <= 0) {
      out.push({
        level: 'error',
        message: `${n.id}: widthは正整数が必要`,
        nodeId: n.id,
      });
    }
    if (p.mode !== 'or' && p.mode !== 'xor' && p.mode !== 'sum') {
      out.push({
        level: 'error',
        message: `${n.id}: 未知のaddモード ${String(p.mode)}(有効: or/xor/sum)`,
        nodeId: n.id,
      });
    }
    return;
  }
  if (n.type === 'conv2d') {
    const shape = conv2dOutShape(p);
    if (!shape) {
      out.push({
        level: 'error',
        message: `${n.id}: conv2dのin_shape/kernel/stride/padding/out_channelsが不正`,
        nodeId: n.id,
      });
    } else if (shape[1] <= 0 || shape[2] <= 0) {
      out.push({
        level: 'error',
        message: `${n.id}: 出力形状が不正(H_out=${shape[1]}, W_out=${shape[2]})`,
        nodeId: n.id,
      });
    }
    return;
  }
  if (n.type === 'or_pool2d') {
    const shape = orPool2dOutShape(p);
    if (!shape) {
      out.push({
        level: 'error',
        message: `${n.id}: or_pool2dのin_shape/kernel_size/strideが不正`,
        nodeId: n.id,
      });
    } else if (shape[1] <= 0 || shape[2] <= 0) {
      out.push({
        level: 'error',
        message: `${n.id}: 出力形状が不正(H_out=${shape[1]}, W_out=${shape[2]})`,
        nodeId: n.id,
      });
    }
    return;
  }
  if (n.type === 'constant') {
    const width = p.width;
    const bits = p.bits;
    if (typeof width !== 'number' || width <= 0 || !Number.isInteger(width)) {
      out.push({
        level: 'error',
        message: `${n.id}: widthは正整数が必要`,
        nodeId: n.id,
      });
    }
    if (
      !Array.isArray(bits) ||
      bits.length !== width ||
      bits.some((b) => Number(b) !== 0 && Number(b) !== 1)
    ) {
      out.push({
        level: 'error',
        message: `${n.id}: bitsはwidth要素の0/1リストが必要`,
        nodeId: n.id,
      });
    }
    return;
  }
  if (n.type === 'upsample2d') {
    const shape = upsample2dOutShape(p);
    if (!shape) {
      out.push({
        level: 'error',
        message: `${n.id}: upsample2dのin_shape/scaleが不正`,
        nodeId: n.id,
      });
    }
    return;
  }
  if (n.type === 'permute') {
    if (permuteWidth(p) === null) {
      out.push({
        level: 'error',
        message: `${n.id}: permuteのin_shapeは正整数リストが必要`,
        nodeId: n.id,
      });
    } else if (!isValidPerm(p)) {
      out.push({
        level: 'error',
        message: `${n.id}: permuteのpermが有効な順列ではない`,
        nodeId: n.id,
      });
    }
  }
}

/** slot整合の診断(重複/欠番/範囲外/arity)。validate.pyの_check_slotsと同一規約 */
function slotDiagnostics(
  n: IRNodeJson,
  inEdges: IREdgeJson[],
  out: ShapeDiagnostic[],
): void {
  const [lo, hi] = NODE_ARITY[n.type] ?? [1, 1];

  if (hi === 0) {
    if (inEdges.length > 0) {
      out.push({
        level: 'error',
        message: `${n.id}: ${n.type}ノードは入力エッジを持てません`,
        nodeId: n.id,
      });
    }
    return;
  }

  if (inEdges.length === 0) {
    out.push({ level: 'error', message: `${n.id}: 入力エッジがない`, nodeId: n.id });
    return;
  }

  const slots = inEdges.map(edgeSlot).sort((a, b) => a - b);
  const seen = new Set<number>();
  for (const s of slots) {
    if (seen.has(s)) {
      out.push({
        level: 'error',
        message: `${n.id}: slot ${s} に複数のエッジが接続されている(重複)`,
        nodeId: n.id,
      });
    }
    seen.add(s);
  }

  if (n.type === 'concat') {
    const widths = n.params.in_widths;
    const nSlots = Array.isArray(widths) ? widths.length : 0;
    for (const s of seen) {
      if (s < 0 || s >= nSlots) {
        out.push({
          level: 'error',
          message: `${n.id}: slot ${s} は範囲外(in_widthsは${nSlots}要素)`,
          nodeId: n.id,
        });
      }
    }
    for (let s = 0; s < nSlots; s++) {
      if (!seen.has(s)) {
        out.push({
          level: 'error',
          message: `${n.id}: slot ${s} が欠番`,
          nodeId: n.id,
        });
      }
    }
    return;
  }

  if (n.type === 'add') {
    if (slots.length < lo) {
      out.push({
        level: 'error',
        message: `${n.id}: addノードは入力エッジが${lo}本以上必要(現在${slots.length}本)`,
        nodeId: n.id,
      });
    }
    // slotは0..m-1の連番であること
    for (let i = 0; i < slots.length; i++) {
      if (slots[i] !== i) {
        out.push({
          level: 'error',
          message: `${n.id}: slot ${i} が欠番`,
          nodeId: n.id,
        });
        break;
      }
    }
    return;
  }

  // 単一入力ノード: slot 0 のみ1本
  if (hi !== null && inEdges.length > hi) {
    out.push({
      level: 'error',
      message: `${n.id}: 入力エッジが複数(${n.type}は単一入力のみ)`,
      nodeId: n.id,
    });
  }
  if (slots[0] !== 0) {
    out.push({
      level: 'error',
      message: `${n.id}: ${n.type}ノードの入力はslot 0のみ(slot ${slots[0]} は不正)`,
      nodeId: n.id,
    });
  }
}

export function inferShapes(
  nodes: IRNodeJson[],
  edges: IREdgeJson[],
  inputs: IRPortJson[],
): ShapeResult {
  const diagnostics: ShapeDiagnostic[] = [];
  const widths: Record<string, NodeWidths> = {};
  const nodeById = new Map(nodes.map((n) => [n.id, n]));
  const inputById = new Map(inputs.map((p) => [p.id, p]));

  for (const n of nodes) {
    widths[n.id] = nodeWidths(n);
  }

  const srcWidth = (id: string): number | null => {
    const port = inputById.get(id);
    if (port) return port.width;
    return widths[id]?.out ?? null;
  };

  // エッジ幅整合(slot別。出力ポート行きはノード側の導出幅を使うため対象外)
  for (const e of edges) {
    const dstNode = nodeById.get(e.dst);
    if (!dstNode) continue;
    const slot = edgeSlot(e);
    const wSrc = srcWidth(e.src);
    const wDst = nodeSlotInWidth(dstNode, slot);
    if (wSrc !== null && wDst !== null && wSrc !== wDst) {
      const slotLabel = slot > 0 || dstNode.type === 'concat' || dstNode.type === 'add'
        ? `.slot${slot}`
        : '';
      diagnostics.push({
        level: 'error',
        message: `幅不整合: ${e.src}(out=${wSrc}) → ${e.dst}${slotLabel}(in=${wDst})`,
        nodeId: e.dst,
        edgeId: `${e.src}->${e.dst}`,
      });
    }
  }

  // ノード固有params + slot整合(arity/重複/欠番/範囲外)
  const inEdgesOf = new Map<string, IREdgeJson[]>();
  for (const e of edges) {
    if (nodeById.has(e.dst)) {
      const list = inEdgesOf.get(e.dst) ?? [];
      list.push(e);
      inEdgesOf.set(e.dst, list);
    }
  }

  for (const n of nodes) {
    slotDiagnostics(n, inEdgesOf.get(n.id) ?? [], diagnostics);
    nodeParamDiagnostics(n, diagnostics);
  }

  // 出力未接続は警告(訓練は可能だがexport結果が変わるため)。fanoutは合法
  const usedAsSrc = new Set(edges.map((e) => e.src));
  for (const n of nodes) {
    if (!usedAsSrc.has(n.id)) {
      diagnostics.push({
        level: 'warning',
        message: `${n.id}: 出力が未使用`,
        nodeId: n.id,
      });
    }
  }

  return { widths, diagnostics };
}

export function hasShapeErrors(diagnostics: ShapeDiagnostic[]): boolean {
  return diagnostics.some((d) => d.level === 'error');
}
