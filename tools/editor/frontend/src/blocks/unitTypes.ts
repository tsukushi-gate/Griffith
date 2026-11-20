// Unit(引数付き再利用サブネット)のデータモデルと展開ロジック。
// 設計の正: docs/architecture/editor-unit-design.md。要点:
//   - Unit定義は metadata.editor.units[]、インスタンス台帳は metadata.editor.unitInstances[]
//   - IRへは常に展開済みで保存(nodes/edges には既存ノード型しか現れない)
//   - 各展開ノードに meta.editor.unit = { unitId, instanceId } の由来タグを付与
//   - 引数は int / enum のみ。テンプレート内の params の {$arg:'name'} を代入で解決
//   - 擬似IOノード(unit-in/unit-out)は展開時に消え、外部との接続は空きハンドルになる
//   - ネストUnit(Unit内Unit)は禁止(検出してエラー)

import type { IREdgeJson, IRNodeJson } from '../ir/schema';

/** 引数の型(まずこの2種のみ。P2でboolean等)。 */
export type UnitArgType = 'int' | 'enum';

/** パラメータ宣言。 */
export interface UnitArg {
  name: string;
  type: UnitArgType;
  default: number | string;
  min?: number;
  max?: number;
  options?: string[]; // enum時の選択肢
}

/** 擬似IOノードの型(展開時に消える表示専用ノード)。 */
export const UNIT_IN_TYPE = 'unit-in';
export const UNIT_OUT_TYPE = 'unit-out';

/** テンプレートノード。params内に {$arg:'name'} 参照を含みうる(IRNodeJsonを緩めた型)。 */
export interface UnitTemplateNode {
  id: string; // Unitローカルid(展開時に採番し直す)
  type: string; // 既存IRNodeType または 'unit-in'/'unit-out'
  params: Record<string, unknown>;
  train_info?: Record<string, unknown>;
  meta?: Record<string, unknown>;
}

/** Unit定義: 引数付きサブグラフのテンプレート。 */
export interface EditorUnit {
  id: string; // 'unit:<slug>' 一意
  name: string; // 表示名
  description?: string;
  args: UnitArg[];
  template: {
    nodes: UnitTemplateNode[];
    edges: IREdgeJson[];
  };
}

/** インスタンス台帳: どのUnitをどの引数で展開したか。 */
export interface UnitInstance {
  instanceId: string; // 'uinst:<n>' 一意
  unitId: string;
  argValues: Record<string, number | string>;
  expandedNodeIds: string[]; // 展開で生成したノードid列
  edited: boolean; // 展開後に手編集されたら true(再展開時に警告)
  position?: { x: number; y: number }; // 展開の基準座標(再展開で同位置に戻す)
}

/** 展開ノードに付く由来タグ。 */
export interface UnitOriginTag {
  unitId: string;
  instanceId: string;
}

// --- 引数参照({$arg:'name'})の解決 ---

interface ArgRef {
  $arg: string;
}

function isArgRef(v: unknown): v is ArgRef {
  return (
    typeof v === 'object' &&
    v !== null &&
    typeof (v as Record<string, unknown>).$arg === 'string'
  );
}

/** params(またはネスト値)中の {$arg:'name'} を argValues で再帰的に置換する純関数。 */
export function resolveArgs(value: unknown, argValues: Record<string, number | string>): unknown {
  if (isArgRef(value)) {
    const resolved = argValues[value.$arg];
    if (resolved === undefined) {
      throw new Error(`Unit引数が未指定です: ${value.$arg}`);
    }
    return resolved;
  }
  if (Array.isArray(value)) {
    return value.map((v) => resolveArgs(v, argValues));
  }
  if (typeof value === 'object' && value !== null) {
    const out: Record<string, unknown> = {};
    for (const [k, v] of Object.entries(value)) out[k] = resolveArgs(v, argValues);
    return out;
  }
  return value;
}

/** 引数の既定値でargValuesを補完する(未指定引数を既定で埋める)。 */
export function withArgDefaults(
  unit: EditorUnit,
  argValues: Record<string, number | string>,
): Record<string, number | string> {
  const out: Record<string, number | string> = {};
  for (const arg of unit.args) {
    out[arg.name] = argValues[arg.name] ?? arg.default;
  }
  return out;
}

// --- 定義バリデーション ---

/** 擬似IOノードか。 */
export function isPseudoIo(node: { type: string }): boolean {
  return node.type === UNIT_IN_TYPE || node.type === UNIT_OUT_TYPE;
}

/**
 * Unit定義の妥当性検証。エラー文字列の配列を返す(空=妥当)。
 *   - 少なくとも1つの実ノードを持つ
 *   - ネストUnit禁止: 実ノードに meta.editor.unit タグがあってはならない
 *   - 引数名は重複しない、enumはoptions必須
 *   - テンプレートエッジの端点がテンプレート内idを指す
 */
export function validateUnit(unit: EditorUnit): string[] {
  const errors: string[] = [];
  const realNodes = unit.template.nodes.filter((n) => !isPseudoIo(n));
  if (realNodes.length === 0) {
    errors.push('Unitには少なくとも1つのノードが必要です');
  }
  // ネスト禁止
  for (const n of realNodes) {
    const editor = (n.meta?.editor ?? {}) as Record<string, unknown>;
    if (editor.unit) {
      errors.push(`ネストしたUnitは使用できません(ノード ${n.id})`);
    }
  }
  // 引数
  const seen = new Set<string>();
  for (const arg of unit.args) {
    if (!arg.name.trim()) errors.push('引数名が空です');
    if (seen.has(arg.name)) errors.push(`引数名が重複しています: ${arg.name}`);
    seen.add(arg.name);
    if (arg.type === 'enum' && (!arg.options || arg.options.length === 0)) {
      errors.push(`enum引数 ${arg.name} には options が必要です`);
    }
  }
  // エッジ端点
  const ids = new Set(unit.template.nodes.map((n) => n.id));
  for (const e of unit.template.edges) {
    if (!ids.has(e.src)) errors.push(`エッジ始点が未定義です: ${e.src}`);
    if (!ids.has(e.dst)) errors.push(`エッジ終点が未定義です: ${e.dst}`);
  }
  return errors;
}

// --- 展開(インスタンス化) ---

/** 展開ノードid採番: instanceIdベースの決定的採番(再展開でも同一idになる)。 */
export function unitNodeId(instanceId: string, localId: string): string {
  const inst = instanceId.replace(/[^a-zA-Z0-9]/g, '');
  const loc = localId.replace(/[^a-zA-Z0-9]/g, '');
  return `${inst}__${loc}`;
}

export interface ExpandResult {
  nodes: IRNodeJson[];
  edges: IREdgeJson[];
  expandedNodeIds: string[];
  /** 外部入力を接続する候補ノードid(unit-in から繋がる実ノード) */
  entryIds: string[];
  /** 外部出力へ接続する候補ノードid(unit-out へ繋がる実ノード) */
  exitIds: string[];
}

/**
 * Unitを引数で展開して {nodes, edges} を返す純関数(addSubgraph経路へ渡す)。
 *   - params中の {$arg} を解決
 *   - 擬似IOノードは落とし、それに触れるエッジは境界(entry/exit)として記録
 *   - 実ノードidは instanceId ベースで決定的採番(再展開id安定)
 *   - 各ノードに meta.editor.unit = { unitId, instanceId } を付与
 */
export function expandUnit(
  unit: EditorUnit,
  argValuesRaw: Record<string, number | string>,
  instanceId: string,
): ExpandResult {
  const argValues = withArgDefaults(unit, argValuesRaw);
  const realNodes = unit.template.nodes.filter((n) => !isPseudoIo(n));
  const pseudoIn = new Set(
    unit.template.nodes.filter((n) => n.type === UNIT_IN_TYPE).map((n) => n.id),
  );
  const pseudoOut = new Set(
    unit.template.nodes.filter((n) => n.type === UNIT_OUT_TYPE).map((n) => n.id),
  );

  const idMap = new Map<string, string>();
  for (const n of realNodes) idMap.set(n.id, unitNodeId(instanceId, n.id));

  const nodes: IRNodeJson[] = realNodes.map((n) => {
    const params = resolveArgs(n.params, argValues) as Record<string, unknown>;
    const editor = (n.meta?.editor ?? {}) as Record<string, unknown>;
    return {
      id: idMap.get(n.id)!,
      type: n.type as IRNodeJson['type'],
      params,
      train_info: n.train_info ? { ...n.train_info } : {},
      meta: {
        ...(n.meta ?? {}),
        editor: { ...editor, unit: { unitId: unit.id, instanceId } },
      },
    };
  });

  const edges: IREdgeJson[] = [];
  const entryIds = new Set<string>();
  const exitIds = new Set<string>();
  for (const e of unit.template.edges) {
    const srcPseudoIn = pseudoIn.has(e.src);
    const dstPseudoOut = pseudoOut.has(e.dst);
    // 擬似IO入力から実ノードへ → その実ノードは外部入力の接続先(entry)
    if (srcPseudoIn && idMap.has(e.dst)) {
      entryIds.add(idMap.get(e.dst)!);
      continue;
    }
    // 実ノードから擬似IO出力へ → その実ノードは外部出力の供給元(exit)
    if (dstPseudoOut && idMap.has(e.src)) {
      exitIds.add(idMap.get(e.src)!);
      continue;
    }
    // 擬似IO同士や未対応は無視
    if (!idMap.has(e.src) || !idMap.has(e.dst)) continue;
    edges.push({
      src: idMap.get(e.src)!,
      dst: idMap.get(e.dst)!,
      ...(e.dst_slot ? { dst_slot: e.dst_slot } : {}),
    });
  }

  return {
    nodes,
    edges,
    expandedNodeIds: nodes.map((n) => n.id),
    entryIds: [...entryIds],
    exitIds: [...exitIds],
  };
}

// --- metadata アクセサ(groups と同じ非破壊metadata方式) ---

import type { IRGraphJson } from '../ir/schema';

function editorNs(ir: IRGraphJson): Record<string, unknown> {
  return (ir.metadata?.editor as Record<string, unknown>) ?? {};
}

/** metadata.editor.units を読み出す(なければ空配列)。 */
export function unitsOf(ir: IRGraphJson): EditorUnit[] {
  return (editorNs(ir).units as EditorUnit[] | undefined) ?? [];
}

/** units を差し替えた新IRを返す(構造は不変)。 */
export function withUnits(ir: IRGraphJson, units: EditorUnit[]): IRGraphJson {
  return {
    ...ir,
    metadata: { ...ir.metadata, editor: { ...editorNs(ir), units } },
  };
}

/** metadata.editor.unitInstances を読み出す(なければ空配列)。 */
export function unitInstancesOf(ir: IRGraphJson): UnitInstance[] {
  return (editorNs(ir).unitInstances as UnitInstance[] | undefined) ?? [];
}

/** unitInstances を差し替えた新IRを返す(構造は不変)。 */
export function withUnitInstances(ir: IRGraphJson, instances: UnitInstance[]): IRGraphJson {
  return {
    ...ir,
    metadata: { ...ir.metadata, editor: { ...editorNs(ir), unitInstances: instances } },
  };
}

/**
 * Unitを既定引数で単体モデルIR化する(ギャラリー用、templateToGraph相当)。
 * x(inWidth) → entry … exit → y(outWidth)。台帳(unitInstances)と定義(units)も同梱する。
 */
export function unitToStandaloneGraph(unit: EditorUnit): IRGraphJson {
  const exp = expandUnit(unit, {}, 'uinst:1');
  const entry = exp.entryIds[0] ?? exp.nodes[0]?.id;
  const exit = exp.exitIds[0] ?? exp.nodes[exp.nodes.length - 1]?.id;
  const entryNode = exp.nodes.find((n) => n.id === entry);
  const exitNode = exp.nodes.find((n) => n.id === exit);
  const inWidth = Number((entryNode?.params.in_width as number) ?? 8);
  const outWidth = Number((exitNode?.params.out_width as number) ?? 8);
  const instance: UnitInstance = {
    instanceId: 'uinst:1',
    unitId: unit.id,
    argValues: withArgDefaults(unit, {}),
    expandedNodeIds: exp.expandedNodeIds,
    edited: false,
    position: { x: 0, y: 0 },
  };
  const graph: IRGraphJson = {
    format: 'magicdelta-ir',
    version: '0.2.0',
    metadata: { name: `unit_${unit.name}` },
    inputs: [{ id: 'x', width: inWidth, shape: null, encoding: null, semantics: null }],
    outputs: [{ id: 'y', width: outWidth, shape: null, encoding: null, semantics: null }],
    nodes: exp.nodes,
    edges: [
      ...(entry ? [{ src: 'x', dst: entry }] : []),
      ...exp.edges,
      ...(exit ? [{ src: exit, dst: 'y' }] : []),
    ],
  };
  return withUnitInstances(withUnits(graph, [unit]), [instance]);
}

/** ノードの由来タグ(meta.editor.unit)を読む。なければnull。 */
export function unitTagOf(node: IRNodeJson): UnitOriginTag | null {
  const editor = (node.meta?.editor ?? {}) as Record<string, unknown>;
  const tag = editor.unit as UnitOriginTag | undefined;
  return tag && tag.instanceId ? tag : null;
}

/**
 * インスタンスの構造シグネチャ(タッチ検出用)。
 * 現存する展開ノードid集合 + そのノードに接続するエッジ集合を安定文字列化する。
 * ノード削除・エッジ増減で値が変わる → 手編集(edited)検出に使う。
 */
export function instanceSignature(ir: IRGraphJson, inst: UnitInstance): string {
  const owned = new Set(inst.expandedNodeIds);
  const nodeIds = ir.nodes.filter((n) => owned.has(n.id)).map((n) => n.id);
  nodeIds.sort();
  const touching = ir.edges
    .filter((e) => owned.has(e.src) || owned.has(e.dst))
    .map((e) => `${e.src}>${e.dst}:${e.dst_slot ?? 0}`);
  touching.sort();
  return JSON.stringify({ nodeIds, touching });
}

/**
 * 構造編集前後のIRを比べ、手編集されたインスタンスに edited=true を立てた台帳を返す。
 * 既に edited のものは維持。シグネチャ不変のものはそのまま。
 */
export function markEditedInstances(prevIr: IRGraphJson, nextIr: IRGraphJson): UnitInstance[] {
  const instances = unitInstancesOf(nextIr);
  if (instances.length === 0) return instances;
  // 直前の台帳に存在したインスタンスのみ評価する(新規追加の展開ノード「出現」を
  // 手編集と誤検知しないため)。
  const prevIds = new Set(unitInstancesOf(prevIr).map((i) => i.instanceId));
  return instances.map((inst) => {
    if (inst.edited || !prevIds.has(inst.instanceId)) return inst;
    const before = instanceSignature(prevIr, inst);
    const after = instanceSignature(nextIr, inst);
    return before === after ? inst : { ...inst, edited: true };
  });
}

/** 次のインスタンスid(`uinst:<n>`)。既存台帳と衝突しない連番。 */
export function nextInstanceId(instances: UnitInstance[]): string {
  let max = 0;
  for (const inst of instances) {
    const m = /^uinst:(\d+)$/.exec(inst.instanceId);
    if (m) max = Math.max(max, Number(m[1]));
  }
  return `uinst:${max + 1}`;
}

/** 次のUnit id(`unit:<n>`)。既存定義と衝突しない連番(名前slugが空/重複の保険)。 */
export function nextUnitId(units: EditorUnit[]): string {
  let max = 0;
  for (const u of units) {
    const m = /^unit:(\d+)$/.exec(u.id);
    if (m) max = Math.max(max, Number(m[1]));
  }
  return `unit:${max + 1}`;
}

/** 展開ノードにドロップ位置オフセットを適用する(テンプレ相対座標 → キャンバス座標)。 */
export function offsetExpandedNodes(
  nodes: IRNodeJson[],
  origin: { x: number; y: number },
): IRNodeJson[] {
  return nodes.map((n) => {
    const editor = (n.meta?.editor ?? {}) as Record<string, unknown>;
    const rel = (editor.position ?? { x: 0, y: 0 }) as { x: number; y: number };
    return {
      ...n,
      meta: {
        ...n.meta,
        editor: {
          ...editor,
          position: { x: Math.round(origin.x + rel.x), y: Math.round(origin.y + rel.y) },
        },
      },
    };
  });
}

export interface ReexpandResult {
  ir: IRGraphJson;
  reexpandedInstanceIds: string[];
  /** 手編集済みでスキップしたインスタンスの警告(表示用) */
  skipped: { instanceId: string; reason: string }[];
  /** 再接続に失敗して落とした外部エッジ(診断用) */
  droppedEdges: IREdgeJson[];
}

/** 何も再展開しなかった空の結果(新規定義保存時など)。 */
export function reexpandInfoEmpty(): ReexpandResult {
  return { ir: null as never, reexpandedInstanceIds: [], skipped: [], droppedEdges: [] };
}

/**
 * 定義変更の伝播: 同一unitIdの全インスタンスを再展開する純関数。
 *   - edited=false のインスタンス: 旧ノード群を削除し同じ位置・同じidで再展開
 *     (instanceIdベースの決定的採番なのでノードidは安定 → 外部接続エッジは自動的に生存)
 *   - edited=true のインスタンス: スキップして警告に載せる
 *   - 再展開後に端点が消えた外部エッジは落として droppedEdges に記録する
 */
export function reexpandUnit(ir: IRGraphJson, unit: EditorUnit): ReexpandResult {
  const instances = unitInstancesOf(ir);
  const reexpandedInstanceIds: string[] = [];
  const skipped: { instanceId: string; reason: string }[] = [];

  let nodes = [...ir.nodes];
  let edges = [...ir.edges];
  const nextInstances: UnitInstance[] = [];

  for (const inst of instances) {
    if (inst.unitId !== unit.id) {
      nextInstances.push(inst);
      continue;
    }
    if (inst.edited) {
      skipped.push({ instanceId: inst.instanceId, reason: '手編集済みのため再展開をスキップ' });
      nextInstances.push(inst);
      continue;
    }
    const oldIds = new Set(inst.expandedNodeIds);
    // 旧ノード群と内部エッジ(両端が旧集合)を除去。外部エッジは残す。
    nodes = nodes.filter((n) => !oldIds.has(n.id));
    edges = edges.filter((e) => !(oldIds.has(e.src) && oldIds.has(e.dst)));

    // 同一instanceIdで再展開(決定的idで同じ位置に戻す)
    const origin = inst.position ?? { x: 0, y: 0 };
    const exp = expandUnit(unit, inst.argValues, inst.instanceId);
    const placed = offsetExpandedNodes(exp.nodes, origin);
    nodes = [...nodes, ...placed];
    edges = [...edges, ...exp.edges];
    reexpandedInstanceIds.push(inst.instanceId);
    nextInstances.push({ ...inst, expandedNodeIds: exp.expandedNodeIds, edited: false });
  }

  // 端点が存在しなくなった外部エッジを掃除する
  const nodeIds = new Set(nodes.map((n) => n.id));
  const portIds = new Set([...ir.inputs, ...ir.outputs].map((p) => p.id));
  const valid = (id: string) => nodeIds.has(id) || portIds.has(id);
  const droppedEdges: IREdgeJson[] = [];
  const keptEdges = edges.filter((e) => {
    const ok = valid(e.src) && valid(e.dst);
    if (!ok) droppedEdges.push(e);
    return ok;
  });

  const nextIr = withUnitInstances({ ...ir, nodes, edges: keptEdges }, nextInstances);
  return { ir: nextIr, reexpandedInstanceIds, skipped, droppedEdges };
}
