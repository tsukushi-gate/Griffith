// カード縦積み/整列の共通トポロジカル順序(/ 単一情報源)。
// 入力ポート → 中間ノード(トポロジカル順) → 出力ポート の並びを返す。
// カード表示と整列コマンドが同じ規約を使うことで見た目と座標を一致させる。

import type { IRGraphJson } from '../../ir/schema';

/**
 * ノード間エッジ(ポートを除く)でトポロジカルソートし、
 * [入力ポートid..., ノードid(topo)..., 出力ポートid...] の順に並べたid列を返す。
 * 循環がある場合(通常はvalidateで弾かれる)は残りを元の順で末尾に付与する。
 */
export function topologicalOrder(ir: IRGraphJson): string[] {
  const nodeIds = new Set(ir.nodes.map((n) => n.id));
  const indeg = new Map<string, number>();
  const succ = new Map<string, string[]>();
  for (const n of ir.nodes) indeg.set(n.id, 0);

  // ノード↔ノードのエッジのみを対象(ポート行き/ポート発は順序に影響させない)
  for (const e of ir.edges) {
    if (!nodeIds.has(e.src) || !nodeIds.has(e.dst)) continue;
    succ.set(e.src, [...(succ.get(e.src) ?? []), e.dst]);
    indeg.set(e.dst, (indeg.get(e.dst) ?? 0) + 1);
  }

  // 入次数0を元の定義順で初期投入(決定的な並びにする)
  const queue = ir.nodes.filter((n) => (indeg.get(n.id) ?? 0) === 0).map((n) => n.id);
  const ordered: string[] = [];
  const seen = new Set<string>();
  while (queue.length > 0) {
    const id = queue.shift()!;
    if (seen.has(id)) continue;
    seen.add(id);
    ordered.push(id);
    for (const nxt of succ.get(id) ?? []) {
      const d = (indeg.get(nxt) ?? 0) - 1;
      indeg.set(nxt, d);
      if (d <= 0) queue.push(nxt);
    }
  }
  // 循環等で未処理のノードは元順で末尾へ
  for (const n of ir.nodes) if (!seen.has(n.id)) ordered.push(n.id);

  return [...ir.inputs.map((p) => p.id), ...ordered, ...ir.outputs.map((p) => p.id)];
}
