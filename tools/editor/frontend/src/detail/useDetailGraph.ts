// 詳細ゲートビューの合成フック。
// グラフ構造(nodes/edges/ir)か展開状態が変わった時のみ再合成する。
// 訓練中の5Hz再着色はGateNodeが自身でsummariesを購読するため、ここは再実行されない。
// 層別キャッシュにより、ドラッグ/選択等の非構造変化の再合成では
// 子ゲートノード/詳細エッジのオブジェクトを再利用する。

import { useMemo, useState } from 'react';
import type { EditorNode } from '../ir/irToGraph';
import { applyGrouping, readGroups } from '../editor/grouping';
import { useGraphStore } from '../store/graphStore';
import { useUIStore } from '../store/uiStore';
import {
  composeDetailGraph,
  createDetailComposeCache,
  type DetailGraph,
} from './irToDetailGraph';

export function useDetailGraph(autoExpand?: Set<string>): DetailGraph {
  const nodes = useGraphStore((s) => s.nodes);
  const edges = useGraphStore((s) => s.edges);
  const ir = useGraphStore((s) => s.ir);
  const expandedLayers = useUIStore((s) => s.expandedLayers);
  // layerIdごとの再利用キャッシュ(マウント単位で保持)
  const [cache] = useState(createDetailComposeCache);

  return useMemo(() => {
    // サブサーキット化: 詳細合成の前にグループ折りたたみを適用する。
    // 折りたたみメンバーは除去済みなので詳細展開の対象にならず、合成グループ
    // ノードはcomposeDetailGraphを素通りする(表示専用、IRは不変)。
    const groups = readGroups(ir);
    const grouped = applyGrouping(nodes, edges, groups);
    return composeDetailGraph(
      grouped.nodes as EditorNode[],
      grouped.edges,
      ir,
      expandedLayers,
      cache,
      autoExpand,
    );
  }, [nodes, edges, ir, expandedLayers, cache, autoExpand]);
}
