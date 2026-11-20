// アプリルート: 訓練WSライフサイクル管理と、消失ノードのビュー復帰のみ担当。
// レイアウトは layout/AppShell に委譲する。

import { useEffect } from 'react';
import { useTrainingSocket } from './api/useTrainingSocket';
import { AppShell } from './layout/AppShell';
import { useGraphStore } from './store/graphStore';
import { useViewStore } from './store/viewStore';

export default function App() {
  useTrainingSocket(); // 訓練WSの接続ライフサイクルをアプリ全体で1つ管理する

  const view = useViewStore((s) => s.view);
  const setView = useViewStore((s) => s.setView);
  const ir = useGraphStore((s) => s.ir);

  // build等でノードidが張り替わり対象が消えたらグラフビューへ戻す
  useEffect(() => {
    const lowering = ir?.metadata?.editor;
    const loweringMap =
      lowering && typeof lowering === 'object' && 'lowering' in lowering
        ? (lowering as { lowering?: unknown }).lowering
        : null;
    const hasLoweringNode =
      loweringMap != null &&
      typeof loweringMap === 'object' &&
      view.kind === 'l0' &&
      view.nodeId in loweringMap;
    if (
      view.kind === 'l0' &&
      !ir?.nodes.some((n) => n.id === view.nodeId) &&
      !hasLoweringNode
    ) {
      setView({ kind: 'graph' });
    }
  }, [view, ir, setView]);

  return <AppShell />;
}
