// 画面構成: ツールバー / 左=ツリーキャンバス / 右=詳細パネル / 検索パレット(⌘K)。

import { useEffect } from 'react';
import { DetailPanel } from './panel/DetailPanel';
import { SearchPalette } from './search/SearchPalette';
import { applyTheme, TREE, useAppStore } from './store';
import { Toolbar } from './Toolbar';
import { TreeCanvas } from './tree/TreeCanvas';

export default function App() {
  const theme = useAppStore((s) => s.theme);
  useEffect(() => {
    applyTheme(theme);
  }, [theme]);

  return (
    <div className="flex h-full flex-col" data-testid="app-root">
      <Toolbar />
      <div className="flex min-h-0 flex-1">
        <main className="min-w-0 flex-1" data-testid="canvas-area">
          <TreeCanvas tree={TREE} />
        </main>
        <DetailPanel />
      </div>
      <SearchPalette />
    </div>
  );
}
