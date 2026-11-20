// アプリ全体レイアウトシェル(NNC/ComfyUI風):
// ヘッダ + [左パレット | 中央(キャンバス+下部ドック) | 右インスペクタ]。
// 各パネルはリサイズ可能(寸法はautoSaveIdでlocalStorage永続化)、開閉はuiStore管理。

import { lazy, Suspense } from 'react';
import {
  ResizableHandle,
  ResizablePanel,
  ResizablePanelGroup,
} from '@/components/ui/resizable';
import { Toaster } from '@/components/ui/sonner';
import { TooltipProvider } from '@/components/ui/tooltip';
import { CommandPalette } from '../commands/CommandPalette';
import { CanvasEmptyState } from '../editor/CanvasEmptyState';
import { FlowCanvas } from '../editor/FlowCanvas';
import { Palette } from '../editor/Palette';
import { useShortcuts } from '../hooks/useShortcuts';
import { useTopTabHash } from '../hooks/useTopTabHash';
import { InspectorPanel } from '../inspector/InspectorPanel';
import { XorGuidePanel } from '../onboarding/XorGuidePanel';
import { useGraphStore } from '../store/graphStore';
import { useUIStore } from '../store/uiStore';
import { useViewStore } from '../store/viewStore';
import { AppHeader } from './AppHeader';
import { BottomDock } from './BottomDock';
import { DatasetTab } from '../dataset/DatasetTab';
import { ConfigTab } from '../config/ConfigTab';
import { EvaluationTab } from '../evaluation/EvaluationTab';
import { TopTabBar } from './TopTabBar';
import { TrainingTab } from './TrainingTab';

// deck.gl(約1MB)はL0ドリルダウン時のみ必要なので遅延ロードする
const DeckCanvas = lazy(() =>
  import('../l0/DeckCanvas').then((m) => ({ default: m.DeckCanvas })),
);

export function AppShell() {
  useShortcuts(); // グローバルショートカット(⌘K/⌘Z/F/L/1〜5=タブ/Q,W,E=レンズ)
  useTopTabHash(); // URLハッシュ⇔topTab双方向同期

  const view = useViewStore((s) => s.view);
  const hasIr = useGraphStore((s) => s.ir !== null);
  const topTab = useUIStore((s) => s.topTab);
  const leftOpen = useUIStore((s) => s.leftOpen);
  const rightOpen = useUIStore((s) => s.rightOpen);
  const bottomOpen = useUIStore((s) => s.bottomOpen);

  // EDITタブの3ペイン構成(パレット | キャンバス+下部ドック | インスペクタ)。
  const editLayout = (
    <ResizablePanelGroup
      direction="horizontal"
      autoSaveId="md-editor-h"
      className="h-full"
    >
      {leftOpen && (
            <>
              <ResizablePanel id="left" order={1} defaultSize={15} minSize={10} maxSize={28}>
                <aside className="h-full overflow-y-auto border-r bg-[var(--panel-bg)]">
                  <Palette />
                </aside>
              </ResizablePanel>
              <ResizableHandle />
            </>
          )}
          <ResizablePanel id="center" order={2} defaultSize={62} minSize={30}>
            <ResizablePanelGroup direction="vertical" autoSaveId="md-editor-v">
              <ResizablePanel id="canvas" order={1} defaultSize={68} minSize={25}>
                <main className="relative h-full min-h-0">
                  {view.kind === 'graph' ? (
                    <FlowCanvas />
                  ) : (
                    <Suspense
                      fallback={
                        <div className="flex h-full items-center justify-center text-xs text-muted-foreground">
                          L0ビューを読込中…
                        </div>
                      }
                    >
                      <DeckCanvas nodeId={view.nodeId} />
                    </Suspense>
                  )}
                  {!hasIr && view.kind === 'graph' && <CanvasEmptyState />}
                </main>
              </ResizablePanel>
              {bottomOpen && (
                <>
                  <ResizableHandle />
                  <ResizablePanel id="bottom" order={2} defaultSize={32} minSize={12} maxSize={70}>
                    <BottomDock />
                  </ResizablePanel>
                </>
              )}
            </ResizablePanelGroup>
          </ResizablePanel>
      {rightOpen && (
        <>
          <ResizableHandle />
          <ResizablePanel id="right" order={3} defaultSize={23} minSize={15} maxSize={40}>
            <aside className="h-full border-l bg-[var(--panel-bg)]">
              <InspectorPanel />
            </aside>
          </ResizablePanel>
        </>
      )}
    </ResizablePanelGroup>
  );

  return (
    <TooltipProvider>
      <div className="flex h-full flex-col">
        <AppHeader />
        <TopTabBar />
        {/* EDITは常時マウントし表示切替のみ(キャンバスのズーム/選択状態を保持)。
            他タブは非アクティブ時アンマウントし、DOM重複(同一testid/tabpanel)を避ける。
            学習セッション・グラフ等の本質状態はzustandストアに保持されるため往復で失われない。 */}
        <div className="relative min-h-0 flex-1">
          <div className={`absolute inset-0${topTab === 'edit' ? '' : ' hidden'}`}>
            {editLayout}
          </div>
          {topTab === 'training' && (
            <div className="absolute inset-0">
              <TrainingTab />
            </div>
          )}
          {topTab === 'evaluation' && (
            <div className="absolute inset-0">
              <EvaluationTab />
            </div>
          )}
          {topTab === 'dataset' && (
            <div className="absolute inset-0">
              <DatasetTab />
            </div>
          )}
          {topTab === 'config' && (
            <div className="absolute inset-0">
              <ConfigTab />
            </div>
          )}
        </div>
        <CommandPalette />
        <XorGuidePanel />
        <Toaster position="bottom-right" />
      </div>
    </TooltipProvider>
  );
}
