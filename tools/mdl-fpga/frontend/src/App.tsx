// アプリ骨格(エディタ AppShell と同型): TooltipProvider > ヘッダ > リサイズ可能
// パネル(左サイド=入力/実行、右=キャンバス)+ sonner Toaster。
import { useEffect, useState } from 'react';
import { TooltipProvider } from '@/components/ui/tooltip';
import {
  ResizableHandle,
  ResizablePanel,
  ResizablePanelGroup,
} from '@/components/ui/resizable';
import { Toaster } from '@/components/ui/sonner';
import { bootApp } from './boot/bootApp';
import { SchematicCanvas } from './canvas/SchematicCanvas';
import { BoardPanel } from './controls/BoardPanel';
import { InputPanel } from './controls/InputPanel';
import { ProjectPanel } from './controls/ProjectPanel';
import { StepControls } from './controls/StepControls';
import { AppHeader } from './layout/AppHeader';
import { useT } from './i18n/useT';
import { useSimStore } from './store/simStore';
import { useViewerStore } from './store/viewerStore';

export default function App() {
  const [error, setError] = useState<string | null>(null);
  const ready = useSimStore((s) => s.meta !== null);
  const loaded = useViewerStore((s) => s.loaded);

  useEffect(() => {
    bootApp().catch((e) => setError(String(e)));
  }, []);

  if (error) {
    return (
      <div className="boot-error">
        起動失敗: {error}
        <br />
        バックエンド(mdl-fpga view)が起動しているか確認してください。
      </div>
    );
  }
  if (!ready || !loaded) {
    return <div className="boot-loading">シミュレーションに接続中...</div>;
  }
  return (
    <TooltipProvider>
      <div className="app">
        <AppHeader />
        <div className="relative min-h-0 flex-1">
          <MainView />
        </div>
        <Toaster position="bottom-right" />
      </div>
    </TooltipProvider>
  );
}

function MainView() {
  const t = useT();
  const allowSetX = useSimStore((s) => s.meta?.allow_set_x !== false);
  return (
    <ResizablePanelGroup direction="horizontal" autoSaveId="mdl-fpga-h">
      <ResizablePanel defaultSize={18} minSize={15} maxSize={32}>
        <aside className="sidebar-pane border-r">
          <ProjectPanel />
          <BoardPanel />
          {allowSetX ? (
            <InputPanel />
          ) : (
            <div className="rounded-lg border bg-[var(--node-bg)] p-3">
              <div className="mb-2 text-xs text-muted-foreground">
                {t('input.title')}
              </div>
              <div className="text-xs text-muted-foreground">
                {t('input.rom')}
              </div>
            </div>
          )}
          <StepControls />
        </aside>
      </ResizablePanel>
      <ResizableHandle />
      <ResizablePanel defaultSize={82} minSize={40}>
        <main className="canvas-wrap">
          <SchematicCanvas />
        </main>
      </ResizablePanel>
    </ResizablePanelGroup>
  );
}
