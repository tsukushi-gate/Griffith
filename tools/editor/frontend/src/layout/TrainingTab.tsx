// TRAININGタブ: NNC風の2ペイン学習監視画面。
// 左 = Results History(RunsPanel: run一覧+選択runの重ね描き比較)
//   右 = 上:学習設定(TrainSettingsForm常設)+学習曲線(TrainCharts)
//        下:タイムライン(MonitorSlot: pause/スクラブ/逆伝播リプレイ/branch) と コアエンジンログ の切替
// 学習開始/一時停止/停止のTrainControlsはヘッダに常設済み(全タブから可視・Popover非依存)。
// 学習設定フォーム(TrainSettingsForm)はTrainChartsが左側に常設し、Popover埋没を解消する(UX設計§5)。
// 裁定1: タイムライン・逆伝播リプレイも学習監視系としてTRAININGタブへ集約する。
// ペイン分割は既存 ResizablePanelGroup を流用(寸法はautoSaveIdで永続化)。

import {
  ResizableHandle,
  ResizablePanel,
  ResizablePanelGroup,
} from '@/components/ui/resizable';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { NasPanel } from '../nas/NasPanel';
import { RunsPanel } from '../runs/RunsPanel';
import { TrainCharts } from '../training/TrainCharts';
import { TrainLog } from '../training/TrainLog';
import { MonitorSlot } from './MonitorSlot';

export function TrainingTab() {
  return (
    <ResizablePanelGroup
      direction="horizontal"
      autoSaveId="md-editor-training-h"
      className="h-full"
    >
      {/* 左: Results History(run一覧+比較) */}
      <ResizablePanel id="training-runs" order={1} defaultSize={38} minSize={22}>
        <aside className="h-full overflow-hidden border-r bg-[var(--panel-bg)]">
          <RunsPanel />
        </aside>
      </ResizablePanel>
      <ResizableHandle />
      {/* 右: 学習曲線 / タイムライン・ログ */}
      <ResizablePanel id="training-main" order={2} defaultSize={62} minSize={30}>
        <ResizablePanelGroup direction="vertical" autoSaveId="md-editor-training-v">
          <ResizablePanel id="training-charts" order={1} defaultSize={62} minSize={25}>
            {/* 学習設定フォーム+学習曲線(実行コントロールはヘッダ常設) */}
            <TrainCharts />
          </ResizablePanel>
          <ResizableHandle />
          <ResizablePanel id="training-bottom" order={2} defaultSize={38} minSize={12}>
            <Tabs
              defaultValue="timeline"
              className="flex h-full flex-col gap-0 bg-[var(--panel-bg)]"
            >
              <TabsList className="h-8 w-fit shrink-0 rounded-none border-b bg-transparent p-0 px-2">
                <TabsTrigger value="timeline" className="h-7 rounded-sm px-3 text-xs">
                  タイムライン
                </TabsTrigger>
                <TabsTrigger value="log" className="h-7 rounded-sm px-3 text-xs">
                  ログ
                </TabsTrigger>
                <TabsTrigger value="nas" className="h-7 rounded-sm px-3 text-xs">
                  構造探索
                </TabsTrigger>
              </TabsList>
              <TabsContent value="timeline" className="min-h-0 flex-1">
                <MonitorSlot />
              </TabsContent>
              <TabsContent value="log" className="min-h-0 flex-1">
                <TrainLog />
              </TabsContent>
              <TabsContent value="nas" className="min-h-0 flex-1">
                <NasPanel />
              </TabsContent>
            </Tabs>
          </ResizablePanel>
        </ResizablePanelGroup>
      </ResizablePanel>
    </ResizablePanelGroup>
  );
}
