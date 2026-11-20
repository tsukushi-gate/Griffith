// 下部ドック: トップタブ(N1)ごとに表示タブ集合を切り替える。
// EDIT: [実行(プレイグラウンド)] [コード] [出荷]。TRAINING配下の学習/タイムライン/Run一覧は
// 専用の2ペインUI(TrainingTab)へ移設済み。タブ集合の真実源は uiStore の TABS_BY_TOP。

import type { ComponentType } from 'react';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { CodePanel } from '../editor/CodePanel';
import { PlaygroundPanel } from '../playground/PlaygroundPanel';
import { RunsPanel } from '../runs/RunsPanel';
import { ShipPanel } from '../ship/ShipPanel';
import { TABS_BY_TOP, useUIStore, type BottomTab } from '../store/uiStore';
import { TrainCharts } from '../training/TrainCharts';
import { MonitorSlot } from './MonitorSlot';

// 下部タブの表示ラベルとパネル実体(単一情報源)。
const BOTTOM_TAB_META: Record<BottomTab, { label: string; Panel: ComponentType }> = {
  train: { label: '学習', Panel: TrainCharts },
  timeline: { label: 'タイムライン', Panel: MonitorSlot },
  playground: { label: '実行', Panel: PlaygroundPanel },
  runs: { label: 'Run一覧', Panel: RunsPanel },
  code: { label: 'コード', Panel: CodePanel },
  ship: { label: '出荷', Panel: ShipPanel },
};

export function BottomDock() {
  const bottomTab = useUIStore((s) => s.bottomTab);
  const setBottomTab = useUIStore((s) => s.setBottomTab);

  // BottomDockはEDITタブ専用のドック。学習/タイムライン/Run一覧はTRAININGタブ(TrainingTab)へ
  // 移設済みなので、ここでは常にEDITの下部タブ集合(コード/実行/出荷)のみを露出する。
  const tabs = TABS_BY_TOP.edit;

  // 永続化された bottomTab が現タブ集合に無い場合は先頭へフォールバック。
  const activeTab = tabs.includes(bottomTab) ? bottomTab : tabs[0];

  return (
    <Tabs
      value={activeTab}
      onValueChange={(v) => setBottomTab(v as BottomTab)}
      className="flex h-full flex-col gap-0 bg-[var(--panel-bg)]"
    >
      <TabsList className="h-8 w-fit shrink-0 rounded-none border-b bg-transparent p-0 px-2">
        {tabs.map((t) => (
          <TabsTrigger key={t} value={t} className="h-7 rounded-sm px-3 text-xs">
            {BOTTOM_TAB_META[t].label}
          </TabsTrigger>
        ))}
      </TabsList>
      {tabs.map((t) => {
        const { Panel } = BOTTOM_TAB_META[t];
        return (
          <TabsContent key={t} value={t} className="min-h-0 flex-1">
            <Panel />
          </TabsContent>
        );
      })}
    </Tabs>
  );
}
