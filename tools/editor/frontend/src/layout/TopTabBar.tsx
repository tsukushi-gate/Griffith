// トップタブ帯: NNC同型の5タブ EDIT / TRAINING / EVALUATION / DATASET / CONFIG。
// ヘッダ直下に常設し、アクティブタブを切り替える(状態は保持=表示切替)。

import { Tabs, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { useT } from '../i18n';
import type { MessageKey } from '../i18n';
import { TOP_TAB_ORDER, useUIStore, type TopTab } from '../store/uiStore';

// タブ→辞書キー(ラベル/ツールチップ)の対応(文言はi18n辞書が単一情報源)。
const TOP_TAB_KEYS: Record<TopTab, { label: MessageKey; hint: MessageKey }> = {
  edit: { label: 'toptab.edit.label', hint: 'toptab.edit.hint' },
  training: { label: 'toptab.training.label', hint: 'toptab.training.hint' },
  evaluation: { label: 'toptab.evaluation.label', hint: 'toptab.evaluation.hint' },
  dataset: { label: 'toptab.dataset.label', hint: 'toptab.dataset.hint' },
  config: { label: 'toptab.config.label', hint: 'toptab.config.hint' },
};

export function TopTabBar() {
  const topTab = useUIStore((s) => s.topTab);
  const setTopTab = useUIStore((s) => s.setTopTab);
  const t = useT();

  return (
    <Tabs
      value={topTab}
      onValueChange={(v) => setTopTab(v as TopTab)}
      className="shrink-0 gap-0"
    >
      <TabsList className="h-9 w-full justify-start rounded-none border-b bg-[var(--panel-bg)] p-0 px-2">
        {TOP_TAB_ORDER.map((tab) => {
          const keys = TOP_TAB_KEYS[tab];
          return (
            <TabsTrigger
              key={tab}
              value={tab}
              data-testid={`toptab-${tab}`}
              title={t(keys.hint)}
              className="h-8 rounded-none border-b-2 border-transparent px-3 text-xs font-semibold tracking-wide data-[state=active]:border-primary data-[state=active]:text-primary"
            >
              {t(keys.label)}
            </TabsTrigger>
          );
        })}
      </TabsList>
    </Tabs>
  );
}
