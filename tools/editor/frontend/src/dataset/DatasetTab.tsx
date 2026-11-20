// DATASETタブ本体。NNCのDATASET画面に相当。
// 左=カタログ(組込+取込)、右=取込パネル+サンプルプレビューグリッド。
// AppShellのプレースホルダ差し替えはメインが配線する(named export: DatasetTab)。

import { useState } from 'react';
import type { DatasetInfo } from '../api/rest';
import { invalidateDatasetsCache } from '../training/useResolvedTrainConfig';
import { DatasetCatalog } from './DatasetCatalog';
import { DatasetImportPanel } from './DatasetImportPanel';
import { SamplePreviewGrid } from './SamplePreviewGrid';

export function DatasetTab() {
  const [selected, setSelected] = useState<string | null>(null);
  // 取込・設定更新でカタログを再取得させるためのキー
  const [refreshKey, setRefreshKey] = useState(0);

  const handleImported = (ds: DatasetInfo) => {
    setSelected(ds.name);
    setRefreshKey((k) => k + 1);
    // 学習フォームのカタログキャッシュを破棄し、取込データを学習セレクタへ露出
    invalidateDatasetsCache();
  };

  return (
    <div className="flex h-full w-full" data-testid="dataset-tab">
      <DatasetCatalog selected={selected} onSelect={setSelected} refreshKey={refreshKey} />
      <div className="flex min-w-0 flex-1 flex-col">
        <DatasetImportPanel selected={selected} onImported={handleImported} />
        <SamplePreviewGrid datasetId={selected} />
      </div>
    </div>
  );
}
