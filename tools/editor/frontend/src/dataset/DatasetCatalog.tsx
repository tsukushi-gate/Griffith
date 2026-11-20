// DATASETタブ左ペイン: データセットカタログ。
// 組込(mnist/logic:*/llvit:*)+取込(imported:)を GET /api/datasets 合成で一覧し、
// 選択を親へ通知する。取込データセットには「取込」バッジを付す。

import { useEffect, useState } from 'react';
import { getDatasets, type DatasetInfo } from '../api/rest';

interface DatasetCatalogProps {
  selected: string | null;
  onSelect: (id: string) => void;
  /** 取込直後などにカタログを再取得させるためのキー。 */
  refreshKey?: number;
}

export function DatasetCatalog({ selected, onSelect, refreshKey }: DatasetCatalogProps) {
  const [datasets, setDatasets] = useState<DatasetInfo[]>([]);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let disposed = false;
    getDatasets()
      .then((items) => {
        if (!disposed) {
          setDatasets(items);
          setError(null);
        }
      })
      .catch((e) => {
        if (!disposed) setError(String(e instanceof Error ? e.message : e));
      });
    return () => {
      disposed = true;
    };
  }, [refreshKey]);

  return (
    <div
      className="flex w-64 shrink-0 flex-col gap-1 overflow-y-auto border-r p-2"
      data-testid="dataset-catalog"
    >
      <div className="px-1 py-1 text-[11px] font-semibold text-muted-foreground">
        データセット
      </div>
      {error && <p className="px-1 text-xs text-destructive">{error}</p>}
      {datasets.map((d) => (
        <button
          key={d.name}
          type="button"
          data-testid={`dataset-cat-${d.name}`}
          data-active={selected === d.name}
          onClick={() => onSelect(d.name)}
          className={`flex flex-col items-start gap-0.5 rounded-md border px-2 py-1.5 text-left text-xs transition-colors ${
            selected === d.name
              ? 'border-primary bg-primary/10'
              : 'border-transparent hover:bg-muted/50'
          }`}
        >
          <span className="flex w-full items-center justify-between gap-1">
            <span className="truncate font-medium">{d.label}</span>
            {d.imported && (
              <span className="shrink-0 rounded-full border px-1.5 text-[9px] text-muted-foreground">
                取込
              </span>
            )}
          </span>
          <span className="text-[10px] text-muted-foreground">
            {d.kind} · {d.in_width}→{d.out_width}
          </span>
        </button>
      ))}
    </div>
  );
}
