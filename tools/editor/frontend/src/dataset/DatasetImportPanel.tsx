// DATASETタブ右上: 取込パネル(CSV / 画像フォルダzip)+
// 選択中の取込データセットのsplit・augmentation設定。
// CSV/zipはファイル選択で即取込し、登録されたデータセットを親へ通知する。

import { useRef, useState } from 'react';
import {
  importCsvDataset,
  importImagesDataset,
  updateImportedDatasetConfig,
  type DatasetInfo,
} from '../api/rest';

interface DatasetImportPanelProps {
  /** 現在選択中のデータセットID(imported: のときのみ設定UIを出す)。 */
  selected: string | null;
  /** 取込成功時: 登録データセットを渡す(親はカタログ再取得+選択する)。 */
  onImported: (ds: DatasetInfo) => void;
}

export function DatasetImportPanel({ selected, onImported }: DatasetImportPanelProps) {
  const csvRef = useRef<HTMLInputElement>(null);
  const zipRef = useRef<HTMLInputElement>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // 画像前処理パラメータ
  const [resize, setResize] = useState(16);
  const [grayscale, setGrayscale] = useState(true);
  const [binarize, setBinarize] = useState(0.5);

  const runImport = async (fn: () => Promise<{ dataset: DatasetInfo }>) => {
    setBusy(true);
    setError(null);
    try {
      const { dataset } = await fn();
      onImported(dataset);
    } catch (e) {
      setError(String(e instanceof Error ? e.message : e));
    } finally {
      setBusy(false);
    }
  };

  const onCsvChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) void runImport(() => importCsvDataset(file));
    e.target.value = '';
  };

  const onZipChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      void runImport(() =>
        importImagesDataset(file, { resize, grayscale, binarize }),
      );
    }
    e.target.value = '';
  };

  return (
    <div className="flex flex-col gap-2 border-b p-3" data-testid="dataset-import-panel">
      <div className="text-[11px] font-semibold text-muted-foreground">データ取込</div>
      {error && <p className="text-xs text-destructive">{error}</p>}
      <div className="flex flex-wrap items-center gap-2">
        <button
          type="button"
          className="rounded border px-2 py-1 text-xs hover:bg-muted/50 disabled:opacity-40"
          disabled={busy}
          onClick={() => csvRef.current?.click()}
        >
          CSV取込
        </button>
        <input
          ref={csvRef}
          type="file"
          accept=".csv,text/csv"
          data-testid="import-csv-input"
          className="hidden"
          onChange={onCsvChange}
        />
        <button
          type="button"
          className="rounded border px-2 py-1 text-xs hover:bg-muted/50 disabled:opacity-40"
          disabled={busy}
          onClick={() => zipRef.current?.click()}
        >
          画像zip取込
        </button>
        <input
          ref={zipRef}
          type="file"
          accept=".zip,application/zip"
          data-testid="import-zip-input"
          className="hidden"
          onChange={onZipChange}
        />
        {busy && <span className="text-xs text-muted-foreground">取込中…</span>}
      </div>

      {/* 画像前処理パラメータ */}
      <div className="flex flex-wrap items-center gap-3 text-xs text-muted-foreground">
        <label className="flex items-center gap-1">
          リサイズ
          <input
            type="number"
            min={4}
            max={64}
            value={resize}
            onChange={(e) => setResize(Math.max(4, Number(e.target.value)))}
            className="w-14 rounded border bg-background px-1 py-0.5"
          />
        </label>
        <label className="flex items-center gap-1">
          <input
            type="checkbox"
            checked={grayscale}
            onChange={(e) => setGrayscale(e.target.checked)}
          />
          グレースケール
        </label>
        <label className="flex items-center gap-1">
          2値化閾値
          <input
            type="range"
            min={0}
            max={1}
            step={0.05}
            value={binarize}
            onChange={(e) => setBinarize(Number(e.target.value))}
          />
          <span className="w-8 tabular-nums">{binarize.toFixed(2)}</span>
        </label>
      </div>

      {selected?.startsWith('imported:') && (
        // key=datasetId でデータセット切替時にフォーム状態を初期化(remount)
        <ImportedDatasetConfig key={selected} datasetId={selected} onUpdated={onImported} />
      )}
    </div>
  );
}

/** 取込データセットのsplit+augmentation設定。*/
function ImportedDatasetConfig({
  datasetId,
  onUpdated,
}: {
  datasetId: string;
  onUpdated: (ds: DatasetInfo) => void;
}) {
  const [trainRatio, setTrainRatio] = useState(0.8);
  const [seed, setSeed] = useState(0);
  const [flip, setFlip] = useState(false);
  const [shift, setShift] = useState(0);
  const [bitFlipNoise, setBitFlipNoise] = useState(0);
  const [saving, setSaving] = useState(false);

  const save = async () => {
    setSaving(true);
    try {
      const val = Math.max(0, 1 - trainRatio);
      const updated = await updateImportedDatasetConfig(datasetId, {
        split: { train: trainRatio, val, test: 0 },
        seed,
        augmentation: { flip, shift, bit_flip_noise: bitFlipNoise },
      });
      onUpdated(updated);
    } finally {
      setSaving(false);
    }
  };

  return (
    <div className="mt-1 flex flex-col gap-2 rounded-md border bg-muted/20 p-2" data-testid="dataset-config">
      <div className="text-[10px] font-semibold text-muted-foreground">
        split / augmentation
      </div>
      <div className="flex flex-wrap items-center gap-3 text-xs text-muted-foreground">
        <label className="flex items-center gap-1">
          train比
          <input
            type="range"
            min={0.1}
            max={1}
            step={0.05}
            value={trainRatio}
            onChange={(e) => setTrainRatio(Number(e.target.value))}
          />
          <span className="w-8 tabular-nums">{trainRatio.toFixed(2)}</span>
        </label>
        <label className="flex items-center gap-1">
          seed
          <input
            type="number"
            value={seed}
            onChange={(e) => setSeed(Number(e.target.value))}
            className="w-14 rounded border bg-background px-1 py-0.5"
          />
        </label>
      </div>
      <div className="flex flex-wrap items-center gap-3 text-xs text-muted-foreground">
        <label className="flex items-center gap-1">
          <input type="checkbox" checked={flip} onChange={(e) => setFlip(e.target.checked)} />
          flip
        </label>
        <label className="flex items-center gap-1">
          shift
          <input
            type="number"
            min={0}
            max={8}
            value={shift}
            onChange={(e) => setShift(Math.max(0, Number(e.target.value)))}
            className="w-12 rounded border bg-background px-1 py-0.5"
          />
        </label>
        <label className="flex items-center gap-1">
          bit反転noise
          <input
            type="range"
            min={0}
            max={0.5}
            step={0.01}
            value={bitFlipNoise}
            onChange={(e) => setBitFlipNoise(Number(e.target.value))}
          />
          <span className="w-8 tabular-nums">{bitFlipNoise.toFixed(2)}</span>
        </label>
        <button
          type="button"
          className="rounded border px-2 py-0.5 text-xs hover:bg-muted/50 disabled:opacity-40"
          disabled={saving}
          onClick={() => void save()}
          data-testid="dataset-config-save"
        >
          適用
        </button>
      </div>
    </div>
  );
}
