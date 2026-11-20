// DATASETタブ右ペイン: サンプルプレビューグリッド。
// GET .../samples をページング表示し、画像はサムネイル、論理はビット列表で見せる。
// クラス分布ミニ棒グラフ、および入力エンコーディングプレビュー(2値化/thermometer後の
// 0/1グリッド、=論理ゲート独自価値)のトグルを備える。

import { useEffect, useState } from 'react';
import {
  encodingPreview,
  getDatasetSamples,
  type DatasetSamplesResponse,
  type EncodingPreviewResponse,
} from '../api/rest';

interface SamplePreviewGridProps {
  datasetId: string | null;
}

const PAGE_SIZE = 12;
const ENCODE_MODES = ['binarize', 'thermometer'] as const;

/** クラス分布ミニ棒グラフ。 */
function ClassDistribution({ dist }: { dist: Record<string, number> }) {
  const entries = Object.entries(dist).sort((a, b) => (a[0] < b[0] ? -1 : 1));
  const max = Math.max(1, ...entries.map(([, c]) => c));
  return (
    <div className="flex flex-col gap-1" data-testid="class-dist">
      <div className="text-[10px] font-semibold text-muted-foreground">クラス分布</div>
      <div className="flex flex-wrap gap-2">
        {entries.map(([label, count]) => (
          <div key={label} className="flex flex-col items-center gap-0.5">
            <div className="flex h-10 w-4 items-end rounded bg-muted/40">
              <div
                className="w-full rounded bg-primary/70"
                style={{ height: `${(count / max) * 100}%` }}
              />
            </div>
            <span className="text-[9px] text-muted-foreground">{label}</span>
            <span className="text-[9px] text-muted-foreground/70">{count}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

/** 0/1ビット列を小さなセルグリッドで表示。 */
function BitRow({ bits }: { bits: number[] }) {
  return (
    <div className="flex flex-wrap gap-px">
      {bits.map((b, i) => (
        <span
          key={i}
          className={`inline-block size-3 rounded-[2px] text-center text-[8px] leading-3 ${
            b ? 'bg-primary text-primary-foreground' : 'bg-muted text-muted-foreground/50'
          }`}
        >
          {b}
        </span>
      ))}
    </div>
  );
}

export function SamplePreviewGrid({ datasetId }: SamplePreviewGridProps) {
  const [page, setPage] = useState(0);
  const [data, setData] = useState<DatasetSamplesResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  // エンコーディングプレビュー
  const [encodeOn, setEncodeOn] = useState(false);
  const [mode, setMode] = useState<(typeof ENCODE_MODES)[number]>('binarize');
  const [threshold, setThreshold] = useState(0.5);
  const [bits, setBits] = useState(4);
  const [encoded, setEncoded] = useState<EncodingPreviewResponse | null>(null);

  // データセット切替でページを先頭へ(レンダー中の派生状態調整=推奨パターン)
  const [prevDatasetId, setPrevDatasetId] = useState(datasetId);
  if (datasetId !== prevDatasetId) {
    setPrevDatasetId(datasetId);
    setPage(0);
  }

  // サンプル取得(setStateは全てpromiseコールバック内=同期setStateをeffectに置かない)
  useEffect(() => {
    if (!datasetId) return;
    let disposed = false;
    getDatasetSamples(datasetId, { page, pageSize: PAGE_SIZE })
      .then((resp) => {
        if (!disposed) {
          setData(resp);
          setError(null);
        }
      })
      .catch((e) => {
        if (!disposed) setError(String(e instanceof Error ? e.message : e));
      });
    return () => {
      disposed = true;
    };
  }, [datasetId, page]);

  // エンコードプレビュー: 先頭サンプルの入力列を変換して表示
  const encodeValues =
    encodeOn && data && data.samples.length > 0 ? (data.samples[0].inputs ?? []) : [];
  const encodeKey = encodeValues.join(',');
  useEffect(() => {
    if (encodeValues.length === 0) return;
    let disposed = false;
    encodingPreview(encodeValues, { mode, threshold, bits })
      .then((r) => {
        if (!disposed) setEncoded(r);
      })
      .catch(() => undefined);
    return () => {
      disposed = true;
    };
    // encodeKey で入力列の変化を検知(encodeValues配列は毎回新規生成のため)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [encodeKey, mode, threshold, bits]);

  if (!datasetId) {
    return (
      <div className="flex flex-1 items-center justify-center text-sm text-muted-foreground">
        左のカタログからデータセットを選択してください
      </div>
    );
  }

  return (
    <div className="flex flex-1 flex-col gap-3 overflow-y-auto p-3" data-testid="sample-preview-grid">
      {error && <p className="text-xs text-destructive">{error}</p>}
      {data && (
        <>
          <div className="flex items-center justify-between gap-2">
            <div className="text-xs text-muted-foreground">
              {datasetId} · {data.num_data}件 · {data.num_columns}列
              {data.class_distribution_approx && '(分布は近似)'}
            </div>
            <div className="flex items-center gap-2">
              <button
                type="button"
                className="rounded border px-2 py-0.5 text-xs disabled:opacity-40"
                disabled={page <= 0}
                onClick={() => setPage((p) => Math.max(0, p - 1))}
              >
                前へ
              </button>
              <span className="text-xs text-muted-foreground">
                {data.page + 1}/{data.num_pages}
              </span>
              <button
                type="button"
                className="rounded border px-2 py-0.5 text-xs disabled:opacity-40"
                disabled={data.page + 1 >= data.num_pages}
                onClick={() => setPage((p) => p + 1)}
              >
                次へ
              </button>
            </div>
          </div>

          <ClassDistribution dist={data.class_distribution} />

          {/* 入力エンコーディングプレビュー */}
          <div className="flex flex-wrap items-center gap-2 rounded-md border bg-muted/20 px-2 py-1.5">
            <label className="flex items-center gap-1 text-xs">
              <input
                type="checkbox"
                data-testid="encoding-toggle"
                checked={encodeOn}
                onChange={(e) => setEncodeOn(e.target.checked)}
              />
              エンコード後ビット列
            </label>
            {encodeOn && (
              <>
                <select
                  data-testid="encoding-mode"
                  value={mode}
                  onChange={(e) => setMode(e.target.value as (typeof ENCODE_MODES)[number])}
                  className="rounded border bg-background px-1 py-0.5 text-xs"
                >
                  {ENCODE_MODES.map((m) => (
                    <option key={m} value={m}>
                      {m}
                    </option>
                  ))}
                </select>
                {mode === 'binarize' ? (
                  <label className="flex items-center gap-1 text-xs">
                    閾値
                    <input
                      type="range"
                      min={0}
                      max={1}
                      step={0.05}
                      value={threshold}
                      onChange={(e) => setThreshold(Number(e.target.value))}
                    />
                    <span className="w-8 tabular-nums">{threshold.toFixed(2)}</span>
                  </label>
                ) : (
                  <label className="flex items-center gap-1 text-xs">
                    ビット数
                    <input
                      type="number"
                      min={1}
                      max={16}
                      value={bits}
                      onChange={(e) => setBits(Math.max(1, Number(e.target.value)))}
                      className="w-14 rounded border bg-background px-1 py-0.5 text-xs"
                    />
                  </label>
                )}
              </>
            )}
          </div>
          {encodeOn && encoded && (
            <div className="rounded-md border p-2" data-testid="encoding-grid">
              <div className="mb-1 text-[10px] text-muted-foreground">
                先頭サンプルのエンコード結果({encoded.mode}, {encoded.bits_per_value}bit/値)
              </div>
              <div className="flex flex-col gap-0.5">
                {encoded.grid.map((row, i) => (
                  <BitRow key={i} bits={row} />
                ))}
              </div>
            </div>
          )}

          {/* サンプルグリッド */}
          <div className="grid grid-cols-[repeat(auto-fill,minmax(96px,1fr))] gap-2">
            {data.samples.map((s) => (
              <div
                key={s.index}
                data-testid="sample-cell"
                className="flex flex-col gap-1 rounded-md border p-1.5"
              >
                {s.image ? (
                  <img
                    src={s.image}
                    alt={`sample ${s.index}`}
                    className="image-render-pixel w-full rounded bg-muted"
                    style={{ imageRendering: 'pixelated' }}
                  />
                ) : (
                  <div className="flex flex-col gap-1">
                    {s.inputs && <BitRow bits={s.inputs} />}
                    {s.outputs && (
                      <div className="flex items-center gap-1">
                        <span className="text-[9px] text-muted-foreground">→</span>
                        <BitRow bits={s.outputs} />
                      </div>
                    )}
                  </div>
                )}
                <div className="flex items-center justify-between text-[9px] text-muted-foreground">
                  <span>#{s.index}</span>
                  {s.label != null && <span>y={s.label}</span>}
                </div>
              </div>
            ))}
          </div>
        </>
      )}
    </div>
  );
}
