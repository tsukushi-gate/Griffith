// EVALUATIONタブ本体。NNCのEVALUATION画面に相当。
// 現IR or 学習run × dataset を一括評価し、サンプル毎結果一覧(ResultTable)と
// 混同行列(ConfusionMatrix)を切替表示する。誤答行クリックで反例トレース
// (playgroundStore注入+EDIT/Probe遷移=)。CSVエクスポート。
// AppShellのプレースホルダ差し替えはメインが配線する(named export: EvaluationTab)。

import { useEffect, useMemo, useState } from 'react';
import { toast } from 'sonner';
import { getDatasets, getRuns, type DatasetInfo, type RunSummary } from '../api/rest';
import { DrawDemoPanel } from '../demo/DrawDemoPanel';
import { evaluateAll, exportCsv, type EvalItem, type EvaluateResponse } from './api';
import { buildConfusion } from './confusionMetrics';
import { ConfusionMatrix, type ConfusionCellFilter } from './ConfusionMatrix';
import { injectCounterexample } from './counterexampleInject';
import { ResultTable } from './ResultTable';

type SourceKind = 'current' | 'run';
type ViewMode = 'table' | 'confusion';

export function EvaluationTab() {
  const [datasets, setDatasets] = useState<DatasetInfo[]>([]);
  const [runs, setRuns] = useState<RunSummary[]>([]);
  const [sourceKind, setSourceKind] = useState<SourceKind>('current');
  const [runId, setRunId] = useState<string>('');
  const [dataset, setDataset] = useState<string>('');
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [meta, setMeta] = useState<EvaluateResponse | null>(null);
  const [items, setItems] = useState<EvalItem[]>([]);

  // フィルタ・表示状態
  const [errorsOnly, setErrorsOnly] = useState(false);
  const [mismatchOnly, setMismatchOnly] = useState(false);
  const [cellFilter, setCellFilter] = useState<ConfusionCellFilter | null>(null);
  const [view, setView] = useState<ViewMode>('table');
  // 手描き推論デモのアコーディオン開閉
  const [drawDemoOpen, setDrawDemoOpen] = useState(false);

  useEffect(() => {
    getDatasets()
      .then((ds) => {
        setDatasets(ds);
        if (ds.length > 0) setDataset((cur) => cur || ds[0].name);
      })
      .catch(() => setDatasets([]));
    getRuns()
      .then((r) => setRuns(r.runs))
      .catch(() => setRuns([]));
  }, []);

  const runEvaluation = async () => {
    if (!dataset) {
      setError('データセットを選択してください');
      return;
    }
    if (sourceKind === 'run' && !runId) {
      setError('runを選択してください');
      return;
    }
    setRunning(true);
    setError(null);
    setCellFilter(null);
    try {
      const { meta: m, items: rows } = await evaluateAll({
        source: { kind: sourceKind, run_id: sourceKind === 'run' ? runId : null },
        dataset,
        page_size: 500,
      });
      setMeta(m);
      setItems(rows);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      setMeta(null);
      setItems([]);
    } finally {
      setRunning(false);
    }
  };

  const filtered = useMemo(() => {
    let out = items;
    if (errorsOnly) out = out.filter((it) => !it.correct);
    if (mismatchOnly) out = out.filter((it) => it.hard_soft_mismatch);
    if (cellFilter && meta?.kind === 'classification') {
      out = out.filter((it) => it.label === cellFilter.label && it.pred === cellFilter.pred);
    }
    return out;
  }, [items, errorsOnly, mismatchOnly, cellFilter, meta]);

  const confusion = useMemo(() => {
    if (!meta || meta.kind !== 'classification') return null;
    if (meta.confusion) return meta.confusion;
    if (meta.num_classes) return buildConfusion(items, meta.num_classes);
    return null;
  }, [meta, items]);

  const handleRowClick = (it: EvalItem) => {
    injectCounterexample(it);
  };

  const handleExport = async () => {
    try {
      const csv = await exportCsv({
        source: { kind: sourceKind, run_id: sourceKind === 'run' ? runId : null },
        dataset,
      });
      const blob = new Blob([csv], { type: 'text/csv' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = 'evaluation.csv';
      a.click();
      URL.revokeObjectURL(url);
    } catch (e) {
      toast.error(e instanceof Error ? e.message : String(e));
    }
  };

  return (
    <div className="flex h-full w-full flex-col gap-3 overflow-auto p-4" data-testid="evaluation-tab">
      {/* 評価対象の指定 */}
      <div className="flex flex-wrap items-end gap-3">
        <label className="flex flex-col gap-1 text-xs">
          <span className="text-muted-foreground">ソース</span>
          <select
            className="rounded border border-border bg-background px-2 py-1"
            value={sourceKind}
            onChange={(e) => setSourceKind(e.target.value as SourceKind)}
            data-testid="eval-source-select"
          >
            <option value="current">現在のモデル(current)</option>
            <option value="run">学習run</option>
          </select>
        </label>
        {sourceKind === 'run' && (
          <label className="flex flex-col gap-1 text-xs">
            <span className="text-muted-foreground">run</span>
            <select
              className="rounded border border-border bg-background px-2 py-1"
              value={runId}
              onChange={(e) => setRunId(e.target.value)}
              data-testid="eval-run-select"
            >
              <option value="">選択…</option>
              {runs.map((r) => (
                <option key={r.run_id} value={r.run_id}>
                  {r.model_name} / {r.dataset} (acc={r.final_acc ?? '—'})
                </option>
              ))}
            </select>
          </label>
        )}
        <label className="flex flex-col gap-1 text-xs">
          <span className="text-muted-foreground">データセット</span>
          <select
            className="rounded border border-border bg-background px-2 py-1"
            value={dataset}
            onChange={(e) => setDataset(e.target.value)}
            data-testid="eval-dataset-select"
          >
            {datasets.map((d) => (
              <option key={d.name} value={d.name}>
                {d.label}
              </option>
            ))}
          </select>
        </label>
        <button
          type="button"
          className="rounded bg-primary px-3 py-1.5 text-xs font-semibold text-primary-foreground disabled:opacity-50"
          onClick={runEvaluation}
          disabled={running}
          data-testid="eval-run-button"
        >
          {running ? '評価中…' : '評価を実行'}
        </button>
      </div>

      {error && (
        <div className="rounded border border-red-500/40 bg-red-500/10 px-3 py-2 text-xs text-red-600" data-testid="eval-error">
          {error}
        </div>
      )}

      {meta && (
        <>
          {/* サマリ + 表示切替 + フィルタ */}
          <div className="flex flex-wrap items-center gap-3 text-xs" data-testid="eval-summary">
            <span>
              全{meta.num_samples}件
              {meta.accuracy != null && (
                <>
                  {' '}/ acc <b>{(meta.accuracy * 100).toFixed(1)}%</b>
                </>
              )}
              {meta.num_errors != null && <> / 誤答 {meta.num_errors}</>}
              {meta.num_mismatch != null && <> / hard-soft不一致 {meta.num_mismatch}</>}
            </span>
            {meta.on_demand && (
              <span className="rounded bg-amber-500/20 px-2 py-0.5 text-amber-700" data-testid="eval-ondemand">
                大規模データセットのため先頭ページのみ表示(逐次評価)
              </span>
            )}
            <label className="flex items-center gap-1">
              <input
                type="checkbox"
                checked={errorsOnly}
                onChange={(e) => setErrorsOnly(e.target.checked)}
                data-testid="filter-errors"
              />
              誤答のみ
            </label>
            <label className="flex items-center gap-1">
              <input
                type="checkbox"
                checked={mismatchOnly}
                onChange={(e) => setMismatchOnly(e.target.checked)}
                data-testid="filter-mismatch"
              />
              hard/soft不一致のみ
            </label>
            {cellFilter && (
              <button
                type="button"
                className="rounded bg-blue-500/15 px-2 py-0.5 text-blue-700"
                onClick={() => setCellFilter(null)}
                data-testid="clear-cell-filter"
              >
                セルフィルタ解除(実際{cellFilter.label}→予測{cellFilter.pred})✕
              </button>
            )}
            {confusion && (
              <button
                type="button"
                className="rounded border border-border px-2 py-0.5 hover:bg-accent"
                onClick={() => setView((v) => (v === 'table' ? 'confusion' : 'table'))}
                data-testid="toggle-view"
              >
                {view === 'table' ? '混同行列' : '一覧'}へ切替
              </button>
            )}
            <button
              type="button"
              className="ml-auto rounded border border-border px-2 py-0.5 hover:bg-accent"
              onClick={handleExport}
              data-testid="export-csv"
            >
              CSV
            </button>
          </div>

          {view === 'confusion' && confusion ? (
            <ConfusionMatrix
              confusion={confusion}
              active={cellFilter}
              onCellClick={(f) => {
                setCellFilter(f);
                setView('table');
              }}
            />
          ) : (
            <ResultTable items={filtered} kind={meta.kind} onRowClick={handleRowClick} />
          )}
        </>
      )}

      {/* 手描きキャンバス推論デモ(アコーディオン) */}
      <div className="mt-2 rounded border border-border" data-testid="draw-demo-section">
        <button
          type="button"
          className="flex w-full items-center justify-between px-3 py-2 text-xs font-semibold hover:bg-accent"
          onClick={() => setDrawDemoOpen((o) => !o)}
          data-testid="draw-demo-toggle"
        >
          <span>手描きデモ(描いて即推論)</span>
          <span className="text-muted-foreground">{drawDemoOpen ? '▲' : '▼'}</span>
        </button>
        {drawDemoOpen && (
          <div className="border-t border-border p-3">
            <DrawDemoPanel />
          </div>
        )}
      </div>
    </div>
  );
}
