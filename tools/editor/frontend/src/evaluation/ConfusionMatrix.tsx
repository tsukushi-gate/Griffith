// 混同行列ビュー。canvasヒートマップ+precision/recall/F1行、
// 全体Accuracy/Avg.Precision/Avg.Recall。セルクリックで ResultTable を
// (実際=label, 予測=pred)へフィルタする。
// 規約: confusion[label][pred] = 件数(行=実際, 列=予測)。

import { useEffect, useRef } from 'react';
import { computeConfusionMetrics } from './confusionMetrics';

export interface ConfusionCellFilter {
  label: number;
  pred: number;
}

interface ConfusionMatrixProps {
  confusion: number[][];
  /** セルクリック(label=実際, pred=予測) */
  onCellClick?: (filter: ConfusionCellFilter) => void;
  /** 現在フィルタ中のセル(ハイライト) */
  active?: ConfusionCellFilter | null;
}

const CELL = 34;

export function ConfusionMatrix({ confusion, onCellClick, active }: ConfusionMatrixProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const metrics = computeConfusionMetrics(confusion);
  const n = metrics.n;
  const max = Math.max(1, ...confusion.flat());

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    for (let label = 0; label < n; label++) {
      for (let pred = 0; pred < n; pred++) {
        const c = confusion[label][pred];
        const t = c / max;
        // 対角(正解)は緑系、非対角(誤り)は赤系
        const hue = label === pred ? 150 : 8;
        ctx.fillStyle = `hsl(${hue} 70% ${92 - t * 55}%)`;
        ctx.fillRect(pred * CELL, label * CELL, CELL - 1, CELL - 1);
        if (active && active.label === label && active.pred === pred) {
          ctx.strokeStyle = '#2563eb';
          ctx.lineWidth = 2;
          ctx.strokeRect(pred * CELL + 1, label * CELL + 1, CELL - 3, CELL - 3);
        }
        ctx.fillStyle = t > 0.6 ? '#fff' : '#334155';
        ctx.font = '10px ui-sans-serif, system-ui';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(String(c), pred * CELL + CELL / 2, label * CELL + CELL / 2);
      }
    }
  }, [confusion, n, max, active]);

  const handleClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!onCellClick) return;
    const rect = e.currentTarget.getBoundingClientRect();
    const pred = Math.floor(((e.clientX - rect.left) / rect.width) * n);
    const label = Math.floor(((e.clientY - rect.top) / rect.height) * n);
    if (pred >= 0 && pred < n && label >= 0 && label < n) {
      onCellClick({ label, pred });
    }
  };

  const pct = (v: number) => `${(v * 100).toFixed(1)}%`;

  return (
    <div className="flex flex-col gap-3" data-testid="confusion-matrix">
      <div className="flex gap-4 text-xs">
        <span data-testid="cm-accuracy">
          Accuracy: <b>{pct(metrics.accuracy)}</b>
        </span>
        <span>Avg.Precision: {pct(metrics.avgPrecision)}</span>
        <span>Avg.Recall: {pct(metrics.avgRecall)}</span>
      </div>
      <div className="flex items-start gap-2">
        <div className="flex flex-col items-center">
          <div className="text-[10px] text-muted-foreground">実際 \ 予測</div>
          <canvas
            ref={canvasRef}
            width={n * CELL}
            height={n * CELL}
            onClick={handleClick}
            className="cursor-pointer rounded border border-border"
            data-testid="confusion-canvas"
          />
        </div>
        {/* クラス別 precision/recall/F1 表 */}
        <table className="text-[10px]" data-testid="confusion-metrics-table">
          <thead>
            <tr className="text-muted-foreground">
              <th className="px-1 text-left">class</th>
              <th className="px-1">Prec</th>
              <th className="px-1">Rec</th>
              <th className="px-1">F1</th>
              <th className="px-1">n</th>
            </tr>
          </thead>
          <tbody>
            {Array.from({ length: n }, (_, i) => (
              <tr key={i}>
                <td className="px-1 font-mono">{i}</td>
                <td className="px-1 text-right tabular-nums">{pct(metrics.precision[i])}</td>
                <td className="px-1 text-right tabular-nums">{pct(metrics.recall[i])}</td>
                <td className="px-1 text-right tabular-nums">{pct(metrics.f1[i])}</td>
                <td className="px-1 text-right tabular-nums text-muted-foreground">
                  {metrics.support[i]}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
