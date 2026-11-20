// Trade-offグラフ: 横軸=総LUT数(回路面積)、縦軸=hard acc の散布図。
// パレートフロントを線で結んで強調し、partial:true は中空マーカー(裁定6=判定除外)。
// 点クリックでそのtrialのIRをEDITに開く(onSelectTrial)。現モデル位置マーカーも重畳可。
// uPlotのfacet(mode:2)より堅牢・テスト容易なSVG実装(色はCSS変数でテーマ追従)。

import { useMemo } from 'react';
import { paretoFront, paretoLine } from './pareto';
import type { NasTrialRow } from './types';

const W = 320;
const H = 220;
const PAD = { top: 12, right: 12, bottom: 28, left: 34 };

interface Props {
  trials: NasTrialRow[];
  onSelectTrial?: (trialId: number) => void;
  /** 現在編集中モデルの位置。acc未知はnull。*/
  currentModel?: { acc: number | null; total_luts: number } | null;
}

export function TradeoffChart({ trials, onSelectTrial, currentModel }: Props) {
  const done = trials.filter((t) => t.state === 'done');
  const frontIds = useMemo(() => new Set(paretoFront(trials)), [trials]);
  const line = useMemo(() => paretoLine(trials), [trials]);

  const lutValues = done.map((t) => t.total_luts);
  if (currentModel) lutValues.push(currentModel.total_luts);
  const maxLut = Math.max(1, ...lutValues);
  // accは[0,1]固定軸(hard accの定義域)
  const plotW = W - PAD.left - PAD.right;
  const plotH = H - PAD.top - PAD.bottom;

  const sx = (lut: number) => PAD.left + (maxLut > 0 ? (lut / maxLut) * plotW : 0);
  const sy = (acc: number) => PAD.top + (1 - acc) * plotH;

  if (done.length === 0) {
    return (
      <div className="hint-text" data-testid="tradeoff-empty">
        trial結果がありません(探索を開始してください)
      </div>
    );
  }

  const linePath = line.map((t) => `${sx(t.total_luts)},${sy(t.acc)}`).join(' ');

  return (
    <svg
      viewBox={`0 0 ${W} ${H}`}
      className="w-full"
      style={{ maxHeight: H }}
      data-testid="tradeoff-chart"
      role="img"
      aria-label="精度×LUT数トレードオフ散布図"
    >
      {/* 軸 */}
      <line
        x1={PAD.left} y1={PAD.top} x2={PAD.left} y2={PAD.top + plotH}
        stroke="var(--chart-axis, #888)" strokeWidth={1}
      />
      <line
        x1={PAD.left} y1={PAD.top + plotH} x2={PAD.left + plotW} y2={PAD.top + plotH}
        stroke="var(--chart-axis, #888)" strokeWidth={1}
      />
      {/* Y軸目盛(0/0.5/1.0) */}
      {[0, 0.5, 1].map((a) => (
        <text
          key={a} x={PAD.left - 4} y={sy(a) + 3}
          textAnchor="end" fontSize={8} fill="var(--chart-axis, #888)"
        >
          {a.toFixed(1)}
        </text>
      ))}
      <text
        x={PAD.left + plotW / 2} y={H - 4}
        textAnchor="middle" fontSize={9} fill="var(--chart-axis, #888)"
      >
        総LUT数(回路面積)
      </text>

      {/* パレートフロント線 */}
      {line.length >= 2 && (
        <polyline
          points={linePath}
          fill="none"
          stroke="var(--primary, #6366f1)"
          strokeWidth={1.5}
          strokeDasharray="4 3"
          data-testid="pareto-line"
        />
      )}

      {/* trial点 */}
      {done.map((t) => {
        const onFront = frontIds.has(t.trial_id);
        const cx = sx(t.total_luts);
        const cy = sy(t.acc);
        return (
          <circle
            key={t.trial_id}
            cx={cx}
            cy={cy}
            r={onFront ? 5 : 3.5}
            fill={t.partial ? 'transparent' : onFront ? 'var(--primary, #6366f1)' : 'var(--chart-axis, #888)'}
            stroke={onFront ? 'var(--primary, #6366f1)' : 'var(--chart-axis, #888)'}
            strokeWidth={t.partial ? 1.5 : 1}
            style={{ cursor: onSelectTrial ? 'pointer' : 'default' }}
            onClick={() => onSelectTrial?.(t.trial_id)}
            data-testid={`tradeoff-point-${t.trial_id}`}
            data-pareto={onFront ? 'true' : 'false'}
          >
            <title>{`trial ${t.trial_id}: acc=${t.acc.toFixed(3)} LUT=${t.total_luts}${t.partial ? ' (部分値)' : ''}`}</title>
          </circle>
        );
      })}

      {/* 現モデル位置 */}
      {currentModel && currentModel.acc !== null && (
        <g data-testid="tradeoff-current">
          <circle
            cx={sx(currentModel.total_luts)}
            cy={sy(currentModel.acc)}
            r={5}
            fill="none"
            stroke="var(--destructive, #ef4444)"
            strokeWidth={2}
          />
          <title>現在のモデル</title>
        </g>
      )}
    </svg>
  );
}
