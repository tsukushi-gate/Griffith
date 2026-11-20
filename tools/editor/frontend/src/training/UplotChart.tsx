// uPlotのReactラッパー(ストリーミング追記はsetDataで反映)。
// 下部ドックのリサイズに追従するため、コンテナ幅をResizeObserverで監視する。

import { useEffect, useRef } from 'react';
import uPlot from 'uplot';
import { cssVar } from '../theme/theme';

const DEFAULT_HEIGHT = 140;
const DEFAULT_WIDTH = 336;

interface SeriesDef {
  label: string;
  stroke: string;
  /** 破線パターン(val系列の重ね描き用) */
  dash?: number[];
  /** 点マーカー表示(疎なval系列用) */
  points?: boolean;
  /** null(欠測)をまたいで線を繋ぐ(疎系列用) */
  spanGaps?: boolean;
}

interface Props {
  title: string;
  series: SeriesDef[];
  data: uPlot.AlignedData;
  height?: number;
  /** リプレイ時のカーソルstep(x値)。nullで非表示 */
  cursorStep?: number | null;
}

function hasRenderableData(data: uPlot.AlignedData): boolean {
  const [x, ...ys] = data;
  if (!Array.isArray(x) || x.length === 0) return false;
  return ys.some((series) => series.some((v) => typeof v === 'number' && Number.isFinite(v)));
}

export function UplotChart({
  title,
  series,
  data,
  height = DEFAULT_HEIGHT,
  cursorStep = null,
}: Props) {
  const containerRef = useRef<HTMLDivElement>(null);
  const plotRef = useRef<uPlot | null>(null);
  const dataRef = useRef(data);
  // eslint-disable-next-line react-hooks/refs -- チャート再生成effectが依存に含めず最新データを参照するlatest-refパターン(意図的)
  dataRef.current = data;
  const cursorStepRef = useRef<number | null>(cursorStep);
  const renderable = hasRenderableData(data);

  // 系列構成が変わったらチャートを作り直す(weight_deltaのレイヤ追加等)
  const seriesKey = series.map((s) => s.label).join('|');

  useEffect(() => {
    const el = containerRef.current;
    if (!el || !renderable) {
      plotRef.current?.destroy();
      plotRef.current = null;
      return;
    }
    const width = el.clientWidth || DEFAULT_WIDTH;
    const axisColor = cssVar('--chart-axis');
    const gridColor = cssVar('--chart-grid');
    const opts: uPlot.Options = {
      title,
      width,
      height,
      series: [
        {},
        ...series.map((s) => ({
          label: s.label,
          stroke: s.stroke,
          width: 1.5,
          ...(s.dash ? { dash: s.dash } : {}),
          ...(s.spanGaps !== undefined ? { spanGaps: s.spanGaps } : {}),
          ...(s.points ? { points: { show: true, size: 5, fill: s.stroke } } : {}),
        })),
      ],
      axes: [
        { stroke: axisColor, grid: { stroke: gridColor }, ticks: { stroke: gridColor } },
        { stroke: axisColor, grid: { stroke: gridColor }, ticks: { stroke: gridColor } },
      ],
      legend: { show: series.length > 1 },
      cursor: { show: false },
      // リプレイカーソル(縦線)を描画するフックプラグイン
      hooks: {
        draw: [
          (u: uPlot) => {
            const step = cursorStepRef.current;
            if (step === null) return;
            const x = u.valToPos(step, 'x', true);
            if (x < u.bbox.left || x > u.bbox.left + u.bbox.width) return;
            const ctx = u.ctx;
            ctx.save();
            ctx.strokeStyle = cssVar('--primary');
            ctx.lineWidth = 1.5;
            ctx.setLineDash([4, 3]);
            ctx.beginPath();
            ctx.moveTo(x, u.bbox.top);
            ctx.lineTo(x, u.bbox.top + u.bbox.height);
            ctx.stroke();
            ctx.restore();
          },
        ],
      },
    };
    plotRef.current = new uPlot(opts, dataRef.current, el);

    // コンテナ幅の変化に追従(高さは固定)
    const observer = new ResizeObserver((entries) => {
      const w = Math.floor(entries[0]?.contentRect.width ?? 0);
      if (w > 0 && plotRef.current && w !== plotRef.current.width) {
        plotRef.current.setSize({ width: w, height });
      }
    });
    observer.observe(el);
    return () => {
      observer.disconnect();
      plotRef.current?.destroy();
      plotRef.current = null;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [seriesKey, title, height, renderable]);

  useEffect(() => {
    if (!renderable) {
      plotRef.current?.destroy();
      plotRef.current = null;
      return;
    }
    plotRef.current?.setData(data);
  }, [data, renderable]);

  // カーソルstep変化で再描画(drawフックが縦線を引く)
  useEffect(() => {
    cursorStepRef.current = cursorStep;
    plotRef.current?.redraw();
  }, [cursorStep]);

  return <div ref={containerRef} className="chart" style={{ minHeight: height }} />;
}
