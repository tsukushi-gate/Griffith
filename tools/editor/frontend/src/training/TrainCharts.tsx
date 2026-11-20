// 学習チャート(下部ドック「学習」タブ): loss / acc / レイヤ別‖ΔW‖。
// hard eval(離散化)のval_loss/val_accを点+破線で重ね描きする
//   (train曲線との乖離=離散化ギャップが見える)。
// classificationタスクでは混同行列ヒートマップを表示する。
// 実行コントロールはヘッダの TrainControls が担当する。

import { useEffect, useMemo, useRef, useState } from 'react';
import type uPlot from 'uplot';
import { getTrainConfusion, type ConfusionResponse } from '../api/rest';
import { useChartCursorStep } from '../store/selectors';
import { alignValSeries, useTrainingStore } from '../store/trainingStore';
import { cssVar } from '../theme/theme';
import { SelectionGapPanel } from './SelectionGapPanel';
import { TrainSettingsForm } from './TrainSettingsForm';
import { UplotChart } from './UplotChart';

const SERIES_COUNT = 8;
const VAL_DASH = [6, 4]; // val系列の破線パターン
const CONFUSION_CELL_PX = 16;
const CONFUSION_MARGIN_PX = 18; // 軸ラベル領域

/** レイヤ別系列の表示色(--chart-series-0..7トークン)。 */
function seriesColor(i: number): string {
  return cssVar(`--chart-series-${i % SERIES_COUNT}`);
}

/** 混同行列ヒートマップの描画(白→primary濃淡+正解/予測ラベル軸)。 */
function drawConfusion(canvas: HTMLCanvasElement, matrix: number[][]): void {
  const n = matrix.length;
  const size = CONFUSION_MARGIN_PX + n * CONFUSION_CELL_PX;
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext('2d');
  if (!ctx) return;

  const primary = cssVar('--primary');
  const axis = cssVar('--chart-axis');
  const max = Math.max(1, ...matrix.map((row) => Math.max(...row)));

  ctx.clearRect(0, 0, size, size);
  for (let r = 0; r < n; r++) {
    for (let c = 0; c < n; c++) {
      const x = CONFUSION_MARGIN_PX + c * CONFUSION_CELL_PX;
      const y = CONFUSION_MARGIN_PX + r * CONFUSION_CELL_PX;
      ctx.fillStyle = '#ffffff';
      ctx.fillRect(x, y, CONFUSION_CELL_PX, CONFUSION_CELL_PX);
      const v = matrix[r][c];
      if (v > 0) {
        ctx.globalAlpha = Math.max(0.08, v / max);
        ctx.fillStyle = primary;
        ctx.fillRect(x, y, CONFUSION_CELL_PX, CONFUSION_CELL_PX);
        ctx.globalAlpha = 1;
      }
    }
  }
  // ラベル軸(行=正解、列=予測)
  ctx.fillStyle = axis;
  ctx.font = '9px sans-serif';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  for (let i = 0; i < n; i++) {
    const center = CONFUSION_MARGIN_PX + i * CONFUSION_CELL_PX + CONFUSION_CELL_PX / 2;
    ctx.fillText(String(i), center, CONFUSION_MARGIN_PX / 2); // 予測(列)
    ctx.fillText(String(i), CONFUSION_MARGIN_PX / 2, center); // 正解(行)
  }
}

/** 混同行列パネル。binary_vector等で未生成(404)の間は非表示。*/
function ConfusionPanel() {
  const sessionId = useTrainingStore((s) => s.sessionId);
  const version = useTrainingStore((s) => s.version);
  const [confusion, setConfusion] = useState<ConfusionResponse | null>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  // eval実施数(valAccペア数)の変化をトリガに最新の混同行列を取得する
  const valCount = useTrainingStore.getState().valAcc.length;
  void version;

  useEffect(() => {
    if (!sessionId || valCount === 0) {
      // eslint-disable-next-line react-hooks/set-state-in-effect -- セッション切替時の表示リセット(意図的)
      setConfusion(null);
      return;
    }
    let disposed = false;
    getTrainConfusion(sessionId)
      .then((c) => {
        if (!disposed) setConfusion(c);
      })
      .catch(() => {
        // 404 = binary_vectorタスク or 未評価 → パネル非表示
        if (!disposed) setConfusion(null);
      });
    return () => {
      disposed = true;
    };
  }, [sessionId, valCount]);

  useEffect(() => {
    if (canvasRef.current && confusion) {
      drawConfusion(canvasRef.current, confusion.matrix);
    }
  }, [confusion]);

  if (!confusion) return null;
  return (
    <div className="flex flex-col gap-1">
      <div className="text-[11px] text-muted-foreground">
        混同行列(hard eval @ step {confusion.step ?? '-'}
        {confusion.val_acc != null && ` ・ val_acc ${(confusion.val_acc * 100).toFixed(1)}%`})
        <span className="ml-1">縦=正解 / 横=予測</span>
      </div>
      <canvas ref={canvasRef} className="w-fit" />
    </div>
  );
}

export function TrainCharts() {
  const version = useTrainingStore((s) => s.version);
  const cursorStep = useChartCursorStep(); // リプレイ時にチャートへ縦線を出す

  // チャートデータ: 配列はin-place追記されるためversionでmemoを更新する
  const store = useTrainingStore.getState();
  const lossData = useMemo(
    () =>
      [
        store.steps,
        store.loss,
        alignValSeries(store.steps, store.valLoss),
      ] as uPlot.AlignedData,
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [version],
  );
  const accData = useMemo(
    () =>
      [
        store.steps,
        store.acc,
        alignValSeries(store.steps, store.valAcc),
        alignValSeries(store.steps, store.valAccSoft),
      ] as uPlot.AlignedData,
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [version],
  );
  const deltaKeys = useMemo(
    () => Object.keys(store.deltaSeries).sort(),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [version],
  );
  // 系列ゼロのuPlotは作れないため、データ未着時はプレースホルダ系列を渡す
  const deltaData = useMemo(
    () =>
      (deltaKeys.length > 0
        ? [store.deltaSteps, ...deltaKeys.map((k) => store.deltaSeries[k])]
        : [[], []]) as uPlot.AlignedData,
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [version, deltaKeys],
  );
  const deltaSeriesDefs =
    deltaKeys.length > 0
      ? deltaKeys.map((k, i) => ({
          label: k,
          stroke: seriesColor(i),
        }))
      : [{ label: 'ΔW', stroke: seriesColor(0) }];

  const lastStep = store.steps.length > 0 ? store.steps[store.steps.length - 1] : 0;
  const lastLoss = store.loss.length > 0 ? store.loss[store.loss.length - 1] : null;
  const lastAcc = store.acc.length > 0 ? store.acc[store.acc.length - 1] : null;
  const lastVal =
    store.valAcc.length > 0 ? store.valAcc[store.valAcc.length - 1] : null;

  return (
    <div className="flex h-full min-h-0">
      {/* §2: 学習設定の常設フォーム(旧ヘッダPopoverから移設) */}
      <TrainSettingsForm />
      <div className="flex flex-1 flex-col gap-1 overflow-y-auto p-3">
        <div className="text-[11px] text-muted-foreground">
          step {lastStep}
          {lastLoss !== null && ` ・ loss ${lastLoss.toFixed(4)}`}
          {lastAcc !== null && ` ・ acc ${((lastAcc ?? 0) * 100).toFixed(1)}%`}
          {lastVal !== null && ` ・ val(hard) ${(lastVal[1] * 100).toFixed(1)}%`}
        </div>
        <div className="grid grid-cols-1 gap-4 lg:grid-cols-3">
          <UplotChart
            title="loss (train / val)"
            series={[
              { label: 'loss', stroke: cssVar('--chart-loss') },
              {
                label: 'val_loss',
                stroke: cssVar('--chart-loss'),
                dash: VAL_DASH,
                points: true,
                spanGaps: true,
              },
            ]}
            data={lossData}
            cursorStep={cursorStep}
          />
          <UplotChart
            title="acc (train / val hard / val soft)"
            series={[
              { label: 'acc', stroke: cssVar('--chart-acc') },
              {
                label: 'val_acc (hard)',
                stroke: cssVar('--chart-acc'),
                dash: VAL_DASH,
                points: true,
                spanGaps: true,
              },
              {
                label: 'val_acc (soft)',
                stroke: cssVar('--chart-loss'),
                dash: VAL_DASH,
                points: true,
                spanGaps: true,
              },
            ]}
            data={accData}
            cursorStep={cursorStep}
          />
          <UplotChart
            title="‖ΔW‖ (レイヤ別)"
            series={deltaSeriesDefs}
            data={deltaData}
            cursorStep={cursorStep}
          />
        </div>
        <SelectionGapPanel />
        <ConfusionPanel />
      </div>
    </div>
  );
}
