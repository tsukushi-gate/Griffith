// ゲートID分布のミニバー(canvas 2D)。訓練中はlayer_summaryでライブ更新される。

import { useEffect, useRef } from 'react';
import { gateColor, rgbCss } from '../../ir/gates';
import { cssVar } from '../../theme/theme';

const N2_BINS = 16;
const MAX_BINS = 32; // N>2のときの表示ビン上限

interface Props {
  gateIds: number[] | null;
  nInputs: number;
  width?: number;
  height?: number;
}

export function GateHistogram({ gateIds, nInputs, width = 176, height = 40 }: Props) {
  const ref = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = ref.current;
    const ctx = canvas?.getContext('2d');
    if (!canvas || !ctx) return;
    ctx.clearRect(0, 0, width, height);

    if (!gateIds || gateIds.length === 0) {
      ctx.fillStyle = cssVar('--muted-foreground');
      ctx.font = '10px sans-serif';
      ctx.fillText('no data', 4, height / 2 + 3);
      return;
    }

    if (nInputs === 2) {
      // 16ゲート固定ビン
      const counts = new Array<number>(N2_BINS).fill(0);
      for (const g of gateIds) counts[g & (N2_BINS - 1)] += 1;
      const max = Math.max(...counts, 1);
      const bw = width / N2_BINS;
      counts.forEach((c, g) => {
        const h = (c / max) * (height - 2);
        ctx.fillStyle = rgbCss(gateColor(g, 2));
        ctx.fillRect(g * bw + 1, height - h, bw - 2, h);
      });
      return;
    }

    // N>2: 出現頻度上位のIDをビンにする
    const counts = new Map<number, number>();
    for (const g of gateIds) counts.set(g, (counts.get(g) ?? 0) + 1);
    const entries = [...counts.entries()].sort((a, b) => b[1] - a[1]).slice(0, MAX_BINS);
    const max = entries[0]?.[1] ?? 1;
    const bw = width / Math.max(entries.length, 1);
    entries.forEach(([g, c], i) => {
      const h = (c / max) * (height - 2);
      ctx.fillStyle = rgbCss(gateColor(g, nInputs));
      ctx.fillRect(i * bw + 1, height - h, Math.max(bw - 2, 1), h);
    });
  }, [gateIds, nInputs, width, height]);

  return <canvas ref={ref} width={width} height={height} className="gate-histogram" />;
}
