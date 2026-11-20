// 選択レイヤの先頭〜256ニューロンのゲートID/確信度ヒートマップ(canvas 2D)。
// セル色=ゲート色、明度=確信度。live=layer_summary / replay=履歴ビュー(最大5Hz更新)。

import { useEffect, useMemo, useRef } from 'react';
import { blendToBg, gateColor, GATE_NAMES, rgbCss, ttToGateIdN2 } from '../ir/gates';
import type { LutLayerParams } from '../ir/schema';
import { useGraphStore } from '../store/graphStore';
import { useLayerView } from '../store/selectors';
import { cssVar, cssVarRGB } from '../theme/theme';

const MAX_NEURONS = 256;
const COLS = 16;
const CELL = 20; // セルのピクセルサイズ(隙間1px込み)

interface Props {
  nodeId: string;
}

export function LutHeatmap({ nodeId }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const view = useLayerView(nodeId);
  const ir = useGraphStore((s) => s.ir);

  // 初期表示用: IRの真理値表からゲートIDを導出(確信度は1.0扱い)
  const initial = useMemo(() => {
    const node = ir?.nodes.find((n) => n.id === nodeId);
    if (!node || node.type !== 'lut_layer') return null;
    const p = node.params as unknown as LutLayerParams;
    if (p.n_inputs !== 2) return { gateIds: null, nInputs: p.n_inputs };
    const gateIds = p.truth_tables
      .slice(0, MAX_NEURONS)
      .map((t) => ttToGateIdN2(parseInt(t, 16)));
    return { gateIds, nInputs: p.n_inputs };
  }, [ir, nodeId]);

  const nInputs = view.nInputs ?? initial?.nInputs ?? 2;
  const gateIds = view.gateIds ?? initial?.gateIds ?? null;
  const confidence = view.confidence;

  useEffect(() => {
    const canvas = canvasRef.current;
    const ctx = canvas?.getContext('2d');
    if (!canvas || !ctx) return;

    const bg = cssVarRGB('--canvas-bg', [255, 255, 255]);
    const n = Math.min(gateIds?.length ?? 0, MAX_NEURONS);
    const rows = Math.max(Math.ceil(n / COLS), 1);
    canvas.height = rows * CELL;
    ctx.fillStyle = rgbCss(bg);
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    if (!gateIds || n === 0) {
      ctx.fillStyle = cssVar('--muted-foreground');
      ctx.font = '11px sans-serif';
      ctx.fillText('データなし', 6, 16);
      return;
    }

    // 白背景では淡色セルが背景に溶けるため、各セルに1pxのボーダーリングを描く
    ctx.strokeStyle = cssVar('--border');
    ctx.lineWidth = 1;
    for (let i = 0; i < n; i++) {
      const conf = confidence ? Math.min(Math.max(confidence[i], 0), 1) : 1;
      const color = blendToBg(gateColor(gateIds[i], nInputs), bg, 0.15 + 0.85 * conf);
      const x = (i % COLS) * CELL;
      const y = Math.floor(i / COLS) * CELL;
      ctx.fillStyle = rgbCss(color);
      ctx.fillRect(x + 1, y + 1, CELL - 2, CELL - 2);
      ctx.strokeRect(x + 1.5, y + 1.5, CELL - 3, CELL - 3);
    }
  }, [gateIds, confidence, nInputs]);

  return (
    <div className="lut-heatmap">
      <canvas ref={canvasRef} width={COLS * CELL} height={CELL} />
      <div className="hint-text">
        先頭{Math.min(gateIds?.length ?? 0, MAX_NEURONS)}ニューロン ・ 明度=確信度
        {view.step !== null
          ? ` ・ step ${view.step}${view.source === 'replay' ? '(リプレイ)' : ''}`
          : ' ・ 初期状態(IR由来)'}
      </div>
      {nInputs === 2 && (
        <div className="gate-legend">
          {Object.entries(GATE_NAMES).map(([id, name]) => (
            <span key={id} className="gate-chip">
              <i style={{ background: rgbCss(gateColor(Number(id), 2)) }} />
              {name}
            </span>
          ))}
        </div>
      )}
    </div>
  );
}
