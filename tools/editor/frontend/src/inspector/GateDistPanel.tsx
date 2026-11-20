// 選択ニューロンの16ゲート確率分布パネル。
// live時はWSのgate_distribution、replay時は履歴ビューから表示する。
// 分布はソフトマックス確率そのもの(現行layer_summaryのmax潰しの非集約版)。

import { GATE_NAMES, gateColor, rgbCss } from '../ir/gates';
import { useLayerView } from '../store/selectors';
import { useUIStore } from '../store/uiStore';

interface Props {
  nodeId: string;
}

export function GateDistPanel({ nodeId }: Props) {
  const view = useLayerView(nodeId);
  const selectedNeuron = useUIStore((s) => s.selectedNeuron);
  const neuron =
    selectedNeuron?.nodeId === nodeId ? selectedNeuron.index : 0;

  if (!view.gateProbs || view.probBins === 0) {
    return (
      <div className="hint-text">
        確率分布データ未受信(学習を開始すると小規模層の分布がストリーミングされます)
      </div>
    );
  }

  const bins = view.probBins;
  const outDim = view.gateProbs.length / bins;
  const idx = Math.min(Math.max(neuron, 0), outDim - 1);
  const probs = Array.from(view.gateProbs.slice(idx * bins, (idx + 1) * bins));
  const maxP = Math.max(...probs, 1e-6);
  const entropy = -probs.reduce(
    (acc, p) => acc + (p > 1e-12 ? p * Math.log(p) : 0),
    0,
  );

  return (
    <div className="flex flex-col gap-1">
      <div className="text-[11px] text-muted-foreground">
        ニューロン #{idx}(詳細ビューでゲートをクリックすると切替) ・ H=
        {entropy.toFixed(3)} ・ {view.source === 'replay' ? `step ${view.step}` : 'live'}
      </div>
      <div className="flex flex-col gap-px">
        {probs.map((p, g) => (
          <div key={g} className="flex items-center gap-1.5 text-[10px]">
            <span className="w-10 shrink-0 text-right font-mono text-muted-foreground">
              {bins === 16 ? (GATE_NAMES[g] ?? `#${g}`) : `#${g}`}
            </span>
            <div className="h-[10px] flex-1 overflow-hidden rounded-sm bg-muted">
              <div
                className="h-full rounded-sm"
                style={{
                  width: `${(p / maxP) * 100}%`,
                  background: rgbCss(gateColor(g, bins === 16 ? 2 : 3)),
                }}
              />
            </div>
            <span className="w-11 shrink-0 font-mono text-muted-foreground">
              {(p * 100).toFixed(1)}%
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}
