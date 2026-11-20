// 手動配線エディタ。選択lut_layerノードのニューロン1個のNスロット
// (入力インデックス)を数値入力で編集し、SVG二部グラフでプレビューする。
// - バリデーション: 0 <= v < in_width。不正スロットがある間は適用不可
// - out_width > L0_EDIT_MAX はreadonly表示(大規模層の全IR PUTを避ける)
// - ビルド済みなら単一ニューロンAPIでin-place同期(逆引き再計算はバックエンド)、
//   未ビルドはIR更新+デバウンスPUT(TruthTableEditorと同じ二経路)

import { useEffect, useState } from 'react';
import { postNodeIndices } from '../api/rest';
import { useGraphStore } from '../store/graphStore';
import { useTrainingStore } from '../store/trainingStore';
import type { LutLayerParams } from '../ir/schema';
import { useUIStore } from '../store/uiStore';
import { cssVar } from '../theme/theme';

// これを超えるout_widthの層は配線編集をreadonlyにする
export const L0_EDIT_MAX = 500;

// SVGプレビュー寸法
const SVG_W = 320;
const SVG_H = 150;
const PAD = 14;
const SLOT_COUNT = 6;

/** スロットjの表示色(--slot-0..5トークン)。SVG属性はvar()非対応のため実値を読む。 */
function slotColor(j: number): string {
  return cssVar(`--slot-${j % SLOT_COUNT}`);
}

interface Props {
  nodeId: string;
}

/** 入力インデックスv(0..inWidth-1)を入力列のy座標へ写像する。 */
function inputY(v: number, inWidth: number): number {
  const t = inWidth <= 1 ? 0.5 : v / (inWidth - 1);
  return PAD + t * (SVG_H - 2 * PAD);
}

export function IndicesEditor({ nodeId }: Props) {
  const ir = useGraphStore((s) => s.ir);
  const needsBuild = useGraphStore((s) => s.needsBuild);
  const updateLutParams = useGraphStore((s) => s.updateLutParams);
  const trainStatus = useTrainingStore((s) => s.status);

  const selectedNeuron = useUIStore((s) => s.selectedNeuron);

  const [neuron, setNeuron] = useState(0);
  const [drafts, setDrafts] = useState<string[]>([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    // eslint-disable-next-line react-hooks/set-state-in-effect -- ノード切替時のローカル編集状態リセット(意図的なprop駆動setState)
    setNeuron(0);
    setDrafts([]);
    setError(null);
  }, [nodeId]);

  // 詳細ゲートビューでゲートをクリックしたら当該ニューロンへ連動
  useEffect(() => {
    if (selectedNeuron && selectedNeuron.nodeId === nodeId) {
      // eslint-disable-next-line react-hooks/set-state-in-effect -- 詳細ビューのゲート選択(外部store)との連動(意図的)
      setNeuron(selectedNeuron.index);
      setDrafts([]);
      setError(null);
    }
  }, [selectedNeuron, nodeId]);

  const node = ir?.nodes.find((n) => n.id === nodeId);
  if (!node || node.type !== 'lut_layer') return null;

  const p = node.params as unknown as LutLayerParams;
  const { in_width: inWidth, out_width: outWidth, n_inputs: nInputs } = p;
  const readonly = outWidth > L0_EDIT_MAX;
  const running = trainStatus === 'running' || trainStatus === 'starting';

  const safeNeuron = Math.min(Math.max(neuron, 0), outWidth - 1);
  const current = Array.from({ length: nInputs }, (_, j) => p.indices[j]?.[safeNeuron] ?? 0);

  // draft未入力のスロットはIRの現在値を表示する
  const slotText = (j: number): string => drafts[j] ?? String(current[j]);
  const parsedSlots = Array.from({ length: nInputs }, (_, j) => {
    const v = Number(slotText(j));
    return Number.isInteger(v) && v >= 0 && v < inWidth ? v : null;
  });
  const allValid = parsedSlots.every((v) => v !== null);
  const changed = parsedSlots.some((v, j) => v !== current[j]);
  const canApply = !readonly && !busy && !running && allValid && changed;

  const selectNeuron = (v: number) => {
    setNeuron(v);
    setDrafts([]);
    setError(null);
  };

  const handleApply = async () => {
    if (!canApply) return;
    const slots = parsedSlots as number[];
    setError(null);
    if (!needsBuild) {
      // ビルド済み: バックエンドAPIがIR+indices+逆引きを同時更新(PUT不要)
      setBusy(true);
      try {
        await postNodeIndices(nodeId, safeNeuron, slots);
        updateLutParams(
          nodeId,
          (params) => {
            slots.forEach((v, j) => {
              params.indices[j][safeNeuron] = v;
            });
          },
          { sync: false },
        );
        setDrafts([]);
      } catch (e) {
        setError(e instanceof Error ? e.message : String(e));
      } finally {
        setBusy(false);
      }
    } else {
      updateLutParams(nodeId, (params) => {
        slots.forEach((v, j) => {
          params.indices[j][safeNeuron] = v;
        });
      });
      setDrafts([]);
    }
  };

  // プレビューは有効な入力値を優先表示(編集途中の配線を確認できる)
  const previewSlots = parsedSlots.map((v, j) => v ?? current[j]);
  const neuronX = SVG_W - PAD * 2;
  const neuronY = SVG_H / 2;

  return (
    <div className="indices-editor">
      <h3>接続編集 (indices)</h3>
      <div className="form-row">
        <label>
          ニューロン番号 (0〜{outWidth - 1})
          <input
            type="number"
            min={0}
            max={outWidth - 1}
            value={safeNeuron}
            onChange={(e) => {
              const v = Number(e.target.value);
              if (Number.isInteger(v)) selectNeuron(v);
            }}
          />
        </label>
      </div>
      {readonly && (
        <div className="hint-text">
          out_width({outWidth}) &gt; {L0_EDIT_MAX} のため配線はreadonly表示です
        </div>
      )}
      {running && !readonly && (
        <div className="hint-text">訓練running中は編集不可(pauseで編集可能)</div>
      )}
      <div className="slot-row">
        {Array.from({ length: nInputs }, (_, j) => (
          <label key={j}>
            s{j}
            <input
              type="number"
              min={0}
              max={inWidth - 1}
              value={slotText(j)}
              disabled={readonly || busy}
              className={parsedSlots[j] === null ? 'invalid' : undefined}
              onChange={(e) => {
                const next = [...drafts];
                next[j] = e.target.value;
                setDrafts(next);
              }}
            />
          </label>
        ))}
        {!readonly && (
          <button disabled={!canApply} onClick={() => void handleApply()}>
            適用
          </button>
        )}
      </div>
      {!allValid && (
        <div className="error-text">スロット値は 0〜{inWidth - 1} の整数</div>
      )}
      {error && <div className="error-text">{error}</div>}
      <svg width={SVG_W} height={SVG_H} role="img" aria-label="配線プレビュー">
        {/* 入力列(左)の軸 */}
        <line
          x1={PAD}
          y1={PAD}
          x2={PAD}
          y2={SVG_H - PAD}
          stroke={cssVar('--edge-default')}
          strokeWidth={4}
          strokeLinecap="round"
        />
        <text x={PAD - 8} y={PAD + 3} fill={cssVar('--muted-foreground')} fontSize={9} textAnchor="end">
          0
        </text>
        <text x={PAD - 8} y={SVG_H - PAD + 3} fill={cssVar('--muted-foreground')} fontSize={9} textAnchor="end">
          {inWidth - 1}
        </text>
        {/* 選択ニューロン(右)と配線 */}
        {previewSlots.map((v, j) => {
          const color = slotColor(j);
          const y = inputY(v, inWidth);
          return (
            <g key={j}>
              <line x1={PAD} y1={y} x2={neuronX} y2={neuronY} stroke={color} strokeWidth={1.5} opacity={0.85} />
              <circle cx={PAD} cy={y} r={4} fill={color} />
              <text x={PAD + 8} y={y - 4} fill={color} fontSize={9}>
                s{j}={v}
              </text>
            </g>
          );
        })}
        <circle
          cx={neuronX}
          cy={neuronY}
          r={7}
          fill={cssVar('--canvas-bg')}
          stroke={cssVar('--primary')}
          strokeWidth={1.5}
        />
        <text x={neuronX} y={neuronY - 12} fill={cssVar('--muted-foreground')} fontSize={9} textAnchor="middle">
          #{safeNeuron}
        </text>
      </svg>
    </div>
  );
}
