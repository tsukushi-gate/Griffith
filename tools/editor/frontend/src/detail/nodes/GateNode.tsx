// 個別ゲートカード。記号+ゲート名+真理値表ミニグリッド+確信度バー。
// 訓練中は layer_summary を自身で購読して再着色する(グラフ再合成は走らない)。
// probs は gate_distribution フレーム用に予約(データが来れば分布バーへ差替)。
// ズームがDETAIL_LOD_ZOOM未満の間は枠+ゲート名だけの軽量表示へ切り替え、
// store購読とDOMを間引く(しきい値を跨いだ時だけ再レンダされるようbool購読にする)。

import { Handle, Position, useStore, type NodeProps } from '@xyflow/react';
import { toast } from 'sonner';
import { gateNodeId } from '../../anim/edgeGradMap';
import { blendToBg, gateColor, GATE_NAMES, rgbCss, ttToGateIdN2 } from '../../ir/gates';
import { ttBit, ttParse } from '../../ir/hexTT';
import { usePlaygroundStore } from '../../playground/playgroundStore';
import { useLayerView } from '../../store/selectors';
import { useTrainingStore } from '../../store/trainingStore';
import { useUIStore } from '../../store/uiStore';
import { useViewStore } from '../../store/viewStore';
import { cssVarRGB } from '../../theme/theme';
import { DETAIL_LOD_ZOOM } from '../detailConfig';
import type { GateFlowNode } from '../irToDetailGraph';
import { canEditTT, toggleNeuronBit } from '../ttEdit';
import { GateSymbol } from './GateSymbol';

const SLOT_COUNT = 6;

/** N=2真理値表グリッドの表示順(TruthTableEditorと同じk昇順・2列) */
const TT_GRID_ORDER = [0, 1, 2, 3];

/** IR真理値表由来のゲートid(学習データ未着時・軽量表示のフォールバック) */
function irGateIdOf(tt: string, nInputs: number): number {
  return nInputs === 2 ? ttToGateIdN2(Number(ttParse(tt) & 0xfn)) : -1;
}

function gateNameOf(gateId: number, nInputs: number): string {
  return nInputs === 2 ? (GATE_NAMES[gateId] ?? `G${gateId}`) : `G${gateId}`;
}

/** スロット入力+出力ハンドル。エッジの接続点なので軽量表示でも同一構成を維持する */
function GateHandles({ nInputs }: { nInputs: number }) {
  // スロット入力ハンドルの縦位置(カード左辺に等間隔)
  const handleTop = (j: number) => `${((j + 1) / (nInputs + 1)) * 100}%`;
  return (
    <>
      {Array.from({ length: nInputs }, (_, j) => (
        <Handle
          key={j}
          id={`s${j}`}
          type="target"
          position={Position.Left}
          isConnectable={false}
          style={{
            top: handleTop(j),
            width: 6,
            height: 6,
            background: `var(--slot-${j % SLOT_COUNT})`,
            border: 'none',
          }}
        />
      ))}
      <Handle
        id="out"
        type="source"
        position={Position.Right}
        isConnectable={false}
        style={{ width: 6, height: 6, background: 'var(--edge-default)', border: 'none' }}
      />
    </>
  );
}

/** 軽量表示(LOD): 枠+ゲート名のみ。store購読を持たず訓練中5Hzでも再レンダしない */
function GateNodeLite({ data }: NodeProps<GateFlowNode>) {
  const { neuron, nInputs, tt } = data;
  const gateName = gateNameOf(irGateIdOf(tt, nInputs), nInputs);
  return (
    <div className="relative flex h-[96px] w-[114px] flex-col rounded-md border bg-card px-1.5 py-1 shadow-xs">
      <GateHandles nInputs={nInputs} />
      <div className="flex h-full flex-col items-center justify-center gap-1">
        <span className="text-[10px] leading-none text-muted-foreground">#{neuron}</span>
        <span className="text-sm leading-none font-semibold">{gateName}</span>
      </div>
    </div>
  );
}

function GateNodeFull({ data }: NodeProps<GateFlowNode>) {
  const { layerId, neuron, nInputs, tt, slotSources, showSlotLabels, parameterization } =
    data;
  // live/replayを自動解決(タイムラインスクラブで過去stepの状態に同期する)
  const view = useLayerView(layerId);
  const trainStatus = useTrainingStore((s) => s.status);
  const selectedNeuron = useUIStore((s) => s.selectedNeuron);
  // プレイグラウンド実行値。off時はundefinedで従来表示と完全同一
  const pgValue = usePlaygroundStore((s) =>
    s.enabled && s.values ? s.values[layerId]?.[neuron] : undefined,
  );
  // 逆伝播リプレイ中の自層アクティブ時のみ、grad_wノルム由来の発光強度[0,1]
  const bpGlow = useViewStore((s) =>
    s.bpAnim.activeLayer === layerId && s.bpAnim.gateIntensity
      ? (s.bpAnim.gateIntensity.get(gateNodeId(layerId, neuron)) ?? 0)
      : null,
  );

  // 学習データがあればgate_id/confidenceを優先、なければIR真理値表から導出
  const gateId = view.gateIds?.[neuron] ?? irGateIdOf(tt, nInputs);
  const confidence = view.confidence?.[neuron] ?? 1;
  // 16ゲート確率分布(gate_distributionフレーム受信時のみ)
  const probs =
    view.gateProbs && view.probBins > 0
      ? view.gateProbs.subarray(neuron * view.probBins, (neuron + 1) * view.probBins)
      : null;

  const bg = cssVarRGB('--canvas-bg', [255, 255, 255]);
  const color = blendToBg(
    gateColor(gateId, nInputs),
    bg,
    0.25 + 0.75 * Math.min(Math.max(confidence, 0), 1),
  );
  const gateName = gateNameOf(gateId, nInputs);

  const running = trainStatus === 'running' || trainStatus === 'starting';
  const editable = canEditTT(parameterization, nInputs) && !running;
  const isSelected =
    selectedNeuron?.nodeId === layerId && selectedNeuron.index === neuron;

  const handleToggle = async (k: number) => {
    if (!editable) return;
    try {
      await toggleNeuronBit(layerId, neuron, tt, k);
    } catch (e) {
      toast.error(e instanceof Error ? e.message : String(e));
    }
  };

  // プレイグラウンド実行時の枠着色(0=無色/1=primary系、softは値で補間)
  const pgT = pgValue !== undefined ? Math.min(Math.max(pgValue, 0), 1) : 0;
  const [pr, pg, pb] = cssVarRGB('--primary', [37, 99, 235]);
  const pgBorderStyle =
    pgValue !== undefined && !isSelected
      ? { borderColor: `rgba(${pr}, ${pg}, ${pb}, ${0.12 + 0.88 * pgT})` }
      : undefined;

  // 逆伝播リプレイ中の発光: grad_w強度でspread/αを増やす(既存の枠に追加)
  const [lr, lg, lb] = cssVarRGB('--chart-loss', [235, 104, 52]);
  const bpGlowStyle =
    bpGlow !== null
      ? {
          boxShadow: `0 0 0 ${1 + 2 * bpGlow}px rgba(${lr}, ${lg}, ${lb}, ${
            0.35 + 0.65 * bpGlow
          }), 0 0 ${4 + 14 * bpGlow}px rgba(${lr}, ${lg}, ${lb}, ${0.25 + 0.55 * bpGlow})`,
        }
      : undefined;

  return (
    <div
      className={`relative flex h-[96px] w-[114px] flex-col gap-0.5 rounded-md border bg-card px-1.5 py-1 text-[10px] shadow-xs ${
        isSelected ? 'border-primary ring-1 ring-primary' : ''
      }`}
      style={{ ...pgBorderStyle, ...bpGlowStyle }}
    >
      {/* 実行値バッジ: hardは0/1、softは確率2桁 */}
      {pgValue !== undefined && (
        <span
          className="absolute -top-2 -right-2 rounded-full border bg-card px-1 font-mono text-[9px] leading-4 font-bold shadow-xs"
          style={{
            color: pgT >= 0.5 ? 'var(--chart-acc)' : 'var(--muted-foreground)',
          }}
          title="プレイグラウンド実行値"
        >
          {pgValue === 0 || pgValue === 1 ? pgValue : pgValue.toFixed(2)}
        </span>
      )}
      <GateHandles nInputs={nInputs} />
      <div className="flex items-baseline justify-between leading-none">
        <span className="text-muted-foreground">#{neuron}</span>
        <span className="font-semibold">{gateName}</span>
      </div>
      <div className="flex items-center justify-between">
        <GateSymbol gateId={gateId} color={color} />
        {nInputs === 2 ? (
          <div
            className={`nodrag grid grid-cols-2 gap-px ${editable ? '' : 'pointer-events-none opacity-70'}`}
            title={editable ? '真理値表をクリックでトグル' : undefined}
          >
            {TT_GRID_ORDER.map((k) => {
              const bit = ttBit(tt, k);
              return (
                <button
                  key={k}
                  type="button"
                  data-slot="gate-tt-cell"
                  className={`flex h-[15px] w-[15px] cursor-pointer items-center justify-center rounded-[2px] border font-mono leading-none ${
                    bit
                      ? 'border-[#86efac] bg-[#dcfce7] font-bold text-[var(--status-ok)]'
                      : 'bg-background text-muted-foreground'
                  }`}
                  onClick={() => void handleToggle(k)}
                >
                  {bit}
                </button>
              );
            })}
          </div>
        ) : (
          <span className="max-w-[52px] truncate font-mono text-muted-foreground">{tt}</span>
        )}
      </div>
      {/* 16ゲート確率分布の積み上げバー(gate_distribution受信時)。なければ確信度バー */}
      {probs ? (
        <div
          className="flex h-[4px] w-full overflow-hidden rounded-full bg-muted"
          title="16ゲート確率分布(色=ゲート種別、幅=確率)"
        >
          {Array.from(probs).map((p, g) => (
            <div
              key={g}
              className="h-full"
              style={{
                width: `${p * 100}%`,
                background: rgbCss(gateColor(g, nInputs)),
              }}
            />
          ))}
        </div>
      ) : (
        <div className="h-[4px] w-full overflow-hidden rounded-full bg-muted">
          <div
            className="h-full rounded-full"
            style={{
              width: `${Math.round(confidence * 100)}%`,
              background: `rgb(${color[0]}, ${color[1]}, ${color[2]})`,
            }}
          />
        </div>
      )}
      {showSlotLabels && (
        <div className="truncate leading-none text-muted-foreground">
          {slotSources.map((s, j) => `s${j}=${s}`).join(' ')}
        </div>
      )}
    </div>
  );
}

export function GateNode(props: NodeProps<GateFlowNode>) {
  // transform[2](ズーム)の直読みはレンダ毎購読になるため、
  // しきい値を跨いだ時だけ値が変わるbooleanセレクタで購読する
  const lite = useStore((s) => s.transform[2] < DETAIL_LOD_ZOOM);
  return lite ? <GateNodeLite {...props} /> : <GateNodeFull {...props} />;
}
