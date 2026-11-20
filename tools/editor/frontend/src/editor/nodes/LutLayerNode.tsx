// LutLayerのカスタムノード: 名前・in/out幅・N・ゲート分布ミニバー。
// 小規模層(DETAIL_MANUAL_MAX以下)は詳細ゲートビューへ「展開」できる。

import { Handle, Position, type NodeProps } from '@xyflow/react';
import { DETAIL_MANUAL_MAX } from '../../detail/detailConfig';
import { LayerTiles } from '../../detail/LayerTiles';
import type { LutLayerFlowNode } from '../../ir/irToGraph';
import { useLayerView } from '../../store/selectors';
import { useUIStore } from '../../store/uiStore';
import { useViewStore } from '../../store/viewStore';
import { GateHistogram } from './GateHistogram';

export function LutLayerNode({ id, data, selected }: NodeProps<LutLayerFlowNode>) {
  // live=layer_summary / replay=履歴ビュー(useLayerViewが解決)。なければIR由来
  const view = useLayerView(id);
  const setView = useViewStore((s) => s.setView);
  const setLayerExpanded = useUIStore((s) => s.setLayerExpanded);
  // 誤差逆伝播リプレイ中の発光
  const bpActive = useViewStore((s) => s.bpAnim.activeLayer === id);
  // §3 L1: 俯瞰(overview)はゲート分布ミニバー、タイル/ゲート段階は集約タイル表示
  const showTiles = useViewStore((s) => s.lodStage !== 'overview');
  const gateIds = view.gateIds ?? data.initialGateIds;
  const expandable = data.outWidth <= DETAIL_MANUAL_MAX;

  return (
    <div
      className={`editor-node lut-node${selected ? ' selected' : ''}`}
      style={
        bpActive
          ? { boxShadow: '0 0 0 3px var(--chart-loss), 0 0 16px var(--chart-loss)' }
          : undefined
      }
    >
      <Handle type="target" position={Position.Left} />
      <div className="node-title">
        {data.label}
        {!data.initialized && <span className="node-draft">draft</span>}
      </div>
      <div className="node-sub">
        {data.inWidth} → {data.outWidth} ・ N={data.nInputs} ・ {data.parameterization}
      </div>
      {showTiles ? (
        <LayerTiles layerId={id} outWidth={data.outWidth} />
      ) : (
        <GateHistogram gateIds={gateIds} nInputs={data.nInputs} />
      )}
      {expandable && (
        <button
          className="drill-btn nodrag"
          title="個別ゲート(記号・真理値表・配線)をキャンバス上に展開"
          onClick={(e) => {
            e.stopPropagation();
            setLayerExpanded(id, true);
          }}
        >
          展開
        </button>
      )}
      <button
        className="drill-btn nodrag"
        title="全ゲートをdeck.glで表示"
        onClick={(e) => {
          e.stopPropagation();
          setView({ kind: 'l0', nodeId: id });
        }}
      >
        ドリルダウン
      </button>
      <Handle type="source" position={Position.Right} />
    </div>
  );
}
