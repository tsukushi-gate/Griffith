// 展開中lut_layerのグループ枠。タイトルバー+「畳む」ボタン。
// 子ゲートノードはReact Flowがparent-child関係で内側に描画する。

import { Handle, Position, type NodeProps } from '@xyflow/react';
import { Minimize2 } from 'lucide-react';
import { useUIStore } from '../../store/uiStore';
import { useViewStore } from '../../store/viewStore';
import type { LutLayerGroupFlowNode } from '../irToDetailGraph';

export function LutLayerGroupNode({ id, data, selected }: NodeProps<LutLayerGroupFlowNode>) {
  const setLayerExpanded = useUIStore((s) => s.setLayerExpanded);
  // 誤差逆伝播リプレイ中の発光
  const bpActive = useViewStore((s) => s.bpAnim.activeLayer === id);

  return (
    <div
      className={`h-full w-full rounded-lg border-2 bg-[color-mix(in_srgb,var(--panel-bg)_60%,transparent)] ${
        selected ? 'border-primary' : 'border-[var(--slot-0)]/40'
      }`}
      style={
        bpActive
          ? { boxShadow: '0 0 0 3px var(--chart-loss), 0 0 20px var(--chart-loss)' }
          : undefined
      }
    >
      <Handle type="target" position={Position.Left} />
      <div className="flex h-[30px] items-center gap-2 border-b bg-[var(--panel-bg)] px-2 rounded-t-lg">
        <span className="text-xs font-semibold">{data.label}</span>
        {!data.initialized && <span className="node-draft">draft</span>}
        <span className="text-[10px] text-muted-foreground">
          {data.inWidth} → {data.outWidth} ・ N={data.nInputs} ・ {data.parameterization}
        </span>
        <div className="flex-1" />
        <button
          type="button"
          data-slot="group-collapse"
          className="nodrag flex cursor-pointer items-center gap-1 rounded border bg-background px-1.5 py-0.5 text-[10px] text-muted-foreground hover:bg-secondary"
          title="抽象表示へ畳む"
          onClick={(e) => {
            e.stopPropagation();
            setLayerExpanded(id, false);
          }}
        >
          <Minimize2 className="size-3" />
          畳む
        </button>
      </div>
      <Handle type="source" position={Position.Right} />
    </div>
  );
}
