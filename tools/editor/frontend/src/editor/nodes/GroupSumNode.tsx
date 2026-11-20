// GroupSumのカスタムノード

import { Handle, Position, type NodeProps } from '@xyflow/react';
import type { GroupSumFlowNode } from '../../ir/irToGraph';

export function GroupSumNode({ data, selected }: NodeProps<GroupSumFlowNode>) {
  return (
    <div className={`editor-node groupsum-node${selected ? ' selected' : ''}`}>
      <Handle type="target" position={Position.Left} />
      <div className="node-title">{data.label}</div>
      <div className="node-sub">
        GroupSum ・ {data.inWidth} → {data.k} ・ τ={data.tau}
      </div>
      <Handle type="source" position={Position.Right} />
    </div>
  );
}
