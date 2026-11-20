// conv2d / or_pool2d の専用カード表示。
// shape(C,H,W)・kernel・stride・paddingと flatten幅(C*H*W)を表示する。
// params編集は非対応(表示・検証・往復のみ)。

import { Handle, Position, type NodeProps } from '@xyflow/react';
import type { ConvFlowNode } from '../../ir/irToGraph';

function shapeLabel(shape: number[] | null): string {
  return shape ? `(${shape.join(',')})` : '?';
}

export function ConvNode({ data, selected }: NodeProps<ConvFlowNode>) {
  return (
    <div
      className={`editor-node conv-node${selected ? ' selected' : ''}`}
      style={{ borderLeft: '3px solid var(--slot-4)' }}
    >
      <Handle type="target" position={Position.Left} />
      <div className="node-title">{data.label}</div>
      <div className="node-sub">
        {data.kind} ・ {shapeLabel(data.inShape)} → {shapeLabel(data.outShape)}
      </div>
      <div style={{ fontSize: 10, color: 'var(--muted-foreground)' }}>
        k={data.kernel} ・ s={data.stride} ・ p={data.padding}
      </div>
      <div style={{ fontSize: 10, color: 'var(--muted-foreground)' }}>
        width: {data.inWidth} → {data.outWidth ?? '?'}
      </div>
      <Handle type="source" position={Position.Right} />
    </div>
  );
}
