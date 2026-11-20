// その他ノード(binarize/flatten/blackbox)と入出力ポートの汎用表示

import { Handle, Position, type NodeProps } from '@xyflow/react';
import type { GenericFlowNode } from '../../ir/irToGraph';
import { nodeInputKind, nodeOutputKind, realHandleStyle } from '../edgeStyle';

export function GenericNode({ data, selected }: NodeProps<GenericFlowNode>) {
  const isInput = data.kind === 'input';
  const isOutput = data.kind === 'output';
  const isConstant = data.kind === 'constant';
  // §5 信号種でハンドル(ポート)を色分け: blackbox=実数域、binarize入力=ToLogic境界
  const inStyle =
    nodeInputKind(data.kind) === 'real' ? realHandleStyle() : undefined;
  const outStyle =
    nodeOutputKind(data.kind) === 'real' ? realHandleStyle() : undefined;
  return (
    <div
      className={`editor-node generic-node kind-${data.kind}${selected ? ' selected' : ''}`}
    >
      {!isInput && !isConstant && (
        <Handle type="target" position={Position.Left} style={inStyle} />
      )}
      <div className="node-title">{data.label}</div>
      <div className="node-sub">{data.subtitle}</div>
      {!isOutput && <Handle type="source" position={Position.Right} style={outStyle} />}
    </div>
  );
}
