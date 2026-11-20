// addノード: 複数入力のビット単位マージ(or/xor)または整数和(sum)。
// 全slot同幅。modeセレクタをノード内に持ち、slot 1以降の入力エッジは
// 残差スタイル(破線)で描画される。

import { Handle, Position, type NodeProps } from '@xyflow/react';
import type { AddFlowNode } from '../../ir/irToGraph';
import { ADD_MODES, type AddMode } from '../../ir/schema';
import { useGraphStore } from '../../store/graphStore';
import { nodeOutputKind, realHandleStyle } from '../edgeStyle';
import { SlotHandles } from './SlotHandles';
import { useInputSlots } from './useInputSlots';

export function AddNode({ id, data, selected }: NodeProps<AddFlowNode>) {
  const slots = useInputSlots(id);
  const updateNodeData = useGraphStore((s) => s.updateNodeData);
  // §5 sumモードの出力は実数域(HDL不可)。出力ハンドルを実数域色にする
  const outStyle =
    nodeOutputKind('add', data.mode) === 'real' ? realHandleStyle() : undefined;

  return (
    <div
      className={`editor-node add-node${selected ? ' selected' : ''}`}
      style={{ borderLeft: '3px solid var(--slot-2)' }}
    >
      <div className="node-title">{data.label}</div>
      <div className="node-sub">
        add ・ width={data.width}
        <select
          className="nodrag"
          value={data.mode}
          title="マージ演算(sumはHDL生成不可)"
          style={{ marginLeft: 6, fontSize: 10 }}
          onClick={(e) => e.stopPropagation()}
          onChange={(e) =>
            updateNodeData(id, { mode: e.target.value as AddMode })
          }
        >
          {ADD_MODES.map((m) => (
            <option key={m} value={m}>
              {m}
            </option>
          ))}
        </select>
      </div>
      <SlotHandles slots={slots} />
      <Handle type="source" position={Position.Right} style={outStyle} />
    </div>
  );
}
