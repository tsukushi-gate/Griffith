// concatノード: 複数入力をビット連結する。slot 0が出力ビット列の先頭。
// 入力ハンドルは接続数+1個を縦に動的表示する。

import { Handle, Position, type NodeProps } from '@xyflow/react';
import type { ConcatFlowNode } from '../../ir/irToGraph';
import { SlotHandles } from './SlotHandles';
import { useInputSlots } from './useInputSlots';

export function ConcatNode({ id, data, selected }: NodeProps<ConcatFlowNode>) {
  const slots = useInputSlots(id);

  return (
    <div
      className={`editor-node concat-node${selected ? ' selected' : ''}`}
      style={{ borderLeft: '3px solid var(--slot-1)' }}
    >
      <div className="node-title">{data.label}</div>
      <div className="node-sub">concat ・ →{data.outWidth}</div>
      <SlotHandles
        slots={slots}
        slotLabel={(slot) =>
          data.inWidths[slot] !== undefined ? `w=${data.inWidths[slot]}` : '(空き)'
        }
      />
      <Handle type="source" position={Position.Right} />
    </div>
  );
}
