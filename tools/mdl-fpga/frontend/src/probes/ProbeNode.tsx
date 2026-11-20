// キャンバス上のプローブ(React Flow カスタムノード)。
// 回路図と一緒に拡大縮小/移動する「あたかも回路の一部」の値表示部。
import { Handle, Position, type NodeProps } from '@xyflow/react';
import { ProbeCard } from './ProbeCard';

export function ProbeNode({ data }: NodeProps) {
  const probeId = (data as { probeId: string }).probeId;
  return (
    <div className="probe-node nodrag-inputs">
      <Handle type="target" position={Position.Left} className="probe-handle" />
      <ProbeCard probeId={probeId} />
    </div>
  );
}
