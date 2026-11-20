// 入出力ポート(x / y)。エディタの .editor-node + kind-input/kind-output 契約。
// 出力ポートにも source ハンドルを置き、そこから引き出すと y_comb プローブになる。
import { Handle, Position, type NodeProps } from '@xyflow/react';
import type { PortData } from '../graph/metaToFlow';
import { formatHex } from '../graph/hexBits';
import { useT } from '../i18n/useT';
import { useSimStore } from '../store/simStore';

export function PortNode({ data, selected }: NodeProps) {
  const t = useT();
  const d = data as PortData;
  const hex = useSimStore((s) => s.frame?.values[d.portId]);
  const kind = d.direction === 'in' ? 'kind-input' : 'kind-output';
  return (
    <div className={`editor-node generic-node ${kind}${selected ? ' selected' : ''}`}>
      {d.direction === 'out' && <Handle type="target" position={Position.Left} />}
      <div className="node-title">
        {d.direction === 'in' ? t('input.title') : '出力'} {d.portId}
      </div>
      <div className="node-sub">{d.width}bit</div>
      <div className="node-value">{formatHex(hex, d.width)}</div>
      <Handle type="source" position={Position.Right} />
    </div>
  );
}
