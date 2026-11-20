// 展開時の個別ゲート(LUT 1 個)。出力 bit の実値で点灯する。
import { Handle, Position, type NodeProps } from '@xyflow/react';
import type { GateData } from '../graph/expandGraph';
import { gateLabel } from '../graph/gateNames';
import { bitOf } from '../graph/hexBits';
import { useSimStore } from '../store/simStore';

export function GateNode({ data }: NodeProps) {
  const d = data as GateData;
  const out = useSimStore((s) =>
    s.frame ? bitOf(s.frame.values[d.layerId], d.gateIndex) : 0,
  );
  const inBits = useSimStore((s) =>
    s.frame
      ? d.srcBits.map((b) => bitOf(s.frame!.values[d.srcId], b)).join('')
      : '?',
  );
  // ゲート色はエディタの --gate-0..15 トークン(N=2 は tt 値がそのままゲートID)
  const accent = d.nInputs === 2 && d.tt >= 0 && d.tt <= 15
    ? `var(--gate-${d.tt})`
    : 'var(--edge-default)';
  return (
    <div
      className={`gate-node${out ? ' on' : ''}`}
      style={{ borderLeftColor: accent }}
      title={`gate ${d.gateIndex}: 入力 bit ${d.srcBits.join(',')} = ${inBits} → 出力 ${out}`}
    >
      <Handle type="target" position={Position.Left} />
      <div className="gate-label">{gateLabel(d.tt, d.nInputs)}</div>
      <div className="gate-out">{out}</div>
      <Handle type="source" position={Position.Right} />
    </div>
  );
}
