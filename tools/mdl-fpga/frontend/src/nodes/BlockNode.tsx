// 回路ブロック(lowered IR ノード 1 個)。エディタの .editor-node 視覚契約に準拠。
// 値チップは RTL 実値(WS)を購読。
import { Handle, Position, type NodeProps } from '@xyflow/react';
import { Button } from '@/components/ui/button';
import { EXPAND_MAX_WIDTH } from '../graph/expandGraph';
import type { BlockData } from '../graph/metaToFlow';
import { formatHex } from '../graph/hexBits';
import { useT } from '../i18n/useT';
import { useDetailStore } from '../store/detailStore';
import { useSimStore } from '../store/simStore';
import { useViewerStore } from '../store/viewerStore';

const TYPE_LABEL: Record<string, string> = {
  lut_layer: 'LUT layer',
  group_sum: 'GroupSum',
  concat: 'Concat',
  add: 'Add',
  constant: 'Const',
  flatten: 'Flatten',
  binarize: 'Binarize',
  system: 'System',
};

const TYPE_CLASS: Record<string, string> = {
  lut_layer: 'lut-node',
  group_sum: 'groupsum-node',
  system: 'system-node',
};

export function BlockNode({ data, selected }: NodeProps) {
  const t = useT();
  const d = data as BlockData;
  const hex = useSimStore((s) => s.frame?.values[d.nodeId]);
  const hasXz = useSimStore(
    (s) => (s.frame?.xz ?? []).includes(d.nodeId),
  );
  const toggleExpanded = useViewerStore((s) => s.toggleExpanded);
  const fetchParams = useDetailStore((s) => s.fetchParams);
  const expandable =
    d.nodeType === 'lut_layer' && d.width <= EXPAND_MAX_WIDTH;

  const valueText = formatHex(hex, d.width);
  const shown =
    valueText.length > 18 ? valueText.slice(0, 16) + '…' : valueText;
  const slots = Array.from({ length: d.inSlots }, (_, i) => i);
  const cls = TYPE_CLASS[d.nodeType] ?? 'generic-node';

  return (
    <div className={`editor-node ${cls}${selected ? ' selected' : ''}`}>
      {slots.map((s) => (
        <Handle
          key={s}
          type="target"
          position={Position.Left}
          id={d.inSlots > 1 ? `s${s}` : undefined}
          style={
            d.inSlots > 1
              ? { top: `${((s + 1) / (d.inSlots + 1)) * 100}%` }
              : undefined
          }
        />
      ))}
      <div className="node-title">
        <span>{d.label ?? d.nodeId}</span>
        {hasXz && <span className="xz-badge" title="X/Z を含む">X</span>}
        {expandable && (
          <Button
            size="xs"
            variant="ghost"
            className="h-5 px-1.5 text-[10px]"
            title="ゲート単位に展開"
            onClick={() => {
              void fetchParams(d.nodeId);
              toggleExpanded(d.nodeId);
            }}
          >
            {t('probe.expand')}
          </Button>
        )}
        {d.nodeType === 'lut_layer' && !expandable && (
          <span
            className="node-sub"
            title={`大規模層(${d.width}ゲート > ${EXPAND_MAX_WIDTH})は展開不可。プローブで値を観測してください`}
          >
            ⊞
          </span>
        )}
      </div>
      <div className="node-sub">
        {TYPE_LABEL[d.nodeType] ?? d.nodeType} · {d.width}bit
        {d.alias ? ' · alias' : ''}
        {d.scoreBits ? ` · score${d.scoreBits}bit` : ''}
      </div>
      <div className="node-value" title={valueText}>
        {shown}
      </div>
      <Handle type="source" position={Position.Right} />
    </div>
  );
}
