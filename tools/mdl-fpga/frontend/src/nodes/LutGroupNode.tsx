// 展開された lut_layer のグループ枠(子ゲートは expandGraph が配置)。
import { Handle, Position, type NodeProps } from '@xyflow/react';
import { Button } from '@/components/ui/button';
import type { BlockData } from '../graph/metaToFlow';
import { useT } from '../i18n/useT';
import { useViewerStore } from '../store/viewerStore';

export function LutGroupNode({ data }: NodeProps) {
  const t = useT();
  const d = data as BlockData;
  const toggleExpanded = useViewerStore((s) => s.toggleExpanded);
  return (
    <div className="lut-group">
      <div className="lut-group-head">
        <span className="node-title">{d.nodeId}</span>
        <span className="node-sub">LUT layer · {d.width} gates</span>
        <Button size="xs" variant="ghost" className="h-5 px-1.5 text-[10px]"
                onClick={() => toggleExpanded(d.nodeId)}>
          {t('probe.collapse')}
        </Button>
      </div>
      <Handle type="source" position={Position.Right} />
    </div>
  );
}
