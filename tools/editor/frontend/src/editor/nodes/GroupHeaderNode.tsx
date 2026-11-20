// 展開中サブサーキットのヘッダノード。メンバー外接矩形の左上へ浮かべ、
// 折りたたみ/解除の操作を提供する。表示専用で、ドラッグ・選択・削除は不可。

import { Boxes, Minimize2, Ungroup } from 'lucide-react';
import type { NodeProps } from '@xyflow/react';
import type { GroupHeaderNode as GroupHeaderNodeType } from '../grouping';
import { useGraphStore } from '../../store/graphStore';

export function GroupHeaderNode({ data }: NodeProps<GroupHeaderNodeType>) {
  const toggleGroupCollapse = useGraphStore((s) => s.toggleGroupCollapse);
  const ungroup = useGraphStore((s) => s.ungroup);

  return (
    <div className="group-header-node" data-slot="group-header">
      <Boxes className="size-3.5 shrink-0" />
      <span className="group-header-name">{data.name}</span>
      <span className="group-header-count">({data.memberCount})</span>
      <button
        type="button"
        data-slot="group-collapse"
        title="折りたたむ"
        onClick={() => toggleGroupCollapse(data.groupId)}
      >
        <Minimize2 className="size-3" />
      </button>
      <button type="button" title="グループ解除" onClick={() => ungroup(data.groupId)}>
        <Ungroup className="size-3" />
      </button>
    </div>
  );
}
