// 折りたたみサブサーキットの合成ノード。表示専用で、IRのノードではない。
// 外部入出力だけを単一ハンドルとして見せ、操作(展開/解除/名前変更/テンプレート保存)を提供する。

import { Handle, Position, type NodeProps } from '@xyflow/react';
import { Boxes, Save, Ungroup } from 'lucide-react';
import { toast } from 'sonner';
import type { GroupCollapsedNode } from '../grouping';
import { useGraphStore } from '../../store/graphStore';

export function GroupNode({ data }: NodeProps<GroupCollapsedNode>) {
  const toggleGroupCollapse = useGraphStore((s) => s.toggleGroupCollapse);
  const ungroup = useGraphStore((s) => s.ungroup);
  const renameGroup = useGraphStore((s) => s.renameGroup);
  const saveGroupAsTemplate = useGraphStore((s) => s.saveGroupAsTemplate);

  const onRename = () => {
    const name = window.prompt('サブサーキット名', data.name);
    if (name !== null) renameGroup(data.groupId, name);
  };
  const onSave = () => {
    const name = window.prompt('マイブロック名', data.name);
    if (name === null) return;
    saveGroupAsTemplate(data.groupId, name);
    toast.success('マイブロックに保存しました(パレットから再配置できます)');
  };

  return (
    <div className="editor-node group-node" data-slot="group-node">
      <Handle type="target" position={Position.Left} />
      <div className="group-node-head">
        <Boxes className="size-3.5 shrink-0" />
        <button
          type="button"
          className="group-node-name"
          title="ダブルクリックで名前変更"
          onDoubleClick={onRename}
        >
          {data.name}
        </button>
      </div>
      <div className="node-sub">
        {data.memberCount}ノード ・ in {data.inputCount} / out {data.outputCount}
      </div>
      <div className="group-node-actions">
        <button
          type="button"
          data-slot="group-expand"
          onClick={() => toggleGroupCollapse(data.groupId)}
        >
          展開
        </button>
        <button type="button" title="テンプレート保存" onClick={onSave}>
          <Save className="size-3" />
        </button>
        <button type="button" title="グループ解除" onClick={() => ungroup(data.groupId)}>
          <Ungroup className="size-3" />
        </button>
      </div>
      <Handle type="source" position={Position.Right} />
    </div>
  );
}
