// カスタムノード(root / category / paper / report)。
// LOD段階(overview/tile/full)で描画量を落とす(editorのdetail/lod.tsと同じ考え方)。

import { Handle, Position, type NodeProps, type Node } from '@xyflow/react';
import { useAppStore } from '../../store';
import type { CategoryNode, PaperNode, ReportNode } from '../../types';

export type TreeFlowNode = Node<{
  /** tree.jsonの元ノード(不変) */
  raw: CategoryNode | PaperNode | ReportNode | { id: string; title: string; kind: 'root' };
  childCount?: number;
  collapsed?: boolean;
}>;

const HANDLE_STYLE = { opacity: 0, width: 1, height: 1, border: 'none' } as const;

/** 入出力ハンドル(見た目は消し、エッジの接続点としてのみ使う) */
function Handles() {
  return (
    <>
      <Handle type="target" position={Position.Left} style={HANDLE_STYLE} isConnectable={false} />
      <Handle type="source" position={Position.Right} style={HANDLE_STYLE} isConnectable={false} />
    </>
  );
}

function useLod() {
  return useAppStore((s) => s.lod);
}

const baseCard =
  'rounded-lg border shadow-sm transition-colors select-none box-border h-full w-full';

export function RootFlowNode({ data, selected }: NodeProps<TreeFlowNode>) {
  return (
    <div
      className={`${baseCard} flex items-center justify-center px-3 py-2 font-bold`}
      style={{
        width: 160,
        height: 56,
        background: 'var(--node-bg)',
        borderColor: selected ? 'var(--ring)' : 'var(--accent-root)',
        color: 'var(--accent-root)',
        borderWidth: 2,
      }}
      data-testid="node-root"
    >
      <Handles />
      {data.raw.title}
    </div>
  );
}

export function CategoryFlowNode({ id, data, selected }: NodeProps<TreeFlowNode>) {
  const lod = useLod();
  const toggleCollapse = useAppStore((s) => s.toggleCollapse);
  const collapsed = data.collapsed ?? false;
  const count = data.childCount ?? 0;
  return (
    <div
      className={`${baseCard} flex items-center gap-2 px-3 py-2`}
      style={{
        width: 200,
        height: 56,
        background: 'var(--node-bg)',
        borderColor: selected ? 'var(--ring)' : 'var(--accent-category)',
        color: 'var(--foreground)',
        borderWidth: 2,
      }}
      data-testid="node-category"
      data-node-id={id}
    >
      <Handles />
      <button
        type="button"
        aria-label={collapsed ? 'カテゴリを開く' : 'カテゴリを折りたたむ'}
        title={collapsed ? '開く' : '折りたたむ'}
        className="nodrag shrink-0 rounded px-1 text-xs leading-none"
        style={{ border: '1px solid var(--border)', color: 'var(--muted-foreground)' }}
        onClick={(e) => {
          e.stopPropagation();
          toggleCollapse(id);
        }}
        data-testid="collapse-toggle"
      >
        {collapsed ? '+' : '−'}
      </button>
      <div className="min-w-0 flex-1">
        <div className="truncate font-semibold" style={{ color: 'var(--accent-category)' }}>
          {data.raw.title}
        </div>
        {lod !== 'overview' && (
          <div className="text-[11px]" style={{ color: 'var(--muted-foreground)' }}>
            {count} 件{collapsed ? '(折りたたみ中)' : ''}
          </div>
        )}
      </div>
    </div>
  );
}

export function PaperFlowNode({ id, data, selected }: NodeProps<TreeFlowNode>) {
  const lod = useLod();
  const paper = data.raw as PaperNode;
  const stub = paper.status === 'stub';
  return (
    <div
      className={`${baseCard} flex flex-col justify-center gap-1 px-3 py-2`}
      style={{
        width: 240,
        height: 84,
        background: stub ? 'var(--node-stub-bg)' : 'var(--node-bg)',
        borderColor: selected ? 'var(--ring)' : stub ? 'var(--border)' : 'var(--accent-paper)',
        color: stub ? 'var(--node-stub-fg)' : 'var(--foreground)',
        borderWidth: selected ? 2 : 1,
        borderStyle: stub ? 'dashed' : 'solid',
      }}
      data-testid="node-paper"
      data-status={paper.status}
      data-node-id={id}
      title={paper.title}
    >
      <Handles />
      {lod === 'overview' ? (
        <div className="truncate text-[15px] font-semibold">{paper.title}</div>
      ) : (
        <>
          <div className="line-clamp-2 text-[12px] leading-snug font-semibold">{paper.title}</div>
          {lod === 'full' && (
            <div
              className="flex items-center gap-2 text-[10px]"
              style={{ color: 'var(--muted-foreground)' }}
            >
              <span>{paper.date?.slice(0, 4) ?? '----'}</span>
              <span className="truncate">{paper.categories.join(', ')}</span>
              {stub && <span className="ml-auto shrink-0">stub</span>}
            </div>
          )}
        </>
      )}
    </div>
  );
}

export function ReportFlowNode({ id, data, selected }: NodeProps<TreeFlowNode>) {
  const lod = useLod();
  const report = data.raw as ReportNode;
  return (
    <div
      className={`${baseCard} flex flex-col justify-center gap-1 px-3 py-2`}
      style={{
        width: 240,
        height: 68,
        background: 'var(--node-bg)',
        borderColor: selected ? 'var(--ring)' : 'var(--accent-report)',
        color: 'var(--foreground)',
        borderWidth: selected ? 2 : 1,
      }}
      data-testid="node-report"
      data-node-id={id}
      title={report.title}
    >
      <Handles />
      <div className="line-clamp-2 text-[12px] leading-snug font-semibold">{report.title}</div>
      {lod === 'full' && (
        <div className="truncate text-[10px]" style={{ color: 'var(--muted-foreground)' }}>
          {report.branch ?? 'branch不明'} / {report.type}
        </div>
      )}
    </div>
  );
}

export const NODE_TYPES = {
  root: RootFlowNode,
  category: CategoryFlowNode,
  paper: PaperFlowNode,
  report: ReportFlowNode,
};
