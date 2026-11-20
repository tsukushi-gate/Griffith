// 右パネル: 選択ノードの落合式Markdownと外部リンク(arXiv/PDF/レポート)を表示する。

import ReactMarkdown from 'react-markdown';
import rehypeKatex from 'rehype-katex';
import remarkGfm from 'remark-gfm';
import remarkMath from 'remark-math';
import 'katex/dist/katex.min.css';
import { getNode, useAppStore } from '../store';
import { isPaper, isReport, type PaperNode, type ReportNode, type TreeNode } from '../types';

/** リポジトリ外に置くPDFはパスだけを持つ。GitHub等に置かないのでパス表示に留める。*/
function PdfRef({ path }: { path: string }) {
  return (
    <div className="text-[11px] break-all" style={{ color: 'var(--muted-foreground)' }}>
      PDF: <code>{path}</code>
    </div>
  );
}

function LinkChip({ href, children }: { href: string; children: React.ReactNode }) {
  return (
    <a
      href={href}
      target="_blank"
      rel="noreferrer"
      className="rounded border px-2 py-0.5 text-[11px] hover:opacity-80"
      style={{ borderColor: 'var(--border)', color: 'var(--primary)' }}
    >
      {children}
    </a>
  );
}

function NodeChip({ id }: { id: string }) {
  const focusNode = useAppStore((s) => s.focusNode);
  const node = getNode(id);
  return (
    <button
      type="button"
      onClick={() => focusNode(id)}
      className="max-w-full truncate rounded border px-2 py-0.5 text-left text-[11px] hover:opacity-80"
      style={{ borderColor: 'var(--border)' }}
      data-testid="related-chip"
      title={node?.title ?? id}
    >
      {node?.title ?? id}
    </button>
  );
}

function Section({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="mt-3">
      <div
        className="mb-1 text-[10px] font-semibold tracking-wide uppercase"
        style={{ color: 'var(--muted-foreground)' }}
      >
        {label}
      </div>
      <div className="flex flex-wrap gap-1">{children}</div>
    </div>
  );
}

function PaperDetail({ node }: { node: PaperNode }) {
  return (
    <>
      <div className="flex flex-wrap items-center gap-2">
        <span
          className="rounded px-1.5 py-0.5 text-[10px] font-semibold"
          style={{
            background: node.status === 'stub' ? 'var(--muted)' : 'var(--accent-paper)',
            color: node.status === 'stub' ? 'var(--muted-foreground)' : 'var(--primary-foreground)',
          }}
          data-testid="panel-status"
        >
          {node.status}
        </span>
        <span className="text-[11px]" style={{ color: 'var(--muted-foreground)' }}>
          {node.date ?? '日付不明'} / {node.categories.join(', ')}
        </span>
      </div>
      <div className="mt-2 flex flex-wrap gap-1">
        {node.arxiv && <LinkChip href={`https://arxiv.org/abs/${node.arxiv}`}>arXiv</LinkChip>}
        {node.url && <LinkChip href={node.url}>原文リンク</LinkChip>}
      </div>
      {node.pdf && <div className="mt-1">{<PdfRef path={node.pdf} />}</div>}
      {node.next.length > 0 && (
        <Section label="次に読むべき論文">
          {node.next.map((id) => (
            <NodeChip key={id} id={id} />
          ))}
        </Section>
      )}
      {node.reports.length > 0 && (
        <Section label="関連するレポート">
          {node.reports.map((id) => (
            <NodeChip key={id} id={id} />
          ))}
        </Section>
      )}
      <hr className="my-3" style={{ borderColor: 'var(--border)' }} />
      <div className="md-body" data-testid="panel-markdown">
        <ReactMarkdown remarkPlugins={[remarkGfm, remarkMath]} rehypePlugins={[rehypeKatex]}>{node.body}</ReactMarkdown>
      </div>
    </>
  );
}

function ReportDetail({ node }: { node: ReportNode }) {
  return (
    <>
      <div className="text-[11px]" style={{ color: 'var(--muted-foreground)' }}>
        {node.date ?? '日付不明'} / branch: {node.branch ?? '不明'} / type: {node.type}
      </div>
      {node.path && (
        <div className="mt-1 text-[11px] break-all" style={{ color: 'var(--muted-foreground)' }}>
          <code>{node.path}</code>
        </div>
      )}
      {node.artifactUrl && (
        <div className="mt-2">
          <LinkChip href={node.artifactUrl}>公開Artifact</LinkChip>
        </div>
      )}
      {node.papers.length > 0 && (
        <Section label="関連論文">
          {node.papers.map((id) => (
            <NodeChip key={id} id={id} />
          ))}
        </Section>
      )}
      <hr className="my-3" style={{ borderColor: 'var(--border)' }} />
      <div className="md-body" data-testid="panel-markdown">
        <ReactMarkdown remarkPlugins={[remarkGfm, remarkMath]} rehypePlugins={[rehypeKatex]}>{node.body}</ReactMarkdown>
      </div>
    </>
  );
}

function CategoryDetail({ node }: { node: TreeNode }) {
  return (
    <div className="text-[12px]" style={{ color: 'var(--muted-foreground)' }}>
      カテゴリノード。ノード左の「−/+」で配下の折りたたみを切り替えられる。
      <div className="mt-1">
        id: <code>{node.id}</code>
      </div>
    </div>
  );
}

export function DetailPanel() {
  const selectedId = useAppStore((s) => s.selectedId);
  const node = getNode(selectedId);

  return (
    <aside
      className="flex h-full w-[420px] shrink-0 flex-col overflow-hidden border-l"
      style={{ borderColor: 'var(--border)', background: 'var(--panel-bg)' }}
      data-testid="detail-panel"
    >
      {!node ? (
        <div className="p-4 text-[12px]" style={{ color: 'var(--muted-foreground)' }}>
          ノードをクリックすると、ここに落合式要約が表示される。
          <div className="mt-2">
            検索は <kbd>⌘K</kbd> / <kbd>Ctrl+K</kbd>。
          </div>
        </div>
      ) : (
        <div className="overflow-y-auto p-4">
          <h2 className="text-[15px] leading-snug font-bold" data-testid="panel-title">
            {node.title}
          </h2>
          <div className="mt-0.5 text-[10px]" style={{ color: 'var(--muted-foreground)' }}>
            {node.id}
          </div>
          {isPaper(node) ? (
            <PaperDetail node={node} />
          ) : isReport(node) ? (
            <ReportDetail node={node} />
          ) : (
            <CategoryDetail node={node} />
          )}
        </div>
      )}
    </aside>
  );
}
