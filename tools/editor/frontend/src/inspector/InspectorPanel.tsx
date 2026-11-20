// インスペクタ(右サイドバー): 診断一覧+選択ノード情報+Accordionセクション
// (パラメータ/ヒートマップ/真理値表/配線)。同期バッジはヘッダへ移設済み。

import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from '@/components/ui/accordion';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { nodeCapabilities } from '../ir/schema';
import { commentsRecordOf, useGraphStore } from '../store/graphStore';
import { GateDistPanel } from './GateDistPanel';
import { HierarchyPanel } from './HierarchyPanel';
import { IndicesEditor } from './IndicesEditor';
import { LutHeatmap } from './LutHeatmap';
import { NodeParamForm } from './NodeParamForm';
import { StatsPanel } from './StatsPanel';
import { TruthTableEditor } from './TruthTableEditor';

/** レイヤーコメント編集: metadata.editor.comments へ非破壊で永続化する。*/
function CommentEditor({ nodeId }: { nodeId: string }) {
  const ir = useGraphStore((s) => s.ir);
  const setComment = useGraphStore((s) => s.setComment);
  const value = ir ? (commentsRecordOf(ir)[nodeId] ?? '') : '';
  return (
    <div className="mt-2">
      <label className="text-[11px] font-medium text-muted-foreground" htmlFor="node-comment">
        コメント
      </label>
      <textarea
        id="node-comment"
        data-testid="node-comment-input"
        className="mt-1 w-full resize-none rounded border bg-transparent px-2 py-1 text-xs"
        rows={2}
        placeholder="このレイヤーへのメモ(カードに付箋表示)"
        value={value}
        onChange={(e) => setComment(nodeId, e.target.value)}
      />
    </div>
  );
}

export function InspectorPanel() {
  const selectedId = useGraphStore((s) => s.selectedNodeId);
  const ir = useGraphStore((s) => s.ir);
  const diagnostics = useGraphStore((s) => s.diagnostics);
  const syncError = useGraphStore((s) => s.syncError);

  if (!ir) {
    return (
      <div className="flex h-full flex-col gap-2 p-3">
        <h2 className="text-[11px] font-semibold tracking-widest text-muted-foreground uppercase">
          インスペクタ
        </h2>
        <div className="hint-text">モデルを読み込むとグラフが表示されます</div>
      </div>
    );
  }

  const node = ir.nodes.find((n) => n.id === selectedId);
  const outPort = ir.outputs.find((p) => p.id === selectedId);
  const inPort = ir.inputs.find((p) => p.id === selectedId);
  const isLut = node?.type === 'lut_layer';
  const caps = node ? nodeCapabilities(node) : null;
  const capBadges = caps
    ? [
        ['Python', caps.python_executable],
        ['Train', caps.trainable],
        ['IR build', caps.ir_buildable],
        ['Gate', caps.gate_expandable],
        ['HDL', caps.hdl_exportable],
      ] as const
    : [];

  return (
    <div className="flex h-full flex-col gap-1 overflow-y-auto p-3">
      <h2 className="text-[11px] font-semibold tracking-widest text-muted-foreground uppercase">
        インスペクタ
      </h2>
      {syncError && <div className="error-text">{syncError}</div>}
      {diagnostics.length > 0 && (
        <ul className="diagnostics">
          {diagnostics.map((d, i) => (
            <li key={i} className={`diag-${d.level}`}>
              {d.level === 'error' ? '✖' : '△'} {d.message}
            </li>
          ))}
        </ul>
      )}
      {/* 統計パネル: グラフ全体の指標を常設表示。選択ノード型の寄与をハイライト */}
      <section className="rounded-md border p-2">
        <h3 className="mb-1 text-[10px] font-semibold tracking-widest text-muted-foreground uppercase">
          統計
        </h3>
        <StatsPanel />
      </section>
      <Tabs defaultValue="node" className="min-h-0 flex-1">
        <TabsList variant="line" className="h-8 rounded-none bg-transparent p-0">
          <TabsTrigger value="node" className="h-7 px-2 text-xs">
            ノード
          </TabsTrigger>
          <TabsTrigger value="hierarchy" className="h-7 px-2 text-xs">
            階層
          </TabsTrigger>
        </TabsList>
        <TabsContent value="node" className="mt-2 space-y-2">
          {!node && !inPort && !outPort && (
            <div className="hint-text">ノードを選択してください</div>
          )}
          {outPort && (
            <dl className="param-list">
              <dt>id</dt>
              <dd>{outPort.id}</dd>
              <dt>width(導出)</dt>
              <dd>{outPort.width}</dd>
              {outPort.semantics && (
                <>
                  <dt>semantics</dt>
                  <dd>{outPort.semantics}</dd>
                </>
              )}
            </dl>
          )}
          {inPort && (
            <dl className="param-list">
              <dt>id</dt>
              <dd>{inPort.id}</dd>
              {inPort.encoding && (
                <>
                  <dt>encoding</dt>
                  <dd>{inPort.encoding}</dd>
                </>
              )}
            </dl>
          )}
          {node && (
            <dl className="param-list">
              <dt>id</dt>
              <dd>{node.id}</dd>
              <dt>type</dt>
              <dd>{node.type}</dd>
              {capBadges.length > 0 && (
                <>
                  <dt>capabilities</dt>
                  <dd className="flex flex-wrap gap-1">
                    {capBadges.map(([label, enabled]) => (
                      <Badge
                        key={label}
                        variant={enabled ? 'secondary' : 'outline'}
                        className="rounded-md text-[10px]"
                      >
                        {enabled ? label : `${label}: no`}
                      </Badge>
                    ))}
                  </dd>
                </>
              )}
            </dl>
          )}
          {node && <CommentEditor nodeId={node.id} />}
          {(node || inPort) && (
            <Accordion
              type="multiple"
              defaultValue={['params', 'heatmap']}
              className="w-full"
            >
              <AccordionItem value="params">
                <AccordionTrigger className="py-2 text-xs">パラメータ</AccordionTrigger>
                <AccordionContent>
                  <NodeParamForm />
                </AccordionContent>
              </AccordionItem>
              {isLut && node && (
                <>
                  <AccordionItem value="heatmap">
                    <AccordionTrigger className="py-2 text-xs">
                      ゲートヒートマップ
                    </AccordionTrigger>
                    <AccordionContent>
                      <LutHeatmap nodeId={node.id} />
                    </AccordionContent>
                  </AccordionItem>
                  <AccordionItem value="dist">
                    <AccordionTrigger className="py-2 text-xs">
                      ゲート確率分布
                    </AccordionTrigger>
                    <AccordionContent>
                      <GateDistPanel nodeId={node.id} />
                    </AccordionContent>
                  </AccordionItem>
                  <AccordionItem value="tt">
                    <AccordionTrigger className="py-2 text-xs">真理値表</AccordionTrigger>
                    <AccordionContent>
                      <TruthTableEditor nodeId={node.id} />
                    </AccordionContent>
                  </AccordionItem>
                  <AccordionItem value="indices">
                    <AccordionTrigger className="py-2 text-xs">配線 (indices)</AccordionTrigger>
                    <AccordionContent>
                      <IndicesEditor nodeId={node.id} />
                    </AccordionContent>
                  </AccordionItem>
                </>
              )}
            </Accordion>
          )}
        </TabsContent>
        <TabsContent value="hierarchy" className="mt-2">
          <HierarchyPanel selectedNode={node} />
        </TabsContent>
      </Tabs>
    </div>
  );
}
