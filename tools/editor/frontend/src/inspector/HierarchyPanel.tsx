import { useEffect, useMemo, useState } from 'react';
import { ChevronRight, ExternalLink, GitBranch, Layers3 } from 'lucide-react';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { getLowering, getModelHierarchy, type HierarchyNode, type LoweringView } from '../api/rest';
import type { IRNodeJson } from '../ir/schema';
import { useGraphStore } from '../store/graphStore';
import { useViewStore } from '../store/viewStore';

function hierarchyIdOf(node: IRNodeJson | undefined): string | null {
  const editor = node?.meta?.editor;
  if (editor && typeof editor === 'object' && 'hierarchy_id' in editor) {
    const value = (editor as { hierarchy_id?: unknown }).hierarchy_id;
    return typeof value === 'string' ? value : null;
  }
  return node?.id ?? null;
}

function shapeText(shape: Record<string, unknown> | null | undefined): string | null {
  if (!shape) return null;
  return Object.entries(shape)
    .map(([k, v]) => `${k}:${Array.isArray(v) ? `[${v.join(',')}]` : String(v)}`)
    .join(' / ');
}

function capabilityBadges(caps: Record<string, boolean> | null | undefined) {
  if (!caps) return null;
  const labels: [string, string][] = [
    ['python_executable', 'Python'],
    ['trainable', 'Train'],
    ['ir_buildable', 'IR'],
    ['gate_expandable', 'Gate'],
    ['hdl_exportable', 'HDL'],
  ];
  return (
    <div className="flex flex-wrap gap-1">
      {labels.map(([key, label]) => {
        const enabled = Boolean(caps[key]);
        return (
          <Badge key={key} variant={enabled ? 'secondary' : 'outline'} className="rounded-md text-[10px]">
            {enabled ? label : `${label}: no`}
          </Badge>
        );
      })}
    </div>
  );
}

export function HierarchyPanel({ selectedNode }: { selectedNode?: IRNodeJson }) {
  const ir = useGraphStore((s) => s.ir);
  const setView = useViewStore((s) => s.setView);
  const selectedHierarchyId = hierarchyIdOf(selectedNode);
  const [nodes, setNodes] = useState<HierarchyNode[]>([]);
  const [paperRef, setPaperRef] = useState<Record<string, unknown> | null>(null);
  const [focusId, setFocusId] = useState<string | null>(selectedHierarchyId);
  const [lowering, setLowering] = useState<LoweringView | null>(null);
  const [error, setError] = useState<string | null>(null);

  // irがnullの間はfetchしない(irが無いときはコンポーネントが早期returnで
  // ヒントを描画するため、nodes/paperRefのクリアは不要。irが戻れば再fetchされる)
  useEffect(() => {
    if (!ir) return;
    let alive = true;
    getModelHierarchy()
      .then((res) => {
        if (!alive) return;
        setNodes(res.nodes);
        setPaperRef(res.paper_ref ?? null);
        setError(null);
      })
      .catch((e) => {
        if (!alive) return;
        setNodes([]);
        setPaperRef(null);
        setError(e instanceof Error ? e.message : String(e));
      });
    return () => {
      alive = false;
    };
  }, [ir]);

  // 選択ノード(prop)の変化に追従してfocus/loweringをリセットする。
  // effectでのsetStateを避け、prop変化時のstate調整パターンで描画中に反映する
  const [prevSelected, setPrevSelected] = useState<string | null>(selectedHierarchyId);
  if (prevSelected !== selectedHierarchyId) {
    setPrevSelected(selectedHierarchyId);
    setFocusId(selectedHierarchyId);
    setLowering(null);
  }

  const byId = useMemo(() => new Map(nodes.map((n) => [n.id, n])), [nodes]);
  const roots = useMemo(() => nodes.filter((n) => !n.parent), [nodes]);
  const focus = focusId ? byId.get(focusId) : roots[0];
  const children = focus ? nodes.filter((n) => n.parent === focus.id) : roots;
  const breadcrumb = useMemo(() => {
    if (!focus) return [];
    const out: HierarchyNode[] = [];
    let cur: HierarchyNode | undefined = focus;
    while (cur) {
      out.unshift(cur);
      cur = cur.parent ? byId.get(cur.parent) : undefined;
    }
    return out;
  }, [focus, byId]);

  async function previewLowering(nodeId: string) {
    try {
      const view = await getLowering(nodeId, 'L0');
      setLowering(view);
      setError(null);
    } catch (e) {
      setLowering(null);
      setError(e instanceof Error ? e.message : String(e));
    }
  }

  if (!ir) return <div className="hint-text">モデルを読み込むと階層が表示されます</div>;
  if (error && nodes.length === 0) return <div className="error-text">{error}</div>;
  if (nodes.length === 0) {
    return <div className="hint-text">このモデルには階層メタデータがありません</div>;
  }

  const canOpenL0 = Boolean(focus?.capabilities?.gate_expandable && focus.kind === 'lut_layer');

  return (
    <div className="flex flex-col gap-3 text-xs">
      {paperRef && (
        <div className="rounded-md border bg-background/60 p-2">
          <div className="flex items-center gap-1 font-medium">
            <Layers3 className="size-3.5" />
            {String(paperRef.title ?? 'Paper config')}
          </div>
          <div className="mt-1 text-muted-foreground">
            baseline {String(paperRef.baseline ?? 'n/a')} / D=192 / encoder=12 / patch=16
          </div>
        </div>
      )}

      <div className="flex flex-wrap items-center gap-1 text-[11px] text-muted-foreground">
        {breadcrumb.map((n, i) => (
          <button
            key={n.id}
            type="button"
            className="inline-flex items-center gap-1 rounded-sm px-1.5 py-0.5 hover:bg-accent hover:text-foreground"
            onClick={() => {
              setFocusId(n.id);
              setLowering(null);
            }}
          >
            {i > 0 && <ChevronRight className="size-3" />}
            {n.label}
          </button>
        ))}
      </div>

      {focus && (
        <div className="space-y-2 rounded-md border bg-background p-2">
          <div className="flex items-start justify-between gap-2">
            <div>
              <div className="font-semibold">{focus.label}</div>
              <div className="text-[11px] text-muted-foreground">
                {focus.level} / {focus.kind}
              </div>
            </div>
            {canOpenL0 && (
              <Button
                size="xs"
                variant="secondary"
                onClick={() => setView({ kind: 'l0', nodeId: focus.id })}
              >
                <ExternalLink className="size-3" />
                L0で開く
              </Button>
            )}
          </div>
          {focus.detail && <div className="text-muted-foreground">{focus.detail}</div>}
          {focus.kind === 'll_channel_mixer' && (
            <div className="grid gap-1 rounded-md border bg-muted/30 p-2 text-[11px]">
              <div>元MLP: <code>GELU(zW1)W2</code></div>
              <div>LL置換: <code>thermometer -&gt; DWN/LUT -&gt; DWN/LUT -&gt; conditional sum</code></div>
            </div>
          )}
          {focus.kind === 'mha_token_mixer' && (
            <div className="rounded-md border bg-muted/30 p-2 text-[11px] text-muted-foreground">
              MHAは第1段階ではPython実行のみです。ゲート展開はChannel MixerのLUT層で行います。
            </div>
          )}
          {shapeText(focus.shape) && (
            <div className="break-words text-[11px] text-muted-foreground">
              {shapeText(focus.shape)}
            </div>
          )}
          {capabilityBadges(focus.capabilities)}
          {focus.paper_note && <div className="text-[11px] text-muted-foreground">{focus.paper_note}</div>}
          {canOpenL0 && (
            <Button size="xs" variant="outline" onClick={() => previewLowering(focus.id)}>
              <GitBranch className="size-3" />
              Lowering確認
            </Button>
          )}
        </div>
      )}

      {children.length > 0 && (
        <div className="space-y-1">
          <div className="text-[11px] font-medium text-muted-foreground">子要素</div>
          {children.map((child) => (
            <button
              key={child.id}
              type="button"
              className="flex w-full items-center justify-between gap-2 rounded-md border bg-background px-2 py-1.5 text-left hover:bg-accent"
              onClick={() => {
                setFocusId(child.id);
                setLowering(null);
              }}
            >
              <span>
                <span className="block font-medium">{child.label}</span>
                <span className="block text-[11px] text-muted-foreground">
                  {child.level} / {child.kind}
                </span>
              </span>
              <ChevronRight className="size-3.5 text-muted-foreground" />
            </button>
          ))}
        </div>
      )}

      {lowering && (
        <div className="rounded-md border bg-background p-2 text-[11px]">
          <div className="font-medium">{lowering.title}</div>
          <div className="mt-1 text-muted-foreground">
            {lowering.summary
              ? `L0 gate ${lowering.summary.gate_ids.length.toLocaleString()} / N=${lowering.summary.n_inputs}`
              : 'Lowering summary is not available'}
          </div>
        </div>
      )}
      {error && <div className="error-text">{error}</div>}
    </div>
  );
}
