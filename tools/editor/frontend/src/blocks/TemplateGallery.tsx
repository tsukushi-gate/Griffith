// テンプレートギャラリー: サンプルIR+複合ブロックのプリセットを
// 一覧から選んで新規モデルとして読み込む。ブロックプリセットは
// templates.tsの生成器を既定パラメータで単体IR化(templateToGraph)して投入する。

import { useEffect, useState } from 'react';
import { Boxes, FilePlus, Package } from 'lucide-react';
import type { LucideIcon } from 'lucide-react';
import { toast } from 'sonner';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog';
import { ScrollArea } from '@/components/ui/scroll-area';
import {
  createProject,
  getTemplates,
  instantiateTemplate,
  putCurrentIR,
  type TemplateSummary,
} from '../api/rest';
import type { IRGraphJson } from '../ir/schema';
import { loadSampleModel } from '../samples/loadSample';
import { SAMPLES, type SampleDef } from '../samples/sampleIr';
import { useNetworkStore } from '../store/networkStore';
import {
  BLOCK_DEFS,
  blockIoWidths,
  generateBlock,
  templateToGraph,
  type BlockParams,
} from './templates';
import { MYUNITS_CHANGED_EVENT, listMyUnits } from './myBlocks';
import { unitToStandaloneGraph, type EditorUnit } from './unitTypes';

/**
 * テンプレ/新規プロジェクトを生成した際に発火するイベント。
 * CONFIGタブ(ConfigTab)がプロジェクト一覧を再取得し、当該プロジェクトを選択する。
 * detail.id にサーバ保存されたプロジェクトIDを載せる。
 */
export const PROJECT_INSTANTIATED_EVENT = 'md-project-instantiated';

/** テンプレの検証水準バッジ(verification に対応)。*/
function verificationBadge(v: TemplateSummary['verification']): {
  label: string;
  className: string;
} {
  if (v === 'smoke')
    return { label: '学習検証', className: 'border-[var(--status-ok)]/40 text-[var(--status-ok)]' };
  if (v === 'transfer')
    return { label: '重み転写', className: 'border-primary/30 text-primary' };
  if (v === 'build')
    return { label: 'build検証', className: 'border-primary/30 text-primary' };
  return { label: 'validateのみ', className: 'border-muted-foreground/30 text-muted-foreground' };
}

/** 空プロジェクト用の最小スターターIR(x8→y8、ノードは空。ユーザーが組み始める起点)。 */
function emptyStarterGraph(): IRGraphJson {
  return {
    format: 'magicdelta-ir',
    version: '0.2.0',
    metadata: { name: 'empty' },
    inputs: [{ id: 'x', width: 8, shape: null, encoding: null, semantics: null }],
    outputs: [{ id: 'y', width: 8, shape: null, encoding: null, semantics: null }],
    nodes: [],
    edges: [],
  };
}

/** ブロックプリセット(既定パラメータ)。細かい調整はパレットのドラッグ配置で行う。
 *  label/icon 未指定時は BLOCK_DEFS から解決する(複合ブロックは自前で指定する)。 */
const BLOCK_PRESETS: { spec: BlockParams; desc: string; label?: string; icon?: LucideIcon }[] = [
  {
    spec: { kind: 'mlp', params: { inWidth: 8, hidden: [16], outWidth: 8 } },
    desc: '8→16→8のLutLayer連鎖(draft)',
  },
  {
    spec: { kind: 'residual', params: { width: 8, innerLayers: 1, merge: 'xor' } },
    desc: '幅8・内側1層・xorマージ',
  },
  {
    spec: { kind: 'classifier', params: { inWidth: 60, k: 10, tau: 1.0 } },
    desc: '60→GroupSum(k=10)',
  },
  {
    // LL-ViT複合(パッチ埋込+PE+Transformer×2+分類ヘッド)。
    // blackbox含むためvalidateのみ(学習はPython import経路)。
    spec: {
      kind: 'llvit',
      params: {
        seqLen: 4,
        dEmbed: 6,
        posBits: 2,
        depth: 2,
        nHeads: 2,
        patchBits: 12,
        numClasses: 10,
      },
    },
    desc: 'パッチ埋込→PE→Transformer×2→分類(blackbox=validateのみ)',
    label: 'LL-ViT複合',
    icon: Boxes,
  },
];

const SAMPLE_GROUPS: { key: SampleDef['category']; label: string; desc: string }[] = [
  { key: 'hybrid', label: 'ハイブリッド', desc: 'Python由来の実行可能モデル' },
  { key: 'learning', label: '学習デモ', desc: 'draft層から論理を獲得するサンプル' },
  { key: 'sequence', label: '系列・状態', desc: 'RNN風の状態更新をLUTで可視化' },
  { key: 'logic', label: '参照回路', desc: '正解構造を分解して確認する小回路' },
  { key: 'dag', label: 'DAG', desc: 'fanout / skip / add を含む構造' },
  { key: 'vision', label: '画像', desc: 'MNIST向けの軽量学習デモ' },
];

function variantBadge(sample: SampleDef): { label: string; className: string } {
  if (sample.variant === 'train') {
    return {
      label: '学習',
      className: 'border-[var(--status-ok)]/40 text-[var(--status-ok)]',
    };
  }
  if (sample.variant === 'reference') {
    return {
      label: '参照',
      className: 'border-muted-foreground/30 text-muted-foreground',
    };
  }
  return {
    label: 'デモ',
    className: 'border-primary/30 text-primary',
  };
}

/** ブロックプリセット→単体モデルIR(u{seq}_採番はプリセット内で完結) */
function presetToGraph(spec: BlockParams): IRGraphJson {
  let seq = 1;
  const tpl = generateBlock(spec, (suffix) => `u${seq++}_${suffix}`);
  const io = blockIoWidths(spec);
  return templateToGraph(`block_${spec.kind}`, tpl, io.in, io.out);
}

interface Props {
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

export function TemplateGallery({ open, onOpenChange }: Props) {
  // マイUnitをギャラリーにも表示(単体モデルとして展開ロード)。
  const [myUnits, setMyUnits] = useState<EditorUnit[]>([]);
  useEffect(() => {
    const refresh = () => setMyUnits(listMyUnits());
    refresh();
    window.addEventListener(MYUNITS_CHANGED_EVENT, refresh);
    return () => window.removeEventListener(MYUNITS_CHANGED_EVENT, refresh);
  }, []);

  // テンプレプロジェクト一覧(ダイアログを開くたびにGET /api/templates)。
  const [templates, setTemplates] = useState<TemplateSummary[]>([]);
  const [busy, setBusy] = useState(false);
  useEffect(() => {
    if (!open) return;
    let cancelled = false;
    void getTemplates()
      .then((res) => {
        if (!cancelled) setTemplates(res.templates);
      })
      .catch(() => {
        // テンプレAPI未配線時(main.py未統合)はテンプレ節を非表示にするだけ
        if (!cancelled) setTemplates([]);
      });
    return () => {
      cancelled = true;
    };
  }, [open]);

  // テンプレ→instantiate→networksをエディタへロード+CONFIGへ反映。
  const loadTemplate = async (t: TemplateSummary) => {
    setBusy(true);
    try {
      const project = await instantiateTemplate(t.id);
      const networks = project.networks as Record<string, IRGraphJson>;
      // アクティブネットワーク(先頭)をbackendへ同期してからロードする
      // (loadSampleModelと同じ流儀。学習・統計APIがそのまま使える状態にする)。
      const activeName = Object.keys(networks)[0];
      if (activeName) await putCurrentIR(networks[activeName]);
      useNetworkStore.getState().loadNetworks(networks);
      window.dispatchEvent(
        new CustomEvent(PROJECT_INSTANTIATED_EVENT, { detail: { id: project.id } }),
      );
      onOpenChange(false);
      toast.success(`テンプレ「${t.name}」からプロジェクトを作成しました`);
    } catch (e) {
      toast.error(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  // 空プロジェクト: サーバに空プロジェクトを作成し、白紙キャンバスをロードする。
  const createEmptyProject = async () => {
    setBusy(true);
    try {
      const project = await createProject('新規プロジェクト');
      useNetworkStore.getState().loadNetworks({ Main: emptyStarterGraph() });
      window.dispatchEvent(
        new CustomEvent(PROJECT_INSTANTIATED_EVENT, { detail: { id: project.id } }),
      );
      onOpenChange(false);
      toast.success('空プロジェクトを作成しました');
    } catch (e) {
      toast.error(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  const loadUnit = async (unit: EditorUnit) => {
    const ok = await loadSampleModel({
      key: `unit_${unit.id}`,
      label: unit.name,
      desc: unit.description ?? '',
      icon: Package,
      category: 'dag',
      variant: 'demo',
      ir: unitToStandaloneGraph(unit),
    });
    if (ok) onOpenChange(false);
  };

  const loadPreset = async (spec: BlockParams, label: string) => {
    const ok = await loadSampleModel({
      key: `block_${spec.kind}`,
      label,
      desc: '',
      icon: BLOCK_DEFS[0].icon,
      category: 'dag',
      variant: 'demo',
      ir: presetToGraph(spec),
    });
    if (ok) {
      onOpenChange(false);
      toast.info('draft層はbuildでランダム初期化されます');
    }
  };

  const loadSample = async (key: string) => {
    const sample = SAMPLES.find((s) => s.key === key);
    if (!sample) return;
    if (await loadSampleModel(sample)) onOpenChange(false);
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-h-[82vh] max-w-4xl overflow-hidden p-0">
        <DialogHeader className="border-b px-5 py-4">
          <DialogTitle>テンプレートギャラリー</DialogTitle>
          <DialogDescription>
            サンプル回路またはブロックテンプレートから新規モデルを作成します。
          </DialogDescription>
        </DialogHeader>

        <ScrollArea className="max-h-[calc(82vh-92px)]">
          <div className="grid gap-5 px-5 py-4">
            {/* テンプレプロジェクト節+空プロジェクト。*/}
            <section className="grid gap-2" data-testid="gallery-templates">
              <div className="flex items-end justify-between gap-3">
                <div className="min-w-0">
                  <h3 className="text-xs font-semibold tracking-wide text-foreground">
                    テンプレプロジェクト
                  </h3>
                  <p className="truncate text-[11px] text-muted-foreground">
                    logistic / MLP / LeNet / AutoEncoder / LSTM / Transformer
                  </p>
                </div>
                <Button
                  variant="outline"
                  size="sm"
                  className="h-8 rounded-md"
                  disabled={busy}
                  data-testid="template-empty"
                  onClick={() => void createEmptyProject()}
                >
                  <FilePlus className="size-4" />
                  空プロジェクト
                </Button>
              </div>
              <div className="grid grid-cols-1 gap-2 sm:grid-cols-2">
                {templates.map((t) => {
                  const badge = verificationBadge(t.verification);
                  return (
                    <Button
                      key={t.id}
                      variant="outline"
                      className="h-[88px] justify-start rounded-md p-3 text-left"
                      disabled={busy}
                      data-testid={`template-${t.id}`}
                      onClick={() => void loadTemplate(t)}
                    >
                      <Package className="size-4 shrink-0 text-muted-foreground" />
                      <span className="flex min-w-0 flex-1 flex-col items-start gap-1">
                        <span className="flex max-w-full items-center gap-1.5">
                          <span className="truncate font-semibold">{t.name}</span>
                          <Badge
                            variant="outline"
                            className={`h-4 rounded px-1 text-[9px] ${badge.className}`}
                          >
                            {badge.label}
                          </Badge>
                        </span>
                        <span className="max-w-full truncate text-[11px] font-normal text-muted-foreground">
                          {t.description}
                        </span>
                        {t.recommended_dataset && (
                          <span className="max-w-full truncate text-[10px] font-normal text-muted-foreground">
                            推奨データセット: {t.recommended_dataset}
                          </span>
                        )}
                      </span>
                    </Button>
                  );
                })}
              </div>
            </section>

            {SAMPLE_GROUPS.map((group) => {
              const samples = SAMPLES.filter((s) => s.category === group.key);
              if (samples.length === 0) return null;
              return (
                <section key={group.key} className="grid gap-2">
                  <div className="flex items-end justify-between gap-3">
                    <div className="min-w-0">
                      <h3 className="text-xs font-semibold tracking-wide text-foreground">
                        {group.label}
                      </h3>
                      <p className="truncate text-[11px] text-muted-foreground">
                        {group.desc}
                      </p>
                    </div>
                    <Badge variant="secondary" className="rounded-md">
                      {samples.length}
                    </Badge>
                  </div>
                  <div className="grid grid-cols-1 gap-2 sm:grid-cols-2">
                    {samples.map((s) => {
                      const badge = variantBadge(s);
                      return (
                        <Button
                          key={s.key}
                          variant="outline"
                          className="h-[88px] justify-start rounded-md p-3 text-left"
                          onClick={() => void loadSample(s.key)}
                        >
                          <s.icon className="size-4 shrink-0 text-muted-foreground" />
                          <span className="flex min-w-0 flex-1 flex-col items-start gap-1">
                            <span className="flex max-w-full items-center gap-1.5">
                              <span className="truncate font-semibold">{s.label}</span>
                              <Badge
                                variant="outline"
                                className={`h-4 rounded px-1 text-[9px] ${badge.className}`}
                              >
                                {badge.label}
                              </Badge>
                            </span>
                            <span className="max-w-full truncate text-[11px] font-normal text-muted-foreground">
                              {s.desc}
                            </span>
                            {s.recommendedDataset && (
                              <span className="max-w-full truncate text-[10px] font-normal text-muted-foreground">
                                {s.recommendedDataset}
                              </span>
                            )}
                          </span>
                        </Button>
                      );
                    })}
                  </div>
                </section>
              );
            })}

            <section className="grid gap-2 border-t pt-4">
              <div className="min-w-0">
                <h3 className="text-xs font-semibold tracking-wide text-foreground">
                  ブロックテンプレート
                </h3>
                <p className="truncate text-[11px] text-muted-foreground">
                  既定パラメータで複合ブロックを単体モデル化
                </p>
              </div>
              <div className="grid grid-cols-1 gap-2 sm:grid-cols-3">
                {BLOCK_PRESETS.map(({ spec, desc, label, icon }) => {
                  const def = BLOCK_DEFS.find((d) => d.kind === spec.kind);
                  const presetLabel = label ?? def?.label ?? spec.kind;
                  const Icon = icon ?? def?.icon ?? BLOCK_DEFS[0].icon;
                  return (
                    <Button
                      key={spec.kind}
                      variant="outline"
                      className="h-[76px] justify-start rounded-md p-3 text-left"
                      data-testid={`preset-${spec.kind}`}
                      onClick={() => void loadPreset(spec, presetLabel)}
                    >
                      <Icon className="size-4 shrink-0 text-muted-foreground" />
                      <span className="flex min-w-0 flex-col items-start">
                        <span className="truncate font-semibold">{presetLabel}</span>
                        <span className="max-w-full truncate text-[11px] font-normal text-muted-foreground">
                          {desc}
                        </span>
                      </span>
                    </Button>
                  );
                })}
              </div>
            </section>

            {myUnits.length > 0 && (
              <section className="grid gap-2 border-t pt-4" data-testid="gallery-units">
                <div className="min-w-0">
                  <h3 className="text-xs font-semibold tracking-wide text-foreground">
                    マイUnit
                  </h3>
                  <p className="truncate text-[11px] text-muted-foreground">
                    定義したUnitを既定引数で単体モデル化
                  </p>
                </div>
                <div className="grid grid-cols-1 gap-2 sm:grid-cols-3">
                  {myUnits.map((unit) => (
                    <Button
                      key={unit.id}
                      variant="outline"
                      className="h-[76px] justify-start rounded-md p-3 text-left"
                      onClick={() => void loadUnit(unit)}
                    >
                      <Package className="size-4 shrink-0 text-muted-foreground" />
                      <span className="flex min-w-0 flex-col items-start">
                        <span className="truncate font-semibold">{unit.name}</span>
                        <span className="max-w-full truncate text-[11px] font-normal text-muted-foreground">
                          引数 {unit.args.length}個
                        </span>
                      </span>
                    </Button>
                  ))}
                </div>
              </section>
            )}
          </div>
        </ScrollArea>
      </DialogContent>
    </Dialog>
  );
}
