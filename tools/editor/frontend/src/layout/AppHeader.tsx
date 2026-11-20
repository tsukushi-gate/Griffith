// アプリヘッダ: ブランド | モデルメニュー | パンくず ‖ build | 同期バッジ | 学習コントロール | パネル開閉

import {
  ChevronRight,
  FlaskConical,
  Hammer,
  PanelBottom,
  PanelLeft,
  PanelRight,
  Redo2,
  Ship,
  Undo2,
  type LucideIcon,
} from 'lucide-react';
import { useState } from 'react';
import { toast } from 'sonner';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Separator } from '@/components/ui/separator';
import { ToggleGroup, ToggleGroupItem } from '@/components/ui/toggle-group';
import { runBuild } from '../commands/actions';
import { useT } from '../i18n';
import { LANGUAGES, type Language } from '../i18n/messages';
import { hasShapeErrors } from '../ir/shapeInference';
import { ModelMenu } from '../model/ModelMenu';
import { redoGraph, undoGraph, useGraphStore } from '../store/graphStore';
import { applyLens } from '../store/lens';
import { LENS_ORDER, useUIStore, type Lens } from '../store/uiStore';
import { useViewStore } from '../store/viewStore';
import { TrainControls } from '../training/TrainControls';

/** 表示言語トグル: ja/enを小さなセグメントで切替。i18n足場の言語切替UI。*/
function LanguageToggle() {
  const language = useUIStore((s) => s.language);
  const setLanguage = useUIStore((s) => s.setLanguage);
  const t = useT();
  return (
    <ToggleGroup
      type="single"
      value={language}
      variant="outline"
      size="sm"
      className="gap-0"
      title={t('header.language')}
      onValueChange={(v) => {
        if (v) setLanguage(v as Language);
      }}
    >
      {LANGUAGES.map((lang) => (
        <ToggleGroupItem
          key={lang}
          value={lang}
          data-testid={`lang-${lang}`}
          aria-label={lang}
          className="h-7 w-8 px-1 text-[11px] font-semibold uppercase data-[state=on]:bg-primary data-[state=on]:text-primary-foreground"
        >
          {lang}
        </ToggleGroupItem>
      ))}
    </ToggleGroup>
  );
}

// レンズのアイコン・ラベル・ショートカット表示(§2)。裁定1でtrainレンズは廃止(TRAININGタブへ)
const LENS_META: Record<Lens, { icon: LucideIcon; label: string }> = {
  build: { icon: Hammer, label: '構築' },
  probe: { icon: FlaskConical, label: '検証' },
  ship: { icon: Ship, label: '出荷' },
};

/** ヘッダのレンズ切替セグメントコントロール(アイコン+ラベル、現レンズ強調) */
function LensSwitcher() {
  const lens = useUIStore((s) => s.lens);
  return (
    <ToggleGroup
      type="single"
      value={lens}
      variant="outline"
      size="sm"
      className="gap-0"
      onValueChange={(v) => {
        if (v) applyLens(v as Lens);
      }}
    >
      {LENS_ORDER.map((l, i) => {
        const { icon: Icon, label } = LENS_META[l];
        return (
          <ToggleGroupItem
            key={l}
            value={l}
            data-testid={`lens-${l}`}
            title={`${label}レンズ (${i + 1})`}
            className="h-7 gap-1 px-2.5 text-xs data-[state=on]:bg-primary data-[state=on]:text-primary-foreground"
          >
            <Icon className="size-3.5" />
            {label}
          </ToggleGroupItem>
        );
      })}
    </ToggleGroup>
  );
}

export function AppHeader() {
  const view = useViewStore((s) => s.view);
  const setView = useViewStore((s) => s.setView);
  const ir = useGraphStore((s) => s.ir);
  const dirty = useGraphStore((s) => s.dirty);
  const syncState = useGraphStore((s) => s.syncState);
  const needsBuild = useGraphStore((s) => s.needsBuild);
  const diagnostics = useGraphStore((s) => s.diagnostics);
  const topTab = useUIStore((s) => s.topTab);
  const leftOpen = useUIStore((s) => s.leftOpen);
  const rightOpen = useUIStore((s) => s.rightOpen);
  const bottomOpen = useUIStore((s) => s.bottomOpen);
  const toggleLeft = useUIStore((s) => s.toggleLeft);
  const toggleRight = useUIStore((s) => s.toggleRight);
  const toggleBottom = useUIStore((s) => s.toggleBottom);
  const t = useT();

  const [building, setBuilding] = useState(false);

  const handleBuild = async () => {
    setBuilding(true);
    try {
      await runBuild();
      toast.success('build完了(draft確定・ノードid更新)');
    } catch (e) {
      toast.error(e instanceof Error ? e.message : String(e));
    } finally {
      setBuilding(false);
    }
  };

  return (
    <header className="flex h-12 shrink-0 items-center gap-2 border-b bg-[var(--panel-bg)] px-3">
      <span className="text-sm font-bold whitespace-nowrap">Griffith Editor</span>
      <ModelMenu />
      <nav className="flex items-center gap-1 text-xs">
        <Button
          variant="ghost"
          size="sm"
          className="h-7 px-2"
          disabled={view.kind === 'graph'}
          onClick={() => setView({ kind: 'graph' })}
        >
          {t('header.model')}
        </Button>
        {view.kind === 'l0' && (
          <>
            <ChevronRight className="size-3.5 text-muted-foreground" />
            <span className="font-semibold text-primary">{view.nodeId}</span>
          </>
        )}
      </nav>

      {/* レンズはEDITタブ内のサブ文脈。他タブでは非表示にする(裁定1) */}
      {topTab === 'edit' && (
        <>
          <Separator orientation="vertical" className="!h-6" />
          <LensSwitcher />
        </>
      )}

      <div className="flex-1" />

      <Button variant="ghost" size="sm" title={`${t('header.undo')} (⌘Z)`} onClick={() => undoGraph()}>
        <Undo2 className="size-4" />
      </Button>
      <Button variant="ghost" size="sm" title={`${t('header.redo')} (⇧⌘Z)`} onClick={() => redoGraph()}>
        <Redo2 className="size-4" />
      </Button>
      <Separator orientation="vertical" className="!h-6" />

      <Button
        size="sm"
        variant={needsBuild ? 'default' : 'outline'}
        disabled={!ir || building || hasShapeErrors(diagnostics)}
        title={needsBuild ? t('header.build.hintNeedsBuild') : t('header.build.hintRebuild')}
        onClick={() => void handleBuild()}
      >
        <Hammer className="size-4" />
        {t('header.build')}
      </Button>
      <Badge
        variant="outline"
        className={
          syncState === 'error'
            ? 'text-destructive border-destructive/40'
            : dirty
              ? 'text-[var(--status-warn)] border-[var(--status-warn)]/40'
              : 'text-muted-foreground'
        }
        title="IR同期状態(編集は800msデバウンスでサーバへPUTされます)"
      >
        {dirty ? '未同期' : `同期 ${syncState}`}
      </Badge>

      <Separator orientation="vertical" className="!h-6" />
      <TrainControls />
      <Separator orientation="vertical" className="!h-6" />
      <LanguageToggle />
      <Separator orientation="vertical" className="!h-6" />

      <div className="flex items-center">
        <Button
          variant="ghost"
          size="sm"
          className={leftOpen ? '' : 'text-muted-foreground/50'}
          title={t('header.togglePalette')}
          onClick={toggleLeft}
        >
          <PanelLeft className="size-4" />
        </Button>
        <Button
          variant="ghost"
          size="sm"
          className={bottomOpen ? '' : 'text-muted-foreground/50'}
          title={t('header.toggleBottom')}
          onClick={toggleBottom}
        >
          <PanelBottom className="size-4" />
        </Button>
        <Button
          variant="ghost"
          size="sm"
          className={rightOpen ? '' : 'text-muted-foreground/50'}
          title={t('header.toggleInspector')}
          onClick={toggleRight}
        >
          <PanelRight className="size-4" />
        </Button>
      </div>
    </header>
  );
}
