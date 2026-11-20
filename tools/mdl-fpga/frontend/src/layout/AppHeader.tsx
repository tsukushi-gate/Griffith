// アプリヘッダ(エディタ AppHeader と同型: h-12 / border-b / panel-bg / px-3)。
// 旧 StatusBar の内容(接続・cycle・latency・step_until 結果)を Badge で吸収。
import { Moon, Sun } from 'lucide-react';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Separator } from '@/components/ui/separator';
import { ToggleGroup, ToggleGroupItem } from '@/components/ui/toggle-group';
import { Tooltip, TooltipContent, TooltipTrigger } from '@/components/ui/tooltip';
import { useT, useLanguageStore } from '../i18n/useT';
import { LANGUAGES, type Language } from '../i18n/messages';
import { useSimStore } from '../store/simStore';
import { useThemeStore } from '../store/themeStore';

export function AppHeader() {
  const t = useT();
  const meta = useSimStore((s) => s.meta);
  const frame = useSimStore((s) => s.frame);
  const connected = useSimStore((s) => s.connected);
  const theme = useThemeStore((s) => s.theme);
  const toggleTheme = useThemeStore((s) => s.toggle);
  const language = useLanguageStore((s) => s.language);
  const setLanguage = useLanguageStore((s) => s.setLanguage);

  return (
    <header className="flex h-12 shrink-0 items-center gap-2 border-b bg-[var(--panel-bg)] px-3">
      <span className="text-sm font-bold whitespace-nowrap">
        {t('header.brand')}
      </span>
      {meta && (
        <>
          <Separator orientation="vertical" className="!h-6" />
          <span className="text-xs text-muted-foreground whitespace-nowrap">
            {meta.name} — in {meta.in_width}b / out {meta.out_width}b
            {meta.k ? ` / ${meta.k}cls` : ''}
            {meta.latency ? ` / ${meta.latency}clk` : ''}
          </span>
          {meta.mode === 'system' && (
            <Badge variant="outline" className="text-primary">
              {t('header.system')}
            </Badge>
          )}
        </>
      )}
      {frame && frame.xz.length > 0 && (
        <Badge variant="outline" className="text-[var(--status-warn)]"
               title={frame.xz.join(', ')}>
          X/Z
        </Badge>
      )}
      <div className="flex-1" />
      <Badge variant="outline" className="font-mono text-xs">
        cycle {frame?.cycle ?? '-'}
      </Badge>
      <Badge variant="outline" className="font-mono text-xs text-muted-foreground">
        {frame?.latency_ms ?? '-'} ms
      </Badge>
      {frame?.steps !== undefined && (
        <Badge
          variant="outline"
          className={frame.changed
            ? 'text-[var(--status-ok)]'
            : 'text-[var(--status-warn)]'}
          title={`watch: ${frame.watch}`}
        >
          ⏭ +{frame.steps}clk
        </Badge>
      )}
      <Badge
        variant="outline"
        className={connected ? 'text-[var(--status-ok)]' : 'text-destructive'}
      >
        {connected ? `● ${t('header.connected')}` : `○ ${t('header.disconnected')}`}
      </Badge>
      <Separator orientation="vertical" className="!h-6" />
      <Tooltip>
        <TooltipTrigger asChild>
          <Button variant="ghost" size="icon-sm" onClick={toggleTheme}
                  data-testid="theme-toggle">
            {theme === 'light' ? <Moon /> : <Sun />}
          </Button>
        </TooltipTrigger>
        <TooltipContent>{t('header.theme')}</TooltipContent>
      </Tooltip>
      <ToggleGroup
        type="single"
        variant="outline"
        size="sm"
        value={language}
        onValueChange={(v) => v && setLanguage(v as Language)}
      >
        {LANGUAGES.map((lang) => (
          <ToggleGroupItem
            key={lang}
            value={lang}
            className="h-7 w-8 text-[11px] font-semibold uppercase"
            data-testid={`lang-${lang}`}
          >
            {lang}
          </ToggleGroupItem>
        ))}
      </ToggleGroup>
    </header>
  );
}
