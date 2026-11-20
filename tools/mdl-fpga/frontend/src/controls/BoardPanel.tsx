// ボードペイン: DB からの選択+「認識」ボタン+詳細+搭載可能性(advise)表示。
// 認識(JTAG スキャン)は排他資源のため必ずボタン起点 — マウント時の自動実行は
// 絶対にしない(refresh は DB 一覧と選択状態の取得のみで JTAG に触らない)。
import { useEffect } from 'react';
import { ScanSearch } from 'lucide-react';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import {
  Select, SelectContent, SelectItem, SelectTrigger, SelectValue,
} from '@/components/ui/select';
import { useT } from '../i18n/useT';
import { useBoardStore } from '../store/boardStore';
import { BoardDetectDialog } from './BoardDetectDialog';

export function BoardPanel() {
  const t = useT();
  const boards = useBoardStore((s) => s.boards);
  const state = useBoardStore((s) => s.state);
  const detecting = useBoardStore((s) => s.detecting);
  const detectResult = useBoardStore((s) => s.detectResult);
  const error = useBoardStore((s) => s.error);

  useEffect(() => {
    void useBoardStore.getState().refresh();
  }, []);

  const selected = boards.find((b) => b.key === state?.key) ?? null;
  const unverified = selected
    ? selected.unverified === true || selected.tier !== 1
    : false;
  const advice = state?.advice ?? null;
  const unit = selected?.resources?.unit;
  const unitTotal =
    unit && typeof selected?.resources?.[unit] === 'number'
      ? (selected.resources[unit] as number)
      : null;

  return (
    <div className="rounded-lg border bg-[var(--node-bg)] p-3">
      <div className="mb-2 flex items-center justify-between">
        <span className="text-xs text-muted-foreground">
          {t('board.title')}
        </span>
        <Button
          size="sm"
          variant="outline"
          disabled={detecting}
          onClick={() => void useBoardStore.getState().detect()}
        >
          <ScanSearch />
          {detecting ? t('board.detecting') : t('board.detect')}
        </Button>
      </div>
      <Select
        value={state?.key ?? ''}
        onValueChange={(v) => void useBoardStore.getState().select(v)}
      >
        <SelectTrigger size="sm" className="w-full">
          <SelectValue placeholder={t('board.select.placeholder')} />
        </SelectTrigger>
        <SelectContent>
          {boards.map((b) => (
            <SelectItem key={b.key} value={b.key}>
              {b.name}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>

      {selected ? (
        <div className="mt-2 flex flex-col gap-1 text-xs">
          <div className="flex items-center gap-1">
            <span className="text-muted-foreground">{selected.device}</span>
            {unverified ? (
              <Badge variant="destructive">{t('board.unverified')}</Badge>
            ) : (
              <Badge variant="outline">Tier1</Badge>
            )}
          </div>
          <div className="text-muted-foreground">
            {selected.family}
            {unit && unitTotal != null
              ? ` / ${unit.toUpperCase()} ${unitTotal.toLocaleString()}`
              : ''}
          </div>
        </div>
      ) : null}

      {advice ? (
        <div className="mt-2 border-t pt-2 text-xs">
          <div className="mb-1 flex items-center gap-1">
            <span className="text-muted-foreground">
              {t('board.fits.title')}
            </span>
            {/* 3段階: 有り(spatial可)/やや有り(overlayのみ可)/無し */}
            <Badge
              variant={
                !advice.spatial_ok && !advice.overlay_ok
                  ? 'destructive'
                  : 'outline'
              }
            >
              {advice.spatial_ok
                ? t('board.fits.yes')
                : advice.overlay_ok
                  ? t('board.fits.partial')
                  : t('board.fits.no')}
            </Badge>
          </div>
          <ul className="flex list-disc flex-col gap-0.5 pl-4 text-muted-foreground">
            {advice.messages.map((m, i) => (
              <li key={i}>{m}</li>
            ))}
          </ul>
        </div>
      ) : null}
      {state?.advice_error ? (
        <div className="mt-1 text-xs text-muted-foreground">
          {t('board.advice.unavailable')}
        </div>
      ) : null}

      {detectResult && !detectResult.best
        && detectResult.candidates.length === 0 ? (
        <div className="mt-2 text-xs text-muted-foreground">
          {t('board.detect.none')}
          {detectResult.observation.notes.map((n, i) => (
            <div key={i}>{n}</div>
          ))}
        </div>
      ) : null}
      {error ? (
        <div className="mt-1 text-xs text-destructive">{error}</div>
      ) : null}
      <BoardDetectDialog />
    </div>
  );
}
