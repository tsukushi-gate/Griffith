// 検出候補が複数のときの確認ダイアログ。ボード画像は使わず、基板のシルク印字
// (silk_hint)の案内テキストで区別する(同一 FPGA 搭載ボードは原理的に
// JTAG では区別できないため、最後はユーザーの目視確認)。
import { Button } from '@/components/ui/button';
import {
  Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle,
} from '@/components/ui/dialog';
import { useT } from '../i18n/useT';
import { useBoardStore } from '../store/boardStore';

export function BoardDetectDialog() {
  const t = useT();
  const candidates = useBoardStore((s) => s.candidates);

  return (
    <Dialog
      open={candidates !== null}
      onOpenChange={(open) => {
        if (!open) useBoardStore.getState().dismissCandidates();
      }}
    >
      <DialogContent>
        <DialogHeader>
          <DialogTitle>{t('board.candidates.title')}</DialogTitle>
          <DialogDescription>{t('board.candidates.desc')}</DialogDescription>
        </DialogHeader>
        <div className="flex flex-col gap-2">
          {(candidates ?? []).map((c) => (
            <Button
              key={c.key}
              variant="outline"
              className="h-auto w-full flex-col items-start gap-0.5"
              onClick={() => void useBoardStore.getState().select(c.key)}
            >
              <span>
                {c.name}
                {c.unverified ? `(${t('board.unverified')})` : ''}
              </span>
              {c.silk_hint ? (
                <span className="text-xs font-normal text-muted-foreground">
                  {c.silk_hint}
                </span>
              ) : null}
            </Button>
          ))}
        </div>
      </DialogContent>
    </Dialog>
  );
}
