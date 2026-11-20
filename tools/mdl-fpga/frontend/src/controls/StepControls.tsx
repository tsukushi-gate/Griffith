// クリック=1clk の実行制御(shadcn+lucide、エディタの部品規約)。
import { CirclePlay, FastForward, Play, RotateCcw, Square } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Label } from '@/components/ui/label';
import { useT } from '../i18n/useT';
import { useSimStore } from '../store/simStore';

export function StepControls() {
  const t = useT();
  const send = useSimStore((s) => s.send);
  const running = useSimStore((s) => s.running);
  const setRunning = useSimStore((s) => s.setRunning);
  const autoStep = useSimStore((s) => s.autoStep);
  const setAutoStep = useSimStore((s) => s.setAutoStep);
  const setInputX = useSimStore((s) => s.setInputX);

  return (
    <div className="rounded-lg border bg-[var(--node-bg)] p-3">
      <div className="flex flex-wrap gap-1.5">
        <Button size="sm" onClick={() => send({ cmd: 'step', n: 1 })}>
          <Play /> {t('controls.step')}
        </Button>
        <Button size="sm" variant="outline"
                onClick={() => send({ cmd: 'step', n: 10 })}>
          <FastForward /> {t('controls.step10')}
        </Button>
        <Button
          size="sm"
          variant={running ? 'destructive' : 'outline'}
          onClick={() => {
            const next = !running;
            setRunning(next);
            if (next) send({ cmd: 'step', n: 1 }); // 初回キック(以後フレーム駆動)
          }}
        >
          {running ? <Square /> : <CirclePlay />}
          {running ? t('controls.stop') : t('controls.run')}
        </Button>
        <Button
          size="sm"
          variant="ghost"
          onClick={() => {
            setRunning(false);
            setInputX(0n);
            send({ cmd: 'reset' });
          }}
        >
          <RotateCcw /> {t('controls.reset')}
        </Button>
      </div>
      <Label className="mt-2 flex items-center gap-2 text-xs font-normal text-muted-foreground">
        <input
          type="checkbox"
          className="accent-[var(--primary)]"
          checked={autoStep}
          onChange={(e) => setAutoStep(e.target.checked)}
        />
        {t('controls.autostep')}
      </Label>
    </div>
  );
}
