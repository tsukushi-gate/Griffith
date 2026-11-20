// プローブカード本体(キャンバス上のノードとしても、ピン留めパネルでも使う)。
// 部品は shadcn プリミティブ+lucide(エディタとUI基盤を共有)。
import { Minus, Pin, PinOff, Plus, StepForward, X } from 'lucide-react';
import { useState } from 'react';
import { Button } from '@/components/ui/button';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { useT } from '../i18n/useT';
import { useSimStore } from '../store/simStore';
import { useViewerStore } from '../store/viewerStore';
import { NumberField } from './NumberField';
import { parseSlice, sliceLabel } from './parseSlice';
import { PayloadView } from './PayloadView';

export function ProbeCard({ probeId }: { probeId: string }) {
  const t = useT();
  const probe = useViewerStore((s) => s.probes.find((p) => p.id === probeId));
  const updateProbe = useViewerStore((s) => s.updateProbe);
  const removeProbe = useViewerStore((s) => s.removeProbe);
  const payload = useSimStore((s) => s.frame?.probes[probeId]);
  const meta = useSimStore((s) => s.meta);
  const renderers = useSimStore((s) => s.renderers);
  const send = useSimStore((s) => s.send);
  const [sliceText, setSliceText] = useState<string | null>(null);
  const [sliceError, setSliceError] = useState<string | null>(null);

  if (!probe || !meta) return null;

  const width =
    probe.target === 'x'
      ? meta.in_width
      : probe.target === 'y' || probe.target === 'y_comb'
        ? meta.out_width
        : (meta.nodes.find((n) => n.id === probe.target)?.width ?? 0);

  const commitSlice = (text: string) => {
    try {
      const sl = parseSlice(text, width);
      updateProbe(probe.id, { slice: sl });
      setSliceError(null);
      setSliceText(null);
    } catch (e) {
      setSliceError((e as Error).message);
    }
  };

  const rendererNames = renderers
    ? [...renderers.builtins, ...renderers.sidecar]
    : ['auto'];
  const showCols = ['bits', 'image', 'auto'].includes(probe.renderer);
  const showOrder = ['bits', 'auto'].includes(probe.renderer);
  const scale = probe.scale || 1;

  return (
    <div className="probe-card" style={{ zoom: scale }}>
      <div className="probe-head">
        <span className="probe-target" title={probe.target}>
          {probe.label ?? probe.target}
        </span>
        {(probe.target === 'y' || probe.target === 'y_comb') && (
          <Button
            size="xs"
            variant="outline"
            title="出力レジスタ(y)と組み合わせ出力(y_comb)を切替"
            onClick={() =>
              updateProbe(probe.id, {
                target: probe.target === 'y' ? 'y_comb' : 'y',
              })
            }
          >
            {probe.target}
          </Button>
        )}
        <input
          className={`slice-input${sliceError ? ' bad' : ''}`}
          placeholder="[msb:lsb]"
          value={sliceText ?? sliceLabel(probe.slice)}
          onChange={(e) => setSliceText(e.target.value)}
          onBlur={(e) => commitSlice(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter') commitSlice((e.target as HTMLInputElement).value);
          }}
          title={`ビットスライス(バス幅 ${width})。空=全体、例: 5 / 7:0`}
        />
        <Select
          value={probe.renderer}
          onValueChange={(v) => updateProbe(probe.id, { renderer: v })}
        >
          <SelectTrigger size="sm" className="h-6 w-[92px] px-2 text-[11px]">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            {rendererNames.map((r) => (
              <SelectItem key={r} value={r} className="text-xs">
                {r}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
        <Button
          size="icon-xs"
          variant="ghost"
          title={probe.pinned ? t('probe.unpin') : t('probe.pin')}
          onClick={() => updateProbe(probe.id, { pinned: !probe.pinned })}
        >
          {probe.pinned ? <PinOff /> : <Pin />}
        </Button>
        <Button
          size="icon-xs"
          variant="ghost"
          className="text-destructive"
          title={t('probe.delete')}
          onClick={() => removeProbe(probe.id)}
        >
          <X />
        </Button>
      </div>
      <div className="probe-opts">
        {showCols && (
          <label title="表示の横幅(bit数)。空=自動(imageは正方形)">
            {t('probe.cols')}
            <NumberField
              value={probe.args.cols ?? null}
              min={1}
              max={Math.max(1, width)}
              placeholder="auto"
              onCommit={(v) =>
                updateProbe(probe.id, {
                  args: { ...probe.args, cols: v ?? undefined },
                })
              }
            />
          </label>
        )}
        {showOrder && (
          <Button
            size="xs"
            variant="outline"
            className="h-5 px-1.5 font-mono text-[10px]"
            title="ビットの並び順を切替"
            onClick={() =>
              updateProbe(probe.id, {
                args: {
                  ...probe.args,
                  order: probe.args.order === 'lsb' ? 'msb' : 'lsb',
                },
              })
            }
          >
            {probe.args.order === 'lsb' ? '[lsb:msb]' : '[msb:lsb]'}
          </Button>
        )}
        <label title="表示キュー段数(過去のクロックの値も保持)">
          {t('probe.depth')}
          <NumberField
            value={probe.depth}
            min={1}
            max={16}
            onCommit={(v) => updateProbe(probe.id, { depth: v ?? 1 })}
          />
        </label>
        <span className="scale-ctl" title="カード個別の拡大縮小">
          <Button size="icon-xs" variant="ghost"
            onClick={() =>
              updateProbe(probe.id, {
                scale: Math.max(0.5, Math.round((scale - 0.25) * 4) / 4),
              })
            }
          >
            <Minus />
          </Button>
          {scale.toFixed(2)}×
          <Button size="icon-xs" variant="ghost"
            onClick={() =>
              updateProbe(probe.id, {
                scale: Math.min(3, Math.round((scale + 0.25) * 4) / 4),
              })
            }
          >
            <Plus />
          </Button>
        </span>
        <Button
          size="xs"
          variant="outline"
          className="h-5 px-1.5 text-[10px] text-primary"
          title={t('probe.until.hint')}
          onClick={() =>
            send({
              cmd: 'step_until',
              target: probe.target,
              slice: probe.slice,
              max: 1000,
            })
          }
        >
          <StepForward /> {t('probe.until')}
        </Button>
      </div>
      {sliceError && <div className="payload-error">{sliceError}</div>}
      <PayloadView payload={payload} args={probe.args} />
    </div>
  );
}
