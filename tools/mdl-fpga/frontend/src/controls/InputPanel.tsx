// 入力編集(≤64bit: トグル格子 / 2D shape・784bit: 手描きキャンバス)。
import { useEffect, useRef, useState } from 'react';
import { Button } from '@/components/ui/button';
import { useT } from '../i18n/useT';
import { useSimStore } from '../store/simStore';

export function InputPanel() {
  const t = useT();
  const meta = useSimStore((s) => s.meta);
  if (!meta) return null;
  const shape2d =
    meta.input_shape && meta.input_shape.length === 2
      ? (meta.input_shape as [number, number])
      : meta.in_width === 784
        ? ([28, 28] as [number, number])
        : null;
  return (
    <div className="rounded-lg border bg-[var(--node-bg)] p-3">
      <div className="mb-2 text-xs text-muted-foreground">
        {t('input.title')}({meta.in_width}bit)
      </div>
      {meta.in_width <= 64 && !shape2d ? (
        <BitToggles width={meta.in_width} />
      ) : (
        <PaintCanvas
          width={meta.in_width}
          side={shape2d ? shape2d[1] : Math.ceil(Math.sqrt(meta.in_width))}
        />
      )}
    </div>
  );
}

function BitToggles({ width }: { width: number }) {
  const inputX = useSimStore((s) => s.inputX);
  const setInputX = useSimStore((s) => s.setInputX);
  const pushInput = useSimStore((s) => s.pushInput);
  return (
    <div className="bit-toggles">
      {Array.from({ length: width }, (_, i) => {
        const on = ((inputX >> BigInt(i)) & 1n) === 1n;
        return (
          <button
            key={i}
            className={`bit-toggle${on ? ' on' : ''}`}
            title={`x[${i}]`}
            onClick={() => {
              setInputX(inputX ^ (1n << BigInt(i)));
              // setState は同期なので直後の push で新値が送られる
              pushInput();
            }}
          >
            {i}
          </button>
        );
      })}
    </div>
  );
}

function PaintCanvas({ width, side }: { width: number; side: number }) {
  const t = useT();
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const inputX = useSimStore((s) => s.inputX);
  const setInputX = useSimStore((s) => s.setInputX);
  const pushInput = useSimStore((s) => s.pushInput);
  const [drawing, setDrawing] = useState<0 | 1 | 2>(0);
  const scale = Math.max(4, Math.floor(252 / side));

  useEffect(() => {
    const cv = canvasRef.current;
    if (!cv) return;
    const g = cv.getContext('2d')!;
    g.fillStyle = '#232a3a';
    g.fillRect(0, 0, cv.width, cv.height);
    g.fillStyle = '#4ade80';
    for (let i = 0; i < width; i++) {
      if (((inputX >> BigInt(i)) & 1n) === 1n) {
        g.fillRect((i % side) * scale, Math.floor(i / side) * scale,
                   scale - 1, scale - 1);
      }
    }
  }, [inputX, width, side, scale]);

  const apply = (e: React.PointerEvent, erase: boolean) => {
    const cv = canvasRef.current!;
    const r = cv.getBoundingClientRect();
    // CSS でペイン幅に追従して縮尺が変わるため、実表示サイズから逆算する
    const cell = r.width / side;
    const px = Math.floor((e.clientX - r.left) / cell);
    const py = Math.floor((e.clientY - r.top) / cell);
    if (px < 0 || py < 0 || px >= side || py >= side) return;
    const i = py * side + px;
    if (i >= width) return;
    const cur = useSimStore.getState().inputX;
    setInputX(erase ? cur & ~(1n << BigInt(i)) : cur | (1n << BigInt(i)));
  };

  return (
    <div className="overflow-x-auto">
      <canvas
        ref={canvasRef}
        className="paint-canvas"
        style={{ maxWidth: side * scale }}
        width={side * scale}
        height={side * scale}
        onPointerDown={(e) => {
          const mode = e.button === 2 ? 2 : 1;
          setDrawing(mode);
          // ポインタをキャプチャ: パレット外でボタンを離しても pointerup が
          // このキャンバスに届く(押下判定の取り残し→勝手に描画される防止)
          (e.target as HTMLCanvasElement).setPointerCapture?.(e.pointerId);
          apply(e, mode === 2);
          e.preventDefault();
        }}
        onPointerMove={(e) => {
          if (!drawing) return;
          if (e.buttons === 0) {
            // 保険: キャプチャ外でボタンが既に離されていたら描画状態を解除
            setDrawing(0);
            pushInput();
            return;
          }
          apply(e, drawing === 2);
        }}
        onPointerUp={(e) => {
          if (drawing) {
            setDrawing(0);
            pushInput();
          }
          (e.target as HTMLCanvasElement).releasePointerCapture?.(e.pointerId);
        }}
        onPointerCancel={() => setDrawing(0)}
        onContextMenu={(e) => e.preventDefault()}
      />
      <div className="mt-2 flex items-center gap-2">
        <Button
          size="xs"
          variant="outline"
          onClick={() => {
            setInputX(0n);
            pushInput();
          }}
        >
          {t('input.clear')}
        </Button>
        <span className="text-[10px] text-muted-foreground">
          {t('input.painthint')}
        </span>
      </div>
    </div>
  );
}
