// 誤差逆伝播リプレイ。
// grad_flowトレース(1ステップ分のbackwardを層分解したもの)を、
// 損失側(order=0)→入力側の順に再生し、viewStore.bpAnimへ発光中レイヤと
// 勾配強度Map(edge_grad/grad_w由来)を書き込む。FlowCanvasのエッジ太さ・
// ゲート発光・粒子オーバーレイ、DeckCanvasのL0着色がこれを購読する。
// トランスポート: 速度(0.25x〜4x)・一時停止/再開・ループ・層クリックで位相ジャンプ。

import { Pause, Play, Repeat, Square, Zap } from 'lucide-react';
import { useCallback, useEffect, useRef, useState } from 'react';
import { Button } from '@/components/ui/button';
import { buildBpIntensityMaps, type BpIntensityMaps } from '../anim/edgeGradMap';
import { decodeU8 } from '../api/u8';
import { useTrainingStore, type GradFlowFrame } from '../store/trainingStore';
import { BP_ANIM_IDLE, useViewStore } from '../store/viewStore';

const PHASE_MS = 450; // 1層あたりの発光時間(1x時)

const SPEED_OPTIONS = [0.25, 0.5, 1, 2, 4];

/** 現在表示すべきgrad_flowトレース(replay優先)。 */
function useCurrentTrace(): GradFlowFrame | null {
  const displayMode = useTrainingStore((s) => s.displayMode);
  const liveTrace = useTrainingStore((s) => s.liveDetail.gradFlow);
  const replayFrame = useTrainingStore((s) => s.replayFrame);
  if (displayMode === 'replay') {
    const trace = replayFrame?.trace as GradFlowFrame | null | undefined;
    return trace ?? null;
  }
  return liveTrace;
}

export function BackpropPlayer() {
  const trace = useCurrentTrace();
  const setBpAnim = useViewStore((s) => s.setBpAnim);
  const [playing, setPlaying] = useState(false);
  const [paused, setPaused] = useState(false);
  const [loop, setLoop] = useState(false);
  const [speed, setSpeed] = useState(1);
  const [phase, setPhase] = useState(-1);
  const phaseRef = useRef(-1);
  const loopRef = useRef(false);
  const mapsRef = useRef<BpIntensityMaps | null>(null);

  /** 現在位相をviewStoreへ書き込む(層発光+勾配強度Mapの供給点)。 */
  const publish = useCallback(
    (p: number, t: GradFlowFrame) => {
      setBpAnim({
        activeLayer: t.layers[p]?.layer ?? null,
        phase: p,
        frame: t,
        edgeIntensity: mapsRef.current?.edgeIntensity ?? null,
        gateIntensity: mapsRef.current?.gateIntensity ?? null,
      });
    },
    [setBpAnim],
  );

  const stop = useCallback(() => {
    setPlaying(false);
    setPaused(false);
    setPhase(-1);
    phaseRef.current = -1;
    mapsRef.current = null;
    setBpAnim(BP_ANIM_IDLE);
  }, [setBpAnim]);

  const start = () => {
    if (!trace) return;
    mapsRef.current = buildBpIntensityMaps(trace);
    phaseRef.current = 0;
    setPhase(0);
    setPaused(false);
    setPlaying(true);
    publish(0, trace);
  };

  /** 層リストクリックで位相ジャンプ(停止中は一時停止状態で固定表示する)。 */
  const jumpTo = (i: number) => {
    if (!trace) return;
    if (!playing) {
      mapsRef.current = buildBpIntensityMaps(trace);
      setPlaying(true);
      setPaused(true);
    }
    phaseRef.current = i;
    setPhase(i);
    publish(i, trace);
  };

  const toggleLoop = () => {
    loopRef.current = !loop;
    setLoop(!loop);
  };

  // 再生ティック: order昇順(損失側→入力側)に層を進める。速度変更は間隔へ即反映
  useEffect(() => {
    if (!playing || paused || !trace) return;
    const timer = window.setInterval(() => {
      const next = phaseRef.current + 1;
      if (next >= trace.layers.length) {
        if (loopRef.current) {
          phaseRef.current = 0;
          setPhase(0);
          publish(0, trace);
        } else {
          stop();
        }
        return;
      }
      phaseRef.current = next;
      setPhase(next);
      publish(next, trace);
    }, PHASE_MS / speed);
    return () => window.clearInterval(timer);
  }, [playing, paused, speed, trace, publish, stop]);

  // 再生中にトレースが差し替わったら(live更新)強度Mapを再構築し、位相をクランプ
  useEffect(() => {
    if (!playing) return;
    if (!trace) {
      // eslint-disable-next-line react-hooks/set-state-in-effect -- トレース消失(セッションリセット等)時に再生を強制停止する(意図的)
      stop();
      return;
    }
    mapsRef.current = buildBpIntensityMaps(trace);
    const p = Math.min(Math.max(phaseRef.current, 0), trace.layers.length - 1);
    phaseRef.current = p;
    publish(p, trace);
  }, [playing, trace, publish, stop]);

  // アンマウント時に発光を消す
  useEffect(() => () => setBpAnim(BP_ANIM_IDLE), [setBpAnim]);

  if (!trace) {
    return (
      <div className="flex items-center gap-2 border-t pt-2 text-[11px] text-muted-foreground">
        <Zap className="size-3.5 opacity-50" />
        誤差逆伝播トレース未受信(小規模モデル=capture full で学習すると記録されます)
      </div>
    );
  }

  const maxNorm = Math.max(...Object.values(trace.layer_grad_norm), 1e-12);

  return (
    <div className="flex flex-col gap-1.5 border-t pt-2">
      <div className="flex items-center gap-2">
        {playing ? (
          <>
            <Button size="sm" variant="outline" onClick={stop} title="再生停止">
              <Square className="size-3.5" />
            </Button>
            {paused ? (
              <Button
                size="sm"
                variant="outline"
                onClick={() => setPaused(false)}
                title="再開"
              >
                <Play className="size-3.5" />
              </Button>
            ) : (
              <Button
                size="sm"
                variant="outline"
                onClick={() => setPaused(true)}
                title="一時停止"
              >
                <Pause className="size-3.5" />
              </Button>
            )}
          </>
        ) : (
          <Button
            size="sm"
            variant="outline"
            onClick={start}
            title="損失側から入力側へ、誤差の伝播を層順に再生"
          >
            <Play className="size-3.5" />
            逆伝播を再生
          </Button>
        )}
        <select
          className="h-7 rounded border bg-background px-1 text-[11px]"
          value={speed}
          onChange={(e) => setSpeed(Number(e.target.value))}
          title="再生速度"
        >
          {SPEED_OPTIONS.map((s) => (
            <option key={s} value={s}>
              {s}x
            </option>
          ))}
        </select>
        <Button
          size="sm"
          variant={loop ? 'secondary' : 'outline'}
          onClick={toggleLoop}
          title={loop ? 'ループ再生: on' : 'ループ再生: off'}
        >
          <Repeat className="size-3.5" />
        </Button>
        <span className="text-[11px] text-muted-foreground">
          step {trace.step} ・ loss {trace.loss.toFixed(4)} ・ 損失側→入力側
          {playing && paused && ' ・ 一時停止中'}
        </span>
      </div>
      <div className="flex flex-col gap-1">
        {trace.layers.map((entry, i) => {
          const norm = trace.layer_grad_norm[entry.layer] ?? 0;
          const gradOut = entry.grad_out ? decodeU8(entry.grad_out) : null;
          const meanOut = gradOut
            ? gradOut.reduce((a, b) => a + b, 0) / gradOut.length
            : null;
          const active = playing && phase === i;
          return (
            <button
              type="button"
              key={entry.layer}
              onClick={() => jumpTo(i)}
              title="クリックでこの層へジャンプ"
              className={`flex cursor-pointer items-center gap-2 rounded px-1.5 py-0.5 text-left text-[11px] transition-colors hover:bg-secondary/60 ${
                active ? 'bg-[var(--chart-loss)]/15 ring-1 ring-[var(--chart-loss)]' : ''
              }`}
            >
              <span className="w-6 text-right font-mono text-muted-foreground">
                {i === 0 ? 'L' : `-${i}`}
              </span>
              <span className="w-28 truncate font-mono">{entry.layer}</span>
              <div className="h-[8px] flex-1 overflow-hidden rounded-sm bg-muted">
                <div
                  className="h-full rounded-sm"
                  style={{
                    width: `${Math.max((norm / maxNorm) * 100, 2)}%`,
                    background: active ? 'var(--chart-loss)' : 'var(--chart-acc)',
                  }}
                />
              </div>
              <span className="w-20 shrink-0 text-right font-mono text-muted-foreground">
                ‖∇W‖ {norm.toExponential(1)}
              </span>
              {meanOut !== null && (
                <span className="w-24 shrink-0 text-right font-mono text-muted-foreground">
                  |∂L/∂y| {meanOut.toExponential(1)}
                </span>
              )}
            </button>
          );
        })}
      </div>
    </div>
  );
}
