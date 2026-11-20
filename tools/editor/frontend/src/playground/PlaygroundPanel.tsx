// フォワード実行プレイグラウンド。下部ドック「実行」タブの中身。
// 左: 実行on/off・hard/soft切替・入力トグル・出力表示 / 右: 真理値表検証テーブル。
// 学習ライブ/リプレイのstep変化はここでuseEffect購読してrecomputeする。

import { useEffect } from 'react';
import { Button } from '@/components/ui/button';
import { useGraphStore } from '../store/graphStore';
import { useDisplayStep } from '../store/selectors';
import { useTrainingStore } from '../store/trainingStore';
import { CounterexampleTracePanel } from './CounterexampleTracePanel';
import type { EvalMode } from './evalCircuit';
import { InputTogglePanel } from './InputTogglePanel';
import { usePlaygroundStore } from './playgroundStore';
import { TruthTableMatrix } from './TruthTableMatrix';

/** 出力値の表示形式: hardの0/1はビット列、それ以外(スコア/確率)は数値列 */
function formatOutput(values: Float32Array, mode: EvalMode): string {
  const arr = Array.from(values);
  if (mode === 'hard') {
    return arr.every((v) => v === 0 || v === 1)
      ? arr.join('')
      : arr.map((v) => `${v}`).join(' ');
  }
  return arr.map((v) => v.toFixed(2)).join(' ');
}

export function PlaygroundPanel() {
  const ir = useGraphStore((s) => s.ir);
  const enabled = usePlaygroundStore((s) => s.enabled);
  const mode = usePlaygroundStore((s) => s.mode);
  const values = usePlaygroundStore((s) => s.values);
  const error = usePlaygroundStore((s) => s.error);
  const setEnabled = usePlaygroundStore((s) => s.setEnabled);
  const setMode = usePlaygroundStore((s) => s.setMode);
  const recompute = usePlaygroundStore((s) => s.recompute);
  const scheduleRecompute = usePlaygroundStore((s) => s.scheduleRecompute);

  // enabled切替は即時再評価
  useEffect(() => {
    if (enabled) recompute();
  }, [enabled, recompute]);

  // 学習ライブ(gateDist)・リプレイ(replayFrame)・step変化に追随して再評価。
  // gate_distは高頻度で届くためスロットリング(末尾集約)して間引く(§4-2)。
  const displayStep = useDisplayStep();
  const gateDist = useTrainingStore((s) => s.liveDetail.gateDist);
  const replayFrame = useTrainingStore((s) => s.replayFrame);
  useEffect(() => {
    if (enabled) scheduleRecompute();
  }, [enabled, displayStep, gateDist, replayFrame, scheduleRecompute]);

  if (!ir) {
    return (
      <div className="flex h-full items-center justify-center text-xs text-muted-foreground">
        モデルを読み込むとフォワード実行を試せます
      </div>
    );
  }

  return (
    <div className="flex h-full min-h-0 gap-4 overflow-hidden p-3">
      <div className="flex w-[300px] shrink-0 flex-col gap-3 overflow-y-auto pr-1">
        <div className="flex items-center gap-2">
          <Button
            size="xs"
            variant={enabled ? 'default' : 'outline'}
            onClick={() => setEnabled(!enabled)}
          >
            {enabled ? '実行モード ON' : '実行モード OFF'}
          </Button>
          <div className="flex overflow-hidden rounded-md border">
            {(['hard', 'soft'] as const).map((m) => (
              <button
                key={m}
                type="button"
                disabled={!enabled}
                className={`cursor-pointer px-2 py-1 text-[11px] transition-colors disabled:cursor-default disabled:opacity-50 ${
                  mode === m
                    ? 'bg-primary text-primary-foreground'
                    : 'bg-background text-muted-foreground hover:bg-muted'
                }`}
                onClick={() => setMode(m)}
              >
                {m}
              </button>
            ))}
          </div>
        </div>
        {enabled ? (
          <>
            <InputTogglePanel />
            <div className="flex flex-col gap-1">
              <div className="text-[11px] font-semibold">出力</div>
              {error ? (
                <div className="text-[11px] text-destructive">{error}</div>
              ) : (
                ir.outputs.map((port) => {
                  const v = values?.[port.id];
                  return (
                    <div key={port.id} className="rounded-md border bg-card px-3 py-2">
                      <div className="text-[10px] text-muted-foreground">
                        {port.id} ({port.width}bit{mode === 'soft' ? '・確率' : ''})
                      </div>
                      <div className="font-mono text-2xl font-bold tracking-widest">
                        {v ? formatOutput(v, mode) : '—'}
                      </div>
                    </div>
                  );
                })
              )}
            </div>
            <CounterexampleTracePanel />
          </>
        ) : (
          <div className="text-[11px] leading-relaxed text-muted-foreground">
            実行モードをONにすると、入力ビットをトグルして信号の流れを
            キャンバス上でライブ観察できます(ゲート出力バッジ+エッジ着色)。
          </div>
        )}
      </div>
      <div className="min-h-0 min-w-0 flex-1">
        {enabled ? (
          <TruthTableMatrix />
        ) : (
          <div className="flex h-full items-center justify-center text-xs text-muted-foreground">
            実行モードON時に真理値表検証テーブルを表示します
          </div>
        )}
      </div>
    </div>
  );
}
