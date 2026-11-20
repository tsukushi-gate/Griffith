// 手描きキャンバス推論デモ+学習中ライブ推論。
// 28×28相当のキャンバスへ手描き→2値化(閾値スライダ)→推論→クラス確率バー。
// 推論経路は2系統をトグルで切替:
//   - probe: 既存 POST /api/probe(バックのゴールデン評価)
//   - live : ブラウザ内 evalCircuit + 学習スナップショット確率(readTrainingProbs)。
//            学習が進むほど確率が変わる様子を体感できる。大規模IRは自動でprobeへフォールバック。
// EVALUATIONタブ内のアコーディオンに埋め込む(組み込みはEvaluationTab側)。

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { DETAIL_TOTAL_MAX } from '../detail/detailConfig';
import { evalCircuit } from '../playground/evalCircuit';
import { readTrainingProbs } from '../playground/trainingProbs';
import { useGraphStore } from '../store/graphStore';
import { useTrainingStore } from '../store/trainingStore';
import {
  argmax,
  binarize,
  browserEvalCost,
  downsampleGray,
  isMnistShapedIR,
  MNIST_GRID,
  normalizeProbs,
  outputSourceNodeId,
} from './drawUtils';
import { probeInference } from './probeApi';

/** キャンバス表示解像度(1辺px)。MNIST_GRID の整数倍にする。 */
const CANVAS_SIZE = 280;
/** ペン太さ(表示px)。数字が潰れない程度に太め。 */
const PEN_WIDTH = 22;
/** ライブ再評価の間引き間隔(ms)。高頻度なgate_dist更新を間引く。 */
const LIVE_THROTTLE_MS = 200;

type InferMode = 'probe' | 'live';

/** キャンバスの描画強度([0,1]、1=描画あり)を読み取り 28×28 へ縮小する。 */
function readGrid(canvas: HTMLCanvasElement): number[] {
  const ctx = canvas.getContext('2d');
  if (!ctx) return new Array<number>(MNIST_GRID * MNIST_GRID).fill(0);
  const { width, height } = canvas;
  const img = ctx.getImageData(0, 0, width, height).data;
  const gray = new Float32Array(width * height);
  for (let i = 0; i < width * height; i++) {
    // 白背景・黒ペン。輝度が低い(暗い)ほど描画強度が高い。
    const r = img[i * 4];
    const g = img[i * 4 + 1];
    const b = img[i * 4 + 2];
    const lum = (0.299 * r + 0.587 * g + 0.114 * b) / 255;
    gray[i] = 1 - lum;
  }
  return downsampleGray(gray, width, MNIST_GRID);
}

export function DrawDemoPanel() {
  const ir = useGraphStore((s) => s.ir);
  // 学習更新(metrics/gate_dist)のたびに増える。ライブ再評価のトリガに使う。
  const trainingVersion = useTrainingStore((s) => s.version);

  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const drawingRef = useRef(false);
  const lastPtRef = useRef<{ x: number; y: number } | null>(null);
  const throttleRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const [threshold, setThreshold] = useState(0.5);
  const [mode, setMode] = useState<InferMode>('probe');
  const [grid, setGrid] = useState<number[]>(() => new Array<number>(MNIST_GRID * MNIST_GRID).fill(0));
  const [probs, setProbs] = useState<number[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [inferring, setInferring] = useState(false);
  // ライブ評価が重すぎてprobeへ切り替わった場合の注記
  const [fellBack, setFellBack] = useState(false);

  const enabled = isMnistShapedIR(ir);
  const bits = useMemo(() => binarize(grid, threshold), [grid, threshold]);
  const hasInk = useMemo(() => bits.some((b) => b === 1), [bits]);

  // --- キャンバス初期化(白背景) ---
  const clearCanvas = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    ctx.fillStyle = '#ffffff';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    setGrid(new Array<number>(MNIST_GRID * MNIST_GRID).fill(0));
    setProbs(null);
    setError(null);
    setFellBack(false);
  }, []);

  useEffect(() => {
    clearCanvas();
  }, [clearCanvas]);

  // --- 描画 ---
  const pointFromEvent = (e: React.PointerEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return { x: 0, y: 0 };
    const rect = canvas.getBoundingClientRect();
    return {
      x: ((e.clientX - rect.left) / rect.width) * canvas.width,
      y: ((e.clientY - rect.top) / rect.height) * canvas.height,
    };
  };

  const strokeTo = (x: number, y: number) => {
    const canvas = canvasRef.current;
    const ctx = canvas?.getContext('2d');
    if (!ctx) return;
    ctx.strokeStyle = '#000000';
    ctx.lineWidth = PEN_WIDTH;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.beginPath();
    const from = lastPtRef.current ?? { x, y };
    ctx.moveTo(from.x, from.y);
    ctx.lineTo(x, y);
    ctx.stroke();
    lastPtRef.current = { x, y };
  };

  const handlePointerDown = (e: React.PointerEvent<HTMLCanvasElement>) => {
    if (!enabled) return;
    drawingRef.current = true;
    const p = pointFromEvent(e);
    lastPtRef.current = p;
    // 点打ちも描けるよう同点へストローク
    strokeTo(p.x, p.y);
    canvasRef.current?.setPointerCapture(e.pointerId);
  };

  const handlePointerMove = (e: React.PointerEvent<HTMLCanvasElement>) => {
    if (!drawingRef.current) return;
    const p = pointFromEvent(e);
    strokeTo(p.x, p.y);
  };

  const endStroke = () => {
    if (!drawingRef.current) return;
    drawingRef.current = false;
    lastPtRef.current = null;
    const canvas = canvasRef.current;
    if (canvas) setGrid(readGrid(canvas));
  };

  // --- 推論本体 ---
  const runLive = useCallback(
    (inputBits: number[]): { probs: number[]; fellBack: boolean } | Promise<{ probs: number[]; fellBack: boolean }> => {
      if (!ir) throw new Error('IR未設定');
      const outNode = outputSourceNodeId(ir);
      // ブラウザ評価が重い場合はprobe APIへフォールバック
      if (browserEvalCost(ir) > DETAIL_TOTAL_MAX) {
        if (!outNode) throw new Error('出力ノードが特定できません');
        return probeInference(inputBits, outNode, 'soft').then((scores) => ({
          probs: normalizeProbs(scores),
          fellBack: true,
        }));
      }
      const portId = ir.inputs[0].id;
      const result = evalCircuit(ir, { [portId]: inputBits }, 'soft', readTrainingProbs(ir));
      return { probs: normalizeProbs(result.output), fellBack: false };
    },
    [ir],
  );

  const runInference = useCallback(async () => {
    if (!ir || !enabled) return;
    setInferring(true);
    setError(null);
    try {
      if (mode === 'probe') {
        const outNode = outputSourceNodeId(ir);
        if (!outNode) throw new Error('出力ノードが特定できません');
        const scores = await probeInference(bits, outNode, 'soft');
        setProbs(normalizeProbs(scores));
        setFellBack(false);
      } else {
        const res = await runLive(bits);
        setProbs(res.probs);
        setFellBack(res.fellBack);
      }
    } catch (e) {
      setProbs(null);
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setInferring(false);
    }
  }, [ir, enabled, mode, bits, runLive]);

  // --- 学習中ライブ更新: version変化でスロットリング再評価(liveモード・描画済みのみ) ---
  useEffect(() => {
    if (mode !== 'live' || !enabled || !hasInk) return;
    if (throttleRef.current) return;
    throttleRef.current = setTimeout(() => {
      throttleRef.current = null;
      void runInference();
    }, LIVE_THROTTLE_MS);
    return () => {
      if (throttleRef.current) {
        clearTimeout(throttleRef.current);
        throttleRef.current = null;
      }
    };
  }, [trainingVersion, mode, enabled, hasInk, bits, runInference]);

  if (!enabled) {
    return (
      <div
        className="rounded border border-amber-500/40 bg-amber-500/10 px-3 py-4 text-xs text-amber-700"
        data-testid="draw-demo-disabled"
      >
        手描きデモは入力784(28×28)のMNIST系モデルで利用できます。MNIST系モデルを読み込んでください。
      </div>
    );
  }

  const predicted = probs ? argmax(probs) : -1;

  return (
    <div className="flex flex-wrap gap-4" data-testid="draw-demo">
      {/* 左: 描画キャンバス + コントロール */}
      <div className="flex flex-col gap-2">
        <canvas
          ref={canvasRef}
          width={CANVAS_SIZE}
          height={CANVAS_SIZE}
          className="touch-none rounded border border-border bg-white"
          style={{ width: CANVAS_SIZE, height: CANVAS_SIZE, cursor: 'crosshair' }}
          onPointerDown={handlePointerDown}
          onPointerMove={handlePointerMove}
          onPointerUp={endStroke}
          onPointerLeave={endStroke}
          data-testid="draw-canvas"
        />
        <div className="flex items-center gap-2 text-xs">
          <button
            type="button"
            className="rounded border border-border px-2 py-1 hover:bg-accent"
            onClick={clearCanvas}
            data-testid="draw-clear"
          >
            クリア
          </button>
          <button
            type="button"
            className="rounded bg-primary px-3 py-1 font-semibold text-primary-foreground disabled:opacity-50"
            onClick={runInference}
            disabled={inferring || !hasInk}
            data-testid="draw-infer"
          >
            {inferring ? '推論中…' : '推論実行'}
          </button>
        </div>
        <label className="flex flex-col gap-1 text-xs">
          <span className="text-muted-foreground">2値化しきい値: {threshold.toFixed(2)}</span>
          <input
            type="range"
            min={0}
            max={1}
            step={0.05}
            value={threshold}
            onChange={(e) => setThreshold(Number(e.target.value))}
            data-testid="draw-threshold"
          />
        </label>
        <div className="flex items-center gap-2 text-xs">
          <span className="text-muted-foreground">推論経路</span>
          <label className="flex items-center gap-1">
            <input
              type="radio"
              name="draw-mode"
              checked={mode === 'probe'}
              onChange={() => setMode('probe')}
              data-testid="draw-mode-probe"
            />
            probe API
          </label>
          <label className="flex items-center gap-1">
            <input
              type="radio"
              name="draw-mode"
              checked={mode === 'live'}
              onChange={() => setMode('live')}
              data-testid="draw-mode-live"
            />
            ライブ評価(学習中)
          </label>
        </div>
        {fellBack && (
          <span className="text-[10px] text-amber-700" data-testid="draw-fellback">
            大規模モデルのためprobe APIへフォールバックしました
          </span>
        )}
      </div>

      {/* 中: 2値化プレビュー(28×28) */}
      <div className="flex flex-col gap-1">
        <span className="text-xs text-muted-foreground">2値化後(回路への入力)</span>
        <div
          className="grid border border-border"
          style={{
            gridTemplateColumns: `repeat(${MNIST_GRID}, 4px)`,
            gridTemplateRows: `repeat(${MNIST_GRID}, 4px)`,
          }}
          data-testid="draw-binary-preview"
        >
          {bits.map((b, i) => (
            <div
              key={i}
              style={{ width: 4, height: 4, background: b ? '#111827' : '#f3f4f6' }}
            />
          ))}
        </div>
      </div>

      {/* 右: クラス確率バー(0-9) */}
      <div className="flex min-w-[200px] flex-1 flex-col gap-1" data-testid="draw-probs">
        <span className="text-xs text-muted-foreground">
          クラス確率{predicted >= 0 && <> / 予測 <b>{predicted}</b></>}
        </span>
        {error && (
          <div className="rounded border border-red-500/40 bg-red-500/10 px-2 py-1 text-xs text-red-600" data-testid="draw-error">
            {error}
          </div>
        )}
        {(probs ?? new Array<number>(10).fill(0)).map((p, cls) => (
          <div key={cls} className="flex items-center gap-2 text-xs" data-testid="class-prob-bar">
            <span className="w-4 text-right tabular-nums">{cls}</span>
            <div className="h-3 flex-1 overflow-hidden rounded bg-muted">
              <div
                className={cls === predicted ? 'h-full bg-primary' : 'h-full bg-blue-400/60'}
                style={{ width: `${Math.round(p * 100)}%` }}
              />
            </div>
            <span className="w-10 text-right tabular-nums text-muted-foreground">
              {(p * 100).toFixed(1)}%
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}
