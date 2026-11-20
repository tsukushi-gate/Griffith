// 学習監視スロット(下部ドック「タイムライン」タブ)。
// 上段: タイムラインスクラバー(step巻き戻し・再生・branch再学習)
// 中段: 誤差逆伝播リプレイ(層順の勾配フロー再生)
// 下段: 層別activationsパネル(平均活性ヒートストリップ+飽和率ゲージ)。
//   live時はWSのactivationsフレーム、replay時は履歴スナップショットのact_meanを使う
//   (useLayerViewがlive/replayを自動解決する)。

import { useEffect, useRef } from 'react';
import { useGraphStore } from '../store/graphStore';
import { useLayerView } from '../store/selectors';
import { useTrainingStore } from '../store/trainingStore';
import { cssVar } from '../theme/theme';
import { BackpropPlayer } from '../training/BackpropPlayer';
import { TimelineScrubber } from '../training/TimelineScrubber';

// 飽和判定: |活性-0.5| がこの帯域を超えたニューロンを「飽和」とみなす
// (backend monitor.py build_activations と同一定義)
const SATURATION_BAND = 0.45;
const STRIP_HEIGHT_PX = 12;

/** 平均活性から飽和率(0/1に張り付いたニューロンの割合)を計算する。 */
function saturationOf(act: Float32Array): number {
  if (act.length === 0) return 0;
  let n = 0;
  for (const v of act) {
    if (Math.abs(v - 0.5) > SATURATION_BAND) n++;
  }
  return n / act.length;
}

/** ニューロン別平均活性のヒートストリップ(1ニューロン=1px列、白→primary濃淡)。 */
function ActivationStrip({ act }: { act: Float32Array }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    canvas.width = act.length;
    canvas.height = 1;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    const primary = cssVar('--primary');
    ctx.fillStyle = '#ffffff';
    ctx.fillRect(0, 0, act.length, 1);
    ctx.fillStyle = primary;
    for (let i = 0; i < act.length; i++) {
      ctx.globalAlpha = Math.min(1, Math.max(0, act[i]));
      ctx.fillRect(i, 0, 1, 1);
    }
    ctx.globalAlpha = 1;
  }, [act]);

  return (
    <canvas
      ref={canvasRef}
      className="min-w-0 flex-1 rounded-sm border"
      style={{ height: STRIP_HEIGHT_PX, imageRendering: 'pixelated' }}
    />
  );
}

/** 1レイヤ分の行: 名前+ヒートストリップ+飽和率ゲージ。 */
function ActivationRow({ layerId }: { layerId: string }) {
  const view = useLayerView(layerId);
  const act = view.actMean;
  const sat = act ? saturationOf(act) : null;

  return (
    <div className="flex items-center gap-2">
      <span
        className="w-32 shrink-0 truncate text-[11px] text-muted-foreground"
        title={layerId}
      >
        {layerId}
      </span>
      {act ? (
        <ActivationStrip act={act} />
      ) : (
        <span className="flex-1 text-[10px] text-muted-foreground">
          活性データ未受信(fullキャプチャ時のみ)
        </span>
      )}
      <div
        className="h-1.5 w-20 shrink-0 overflow-hidden rounded-full bg-muted"
        title="飽和率(平均活性が0/1に張り付いたニューロンの割合)"
      >
        {sat !== null && (
          <div
            className="h-full bg-[var(--status-warn)]"
            style={{ width: `${sat * 100}%` }}
          />
        )}
      </div>
      <span className="w-10 shrink-0 text-right text-[11px] tabular-nums">
        {sat !== null ? `${(sat * 100).toFixed(0)}%` : '-'}
      </span>
    </div>
  );
}

/** 層別activationsパネル。*/
function ActivationsPanel() {
  const sessionId = useTrainingStore((s) => s.sessionId);
  const nodes = useGraphStore((s) => s.nodes);
  const layerIds = nodes.filter((n) => n.type === 'lutLayer').map((n) => n.id);

  if (!sessionId || layerIds.length === 0) return null;
  return (
    <div className="flex flex-col gap-1.5">
      <div className="text-[11px] font-medium text-muted-foreground">
        活性モニタ(層別平均活性+飽和率)
      </div>
      {layerIds.map((id) => (
        <ActivationRow key={id} layerId={id} />
      ))}
    </div>
  );
}

export function MonitorSlot() {
  return (
    <div className="flex h-full flex-col overflow-y-auto">
      <TimelineScrubber />
      <div className="flex flex-col gap-3 px-3 pb-3">
        <BackpropPlayer />
        <ActivationsPanel />
      </div>
    </div>
  );
}
