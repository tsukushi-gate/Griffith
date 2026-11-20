// 誤差逆伝播の粒子オーバーレイ。
// React Flowビューポートに追従するcanvasを1枚重ね、rAFループで
// アクティブ層の詳細エッジパスに沿って粒子を逆流(target→source=勾配の流れ)させる。
// 粒子の密度・速度は edgeIntensity(層内max正規化のedge_grad)に比例する。
//
// エッジ幾何の取得方式: React Flowが描画した実SVGパス
// (svg.react-flow__edge[data-id] 内の path.react-flow__edge-path、フロー座標系)を
// getPointAtLengthで等間隔サンプルしポリライン化してキャッシュする。
// ノード位置からのベジェ近似ではなくDOM取得を選んだ理由:
// 子ゲートノードはparentId相対座標のため絶対位置+ハンドル位置の再計算が必要で、
// さらにReact Flow内部のベジェ制御点ロジックの複製が要る。実パスなら描画と
// 完全一致し、サンプルは層切替時に1回だけなのでrAF内のDOMアクセスもゼロにできる。
//
// React再レンダを介さない設計: 粒子状態はすべてeffect内ローカル、
// bpAnim変更はzustandのトランジェント購読(subscribe)で受け、
// ビューポートはrAF内で getViewport() を直接読む。

import { useReactFlow } from '@xyflow/react';
import { useEffect, useRef } from 'react';
import { useGraphStore } from '../store/graphStore';
import { useViewStore, type BpAnimState } from '../store/viewStore';
import { cssVarRGB } from '../theme/theme';
import { buildPolyline, pointAtFraction, type Polyline } from './particlePath';

/** 粒子総数の上限(60fps維持のための予算) */
const MAX_PARTICLES = 500;
/** エッジパスのサンプル点数(ベジェを折れ線近似する分割数) */
const PATH_SAMPLES = 24;
/** rAF間隔の上限(タブ非表示復帰などでの粒子ワープ防止) */
const MAX_FRAME_DT_MS = 64;
/** 走行時間: 強度0で900ms、強度1で400ms(速度∝強度) */
const TRAVEL_MS_SLOW = 900;
const TRAVEL_MS_RANGE = 500;
/** 詳細エッジのスポーンレート(個/秒/エッジ、密度∝強度) */
const DETAIL_RATE_SCALE = 8;
/** スポーン対象とする最小強度(ほぼ勾配ゼロのエッジは流さない) */
const MIN_INTENSITY = 0.02;
/** 集約エッジ(未展開層のフォールバック)のスポーンレート */
const AGG_RATE_BASE = 10;
const AGG_RATE_SCALE = 30;

interface Track {
  line: Polyline;
  intensity: number;
  /** スポーンレート(個/秒) */
  rate: number;
  /** スポーン積算器(1を超えた分だけ生成) */
  spawnAcc: number;
}

interface Particle {
  track: number;
  /** 弧長割合(1=target側から出発し、0=source側で消滅=勾配の逆流) */
  f: number;
  /** fの減少速度(1/ms) */
  speed: number;
}

/** SVGパスを等間隔サンプルした点列(フロー座標)を返す。 */
function samplePath(path: SVGPathElement): Float32Array {
  const total = path.getTotalLength();
  if (!(total > 0)) return new Float32Array(0);
  const pts = new Float32Array((PATH_SAMPLES + 1) * 2);
  for (let i = 0; i <= PATH_SAMPLES; i++) {
    const p = path.getPointAtLength((i / PATH_SAMPLES) * total);
    pts[i * 2] = p.x;
    pts[i * 2 + 1] = p.y;
  }
  return pts;
}

/** DOM上の全React Flowエッジパスをdata-idで索引する(属性セレクタのエスケープ回避)。 */
function indexEdgePaths(container: HTMLElement): Map<string, SVGPathElement> {
  const map = new Map<string, SVGPathElement>();
  for (const el of container.querySelectorAll('.react-flow__edge')) {
    const id = el.getAttribute('data-id');
    const path = el.querySelector<SVGPathElement>('.react-flow__edge-path');
    if (id && path) map.set(id, path);
  }
  return map;
}

export function GradParticleOverlay() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const { getViewport } = useReactFlow();

  useEffect(() => {
    const canvas = canvasRef.current;
    const container = canvas?.closest<HTMLElement>('.react-flow') ?? null;
    const ctx = canvas?.getContext('2d') ?? null;
    if (!canvas || !container || !ctx) return;

    let tracks: Track[] = [];
    let particles: Particle[] = [];
    const [cr, cg, cb] = cssVarRGB('--chart-loss', [235, 104, 52]);

    /** アクティブ層の変化でエッジ幾何と強度を再構築する(rAF外で1回だけ)。 */
    const rebuildTracks = (anim: BpAnimState) => {
      tracks = [];
      particles = [];
      const { frame, activeLayer, edgeIntensity } = anim;
      if (!frame || !activeLayer) return;

      const pathById = indexEdgePaths(container);
      const addTrack = (path: SVGPathElement, intensity: number, rate: number) => {
        const line = buildPolyline(samplePath(path));
        if (line) tracks.push({ line, intensity, rate, spawnAcc: Math.random() });
      };

      // 1) 詳細エッジ(層が展開されている場合): d:${layerId}:${neuron}:${slot}
      if (edgeIntensity) {
        const prefix = `d:${activeLayer}:`;
        for (const [id, intensity] of edgeIntensity) {
          if (!id.startsWith(prefix) || intensity < MIN_INTENSITY) continue;
          const path = pathById.get(id);
          if (path) addTrack(path, intensity, DETAIL_RATE_SCALE * intensity);
        }
      }
      if (tracks.length > 0) return;

      // 2) フォールバック: 未展開層は抽象層ノード間の集約エッジ上を
      //    層勾配ノルム(全層max正規化)の強度で流す
      const norms = Object.values(frame.layer_grad_norm);
      const maxNorm = Math.max(...norms, 1e-12);
      const intensity = Math.min(
        Math.max((frame.layer_grad_norm[activeLayer] ?? 0) / maxNorm, 0),
        1,
      );
      for (const e of useGraphStore.getState().edges) {
        if (e.target !== activeLayer) continue;
        const path = pathById.get(e.id);
        if (path) addTrack(path, intensity, AGG_RATE_BASE + AGG_RATE_SCALE * intensity);
      }
    };

    rebuildTracks(useViewStore.getState().bpAnim);
    // トランジェント購読: 発光層/トレース/強度Mapの実質変化時のみ再構築
    // (再生ティックごとのbpAnimオブジェクト差し替えでは粒子を消さない)
    const unsubscribe = useViewStore.subscribe((state, prev) => {
      const a = state.bpAnim;
      const b = prev.bpAnim;
      if (
        a.activeLayer !== b.activeLayer ||
        a.frame !== b.frame ||
        a.edgeIntensity !== b.edgeIntensity
      ) {
        rebuildTracks(a);
      }
    });

    let raf = 0;
    let last = performance.now();
    const pos = { x: 0, y: 0 };

    const loop = (now: number) => {
      raf = requestAnimationFrame(loop);
      const dt = Math.min(now - last, MAX_FRAME_DT_MS);
      last = now;

      // キャンバス実寸をコンテナへ追従(ResizeObserver不要の毎フレーム比較)
      const dpr = window.devicePixelRatio || 1;
      const w = Math.round(container.clientWidth * dpr);
      const h = Math.round(container.clientHeight * dpr);
      if (canvas.width !== w || canvas.height !== h) {
        canvas.width = w;
        canvas.height = h;
      }
      ctx.setTransform(1, 0, 0, 1, 0, 0);
      ctx.clearRect(0, 0, w, h);
      if (tracks.length === 0) return;

      // スポーン(密度∝強度、総数上限あり)
      for (let i = 0; i < tracks.length && particles.length < MAX_PARTICLES; i++) {
        const tr = tracks[i];
        tr.spawnAcc += (tr.rate * dt) / 1000;
        while (tr.spawnAcc >= 1 && particles.length < MAX_PARTICLES) {
          tr.spawnAcc -= 1;
          const travelMs = TRAVEL_MS_SLOW - TRAVEL_MS_RANGE * tr.intensity;
          particles.push({ track: i, f: 1, speed: 1 / travelMs });
        }
      }

      // 前進(f減少=逆流)+描画。フロー座標系をctx.setTransformで直接使う
      const vp = getViewport();
      ctx.setTransform(dpr * vp.zoom, 0, 0, dpr * vp.zoom, dpr * vp.x, dpr * vp.y);
      let alive = 0;
      for (const pa of particles) {
        pa.f -= pa.speed * dt;
        if (pa.f <= 0) continue; // source側へ到達した粒子は消滅
        particles[alive++] = pa;
        const tr = tracks[pa.track];
        pointAtFraction(tr.line, pa.f, pos);
        // 画面上でほぼ一定サイズになるようzoomで割る(強度で微増)
        const radius = (1.5 + 2.5 * tr.intensity) / vp.zoom;
        ctx.beginPath();
        ctx.arc(pos.x, pos.y, radius, 0, Math.PI * 2);
        ctx.fillStyle = `rgba(${cr}, ${cg}, ${cb}, ${0.45 + 0.55 * tr.intensity})`;
        ctx.fill();
      }
      particles.length = alive;
    };
    raf = requestAnimationFrame(loop);

    return () => {
      cancelAnimationFrame(raf);
      unsubscribe();
    };
  }, [getViewport]);

  return (
    <canvas
      ref={canvasRef}
      className="pointer-events-none absolute inset-0"
      style={{ zIndex: 4 }}
      aria-hidden
    />
  );
}
