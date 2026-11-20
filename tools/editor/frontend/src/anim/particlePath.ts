// 粒子オーバーレイのポリライン幾何。
// SVGエッジパスをgetPointAtLengthで等間隔サンプルした点列を弧長パラメータ化し、
// rAFループ内で弧長割合→座標の補間を(DOMアクセスなしで)高速に引けるようにする。

export interface Polyline {
  /** [x0,y0,x1,y1,...] のフロー座標点列 */
  pts: Float32Array;
  /** 各点までの累積弧長(cum[0]=0) */
  cum: Float32Array;
  /** 総弧長 */
  length: number;
}

/** 点列から累積弧長つきポリラインを構築する(点が2未満ならnull)。 */
export function buildPolyline(pts: Float32Array): Polyline | null {
  const n = pts.length / 2;
  if (n < 2) return null;
  const cum = new Float32Array(n);
  let total = 0;
  for (let i = 1; i < n; i++) {
    const dx = pts[i * 2] - pts[(i - 1) * 2];
    const dy = pts[i * 2 + 1] - pts[(i - 1) * 2 + 1];
    total += Math.hypot(dx, dy);
    cum[i] = total;
  }
  if (total <= 0) return null;
  return { pts, cum, length: total };
}

/**
 * 弧長割合f(0=始点,1=終点)の座標をoutへ書き込む(線形補間)。
 * fは[0,1]へクランプされる。セグメント数は高々数十なので線形走査で十分。
 */
export function pointAtFraction(
  line: Polyline,
  f: number,
  out: { x: number; y: number },
): void {
  const target = Math.min(Math.max(f, 0), 1) * line.length;
  const { pts, cum } = line;
  const n = cum.length;
  let i = 1;
  while (i < n - 1 && cum[i] < target) i++;
  const segLen = cum[i] - cum[i - 1];
  const t = segLen > 0 ? (target - cum[i - 1]) / segLen : 0;
  out.x = pts[(i - 1) * 2] + (pts[i * 2] - pts[(i - 1) * 2]) * t;
  out.y = pts[(i - 1) * 2 + 1] + (pts[i * 2 + 1] - pts[(i - 1) * 2 + 1]) * t;
}
