// 16ゲートの表示色をデザイントークン(--gate-0..15)から供給する。
// ゲートID/名前の規約は backend extract.py と同一だが、表示色はフロントテーマの所管。

import { cssVarRGB, type RGB } from './theme';

export const DEFAULT_GATE_COLOR: RGB = [100, 100, 200];

/** ゲートIDから表示色を返す。N>2はハッシュベースの色分け(白背景向けに彩度/明度を調整)。 */
export function gateColor(gateId: number, nInputs: number): RGB {
  if (nInputs === 2) {
    return cssVarRGB(`--gate-${gateId & 15}`, DEFAULT_GATE_COLOR);
  }
  // N>2: extractor.py移植のHSVハッシュ。白背景で判別できるよう s=0.55 / v=0.72
  const h = (gateId * 137 + 43) % 360;
  const s = 0.55;
  const v = 0.72;
  const c = v * s;
  const x = c * (1 - Math.abs(((h / 60) % 2) - 1));
  const m = v - c;
  let r: number;
  let g: number;
  let b: number;
  if (h < 60) [r, g, b] = [c, x, 0];
  else if (h < 120) [r, g, b] = [x, c, 0];
  else if (h < 180) [r, g, b] = [0, c, x];
  else if (h < 240) [r, g, b] = [0, x, c];
  else if (h < 300) [r, g, b] = [x, 0, c];
  else [r, g, b] = [c, 0, x];
  return [
    Math.round((r + m) * 255),
    Math.round((g + m) * 255),
    Math.round((b + m) * 255),
  ];
}
