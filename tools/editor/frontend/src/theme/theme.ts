// CSS変数(デザイントークン)をCSS外の描画系(canvas 2D / SVG / deck.gl / uPlot)へ
// 供給するモジュール。トークンは index.css の :root にhex表記で定義されている前提。
// getComputedStyleは比較的重いため、初回読取後はキャッシュする。

export type RGB = [number, number, number];
export type RGBA = [number, number, number, number];

const cache = new Map<string, string>();

/** トークン名('--gate-0'等)の生の値(hex文字列)を返す。 */
export function cssVar(name: string): string {
  const hit = cache.get(name);
  if (hit !== undefined) return hit;
  const value = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  cache.set(name, value);
  return value;
}

/** '#rrggbb' をRGBタプルへ変換する。トークン未定義や不正値はフォールバック色。 */
export function cssVarRGB(name: string, fallback: RGB = [100, 100, 100]): RGB {
  const hex = cssVar(name);
  const m = /^#([0-9a-fA-F]{6})$/.exec(hex);
  if (!m) return fallback;
  const v = parseInt(m[1], 16);
  return [(v >> 16) & 0xff, (v >> 8) & 0xff, v & 0xff];
}

/** deck.gl用のRGBA配列を返す(alphaは0..255)。 */
export function cssVarRGBA(name: string, alpha = 255, fallback: RGB = [100, 100, 100]): RGBA {
  const [r, g, b] = cssVarRGB(name, fallback);
  return [r, g, b, alpha];
}

/** 将来のテーマ切替(トークン差替え)時にキャッシュを破棄するためのフック。 */
export function invalidateThemeCache(): void {
  cache.clear();
}
