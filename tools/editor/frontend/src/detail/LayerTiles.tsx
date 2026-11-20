// L1(中間LOD)のニューロン集約タイル(§3)。lut_layerノード内に、各ニューロンを
// 小セルで並べ、セル色=確信度(低=迷い/赤 → 高=確信/緑)で表示する。
// データは trainingStore の layer_summary(useLayerView.confidence)を流用し、
// ニューロン数が上限を超えたら等間隔で間引く(軽量なCSSグリッド描画)。

import { useMemo } from 'react';
import { useLayerView } from '../store/selectors';

/** 表示するタイルセル数の上限(超過分は等間隔サンプリングで間引く) */
const TILE_MAX_CELLS = 64;
/** セルの一辺(px)。ノード幅(約176px)内に収まるよう固定小サイズにする */
const TILE_CELL_PX = 12;

interface Props {
  layerId: string;
  outWidth: number;
}

/** out_width個のニューロンから、確信度配列(なければnull)を最大cap個へ間引く。 */
function sampleTiles(
  outWidth: number,
  confidence: number[] | null,
  cap: number,
): (number | null)[] {
  const shown = Math.min(outWidth, cap);
  if (shown <= 0) return [];
  const out: (number | null)[] = [];
  for (let i = 0; i < shown; i++) {
    // 等間隔サンプリング(端を含む)
    const src = outWidth <= cap ? i : Math.round((i * (outWidth - 1)) / (shown - 1 || 1));
    out.push(confidence?.[src] ?? null);
  }
  return out;
}

/** 確信度[0,1]→色。低=迷い(赤)〜高=確信(緑)。データ無しはニュートラル。 */
function tileColor(c: number | null): string {
  if (c === null || Number.isNaN(c)) return 'var(--border)';
  const t = Math.min(Math.max(c, 0), 1);
  return `hsl(${Math.round(140 * t)}, 70%, 52%)`;
}

export function LayerTiles({ layerId, outWidth }: Props) {
  const view = useLayerView(layerId);
  const cells = useMemo(
    () => sampleTiles(outWidth, view.confidence, TILE_MAX_CELLS),
    [outWidth, view.confidence],
  );
  const cols = Math.max(Math.ceil(Math.sqrt(cells.length)), 1);

  return (
    <div
      className="mt-1 grid gap-[1px] rounded bg-[var(--panel-bg)] p-[2px]"
      style={{ gridTemplateColumns: `repeat(${cols}, ${TILE_CELL_PX}px)` }}
      title={`${outWidth}ニューロン ・ セル色=確信度(赤=迷い / 緑=確信)`}
    >
      {cells.map((c, i) => (
        <div
          key={i}
          className="rounded-[1px]"
          style={{ width: TILE_CELL_PX, height: TILE_CELL_PX, backgroundColor: tileColor(c) }}
        />
      ))}
    </div>
  );
}
