// 単一lut_layerのドリルダウンレイアウト。
// 入力列(左)×ゲート列(右)の2列に配置し、IRのindicesインラインから
// エッジを構築する。訓練中はlayer_summary(gate_ids/confidence)で再着色する。

import { blendToBg, gateColor, ttToGateIdN2 } from '../ir/gates';
import { ttParse } from '../ir/hexTT';
import type { IRNodeJson, LutLayerParams } from '../ir/schema';
import type { LayerSummaryFrame } from '../store/trainingStore';
import { cssVarRGB, cssVarRGBA } from '../theme/theme';

const SPACING = 4;            // ノード間隔(common単位、半径1.2に対して)
const MIN_COLUMN_GAP = 120;   // 入力列⇔ゲート列の最小距離
const COLUMN_GAP_RATIO = 0.4; // 列間距離 = 列高さ × この係数
const CONF_FLOOR = 0.15;      // 確信度0でも沈み切らない下限

export interface L0Node {
  position: [number, number];
  gateId: number;
  confidence: number;
  color: [number, number, number, number];
  index: number; // 入力番号 or ニューロン番号
  kind: 'input' | 'gate';
}

export interface L0Edge {
  source: [number, number];
  target: [number, number];
  index: number;
}

export interface L0Data {
  nodes: L0Node[];
  edges: L0Edge[];
  nInputs: number;
  inWidth: number;
  outWidth: number;
  bounds: { xMin: number; xMax: number; yMin: number; yMax: number };
}

/** IRの真理値表hexからゲートIDを導出(backend/extract.pyのget_gate_idsと同規約)。 */
function gateIdFromTT(hex: string, nInputs: number): number {
  const tt = ttParse(hex);
  if (nInputs === 2) return ttToGateIdN2(Number(tt & 0xfn));
  return Number(tt & 0xffffn); // N>2は色分け用の安定なintに丸める
}

/** i番目(全count個)を高さHの列上へ等間隔配置する。 */
function columnY(i: number, count: number, height: number): number {
  if (count <= 1) return height / 2;
  return (i / (count - 1)) * height;
}

/**
 * lut_layerノードのIR paramsからドリルダウン表示データを構築する。
 * summaryがあればgate_ids/confidenceを優先(訓練中のライブ再着色)。
 */
export function buildL0Data(
  node: IRNodeJson,
  summary: LayerSummaryFrame | undefined,
): L0Data {
  const p = node.params as unknown as LutLayerParams;
  const { in_width: inWidth, out_width: outWidth, n_inputs: nInputs } = p;

  const height = Math.max(inWidth, outWidth) * SPACING;
  const gap = Math.max(height * COLUMN_GAP_RATIO, MIN_COLUMN_GAP);
  // キャンバス背景(確信度ブレンド先)と入力ノード色はトークンから取得
  const bg = cssVarRGB('--canvas-bg', [255, 255, 255]);
  const inputColor = cssVarRGBA('--slot-3');

  const nodes: L0Node[] = [];
  for (let i = 0; i < inWidth; i++) {
    nodes.push({
      position: [0, columnY(i, inWidth, height)],
      gateId: -1,
      confidence: 1,
      color: inputColor,
      index: i,
      kind: 'input',
    });
  }

  const useSummary = summary !== undefined && summary.gate_ids.length === outWidth;
  for (let o = 0; o < outWidth; o++) {
    const gateId = useSummary
      ? summary.gate_ids[o]
      : gateIdFromTT(p.truth_tables[o] ?? '0x0', nInputs);
    const conf = useSummary
      ? Math.min(Math.max(summary.confidence[o] ?? 1, 0), 1)
      : 1;
    const rgb = blendToBg(
      gateColor(gateId, nInputs),
      bg,
      CONF_FLOOR + (1 - CONF_FLOOR) * conf,
    );
    nodes.push({
      position: [gap, columnY(o, outWidth, height)],
      gateId,
      confidence: conf,
      color: [rgb[0], rgb[1], rgb[2], 255],
      index: o,
      kind: 'gate',
    });
  }

  // エッジ: indices[j][o] = ニューロンoのスロットjが見る入力番号
  const edges: L0Edge[] = [];
  for (let o = 0; o < outWidth; o++) {
    for (let j = 0; j < nInputs; j++) {
      const src = p.indices[j]?.[o];
      if (typeof src !== 'number' || src < 0 || src >= inWidth) continue;
      edges.push({
        source: [0, columnY(src, inWidth, height)],
        target: [gap, columnY(o, outWidth, height)],
        index: o * nInputs + j,
      });
    }
  }

  return {
    nodes,
    edges,
    nInputs,
    inWidth,
    outWidth,
    bounds: { xMin: 0, xMax: gap, yMin: 0, yMax: height },
  };
}
