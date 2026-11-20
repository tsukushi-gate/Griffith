// 手描きキャンバス推論デモの純関数群。
// キャンバス描画→グレースケール縮小→2値化→クラス確率正規化までを
// UI非依存の副作用なし関数として切り出す(vitestで単体検証する)。

import type { IRGraphJson } from '../ir/schema';

/** MNIST入力の1辺(28×28=784)。 */
export const MNIST_GRID = 28;
/** MNIST入力の平坦ビット幅。 */
export const MNIST_INPUT_WIDTH = MNIST_GRID * MNIST_GRID;

/**
 * 高解像度キャンバスのグレースケール強度([0,1]、1=描画あり)を
 * gridSize×gridSize へ平均プーリングで縮小する。
 *
 * @param gray 長さ srcSize*srcSize の強度配列(row-major、[0,1])
 * @param srcSize 元画像の1辺
 * @param gridSize 出力の1辺(既定=28)
 * @returns 長さ gridSize*gridSize の縮小強度([0,1])
 */
export function downsampleGray(
  gray: ArrayLike<number>,
  srcSize: number,
  gridSize: number = MNIST_GRID,
): number[] {
  if (gray.length !== srcSize * srcSize) {
    throw new Error(`downsampleGray: 入力長${gray.length}がsrcSize^2=${srcSize * srcSize}と不一致`);
  }
  if (gridSize <= 0 || srcSize <= 0) {
    throw new Error('downsampleGray: サイズは正の値が必要');
  }
  const out = new Array<number>(gridSize * gridSize).fill(0);
  const scale = srcSize / gridSize;
  for (let gy = 0; gy < gridSize; gy++) {
    for (let gx = 0; gx < gridSize; gx++) {
      // このセルが覆う元ピクセル範囲を平均する
      const x0 = Math.floor(gx * scale);
      const x1 = Math.max(x0 + 1, Math.floor((gx + 1) * scale));
      const y0 = Math.floor(gy * scale);
      const y1 = Math.max(y0 + 1, Math.floor((gy + 1) * scale));
      let sum = 0;
      let count = 0;
      for (let y = y0; y < y1 && y < srcSize; y++) {
        for (let x = x0; x < x1 && x < srcSize; x++) {
          sum += gray[y * srcSize + x];
          count++;
        }
      }
      out[gy * gridSize + gx] = count > 0 ? sum / count : 0;
    }
  }
  return out;
}

/**
 * グレースケール強度を閾値で2値化する。強度 > threshold のセルを1にする。
 *
 * @param gray 強度配列([0,1])
 * @param threshold しきい値([0,1])
 * @returns 0/1 のビット配列(gray と同じ長さ)
 */
export function binarize(gray: ArrayLike<number>, threshold: number): number[] {
  const out = new Array<number>(gray.length);
  for (let i = 0; i < gray.length; i++) {
    out[i] = gray[i] > threshold ? 1 : 0;
  }
  return out;
}

/**
 * クラススコア(group_sum等の出力)を数値安定なsoftmaxで確率へ正規化する。
 * 全要素が有限でない/空の場合は一様分布を返す。
 *
 * @param scores クラススコア列
 * @returns 合計1・各要素[0,1]の確率列(scores と同じ長さ)
 */
export function normalizeProbs(scores: ArrayLike<number>): number[] {
  const n = scores.length;
  if (n === 0) return [];
  let max = -Infinity;
  for (let i = 0; i < n; i++) {
    const v = scores[i];
    if (Number.isFinite(v) && v > max) max = v;
  }
  if (!Number.isFinite(max)) {
    // 全て非有限 → 一様分布
    return new Array<number>(n).fill(1 / n);
  }
  const exp = new Array<number>(n);
  let sum = 0;
  for (let i = 0; i < n; i++) {
    const v = scores[i];
    const e = Number.isFinite(v) ? Math.exp(v - max) : 0;
    exp[i] = e;
    sum += e;
  }
  if (sum <= 0) return new Array<number>(n).fill(1 / n);
  for (let i = 0; i < n; i++) exp[i] /= sum;
  return exp;
}

/** 引数最大(確率が最大)のインデックスを返す。空なら-1。 */
export function argmax(values: ArrayLike<number>): number {
  let best = -1;
  let bestVal = -Infinity;
  for (let i = 0; i < values.length; i++) {
    if (values[i] > bestVal) {
      bestVal = values[i];
      best = i;
    }
  }
  return best;
}

/** IRの入力ポート幅の総和(平坦ビット幅)を返す。 */
export function irInputWidth(ir: IRGraphJson): number {
  return ir.inputs.reduce((acc, p) => acc + p.width, 0);
}

/** 現IRがMNIST系(単一入力ポート・幅784)かどうかを判定する。 */
export function isMnistShapedIR(ir: IRGraphJson | null | undefined): boolean {
  if (!ir) return false;
  return ir.inputs.length === 1 && ir.inputs[0].width === MNIST_INPUT_WIDTH;
}

/** 出力ポート(先頭)へ接続されたソースノードidを返す(probe応答から出力値を取り出す用)。 */
export function outputSourceNodeId(ir: IRGraphJson): string | null {
  const port = ir.outputs[0];
  if (!port) return null;
  const edge = ir.edges.find((e) => e.dst === port.id);
  return edge ? edge.src : null;
}

/**
 * ブラウザ内評価(evalCircuit)の負荷指標としてlut_layer出力幅の総和を返す。
 * これが上限を超える場合はライブ評価を諦めてprobe APIへフォールバックする。
 */
export function browserEvalCost(ir: IRGraphJson): number {
  let cost = 0;
  for (const node of ir.nodes) {
    if (node.type === 'lut_layer') {
      const outWidth = (node.params as { out_width?: number }).out_width ?? 0;
      cost += outWidth;
    }
  }
  return cost;
}
