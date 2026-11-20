// エンジン共通インターフェース(WB-4/5/6)。
// predictは「int16復元済み生値 → クラス予測」のエンドツーエンド
// (前処理=標準化/thermometerを含む。実運用のセンサ入力経路と同じ)。
import type { IRGraphJson, LutLayerParams } from '../shared';
import type { TestData } from '../types';

export interface Engine {
  readonly name: string;
  /** [from, from+count) のサンプルを推論しクラス予測を返す */
  predict(data: TestData, from: number, count: number): Promise<Int32Array>;
  /** テスト用: クラススコア(整数和)を返す(DLGN系のみ) */
  scores?(data: TestData, from: number, count: number): Int32Array[];
}

/** シリアルdense DLGN(flatten*→lut_layer+→group_sum)のレイヤ抽出結果 */
export interface DenseLayers {
  inWidth: number;
  luts: LutLayerParams[];
  groupK: number;
}

/** IRからシリアルdense構成を抽出する(対応外のIRはthrow)。
    エッジのdst_slot=0直列のみ対応(uci-har系train.pyのexport_ir出力形)。 */
export function extractDense(ir: IRGraphJson): DenseLayers {
  if (ir.inputs.length !== 1) {
    throw new Error(`単一入力ポートのみ対応(実際 ${ir.inputs.length})`);
  }
  // src→dstの直列鎖をたどる
  const bySrc = new Map<string, string>();
  for (const e of ir.edges) {
    if (bySrc.has(e.src)) throw new Error(`分岐エッジは非対応: ${e.src}`);
    bySrc.set(e.src, e.dst);
  }
  const nodeById = new Map(ir.nodes.map((n) => [n.id, n]));
  const luts: LutLayerParams[] = [];
  let groupK = -1;
  let cur = bySrc.get(ir.inputs[0].id);
  while (cur !== undefined && nodeById.has(cur)) {
    const node = nodeById.get(cur)!;
    if (node.type === 'lut_layer') {
      luts.push(node.params as unknown as LutLayerParams);
    } else if (node.type === 'group_sum') {
      groupK = (node.params as { k: number }).k;
    } else if (node.type !== 'flatten') {
      throw new Error(`未対応ノード型: ${node.type}(dense直列のみ)`);
    }
    cur = bySrc.get(node.id);
  }
  if (luts.length === 0 || groupK <= 0) {
    throw new Error('lut_layer+group_sumの直列構成が見つかりません');
  }
  return { inWidth: ir.inputs[0].width, luts, groupK };
}
