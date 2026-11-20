// 真理値表トグルの共有ヘルパ(TruthTableEditorと同じ二経路)。
// - ビルド済み: 単一ニューロンAPIでin-place同期(バックエンドがIRも更新、PUT不要)
// - 未ビルド(draft): IRのみ更新し既存デバウンスPUTで同期

import { postNodeTruthTable } from '../api/rest';
import { ttToggleBit } from '../ir/hexTT';
import { useGraphStore } from '../store/graphStore';

// コア規約のコピー(griffith_logic/ir/weights.py TT_EDITABLE)
const TT_EDITABLE = new Set(['softmax16g', 'independent', 'raw']);

/** parameterization×Nから実効パラメータ化名(softmax N=2は16ゲートモード) */
export function effectiveParam(param: string, nInputs: number): string {
  return param === 'softmax' && nInputs === 2 ? 'softmax16g' : param;
}

/** この層の真理値表がUI編集可能か */
export function canEditTT(param: string, nInputs: number): boolean {
  return TT_EDITABLE.has(effectiveParam(param, nInputs));
}

/** ニューロンの真理値表ビットkをトグルする(失敗時はthrow)。 */
export async function toggleNeuronBit(
  nodeId: string,
  neuron: number,
  hex: string,
  k: number,
): Promise<void> {
  const { needsBuild, updateLutParams } = useGraphStore.getState();
  const next = ttToggleBit(hex, k);
  if (!needsBuild) {
    await postNodeTruthTable(nodeId, neuron, next);
    updateLutParams(
      nodeId,
      (params) => {
        params.truth_tables[neuron] = next;
      },
      { sync: false },
    );
  } else {
    updateLutParams(nodeId, (params) => {
      params.truth_tables[neuron] = next;
    });
  }
}
