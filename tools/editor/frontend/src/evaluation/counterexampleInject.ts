// 反例トレース注入。誤答サンプルの入力ビット列を playgroundStore へ注入し、
// EDITタブ + Probeレンズ + プレイグラウンド着色ON をまとめて適用する。
// コンポーネントと別ファイルにしてfast-refresh(react-refresh)制約を満たす。

import { toast } from 'sonner';
import { usePlaygroundStore } from '../playground/playgroundStore';
import { applyLens } from '../store/lens';
import type { EvalItem } from './api';

/** プレイグラウンドへ注入可能な入力ビット幅の上限(これを超える画像入力等は注入しない)。 */
export const INJECT_MAX_WIDTH = 24;

/**
 * 入力ビット列を playgroundStore へ注入し EDIT/Probe へ遷移する。
 * 注入した場合 true、画像入力/過大幅でスキップした場合 false を返す。
 */
export function injectCounterexample(item: EvalItem): boolean {
  if (!item.input_bits) {
    toast.warning('画像入力は反例トレースに注入できません(EDIT/Probeで手動確認してください)');
    return false;
  }
  if (item.input_bits.length > INJECT_MAX_WIDTH) {
    toast.warning(`入力幅${item.input_bits.length}bitは大きいため注入をスキップしました`);
    return false;
  }
  const k = item.input_bits.reduce((acc, b, i) => (b ? acc | (1 << i) : acc), 0);
  applyLens('probe');
  usePlaygroundStore.getState().pinPattern(k);
  return true;
}
