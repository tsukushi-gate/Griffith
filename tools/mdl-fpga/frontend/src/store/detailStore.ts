// ゲート展開用のノード params キャッシュ(非永続。展開 id は viewerStore が永続)。
import { create } from 'zustand';
import { getNodeDetail } from '../api/rest';

export interface LutParams {
  in_width: number;
  out_width: number;
  n_inputs: number;
  indices: number[][]; // [slot][gate] → 入力バスの bit 番号
  truth_tables: (string | number)[];
}

interface DetailState {
  params: Record<string, LutParams>;
  errors: Record<string, string>;
  fetchParams: (nodeId: string) => Promise<void>;
}

export const useDetailStore = create<DetailState>((set, get) => ({
  params: {},
  errors: {},
  fetchParams: async (nodeId) => {
    if (get().params[nodeId]) return;
    try {
      const detail = await getNodeDetail(nodeId);
      set({
        params: {
          ...get().params,
          [nodeId]: detail.params as unknown as LutParams,
        },
      });
    } catch (e) {
      set({ errors: { ...get().errors, [nodeId]: String(e) } });
    }
  },
}));

export function ttToInt(tt: string | number): number {
  return typeof tt === 'number' ? tt : parseInt(tt, 16);
}
