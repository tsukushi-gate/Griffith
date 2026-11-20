// Unit定義レジストリのストア。
// アクティブIRの metadata.editor.units をフロント側で保持・編集し、graphStoreへ書き戻す。
// - Unit編集モード(editing)の下書き状態を持つ
// - upsertUnit で定義を保存し、既存インスタンスへ再展開を伝播する(graphStore経由)
// タブ切替(networkStore)でアクティブIRが変わったら loadFromActiveIr で同期する。

import { create } from 'zustand';
import {
  nextUnitId,
  reexpandInfoEmpty,
  unitsOf,
  validateUnit,
  type EditorUnit,
  type ReexpandResult,
  type UnitArg,
} from '../blocks/unitTypes';
import { useGraphStore } from './graphStore';

/** 新規Unitの空テンプレート(unit-in → lut_layer → unit-out の最小形)。 */
export function blankUnit(id: string): EditorUnit {
  return {
    id,
    name: '新しいUnit',
    description: '',
    args: [],
    template: {
      nodes: [
        { id: 'in', type: 'unit-in', params: {}, meta: { editor: { position: { x: 0, y: 0 } } } },
        {
          id: 'lut0',
          type: 'lut_layer',
          params: {
            in_width: 8,
            out_width: 8,
            n_inputs: 2,
            indices: [
              [0, 1],
              [2, 3],
            ],
            truth_tables: ['0x6'],
          },
          train_info: { parameterization: 'softmax', connections: 'random', selection: 'soft' },
          meta: { editor: { position: { x: 220, y: 0 } } },
        },
        { id: 'out', type: 'unit-out', params: {}, meta: { editor: { position: { x: 440, y: 0 } } } },
      ],
      edges: [
        { src: 'in', dst: 'lut0' },
        { src: 'lut0', dst: 'out' },
      ],
    },
  };
}

interface UnitState {
  units: EditorUnit[];
  editing: EditorUnit | null; // Unit編集モードで開いている下書き(null=非編集)
  /** アクティブIRのmetadataからunit定義を取り込む(タブ切替/モデル読込時)。 */
  loadFromActiveIr: () => void;
  /** 新規Unitの編集を開始する。 */
  startNew: () => void;
  /** 既存Unitの編集を開始する。 */
  startEdit: (id: string) => void;
  /** 編集中の下書きを部分更新する。 */
  patchEditing: (patch: Partial<EditorUnit>) => void;
  /** 引数宣言を丸ごと差し替える。 */
  setEditingArgs: (args: UnitArg[]) => void;
  /** 編集を破棄する。 */
  cancelEdit: () => void;
  /**
   * 編集中の定義を保存する。既存なら再展開伝播する。
   * 戻り値: エラー(バリデーション)または再展開結果。
   */
  saveEditing: () => { errors: string[]; reexpand: ReexpandResult };
  /** 定義を削除する(インスタンスは残る=展開済みノードはそのまま)。 */
  removeUnit: (id: string) => void;
}

export const useUnitStore = create<UnitState>((set, get) => ({
  units: [],
  editing: null,

  loadFromActiveIr: () => {
    const ir = useGraphStore.getState().ir;
    const next = ir ? unitsOf(ir) : [];
    // 参照が同じ(=metadata.editor.units未変更)なら再セットしない(編集中の再描画抑制)
    if (next !== get().units) set({ units: next });
  },

  startNew: () => {
    const id = nextUnitId(get().units);
    set({ editing: blankUnit(id) });
  },

  startEdit: (id) => {
    const unit = get().units.find((u) => u.id === id);
    if (unit) set({ editing: structuredClone(unit) });
  },

  patchEditing: (patch) => {
    const editing = get().editing;
    if (!editing) return;
    set({ editing: { ...editing, ...patch } });
  },

  setEditingArgs: (args) => {
    const editing = get().editing;
    if (!editing) return;
    set({ editing: { ...editing, args } });
  },

  cancelEdit: () => set({ editing: null }),

  saveEditing: () => {
    const editing = get().editing;
    if (!editing) return { errors: ['編集中のUnitがありません'], reexpand: reexpandInfoEmpty() };
    const errors = validateUnit(editing);
    if (errors.length > 0) return { errors, reexpand: reexpandInfoEmpty() };

    const units = get().units;
    const isExisting = units.some((u) => u.id === editing.id);
    const nextUnits = isExisting
      ? units.map((u) => (u.id === editing.id ? editing : u))
      : [...units, editing];

    // metadata.editor.units へ書き戻す(同期予約)。
    useGraphStore.getState().setUnits(nextUnits);
    // 既存インスタンスへ再展開を伝播する(新規なら対象なし)。
    const reexpand = isExisting
      ? useGraphStore.getState().reexpandUnitInstances(editing)
      : reexpandInfoEmpty();
    set({ units: nextUnits, editing: null });
    return { errors: [], reexpand };
  },

  removeUnit: (id) => {
    const nextUnits = get().units.filter((u) => u.id !== id);
    useGraphStore.getState().setUnits(nextUnits);
    set({ units: nextUnits, editing: null });
  },
}));

// アクティブIRが差し替わったら(モデル読込/タブ切替)Unit定義を自動同期する。
useGraphStore.subscribe((state, prev) => {
  if (state.ir !== prev.ir) useUnitStore.getState().loadFromActiveIr();
});
