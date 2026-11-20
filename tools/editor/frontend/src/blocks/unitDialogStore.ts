// Unitインスタンス化のドロップ→引数ダイアログ連携store(blockDialogStoreと同型)。
// FlowCanvasのonDropが unitId とドロップ位置を渡し、ダイアログ本体(UnitInstanceDialog)は
// Palette側にマウントする。

import { create } from 'zustand';

export const UNIT_DND_MIME = 'application/x-magicdelta-unit';

interface UnitDialogState {
  pending: { unitId: string; position: { x: number; y: number } } | null;
  open: (unitId: string, position: { x: number; y: number }) => void;
  close: () => void;
}

export const useUnitDialogStore = create<UnitDialogState>((set) => ({
  pending: null,
  open: (unitId, position) => set({ pending: { unitId, position } }),
  close: () => set({ pending: null }),
}));
