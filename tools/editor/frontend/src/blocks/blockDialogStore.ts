// 複合ブロックのドロップ→パラメータダイアログ連携store。
// FlowCanvasのonDropはドロップ位置(flow座標)とブロック種別をここへ渡すだけにし、
// ダイアログ本体(BlockParamDialog)はPalette側にマウントする。

import { create } from 'zustand';
import type { BlockKind } from './templates';

export const BLOCK_DND_MIME = 'application/x-magicdelta-block';

interface BlockDialogState {
  pending: { kind: BlockKind; position: { x: number; y: number } } | null;
  open: (kind: BlockKind, position: { x: number; y: number }) => void;
  close: () => void;
}

export const useBlockDialogStore = create<BlockDialogState>((set) => ({
  pending: null,
  open: (kind, position) => set({ pending: { kind, position } }),
  close: () => set({ pending: null }),
}));
