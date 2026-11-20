// シミュレーション状態(WS フレーム+接続+入力ビット+RUN)。
// 真実は常に RTL 側 — このストアは最新フレームの写しに過ぎない。
import { create } from 'zustand';
import type { Meta, RendererListing, SimFrame } from '../api/types';

export type SimCommand =
  | { cmd: 'set_step'; hex: string; n: number }
  | { cmd: 'set_x'; hex: string }
  | { cmd: 'step'; n: number }
  | { cmd: 'step_until'; target: string;
      slice: [number, number] | null; max: number }
  | { cmd: 'reset' }
  | { cmd: 'read' };

interface SimState {
  meta: Meta | null;
  renderers: RendererListing | null;
  frame: SimFrame | null;
  connected: boolean;
  running: boolean;
  autoStep: boolean;
  inputX: bigint; // ユーザーが編集中の入力(送信はコマンド経由)
  send: (cmd: SimCommand) => void; // simSocket が差し替える
  setMeta: (meta: Meta) => void;
  setRenderers: (renderers: RendererListing) => void;
  setFrame: (frame: SimFrame) => void;
  setConnected: (connected: boolean) => void;
  setRunning: (running: boolean) => void;
  setAutoStep: (autoStep: boolean) => void;
  setInputX: (x: bigint) => void;
  pushInput: () => void; // inputX を回路へ(autoStep なら 1clk 付き)
}

export const useSimStore = create<SimState>((set, get) => ({
  meta: null,
  renderers: null,
  frame: null,
  connected: false,
  running: false,
  autoStep: true,
  inputX: 0n,
  send: () => {},
  setMeta: (meta) => set({ meta }),
  setRenderers: (renderers) => set({ renderers }),
  setFrame: (frame) => set({ frame }),
  setConnected: (connected) => set({ connected }),
  setRunning: (running) => set({ running }),
  setAutoStep: (autoStep) => set({ autoStep }),
  setInputX: (inputX) => set({ inputX }),
  pushInput: () => {
    const { inputX, autoStep, send } = get();
    const hex = inputX.toString(16);
    send(autoStep ? { cmd: 'set_step', hex, n: 1 } : { cmd: 'set_x', hex });
  },
}));
