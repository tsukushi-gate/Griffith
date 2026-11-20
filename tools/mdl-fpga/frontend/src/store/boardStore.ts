// ボードペインの状態(DB 一覧/選択+advise/検出)。
// 検出(JTAG)は必ずユーザー操作(認識ボタン)起点 — ここから自動実行しない。
import { create } from 'zustand';
import { getBoard, getBoards, postDetect, putBoard } from '../api/rest';
import type {
  BoardDoc, BoardState as BoardStateDoc, DetectCandidate, DetectResultDoc,
} from '../api/types';

interface BoardStoreState {
  boards: BoardDoc[];
  state: BoardStateDoc | null;      // {key, advice, advice_error?}
  detecting: boolean;
  detectResult: DetectResultDoc | null;
  candidates: DetectCandidate[] | null; // 複数候補 → 選択ダイアログを開く
  error: string | null;
  refresh: () => Promise<void>;
  select: (key: string | null) => Promise<void>;
  detect: () => Promise<void>;
  dismissCandidates: () => void;
}

export const useBoardStore = create<BoardStoreState>((set, get) => ({
  boards: [],
  state: null,
  detecting: false,
  detectResult: null,
  candidates: null,
  error: null,
  refresh: async () => {
    try {
      const [{ boards }, state] = await Promise.all([getBoards(), getBoard()]);
      set({ boards, state });
    } catch (e) {
      console.warn('board 取得失敗:', e);
    }
  },
  select: async (key) => {
    set({ error: null, candidates: null });
    try {
      const state = await putBoard(key);
      set({ state });
    } catch (e) {
      set({ error: String(e instanceof Error ? e.message : e) });
    }
  },
  detect: async () => {
    if (get().detecting) return;
    set({ detecting: true, error: null, detectResult: null,
          candidates: null });
    try {
      const res = await postDetect();
      set({ detectResult: res });
      if (res.best) {
        // 一意に絞れた → 自動選択(サーバは選択しない設計。選択はここで)
        await get().select(res.best.key);
      } else if (res.candidates.length > 1) {
        set({ candidates: res.candidates }); // ダイアログでユーザーが確認
      }
    } catch (e) {
      set({ error: String(e instanceof Error ? e.message : e) });
    } finally {
      set({ detecting: false });
    }
  },
  dismissCandidates: () => set({ candidates: null }),
}));
