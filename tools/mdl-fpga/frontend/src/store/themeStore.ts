// ライト/ダーク切替(既定=ライト: エディタと並べたときの統一感を優先)。
// エディタは現状ライト固定のため、切替機構はビューア側のみ。トークン名は
// 共有契約なので、.dark ブロックごと将来エディタへ逆移植できる。
import { create } from 'zustand';

export type Theme = 'light' | 'dark';

const STORAGE_KEY = 'mdl-fpga-theme';

function applyTheme(theme: Theme) {
  document.documentElement.classList.toggle('dark', theme === 'dark');
}

function initialTheme(): Theme {
  try {
    const saved = localStorage.getItem(STORAGE_KEY);
    if (saved === 'light' || saved === 'dark') return saved;
  } catch {
    /* localStorage 不可の環境(テスト等)は既定値 */
  }
  return 'light';
}

interface ThemeState {
  theme: Theme;
  setTheme: (theme: Theme) => void;
  toggle: () => void;
}

export const useThemeStore = create<ThemeState>((set, get) => ({
  theme: initialTheme(),
  setTheme: (theme) => {
    applyTheme(theme);
    try {
      localStorage.setItem(STORAGE_KEY, theme);
    } catch {
      /* noop */
    }
    set({ theme });
  },
  toggle: () => get().setTheme(get().theme === 'light' ? 'dark' : 'light'),
}));

/** アプリ起動時に <html> へ初期テーマを反映する。 */
export function initTheme(): void {
  applyTheme(useThemeStore.getState().theme);
}
