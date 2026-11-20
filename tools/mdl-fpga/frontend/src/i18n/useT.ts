import { create } from 'zustand';
import {
  DEFAULT_LANGUAGE,
  translate,
  type Language,
  type MessageKey,
} from './messages';

const STORAGE_KEY = 'mdl-fpga-language';

function initialLanguage(): Language {
  try {
    const saved = localStorage.getItem(STORAGE_KEY);
    if (saved === 'ja' || saved === 'en') return saved;
  } catch {
    /* noop */
  }
  return DEFAULT_LANGUAGE;
}

interface LangState {
  language: Language;
  setLanguage: (language: Language) => void;
}

export const useLanguageStore = create<LangState>((set) => ({
  language: initialLanguage(),
  setLanguage: (language) => {
    try {
      localStorage.setItem(STORAGE_KEY, language);
    } catch {
      /* noop */
    }
    set({ language });
  },
}));

/** 翻訳フック(エディタの useT と同シグネチャ)。 */
export function useT() {
  const language = useLanguageStore((s) => s.language);
  return (key: MessageKey) => translate(key, language);
}
