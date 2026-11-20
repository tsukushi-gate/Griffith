// useT() フック: uiStoreの language を購読し、現在言語に束縛した t(key) を返す。
// 言語切替でlanguageが変わると購読コンポーネントが再描画され、文言が切り替わる。

import { useUIStore } from '../store/uiStore';
import { translate, type MessageKey } from './messages';

/**
 * 現在の表示言語に束縛した翻訳関数を返すフック。
 * `const t = useT(); t('header.build')` の形で使う。
 */
export function useT(): (key: MessageKey) => string {
  const language = useUIStore((s) => s.language);
  return (key: MessageKey) => translate(key, language);
}
