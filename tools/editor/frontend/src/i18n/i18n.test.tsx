// @vitest-environment jsdom
// i18n足場のテスト。
// (1)純翻訳器translate: en切替・未訳(en欠落)キーのjaフォールバック・未定義キーの安全策。
// (2)言語切替でTopTabBarのラベル/ツールチップがen辞書へ切り替わる。
// (3)言語設定がlocalStorage(md-editor-ui)へ永続化される。
// uiStoreはpersistミドルウェアを使うため、import前にインメモリlocalStorageを注入する。

import { act, cleanup, render, screen } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

const memStore: Record<string, string> = {};
vi.stubGlobal('localStorage', {
  getItem: (k: string) => memStore[k] ?? null,
  setItem: (k: string, v: string) => {
    memStore[k] = v;
  },
  removeItem: (k: string) => {
    delete memStore[k];
  },
  clear: () => {
    for (const k of Object.keys(memStore)) delete memStore[k];
  },
});

const { translate } = await import('./messages');
const { useUIStore } = await import('../store/uiStore');
const { TopTabBar } = await import('../layout/TopTabBar');

beforeEach(() => {
  useUIStore.setState({ language: 'ja' });
});

afterEach(() => {
  cleanup();
});

describe('translate(純関数)', () => {
  it('言語ごとに辞書文言を返す', () => {
    expect(translate('header.model', 'ja')).toBe('モデル');
    expect(translate('header.model', 'en')).toBe('Model');
    expect(translate('toptab.training.hint', 'en')).toBe('Training monitor');
  });

  it('未訳(en欠落)キーはjaへフォールバックする', () => {
    // demo.untranslated はenを持たない足場見本 → en要求でもjaを返す
    expect(translate('demo.untranslated', 'en')).toBe('未翻訳サンプル');
    expect(translate('demo.untranslated', 'ja')).toBe('未翻訳サンプル');
  });

  it('辞書に無いキーはキー文字列を返す(表示が空にならない)', () => {
    // 型を外れた未知キーでもクラッシュせずキー名を返す
    expect(translate('no.such.key' as never, 'en')).toBe('no.such.key');
  });
});

describe('言語切替のUI反映', () => {
  it('ja既定ではツールチップが日本語、enへ切替で英語辞書になる', () => {
    render(<TopTabBar />);
    // 既定ja: EDITタブのツールチップは日本語
    expect(screen.getByTestId('toptab-edit').getAttribute('title')).toBe('構造編集');

    // enへ切替 → 英語辞書のツールチップへ(ストア更新でReactを再描画)
    act(() => useUIStore.getState().setLanguage('en'));
    expect(screen.getByTestId('toptab-training').getAttribute('title')).toBe(
      'Training monitor',
    );
    // ラベル(NNC英大文字)は日英共通で維持される
    expect(screen.getByTestId('toptab-training').textContent).toBe('TRAINING');
  });
});

describe('言語設定の永続化', () => {
  it('setLanguageがlocalStorage(md-editor-ui)へ書き込まれる', () => {
    useUIStore.getState().setLanguage('en');
    const persisted = JSON.parse(memStore['md-editor-ui'] ?? '{}');
    expect(persisted.state.language).toBe('en');

    useUIStore.getState().setLanguage('ja');
    const persisted2 = JSON.parse(memStore['md-editor-ui'] ?? '{}');
    expect(persisted2.state.language).toBe('ja');
  });
});
