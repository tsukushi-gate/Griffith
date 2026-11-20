// @vitest-environment jsdom
// useTopTabHashの単体テスト。hash⇔topTabの双方向同期・不正hashのフォールバック・
// hashchange追従・無限ループしないことを検証する。

import { cleanup, renderHook } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

// persistのsetItem/getItemを受けるインメモリlocalStorage(store読込より前に注入。uiStore.test.ts同様)
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

// localStorage注入後に評価するため動的import(persistのrehydrateを安全にする)
const { useTopTabHash } = await import('./useTopTabHash');
const { useUIStore } = await import('../store/uiStore');

function setHash(hash: string) {
  window.location.hash = hash;
  window.dispatchEvent(new Event('hashchange'));
}

beforeEach(() => {
  useUIStore.setState({ topTab: 'edit' });
  window.history.replaceState(null, '', '#');
});

afterEach(() => {
  cleanup();
});

describe('useTopTabHash', () => {
  it('マウント時に妥当なhashがあればtopTabへ反映する', () => {
    window.history.replaceState(null, '', '#training');
    renderHook(() => useTopTabHash());
    expect(useUIStore.getState().topTab).toBe('training');
  });

  it('不正/空なhashのときはpersist済みtopTabを優先しhashへ書き戻す', () => {
    useUIStore.setState({ topTab: 'evaluation' });
    window.history.replaceState(null, '', '#not-a-tab');
    renderHook(() => useTopTabHash());
    expect(useUIStore.getState().topTab).toBe('evaluation');
    expect(window.location.hash).toBe('#evaluation');
  });

  it('topTabが変化するとhashが追従する(replaceState)', () => {
    renderHook(() => useTopTabHash());
    useUIStore.getState().setTopTab('dataset');
    expect(window.location.hash).toBe('#dataset');
  });

  it('hashchangeイベントでtopTabが追従する', () => {
    renderHook(() => useTopTabHash());
    setHash('#config');
    expect(useUIStore.getState().topTab).toBe('config');
  });

  it('自分で書き込んだhashが再度hashchangeで届いても無限ループ・状態破壊しない', () => {
    renderHook(() => useTopTabHash());
    useUIStore.getState().setTopTab('training');
    expect(window.location.hash).toBe('#training');
    // ブラウザが自分自身のreplaceStateに対してhashchangeを発火させても
    // lastWrittenHashガードにより無視され、topTabはtrainingのまま安定する
    window.dispatchEvent(new Event('hashchange'));
    expect(useUIStore.getState().topTab).toBe('training');
  });

  it('ガード残留で後続の同値手動hash編集が無視されない(1回消費型)', () => {
    renderHook(() => useTopTabHash());
    // タブ操作でガードに '#training' が入る
    useUIStore.getState().setTopTab('training');
    // 手動編集で別タブへ(同期されるがガード値は書き換わらない状況を再現)
    setHash('#evaluation');
    expect(useUIStore.getState().topTab).toBe('evaluation');
    // 再び '#training' へ手動編集: 古いガードが残留していると無視されてしまう
    setHash('#training');
    expect(useUIStore.getState().topTab).toBe('training');
  });
});
