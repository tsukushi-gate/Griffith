// @vitest-environment jsdom
// トップタブ帯とプレースホルダの描画テスト。
// 5タブがクラッシュせず描画され、クリックでtopTabストアが切り替わることを確認する。
// uiStoreはpersistミドルウェアを使うため、import前にインメモリlocalStorageを注入する。

import { cleanup, fireEvent, render, screen } from '@testing-library/react';
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

// localStorage注入後に評価するため動的import(persistのstorage束縛を安全にする)
const { useUIStore } = await import('../store/uiStore');
const { TopTabBar } = await import('./TopTabBar');
const { PlaceholderTab } = await import('./PlaceholderTab');

beforeEach(() => {
  useUIStore.setState({ topTab: 'edit' });
});

afterEach(() => {
  cleanup();
});

describe('TopTabBar', () => {
  it('5トップタブすべてがクラッシュせず描画される', () => {
    render(<TopTabBar />);
    for (const label of ['EDIT', 'TRAINING', 'EVALUATION', 'DATASET', 'CONFIG']) {
      expect(screen.getByText(label)).toBeTruthy();
    }
  });

  it('タブクリックでtopTabストアが切り替わり、localStorageへ永続化される', () => {
    render(<TopTabBar />);
    // Radix Tabsのトリガはmousedownでアクティブ化する
    fireEvent.mouseDown(screen.getByTestId('toptab-training'));
    expect(useUIStore.getState().topTab).toBe('training');
    // 永続化キー md-editor-ui に topTab が書き込まれる(jsdomのwindow.localStorage=memStore)
    const persisted = JSON.parse(memStore['md-editor-ui'] ?? '{}');
    expect(persisted.state.topTab).toBe('training');

    fireEvent.mouseDown(screen.getByTestId('toptab-config'));
    expect(useUIStore.getState().topTab).toBe('config');
  });
});

describe('PlaceholderTab', () => {
  it('タイトルと「実装予定」バッジを表示する', () => {
    render(<PlaceholderTab title="EVALUATION" description="評価をここに実装します。" />);
    expect(screen.getByText('EVALUATION')).toBeTruthy();
    expect(screen.getByText('実装予定')).toBeTruthy();
  });
});
