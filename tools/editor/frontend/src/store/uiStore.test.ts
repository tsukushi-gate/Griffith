// uiStoreのレンズ状態遷移(§2)とapplyLensオーケストレーションの単体テスト。
// viewStore/trainingStoreのテスト同様、getState()でアクションを直接呼ぶ。
// uiStoreはpersistミドルウェアを使うため、node環境向けにlocalStorageを差し込む。

import { beforeEach, describe, expect, it, vi } from 'vitest';
import type { Lens } from './uiStore';

// persistのsetItem/getItemを受けるインメモリlocalStorage(store読込より前に注入)
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
const { usePlaygroundStore } = await import('../playground/playgroundStore');
const { applyLens } = await import('./lens');
const {
  LENS_ORDER,
  LENS_TO_TAB,
  LENS_WITHIN_EDIT,
  TOP_TAB_ORDER,
  TABS_BY_TOP,
  topTabForBottom,
  useUIStore,
} = await import('./uiStore');

beforeEach(() => {
  useUIStore.setState({
    lens: 'build',
    bottomTab: 'code',
    bottomOpen: true,
    topTab: 'edit',
  });
  usePlaygroundStore.setState({ enabled: false });
});

describe('uiStore レンズ', () => {
  it('LENS_ORDERは build/probe/ship の3つ(裁定1: trainレンズ廃止)', () => {
    expect(LENS_ORDER).toEqual(['build', 'probe', 'ship']);
  });

  it('LENS_WITHIN_EDITはLENS_ORDERと同一のEDIT内レンズ集合', () => {
    expect(LENS_WITHIN_EDIT).toEqual(['build', 'probe', 'ship']);
  });

  it('LENS_TO_TABは各レンズを既定タブへ対応づける', () => {
    expect(LENS_TO_TAB).toEqual({
      build: 'code',
      probe: 'playground',
      ship: 'ship',
    });
  });

  it('setLensはlens・下部タブ・下部ドック開・EDITタブを一括で設定する', () => {
    const cases: { lens: Lens; tab: string }[] = [
      { lens: 'build', tab: 'code' },
      { lens: 'probe', tab: 'playground' },
      { lens: 'ship', tab: 'ship' },
    ];
    for (const { lens, tab } of cases) {
      useUIStore.setState({ bottomOpen: false, topTab: 'training' });
      useUIStore.getState().setLens(lens);
      const s = useUIStore.getState();
      expect(s.lens).toBe(lens);
      expect(s.bottomTab).toBe(tab);
      expect(s.bottomOpen).toBe(true);
      expect(s.topTab).toBe('edit');
    }
  });
});

describe('uiStore トップタブ(N1)', () => {
  it('TOP_TAB_ORDERは5タブ edit/training/evaluation/dataset/config', () => {
    expect(TOP_TAB_ORDER).toEqual([
      'edit',
      'training',
      'evaluation',
      'dataset',
      'config',
    ]);
  });

  it('setTopTabでtopTabが切り替わる(localStorage永続化はTopTabBar.test.tsxで検証)', () => {
    expect(useUIStore.getState().topTab).toBe('edit');
    useUIStore.getState().setTopTab('training');
    expect(useUIStore.getState().topTab).toBe('training');
  });

  it('TABS_BY_TOPはEDIT/TRAININGへ下部タブを排他的に振り分ける', () => {
    expect(TABS_BY_TOP.edit).toEqual(['code', 'playground', 'ship']);
    expect(TABS_BY_TOP.training).toEqual(['train', 'timeline', 'runs']);
    expect(TABS_BY_TOP.evaluation).toEqual([]);
    // 学習/Run一覧はEDITに露出しない
    expect(TABS_BY_TOP.edit).not.toContain('train');
    expect(TABS_BY_TOP.edit).not.toContain('runs');
  });

  it('topTabForBottomは下部タブの所属トップタブを返す', () => {
    expect(topTabForBottom('ship')).toBe('edit');
    expect(topTabForBottom('code')).toBe('edit');
    expect(topTabForBottom('train')).toBe('training');
    expect(topTabForBottom('runs')).toBe('training');
  });

  it('openBottomTabは所属トップタブへ自動遷移してタブを開く', () => {
    // TRAINING所属のtrainを開くとtopTab=training、EDIT所属のshipでtopTab=edit
    useUIStore.setState({ topTab: 'evaluation', bottomOpen: false });
    useUIStore.getState().openBottomTab('train');
    expect(useUIStore.getState().topTab).toBe('training');
    expect(useUIStore.getState().bottomTab).toBe('train');
    expect(useUIStore.getState().bottomOpen).toBe(true);

    useUIStore.getState().openBottomTab('ship');
    expect(useUIStore.getState().topTab).toBe('edit');
    expect(useUIStore.getState().bottomTab).toBe('ship');
  });
});

describe('applyLens オーケストレーション', () => {
  it('probeレンズだけプレイグラウンド着色を有効化する', () => {
    applyLens('probe');
    expect(useUIStore.getState().lens).toBe('probe');
    expect(useUIStore.getState().bottomTab).toBe('playground');
    expect(usePlaygroundStore.getState().enabled).toBe(true);

    applyLens('build');
    expect(useUIStore.getState().lens).toBe('build');
    expect(usePlaygroundStore.getState().enabled).toBe(false);

    applyLens('ship');
    expect(usePlaygroundStore.getState().enabled).toBe(false);
  });
});
