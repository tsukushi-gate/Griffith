// UIレイアウト状態(パネル開閉・下部ドックのタブ・詳細ビュー展開・選択ニューロン)。
// パネルの開閉とタブのみlocalStorageへ永続化する(寸法はreact-resizable-panelsのautoSaveId任せ)。

import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import { DEFAULT_LANGUAGE, LANGUAGES, type Language } from '../i18n/messages';

export type BottomTab = 'train' | 'timeline' | 'code' | 'playground' | 'runs' | 'ship';

// トップタブ(N1: NNC同型の5タブ体系)。EDIT=構造編集(既存レンズを内包)、
// TRAINING=学習監視、EVALUATION=評価、DATASET=データ管理、CONFIG=プロジェクト設定。
export type TopTab = 'edit' | 'training' | 'evaluation' | 'dataset' | 'config';

/** トップタブ帯とキーボード1〜5の並び順 */
export const TOP_TAB_ORDER: readonly TopTab[] = [
  'edit',
  'training',
  'evaluation',
  'dataset',
  'config',
];

// レンズ(§2 情報アーキテクチャ): EDITタブ内でのみ有効なオーバーレイ文脈。
// グラフ自体は同一で、着色・バッジ・下部ドックのタブを一括で切り替える。
// 裁定1: trainレンズは廃止しTRAININGタブへ吸収。EDIT内は Build/Probe/Ship の3レンズへ縮約。
export type Lens = 'build' | 'probe' | 'ship';

/** ヘッダのセグメントコントロールとEDIT内レンズ切替キーの並び順 */
export const LENS_ORDER: readonly Lens[] = ['build', 'probe', 'ship'];

/** EDITタブ内でのみ有効なレンズ集合(単一情報源。裁定1のレンズ降格マッピング) */
export const LENS_WITHIN_EDIT: readonly Lens[] = LENS_ORDER;

/** レンズ→下部ドックの初期タブ(切替時プリセット。以降のタブ手動切替は自由) */
export const LENS_TO_TAB: Record<Lens, BottomTab> = {
  build: 'code',
  probe: 'playground',
  ship: 'ship',
};

/**
 * 下部ドックのタブ→所属トップタブ(単一情報源)。
 * EDIT: コード/実行/出荷。TRAINING: 学習/タイムライン/Run一覧。他タブは専用UI(下部ドック非使用)。
 */
export const TABS_BY_TOP: Record<TopTab, BottomTab[]> = {
  edit: ['code', 'playground', 'ship'],
  training: ['train', 'timeline', 'runs'],
  evaluation: [],
  dataset: [],
  config: [],
};

/** 下部タブが属するトップタブを引く(openBottomTabの自動遷移用)。未割当は 'edit' */
export function topTabForBottom(tab: BottomTab): TopTab {
  for (const t of TOP_TAB_ORDER) {
    if (TABS_BY_TOP[t].includes(tab)) return t;
  }
  return 'edit';
}

export interface SelectedNeuron {
  nodeId: string;
  index: number;
}

interface UIState {
  leftOpen: boolean;
  rightOpen: boolean;
  bottomOpen: boolean;
  bottomTab: BottomTab;
  /** 現在のトップタブ(N1)。切替はトップタブ帯/キーボード1〜5。永続化対象 */
  topTab: TopTab;
  setTopTab: (tab: TopTab) => void;
  /** 表示言語(i18n足場)。既定ja、AppHeaderのトグルで切替。永続化対象 */
  language: Language;
  setLanguage: (language: Language) => void;
  /** 現在のレンズ(§2、EDITタブ内のみ)。切替はセグメントコントロール/キーボード */
  lens: Lens;
  /** 詳細ゲートビューの手動展開override。非永続 */
  expandedLayers: Record<string, boolean>;
  /** 詳細ビューで選択中のゲート(インスペクタ連動用)。非永続 */
  selectedNeuron: SelectedNeuron | null;
  /** コマンドパレット(⌘K)の開閉。非永続 */
  cmdOpen: boolean;
  setCmdOpen: (open: boolean) => void;
  /** オンボーディング: 初回起動でTemplateGalleryを自動表示済みか(永続) */
  onboardingVisited: boolean;
  markOnboardingVisited: () => void;
  /** XORチュートリアルガイドパネルの開閉(永続、閉じ/再表示可能) */
  xorGuideOpen: boolean;
  setXorGuideOpen: (open: boolean) => void;
  toggleLeft: () => void;
  toggleRight: () => void;
  toggleBottom: () => void;
  setBottomTab: (tab: BottomTab) => void;
  /** 下部ドックを開いた上で指定タブへ切り替える */
  openBottomTab: (tab: BottomTab) => void;
  /** レンズを切り替え、下部ドックを開いてレンズ既定タブへ寄せる(プリセット) */
  setLens: (lens: Lens) => void;
  setLayerExpanded: (nodeId: string, expanded: boolean) => void;
  setSelectedNeuron: (sel: SelectedNeuron | null) => void;
}

export const useUIStore = create<UIState>()(
  persist(
    (set) => ({
      leftOpen: true,
      rightOpen: true,
      bottomOpen: true,
      bottomTab: 'code',
      topTab: 'edit',
      setTopTab: (tab) => set({ topTab: tab }),
      language: DEFAULT_LANGUAGE,
      setLanguage: (language) => set({ language }),
      lens: 'build',
      expandedLayers: {},
      selectedNeuron: null,
      cmdOpen: false,
      setCmdOpen: (open) => set({ cmdOpen: open }),
      onboardingVisited: false,
      markOnboardingVisited: () => set({ onboardingVisited: true }),
      xorGuideOpen: false,
      setXorGuideOpen: (open) => set({ xorGuideOpen: open }),
      toggleLeft: () => set((s) => ({ leftOpen: !s.leftOpen })),
      toggleRight: () => set((s) => ({ rightOpen: !s.rightOpen })),
      toggleBottom: () => set((s) => ({ bottomOpen: !s.bottomOpen })),
      setBottomTab: (tab) => set({ bottomTab: tab }),
      // 下部ドックを開いた上で指定タブへ切り替える。タブ所属のトップタブへ自動遷移する
      openBottomTab: (tab) =>
        set({ topTab: topTabForBottom(tab), bottomOpen: true, bottomTab: tab }),
      // レンズを切り替え、下部ドックを開いてレンズ既定タブへ寄せる。EDITタブ内限定(裁定1)
      setLens: (lens) =>
        set({ lens, topTab: 'edit', bottomOpen: true, bottomTab: LENS_TO_TAB[lens] }),
      setLayerExpanded: (nodeId, expanded) =>
        set((s) => ({ expandedLayers: { ...s.expandedLayers, [nodeId]: expanded } })),
      setSelectedNeuron: (sel) => set({ selectedNeuron: sel }),
    }),
    {
      name: 'md-editor-ui',
      // v1: 裁定1でtrainレンズ廃止。旧localStorageの lens:'train' 等の廃止値を安全側へ吸収する
      version: 1,
      migrate: (persisted) => {
        const s = (persisted ?? {}) as Partial<UIState> & {
          lens?: string;
          topTab?: string;
          language?: string;
        };
        if (!LENS_ORDER.includes(s.lens as Lens)) s.lens = 'build';
        if (!TOP_TAB_ORDER.includes(s.topTab as TopTab)) s.topTab = 'edit';
        // 旧localStorage(language未保存)や不正値は既定言語へ吸収する
        if (!LANGUAGES.includes(s.language as Language)) s.language = DEFAULT_LANGUAGE;
        return s as UIState;
      },
      partialize: (s) => ({
        leftOpen: s.leftOpen,
        rightOpen: s.rightOpen,
        bottomOpen: s.bottomOpen,
        bottomTab: s.bottomTab,
        topTab: s.topTab,
        language: s.language,
        lens: s.lens,
        onboardingVisited: s.onboardingVisited,
        xorGuideOpen: s.xorGuideOpen,
      }),
    },
  ),
);
