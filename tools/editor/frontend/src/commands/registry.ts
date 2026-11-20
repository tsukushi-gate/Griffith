// コマンドパレット(⌘K)のコマンド定義。
// run実行時に最新のstore/flowApiを参照するため、定義は関数で都度組み立てる。

import { toast } from 'sonner';
import { flowApi } from '../editor/flowApi';
import {
  groupableSelection,
  redoGraph,
  undoGraph,
  useGraphStore,
} from '../store/graphStore';
import { TOP_TAB_ORDER, useUIStore, type TopTab } from '../store/uiStore';
import { runBuild } from './actions';

export interface CommandDef {
  id: string;
  group: string;
  label: string;
  shortcut?: string;
  disabled?: boolean;
  run: () => void | Promise<void>;
}

// トップタブ→コマンドパレット上のラベル(単一情報源)
const TOP_TAB_LABELS: Record<TopTab, string> = {
  edit: '編集',
  training: '学習',
  evaluation: '評価',
  dataset: 'データセット',
  config: '設定',
};

export function buildCommands(): CommandDef[] {
  const hasIr = useGraphStore.getState().ir !== null;
  const groupable = groupableSelection().length;
  const ui = useUIStore.getState();

  // タブ移動コマンド: 1〜5でトップタブ切替(useShortcuts.tsと同じ割当)
  const tabCommands: CommandDef[] = TOP_TAB_ORDER.map((tab, i) => ({
    id: `tab-${tab}`,
    group: 'タブ',
    label: `タブ: ${TOP_TAB_LABELS[tab]}へ`,
    shortcut: String(i + 1),
    run: () => ui.setTopTab(tab),
  }));

  return [
    ...tabCommands,
    {
      id: 'add-lut',
      group: 'ノード追加',
      label: 'LutLayerを追加',
      disabled: !hasIr,
      run: () => flowApi.current?.addAtCenter('lut_layer'),
    },
    {
      id: 'add-groupsum',
      group: 'ノード追加',
      label: 'GroupSumを追加',
      disabled: !hasIr,
      run: () => flowApi.current?.addAtCenter('group_sum'),
    },
    {
      id: 'add-binarize',
      group: 'ノード追加',
      label: 'Binarizeを追加',
      disabled: !hasIr,
      run: () => flowApi.current?.addAtCenter('binarize'),
    },
    {
      id: 'add-constant',
      group: 'ノード追加',
      label: 'Constantを追加',
      disabled: !hasIr,
      run: () => flowApi.current?.addAtCenter('constant'),
    },
    {
      id: 'undo',
      group: '編集',
      label: '元に戻す',
      shortcut: '⌘Z',
      run: () => {
        if (!undoGraph()) toast.info('戻す履歴がありません');
      },
    },
    {
      id: 'redo',
      group: '編集',
      label: 'やり直す',
      shortcut: '⇧⌘Z',
      run: () => {
        if (!redoGraph()) toast.info('やり直す履歴がありません');
      },
    },
    {
      id: 'group-selection',
      group: '編集',
      label: '選択ノードをグループ化',
      disabled: groupable < 2,
      run: () => {
        const gid = useGraphStore.getState().createGroup();
        if (gid) toast.success('サブサーキット化しました');
        else toast.info('グループ化するには2つ以上のノードを選択してください');
      },
    },
    {
      id: 'sync',
      group: '編集',
      label: '今すぐサーバへ同期',
      shortcut: '⌘S',
      disabled: !hasIr,
      run: () => void useGraphStore.getState().syncNow(),
    },
    {
      id: 'build',
      group: 'モデル',
      label: 'build(モデル確定)',
      disabled: !hasIr,
      run: async () => {
        try {
          await runBuild();
          toast.success('build完了(draft確定・ノードid更新)');
        } catch (e) {
          toast.error(e instanceof Error ? e.message : String(e));
        }
      },
    },
    {
      id: 'auto-layout',
      group: '表示',
      label: '自動整列',
      shortcut: 'L',
      disabled: !hasIr,
      run: () => void flowApi.current?.autoLayout(),
    },
    {
      id: 'align-vertical',
      group: '表示',
      label: 'NNC縦整列',
      shortcut: 'A',
      disabled: !hasIr,
      run: () => useGraphStore.getState().alignVerticalNnc(),
    },
    {
      id: 'fit-view',
      group: '表示',
      label: '全体表示',
      shortcut: 'F',
      run: () => flowApi.current?.fitView(),
    },
    {
      id: 'toggle-left',
      group: 'パネル',
      label: 'パレットの表示切替',
      run: () => ui.toggleLeft(),
    },
    {
      id: 'toggle-right',
      group: 'パネル',
      label: 'インスペクタの表示切替',
      run: () => ui.toggleRight(),
    },
    {
      id: 'toggle-bottom',
      group: 'パネル',
      label: '下部ドックの表示切替',
      run: () => ui.toggleBottom(),
    },
    {
      id: 'tab-train',
      group: 'パネル',
      label: '下部タブ: 学習',
      run: () => ui.openBottomTab('train'),
    },
    {
      id: 'tab-timeline',
      group: 'パネル',
      label: '下部タブ: タイムライン',
      run: () => ui.openBottomTab('timeline'),
    },
    {
      id: 'tab-code',
      group: 'パネル',
      label: '下部タブ: コード',
      run: () => ui.openBottomTab('code'),
    },
  ];
}
