// グローバルキーボードショートカット(キー再割当)。
// input/textarea/select/contentEditableフォーカス中は無効。
//
// キー割当表(単一情報源):
//   1〜5     = トップタブ切替(TOP_TAB_ORDER準拠: edit/training/evaluation/dataset/config)。全タブから常時有効
//   Q/W/E    = EDIT内レンズ切替(LENS_ORDER準拠: build/probe/ship)。EDITタブ内限定
//   F        = 全体表示
//   L        = 自動整列
//   A        = NNC縦整列(EDITタブ内限定)
//   ⌘K       = コマンドパレット
//   ⌘Z/⇧⌘Z  = undo/redo
//   ⌘S       = 即時同期

import { useEffect } from 'react';
import { flowApi } from '../editor/flowApi';
import { redoGraph, undoGraph, useGraphStore } from '../store/graphStore';
import { applyLens } from '../store/lens';
import { LENS_ORDER, TOP_TAB_ORDER, useUIStore, type TopTab } from '../store/uiStore';

// 数字キー→トップタブ。全タブから常時有効。
const TOP_TAB_KEYS: Record<string, TopTab> = {
  '1': TOP_TAB_ORDER[0],
  '2': TOP_TAB_ORDER[1],
  '3': TOP_TAB_ORDER[2],
  '4': TOP_TAB_ORDER[3],
  '5': TOP_TAB_ORDER[4],
};

// Q/W/E→EDIT内レンズ(build/probe/ship)。数字キーがトップタブへ格上げされたため退避先。
const LENS_KEYS: Record<string, (typeof LENS_ORDER)[number]> = {
  q: LENS_ORDER[0],
  w: LENS_ORDER[1],
  e: LENS_ORDER[2],
};

export function useShortcuts() {
  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      const target = e.target as HTMLElement | null;
      if (target && target.closest('input, textarea, select, [contenteditable="true"]')) {
        return;
      }
      const mod = e.metaKey || e.ctrlKey;
      const key = e.key.toLowerCase();

      if (mod && key === 'k') {
        e.preventDefault();
        const ui = useUIStore.getState();
        ui.setCmdOpen(!ui.cmdOpen);
        return;
      }
      if (mod && key === 'z') {
        e.preventDefault();
        if (e.shiftKey) redoGraph();
        else undoGraph();
        return;
      }
      if (mod && key === 's') {
        e.preventDefault();
        void useGraphStore.getState().syncNow();
        return;
      }
      if (mod) return;

      if (key === 'f') {
        flowApi.current?.fitView();
      } else if (key === 'l') {
        void flowApi.current?.autoLayout();
      } else if (key === 'a') {
        // NNC縦整列: トポロジカル順に座標だけ更新(EDITタブ内で有効)
        if (useUIStore.getState().topTab === 'edit') {
          useGraphStore.getState().alignVerticalNnc();
        }
      } else if (key in TOP_TAB_KEYS) {
        // トップタブ切替: 全タブから常時有効
        useUIStore.getState().setTopTab(TOP_TAB_KEYS[key]);
      } else if (key in LENS_KEYS) {
        // レンズ切替はEDITタブ内でのみ有効(裁定1)
        if (useUIStore.getState().topTab === 'edit') applyLens(LENS_KEYS[key]);
      }
    };
    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, []);
}
