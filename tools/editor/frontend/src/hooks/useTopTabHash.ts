// URLハッシュ⇔トップタブ双方向同期フック。
// `#edit`/`#training`/`#evaluation`/`#dataset`/`#config` でディープリンク可能にする。
// マウント時: hashがTOP_TAB_ORDERに含まれればそれをtopTabへ反映。
// 不正/空ならpersist済みのtopTab値を優先し、hash側をそれに合わせて書き換える。
// 以降: hashchange(戻る/進む・手動編集)→topTab、topTab変化→hash、を双方向に追従する。
// pushStateだと戻るボタンがタブ間ナビゲーションになりUXと衝突するため、
// hash書き換えは常にreplaceStateを使う。
// 自分が書いたhashをhashchangeで再度拾って無限ループしないよう、
// 直近で書き込んだhash値をrefに覚えて無視するガードを入れる。

import { useEffect, useRef } from 'react';
import { TOP_TAB_ORDER, useUIStore, type TopTab } from '../store/uiStore';

function readHashTab(): TopTab | null {
  const raw = window.location.hash.replace(/^#/, '');
  return (TOP_TAB_ORDER as readonly string[]).includes(raw) ? (raw as TopTab) : null;
}

export function useTopTabHash(): void {
  // 直近で自分がreplaceStateしたhash値。hashchangeハンドラの自己反応防止ガード
  const lastWrittenHash = useRef<string | null>(null);

  useEffect(() => {
    // マウント時: hashが妥当ならtopTabへ反映、そうでなければpersist済みtopTabをhashへ反映
    const hashTab = readHashTab();
    if (hashTab !== null) {
      useUIStore.getState().setTopTab(hashTab);
    } else {
      const currentTab = useUIStore.getState().topTab;
      lastWrittenHash.current = `#${currentTab}`;
      window.history.replaceState(null, '', `#${currentTab}`);
    }

    const onHashChange = () => {
      const hash = window.location.hash;
      // ガードは1回消費型: 残留すると後で同じ値への手動編集まで無視してしまう
      const ignored = hash === lastWrittenHash.current;
      lastWrittenHash.current = null;
      if (ignored) return;
      const tab = readHashTab();
      if (tab !== null) {
        useUIStore.getState().setTopTab(tab);
      }
    };
    window.addEventListener('hashchange', onHashChange);
    return () => window.removeEventListener('hashchange', onHashChange);
  }, []);

  useEffect(() => {
    // topTab変化をhashへ反映。unsubscribeはクリーンアップで解除
    const unsubscribe = useUIStore.subscribe((state, prevState) => {
      if (state.topTab === prevState.topTab) return;
      const next = `#${state.topTab}`;
      if (window.location.hash === next) return;
      lastWrittenHash.current = next;
      window.history.replaceState(null, '', next);
    });
    return unsubscribe;
  }, []);
}
