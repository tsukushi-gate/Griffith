// ブート処理(App.tsx から抽出)+プロジェクト切替後の全再構築(rehydrate)。
// 切替の主経路は「open 成功 → rehydrateApp = WS を静かに閉じ→全ストア初期化→
// ブートし直し」。サーバの session_changed は他タブ向けのバックストップ。
import { getLayout, getMeta, getRenderers } from '../api/rest';
import { closeSimSocket, connectSimSocket } from '../api/simSocket';
import { layoutWithElk } from '../graph/autoLayout';
import { metaToFlow } from '../graph/metaToFlow';
import { useSimStore } from '../store/simStore';
import { useViewerStore } from '../store/viewerStore';

export async function bootApp(): Promise<void> {
  const [meta, layout, renderers] = await Promise.all([
    getMeta(),
    getLayout(),
    getRenderers(),
  ]);
  useSimStore.getState().setMeta(meta);
  useSimStore.getState().setRenderers(renderers);
  useViewerStore.getState().hydrate(layout);
  // 永続化された展開状態の params を先読み
  const { useDetailStore } = await import('../store/detailStore');
  for (const id of layout.expanded) {
    void useDetailStore.getState().fetchParams(id);
  }

  // 位置未保存のノードがあれば elk で初回整列
  const ids = [
    ...meta.inputs.map((p) => p.id),
    ...meta.nodes.map((n) => n.id),
    ...meta.outputs.map((p) => p.id),
  ];
  const missing = ids.some((id) => !(id in layout.positions));
  if (missing) {
    const { nodes, edges } = metaToFlow(meta, layout.positions);
    const pos = await layoutWithElk(nodes, edges);
    useViewerStore.getState().setPositions(Object.fromEntries(pos));
  }
  connectSimSocket();
}

let rehydrating = false;

export async function rehydrateApp(): Promise<void> {
  if (rehydrating) return; // open 成功と session_changed の二重発火を抑止
  rehydrating = true;
  try {
    closeSimSocket({ silent: true });
    useViewerStore.getState().reset(); // loaded:false → ローディング表示に戻る
    useSimStore.setState({
      meta: null,
      frame: null,
      renderers: null,
      running: false,
      inputX: 0n,
    });
    const { useDetailStore } = await import('../store/detailStore');
    useDetailStore.setState({ params: {}, errors: {} }); // 旧モデルの params 破棄
    await bootApp();
    const { useProjectStore } = await import('../store/projectStore');
    void useProjectStore.getState().refresh();
    const { useBoardStore } = await import('../store/boardStore');
    void useBoardStore.getState().refresh();
  } finally {
    rehydrating = false;
  }
}
