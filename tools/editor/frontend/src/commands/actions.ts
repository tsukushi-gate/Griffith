// ヘッダ/コマンドパレット共有のアクション

import { buildModel } from '../api/rest';
import { useGraphStore } from '../store/graphStore';

/** 未同期編集をflushしてbuild(draft確定+ノードid同期)する。失敗時はthrow。 */
export async function runBuild(): Promise<void> {
  const graph = useGraphStore.getState();
  await graph.syncNow();
  if (useGraphStore.getState().syncError) {
    throw new Error(useGraphStore.getState().syncError ?? '同期失敗');
  }
  const res = await buildModel();
  useGraphStore.getState().setIR(res.ir, { needsBuild: false });
}
