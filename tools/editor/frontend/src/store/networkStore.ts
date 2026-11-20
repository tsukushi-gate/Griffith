// 複数ネットワークタブのストア。
// 1プロジェクトが複数ネットワーク(Main/Runtime 等)を持つ器。フロント側は
// networks{ name → IRGraphJson } を保持し、タブ切替でアクティブIRを graphStore へ差し替える。
// 切替前に現タブのIR(graphStore.ir)を networks へ書き戻す(未保存編集を失わない)。
// 学習対象は従来どおり graphStore(STATE.current)に載っているアクティブネットワーク。
//
// プロジェクト保存/読込はCONFIGタブ経由(既存)。本ストアは loadNetworks/getNetworks で
// networksマップの読み書き口を提供する(将来のCONFIGタブ統合の接続点)。

import { create } from 'zustand';
import type { IRGraphJson } from '../ir/schema';
import { useGraphStore } from './graphStore';

/** 既定の主ネットワーク名。 */
export const MAIN_NETWORK = 'Main';

interface NetworkState {
  networks: Record<string, IRGraphJson>;
  order: string[]; // タブ表示順
  active: string | null;
  /** graphStoreに既にIRがあり networks が空なら Main として取り込む(初期化)。 */
  ensureSeeded: () => void;
  /** 現在のアクティブタブへ graphStore.ir を書き戻す(構造は複製で退避)。 */
  writeBackActive: () => void;
  /** タブ切替: 現タブへ書き戻し → 対象IRを graphStore へロードして active を更新。 */
  switchTo: (name: string) => void;
  /** 新規ネットワークタブを追加する(空 or 現IRの複製)。追加後アクティブ化する。 */
  addNetwork: (name?: string, clone?: boolean) => string;
  /** ネットワークを改名する。 */
  renameNetwork: (from: string, to: string) => void;
  /** ネットワークを削除する(最後の1本は削除しない)。 */
  removeNetwork: (name: string) => void;
  /** プロジェクトから networks マップを読み込み、activeを graphStore へロードする。 */
  loadNetworks: (networks: Record<string, IRGraphJson>, active?: string) => void;
  /** 現在の networks マップ(アクティブ分は最新の graphStore.ir を反映)を返す。 */
  getNetworks: () => Record<string, IRGraphJson>;
}

/** IRを安全に複製する(退避時の共有参照を避ける)。 */
function cloneIr(ir: IRGraphJson): IRGraphJson {
  return structuredClone(ir);
}

/** 一意なネットワーク名を作る(重複時に連番を付す)。 */
function uniqueName(base: string, existing: string[]): string {
  if (!existing.includes(base)) return base;
  let i = 2;
  while (existing.includes(`${base}${i}`)) i += 1;
  return `${base}${i}`;
}

export const useNetworkStore = create<NetworkState>((set, get) => ({
  networks: {},
  order: [],
  active: null,

  ensureSeeded: () => {
    const { active, order } = get();
    if (order.length > 0) return;
    const ir = useGraphStore.getState().ir;
    if (!ir) return;
    set({
      networks: { [MAIN_NETWORK]: cloneIr(ir) },
      order: [MAIN_NETWORK],
      active: active ?? MAIN_NETWORK,
    });
  },

  writeBackActive: () => {
    const { active, networks } = get();
    if (!active) return;
    const ir = useGraphStore.getState().ir;
    if (!ir) return;
    set({ networks: { ...networks, [active]: cloneIr(ir) } });
  },

  switchTo: (name) => {
    const { active, networks } = get();
    if (name === active) return;
    const target = networks[name];
    if (!target) return;
    // 現タブへ書き戻してから対象をロード
    get().writeBackActive();
    set({ active: name });
    useGraphStore.getState().setIR(cloneIr(target), { needsBuild: true });
  },

  addNetwork: (name, clone = false) => {
    const { order, networks } = get();
    const finalName = uniqueName(name?.trim() || 'Network', order);
    // 現タブを書き戻してからベースIRを決める
    get().writeBackActive();
    const baseIr = useGraphStore.getState().ir;
    const newIr =
      clone && baseIr
        ? cloneIr(baseIr)
        : baseIr
          ? emptyLikeIr(baseIr, finalName)
          : undefined;
    if (!newIr) return finalName;
    set({
      networks: { ...networks, [finalName]: newIr },
      order: [...order, finalName],
      active: finalName,
    });
    useGraphStore.getState().setIR(cloneIr(newIr), { needsBuild: true });
    return finalName;
  },

  renameNetwork: (from, to) => {
    const { networks, order, active } = get();
    const finalTo = to.trim();
    if (!finalTo || from === finalTo || !networks[from] || networks[finalTo]) return;
    const nextNetworks: Record<string, IRGraphJson> = {};
    for (const [k, v] of Object.entries(networks)) nextNetworks[k === from ? finalTo : k] = v;
    set({
      networks: nextNetworks,
      order: order.map((n) => (n === from ? finalTo : n)),
      active: active === from ? finalTo : active,
    });
  },

  removeNetwork: (name) => {
    const { networks, order, active } = get();
    if (order.length <= 1 || !networks[name]) return;
    const nextNetworks = { ...networks };
    delete nextNetworks[name];
    const nextOrder = order.filter((n) => n !== name);
    const nextActive = active === name ? nextOrder[0] : active;
    set({ networks: nextNetworks, order: nextOrder, active: nextActive });
    if (active === name && nextActive) {
      useGraphStore.getState().setIR(cloneIr(nextNetworks[nextActive]), { needsBuild: true });
    }
  },

  loadNetworks: (networks, active) => {
    const order = Object.keys(networks);
    if (order.length === 0) {
      set({ networks: {}, order: [], active: null });
      return;
    }
    const activeName = active && networks[active] ? active : order[0];
    set({
      networks: Object.fromEntries(order.map((k) => [k, cloneIr(networks[k])])),
      order,
      active: activeName,
    });
    useGraphStore.getState().setIR(cloneIr(networks[activeName]), { needsBuild: true });
  },

  getNetworks: () => {
    get().writeBackActive();
    return get().networks;
  },
}));

/** ベースIRの入出力ポートだけ引き継いだ空ネットワークIRを作る(新規タブの雛形)。 */
function emptyLikeIr(base: IRGraphJson, name: string): IRGraphJson {
  return {
    format: base.format,
    version: base.version,
    metadata: { name },
    inputs: base.inputs.map((p) => ({ ...p })),
    outputs: base.outputs.map((p) => ({ ...p })),
    nodes: [],
    edges: [],
  };
}
