// ビューア構成(プローブ・ノード位置・展開・viewport)。
// 変更は 500ms デバウンスで PUT /api/layout へ保存(単一ライター)。
// サーバ側はモデル横の <model>.viewer.json に永続化する。
import { create } from 'zustand';
import { putLayout } from '../api/rest';
import type { ProbeSpec, ViewerConfig } from '../api/types';

let saveTimer: ReturnType<typeof setTimeout> | null = null;
let probeSeq = 0;

export function cancelPendingSave(): void {
  // プロジェクト切替時に呼ぶ: 旧モデルの遅延保存が新モデルのサイドカーを
  // 汚さないように(サーバ側 epoch ガードとの二重防御)
  if (saveTimer) {
    clearTimeout(saveTimer);
    saveTimer = null;
  }
}

function scheduleSave() {
  if (saveTimer) clearTimeout(saveTimer);
  saveTimer = setTimeout(async () => {
    const s = useViewerStore.getState();
    const cfg: ViewerConfig = {
      version: 1,
      probes: s.probes,
      positions: s.positions,
      expanded: s.expanded,
      viewport: s.viewport,
    };
    const { useSimStore } = await import('./simStore');
    const epoch = useSimStore.getState().meta?.epoch;
    putLayout(cfg, epoch)
      .then(async () => {
        // 表示設定の変更を step なしで反映(サーバが現値で再整形して返す)
        const { useSimStore } = await import('./simStore');
        useSimStore.getState().send({ cmd: 'read' });
      })
      .catch(async (e) => {
        console.warn('layout 保存失敗:', e);
        const { toast } = await import('sonner');
        const { translate } = await import('../i18n/messages');
        const { useLanguageStore } = await import('../i18n/useT');
        toast.error(translate('toast.saved.error',
                              useLanguageStore.getState().language));
      });
  }, 500);
}

interface ViewerState {
  probes: ProbeSpec[];
  positions: Record<string, { x: number; y: number }>;
  expanded: string[];
  viewport: { x: number; y: number; zoom: number } | null;
  loaded: boolean;
  hydrate: (cfg: ViewerConfig) => void;
  reset: () => void;
  addProbe: (target: string, pos: { x: number; y: number }) => void;
  updateProbe: (id: string, patch: Partial<ProbeSpec>) => void;
  removeProbe: (id: string) => void;
  setNodePosition: (id: string, pos: { x: number; y: number }) => void;
  setPositions: (positions: Record<string, { x: number; y: number }>) => void;
  toggleExpanded: (id: string) => void;
  setViewport: (vp: { x: number; y: number; zoom: number }) => void;
}

export const useViewerStore = create<ViewerState>((set, get) => ({
  probes: [],
  positions: {},
  expanded: [],
  viewport: null,
  loaded: false,
  hydrate: (cfg) => {
    probeSeq = cfg.probes.reduce((m, p) => {
      const n = /^p(\d+)$/.exec(p.id);
      return n ? Math.max(m, Number(n[1])) : m;
    }, 0);
    set({
      probes: cfg.probes,
      positions: cfg.positions,
      expanded: cfg.expanded,
      viewport: cfg.viewport,
      loaded: true,
    });
  },
  reset: () => {
    cancelPendingSave();
    set({ probes: [], positions: {}, expanded: [], viewport: null,
          loaded: false });
  },
  addProbe: (target, pos) => {
    probeSeq += 1;
    const probe: ProbeSpec = {
      id: `p${probeSeq}`,
      target,
      slice: null,
      renderer: 'auto',
      args: {},
      pos,
      pinned: false,
      label: null,
      depth: 1,
      scale: 1,
    };
    set({ probes: [...get().probes, probe] });
    scheduleSave();
  },
  updateProbe: (id, patch) => {
    set({
      probes: get().probes.map((p) => (p.id === id ? { ...p, ...patch } : p)),
    });
    scheduleSave();
  },
  removeProbe: (id) => {
    set({ probes: get().probes.filter((p) => p.id !== id) });
    scheduleSave();
  },
  setNodePosition: (id, pos) => {
    set({ positions: { ...get().positions, [id]: pos } });
    scheduleSave();
  },
  setPositions: (positions) => {
    set({ positions: { ...get().positions, ...positions } });
    scheduleSave();
  },
  toggleExpanded: (id) => {
    const cur = get().expanded;
    set({
      expanded: cur.includes(id)
        ? cur.filter((e) => e !== id)
        : [...cur, id],
    });
    scheduleSave();
  },
  setViewport: (viewport) => {
    set({ viewport });
    scheduleSave();
  },
}));
