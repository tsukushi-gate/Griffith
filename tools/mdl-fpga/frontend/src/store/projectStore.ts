// プロジェクトペインの状態(現在プロジェクト/イグザンプル一覧/切替)。
// examples のフェッチは「開く」ダイアログを開いたときだけ(ブート時に叩かない)。
import { create } from 'zustand';
import { getExamples, getProject, openProject } from '../api/rest';
import type { ExampleEntry, OpenRequest, ProjectInfo } from '../api/types';

interface ProjectState {
  project: ProjectInfo | null;
  examples: ExampleEntry[];
  examplesLoaded: boolean;
  opening: boolean;
  error: string | null;
  refresh: () => Promise<void>;
  fetchExamples: () => Promise<void>;
  open: (req: OpenRequest) => Promise<boolean>;
}

export const useProjectStore = create<ProjectState>((set, get) => ({
  project: null,
  examples: [],
  examplesLoaded: false,
  opening: false,
  error: null,
  refresh: async () => {
    try {
      set({ project: await getProject() });
    } catch (e) {
      console.warn('project 取得失敗:', e);
    }
  },
  fetchExamples: async () => {
    try {
      const { examples } = await getExamples();
      set({ examples, examplesLoaded: true });
    } catch (e) {
      set({ error: String(e instanceof Error ? e.message : e) });
    }
  },
  open: async (req) => {
    if (get().opening) return false;
    set({ opening: true, error: null });
    try {
      await openProject(req);
      // 切替成功: 旧レイアウトの遅延保存を破棄してから全再構築
      const { cancelPendingSave } = await import('./viewerStore');
      cancelPendingSave();
      const { rehydrateApp } = await import('../boot/bootApp');
      await rehydrateApp();
      const { toast } = await import('sonner');
      const { translate } = await import('../i18n/messages');
      const { useLanguageStore } = await import('../i18n/useT');
      toast.success(translate('toast.project.opened',
                              useLanguageStore.getState().language));
      return true;
    } catch (e) {
      set({ error: String(e instanceof Error ? e.message : e) });
      return false;
    } finally {
      set({ opening: false });
    }
  },
}));
