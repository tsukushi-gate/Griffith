// @vitest-environment jsdom
// TemplateGallery のテンプレプロジェクト節の単体テスト。
// バックエンドAPI(../api/rest)はモックし、instantiate→loadNetworks→イベント発火を検証する。

import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';
import type { ProjectResponse, TemplateSummary } from '../api/rest';

const rest = vi.hoisted(() => ({
  getTemplates: vi.fn(),
  instantiateTemplate: vi.fn(),
  createProject: vi.fn(),
  putCurrentIR: vi.fn().mockResolvedValue({ ok: true, warnings: [] }),
}));

vi.mock('../api/rest', () => rest);
vi.mock('sonner', () => ({ toast: { success: vi.fn(), error: vi.fn(), info: vi.fn() } }));
// myBlocks は localStorage 依存(マイUnit)。本テストの対象外なので空実装にする。
vi.mock('./myBlocks', () => ({
  MYUNITS_CHANGED_EVENT: 'md-myunits-changed',
  listMyUnits: () => [],
}));

// networkStore.loadNetworks を監視するためストアはモックせず、実体の呼び出しを見張る。
import { useNetworkStore } from '../store/networkStore';

const { TemplateGallery, PROJECT_INSTANTIATED_EVENT } = await import('./TemplateGallery');

afterEach(() => {
  cleanup();
  vi.clearAllMocks();
});

const TEMPLATES: TemplateSummary[] = [
  {
    id: 'logreg',
    name: 'ロジスティック回帰',
    description: '784→LutLayer→GroupSum(10)',
    category: 'basic',
    verification: 'smoke',
    recommended_dataset: 'mnist',
  },
  {
    id: 'lenet',
    name: 'LeNet風CNN',
    description: 'conv2d→OrPool2d…',
    category: 'vision',
    verification: 'smoke',
    recommended_dataset: 'mnist',
  },
  {
    id: 'transformer',
    name: 'Transformer(blackbox)',
    description: 'blackboxノード',
    category: 'advanced',
    verification: 'validate',
    recommended_dataset: null,
  },
];

const PROJECT: ProjectResponse = {
  id: 'new-id',
  name: 'LeNet風CNN',
  description: '',
  networks: { Main: { format: 'magicdelta-ir', version: '0.2.0', metadata: { name: 'lenet' }, inputs: [{ id: 'x', width: 784, shape: null, encoding: null, semantics: null }], outputs: [{ id: 'y', width: 10, shape: null, encoding: null, semantics: null }], nodes: [], edges: [] } },
  train_configs: {},
  dataset_refs: [],
  run_refs: [],
  optimizers: {},
  monitors: {},
  executors: {},
  created_at: 1,
  updated_at: 1,
  deleted: false,
};

describe('TemplateGallery テンプレプロジェクト節', () => {
  it('GET /api/templates の結果を一覧表示する', async () => {
    rest.getTemplates.mockResolvedValue({ templates: TEMPLATES });
    render(<TemplateGallery open onOpenChange={() => {}} />);
    await waitFor(() => expect(screen.getByTestId('template-logreg')).toBeTruthy());
    expect(screen.getByTestId('template-lenet')).toBeTruthy();
    expect(screen.getByTestId('template-transformer')).toBeTruthy();
    expect(screen.getByTestId('template-empty')).toBeTruthy();
  });

  it('テンプレ選択で instantiate→loadNetworks→イベント発火→ダイアログを閉じる', async () => {
    rest.getTemplates.mockResolvedValue({ templates: TEMPLATES });
    rest.instantiateTemplate.mockResolvedValue(PROJECT);
    const onOpenChange = vi.fn();
    const loadSpy = vi.spyOn(useNetworkStore.getState(), 'loadNetworks');
    const evSpy = vi.fn();
    window.addEventListener(PROJECT_INSTANTIATED_EVENT, evSpy);

    render(<TemplateGallery open onOpenChange={onOpenChange} />);
    await waitFor(() => expect(screen.getByTestId('template-lenet')).toBeTruthy());
    fireEvent.click(screen.getByTestId('template-lenet'));

    await waitFor(() => expect(rest.instantiateTemplate).toHaveBeenCalledWith('lenet'));
    // アクティブネットワーク(先頭)がbackendへ同期される(loadSampleModelと同じ流儀)
    await waitFor(() => expect(rest.putCurrentIR).toHaveBeenCalledWith(PROJECT.networks.Main));
    await waitFor(() => expect(loadSpy).toHaveBeenCalledWith(PROJECT.networks));
    await waitFor(() => expect(evSpy).toHaveBeenCalled());
    expect(onOpenChange).toHaveBeenCalledWith(false);
    window.removeEventListener(PROJECT_INSTANTIATED_EVENT, evSpy);
  });

  it('空プロジェクトは createProject を呼ぶ', async () => {
    rest.getTemplates.mockResolvedValue({ templates: TEMPLATES });
    rest.createProject.mockResolvedValue({ ...PROJECT, id: 'empty-id' });
    render(<TemplateGallery open onOpenChange={() => {}} />);
    await waitFor(() => expect(screen.getByTestId('template-empty')).toBeTruthy());
    fireEvent.click(screen.getByTestId('template-empty'));
    await waitFor(() => expect(rest.createProject).toHaveBeenCalled());
  });
});
