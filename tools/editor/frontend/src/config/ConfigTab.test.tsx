// @vitest-environment jsdom
// ConfigTabの描画・保存フローの単体テスト。
// バックエンドAPI(../api/rest)は直接モックし、fetchは呼ばせない。

import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';
import type { ProjectResponse, ProjectSummary } from '../api/rest';

// jsdomにはscrollIntoView/PointerEvent系APIが無く、Radix Selectが内部で
// 呼び出すため未実装のままだとクラッシュする(ブラウザでは問題なし)。
Element.prototype.scrollIntoView = vi.fn();
if (!Element.prototype.hasPointerCapture) {
  Element.prototype.hasPointerCapture = vi.fn().mockReturnValue(false);
}
if (!Element.prototype.setPointerCapture) {
  Element.prototype.setPointerCapture = vi.fn();
}
if (!Element.prototype.releasePointerCapture) {
  Element.prototype.releasePointerCapture = vi.fn();
}

const rest = vi.hoisted(() => ({
  listProjects: vi.fn(),
  createProject: vi.fn(),
  getProject: vi.fn(),
  updateProject: vi.fn(),
  deleteProject: vi.fn(),
  getConfigSchema: vi.fn(),
}));

vi.mock('../api/rest', () => rest);
vi.mock('sonner', () => ({ toast: { success: vi.fn(), error: vi.fn() } }));

const { ConfigTab } = await import('./ConfigTab');

afterEach(() => {
  cleanup();
  vi.clearAllMocks();
});

const SUMMARY: ProjectSummary = {
  id: 'p1',
  name: 'proj-a',
  description: '',
  created_at: 1,
  updated_at: 1,
};

const PROJECT: ProjectResponse = {
  id: 'p1',
  name: 'proj-a',
  description: '',
  networks: {},
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

const CONFIG_SCHEMA = {
  $defs: {},
  type: 'object',
  properties: {
    data: {
      type: 'object',
      title: 'DataConfig',
      properties: {
        name: { type: 'string', title: 'Name', default: 'mnist' },
      },
    },
  },
};

describe('ConfigTab', () => {
  it('プロジェクト未選択時はGlobal Config等を出さずクラッシュしない', async () => {
    rest.listProjects.mockResolvedValue([SUMMARY]);
    rest.getConfigSchema.mockResolvedValue(CONFIG_SCHEMA);

    render(<ConfigTab />);

    await waitFor(() => expect(rest.listProjects).toHaveBeenCalled());
    expect(screen.getByText('プロジェクトを選択または新規作成してください')).toBeTruthy();
    expect(screen.queryByText('Global Config')).toBeNull();
  });

  it('新規プロジェクト作成でcreateProjectが呼ばれ、選択状態になる', async () => {
    rest.listProjects.mockResolvedValue([]);
    rest.getConfigSchema.mockResolvedValue(CONFIG_SCHEMA);
    rest.createProject.mockResolvedValue(PROJECT);
    rest.getProject.mockResolvedValue(PROJECT);

    render(<ConfigTab />);
    await waitFor(() => expect(rest.listProjects).toHaveBeenCalled());

    fireEvent.change(screen.getByPlaceholderText('新規プロジェクト名'), {
      target: { value: 'proj-a' },
    });
    fireEvent.click(screen.getByText('新規作成'));

    await waitFor(() => expect(rest.createProject).toHaveBeenCalledWith('proj-a'));
    await waitFor(() => expect(screen.getByText('Global Config')).toBeTruthy());
  });

  it('Global Config編集でupdateProjectがtrain_configs.globalを含めて呼ばれる', async () => {
    rest.listProjects.mockResolvedValue([SUMMARY]);
    rest.getConfigSchema.mockResolvedValue(CONFIG_SCHEMA);
    rest.getProject.mockResolvedValue(PROJECT);
    rest.updateProject.mockResolvedValue(PROJECT);

    render(<ConfigTab />);
    await waitFor(() => expect(rest.listProjects).toHaveBeenCalled());

    // プロジェクトを選択(Radix Selectはjsdomでのuser操作が複雑なため、
    // テスト容易性のためselect要素と同様のvalue設定はコンポーネント内部に閉じている。
    // ここではselectedIdがuseStateで管理されているため、Selectのトリガをクリックして開く必要がある)
    fireEvent.click(screen.getByTestId('project-select'));
    await waitFor(() => screen.queryAllByText('proj-a').length > 0);
    const options = screen.getAllByText('proj-a');
    fireEvent.click(options[options.length - 1]);

    await waitFor(() => expect(rest.getProject).toHaveBeenCalledWith('p1'));
    await waitFor(() => expect(screen.getByText('Global Config')).toBeTruthy());

    const nameInput = await screen.findByLabelText('Name');
    fireEvent.change(nameInput, { target: { value: 'logic:xor' } });

    await waitFor(() => expect(rest.updateProject).toHaveBeenCalled());
    const [pid, patch] = rest.updateProject.mock.calls[0] as [
      string,
      { train_configs: { global: { data: { name: string } } } },
    ];
    expect(pid).toBe('p1');
    expect(patch.train_configs.global.data.name).toBe('logic:xor');
  });

  it('Optimizer定義の追加でupdateProjectがoptimizersを含めて呼ばれる', async () => {
    rest.listProjects.mockResolvedValue([SUMMARY]);
    rest.getConfigSchema.mockResolvedValue(CONFIG_SCHEMA);
    rest.getProject.mockResolvedValue(PROJECT);
    rest.updateProject.mockResolvedValue(PROJECT);

    render(<ConfigTab />);
    await waitFor(() => expect(rest.listProjects).toHaveBeenCalled());
    fireEvent.click(screen.getByTestId('project-select'));
    const options = await screen.findAllByText('proj-a');
    fireEvent.click(options[options.length - 1]);
    await waitFor(() => expect(rest.getProject).toHaveBeenCalledWith('p1'));

    const nameInputs = await screen.findAllByPlaceholderText('名前');
    fireEvent.change(nameInputs[0], { target: { value: 'opt1' } });
    fireEvent.click(screen.getByText('Optimizerを追加'));

    await waitFor(() =>
      expect(rest.updateProject).toHaveBeenCalledWith(
        'p1',
        expect.objectContaining({
          optimizers: expect.objectContaining({ opt1: expect.any(Object) }),
        }),
      ),
    );
  });
});
