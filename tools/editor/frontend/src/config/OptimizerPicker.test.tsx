// @vitest-environment jsdom
// OptimizerPickerの単体テスト。
// TrainSettingsFormへの最小フック用の独立コンポーネントなので、
// zustandストアには触れずonApplyコールバックの引数のみ検証する。

import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';
import type { ProjectResponse, ProjectSummary } from '../api/rest';

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
  getProject: vi.fn(),
}));

vi.mock('../api/rest', () => rest);
vi.mock('sonner', () => ({ toast: { success: vi.fn(), error: vi.fn() } }));

const { OptimizerPicker } = await import('./OptimizerPicker');

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
  optimizers: {
    fast: {
      dataset: 'logic:xor',
      method: 'bp',
      loss: 'bce',
      optimizer: 'sgd',
      lr: 0.05,
      logic_lr: null,
      ann_lr: null,
      batch_size: 64,
      epochs: 5,
    },
  },
  monitors: {},
  executors: {},
  created_at: 1,
  updated_at: 1,
  deleted: false,
};

describe('OptimizerPicker', () => {
  it('プロジェクト一覧を取得しクラッシュせず描画する', async () => {
    rest.listProjects.mockResolvedValue([SUMMARY]);

    render(<OptimizerPicker onApply={() => {}} />);
    await waitFor(() => expect(rest.listProjects).toHaveBeenCalled());
    expect(screen.getByText('Optimizer定義から適用')).toBeTruthy();
  });

  it('プロジェクト選択→Optimizer選択→適用でonApplyがTrainingPreset形の値を伴って呼ばれる', async () => {
    rest.listProjects.mockResolvedValue([SUMMARY]);
    rest.getProject.mockResolvedValue(PROJECT);
    const onApply = vi.fn();

    render(<OptimizerPicker onApply={onApply} />);
    await waitFor(() => expect(rest.listProjects).toHaveBeenCalled());

    const selects = screen.getAllByRole('combobox');
    fireEvent.click(selects[0]); // プロジェクトSelect
    const projectOptions = await screen.findAllByText('proj-a');
    fireEvent.click(projectOptions[projectOptions.length - 1]);

    await waitFor(() => expect(rest.getProject).toHaveBeenCalledWith('p1'));

    const selectsAfter = screen.getAllByRole('combobox');
    fireEvent.click(selectsAfter[1]); // OptimizerSelect
    const optimizerOptions = await screen.findAllByText('fast');
    fireEvent.click(optimizerOptions[optimizerOptions.length - 1]);

    fireEvent.click(screen.getByText('適用'));

    expect(onApply).toHaveBeenCalledTimes(1);
    const [values, label] = onApply.mock.calls[0] as [Record<string, unknown>, string];
    expect(label).toBe('fast');
    expect(values).toEqual({
      dataset: 'logic:xor',
      loss: 'bce',
      method: 'bp',
      optimizer: 'sgd',
      lr: 0.05,
      logicLr: null,
      annLr: null,
      batchSize: 64,
      epochs: 5,
    });
  });
});
