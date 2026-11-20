// @vitest-environment jsdom
// DATASETタブのRTLテスト。rest APIはモックし、
// カタログ選択→プレビュー描画、CSV取込→カタログ更新、エンコードプレビュー
// トグルを検証する。

import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type {
  DatasetInfo,
  DatasetSamplesResponse,
  EncodingPreviewResponse,
} from '../api/rest';

const CATALOG: DatasetInfo[] = [
  { name: 'logic:xor', label: 'XOR', in_width: 2, out_width: 1, kind: 'binary_vector' },
];

const IMPORTED: DatasetInfo = {
  name: 'imported:mycsv',
  label: 'mycsv(取込)',
  in_width: 2,
  out_width: 1,
  kind: 'binary_vector',
  imported: true,
};

const SAMPLES: DatasetSamplesResponse = {
  dataset_id: 'logic:xor',
  kind: 'logic',
  num_data: 4,
  num_columns: 2,
  page: 0,
  page_size: 12,
  num_pages: 1,
  samples: [
    { index: 0, inputs: [0, 0], outputs: [0] },
    { index: 1, inputs: [0, 1], outputs: [1] },
  ],
  class_distribution: { '0': 2, '1': 2 },
  class_distribution_approx: false,
};

const ENCODED: EncodingPreviewResponse = { mode: 'binarize', bits_per_value: 1, grid: [[0], [0]] };

const getDatasets = vi.fn();
const getDatasetSamples = vi.fn();
const importCsvDataset = vi.fn();
const encodingPreview = vi.fn();

vi.mock('../api/rest', () => ({
  getDatasets: () => getDatasets(),
  getDatasetSamples: (...a: unknown[]) => getDatasetSamples(...a),
  importCsvDataset: (...a: unknown[]) => importCsvDataset(...a),
  importImagesDataset: vi.fn(),
  updateImportedDatasetConfig: vi.fn(),
  encodingPreview: (...a: unknown[]) => encodingPreview(...a),
}));

const { DatasetTab } = await import('./DatasetTab');

beforeEach(() => {
  getDatasets.mockResolvedValue(CATALOG);
  getDatasetSamples.mockResolvedValue(SAMPLES);
  encodingPreview.mockResolvedValue(ENCODED);
  importCsvDataset.mockResolvedValue({ dataset: IMPORTED });
});

afterEach(() => {
  cleanup();
  vi.clearAllMocks();
});

describe('DatasetTab', () => {
  it('カタログを一覧し、選択でプレビューグリッドを描画する', async () => {
    render(<DatasetTab />);
    const item = await screen.findByTestId('dataset-cat-logic:xor');
    fireEvent.click(item);
    await waitFor(() => expect(getDatasetSamples).toHaveBeenCalledWith('logic:xor', expect.anything()));
    expect(await screen.findByTestId('sample-preview-grid')).toBeTruthy();
    expect(screen.getByTestId('class-dist')).toBeTruthy();
    expect(screen.getAllByTestId('sample-cell').length).toBe(2);
  });

  it('CSV取込で登録データセットが選択され、カタログが再取得される', async () => {
    render(<DatasetTab />);
    await screen.findByTestId('dataset-cat-logic:xor');
    // 取込後のカタログには imported を含める
    getDatasets.mockResolvedValue([...CATALOG, IMPORTED]);
    const input = screen.getByTestId('import-csv-input') as HTMLInputElement;
    const file = new File(['x__0,x__1,y\n0,0,0\n'], 'mycsv.csv', { type: 'text/csv' });
    fireEvent.change(input, { target: { files: [file] } });
    await waitFor(() => expect(importCsvDataset).toHaveBeenCalled());
    // 取込済みデータセットがカタログに現れる
    expect(await screen.findByTestId('dataset-cat-imported:mycsv')).toBeTruthy();
  });

  it('エンコードプレビューをトグルするとencodingPreview APIを呼ぶ', async () => {
    render(<DatasetTab />);
    fireEvent.click(await screen.findByTestId('dataset-cat-logic:xor'));
    await screen.findByTestId('sample-preview-grid');
    fireEvent.click(screen.getByTestId('encoding-toggle'));
    await waitFor(() => expect(encodingPreview).toHaveBeenCalled());
    expect(await screen.findByTestId('encoding-grid')).toBeTruthy();
  });
});
