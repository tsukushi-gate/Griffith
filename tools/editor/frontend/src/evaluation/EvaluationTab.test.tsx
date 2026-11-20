// @vitest-environment jsdom
// EVALUATIONタブのRTLテスト。api/storeはモックし、
// 評価実行→一覧描画、誤答フィルタ、混同行列セルクリック→フィルタ、
// 行クリック→反例トレース注入(playgroundStore+EDIT/Probe遷移)を検証する。

import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type { DatasetInfo, RunSummary } from '../api/rest';
import type { EvalItem, EvaluateResponse } from './api';

const CATALOG: DatasetInfo[] = [
  { name: 'logic:xor', label: 'XOR', in_width: 2, out_width: 1, kind: 'binary_vector' },
];

const RUNS: RunSummary[] = [];

// binary_vector 評価結果(1件誤答を含む)
const ITEMS: EvalItem[] = [
  {
    index: 0, label: null, pred: null, label_bits: [0], pred_bits: [0],
    correct: true, soft_confidence: 0.9, soft_margin: 0.9, hard_soft_mismatch: false,
    input_bits: [0, 0], image: null,
  },
  {
    index: 1, label: null, pred: null, label_bits: [1], pred_bits: [0],
    correct: false, soft_confidence: 0.4, soft_margin: 0.1, hard_soft_mismatch: true,
    input_bits: [0, 1], image: null,
  },
];

const META: EvaluateResponse = {
  source_kind: 'current', dataset: 'logic:xor', executor: null, kind: 'binary_vector',
  num_classes: null, num_samples: 2, page: 0, page_size: 500, num_pages: 1,
  cached: false, on_demand: false, cap: 10000, accuracy: 0.5, num_correct: 1,
  num_errors: 1, num_mismatch: 1, confusion: null, fingerprint: 'abc', items: ITEMS,
};

const getDatasets = vi.fn();
const getRuns = vi.fn();
const evaluateAll = vi.fn();
const exportCsv = vi.fn();
const pinPattern = vi.fn();

vi.mock('../api/rest', () => ({
  getDatasets: () => getDatasets(),
  getRuns: () => getRuns(),
}));

vi.mock('./api', () => ({
  evaluateAll: (...a: unknown[]) => evaluateAll(...a),
  exportCsv: (...a: unknown[]) => exportCsv(...a),
}));

vi.mock('../playground/playgroundStore', () => ({
  usePlaygroundStore: { getState: () => ({ pinPattern }) },
}));

const applyLens = vi.fn();
vi.mock('../store/lens', () => ({ applyLens: (...a: unknown[]) => applyLens(...a) }));

const toastWarning = vi.fn();
vi.mock('sonner', () => ({ toast: { warning: (...a: unknown[]) => toastWarning(...a), error: vi.fn() } }));

const { EvaluationTab } = await import('./EvaluationTab');

beforeEach(() => {
  getDatasets.mockResolvedValue(CATALOG);
  getRuns.mockResolvedValue({ runs: RUNS });
  evaluateAll.mockResolvedValue({ meta: META, items: ITEMS });
});

afterEach(() => {
  cleanup();
  vi.clearAllMocks();
});

async function runEval() {
  render(<EvaluationTab />);
  await screen.findByTestId('eval-dataset-select');
  fireEvent.click(screen.getByTestId('eval-run-button'));
  await screen.findByTestId('eval-summary');
}

describe('EvaluationTab', () => {
  it('評価を実行すると全件が一覧表示される', async () => {
    await runEval();
    expect(screen.getByTestId('result-table')).toBeTruthy();
    expect(screen.getByTestId('result-row-0')).toBeTruthy();
    expect(screen.getByTestId('result-row-1')).toBeTruthy();
    // サマリに誤答件数が出る
    expect(screen.getByTestId('eval-summary').textContent).toContain('誤答 1');
  });

  it('誤答のみフィルタで正解行が消える', async () => {
    await runEval();
    fireEvent.click(screen.getByTestId('filter-errors'));
    await waitFor(() => expect(screen.queryByTestId('result-row-0')).toBeNull());
    expect(screen.getByTestId('result-row-1')).toBeTruthy();
  });

  it('hard/soft不一致フィルタが効く', async () => {
    await runEval();
    fireEvent.click(screen.getByTestId('filter-mismatch'));
    await waitFor(() => expect(screen.queryByTestId('result-row-0')).toBeNull());
    expect(screen.getByTestId('result-row-1')).toBeTruthy();
  });

  it('誤答行クリックで反例トレース注入(pinPattern+Probeレンズ)', async () => {
    await runEval();
    fireEvent.click(screen.getByTestId('result-row-1'));
    // input_bits=[0,1] → k=2
    expect(pinPattern).toHaveBeenCalledWith(2);
    expect(applyLens).toHaveBeenCalledWith('probe');
  });
});

describe('EvaluationTab confusion', () => {
  const CLS_ITEMS: EvalItem[] = [
    { index: 0, label: 0, pred: 0, label_bits: null, pred_bits: null, correct: true, soft_confidence: 0.9, soft_margin: 0.8, hard_soft_mismatch: false, input_bits: null, image: 'data:image/png;base64,AAAA' },
    { index: 1, label: 1, pred: 0, label_bits: null, pred_bits: null, correct: false, soft_confidence: 0.5, soft_margin: 0.1, hard_soft_mismatch: false, input_bits: null, image: 'data:image/png;base64,AAAA' },
  ];
  const CLS_META: EvaluateResponse = {
    ...META, kind: 'classification', num_classes: 2,
    confusion: [[1, 0], [1, 0]], items: CLS_ITEMS, num_errors: 1,
  };

  it('混同行列に切替→セルクリックで一覧が該当セルにフィルタ', async () => {
    evaluateAll.mockResolvedValue({ meta: CLS_META, items: CLS_ITEMS });
    render(<EvaluationTab />);
    await screen.findByTestId('eval-dataset-select');
    fireEvent.click(screen.getByTestId('eval-run-button'));
    await screen.findByTestId('eval-summary');
    // 混同行列へ切替
    fireEvent.click(screen.getByTestId('toggle-view'));
    const canvas = await screen.findByTestId('confusion-canvas');
    // jsdomはgetBoundingClientRectが0を返すため、2x2セル(68x68)の矩形をスタブする
    canvas.getBoundingClientRect = () =>
      ({ left: 0, top: 0, width: 68, height: 68, right: 68, bottom: 68, x: 0, y: 0, toJSON: () => ({}) }) as DOMRect;
    // 左上セル=(実際0, 予測0)をクリック
    fireEvent.click(canvas, { clientX: 5, clientY: 5 });
    // (label0,pred0)= index0 のみ表示
    await waitFor(() => expect(screen.getByTestId('result-row-0')).toBeTruthy());
    expect(screen.queryByTestId('result-row-1')).toBeNull();
  });

  it('画像入力の行クリックは注入せず警告トースト', async () => {
    evaluateAll.mockResolvedValue({ meta: CLS_META, items: CLS_ITEMS });
    render(<EvaluationTab />);
    await screen.findByTestId('eval-dataset-select');
    fireEvent.click(screen.getByTestId('eval-run-button'));
    await screen.findByTestId('eval-summary');
    fireEvent.click(screen.getByTestId('result-row-1'));
    expect(toastWarning).toHaveBeenCalled();
    expect(pinPattern).not.toHaveBeenCalled();
  });
});
