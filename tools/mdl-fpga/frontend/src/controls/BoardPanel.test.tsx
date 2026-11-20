// @vitest-environment jsdom
// ボードペイン: 認識が自動実行されない/一意候補は自動選択/複数候補は
// silk_hint 付きダイアログ/実機未検証バッジ/advise 表示/検出失敗の文言。
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { BoardPanel } from './BoardPanel';
import { useBoardStore } from '../store/boardStore';

afterEach(cleanup);

const DE0CV = {
  key: 'de0-cv', name: 'Terasic DE0-CV', tier: 1,
  device: '5CEBA4F23C7', family: 'Cyclone V',
  detection: { silk_hint: '基板に『DE0-CV』と印字' },
  resources: { unit: 'alm', alm: 18480 },
};
const DE10 = {
  key: 'de10-lite', name: 'Terasic DE10-Lite', tier: 2, unverified: true,
  device: '10M50DAF484C7G', family: 'MAX 10',
  detection: { silk_hint: '基板に『DE10-Lite』と印字' },
  resources: { unit: 'le', le: 49760 },
};
const ADVICE = {
  gates: 3, spatial_ok: true, overlay_ok: true, spatial_alm_needed: 2,
  messages: ['空間焼込み(spatial): 搭載可能'], unit: 'alm',
  calibrated: true,
};

let detectResponse: { status: number; body: unknown };
let putBoardCalls: unknown[];
let detectCalls: number;

beforeEach(() => {
  putBoardCalls = [];
  detectCalls = 0;
  detectResponse = {
    status: 200,
    body: {
      observation: { cables: [], jtag: [], serial: [], notes: [] },
      candidates: [{ key: 'de0-cv', name: 'Terasic DE0-CV', tier: 1,
                     confidence: 0.6, reasons: ['部品名一致'],
                     silk_hint: '基板に『DE0-CV』と印字' }],
      best: { key: 'de0-cv', name: 'Terasic DE0-CV', tier: 1,
              confidence: 0.6, reasons: ['部品名一致'] },
    },
  };
  useBoardStore.setState({ boards: [], state: null, detecting: false,
                           detectResult: null, candidates: null,
                           error: null });
  vi.stubGlobal('fetch', vi.fn(
    async (url: string, init?: { method?: string; body?: string }) => {
      if (url === '/api/boards') {
        return { ok: true,
                 json: async () => ({ boards: [DE0CV, DE10] }) } as Response;
      }
      if (url === '/api/board' && init?.method === 'PUT') {
        const req = JSON.parse(init.body ?? '{}') as { key: string | null };
        putBoardCalls.push(req.key);
        const advice = req.key ? { ...ADVICE,
                                   calibrated: req.key === 'de0-cv' } : null;
        return { ok: true,
                 json: async () => ({ ok: true, key: req.key, advice }),
        } as Response;
      }
      if (url === '/api/board') {
        return { ok: true,
                 json: async () => ({ key: null, advice: null }) } as Response;
      }
      if (url === '/api/board/detect' && init?.method === 'POST') {
        detectCalls += 1;
        return { ok: detectResponse.status < 400,
                 status: detectResponse.status,
                 json: async () => detectResponse.body } as Response;
      }
      throw new Error(`unexpected fetch: ${url}`);
    }));
});

describe('BoardPanel', () => {
  it('マウント時に detect を自動実行しない(一覧と選択状態のみ取得)', async () => {
    render(<BoardPanel />);
    await waitFor(() => {
      expect(useBoardStore.getState().boards.length).toBe(2);
    });
    expect(detectCalls).toBe(0);
  });

  it('認識ボタン → 一意候補なら自動選択(PUT /api/board)', async () => {
    render(<BoardPanel />);
    fireEvent.click(screen.getByText('認識'));
    await waitFor(() => {
      expect(detectCalls).toBe(1);
      expect(putBoardCalls).toEqual(['de0-cv']);
    });
    // 搭載可能性の表示: バッジ=有り+advisor メッセージ
    await waitFor(() => {
      expect(screen.getByText('有り')).toBeDefined();
      expect(screen.getByText(/空間焼込み/)).toBeDefined();
    });
  });

  it('複数候補は silk_hint 付きダイアログ → 選択で PUT', async () => {
    const cand2 = { key: 'de10-lite', name: 'Terasic DE10-Lite', tier: 2,
                    confidence: 0.6, reasons: ['IDCODE一致'],
                    silk_hint: '基板に『DE10-Lite』と印字',
                    unverified: true };
    detectResponse = {
      status: 200,
      body: {
        observation: { cables: [], jtag: [], serial: [], notes: [] },
        candidates: [
          { key: 'de0-cv', name: 'Terasic DE0-CV', tier: 1, confidence: 0.6,
            reasons: [], silk_hint: '基板に『DE0-CV』と印字' },
          cand2,
        ],
        best: null,
      },
    };
    render(<BoardPanel />);
    fireEvent.click(screen.getByText('認識'));
    await waitFor(() => {
      expect(screen.getByText(/シルク印字/)).toBeDefined();
      expect(screen.getByText('基板に『DE10-Lite』と印字')).toBeDefined();
    });
    expect(putBoardCalls).toEqual([]);   // ユーザー確認まで選択しない
    fireEvent.click(screen.getByText('基板に『DE0-CV』と印字'));
    await waitFor(() => {
      expect(putBoardCalls).toEqual(['de0-cv']);
    });
  });

  it('実機未検証ボードはバッジ表示', async () => {
    useBoardStore.setState({
      boards: [DE0CV, DE10],
      state: { key: 'de10-lite', advice: { ...ADVICE, calibrated: false } },
    });
    render(<BoardPanel />);
    expect(screen.getByText('実機未検証')).toBeDefined();
    expect(screen.getByText(/10M50DAF484C7G/)).toBeDefined();
  });

  it('搭載可能性バッジは3段階(有り/やや有り/無し)', async () => {
    useBoardStore.setState({
      boards: [DE0CV],
      state: { key: 'de0-cv',
               advice: { ...ADVICE, spatial_ok: false, overlay_ok: true } },
    });
    const { unmount } = render(<BoardPanel />);
    expect(screen.getByText(/やや有り/)).toBeDefined();
    unmount();
    useBoardStore.setState({
      boards: [DE0CV],
      state: { key: 'de0-cv',
               advice: { ...ADVICE, spatial_ok: false, overlay_ok: false } },
    });
    render(<BoardPanel />);
    expect(screen.getByText('無し')).toBeDefined();
  });

  it('検出失敗はエラー文言を表示', async () => {
    detectResponse = { status: 501, body: { detail: 'detect 未対応' } };
    render(<BoardPanel />);
    fireEvent.click(screen.getByText('認識'));
    await waitFor(() => {
      expect(screen.getByText(/detect 未対応/)).toBeDefined();
    });
  });
});
