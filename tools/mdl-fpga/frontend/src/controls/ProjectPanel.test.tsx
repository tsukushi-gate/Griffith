// @vitest-environment jsdom
// プロジェクトペイン: 表示/開くダイアログ(examples はダイアログ開時フェッチ)/
// 未ビルド行の disabled+hint/切替成功で rehydrate/失敗でダイアログ内エラー。
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

const rehydrateApp = vi.fn(async () => {});
vi.mock('../boot/bootApp', () => ({
  bootApp: vi.fn(async () => {}),
  rehydrateApp,
}));

import { ProjectPanel } from './ProjectPanel';
import { useProjectStore } from '../store/projectStore';
import { useSimStore } from '../store/simStore';

afterEach(cleanup);

const PROJECT = {
  model: 'examples/full_adder.ir.json', name: 'full_adder', mode: 'core',
  system_dir: null, pipelined: false, epoch: 0,
  recent: [{ model: 'old/parity4.ir.json', name: 'parity4', system: null }],
};

const EXAMPLES = {
  examples: [
    { key: 'full_adder', name: 'full_adder',
      path: 'examples/full_adder.ir.json',
      modes: [{ mode: 'core', ready: true }] },
    { key: 'mnist/mnist24k', name: 'mnist24k',
      path: 'examples/mnist/mnist24k.ir.json',
      modes: [
        { mode: 'core', ready: true },
        { mode: 'system', ready: false, hint: 'build_mnist_demo.py で生成' },
      ] },
  ],
};

let openResponse: { status: number; body: unknown };
let openCalls: { url: string; body: unknown }[];

beforeEach(() => {
  rehydrateApp.mockClear();
  openCalls = [];
  openResponse = { status: 200,
                   body: { ok: true, name: 'mnist24k', mode: 'core',
                           epoch: 1 } };
  useSimStore.setState({
    meta: { name: 'full_adder', mode: 'core' } as never,
  });
  useProjectStore.setState({ project: null, examples: [],
                             examplesLoaded: false, opening: false,
                             error: null });
  vi.stubGlobal('fetch', vi.fn(
    async (url: string, init?: { method?: string; body?: string }) => {
      if (url === '/api/project') {
        return { ok: true, json: async () => PROJECT } as Response;
      }
      if (url === '/api/examples') {
        return { ok: true, json: async () => EXAMPLES } as Response;
      }
      if (url === '/api/project/open' && init?.method === 'POST') {
        openCalls.push({ url, body: JSON.parse(init.body ?? '{}') });
        return {
          ok: openResponse.status < 400,
          status: openResponse.status,
          json: async () => openResponse.body,
        } as Response;
      }
      throw new Error(`unexpected fetch: ${url}`);
    }));
});

describe('ProjectPanel', () => {
  it('プロジェクト名を表示し、ダイアログを開くと examples をフェッチする', async () => {
    render(<ProjectPanel />);
    expect(screen.getByText('full_adder')).toBeDefined();
    // ダイアログを開くまで /api/examples は叩かれない
    expect((globalThis.fetch as ReturnType<typeof vi.fn>).mock.calls
      .map((c) => c[0])).not.toContain('/api/examples');

    fireEvent.click(screen.getByText('開く…'));
    await waitFor(() => {
      // mnist24k は core / system の 2 行に出る
      expect(screen.getAllByText('mnist24k').length).toBe(2);
    });
    // 未ビルドの system 行は disabled+hint(テキストは複数ノードに分割される
    // ため関数マッチャで textContent 全体を照合)
    const systemLabel = screen.getByText(
      (_, el) => el?.textContent === 'system(未ビルド)');
    expect(systemLabel.closest('button')?.hasAttribute('disabled')).toBe(true);
    expect(screen.getByText(/build_mnist_demo\.py/)).toBeDefined();
  });

  it('イグザンプル選択で POST → 成功なら rehydrate される', async () => {
    render(<ProjectPanel />);
    fireEvent.click(screen.getByText('開く…'));
    await waitFor(() => {
      expect(screen.getAllByText('mnist24k').length).toBeGreaterThan(0);
    });
    fireEvent.click(screen.getAllByText('mnist24k')[0]);
    await waitFor(() => {
      expect(openCalls.length).toBe(1);
    });
    expect(openCalls[0].body).toEqual({ example: 'mnist/mnist24k',
                                        mode: 'core' });
    await waitFor(() => {
      expect(rehydrateApp).toHaveBeenCalledTimes(1);
    });
  });

  it('切替失敗はダイアログ内にエラー表示(rehydrate しない)', async () => {
    openResponse = { status: 400,
                     body: { detail: '起動失敗: X(現在のプロジェクトは継続)' } };
    render(<ProjectPanel />);
    fireEvent.click(screen.getByText('開く…'));
    await waitFor(() => {
      expect(screen.getAllByText('full_adder').length).toBeGreaterThan(0);
    });
    fireEvent.click(screen.getAllByText('full_adder')[1]); // ダイアログ内の行
    await waitFor(() => {
      expect(screen.getByText(/現在のプロジェクトは継続/)).toBeDefined();
    });
    expect(rehydrateApp).not.toHaveBeenCalled();
  });
});
