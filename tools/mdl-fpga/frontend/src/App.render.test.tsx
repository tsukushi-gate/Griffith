// @vitest-environment jsdom
// App ブートの描画スモーク(実サーバの full_adder レスポンスを忠実に再現)。
// 「シミュレーションに接続中...」から先へ進み、回路図とプローブが出ることを保証する。
import { cleanup, render, screen, waitFor } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

afterEach(cleanup);   // vitest globals 無効のため自動クリーンアップは効かない
import App from './App';
import { useSimStore } from './store/simStore';
import { useViewerStore } from './store/viewerStore';

const META = {
  name: 'full_adder', in_width: 3, out_width: 20, score_bits: 5, k: 4,
  input_semantics: null, input_shape: null,
  nodes: [
    { id: 'n0_lutlayer', wire: 'w_n0_lutlayer', width: 64, type: 'lut_layer', alias: false, score_bits: null },
    { id: 'n1_lutlayer', wire: 'w_n1_lutlayer', width: 64, type: 'lut_layer', alias: false, score_bits: null },
    { id: 'n2_groupsum', wire: 'w_n2_groupsum', width: 20, type: 'group_sum', alias: false, score_bits: 5 },
  ],
  inputs: [{ id: 'x', width: 3, shape: null, semantics: null }],
  outputs: [{ id: 'y', width: 4, semantics: null }],
  edges: [
    { src: 'x', dst: 'n0_lutlayer', slot: 0 },
    { src: 'n0_lutlayer', dst: 'n1_lutlayer', slot: 0 },
    { src: 'n1_lutlayer', dst: 'n2_groupsum', slot: 0 },
    { src: 'n2_groupsum', dst: 'y', slot: 0 },
  ],
  cycle: 0, workdir: 'runs/x',
};

const LAYOUT = {
  version: 1,
  probes: [
    { id: 'p1', target: 'y_comb', slice: null, renderer: 'carry_sum',
      args: {}, pos: { x: 0, y: 0 }, pinned: false, label: null,
      depth: 1, scale: 1 },
    { id: 'p2', target: 'x', slice: null, renderer: 'adder_inputs',
      args: {}, pos: { x: 0, y: 0 }, pinned: false, label: null,
      depth: 1, scale: 1 },
  ],
  positions: {
    x: { x: 12, y: 22 }, n0_lutlayer: { x: 242, y: 12 },
    n1_lutlayer: { x: 512, y: 12 }, n2_groupsum: { x: 782, y: 12 },
    y: { x: 1052, y: 22 },
  },
  expanded: [], viewport: null,
};

const RENDERERS = {
  builtins: ['auto', 'bits', 'hex', 'uint', 'bars', 'image'],
  sidecar: ['adder_inputs', 'carry_sum'],
  sidecar_path: 'renderers.py', sidecar_error: null,
};

// ProjectPanel がマウント時に取得する(examples はダイアログ開時のみ=
// この throw するモックが「ブート時に叩かない」設計の回帰固定になっている)
const PROJECT = {
  model: 'examples/full_adder.ir.json', name: 'full_adder', mode: 'core',
  system_dir: null, pipelined: false, epoch: 0, recent: [],
};

// BoardPanel がマウント時に取得(detect はボタン押下時のみ=モックに無い URL
// への fetch は throw するので「自動実行しない」設計の回帰固定になっている)
const BOARDS = {
  boards: [
    { key: 'de0-cv', name: 'Terasic DE0-CV', tier: 1,
      device: '5CEBA4F23C7', family: 'Cyclone V',
      detection: { silk_hint: '基板に『DE0-CV』と印字' },
      resources: { unit: 'alm', alm: 18480 } },
  ],
};
const BOARD_STATE = { key: null, advice: null };

class FakeWebSocket {
  static OPEN = 1;
  readyState = 1;
  onopen: (() => void) | null = null;
  onclose: (() => void) | null = null;
  onmessage: ((ev: { data: string }) => void) | null = null;
  constructor(public url: string) {
    setTimeout(() => {
      this.onopen?.();
      this.onmessage?.({
        data: JSON.stringify({
          type: 'values', ok: true, cycle: 0,
          values: { x: '0', y: '0', y_comb: '0', n0_lutlayer: '0',
                    n1_lutlayer: '0', n2_groupsum: '0' },
          xz: [],
          probes: {
            p1: { kind: 'text', text: 'carry=0 sum=0(class 0)' },
            p2: { kind: 'text', text: '0 + 0 + cin0 = 0' },
          },
        }),
      });
    }, 0);
  }
  send() {}
  close() {}
}

beforeEach(() => {
  // ストアはモジュールシングルトンのためテスト間でリセット
  useSimStore.setState({ meta: null, frame: null, renderers: null });
  useViewerStore.setState({ probes: [], positions: {}, expanded: [],
                            viewport: null, loaded: false });
  vi.stubGlobal('WebSocket', FakeWebSocket as unknown as typeof WebSocket);
  vi.stubGlobal('ResizeObserver', class {
    observe() {}
    unobserve() {}
    disconnect() {}
  });
  vi.stubGlobal('fetch', vi.fn(async (url: string) => {
    const body =
      url === '/api/meta' ? META
      : url === '/api/layout' ? LAYOUT
      : url === '/api/renderers' ? RENDERERS
      : url === '/api/project' ? PROJECT
      : url === '/api/boards' ? BOARDS
      : url === '/api/board' ? BOARD_STATE
      : null;
    if (body === null) throw new Error(`unexpected fetch: ${url}`);
    return { ok: true, json: async () => body } as Response;
  }));
});

describe('App boot (system mode)', () => {
  it('allow_set_x=false で入力パネルが ROM 案内に置き換わる', async () => {
    const meta = {
      ...META, mode: 'system', allow_set_x: false, latency: 7,
      nodes: [...META.nodes,
              { id: 'img_idx', width: 10, type: 'system',
                path: 'u_sys.img_idx', label: 'ROM index' }],
    };
    (globalThis.fetch as ReturnType<typeof vi.fn>).mockImplementation(
      async (url: string) => {
        const body =
          url === '/api/meta' ? meta
          : url === '/api/layout' ? LAYOUT
          : url === '/api/renderers' ? RENDERERS
          : url === '/api/project' ? { ...PROJECT, mode: 'system' }
          : url === '/api/boards' ? BOARDS
          : url === '/api/board' ? BOARD_STATE
          : null;
        if (body === null) throw new Error(`unexpected fetch: ${url}`);
        return { ok: true, json: async () => body } as Response;
      });
    render(<App />);
    await waitFor(() => {
      expect(screen.getByText(/ROM から供給/)).toBeDefined();
    }, { timeout: 5000 });
    expect(screen.getByText('ROM index')).toBeDefined();
    expect(screen.queryByTitle('x[0]')).toBeNull();   // 入力トグルなし
  });
});

describe('App boot', () => {
  it('ローディングから回路図+プローブ表示まで到達する', async () => {
    render(<App />);
    expect(screen.getByText(/接続中/)).toBeDefined();

    await waitFor(() => {
      expect(screen.getByText('n0_lutlayer')).toBeDefined();
    }, { timeout: 5000 });

    // ポート・GroupSum ブロック・プローブカードが描画されている
    expect(screen.getAllByText(/入力 x/).length).toBeGreaterThan(0);
    expect(screen.getAllByText(/GroupSum/).length).toBeGreaterThan(0);
    await waitFor(() => {
      expect(screen.getByText('carry=0 sum=0(class 0)')).toBeDefined();
    }, { timeout: 5000 });
  });
});
