// @vitest-environment jsdom
// NodeParamForm のDWN kindバッジ + blackbox汎用パラメータエディタのRTLテスト。

import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type { IRGraphJson } from '../ir/schema';

const postWiringGenerate = vi.fn();

vi.mock('../api/rest', () => ({
  putCurrentIR: vi.fn().mockResolvedValue({ ok: true, warnings: [] }),
  postWiringGenerate: (req: unknown) => postWiringGenerate(req),
  ApiError: class ApiError extends Error {},
}));

const { useGraphStore } = await import('../store/graphStore');
const { NodeParamForm } = await import('./NodeParamForm');

/** 学習済み lut_layer(12→8、N=2)+ group_sum(k=2)の直列 IR(構造化配線のフォーム用) */
function irWithLutAndGroupSum(): IRGraphJson {
  return {
    format: 'magicdelta-ir',
    version: '0.2.0',
    metadata: {},
    inputs: [{ id: 'x', width: 12, shape: null, encoding: null, semantics: null }],
    outputs: [{ id: 'y', width: 2, shape: null, encoding: null, semantics: null }],
    nodes: [
      {
        id: 'lut',
        type: 'lut_layer',
        params: {
          in_width: 12,
          out_width: 8,
          n_inputs: 2,
          indices: [
            Array.from({ length: 8 }, (_, o) => (o * 2) % 12),
            Array.from({ length: 8 }, (_, o) => (o * 2 + 1) % 12),
          ],
          truth_tables: new Array(8).fill('0x6'),
        },
        train_info: { parameterization: 'softmax', connections: 'random', selection: 'soft' },
        meta: { editor: { position: { x: 4, y: 0 } } },
      },
      {
        id: 'gs',
        type: 'group_sum',
        params: { in_width: 8, k: 2, tau: 1.0 },
        train_info: {},
        meta: { editor: { position: { x: 300, y: 0 } } },
      },
    ],
    edges: [
      { src: 'x', dst: 'lut' },
      { src: 'lut', dst: 'gs' },
      { src: 'gs', dst: 'y' },
    ],
  };
}

function irWithBlackbox(): IRGraphJson {
  return {
    format: 'magicdelta-ir',
    version: '0.2.0',
    metadata: {},
    inputs: [{ id: 'x', width: 12, shape: null, encoding: null, semantics: null }],
    outputs: [{ id: 'y', width: 12, shape: null, encoding: null, semantics: null }],
    nodes: [
      {
        id: 'bb',
        type: 'blackbox',
        params: {
          torch_class: 'LogicAttention',
          config: { d_model: 4, n_heads: 1, mode: 'soft', similarity: 'xnor' },
          in_width: 12,
          out_width: 12,
        },
        train_info: {
          contract: {
            input: '(B, S, 4)のNCHW flatten(S=3)',
            output: '(B, S, 4)のflatten',
            hdl: 'softmax/argmaxを含むためHDL生成不可',
          },
        },
        meta: { editor: { position: { x: 4, y: 0 }, detail: 'LogicAttention d=4 h=1 soft' } },
      },
    ],
    edges: [{ src: 'x', dst: 'bb' }, { src: 'bb', dst: 'y' }],
  };
}

function irWithDwn(): IRGraphJson {
  return {
    format: 'magicdelta-ir',
    version: '0.2.0',
    metadata: {},
    inputs: [{ id: 'x', width: 12, shape: null, encoding: null, semantics: null }],
    outputs: [{ id: 'y', width: 8, shape: null, encoding: null, semantics: null }],
    nodes: [
      {
        id: 'dwn',
        type: 'lut_layer',
        params: {
          in_width: 12,
          out_width: 8,
          n_inputs: 6,
          indices: [[0], [1], [2], [3], [4], [5]].map((_, j) =>
            Array.from({ length: 8 }, (__, o) => (o * 6 + j) % 12),
          ),
          truth_tables: new Array(8).fill('0xaaaaaaaaaaaaaaaa'),
        },
        train_info: { kind: 'dwn', mapping: 'random', initialized: false },
        meta: { editor: { position: { x: 4, y: 0 } } },
      },
    ],
    edges: [{ src: 'x', dst: 'dwn' }, { src: 'dwn', dst: 'y' }],
  };
}

afterEach(() => cleanup());

describe('NodeParamForm blackbox editor', () => {
  beforeEach(() => {
    useGraphStore.getState().setIR(irWithBlackbox());
    useGraphStore.setState({ selectedNodeId: 'bb' });
  });

  it('torch_class/config/contract を表示し、in/out幅は編集不可', () => {
    render(<NodeParamForm />);
    const torch = screen.getByTestId('blackbox-torch-class') as HTMLInputElement;
    expect(torch.value).toBe('LogicAttention');
    const config = screen.getByTestId('blackbox-config') as HTMLTextAreaElement;
    expect(config.value).toContain('d_model');
    expect(screen.getByTestId('blackbox-contract').textContent).toContain('HDL生成不可');
  });

  it('torch_class編集がparamsへ反映される', () => {
    render(<NodeParamForm />);
    const torch = screen.getByTestId('blackbox-torch-class');
    fireEvent.change(torch, { target: { value: 'MyCustomModule' } });
    const node = useGraphStore.getState().ir!.nodes.find((n) => n.id === 'bb')!;
    expect(node.params.torch_class).toBe('MyCustomModule');
    // contract(in/out幅)は不変
    expect(node.params.in_width).toBe(12);
    expect(node.params.out_width).toBe(12);
  });

  it('config JSON編集がparamsへ反映され、不正JSONはエラー表示', () => {
    render(<NodeParamForm />);
    const config = screen.getByTestId('blackbox-config');
    fireEvent.change(config, { target: { value: '{"d_model": 8}' } });
    let node = useGraphStore.getState().ir!.nodes.find((n) => n.id === 'bb')!;
    expect(node.params.config).toEqual({ d_model: 8 });
    // 不正JSON: エラー表示され、paramsは更新されない(直前の有効値を保持)
    fireEvent.change(config, { target: { value: '{bad json' } });
    expect(screen.getByTestId('blackbox-config-error')).toBeTruthy();
    node = useGraphStore.getState().ir!.nodes.find((n) => n.id === 'bb')!;
    expect(node.params.config).toEqual({ d_model: 8 });
  });
});

describe('NodeParamForm DWN kindバッジ', () => {
  beforeEach(() => {
    useGraphStore.getState().setIR(irWithDwn());
    useGraphStore.setState({ selectedNodeId: 'dwn' });
  });

  it('kindバッジとmethod提案を表示する', () => {
    render(<NodeParamForm />);
    expect(screen.getByTestId('node-kind-badge').textContent).toContain('DWN');
    expect(screen.getByTestId('dwn-method-hint')).toBeTruthy();
  });
});

describe('NodeParamForm 構造化配線', () => {
  beforeEach(() => {
    postWiringGenerate.mockReset();
    useGraphStore.getState().setIR(irWithLutAndGroupSum());
    useGraphStore.setState({ selectedNodeId: 'lut' });
  });

  it('構造化配線を選ぶと引数フォームが出て、train_info に既定の引数が入る', () => {
    render(<NodeParamForm />);
    expect(screen.queryByTestId('wiring-spec-form')).toBeNull();
    fireEvent.change(screen.getByTestId('lut-connections'), { target: { value: 'butterfly' } });
    expect(screen.getByTestId('wiring-spec-form')).toBeTruthy();
    expect((screen.getByTestId('wiring-field-stride') as HTMLInputElement).value).toBe('1');
    const node = useGraphStore.getState().ir!.nodes.find((n) => n.id === 'lut')!;
    expect(node.train_info.connections).toBe('butterfly');
    expect(node.train_info.wiring).toEqual({ stride: 1 });
    // random へ戻すと引数が消える
    fireEvent.change(screen.getByTestId('lut-connections'), { target: { value: 'random' } });
    const back = useGraphStore.getState().ir!.nodes.find((n) => n.id === 'lut')!;
    expect(back.train_info.connections).toBe('random');
    expect('wiring' in back.train_info).toBe(false);
  });

  it('「配線を生成して適用」が API を呼び、indices と引数を IR へ書く', async () => {
    const indices = [
      Array.from({ length: 8 }, (_, o) => o % 12),
      Array.from({ length: 8 }, (_, o) => (o + 4) % 12),
    ];
    postWiringGenerate.mockResolvedValue({
      kind: 'butterfly',
      indices,
      stats: { coverage: 0.5, usage_hist: { '0': 6, '1': 6, '2': 0, '3+': 0 }, same_input: 0, n_gates: 8 },
    });
    render(<NodeParamForm />);
    fireEvent.change(screen.getByTestId('lut-connections'), { target: { value: 'butterfly' } });
    fireEvent.change(screen.getByTestId('wiring-field-stride'), { target: { value: '4' } });
    fireEvent.click(screen.getByTestId('wiring-dedupe'));
    fireEvent.click(screen.getByTestId('wiring-generate'));
    await waitFor(() => expect(screen.getByTestId('wiring-stats')).toBeTruthy());

    expect(postWiringGenerate).toHaveBeenCalledWith({
      kind: 'butterfly', in_width: 12, out_width: 8, n_inputs: 2,
      spec: { stride: 4 }, dedupe: true,
    });
    const node = useGraphStore.getState().ir!.nodes.find((n) => n.id === 'lut')!;
    expect(node.params.indices).toEqual(indices);
    expect(node.params.truth_tables).toEqual(new Array(8).fill('0x6')); // 真理値表は保たれる
    expect(node.train_info.wiring).toEqual({ stride: 4, dedupe: true });
    expect(screen.getByTestId('wiring-stats').textContent).toContain('被覆 6/12');
  });

  it('引数が不正なら API を呼ばずエラーを出す', () => {
    render(<NodeParamForm />);
    fireEvent.change(screen.getByTestId('lut-connections'), { target: { value: 'butterfly' } });
    fireEvent.change(screen.getByTestId('wiring-field-stride'), { target: { value: 'x' } });
    fireEvent.click(screen.getByTestId('wiring-generate'));
    expect(screen.getByTestId('wiring-error')).toBeTruthy();
    expect(postWiringGenerate).not.toHaveBeenCalled();
  });
});

describe('NodeParamForm group_sum の bias', () => {
  beforeEach(() => {
    useGraphStore.getState().setIR(irWithLutAndGroupSum());
    useGraphStore.setState({ selectedNodeId: 'gs' });
  });

  it('長さ k の整数列を params.bias へ書き、不正ならエラーで据え置く', () => {
    render(<NodeParamForm />);
    fireEvent.change(screen.getByTestId('group-sum-bias'), { target: { value: '0,1' } });
    let gs = useGraphStore.getState().ir!.nodes.find((n) => n.id === 'gs')!;
    expect(gs.params.bias).toEqual([0, 1]);
    fireEvent.change(screen.getByTestId('group-sum-bias'), { target: { value: '1,2,3' } });
    expect(screen.getByTestId('group-sum-bias-error')).toBeTruthy();
    gs = useGraphStore.getState().ir!.nodes.find((n) => n.id === 'gs')!;
    expect(gs.params.bias).toEqual([0, 1]);
    // 空に戻すとバイアス無し
    fireEvent.change(screen.getByTestId('group-sum-bias'), { target: { value: '' } });
    gs = useGraphStore.getState().ir!.nodes.find((n) => n.id === 'gs')!;
    expect('bias' in gs.params).toBe(false);
  });
});
