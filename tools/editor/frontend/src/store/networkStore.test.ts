// 複数ネットワークタブのストアテスト。
// タブ切替でアクティブIRが入れ替わること、切替前に現タブへ書き戻されること、
// プロジェクトからの読込で復元されることを固定する。

import { beforeEach, describe, expect, it, vi } from 'vitest';
import type { IRGraphJson } from '../ir/schema';
import { useGraphStore } from './graphStore';
import { MAIN_NETWORK, useNetworkStore } from './networkStore';

vi.mock('../api/rest', () => ({
  putCurrentIR: vi.fn().mockResolvedValue({ ok: true, warnings: [] }),
}));

function irNamed(name: string, outWidth: number): IRGraphJson {
  return {
    format: 'magicdelta-ir',
    version: '0.2.0',
    metadata: { name, editor: {} },
    inputs: [{ id: 'x', width: 4, shape: null, encoding: null, semantics: null }],
    outputs: [{ id: 'y', width: outWidth, shape: null, encoding: null, semantics: null }],
    nodes: [],
    edges: [],
  };
}

beforeEach(() => {
  useNetworkStore.setState({ networks: {}, order: [], active: null });
  useGraphStore.getState().setIR(irNamed('Main', 8));
});

describe('networkStore', () => {
  it('ensureSeededで現IRをMainタブとして取り込む', () => {
    useNetworkStore.getState().ensureSeeded();
    const s = useNetworkStore.getState();
    expect(s.order).toEqual([MAIN_NETWORK]);
    expect(s.active).toBe(MAIN_NETWORK);
    expect(s.networks[MAIN_NETWORK].outputs[0].width).toBe(8);
  });

  it('addNetwork+switchToでアクティブIRが入れ替わる', () => {
    const net = useNetworkStore.getState();
    net.ensureSeeded();
    net.addNetwork('Runtime'); // 空ネットワークを追加しアクティブ化
    expect(useNetworkStore.getState().active).toBe('Runtime');
    expect(useGraphStore.getState().ir!.nodes).toEqual([]);

    // Mainへ戻すとMainのIR(width=8)が復帰する
    net.switchTo(MAIN_NETWORK);
    expect(useNetworkStore.getState().active).toBe(MAIN_NETWORK);
    expect(useGraphStore.getState().ir!.outputs[0].width).toBe(8);
  });

  it('切替前に現タブへ書き戻される(未保存編集を失わない)', () => {
    const net = useNetworkStore.getState();
    net.ensureSeeded();
    net.addNetwork('Runtime');
    // RuntimeタブでIRを編集(出力幅を変更したIRを差し替え)
    useGraphStore.getState().setIR(irNamed('Runtime', 3));
    // Mainへ切替 → Runtimeの編集が networks に退避される
    net.switchTo(MAIN_NETWORK);
    net.switchTo('Runtime');
    expect(useGraphStore.getState().ir!.outputs[0].width).toBe(3);
  });

  it('loadNetworksでプロジェクトから復元しactiveをロードする', () => {
    useNetworkStore.getState().loadNetworks(
      { Main: irNamed('Main', 8), Runtime: irNamed('Runtime', 3) },
      'Runtime',
    );
    const s = useNetworkStore.getState();
    expect(s.order).toEqual(['Main', 'Runtime']);
    expect(s.active).toBe('Runtime');
    expect(useGraphStore.getState().ir!.outputs[0].width).toBe(3);
  });
});
