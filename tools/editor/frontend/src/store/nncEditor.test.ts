//(レイヤーコメント)(NNC縦整列)のgraphStoreテスト。
// コメントはmetadata.editor.commentsへ非破壊で往復保存され、
// 縦整列は座標のみ更新しIR構造(nodes/edges)を変えないことを確認する。

import { beforeEach, describe, expect, it, vi } from 'vitest';
import type { IRGraphJson } from '../ir/schema';
import { commentsRecordOf, useGraphStore } from './graphStore';

vi.mock('../api/rest', () => ({
  putCurrentIR: vi.fn().mockResolvedValue({ ok: true, warnings: [] }),
}));

const pos = (x: number, y: number) => ({ editor: { position: { x, y } } });

function ir(): IRGraphJson {
  return {
    format: 'magicdelta-ir',
    version: '0.2.0',
    metadata: {
      editor: { ports: { x: { position: { x: 0, y: 0 } }, y: { position: { x: 600, y: 0 } } } },
    },
    inputs: [{ id: 'x', width: 2, shape: null, encoding: null, semantics: null }],
    outputs: [{ id: 'y', width: 2, shape: null, encoding: null, semantics: null }],
    nodes: [
      {
        id: 'n1',
        type: 'lut_layer',
        params: {
          in_width: 2,
          out_width: 2,
          n_inputs: 2,
          indices: [[0, 0], [1, 1]],
          truth_tables: ['0x6', '0x6'],
        },
        train_info: { parameterization: 'softmax', connections: 'random', selection: 'soft' },
        meta: pos(200, 300),
      },
    ],
    edges: [
      { src: 'x', dst: 'n1' },
      { src: 'n1', dst: 'y' },
    ],
  };
}

beforeEach(() => {
  useGraphStore.getState().setIR(ir());
});

describe('setComment', () => {
  it('metadata.editor.commentsへ保存され、空文字で削除される(roundtrip)', () => {
    useGraphStore.getState().setComment('n1', '  重要な層  ');
    const saved = useGraphStore.getState().ir!;
    expect(commentsRecordOf(saved)['n1']).toBe('重要な層'); // trim保存

    // roundtrip: 同じIRを再ロードしてもコメントが保持される
    useGraphStore.getState().setIR(saved);
    expect(commentsRecordOf(useGraphStore.getState().ir!)['n1']).toBe('重要な層');

    // 空文字でキー削除
    useGraphStore.getState().setComment('n1', '   ');
    expect(commentsRecordOf(useGraphStore.getState().ir!)['n1']).toBeUndefined();
  });

  it('コメント編集でIRのノード構造は不変', () => {
    const before = useGraphStore.getState().ir!;
    useGraphStore.getState().setComment('n1', 'メモ');
    const after = useGraphStore.getState().ir!;
    expect(after.nodes).toEqual(before.nodes);
    expect(after.edges).toEqual(before.edges);
  });
});

describe('alignVerticalNnc', () => {
  it('全ノードを同一xで縦積みし、IR構造は不変', () => {
    useGraphStore.getState().alignVerticalNnc();
    const s = useGraphStore.getState();
    const xs = new Set(s.nodes.map((n) => Math.round(n.position.x)));
    expect(xs.size).toBe(1); // 全ノード同一x
    // 座標は変わるが、ノード/エッジの本数は不変
    expect(s.ir!.nodes).toHaveLength(1);
    expect(s.ir!.edges).toHaveLength(2);
    // 入力ポート(rank0) は n1(rank1) より上(yが小さい)
    const yx = s.nodes.find((n) => n.id === 'x')!.position.y;
    const yn1 = s.nodes.find((n) => n.id === 'n1')!.position.y;
    expect(yx).toBeLessThan(yn1);
  });
});
