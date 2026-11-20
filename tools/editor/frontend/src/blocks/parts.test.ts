// モデル部品ジェネレータ(Attention/Transformer/PE/Embedding/DWN/UpSample)の単体テスト。
// - blackbox params/contract が to_ir.py の _emit_attention_blackbox 規約と一致
// - positionCodes/positionBitsFlat が positional.py position_codes とビット一致
//   (期待値は Python position_codes から採取した参照)
// - 生成部分グラフが shapeInference 診断ゼロで接続できる
// - blockIoWidths の整合

import { describe, expect, it } from 'vitest';
import { inferShapes } from '../ir/shapeInference';
import type { IRGraphJson, IRNodeJson } from '../ir/schema';
import {
  attentionBlackboxBlock,
  blockIoWidths,
  generateBlock,
  positionBitsFlat,
  positionCodes,
  positionalCodeValid,
  templateToGraph,
  type BlockParams,
  type IdAllocator,
} from './templates';

function makeAlloc(start = 1): IdAllocator {
  let seq = start;
  return (suffix) => `u${seq++}_${suffix}`;
}

function diagnosticsOf(ir: IRGraphJson) {
  return inferShapes(ir.nodes, ir.edges, ir.inputs).diagnostics;
}

describe('positionCodes(positional.py position_codes と一致)', () => {
  // 参照値は Python position_codes から採取(位置優先flatten: index p*width+j)
  it('binary(seq_len=4, width=3)', () => {
    expect(positionBitsFlat(4, 3, 'binary')).toEqual([
      0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0,
    ]);
  });
  it('gray(seq_len=4, width=3)', () => {
    expect(positionBitsFlat(4, 3, 'gray')).toEqual([
      0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0,
    ]);
  });
  it('thermometer(seq_len=4, width=4)', () => {
    expect(positionBitsFlat(4, 4, 'thermometer')).toEqual([
      0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0,
    ]);
  });
  it('テーブル形状は (seqLen, width)', () => {
    expect(positionCodes(4, 3, 'binary').length).toBe(4);
    expect(positionCodes(4, 3, 'binary')[0].length).toBe(3);
  });
  it('コード制約(binary/gray: S≦2^w、thermometer: w≧S-1)', () => {
    expect(positionalCodeValid(4, 2, 'binary')).toBe(true); // 4≦2^2
    expect(positionalCodeValid(5, 2, 'binary')).toBe(false); // 5>4
    expect(positionalCodeValid(4, 3, 'thermometer')).toBe(true); // 3≧3
    expect(positionalCodeValid(4, 2, 'thermometer')).toBe(false); // 2<3
  });
});

describe('attention/transformer blackbox', () => {
  it('LogicAttention blackboxの params/contract が to_ir規約と一致', () => {
    const tpl = attentionBlackboxBlock(
      { dModel: 4, seqLen: 3, nHeads: 2, mode: 'soft' },
      makeAlloc(),
    );
    expect(tpl.nodes).toHaveLength(1);
    const n = tpl.nodes[0];
    expect(n.type).toBe('blackbox');
    expect(n.params.torch_class).toBe('LogicAttention');
    expect(n.params.config).toEqual({
      d_model: 4,
      n_heads: 2,
      mode: 'soft',
      similarity: 'xnor',
    });
    expect(n.params.in_width).toBe(12);
    expect(n.params.out_width).toBe(12);
    const contract = (n.train_info.contract ?? {}) as Record<string, string>;
    expect(contract.input).toBe('(B, S, 4)のNCHW flatten(S=3)');
    expect(contract.output).toBe('(B, S, 4)のflatten');
    expect(contract.hdl).toContain('HDL生成不可');
  });

  it('generateBlock(transformer)は幅不変blackboxを生成', () => {
    const spec: BlockParams = {
      kind: 'transformer',
      params: { dModel: 8, seqLen: 2, nHeads: 2, norm: 'layernorm', merge: 'xor' },
    };
    const tpl = generateBlock(spec, makeAlloc());
    const n = tpl.nodes[0];
    expect(n.params.torch_class).toBe('LogicTransformerBlock');
    expect(n.params.config).toEqual({
      d_model: 8,
      n_heads: 2,
      norm: 'layernorm',
      merge: 'xor',
    });
    expect(n.params.in_width).toBe(16);
    expect(n.params.out_width).toBe(16);
    expect(blockIoWidths(spec)).toEqual({ in: 16, out: 16 });
  });
});

describe('positional encoding', () => {
  it('constant + concat を生成し、bitsが position_codes と一致', () => {
    const spec: BlockParams = {
      kind: 'positional_encoding',
      params: { inWidth: 8, seqLen: 4, codeWidth: 3, code: 'binary' },
    };
    const tpl = generateBlock(spec, makeAlloc());
    expect(tpl.nodes.map((n) => n.type).sort()).toEqual(['concat', 'constant']);
    const constNode = tpl.nodes.find((n) => n.type === 'constant') as IRNodeJson;
    expect(constNode.params.width).toBe(12);
    expect(constNode.params.bits).toEqual(positionBitsFlat(4, 3, 'binary'));
    const catNode = tpl.nodes.find((n) => n.type === 'concat') as IRNodeJson;
    expect(catNode.params.in_widths).toEqual([8, 12]);
    // 内部エッジ: constant → concat slot1
    expect(tpl.edges).toEqual([
      { src: constNode.id, dst: catNode.id, dst_slot: 1 },
    ]);
    expect(blockIoWidths(spec)).toEqual({ in: 8, out: 20 });
  });

  it('接続後に shapeInference 診断ゼロ', () => {
    const spec: BlockParams = {
      kind: 'positional_encoding',
      params: { inWidth: 8, seqLen: 4, codeWidth: 3, code: 'gray' },
    };
    const tpl = generateBlock(spec, makeAlloc());
    const ir = templateToGraph('pe', tpl, 8, 20);
    expect(diagnosticsOf(ir).filter((d) => d.level === 'error')).toEqual([]);
  });
});

describe('embedding / dwn / upsample', () => {
  it('embeddingは lut_layer(vocab→dim)', () => {
    const spec: BlockParams = { kind: 'embedding', params: { vocab: 16, dim: 8 } };
    const tpl = generateBlock(spec, makeAlloc());
    const n = tpl.nodes[0];
    expect(n.type).toBe('lut_layer');
    expect(n.params.in_width).toBe(16);
    expect(n.params.out_width).toBe(8);
    expect(n.train_info.kind).toBe('embedding');
    expect(blockIoWidths(spec)).toEqual({ in: 16, out: 8 });
  });

  it('dwnは lut_layer + train_info.kind=dwn', () => {
    const spec: BlockParams = {
      kind: 'dwn',
      params: { inWidth: 16, outWidth: 12, nInputs: 6, mapping: 'random' },
    };
    const tpl = generateBlock(spec, makeAlloc());
    const n = tpl.nodes[0];
    expect(n.type).toBe('lut_layer');
    expect(n.train_info.kind).toBe('dwn');
    expect(n.train_info.mapping).toBe('random');
    expect(n.params.n_inputs).toBe(6);
    expect(n.params.out_width).toBe(12);
    // draft(build時初期化)
    expect(n.train_info.initialized).toBe(false);
    expect(blockIoWidths(spec)).toEqual({ in: 16, out: 12 });
  });

  it('llvit複合は埋込lut+PE(const/concat)+Transformer×N+分類ヘッド', () => {
    const spec: BlockParams = {
      kind: 'llvit',
      params: {
        seqLen: 4,
        dEmbed: 6,
        posBits: 2,
        depth: 2,
        nHeads: 2,
        patchBits: 12,
        numClasses: 10,
      },
    };
    const tpl = generateBlock(spec, makeAlloc());
    const byType = tpl.nodes.reduce<Record<string, IRNodeJson[]>>((acc, n) => {
      (acc[n.type] ??= []).push(n);
      return acc;
    }, {});
    // 構成ノード型の存在
    expect(byType.constant).toHaveLength(1); // PE定数線
    expect(byType.concat).toHaveLength(1); // PE連結
    expect(byType.blackbox).toHaveLength(2); // Transformer×depth
    expect(byType.lut_layer).toHaveLength(2); // patch_embed + head
    expect(byType.group_sum).toHaveLength(1);

    const dModel = 6 + 2; // dEmbed + posBits
    const tokenWidth = 4 * dModel; // 32
    // PE: 埋込幅 + 位置幅 = トークン幅
    expect(byType.concat[0].params.in_widths).toEqual([4 * 6, 4 * 2]);
    expect(byType.constant[0].params.width).toBe(4 * 2);
    expect(byType.constant[0].params.bits).toEqual(positionBitsFlat(4, 2, 'binary'));
    // 全 Transformer blackbox は幅不変 tokenWidth
    for (const bb of byType.blackbox) {
      expect(bb.params.torch_class).toBe('LogicTransformerBlock');
      expect(bb.params.in_width).toBe(tokenWidth);
      expect(bb.params.out_width).toBe(tokenWidth);
    }
    // 分類ヘッド: tokenWidth入力 → GroupSum(k=10)
    expect(byType.group_sum[0].params.k).toBe(10);
    // entry=patch_embed(lut), exit=group_sum
    const entry = tpl.nodes.find((n) => n.id === tpl.entryId)!;
    const exit = tpl.nodes.find((n) => n.id === tpl.exitId)!;
    expect(entry.type).toBe('lut_layer');
    expect(entry.train_info.kind).toBe('patch_embed');
    expect(exit.type).toBe('group_sum');
    // I/O幅
    expect(blockIoWidths(spec)).toEqual({ in: 4 * 12, out: 10 });
  });

  it('llvit複合は接続後に shapeInference エラー診断ゼロ(blackbox幅整合)', () => {
    const spec: BlockParams = {
      kind: 'llvit',
      params: {
        seqLen: 4,
        dEmbed: 6,
        posBits: 2,
        depth: 2,
        nHeads: 2,
        patchBits: 12,
        numClasses: 10,
      },
    };
    const tpl = generateBlock(spec, makeAlloc());
    const io = blockIoWidths(spec);
    const ir = templateToGraph('llvit', tpl, io.in, io.out);
    expect(diagnosticsOf(ir).filter((d) => d.level === 'error')).toEqual([]);
  });

  it('upsample2dは n_inputs=2 の恒等配線 lut_layer(build可能)', () => {
    const spec: BlockParams = {
      kind: 'upsample2d',
      params: { inC: 1, inH: 2, inW: 2, scale: 2 },
    };
    const tpl = generateBlock(spec, makeAlloc());
    const n = tpl.nodes[0];
    expect(n.type).toBe('lut_layer');
    expect(n.params.n_inputs).toBe(2); // LutLayer制約(2,4,6)に載る
    expect(n.params.in_width).toBe(4);
    expect(n.params.out_width).toBe(16);
    // nearest複製: out画素(0,0),(0,1),(1,0),(1,1)は全て入力(0,0)=index0
    const row = (n.params.indices as number[][])[0];
    expect(row[0]).toBe(0);
    expect(row[1]).toBe(0);
    expect(row[4]).toBe(0); // (y=1,x=0) -> in(0,0)
    expect(n.train_info.kind).toBe('binarybrain');
    expect(blockIoWidths(spec)).toEqual({ in: 4, out: 16 });
  });
});
