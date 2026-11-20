// エッジ基底スタイル(§5)の純関数テスト: ビット幅→太さ・信号種判定・マップ構築。

import { describe, expect, it } from 'vitest';
import type { IRGraphJson, IRNodeJson } from '../ir/schema';
import {
  MAX_EDGE_WIDTH,
  MIN_EDGE_WIDTH,
  bitWidthToStrokeWidth,
  buildEdgeBaseStyles,
  edgeBaseStyle,
  edgeSignalKind,
  nodeInputKind,
  nodeOutputKind,
} from './edgeStyle';

describe('bitWidthToStrokeWidth', () => {
  it('1bit以下・不正値は下限', () => {
    expect(bitWidthToStrokeWidth(1)).toBe(MIN_EDGE_WIDTH);
    expect(bitWidthToStrokeWidth(0)).toBe(MIN_EDGE_WIDTH);
    expect(bitWidthToStrokeWidth(null)).toBe(MIN_EDGE_WIDTH);
    expect(bitWidthToStrokeWidth(undefined)).toBe(MIN_EDGE_WIDTH);
    expect(bitWidthToStrokeWidth(Infinity)).toBe(MIN_EDGE_WIDTH);
  });

  it('2倍ごとに+1px(log2スケール)', () => {
    expect(bitWidthToStrokeWidth(2)).toBeCloseTo(2);
    expect(bitWidthToStrokeWidth(4)).toBeCloseTo(3);
    expect(bitWidthToStrokeWidth(8)).toBeCloseTo(4);
  });

  it('上限MAX_EDGE_WIDTHで飽和', () => {
    expect(bitWidthToStrokeWidth(16)).toBe(MAX_EDGE_WIDTH);
    expect(bitWidthToStrokeWidth(1024)).toBe(MAX_EDGE_WIDTH);
  });
});

describe('nodeOutputKind / nodeInputKind', () => {
  it('blackboxは入出力とも実数域', () => {
    expect(nodeOutputKind('blackbox')).toBe('real');
    expect(nodeInputKind('blackbox')).toBe('real');
  });

  it('binarizeは入力=実数域・出力=論理(ToLogic境界)', () => {
    expect(nodeInputKind('binarize')).toBe('real');
    expect(nodeOutputKind('binarize')).toBe('logic');
  });

  it('add(sum)の出力は実数域、or/xorは論理', () => {
    expect(nodeOutputKind('add', 'sum')).toBe('real');
    expect(nodeOutputKind('add', 'or')).toBe('logic');
    expect(nodeOutputKind('add', 'xor')).toBe('logic');
  });

  it('lut_layer等の論理ノードは論理', () => {
    expect(nodeOutputKind('lut_layer')).toBe('logic');
    expect(nodeInputKind('lut_layer')).toBe('logic');
    expect(nodeInputKind(undefined)).toBe('logic');
  });
});

describe('edgeSignalKind', () => {
  it('ソースがblackboxなら実数域', () => {
    expect(edgeSignalKind('blackbox', 'lut_layer')).toBe('real');
  });

  it('binarizeへ入るエッジは実数域(ToLogic境界の実数側)', () => {
    expect(edgeSignalKind('input', 'binarize')).toBe('real');
  });

  it('論理→論理は論理', () => {
    expect(edgeSignalKind('lut_layer', 'lut_layer')).toBe('logic');
    expect(edgeSignalKind('input', 'lut_layer')).toBe('logic');
  });
});

describe('edgeBaseStyle', () => {
  it('信号種で色、ビット幅で太さを与える', () => {
    expect(edgeBaseStyle(1, 'logic')).toEqual({
      stroke: 'var(--edge-default)',
      strokeWidth: MIN_EDGE_WIDTH,
    });
    expect(edgeBaseStyle(8, 'real')).toEqual({
      stroke: 'var(--edge-real)',
      strokeWidth: 4,
    });
  });
});

function lut(id: string, inW: number, outW: number): IRNodeJson {
  return {
    id,
    type: 'lut_layer',
    params: { in_width: inW, out_width: outW, n_inputs: 2 },
    train_info: {},
    meta: {},
  };
}

describe('buildEdgeBaseStyles', () => {
  it('nullは空マップ', () => {
    expect(buildEdgeBaseStyles(null).size).toBe(0);
  });

  it('ソース出力幅→太さ・信号種→色をedgeId別に返す', () => {
    const ir: IRGraphJson = {
      format: 'magicdelta-ir',
      version: '0.2',
      metadata: {},
      inputs: [{ id: 'x', width: 4 }],
      outputs: [{ id: 'y', width: 2 }],
      nodes: [lut('h', 4, 8), lut('o', 8, 2)],
      edges: [
        { src: 'x', dst: 'h' },
        { src: 'h', dst: 'o' },
        { src: 'o', dst: 'y' },
      ],
    };
    const map = buildEdgeBaseStyles(ir);
    // x(width=4)→h: 太さ=3, 論理
    expect(map.get('x->h')).toEqual({
      stroke: 'var(--edge-default)',
      strokeWidth: 3,
    });
    // h(out=8)→o: 太さ=4
    expect(map.get('h->o')?.strokeWidth).toBe(4);
  });

  it('blackboxソースは実数域色', () => {
    const ir: IRGraphJson = {
      format: 'magicdelta-ir',
      version: '0.2',
      metadata: {},
      inputs: [],
      outputs: [],
      nodes: [
        { id: 'bb', type: 'blackbox', params: { in_width: 2, out_width: 2 }, train_info: {}, meta: {} },
        lut('l', 2, 2),
      ],
      edges: [{ src: 'bb', dst: 'l' }],
    };
    expect(buildEdgeBaseStyles(ir).get('bb->l')?.stroke).toBe('var(--edge-real)');
  });
});
