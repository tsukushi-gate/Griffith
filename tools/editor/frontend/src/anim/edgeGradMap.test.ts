// edgeGradMapの単体テスト。
// 既知のu8エンベロープからのMap構築・層内max正規化・エッジ/ゲートid規約を固定する。

import { describe, expect, it } from 'vitest';
import type { U8Envelope } from '../api/rest';
import type { GradFlowFrame, GradFlowLayerTrace } from '../store/trainingStore';
import {
  buildBpIntensityMaps,
  detailEdgeId,
  gateNodeId,
  layerEdgeGradIntensity,
} from './edgeGradMap';

/** バイト列から u8エンベロープを組み立てる(monitor.py encode_u8 と同形式)。 */
function env(bytes: number[], scale: number, shape: number[]): U8Envelope {
  return {
    encoding: 'u8',
    shape,
    scale,
    data: btoa(String.fromCharCode(...bytes)),
  };
}

function frame(layers: GradFlowLayerTrace[]): GradFlowFrame {
  return { type: 'grad_flow', step: 10, loss: 0.5, layers, layer_grad_norm: {} };
}

describe('detailEdgeId / gateNodeId', () => {
  it('詳細エッジid規約 d:${layerId}:${neuron}:${slot} に一致する', () => {
    // irToDetailGraph.ts の `d:${fn.id}:${o}:${j}` と同一規約
    expect(detailEdgeId('layerA', 3, 1)).toBe('d:layerA:3:1');
  });

  it('ゲートid規約 ${layerId}::g${neuron} に一致する', () => {
    // irToDetailGraph.ts の `${fn.id}::g${o}` と同一規約
    expect(gateNodeId('layerA', 7)).toBe('layerA::g7');
  });
});

describe('layerEdgeGradIntensity', () => {
  it('edge_gradを層内maxで[0,1]へ正規化した平坦配列(o×n_inputs+j)を返す', () => {
    // (2,3): [[0,51,102],[153,204,255]] scale=2 → 値[0,0.4,...,2] → 正規化[0,0.2,...,1]
    const entry: GradFlowLayerTrace = {
      layer: 'l1',
      order: 0,
      edge_grad: env([0, 51, 102, 153, 204, 255], 2, [2, 3]),
    };
    const out = layerEdgeGradIntensity(entry)!;
    expect(out).toHaveLength(6);
    for (const [i, want] of [0, 0.2, 0.4, 0.6, 0.8, 1].entries()) {
      expect(out[i]).toBeCloseTo(want, 5);
    }
    // 平坦index規約: neuron=1, slot=2 → 1×3+2=5
    expect(out[1 * 3 + 2]).toBeCloseTo(1, 5);
  });

  it('edge_grad欠落・非2次元shapeはnullを返す', () => {
    expect(layerEdgeGradIntensity({ layer: 'l1', order: 0 })).toBeNull();
    expect(
      layerEdgeGradIntensity({
        layer: 'l1',
        order: 0,
        edge_grad: env([1, 2], 1, [2]),
      }),
    ).toBeNull();
  });

  it('全ゼロ勾配は全て0のまま(NaNを出さない)', () => {
    const out = layerEdgeGradIntensity({
      layer: 'l1',
      order: 0,
      edge_grad: env([0, 0, 0, 0], 1, [2, 2]),
    })!;
    expect(Array.from(out)).toEqual([0, 0, 0, 0]);
  });
});

describe('buildBpIntensityMaps', () => {
  it('edge_gradから詳細エッジid→強度Mapを構築する(キー規約・件数)', () => {
    const { edgeIntensity } = buildBpIntensityMaps(
      frame([
        { layer: 'l1', order: 0, edge_grad: env([0, 51, 102, 153, 204, 255], 2, [2, 3]) },
      ]),
    );
    expect(edgeIntensity.size).toBe(6);
    expect(edgeIntensity.get('d:l1:0:0')).toBeCloseTo(0, 5);
    expect(edgeIntensity.get('d:l1:0:1')).toBeCloseTo(0.2, 5);
    expect(edgeIntensity.get('d:l1:1:2')).toBeCloseTo(1, 5);
  });

  it('grad_wからゲートid→強度Mapを構築する(層内max正規化)', () => {
    const { gateIntensity } = buildBpIntensityMaps(
      frame([{ layer: 'l1', order: 0, grad_w: env([255, 51], 4, [2]) }]),
    );
    expect(gateIntensity.size).toBe(2);
    expect(gateIntensity.get('l1::g0')).toBeCloseTo(1, 5);
    expect(gateIntensity.get('l1::g1')).toBeCloseTo(0.2, 5);
  });

  it('複数層はそれぞれ層内maxで独立に正規化される', () => {
    const { edgeIntensity } = buildBpIntensityMaps(
      frame([
        // l1のmaxは255(scale=1)、l2のmaxは51(scale=1)だが両層とも最大は1.0になる
        { layer: 'l1', order: 0, edge_grad: env([255, 51], 1, [1, 2]) },
        { layer: 'l2', order: 1, edge_grad: env([51, 17], 1, [1, 2]) },
      ]),
    );
    expect(edgeIntensity.get('d:l1:0:0')).toBeCloseTo(1, 5);
    expect(edgeIntensity.get('d:l1:0:1')).toBeCloseTo(51 / 255, 5);
    expect(edgeIntensity.get('d:l2:0:0')).toBeCloseTo(1, 5);
    expect(edgeIntensity.get('d:l2:0:1')).toBeCloseTo(17 / 51, 5);
  });

  it('edge_grad欠落層はエッジMapに寄与しない(grad_wのみ反映)', () => {
    const { edgeIntensity, gateIntensity } = buildBpIntensityMaps(
      frame([{ layer: 'l1', order: 0, grad_w: env([255], 1, [1]) }]),
    );
    expect(edgeIntensity.size).toBe(0);
    expect(gateIntensity.size).toBe(1);
  });

  it('edge_gradとgrad_wの両方があれば両Mapが同時に構築される', () => {
    const { edgeIntensity, gateIntensity } = buildBpIntensityMaps(
      frame([
        {
          layer: 'l1',
          order: 0,
          edge_grad: env([255, 0, 0, 255], 1, [2, 2]),
          grad_w: env([255, 0], 1, [2]),
        },
      ]),
    );
    expect(edgeIntensity.size).toBe(4);
    expect(gateIntensity.size).toBe(2);
    expect(edgeIntensity.get('d:l1:0:0')).toBeCloseTo(1, 5);
    expect(edgeIntensity.get('d:l1:1:0')).toBeCloseTo(0, 5);
    expect(gateIntensity.get('l1::g1')).toBeCloseTo(0, 5);
  });

  it('空フレームは空Mapを返す', () => {
    const { edgeIntensity, gateIntensity } = buildBpIntensityMaps(frame([]));
    expect(edgeIntensity.size).toBe(0);
    expect(gateIntensity.size).toBe(0);
  });

  it('強度は全て[0,1]の範囲に収まる', () => {
    const bytes = Array.from({ length: 12 }, (_, i) => (i * 37) % 256);
    const { edgeIntensity } = buildBpIntensityMaps(
      frame([{ layer: 'l1', order: 0, edge_grad: env(bytes, 3.5, [4, 3]) }]),
    );
    for (const v of edgeIntensity.values()) {
      expect(v).toBeGreaterThanOrEqual(0);
      expect(v).toBeLessThanOrEqual(1);
    }
  });
});
