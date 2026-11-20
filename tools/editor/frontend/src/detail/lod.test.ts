// LOD判定純関数の単体テスト(§3 セマンティックズーム)。
// ズーム×ゲート実寸→段階、可視領域×折りたたみ→自動展開対象を固定する。

import { describe, expect, it } from 'vitest';
import { GATE_W } from './detailConfig';
import {
  GATE_VISIBLE_PX,
  TILE_VISIBLE_PX,
  apparentGatePx,
  layerIntersectsView,
  lodStageForZoom,
  sameIdSet,
  selectAutoExpandLayers,
  viewportToFlowRect,
  type LodLayerInfo,
} from './lod';

// ちょうど閾値のズーム(見かけサイズ = しきい値pxになる点)
const ZOOM_GATE = GATE_VISIBLE_PX / GATE_W;
const ZOOM_TILE = TILE_VISIBLE_PX / GATE_W;

describe('apparentGatePx / lodStageForZoom', () => {
  it('見かけサイズ = ズーム × ゲート実寸', () => {
    expect(apparentGatePx(1, GATE_W)).toBe(GATE_W);
    expect(apparentGatePx(0.5, 100)).toBe(50);
  });

  it('しきい値以上でゲート、その手前でタイル、さらに手前で俯瞰', () => {
    expect(lodStageForZoom(ZOOM_GATE)).toBe('gate');
    expect(lodStageForZoom(ZOOM_GATE - 1e-6)).toBe('tile');
    expect(lodStageForZoom(ZOOM_TILE)).toBe('tile');
    expect(lodStageForZoom(ZOOM_TILE - 1e-6)).toBe('overview');
  });

  it('段階はズーム単調(俯瞰→タイル→ゲート)', () => {
    expect(lodStageForZoom(0.05)).toBe('overview');
    expect(lodStageForZoom(0.25)).toBe('tile');
    expect(lodStageForZoom(1)).toBe('gate');
  });
});

describe('viewportToFlowRect', () => {
  it('原点・等倍では画面寸法がそのままflow矩形になる', () => {
    expect(viewportToFlowRect({ x: 0, y: 0, zoom: 1 }, 800, 600)).toEqual({
      minX: 0,
      minY: 0,
      maxX: 800,
      maxY: 600,
    });
  });

  it('パン(offset)とズームを逆変換する', () => {
    // zoom=2, offset=(100,50): 画面(0,0)→flow(-50,-25), 画面(800,600)→flow(350,275)
    expect(viewportToFlowRect({ x: 100, y: 50, zoom: 2 }, 800, 600)).toEqual({
      minX: -50,
      minY: -25,
      maxX: 350,
      maxY: 275,
    });
  });
});

describe('layerIntersectsView', () => {
  const view = { minX: 0, minY: 0, maxX: 100, maxY: 100 };
  const mk = (x: number, y: number): LodLayerInfo => ({
    id: 'l',
    x,
    y,
    w: 20,
    h: 20,
    expandable: true,
    collapsed: false,
  });

  it('可視領域内は交差、外は非交差', () => {
    expect(layerIntersectsView(mk(10, 10), view)).toBe(true);
    expect(layerIntersectsView(mk(200, 10), view)).toBe(false);
    expect(layerIntersectsView(mk(-30, 10), view)).toBe(false);
  });
});

describe('selectAutoExpandLayers', () => {
  const view = { minX: 0, minY: 0, maxX: 1000, maxY: 1000 };
  const layers: LodLayerInfo[] = [
    { id: 'a', x: 0, y: 0, w: 50, h: 50, expandable: true, collapsed: false },
    { id: 'b', x: 100, y: 0, w: 50, h: 50, expandable: false, collapsed: false }, // 大きすぎ
    { id: 'c', x: 200, y: 0, w: 50, h: 50, expandable: true, collapsed: true }, // 折りたたみ内
    { id: 'd', x: 5000, y: 0, w: 50, h: 50, expandable: true, collapsed: false }, // 画面外
  ];

  it('ゲート段階では 展開可能かつ非折りたたみかつ可視 の層のみ返す', () => {
    const ids = selectAutoExpandLayers(layers, 1, view);
    expect([...ids]).toEqual(['a']);
  });

  it('タイル/俯瞰段階では空集合(ゲートを展開しない)', () => {
    expect(selectAutoExpandLayers(layers, ZOOM_TILE, view).size).toBe(0);
    expect(selectAutoExpandLayers(layers, 0.01, view).size).toBe(0);
  });

  it('パンで可視領域が移れば展開対象が入れ替わる', () => {
    const shifted = { minX: 4900, minY: 0, maxX: 5100, maxY: 1000 };
    const ids = selectAutoExpandLayers(layers, 1, shifted);
    expect([...ids]).toEqual(['d']);
  });
});

describe('sameIdSet', () => {
  it('内容一致は順不同でtrue、差異はfalse', () => {
    expect(sameIdSet(new Set(['a', 'b']), new Set(['b', 'a']))).toBe(true);
    expect(sameIdSet(new Set(['a']), new Set(['a', 'b']))).toBe(false);
    expect(sameIdSet(new Set(['a']), new Set(['b']))).toBe(false);
  });
});
