// ページ設定の不変条件。
import { describe, expect, it } from 'vitest';

import type { ControlSpec } from './controls';
import { NAV } from './nav';
import { DEFAULT_DLGN_BACKEND, PAGES } from './pages';

const DET_PAGES = ['det', 'detmono', 'detface', 'detreal'];

/** group にネストしたコントロールも含めてIDで探す(webcamの計算方法セレクタは
    手法セレクタの値に応じて出し分ける group の子になっている)。*/
function findControl(controls: readonly ControlSpec[], id: string): ControlSpec | undefined {
  for (const c of controls) {
    if (c.id === id) return c;
    if (c.kind === 'group') {
      const found = findControl(c.children, id);
      if (found) return found;
    }
  }
  return undefined;
}

describe('ページ設定', () => {
  it('全ページがサイドバーの一覧に載っている', () => {
    const navIds = new Set(NAV.map((n) => n.id));
    for (const [key, cfg] of Object.entries(PAGES)) {
      expect(navIds.has(cfg.id), key).toBe(true);
      expect(cfg.id, `${key}: キーと id を揃える`).toBe(key);
    }
  });

  it('検出画面のしきい値は0〜100の整数(表示用フィルタの実測較正値)', () => {
    for (const key of DET_PAGES) {
      const thr = PAGES[key].controls.find((c) => c.id === 'thr');
      expect(thr, key).toBeDefined();
      if (thr?.kind !== 'range') throw new Error(`${key}: thr は range`);
      expect(Number.isInteger(thr.value), key).toBe(true);
      expect(thr.value).toBeGreaterThanOrEqual(thr.min);
      expect(thr.value).toBeLessThanOrEqual(thr.max);
    }
  });

  it('行動認識は再生ボタンと窓スライダーを持つ', () => {
    const ids = PAGES.har.controls.map((c) => c.id);
    expect(ids).toContain('play');
    expect(ids).toContain('window');
  });

  it('計算方法セレクタの値は実装が期待する文字列のまま、既定はDEFAULT_DLGN_BACKEND', () => {
    for (const [key, cfg] of Object.entries(PAGES)) {
      const dlgn = findControl(cfg.controls, key === 'webcam' ? 'backend' : 'dlgnBackend');
      if (!dlgn) continue;
      if (dlgn.kind !== 'select') throw new Error(`${key}: DLGN計算方法は select`);
      expect(dlgn.options.map((o) => o.value)).toEqual(['js', 'wasm', 'webgpu']);
      const selected = dlgn.options.filter((o) => o.selected);
      expect(selected, key).toHaveLength(1);
      expect(selected[0].value, key).toBe(DEFAULT_DLGN_BACKEND);
    }
  });

  it('webcamは「手法」を1つだけ選ぶセレクタ(比較列のチェックボックスではない)', () => {
    // 以前は比較手法をチェックボックスで並べる列だったが、動かす手法
    // そのものを1つだけ選ぶセレクタ方式に改めた。DLGN/ONNX Runtime Web/
    // TensorFlow.js/MediaPipeの4択で、既定はDLGN。
    const method = findControl(PAGES.webcam.controls, 'method');
    if (method?.kind !== 'select') throw new Error('method は select');
    expect(method.options.map((o) => o.value)).toEqual(['dlgn', 'onnx', 'tfjs', 'mediapipe']);
    const selected = method.options.filter((o) => o.selected);
    expect(selected).toHaveLength(1);
    expect(selected[0].value).toBe('dlgn');

    // チェックボックスの比較列(旧UI)は残っていないこと。
    const ids = PAGES.webcam.controls.map((c) => c.id);
    expect(ids).not.toContain('cmpMediapipe');
    expect(ids).not.toContain('cmpOnnx');
    expect(ids).not.toContain('cmpTfjs');
  });

  it('webcamのDLGN/ONNX計算方法セレクタは手法セレクタの子ではなく独立したgroup', () => {
    // 「手法→(必要なら)計算方法」の最大2段構成であることを、
    // dlgnBackendWrap/onnxBackendWrapという別groupの存在で確認する。
    const dlgnWrap = PAGES.webcam.controls.find((c) => c.id === 'dlgnBackendWrap');
    const onnxWrap = PAGES.webcam.controls.find((c) => c.id === 'onnxBackendWrap');
    expect(dlgnWrap?.kind).toBe('group');
    expect(onnxWrap?.kind).toBe('group');
  });

  it('カメラのライブ実行は顔検出とセグメンテーションの2タスク', () => {
    const t = PAGES.webcam.controls.find((c) => c.id === 'task');
    if (t?.kind !== 'select') throw new Error('task は select');
    expect(t.options.map((o) => o.value).sort()).toEqual(['face', 'seg']);
  });
});
