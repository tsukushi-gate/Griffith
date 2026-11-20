// `?model=` によるDLGNモデル差し替えの解釈契約。
//
// なぜテストするか: この解釈は「配信ルート配下の models/ 以外を読ませない」
// という安全条件を兼ねている(クエリ由来の文字列をそのまま fetch する経路)。
// 5つのデモ main が同じ関数を使うので、ここが壊れると計測が黙って別モデルを
// 読む(= 速度比較が無意味になる)。href を明示に渡してブラウザ外でも縛る。
import { describe, expect, it } from 'vitest';

import { modelAsset, modelVariantFromUrl } from './basePath';

const DEF = '/models/det_multimnist_lut_k256.json';
const url = (q: string): string => `http://localhost:8787/det.html?${q}`;

describe('modelVariantFromUrl', () => {
  it('指定なしは null(既定モデル)', () => {
    expect(modelVariantFromUrl(url('autobench=1&backend=wasm'))).toBeNull();
  });
  it('変種サフィックスを受け付ける', () => {
    expect(modelVariantFromUrl(url('model=_ll'))).toBe('_ll');
    expect(modelVariantFromUrl(url('model=_dc'))).toBe('_dc');
  });
  it('完全なファイル名を受け付ける', () => {
    expect(modelVariantFromUrl(url('model=seg_convdlgn_k128_ll.json')))
      .toBe('seg_convdlgn_k128_ll.json');
  });
  it('パス区切り・親参照・拡張子違いは拒否して既定へ落とす', () => {
    for (const bad of [
      'model=..%2F..%2Fetc%2Fpasswd',
      'model=%2Fetc%2Fpasswd',
      'model=sub%2Fdir%2Fa.json',
      'model=a.onnx',
      'model=.hidden.json',
      'model=',
    ]) {
      expect(modelVariantFromUrl(url(bad))).toBeNull();
    }
  });
});

describe('modelAsset', () => {
  it('指定なしは既定パスのまま', () => {
    expect(modelAsset(DEF, url('autobench=1'))).toBe(DEF);
  });
  it('サフィックスは .json の直前へ挿入される(タスク非依存)', () => {
    expect(modelAsset(DEF, url('model=_ll')))
      .toBe('/models/det_multimnist_lut_k256_ll.json');
    expect(modelAsset('/models/seg_convdlgn_k128.json', url('model=_dc')))
      .toBe('/models/seg_convdlgn_k128_dc.json');
  });
  it('完全なファイル名は models/ 直下を指す', () => {
    expect(modelAsset(DEF, url('model=det_face_lut_k256_dc.json')))
      .toBe('/models/det_face_lut_k256_dc.json');
  });
  it('不正値は既定へフォールバックする', () => {
    expect(modelAsset(DEF, url('model=..%2Fsecret.json'))).toBe(DEF);
  });
});
