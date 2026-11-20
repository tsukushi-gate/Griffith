// 手元の画像アップロード共通処理の純粋部分(DOM非依存)。
import { describe, expect, it } from 'vitest';

import { coverCropRect, isImageFile, toGrayRgba } from './upload';

describe('coverCropRect', () => {
  it('横長の画像は左右を切り落として中央の正方形にする', () => {
    expect(coverCropRect(200, 100)).toEqual({ sx: 50, sy: 0, sSide: 100 });
  });

  it('縦長の画像は上下を切り落として中央の正方形にする', () => {
    expect(coverCropRect(100, 200)).toEqual({ sx: 0, sy: 50, sSide: 100 });
  });

  it('正方形の画像は切り出さない(恒等変換)', () => {
    expect(coverCropRect(64, 64)).toEqual({ sx: 0, sy: 0, sSide: 64 });
  });
});

describe('isImageFile', () => {
  it('MIMEタイプが image/ で始まれば true', () => {
    expect(isImageFile({ type: 'image/png', name: 'a.png' })).toBe(true);
    expect(isImageFile({ type: 'image/jpeg', name: 'a.unknown' })).toBe(true);
  });

  it('MIMEタイプが画像以外なら false(拡張子が画像でも type を優先)', () => {
    expect(isImageFile({ type: 'application/pdf', name: 'a.pdf' })).toBe(false);
    expect(isImageFile({ type: 'text/plain', name: 'photo.png' })).toBe(false);
  });

  it('type が空ならファイル名の拡張子で判定する(大文字小文字を無視)', () => {
    expect(isImageFile({ type: '', name: 'photo.JPG' })).toBe(true);
    expect(isImageFile({ type: '', name: 'scan.webp' })).toBe(true);
    expect(isImageFile({ type: '', name: 'notes.txt' })).toBe(false);
  });
});

describe('toGrayRgba', () => {
  it('Rec.601輝度でr=g=bに揃え、alphaは変えない', () => {
    // 200,150,50 → 0.299*200+0.587*150+0.114*50 = 153.55 → 154
    // 10,20,30   → 0.299*10 +0.587*20 +0.114*30  = 18.15  → 18
    const rgba = new Uint8ClampedArray([200, 150, 50, 255, 10, 20, 30, 128]);
    toGrayRgba(rgba);
    expect(Array.from(rgba)).toEqual([154, 154, 154, 255, 18, 18, 18, 128]);
  });

  it('既にグレー(r=g=b)の画素は値を変えない', () => {
    const rgba = new Uint8ClampedArray([100, 100, 100, 255]);
    toGrayRgba(rgba);
    expect(Array.from(rgba)).toEqual([100, 100, 100, 255]);
  });
});
