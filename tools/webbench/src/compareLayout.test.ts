// W6: TF.js(NHWC)⇔ DLGN/ONNX(NCHW)橋渡し純関数のユニットテスト。
import { describe, expect, it } from 'vitest';

import { argmaxSpatial, nchwToNhwc, nhwcToNchwPlane } from './compareLayout';

describe('argmaxSpatial', () => {
  it('NHWC[1,2,2,3] を per-pixel argmax できる', () => {
    // 4 画素 × 3 クラス、HWC 並び。各画素の最大クラス = [2,0,1,2]
    const data = [
      0.1, 0.2, 0.9, // px0 -> 2
      0.8, 0.1, 0.0, // px1 -> 0
      0.0, 0.7, 0.3, // px2 -> 1
      0.2, 0.2, 0.5, // px3 -> 2
    ];
    expect([...argmaxSpatial(data, [1, 2, 2, 3])]).toEqual([2, 0, 1, 2]);
  });

  it('NCHW[1,3,2,2] は同じ論理画素で同じ結果', () => {
    // 上と同じ画素を CHW 並びに置換(c*hw + p)
    const data = [
      0.1, 0.8, 0.0, 0.2, // class0 の 4 画素
      0.2, 0.1, 0.7, 0.2, // class1
      0.9, 0.0, 0.3, 0.5, // class2
    ];
    expect([...argmaxSpatial(data, [1, 3, 2, 2])]).toEqual([2, 0, 1, 2]);
  });
});

describe('NHWC <-> NCHW plane', () => {
  it('nhwcToNchwPlane は c*hw+p 並びへ移す', () => {
    // 2 画素 × 2 ch, HWC: [p0c0,p0c1, p1c0,p1c1]
    const nhwc = [10, 11, 20, 21];
    const plane = nhwcToNchwPlane(nhwc, 1, 2, 2);
    // NCHW: [c0p0,c0p1, c1p0,c1p1] = [10,20, 11,21]
    expect([...plane]).toEqual([10, 20, 11, 21]);
  });

  it('nchwToNhwc は nhwcToNchwPlane の逆写像', () => {
    const nchw = [10, 20, 11, 21]; // c*hw+p, 2ch 2px
    const nhwc = nchwToNhwc(nchw, 1, 2, 2);
    expect([...nhwc]).toEqual([10, 11, 20, 21]);
    // 往復で戻る
    const back = nhwcToNchwPlane(nhwc, 1, 2, 2);
    expect([...back]).toEqual(nchw);
  });
});
