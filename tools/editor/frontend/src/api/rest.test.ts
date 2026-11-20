// データセットカタログのfilterロジック(I/O幅マッチ)と
// loss/methodセレクタの選択肢ロジックの単体テスト。

import { describe, expect, it } from 'vitest';
import {
  ApiError,
  datasetMismatchReason,
  lossOptionsForKind,
  methodDisabledReason,
  type DatasetInfo,
} from './rest';

const mnist: DatasetInfo = {
  name: 'mnist',
  label: 'MNIST',
  in_width: 784,
  out_width: 10,
  kind: 'classification',
};

const xor: DatasetInfo = {
  name: 'logic:xor',
  label: 'XOR',
  in_width: 2,
  out_width: 1,
  kind: 'binary_vector',
};

describe('datasetMismatchReason', () => {
  it('I/O幅が一致すればnull(選択可能)', () => {
    expect(datasetMismatchReason(mnist, 784, 10)).toBeNull();
    expect(datasetMismatchReason(xor, 2, 1)).toBeNull();
  });

  it('入力幅不一致は理由に両方の幅を含む', () => {
    const reason = datasetMismatchReason(xor, 784, 1);
    expect(reason).toContain('入力幅2');
    expect(reason).toContain('784');
  });

  it('出力幅不一致は理由に両方の幅を含む', () => {
    const reason = datasetMismatchReason(mnist, 784, 1);
    expect(reason).toContain('出力幅10');
    expect(reason).toContain('現在1');
  });

  it('入出力とも不一致なら理由を併記する', () => {
    const reason = datasetMismatchReason(mnist, 2, 1);
    expect(reason).toContain('入力幅784');
    expect(reason).toContain('出力幅10');
  });

  it('モデル未読込(幅不明)は選択不可', () => {
    expect(datasetMismatchReason(xor, undefined, undefined)).toContain('モデル');
  });
});

describe('lossOptionsForKind', () => {
  it('classificationはceのみ', () => {
    expect(lossOptionsForKind('classification')).toEqual(['ce']);
  });

  it('binary_vectorはbce/mse', () => {
    expect(lossOptionsForKind('binary_vector')).toEqual(['bce', 'mse']);
  });
});

describe('methodDisabledReason', () => {
  it('apはbinary_vectorで選択不可', () => {
    expect(methodDisabledReason('ap', 'binary_vector')).toContain('AP法');
  });

  it('apはclassificationで選択可', () => {
    expect(methodDisabledReason('ap', 'classification')).toBeNull();
  });

  it('bp/bbはタスク種別によらず選択可', () => {
    expect(methodDisabledReason('bp', 'binary_vector')).toBeNull();
    expect(methodDisabledReason('bb', 'binary_vector')).toBeNull();
    expect(methodDisabledReason('bp', 'classification')).toBeNull();
  });
});

describe('ApiError', () => {
  it('HTTPステータスを保持する(404でセッション消失を判定できる)', () => {
    const e = new ApiError('セッションが存在しない: abc', 404);
    expect(e).toBeInstanceOf(Error);
    expect(e.status).toBe(404);
    expect(e.message).toContain('セッション');
  });
});
