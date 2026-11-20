// playgroundStoreの単体テスト。
// コンポーネント不要: getState()でアクションを直接呼び、graphStore/trainingStoreは
// setState/actionで状態を注入する。

import { beforeEach, describe, expect, it } from 'vitest';
import type { HistoryStepView, U8Envelope } from '../api/rest';
import type { IRGraphJson } from '../ir/schema';
import { SAMPLE_PARITY3, SAMPLE_XOR } from '../samples/sampleIr';
import { useGraphStore } from '../store/graphStore';
import {
  useTrainingStore,
  type GateDistributionFrame,
} from '../store/trainingStore';
import { usePlaygroundStore } from './playgroundStore';

/** [0,1]値配列をu8量子化エンベロープへ(decodeU8の逆変換) */
function u8Env(values: number[], shape: number[]): U8Envelope {
  const bytes = values.map((v) => Math.round(v * 255));
  return {
    encoding: 'u8',
    shape,
    scale: 1,
    data: btoa(String.fromCharCode(...bytes)),
  };
}

/** OR(g=7)one-hotの16ゲート分布(1ニューロン分) */
const OR16: number[] = Object.assign(new Array<number>(16).fill(0), { 7: 1 });

function orDistEnv(neurons = 1): U8Envelope {
  const probs: number[] = [];
  for (let i = 0; i < neurons; i++) probs.push(...OR16);
  return u8Env(probs, [neurons, 16]);
}

function setIr(ir: IRGraphJson | null) {
  useGraphStore.setState({ ir });
}

function pg() {
  return usePlaygroundStore.getState();
}

beforeEach(() => {
  usePlaygroundStore.setState({
    enabled: false,
    mode: 'hard',
    inputBits: {},
    values: null,
    error: null,
    tableRows: null,
    expectedTable: null,
  });
  setIr(null);
  useTrainingStore.getState().resetAll();
});

describe('入力トグルと評価', () => {
  it('setEnabled(true)で評価され、toggleBitで出力が変わる(XOR)', () => {
    setIr(SAMPLE_XOR);
    pg().setEnabled(true);
    expect(pg().inputBits).toEqual({ x: [0, 0] });
    expect(Array.from(pg().values!.y)).toEqual([0]);

    pg().toggleBit('x', 0);
    expect(Array.from(pg().values!.y)).toEqual([1]);

    pg().toggleBit('x', 1);
    expect(Array.from(pg().values!.y)).toEqual([0]);
  });

  it('無効時は評価しない(values=null)', () => {
    setIr(SAMPLE_XOR);
    pg().setEnabled(true);
    pg().setEnabled(false);
    expect(pg().values).toBeNull();
    expect(pg().tableRows).toBeNull();
  });

  it('IR差し替えに追随して入力幅を正規化し再評価する', () => {
    setIr(SAMPLE_XOR);
    pg().setEnabled(true);
    pg().toggleBit('x', 1);
    setIr(SAMPLE_PARITY3); // graphStore購読で自動recompute
    expect(pg().inputBits.x).toEqual([0, 1, 0]); // 既存値は幅の範囲で保持
    expect(pg().values).not.toBeNull();
    expect(Array.from(pg().values!.y)).toEqual([1]); // parity(0,1,0)=1
  });

  it('pinPatternで入力パターンを一括設定する', () => {
    setIr(SAMPLE_XOR);
    pg().setEnabled(true);
    pg().pinPattern(2); // bit j = (k>>j)&1 → [0,1]
    expect(pg().inputBits).toEqual({ x: [0, 1] });
    expect(Array.from(pg().values!.y)).toEqual([1]);
  });
});

describe('検証テーブルと期待値', () => {
  it('tableRowsが全列挙され、expectedTableは現在出力のコピーで初期化される', () => {
    setIr(SAMPLE_XOR);
    pg().setEnabled(true);
    expect(pg().tableRows).toHaveLength(4);
    expect(pg().expectedTable).toEqual({ 0: [0], 1: [1], 2: [1], 3: [0] });
  });

  it('期待値の編集はrecomputeを跨いで保持され、resetExpectedで戻る', () => {
    setIr(SAMPLE_XOR);
    pg().setEnabled(true);
    pg().toggleExpectedBit(0, 0);
    expect(pg().expectedTable![0]).toEqual([1]);

    pg().toggleBit('x', 0); // recomputeが走っても再初期化されない
    expect(pg().expectedTable![0]).toEqual([1]);

    pg().resetExpected();
    expect(pg().expectedTable![0]).toEqual([0]);
  });
});

describe('学習/リプレイ同期', () => {
  const orFrame: GateDistributionFrame = {
    type: 'gate_distribution',
    layer: 'n1_lutlayer',
    step: 5,
    n_inputs: 2,
    parameterization: 'softmax',
    mode: 'dense',
    probs: orDistEnv(),
    entropy_mean: null,
  };

  it('liveのdense分布が届くとargmax真理値表で評価される(XOR層→OR動作)', () => {
    setIr(SAMPLE_XOR);
    pg().setEnabled(true);
    useTrainingStore.getState().applyDetailFrame(orFrame);
    pg().recompute();
    const outputs = pg().tableRows!.map((r) => r.outputBits[0]);
    expect(outputs).toEqual([0, 1, 1, 1]); // OR真理値(IRのXORを上書き)
  });

  it('soft時はゲート分布の期待値で評価される', () => {
    setIr(SAMPLE_XOR);
    pg().setEnabled(true);
    pg().setMode('soft');
    useTrainingStore.getState().applyDetailFrame(orFrame);
    pg().pinPattern(3); // (1,1): OR→1(XORなら0)
    expect(pg().values!.y[0]).toBeCloseTo(1, 6);
  });

  it('replay時はreplayFrameのgate_distributionを使う', () => {
    setIr(SAMPLE_XOR);
    pg().setEnabled(true);
    const view: HistoryStepView = {
      resolved_step: 3,
      method: 'exact',
      metrics: { step: 3, epoch: 0, loss: 0.5, acc: 0.5 },
      layers: { n1_lutlayer: { gate_distribution: orDistEnv(), n_inputs: 2 } },
    };
    useTrainingStore.getState().enterReplay(3);
    useTrainingStore.getState().applyReplayFrame(view);
    pg().recompute();
    const outputs = pg().tableRows!.map((r) => r.outputBits[0]);
    expect(outputs).toEqual([0, 1, 1, 1]);

    // replay終了でIRのtruth_tables(XOR)へ戻る
    useTrainingStore.getState().exitReplay();
    pg().recompute();
    expect(pg().tableRows!.map((r) => r.outputBits[0])).toEqual([0, 1, 1, 0]);
  });

  it('確率が届いていない層はIRのtruth_tablesにフォールバックする', () => {
    setIr(SAMPLE_PARITY3);
    pg().setEnabled(true);
    // n1(2ニューロン)にのみOR分布を注入、n2は確率なし(IRのXORのまま)
    useTrainingStore.getState().applyDetailFrame({ ...orFrame, probs: orDistEnv(2) });
    pg().pinPattern(3); // (1,1,0): n1がORなら XOR(1|1, 0)=1、元のXOR層なら XOR(1^1, 0)=0
    expect(pg().values!.y[0]).toBe(1);
  });
});
