// 変換ウィザード状態機械の単体テスト。
// rest APIをモックし、start→statusポーリング→done(IRロード)/error/cancel/abort の
// 各分岐と、phaseラベル・終端判定を検証する(fetch実体・DOMには依存しない)。

import { beforeEach, describe, expect, it, vi } from 'vitest';
import type { IRGraphJson } from '../ir/schema';
import {
  isTerminalPhase,
  phaseLabel,
  requestCancel,
  runConversion,
} from './convertMachine';
import * as rest from '../api/rest';

vi.mock('../api/rest');

const mockedRest = vi.mocked(rest);

const FAKE_IR = { format: 'magicdelta-ir', nodes: [] } as unknown as IRGraphJson;

function statusFrame(overrides: Partial<rest.ConvertStatus>): rest.ConvertStatus {
  return {
    convert_id: 'cid-1',
    phase: 'preparing',
    progress: 0,
    metrics: {},
    error: null,
    has_ir: false,
    source: 'ckpt:foo',
    dataset: 'mnist',
    ...overrides,
  };
}

const noSleep = () => Promise.resolve();

beforeEach(() => {
  vi.clearAllMocks();
  mockedRest.startConversion.mockResolvedValue({ convert_id: 'cid-1' });
  mockedRest.getCurrentIR.mockResolvedValue({ ir: FAKE_IR, dirty: false });
});

describe('phaseLabel / isTerminalPhase', () => {
  it('日本語ラベルへ変換(未知はそのまま)', () => {
    expect(phaseLabel('pruning')).toBe('枝刈り+微調整');
    expect(phaseLabel('lowering')).toBe('回路化');
    expect(phaseLabel('verifying')).toBe('検証');
    expect(phaseLabel('done')).toBe('完了');
    expect(phaseLabel('unknown-phase')).toBe('unknown-phase');
  });

  it('done/error/cancelledが終端', () => {
    expect(isTerminalPhase('done')).toBe(true);
    expect(isTerminalPhase('error')).toBe(true);
    expect(isTerminalPhase('cancelled')).toBe(true);
    expect(isTerminalPhase('pruning')).toBe(false);
  });
});

describe('runConversion: doneフロー', () => {
  it('start→複数phaseポーリング→done時にgetCurrentIR+onIR', async () => {
    mockedRest.getConversionStatus
      .mockResolvedValueOnce(statusFrame({ phase: 'preparing', progress: 0.1 }))
      .mockResolvedValueOnce(statusFrame({ phase: 'pruning', progress: 0.5 }))
      .mockResolvedValueOnce(statusFrame({ phase: 'done', progress: 1, has_ir: true }));

    const onStatus = vi.fn();
    const onIR = vi.fn();

    const result = await runConversion(
      { source: 'ckpt:foo', preset: 'mnist-intlower-v1', overrides: { epochs_per_stage: 2 } },
      { onStatus, onIR, sleep: noSleep },
    );

    expect(result.kind).toBe('done');
    expect(mockedRest.startConversion).toHaveBeenCalledWith({
      source: 'ckpt:foo',
      preset: 'mnist-intlower-v1',
      overrides: { epochs_per_stage: 2 },
    });
    // 3回ポーリング、各回onStatus
    expect(mockedRest.getConversionStatus).toHaveBeenCalledTimes(3);
    expect(onStatus).toHaveBeenCalledTimes(3);
    // done時にIR取得→onIR
    expect(mockedRest.getCurrentIR).toHaveBeenCalledTimes(1);
    expect(onIR).toHaveBeenCalledWith(FAKE_IR);
  });
});

describe('runConversion: error分岐', () => {
  it('phase=errorでエラーメッセージを返しIRは取得しない', async () => {
    mockedRest.getConversionStatus.mockResolvedValueOnce(
      statusFrame({ phase: 'error', error: 'teacher読込失敗' }),
    );
    const onIR = vi.fn();

    const result = await runConversion(
      { source: 'ckpt:bad', preset: 'mnist-intlower-v1' },
      { onStatus: vi.fn(), onIR, sleep: noSleep },
    );

    expect(result).toMatchObject({ kind: 'error', message: 'teacher読込失敗' });
    expect(onIR).not.toHaveBeenCalled();
    expect(mockedRest.getCurrentIR).not.toHaveBeenCalled();
  });

  it('startの例外(409学習中など)はそのままthrowされる', async () => {
    // 409はrest層でApiErrorとして投げられる。ここではその伝播のみ検証(実体はplain Error)。
    mockedRest.startConversion.mockRejectedValueOnce(
      new Error('学習中は変換できません'),
    );

    await expect(
      runConversion(
        { source: '', preset: 'mnist-intlower-v1' },
        { onStatus: vi.fn(), onIR: vi.fn(), sleep: noSleep },
      ),
    ).rejects.toThrow('学習中');
    expect(mockedRest.getConversionStatus).not.toHaveBeenCalled();
  });
});

describe('runConversion: cancel/abort分岐', () => {
  it('phase=cancelledでcancelled outcome', async () => {
    mockedRest.getConversionStatus.mockResolvedValueOnce(
      statusFrame({ phase: 'cancelled' }),
    );
    const result = await runConversion(
      { source: '', preset: 'mnist-intlower-v1' },
      { onStatus: vi.fn(), onIR: vi.fn(), sleep: noSleep },
    );
    expect(result.kind).toBe('cancelled');
  });

  it('shouldAbort()=trueで即abort(ポーリングしない)', async () => {
    const result = await runConversion(
      { source: '', preset: 'mnist-intlower-v1' },
      { onStatus: vi.fn(), onIR: vi.fn(), sleep: noSleep, shouldAbort: () => true },
    );
    expect(result.kind).toBe('aborted');
    expect(mockedRest.getConversionStatus).not.toHaveBeenCalled();
  });

  it('ポーリング途中でabortに切り替わったら打ち切る', async () => {
    let aborted = false;
    mockedRest.getConversionStatus.mockImplementation(() => {
      aborted = true; // 1回目のstatus取得後にabortさせる
      return Promise.resolve(statusFrame({ phase: 'pruning' }));
    });
    const result = await runConversion(
      { source: '', preset: 'mnist-intlower-v1' },
      {
        onStatus: vi.fn(),
        onIR: vi.fn(),
        sleep: noSleep,
        shouldAbort: () => aborted,
      },
    );
    // 1回statusを取り、次ループ冒頭のshouldAbortでabort
    expect(result.kind).toBe('aborted');
    expect(mockedRest.getConversionStatus).toHaveBeenCalledTimes(1);
  });
});

describe('requestCancel', () => {
  it('cancelConversionを呼ぶ(失敗しても例外を投げない)', async () => {
    mockedRest.cancelConversion.mockRejectedValueOnce(new Error('boom'));
    await expect(requestCancel('cid-1')).resolves.toBeUndefined();
    expect(mockedRest.cancelConversion).toHaveBeenCalledWith('cid-1');
  });
});
