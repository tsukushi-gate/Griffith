// side-by-side比較デモの推論計測を「同時1本」に揃えるための逐次実行ヘルパーのテスト。
// 資源競合を避ける本題である「同時実行数が常に1であること」を主眼に、
// 実行順序の保持・1列の失敗が残りを止めないことも検証する。
import { describe, expect, it } from 'vitest';

import { runSequential } from './sequentialRun';

function delay(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

describe('runSequential', () => {
  it('同時に in-flight な runFn 呼び出しは常に1つ(最大同時実行数=1)', async () => {
    const items = [10, 5, 8, 3, 12];
    let inFlight = 0;
    let maxInFlight = 0;
    await runSequential(items, async (ms) => {
      inFlight++;
      maxInFlight = Math.max(maxInFlight, inFlight);
      await delay(ms);
      inFlight--;
      return ms;
    });
    expect(maxInFlight).toBe(1);
  });

  it('実行順序は items の順のまま', async () => {
    const items = ['a', 'b', 'c', 'd'];
    const order: string[] = [];
    // 先頭ほど遅く完了するようにしても、開始順=items順が守られていれば
    // orderは items と一致するはず(逐次なので前が終わるまで次は始まらない)。
    const delays: Record<string, number> = { a: 30, b: 10, c: 20, d: 5 };
    await runSequential(items, async (item) => {
      order.push(item);
      await delay(delays[item]);
      return item;
    });
    expect(order).toEqual(items);
  });

  it('outcome の順序・内容も items 順で成功/失敗を保持する', async () => {
    const items = [1, 2, 3];
    const outcomes = await runSequential(items, async (n) => {
      if (n === 2) throw new Error('boom-2');
      return n * 10;
    });
    expect(outcomes).toEqual([
      { ok: true, value: 10 },
      { ok: false, error: new Error('boom-2') },
      { ok: true, value: 30 },
    ]);
  });

  it('1列が例外を投げても残りの列は実行される', async () => {
    const items = ['x', 'y', 'z'];
    const ran: string[] = [];
    const outcomes = await runSequential(items, async (item) => {
      ran.push(item);
      if (item === 'y') throw new Error('fail-y');
      return `ok-${item}`;
    });
    expect(ran).toEqual(items); // 全列が実行された(yで止まっていない)
    expect(outcomes[0]).toEqual({ ok: true, value: 'ok-x' });
    expect(outcomes[1].ok).toBe(false);
    expect(outcomes[2]).toEqual({ ok: true, value: 'ok-z' });
  });

  it('1列が reject しても残りの列は実行される(非Error例外も含む)', async () => {
    const items = [1, 2, 3];
    const ran: number[] = [];
    const outcomes = await runSequential(items, async (n) => {
      ran.push(n);
      if (n === 1) return Promise.reject('string-rejection');
      return n;
    });
    expect(ran).toEqual(items);
    expect(outcomes[0]).toEqual({ ok: false, error: 'string-rejection' });
    expect(outcomes[1]).toEqual({ ok: true, value: 2 });
    expect(outcomes[2]).toEqual({ ok: true, value: 3 });
  });

  it('onSettled は各列の完了ごとに(次の列の開始前に)呼ばれる', async () => {
    const items = [1, 2, 3];
    const events: string[] = [];
    await runSequential(
      items,
      async (n) => {
        events.push(`run:${n}`);
        return n;
      },
      (n, _i, outcome) => {
        events.push(`settled:${n}:${outcome.ok}`);
      },
    );
    expect(events).toEqual([
      'run:1', 'settled:1:true',
      'run:2', 'settled:2:true',
      'run:3', 'settled:3:true',
    ]);
  });

  it('空配列は何もせず空配列を返す', async () => {
    const outcomes = await runSequential([], async () => 1);
    expect(outcomes).toEqual([]);
  });
});
