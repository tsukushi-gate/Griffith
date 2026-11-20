// parseAutobenchParams のURLクエリ解釈を検証する(純粋関数、DOM不要。
// href を明示的に渡すことで vitest(environment: node)でも window に触れず動く)。
import { describe, expect, it } from 'vitest';

import { parseAutobenchParams, pipelineEnabledFromUrl } from './autobench';

describe('parseAutobenchParams', () => {
  it('autobench=1 が無ければ null(通常UIをそのまま動かす)', () => {
    expect(parseAutobenchParams('det', 'https://x.test/det.html')).toBeNull();
    expect(parseAutobenchParams('det', 'https://x.test/det.html?backend=wasm')).toBeNull();
  });

  it('backend/framesを指定通りに解釈する', () => {
    const p = parseAutobenchParams(
      'det', 'https://x.test/det.html?autobench=1&backend=webgpu&frames=50',
    );
    expect(p).toEqual({ task: 'det', backend: 'webgpu', frames: 50 });
  });

  it('backend未指定/不正値は既定でwasmへ落ちる', () => {
    expect(
      parseAutobenchParams('seg', 'https://x.test/seg.html?autobench=1&frames=10'),
    ).toEqual({ task: 'seg', backend: 'wasm', frames: 10 });
    expect(
      parseAutobenchParams(
        'seg', 'https://x.test/seg.html?autobench=1&backend=nope&frames=10',
      ),
    ).toEqual({ task: 'seg', backend: 'wasm', frames: 10 });
  });

  it('frames未指定/不正値は既定で50へ落ちる', () => {
    expect(
      parseAutobenchParams('detmono', 'https://x.test/detmono.html?autobench=1&backend=js'),
    ).toEqual({ task: 'detmono', backend: 'js', frames: 50 });
    expect(
      parseAutobenchParams(
        'detmono', 'https://x.test/detmono.html?autobench=1&backend=js&frames=-3',
      ),
    ).toEqual({ task: 'detmono', backend: 'js', frames: 50 });
  });

  it('task引数はそのまま結果に反映される(呼び出し元ごとの識別用)', () => {
    const p = parseAutobenchParams('detface', 'https://x.test/detface.html?autobench=1');
    expect(p?.task).toBe('detface');
  });

  it('pipeline=1 は parseAutobenchParams の戻り値を変えない(既存契約の保護)', () => {
    expect(
      parseAutobenchParams(
        'det', 'https://x.test/det.html?autobench=1&backend=webgpu&frames=50&pipeline=1',
      ),
    ).toEqual({ task: 'det', backend: 'webgpu', frames: 50 });
  });
});

describe('pipelineEnabledFromUrl', () => {
  it('pipeline=1 のときだけ true(既定オフ=既存セルの計測内容を変えない)', () => {
    expect(pipelineEnabledFromUrl('https://x.test/det.html?autobench=1&pipeline=1')).toBe(true);
    expect(pipelineEnabledFromUrl('https://x.test/det.html?autobench=1')).toBe(false);
    expect(pipelineEnabledFromUrl('https://x.test/det.html?autobench=1&pipeline=0')).toBe(false);
    expect(pipelineEnabledFromUrl('https://x.test/det.html?autobench=1&pipeline=yes')).toBe(false);
  });
});
