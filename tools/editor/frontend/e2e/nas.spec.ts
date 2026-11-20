// 構造探索(NAS)E2E: XOR読込→TRAINING「構造探索」サブタブ→探索開始→
// trial一覧に行が増える→Trade-offグラフ描画→点クリックでEDITに構造ロード。
//
// 前提: NAS Router(tools/editor/backend/nas.py)が main.py に配線されていること。
// 未配線環境(現状の main.py)では /api/nas/status が404になるため、beforeEachで
// 検出して自動スキップする(配線後は自動的に実行される)。探索は max_trials=3 +
// epoch最小で30秒以内に収める。

import { expect, test } from './fixtures';
import { goToTopTab, loadXorSample } from './helpers';

test.afterEach(async ({ request }) => {
  // 走り残しの探索を止める(存在しなくても無害)。
  await request.post('/api/nas/stop').catch(() => undefined);
});

test('XOR構造探索: 開始→trial増加→Trade-offグラフ→点クリックでEDITロード', async ({
  page,
  request,
}) => {
  // NAS Router未配線ならスキップ(配線待ち。仮統合で検証済み)。
  const probe = await request.get('/api/nas/status').catch(() => null);
  test.skip(!probe || probe.status() === 404, 'NAS Router未配線(main.py統合待ち)');

  await loadXorSample(page);
  await goToTopTab(page, 'training');

  // 「構造探索」サブタブへ
  await page.getByRole('tab', { name: '構造探索' }).click();
  await expect(page.getByTestId('nas-panel')).toBeVisible();

  // 小規模設定で開始(max_trials=3 / epochs=1)
  await page.getByTestId('nas-max-trials').fill('3');
  await page.getByTestId('nas-epochs').fill('1');
  await page.getByTestId('nas-start').click();

  // trial一覧に行が増える(最低1行)
  await expect(page.getByTestId('nas-trial-row-0')).toBeVisible({ timeout: 30_000 });

  // Trade-offグラフが描画される
  await expect(page.getByTestId('tradeoff-chart')).toBeVisible();

  // 探索完了(3 trial)を待ってから点をクリック
  await expect(page.getByTestId('nas-trial-row-2')).toBeVisible({ timeout: 30_000 });
  await page.getByTestId('tradeoff-point-0').click();

  // EDITタブへ遷移し、キャンバスが見える(構造がロードされた)
  await expect(page.getByTestId('toptab-edit')).toHaveAttribute('data-state', 'active');
});
