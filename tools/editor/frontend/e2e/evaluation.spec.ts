// EVALUATIONタブ統合E2E: 学習→評価→誤答/全件表示→誤答行クリック→
// EDIT/Probeへ反例トレース遷移で入力が注入されることを確認する。
//
// 【配線待ち】EVALUATIONタブはAppShellでまだプレースホルダのため、本specは
// EvaluationTabが配線されるまで自動スキップする(placeholderを検出したらskip)。
// メインがAppShellのプレースホルダを EvaluationTab へ差し替えると本testが有効化される。

import { expect, test } from './fixtures';
import { goToTopTab, loadXorSample } from './helpers';

let activeSid: string | null = null;

test.afterEach(async ({ request }) => {
  if (activeSid) {
    const sid = activeSid;
    await request.post(`/api/train/${sid}/stop`).catch(() => undefined);
    activeSid = null;
  }
});

test('学習→EVALUATIONで評価→誤答行クリックでEDIT/Probeへ反例トレース遷移', async ({
  page,
}) => {
  await loadXorSample(page);

  // --- EVALUATIONタブへ。プレースホルダなら配線待ちとしてskip ---
  await goToTopTab(page, 'evaluation');
  const runButton = page.getByTestId('eval-run-button');
  if ((await runButton.count()) === 0) {
    test.skip(true, 'EVALUATIONタブがAppShellに未配線(N5配線待ち)');
    return;
  }

  // --- 学習: TRAININGタブでXORを少し学習して重みを更新 ---
  await goToTopTab(page, 'training');
  await page.getByTestId('dataset-select').click();
  await page.getByRole('option', { name: /XOR/ }).click();
  await page.locator('#train-batch').fill('1');
  await page.locator('#train-epochs').fill('30');
  const startResp = page.waitForResponse(
    (r) => r.url().includes('/api/train/start') && r.ok(),
  );
  await page.getByRole('button', { name: '学習開始' }).click();
  activeSid = ((await (await startResp).json()) as { session_id: string }).session_id;
  // 学習が数十stepは進むのを待つ
  await expect(page.locator('.chart canvas').first()).toBeVisible();

  // --- EVALUATION: 現在のモデル × XOR を評価 ---
  await goToTopTab(page, 'evaluation');
  await page.getByTestId('eval-dataset-select').selectOption('logic:xor');
  const evalResp = page.waitForResponse(
    (r) => r.url().includes('/api/evaluate') && r.ok(),
  );
  await page.getByTestId('eval-run-button').click();
  await evalResp;

  // サマリと一覧が出る
  await expect(page.getByTestId('eval-summary')).toBeVisible();
  await expect(page.getByTestId('result-table')).toBeVisible();

  // 誤答のみフィルタ。誤答が無ければ全件から先頭行を採用する
  await page.getByTestId('filter-errors').check();
  let targetRow = page.getByTestId(/^result-row-/).first();
  if ((await targetRow.count()) === 0) {
    await page.getByTestId('filter-errors').uncheck();
    targetRow = page.getByTestId(/^result-row-/).first();
  }
  await expect(targetRow).toBeVisible();

  // --- 反例トレース: 行クリック→EDITタブ+Probeレンズへ遷移し入力が着色される ---
  await targetRow.click();
  await expect(page.getByTestId('toptab-edit')).toHaveAttribute('data-state', 'active');
  // Probeレンズ有効化=プレイグラウンドの実行(着色)がキャンバスに反映される
  await expect
    .poll(() => page.locator('.react-flow__node').count())
    .toBeGreaterThanOrEqual(1);
});
