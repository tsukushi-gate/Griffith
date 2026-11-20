// 5タブ横断PlaywrightジャーニーE2E。
// 「NNCユーザーが迷わず操作できる」ことの機械検証(ROADMAP-v5 付録A台本)。
// CSV取込→組む→学習→評価→出荷が1本で通ることを検証する。
// 手描きデモ・テンプレギャラリー(N9、並行実装中)はジャーニーに含めない(付録Aの
// 「新規プロジェクトダイアログ」も未実装のためEDITタブ直起動で代替する)。

import { expect, test } from './fixtures';
import {
  createAndSelectProject,
  goToTopTab,
  importCsvViaDatasetTab,
  loadXorSample,
} from './helpers';

const XOR_CSV = 'x__0,x__1,y\n0,0,0\n0,1,1\n1,0,1\n1,1,0\n';

/** 右インスペクタの統計パネルから「合計ノード」数を読む。*/
async function nodeTotal(page: import('@playwright/test').Page): Promise<number> {
  const el = page.getByTestId('stat-nodes-total');
  await expect(el).toBeVisible();
  const txt = await el.locator('span').last().innerText();
  return parseInt(txt.replace(/[^0-9]/g, ''), 10);
}

let activeSid: string | null = null;

test.afterEach(async ({ request }) => {
  if (activeSid) {
    const sid = activeSid;
    await request.post(`/api/train/${sid}/stop`).catch(() => undefined);
    activeSid = null;
  }
});

test('5タブ横断: CSV取込→組む→学習→評価→出荷が1本で通る', async ({ page }) => {
  // --- 1. 起動: XORミニを読み込み、5タブ帯が描画される ---
  await loadXorSample(page);
  for (const tab of ['edit', 'training', 'evaluation', 'dataset', 'config'] as const) {
    await expect(page.getByTestId(`toptab-${tab}`)).toBeVisible();
  }

  // --- 2. DATASET: XOR相当CSVを取込 → カタログに出現+プレビューグリッドに4行 ---
  await goToTopTab(page, 'dataset');
  await importCsvViaDatasetTab(page, 'xor-journey.csv', XOR_CSV);
  await expect(page.getByTestId('sample-cell')).toHaveCount(4);
  await expect(page.getByTestId('class-dist')).toBeVisible();
  // 取込データセットがカタログに現れる(プレビュー対象はプレビューグリッド側で確認済み)
  const importedCatalogEntry = page.locator('[data-testid^="dataset-cat-imported:"]').first();
  await expect(importedCatalogEntry).toBeVisible();

  // --- 3. DATASET: エンコーディングプレビュー切替 → 2値化後ビット列が表示される ---
  await page.getByTestId('encoding-toggle').check();
  await expect(page.getByTestId('encoding-grid')).toBeVisible();
  await page.getByTestId('encoding-toggle').uncheck();

  // --- 4. EDIT: XORミニ(既読込)をカード表示に切替 → 縦積みカード+形状注記 ---
  await goToTopTab(page, 'edit');
  const nodesBefore = await nodeTotal(page);
  expect(nodesBefore).toBeGreaterThanOrEqual(1);
  await page.getByTestId('displaymode-nnc-card').click();
  await expect(page.getByTestId('nnc-card-canvas')).toBeVisible();
  await expect(page.locator('[data-drill="true"]').first()).toBeVisible();

  // --- 5. EDIT: ノードを1個追加 → 統計パネルのtotal_lutsが増加 ---
  await page.getByTestId('displaymode-graph').click();
  await expect(page.locator('.react-flow')).toBeVisible();
  await page.keyboard.press('ControlOrMeta+KeyK');
  await page.getByPlaceholder('コマンドを検索…').fill('Constant');
  await page.getByText('Constantを追加').click();
  await expect.poll(() => nodeTotal(page), { timeout: 8000 }).toBeGreaterThan(nodesBefore);

  // --- 6. EDIT: カードをダブルクリック → ゲート詳細(L0)へ → パンくずで復帰 ---
  await page.getByTestId('displaymode-nnc-card').click();
  await expect(page.getByTestId('nnc-card-canvas')).toBeVisible();
  await page.locator('[data-drill="true"]').first().dblclick();
  await expect(page.getByTestId('l0-breadcrumb-home')).toBeVisible();
  await page.getByTestId('l0-breadcrumb-home').click();
  await expect(page.getByTestId('nnc-card-canvas')).toBeVisible();
  // カードモードは表示専用(設計裁定2)。以降の反例トレース確認はグラフモード前提のため戻す
  await page.getByTestId('displaymode-graph').click();
  await expect(page.locator('.react-flow')).toBeVisible();

  // --- 7. CONFIG: プロジェクト作成 → Max Epoch設定 → Optimizer定義 ---
  await goToTopTab(page, 'config');
  await createAndSelectProject(page, `journey-${Date.now()}`);
  const epochsField = page.getByLabel('Epochs');
  await expect(epochsField).toBeVisible();
  await epochsField.fill('30');
  await expect(epochsField).toHaveValue('30');
  const optimizerSection = page.locator('section', { hasText: 'Optimizer定義' });
  await optimizerSection.getByPlaceholder('名前').fill('opt-journey');
  await optimizerSection.getByRole('button', { name: 'Optimizerを追加' }).click();
  await expect(optimizerSection.getByText('opt-journey', { exact: true })).toBeVisible();

  // --- 8. TRAINING: 取込データセットを選択して学習開始 ---
  await goToTopTab(page, 'training');
  await page.getByTestId('dataset-select').click();
  await page.getByRole('option', { name: /取込/ }).first().click();
  await page.locator('#train-batch').fill('4');
  await page.locator('#train-epochs').fill('300');
  const startResp = page.waitForResponse(
    (r) => r.url().includes('/api/train/start') && r.ok(),
  );
  await page.getByRole('button', { name: '学習開始' }).click();
  activeSid = ((await (await startResp).json()) as { session_id: string }).session_id;

  // --- 9. TRAINING: acc閾値到達を待つ ---
  await expect
    .poll(
      async () => {
        const resp = await page.request.get(`/api/train/${activeSid}/status`);
        if (!resp.ok()) return 0;
        const body = (await resp.json()) as { acc?: number | null };
        return body.acc ?? 0;
      },
      { timeout: 60_000, intervals: [500] },
    )
    .toBeGreaterThan(0.9);

  // --- 10. EVALUATION: 一括評価実行 → 一覧+混同行列切替可 ---
  await goToTopTab(page, 'evaluation');
  await page.getByTestId('eval-dataset-select').selectOption('logic:xor');
  const evalResp = page.waitForResponse(
    (r) => r.url().includes('/api/evaluate') && r.ok(),
  );
  await page.getByTestId('eval-run-button').click();
  await evalResp;
  await expect(page.getByTestId('eval-summary')).toBeVisible();
  await expect(page.getByTestId('result-table')).toBeVisible();

  // --- 11. EVALUATION: 誤答フィルタ→行クリック→EDIT/Probeへ反例トレース遷移 ---
  await page.getByTestId('filter-errors').check();
  let targetRow = page.getByTestId(/^result-row-/).first();
  if ((await targetRow.count()) === 0) {
    await page.getByTestId('filter-errors').uncheck();
    targetRow = page.getByTestId(/^result-row-/).first();
  }
  await expect(targetRow).toBeVisible();
  await targetRow.click();
  await expect(page.getByTestId('toptab-edit')).toHaveAttribute('data-state', 'active');
  await expect
    .poll(() => page.locator('.react-flow__node').count())
    .toBeGreaterThanOrEqual(1);

  // --- 12. EDIT(Ship): 出荷パネルでpreflight表示 ---
  await page.getByRole('tab', { name: '出荷' }).click();
  await expect(page.getByText('出荷プリフライト')).toBeVisible();
  await expect.poll(
    async () => (await page.getByText(/HDL出力/).count()) > 0,
    { timeout: 10_000 },
  ).toBe(true);

  // --- 13. CONFIG: プロジェクト保存→再読込→設定が復元される ---
  await goToTopTab(page, 'config');
  await createAndSelectProject(page, `journey-reload-check-${Date.now()}`);
  // project-selectで元プロジェクトへ戻し、サーバから再取得させる(=再読込相当)
  await page.getByTestId('project-select').click();
  await page.getByRole('option', { name: /^journey-\d+$/ }).first().click();
  await expect(page.getByLabel('Epochs')).toHaveValue('30');
  await expect(
    page.locator('section', { hasText: 'Optimizer定義' }).getByText('opt-journey', { exact: true }),
  ).toBeVisible();
});
