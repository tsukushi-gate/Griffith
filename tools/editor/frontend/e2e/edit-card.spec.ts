// カードモードEDIT体験の統合E2E:
// graph→nnc-card切替 → グラフモードでノード追加 → /api/stats(統計)が更新 →
// カードのダブルクリックでL0ドリルダウン → パンくずで俯瞰へ復帰、を1本で踏む。

import { expect, test } from './fixtures';
import { loadXorSample } from './helpers';

/** 右インスペクタの統計パネルから「合計ノード」数を読む。 */
async function nodeTotal(page: import('@playwright/test').Page): Promise<number> {
  const el = page.getByTestId('stat-nodes-total');
  await expect(el).toBeVisible();
  const txt = await el.locator('span').last().innerText();
  return parseInt(txt.replace(/[^0-9]/g, ''), 10);
}

test('カード切替→ノード追加で統計増加→ドリルダウン→パンくず復帰', async ({ page }) => {
  await loadXorSample(page);

  // 統計パネルが初期値を表示する(合計ノード数を記録)
  const before = await nodeTotal(page);
  expect(before).toBeGreaterThanOrEqual(1);

  // グラフ → カードモード切替
  await page.getByTestId('displaymode-nnc-card').click();
  await expect(page.getByTestId('nnc-card-canvas')).toBeVisible();
  // ドリルダウン可能なカード(ポート以外)が存在する
  await expect(page.locator('[data-drill="true"]').first()).toBeVisible();

  // カードモードは表示専用 → いったんグラフモードへ戻してノード追加
  await page.getByTestId('displaymode-graph').click();
  await expect(page.locator('.react-flow')).toBeVisible();

  // コマンドパレットからConstant(入力不要=単体で妥当)を追加
  await page.keyboard.press('ControlOrMeta+KeyK');
  await page.getByPlaceholder('コマンドを検索…').fill('Constant');
  await page.getByText('Constantを追加').click();

  // 統計が増える(ローカル追加→デバウンスPUT→/api/stats再取得で確定)
  await expect.poll(() => nodeTotal(page), { timeout: 8000 }).toBeGreaterThan(before);

  // カードモードへ戻り、レイヤーカードをダブルクリックしてL0へドリルダウン
  await page.getByTestId('displaymode-nnc-card').click();
  await expect(page.getByTestId('nnc-card-canvas')).toBeVisible();
  await page.locator('[data-drill="true"]').first().dblclick();

  // L0ビューのパンくずが出る → 「全体」で俯瞰(カードモード)へ復帰
  await expect(page.getByTestId('l0-breadcrumb-home')).toBeVisible();
  await page.getByTestId('l0-breadcrumb-home').click();
  await expect(page.getByTestId('nnc-card-canvas')).toBeVisible();
});
