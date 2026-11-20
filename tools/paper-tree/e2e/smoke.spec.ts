// スモーク: 描画・ノードクリック→パネル・検索・フィルタ・折りたたみ・両テーマ。
// data/ は他レーンが並行で増やすため、「見本ノード(ptree-sample-*)が存在すること」だけを
// 固定の前提にし、総ノード数には依存しない。

import { expect, test, type Page } from '@playwright/test';

const SAMPLE_PAPER = 'ptree-sample-paper';
const SAMPLE_STUB = 'ptree-sample-stub-2025';

async function waitForCanvas(page: Page) {
  await page.goto('/');
  await expect(page.getByTestId('toolbar')).toBeVisible();
  // elkレイアウト完了後にノードが並ぶ
  await expect(page.locator('.react-flow__node').first()).toBeVisible();
}

test('ツリーが描画される(root/カテゴリ/論文/エッジ)', async ({ page }) => {
  await waitForCanvas(page);
  await expect(page.getByTestId('node-root')).toHaveCount(1);
  expect(await page.getByTestId('node-category').count()).toBeGreaterThan(0);
  expect(await page.getByTestId('node-paper').count()).toBeGreaterThan(0);
  expect(await page.locator('.react-flow__edge').count()).toBeGreaterThan(0);
  await expect(page.locator('.react-flow__minimap')).toBeVisible();
});

test('スタブはグレー(dashed)で描画される', async ({ page }) => {
  await waitForCanvas(page);
  const stub = page.locator(`[data-node-id="${SAMPLE_STUB}"]`);
  await expect(stub).toHaveAttribute('data-status', 'stub');
  await expect(stub).toHaveCSS('border-style', 'dashed');
});

test('ノードクリックで右パネルに落合式Markdownが出る', async ({ page }) => {
  await waitForCanvas(page);
  await page.locator(`[data-node-id="${SAMPLE_PAPER}"]`).click();
  await expect(page.getByTestId('panel-title')).toContainText('落合式');
  await expect(page.getByTestId('panel-status')).toHaveText('summarized');
  // Markdownの見出しがHTMLとして描画されている
  await expect(page.getByTestId('panel-markdown').locator('h3').first()).toContainText('どんなもの');
  // 関連チップ(next / reports)
  expect(await page.getByTestId('related-chip').count()).toBeGreaterThan(0);
});

test('検索(⌘K)でノードへ飛べる', async ({ page }) => {
  await waitForCanvas(page);
  await page.getByTestId('search-button').click();
  // 他レーンのdata/が増えても一意に当たるよう見本のidで引く
  await page.getByTestId('search-input').fill(SAMPLE_STUB);
  const items = page.getByTestId('search-item');
  await expect(items).toHaveCount(1);
  await items.first().click();
  await expect(page.getByTestId('search-overlay')).toHaveCount(0);
  await expect(page.getByTestId('panel-title')).toContainText('スタブ');
});

test('statusフィルタとカテゴリ折りたたみが効く', async ({ page }) => {
  await waitForCanvas(page);
  const before = await page.getByTestId('node-paper').count();

  await page.getByTestId('status-filter').selectOption('stub');
  await expect(page.locator(`[data-node-id="${SAMPLE_PAPER}"]`)).toHaveCount(0);
  await expect(page.locator(`[data-node-id="${SAMPLE_STUB}"]`)).toHaveCount(1);

  await page.getByTestId('status-filter').selectOption('all');
  await expect(page.getByTestId('node-paper')).toHaveCount(before);

  await page.getByTestId('collapse-all').click();
  await expect(page.getByTestId('node-paper')).toHaveCount(0);
  await page.getByTestId('expand-all').click();
  await expect(page.getByTestId('node-paper')).toHaveCount(before);
});

// 公開版は論文データを同梱しない(見本 2 枚のみ)ため規模テストは対象外
test.skip('全論文規模(300+ノード)がレイアウト破綻なく描画される', async ({ page }) => {
  await waitForCanvas(page);
  // P1/S1〜S8統合後の実データ(論文230+ / レポート93 / 合計340+ノード)を下回ったら退行とみなす
  expect(await page.getByTestId('node-paper').count()).toBeGreaterThan(200);
  expect(await page.getByTestId('node-report').count()).toBeGreaterThan(80);
  expect(await page.getByTestId('node-category').count()).toBeGreaterThan(10);

  // elkが失敗して全ノードが原点へ重なる/巨大な縦一列になる破綻を検出する
  const boxes = await page.locator('.react-flow__node').evaluateAll((els) =>
    els.map((el) => {
      const r = el.getBoundingClientRect();
      return { x: r.x, y: r.y, w: r.width, h: r.height };
    }),
  );
  expect(boxes.length).toBeGreaterThan(300);
  expect(boxes.every((b) => b.w > 0 && b.h > 0)).toBe(true);
  const xs = new Set(boxes.map((b) => Math.round(b.x)));
  const ys = new Set(boxes.map((b) => Math.round(b.y)));
  // 2次元に敷き詰められている(縦一列でも横一列でもない)
  expect(xs.size).toBeGreaterThan(5);
  expect(ys.size).toBeGreaterThan(5);

  await page.screenshot({ path: 'test-results/tree-overview-dark.png', fullPage: false });
});

test('ライト/ダーク両テーマで描画できる', async ({ page }) => {
  await waitForCanvas(page);
  // 既定はdark
  await expect(page.locator('html')).toHaveClass(/dark/);
  const darkBg = await page.evaluate(() => getComputedStyle(document.body).backgroundColor);

  await page.getByTestId('theme-toggle').click();
  await expect(page.locator('html')).not.toHaveClass(/dark/);
  const lightBg = await page.evaluate(() => getComputedStyle(document.body).backgroundColor);
  expect(lightBg).not.toBe(darkBg);
  await expect(page.getByTestId('node-root')).toBeVisible();

  // 再読み込みしてもテーマが保持される
  await page.reload();
  await expect(page.locator('html')).not.toHaveClass(/dark/);
  await expect(page.locator('.react-flow__node').first()).toBeVisible();
});
