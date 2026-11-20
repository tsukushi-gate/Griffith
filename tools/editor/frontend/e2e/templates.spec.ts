// E2E: 新規プロジェクトダイアログ(テンプレギャラリー)→ LeNet風テンプレ選択
// → networksがEDITキャンバスへロード → TRAININGタブで学習開始まで。
//
// テンプレAPI(GET /api/templates, POST /api/templates/{id}/instantiate)が
// main.pyへ統合されている前提(backend/templates.py の include_router)。

import { expect, test } from './fixtures';
import { goToTopTab } from './helpers';

test('テンプレギャラリーからLeNet風を選択→EDITにロード→学習開始準備まで', async ({ page }) => {
  // 初回起動でTemplateGalleryが自動表示される(onboardingVisited未設定)。
  await page.goto('/');
  await expect(page.getByRole('heading', { name: 'テンプレートギャラリー' })).toBeVisible();

  // テンプレプロジェクト節にLeNet風テンプレが並ぶ(GET /api/templates)。
  const lenet = page.getByTestId('template-lenet');
  await expect(lenet).toBeVisible();

  // 選択→instantiate→networksをエディタへロード(IRはbackendへdebounce PUT同期)。
  const irSync = page.waitForResponse(
    (r) => r.url().includes('/api/models/current/ir') && r.ok(),
  );
  await lenet.click();

  // networksがEDITキャンバスへロードされる(conv/pool/lut/group_sum=4ノード以上)。
  await expect
    .poll(() => page.locator('.react-flow__node').count())
    .toBeGreaterThanOrEqual(4);
  await irSync; // backendへIRが同期された(学習前提のモデルが載る)

  // TRAININGタブへ遷移し、MNISTを選ぶと「学習開始」が押下可能になる(=学習開始まで到達)。
  await goToTopTab(page, 'training');
  await page.getByTestId('dataset-select').click();
  await page.getByRole('option', { name: /MNIST/ }).click();
  await expect(page.getByRole('button', { name: '学習開始' })).toBeEnabled();
});
