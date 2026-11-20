import { expect, test } from './fixtures';
import { loadSample } from './helpers';

test('LL-ViT I-ViT-T CIFAR-10を読み込み、InspectorからL0 LUTまで降りられる', async ({ page }) => {
  await loadSample(page, /LL-ViT I-ViT-T CIFAR-10/);

  await expect(page.getByRole('button', { name: '学習開始' })).toBeDisabled();
  await expect(page.getByRole('button', { name: '学習開始' })).toHaveAttribute(
    'title',
    /Modal\/experiments\/train_llvit.py/,
  );

  await page.getByRole('tab', { name: '階層' }).click();
  await page.getByRole('button', { name: /Transformer Encoder x12/ }).click();
  await page.getByRole('button', { name: /Encoder 0/ }).click();
  await page.getByRole('button', { name: /MHA Token Mixer/ }).click();
  await expect(page.getByText('MHAは第1段階ではPython実行のみです')).toBeVisible();

  await page.getByRole('button', { name: /Encoder 0/ }).click();
  await page.getByRole('button', { name: /LL Channel Mixer/ }).click();
  await expect(page.getByText('元MLP:')).toBeVisible();
  await expect(
    page.getByText('thermometer -> DWN/LUT -> DWN/LUT -> conditional sum', { exact: true }),
  ).toBeVisible();

  await page.getByRole('button', { name: /LUT Layer 0/ }).click();
  await page.getByRole('button', { name: /L0で開く/ }).click();
  // でL0にパンくず(同名のレイヤ名)を追加したため、オーバーレイ本文に限定する
  await expect(
    page.locator('.l0-overlay').getByText(/Encoder 0 LUT Layer 0/),
  ).toBeVisible();
  await expect(page.getByText(/入力 1536 ・ ゲート 768 ・ N=6/)).toBeVisible();
});
