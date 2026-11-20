// プレイグラウンドE2E: 実行タブ→実行モードON→入力トグル→出力変化→検証テーブル。
// フォワード実行はフロント内評価(evalCircuit)だが、IR読込(PUT)はバックエンド経由。

import { expect, test } from './fixtures';
import { loadXorSample } from './helpers';

test('実行モードで入力トグルにより出力が変化し、検証テーブルに4行出る', async ({
  page,
}) => {
  await loadXorSample(page);

  // 「実行」タブ → 実行モードON
  await page.getByRole('tab', { name: '実行' }).click();
  await page.getByRole('button', { name: '実行モード OFF' }).click();
  await expect(page.getByRole('button', { name: '実行モード ON' })).toBeVisible();

  // 初期入力(x0=0, x1=0) → XOR=0
  const output = page.locator('.font-mono.text-2xl');
  await expect(output).toHaveText('0');

  // 入力トグルx0クリック → (1,0) → XOR=1 に出力表示が変化
  // (キャンバス上のInputBitsノードにも同titleのトグルがあるため下部ドックへスコープ)
  const dock = page.getByRole('tabpanel');
  await dock.getByTitle('x0 をトグル').click();
  await expect(output).toHaveText('1');

  // 検証テーブル: 2bit入力の全列挙=4行
  await expect(page.locator('tbody tr')).toHaveCount(4);
});
