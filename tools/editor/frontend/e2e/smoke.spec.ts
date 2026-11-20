// スモークE2E: トップ表示→空状態カード→XORミニ読込→キャンバス表示→コード生成。
// 過去の「yしか表示されない」表示バグ(ノードが1個しか描画されない)の直接回帰として、
// React Flowノード総数と inputBits/gate/出力ノードそれぞれの存在を検証する。

import { expect, test } from './fixtures';
import { loadXorSample } from './helpers';

test('XORミニ読込でノードが3個以上レンダリングされ、コード生成できる', async ({ page }) => {
  await loadXorSample(page);

  // 表示バグ回帰: ノード総数(XORミニは入力/ゲート層/出力で3個以上)
  await expect
    .poll(() => page.locator('.react-flow__node').count())
    .toBeGreaterThanOrEqual(3);
  // 種別ごとの存在: 入力ビット展開ノード / XORゲートカード / 出力ポートy
  await expect(page.locator('.react-flow__node-inputBits')).toHaveCount(1);
  await expect(page.locator('.react-flow__node-gate').first()).toBeVisible();
  await expect(
    page.locator('.react-flow__node', { hasText: '出力' }).first(),
  ).toBeVisible();

  // コードタブ → 生成コードにモデル構築コードが出る
  await page.getByRole('tab', { name: 'コード' }).click();
  await page.getByRole('button', { name: 'コード生成' }).click();
  const code = page.locator('pre.code-block');
  await expect(code).toContainText('LutLayer');
  await expect(code).toContainText('build_model');
});
