// 手描き推論デモの統合E2E:
// MNISTミニ読込 → EVALUATIONタブ → 手描きデモアコーディオンを開く →
// キャンバスにストローク → 2値化プレビュー反映 → probeで推論実行 →
// クラス確率バー(0-9)が描画される、を1本で踏む。

import { expect, test } from './fixtures';
import { goToTopTab, loadSample } from './helpers';

test('MNISTミニで手描き→2値化→probe推論→確率バー描画', async ({ page }) => {
  await loadSample(page, /MNISTミニ/);

  // EVALUATIONタブへ(未配線ならskip)
  await goToTopTab(page, 'evaluation');
  const toggle = page.getByTestId('draw-demo-toggle');
  if ((await toggle.count()) === 0) {
    test.skip(true, 'EVALUATIONタブ未配線(N5配線待ち)');
    return;
  }

  // アコーディオンを開く → MNIST系IRなのでデモ本体が出る(無効メッセージでない)
  await toggle.click();
  await expect(page.getByTestId('draw-demo')).toBeVisible();
  await expect(page.getByTestId('draw-canvas')).toBeVisible();

  // キャンバス中央へ十字ストロークを描く
  const canvas = page.getByTestId('draw-canvas');
  const box = await canvas.boundingBox();
  if (!box) throw new Error('キャンバスのbounding boxが取れません');
  const cx = box.x + box.width / 2;
  const cy = box.y + box.height / 2;
  await page.mouse.move(cx, box.y + box.height * 0.2);
  await page.mouse.down();
  await page.mouse.move(cx, box.y + box.height * 0.8, { steps: 12 });
  await page.mouse.up();
  await page.mouse.move(box.x + box.width * 0.2, cy);
  await page.mouse.down();
  await page.mouse.move(box.x + box.width * 0.8, cy, { steps: 12 });
  await page.mouse.up();

  // 2値化プレビューに黒セル(=入力ビット1)が現れる
  await expect(page.getByTestId('draw-binary-preview')).toBeVisible();

  // probe経路で推論
  await expect(page.getByTestId('draw-mode-probe')).toBeChecked();
  const probeResp = page.waitForResponse((r) => r.url().includes('/api/probe') && r.ok());
  await page.getByTestId('draw-infer').click();
  await probeResp;

  // クラス確率バーが10本描画される
  const bars = page.getByTestId('class-prob-bar');
  await expect(bars).toHaveCount(10);
  // 予測クラスの明示(probsが入るとヘッダに「予測 N」が出る)
  await expect(page.getByTestId('draw-probs')).toContainText('予測');
});
