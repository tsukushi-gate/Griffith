// E2E: Conv2dをパレットからドロップ → パラメータダイアログ → 配置 →
// conv-nodeがキャンバスへ出現 → 形状診断(出力shapeが解決される)を確認する。

import type { Page } from '@playwright/test';
import { expect, test } from './fixtures';
import { loadXorSample } from './helpers';

/**
 * パレットカード(BLOCK_DND_MIME)をReactFlowペインへドロップする。
 * 実コード経路(onDragStart→onDrop)を、共有DataTransferを持つ
 * ネイティブDragEventで駆動する。
 */
async function dragPaletteToCanvas(page: Page, testid: string): Promise<void> {
  const source = page.getByTestId(testid);
  await expect(source).toBeVisible();
  const pane = page.locator('.react-flow__pane').first();
  await expect(pane).toBeVisible();
  const box = await pane.boundingBox();
  if (!box) throw new Error('react-flow pane not found');
  const cx = box.x + box.width / 2;
  const cy = box.y + box.height / 2;

  await page.evaluate(
    ({ testid, cx, cy }) => {
      const src = document.querySelector(`[data-testid="${testid}"]`);
      const paneEl = document.querySelector('.react-flow__pane');
      if (!src || !paneEl) throw new Error('drag要素が見つからない');
      const dt = new DataTransfer();
      src.dispatchEvent(
        new DragEvent('dragstart', { bubbles: true, cancelable: true, dataTransfer: dt }),
      );
      paneEl.dispatchEvent(
        new DragEvent('dragover', {
          bubbles: true,
          cancelable: true,
          dataTransfer: dt,
          clientX: cx,
          clientY: cy,
        }),
      );
      paneEl.dispatchEvent(
        new DragEvent('drop', {
          bubbles: true,
          cancelable: true,
          dataTransfer: dt,
          clientX: cx,
          clientY: cy,
        }),
      );
    },
    { testid, cx, cy },
  );
}

test('Conv2dをパレットからドロップ→ダイアログ→配置→形状診断OK', async ({ page }) => {
  await loadXorSample(page);

  const before = await page.locator('.react-flow__node').count();

  // パレットのConv2dカードをキャンバスへドロップ → パラメータダイアログが開く
  await dragPaletteToCanvas(page, 'palette-conv2d');
  await expect(page.getByRole('dialog')).toBeVisible();
  await expect(page.getByText('Conv2d のパラメータ')).toBeVisible();

  // 初期化プリセットセレクタが存在する
  await expect(page.getByTestId('init-preset-select')).toBeVisible();

  // 既定値(C=1,H=8,W=8,out=4,k=3,s=1,p=0,depth=2,n=2)のまま配置
  await page.getByRole('button', { name: '配置' }).click();

  // conv-nodeがキャンバスへ出現する
  const conv = page.locator('.conv-node');
  await expect(conv.first()).toBeVisible();
  await expect
    .poll(() => page.locator('.react-flow__node').count())
    .toBeGreaterThan(before);

  // 形状診断: 8x8 → conv(k3,s1) → 6x6, out_channels=4 が解決される(出力shapeが'?'でない)
  await expect(conv.first()).toContainText('(4,6,6)');
});
