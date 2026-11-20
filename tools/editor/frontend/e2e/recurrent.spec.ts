// E2E: リカレント時間展開(LSTM unroll)のGUI配置スモーク。
// パレット「RNN/LSTM(unroll)」ドロップ → ダイアログでLSTM/unroll=3を確定 → ノード群が展開 →
// 形状診断OK(内部整合・配置エラーなし) → 統計パネルが機能し続ける、を1本で踏む。
//
// 注記: パレット配置ブロックは外部入力へ未接続の状態でドロップされるため、入口ノードは
// 「入力エッジがない」形状診断となり graphStore は同期を保留する(conv-palette.spec と同じ
// エディタ既定挙動)。内部配線の形状整合はvitest(recurrentUnit.test.ts inferShapes=エラーゼロ)、
// scan_cellとのビット等価は pytest(test_editor_recurrent_n7.py)で担保する。

import type { Page } from '@playwright/test';
import { expect, test } from './fixtures';
import { loadXorSample } from './helpers';

/** パレットカードをReactFlowペインへドロップする(実onDrop経路を駆動、parts-palette.specと同型)。 */
async function dragPaletteToCanvas(page: Page, testid: string): Promise<void> {
  const source = page.getByTestId(testid);
  await expect(source).toBeVisible();
  const pane = page.locator('.react-flow__pane').first();
  await expect(pane).toBeVisible();
  const box = await pane.boundingBox();
  if (!box) throw new Error('react-flow pane not found');
  const cx = box.x + box.width * 0.3;
  const cy = box.y + box.height * 0.3;
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
          bubbles: true, cancelable: true, dataTransfer: dt, clientX: cx, clientY: cy,
        }),
      );
      paneEl.dispatchEvent(
        new DragEvent('drop', {
          bubbles: true, cancelable: true, dataTransfer: dt, clientX: cx, clientY: cy,
        }),
      );
    },
    { testid, cx, cy },
  );
}

test('リカレント: LSTM(unroll=3)ドロップ→ノード群が展開→形状診断OK→統計パネル', async ({ page }) => {
  await loadXorSample(page);
  const nodesBefore = await page.locator('.react-flow__node').count();
  // 統計パネル(合計ノード)が存在する
  await expect(page.getByTestId('stat-nodes-total')).toBeVisible();

  // パレット「RNN/LSTM(unroll)」をドロップ → パラメータダイアログが開く
  await dragPaletteToCanvas(page, 'palette-recurrent');
  await expect(page.getByRole('dialog')).toBeVisible();
  await expect(page.getByText('RNN/LSTM(unroll) のパラメータ')).toBeVisible();
  await expect(page.getByTestId('recurrent-cell-select')).toBeVisible();

  // セル種別=LSTM、unroll=3(既定)のまま配置
  await page.getByTestId('recurrent-cell-select').selectOption('lstm');
  await page.getByRole('button', { name: '配置' }).click();

  // 配置成功トースト(入口/出口ノードidを含む)
  await expect(page.getByText(/RNN\/LSTM\(unroll\) を配置しました/)).toBeVisible();

  // ノード群が展開: セル1step(≒16ノード)×unroll=3 + 初期状態/入力分配。
  // ReactFlowはビューポート外をカリングするため、少なくとも複数ノードの増加を確認する。
  await expect
    .poll(() => page.locator('.react-flow__node').count(), { timeout: 10000 })
    .toBeGreaterThan(nodesBefore);

  // 形状診断OK: 配置エラー(パラメータ不正/内部形状不整合)トーストが出ていない。
  // 内部配線の整合は vitest(inferShapes=エラーゼロ)で厳密担保。
  await expect(page.locator('[data-sonner-toast][data-type="error"]')).toHaveCount(0);

  // 統計パネルは機能し続ける(合計ノード数が数値として読める)。
  const total = await page.getByTestId('stat-nodes-total').locator('span').last().innerText();
  expect(parseInt(total.replace(/[^0-9]/g, ''), 10)).toBeGreaterThanOrEqual(1);
});
