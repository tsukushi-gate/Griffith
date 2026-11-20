// E2E: モデル部品パレット(PositionalEncoding / DWN)の統合スモーク。
// PositionalEncodingドロップ → constant+concat出現(形状表示) → concat削除で
// PE由来のconstantが妥当グラフとして同期 → 統計パネル(合計ノード)へ反映 →
// DWNドロップ → lut_layerノード出現 → インスペクタにkind=DWNバッジ表示、を1本で踏む。

import type { Page } from '@playwright/test';
import { expect, test } from './fixtures';
import { loadXorSample } from './helpers';

/** パレットカード(BLOCK_DND_MIME)をReactFlowペインへドロップする(実onDrop経路を駆動)。 */
async function dragPaletteToCanvas(page: Page, testid: string): Promise<void> {
  const source = page.getByTestId(testid);
  await expect(source).toBeVisible();
  const pane = page.locator('.react-flow__pane').first();
  await expect(pane).toBeVisible();
  const box = await pane.boundingBox();
  if (!box) throw new Error('react-flow pane not found');
  // 右下のminimapと重なるとノードのクリックが妨げられるため、左上寄りへ配置する。
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

/** 統計パネルの「合計ノード」数を読む。 */
async function nodeTotal(page: Page): Promise<number> {
  const el = page.getByTestId('stat-nodes-total');
  await expect(el).toBeVisible();
  const txt = await el.locator('span').last().innerText();
  return parseInt(txt.replace(/[^0-9]/g, ''), 10);
}

/** 配置トーストから入口ノードidを取り出す。 */
async function entryIdFromToast(page: Page, labelRe: RegExp): Promise<string> {
  const toast = page.getByText(labelRe);
  await expect(toast).toBeVisible();
  const text = await toast.innerText();
  const m = /入口:\s*(\S+?)\s*\//.exec(text);
  if (!m) throw new Error(`入口ノードidをトーストから取得できない: ${text}`);
  return m[1];
}

test('部品パレット: PE/DWNドロップ→ノード出現→形状診断OK→統計反映', async ({ page }) => {
  await loadXorSample(page); // x入力幅=2
  const before = await nodeTotal(page);

  // --- PositionalEncoding: ドロップ → constant + concat が出現 ---
  const nodesBefore = await page.locator('.react-flow__node').count();
  await dragPaletteToCanvas(page, 'palette-positional_encoding');
  await expect(page.getByRole('dialog')).toBeVisible();
  await expect(page.getByText('PositionalEncoding のパラメータ')).toBeVisible();
  await expect(page.getByTestId('pe-code-select')).toBeVisible();
  // 既定(入力幅8, S4, コード幅3, binary)のまま配置 → constant+concat の2ノード追加
  await page.getByRole('button', { name: '配置' }).click();
  const concatId = await entryIdFromToast(page, /PositionalEncoding を配置しました/);
  await expect
    .poll(() => page.locator('.react-flow__node').count())
    .toBeGreaterThanOrEqual(nodesBefore + 2);
  // 形状表示: constantノードが width を表示(形状注記が解決される)
  await expect(page.locator('.react-flow__node').filter({ hasText: 'constant' }).first())
    .toBeVisible();

  // --- 統計反映: concatを削除するとPE由来のconstantが妥当グラフとして同期される ---
  // (concatはslot0未接続で形状エラー→同期を止めるため、削除して定数線のみ残す)
  await page.locator(`.react-flow__node[data-id="${concatId}"]`).click();
  await page.keyboard.press('Delete');
  // constantノード(入力不要=形状エラーなし)が加わり合計ノードが増える
  await expect.poll(() => nodeTotal(page), { timeout: 10000 }).toBeGreaterThan(before);

  // --- DWN: XOR入力(幅2)整合でドロップ → lut_layerノード出現 → kind=DWNバッジ ---
  await dragPaletteToCanvas(page, 'palette-dwn');
  await expect(page.getByRole('dialog')).toBeVisible();
  await expect(page.getByText('DWN層 のパラメータ')).toBeVisible();
  const nums = page.getByRole('dialog').getByRole('spinbutton');
  await nums.nth(0).fill('2'); // 入力幅
  await nums.nth(1).fill('2'); // 出力幅
  await nums.nth(2).fill('2'); // n_inputs
  await page.getByRole('button', { name: '配置' }).click();
  const dwnId = await entryIdFromToast(page, /DWN層 を配置しました/);
  await expect(page.locator(`.react-flow__node[data-id="${dwnId}"]`)).toBeVisible();
  // 小さいlut層は詳細ゲートカードへ自動展開される。子ゲートのクリックで親層を選択する
  // (FlowCanvas.onNodeClick: 詳細子ノード→setSelected(layerId))。
  await page.locator(`.react-flow__node[data-id="${dwnId}::g0"]`).click();
  // インスペクタで DWN kindバッジ + method提案が表示される。
  await expect(page.getByTestId('node-kind-badge')).toContainText('DWN');
  await expect(page.getByTestId('dwn-method-hint')).toBeVisible();
});
