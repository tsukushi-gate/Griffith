// トップタブ遷移スモークE2E: EDIT→TRAINING→EVALUATION→DATASET→CONFIG を順に踏み、
// 各タブの目印要素が描画されることを検証する。タブ切替後もキャンバス状態が保持されることも確認。
// キーボード1〜5でのタブ切替、: URLハッシュディープリンクも検証する。

import { expect, test } from './fixtures';
import { goToTopTab, loadXorSample } from './helpers';

test('5トップタブを順に遷移でき、各タブの目印要素が描画される', async ({ page }) => {
  await loadXorSample(page);

  // EDIT: キャンバスにノードが描画されている(既定タブ)
  await goToTopTab(page, 'edit');
  await expect
    .poll(() => page.locator('.react-flow__node').count())
    .toBeGreaterThanOrEqual(3);

  // TRAINING: 学習設定フォーム(dataset-select)が常設で見える
  await goToTopTab(page, 'training');
  await expect(page.getByTestId('dataset-select')).toBeVisible();
  // タイムライン/ログのサブタブが存在する
  await expect(page.getByRole('tab', { name: 'タイムライン', exact: true })).toBeVisible();

  // EVALUATION: 本実装のタブ本体が描画される
  await goToTopTab(page, 'evaluation');
  await expect(page.getByTestId('evaluation-tab')).toBeVisible();

  // DATASET: 本実装のタブ本体が描画される
  await goToTopTab(page, 'dataset');
  await expect(page.getByTestId('dataset-tab')).toBeVisible();

  // CONFIG: 本実装のプロジェクトセレクタが描画される
  await goToTopTab(page, 'config');
  await expect(page.getByTestId('project-select')).toBeVisible();

  // EDITへ戻るとキャンバス状態(ノード)が保持されている(アンマウントしていない)
  await goToTopTab(page, 'edit');
  await expect
    .poll(() => page.locator('.react-flow__node').count())
    .toBeGreaterThanOrEqual(3);
});

test('キーボード1〜5でトップタブが切り替わる', async ({ page }) => {
  await loadXorSample(page);

  // キャンバス領域をクリックしてフォーカスを入力欄から外してからキー送信する
  await page.locator('.react-flow__pane').click();

  const order: Array<[string, string]> = [
    ['2', 'training'],
    ['3', 'evaluation'],
    ['4', 'dataset'],
    ['5', 'config'],
    ['1', 'edit'],
  ];
  for (const [key, tab] of order) {
    await page.keyboard.press(key);
    await expect(page.getByTestId(`toptab-${tab}`)).toHaveAttribute('data-state', 'active');
  }
});

test('URLハッシュで直接開くと対応するトップタブがアクティブになる', async ({ page }) => {
  // オンボーディングのTemplateGalleryが出ないよう先に既訪問フラグを仕込む
  await page.addInitScript(() => {
    try {
      window.localStorage.setItem(
        'md-editor-ui',
        JSON.stringify({ state: { onboardingVisited: true }, version: 0 }),
      );
    } catch {
      /* localStorage不可の環境は無視 */
    }
  });
  await page.goto('/#evaluation');
  await expect(page.getByTestId('toptab-evaluation')).toHaveAttribute('data-state', 'active');
  await expect(page.getByTestId('evaluation-tab')).toBeVisible();
});
