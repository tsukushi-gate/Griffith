// E2E共通ヘルパ。
// セレクタはrole/テキストベースを優先し、タイミングはexpect(自動リトライ)/
// expect.pollで安定化する。

import { expect, type Page } from '@playwright/test';

/** トップを開き、空状態またはギャラリーから指定サンプルを読み込む。 */
export async function loadSample(page: Page, name: string | RegExp): Promise<void> {
  // 既訪問フラグを事前投入し、初回起動のTemplateGallery自動表示を抑止する
  // (空状態カード起点の既存フローを維持するため)。
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
  await page.goto('/');
  // 空状態オンボーディングカード
  await expect(
    page.getByRole('heading', { name: 'サンプルライブラリ' }),
  ).toBeVisible();
  const direct = page.getByRole('button', { name });
  if ((await direct.count()) > 0 && await direct.first().isVisible()) {
    await direct.first().click();
  } else {
    await page.getByRole('button', { name: 'すべてのサンプル' }).click();
    await page.getByRole('button', { name }).first().click();
  }
  // 読み込み完了 = React Flowノードがキャンバスへレンダリングされる
  await expect
    .poll(() => page.locator('.react-flow__node').count())
    .toBeGreaterThanOrEqual(3);
  // XORチュートリアルガイドが開いていたら閉じる(後続操作の邪魔をしない)
  const closeGuide = page.getByTitle('ガイドを閉じる');
  if ((await closeGuide.count()) > 0) {
    await closeGuide.click().catch(() => undefined);
  }
}

/** トップを開き、空状態カードから「XORミニ」サンプルを読み込む。 */
export async function loadXorSample(page: Page): Promise<void> {
  await loadSample(page, /XORミニ/);
}

/** トップタブ(N1: EDIT/TRAINING/EVALUATION/DATASET/CONFIG)を切り替える。 */
export async function goToTopTab(
  page: Page,
  tab: 'edit' | 'training' | 'evaluation' | 'dataset' | 'config',
): Promise<void> {
  const trigger = page.getByTestId(`toptab-${tab}`);
  await trigger.click();
  await expect(trigger).toHaveAttribute('data-state', 'active');
}

/**
 * DATASETタブでCSVを取込む。取込用の隠しinputへファイルを流し込み、
 * プレビューグリッドが出るまで待つ。呼び出し前に DATASET タブへ遷移しておくこと。
 */
export async function importCsvViaDatasetTab(
  page: Page,
  fileName: string,
  csvContent: string,
): Promise<void> {
  await page.getByTestId('import-csv-input').setInputFiles({
    name: fileName,
    mimeType: 'text/csv',
    buffer: Buffer.from(csvContent, 'utf-8'),
  });
  // 取込成功でプレビューグリッド(=選択済み)が現れる
  await expect(page.getByTestId('sample-preview-grid')).toBeVisible();
}

/**
 * CONFIGタブでプロジェクトを新規作成し、選択状態にする。
 * 呼び出し前に CONFIG タブへ遷移しておくこと。作成後は自動選択されるため
 * Global Config フォーム等がその場で描画される。
 */
export async function createAndSelectProject(page: Page, name: string): Promise<void> {
  await page.getByPlaceholder('新規プロジェクト名').fill(name);
  await page.getByRole('button', { name: '新規作成' }).click();
  await expect(page.getByTestId('project-select')).toContainText(name);
}
