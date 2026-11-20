// 学習E2E: dataset選択→学習開始→チャート描画→タイムライン→pause→
// スライダースクラブ→branch(「ここから再学習」)ボタンの存在。
//
// pause操作を確実にするため、下部ドック「学習」タブの常設フォームで
// batch=1/epochs=200(=数万step)にして学習を長めに走らせる
// (1ゲートモデルなので負荷は無視できる。テスト末尾でstopする)。

import { expect, test } from './fixtures';
import { goToTopTab, loadSample, loadXorSample } from './helpers';

// リトライ時に生き残ったセッションを止めるため、開始レスポンスからsidを控える
let activeSid: string | null = null;

test.afterEach(async ({ request }) => {
  if (activeSid) {
    const sid = activeSid;
    await request.post(`/api/train/${sid}/stop`).catch(() => undefined);
    const deadline = Date.now() + 10_000;
    while (Date.now() < deadline) {
      const status = await request.get(`/api/train/${sid}/status`).catch(() => null);
      if (!status || status.status() === 404) break;
      const body = (await status.json().catch(() => null)) as { state?: string } | null;
      if (body?.state && ['stopped', 'finished', 'error'].includes(body.state)) break;
      await new Promise((resolve) => setTimeout(resolve, 100));
    }
    activeSid = null;
  }
});

test('XORデータセットで学習し、タイムラインでpause→スクラブ→branchボタンが出る', async ({
  page,
}) => {
  await loadXorSample(page);

  // 学習監視はTRAININGタブへ集約(N1)。まずTRAININGへ遷移する
  await goToTopTab(page, 'training');

  // TRAININGタブの学習設定フォーム(常設)でdataset「XOR」を選択
  await page.getByTestId('dataset-select').click();
  await page.getByRole('option', { name: /XOR/ }).click();

  // 長めに走る設定(batch=1で256step/epoch)。pauseの余地を確保する
  await page.locator('#train-batch').fill('1');
  await page.locator('#train-epochs').fill('200');

  // 学習開始(session_idを控えてafterEachで確実にstopできるようにする)
  const startResp = page.waitForResponse(
    (r) => r.url().includes('/api/train/start') && r.ok(),
  );
  await page.getByRole('button', { name: '学習開始' }).click();
  activeSid = ((await (await startResp).json()) as { session_id: string })
    .session_id;

  // status badge変化(ヘッダの状態バッジ)
  const header = page.locator('header');
  await expect(header.getByText('running', { exact: true })).toBeVisible();

  // チャートにデータが乗る: uPlotのcanvasが存在し、step表示が2桁以上まで進む
  // (タイムラインのMonitorSlotにも同様のstep表示があるためチャートペインへスコープ)
  await expect(page.locator('.chart canvas').first()).toBeVisible();
  await expect(
    page.locator('#training-charts').getByText(/step \d{2,} ・ loss/),
  ).toBeVisible();

  // タイムラインタブ → pause
  await page.getByRole('tab', { name: 'タイムライン' }).click();
  // 実行コントロールはヘッダに常設(RunsPanelのActiveRunsにも停止ボタンがあるためheaderへスコープ)
  await header.getByTitle('一時停止').click();
  await expect(header.getByText('paused', { exact: true })).toBeVisible();

  // スライダー操作でstep表示が変わる(replayモードへ入り、カーソルstepを表示する)
  const slider = page.locator('input[type="range"]');
  await expect(slider).toBeEnabled();
  const stepLabel = page.locator('b.text-foreground');
  const before = await stepLabel.textContent();
  expect(Number(before)).toBeGreaterThanOrEqual(10); // pause時点のlive step
  await slider.fill('1'); // 最初のcaptured stepへスクラブ
  await expect(stepLabel).toHaveText('1');

  // branch(「ここから再学習」)ボタンが存在する(paused+replayなので有効)
  await expect(
    page.getByRole('button', { name: 'ここから再学習' }),
  ).toBeVisible();

  // 後始末: 停止して次specへ影響を残さない
  await header.getByRole('button', { name: '停止', exact: true }).click();
  await expect(header.getByText('stopped', { exact: true })).toBeVisible();
  activeSid = null;
});

test('全加算器 学習は読み込み直後に開始でき、stepが十分進む', async ({ page }) => {
  await loadSample(page, /全加算器 学習/);

  const startButton = page.getByRole('button', { name: '学習開始' });
  await expect(startButton).toBeEnabled();

  const startResp = page.waitForResponse(
    (r) => r.url().includes('/api/train/start') && r.ok(),
  );
  await startButton.click();
  activeSid = ((await (await startResp).json()) as { session_id: string })
    .session_id;

  const header = page.locator('header');
  await expect(header.getByText('running', { exact: true })).toBeVisible();
  await expect(header.getByText(/\/3840 step/)).toBeVisible();

  await expect
    .poll(async () => {
      const text = await header.getByText(/\/3840 step/).textContent();
      return Number(text?.match(/^(\d+)/)?.[1] ?? 0);
    }, { timeout: 30_000 })
    .toBeGreaterThanOrEqual(20);

  await header.getByRole('button', { name: '停止', exact: true }).click();
  await expect(header.getByText('stopped', { exact: true })).toBeVisible();
  activeSid = null;
});
