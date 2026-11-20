// 並列学習ランE2E: 学習開始を2回押すと2ランが同時に走り(以前は2回目が409/無反応)、
// Run一覧の「アクティブ学習」に両方が並び、行ごとに停止/モデルへ適用できることを検証する。

import { expect, test } from './fixtures';
import { goToTopTab, loadXorSample } from './helpers';

// リトライ時に生き残ったセッションを確実に止めるため、開始sidを配列で控える
const activeSids: string[] = [];

test.afterEach(async ({ request }) => {
  for (const sid of activeSids) {
    await request.post(`/api/train/${sid}/stop`).catch(() => undefined);
  }
  activeSids.length = 0;
});

async function startRun(page: import('@playwright/test').Page): Promise<string> {
  const startResp = page.waitForResponse(
    (r) => r.url().includes('/api/train/start') && r.ok(),
  );
  await page.getByRole('button', { name: '学習開始' }).click();
  const sid = ((await (await startResp).json()) as { session_id: string }).session_id;
  activeSids.push(sid);
  return sid;
}

test('学習開始×2で2ランが並列実行され、Run一覧で停止/適用できる', async ({
  page,
  request,
}) => {
  await loadXorSample(page);

  // 学習監視はTRAININGタブへ集約(N1)。Run一覧(RunsPanel)は左ペインに常設される
  await goToTopTab(page, 'training');

  // dataset=XOR、長めに走る設定(batch=1で256step/epoch)にして両ランをrunningに保つ
  await page.getByTestId('dataset-select').click();
  await page.getByRole('option', { name: /XOR/ }).click();
  await page.locator('#train-batch').fill('1');
  await page.locator('#train-epochs').fill('200');

  // 2回連続で開始(2回目が409にならず新sidが返る=本修正の核)
  const sid1 = await startRun(page);
  const sid2 = await startRun(page);
  expect(sid1).not.toEqual(sid2);

  // バックエンドが2ランを並列保持していることを直接確認
  await expect
    .poll(async () => {
      const body = (await (await request.get('/api/train/sessions')).json()) as {
        sessions: { session_id: string }[];
      };
      return body.sessions.map((s) => s.session_id);
    })
    .toEqual(expect.arrayContaining([sid1, sid2]));

  // TRAININGタブ左ペインのRunsPanel(常設)のアクティブ学習に2行以上並ぶ
  await expect(page.getByTestId('active-runs-panel')).toBeVisible();
  await expect
    .poll(() => page.getByTestId('active-run-row').count())
    .toBeGreaterThanOrEqual(2);

  // run1の行で「停止」→ 当該runが終端状態へ
  const row1 = page.getByTestId('active-run-row').filter({ hasText: sid1.slice(0, 8) });
  await row1.getByRole('button', { name: '停止' }).click();
  await expect
    .poll(async () => {
      const body = (await (await request.get(`/api/train/${sid1}/status`)).json()) as {
        state: string;
      };
      return body.state;
    })
    .toMatch(/stopped|finished|error/);

  // 停止したrunの「適用」→ 学習済み重みが編集中モデルへ反映(POST /apply が成功)
  const applyResp = page.waitForResponse(
    (r) => r.url().includes(`/api/train/${sid1}/apply`) && r.ok(),
  );
  await row1.getByRole('button', { name: '適用' }).click();
  expect((await applyResp).ok()).toBeTruthy();
  // run2はafterEachでstopする(activeSidsに記録済み)
});
