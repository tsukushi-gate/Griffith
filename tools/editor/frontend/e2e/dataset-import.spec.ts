// DATASET取込E2E: XOR相当CSVを取込→DATASETプレビュー確認→
// TRAININGタブで当該データを選択→学習→accが閾値到達、までを1本で踏む。
// 北極星: 取込データだけで学習が完結する(DATASET→TRAINING地続き)。
//
// 前提: DATASETタブがAppShellへ配線済みであること(差し替え=メインが配線)。
//   未配線の間は本specはDATASETタブUIが出ずに失敗するため「AppShell配線待ち」。
//
// XOR相当CSV: in_width=2 / out_width=1(binary_vector) → XORミニ・モデルとI/O一致。

import { expect, test } from './fixtures';
import { goToTopTab, importCsvViaDatasetTab, loadXorSample } from './helpers';

const XOR_CSV = 'x__0,x__1,y\n0,0,0\n0,1,1\n1,0,1\n1,1,0\n';

let activeSid: string | null = null;

test.afterEach(async ({ request }) => {
  if (activeSid) {
    const sid = activeSid;
    await request.post(`/api/train/${sid}/stop`).catch(() => undefined);
    activeSid = null;
  }
});

test('取込CSVで学習が地続きに完結する(取込→プレビュー→選択→学習→acc到達)', async ({
  page,
}) => {
  // I/O幅を取込XOR(2→1)に合わせるためXORミニ・モデルを読み込む
  await loadXorSample(page);

  // DATASETタブでCSV取込 → プレビューグリッド表示
  await goToTopTab(page, 'dataset');
  await importCsvViaDatasetTab(page, 'xor.csv', XOR_CSV);
  // 論理ビット列サンプルが4件見える
  await expect(page.getByTestId('sample-cell').first()).toBeVisible();
  // クラス分布ミニバーが出る
  await expect(page.getByTestId('class-dist')).toBeVisible();

  // TRAININGタブで取込データセットを選択(カタログへ露出)
  await goToTopTab(page, 'training');
  await page.getByTestId('dataset-select').click();
  // 取込データセットは「(取込)」ラベルで現れる
  await page.getByRole('option', { name: /取込/ }).first().click();

  // 全列挙4行を素早く回すため小バッチ・多epoch
  await page.locator('#train-batch').fill('4');
  await page.locator('#train-epochs').fill('300');

  const startResp = page.waitForResponse(
    (r) => r.url().includes('/api/train/start') && r.ok(),
  );
  await page.getByRole('button', { name: '学習開始' }).click();
  activeSid = ((await (await startResp).json()) as { session_id: string }).session_id;

  // acc到達(hard eval)。XORは1ゲートモデルで100%到達可能。緩めの閾値で安定化。
  await expect
    .poll(
      async () => {
        const resp = await page.request.get(`/api/train/${activeSid}/status`);
        if (!resp.ok()) return 0;
        const body = (await resp.json()) as { acc?: number | null };
        return body.acc ?? 0;
      },
      { timeout: 60_000, intervals: [500] },
    )
    .toBeGreaterThan(0.9);
});
