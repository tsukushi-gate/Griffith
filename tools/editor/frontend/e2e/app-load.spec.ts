// アプリ読込クラッシュのE2E回帰ガード。
// MNISTミニ(out=60の展開可能層を1つ持つ)を読み込むと、セマンティックズームの
// fit⇔自動展開の双安定振動でReactが "Maximum update depth exceeded" を起こし白画面化する
// 既存バグ(コミット4c1747b由来)への直接回帰。fixtures.tsのcrashGuardが致命エラーを検出する。
// 併せて全BottomDockタブを巡回し、各パネルのマウントで実行時エラーが出ないことを確認する。

import { expect, test } from './fixtures';
import { goToTopTab, loadSample } from './helpers';

// N1: 下部タブはトップタブごとに分割された。EDIT=実行/コード/出荷、TRAINING=タイムライン/ログ。
const EDIT_TABS = ['実行', 'コード', '出荷'] as const;
const TRAINING_TABS = ['タイムライン', 'ログ'] as const;

// 双安定振動はキャンバスのfitズームがLOD閾値(zoom≈0.393)を跨ぐ時のみ起きるため、
// 実利用に近い広めのビューポートで再現性を確保する(小さい枠だとfitズームが閾値未満で素通り)。
test.use({ viewport: { width: 2560, height: 1440 } });

test('MNISTミニを読み込んでも無限ループでクラッシュせず、全タブを巡回できる', async ({
  page,
}) => {
  // 読込＝キャンバス描画で振動が起きるとここで "Maximum update depth" が発火する
  await loadSample(page, /MNISTミニ/);

  // キャンバスがクラッシュ(白画面)せずノードが描画されている
  await expect
    .poll(() => page.locator('.react-flow__node').count())
    .toBeGreaterThanOrEqual(3);

  // EDITタブ配下の下部タブを巡回(各パネルのマウントで実行時エラーが出ないこと。crashGuardが監視)
  await goToTopTab(page, 'edit');
  for (const name of EDIT_TABS) {
    await page.getByRole('tab', { name, exact: true }).click();
    await expect(page.getByRole('tab', { name, exact: true })).toHaveAttribute(
      'data-state',
      'active',
    );
  }

  // TRAININGタブへ遷移し、学習監視系サブタブ(タイムライン/ログ)を巡回
  await goToTopTab(page, 'training');
  for (const name of TRAINING_TABS) {
    await page.getByRole('tab', { name, exact: true }).click();
    await expect(page.getByRole('tab', { name, exact: true })).toHaveAttribute(
      'data-state',
      'active',
    );
  }

  // EVALUATION/DATASET/CONFIGのプレースホルダもクラッシュせず描画される
  for (const tab of ['evaluation', 'dataset', 'config'] as const) {
    await goToTopTab(page, tab);
  }
});
