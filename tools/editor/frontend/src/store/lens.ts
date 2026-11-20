// レンズ切替のオーケストレーション(§2)。
// uiStoreのlens/下部タブと、レンズ依存のオーバーレイ(プレイグラウンド着色)を
// 一括で「初期プリセット」として適用する。切替後にユーザーが個別トグルを
// 変えるのは自由で、強制同期はしない(applyLensは切替の瞬間だけ呼ばれる)。
//
// レンズ→状態マッピング:
//   build : 下部=コード       / プレイグラウンド着色 off(形状診断バッジは常時)
//   train : 下部=学習         / プレイグラウンド着色 off(学習系オーバーレイはbpAnim経由で既存動作)
//   probe : 下部=実行         / プレイグラウンド着色 on
//   ship  : 下部=出荷(W3)    / プレイグラウンド着色 off

import { usePlaygroundStore } from '../playground/playgroundStore';
import { useUIStore, type Lens } from './uiStore';

export function applyLens(lens: Lens): void {
  useUIStore.getState().setLens(lens);
  // Probeレンズだけ入力トグル→ビット値伝播の着色を有効化する
  usePlaygroundStore.getState().setEnabled(lens === 'probe');
}
