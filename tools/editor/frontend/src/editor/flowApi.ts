// React Flowインスタンスへの外部アクセス窓口。
// コマンドパレット/ショートカットはReactFlowProviderの外にいるため、
// FlowCanvasがマウント時にここへ操作APIを登録する(モジュールシングルトン)。

import type { PaletteNodeType } from '../store/graphStore';

export interface FlowApi {
  /** ビューポート中央に指定型のノードを追加する */
  addAtCenter: (type: PaletteNodeType) => void;
  fitView: () => void;
  autoLayout: () => Promise<void>;
  /** 指定ノードをビューポート中央へ寄せる(選択ギャップTop-Nジャンプ用、§4-1) */
  focusNode: (nodeId: string) => void;
}

export const flowApi: { current: FlowApi | null } = { current: null };
