// 表示ビュー状態: グラフ全体 or 単一lut_layerのL0ドリルダウン。
// 誤差逆伝播リプレイのアニメ位相(発光中レイヤ)もここで持つ
// (FlowCanvas/DeckCanvasが参照して層を発光させる)。
// 再生中のGradFlowFrameと勾配強度Map(エッジ太さ・ゲート発光・粒子密度の
// データ源)も保持する(BackpropPlayerが再生開始時に構築、停止でnullへ戻す)。

import { create } from 'zustand';
import type { LodStage } from '../detail/lod';
import type { GradFlowFrame } from './trainingStore';

export type EditorView = { kind: 'graph' } | { kind: 'l0'; nodeId: string };

// EDITタブの表示モード: 'graph'=React Flow編集ビュー、
// 'nnc-card'=NNC風レイヤーカード縦積み(表示専用)。
// カードモードはIR/graphStoreを一切変更しない派生描画で、編集はグラフモードで行う。
// 非永続(EDIT限定のセッション状態。リロードでgraphへ戻す)。
export type DisplayMode = 'graph' | 'nnc-card';

export interface BpAnimState {
  /** 逆伝播リプレイで現在発光中のレイヤ(null=非再生) */
  activeLayer: string | null;
  /** 再生位相(0=損失側から順に増える。表示用) */
  phase: number;
  /** 再生中のgrad_flowトレース(null=非再生。L0着色や粒子生成の元データ) */
  frame: GradFlowFrame | null;
  /** 詳細エッジid `d:${layerId}:${neuron}:${slot}` → 正規化勾配強度[0,1] */
  edgeIntensity: Map<string, number> | null;
  /** ゲートid `${layerId}::g${neuron}` → 正規化勾配強度[0,1] */
  gateIntensity: Map<string, number> | null;
}

export const BP_ANIM_IDLE: BpAnimState = {
  activeLayer: null,
  phase: 0,
  frame: null,
  edgeIntensity: null,
  gateIntensity: null,
};

interface ViewStoreState {
  view: EditorView;
  /** EDITキャンバスの表示モード。非永続。カードモードは表示専用 */
  displayMode: DisplayMode;
  setDisplayMode: (mode: DisplayMode) => void;
  bpAnim: BpAnimState;
  /** セマンティックズームの現在段階(§3。LODインジケータ/タイル表示が購読) */
  lodStage: LodStage;
  /** 見かけサイズ基準の自動LOD遷移のon/off。offでも手動展開/畳むは有効 */
  autoLod: boolean;
  setView: (view: EditorView) => void;
  setBpAnim: (anim: BpAnimState) => void;
  /** 段階が変わった時だけ更新する(不変なら参照維持で無用な再描画を避ける) */
  setLodStage: (stage: LodStage) => void;
  setAutoLod: (on: boolean) => void;
}

export const useViewStore = create<ViewStoreState>((set) => ({
  view: { kind: 'graph' },
  displayMode: 'graph',
  setDisplayMode: (displayMode) => set({ displayMode }),
  bpAnim: BP_ANIM_IDLE,
  lodStage: 'overview',
  autoLod: true,
  setView: (view) => set({ view }),
  setBpAnim: (bpAnim) => set({ bpAnim }),
  setLodStage: (stage) =>
    set((s) => (s.lodStage === stage ? s : { lodStage: stage })),
  setAutoLod: (autoLod) => set({ autoLod }),
}));
