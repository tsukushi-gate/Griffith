// 詳細ゲートビューのしきい値定数

/** この幅以下のlut_layerは読込時に自動で詳細展開する(in/out両方が条件) */
export const DETAIL_AUTO_MAX = 64;

/** 手動展開を許可する上限。超える層はdeck.glドリルダウンへ誘導 */
export const DETAIL_MANUAL_MAX = 128;

/** 入力ポートのビット展開を行う幅の上限 */
export const INPUT_BITS_MAX = 64;

/** 合成後の総ノード数上限(超過した層は展開を拒否しトースト通知)。
    onlyRenderVisibleElements+ズームLOD+層別キャッシュ導入により
    300→1000へ引き上げ(合成960ゲートで数ms、テストで<200msを担保) */
export const DETAIL_TOTAL_MAX = 1000;

/** このズーム未満ではGateNodeを枠+ゲート名だけの軽量表示に切り替える */
export const DETAIL_LOD_ZOOM = 0.4;

/** ゲートカードのグリッドセル間隔(px)。LutLayerGroupNodeの枠計算と共有 */
export const GATE_W = 122;
export const GATE_H = 104;
export const GATE_GAP = 10;
export const GROUP_HEADER = 34;
export const GROUP_PAD = 12;

/** ゲートカード自体の実寸(px)。GateNode.tsx の w-[114px] h-[96px] と一致させること。
    React Flowの可視判定(nodeHasDimensions)を即時に満たすためノード直下のwidth/heightに使う */
export const GATE_CARD_W = 114;
export const GATE_CARD_H = 96;

/** InputBitsノードの寸法 */
export const BITS_HEADER = 30;
export const BITS_ROW = 18;
export const BITS_W = 96;
