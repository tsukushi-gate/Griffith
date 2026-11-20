// NNC風レイヤーカードのスタイル単一情報源。
// ノード型 → (色カテゴリ / 頭文字1字 / 主要パラメータ抽出) の対応表。
// カードUI(NncCardNode)・凡例・テストはすべてこの表を参照し、色や頭文字を直書きしない。
//
// NNC配色の踏襲: Input=黒 / 演算層=青 / 活性化系=赤 /
// pooling=茶 / その他=グレー。論理ゲートNNのノード型へ割り当てる。

import type { IRNodeJson, IRNodeType } from '../../ir/schema';

/** カード配色カテゴリ(NNC踏襲) */
export type CardCategory = 'input' | 'compute' | 'activation' | 'pool' | 'other';

/** ポートを含めたカード種別(入出力ポートはIRNodeTypeに無いため拡張) */
export type CardKind = IRNodeType | 'input' | 'output';

export interface CardCategoryStyle {
  /** 頭文字アイコンの背景色 */
  bg: string;
  /** 頭文字アイコンの文字色 */
  fg: string;
  /** カテゴリの日本語ラベル(凡例・ツールチップ用) */
  label: string;
}

/** カテゴリ→配色(インラインstyleで使うhex。テーマ非依存でNNCの見えを固定する) */
export const CARD_CATEGORY_STYLE: Record<CardCategory, CardCategoryStyle> = {
  input: { bg: '#2b2b2b', fg: '#ffffff', label: '入力' },
  compute: { bg: '#2f6fb2', fg: '#ffffff', label: '演算層' },
  activation: { bg: '#c0392b', fg: '#ffffff', label: '活性化系' },
  pool: { bg: '#8a5a2b', fg: '#ffffff', label: 'プーリング' },
  other: { bg: '#6b7280', fg: '#ffffff', label: 'その他' },
};

export interface CardStyle {
  category: CardCategory;
  /** 頭文字1字の大アイコン */
  letter: string;
}

/** カード種別 → (カテゴリ, 頭文字)。色は CARD_CATEGORY_STYLE から引く */
export const CARD_STYLE_BY_TYPE: Record<CardKind, CardStyle> = {
  input: { category: 'input', letter: 'I' },
  output: { category: 'other', letter: 'O' },
  lut_layer: { category: 'compute', letter: 'L' },
  conv2d: { category: 'compute', letter: 'C' },
  group_sum: { category: 'activation', letter: 'Σ' },
  add: { category: 'activation', letter: '+' },
  binarize: { category: 'activation', letter: 'B' },
  or_pool2d: { category: 'pool', letter: 'P' },
  upsample2d: { category: 'pool', letter: 'U' },
  permute: { category: 'other', letter: '⇄' },
  concat: { category: 'other', letter: '⋈' },
  flatten: { category: 'other', letter: 'F' },
  constant: { category: 'other', letter: 'K' },
  blackbox: { category: 'other', letter: '?' },
};

/** 種別のスタイル(未知種別はother扱いでフォールバック) */
export function cardStyleFor(kind: CardKind): CardStyle {
  return CARD_STYLE_BY_TYPE[kind] ?? { category: 'other', letter: '?' };
}

/** 種別の配色 */
export function cardCategoryStyle(kind: CardKind): CardCategoryStyle {
  return CARD_CATEGORY_STYLE[cardStyleFor(kind).category];
}

function num(v: unknown, fallback = '?'): string {
  return typeof v === 'number' ? String(v) : fallback;
}

function pairStr(v: unknown): string {
  if (Array.isArray(v)) return `${v[0]}×${v[1]}`;
  if (typeof v === 'number') return `${v}×${v}`;
  return '?';
}

/**
 * カードに表示する主要パラメータ文字列(NNCの「KernelShape:5,5」等に相当)。
 * ノード型ごとに最も重要な1〜2項目を抽出する。IRノード専用(ポートは別途幅を表示)。
 */
export function mainParamsFor(node: IRNodeJson): string {
  const p = node.params;
  switch (node.type) {
    case 'lut_layer': {
      // train_info.kind(dwn/binarybrain等)があればカードに種別を併記する。
      const kind = node.train_info?.kind;
      const kindTag = typeof kind === 'string' && kind ? ` [${kind}]` : '';
      return `N=${num(p.n_inputs)} out=${num(p.out_width)}${kindTag}`;
    }
    case 'group_sum':
      return `k=${num(p.k)}`;
    case 'binarize':
    case 'flatten':
      return `w=${num(p.width)}`;
    case 'blackbox':
      return `in=${num(p.in_width)} out=${num(p.out_width)}`;
    case 'concat':
      return Array.isArray(p.in_widths) ? p.in_widths.join('+') : '?';
    case 'add':
      return `${String(p.mode ?? '?')} w=${num(p.width)}`;
    case 'conv2d':
      return `k=${pairStr(p.kernel_size)} oc=${num(p.out_channels)}`;
    case 'or_pool2d':
      return `k=${num(p.kernel_size)} s=${num(p.stride)}`;
    case 'constant':
      return `w=${num(p.width)}`;
    default:
      return '';
  }
}
