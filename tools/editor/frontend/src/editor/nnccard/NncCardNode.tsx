// NNC風レイヤーカード1枚。表示専用。
// 構成: [頭文字1字の大アイコン(型別色)] [レイヤー名+主要パラメータ]
//        [出力形状注記+相対サイズバー] [コメント付箋] 。
// 多入力(concat/add/fanout)は左脇の合流インジケータで表す。
// クリックで選択、ダブルクリックでゲート詳細(L0)へドリルダウンする。

import { cardCategoryStyle, cardStyleFor, type CardKind } from './CARD_STYLE_BY_TYPE';

export interface NncCardNodeProps {
  id: string;
  kind: CardKind;
  /** レイヤー名(表示ラベル) */
  name: string;
  /** 主要パラメータ文字列(空可) */
  params: string;
  /** 入力幅(ポート/導出。null=不定) */
  inWidth: number | null;
  /** 出力幅(導出。null=不定) */
  outWidth: number | null;
  /** サイズバーの相対長算出に使う全カード中の最大出力幅 */
  maxWidth: number;
  /** 形状不整合(赤バッジ) */
  hasError: boolean;
  /** 入力エッジ本数(≥2で合流インジケータ) */
  inDegree: number;
  /** レイヤーコメント(あれば付箋アイコン) */
  comment?: string;
  /** ゲート詳細へドリルダウン可能か(ポートは不可) */
  drillable: boolean;
  selected: boolean;
  onSelect: () => void;
  onDrill: () => void;
}

export function NncCardNode({
  id,
  kind,
  name,
  params,
  inWidth,
  outWidth,
  maxWidth,
  hasError,
  inDegree,
  comment,
  drillable,
  selected,
  onSelect,
  onDrill,
}: NncCardNodeProps) {
  const { letter } = cardStyleFor(kind);
  const { bg, fg } = cardCategoryStyle(kind);
  // 相対サイズバー(出力幅 / 全カード最大幅)。最小2%は視認用の下限
  const barPct =
    outWidth && maxWidth > 0 ? Math.max(2, (outWidth / maxWidth) * 100) : 0;
  const shape =
    inWidth !== null || outWidth !== null
      ? `${inWidth ?? '?'} → ${outWidth ?? '?'}`
      : '';

  return (
    <div
      data-testid={`nnc-card-${id}`}
      data-kind={kind}
      data-drill={drillable}
      onClick={onSelect}
      onDoubleClick={() => drillable && onDrill()}
      className={`flex cursor-pointer items-stretch gap-2 rounded-md border bg-[var(--panel-bg)] px-2 py-1 shadow-sm ${
        selected ? 'ring-2 ring-primary' : ''
      } ${hasError ? 'border-[var(--edge-error)]' : ''}`}
      style={{ minHeight: 48 }}
    >
      {/* 合流インジケータ(多入力): 入力本数を左脇に表示 */}
      {inDegree >= 2 && (
        <div
          className="flex w-4 shrink-0 items-center justify-center rounded bg-muted text-[10px] font-semibold text-muted-foreground"
          title={`${inDegree}入力の合流`}
          data-testid={`nnc-merge-${id}`}
        >
          {inDegree}
        </div>
      )}
      {/* 頭文字1字の大アイコン(型別色) */}
      <div
        className="flex w-10 shrink-0 items-center justify-center rounded text-lg font-bold"
        style={{ backgroundColor: bg, color: fg }}
        aria-hidden
      >
        {letter}
      </div>
      {/* レイヤー名+主要パラメータ */}
      <div className="flex min-w-0 flex-1 flex-col justify-center">
        <div className="flex items-center gap-1 truncate text-xs font-semibold">
          <span className="truncate">{name}</span>
          {comment && (
            <span title={comment} data-testid={`nnc-comment-${id}`}>
              🗒
            </span>
          )}
        </div>
        {params && <div className="truncate text-[10px] text-muted-foreground">{params}</div>}
      </div>
      {/* 出力形状注記+相対サイズバー */}
      <div className="flex w-24 shrink-0 flex-col justify-center gap-0.5">
        <div className="flex items-center justify-end gap-1 text-[10px] tabular-nums text-muted-foreground">
          {hasError && (
            <span
              className="rounded bg-[var(--edge-error)] px-1 text-[9px] font-bold text-white"
              data-testid={`nnc-error-${id}`}
              title="形状不整合"
            >
              !
            </span>
          )}
          <span>{shape}</span>
        </div>
        <div className="h-1.5 w-full overflow-hidden rounded-full bg-muted">
          <div
            className="h-full rounded-full"
            style={{
              width: `${barPct}%`,
              backgroundColor: hasError ? 'var(--edge-error)' : bg,
            }}
          />
        </div>
      </div>
    </div>
  );
}
