// サンプル毎結果一覧表。軽量自作の仮想スクロール。
// 列=Index/入力(画像サムネイル or ビット列)/label/pred/正誤/確信度/離散化ギャップ。
// ソート(確信度・マージン昇順で「迷いサンプル」上位)を内蔵。フィルタは親が適用済み。
// 行クリックで onRowClick(反例トレース連携=)。

import { useRef, useState } from 'react';
import type { EvalItem } from './api';

export type SortKey = 'index' | 'confidence' | 'margin';

interface ResultTableProps {
  items: EvalItem[];
  kind: 'classification' | 'binary_vector';
  onRowClick?: (item: EvalItem) => void;
}

const ROW_H = 30;
const VIEW_H = 480;
const OVERSCAN = 6;

function bitsLabel(bits: number[] | null): string {
  return bits ? bits.join('') : '';
}

/** ビット列のコンパクト表示。 */
function BitCells({ bits }: { bits: number[] }) {
  return (
    <span className="inline-flex gap-px align-middle">
      {bits.map((b, i) => (
        <span
          key={i}
          className={`inline-block size-3 rounded-[2px] text-center text-[8px] leading-3 ${
            b ? 'bg-primary text-primary-foreground' : 'bg-muted text-muted-foreground/50'
          }`}
        >
          {b}
        </span>
      ))}
    </span>
  );
}

export function ResultTable({ items, kind, onRowClick }: ResultTableProps) {
  const [sortKey, setSortKey] = useState<SortKey>('index');
  const [scrollTop, setScrollTop] = useState(0);
  const scrollRef = useRef<HTMLDivElement | null>(null);

  const sorted = [...items];
  if (sortKey === 'confidence') sorted.sort((a, b) => a.soft_confidence - b.soft_confidence);
  else if (sortKey === 'margin') sorted.sort((a, b) => a.soft_margin - b.soft_margin);
  else sorted.sort((a, b) => a.index - b.index);

  const total = sorted.length;
  const start = Math.max(0, Math.floor(scrollTop / ROW_H) - OVERSCAN);
  const end = Math.min(total, Math.ceil((scrollTop + VIEW_H) / ROW_H) + OVERSCAN);
  const visible = sorted.slice(start, end);

  const cycleSort = (key: SortKey) => setSortKey((cur) => (cur === key ? 'index' : key));

  return (
    <div className="flex flex-col text-xs" data-testid="result-table">
      {/* ヘッダ */}
      <div className="flex items-center gap-2 border-b border-border bg-muted/30 px-2 py-1 font-semibold text-muted-foreground">
        <span className="w-12">#</span>
        <span className="w-24">入力</span>
        <span className="w-16">label</span>
        <span className="w-16">pred</span>
        <span className="w-10">正誤</span>
        <button
          type="button"
          className="w-20 text-left hover:text-foreground"
          onClick={() => cycleSort('confidence')}
          data-testid="sort-confidence"
        >
          確信度{sortKey === 'confidence' ? ' ↑' : ''}
        </button>
        <button
          type="button"
          className="w-24 text-left hover:text-foreground"
          onClick={() => cycleSort('margin')}
          data-testid="sort-margin"
        >
          離散化ギャップ{sortKey === 'margin' ? ' ↑' : ''}
        </button>
      </div>
      {/* 仮想スクロール本体 */}
      <div
        ref={scrollRef}
        className="overflow-y-auto"
        style={{ height: VIEW_H }}
        onScroll={(e) => setScrollTop(e.currentTarget.scrollTop)}
        data-testid="result-scroll"
      >
        <div style={{ height: total * ROW_H, position: 'relative' }}>
          {visible.map((it) => {
            const top = (start + visible.indexOf(it)) * ROW_H;
            const gapWarn = it.hard_soft_mismatch || it.soft_margin < 0.2;
            return (
              <div
                key={it.index}
                className={`absolute left-0 flex w-full items-center gap-2 px-2 hover:bg-accent/40 ${
                  onRowClick ? 'cursor-pointer' : ''
                } ${it.correct ? '' : 'bg-red-500/5'}`}
                style={{ top, height: ROW_H }}
                onClick={() => onRowClick?.(it)}
                data-testid={`result-row-${it.index}`}
                data-correct={it.correct ? '1' : '0'}
              >
                <span className="w-12 font-mono text-muted-foreground">{it.index}</span>
                <span className="w-24 truncate">
                  {it.image ? (
                    <img src={it.image} alt="" className="h-6 w-6 rounded object-contain" />
                  ) : it.input_bits ? (
                    <BitCells bits={it.input_bits} />
                  ) : (
                    <span className="text-muted-foreground/50">—</span>
                  )}
                </span>
                <span className="w-16 font-mono">
                  {kind === 'classification' ? it.label : bitsLabel(it.label_bits)}
                </span>
                <span className="w-16 font-mono">
                  {kind === 'classification' ? it.pred : bitsLabel(it.pred_bits)}
                </span>
                <span className="w-10">
                  {it.correct ? (
                    <span className="text-green-600">○</span>
                  ) : (
                    <span className="font-semibold text-red-600">×</span>
                  )}
                </span>
                <span className="w-20 tabular-nums">{it.soft_confidence.toFixed(3)}</span>
                <span
                  className={`w-24 tabular-nums ${gapWarn ? 'font-semibold text-red-600' : ''}`}
                  title={it.hard_soft_mismatch ? 'hard/soft不一致' : undefined}
                >
                  {it.soft_margin.toFixed(3)}
                  {it.hard_soft_mismatch ? ' ⚠' : ''}
                </span>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
