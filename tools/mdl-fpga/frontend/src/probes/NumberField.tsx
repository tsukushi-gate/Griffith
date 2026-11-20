// 数値入力(−/+ ボタン+クリックで全選択のテキスト欄)。
// ネイティブ input[type=number] のスピナーは ReactFlow ノード内で mouseup を
// 取りこぼして増減が止まらなくなる実測不具合があるため使わない。
// − / + は 1 クリックで必ず ±1。テキスト部はクリックで常に全選択され、
// そのまま数字を打てば上書きになる(「1」の後ろに「8」が付いて 18 になる事故防止)。
import { Minus, Plus } from 'lucide-react';
import { useState } from 'react';
import { Button } from '@/components/ui/button';

export function NumberField({
  value,
  min,
  max,
  placeholder,
  title,
  onCommit,
}: {
  value: number | null; // null = 未指定(placeholder 表示。例 'auto')
  min: number;
  max: number;
  placeholder?: string;
  title?: string;
  onCommit: (v: number | null) => void;
}) {
  const [text, setText] = useState<string | null>(null); // null = 非編集中
  const shown = text ?? (value == null ? '' : String(value));
  const clamp = (v: number) => Math.max(min, Math.min(max, v));

  const commitText = (raw: string) => {
    const trimmed = raw.trim();
    if (trimmed === '') {
      onCommit(placeholder ? null : clamp(min));
    } else {
      const n = Number(trimmed);
      if (Number.isFinite(n)) onCommit(clamp(Math.round(n)));
    }
    setText(null);
  };

  const stepBy = (d: number) => {
    setText(null);
    onCommit(value == null ? min : clamp(value + d));
  };

  return (
    <span className="num-field nodrag" title={title}>
      <Button size="icon-xs" variant="ghost" tabIndex={-1} title="-1"
              onClick={() => stepBy(-1)}>
        <Minus />
      </Button>
      <input
        inputMode="numeric"
        value={shown}
        placeholder={placeholder}
        onChange={(e) => setText(e.target.value)}
        onFocus={(e) => e.target.select()}
        onMouseUp={(e) => {
          // クリックのたびに全選択を維持(ブラウザの選択解除を抑止)
          e.preventDefault();
          (e.target as HTMLInputElement).select();
        }}
        onBlur={(e) => commitText(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === 'Enter') (e.target as HTMLInputElement).blur();
          if (e.key === 'ArrowUp') {
            e.preventDefault();
            stepBy(1);
          }
          if (e.key === 'ArrowDown') {
            e.preventDefault();
            stepBy(-1);
          }
        }}
      />
      <Button size="icon-xs" variant="ghost" tabIndex={-1} title="+1"
              onClick={() => stepBy(1)}>
        <Plus />
      </Button>
    </span>
  );
}
