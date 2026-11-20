// SchemaFormの未知型(array等)fallback用テキストエリア。
// 現在値をJSON.stringifyして表示し、編集後JSON.parseで書き戻す。
// パース失敗時はエラー表示のみでonChangeは呼ばない(壊れた値を親に伝播させない)。

import { useState } from 'react';

interface Props {
  id: string;
  value: unknown;
  onChange: (v: unknown) => void;
}

export function JsonFallbackField({ id, value, onChange }: Props) {
  const [text, setText] = useState(() => JSON.stringify(value ?? null, null, 1));
  const [error, setError] = useState<string | null>(null);

  return (
    <div className="grid gap-1">
      <textarea
        id={id}
        className="min-h-16 w-full rounded-md border border-input bg-transparent px-3 py-1.5 font-mono text-xs shadow-xs outline-none focus-visible:border-ring focus-visible:ring-[3px] focus-visible:ring-ring/50"
        value={text}
        onChange={(e) => {
          const raw = e.target.value;
          setText(raw);
          try {
            const parsed = JSON.parse(raw) as unknown;
            setError(null);
            onChange(parsed);
          } catch {
            setError('JSONとして解釈できません(未反映)');
          }
        }}
      />
      {error && <p className="text-[11px] text-destructive">{error}</p>}
    </div>
  );
}
