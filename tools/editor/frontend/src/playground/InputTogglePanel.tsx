// 入力トグルパネル: 入力ポートの各ビットをボタンで0/1トグルする。
// 幅が上限を超えるポートは操作不可(「大規模入力は未対応」表示)。

import { useGraphStore } from '../store/graphStore';
import { usePlaygroundStore } from './playgroundStore';

/** トグルUIを出す入力ポート幅の上限 */
export const INPUT_TOGGLE_MAX = 32;

export function InputTogglePanel() {
  const ir = useGraphStore((s) => s.ir);
  const inputBits = usePlaygroundStore((s) => s.inputBits);
  const toggleBit = usePlaygroundStore((s) => s.toggleBit);

  if (!ir || ir.inputs.length === 0) {
    return <div className="text-[11px] text-muted-foreground">入力ポートがありません</div>;
  }

  return (
    <div className="flex flex-col gap-2">
      {ir.inputs.map((port) => (
        <div key={port.id} className="flex flex-col gap-1">
          <div className="text-[11px] font-semibold">
            {port.id}
            <span className="ml-1 font-normal text-muted-foreground">
              ({port.width}bit)
            </span>
          </div>
          {port.width > INPUT_TOGGLE_MAX ? (
            <div className="text-[11px] text-muted-foreground">
              大規模入力({port.width}bit)は未対応
            </div>
          ) : (
            <div className="flex flex-wrap gap-1">
              {Array.from({ length: port.width }, (_, i) => {
                const on = (inputBits[port.id]?.[i] ?? 0) === 1;
                return (
                  <button
                    key={i}
                    type="button"
                    title={`x${i} をトグル`}
                    className={`flex h-9 w-8 cursor-pointer flex-col items-center justify-center rounded-md border font-mono leading-none transition-colors ${
                      on
                        ? 'border-primary bg-primary text-primary-foreground'
                        : 'bg-background text-muted-foreground hover:bg-muted'
                    }`}
                    onClick={() => toggleBit(port.id, i)}
                  >
                    <span className="text-sm font-bold">{on ? 1 : 0}</span>
                    <span className="text-[9px] opacity-70">x{i}</span>
                  </button>
                );
              })}
            </div>
          )}
        </div>
      ))}
    </div>
  );
}
