// 入力ポートのビット展開ノード。
// 幅≦INPUT_BITS_MAXの入力ポートが詳細展開層へ直結する場合に、
// ビットごとのソースハンドル(b0..b{w-1})を持つ縦長ノードとして描画する。
// 子ノードは作らず、1ノード+複数ハンドルで実現する(合成コスト最小化)。
// プレイグラウンド実行時は各ビットの現在値を表示し、クリックでトグルする。

import { Handle, Position, type NodeProps } from '@xyflow/react';
import { usePlaygroundStore } from '../../playground/playgroundStore';
import { BITS_HEADER, BITS_ROW } from '../detailConfig';
import type { InputBitsFlowNode } from '../irToDetailGraph';

export function InputBitsNode({ data, selected }: NodeProps<InputBitsFlowNode>) {
  const pgEnabled = usePlaygroundStore((s) => s.enabled);
  const bits = usePlaygroundStore((s) =>
    s.enabled ? s.inputBits[data.label] : undefined,
  );
  const toggleBit = usePlaygroundStore((s) => s.toggleBit);

  return (
    <div
      className={`h-full w-full rounded-md border bg-card text-[10px] shadow-xs ${
        selected ? 'border-primary ring-1 ring-primary' : ''
      }`}
      style={{ borderLeft: '3px solid var(--slot-3)' }}
    >
      <div
        className="flex items-center border-b px-2 font-semibold"
        style={{ height: BITS_HEADER }}
      >
        {data.label}
      </div>
      {Array.from({ length: data.width }, (_, i) => {
        const top = BITS_HEADER + i * BITS_ROW + BITS_ROW / 2;
        const on = (bits?.[i] ?? 0) === 1;
        return (
          <div key={i}>
            {/* 実行時のみ: 現在値ボタン(クリックでトグル) */}
            {pgEnabled && (
              <button
                type="button"
                title={`x${i} をトグル`}
                className={`nodrag absolute left-2 flex h-[14px] w-[14px] cursor-pointer items-center justify-center rounded-[3px] border font-mono text-[9px] leading-none ${
                  on
                    ? 'border-primary bg-primary font-bold text-primary-foreground'
                    : 'bg-background text-muted-foreground'
                }`}
                style={{ top: top - 7 }}
                onClick={() => toggleBit(data.label, i)}
              >
                {on ? 1 : 0}
              </button>
            )}
            <div
              className="absolute right-2 font-mono text-muted-foreground"
              style={{ top: top - 7 }}
            >
              x{i}
            </div>
            <Handle
              id={`b${i}`}
              type="source"
              position={Position.Right}
              isConnectable={false}
              style={{
                top,
                width: 6,
                height: 6,
                background: 'var(--slot-3)',
                border: 'none',
              }}
            />
          </div>
        );
      })}
    </div>
  );
}
