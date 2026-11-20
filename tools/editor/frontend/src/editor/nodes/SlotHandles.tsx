// 多入力ノード共通のslotハンドル列(縦配置)表示。
// 各行の左端にtargetハンドル(id='s{slot}')を置き、接続済みは塗り・
// 未接続(空き)は破線リングで描き分ける。styles.cssには依存しない。

import { Handle, Position } from '@xyflow/react';
import type { CSSProperties } from 'react';
import type { SlotHandle } from './useInputSlots';

const SLOT_COLOR_COUNT = 6;

const rowStyle: CSSProperties = {
  position: 'relative',
  display: 'flex',
  alignItems: 'center',
  gap: 6,
  height: 18,
  fontSize: 10,
  color: 'var(--muted-foreground)',
};

function handleStyle(slot: number, connected: boolean): CSSProperties {
  const color = `var(--slot-${slot % SLOT_COLOR_COUNT})`;
  return {
    position: 'absolute',
    // ノードのpadding(10px)+ボーダー分だけ左へ出して枠線上に置く
    left: -15,
    top: '50%',
    transform: 'translateY(-50%)',
    width: 8,
    height: 8,
    background: connected ? color : 'var(--node-bg, #fff)',
    border: `2px ${connected ? 'solid' : 'dashed'} ${color}`,
  };
}

interface Props {
  slots: SlotHandle[];
  /** slotごとの補足表示(concatの幅等)。省略時はslot番号のみ */
  slotLabel?: (slot: number) => string;
}

export function SlotHandles({ slots, slotLabel }: Props) {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', marginTop: 2 }}>
      {slots.map(({ slot, connected }) => (
        <div key={slot} style={rowStyle}>
          <Handle
            type="target"
            id={`s${slot}`}
            position={Position.Left}
            style={handleStyle(slot, connected)}
          />
          <span style={{ fontWeight: 600 }}>s{slot}</span>
          <span>{slotLabel ? slotLabel(slot) : connected ? '' : '(空き)'}</span>
        </div>
      ))}
    </div>
  );
}
