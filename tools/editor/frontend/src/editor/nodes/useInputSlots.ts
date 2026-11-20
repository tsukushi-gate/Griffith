// 多入力ノード(concat/add)の入力slotハンドル計算フック。
// 接続済みslot+1個の空きハンドルを動的表示する(最低2slot)。
// ハンドル数が変わったらReact Flowへ内部再計測を通知する。

import { useUpdateNodeInternals } from '@xyflow/react';
import { useEffect, useMemo } from 'react';
import { slotFromHandle } from '../../ir/graphToIr';
import { useGraphStore } from '../../store/graphStore';

export interface SlotHandle {
  slot: number;
  connected: boolean;
}

export function useInputSlots(nodeId: string): SlotHandle[] {
  const edges = useGraphStore((s) => s.edges);
  const updateNodeInternals = useUpdateNodeInternals();

  const slots = useMemo(() => {
    const connected = new Set<number>();
    for (const e of edges) {
      if (e.target === nodeId) connected.add(slotFromHandle(e.targetHandle));
    }
    const maxSlot = connected.size > 0 ? Math.max(...connected) : -1;
    // 接続済みslotの次に空きハンドルを1つ出す(欠番slotも表示して修復可能にする)
    const count = Math.max(maxSlot + 2, connected.size + 1, 2);
    return Array.from({ length: count }, (_, s) => ({
      slot: s,
      connected: connected.has(s),
    }));
  }, [edges, nodeId]);

  useEffect(() => {
    updateNodeInternals(nodeId);
  }, [slots.length, nodeId, updateNodeInternals]);

  return slots;
}
