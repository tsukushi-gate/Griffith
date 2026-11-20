// プローブ → 破線エッジ(回路のように見せるが、回路には存在しない表示専用線)。
import type { Edge, Node } from '@xyflow/react';
import type { ProbeSpec } from '../api/types';
import { sliceLabel } from '../probes/parseSlice';

/** プローブ対象 → 接続元ノード id(y/y_comb は出力ポートに繋ぐ)。 */
export function probeSourceId(target: string): string {
  return target === 'y_comb' ? 'y' : target;
}

export function probeNodes(probes: ProbeSpec[]): Node[] {
  return probes
    .filter((p) => !p.pinned)
    .map((p) => ({
      id: `probe:${p.id}`,
      type: 'probe',
      position: p.pos,
      zIndex: 10,
      data: { probeId: p.id },
    }));
}

export function probeEdges(probes: ProbeSpec[], knownIds: Set<string>): Edge[] {
  return probes
    .filter((p) => !p.pinned && knownIds.has(probeSourceId(p.target)))
    .map((p) => ({
      id: `probe-edge:${p.id}`,
      source: probeSourceId(p.target),
      target: `probe:${p.id}`,
      animated: true,
      style: {
        stroke: 'var(--probe)',
        strokeWidth: 1.2,
        strokeDasharray: '6 4',
      },
      label: sliceLabel(p.slice) || undefined,
    }));
}
