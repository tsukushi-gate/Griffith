// ピン留めプローブ(画面固定)。キャンバスの pan/zoom に追従しない。
import { useViewerStore } from '../store/viewerStore';
import { ProbeCard } from './ProbeCard';

export function PinnedProbes() {
  // 注意: セレクタ内で filter しない(毎回新配列 → useSyncExternalStore が
  // 無限再レンダリングに陥る。実測でアプリ全体が固まった)
  const probes = useViewerStore((s) => s.probes);
  const pinned = probes.filter((p) => p.pinned);
  if (pinned.length === 0) return null;
  return (
    <div className="pinned-probes">
      {pinned.map((p) => (
        <ProbeCard key={p.id} probeId={p.id} />
      ))}
    </div>
  );
}
