// LODインジケータ(§3)。キャンバス隅に現在のセマンティックズーム段階
// (俯瞰/タイル/ゲート)と次段階への遷移ヒントを常時表示する小バッジ。
// 自動LODのon/offトグル(手動展開ボタンとの共存)と、大規模用deck.gl
// ドリルダウンへの導線を持つ。

import { Layers } from 'lucide-react';
import { useGraphStore } from '../store/graphStore';
import { useViewStore } from '../store/viewStore';
import type { LodStage } from './lod';

const STAGE_LABEL: Record<LodStage, string> = {
  overview: '俯瞰',
  tile: 'タイル',
  gate: 'ゲート',
};

/** 次にズームインすると何が起きるかのヒント。ゲート段階では最深なのでヒントなし。 */
const STAGE_HINT: Record<LodStage, string> = {
  overview: 'ズームインでタイル表示',
  tile: 'ズームインでゲート展開',
  gate: '個別ゲートを表示中',
};

export function LodIndicator() {
  const lodStage = useViewStore((s) => s.lodStage);
  const autoLod = useViewStore((s) => s.autoLod);
  const setAutoLod = useViewStore((s) => s.setAutoLod);
  const setView = useViewStore((s) => s.setView);
  // deck.glドリルダウンは対象lut_layerが要るので、選択中のlut_layerに導線を出す
  const selectedNodeId = useGraphStore((s) => s.selectedNodeId);
  const drillTarget = useGraphStore((s) =>
    selectedNodeId &&
    s.ir?.nodes.some((n) => n.id === selectedNodeId && n.type === 'lut_layer')
      ? selectedNodeId
      : null,
  );

  return (
    <div className="flex flex-col items-start gap-1 rounded-md border bg-[var(--panel-bg)]/90 px-2 py-1.5 text-[10px] shadow-sm">
      <div className="flex items-center gap-1.5">
        <Layers className="size-3 text-muted-foreground" />
        <span className="font-semibold">LOD: {STAGE_LABEL[lodStage]}</span>
      </div>
      <span className="text-muted-foreground">{STAGE_HINT[lodStage]}</span>
      <label className="flex cursor-pointer items-center gap-1 text-muted-foreground">
        <input
          type="checkbox"
          className="size-3"
          checked={autoLod}
          onChange={(e) => setAutoLod(e.target.checked)}
        />
        自動LOD
      </label>
      {drillTarget && (
        <button
          type="button"
          className="nodrag cursor-pointer text-primary hover:underline"
          title="選択中の層を大規模用deck.glビューで表示"
          onClick={() => setView({ kind: 'l0', nodeId: drillTarget })}
        >
          deck.glで大規模表示 →
        </button>
      )}
    </div>
  );
}
