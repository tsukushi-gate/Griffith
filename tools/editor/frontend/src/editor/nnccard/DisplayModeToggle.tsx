// EDITキャンバスの表示モード切替トグル。
// グラフ(React Flow編集) ⇔ NNC風レイヤーカード(表示専用) を切り替える。
// グラフモード・カードモード双方の隅に常設し、どちらからでも往復できるようにする。

import { useViewStore, type DisplayMode } from '../../store/viewStore';

const OPTIONS: { mode: DisplayMode; label: string; hint: string }[] = [
  { mode: 'graph', label: 'グラフ', hint: 'React Flowで編集' },
  { mode: 'nnc-card', label: 'カード', hint: 'NNC風カード(表示専用)' },
];

export function DisplayModeToggle() {
  const displayMode = useViewStore((s) => s.displayMode);
  const setDisplayMode = useViewStore((s) => s.setDisplayMode);

  return (
    <div
      className="flex overflow-hidden rounded-md border bg-[var(--panel-bg)] text-xs shadow-sm"
      role="group"
      aria-label="表示モード"
    >
      {OPTIONS.map((o) => (
        <button
          key={o.mode}
          type="button"
          title={o.hint}
          data-testid={`displaymode-${o.mode}`}
          data-active={displayMode === o.mode}
          onClick={() => setDisplayMode(o.mode)}
          className={`px-2 py-1 font-medium ${
            displayMode === o.mode
              ? 'bg-primary text-primary-foreground'
              : 'text-muted-foreground hover:bg-muted'
          }`}
        >
          {o.label}
        </button>
      ))}
    </div>
  );
}
