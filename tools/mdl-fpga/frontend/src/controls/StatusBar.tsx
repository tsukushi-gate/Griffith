// 【廃止】接続・サイクル等の表示は layout/AppHeader.tsx へ統合済み(UI統一)。
// 参照ゼロの旧実装。削除は行わない方針のため据え置き(履歴用)。
import { useSimStore } from '../store/simStore';

export function StatusBar() {
  const meta = useSimStore((s) => s.meta);
  const frame = useSimStore((s) => s.frame);
  const connected = useSimStore((s) => s.connected);
  return (
    <header className="statusbar">
      <h1>mdl-fpga viewer</h1>
      {meta && (
        <span className="dim">
          {meta.name} — in {meta.in_width}b / out {meta.out_width}b
          {meta.k ? ` / ${meta.k} クラス` : ''}
          {meta.latency ? ` / latency ${meta.latency}clk` : ''}
          {meta.mode === 'system' ? ' / SYSTEM' : ''}
        </span>
      )}
      {frame && frame.xz.length > 0 && (
        <span className="xz-badge" title={frame.xz.join(', ')}>
          ⚠ X/Z
        </span>
      )}
      <div className="spacer" />
      <span className="dim">
        cycle <b>{frame?.cycle ?? '-'}</b>
      </span>
      <span className="dim">
        step <b>{frame?.latency_ms ?? '-'}</b> ms
      </span>
      {frame?.steps !== undefined && (
        <span className={frame.changed ? 'conn-ok' : 'conn-ng'}
              title={`watch: ${frame.watch}`}>
          ⏭ +{frame.steps}clk {frame.changed ? '変化' : '上限打切'}
        </span>
      )}
      <span className={connected ? 'conn-ok' : 'conn-ng'}>
        {connected ? '● 接続' : '○ 切断'}
      </span>
    </header>
  );
}
