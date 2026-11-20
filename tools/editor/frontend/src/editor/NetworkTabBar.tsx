// 複数ネットワークタブバー。EDITのキャンバス上部に薄く重ねるグループタブ。
// タブ切替で networkStore.switchTo → graphStore.ir を差し替える(切替前に現タブへ書き戻し)。
// AppShell(タブ器)は変更不可のため、EDIT内キャンバス(FlowCanvas)上にマウントする。

import { Plus, X } from 'lucide-react';
import { useEffect } from 'react';
import { useGraphStore } from '../store/graphStore';
import { useNetworkStore } from '../store/networkStore';

export function NetworkTabBar() {
  const hasIr = useGraphStore((s) => s.ir !== null);
  const order = useNetworkStore((s) => s.order);
  const active = useNetworkStore((s) => s.active);
  const ensureSeeded = useNetworkStore((s) => s.ensureSeeded);
  const switchTo = useNetworkStore((s) => s.switchTo);
  const addNetwork = useNetworkStore((s) => s.addNetwork);
  const removeNetwork = useNetworkStore((s) => s.removeNetwork);

  // IRがある(モデル読込済み)なら Main タブを自動シードする。
  useEffect(() => {
    if (hasIr) ensureSeeded();
  }, [hasIr, ensureSeeded]);

  // タブが1つ以下(=単一ネットワーク)のときはバーを出さない(既存UXを邪魔しない)。
  if (!hasIr || order.length === 0) return null;

  return (
    <div
      className="pointer-events-auto absolute top-2 left-1/2 z-10 flex -translate-x-1/2 items-center gap-1 rounded-md border bg-[var(--panel-bg)]/95 px-1 py-1 shadow-sm backdrop-blur"
      data-testid="network-tabbar"
    >
      {order.map((name) => (
        <div key={name} className="flex items-center">
          <button
            type="button"
            data-testid={`network-tab-${name}`}
            className={`rounded-sm px-2.5 py-1 text-xs font-medium transition-colors ${
              name === active
                ? 'bg-primary/15 text-primary'
                : 'text-muted-foreground hover:bg-accent'
            }`}
            onClick={() => switchTo(name)}
          >
            {name}
          </button>
          {order.length > 1 && name === active && (
            <button
              type="button"
              className="ml-0.5 rounded-sm p-0.5 text-muted-foreground hover:bg-accent"
              title="このネットワークを削除"
              data-testid={`network-remove-${name}`}
              onClick={() => removeNetwork(name)}
            >
              <X className="size-3" />
            </button>
          )}
        </div>
      ))}
      <button
        type="button"
        className="rounded-sm p-1 text-muted-foreground hover:bg-accent"
        title="ネットワークを追加"
        data-testid="network-add"
        onClick={() => addNetwork()}
      >
        <Plus className="size-3.5" />
      </button>
    </div>
  );
}
