// キャンバス右クリックのクイック追加メニュー。
// カーソル位置にノードを配置する(座標変換は呼び出し元FlowCanvasが行う)。

import { useEffect, useRef } from 'react';
import { Boxes } from 'lucide-react';
import { toast } from 'sonner';
import { PALETTE_ITEMS } from './Palette';
import {
  groupableSelection,
  useGraphStore,
  type PaletteNodeType,
} from '../store/graphStore';

interface Props {
  /** 画面座標(クリック位置)。nullなら非表示 */
  at: { x: number; y: number } | null;
  onSelect: (type: PaletteNodeType, screen: { x: number; y: number }) => void;
  onClose: () => void;
}

export function QuickAddMenu({ at, onSelect, onClose }: Props) {
  const ref = useRef<HTMLDivElement>(null);
  // 選択ノードの変化に追従してグループ化可否を判定する
  const nodes = useGraphStore((s) => s.nodes);
  const createGroup = useGraphStore((s) => s.createGroup);
  const groupableCount = at ? groupableSelection().length : 0;
  void nodes;

  // メニュー外クリック/Escで閉じる
  useEffect(() => {
    if (!at) return;
    const onDown = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) onClose();
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };
    window.addEventListener('mousedown', onDown);
    window.addEventListener('keydown', onKey);
    return () => {
      window.removeEventListener('mousedown', onDown);
      window.removeEventListener('keydown', onKey);
    };
  }, [at, onClose]);

  if (!at) return null;

  return (
    <div
      ref={ref}
      className="fixed z-50 w-52 rounded-md border bg-popover p-1 shadow-md"
      style={{ left: at.x, top: at.y }}
    >
      {groupableCount >= 2 && (
        <>
          <button
            type="button"
            data-slot="quick-add-group"
            className="flex w-full cursor-pointer items-center gap-2 rounded-sm px-2 py-1.5 text-left text-xs hover:bg-accent"
            onClick={() => {
              const gid = createGroup();
              if (gid) toast.success(`${groupableCount}ノードをグループ化しました`);
              onClose();
            }}
          >
            <Boxes className="size-3.5 text-muted-foreground" />
            <span className="font-medium">選択をグループ化</span>
            <span className="truncate text-[10px] text-muted-foreground">
              {groupableCount}ノード
            </span>
          </button>
          <div className="my-1 border-t" />
        </>
      )}
      <div className="px-2 py-1 text-[10px] font-semibold tracking-wider text-muted-foreground uppercase">
        ノードを追加
      </div>
      {PALETTE_ITEMS.map((item) => (
        <button
          key={item.type}
          type="button"
          data-slot="quick-add-item"
          className="flex w-full cursor-pointer items-center gap-2 rounded-sm px-2 py-1.5 text-left text-xs hover:bg-accent"
          onClick={() => {
            onSelect(item.type, at);
            onClose();
          }}
        >
          <item.icon className="size-3.5 text-muted-foreground" />
          <span className="font-medium">{item.label}</span>
          <span className="truncate text-[10px] text-muted-foreground">{item.desc}</span>
        </button>
      ))}
    </div>
  );
}
