// ノードパレット(左サイドバー): キャンバスへのドラッグ&ドロップ配置元。
// dataTransferにノード型を載せ、FlowCanvasのonDropで受ける。
// 複合ブロックセクション(ドロップ時にパラメータダイアログ→サブグラフ展開)。

import { Binary, Boxes, Layers, Package, Pencil, Plus, SigmaSquare, ToggleLeft, X } from 'lucide-react';
import type { LucideIcon } from 'lucide-react';
import { useEffect, useState } from 'react';
import { BlockParamDialog } from '../blocks/BlockParamDialog';
import { BLOCK_DND_MIME } from '../blocks/blockDialogStore';
import {
  MYBLOCKS_CHANGED_EVENT,
  MYBLOCK_DND_MIME,
  MYUNITS_CHANGED_EVENT,
  deleteMyBlock,
  deleteMyUnit,
  listMyBlocks,
  listMyUnits,
  type MyBlock,
} from '../blocks/myBlocks';
import { ADD_ICON, BLOCK_DEFS, CONCAT_ICON, CONV_DEFS, PARTS_DEFS } from '../blocks/templates';
import { UnitEditorDialog } from '../blocks/UnitEditorDialog';
import { UnitInstanceDialog } from '../blocks/UnitInstanceDialog';
import { UNIT_DND_MIME } from '../blocks/unitDialogStore';
import type { EditorUnit } from '../blocks/unitTypes';
import { useGraphStore, type PaletteNodeType } from '../store/graphStore';
import { useUnitStore } from '../store/unitStore';

export const PALETTE_DND_MIME = 'application/x-magicdelta-node';

export interface PaletteItem {
  type: PaletteNodeType;
  label: string;
  desc: string;
  icon: LucideIcon;
  color: string; // 左ボーダー色(ノード種別色と揃える)
}

// QuickAddMenu(右クリック追加)と共有する
// eslint-disable-next-line react-refresh/only-export-components -- QuickAddMenu/FlowCanvasと共有する定数の意図的な同居
export const PALETTE_ITEMS: PaletteItem[] = [
  { type: 'lut_layer', label: 'LutLayer', desc: 'N入力LUT層', icon: Layers, color: 'var(--slot-0)' },
  { type: 'group_sum', label: 'GroupSum', desc: 'クラス集計(k分割和)', icon: SigmaSquare, color: 'var(--slot-2)' },
  { type: 'binarize', label: 'Binarize', desc: '二値化(HardTanh STE)', icon: ToggleLeft, color: 'var(--edge-default)' },
  { type: 'flatten', label: 'Flatten', desc: '形状を平坦幅へ変換', icon: Layers, color: 'var(--slot-4)' },
  { type: 'constant', label: 'Constant', desc: '固定ビット列(状態/位置)', icon: Binary, color: 'var(--slot-5)' },
  { type: 'concat', label: 'Concat', desc: 'ビット連結(多入力)', icon: CONCAT_ICON, color: 'var(--slot-1)' },
  { type: 'add', label: 'Add', desc: 'マージ(or/xor/sum、多入力)', icon: ADD_ICON, color: 'var(--slot-2)' },
];

export function Palette() {
  const ir = useGraphStore((s) => s.ir);
  // マイブロック: localStorage保存テンプレート。保存/削除イベントで再取得する
  const [myBlocks, setMyBlocks] = useState<MyBlock[]>([]);
  useEffect(() => {
    const refresh = () => setMyBlocks(listMyBlocks());
    refresh();
    window.addEventListener(MYBLOCKS_CHANGED_EVENT, refresh);
    return () => window.removeEventListener(MYBLOCKS_CHANGED_EVENT, refresh);
  }, []);

  // Unit: プロジェクト同梱Unit(アクティブIR)+ localStorageマイUnit。
  const projectUnits = useUnitStore((s) => s.units);
  const startNewUnit = useUnitStore((s) => s.startNew);
  const startEditUnit = useUnitStore((s) => s.startEdit);
  const [myUnits, setMyUnits] = useState<EditorUnit[]>([]);
  useEffect(() => {
    const refresh = () => setMyUnits(listMyUnits());
    refresh();
    window.addEventListener(MYUNITS_CHANGED_EVENT, refresh);
    return () => window.removeEventListener(MYUNITS_CHANGED_EVENT, refresh);
  }, []);
  // 表示用: プロジェクト同梱を優先し、localStorage分をidで重複排除してマージ。
  const unitList: EditorUnit[] = [
    ...projectUnits,
    ...myUnits.filter((u) => !projectUnits.some((p) => p.id === u.id)),
  ];

  const itemClass = (enabled: boolean) =>
    `flex items-center gap-2.5 rounded-md border bg-card px-2.5 py-2 shadow-xs select-none ${
      enabled ? 'cursor-grab hover:border-primary/40 hover:shadow-sm' : 'cursor-not-allowed opacity-40'
    }`;

  return (
    <div className="flex flex-col gap-2 p-3">
      <h2 className="text-[11px] font-semibold tracking-widest text-muted-foreground uppercase">
        パレット
      </h2>
      {!ir && <div className="hint-text">モデル読込後にドラッグで追加できます</div>}
      <div className="flex flex-col gap-1.5">
        {PALETTE_ITEMS.map((item) => (
          <div
            key={item.type}
            className={itemClass(!!ir)}
            style={{ borderLeft: `3px solid ${item.color}` }}
            draggable={!!ir}
            onDragStart={(e) => {
              e.dataTransfer.setData(PALETTE_DND_MIME, item.type);
              e.dataTransfer.effectAllowed = 'move';
            }}
          >
            <item.icon className="size-4 shrink-0 text-muted-foreground" />
            <div className="min-w-0">
              <div className="text-xs font-semibold">{item.label}</div>
              <div className="truncate text-[11px] text-muted-foreground">{item.desc}</div>
            </div>
          </div>
        ))}
        {/* Conv2d/OrPool2d/縮約ツリーを実カード化(ドロップでパラメータダイアログ) */}
        {CONV_DEFS.map((def) => (
          <div
            key={def.kind}
            data-testid={`palette-${def.kind}`}
            className={itemClass(!!ir)}
            style={{ borderLeft: `3px solid ${def.color}` }}
            title="ドロップするとパラメータ入力ダイアログが開きます(indices自動生成)"
            draggable={!!ir}
            onDragStart={(e) => {
              e.dataTransfer.setData(BLOCK_DND_MIME, def.kind);
              e.dataTransfer.effectAllowed = 'move';
            }}
          >
            <def.icon className="size-4 shrink-0 text-muted-foreground" />
            <div className="min-w-0">
              <div className="text-xs font-semibold">{def.label}</div>
              <div className="truncate text-[11px] text-muted-foreground">{def.desc}</div>
            </div>
          </div>
        ))}
      </div>

      <h2 className="mt-2 text-[11px] font-semibold tracking-widest text-muted-foreground uppercase">
        ブロック
      </h2>
      <div className="flex flex-col gap-1.5">
        {BLOCK_DEFS.map((def) => (
          <div
            key={def.kind}
            className={itemClass(!!ir)}
            style={{ borderLeft: `3px solid ${def.color}` }}
            title="ドロップするとパラメータ入力ダイアログが開きます"
            draggable={!!ir}
            onDragStart={(e) => {
              e.dataTransfer.setData(BLOCK_DND_MIME, def.kind);
              e.dataTransfer.effectAllowed = 'move';
            }}
          >
            <def.icon className="size-4 shrink-0 text-muted-foreground" />
            <div className="min-w-0">
              <div className="text-xs font-semibold">{def.label}</div>
              <div className="truncate text-[11px] text-muted-foreground">{def.desc}</div>
            </div>
          </div>
        ))}
      </div>

      {/* モデル部品(Attention/Transformer/PE/Embedding/DWN/UpSample) */}
      <h2 className="mt-2 text-[11px] font-semibold tracking-widest text-muted-foreground uppercase">
        モデル部品
      </h2>
      <div className="flex flex-col gap-1.5">
        {PARTS_DEFS.map((def) => (
          <div
            key={def.kind}
            data-testid={`palette-${def.kind}`}
            className={itemClass(!!ir)}
            style={{ borderLeft: `3px solid ${def.color}` }}
            title="ドロップするとパラメータ入力ダイアログが開きます"
            draggable={!!ir}
            onDragStart={(e) => {
              e.dataTransfer.setData(BLOCK_DND_MIME, def.kind);
              e.dataTransfer.effectAllowed = 'move';
            }}
          >
            <def.icon className="size-4 shrink-0 text-muted-foreground" />
            <div className="min-w-0">
              <div className="text-xs font-semibold">{def.label}</div>
              <div className="truncate text-[11px] text-muted-foreground">{def.desc}</div>
            </div>
          </div>
        ))}
      </div>

      <h2 className="mt-2 text-[11px] font-semibold tracking-widest text-muted-foreground uppercase">
        マイブロック
      </h2>
      {myBlocks.length === 0 && (
        <div className="hint-text">
          グループを「テンプレート保存」するとここに追加されます
        </div>
      )}
      <div className="flex flex-col gap-1.5">
        {myBlocks.map((block) => (
          <div
            key={block.id}
            className={itemClass(!!ir)}
            style={{ borderLeft: '3px solid var(--slot-3)' }}
            title="ドラッグでキャンバスへ配置"
            draggable={!!ir}
            onDragStart={(e) => {
              e.dataTransfer.setData(MYBLOCK_DND_MIME, block.id);
              e.dataTransfer.effectAllowed = 'move';
            }}
          >
            <Boxes className="size-4 shrink-0 text-muted-foreground" />
            <div className="min-w-0 flex-1">
              <div className="truncate text-xs font-semibold">{block.name}</div>
              <div className="truncate text-[11px] text-muted-foreground">
                {block.nodes.length}ノード
              </div>
            </div>
            <button
              type="button"
              className="shrink-0 rounded-sm p-0.5 text-muted-foreground hover:bg-accent"
              title="削除"
              onClick={() => deleteMyBlock(block.id)}
            >
              <X className="size-3.5" />
            </button>
          </div>
        ))}
      </div>

      {/* マイUnit(引数付き再利用サブネット) */}
      <div className="mt-2 flex items-center justify-between">
        <h2 className="text-[11px] font-semibold tracking-widest text-muted-foreground uppercase">
          マイUnit
        </h2>
        <button
          type="button"
          className="flex items-center gap-1 rounded-sm px-1.5 py-0.5 text-[11px] text-muted-foreground hover:bg-accent"
          title="新しいUnitを定義"
          data-testid="unit-new"
          onClick={() => startNewUnit()}
        >
          <Plus className="size-3.5" />
          新規
        </button>
      </div>
      {unitList.length === 0 && (
        <div className="hint-text">「新規」でUnitを定義するとここに追加されます</div>
      )}
      <div className="flex flex-col gap-1.5">
        {unitList.map((unit) => (
          <div
            key={unit.id}
            data-testid={`palette-unit-${unit.id}`}
            className={itemClass(!!ir)}
            style={{ borderLeft: '3px solid var(--slot-5)' }}
            title="ドラッグでキャンバスへ配置(引数ダイアログが開きます)"
            draggable={!!ir}
            onDragStart={(e) => {
              e.dataTransfer.setData(UNIT_DND_MIME, unit.id);
              e.dataTransfer.effectAllowed = 'move';
            }}
          >
            <Package className="size-4 shrink-0 text-muted-foreground" />
            <div className="min-w-0 flex-1">
              <div className="truncate text-xs font-semibold">{unit.name}</div>
              <div className="truncate text-[11px] text-muted-foreground">
                引数 {unit.args.length}個
              </div>
            </div>
            <button
              type="button"
              className="shrink-0 rounded-sm p-0.5 text-muted-foreground hover:bg-accent"
              title="編集"
              data-testid={`unit-edit-${unit.id}`}
              onClick={() => {
                if (projectUnits.some((p) => p.id === unit.id)) {
                  startEditUnit(unit.id);
                } else {
                  // localStorage専用Unitは下書きへ直接読み込んで編集する
                  useUnitStore.setState({ editing: structuredClone(unit) });
                }
              }}
            >
              <Pencil className="size-3.5" />
            </button>
            <button
              type="button"
              className="shrink-0 rounded-sm p-0.5 text-muted-foreground hover:bg-accent"
              title="マイUnitから削除"
              onClick={() => deleteMyUnit(unit.id)}
            >
              <X className="size-3.5" />
            </button>
          </div>
        ))}
      </div>

      {/* ブロックドロップ時のパラメータダイアログ(blockDialogStore経由で開く) */}
      <BlockParamDialog />
      {/* Unit定義編集(unitStore.editing)・Unitインスタンス化(unitDialogStore.pending) */}
      <UnitEditorDialog />
      <UnitInstanceDialog />
    </div>
  );
}
