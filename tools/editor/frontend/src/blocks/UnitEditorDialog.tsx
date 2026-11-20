// Unit定義編集ダイアログ。
// unitStore.editing が非nullのとき開く。名前/説明、引数宣言(name/type/default/min/max/options)、
// テンプレート本体(擬似IO unit-in/unit-out + lut_layer)のパラメータ編集(引数バインド)を扱う。
//
// 設計判断: AppShell(タブ器)は変更不可・FlowCanvasは単一IR前提のため、Unit編集は
// 「同一FlowCanvasのサブタブ」ではなく BlockParamDialog/TemplateGallery と同じ
// ダイアログ形態を採る(既存UI構造に自然)。テンプレートは擬似IO付きの構造化フォームで編集する。

import { useMemo } from 'react';
import { toast } from 'sonner';
import { Plus, Trash2 } from 'lucide-react';
import { Button } from '@/components/ui/button';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { ScrollArea } from '@/components/ui/scroll-area';
import { useUnitStore } from '../store/unitStore';
import { saveMyUnit } from './myBlocks';
import type { EditorUnit, UnitArg, UnitTemplateNode } from './unitTypes';
import { isPseudoIo } from './unitTypes';

/** テンプレート内の最初の lut_layer(構造化フォームの編集対象)。 */
function firstLut(unit: EditorUnit): UnitTemplateNode | undefined {
  return unit.template.nodes.find((n) => n.type === 'lut_layer');
}

/** paramsの値がリテラル数値か引数参照か({$arg}なら参照名、そうでなければnull)。 */
function argRefName(v: unknown): string | null {
  if (typeof v === 'object' && v !== null && typeof (v as { $arg?: unknown }).$arg === 'string') {
    return (v as { $arg: string }).$arg;
  }
  return null;
}

const LITERAL = '__literal__';

export function UnitEditorDialog() {
  const editing = useUnitStore((s) => s.editing);
  const patchEditing = useUnitStore((s) => s.patchEditing);
  const setEditingArgs = useUnitStore((s) => s.setEditingArgs);
  const cancelEdit = useUnitStore((s) => s.cancelEdit);
  const saveEditing = useUnitStore((s) => s.saveEditing);

  const lut = useMemo(() => (editing ? firstLut(editing) : undefined), [editing]);

  if (!editing) return null;

  /** テンプレートの lut_layer params を部分更新する。 */
  const patchLutParams = (patch: Record<string, unknown>) => {
    const nodes = editing.template.nodes.map((n) =>
      n.type === 'lut_layer' ? { ...n, params: { ...n.params, ...patch } } : n,
    );
    patchEditing({ template: { ...editing.template, nodes } });
  };

  /** in_width/out_width の値表現を「リテラル or 引数参照」で更新する。 */
  const setWidthField = (key: 'in_width' | 'out_width', mode: string, literal: string) => {
    if (mode === LITERAL) {
      patchLutParams({ [key]: Number(literal) || 0 });
    } else {
      patchLutParams({ [key]: { $arg: mode } });
    }
  };

  const addArg = () => {
    const name = `arg${editing.args.length + 1}`;
    setEditingArgs([...editing.args, { name, type: 'int', default: 8, min: 1 }]);
  };

  const updateArg = (idx: number, patch: Partial<UnitArg>) => {
    setEditingArgs(editing.args.map((a, i) => (i === idx ? { ...a, ...patch } : a)));
  };

  const removeArg = (idx: number) => {
    setEditingArgs(editing.args.filter((_, i) => i !== idx));
  };

  const handleSave = () => {
    const { errors, reexpand } = saveEditing();
    if (errors.length > 0) {
      toast.error(`Unit定義が不正です: ${errors[0]}`);
      return;
    }
    if (reexpand.reexpandedInstanceIds.length > 0) {
      toast.success(`${reexpand.reexpandedInstanceIds.length}個のインスタンスを再展開しました`);
    }
    if (reexpand.skipped.length > 0) {
      toast.warning(
        `手編集済みの ${reexpand.skipped.length}個は再展開をスキップしました`,
      );
    }
    toast.success(`Unit「${editing.name}」を保存しました`);
  };

  const handleSaveToPalette = () => {
    saveMyUnit(editing);
    toast.success('マイUnitへ保存しました(パレット/ギャラリーに表示されます)');
  };

  const pseudoIn = editing.template.nodes.filter((n) => n.type === 'unit-in');
  const pseudoOut = editing.template.nodes.filter((n) => n.type === 'unit-out');
  const realCount = editing.template.nodes.filter((n) => !isPseudoIo(n)).length;

  const outWidthRef = lut ? argRefName(lut.params.out_width) : null;
  const inWidthRef = lut ? argRefName(lut.params.in_width) : null;

  return (
    <Dialog open onOpenChange={(open) => !open && cancelEdit()}>
      <DialogContent className="max-h-[85vh] max-w-lg overflow-hidden p-0" data-testid="unit-editor-dialog">
        <DialogHeader className="border-b px-5 py-4">
          <DialogTitle>Unit定義の編集</DialogTitle>
          <DialogDescription>
            引数付き再利用サブネットを定義します。IRへは常に展開済みで保存されます。
          </DialogDescription>
        </DialogHeader>

        <ScrollArea className="max-h-[calc(85vh-140px)]">
          <div className="flex flex-col gap-4 px-5 py-4">
            {/* 基本情報 */}
            <div className="grid grid-cols-2 gap-2">
              <div className="flex flex-col gap-1">
                <Label className="text-xs">名前</Label>
                <Input
                  data-testid="unit-name-input"
                  value={editing.name}
                  onChange={(e) => patchEditing({ name: e.target.value })}
                />
              </div>
              <div className="flex flex-col gap-1">
                <Label className="text-xs">説明(任意)</Label>
                <Input
                  value={editing.description ?? ''}
                  onChange={(e) => patchEditing({ description: e.target.value })}
                />
              </div>
            </div>

            {/* 引数宣言パネル */}
            <section className="grid gap-2">
              <div className="flex items-center justify-between">
                <Label className="text-[11px] font-semibold tracking-wide text-muted-foreground uppercase">
                  パラメータ宣言
                </Label>
                <Button variant="outline" size="sm" onClick={addArg} data-testid="unit-add-arg">
                  <Plus className="size-3.5" />
                  引数を追加
                </Button>
              </div>
              {editing.args.length === 0 && (
                <div className="hint-text">引数なし(固定サブネット)。</div>
              )}
              {editing.args.map((a, idx) => (
                <div key={idx} className="grid gap-2 rounded-md border p-2.5">
                  <div className="flex items-center gap-2">
                    <Input
                      className="h-7"
                      placeholder="引数名"
                      value={a.name}
                      onChange={(e) => updateArg(idx, { name: e.target.value })}
                    />
                    <select
                      className="h-7 rounded-md border bg-transparent px-2 text-xs"
                      value={a.type}
                      onChange={(e) =>
                        updateArg(idx, {
                          type: e.target.value as UnitArg['type'],
                          ...(e.target.value === 'enum'
                            ? { options: a.options ?? ['a', 'b'], default: (a.options ?? ['a'])[0] }
                            : { default: 8 }),
                        })
                      }
                    >
                      <option value="int">int</option>
                      <option value="enum">enum</option>
                    </select>
                    <Button
                      variant="ghost"
                      size="icon-xs"
                      title="引数を削除"
                      onClick={() => removeArg(idx)}
                    >
                      <Trash2 className="size-3.5" />
                    </Button>
                  </div>
                  {a.type === 'int' ? (
                    <div className="grid grid-cols-3 gap-2">
                      <LabeledInput
                        label="既定値"
                        value={String(a.default)}
                        onChange={(v) => updateArg(idx, { default: Number(v) || 0 })}
                      />
                      <LabeledInput
                        label="min"
                        value={a.min !== undefined ? String(a.min) : ''}
                        onChange={(v) =>
                          updateArg(idx, { min: v === '' ? undefined : Number(v) })
                        }
                      />
                      <LabeledInput
                        label="max"
                        value={a.max !== undefined ? String(a.max) : ''}
                        onChange={(v) =>
                          updateArg(idx, { max: v === '' ? undefined : Number(v) })
                        }
                      />
                    </div>
                  ) : (
                    <div className="grid gap-1">
                      <Label className="text-[10px] text-muted-foreground">
                        選択肢(カンマ区切り)
                      </Label>
                      <Input
                        className="h-7"
                        value={(a.options ?? []).join(',')}
                        onChange={(e) => {
                          const options = e.target.value
                            .split(',')
                            .map((s) => s.trim())
                            .filter(Boolean);
                          updateArg(idx, {
                            options,
                            default: options.includes(String(a.default))
                              ? a.default
                              : (options[0] ?? ''),
                          });
                        }}
                      />
                    </div>
                  )}
                </div>
              ))}
            </section>

            {/* テンプレート本体(擬似IO + lut_layer) */}
            <section className="grid gap-2">
              <Label className="text-[11px] font-semibold tracking-wide text-muted-foreground uppercase">
                テンプレート本体
              </Label>
              <div className="hint-text">
                入力ポート {pseudoIn.length}個 / 出力ポート {pseudoOut.length}個 / ノード{' '}
                {realCount}個(擬似IOは展開時に境界ハンドルになります)
              </div>
              {lut && (
                <div className="grid gap-2 rounded-md border p-2.5">
                  <div className="text-xs font-semibold">lut_layer</div>
                  <WidthField
                    label="in_width"
                    args={editing.args}
                    refName={inWidthRef}
                    literal={inWidthRef ? '' : String(lut.params.in_width ?? '')}
                    onChange={(mode, literal) => setWidthField('in_width', mode, literal)}
                  />
                  <WidthField
                    label="out_width"
                    args={editing.args}
                    refName={outWidthRef}
                    literal={outWidthRef ? '' : String(lut.params.out_width ?? '')}
                    onChange={(mode, literal) => setWidthField('out_width', mode, literal)}
                  />
                  <LabeledInput
                    label="n_inputs"
                    value={String(lut.params.n_inputs ?? 2)}
                    onChange={(v) => patchLutParams({ n_inputs: Number(v) || 2 })}
                  />
                </div>
              )}
            </section>
          </div>
        </ScrollArea>

        <DialogFooter className="border-t px-5 py-3">
          <Button variant="ghost" onClick={handleSaveToPalette}>
            マイUnitへ保存
          </Button>
          <Button variant="outline" onClick={cancelEdit}>
            キャンセル
          </Button>
          <Button onClick={handleSave} data-testid="unit-save">
            保存
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}

function LabeledInput({
  label,
  value,
  onChange,
}: {
  label: string;
  value: string;
  onChange: (v: string) => void;
}) {
  return (
    <div className="flex flex-col gap-1">
      <Label className="text-[10px] text-muted-foreground">{label}</Label>
      <Input className="h-7" value={value} onChange={(e) => onChange(e.target.value)} />
    </div>
  );
}

/** 幅フィールド: リテラル数値 or 引数参照({$arg})を選ぶ。 */
function WidthField({
  label,
  args,
  refName,
  literal,
  onChange,
}: {
  label: string;
  args: UnitArg[];
  refName: string | null;
  literal: string;
  onChange: (mode: string, literal: string) => void;
}) {
  const mode = refName ?? LITERAL;
  return (
    <div className="flex items-end gap-2">
      <div className="flex flex-1 flex-col gap-1">
        <Label className="text-[10px] text-muted-foreground">{label}</Label>
        <select
          className="h-7 rounded-md border bg-transparent px-2 text-xs"
          value={mode}
          onChange={(e) => onChange(e.target.value, literal)}
          data-testid={`unit-${label}-bind`}
        >
          <option value={LITERAL}>リテラル</option>
          {args.map((a) => (
            <option key={a.name} value={a.name}>
              引数: {a.name}
            </option>
          ))}
        </select>
      </div>
      {mode === LITERAL && (
        <Input
          className="h-7 w-24"
          value={literal}
          onChange={(e) => onChange(LITERAL, e.target.value)}
        />
      )}
    </div>
  );
}
