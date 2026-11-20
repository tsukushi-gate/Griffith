// Unitインスタンス化の引数ダイアログ。
// パレットのUnitをドロップ(unitDialogStore.pending)で開き、宣言された引数
// (int/enum)を入力して graphStore.addUnitInstance で展開配置する。

import { useState } from 'react';
import { toast } from 'sonner';
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
import { useGraphStore } from '../store/graphStore';
import { useUnitStore } from '../store/unitStore';
import { useUnitDialogStore } from './unitDialogStore';
import type { UnitArg } from './unitTypes';

/** 引数の入力値(文字列)を型に応じてパースする。不正なら null。 */
function parseArgValue(arg: UnitArg, raw: string): number | string | null {
  if (arg.type === 'enum') {
    return arg.options?.includes(raw) ? raw : null;
  }
  const n = Number(raw);
  if (!Number.isInteger(n)) return null;
  if (arg.min !== undefined && n < arg.min) return null;
  if (arg.max !== undefined && n > arg.max) return null;
  return n;
}

export function UnitInstanceDialog() {
  const pending = useUnitDialogStore((s) => s.pending);
  const close = useUnitDialogStore((s) => s.close);
  const units = useUnitStore((s) => s.units);
  const addUnitInstance = useGraphStore((s) => s.addUnitInstance);

  const unit = pending ? units.find((u) => u.id === pending.unitId) : undefined;
  // 引数の入力状態(name→文字列)。Unitが変わるたびに既定値で初期化する。
  const [values, setValues] = useState<Record<string, string>>({});
  const [initFor, setInitFor] = useState<string | null>(null);

  if (!pending || !unit) return null;

  // 対象Unitが変わったら既定値で初期化(レンダー中の同期セット)
  if (initFor !== unit.id) {
    const init: Record<string, string> = {};
    for (const a of unit.args) init[a.name] = String(a.default);
    setValues(init);
    setInitFor(unit.id);
    return null;
  }

  const submit = () => {
    const argValues: Record<string, number | string> = {};
    for (const a of unit.args) {
      const parsed = parseArgValue(a, values[a.name] ?? String(a.default));
      if (parsed === null) {
        toast.error(`引数 ${a.name} の値が不正です`);
        return;
      }
      argValues[a.name] = parsed;
    }
    addUnitInstance(unit, argValues, pending.position);
    close();
    toast.success(`Unit「${unit.name}」を配置しました`);
  };

  return (
    <Dialog open onOpenChange={(open) => !open && close()}>
      <DialogContent className="max-w-sm" data-testid="unit-instance-dialog">
        <DialogHeader>
          <DialogTitle>{unit.name} を配置</DialogTitle>
          <DialogDescription>
            {unit.description || '引数を指定してキャンバスへ展開します。'}
          </DialogDescription>
        </DialogHeader>

        <div className="flex flex-col gap-3">
          {unit.args.length === 0 && (
            <div className="hint-text">このUnitに引数はありません。</div>
          )}
          {unit.args.map((a) => (
            <div key={a.name} className="flex flex-col gap-1">
              <Label className="text-xs">
                {a.name} <span className="text-muted-foreground">({a.type})</span>
              </Label>
              {a.type === 'enum' ? (
                <select
                  className="h-8 rounded-md border bg-transparent px-2 text-sm"
                  value={values[a.name] ?? String(a.default)}
                  onChange={(e) => setValues((v) => ({ ...v, [a.name]: e.target.value }))}
                >
                  {(a.options ?? []).map((opt) => (
                    <option key={opt} value={opt}>
                      {opt}
                    </option>
                  ))}
                </select>
              ) : (
                <Input
                  type="number"
                  value={values[a.name] ?? String(a.default)}
                  onChange={(e) => setValues((v) => ({ ...v, [a.name]: e.target.value }))}
                />
              )}
            </div>
          ))}
        </div>

        <DialogFooter>
          <Button variant="outline" onClick={close}>
            キャンセル
          </Button>
          <Button onClick={submit} data-testid="unit-instance-confirm">
            配置
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
