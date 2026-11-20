// モデル操作ダイアログ群: デモ生成 / checkpoint読込 / IR保存 / IR読込。
// サーバパスに加えて、IR JSONのブラウザダウンロード/ファイルアップロードに対応
// (サーバのファイルシステムを意識せずにモデルを持ち出し・復元できる)。

import { useRef, useState } from 'react';
import { toast } from 'sonner';
import { Download, Upload } from 'lucide-react';
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
import {
  createDemoModel,
  importModel,
  importPythonModel,
  loadIRFile,
  putCurrentIR,
  saveIRFile,
} from '../api/rest';
import type { IRGraphJson } from '../ir/schema';
import { useGraphStore } from '../store/graphStore';
import { useTrainingStore } from '../store/trainingStore';
import { ConvertWizard } from './ConvertWizard';

export type ModelDialogKind =
  | 'demo' | 'ckpt' | 'python' | 'convert' | 'ir-save' | 'ir-load' | null;

interface Props {
  kind: ModelDialogKind;
  onClose: () => void;
}

export function ModelDialogs({ kind, onClose }: Props) {
  const [k, setK] = useState('2000');
  const [layers, setLayers] = useState('3');
  const [ckptPath, setCkptPath] = useState('');
  const [pythonPath, setPythonPath] = useState('tools/editor/samples/llvit_tiny.py');
  const [pythonSymbol, setPythonSymbol] = useState('');
  const [irPath, setIrPath] = useState('checkpoints/editor/model_ir.json');
  const [busy, setBusy] = useState(false);

  const setIR = useGraphStore((s) => s.setIR);

  const run = async (fn: () => Promise<string>) => {
    setBusy(true);
    try {
      const message = await fn();
      toast.success(message);
      onClose();
    } catch (e) {
      toast.error(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  const handleDemo = () =>
    run(async () => {
      const res = await createDemoModel(Number(k), Number(layers));
      setIR(res.ir);
      useTrainingStore.getState().resetAll(); // 旧セッションの履歴を破棄
      return `デモモデルを生成しました(k=${k}, layers=${layers})`;
    });

  const handleImport = () =>
    run(async () => {
      const res = await importModel(ckptPath.trim());
      setIR(res.ir);
      useTrainingStore.getState().resetAll();
      return 'チェックポイントを読み込みました';
    });

  const handlePythonImport = () =>
    run(async () => {
      const res = await importPythonModel(pythonPath.trim(), {
        symbol: pythonSymbol.trim() || null,
      });
      setIR(res.ir, { needsBuild: false });
      useTrainingStore.getState().resetAll();
      return 'Pythonモデルを読み込みました';
    });

  const handleSave = () =>
    run(async () => {
      await useGraphStore.getState().syncNow();
      const res = await saveIRFile(irPath.trim());
      return `IRを保存しました: ${res.path}`;
    });

  const handleLoad = () =>
    run(async () => {
      const res = await loadIRFile(irPath.trim());
      setIR(res.ir, { needsBuild: true });
      useTrainingStore.getState().resetAll();
      return 'IRを読み込みました(訓練前にbuildされます)';
    });

  // --- ブラウザDL/アップロード ---

  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleDownload = () =>
    run(async () => {
      const graph = useGraphStore.getState();
      await graph.syncNow();
      const ir = useGraphStore.getState().ir;
      if (!ir) throw new Error('モデル未読込');
      const name = String(ir.metadata?.name ?? 'model');
      const blob = new Blob([JSON.stringify(ir, null, 1)], {
        type: 'application/json',
      });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `${name.replace(/[^0-9A-Za-z_-]+/g, '_')}.ir.json`;
      a.click();
      URL.revokeObjectURL(url);
      return 'IR JSONをダウンロードしました';
    });

  const handleUpload = (file: File) =>
    run(async () => {
      const text = await file.text();
      let ir: IRGraphJson;
      try {
        ir = JSON.parse(text) as IRGraphJson;
      } catch {
        throw new Error('JSONとして解釈できないファイルです');
      }
      if (ir.format !== 'magicdelta-ir') {
        throw new Error(`magicdelta-ir形式ではありません(format=${ir.format})`);
      }
      await putCurrentIR(ir); // サーバ側validateに通す(不正IRは422)
      setIR(ir, { needsBuild: true });
      useTrainingStore.getState().resetAll();
      return `${file.name} を読み込みました(訓練前にbuildされます)`;
    });

  return (
    <>
      <Dialog open={kind === 'demo'} onOpenChange={(o) => !o && onClose()}>
        <DialogContent className="sm:max-w-sm">
          <DialogHeader>
            <DialogTitle>デモモデル生成 (MNIST)</DialogTitle>
            <DialogDescription>
              Flatten → LutLayer×layers → GroupSum の直列モデルを生成します
            </DialogDescription>
          </DialogHeader>
          <div className="grid grid-cols-2 gap-3">
            <div className="grid gap-1.5">
              <Label htmlFor="demo-k">k(ニューロン数)</Label>
              <Input id="demo-k" value={k} onChange={(e) => setK(e.target.value)} />
            </div>
            <div className="grid gap-1.5">
              <Label htmlFor="demo-layers">レイヤ数</Label>
              <Input id="demo-layers" value={layers} onChange={(e) => setLayers(e.target.value)} />
            </div>
          </div>
          <DialogFooter>
            <Button disabled={busy} onClick={() => void handleDemo()}>
              生成
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      <Dialog open={kind === 'python'} onOpenChange={(o) => !o && onClose()}>
        <DialogContent className="sm:max-w-md">
          <DialogHeader>
            <DialogTitle>Pythonモデル読み込み</DialogTitle>
            <DialogDescription>
              @editor_model付きbuilder、またはbuild_model()を読み込みます
            </DialogDescription>
          </DialogHeader>
          <div className="grid gap-3">
            <div className="grid gap-1.5">
              <Label htmlFor="python-path">Pythonファイルパス</Label>
              <Input
                id="python-path"
                value={pythonPath}
                placeholder="tools/editor/samples/llvit_tiny.py"
                onChange={(e) => setPythonPath(e.target.value)}
              />
            </div>
            <div className="grid gap-1.5">
              <Label htmlFor="python-symbol">symbol(任意)</Label>
              <Input
                id="python-symbol"
                value={pythonSymbol}
                placeholder="build_model"
                onChange={(e) => setPythonSymbol(e.target.value)}
              />
            </div>
          </div>
          <DialogFooter>
            <Button
              disabled={busy || pythonPath.trim() === ''}
              onClick={() => void handlePythonImport()}
            >
              読み込み
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      <Dialog open={kind === 'ckpt'} onOpenChange={(o) => !o && onClose()}>
        <DialogContent className="sm:max-w-md">
          <DialogHeader>
            <DialogTitle>チェックポイント読み込み</DialogTitle>
            <DialogDescription>学習済みモデル(.pt)からIRを抽出します</DialogDescription>
          </DialogHeader>
          <div className="grid gap-1.5">
            <Label htmlFor="ckpt-path">checkpointパス</Label>
            <Input
              id="ckpt-path"
              value={ckptPath}
              placeholder="checkpoints/.../model.pt"
              onChange={(e) => setCkptPath(e.target.value)}
            />
          </div>
          <DialogFooter>
            <Button disabled={busy || ckptPath.trim() === ''} onClick={() => void handleImport()}>
              読み込み
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      <ConvertWizard kind={kind} onClose={onClose} />

      <Dialog
        open={kind === 'ir-save' || kind === 'ir-load'}
        onOpenChange={(o) => !o && onClose()}
      >
        <DialogContent className="sm:max-w-md">
          <DialogHeader>
            <DialogTitle>{kind === 'ir-save' ? 'IR保存' : 'IR読込'}</DialogTitle>
            <DialogDescription>
              {kind === 'ir-save'
                ? 'IR(JSON)をブラウザへダウンロード、またはサーバ側パスへ保存します'
                : 'IR(JSON)ファイルをアップロード、またはサーバ側パスから読み込みます'}
            </DialogDescription>
          </DialogHeader>
          {/* 推奨経路: ブラウザDL/アップロード */}
          {kind === 'ir-save' ? (
            <Button variant="outline" disabled={busy} onClick={() => void handleDownload()}>
              <Download className="size-4" />
              ブラウザにダウンロード (.ir.json)
            </Button>
          ) : (
            <>
              <input
                ref={fileInputRef}
                type="file"
                accept=".json,application/json"
                className="hidden"
                onChange={(e) => {
                  const file = e.target.files?.[0];
                  if (file) void handleUpload(file);
                  e.target.value = '';
                }}
              />
              <Button
                variant="outline"
                disabled={busy}
                onClick={() => fileInputRef.current?.click()}
              >
                <Upload className="size-4" />
                ファイルをアップロード (.ir.json)
              </Button>
            </>
          )}
          <div className="flex items-center gap-2 text-[11px] text-muted-foreground">
            <span className="h-px flex-1 bg-border" />
            またはサーバ側パス
            <span className="h-px flex-1 bg-border" />
          </div>
          <div className="grid gap-1.5">
            <Label htmlFor="ir-path">IRファイルパス</Label>
            <Input
              id="ir-path"
              value={irPath}
              placeholder="checkpoints/editor/model_ir.json"
              onChange={(e) => setIrPath(e.target.value)}
            />
          </div>
          <DialogFooter>
            <Button
              disabled={busy || irPath.trim() === ''}
              onClick={() => void (kind === 'ir-save' ? handleSave() : handleLoad())}
            >
              {kind === 'ir-save' ? 'サーバへ保存' : 'サーバから読み込み'}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </>
  );
}
