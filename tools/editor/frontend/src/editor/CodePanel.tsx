// コード生成パネル(下部ドック「コード」タブ)。
// buildはヘッダに常駐、IR保存/読込はモデルメニューへ移設済み(旧CodegenPanelを分割)。

import { Copy, Download, FileCode2 } from 'lucide-react';
import { useState } from 'react';
import { toast } from 'sonner';
import { Button } from '@/components/ui/button';
import { generateCode } from '../api/rest';
import { hasShapeErrors } from '../ir/shapeInference';
import { useGraphStore } from '../store/graphStore';

export function CodePanel() {
  const ir = useGraphStore((s) => s.ir);
  const diagnostics = useGraphStore((s) => s.diagnostics);
  const syncNow = useGraphStore((s) => s.syncNow);

  const [code, setCode] = useState<string | null>(null);
  const [filename, setFilename] = useState('model.py');
  const [busy, setBusy] = useState(false);

  const blocked = !ir || busy || hasShapeErrors(diagnostics);

  const handleCodegen = async () => {
    setBusy(true);
    try {
      await syncNow();
      const res = await generateCode();
      setCode(res.code);
      setFilename(res.filename);
    } catch (e) {
      toast.error(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  const handleCopy = () => {
    if (!code) return;
    void navigator.clipboard.writeText(code);
    toast.success('コードをコピーしました');
  };

  const handleDownload = () => {
    if (!code) return;
    const blob = new Blob([code], { type: 'text/x-python' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="flex h-full flex-col gap-2 p-3">
      <div className="flex items-center gap-2">
        <Button size="sm" disabled={blocked} onClick={() => void handleCodegen()}>
          <FileCode2 className="size-4" />
          コード生成
        </Button>
        {code && (
          <>
            <Button size="sm" variant="outline" onClick={handleCopy}>
              <Copy className="size-4" />
              コピー
            </Button>
            <Button size="sm" variant="outline" onClick={handleDownload}>
              <Download className="size-4" />
              {filename}
            </Button>
          </>
        )}
      </div>
      {code ? (
        <pre className="code-block !max-h-none min-h-0 flex-1">{code}</pre>
      ) : (
        <div className="hint-text">
          「コード生成」で現在のIRからPyTorchモデル構築コードを生成します
        </div>
      )}
    </div>
  );
}
