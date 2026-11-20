// Shipレンズ(HDL一気通貫プリフライト、§4-3、下部ドック「出荷」タブ)。
//   1. HDL出力可否バッジ + 未離散化draftノード数 + 検証(未使用)警告
//   2. ブロッカー一覧(クリックで該当ノードへジャンプ)
//   3. リソース見積り表(LUT数/論理深さ/配線)
//   4. HDLエクスポート(Verilog/VHDL選択→プレビュー+ダウンロード)
//   5. 等価性検証(iverilog/ghdlが利用可能なときのみ実行)
// グラフ編集に追従して同期完了後(dirty=false)に自動再取得する。

import {
  CheckCircle2,
  Copy,
  Download,
  RefreshCw,
  ShieldCheck,
  XCircle,
} from 'lucide-react';
import { useCallback, useEffect, useState } from 'react';
import { toast } from 'sonner';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import {
  exportHdl,
  getPreflight,
  verifyHdl,
  type HdlExportResponse,
  type HdlLang,
  type HdlVerifyResponse,
  type PreflightResponse,
} from '../api/rest';
import { flowApi } from '../editor/flowApi';
import { useGraphStore } from '../store/graphStore';
import { useViewStore } from '../store/viewStore';
import {
  blockerKindLabel,
  estimateRows,
  preflightBadge,
} from './preflight';

/** ブロッカークリック: グラフビューへ戻し、該当ノードを選択+中央寄せする。 */
function jumpToNode(nodeId: string): void {
  useViewStore.getState().setView({ kind: 'graph' });
  useGraphStore.getState().setSelected(nodeId);
  flowApi.current?.focusNode(nodeId);
}

function LangToggle({
  lang,
  onChange,
  disabled,
}: {
  lang: HdlLang;
  onChange: (lang: HdlLang) => void;
  disabled: boolean;
}) {
  return (
    <div className="flex overflow-hidden rounded-md border">
      {(['verilog', 'vhdl'] as const).map((l) => (
        <button
          key={l}
          type="button"
          disabled={disabled}
          onClick={() => onChange(l)}
          className={`px-2 py-1 text-xs disabled:opacity-50 ${
            lang === l
              ? 'bg-primary text-primary-foreground'
              : 'bg-transparent text-muted-foreground hover:bg-accent'
          }`}
        >
          {l === 'verilog' ? 'Verilog' : 'VHDL'}
        </button>
      ))}
    </div>
  );
}

export function ShipPanel() {
  const ir = useGraphStore((s) => s.ir);
  const dirty = useGraphStore((s) => s.dirty);

  const [preflight, setPreflight] = useState<PreflightResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [lang, setLang] = useState<HdlLang>('verilog');
  const [exported, setExported] = useState<HdlExportResponse | null>(null);
  const [verifyResult, setVerifyResult] = useState<HdlVerifyResponse | null>(null);
  const [busy, setBusy] = useState(false);

  const refresh = useCallback(async () => {
    try {
      const res = await getPreflight();
      setPreflight(res);
      setError(null);
    } catch (e) {
      setPreflight(null);
      setError(e instanceof Error ? e.message : String(e));
    }
  }, []);

  // IR同期完了(dirty=false)に追従して自動再取得。編集中(dirty=true)はbackendが
  // 古いため待つ。graphStoreのデバウンスPUT機構(SYNC_DEBOUNCE_MS)に乗る形。
  useEffect(() => {
    if (!ir || dirty) return;
    const timer = setTimeout(() => void refresh(), 200);
    return () => clearTimeout(timer);
  }, [ir, dirty, refresh]);

  const handleExport = async () => {
    setBusy(true);
    try {
      const res = await exportHdl(lang);
      setExported(res);
      setVerifyResult(null);
    } catch (e) {
      toast.error(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  const handleVerify = async () => {
    setBusy(true);
    try {
      const res = await verifyHdl(lang);
      setVerifyResult(res);
      if (!res.available) {
        toast.info(
          `${lang === 'verilog' ? 'iverilog' : 'ghdl'}が未導入のため検証をスキップしました`,
        );
      }
    } catch (e) {
      toast.error(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  const handleCopy = () => {
    if (!exported) return;
    void navigator.clipboard.writeText(exported.source);
    toast.success('HDLソースをコピーしました');
  };

  const handleDownload = () => {
    if (!exported) return;
    const blob = new Blob([exported.source], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = exported.filename;
    a.click();
    URL.revokeObjectURL(url);
  };

  if (!ir) {
    return (
      <div className="hint-text p-3">
        モデルを読み込むとHDL出荷プリフライトを表示します
      </div>
    );
  }

  const verifyAvailable = preflight?.verify_available[
    lang === 'verilog' ? 'iverilog' : 'ghdl'
  ];
  const badge = preflight ? preflightBadge(preflight) : null;

  return (
    <div className="flex h-full flex-col gap-3 overflow-y-auto p-3 text-sm">
      <div className="flex items-center gap-2">
        <span className="font-medium">出荷プリフライト</span>
        {badge && (
          <Badge variant={badge.tone === 'ok' ? 'default' : 'destructive'}>
            {badge.tone === 'ok' ? (
              <CheckCircle2 className="size-3" />
            ) : (
              <XCircle className="size-3" />
            )}
            {badge.text}
          </Badge>
        )}
        {dirty && (
          <span className="text-[11px] text-muted-foreground">同期待ち…</span>
        )}
        <Button
          size="sm"
          variant="ghost"
          className="ml-auto"
          onClick={() => void refresh()}
        >
          <RefreshCw className="size-4" />
          再取得
        </Button>
      </div>

      {error && (
        <div className="rounded-md border border-destructive/40 bg-destructive/10 p-2 text-xs text-destructive">
          {error}
        </div>
      )}

      {preflight && (
        <>
          <div className="flex flex-wrap gap-x-4 gap-y-1 text-xs text-muted-foreground">
            <span>未離散化(draft)ノード: {preflight.draft_node_count}</span>
            {preflight.warnings.length > 0 && (
              <span>警告: {preflight.warnings.length}件</span>
            )}
          </div>

          {preflight.blockers.length > 0 && (
            <div className="flex flex-col gap-1">
              <div className="text-xs font-medium text-destructive">
                ブロッカー(クリックで該当ノードへ)
              </div>
              {preflight.blockers.map((b, i) => (
                <button
                  key={`${b.node_id ?? 'none'}-${i}`}
                  type="button"
                  disabled={!b.node_id}
                  onClick={() => b.node_id && jumpToNode(b.node_id)}
                  className="flex items-start gap-2 rounded-md border border-destructive/30 bg-destructive/5 p-2 text-left text-xs enabled:hover:bg-destructive/10 disabled:cursor-default"
                >
                  <Badge variant="outline" className="shrink-0">
                    {blockerKindLabel(b.kind)}
                  </Badge>
                  <span className="min-w-0 break-words">{b.message}</span>
                </button>
              ))}
            </div>
          )}

          {preflight.estimate && (
            <div className="flex flex-col gap-1">
              <div className="text-xs font-medium">リソース見積り</div>
              <table className="w-full border-collapse text-xs">
                <tbody>
                  {estimateRows(preflight.estimate).map((row) => (
                    <tr key={row.label} className="border-b border-border/50">
                      <td className="py-1 pr-3 text-muted-foreground">
                        {row.label}
                      </td>
                      <td className="py-1 font-mono">{row.value}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}

          <div className="flex flex-wrap items-center gap-2 border-t pt-2">
            <LangToggle lang={lang} onChange={setLang} disabled={busy} />
            <Button
              size="sm"
              disabled={busy || !preflight.hdl_exportable}
              onClick={() => void handleExport()}
            >
              HDL生成
            </Button>
            <Button
              size="sm"
              variant="outline"
              disabled={busy || !preflight.hdl_exportable || !verifyAvailable}
              onClick={() => void handleVerify()}
              title={
                verifyAvailable
                  ? '等価性検証を実行'
                  : `${lang === 'verilog' ? 'iverilog' : 'ghdl'}が未導入のため検証不可`
              }
            >
              <ShieldCheck className="size-4" />
              検証
            </Button>
          </div>

          {verifyResult && verifyResult.available && (
            <div
              className={`rounded-md border p-2 text-xs ${
                verifyResult.passed
                  ? 'border-emerald-500/40 bg-emerald-500/10'
                  : 'border-destructive/40 bg-destructive/10 text-destructive'
              }`}
            >
              {verifyResult.passed
                ? `等価性検証OK(${verifyResult.n_vectors}ベクトル一致)`
                : `等価性検証NG(${verifyResult.n_vectors}ベクトル)`}
            </div>
          )}

          {exported && (
            <div className="flex min-h-0 flex-1 flex-col gap-2">
              <div className="flex items-center gap-2">
                <Button size="sm" variant="outline" onClick={handleCopy}>
                  <Copy className="size-4" />
                  コピー
                </Button>
                <Button size="sm" variant="outline" onClick={handleDownload}>
                  <Download className="size-4" />
                  {exported.filename}
                </Button>
              </div>
              <pre className="code-block !max-h-none min-h-0 flex-1">
                {exported.source}
              </pre>
            </div>
          )}
        </>
      )}

      {!preflight && !error && (
        <div className="hint-text">プリフライトを取得中…</div>
      )}
    </div>
  );
}
