// 事前学習モデルのインポート&変換ウィザード。
// Step1 ソース: プリセット選択(カード)+teacherソース入力
// Step2 パラメータ: preset spec主要値をoverrides編集(数値のみ、簡易)
// Step3 実行&進捗: startConversion→0.5sポーリング→done時にIRをcanvasへ反映
//
// done経路はckptインポート(ModelDialogs.handleImport)と同じ: getCurrentIRで
// backendがSTATE反映済みのIRを取り出し、graphStore.setIR({needsBuild:true})+
// trainingStore.resetAll() でcanvasへ載せる。

import { useEffect, useRef, useState } from 'react';
import { toast } from 'sonner';
import { Loader2, X } from 'lucide-react';
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
  listConvertPresets,
  type ConvertPresetInfo,
  type ConvertStatus,
} from '../api/rest';
import { useGraphStore } from '../store/graphStore';
import { useTrainingStore } from '../store/trainingStore';
import type { ModelDialogKind } from './ModelDialogs';
import {
  isTerminalPhase,
  phaseLabel,
  requestCancel,
  runConversion,
} from './convertMachine';

// overrides編集で提示するspec主要値(数値のみの簡易フォーム。他フィールドはpreset既定のまま)。
const EDITABLE_SPEC_FIELDS: { key: string; label: string }[] = [
  { key: 'epochs_per_stage', label: '各段エポック数' },
  { key: 'lr', label: '学習率(枝刈り)' },
  { key: 'weight_bits_hidden', label: '隠れ層量子化bit' },
  { key: 'weight_bits_final', label: '最終層量子化bit' },
  { key: 'qat_epochs', label: 'QATエポック数' },
  { key: 'gate_acc', label: 'ゲート合格acc' },
  { key: 'batch_size', label: 'バッチサイズ' },
];

type Step = 1 | 2 | 3;

interface Props {
  kind: ModelDialogKind;
  onClose: () => void;
}

export function ConvertWizard({ kind, onClose }: Props) {
  const open = kind === 'convert';
  const [presets, setPresets] = useState<ConvertPresetInfo[] | null>(null);
  const [presetsError, setPresetsError] = useState<string | null>(null);
  const [step, setStep] = useState<Step>(1);
  const [presetName, setPresetName] = useState('');
  const [source, setSource] = useState('');
  const [overrides, setOverrides] = useState<Record<string, string>>({});

  const [running, setRunning] = useState(false);
  const [status, setStatus] = useState<ConvertStatus | null>(null);
  const [outcome, setOutcome] = useState<'error' | 'cancelled' | null>(null);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const convertIdRef = useRef<string | null>(null);
  const abortRef = useRef(false);

  const setIR = useGraphStore((s) => s.setIR);
  const selectedPreset = presets?.find((p) => p.name === presetName) ?? null;

  // ダイアログを開いたらプリセットを取得。閉じたら全state初期化。
  // open(prop)駆動のリセット/フェッチ発火は意図的(外部トリガとの同期)。
  useEffect(() => {
    if (!open) {
      abortRef.current = true;
      resetState();
      return;
    }
    abortRef.current = false;
    listConvertPresets()
      .then(({ presets: all }) => {
        // 現状datasetはmnistのみ対応(fashionは選ばせない)。
        const usable = all.filter((p) => p.dataset === 'mnist');
        setPresets(usable);
        setPresetsError(null);
        if (usable.length > 0) selectPreset(usable[0]);
      })
      .catch((e: unknown) => {
        setPresets([]);
        setPresetsError(e instanceof Error ? e.message : String(e));
      });
  }, [open]);

  function resetState() {
    setStep(1);
    setSource('');
    setOverrides({});
    setRunning(false);
    setStatus(null);
    setOutcome(null);
    setErrorMsg(null);
    convertIdRef.current = null;
  }

  function selectPreset(p: ConvertPresetInfo) {
    setPresetName(p.name);
    // overrides編集フォームの初期値はpreset specの既定値(文字列表示)。
    const init: Record<string, string> = {};
    for (const { key } of EDITABLE_SPEC_FIELDS) {
      const v = p.spec[key];
      if (typeof v === 'number') init[key] = String(v);
    }
    setOverrides(init);
  }

  /** overridesフォームから、preset既定と異なる数値のみを差分dictに変換。 */
  function buildOverrides(): Record<string, number> {
    const out: Record<string, number> = {};
    if (!selectedPreset) return out;
    for (const { key } of EDITABLE_SPEC_FIELDS) {
      const raw = overrides[key];
      if (raw === undefined || raw.trim() === '') continue;
      const num = Number(raw);
      if (!Number.isFinite(num)) continue;
      if (selectedPreset.spec[key] !== num) out[key] = num;
    }
    return out;
  }

  const handleStart = async () => {
    if (!selectedPreset) return;
    abortRef.current = false;
    setRunning(true);
    setStatus(null);
    setOutcome(null);
    setErrorMsg(null);
    setStep(3);
    try {
      const result = await runConversion(
        {
          source: source.trim(),
          preset: selectedPreset.name,
          overrides: buildOverrides(),
        },
        {
          onStatus: (s) => {
            convertIdRef.current = s.convert_id;
            setStatus(s);
          },
          onIR: (ir) => {
            setIR(ir, { needsBuild: true });
            useTrainingStore.getState().resetAll();
          },
          shouldAbort: () => abortRef.current,
        },
      );
      if (result.convertId) convertIdRef.current = result.convertId;
      if (result.kind === 'done') {
        toast.success('変換が完了しました(canvasへ反映、訓練前にbuildされます)');
        onClose();
      } else if (result.kind === 'error') {
        setOutcome('error');
        setErrorMsg(result.message);
      } else if (result.kind === 'cancelled') {
        setOutcome('cancelled');
      }
      // aborted(ダイアログclose)は何もしない
    } catch (e) {
      setOutcome('error');
      setErrorMsg(e instanceof Error ? e.message : String(e));
    } finally {
      setRunning(false);
    }
  };

  const handleCancel = () => {
    const cid = convertIdRef.current;
    if (cid) void requestCancel(cid);
  };

  const handleRetry = () => {
    setStep(2);
    setOutcome(null);
    setErrorMsg(null);
    setStatus(null);
  };

  const progressPct = Math.round((status?.progress ?? 0) * 100);
  const phase = status?.phase ?? 'preparing';
  const metrics = (status?.metrics ?? {}) as Record<string, unknown>;

  return (
    <Dialog
      open={open}
      onOpenChange={(o) => {
        // 実行中はダイアログを閉じない(ポーリングをabortしてから閉じる)
        if (!o) {
          abortRef.current = true;
          onClose();
        }
      }}
    >
      <DialogContent className="sm:max-w-lg">
        <DialogHeader>
          <DialogTitle>インポート&変換 (事前学習モデル→回路)</DialogTitle>
          <DialogDescription>
            事前学習MLPを枝刈り・量子化・回路化し、論理ゲート回路IRへ変換します
            (Step {step}/3)
          </DialogDescription>
        </DialogHeader>

        {/* Step1: ソース */}
        {step === 1 && (
          <div className="grid gap-3">
            {presetsError && (
              <p className="text-sm text-destructive">
                プリセット取得に失敗: {presetsError}
              </p>
            )}
            {presets === null && !presetsError && (
              <p className="text-sm text-muted-foreground">プリセットを読み込み中…</p>
            )}
            {presets !== null && presets.length === 0 && !presetsError && (
              <p className="text-sm text-muted-foreground">
                利用可能なプリセットがありません
              </p>
            )}
            <div className="grid gap-2">
              {presets?.map((p) => (
                <button
                  key={p.name}
                  type="button"
                  onClick={() => selectPreset(p)}
                  className={`rounded-md border p-3 text-left transition-colors ${
                    presetName === p.name
                      ? 'border-primary bg-primary/5'
                      : 'border-border hover:bg-accent'
                  }`}
                >
                  <div className="text-sm font-medium">{p.label}</div>
                  <div className="mt-1 text-[11px] leading-snug text-muted-foreground">
                    {p.description}
                  </div>
                  <div className="mt-1.5 text-[11px] text-muted-foreground/80">
                    ヒント: {p.teacher_hint}
                  </div>
                </button>
              ))}
            </div>
            <div className="grid gap-1.5">
              <Label htmlFor="convert-source">teacherソース(省略時はプリセット既定)</Label>
              <Input
                id="convert-source"
                value={source}
                placeholder={selectedPreset?.teacher_hint ?? 'ckpt:checkpoints/.../best.pt'}
                onChange={(e) => setSource(e.target.value)}
              />
            </div>
          </div>
        )}

        {/* Step2: パラメータ */}
        {step === 2 && (
          <div className="grid gap-3">
            <p className="text-[11px] text-muted-foreground">
              プリセット「{selectedPreset?.label}」の主要値。既定と異なる値のみ上書き送信します。
            </p>
            <div className="grid grid-cols-2 gap-3">
              {EDITABLE_SPEC_FIELDS.filter(
                (f) => typeof selectedPreset?.spec[f.key] === 'number',
              ).map(({ key, label }) => (
                <div key={key} className="grid gap-1.5">
                  <Label htmlFor={`spec-${key}`} className="text-xs">
                    {label}
                  </Label>
                  <Input
                    id={`spec-${key}`}
                    type="number"
                    value={overrides[key] ?? ''}
                    onChange={(e) =>
                      setOverrides((o) => ({ ...o, [key]: e.target.value }))
                    }
                  />
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Step3: 実行&進捗 */}
        {step === 3 && (
          <div className="grid gap-3">
            <div className="flex items-center justify-between text-sm">
              <span className="flex items-center gap-2 font-medium">
                {running && !isTerminalPhase(phase) && (
                  <Loader2 className="size-4 animate-spin" />
                )}
                {phaseLabel(phase)}
              </span>
              <span className="tabular-nums text-muted-foreground">{progressPct}%</span>
            </div>
            <div className="h-2 w-full overflow-hidden rounded-full bg-muted">
              <div
                className={`h-full transition-all ${
                  outcome === 'error'
                    ? 'bg-destructive'
                    : outcome === 'cancelled'
                      ? 'bg-muted-foreground'
                      : 'bg-primary'
                }`}
                style={{ width: `${progressPct}%` }}
              />
            </div>

            <ConvertMetrics metrics={metrics} />

            {outcome === 'error' && (
              <p className="text-sm text-destructive">
                変換に失敗しました: {errorMsg ?? status?.error ?? '不明なエラー'}
              </p>
            )}
            {outcome === 'cancelled' && (
              <p className="text-sm text-muted-foreground">変換をキャンセルしました。</p>
            )}
          </div>
        )}

        <DialogFooter className="gap-2 sm:justify-between">
          <div className="flex gap-2">
            {step === 2 && (
              <Button variant="ghost" onClick={() => setStep(1)}>
                戻る
              </Button>
            )}
            {step === 3 && running && (
              <Button variant="outline" onClick={handleCancel}>
                <X className="size-4" />
                キャンセル
              </Button>
            )}
            {step === 3 && !running && outcome !== null && (
              <Button variant="outline" onClick={handleRetry}>
                再試行
              </Button>
            )}
          </div>
          <div className="flex gap-2">
            {step === 1 && (
              <Button disabled={!selectedPreset} onClick={() => setStep(2)}>
                次へ
              </Button>
            )}
            {step === 2 && (
              <Button disabled={!selectedPreset} onClick={() => void handleStart()}>
                変換を開始
              </Button>
            )}
          </div>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}

/** 途中経過メトリクス(段階acc・回路統計)の簡易表示。 */
function ConvertMetrics({ metrics }: { metrics: Record<string, unknown> }) {
  const rows: { label: string; value: string }[] = [];

  const stageAcc = metrics.pruning_stage_acc;
  if (Array.isArray(stageAcc) && stageAcc.length > 0) {
    rows.push({
      label: '枝刈り段階acc',
      value: stageAcc.map((v) => fmt(v)).join(' → '),
    });
  }
  pushNum(rows, '量子化acc', metrics.quant_acc);
  pushNum(rows, 'QAT acc', metrics.qat_acc);
  pushNum(rows, 'hard acc', metrics.hard_acc);
  pushNum(rows, 'LUT出力総数', metrics.total_lut_outputs);
  pushNum(rows, 'LUT層数', metrics.n_lut_layers);

  if (rows.length === 0) return null;
  return (
    <dl className="grid grid-cols-2 gap-x-4 gap-y-1 text-[11px]">
      {rows.map((r) => (
        <div key={r.label} className="flex justify-between gap-2">
          <dt className="text-muted-foreground">{r.label}</dt>
          <dd className="tabular-nums">{r.value}</dd>
        </div>
      ))}
    </dl>
  );
}

function fmt(v: unknown): string {
  if (typeof v !== 'number') return String(v);
  return Number.isInteger(v) ? String(v) : v.toFixed(4);
}

function pushNum(
  rows: { label: string; value: string }[],
  label: string,
  v: unknown,
) {
  if (typeof v === 'number') rows.push({ label, value: fmt(v) });
}
