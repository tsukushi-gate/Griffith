// 学習設定の常設フォーム(§2 Trainレンズ)。
// 従来ヘッダのPopoverに埋没していたdataset/loss/optimizer/method/lr/batch/epochsを
// 下部ドック「学習」タブ内に常設表示する。実行ボタン(開始/一時停止/停止)と
// ステータスはヘッダのTrainControlsに残す。
// I/O幅不整合でdisableされる理由は、ツールチップ頼みでなくフォーム内に明示する。

import { Check, LineChart } from 'lucide-react';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Field, FieldLabel } from '@/components/ui/field';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Toggle } from '@/components/ui/toggle';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { ToggleGroup, ToggleGroupItem } from '@/components/ui/toggle-group';
import { datasetMismatchReason, methodDisabledReason } from '../api/rest';
import {
  ACTIVE_STATES,
  useTrainingStore,
  type TrainingMethod,
  type TrainingOptimizer,
} from '../store/trainingStore';
import { useResolvedTrainConfig } from './useResolvedTrainConfig';
import { OptimizerPicker } from '../config/OptimizerPicker';

const METHOD_OPTIONS = [
  { value: 'bp', label: 'BP(誤差逆伝播)' },
  { value: 'ap', label: 'AP法' },
  { value: 'bb', label: 'BB(BinaryBrain風)' },
] as const;

const OPTIMIZER_OPTIONS = ['adam', 'sgd'] as const;

export function TrainSettingsForm() {
  const trainConfig = useTrainingStore((s) => s.trainConfig);
  const setTrainConfig = useTrainingStore((s) => s.setTrainConfig);
  const status = useTrainingStore((s) => s.status);
  const {
    datasets,
    inWidth,
    outWidth,
    selectedDataset,
    kind,
    lossOptions,
    loss,
    method,
    disabledReason,
  } = useResolvedTrainConfig();

  const active = ACTIVE_STATES.includes(status);
  const { optimizer, lr, logicLr, annLr, batchSize, epochs, presetLabel } =
    trainConfig;
  const wandbEnabled = trainConfig.wandbEnabled;

  // 実効データセットのI/O不整合理由(明示メッセージ用)
  const selectedMismatch = selectedDataset
    ? datasetMismatchReason(selectedDataset, inWidth, outWidth)
    : null;

  return (
    <div className="flex w-64 shrink-0 flex-col gap-3 overflow-y-auto border-r p-3">
      <div className="text-[11px] font-semibold text-muted-foreground">
        学習設定
      </div>

      {presetLabel && (
        <div className="rounded-md border bg-muted/40 px-2.5 py-2 text-[11px] text-muted-foreground">
          推奨設定: {presetLabel}
        </div>
      )}

      {disabledReason && (
        <Alert variant="destructive" className="px-2.5 py-2">
          <AlertDescription className="text-xs">
            {disabledReason}
          </AlertDescription>
        </Alert>
      )}

      {active && (
        <div className="rounded-md border bg-muted/40 px-2.5 py-2 text-[11px] text-muted-foreground">
          学習中は設定を変更しても次回開始時に反映されます
        </div>
      )}

      <div className="grid gap-1.5">
        <Label>データセット</Label>
        <Select
          value={selectedDataset?.name ?? ''}
          onValueChange={(value) => setTrainConfig({ datasetChoice: value })}
        >
          <SelectTrigger
            size="sm"
            className="w-full"
            data-testid="dataset-select"
          >
            <SelectValue placeholder="データセットを選択" />
          </SelectTrigger>
          <SelectContent>
            {datasets.map((d) => {
              const mismatch = datasetMismatchReason(d, inWidth, outWidth);
              return (
                <SelectItem
                  key={d.name}
                  value={d.name}
                  disabled={mismatch !== null}
                  title={mismatch ?? `${d.label}(I/O幅一致)`}
                >
                  <span className="flex items-center gap-1.5">
                    {mismatch === null && (
                      <Check className="size-3 text-[var(--status-ok)]" />
                    )}
                    {d.label}
                    <span className="text-[10px] text-muted-foreground">
                      {mismatch ?? `${d.in_width}→${d.out_width}`}
                    </span>
                  </span>
                </SelectItem>
              );
            })}
          </SelectContent>
        </Select>
        {/* I/O幅不整合の理由をフォーム内に明示(ツールチップ頼みにしない) */}
        {selectedMismatch && (
          <p className="text-[11px] text-destructive">
            現在のモデルI/O幅({inWidth ?? '?'}→{outWidth ?? '?'})と不整合:{' '}
            {selectedMismatch}
          </p>
        )}
      </div>

      <div className="grid grid-cols-2 gap-2">
        <div className="grid gap-1.5">
          <Label>学習手法</Label>
          <Select
            value={method}
            onValueChange={(v) =>
              setTrainConfig({ methodChoice: v as TrainingMethod })
            }
          >
            <SelectTrigger size="sm" className="w-full">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {METHOD_OPTIONS.map((m) => {
                const reason = methodDisabledReason(m.value, kind);
                return (
                  <SelectItem
                    key={m.value}
                    value={m.value}
                    disabled={reason !== null}
                    title={reason ?? m.label}
                  >
                    {m.label}
                  </SelectItem>
                );
              })}
            </SelectContent>
          </Select>
        </div>
        <div className="grid gap-1.5">
          <Label>損失関数</Label>
          <Select
            value={loss}
            onValueChange={(value) => setTrainConfig({ lossChoice: value })}
          >
            <SelectTrigger size="sm" className="w-full">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {lossOptions.map((l) => (
                <SelectItem key={l} value={l}>
                  {l}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
      </div>

      <Field className="gap-1.5">
        <FieldLabel>Optimizer</FieldLabel>
        <ToggleGroup
          type="single"
          value={optimizer}
          variant="outline"
          size="sm"
          className="grid w-full grid-cols-2"
          onValueChange={(v) => {
            if (v) setTrainConfig({ optimizer: v as TrainingOptimizer });
          }}
        >
          {OPTIMIZER_OPTIONS.map((o) => (
            <ToggleGroupItem key={o} value={o} className="w-full">
              {o}
            </ToggleGroupItem>
          ))}
        </ToggleGroup>
        {/* CONFIGタブの名前付きOptimizer定義から一括適用(バックのTrainStartRequestは無変更) */}
        <OptimizerPicker
          onApply={(values, label) =>
            useTrainingStore.getState().applyTrainingPreset(values, label)
          }
        />
      </Field>

      <div className="grid gap-1.5">
        <Label htmlFor="train-lr">学習率 (lr)</Label>
        <Input
          id="train-lr"
          value={lr}
          onChange={(e) => setTrainConfig({ lr: e.target.value })}
        />
      </div>

      <div className="grid grid-cols-2 gap-2">
        <div className="grid gap-1.5">
          <Label htmlFor="train-logic-lr">Logic LR</Label>
          <Input
            id="train-logic-lr"
            value={logicLr}
            placeholder="未指定"
            onChange={(e) => setTrainConfig({ logicLr: e.target.value })}
          />
        </div>
        <div className="grid gap-1.5">
          <Label htmlFor="train-ann-lr">ANN LR</Label>
          <Input
            id="train-ann-lr"
            value={annLr}
            placeholder="未指定"
            onChange={(e) => setTrainConfig({ annLr: e.target.value })}
          />
        </div>
      </div>

      <div className="grid gap-1.5">
        <Label htmlFor="train-batch">バッチサイズ</Label>
        <Input
          id="train-batch"
          value={batchSize}
          onChange={(e) => setTrainConfig({ batchSize: e.target.value })}
        />
      </div>

      <div className="grid gap-1.5">
        <Label htmlFor="train-epochs">エポック数</Label>
        <Input
          id="train-epochs"
          value={epochs}
          onChange={(e) => setTrainConfig({ epochs: e.target.value })}
        />
      </div>

      {/* wandb連携(opt-in)。有効時のみ学習セッションをwandbへログし、
          Run履歴のwandbリンクが実URLになる。未認証環境では無害に無効化される。 */}
      <Field className="gap-1.5">
        <FieldLabel>実験トラッキング</FieldLabel>
        <Toggle
          variant="outline"
          size="sm"
          pressed={wandbEnabled}
          onPressedChange={(v) => setTrainConfig({ wandbEnabled: v })}
          className="w-full justify-start"
          aria-label="wandb連携"
          title="有効にすると学習をwandbへログします(要WANDB_API_KEY)"
        >
          <LineChart className="size-4" />
          wandbへログ{wandbEnabled ? '(on)' : '(off)'}
        </Toggle>
      </Field>
    </div>
  );
}
