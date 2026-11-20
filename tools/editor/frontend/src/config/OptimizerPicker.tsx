// TrainSettingsFormへ最小フックで挿入できる独立コンポーネント。
// TrainSettingsForm自体は変更しないため、このコンポーネント単体で
// プロジェクト選択→Optimizer選択→適用ボタンまで完結させる。
// 選択されたOptimizer定義は onApply 経由でTrainingPreset相当の値を返すだけで、
// zustandストア(trainingStore)へは一切触れない(呼び出し側がapplyTrainingPresetする)。

import { useEffect, useState } from 'react';
import { toast } from 'sonner';
import { Wand2 } from 'lucide-react';
import { Button } from '@/components/ui/button';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { getProject, listProjects, type ProjectSummary } from '../api/rest';

/** OptimizerPickerがonApplyへ渡す値(trainingStoreのTrainingPresetと同形)。 */
export interface OptimizerApplyValues {
  dataset?: string;
  loss?: string;
  method?: 'bp' | 'ap' | 'bb';
  optimizer?: 'adam' | 'sgd';
  lr?: number;
  logicLr?: number | null;
  annLr?: number | null;
  batchSize?: number;
  epochs?: number;
}

interface OptimizerDef {
  network?: string;
  dataset?: string;
  method?: string;
  loss?: string;
  optimizer?: string;
  lr?: number;
  logic_lr?: number | null;
  ann_lr?: number | null;
  batch_size?: number;
  epochs?: number;
}

const KNOWN_METHODS = new Set(['bp', 'ap', 'bb']);
const KNOWN_OPTIMIZERS = new Set(['adam', 'sgd']);

/** OptimizerDef(スネークケース)をTrainingPreset形(キャメルケース)へ変換する。 */
function toApplyValues(def: OptimizerDef): OptimizerApplyValues {
  const out: OptimizerApplyValues = {};
  if (def.dataset) out.dataset = def.dataset;
  if (def.loss) out.loss = def.loss;
  if (def.method && KNOWN_METHODS.has(def.method)) {
    out.method = def.method as OptimizerApplyValues['method'];
  }
  if (def.optimizer && KNOWN_OPTIMIZERS.has(def.optimizer)) {
    out.optimizer = def.optimizer as OptimizerApplyValues['optimizer'];
  }
  if (typeof def.lr === 'number') out.lr = def.lr;
  if (def.logic_lr !== undefined) out.logicLr = def.logic_lr;
  if (def.ann_lr !== undefined) out.annLr = def.ann_lr;
  if (typeof def.batch_size === 'number') out.batchSize = def.batch_size;
  if (typeof def.epochs === 'number') out.epochs = def.epochs;
  return out;
}

interface Props {
  onApply: (values: OptimizerApplyValues, label: string) => void;
}

/**
 * プロジェクトのOptimizer定義一覧から1つ選び、適用する独立UI。
 * 統合手順はTrainSettingsForm.tsxのコメント/最終報告を参照
 * (TrainSettingsForm自体は変更しない)。
 */
export function OptimizerPicker({ onApply }: Props) {
  const [projects, setProjects] = useState<ProjectSummary[]>([]);
  const [projectId, setProjectId] = useState<string | null>(null);
  const [optimizers, setOptimizers] = useState<Record<string, OptimizerDef>>({});
  const [optimizerName, setOptimizerName] = useState<string | null>(null);

  useEffect(() => {
    void listProjects()
      .then(setProjects)
      .catch((e) => toast.error(String(e)));
  }, []);

  useEffect(() => {
    if (!projectId) {
      // eslint-disable-next-line react-hooks/set-state-in-effect
      setOptimizers({});
      setOptimizerName(null);
      return;
    }
    void getProject(projectId)
      .then((p) => {
        setOptimizers((p.optimizers as Record<string, OptimizerDef>) ?? {});
        setOptimizerName(null);
      })
      .catch((e) => toast.error(String(e)));
  }, [projectId]);

  const names = Object.keys(optimizers);

  const handleApply = () => {
    if (!optimizerName) return;
    const def = optimizers[optimizerName];
    if (!def) return;
    onApply(toApplyValues(def), optimizerName);
  };

  return (
    <div className="grid gap-1.5">
      <div className="text-[11px] font-semibold text-muted-foreground">
        Optimizer定義から適用
      </div>
      <Select value={projectId ?? ''} onValueChange={(v) => setProjectId(v || null)}>
        <SelectTrigger size="sm" className="w-full">
          <SelectValue placeholder="プロジェクトを選択" />
        </SelectTrigger>
        <SelectContent>
          {projects.map((p) => (
            <SelectItem key={p.id} value={p.id}>
              {p.name}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
      <Select
        value={optimizerName ?? ''}
        onValueChange={(v) => setOptimizerName(v || null)}
        disabled={names.length === 0}
      >
        <SelectTrigger size="sm" className="w-full">
          <SelectValue placeholder={names.length === 0 ? 'Optimizer未定義' : 'Optimizerを選択'} />
        </SelectTrigger>
        <SelectContent>
          {names.map((n) => (
            <SelectItem key={n} value={n}>
              {n}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
      <Button
        variant="outline"
        size="sm"
        disabled={!optimizerName}
        onClick={handleApply}
      >
        <Wand2 className="size-4" />
        適用
      </Button>
    </div>
  );
}
