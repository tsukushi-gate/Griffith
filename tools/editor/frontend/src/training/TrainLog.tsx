// コアエンジンlog(TRAININGタブ下段): WSで届く epoch/step/loss/acc ストリームを
// 行テキストとして時系列表示する(NNC TRAINING画面の下部ログ相当)。
// trainingStore の steps/loss/acc から末尾N件を組み立てる派生表示。

import { useMemo } from 'react';
import { useTrainingStore } from '../store/trainingStore';

const MAX_LINES = 200;

export function TrainLog() {
  // version購読でメトリクス追記のたびに再描画する
  const version = useTrainingStore((s) => s.version);
  const status = useTrainingStore((s) => s.status);

  const lines = useMemo(() => {
    void version; // 依存に含めて追記ごとに再計算
    const s = useTrainingStore.getState();
    const n = s.steps.length;
    const start = Math.max(0, n - MAX_LINES);
    const out: string[] = [];
    for (let i = start; i < n; i++) {
      const step = s.steps[i];
      const loss = s.loss[i];
      const acc = s.acc[i];
      const parts = [`step ${step}`];
      if (loss != null) parts.push(`loss ${loss.toFixed(4)}`);
      if (acc != null) parts.push(`acc ${(acc * 100).toFixed(1)}%`);
      out.push(parts.join('  '));
    }
    return out;
  }, [version]);

  return (
    <div className="flex h-full min-h-0 flex-col">
      <div className="shrink-0 border-b px-3 py-1 text-[11px] font-semibold tracking-widest text-muted-foreground uppercase">
        コアエンジンログ ・ {status}
      </div>
      <pre className="min-h-0 flex-1 overflow-auto p-3 font-mono text-[11px] leading-relaxed text-muted-foreground">
        {lines.length > 0
          ? lines.join('\n')
          : '学習を開始するとstep/loss/accのログがここに流れます'}
      </pre>
    </div>
  );
}
