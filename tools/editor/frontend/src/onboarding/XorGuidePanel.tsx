// XORチュートリアルガイド(オンボーディング、Lobe/NNC型)。
// XORサンプルを開いたときに表示する「組む→学習→触る→巻き戻す」4ステップの
// チェックリスト。各ストア(graph/training/playground)の実状態を購読して
// チェックが自動で進む。閉じる/再表示可能(開閉フラグはuiStoreで永続化)。

import { Check, Circle, GraduationCap, X } from 'lucide-react';
import { useEffect, useState } from 'react';
import { Button } from '@/components/ui/button';
import { usePlaygroundStore } from '../playground/playgroundStore';
import { useGraphStore } from '../store/graphStore';
import { useTrainingStore } from '../store/trainingStore';
import { useUIStore } from '../store/uiStore';

/** XORサンプルのIRメタデータ名(このモデルが読み込まれている間だけガイドを出す) */
const XOR_MODEL_NAME = 'sample_xor';

type StepKey = 'build' | 'train' | 'probe' | 'rewind';

const STEPS: { key: StepKey; label: string; hint: string }[] = [
  {
    key: 'build',
    label: '組む',
    hint: 'ヘッダの「学習開始」を押すとdraft層がbuildされ、回路が確定します。',
  },
  {
    key: 'train',
    label: '学習',
    hint: 'ヘッダの「学習開始」で学習セッションが動き出します(TRAININGタブで進捗を確認できます)。',
  },
  {
    key: 'probe',
    label: '触る',
    hint: 'EDITタブの下部ドック「プレイグラウンド」タブで実行をONにし、入力ビットを切り替えます。',
  },
  {
    key: 'rewind',
    label: '巻き戻す',
    hint: 'TRAININGタブの下部ドック「タイムライン」タブでスライダーを動かし、過去stepへ巻き戻します。',
  },
];

export function XorGuidePanel() {
  const xorGuideOpen = useUIStore((s) => s.xorGuideOpen);
  const setXorGuideOpen = useUIStore((s) => s.setXorGuideOpen);

  // 現在のモデルがXORサンプルか(このモデルのときだけガイド/再表示ピルを出す)
  const isXorModel = useGraphStore((s) => s.ir?.metadata?.name === XOR_MODEL_NAME);
  const needsBuild = useGraphStore((s) => s.needsBuild);
  const sessionId = useTrainingStore((s) => s.sessionId);
  const displayMode = useTrainingStore((s) => s.displayMode);
  const playgroundEnabled = usePlaygroundStore((s) => s.enabled);

  // 達成状態は単調(一度達成したら戻さない)。購読中の実状態のORで蓄積し、
  // sessionId消失やreplay→live復帰でチェックが戻らないようにする。
  const [done, setDone] = useState<Record<StepKey, boolean>>({
    build: false,
    train: false,
    probe: false,
    rewind: false,
  });

  useEffect(() => {
    // XORモデル以外へ切り替えたら達成状態をリセットする(モデル跨ぎで
    // 進捗が漏れないように)。XORモデル中は購読状態のORで単調に進める。
    if (!isXorModel) {
      // eslint-disable-next-line react-hooks/set-state-in-effect
      setDone({ build: false, train: false, probe: false, rewind: false });
      return;
    }
    setDone((prev) => {
      const next: Record<StepKey, boolean> = {
        build: prev.build || !needsBuild,
        train: prev.train || sessionId !== null,
        probe: prev.probe || playgroundEnabled,
        rewind: prev.rewind || displayMode === 'replay',
      };
      const changed = (Object.keys(next) as StepKey[]).some(
        (k) => next[k] !== prev[k],
      );
      return changed ? next : prev;
    });
  }, [isXorModel, needsBuild, sessionId, playgroundEnabled, displayMode]);

  if (!isXorModel) return null;

  const completedCount = STEPS.filter((s) => done[s.key]).length;

  // 閉じているときは再表示用の小さなピルだけ出す
  if (!xorGuideOpen) {
    return (
      <Button
        size="sm"
        variant="outline"
        className="fixed bottom-4 left-4 z-50 shadow-md"
        onClick={() => setXorGuideOpen(true)}
        title="XORチュートリアルを再表示"
      >
        <GraduationCap className="size-4" />
        XORガイド ({completedCount}/{STEPS.length})
      </Button>
    );
  }

  return (
    <div className="fixed bottom-4 left-4 z-50 w-72 rounded-lg border bg-card p-3 shadow-lg">
      <div className="mb-2 flex items-center gap-2">
        <GraduationCap className="size-4 text-primary" />
        <h3 className="flex-1 text-xs font-semibold">XORを学習してみる</h3>
        <span className="text-[10px] text-muted-foreground tabular-nums">
          {completedCount}/{STEPS.length}
        </span>
        <Button
          size="icon"
          variant="ghost"
          className="size-6"
          onClick={() => setXorGuideOpen(false)}
          title="ガイドを閉じる"
        >
          <X className="size-3.5" />
        </Button>
      </div>
      <ol className="grid gap-2">
        {STEPS.map((step, i) => {
          const isDone = done[step.key];
          return (
            <li key={step.key} className="flex items-start gap-2">
              {isDone ? (
                <Check className="mt-0.5 size-4 shrink-0 text-[var(--status-ok)]" />
              ) : (
                <Circle className="mt-0.5 size-4 shrink-0 text-muted-foreground/50" />
              )}
              <div className="min-w-0">
                <div
                  className={`text-xs font-medium ${
                    isDone ? 'text-muted-foreground line-through' : 'text-foreground'
                  }`}
                >
                  {i + 1}. {step.label}
                </div>
                {!isDone && (
                  <p className="mt-0.5 text-[11px] leading-snug text-muted-foreground">
                    {step.hint}
                  </p>
                )}
              </div>
            </li>
          );
        })}
      </ol>
      {completedCount === STEPS.length && (
        <p className="mt-2 rounded-md bg-[var(--status-ok)]/10 px-2 py-1.5 text-[11px] text-[var(--status-ok)]">
          全ステップ完了。組む→学習→触る→巻き戻すの一周を体験できました。
        </p>
      )}
    </div>
  );
}
