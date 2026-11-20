// 空状態オンボーディング。モデル未読込時にキャンバス中央へ表示する。
// サンプルは共通ローダ(samples/loadSample)経由で投入する。

import { Blocks, Library, Sparkles } from 'lucide-react';
import { useEffect, useState } from 'react';
import { toast } from 'sonner';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import {
  Empty,
  EmptyDescription,
  EmptyHeader,
  EmptyTitle,
} from '@/components/ui/empty';
import { createDemoModel } from '../api/rest';
import { TemplateGallery } from '../blocks/TemplateGallery';
import { loadSampleModel } from '../samples/loadSample';
import { SAMPLES } from '../samples/sampleIr';
import { useGraphStore } from '../store/graphStore';
import { useTrainingStore } from '../store/trainingStore';
import { useUIStore } from '../store/uiStore';

const PRIMARY_SAMPLES = SAMPLES.filter((s) => s.variant === 'train').slice(0, 6);

export function CanvasEmptyState() {
  const [busy, setBusy] = useState(false);
  const [galleryOpen, setGalleryOpen] = useState(false);
  const setIR = useGraphStore((s) => s.setIR);
  const onboardingVisited = useUIStore((s) => s.onboardingVisited);
  const markOnboardingVisited = useUIStore((s) => s.markOnboardingVisited);

  // オンボーディング: 初回起動(モデル未読込かつ既訪問フラグ無し)は
  // 白紙で放置せずTemplateGalleryを自動表示する(Lobe/NNC型)。
  useEffect(() => {
    if (onboardingVisited) return;
    // 初回のみギャラリーを開いて既訪問フラグを立てる一度きりの同期処理
    // eslint-disable-next-line react-hooks/set-state-in-effect
    setGalleryOpen(true);
    markOnboardingVisited();
  }, [onboardingVisited, markOnboardingVisited]);

  const handleSample = async (key: string) => {
    const sample = SAMPLES.find((s) => s.key === key);
    if (!sample) return;
    setBusy(true);
    try {
      await loadSampleModel(sample);
    } finally {
      setBusy(false);
    }
  };

  const loadDemo = async () => {
    setBusy(true);
    try {
      const res = await createDemoModel(2000, 3);
      setIR(res.ir);
      useTrainingStore.getState().resetAll();
      toast.success('デモモデル(MNIST, k=2000×3層)を生成しました');
    } catch (e) {
      toast.error(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="pointer-events-none absolute inset-0 z-10 flex items-center justify-center">
      <Empty className="pointer-events-auto w-[720px] max-w-[calc(100vw-40px)] items-stretch justify-start gap-0 rounded-lg border bg-card p-0 text-left shadow-lg">
        <EmptyHeader className="max-w-none items-start border-b px-5 py-4 text-left">
          <EmptyTitle role="heading" aria-level={2} className="text-base font-bold">
            サンプルライブラリ
          </EmptyTitle>
          <EmptyDescription className="mt-1 text-xs text-muted-foreground">
            学習用サンプルを読み込むと、推奨データセットと学習設定が自動で適用されます。
          </EmptyDescription>
        </EmptyHeader>
        <div className="grid gap-3 p-5">
          <div className="grid grid-cols-2 gap-2">
            {PRIMARY_SAMPLES.map((s) => (
              <Button
                key={s.key}
                variant="outline"
                className="h-[84px] justify-start rounded-md p-3 text-left"
                disabled={busy}
                onClick={() => void handleSample(s.key)}
              >
                <s.icon className="size-4 shrink-0 text-muted-foreground" />
                <span className="flex min-w-0 flex-1 flex-col items-start gap-1">
                  <span className="flex max-w-full items-center gap-1.5">
                    <span className="truncate font-semibold">{s.label}</span>
                    <Badge
                      variant="outline"
                      className="h-4 rounded px-1 text-[9px] text-[var(--status-ok)]"
                    >
                      学習
                    </Badge>
                  </span>
                  <span className="max-w-full truncate text-[11px] font-normal text-muted-foreground">
                    {s.desc}
                  </span>
                  {s.recommendedDataset && (
                    <span className="max-w-full truncate text-[10px] font-normal text-muted-foreground">
                      {s.recommendedDataset}
                    </span>
                  )}
                </span>
              </Button>
            ))}
          </div>
          <div className="grid grid-cols-3 gap-2 border-t pt-3">
            <Button
              variant="outline"
              className="h-[68px] justify-start rounded-md p-3 text-left"
              disabled={busy}
              onClick={() => setGalleryOpen(true)}
            >
              <Library className="size-4 shrink-0 text-muted-foreground" />
              <span className="flex min-w-0 flex-col items-start">
                <span className="font-semibold">すべてのサンプル</span>
                <span className="max-w-full truncate text-[11px] font-normal text-muted-foreground">
                  参照回路とブロックも選択
                </span>
              </span>
            </Button>
            <Button
              variant="outline"
              className="h-[68px] justify-start rounded-md p-3 text-left"
              disabled={busy}
              onClick={() => void loadDemo()}
            >
              <Sparkles className="size-4 shrink-0 text-muted-foreground" />
              <span className="flex min-w-0 flex-col items-start">
                <span className="font-semibold">デモモデル生成</span>
                <span className="max-w-full truncate text-[11px] font-normal text-muted-foreground">
                  MNIST / k=2000×3層
                </span>
              </span>
            </Button>
            <Button
              variant="outline"
              className="h-[68px] justify-start rounded-md p-3 text-left"
              disabled={busy}
              onClick={() => setGalleryOpen(true)}
            >
              <Blocks className="size-4 shrink-0 text-muted-foreground" />
              <span className="flex min-w-0 flex-col items-start">
                <span className="font-semibold">ブロックから組む</span>
                <span className="max-w-full truncate text-[11px] font-normal text-muted-foreground">
                  MLP / residual / classifier
                </span>
              </span>
            </Button>
          </div>
        </div>
      </Empty>
      <TemplateGallery open={galleryOpen} onOpenChange={setGalleryOpen} />
    </div>
  );
}
