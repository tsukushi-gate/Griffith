// プロジェクトペイン(左サイドバー最上部): 現在のプロジェクト表示+「開く」。
// 搭載可能性(advise)の要約はボード選択とセットのため BoardPanel 側に表示。
import { useEffect, useState } from 'react';
import { FolderOpen } from 'lucide-react';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { useT } from '../i18n/useT';
import { useProjectStore } from '../store/projectStore';
import { useSimStore } from '../store/simStore';
import { ProjectOpenDialog } from './ProjectOpenDialog';

export function ProjectPanel() {
  const t = useT();
  const meta = useSimStore((s) => s.meta);
  const project = useProjectStore((s) => s.project);
  const opening = useProjectStore((s) => s.opening);
  const [dialogOpen, setDialogOpen] = useState(false);

  useEffect(() => {
    void useProjectStore.getState().refresh();
  }, []);

  return (
    <div className="rounded-lg border bg-[var(--node-bg)] p-3">
      <div className="mb-2 flex items-center justify-between">
        <span className="text-xs text-muted-foreground">
          {t('project.title')}
        </span>
        <Badge variant="outline">
          {meta?.mode === 'system' ? t('header.system') : 'core'}
        </Badge>
      </div>
      <div className="truncate text-sm font-medium"
           title={project?.model ?? undefined}>
        {meta?.name ?? '—'}
      </div>
      {meta?.pipelined ? (
        <div className="text-xs text-muted-foreground">pipelined</div>
      ) : null}
      <Button
        size="sm"
        variant="outline"
        className="mt-2 w-full"
        disabled={opening}
        onClick={() => setDialogOpen(true)}
      >
        <FolderOpen />
        {opening ? t('project.opening') : t('project.open')}
      </Button>
      <ProjectOpenDialog open={dialogOpen} onOpenChange={setDialogOpen} />
    </div>
  );
}
