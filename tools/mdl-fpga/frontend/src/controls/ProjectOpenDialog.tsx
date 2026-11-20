// プロジェクトを開くダイアログ(タブ: イグザンプル / ファイル)。
// examples はダイアログを開いたときだけフェッチ(ブート時に叩かない)。
// 未ビルドの system イグザンプルは disabled+hint で入口を塞ぐ。
import { useEffect, useState } from 'react';
import { Button } from '@/components/ui/button';
import {
  Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle,
} from '@/components/ui/dialog';
import { Input } from '@/components/ui/input';
import {
  Tabs, TabsContent, TabsList, TabsTrigger,
} from '@/components/ui/tabs';
import type { OpenRequest } from '../api/types';
import { useT } from '../i18n/useT';
import { useProjectStore } from '../store/projectStore';

export function ProjectOpenDialog({
  open,
  onOpenChange,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
}) {
  const t = useT();
  const examples = useProjectStore((s) => s.examples);
  const project = useProjectStore((s) => s.project);
  const opening = useProjectStore((s) => s.opening);
  const error = useProjectStore((s) => s.error);
  const [path, setPath] = useState('');

  useEffect(() => {
    if (open) void useProjectStore.getState().fetchExamples();
  }, [open]);

  const doOpen = async (req: OpenRequest) => {
    const ok = await useProjectStore.getState().open(req);
    if (ok) onOpenChange(false);
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>{t('project.dialog.title')}</DialogTitle>
          <DialogDescription>{t('project.dialog.desc')}</DialogDescription>
        </DialogHeader>
        <Tabs defaultValue="examples">
          <TabsList className="w-full">
            <TabsTrigger value="examples">
              {t('project.tab.examples')}
            </TabsTrigger>
            <TabsTrigger value="file">{t('project.tab.file')}</TabsTrigger>
          </TabsList>
          <TabsContent value="examples">
            <div className="flex max-h-72 flex-col gap-1 overflow-y-auto">
              {examples.flatMap((ex) =>
                ex.modes.map((m) => (
                  <div key={`${ex.key}:${m.mode}`}>
                    <Button
                      size="sm"
                      variant="ghost"
                      className="w-full justify-start"
                      disabled={!m.ready || opening}
                      onClick={() =>
                        void doOpen({ example: ex.key, mode: m.mode })
                      }
                    >
                      <span className="truncate">{ex.name}</span>
                      <span className="ml-auto text-xs text-muted-foreground">
                        {m.mode}
                        {!m.ready ? `(${t('project.notready')})` : ''}
                      </span>
                    </Button>
                    {!m.ready && m.hint ? (
                      <div className="px-3 pb-1 text-xs text-muted-foreground">
                        {m.hint}
                      </div>
                    ) : null}
                  </div>
                )),
              )}
              {examples.length === 0 ? (
                <div className="p-2 text-xs text-muted-foreground">…</div>
              ) : null}
            </div>
          </TabsContent>
          <TabsContent value="file">
            <div className="flex flex-col gap-2">
              <Input
                value={path}
                placeholder={t('project.path.placeholder')}
                onChange={(e) => setPath(e.target.value)}
              />
              <Button
                size="sm"
                disabled={!path.trim() || opening}
                onClick={() => void doOpen({ model: path.trim() })}
              >
                {t('project.open')}
              </Button>
              {project?.recent.length ? (
                <>
                  <div className="mt-1 text-xs text-muted-foreground">
                    {t('project.recent')}
                  </div>
                  {project.recent.map((r) => (
                    <Button
                      key={`${r.model}:${r.system ?? ''}`}
                      size="sm"
                      variant="ghost"
                      className="w-full justify-start"
                      disabled={opening}
                      title={r.model}
                      onClick={() =>
                        void doOpen({
                          model: r.model,
                          system: r.system ?? undefined,
                          pipelined: r.pipelined,
                        })
                      }
                    >
                      <span className="truncate">{r.name ?? r.model}</span>
                    </Button>
                  ))}
                </>
              ) : null}
            </div>
          </TabsContent>
        </Tabs>
        {opening ? (
          <div className="text-xs text-muted-foreground">
            {t('project.opening.hint')}
          </div>
        ) : null}
        {error ? (
          <div className="text-xs text-destructive">{error}</div>
        ) : null}
      </DialogContent>
    </Dialog>
  );
}
