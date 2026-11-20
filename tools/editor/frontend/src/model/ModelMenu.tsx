// ヘッダの「モデル」ドロップダウンメニュー(旧ModelPanelの入口を集約)。
// サンプルモデルのサブメニュー(モデル読込後もサンプルへ切替可能)。

import {
  ChevronDown, FileCode, FileDown, FileUp, FolderOpen, Shapes, Sparkles, Wand2,
} from 'lucide-react';
import { Fragment, useState } from 'react';
import { Button } from '@/components/ui/button';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuSub,
  DropdownMenuSubContent,
  DropdownMenuSubTrigger,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu';
import { loadSampleModel } from '../samples/loadSample';
import { SAMPLES } from '../samples/sampleIr';
import { useGraphStore } from '../store/graphStore';
import { ModelDialogs, type ModelDialogKind } from './ModelDialogs';

const SAMPLE_MENU_GROUPS = [
  { key: 'hybrid', label: 'ハイブリッド' },
  { key: 'learning', label: '学習デモ' },
  { key: 'sequence', label: '系列・状態' },
  { key: 'logic', label: '参照回路' },
  { key: 'dag', label: 'DAG' },
  { key: 'vision', label: '画像' },
] as const;

export function ModelMenu() {
  const [dialog, setDialog] = useState<ModelDialogKind>(null);
  const ir = useGraphStore((s) => s.ir);

  return (
    <>
      <DropdownMenu>
        <DropdownMenuTrigger asChild>
          <Button variant="ghost" size="sm">
            モデル
            <ChevronDown className="size-3.5" />
          </Button>
        </DropdownMenuTrigger>
        <DropdownMenuContent align="start" className="w-64">
          <DropdownMenuSub>
            <DropdownMenuSubTrigger>
              <Shapes className="mr-2 size-4" />
              サンプルを開く
            </DropdownMenuSubTrigger>
            <DropdownMenuSubContent className="w-64">
              {SAMPLE_MENU_GROUPS.map((group, index) => {
                const samples = SAMPLES.filter((s) => s.category === group.key);
                if (samples.length === 0) return null;
                return (
                  <Fragment key={group.key}>
                    {index > 0 && <DropdownMenuSeparator />}
                    <DropdownMenuLabel className="text-[11px] text-muted-foreground">
                      {group.label}
                    </DropdownMenuLabel>
                    {samples.map((s) => (
                      <DropdownMenuItem
                        key={s.key}
                        onSelect={() => void loadSampleModel(s)}
                      >
                        <s.icon className="size-4" />
                        <span>{s.label}</span>
                        <span className="ml-auto truncate pl-2 text-[10px] text-muted-foreground">
                          {s.variant === 'train' ? '学習' : s.variant === 'reference' ? '参照' : 'デモ'}
                        </span>
                      </DropdownMenuItem>
                    ))}
                  </Fragment>
                );
              })}
            </DropdownMenuSubContent>
          </DropdownMenuSub>
          <DropdownMenuItem onSelect={() => setDialog('demo')}>
            <Sparkles className="size-4" />
            デモモデル生成 (MNIST)…
          </DropdownMenuItem>
          <DropdownMenuItem onSelect={() => setDialog('ckpt')}>
            <FolderOpen className="size-4" />
            チェックポイント読み込み…
          </DropdownMenuItem>
          <DropdownMenuItem onSelect={() => setDialog('python')}>
            <FileCode className="size-4" />
            Pythonモデル読み込み…
          </DropdownMenuItem>
          <DropdownMenuItem onSelect={() => setDialog('convert')}>
            <Wand2 className="size-4" />
            インポート&変換 (事前学習→回路)…
          </DropdownMenuItem>
          <DropdownMenuSeparator />
          <DropdownMenuItem disabled={!ir} onSelect={() => setDialog('ir-save')}>
            <FileDown className="size-4" />
            IR保存…
          </DropdownMenuItem>
          <DropdownMenuItem onSelect={() => setDialog('ir-load')}>
            <FileUp className="size-4" />
            IR読込…
          </DropdownMenuItem>
          {ir && (
            <>
              <DropdownMenuSeparator />
              <div className="px-2 py-1.5 text-[11px] text-muted-foreground">
                {String(ir.metadata.name ?? 'model')} ・ ノード{ir.nodes.length}個
              </div>
            </>
          )}
        </DropdownMenuContent>
      </DropdownMenu>
      <ModelDialogs kind={dialog} onClose={() => setDialog(null)} />
    </>
  );
}
