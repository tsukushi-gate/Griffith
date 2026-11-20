// コマンドパレット(⌘K)。cmdkベースのshadcn CommandDialog。

import { useMemo } from 'react';
import {
  CommandDialog,
  CommandEmpty,
  CommandGroup,
  CommandInput,
  CommandItem,
  CommandList,
  CommandShortcut,
} from '@/components/ui/command';
import { useUIStore } from '../store/uiStore';
import { buildCommands } from './registry';

export function CommandPalette() {
  const cmdOpen = useUIStore((s) => s.cmdOpen);
  const setCmdOpen = useUIStore((s) => s.setCmdOpen);

  // 開いた時点のstore状態でコマンド一覧を組み立てる
  const groups = useMemo(() => {
    if (!cmdOpen) return [];
    const commands = buildCommands();
    const byGroup = new Map<string, typeof commands>();
    for (const c of commands) {
      const list = byGroup.get(c.group) ?? [];
      list.push(c);
      byGroup.set(c.group, list);
    }
    return [...byGroup.entries()];
  }, [cmdOpen]);

  return (
    <CommandDialog
      open={cmdOpen}
      onOpenChange={setCmdOpen}
      title="コマンドパレット"
      description="コマンドを検索して実行"
    >
      <CommandInput placeholder="コマンドを検索…" />
      <CommandList>
        <CommandEmpty>該当するコマンドがありません</CommandEmpty>
        {groups.map(([group, commands]) => (
          <CommandGroup key={group} heading={group}>
            {commands.map((c) => (
              <CommandItem
                key={c.id}
                disabled={c.disabled}
                onSelect={() => {
                  setCmdOpen(false);
                  void c.run();
                }}
              >
                {c.label}
                {c.shortcut && <CommandShortcut>{c.shortcut}</CommandShortcut>}
              </CommandItem>
            ))}
          </CommandGroup>
        ))}
      </CommandList>
    </CommandDialog>
  );
}
