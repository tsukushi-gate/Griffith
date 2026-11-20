// cmdkによる検索パレット(⌘K / Ctrl+K)。タイトル・id・カテゴリ・本文を対象に引く。

import { Command } from 'cmdk';
import { useEffect, useMemo, useState } from 'react';
import { TREE, useAppStore } from '../store';
import { searchText } from '../tree/graph';
import { isCategory, isPaper, isReport } from '../types';

interface Entry {
  id: string;
  title: string;
  hint: string;
  kind: string;
  text: string;
}

function buildEntries(): Entry[] {
  return TREE.nodes
    .filter((n) => n.kind !== 'root')
    .map((n) => ({
      id: n.id,
      title: n.title,
      kind: n.kind,
      hint: isPaper(n)
        ? `${n.date?.slice(0, 4) ?? '----'} · ${n.categories.join(', ')} · ${n.status}`
        : isReport(n)
          ? `report · ${n.branch ?? ''}`
          : isCategory(n)
            ? 'category'
            : '',
      text: searchText(n),
    }));
}

export function SearchPalette() {
  const open = useAppStore((s) => s.searchOpen);
  const setOpen = useAppStore((s) => s.setSearchOpen);
  const focusNode = useAppStore((s) => s.focusNode);
  const entries = useMemo(buildEntries, []);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'k' && (e.metaKey || e.ctrlKey)) {
        e.preventDefault();
        setOpen(!useAppStore.getState().searchOpen);
      }
      if (e.key === 'Escape') setOpen(false);
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [setOpen]);

  if (!open) return null;

  return (
    <div
      className="fixed inset-0 z-50 flex items-start justify-center pt-[12vh]"
      style={{ background: 'rgba(15,23,42,0.45)' }}
      onClick={() => setOpen(false)}
      data-testid="search-overlay"
    >
      <div
        className="w-[640px] max-w-[92vw] overflow-hidden rounded-lg border shadow-xl"
        style={{ borderColor: 'var(--border)', background: 'var(--card)' }}
        onClick={(e) => e.stopPropagation()}
      >
        <Command
          // 本文まで含む長いテキストを対象にするため、フィルタは自前の部分一致にする
          shouldFilter={false}
          loop
        >
          <SearchBody
            entries={entries}
            onPick={(id) => {
              focusNode(id);
              setOpen(false);
            }}
          />
        </Command>
      </div>
    </div>
  );
}

function SearchBody({ entries, onPick }: { entries: Entry[]; onPick: (id: string) => void }) {
  const [query, setQuery] = useState('');
  const results = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (!q) return entries.slice(0, 50);
    const terms = q.split(/\s+/);
    return entries.filter((e) => terms.every((t) => e.text.includes(t))).slice(0, 50);
  }, [entries, query]);

  return (
    <>
      <Command.Input
        autoFocus
        value={query}
        onValueChange={setQuery}
        placeholder="論文・レポートを検索(タイトル/id/カテゴリ/本文)"
        className="w-full border-b px-4 py-3 text-[13px] outline-none"
        style={{ borderColor: 'var(--border)', background: 'transparent', color: 'var(--foreground)' }}
        data-testid="search-input"
      />
      <Command.List className="max-h-[50vh] overflow-y-auto p-1" data-testid="search-list">
        {results.length === 0 && (
          <div className="p-4 text-[12px]" style={{ color: 'var(--muted-foreground)' }}>
            該当なし
          </div>
        )}
        {results.map((e) => (
          <Command.Item
            key={e.id}
            value={e.id}
            onSelect={() => onPick(e.id)}
            className="cursor-pointer rounded px-3 py-2"
            data-testid="search-item"
          >
            <div className="truncate text-[13px]">{e.title}</div>
            <div className="truncate text-[10px]" style={{ color: 'var(--muted-foreground)' }}>
              {e.id} · {e.hint}
            </div>
          </Command.Item>
        ))}
      </Command.List>
    </>
  );
}
