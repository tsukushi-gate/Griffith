// 上部ツールバー: 検索起動・カテゴリ/年/statusフィルタ・折りたたみ一括・テーマ切替・統計。

import { useMemo, useState } from 'react';
import { TREE, useAppStore } from './store';
import { categoryList } from './tree/graph';

const btn =
  'rounded border px-2 py-1 text-[11px] hover:opacity-80 disabled:opacity-40 whitespace-nowrap';
const btnStyle = { borderColor: 'var(--border)', background: 'var(--card)' } as const;

function CategoryFilter() {
  const filters = useAppStore((s) => s.filters);
  const setFilters = useAppStore((s) => s.setFilters);
  const [open, setOpen] = useState(false);
  const cats = useMemo(() => categoryList(TREE), []);
  const selected = filters.categories;

  return (
    <div className="relative">
      <button
        type="button"
        className={btn}
        style={btnStyle}
        onClick={() => setOpen((v) => !v)}
        data-testid="category-filter-button"
      >
        カテゴリ{selected.size > 0 ? ` (${selected.size})` : ' (全て)'}
      </button>
      {open && (
        <div
          className="absolute z-40 mt-1 max-h-[60vh] w-64 overflow-y-auto rounded border p-1 shadow-lg"
          style={{ borderColor: 'var(--border)', background: 'var(--card)' }}
          data-testid="category-filter-menu"
        >
          {cats.map((c) => (
            <label
              key={c.id}
              className="flex cursor-pointer items-center gap-2 rounded px-2 py-1 text-[12px] hover:opacity-80"
            >
              <input
                type="checkbox"
                checked={selected.has(c.id)}
                onChange={() => {
                  const next = new Set(selected);
                  if (next.has(c.id)) next.delete(c.id);
                  else next.add(c.id);
                  setFilters({ categories: next });
                }}
              />
              <span className="truncate">{c.label}</span>
            </label>
          ))}
          <button
            type="button"
            className={`${btn} m-1`}
            style={btnStyle}
            onClick={() => setFilters({ categories: new Set() })}
          >
            クリア
          </button>
        </div>
      )}
    </div>
  );
}

export function Toolbar() {
  const filters = useAppStore((s) => s.filters);
  const setFilters = useAppStore((s) => s.setFilters);
  const resetFilters = useAppStore((s) => s.resetFilters);
  const setSearchOpen = useAppStore((s) => s.setSearchOpen);
  const setAllCollapsed = useAppStore((s) => s.setAllCollapsed);
  const theme = useAppStore((s) => s.theme);
  const toggleTheme = useAppStore((s) => s.toggleTheme);
  const lod = useAppStore((s) => s.lod);
  const catIds = useMemo(() => categoryList(TREE).map((c) => c.id), []);

  return (
    <header
      className="flex shrink-0 flex-wrap items-center gap-2 border-b px-3 py-2"
      style={{ borderColor: 'var(--border)', background: 'var(--panel-bg)' }}
      data-testid="toolbar"
    >
      <span className="mr-1 text-[13px] font-bold">Paper Tree</span>

      <button
        type="button"
        className={btn}
        style={btnStyle}
        onClick={() => setSearchOpen(true)}
        data-testid="search-button"
      >
        検索 ⌘K
      </button>

      <CategoryFilter />

      <select
        className={btn}
        style={btnStyle}
        value={filters.status}
        onChange={(e) => setFilters({ status: e.target.value as typeof filters.status })}
        data-testid="status-filter"
        aria-label="status"
      >
        <option value="all">status: 全て</option>
        <option value="summarized">summarized</option>
        <option value="stub">stub</option>
      </select>

      <label className="flex items-center gap-1 text-[11px]">
        年
        <input
          type="number"
          className={`${btn} w-[74px]`}
          style={btnStyle}
          placeholder="min"
          value={filters.yearMin ?? ''}
          onChange={(e) => setFilters({ yearMin: e.target.value ? Number(e.target.value) : null })}
          data-testid="year-min"
        />
        〜
        <input
          type="number"
          className={`${btn} w-[74px]`}
          style={btnStyle}
          placeholder="max"
          value={filters.yearMax ?? ''}
          onChange={(e) => setFilters({ yearMax: e.target.value ? Number(e.target.value) : null })}
          data-testid="year-max"
        />
      </label>

      <label className="flex items-center gap-1 text-[11px]">
        <input
          type="checkbox"
          checked={filters.showReports}
          onChange={(e) => setFilters({ showReports: e.target.checked })}
          data-testid="show-reports"
        />
        レポート表示
      </label>

      <button type="button" className={btn} style={btnStyle} onClick={resetFilters}>
        フィルタ解除
      </button>

      <span className="mx-1" style={{ color: 'var(--border)' }}>
        |
      </span>
      <button
        type="button"
        className={btn}
        style={btnStyle}
        onClick={() => setAllCollapsed(catIds, true)}
        data-testid="collapse-all"
      >
        全て折りたたむ
      </button>
      <button
        type="button"
        className={btn}
        style={btnStyle}
        onClick={() => setAllCollapsed(catIds, false)}
        data-testid="expand-all"
      >
        全て開く
      </button>

      <div className="ml-auto flex items-center gap-2">
        <span className="text-[11px]" style={{ color: 'var(--muted-foreground)' }}>
          LOD: {lod} / 論文 {TREE.stats.papers}(stub {TREE.stats.stubs}) / レポート{' '}
          {TREE.stats.reports} / エッジ {TREE.stats.edges}
        </span>
        <button
          type="button"
          className={btn}
          style={btnStyle}
          onClick={toggleTheme}
          data-testid="theme-toggle"
        >
          {theme === 'dark' ? '☾ dark' : '☀ light'}
        </button>
      </div>
    </header>
  );
}
