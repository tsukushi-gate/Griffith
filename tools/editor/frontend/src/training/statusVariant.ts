// 学習状態バッジの配色(白テーマ)。TrainControlsと並行run一覧(ActiveRunsPanel)で共用する。
// コンポーネントファイルからは分離する(react-refreshのmixed-export違反/HMR境界の劣化を避ける)。
export function statusVariant(status: string): string {
  if (status === 'running' || status === 'starting')
    return 'text-[var(--status-ok)] border-[var(--status-ok)]/40';
  if (status === 'paused') return 'text-[var(--status-warn)] border-[var(--status-warn)]/40';
  if (status === 'error' || status === 'lost') return 'text-destructive border-destructive/40';
  return 'text-muted-foreground';
}
