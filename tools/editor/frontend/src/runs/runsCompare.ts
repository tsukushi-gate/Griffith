// run比較の系列整列。step格子が異なる複数runの系列を、
// 共通のx軸(全runのstepの和集合)へnullパディングで整列する(uPlot AlignedData用)。

export interface RunSeries {
  steps: number[];
  values: number[];
}

/** 複数runの系列を共通x軸へ整列する。戻り値: [x, run1値, run2値, ...] */
export function alignRunSeries(runs: RunSeries[]): (number | null)[][] {
  const xSet = new Set<number>();
  for (const r of runs) {
    for (const s of r.steps) xSet.add(s);
  }
  const xs = [...xSet].sort((a, b) => a - b);
  const index = new Map(xs.map((x, i) => [x, i]));

  const out: (number | null)[][] = [xs];
  for (const r of runs) {
    const col: (number | null)[] = new Array<number | null>(xs.length).fill(null);
    r.steps.forEach((s, i) => {
      const at = index.get(s);
      if (at !== undefined) col[at] = r.values[i] ?? null;
    });
    out.push(col);
  }
  return out;
}
