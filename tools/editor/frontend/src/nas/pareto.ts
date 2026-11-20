// パレートフロント抽出の純ロジック。
// 目的: hard acc を最大化、total_luts(回路面積)を最小化する2目的の非劣解集合。
// 裁定6: partial:true(blackbox含有=面積過小評価)の trial は判定から除外する。
// state!=='done' の trial も除外する。

import type { NasTrialRow } from './types';

/** a が b を支配するか(accは大きい方が良い / total_lutsは小さい方が良い)。 */
function dominates(a: NasTrialRow, b: NasTrialRow): boolean {
  const betterOrEq = a.acc >= b.acc && a.total_luts <= b.total_luts;
  const strictly = a.acc > b.acc || a.total_luts < b.total_luts;
  return betterOrEq && strictly;
}

/** 判定対象(done かつ partial:false)のみ抽出する。 */
export function paretoCandidates(trials: NasTrialRow[]): NasTrialRow[] {
  return trials.filter((t) => t.state === 'done' && !t.partial);
}

/** 非劣解の trial_id 集合(昇順)を返す。 */
export function paretoFront(trials: NasTrialRow[]): number[] {
  const cands = paretoCandidates(trials);
  const front: number[] = [];
  for (const t of cands) {
    const dominated = cands.some((o) => o.trial_id !== t.trial_id && dominates(o, t));
    if (!dominated) front.push(t.trial_id);
  }
  return front.sort((x, y) => x - y);
}

/** パレート集合を acc昇順(=total_luts昇順)に並べたフロント線用の点列を返す。 */
export function paretoLine(trials: NasTrialRow[]): NasTrialRow[] {
  const ids = new Set(paretoFront(trials));
  return trials
    .filter((t) => ids.has(t.trial_id))
    .sort((a, b) => a.total_luts - b.total_luts || a.acc - b.acc);
}

/**
 * 現在編集中モデルの (acc, total_luts) がパレートフロントに対して
 * 前方(=非劣: フロントを更新しうる)か劣後かを判定する。
 * acc未知(null)の場合は total_luts のみで「同面積以下で最良accを超えるか」を測れないため
 * dominated=false(=前方扱いにしない)を返し、position='unknown' とする。
 */
export function positionVsPareto(
  current: { acc: number | null; total_luts: number },
  trials: NasTrialRow[],
): { position: 'ahead' | 'behind' | 'unknown'; dominatedBy: number | null } {
  const cands = paretoCandidates(trials);
  if (cands.length === 0 || current.acc === null) {
    return { position: 'unknown', dominatedBy: null };
  }
  const me: NasTrialRow = {
    trial_id: -1,
    acc: current.acc,
    best_acc: current.acc,
    total_luts: current.total_luts,
    logic_depth: 0,
    gates: 0,
    num_params: 0,
    partial: false,
    state: 'done',
    parent_id: null,
    genome: { layers: 0, width: 0, n_inputs: 0 },
  };
  const dominator = cands.find((o) => dominates(o, me));
  if (dominator) return { position: 'behind', dominatedBy: dominator.trial_id };
  return { position: 'ahead', dominatedBy: null };
}
