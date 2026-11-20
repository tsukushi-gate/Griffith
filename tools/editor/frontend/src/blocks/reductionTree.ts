// 付録E#9: BinaryBrain流 6-fanin 縮約ツリーの自動生成(lut_layer列ヘルパ)。
// 幅 W → ceil(W/k) → … と fanin=k で段階的に縮約する lut_layer の直列を生成する。
// 各層の出力ゲートは、前層の「局所窓」から k 本の入力を選ぶ(BinaryBrainの
// スパース局所結線に倣う)。indices/truth_tables の生成は nodeInit のプリセットへ委譲。

import { lutInit, type InitPreset } from './nodeInit';
import type { IdAllocator, SubgraphTemplate } from './templates';
import type { IREdgeJson, IRNodeJson } from '../ir/schema';

const COL_X = 220;
const pos = (col: number) => ({ x: col * COL_X, y: 0 });

export interface ReductionTreeOptions {
  fanin?: number; // 1層あたりの縮約率(=n_inputs)。既定6(BinaryBrain)
  minWidth?: number; // これ以下になったら縮約を止める(既定=fanin)
  preset?: InitPreset; // indices/tt初期化プリセット(既定'random')
  seed?: number; // 決定的生成のseed(テスト用)。省略時はランダム窓
}

/** 縮約ツリーの各層の幅列を返す(inWidth を先頭に含む)。 */
export function reductionWidths(
  inWidth: number,
  fanin: number,
  minWidth: number,
): number[] {
  const widths = [inWidth];
  let w = inWidth;
  // 進捗が無い(w<=minWidth もしくは縮約で減らない)場合は停止
  while (w > minWidth) {
    const next = Math.max(minWidth, Math.ceil(w / fanin));
    if (next >= w) break;
    widths.push(next);
    w = next;
  }
  return widths;
}

/**
 * 局所ランダム窓で lut_layer 列(縮約ツリー)を生成する。
 * 返り値は addSubgraph へ渡せる SubgraphTemplate。層が1つも生成されない
 * (inWidth<=minWidth)場合でも、恒等な単層を1つ返して entry/exit を保証する。
 */
export function reductionTreeLuts(
  inWidth: number,
  alloc: IdAllocator,
  opts: ReductionTreeOptions = {},
): SubgraphTemplate {
  const fanin = Math.max(2, opts.fanin ?? 6);
  const minWidth = Math.max(1, opts.minWidth ?? fanin);
  const preset = opts.preset ?? 'random';
  const widths = reductionWidths(inWidth, fanin, minWidth);

  const nodes: IRNodeJson[] = [];
  const edges: IREdgeJson[] = [];

  // 縮約が発生しない場合は幅維持の単層(恒等)を1つ置く
  const layerWidths =
    widths.length >= 2 ? widths : [inWidth, Math.max(1, Math.min(inWidth, minWidth))];

  let rngState = (opts.seed ?? 1) >>> 0;
  const rng = () => {
    // mulberry32 相当(nodeInit.mulberryと同系。ここでは層ごとに前進)
    rngState = (rngState + 0x6d2b79f5) | 0;
    let t = Math.imul(rngState ^ (rngState >>> 15), 1 | rngState);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };

  for (let i = 0; i < layerWidths.length - 1; i++) {
    const li = layerWidths[i];
    const lo = layerWidths[i + 1];
    const nInputs = Math.min(fanin, li);
    const { indices, truth_tables } = lutInit(li, lo, nInputs, preset, rng);
    // 局所窓に寄せる: 出力 o の入力を o に比例した窓へ再配置(random時のみ)
    if (preset === 'random') {
      localizeIndices(indices, li, lo, nInputs, rng);
    }
    const id = alloc('rtlut');
    nodes.push({
      id,
      type: 'lut_layer',
      params: { in_width: li, out_width: lo, n_inputs: nInputs, indices, truth_tables },
      train_info: {
        parameterization: 'softmax',
        connections: 'local',
        selection: 'soft',
        initialized: preset !== 'random',
      },
      meta: { editor: { position: pos(i) } },
    });
    if (i > 0) edges.push({ src: nodes[i - 1].id, dst: id });
  }

  return {
    nodes,
    edges,
    entryId: nodes[0].id,
    exitId: nodes[nodes.length - 1].id,
  };
}

/** 出力 o の入力群を、位置 o に対応する局所窓 [start, start+window) へ寄せる。 */
function localizeIndices(
  indices: number[][],
  inWidth: number,
  outWidth: number,
  nInputs: number,
  rng: () => number,
): void {
  const window = Math.min(inWidth, Math.max(nInputs, Math.ceil((inWidth / outWidth) * nInputs)));
  for (let o = 0; o < outWidth; o++) {
    const center = outWidth > 1 ? Math.floor((o / (outWidth - 1)) * (inWidth - window)) : 0;
    const start = Math.max(0, Math.min(inWidth - window, center));
    const seen = new Set<number>();
    for (let j = 0; j < nInputs; j++) {
      let v = start + Math.floor(rng() * window);
      // 窓内で重複回避(窓がnInputs未満なら重複許容)
      let guard = 0;
      while (seen.has(v) && guard < window) {
        v = start + Math.floor(rng() * window);
        guard++;
      }
      seen.add(v);
      indices[j][o] = Math.min(inWidth - 1, v);
    }
  }
}
