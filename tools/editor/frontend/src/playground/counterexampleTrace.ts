// 反例トレース(§4-2)の純粋ロジック。
// 真理値表マトリクスで期待値≠実出力の行(反例)を選んだとき、
//   1. 誤った出力ポートから入力側へ遡る「到達コーン」(ファンイン集合)を求める
//   2. コーン上のlut_layerゲートを、soft/hardのビット食い違い・soft出力の迷い
//      (二値エントロピー)で順位付けし、原因候補として並べる
// を提供する。canvasの着色・原因リスト表示はこの結果を消費するだけ(表示と分離)。

import type { IRGraphJson } from '../ir/schema';

/** 到達コーン: 誤出力に寄与するノード集合とエッジ集合。 */
export interface TraceCone {
  /** コーン上のIRノードid(入力ポートidを含む) */
  nodes: Set<string>;
  /** コーン上のエッジ(`${src}->${dst}`規約) */
  edges: Set<string>;
}

/** 原因候補ゲート(1ニューロン)の評価結果。 */
export interface CauseCandidate {
  /** 所属するlut_layerノードid(focusNodeのジャンプ先) */
  nodeId: string;
  /** ノード内のニューロン番号 */
  neuron: number;
  /** softのビット(>=0.5)とhardのビットが食い違うか(強い原因シグナル) */
  disagreement: boolean;
  /** soft出力値の二値エントロピー(0=確定 / 1=最大の迷い) */
  entropy: number;
  /** 順位付けスコア(降順): 食い違い(+1)を最優先し、迷いで細分化 */
  score: number;
}

const BIT_ON_THRESHOLD = 0.5;
const ENTROPY_EPSILON = 1e-3;

/** dst→接続元ノードid配列(v0.1エッジはslot昇順・宣言順で安定) */
function buildInSources(ir: IRGraphJson): Map<string, string[]> {
  const grouped = new Map<string, { src: string; slot: number }[]>();
  for (const e of ir.edges) {
    const slot = (e as { dst_slot?: number }).dst_slot ?? 0;
    const list = grouped.get(e.dst) ?? [];
    list.push({ src: e.src, slot });
    grouped.set(e.dst, list);
  }
  const inSrcs = new Map<string, string[]>();
  for (const [dst, list] of grouped) {
    inSrcs.set(
      dst,
      [...list].sort((a, b) => a.slot - b.slot).map((x) => x.src),
    );
  }
  return inSrcs;
}

/**
 * 出力ビット列(全出力ポート連結)と期待値を比較し、
 * ビットが食い違う出力ポートの接続元ノードidを返す(到達コーンの起点)。
 */
export function mismatchedOutputTargets(
  ir: IRGraphJson,
  outputBits: ReadonlyArray<number>,
  expectedBits: ReadonlyArray<number>,
): Set<string> {
  const inSrcs = buildInSources(ir);
  const targets = new Set<string>();
  let offset = 0;
  for (const port of ir.outputs) {
    let mismatched = false;
    for (let i = 0; i < port.width; i++) {
      if (outputBits[offset + i] !== expectedBits[offset + i]) {
        mismatched = true;
        break;
      }
    }
    if (mismatched) {
      const src = inSrcs.get(port.id)?.[0];
      if (src) targets.add(src);
    }
    offset += port.width;
  }
  return targets;
}

/**
 * 起点ノード集合から入力側へ遡る到達コーンを求める(後方BFS)。
 * コーン=その出力に寄与しうる全祖先ノード+その間のエッジ。
 */
export function computeReachabilityCone(
  ir: IRGraphJson,
  targets: Iterable<string>,
): TraceCone {
  const inSrcs = buildInSources(ir);
  const nodes = new Set<string>();
  const edges = new Set<string>();
  const queue: string[] = [];
  for (const t of targets) {
    if (!nodes.has(t)) {
      nodes.add(t);
      queue.push(t);
    }
  }
  while (queue.length > 0) {
    const cur = queue.pop() as string;
    for (const src of inSrcs.get(cur) ?? []) {
      edges.add(`${src}->${cur}`);
      if (!nodes.has(src)) {
        nodes.add(src);
        queue.push(src);
      }
    }
  }
  return { nodes, edges };
}

/** 確率pの二値エントロピー(bit単位、0〜1)。p<=0またはp>=1は0。 */
export function binaryEntropy(p: number): number {
  if (!Number.isFinite(p) || p <= 0 || p >= 1) return 0;
  return -(p * Math.log2(p) + (1 - p) * Math.log2(1 - p));
}

/**
 * コーン上のlut_layerゲートを原因候補として順位付けする。
 * hard/softは同一入力パターンでの全ノード出力(evalCircuit.valuesのnodeId→値)。
 * スコア=食い違い(1 or 0)+ soft出力の迷い。
 * 「食い違いあり」を必ず「食い違いなし」より上位にし、同群は迷いで並べる。
 * 迷いも食い違いも無いゲート(entropy≈0かつ一致)は候補から除く。
 */
export function rankCauseCandidates(
  ir: IRGraphJson,
  cone: TraceCone,
  hardValues: Record<string, ArrayLike<number>>,
  softValues: Record<string, ArrayLike<number>>,
): CauseCandidate[] {
  const candidates: CauseCandidate[] = [];
  for (const node of ir.nodes) {
    if (node.type !== 'lut_layer') continue;
    if (!cone.nodes.has(node.id)) continue;
    const hard = hardValues[node.id];
    const soft = softValues[node.id];
    if (!hard || !soft) continue;
    const count = Math.min(hard.length, soft.length);
    for (let o = 0; o < count; o++) {
      const softVal = soft[o];
      const hardBit = hard[o] >= BIT_ON_THRESHOLD;
      const softBit = softVal >= BIT_ON_THRESHOLD;
      const disagreement = hardBit !== softBit;
      const entropy = binaryEntropy(softVal);
      if (!disagreement && entropy < ENTROPY_EPSILON) continue;
      candidates.push({
        nodeId: node.id,
        neuron: o,
        disagreement,
        entropy,
        score: (disagreement ? 1 : 0) + entropy,
      });
    }
  }
  candidates.sort((a, b) => b.score - a.score);
  return candidates;
}
