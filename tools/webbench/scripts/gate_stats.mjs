#!/usr/bin/env node
// conv2d 重み共有木の命令列化(convProgram.ts)によるゲート削減量を
// public/models/ のモデルJSONについて集計して表示する。
//
//   node scripts/gate_stats.mjs                # 既定5モデル
//   node scripts/gate_stats.mjs det seg        # タスク名指定
//   node scripts/gate_stats.mjs --json         # 機械可読出力
//
// convProgram.ts は実行時importを持たない純TSなので、Node の type stripping
// (Node>=22.18/23、.ts を拡張子付きで import)でそのまま読める。
import { readFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

import { buildConvProgram, verifyConvProgram } from '../src/engines/convProgram.ts';

const here = dirname(fileURLToPath(import.meta.url));
const root = resolve(here, '..');

const MODELS = {
  det: 'det_multimnist_lut_k256.json',
  detmono: 'det_multimnist_mono_lut_k256.json',
  detface: 'det_face_lut_k256.json',
  detreal: 'det_fewclass_lut_k64.json',
  seg: 'seg_convdlgn_k32.json',
};

/** hexTT.ts::ttParse と同じ規約('0x…' または裸のhex)で 0..15 の数値へ。 */
function ttNum(hex) {
  const s = String(hex).trim().toLowerCase();
  const v = Number(s.startsWith('0x') ? s : `0x${s}`);
  return Number.isFinite(v) ? v & 0xf : 0;
}

function specOf(params) {
  const nInternal = (1 << params.tree_depth) - 1;
  const tt = new Uint8Array(params.out_channels * nInternal);
  for (let i = 0; i < tt.length; i++) tt[i] = ttNum(params.truth_tables[i]);
  return {
    outChannels: params.out_channels,
    treeDepth: params.tree_depth,
    indicesFlat: params.indices_flat,
    truthTables: tt,
  };
}

/** タップ平面の枚数(bitplane.ts の dedupキーと同じ: 高速路 (c,dx)、
    strided路 (c,dy,dx))。命令列化で未使用タップが落ちるぶん、実体化する
    平面も減る = カーネルの「タップ収集」コストがそのまま減る。 */
function planeCount(params, tapValues) {
  const [, , W] = params.in_shape;
  const [kH, kW] = Array.isArray(params.kernel_size)
    ? params.kernel_size : [params.kernel_size, params.kernel_size];
  const [sH, sW] = Array.isArray(params.stride) ? params.stride : [params.stride, params.stride];
  const [, pW] = Array.isArray(params.padding) ? params.padding : [params.padding, params.padding];
  const WOut = Math.floor((W + 2 * pW - kW) / sW) + 1;
  const fast = sH === 1 && sW === 1 && WOut === W;
  const khw = kH * kW;
  const keys = new Set();
  for (const pp of tapValues) {
    const rem = pp % khw;
    const c = Math.floor(pp / khw);
    const dy = Math.floor(rem / kW);
    const dx = rem % kW;
    keys.add(fast ? c * kW + dx : c * khw + dy * kW + dx);
  }
  return keys.size;
}

/** [A3判断用] 残った命令の構造を見る:
    - ttヒストグラム(縮約後に残る2入力ゲートの種類分布)
    - 同一オペランド対 (a,b) を共有する命令の数(= 同じ2入力から別関数を
      作っている箇所。NPN同値クラスで見れば重複しており、don't care 再合成や
      「出力反転を親へ吸収する」といった追加最適化の余地を示す)
    - そのうち「出力反転だけの差」(tt2 = ~tt1 & 0xf)である対の数(NPN重複) */
function residualStructure(prog) {
  const ttHist = new Array(16).fill(0);
  const byPair = new Map();
  for (const ins of prog.instrs) {
    ttHist[ins.tt & 0xf]++;
    const key = `${ins.a}:${ins.b}`;
    if (!byPair.has(key)) byPair.set(key, []);
    byPair.get(key).push(ins.tt & 0xf);
  }
  let sharedPairInstrs = 0;
  let npnDupPairs = 0;
  for (const tts of byPair.values()) {
    if (tts.length < 2) continue;
    sharedPairInstrs += tts.length;
    for (let i = 0; i < tts.length; i++) {
      for (let j = i + 1; j < tts.length; j++) {
        if (tts[j] === (~tts[i] & 0xf)) npnDupPairs++;
      }
    }
  }
  return { ttHist, sharedPairInstrs, npnDupPairs, distinctPairs: byPair.size };
}

function analyzeModel(task, file) {
  const doc = JSON.parse(readFileSync(resolve(root, 'public/models', file), 'utf8'));
  const nodes = [];
  const total = {
    gatesBefore: 0, gatesAfter: 0, constFolded: 0, aliasFolded: 0, notFolded: 0,
    cseHits: 0, emitted: 0, deadRemoved: 0, notInstrs: 0, tapsBefore: 0, nTaps: 0, nSlots: 0,
    ocConst: 0, ocTap: 0, planesBefore: 0, planesAfter: 0,
  };
  let verifyMs = 0;
  const ttHist = new Array(16).fill(0);
  let sharedPairInstrs = 0;
  let npnDupPairs = 0;
  for (const node of doc.ir.nodes) {
    if (node.type !== 'conv2d') continue;
    if (node.params.n_inputs !== 2) throw new Error(`${task}/${node.id}: n_inputs!=2`);
    const spec = specOf(node.params);
    const t0 = performance.now();
    const prog = buildConvProgram(spec);
    const t1 = performance.now();
    verifyConvProgram(spec, prog);
    verifyMs += performance.now() - t1;
    const s = {
      ...prog.stats,
      planesBefore: planeCount(node.params, node.params.indices_flat.flat()),
      planesAfter: planeCount(node.params, prog.taps),
    };
    const rs = residualStructure(prog);
    for (let i = 0; i < 16; i++) ttHist[i] += rs.ttHist[i];
    sharedPairInstrs += rs.sharedPairInstrs;
    npnDupPairs += rs.npnDupPairs;
    nodes.push({ id: node.id, oc: node.params.out_channels, buildMs: t1 - t0, ...s });
    for (const k of Object.keys(total)) total[k] += s[k] ?? 0;
  }
  return { task, file, nodes, total, verifyMs, ttHist, sharedPairInstrs, npnDupPairs };
}

function pct(a, b) {
  return b === 0 ? '   n/a' : `${((1 - a / b) * 100).toFixed(1)}%`;
}

const args = process.argv.slice(2);
const asJson = args.includes('--json');
// `--variant=_ll` / `--variant=_dc` で A3 派生モデル(lossless張り替え /
// don't care再合成)のゲート数を同じ集計器で出す。URLクエリ `?model=_ll` と
// 同じ規約(.json の直前へサフィックスを挿入)にそろえてある。
const variantArg = args.find((a) => a.startsWith('--variant='));
const variant = variantArg ? variantArg.slice('--variant='.length) : '';
if (variant && !/^_[A-Za-z0-9_-]+$/.test(variant)) {
  console.error(`不正な --variant: ${variant}(先頭 '_' の英数字のみ)`);
  process.exit(1);
}
const tasks = args.filter((a) => !a.startsWith('--'));
const targets = tasks.length > 0 ? tasks : Object.keys(MODELS);

const results = [];
for (const task of targets) {
  const base = MODELS[task];
  if (!base) {
    console.error(`未知のタスク: ${task} (有効: ${Object.keys(MODELS).join(', ')})`);
    process.exit(1);
  }
  const file = variant ? base.replace(/\.json$/, `${variant}.json`) : base;
  results.push(analyzeModel(variant ? `${task}${variant}` : task, file));
}

if (asJson) {
  console.log(JSON.stringify(results, null, 2));
} else {
  for (const r of results) {
    console.log(`\n=== ${r.task} (${r.file}) ===`);
    console.log(
      'node                oc  before   after  削減   const  alias    not    cse   dead   tap全  tap使 slots',
    );
    for (const n of r.nodes) {
      console.log(
        `${n.id.padEnd(18)}${String(n.oc).padStart(5)}`
        + `${String(n.gatesBefore).padStart(8)}${String(n.gatesAfter).padStart(8)}`
        + `${pct(n.gatesAfter, n.gatesBefore).padStart(7)}`
        + `${String(n.constFolded).padStart(8)}${String(n.aliasFolded).padStart(7)}`
        + `${String(n.notFolded).padStart(7)}${String(n.cseHits).padStart(7)}`
        + `${String(n.deadRemoved).padStart(7)}${String(n.tapsBefore).padStart(7)}`
        + `${String(n.nTaps).padStart(7)}${String(n.nSlots).padStart(6)}`,
      );
    }
    const t = r.total;
    console.log(
      `${'合計'.padEnd(21)}${String(t.gatesBefore).padStart(8)}${String(t.gatesAfter).padStart(8)}`
      + `${pct(t.gatesAfter, t.gatesBefore).padStart(7)}`
      + `${String(t.constFolded).padStart(8)}${String(t.aliasFolded).padStart(7)}`
      + `${String(t.notFolded).padStart(7)}${String(t.cseHits).padStart(7)}`
      + `${String(t.deadRemoved).padStart(7)}${String(t.tapsBefore).padStart(7)}`
      + `${String(t.nTaps).padStart(7)}${String(t.nSlots).padStart(6)}`,
    );
    console.log(
      `  内訳合計チェック: const+alias+not+cse+emitted=${t.constFolded + t.aliasFolded + t.notFolded + t.cseHits + t.emitted} (=before ${t.gatesBefore})`,
    );
    console.log(`  NOT命令数=${t.notInstrs} / 出力が定数のoc=${t.ocConst} / タップ直結のoc=${t.ocTap}`);
    console.log(
      `  タップ平面: ${t.planesBefore} → ${t.planesAfter} (${pct(t.planesAfter, t.planesBefore)}減)`,
    );
    console.log(
      `  残存命令のtt分布: ${r.ttHist.map((v, i) => (v ? `0x${i.toString(16)}:${v}` : null)).filter(Boolean).join(' ')}`,
    );
    console.log(
      `  同一オペランド対を共有する命令=${r.sharedPairInstrs}`
      + ` / うち出力反転だけ違う対(NPN重複)=${r.npnDupPairs}`,
    );
    console.log(`  セルフチェック(全数真理値表照合) ${r.verifyMs.toFixed(1)}ms`);
  }
}
