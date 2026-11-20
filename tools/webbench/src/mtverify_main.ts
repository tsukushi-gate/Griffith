// DLGN CPUエンジン(bitplane-wasm)のマルチスレッド実行が
// 単スレッド実行と **整数完全一致** することをブラウザ上で確認するページ。
//
// vitest(node)には web Worker / SharedArrayBuffer(crossOriginIsolated)が
// 無いため、実スレッドでの一致確認はここが正式な経路になる
// (wasmgpu_verify.html が GPU 実行系に対して果たしている役割の CPU 版)。
// シャード境界そのものの純関数テストは wasmBitplaneOps.test.ts 側にある。
//
// 使い方(Playwright: scripts/mt_verify.mjs):
//   mtverify.html?mt=1  → window.__mtScores に全モデル×全入力の生スコア
//   mtverify.html?mt=4  → 同上
// を別々にロードし、node 側で全要素を突き合わせる。ページ内で2つの
// スレッド数を同時に持たない(?mt= はプロセス全体の既定として効くため、
// 「実際のデモと同じ経路」をそのまま検証できる)。
import { encodeImageAuto, rgbaToNchw } from './convpreproc';
import { loadSegModel } from './segdemo';
import { loadDetModel } from './detdemo';
import { asset } from './basePath';

const $ = (): HTMLElement => document.getElementById('status')!;

function log(line: string): void {
  $().textContent += `${line}\n`;
}

/** 決定的な擬似ランダムRGBA(wasmgpu_verify_main.ts と同じ生成器)。 */
function makeInput(size: number, seed: number): Uint8ClampedArray {
  let s = seed;
  const rnd = (): number => {
    s = (s * 1103515245 + 12345) & 0x7fffffff;
    return s / 0x7fffffff;
  };
  const rgba = new Uint8ClampedArray(size * size * 4);
  for (let i = 0; i < size * size; i++) {
    rgba[i * 4] = (rnd() * 255) | 0;
    rgba[i * 4 + 1] = (rnd() * 255) | 0;
    rgba[i * 4 + 2] = (rnd() * 255) | 0;
    rgba[i * 4 + 3] = 255;
  }
  return rgba;
}

interface Case { name: string; file: string; kind: 'seg' | 'det' }

const CASES: Case[] = [
  { name: 'seg', file: 'seg_convdlgn_k128.json', kind: 'seg' },
  { name: 'det', file: 'det_multimnist_lut_k256.json', kind: 'det' },
  { name: 'detmono', file: 'det_multimnist_mono_lut_k256.json', kind: 'det' },
  { name: 'detface', file: 'det_face_lut_k256.json', kind: 'det' },
  { name: 'detreal', file: 'det_fewclass_lut_k64.json', kind: 'det' },
];

const TRIALS = 3;

declare global {
  interface Window {
    /** タスク名 → 試行ごとの生スコア(Int32を通常配列にしたもの)。 */
    __mtScores?: Record<string, number[][]>;
    __mtInfo?: unknown;
  }
}

async function main(): Promise<void> {
  const out: Record<string, number[][]> = {};
  for (const c of CASES) {
    const url = asset(`/models/${c.file}`);
    const m = c.kind === 'seg'
      ? await loadSegModel(url, 'bitplane-wasm')
      : await loadDetModel(url, 'bitplane-wasm');
    const rt = m.runtime;
    const rows: number[][] = [];
    for (let t = 0; t < TRIALS; t++) {
      const rgba = makeInput(m.w, 20260804 + t * 6151);
      const bits = encodeImageAuto(rgbaToNchw(rgba, m.h, m.w), m.thermo);
      const { data } = await rt.scoresAsync(bits, 1);
      rows.push(Array.from(data));
    }
    out[c.name] = rows;
    log(`${c.name}: ${TRIALS}試行 × ${rows[0].length}要素 取得`
      + ` / threads要求=${rt.threads} 実効=${rt.actualThreads}`
      + `${rt.threadsNote ? ` (${rt.threadsNote})` : ''}`);
    window.__mtInfo = {
      ...(window.__mtInfo as Record<string, unknown> ?? {}),
      [c.name]: {
        requested: rt.threads, actual: rt.actualThreads, note: rt.threadsNote,
      },
    };
  }
  window.__mtScores = out;
  log('DONE');
  document.title = 'MTVERIFY_DONE';
}

main().catch((e) => {
  log(`失敗: ${String(e)}`);
  document.title = 'MTVERIFY_ERROR';
});
