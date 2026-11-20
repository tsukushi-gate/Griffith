// CenterNet デコード一致テスト:
//  1. det_decode_golden.json: モデル非依存の純デコード。Python
//     detection_utils.decode_detections と detdecode.decodeDetections が
//     同一 hm/size/off(float)から同一箱列を出す(topK・NMS・box 復元)。
//  2. det_e2e_golden.json: 小 CenterNetLUT の IR → ConvRuntime(整数 popcount)
//     → 多ヘッド分割 → decode が Python simulate→decode と一致(端到端)。
// float64 で同一手順(-inf パディング maxpool・1e-6 タイブレーク)を踏むため
// 箱の各値は許容誤差 1e-6 内で一致する(conftest 流の実測較正なし固定閾値)。
import { existsSync, readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';

import type { IRGraphJson } from './shared';
import type { DetReadout } from './types';
import { decodeDetections } from './detdecode';
import type { DetBoxPx } from './detdecode';
import { ConvRuntime } from './engines/conv';
import type { ConvEngine } from './engines/conv';
import { loadTestWasmModule } from './engines/wasmTestUtil';
import type { WasmModule } from './engines/wasmConv';

const here = dirname(fileURLToPath(import.meta.url));
const TOL = 1e-6;

function b64ToBits(b64: string, nBits: number): Int32Array {
  const raw = atob(b64);
  const bits = new Int32Array(nBits);
  for (let i = 0; i < nBits; i++) {
    bits[i] = (raw.charCodeAt(i >> 3) >> (i & 7)) & 1;
  }
  return bits;
}

/** 箱列を (cls, x1, y1) 昇順で正規化してから 1:1 照合する。 */
function expectBoxesClose(got: DetBoxPx[], want: DetBoxPx[]): void {
  expect(got.length).toBe(want.length);
  const key = (b: DetBoxPx): string =>
    `${b.cls}|${b.x1.toFixed(4)}|${b.y1.toFixed(4)}`;
  const g = [...got].sort((a, b) => key(a).localeCompare(key(b)));
  const w = [...want].sort((a, b) => key(a).localeCompare(key(b)));
  for (let i = 0; i < w.length; i++) {
    expect(g[i].cls).toBe(w[i].cls);
    expect(g[i].x1).toBeCloseTo(w[i].x1, 5);
    expect(g[i].y1).toBeCloseTo(w[i].y1, 5);
    expect(g[i].x2).toBeCloseTo(w[i].x2, 5);
    expect(g[i].y2).toBeCloseTo(w[i].y2, 5);
    expect(Math.abs(g[i].score - w[i].score)).toBeLessThan(TOL);
  }
}

// ---------------------------------------------------------------------------
interface DecodeCase {
  name: string;
  scores: number[];
  readout: DetReadout;
  boxes: DetBoxPx[];
}
const decodePath = join(here, 'fixtures/det_decode_golden.json');
const hasDecode = existsSync(decodePath);

describe.skipIf(!hasDecode)('det 純デコード: Python decode_detections と一致', () => {
  const golden = JSON.parse(readFileSync(decodePath, 'utf8')) as {
    cases: DecodeCase[];
  };
  for (const c of golden.cases) {
    it(`${c.name} (${c.boxes.length}箱)`, () => {
      const scores = Float64Array.from(c.scores);
      const got = decodeDetections(scores, c.readout);
      expectBoxesClose(got, c.boxes);
    });
  }
});

// ---------------------------------------------------------------------------
interface E2ECase {
  name: string;
  ir: IRGraphJson;
  in_width: number;
  readout: DetReadout;
  frames: { in_bits_b64: string; boxes: DetBoxPx[] }[];
}
const e2ePath = join(here, 'fixtures/det_e2e_golden.json');
const hasE2E = existsSync(e2ePath);
const wasmModule: WasmModule | null = await loadTestWasmModule();
const ENGINES: ConvEngine[] = wasmModule
  ? ['packed', 'irflat', 'packed-wasm', 'bitplane', 'bitplane-wasm']
  : ['packed', 'irflat', 'bitplane'];

async function makeRuntime(ir: IRGraphJson, engine: ConvEngine): Promise<ConvRuntime> {
  if (engine === 'packed-wasm' || engine === 'bitplane-wasm') {
    return ConvRuntime.create(ir, { engine, wasmModule: wasmModule! });
  }
  return new ConvRuntime(ir, { engine });
}

describe.skipIf(!hasE2E)('det 端到端: ConvRuntime→detdecode == Python simulate→decode', () => {
  const golden = JSON.parse(readFileSync(e2ePath, 'utf8')) as {
    cases: E2ECase[];
  };
  for (const c of golden.cases) {
    for (const engine of ENGINES) {
      it(`[${engine}] ${c.name} (${c.frames.length}フレーム)`, async () => {
        const rt = await makeRuntime(c.ir, engine);
        for (const fr of c.frames) {
          const bits = b64ToBits(fr.in_bits_b64, c.in_width);
          const { data } = rt.scores(bits, 1);
          const got = decodeDetections(data, c.readout);
          expectBoxesClose(got, fr.boxes);
        }
      });
    }
  }
});
