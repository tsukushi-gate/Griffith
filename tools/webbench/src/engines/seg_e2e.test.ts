// 実モデル端到端整数一致テスト:
//   実画像RGBA → convpreproc(rgbaToNchw + encodeImageThermometer) →
//   ConvRuntime(実SegConvDLGN IR) → argmaxPerPixel マスクが、
//   Python ir/simulate.py の期待マスクと整数完全一致することを検証する。
// ゴールデンは scripts/export_webbench.py seg-ckpt --golden-image が生成
// (public/models の実モデルJSONと src/fixtures/seg_e2e_golden.json。共に
// 生成物=gitignore。無い場合は skipIf で自動スキップ)。
// packed-wasm(AssemblyScript製WASMカーネル)も実モデル(5conv層、
// depth=3、最大128ch)で packed/irflat と整数完全一致を取る
// (build/release.wasmが無ければpacked-wasm分だけ自動スキップ)。
import { existsSync, readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';

import { encodeImageThermometer, rgbaToNchw } from '../convpreproc';
import type { ImageThermoParams } from '../convpreproc';
import { ConvRuntime } from './conv';
import type { ConvEngine } from './conv';
import { loadTestWasmModule } from './wasmTestUtil';
import type { WasmModule } from './wasmConv';
import type { ConvModelDoc } from '../types';

const wasmModule: WasmModule | null = await loadTestWasmModule();
const ENGINES: ConvEngine[] = wasmModule
  ? ['packed', 'irflat', 'packed-wasm', 'bitplane', 'bitplane-wasm']
  : ['packed', 'irflat', 'bitplane'];

async function makeRuntime(ir: ConvModelDoc['ir'], engine: ConvEngine): Promise<ConvRuntime> {
  if (engine === 'packed-wasm' || engine === 'bitplane-wasm') {
    return ConvRuntime.create(ir, { engine, wasmModule: wasmModule! });
  }
  return new ConvRuntime(ir, { engine });
}

const here = dirname(fileURLToPath(import.meta.url));
const goldenPath = join(here, '../fixtures/seg_e2e_golden.json');
const modelsDir = join(here, '../../public/models');

interface SegE2EGolden {
  model_file: string;
  height: number;
  width: number;
  n_out: number;
  cases: {
    name: string;
    rgba_b64: string;
    expected_mask: number[];
    pet_fraction: number;
  }[];
}

function b64ToBytes(b64: string): Uint8ClampedArray {
  const raw = atob(b64);
  const out = new Uint8ClampedArray(raw.length);
  for (let i = 0; i < raw.length; i++) out[i] = raw.charCodeAt(i);
  return out;
}

const hasGolden = existsSync(goldenPath);
const golden: SegE2EGolden | null = hasGolden
  ? (JSON.parse(readFileSync(goldenPath, 'utf8')) as SegE2EGolden)
  : null;
const modelPath = golden ? join(modelsDir, golden.model_file) : '';
const ready = hasGolden && existsSync(modelPath);

describe.skipIf(!ready)('seg 実モデル端到端: convpreproc→ConvRuntime==Python', () => {
  const doc = JSON.parse(readFileSync(modelPath, 'utf8')) as ConvModelDoc;
  const p = doc.preproc;
  const thermo: ImageThermoParams = {
    mean: p.mean, std: p.std, thresholds: p.thresholds,
    height: p.height, width: p.width,
  };

  for (const engine of ENGINES) {
    for (const c of golden!.cases) {
      it(`[${engine}] ${c.name}: マスク整数一致 (pet率 ${c.pet_fraction.toFixed(3)})`, async () => {
        const rt = await makeRuntime(doc.ir, engine);
        const rgba = b64ToBytes(c.rgba_b64);
        const nchw = rgbaToNchw(rgba, golden!.height, golden!.width);
        const bits = encodeImageThermometer(nchw, thermo);
        const { data } = rt.scores(bits, 1);
        const hw = golden!.height * golden!.width;
        const labels = rt.argmaxPerPixel(data, 1, golden!.n_out, hw);
        expect(Array.from(labels)).toEqual(c.expected_mask);
        // k128 モデル(W21B)で irflat(素朴基準)が既定5秒を掠るため延長。
        // 高速エンジン側は従来どおり1秒未満で終わる。
      }, 30000);
    }
  }

  it('packed と irflat のスコアが完全一致(全画素・全クラス)', async () => {
    const packed = new ConvRuntime(doc.ir, { engine: 'packed' });
    const irflat = new ConvRuntime(doc.ir, { engine: 'irflat' });
    const bitplaneRt = new ConvRuntime(doc.ir, { engine: 'bitplane' });
    const wasmRt = wasmModule
      ? await ConvRuntime.create(doc.ir, { engine: 'packed-wasm', wasmModule })
      : null;
    const hw = golden!.height * golden!.width;
    for (const c of golden!.cases) {
      const rgba = b64ToBytes(c.rgba_b64);
      const nchw = rgbaToNchw(rgba, golden!.height, golden!.width);
      const bits = encodeImageThermometer(nchw, thermo);
      const a = packed.scores(bits, 1);
      const b = irflat.scores(bits, 1);
      expect(a.width).toBe(b.width);
      expect(a.width).toBe(golden!.n_out * hw);
      expect(Array.from(a.data)).toEqual(Array.from(b.data));
      const bp = bitplaneRt.scores(bits, 1);
      expect(Array.from(bp.data), `[bitplane] ${c.name}`).toEqual(Array.from(a.data));
      if (wasmRt) {
        const w = wasmRt.scores(bits, 1);
        expect(Array.from(w.data), `[packed-wasm] ${c.name}`).toEqual(Array.from(a.data));
      }
    }
    // irflat を含む照合は k128 モデルで既定5秒を超えるため延長
  }, 60000);
});
