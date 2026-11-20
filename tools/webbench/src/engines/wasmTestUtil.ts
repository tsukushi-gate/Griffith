// テスト専用: build/release.wasm を node:fs で直接読み込み
// インスタンス化する(本番コードは vite `?url` + fetch を使うが、vitestは
// ブラウザではないため node:fs 経由にする。`npm test` の pretest で
// asbuild:release が必ず先に走るため通常は常に存在する)。
import { existsSync, readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

import { instantiateWasmModule } from './wasmConv';
import type { WasmModule } from './wasmConv';

const here = dirname(fileURLToPath(import.meta.url));
export const releaseWasmPath = join(here, '../../build/release.wasm');
// build/release-simd.wasm(assembly/index-simd.ts、v128版)。ABIは
// release.wasmと完全同一(alloc/evalConv2dBatch/evalConv2dBitplaneBatch)。
export const releaseSimdWasmPath = join(here, '../../build/release-simd.wasm');

/** build/release.wasm が存在すればインスタンス化して返す(無ければnull、
    呼び出し側は describe.skipIf(!mod) で自動スキップする)。 */
export async function loadTestWasmModule(): Promise<WasmModule | null> {
  if (!existsSync(releaseWasmPath)) return null;
  const bytes = readFileSync(releaseWasmPath);
  return instantiateWasmModule(new Uint8Array(bytes));
}

/** build/release-simd.wasm(SIMD/v128版)が存在すればインスタンス化
    して返す(無ければnull、呼び出し側は describe.skipIf(!mod) で自動
    スキップする)。実行環境がWASM SIMDに未対応でも(Node/vitestは対応済み
    前提)インスタンス化自体は「モジュールを検証・リンクする」だけであり、
    SIMD命令の実行を伴わないため通常は問題なく成功する。 */
export async function loadTestWasmSimdModule(): Promise<WasmModule | null> {
  if (!existsSync(releaseSimdWasmPath)) return null;
  const bytes = readFileSync(releaseSimdWasmPath);
  return instantiateWasmModule(new Uint8Array(bytes));
}
