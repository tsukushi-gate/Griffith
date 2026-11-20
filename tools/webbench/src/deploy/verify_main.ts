#!/usr/bin/env node
// `griffith deploy verify-web` の出荷ゲートCLI(dist/verify.mjs)。
//
// frozen contract:
//   node verify.mjs <bundle.json> <cases.json> [--engine js|wasm] [--wasm <path>]
//   stdout: 1行JSON {"ok","engine","mode","outputs","maxAbsDiff","error"?}
//   exit 0 iff ok、それ以外は 1。実行時エラーも {"ok":false,"error":...} + exit1。
//
// 既定エンジンは 'js'(依存ゼロ、wasmバイナリのfetch/読み込みが一切要らない)。
// --engine wasm は node:fsでwasmバイナリを読み込みインスタンス化して
// wasmModuleとして注入する(lib.ts::loadBundle の 'auto'/'wasm' 経路が使う
// fetch()はブラウザ専用でfile:を扱えないため、CLI側で明示的に注入する)。
// --wasm <path> 省略時は verify.mjs 自身と同じディレクトリの
// release-simd.wasm を既定にする(dist/ 配布物では常に隣にコピーされる
// ので通常は指定不要。griffith_logic/deploy/webverify.py も
// --wasm を渡さずに --engine wasm を呼ぶ)。
import { readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { instantiateWasmModule } from '../engines/wasmConv';
import { verifyBundleCases } from './verifyCore';
import type { VerifyCaseFile } from './verifyCore';
import type { LoadBundleOptions } from './lib';

/** --wasm省略時の既定パス: このファイル(dist/verify.mjs)と同じ
    ディレクトリの release-simd.wasm(vite.lib.config.tsのcloseBundleが
    必ずコピーする)。 */
function defaultWasmPath(): string {
  return join(dirname(fileURLToPath(import.meta.url)), 'release-simd.wasm');
}

interface ParsedArgs {
  bundlePath: string;
  casesPath: string;
  engine: 'js' | 'wasm';
  wasmPath: string | null;
}

function parseArgs(argv: string[]): ParsedArgs {
  const positional: string[] = [];
  let engine: 'js' | 'wasm' = 'js';
  let wasmPath: string | null = null;
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a === '--engine') {
      const v = argv[i += 1];
      if (v !== 'js' && v !== 'wasm') {
        throw new Error(`--engine は js|wasm のみ対応(got: ${String(v)})`);
      }
      engine = v;
    } else if (a === '--wasm') {
      wasmPath = argv[i += 1] ?? null;
    } else {
      positional.push(a);
    }
  }
  if (positional.length < 2) {
    throw new Error(
      '使い方: node verify.mjs <bundle.json> <cases.json> [--engine js|wasm] [--wasm <path>]',
    );
  }
  return {
    bundlePath: positional[0], casesPath: positional[1], engine, wasmPath,
  };
}

async function main(): Promise<void> {
  const { bundlePath, casesPath, engine, wasmPath } = parseArgs(process.argv.slice(2));
  const doc: unknown = JSON.parse(readFileSync(bundlePath, 'utf8'));
  const file = JSON.parse(readFileSync(casesPath, 'utf8')) as VerifyCaseFile;

  const opts: LoadBundleOptions = { engine };
  if (engine === 'wasm') {
    const bytes = readFileSync(wasmPath ?? defaultWasmPath());
    opts.wasmModule = await instantiateWasmModule(new Uint8Array(bytes));
  }

  const report = await verifyBundleCases(doc, file, opts);
  process.stdout.write(`${JSON.stringify(report)}\n`);
  // process.exit()を書き込み直後に呼ぶと、出力がパイプバッファ
  // (~64KB)を超えたときstdoutのflushを待たずにプロセスが終了し、
  // 大きめのcases.json(demo_samples複数件等)でJSONの末尾が欠落する
  // (Node既知の落とし穴)。exitCodeだけ設定してイベントループの自然な
  // 終了(=stdoutのdrain完了後)に任せる。
  process.exitCode = report.ok ? 0 : 1;
}

main().catch((e: unknown) => {
  process.stdout.write(`${JSON.stringify({ ok: false, error: String(e) })}\n`);
  process.exitCode = 1;
});
