// WebAssembly SIMD(v128)対応検出。WasmModuleのABI(alloc/
// evalConv2dBatch/evalConv2dBitplaneBatch)はscalar版(build/release.wasm)と
// simd版(build/release-simd.wasm)で完全同一なので、呼び出し側
// (conv.ts::loadWasmModuleForBrowser、wasmTestUtil.ts)はこの検出結果だけで
// どちらのバイナリを fetch/インスタンス化するか選べる。
//
// プローブ用バイト列は wasm-feature-detect (GoogleChromeLabs) の simd() と
// 同一の最小WASMモジュール(i32x4.splatを1命令だけ含む関数)。
// WebAssembly.validate は同期・例外を投げないため、対応判定に安全に使える。
const SIMD_PROBE = new Uint8Array([
  0, 97, 115, 109, 1, 0, 0, 0, 1, 5, 1, 96, 0, 1, 123, 3, 2, 1, 0, 10, 10, 1, 8, 0, 65, 0, 253, 15, 253, 98, 11,
]);

let cached: boolean | null = null;

/** 現在の実行環境(ブラウザ/Node)がWASM SIMD(v128)命令に対応していれば
    true。判定結果はプロセス内でキャッシュする(WebAssembly.validate自体は
    安価だが、モジュール選択のたびに再計算する必要はない)。 */
export function wasmSimdSupported(): boolean {
  if (cached !== null) return cached;
  try {
    cached = typeof WebAssembly !== 'undefined'
      && typeof WebAssembly.validate === 'function'
      && WebAssembly.validate(SIMD_PROBE);
  } catch {
    cached = false;
  }
  return cached;
}

/** テスト用: キャッシュを強制的にリセットする(通常のプロダクションコードは
    呼ぶ必要が無い、環境が実行中に変わることは無いため)。 */
export function resetWasmSimdCacheForTest(): void {
  cached = null;
}
