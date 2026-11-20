// WebGPUデバイス取得ヘルパー。
// conv2dノードごとにdispatch+mapAsync読み戻しを行っていた旧実装
// (WebgpuConvKernel、1フレーム5〜6往復)はW10でグラフ常駐executor
// (webgpuGraph.ts::WebgpuGraphRuntime、全ノードを1つのcommandEncoder/
// 1回queue.submitで連鎖実行し、最終出力だけ1回読み戻す)へ置き換えた。
// requestWebgpuDevice はconv.ts::ConvRuntime.initWebgpuが引き続き使う。
export async function requestWebgpuDevice(): Promise<GPUDevice | null> {
  const nav = navigator as unknown as { gpu?: GPU };
  if (!nav.gpu) return null;
  const adapter = await nav.gpu.requestAdapter();
  if (!adapter) return null;
  // 大モデル(k640=75Mゲート級)向けにadapter実上限まで要求する。
  // 既定値(maxStorageBufferBindingSize=128MiB等)は保険として引き上げるが、
  // k640 box0/0 の主因は dispatch 1次元上限(65,535)であり、そちらは
  // webgpuGraph.ts の dispatch2d が対処する。
  const al = adapter.limits;
  // 対応環境では timestamp-query feature を要求する(webgpuGraph.ts の
  // ノード別GPU時間計測=TimingHelperパターンに使う)。非対応環境では
  // requiredFeatures に含めないだけで黙殺はしない: WebgpuGraphRuntime側が
  // device.features.has('timestamp-query') を見て timestampSupported:false を
  // 結果へ記録し、他の計測(CPU側分解等)は継続する。
  const requiredFeatures: GPUFeatureName[] = adapter.features.has('timestamp-query')
    ? ['timestamp-query']
    : [];
  try {
    return await adapter.requestDevice({
      requiredFeatures,
      requiredLimits: {
        maxStorageBufferBindingSize: al.maxStorageBufferBindingSize,
        maxBufferSize: al.maxBufferSize,
        maxStorageBuffersPerShaderStage: al.maxStorageBuffersPerShaderStage,
      },
    });
  } catch {
    try {
      // requiredLimits要求のみが拒否される環境向けフォールバック
      // (timestamp-query featureは維持する)。
      return await adapter.requestDevice(
        requiredFeatures.length > 0 ? { requiredFeatures } : undefined,
      );
    } catch {
      // 一部環境でfeature要求そのものが拒否される場合は既定で続行
      return adapter.requestDevice();
    }
  }
}
