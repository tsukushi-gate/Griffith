// 文献値の参考表(WB-6)。同一タスク実測ではないランタイムの
// 「本体サイズ・起動・スループット」目安を出典つきで併記する。
// 値は2026-07時点の調査(数値が取れないものはnull=未確認)。

export interface LiteratureRow {
  runtime: string;
  runtimeSizeNote: string;
  startupNote: string;
  throughputNote: string;
  source: string;
}

export const LITERATURE: LiteratureRow[] = [
  {
    runtime: 'ONNX Runtime Web (WASM)',
    runtimeSizeNote: 'wasm約20MB(既定)/最適化ビルド約8MB/minimal約3MB',
    startupNote: 'WASMコンパイル+初期化(CDN+キャッシュ運用が前提)',
    throughputNote: '本ページで実測(右表)。小型モデルはWASMがWebGPUより速い事例多数',
    source: 'https://github.com/microsoft/onnxruntime/discussions/24161',
  },
  {
    runtime: 'MediaPipe Tasks (Web)',
    runtimeSizeNote: 'WASMランタイム約6MB+モデル(タスク別)',
    startupNote: 'ページロード時初期化はレンダブロック(遅延ロード推奨)',
    throughputNote: '顔ランドマーク等はリアルタイム動作(カスタム小型分類は非対応が実情)',
    source: 'https://github.com/google/mediapipe/issues/4537',
  },
  {
    runtime: 'llama.cpp (wllama/WASM)',
    runtimeSizeNote: 'GGUFモデルQ4で数百MB〜(LLM前提)',
    startupNote: '分割DL+OPFSキャッシュでも初回は分オーダー',
    throughputNote: 'LLM専用(小型分類タスクには非適用)',
    source: 'https://github.com/ngxson/wllama',
  },
  {
    runtime: 'TensorFlow.js (WASM backend)',
    runtimeSizeNote: 'SIMD/threads別に複数wasmバイナリを実行時出し分け',
    startupNote: '環境検出+wasmロード',
    throughputNote: 'plain JSの10〜30倍(MobileNetV2で3〜11.5倍)',
    source: 'https://blog.tensorflow.org/2020/03/introducing-webassembly-backend-for-tensorflow-js.html',
  },
  {
    runtime: 'difflogic ネイティブC(参考)',
    runtimeSizeNote: 'モデル=ビット演算コード(KB級)',
    startupNote: 'コンパイル済みバイナリ',
    throughputNote: '単一CPUコアで約100万MNIST画像/秒(64サンプル/64bitワード)',
    source: 'https://arxiv.org/abs/2210.08277',
  },
];
