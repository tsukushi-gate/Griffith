// 6画面ぶんのサイドバー設定。ページ間の差分をここに集約する。
//
// 注意: `<option value>` は従来の 'js' / 'wasm' / 'webgpu' / 'mediapipe' のまま。
// 表示テキストだけを一般的な用語へ置き換えている(*_main.ts 側は
// option[value="webgpu"] を引いて対応状況を反映するため、値は契約の一部)。
import type { ControlSpec, SelectOption } from './controls';
import type { PageConfig } from './shell';

/** DLGN の計算方法。'js'=JavaScript、'wasm'=WebAssembly、'webgpu'=WGSL。 */
export function dlgnBackendOptions(selected: 'js' | 'wasm' | 'webgpu'): SelectOption[] {
  return [
    { value: 'js', label: 'CPU(JavaScript)', selected: selected === 'js' },
    { value: 'wasm', label: 'WebAssembly', selected: selected === 'wasm' },
    { value: 'webgpu', label: 'WebGPU', selected: selected === 'webgpu' },
  ];
}

/** DLGN計算方法の既定値。WebAssembly非対応環境は事実上存在せず、SIMD/非SIMDの
    フォールバックは engines/wasmFeature.ts 側で既に吸収しているため、
    CPU(JavaScript)より速いWebAssemblyを既定にする。この値を唯一の定義点とし、
    各ページの `dlgnBackendOptions(...)` 初期選択と `RuntimeGrid.initialBackend`/
    `bindBackendSelect` のフォールバック値(webgpu非対応時の降格先)を揃える。
    `?backend=` 等のURLクエリ明示指定(autobench.ts::parseAutobenchParams)は
    このUI既定値より優先される既存挙動のまま変更しない。 */
export const DEFAULT_DLGN_BACKEND: 'js' | 'wasm' | 'webgpu' = 'wasm';

// 既定は CPU 実行(ORT Web の CPU 実行プロバイダは wasm)。DLGN 側と条件を
// 揃えないと「CPU の DLGN vs GPU の CNN」を並べることになり比較が歪む。
export const ONNX_BACKEND_OPTIONS: SelectOption[] = [
  { value: 'wasm', label: 'WebAssembly', selected: true },
  { value: 'webgpu', label: 'WebGPU' },
];

/** 検出デモ共通の操作。しきい値の既定値だけページごとに違う。
    この値は **画面に検出枠をいくつ描くか** の表示用フィルタで、実測して較正した。
    モデルの readout.threshold(デコード側の閾値)とは別物なので、
    モデルJSONから供給してはいけない。 */
function detControls(threshold: number): ControlSpec[] {
  return [
    {
      kind: 'select', id: 'dlgnBackend', label: '計算方法(DLGN)',
      options: dlgnBackendOptions(DEFAULT_DLGN_BACKEND),
    },
    {
      kind: 'select', id: 'onnxBackend', label: '計算方法(ONNX)',
      options: ONNX_BACKEND_OPTIONS,
    },
    {
      kind: 'range', id: 'thr', label: '検出のしきい値',
      min: 0, max: 100, value: threshold,
      note: '表示する検出枠を絞るための値です。',
    },
    { kind: 'checkbox', id: 'gt', label: '正解の位置を表示(破線)', checked: true },
    {
      kind: 'file', id: 'file', label: '手元の画像を試す(ドラッグ&ドロップ可)',
      accept: 'image/*', disabled: true,
    },
  ];
}

export const PAGES: Record<string, PageConfig> = {
  index: { id: 'index', controls: [] },

  det: { id: 'det', controls: detControls(28) },

  detmono: { id: 'detmono', controls: detControls(30) },

  detface: { id: 'detface', controls: detControls(25) },

  detreal: { id: 'detreal', controls: detControls(25) },

  seg: {
    id: 'seg',
    controls: [
      {
        kind: 'select', id: 'dlgnBackend', label: '計算方法(DLGN)',
        options: dlgnBackendOptions(DEFAULT_DLGN_BACKEND),
      },
      // 検出デモと揃えて ONNX 側も選べるようにした(従来 seg だけ
      // WebAssembly 固定で、比較条件が他ページと違っていた)。
      {
        kind: 'select', id: 'onnxBackend', label: '計算方法(ONNX)',
        options: ONNX_BACKEND_OPTIONS,
      },
      {
        kind: 'range', id: 'alpha', label: '塗り分けの濃さ',
        min: 0, max: 255, value: 130,
      },
      { kind: 'checkbox', id: 'gt', label: '正解の範囲を表示(黄の輪郭)', checked: true },
      {
        kind: 'file', id: 'file', label: '手元の画像を試す(ドラッグ&ドロップ可)',
        accept: 'image/*', disabled: true,
      },
    ],
  },

  har: {
    id: 'har',
    controls: [
      { kind: 'button', id: 'play', label: '停止' },
      // 記録は同じ行動が 30 秒前後まとまって並んでいるので、実時間のままだと
      // 行動が変わるまで待つことになる。早送りと行動送りを用意する。
      { kind: 'button', id: 'nextActivity', label: '次の行動へ' },
      {
        kind: 'select', id: 'speed', label: '再生速度',
        options: [
          { value: '1', label: '実時間(1.28秒ごと)', selected: true },
          { value: '4', label: '4倍速' },
          { value: '16', label: '16倍速' },
        ],
      },
      {
        kind: 'range', id: 'window', label: 'センサ窓の位置',
        min: 0, max: 100, value: 0,
        note: '読み込み後は実時間で自動再生します。スライダーを動かすと停止して'
          + 'その位置の窓を判定します。',
      },
      {
        kind: 'select', id: 'onnxBackend', label: '計算方法(ONNX)',
        options: ONNX_BACKEND_OPTIONS,
      },
    ],
  },

  webcam: {
    id: 'webcam',
    // 手法セレクタで常に1つのランタイムしか動かさないので、
    // 「1手法ずつ順番に(逐次)行っています」という複数列前提のCONCURRENCY_NOTEは
    // 誤解を招く(同時に何本も動かして比べているように読める)。専用の文言にする。
    concurrencyNote:
      '選んだ手法(DLGN/ONNX Runtime Web/TensorFlow.js/MediaPipe)を1つだけ動かし、'
      + 'その推論時間を計測しています。他の手法と比べたいときは「手法」セレクタで'
      + '切り替えてください。',
    controls: [
      { kind: 'button', id: 'start', label: 'カメラ開始' },
      { kind: 'checkbox', id: 'infer', label: '推論を実行', checked: true },
      // ライブ実行は顔検出とセグメンテーションの2つに絞る
      // (実物体検出はカメラ映像だと学習分布から外れすぎて見せ場にならない)。
      {
        kind: 'select', id: 'task', label: 'タスク',
        options: [
          { value: 'seg', label: 'セグメンテーション', selected: true },
          { value: 'face', label: '顔検出' },
        ],
      },
      // 比較列を横に並べる方式から、動かす手法そのものを1つだけ選ぶ
      // セレクタ方式に変更した。表示は常に1枚(稼働エンジンも常に1つ)。
      // tfjs は seg のみ選べる(webcam_main.ts::applyTaskUi が選択肢を出し分ける)。
      {
        kind: 'select', id: 'method', label: '手法',
        options: [
          { value: 'dlgn', label: 'DLGN(論理ゲート回路)', selected: true },
          { value: 'onnx', label: '比較用のCNN(ONNX Runtime Web)' },
          { value: 'tfjs', label: '比較用のCNN(TensorFlow.js)' },
          { value: 'mediapipe', label: 'MediaPipe' },
        ],
      },
      // 計算方法(js/wasm/webgpu, wasm/webgpu)は選んだ手法に応じて出し分ける
      // (webcam_main.ts::applyMethodUi が表示/非表示を切り替える)。
      // 操作は「手法→(必要なら)計算方法」の最大2段に収める。
      {
        kind: 'group', id: 'dlgnBackendWrap',
        children: [{
          kind: 'select', id: 'backend', label: '計算方法(DLGN)',
          options: dlgnBackendOptions(DEFAULT_DLGN_BACKEND),
        }],
      },
      {
        kind: 'group', id: 'onnxBackendWrap', hidden: true,
        children: [{
          kind: 'select', id: 'onnxBackend', label: '計算方法(ONNX)',
          options: ONNX_BACKEND_OPTIONS,
        }],
      },
      {
        kind: 'group', id: 'thrWrap', hidden: true,
        children: [{
          kind: 'range', id: 'thr', label: '検出のしきい値',
          min: 0, max: 100, value: 25,
          note: '表示する検出枠を絞るための値です。',
        }],
      },
    ],
  },
};
