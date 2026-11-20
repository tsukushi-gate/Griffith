// 行動認識デモ(UCI-HAR)。スマートフォンの加速度・角速度 2.56 秒ぶんから
// 6種類の行動を判定する。窓は 50% 重なりで連続しているので、順に送ると
// センサのストリーム再生になる(読み込み後そのまま実時間で流し始める)。
//
// 注意: 公開されているテスト記録は**同じ行動がまとまって並んでいる**
// (1ブロック 6〜40 窓、中央値 26 窓 ≒ 33 秒)。実時間で流すと行動が変わるまで
// 30 秒前後かかるので、再生速度と「次の行動へ」を用意する。
import { HarDlgnRuntime, HarOnnxRuntime } from './compareHar';
import type { HarRuntime, HarWindow } from './compareHar';
import type { OnnxBackend } from './compareSeg';
import { drawWindow, fetchHarTestData } from './harData';
import type { TestData } from './types';
import { asset, modelAsset } from './basePath';
import { applyModelMeta } from './modelMeta';
import { applyReferences } from './references';
import { runSequential } from './sequentialRun';
import { RuntimeGrid } from './ui/runtimeCard';
import type { SlotSpec } from './ui/runtimeCard';
import { renderShell } from './ui/shell';
import { PAGES } from './ui/pages';

const $ = <T extends HTMLElement>(id: string): T =>
  document.getElementById(id) as T;

const DLGN_URL = modelAsset('/models/har_dlgn_k6000_gumbel.json');
const MLP_URL = asset('/models/har_mlp_fp32.onnx');
const CNN_URL = asset('/models/har_cnn1d_fp32.onnx');
const DISP = 200;
/** 窓の進む間隔。50Hz・128点で 50% 重なりなので実時間では 1.28 秒ごと。 */
const STEP_MS = 1280;

async function init(): Promise<void> {
  renderShell(PAGES.har);

  const data: TestData = await fetchHarTestData();
  const labels = data.meta.labels;
  const truthOf = (i: number): number => data.meta.y_test[i];
  let index = 0;
  let timer: ReturnType<typeof setInterval> | null = null;

  const onnxBackend = RuntimeGrid.initialBackend('onnxBackend', 'wasm');
  const slots: SlotSpec<HarRuntime>[] = [
    {
      kind: 'other',
      label: 'DLGN(論理ゲート回路)',
      create: () => new HarDlgnRuntime(DLGN_URL),
    },
    {
      kind: 'onnx',
      label: '比較用のMLP(ONNX Runtime Web)',
      backend: onnxBackend,
      create: (b) => new HarOnnxRuntime(
        MLP_URL, 'har_mlp_teacher', '比較用のMLP(ONNX Runtime Web)', b as OnnxBackend,
      ),
    },
    {
      kind: 'onnx',
      label: '比較用の1D CNN(ONNX Runtime Web)',
      backend: onnxBackend,
      create: (b) => new HarOnnxRuntime(
        CNN_URL, 'har_cnn1d_teacher', '比較用の1D CNN(ONNX Runtime Web)', b as OnnxBackend,
      ),
    },
  ];

  const grid = new RuntimeGrid<HarRuntime>({
    gridEl: $('grid'),
    display: DISP,
    aspect: 'wide',
    slots,
    onRender: () => renderAll(),
  });

  function updatePosition(): void {
    $('position').textContent = `窓 ${index + 1} / ${data.n}(正解: ${
      labels[truthOf(index)] ?? '—'})`;
    const slider = $<HTMLInputElement>('window');
    if (Number(slider.value) !== index) slider.value = String(index);
  }

  /** 列ごとに厳密に逐次実行する(CPU/GPU資源の競合を避け、真の推論速度に
      近づける。複数ランタイムを Promise.all で同時に走らせると、表示される
      「推論時間」が資源の奪い合いで実際より遅く出てしまうため)。 */
  async function renderAll(): Promise<void> {
    updatePosition();
    const truth = truthOf(index);
    const win: HarWindow = { data, index };
    await runSequential(grid.cols, async (c) => {
      if (c.error || !c.info) return;
      drawWindow(c.base, data, index);
      try {
        const { output, ms } = await RuntimeGrid.timedRun<number, HarWindow>(
          c.rt, win, 3,
        );
        const name = labels[output] ?? String(output);
        grid.setHeadline(c, name, output === truth ? 'ok' : 'ng');
        grid.setInference(c, ms, output === truth ? '正解と一致' : '正解と不一致');
      } catch (e) {
        grid.setRunError(c, String(e));
      }
      // 次の列へ移る前に1フレーム分yieldし、描画を落ち着かせてから次の計測に入る。
      await new Promise((r) => requestAnimationFrame(r));
    });
  }

  function stepMs(): number {
    return STEP_MS / Number($<HTMLSelectElement>('speed').value || '1');
  }

  function setPlaying(on: boolean): void {
    const btn = $<HTMLButtonElement>('play');
    if (timer !== null) { clearInterval(timer); timer = null; }
    if (on) {
      timer = setInterval(() => {
        index = (index + 1) % data.n;
        void renderAll();
      }, stepMs());
    }
    btn.textContent = on ? '停止' : '再生';
  }

  /** 記録は同じ行動がまとまって並んでいるので、次に行動が変わる窓まで飛ぶ。 */
  function jumpToNextActivity(): void {
    const cur = truthOf(index);
    for (let i = 1; i <= data.n; i++) {
      const j = (index + i) % data.n;
      if (truthOf(j) !== cur) { index = j; break; }
    }
    void renderAll();
  }

  grid.build();
  grid.bindBackendSelect('onnx', 'onnxBackend', 'wasm');
  await grid.loadAll();

  // 精度・ゲート数などのページ本文の数値は DLGN モデルJSONから描く。
  applyModelMeta((grid.cols[0].rt as HarDlgnRuntime).doc);
  applyReferences();

  const slider = $<HTMLInputElement>('window');
  slider.max = String(data.n - 1);
  slider.oninput = () => {
    index = Number(slider.value);
    setPlaying(false);
    void renderAll();
  };
  $<HTMLButtonElement>('play').onclick = () => setPlaying(timer === null);
  $<HTMLButtonElement>('nextActivity').onclick = () => jumpToNextActivity();
  $<HTMLSelectElement>('speed').onchange = () => {
    if (timer !== null) setPlaying(true); // 新しい間隔で貼り直す
  };
  await renderAll();
  // 読み込み後はそのまま流し始める(1.28 秒ごと=センサの実時間)。
  setPlaying(true);
}

void init();
