// ランタイム比較カードの共通実装。
//
// 従来は det/detmono/detreal/detface/seg の5つの *_main.ts が、カード生成・
// ロード・バックエンド切替・時間計測をほぼ同一のコードで各々持っていた
// (1ファイルあたり約200行の重複)。ここへ集約し、各ページには
// 「どのランタイムを並べるか」と「結果をどう描くか」だけを残す。
import type { LoadInfo } from '../compare';
import { hasWebGpuApi } from '../compare';
import { MISSING } from '../modelMeta';
import { fmtBytesBig, fmtBytesInline, fmtMetricBig, fmtMs } from './format';
import type { BigValue } from './format';

/** カード内の役割。dlgn/onnx はサイドバーの計算方法セレクタと連動する。 */
export type SlotKind = 'dlgn' | 'onnx' | 'other';

export interface SlotSpec<RT> {
  kind: SlotKind;
  /** カード見出し(平易な表示名)。 */
  label: string;
  /** backend は kind==='other' のとき空文字。 */
  create: (backend: string) => RT;
  /** dlgn/onnx の初期バックエンド。 */
  backend?: string;
}

interface MetricCell {
  root: HTMLElement;
  value: HTMLElement;
  unit: HTMLElement;
  sub: HTMLElement;
}

export interface RuntimeCol<RT> {
  spec: SlotSpec<RT>;
  rt: RT;
  info: LoadInfo | null;
  error: string | null;
  backend: string;
  base: HTMLCanvasElement;
  overlay: HTMLCanvasElement;
  badgeEl: HTMLElement;
  headlineEl: HTMLElement;
  statEl: HTMLElement;
  detailEl: HTMLElement;
  metrics: { infer: MetricCell; size: MetricCell; load: MetricCell; acc: MetricCell };
}

export function backendLabel(b: string): string {
  if (b === 'js') return 'CPU(JavaScript)';
  if (b === 'wasm') return 'WebAssembly';
  if (b === 'webgpu') return 'WebGPU';
  if (b === 'mediapipe') return 'MediaPipe';
  return b;
}

/** 実際に使われたバックエンド(WebGPU→CPU のフォールバック後の値)。 */
function actualBackendOf(rt: unknown, requested: string): string {
  const a = (rt as { actualBackend?: string }).actualBackend;
  return a ?? requested;
}

function el<K extends keyof HTMLElementTagNameMap>(
  tag: K, cls: string, text?: string,
): HTMLElementTagNameMap[K] {
  const e = document.createElement(tag);
  e.className = cls;
  if (text !== undefined) e.textContent = text;
  return e;
}

function metricCell(label: string): MetricCell {
  const root = el('div', 'rounded-lg border border-line bg-panel px-3 py-2');
  root.append(el('div', 'text-xs text-muted', label));
  const line = el('div', 'flex items-baseline gap-1');
  const value = el('span', 'text-2xl font-bold tabular-nums leading-tight', MISSING);
  const unit = el('span', 'text-xs text-muted', '');
  line.append(value, unit);
  const sub = el('div', 'text-xs text-muted', '');
  root.append(line, sub);
  return { root, value, unit, sub };
}

function setCell(cell: MetricCell, v: BigValue, sub = ''): void {
  cell.value.textContent = v.value;
  cell.unit.textContent = v.unit;
  cell.sub.textContent = sub;
}

/** 配信サイズの大きい数字を「モデル+ランタイム」の合計にする(以前は
    モデルのみで、ランタイムはサブラインの注記に留まっていた)。ただし合算する
    2値は必ず同じ基準でなければならない: gzipBytes が取れているときモデル側は
    gzip 実測値なので、ランタイムも同じ CompressionStream 実測
    (runtimeGzipBytes)を使う。gzip 非対応環境(gzipBytes===null)ではモデルが
    非圧縮のまま出ているので、ランタイムも非圧縮の runtimeBytes を使う
    (どちらの分岐でも「非圧縮どうし」「gzipどうし」の組み合わせに揃う)。
    ランタイムの値が不明(null)なときはモデル単体を返す(以前の挙動のまま、
    NaN化を防ぐ)。DOM に依存しない純関数なのでユニットテストできる。 */
export function deliverBytesOf(
  info: Pick<LoadInfo, 'modelBytes' | 'gzipBytes' | 'runtimeBytes' | 'runtimeGzipBytes'>,
): { model: number | null; runtime: number | null; total: number | null } {
  const model = info.gzipBytes ?? info.modelBytes;
  const runtime = info.gzipBytes !== null ? info.runtimeGzipBytes : info.runtimeBytes;
  const total = runtime !== null && model !== null ? model + runtime : model;
  return { model, runtime, total };
}

/** 配信サイズのサブライン文言。deliverBytesOf() が返す model/runtime と
    常に対応させ、「先頭の内訳を足すと配信サイズ(大きい数字)に一致する」形に
    揃える。以前は非圧縮(モデル単体)をサブラインの先頭に置いていたため、
    「配信サイズ(モデル+ランタイム合算)の方が非圧縮(モデル単体)より大きい」
    という一見矛盾した表示になっていた(実際は基準がモデル単体 vs 合算で
    違うだけ)。
    - gzip実測が取れている場合: 「モデル(gzip)+ランタイム(gzip)」を内訳として
      出し、参考として素の非圧縮モデルサイズ(info.modelBytes)を末尾に添える。
      ランタイムのgzipが未計測ならランタイム項は省く。
    - gzip非対応(raw基準、deliverBytesOfがmodelBytes/runtimeBytesにフォール
      バックしている)場合は、非圧縮どうしの内訳のみを出す。非圧縮の注記は
      モデル項と同じ値の重複になるため付けない。 */
export function sizeSublineOf(
  info: Pick<LoadInfo, 'modelBytes' | 'gzipBytes'>,
  model: number | null,
  runtime: number | null,
): string {
  if (model === null) return '';
  const modelPart = `モデル ${fmtBytesInline(model)}`;
  const breakdown = runtime !== null
    ? `${modelPart} + ランタイム ${fmtBytesInline(runtime)}`
    : modelPart;
  if (info.gzipBytes !== null) {
    return `${breakdown} / 非圧縮 ${fmtBytesInline(info.modelBytes)}`;
  }
  return breakdown;
}

export interface RuntimeGridOptions<RT> {
  /** カードを並べる要素(#grid)。 */
  gridEl: HTMLElement;
  /** 表示canvasの一辺(px)。 */
  display: number;
  slots: SlotSpec<RT>[];
  /** 表示領域の縦横比。'square'=画像系(既定)、'wide'=波形など横長、
      'live'=常時1枚のライブ表示(webcam)。1:1である点は'square'と同じだが、
      グリッドの列指定を「1枚でも比較ページ(det/seg等)と同じ
      md:grid-cols-2 xl:grid-cols-3」に固定する。CSS Gridは列テンプレートを
      アイテム数と無関係に確保するため、1枚しか無くてもその1枚は3列レイアウト
      の1列ぶんの幅になり、比較ページのカード1枚と常に同じ大きさになる
      (残りの列は空白のまま。凝った独自CSSは持たず、既存の列クラスを
      流用するだけ)。 */
  aspect?: 'square' | 'wide' | 'live';
  /** バックエンド切替後の再描画。 */
  onRender: () => void | Promise<void>;
  /** バックエンド切替の直前に、実行中の推論(in-flight)が完全に終わるのを
      待つためのフック。ライブ実行(webcam)のように非同期ループを持つページだけ
      指定する(切替中に旧ランタイムへ推論が飛び込むのを防ぐ)。静的な比較ページ
      (det/seg/har)はUIイベント駆動で直列化されるため指定不要。 */
  waitInFlight?: () => Promise<void>;
}

/** RT=ランタイムの具体型。ページ側が modelSize/labels などの固有プロパティを
    使うので型パラメータで受ける。入力の型(画像RGBA / センサ窓)には依存しない。
    dispose は CompareRuntime と同じく任意(switchBackend の排他切替で使う)。 */
export class RuntimeGrid<
  RT extends { load(): Promise<LoadInfo>; dispose?(): void | Promise<void> },
> {
  cols: RuntimeCol<RT>[] = [];

  constructor(private opts: RuntimeGridOptions<RT>) {}

  /** ウォームアップ1回(初回カーネル準備を計測から除く)+ iters 回の中央値。
      表示に使う出力は最後の実行結果。 */
  static async timedRun<O, I>(
    rt: { run(input: I): Promise<{ output: O; inferMs: number }> },
    input: I,
    iters = 3,
  ): Promise<{ output: O; ms: number }> {
    let last = await rt.run(input);
    const times: number[] = [];
    for (let i = 0; i < iters; i++) {
      last = await rt.run(input);
      times.push(last.inferMs);
    }
    const s = [...times].sort((a, b) => a - b);
    return { output: last.output, ms: s[Math.floor(s.length / 2)] };
  }

  build(): void {
    const { gridEl, display, slots } = this.opts;
    gridEl.textContent = '';
    const live = this.opts.aspect === 'live';
    // webcamは常に1枚(slots.length===1)だが、live指定のときは比較
    // ページ(det/seg/har)と同じ列クラスを使う。CSS Gridの列テンプレートは
    // アイテム数と無関係に確保されるため、1枚だけでも3列レイアウトの1列分の
    // 幅になり、比較ページのカード1枚とちょうど同じ大きさで表示される
    // (以前は1枚のとき列指定を外して幅いっぱいに広げていたが、巨大化して
    // 画面に収まらなくなる不具合があったため撤回した)。
    gridEl.className = live || slots.length >= 3
      ? 'mt-4 grid gap-4 md:grid-cols-2 xl:grid-cols-3'
      : slots.length === 1
        ? 'mt-4 grid gap-4'
        : 'mt-4 grid gap-4 md:grid-cols-2';
    this.cols = slots.map((spec) => {
      const backend = spec.backend ?? '';
      const card = el('article', 'rounded-xl border border-line bg-card p-4');

      const header = el('div', 'flex items-center justify-between gap-2');
      header.append(el('h2', 'text-sm font-semibold', spec.label));
      // 計算方法を切り替えられない列(MediaPipe など)はバッジ自体を出さない。
      const badge = el('span',
        spec.kind === 'other'
          ? 'hidden'
          : 'rounded-full border border-line px-2 py-0.5 text-xs text-muted',
        spec.kind === 'other' ? '' : backendLabel(backend));
      header.append(badge);

      const wide = this.opts.aspect === 'wide';
      const stage = el('div',
        `relative mt-3 w-full ${wide ? 'aspect-video' : 'aspect-square'}`);
      const base = document.createElement('canvas');
      base.width = wide ? display * 2 : display;
      base.height = display;
      base.className = 'pixelated absolute inset-0 h-full w-full rounded-lg '
        + 'border border-line';
      const overlay = document.createElement('canvas');
      overlay.width = base.width; overlay.height = base.height;
      overlay.className = 'pixelated absolute inset-0 h-full w-full rounded-lg';
      stage.append(base, overlay);

      // 見出し(行動認識の予測クラスなど、結果を一言で出したいページだけ使う)。
      const headline = el('div', 'mt-3 hidden text-lg font-bold');

      const metrics = {
        infer: metricCell('推論時間'),
        size: metricCell('配信サイズ'),
        load: metricCell('読み込み時間'),
        acc: metricCell('精度'),
      };
      const dl = el('div', 'mt-3 grid grid-cols-2 gap-2');
      dl.append(metrics.infer.root, metrics.size.root, metrics.load.root, metrics.acc.root);

      const stat = el('div', 'mt-2 min-h-[1.5em] text-xs text-muted', '読み込み中…');
      const details = document.createElement('details');
      details.className = 'mt-2 text-xs text-muted';
      const summary = document.createElement('summary');
      summary.className = 'cursor-pointer';
      summary.textContent = '詳細';
      const detail = el('div', 'mt-1 space-y-1');
      details.append(summary, detail);

      card.append(header, stage, headline, dl, stat, details);
      this.opts.gridEl.append(card);

      return {
        spec,
        rt: spec.create(backend),
        info: null,
        error: null,
        backend,
        base,
        overlay,
        badgeEl: badge,
        headlineEl: headline,
        statEl: stat,
        detailEl: detail,
        metrics,
      };
    });
  }

  private applyInfo(col: RuntimeCol<RT>, info: LoadInfo): void {
    col.info = info;
    col.error = null;
    if (col.spec.kind !== 'other') {
      col.badgeEl.textContent = backendLabel(actualBackendOf(col.rt, col.backend));
    }

    const { model, runtime, total: deliver } = deliverBytesOf(info);
    setCell(col.metrics.size, fmtBytesBig(deliver), sizeSublineOf(info, model, runtime));
    setCell(col.metrics.load, fmtMs(info.loadMs), '初回のみ');
    setCell(col.metrics.acc, fmtMetricBig(info.metricValue),
      info.metricValue === null ? '実測していません' : `${info.metricLabel}(評価データ)`);

    col.detailEl.textContent = '';
    col.detailEl.append(el('div', '', info.paramsLabel));
    if (info.note) col.detailEl.append(el('div', '', info.note));
    col.statEl.textContent = '';
  }

  private applyError(col: RuntimeCol<RT>, msg: string): void {
    col.error = msg;
    col.info = null;
    col.statEl.textContent = '';
    setCell(col.metrics.infer, fmtMs(null));
    setCell(col.metrics.size, fmtBytesBig(null));
    setCell(col.metrics.load, fmtMs(null));
    setCell(col.metrics.acc, fmtMetricBig(null));
    col.detailEl.textContent = msg;
    col.badgeEl.textContent = '利用できません';
    col.badgeEl.className =
      'rounded-full border border-danger px-2 py-0.5 text-xs text-danger';
  }

  private resetPending(col: RuntimeCol<RT>): void {
    col.info = null;
    col.error = null;
    col.statEl.textContent = '読み込み中…';
    col.badgeEl.className = col.spec.kind === 'other'
      ? 'hidden'
      : 'rounded-full border border-line px-2 py-0.5 text-xs text-muted';
  }

  async loadAll(): Promise<void> {
    await Promise.all(this.cols.map(async (c) => {
      try {
        this.applyInfo(c, await c.rt.load());
      } catch (e) {
        this.applyError(c, String(e));
      }
    }));
  }

  /** 1フレームぶんの推論結果を反映する(各ページの renderAll から呼ぶ)。
      sub は数値の下に出す補足(既定は「逐次計測・中央値」=列を1つずつ実行した
      結果であることを明示する。ライブ実行では移動平均)。 */
  setInference(col: RuntimeCol<RT>, ms: number, note = '', sub = '逐次計測・中央値'): void {
    setCell(col.metrics.infer, fmtMs(ms), sub);
    col.statEl.textContent = note;
  }

  /** カード中央の見出し(予測クラスなど)。tone で正誤を色分けする。 */
  setHeadline(
    col: RuntimeCol<RT>, text: string, tone: 'plain' | 'ok' | 'ng' = 'plain',
  ): void {
    const color = tone === 'ok' ? 'text-accent' : tone === 'ng' ? 'text-danger' : '';
    col.headlineEl.className = `mt-3 text-lg font-bold ${color}`;
    col.headlineEl.textContent = text;
  }

  setRunError(col: RuntimeCol<RT>, msg: string): void {
    setCell(col.metrics.infer, fmtMs(null));
    col.statEl.textContent = `推論に失敗しました: ${msg}`;
  }

  /** 指定した役割の列(複数可)のバックエンドを差し替えて再ロードする。
      旧ランタイムは新ランタイムのロード完了後に dispose する(webcam の
      ように dispose?() を実装するランタイム(ONNX/TF.js)ではセッション/モデルを
      解放する。未実装のランタイム(DLGN/MediaPipe)は「解放不要」として無視)。 */
  async switchBackend(kind: SlotKind, backend: string): Promise<void> {
    const targets = this.cols.filter((c) => c.spec.kind === kind);
    if (targets.length === 0) return;
    if (this.opts.waitInFlight) await this.opts.waitInFlight();
    await Promise.all(targets.map(async (col) => {
      const prevRt = col.rt;
      col.backend = backend;
      col.rt = col.spec.create(backend);
      this.resetPending(col);
      col.badgeEl.textContent = backendLabel(backend);
      try {
        this.applyInfo(col, await col.rt.load());
      } catch (e) {
        this.applyError(col, String(e));
      }
      try {
        await prevRt.dispose?.();
      } catch {
        // 解放失敗は致命的でない(ページ寿命で最終的に回収される)ため無視する。
      }
    }));
    await this.opts.onRender();
  }

  /** サイドバーの計算方法セレクタを列に結び付ける。WebGPU が使えない環境では
      その選択肢を無効化し、選ばれていたら fallback へ寄せる。 */
  bindBackendSelect(kind: SlotKind, selectId: string, fallback: string): void {
    const sel = document.getElementById(selectId) as HTMLSelectElement | null;
    if (!sel) return;
    if (!hasWebGpuApi()) {
      const opt = sel.querySelector('option[value="webgpu"]') as HTMLOptionElement | null;
      if (opt) {
        opt.disabled = true;
        opt.textContent += '(このブラウザでは使えません)';
      }
      if (sel.value === 'webgpu') sel.value = fallback;
    }
    sel.onchange = () => void this.switchBackend(kind, sel.value);
  }

  /** セレクタの初期値(HTML/設定側の selected)。列生成前に読む。 */
  static initialBackend(selectId: string, fallback: string): string {
    const sel = document.getElementById(selectId) as HTMLSelectElement | null;
    const v = sel?.value;
    if (!v) return fallback;
    if (v === 'webgpu' && !hasWebGpuApi()) return fallback;
    return v;
  }
}
