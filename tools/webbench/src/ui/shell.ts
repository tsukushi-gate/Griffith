// 全ページ共通の外枠(左サイドバー)。各HTMLは <div id="shell"></div> を
// 置くだけで、タスク一覧・操作パネル・注意書きはここが描く。
//
// 狭い画面ではチェックボックス(#navOpen)+ peer 修飾子だけで開閉する。JSの状態を
// 持たないので、推論側でエラーが出てもメニューは動く。
import { renderControls } from './controls';
import type { ControlSpec } from './controls';
import { navItems, navLabel } from './nav';
import type { TaskId } from './nav';

/** 比較実行の注意書き(全ページ共通、文言の唯一の定義)。
    以前は複数ランタイムを同時に動かして計測していたため、CPU・GPUの
    奪い合いで推論時間が実際より長く出ていた。今は1手法ずつ順番(逐次)に
    計測しているので、ここに表示される値がその手法単独の推論時間に近い。 */
export const CONCURRENCY_NOTE =
  '比較用のランタイムは同じブラウザ上で動かしていますが、推論の計測は'
  + '1手法ずつ順番に(逐次)行っています。同時に走らせるとCPU・GPUを'
  + '奪い合って表示される推論時間が実際より遅く出てしまうためです。';

export interface PageConfig {
  id: TaskId;
  controls: ControlSpec[];
  /** CONCURRENCY_NOTE を上書きしたいページだけ指定する。
      webcamは手法セレクタで常に1つのランタイムしか動かさないため、
      「1手法ずつ順番に(逐次)行っています」という複数列前提の文言が
      誤解を招く(専用の文言に差し替える)。 */
  concurrencyNote?: string;
}

function el<K extends keyof HTMLElementTagNameMap>(
  tag: K, cls: string, text?: string,
): HTMLElementTagNameMap[K] {
  const e = document.createElement(tag);
  e.className = cls;
  if (text !== undefined) e.textContent = text;
  return e;
}

function buildNav(current: TaskId, toggle: HTMLInputElement): HTMLElement {
  const nav = el('nav', 'mt-5 space-y-0.5');
  // 遷移前にメニューを閉じておく(戻る操作で開いたまま復元されるのを防ぐ)。
  nav.onclick = () => { toggle.checked = false; };
  for (const item of navItems(current)) {
    const box = item.current
      ? el('span', 'block rounded-md bg-accent/10 px-3 py-2 text-accent')
      : el('a', 'block rounded-md px-3 py-2 text-fg hover:bg-line/60');
    if (!item.current) (box as HTMLAnchorElement).href = item.url;
    else box.setAttribute('aria-current', 'page');
    const label = el('span', item.current
      ? 'block text-sm font-semibold' : 'block text-sm', item.label);
    const hint = el('span', 'block text-xs text-muted', item.hint);
    box.append(label, hint);
    nav.append(box);
  }
  return nav;
}

function buildAside(cfg: PageConfig, toggle: HTMLInputElement): HTMLElement {
  const aside = el('aside',
    'fixed inset-y-0 left-0 z-30 w-64 -translate-x-full overflow-y-auto '
    + 'border-r border-line bg-panel px-4 py-5 transition-transform duration-200 '
    + 'peer-checked:translate-x-0 md:translate-x-0');

  const brand = el('div', '');
  brand.append(
    el('div', 'text-sm font-semibold', 'ブラウザ推論デモ'),
    el('div', 'text-xs text-muted', '論理ゲート回路モデル(DLGN)'),
  );
  aside.append(brand, buildNav(cfg.id, toggle));

  if (cfg.controls.length > 0) {
    const sec = el('section', 'mt-6 space-y-3 border-t border-line pt-5');
    sec.append(el('h2', 'text-xs font-semibold text-muted', '設定'));
    sec.append(renderControls(cfg.controls));
    aside.append(sec);
  }

  const foot = el('p', 'mt-6 border-t border-line pt-4 text-xs leading-relaxed text-muted',
    cfg.concurrencyNote ?? CONCURRENCY_NOTE);
  aside.append(foot);
  return aside;
}

function buildTopBar(cfg: PageConfig, toggle: HTMLInputElement): HTMLElement {
  const bar = el('header',
    'sticky top-0 z-10 flex items-center gap-3 border-b border-line bg-bg/90 '
    + 'px-4 py-3 backdrop-blur md:hidden');
  const burger = document.createElement('button');
  burger.type = 'button';
  burger.className = 'cursor-pointer rounded-md border border-line px-2 py-1 text-sm';
  burger.textContent = '☰';
  burger.setAttribute('aria-label', 'メニューを開く');
  // <label for> による活性化は環境差があるので、状態は明示的に切り替える
  // (見た目は checkbox の :checked を見る CSS が担う)。
  burger.onclick = () => { toggle.checked = !toggle.checked; };
  bar.append(burger, el('span', 'text-sm font-semibold', navLabel(cfg.id)));
  return bar;
}

/** #shell にサイドバー一式を描く。`?autobench=1` の計測経路では呼ばれない。 */
export function renderShell(cfg: PageConfig): void {
  const root = document.getElementById('shell');
  if (!root) return;
  root.textContent = '';

  // peer 修飾子は「兄弟要素」にしか効かないので、チェックボックスは
  // サイドバー/バーと同じ階層の先頭に置く。
  const toggle = document.createElement('input');
  toggle.type = 'checkbox';
  toggle.id = 'navOpen';
  // display:none だとラベル経由の活性化が環境によって効かないため sr-only にする
  // (レイアウト上は存在させたまま視覚的に隠す。キーボード操作もできる)。
  toggle.className = 'peer sr-only';
  toggle.setAttribute('aria-label', 'メニューの開閉');

  const backdrop = document.createElement('div');
  backdrop.className = 'fixed inset-0 z-20 hidden bg-black/40 peer-checked:block md:hidden';
  backdrop.onclick = () => { toggle.checked = false; };

  root.append(toggle, buildTopBar(cfg, toggle), buildAside(cfg, toggle), backdrop);

  // ページ本文側にも同じ注意書きを出す(カード一覧の直前などに置く)。
  document.querySelectorAll('[data-note="concurrency"]').forEach((e) => {
    e.textContent = cfg.concurrencyNote ?? CONCURRENCY_NOTE;
  });
}
