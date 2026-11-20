/**
 * 本文中の図(Mermaid の SVG / コンテンツ画像)を全画面拡大するライトボックス。
 *
 * - 対象は `.sl-markdown-content` 内だけ(ロゴ・サイドバー・ヒーロー等のUI画像には効かせない)
 * - 図の右上に付く拡大ボタンからのみ開く(図本体のクリックは邪魔しない)
 * - Mermaid はクライアントサイド描画なので MutationObserver で後から現れた図も拾う
 * - オーバーレイ内ではホイール/ピンチでズーム、ドラッグでパン、Esc・背景クリックで閉じる
 */

const CONTENT_SELECTOR = '.sl-markdown-content';
const ZOOMABLE_ATTR = 'data-md-zoomable';
/** アイコン・バッジを拾わないための最小サイズ(px) */
const MIN_SIZE = 64;
const MIN_SCALE = 0.2;
const MAX_SCALE = 12;

type Transform = { scale: number; tx: number; ty: number };

let overlay: HTMLDivElement | null = null;
let stage: HTMLDivElement | null = null;
let content: HTMLDivElement | null = null;
let lastFocused: HTMLElement | null = null;
const view: Transform = { scale: 1, tx: 0, ty: 0 };
const pointers = new Map<number, { x: number; y: number }>();
let pinchDistance = 0;
let panning = false;
let panOrigin = { x: 0, y: 0, tx: 0, ty: 0 };

// ---------------------------------------------------------------- 対象の収集

/** 図として扱えるか(リンク内・小さすぎるもの・明示的除外は対象外) */
function isZoomTarget(el: Element): boolean {
	if (el.hasAttribute(ZOOMABLE_ATTR)) return false;
	if (el.closest('a')) return false;
	if (el.closest('[data-md-no-zoom]')) return false;
	if (!el.closest(CONTENT_SELECTOR)) return false;
	const rect = el.getBoundingClientRect();
	// 画像はデコード前だと 0 幅になり得るので、その場合は属性側で判定する
	if (rect.width === 0 && el instanceof HTMLImageElement) {
		return el.naturalWidth === 0 || el.naturalWidth >= MIN_SIZE;
	}
	return rect.width >= MIN_SIZE && rect.height >= MIN_SIZE;
}

/** 図を包む器を作り、右上に拡大ボタンを付ける(図本体はクリックしても何も起きない) */
function mark(el: Element): void {
	if (!isZoomTarget(el)) return;
	el.setAttribute(ZOOMABLE_ATTR, '');

	const parent = el.parentNode;
	if (!parent) return;
	const wrapper = document.createElement('span');
	wrapper.className = 'md-zoomable';
	parent.insertBefore(wrapper, el);
	wrapper.appendChild(el);

	const button = document.createElement('button');
	button.type = 'button';
	button.className = 'md-zoom-btn';
	button.setAttribute('aria-label', '図を拡大表示する');
	button.title = '拡大表示';
	button.innerHTML =
		'<svg viewBox="0 0 24 24" width="18" height="18" aria-hidden="true" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round">' +
		'<circle cx="10.5" cy="10.5" r="6.5"/><path d="M15.5 15.5 L21 21"/><path d="M10.5 7.5v6M7.5 10.5h6"/></svg>';
	button.addEventListener('click', (event) => {
		event.preventDefault();
		event.stopPropagation();
		open(el);
	});
	wrapper.appendChild(button);
}

function scan(): void {
	for (const root of document.querySelectorAll(CONTENT_SELECTOR)) {
		root.querySelectorAll('img').forEach(mark);
		// Mermaid は <div class="mermaid" data-mermaid-rendered> の中に SVG を吐く
		root.querySelectorAll('[data-mermaid-rendered] svg, .mermaid svg').forEach(mark);
	}
}

// ------------------------------------------------------------ オーバーレイ

function buildOverlay(): HTMLDivElement {
	const root = document.createElement('div');
	root.className = 'md-lightbox';
	root.setAttribute('role', 'dialog');
	root.setAttribute('aria-modal', 'true');
	root.setAttribute('aria-label', '図の拡大表示');
	root.innerHTML = `
		<div class="md-lightbox__stage">
			<div class="md-lightbox__content"></div>
		</div>
		<button type="button" class="md-lightbox__close" aria-label="閉じる">✕</button>
		<p class="md-lightbox__hint">ホイール / ピンチで拡大・ドラッグで移動・Esc で閉じる</p>
	`;
	stage = root.querySelector<HTMLDivElement>('.md-lightbox__stage');
	content = root.querySelector<HTMLDivElement>('.md-lightbox__content');
	root.querySelector('.md-lightbox__close')?.addEventListener('click', close);
	root.addEventListener('pointerdown', onPointerDown);
	root.addEventListener('pointermove', onPointerMove);
	root.addEventListener('pointerup', onPointerUp);
	root.addEventListener('pointercancel', onPointerUp);
	root.addEventListener('wheel', onWheel, { passive: false });
	root.addEventListener('dblclick', onDoubleClick);
	root.addEventListener('click', (e) => {
		// 背景(図の外側)クリックで閉じる
		if (e.target === root || e.target === stage) close();
	});
	document.body.appendChild(root);
	return root;
}

/** srcset の中で最も幅(または倍率)が大きい候補を返す。srcset が無ければ現在の src */
function largestSource(img: HTMLImageElement): string {
	const srcset = img.getAttribute('srcset');
	if (!srcset) return img.currentSrc || img.src;
	let best = { url: img.currentSrc || img.src, weight: 0 };
	for (const entry of srcset.split(',')) {
		const [url, descriptor] = entry.trim().split(/\s+/);
		if (!url) continue;
		const weight = descriptor ? Number.parseFloat(descriptor) : 1;
		if (Number.isFinite(weight) && weight >= best.weight) best = { url, weight };
	}
	return best.url;
}

function cloneFigure(el: Element): Element {
	const clone = el.cloneNode(true) as HTMLElement;
	clone.removeAttribute(ZOOMABLE_ATTR);
	// id は**外さない**。mermaid が SVG 内に注入する <style> は `#svgId .node rect { ... }`
	// のように id で絞ったセレクタなので、id を消すとノードが黒塗り・エッジが黒塊になる。
	// marker(矢印)の `url(#...)` 参照も同じ理由で id をそのまま保つ。
	if (clone instanceof HTMLImageElement && el instanceof HTMLImageElement) {
		// 本文は表示幅に最適化された候補を使うが、拡大表示では常に最大解像度を読む。
		// `data-md-zoom-src` があれば最優先(原寸アセットを明示指定したい場合の逃げ道)。
		const zoomSrc = el.dataset.mdZoomSrc ?? largestSource(el);
		if (zoomSrc) clone.src = zoomSrc;
		clone.removeAttribute('srcset');
		clone.removeAttribute('sizes');
		clone.removeAttribute('loading');
		clone.removeAttribute('width');
		clone.removeAttribute('height');
	}
	if (clone instanceof SVGElement) {
		// SVG は内在サイズを持たないので、viewBox の縦横比からステージに収まる実寸を計算して与える
		// (Mermaid が付ける max-width 制限もここで外れる)
		const rect = el.getBoundingClientRect();
		const viewBox = clone.getAttribute('viewBox')?.split(/[\s,]+/).map(Number);
		const boxW = viewBox && viewBox.length === 4 ? viewBox[2] : rect.width;
		const boxH = viewBox && viewBox.length === 4 ? viewBox[3] : rect.height;
		const aspect = boxW && boxH ? boxW / boxH : 1;
		if (!viewBox && rect.width > 0) {
			clone.setAttribute('viewBox', `0 0 ${rect.width} ${rect.height}`);
		}
		clone.setAttribute('preserveAspectRatio', 'xMidYMid meet');
		const availW = window.innerWidth * 0.94 - 32;
		const availH = window.innerHeight * 0.88 - 32;
		const width = Math.min(availW, availH * aspect);
		clone.style.width = `${Math.round(width)}px`;
		clone.style.height = `${Math.round(width / aspect)}px`;
		clone.style.maxWidth = 'none';
		clone.style.maxHeight = 'none';
	}
	return clone;
}

/** 拡大中の SVG(ベクタのまま拡大するので transform ではなく実寸を変える) */
let vector: { el: SVGElement; width: number; height: number } | null = null;

function apply(): void {
	if (!content) return;
	const move = `translate(calc(-50% + ${view.tx}px), calc(-50% + ${view.ty}px))`;
	if (vector) {
		// CSS transform の scale はラスタ化されたレイヤを引き伸ばすため文字がぼやける。
		// SVG は要素の実寸(px)を変えて再描画させ、どの倍率でもシャープに保つ。
		vector.el.style.width = `${vector.width * view.scale}px`;
		vector.el.style.height = `${vector.height * view.scale}px`;
		content.style.transform = move;
		return;
	}
	content.style.transform = `${move} scale(${view.scale})`;
}

function reset(): void {
	view.scale = 1;
	view.tx = 0;
	view.ty = 0;
	apply();
}

function open(el: Element): void {
	overlay ??= buildOverlay();
	if (!content) return;
	const figure = cloneFigure(el);
	content.replaceChildren(figure);
	vector =
		figure instanceof SVGElement
			? {
					el: figure,
					width: Number.parseFloat(figure.style.width),
					height: Number.parseFloat(figure.style.height),
				}
			: null;
	reset();
	lastFocused = document.activeElement as HTMLElement | null;
	overlay.classList.add('is-open');
	document.documentElement.classList.add('md-lightbox-open');
	overlay.querySelector<HTMLButtonElement>('.md-lightbox__close')?.focus();
	document.addEventListener('keydown', onKeyDown);
}

function close(): void {
	if (!overlay) return;
	overlay.classList.remove('is-open');
	document.documentElement.classList.remove('md-lightbox-open');
	document.removeEventListener('keydown', onKeyDown);
	content?.replaceChildren();
	vector = null;
	pointers.clear();
	panning = false;
	lastFocused?.focus?.();
}

// ------------------------------------------------------------ 操作ハンドラ

function zoomAt(nextScale: number, px: number, py: number): void {
	if (!stage) return;
	const rect = stage.getBoundingClientRect();
	const cx = rect.left + rect.width / 2;
	const cy = rect.top + rect.height / 2;
	const clamped = Math.min(MAX_SCALE, Math.max(MIN_SCALE, nextScale));
	// ポインタ位置を固定点にしてスケールする
	const ux = (px - cx - view.tx) / view.scale;
	const uy = (py - cy - view.ty) / view.scale;
	view.scale = clamped;
	view.tx = px - cx - ux * clamped;
	view.ty = py - cy - uy * clamped;
	apply();
}

function onWheel(event: WheelEvent): void {
	event.preventDefault();
	const factor = Math.exp(-event.deltaY * (event.ctrlKey ? 0.01 : 0.0022));
	zoomAt(view.scale * factor, event.clientX, event.clientY);
}

function onDoubleClick(event: MouseEvent): void {
	event.preventDefault();
	zoomAt(view.scale > 1.2 ? 1 : 2.5, event.clientX, event.clientY);
	if (view.scale === 1) reset();
}

function onPointerDown(event: PointerEvent): void {
	if ((event.target as Element).closest('.md-lightbox__close')) return;
	pointers.set(event.pointerId, { x: event.clientX, y: event.clientY });
	if (pointers.size === 2) {
		pinchDistance = distance();
		panning = false;
		return;
	}
	panning = true;
	panOrigin = { x: event.clientX, y: event.clientY, tx: view.tx, ty: view.ty };
	overlay?.classList.add('is-grabbing');
	(event.target as Element).setPointerCapture?.(event.pointerId);
}

function distance(): number {
	const [a, b] = Array.from(pointers.values());
	if (!a || !b) return 0;
	return Math.hypot(a.x - b.x, a.y - b.y);
}

function midpoint(): { x: number; y: number } {
	const [a, b] = Array.from(pointers.values());
	if (!a || !b) return { x: 0, y: 0 };
	return { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 };
}

function onPointerMove(event: PointerEvent): void {
	if (!pointers.has(event.pointerId)) return;
	pointers.set(event.pointerId, { x: event.clientX, y: event.clientY });
	if (pointers.size >= 2) {
		const d = distance();
		if (pinchDistance > 0 && d > 0) {
			const m = midpoint();
			zoomAt(view.scale * (d / pinchDistance), m.x, m.y);
		}
		pinchDistance = d;
		return;
	}
	if (!panning) return;
	view.tx = panOrigin.tx + (event.clientX - panOrigin.x);
	view.ty = panOrigin.ty + (event.clientY - panOrigin.y);
	apply();
}

function onPointerUp(event: PointerEvent): void {
	pointers.delete(event.pointerId);
	if (pointers.size < 2) pinchDistance = 0;
	if (pointers.size === 0) {
		panning = false;
		overlay?.classList.remove('is-grabbing');
	}
}

function onKeyDown(event: KeyboardEvent): void {
	if (event.key === 'Escape') {
		event.preventDefault();
		close();
		return;
	}
	if (!stage) return;
	const rect = stage.getBoundingClientRect();
	const cx = rect.left + rect.width / 2;
	const cy = rect.top + rect.height / 2;
	if (event.key === '+' || event.key === '=') zoomAt(view.scale * 1.25, cx, cy);
	if (event.key === '-') zoomAt(view.scale / 1.25, cx, cy);
	if (event.key === '0') reset();
}

// ------------------------------------------------------------------ 起動

function start(): void {
	scan();
	// Mermaid は非同期描画なので、後から挿入された図も自動で対象にする
	const observer = new MutationObserver(() => scan());
	for (const root of document.querySelectorAll(CONTENT_SELECTOR)) {
		observer.observe(root, { childList: true, subtree: true });
	}
	// 画像の遅延読み込みでサイズが確定するケースに備えて一度だけ再走査する
	window.addEventListener('load', () => scan(), { once: true });
}

if (document.readyState === 'loading') {
	document.addEventListener('DOMContentLoaded', start);
} else {
	start();
}
