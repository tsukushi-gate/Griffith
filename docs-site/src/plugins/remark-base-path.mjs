/**
 * Markdown/MDX のルート相対 URL(`/guides/foo/`・`/images/bar.png`)に Astro の `base` を前置する。
 *
 * Astro は `base` を設定しても **本文中に手で書いたルート相対リンクは書き換えない**。
 * このサイトの本文リンクは全て `/guides/...` 形式なので、GitHub Pages の
 * プロジェクトページ配信(`base: /Griffith`)にすると全リンクが 404 になる。
 * 本プラグインが AST 段階で前置することで、本文を 1 行も書き換えずに
 * 「base あり/なし」の両方へ同じソースから配信できる。
 *
 * 対象:
 *   - link / image / definition ノードの `url`
 *   - MDX の JSX 属性 `href` / `src`(`<LinkCard href="/..." />` など)
 *
 * 対象外(そのまま通す): 外部URL・`//`(プロトコル相対)・アンカー・相対パス・
 * すでに base で始まっているもの。
 */

const SKIP_PREFIXES = ['//', 'http://', 'https://', 'mailto:', 'tel:', '#', 'data:'];

function shouldRewrite(url, base) {
	if (typeof url !== 'string' || url.length === 0) return false;
	if (!url.startsWith('/')) return false;
	if (SKIP_PREFIXES.some((p) => url.startsWith(p))) return false;
	// 二重付与の防止
	return !(url === base || url.startsWith(`${base}/`));
}

/** @param {{ base?: string }} options */
export function remarkBasePath(options = {}) {
	const base = (options.base ?? '').replace(/\/+$/, '');
	// base なし(ルート配信)のときは何もしない
	if (!base) return () => {};

	return (tree) => {
		walk(tree);

		function walk(node) {
			if (!node || typeof node !== 'object') return;

			if (
				(node.type === 'link' || node.type === 'image' || node.type === 'definition') &&
				shouldRewrite(node.url, base)
			) {
				node.url = base + node.url;
			}

			if (
				(node.type === 'mdxJsxFlowElement' || node.type === 'mdxJsxTextElement') &&
				Array.isArray(node.attributes)
			) {
				for (const attr of node.attributes) {
					if (attr.type !== 'mdxJsxAttribute') continue;
					if (attr.name !== 'href' && attr.name !== 'src') continue;
					if (shouldRewrite(attr.value, base)) attr.value = base + attr.value;
				}
			}

			if (Array.isArray(node.children)) node.children.forEach(walk);
		}
	};
}

export default remarkBasePath;
