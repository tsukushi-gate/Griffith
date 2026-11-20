// 配信ベースパスの解決。
//
// GitHub Pages は https://<user>.github.io/<repo>/ のサブパス配下で配信されるため、
// 実行時に `fetch('/models/x.json')` のような**絶対パス文字列**を使っていると
// 全て 404 になる。vite の `base` はビルド時に解決される import / HTML の
// src・href しか書き換えないので、文字列リテラルは自前で前置する必要がある。
//
// import.meta.env.BASE_URL は vite が base から注入する（末尾スラッシュ付き）。
// ローカル(dev/serve)では '/' なので従来と完全に同じ URL になる。
const BASE = (import.meta.env?.BASE_URL ?? '/').replace(/\/+$/, '');

/** 先頭 '/' 始まりの配信パスに base を前置する。'/models/a.json' → '/repo/models/a.json' */
export function asset(path: string): string {
  if (!path.startsWith('/')) return path;
  return `${BASE}${path}`;
}

/** URLクエリ `?model=` によるDLGNモデル差し替えの解決。
 *
 * 各デモの DLGN モデルURLは main側に定数でハードコードされているため、
 * A3が生成した派生モデル(`_ll` losslessリンク張り替え / `_dc` don't care)を
 * 計測するにはファイルを入れ替えるしかなかった。ここで1つのクエリだけで
 * 切り替えられるようにする(`?prog=` `?mt=` `?gpupre=` と同じ流儀)。
 *
 * 受け付ける形式は2つ:
 *   - 変種サフィックス(先頭 '_'): `?model=_ll` → 既定の `x.json` を `x_ll.json` に
 *     置換する。**タスク非依存**なので bench_matrix の 5タスク一括計測でそのまま使える。
 *   - 完全なファイル名: `?model=det_multimnist_lut_k256_ll.json` → `/models/<name>` を
 *     直接指す(単一タスクのピンポイント計測用)。
 *
 * 安全のためパス区切り・親ディレクトリ参照は受け付けない(配信ルート配下の
 * models/ 以外を読ませない)。不正値は黙って既定へ落とす。
 */
const MODEL_NAME_RE = /^[A-Za-z0-9][A-Za-z0-9._-]*\.json$/;
const MODEL_SUFFIX_RE = /^_[A-Za-z0-9_-]+$/;

export function modelVariantFromUrl(href?: string): string | null {
  try {
    if (href === undefined && typeof window === 'undefined') return null;
    const v = new URL(href ?? window.location.href).searchParams.get('model');
    if (!v) return null;
    if (MODEL_SUFFIX_RE.test(v) || MODEL_NAME_RE.test(v)) return v;
    return null;
  } catch {
    return null;
  }
}

/** 既定のモデルパス('/models/x.json')に `?model=` の指定を適用して asset() する。 */
export function modelAsset(path: string, href?: string): string {
  const v = modelVariantFromUrl(href);
  if (v === null) return asset(path);
  if (v.startsWith('_')) return asset(path.replace(/\.json$/, `${v}.json`));
  return asset(`/models/${v}`);
}
