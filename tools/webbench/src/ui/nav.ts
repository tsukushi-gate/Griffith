// サイドバーのタスク一覧(全ページ共通の唯一の定義)。
//
// 従来は同じ <nav> を9枚のHTMLに手でコピーしていたため、ページを追加するたびに
// 全ファイルを編集する必要があった。さらにリンクが絶対パス('/det.html')だったので
// GitHub Pages のサブパス配信(/<repo>/)では全リンクが404になっていた。
// ここで1箇所に集約し、URLは basePath.ts::asset() を通す。
import { asset } from '../basePath';

export type TaskId =
  | 'index' | 'det' | 'detmono' | 'detface' | 'detreal' | 'seg' | 'har' | 'webcam';

export interface NavItem {
  id: TaskId;
  /** 配信ルート基準のパス(asset() を通す前)。 */
  href: string;
  label: string;
  /** サイドバーの2行目に出す一言(データセット名)。 */
  hint: string;
}

/** 検証用ページ(wasmgpu_verify.html / mtverify.html)はここに載せない。
    計測スクリプトから直接URLで叩くだけの開発用ページで、デモの導線ではない。 */
export const NAV: readonly NavItem[] = [
  { id: 'index', href: '/index.html', label: 'ホーム', hint: 'このデモについて' },
  { id: 'det', href: '/det.html', label: 'カラー数字検出', hint: 'CIFAR-10 + 数字' },
  { id: 'detmono', href: '/detmono.html', label: 'モノクロ数字検出', hint: 'MNIST' },
  { id: 'detface', href: '/detface.html', label: '顔検出', hint: 'WIDER FACE' },
  { id: 'detreal', href: '/detreal.html', label: '実物体検出', hint: 'Pascal VOC' },
  { id: 'seg', href: '/seg.html', label: 'セグメンテーション', hint: 'Oxford-IIIT Pet' },
  { id: 'har', href: '/har.html', label: '行動認識', hint: 'UCI-HAR(スマホのセンサ)' },
  { id: 'webcam', href: '/webcam.html', label: 'カメラでライブ実行', hint: '手元のカメラ映像' },
];

export interface NavEntry extends NavItem {
  /** base 込みの実URL。 */
  url: string;
  current: boolean;
}

/** 現在地は URL ではなく **ページ側の設定値** から決める。
    location.pathname だと base 付き配信や '/' と '/index.html' の揺れで外れる。 */
export function navItems(current: TaskId): NavEntry[] {
  return NAV.map((n) => ({ ...n, url: asset(n.href), current: n.id === current }));
}

export function navLabel(id: TaskId): string {
  return NAV.find((n) => n.id === id)?.label ?? '';
}
