// 公開ビルドから再配布できないデータセット由来の埋め込み画像を除去する。
//
// なぜ必要か
// ----------
// デモモデルJSONは `demo_samples` に実画像(base64 RGBA)を埋め込んでいる。
// ローカル研究用途では問題ないが、**公開ホストへ出すと再配布にあたる**。
//   - WIDER FACE: CC BY-NC-ND 4.0。ND(改変禁止)があり、埋め込みは128pxへ
//     リサイズ済み=改変物にあたるため公開不可。detface.html 自身にも
//     「非商用・研究目的での利用に限る」と明記している。
//   - VOC: Flickr由来画像で再配布条件が不明瞭。保守的に除去する。
// Oxford-IIIT Pet(CC BY-SA 4.0、帰属表示済み)と MNIST/CIFAR 合成物は残す。
//
// 画像を外してもデモは壊れない: 各ページに画像アップロード欄があり、
// サンプルボタンは `demo_samples` から動的生成されるため0枚なら出ないだけ。
//
// `--keep-samples` を付けると埋め込み画像を残したまま公開ビルドを作る
// (未使用アセットの削除だけ行う)。**ライセンス上の判断を伴う**ので、
// 使うときは各ページの出典表示が出ていることを必ず確認すること。
import { readFileSync, writeFileSync, existsSync, rmSync, statSync } from 'node:fs';
import { join } from 'node:path';

const args = process.argv.slice(2);
const keepSamples = args.includes('--keep-samples');
const DIST = args.find((a) => !a.startsWith('--'))
  || new URL('../dist', import.meta.url).pathname;
// ファイル名 → 除去理由(公開ビルドで demo_samples を空にする)
const RESTRICTED = {
  'det_face_lut_k256.json': 'WIDER FACE (CC BY-NC-ND 4.0: ND により改変物の再配布不可)',
  'det_fewclass_lut_k64.json': 'PASCAL VOC (Flickr由来、再配布条件が不明瞭なため保守的に除去)',
};

let changed = 0;
if (keepSamples) {
  console.log('--keep-samples: 埋め込みサンプル画像を残す(出典表示の確認が必要)');
}
for (const [file, reason] of Object.entries(RESTRICTED)) {
  if (keepSamples) continue;
  const p = join(DIST, 'models', file);
  if (!existsSync(p)) { console.log(`  skip (無し): ${file}`); continue; }
  const doc = JSON.parse(readFileSync(p, 'utf8'));
  const n = doc.demo_samples?.length ?? 0;
  if (!n) { console.log(`  既に0枚: ${file}`); continue; }
  doc.demo_samples = [];
  doc.demo_samples_removed = { count: n, reason };
  writeFileSync(p, JSON.stringify(doc));
  console.log(`  除去 ${n}枚: ${file}  — ${reason}`);
  changed++;
}
console.log(`公開ビルド: ${changed} ファイルから埋め込み画像を除去した`);

// --- 未使用アセットの除去(公開ビルドの軽量化)-------------------------
// MediaPipe は FilesetResolver が実行時に1変種だけ取得する。実測(detface.html の
// resource timing)で使われるのは vision_wasm_internal.{js,wasm} のみ。
// module 変種(ESM用)は本デモの経路では使われないので公開ビルドから外す。
// nosimd 変種は SIMD 非対応ブラウザのフォールバックなので**残す**。
const UNUSED = ['mediapipe/wasm/vision_wasm_module_internal.wasm',
                'mediapipe/wasm/vision_wasm_module_internal.js'];
let freed = 0;
for (const rel of UNUSED) {
  const p = join(DIST, rel);
  if (!existsSync(p)) { console.log(`  skip (無し): ${rel}`); continue; }
  freed += statSync(p).size;
  rmSync(p);
  console.log(`  削除: ${rel}`);
}
if (freed) console.log(`公開ビルド: 未使用アセット ${(freed / 1048576).toFixed(1)}MB を削除した`);
