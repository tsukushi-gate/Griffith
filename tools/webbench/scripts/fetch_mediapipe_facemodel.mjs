#!/usr/bin/env node
// MediaPipe Face Detection(BlazeFace short-range, 入力128×128)の
// アセットを Google 公式ストレージから取得し public/models/ へ配置する。
// ライセンス: Apache License 2.0(MediaPipe Model Card 参照)。
// 既に存在する場合は再取得しない(--force で強制再取得)。
// public/models/ は .gitignore 済み(モデル実体はコミットしない、既存規約と同一)。
//
// 注記(実装時に判明): MediaPipe公式ドキュメントが例示する
// `.../blaze_face_short_range/float16/1/blaze_face_short_range.task` は
// 404(そのバージョンパスに `.task` は存在しない)。実際に配布されているのは
// `float16/latest/blaze_face_short_range.tflite`(約224KB、`.tflite`
// 拡張子だがメタデータ埋め込み済みでFaceDetector.createFromOptionsの
// modelAssetPathへそのまま渡せる)。
import { existsSync, mkdirSync, writeFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const URL_ = 'https://storage.googleapis.com/mediapipe-models/face_detector/'
  + 'blaze_face_short_range/float16/latest/blaze_face_short_range.tflite';

const here = dirname(fileURLToPath(import.meta.url));
const root = resolve(here, '..');
const outDir = resolve(root, 'public/models');
const outFile = resolve(outDir, 'blaze_face_short_range.tflite');

const force = process.argv.includes('--force');

if (existsSync(outFile) && !force) {
  console.log(`[fetch_mediapipe_facemodel] 既に存在します(スキップ): ${outFile}`);
  console.log('  再取得するには --force を付けてください。');
  process.exit(0);
}

console.log(`[fetch_mediapipe_facemodel] GET ${URL_}`);
const res = await fetch(URL_);
if (!res.ok) {
  console.error(`[fetch_mediapipe_facemodel] 取得失敗: ${res.status} ${res.statusText}`);
  process.exit(1);
}
const buf = new Uint8Array(await res.arrayBuffer());
mkdirSync(outDir, { recursive: true });
writeFileSync(outFile, buf);
console.log(`[fetch_mediapipe_facemodel] 保存: ${outFile} (${buf.byteLength} bytes)`);
