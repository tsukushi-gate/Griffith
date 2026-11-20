#!/usr/bin/env node
// MediaPipe の比較用モデルを Google 公式ストレージから取得し public/models/ へ置く。
//   - 顔検出: BlazeFace short-range(入力128×128)
//   - セグメンテーション: Selfie Segmenter(入力256×256。ブラウザでの
//     人物切り抜きに広く使われている標準モデル)
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

const BASE = 'https://storage.googleapis.com/mediapipe-models';
const MODELS = [
  {
    name: 'blaze_face_short_range.tflite',
    url: `${BASE}/face_detector/blaze_face_short_range/float16/latest/`
      + 'blaze_face_short_range.tflite',
  },
  {
    name: 'selfie_segmenter.tflite',
    url: `${BASE}/image_segmenter/selfie_segmenter/float16/latest/selfie_segmenter.tflite`,
  },
];

const here = dirname(fileURLToPath(import.meta.url));
const root = resolve(here, '..');
const outDir = resolve(root, 'public/models');
const force = process.argv.includes('--force');
const tag = '[fetch_mediapipe_models]';

mkdirSync(outDir, { recursive: true });
for (const m of MODELS) {
  const outFile = resolve(outDir, m.name);
  if (existsSync(outFile) && !force) {
    console.log(`${tag} 既に存在します(スキップ): ${outFile}`);
    continue;
  }
  console.log(`${tag} GET ${m.url}`);
  const res = await fetch(m.url);
  if (!res.ok) {
    console.error(`${tag} 取得失敗: ${res.status} ${res.statusText}`);
    process.exit(1);
  }
  const buf = new Uint8Array(await res.arrayBuffer());
  writeFileSync(outFile, buf);
  console.log(`${tag} 保存: ${outFile} (${buf.byteLength} bytes)`);
}
