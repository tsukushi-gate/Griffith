#!/usr/bin/env node
// @mediapipe/tasks-vision の FilesetResolver.forVisionTasks(basePath) は
// basePath を「ディレクトリURL」として要求し、内部で既知のファイル名
// (vision_wasm_internal.js/.wasm 等)を文字列連結して fetch する。
// Vite の `?url` 明示 import(ハッシュ付き個別ファイル名)では成立しないため、
// node_modules/@mediapipe/tasks-vision/wasm/ をそのまま public/mediapipe/wasm/ へ
// コピーし、実行時は FilesetResolver.forVisionTasks('/mediapipe/wasm') で読む。
// public/mediapipe/ は .gitignore 済み(モデル同様、生成物はコミットしない)。
// npm run dev / build の predev/prebuild から自動実行される(手動コピー禁止)。
import { cpSync, existsSync, mkdirSync, readdirSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const root = resolve(here, '..');
const src = resolve(root, 'node_modules/@mediapipe/tasks-vision/wasm');
const dst = resolve(root, 'public/mediapipe/wasm');

if (!existsSync(src)) {
  console.error(`[copy_mediapipe_assets] 見つかりません: ${src}`);
  console.error('npm i @mediapipe/tasks-vision を先に実行してください。');
  process.exit(1);
}

mkdirSync(dst, { recursive: true });
cpSync(src, dst, { recursive: true });

const files = readdirSync(dst);
console.log(`[copy_mediapipe_assets] ${src} → ${dst} (${files.length} files)`);
