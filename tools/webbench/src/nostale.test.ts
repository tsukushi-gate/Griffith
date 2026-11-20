// 陳腐化の再発防止ガード。
//
// 動機(実際に起きた事故): detbackends.html の説明文が旧世代モデル(小さいk・
// 低いmAP)を指したままで、配信中の実モデルは別のk・別の精度だった。同ページの表は
// モデルJSONから正しい値を出していたので、**説明文だけが嘘**という状態になっていた。
// 同種の埋め込みが det/detmono/detreal/detface/seg/webcam/convbench/detbench/
// segbackends にもあった。
//
// このテストは「モデル固有の数値リテラルがページ本文/ページTSに戻ってくること」を
// 機械的に禁止する。許可リストは持たない。どうしても必要な1箇所(参照値の集約)
// だけをファイル単位で除外し、理由をここに書く。
import { readFileSync, readdirSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';

const here = dirname(fileURLToPath(import.meta.url));
const ROOT = resolve(here, '..');

/** 除外ファイルと理由(ファイル単位のみ。行単位の許可リストは作らない)。 */
const EXCLUDED_TS = new Map<string, string>([
  // 参照値(ONNX/TF.js teacher, MediaPipe)の唯一の集約点。モデルJSONを持たない
  // 比較対象の実測値を、出典(wandb run id)つきでここに1箇所だけ書く。
  ['references.ts', '比較対象teacherの参照値を集約する唯一の場所'],
  // 外部文献の引用値とURL(arxiv ID が小数に見える)。同一タスク実測ではない旨を
  // ページ側にも明示しており、モデル差し替えで陳腐化する類の数値ではない。
  ['literature.ts', '他ランタイムの文献値(同一タスク実測ではない)+ arxiv URL'],
]);

/** モデル固有の指標値に見える数値リテラル(0.5209 / .6209 / 1.2345 …)。
    CSS の 0.85em や 1.05rem(小数2桁)は対象外。 */
const METRIC_LITERAL = /(?<![\w.])\d*\.\d{4,}(?![\d.])/g;

/** 「mAP 0.52」のように指標名の直後に数値が続く形(桁数が少なくても禁止)。
    `<` を含まないので `hard mAP <span data-meta=...>` は当たらない。 */
const METRIC_PHRASE = /(mAP|mIoU|mIou|hard acc|hard_map|hard_miou)[^<\n]{0,14}?\d+\.\d+/g;

function htmlFiles(): string[] {
  return readdirSync(ROOT).filter((f) => f.endsWith('.html')).sort();
}

/** src 直下と src/ui の「ページ側TS」(engines/ 等の数値カーネルは対象外)。
    サイドバー・カードの文言と既定値は src/ui へ移したので、
    走査対象に含めないとガードの穴になる。 */
function pageTsFiles(): string[] {
  const direct = readdirSync(resolve(ROOT, 'src'))
    .filter((f) => f.endsWith('.ts') && !f.endsWith('.test.ts'))
    .filter((f) => !EXCLUDED_TS.has(f));
  const ui = readdirSync(resolve(ROOT, 'src', 'ui'))
    .filter((f) => f.endsWith('.ts') && !f.endsWith('.test.ts'))
    .map((f) => `ui/${f}`);
  return [...direct, ...ui].sort();
}

function matches(text: string, re: RegExp): string[] {
  return [...text.matchAll(new RegExp(re.source, re.flags))].map((m) => m[0]);
}

describe('陳腐化ガード: HTML本文にモデル固有の数値を埋め込まない', () => {
  for (const f of htmlFiles()) {
    it(f, () => {
      const text = readFileSync(resolve(ROOT, f), 'utf8');
      expect(matches(text, METRIC_LITERAL), `${f}: 数値リテラル`).toEqual([]);
      expect(matches(text, METRIC_PHRASE), `${f}: 指標名+数値`).toEqual([]);
    });
  }
});

describe('陳腐化ガード: ページTSにモデル固有の数値を埋め込まない', () => {
  for (const f of pageTsFiles()) {
    it(f, () => {
      const text = readFileSync(resolve(ROOT, 'src', f), 'utf8');
      expect(matches(text, METRIC_LITERAL), `${f}: 数値リテラル`).toEqual([]);
      expect(matches(text, METRIC_PHRASE), `${f}: 指標名+数値`).toEqual([]);
    });
  }
});
