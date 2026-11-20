#!/usr/bin/env node
// data/papers/*.md と data/reports/*.md のfrontmatterを走査して
// src/data/tree.json を生成する。
//
// 設計方針: 複数レーンが並行で data/ を量産するため、
//  - md が増えても「未知のカテゴリ」「未知の追加フィールド」で壊れない(カテゴリは自動生成、
//    未知キーは extra に退避)
//  - 検査は全件走査してから一括レポートし、致命的な問題があるときだけ exit 1
//    (最初の1件で止めない = 他レーンが自分の担当分の不備を一度に把握できる)
//
// 検査(致命=exit 1):
//  1. id重複(papers/reports 横断)
//  2. 必須フィールド欠落(paper: id/title/status/categories、report: id/title)
//  3. 参照先不在(paper.parents / paper.next / paper.reports / report.papers)
//  4. superseded_by の参照先不在 / 自己参照 / 循環
//
// 重複論文の一本化(ファイル削除は禁止のため frontmatter で解決する):
//   重複側の frontmatter に `superseded_by: <正本id>` を書くと、
//    - そのファイルはノードを作らない(ツリーから消える)
//    - 他ファイルが持つ重複idへの参照(parents/next/reports/papers)は正本idへ自動で付け替わる
//   本文・ファイルは残るので、いつでも履歴を追える。
//
// オプション:
//  --allow-dangling  参照先不在を警告に降格し、該当エッジを落として生成を続ける(dev用)
//  --out <path>      出力先(既定 src/data/tree.json)
//  --data <dir>      入力ディレクトリ(既定 data。papers/ と reports/ を含む。検査の自己テスト用)
//  --quiet           警告を抑制

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { basename, dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import matter from 'gray-matter';

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..');

const ROOT_ID = 'root';
const CAT_PREFIX = 'cat:';
const REPORT_PREFIX = 'rep:';
/** レポートを束ねる既定カテゴリ */
const REPORTS_CAT = `${CAT_PREFIX}reports`;

const argv = process.argv.slice(2);
const hasFlag = (name) => argv.includes(name);
const flagValue = (name, fallback) => {
  const i = argv.indexOf(name);
  return i >= 0 && argv[i + 1] ? argv[i + 1] : fallback;
};

const ALLOW_DANGLING = hasFlag('--allow-dangling');
const QUIET = hasFlag('--quiet');
const OUT_PATH = resolve(ROOT, flagValue('--out', join('src', 'data', 'tree.json')));
const DATA_DIR = resolve(ROOT, flagValue('--data', 'data'));
const PAPERS_DIR = join(DATA_DIR, 'papers');
const REPORTS_DIR = join(DATA_DIR, 'reports');

const errors = [];
const warnings = [];
const err = (file, msg) => errors.push(`${file}: ${msg}`);
const warn = (file, msg) => warnings.push(`${file}: ${msg}`);

/** frontmatterの値を文字列配列へ正規化(未指定=[]、単一文字列も配列にする) */
function toList(value) {
  if (value == null) return [];
  const arr = Array.isArray(value) ? value : [value];
  return arr
    .map((v) => (typeof v === 'string' ? v.trim() : v == null ? '' : String(v)))
    .filter((v) => v.length > 0);
}

/** null / "null" / 空文字 を null に畳む */
function toNullableString(value) {
  if (value == null) return null;
  const s = String(value).trim();
  if (s === '' || s === 'null' || s === 'None') return null;
  return s;
}

/** Date型で読まれたfrontmatter(YAMLの日付リテラル)もISO日付文字列へ揃える */
function toDate(value) {
  if (value instanceof Date && !Number.isNaN(value.getTime())) {
    return value.toISOString().slice(0, 10);
  }
  return toNullableString(value);
}

function listMarkdown(dir) {
  let names;
  try {
    names = readdirSync(dir);
  } catch {
    return []; // ディレクトリ未作成でも落とさない
  }
  return names
    .filter((n) => n.endsWith('.md') && !n.startsWith('_') && n !== 'README.md')
    .sort()
    .map((n) => join(dir, n));
}

const KNOWN_PAPER_KEYS = new Set([
  'id',
  'title',
  'date',
  'summarized',
  'status',
  'categories',
  'arxiv',
  'url',
  'pdf',
  'parents',
  'next',
  'reports',
  'superseded_by',
]);
const KNOWN_REPORT_KEYS = new Set([
  'id',
  'title',
  'date',
  'branch',
  'path',
  'artifactUrl',
  'type',
  'papers',
]);

function collectExtra(data, known) {
  const extra = {};
  for (const [k, v] of Object.entries(data)) {
    if (!known.has(k)) extra[k] = v;
  }
  return Object.keys(extra).length > 0 ? extra : undefined;
}

function parseFile(path) {
  const raw = readFileSync(path, 'utf8');
  try {
    const fm = matter(raw);
    return { data: fm.data ?? {}, body: (fm.content ?? '').trim() };
  } catch (e) {
    err(basename(path), `frontmatterのYAMLを解析できない: ${e.message}`);
    return null;
  }
}

// --- 論文の読み込み ---------------------------------------------------------
const allPapers = [];
for (const path of listMarkdown(PAPERS_DIR)) {
  const file = `data/papers/${basename(path)}`;
  const parsed = parseFile(path);
  if (!parsed) continue;
  const { data, body } = parsed;

  const id = toNullableString(data.id);
  const title = toNullableString(data.title);
  if (!id) err(file, '必須フィールド `id` が無い');
  if (!title) err(file, '必須フィールド `title` が無い');

  const status = toNullableString(data.status) ?? (body ? 'summarized' : 'stub');
  if (status !== 'summarized' && status !== 'stub') {
    err(file, `\`status\` は summarized | stub のいずれか(実際: ${status})`);
  }
  const categories = toList(data.categories);
  if (categories.length === 0) err(file, '必須フィールド `categories` が空');

  if (id && id !== basename(path, '.md')) {
    warn(file, `id(${id})とファイル名が一致しない(推奨: ${id}.md)`);
  }

  // parents未指定なら先頭カテゴリを親にする(ツリーが切れないための既定)
  const parents = toList(data.parents);
  const effectiveParents =
    parents.length > 0 ? parents : categories.map((c) => `${CAT_PREFIX}${c}`);

  allPapers.push({
    kind: 'paper',
    file,
    supersededBy: toNullableString(data.superseded_by),
    id: id ?? file,
    title: title ?? basename(path, '.md'),
    date: toDate(data.date),
    summarized: toDate(data.summarized),
    status,
    categories,
    arxiv: toNullableString(data.arxiv),
    url: toNullableString(data.url),
    pdf: toNullableString(data.pdf),
    parents: effectiveParents,
    next: toList(data.next),
    reports: toList(data.reports),
    extra: collectExtra(data, KNOWN_PAPER_KEYS),
    body,
  });
}

// --- superseded_by による重複論文の一本化 -----------------------------------
// 「ファイルを消さずに重複idを畳む」ための仕組み。重複側はノードを作らず、
// 全ファイルが持つ重複idへの参照(parents/next/reports/report.papers)は正本idへ付け替える。
const allPaperIds = new Set(allPapers.map((p) => p.id));
/** 重複id -> 直接の正本id */
const supersededMap = new Map();
for (const p of allPapers) {
  if (!p.supersededBy) continue;
  if (p.supersededBy === p.id) {
    err(p.file, '`superseded_by` が自分自身を指している');
    continue;
  }
  if (!allPaperIds.has(p.supersededBy)) {
    err(p.file, `\`superseded_by\` の参照先 \`${p.supersededBy}\` が存在しない`);
    continue;
  }
  supersededMap.set(p.id, p.supersededBy);
}

/** superseded_by の連鎖を辿って正本idへ解決する(循環は致命) */
function canonicalId(id, file) {
  let cur = id;
  const seen = new Set([cur]);
  while (supersededMap.has(cur)) {
    cur = supersededMap.get(cur);
    if (seen.has(cur)) {
      err(file, `\`superseded_by\` が循環している(${[...seen].join(' -> ')} -> ${cur})`);
      return id;
    }
    seen.add(cur);
  }
  return cur;
}
for (const p of allPapers) if (p.supersededBy) canonicalId(p.id, p.file);

const papers = allPapers.filter((p) => !supersededMap.has(p.id));
for (const p of allPapers) {
  if (!p.supersededBy) continue;
  warn(p.file, `superseded_by により \`${canonicalId(p.id, p.file)}\` へ統合(ノードを生成しない)`);
}

/** 参照id配列を正本idへ解決し、重複を畳む */
function resolveRefs(list, file) {
  const out = [];
  for (const ref of list) {
    const c = canonicalId(ref, file);
    if (!out.includes(c)) out.push(c);
  }
  return out;
}
for (const p of papers) {
  p.parents = resolveRefs(p.parents, p.file).filter((r) => r !== p.id);
  p.next = resolveRefs(p.next, p.file).filter((r) => r !== p.id);
  p.reports = resolveRefs(p.reports, p.file);
  if (p.parents.length === 0) p.parents = p.categories.map((c) => `${CAT_PREFIX}${c}`);
}

// --- レポートの読み込み -----------------------------------------------------
const reports = [];
for (const path of listMarkdown(REPORTS_DIR)) {
  const file = `data/reports/${basename(path)}`;
  const parsed = parseFile(path);
  if (!parsed) continue;
  const { data, body } = parsed;

  const rawId = toNullableString(data.id);
  const title = toNullableString(data.title);
  if (!rawId) err(file, '必須フィールド `id` が無い');
  if (!title) err(file, '必須フィールド `title` が無い');
  // id は `rep:` 接頭辞を正とする(付け忘れは補完して警告)
  let id = rawId ?? file;
  if (rawId && !rawId.startsWith(REPORT_PREFIX)) {
    id = `${REPORT_PREFIX}${rawId}`;
    warn(file, `レポートidには \`${REPORT_PREFIX}\` 接頭辞を付ける(補完: ${id})`);
  }

  reports.push({
    kind: 'report',
    file,
    id,
    title: title ?? basename(path, '.md'),
    date: toDate(data.date),
    branch: toNullableString(data.branch),
    path: toNullableString(data.path),
    artifactUrl: toNullableString(data.artifactUrl),
    type: toNullableString(data.type) ?? 'md',
    papers: toList(data.papers),
    extra: collectExtra(data, KNOWN_REPORT_KEYS),
    body,
  });
}

for (const r of reports) r.papers = resolveRefs(r.papers, r.file);

// --- id重複検査 -------------------------------------------------------------
const byId = new Map();
for (const item of [...papers, ...reports]) {
  const prev = byId.get(item.id);
  if (prev) {
    err(item.file, `id \`${item.id}\` が重複している(先勝ち: ${prev.file})`);
    continue;
  }
  byId.set(item.id, item);
}

// --- カテゴリノードの自動生成 ----------------------------------------------
// 未知のカテゴリが増えても壊れないよう、参照されたカテゴリはその場で作る。
const categories = new Map();
function ensureCategory(catId) {
  if (!categories.has(catId)) {
    const label = catId.slice(CAT_PREFIX.length);
    categories.set(catId, {
      kind: 'category',
      id: catId,
      title: label,
      label,
      parent: ROOT_ID,
    });
  }
  return categories.get(catId);
}
for (const p of papers) {
  for (const c of p.categories) ensureCategory(`${CAT_PREFIX}${c}`);
  for (const parent of p.parents) if (parent.startsWith(CAT_PREFIX)) ensureCategory(parent);
}
if (reports.length > 0) ensureCategory(REPORTS_CAT);

// --- エッジ構築 + 参照先検査 -----------------------------------------------
const edges = [];
const seenEdge = new Set();
function addEdge(source, target, kind) {
  const id = `${kind}:${source}->${target}`;
  if (seenEdge.has(id)) return;
  seenEdge.add(id);
  edges.push({ id, source, target, kind });
}

/** 参照先の存在確認。無ければ致命(--allow-danglingなら警告)としてfalseを返す */
function checkRef(file, field, ref) {
  if (byId.has(ref) || categories.has(ref) || ref === ROOT_ID) return true;
  const msg = `\`${field}\` の参照先 \`${ref}\` が存在しない`;
  if (ALLOW_DANGLING) warn(file, `${msg}(--allow-dangling によりエッジを省略)`);
  else err(file, msg);
  return false;
}

for (const cat of categories.values()) addEdge(ROOT_ID, cat.id, 'parent');

for (const p of papers) {
  let attached = 0;
  for (const parent of p.parents) {
    if (checkRef(p.file, 'parents', parent)) {
      addEdge(parent, p.id, 'parent');
      attached += 1;
    }
  }
  if (attached === 0) {
    // 親が全て不在(--allow-dangling時)でもツリーから孤立させない。カテゴリへ退避する。
    for (const c of p.categories) addEdge(`${CAT_PREFIX}${c}`, p.id, 'parent');
  }
  for (const n of p.next) {
    if (checkRef(p.file, 'next', n)) addEdge(p.id, n, 'next');
  }
  for (const r of p.reports) {
    if (checkRef(p.file, 'reports', r)) addEdge(p.id, r, 'report');
  }
}
for (const r of reports) {
  addEdge(REPORTS_CAT, r.id, 'parent');
  for (const p of r.papers) {
    // レポートエッジは向きを 論文 -> レポート に正規化する
    // (paper.reports と report.papers の双方から同じ関係が来ても二重線にしない)
    if (checkRef(r.file, 'papers', p)) addEdge(p, r.id, 'report');
  }
}

// --- 出力 -------------------------------------------------------------------
const nodes = [
  { kind: 'root', id: ROOT_ID, title: 'Griffith', label: 'Griffith' },
  ...[...categories.values()].sort((a, b) => a.id.localeCompare(b.id)),
  ...papers.map(({ file, supersededBy, ...rest }) => ({ ...rest, file })),
  ...reports.map(({ file, ...rest }) => ({ ...rest, file })),
];

const years = papers.map((p) => (p.date ? Number(p.date.slice(0, 4)) : NaN)).filter(Number.isFinite);
const stubCount = papers.filter((p) => p.status === 'stub').length;
const tree = {
  generatedAt: new Date().toISOString(),
  stats: {
    papers: papers.length,
    reports: reports.length,
    categories: categories.size,
    nodes: nodes.length,
    edges: edges.length,
    stubs: stubCount,
    superseded: supersededMap.size,
    stubRatio: papers.length ? Number((stubCount / papers.length).toFixed(3)) : 0,
    yearMin: years.length ? Math.min(...years) : null,
    yearMax: years.length ? Math.max(...years) : null,
    edgesByKind: edges.reduce((acc, e) => ({ ...acc, [e.kind]: (acc[e.kind] ?? 0) + 1 }), {}),
  },
  nodes,
  edges,
};

if (!QUIET && warnings.length > 0) {
  console.warn(`[build-tree] 警告 ${warnings.length} 件:`);
  for (const w of warnings) console.warn(`  - ${w}`);
}

if (errors.length > 0) {
  console.error(`[build-tree] 検査エラー ${errors.length} 件(tree.jsonは更新しない):`);
  for (const e of errors) console.error(`  - ${e}`);
  process.exit(1);
}

mkdirSync(dirname(OUT_PATH), { recursive: true });
writeFileSync(OUT_PATH, `${JSON.stringify(tree, null, 2)}\n`, 'utf8');
console.log(
  `[build-tree] OK -> ${OUT_PATH.replace(`${ROOT}/`, '')} ` +
    `(papers=${papers.length} reports=${reports.length} categories=${categories.size} ` +
    `nodes=${nodes.length} edges=${edges.length} stubs=${stubCount} superseded=${supersededMap.size})`,
);
