// Mermaid 図の構文検証ワーカー(scripts/check_docs_mermaid.py から呼ばれる)。
//
// 標準入力に {"blocks": [{"id": 0, "source": "graph TD\n..."}, ...]} を受け取り、
// 各ブロックを mermaid.parse() に通して
// {"results": [{"id": 0, "ok": true, "diagramType": "flowchart-v2"} | {"id": 0, "ok": false, "error": "..."}]}
// を標準出力に返す。docs-site/node_modules の mermaid をそのまま使うので、
// **サイトが実際に描画するのと同じバージョン・同じパーサ**で検査できる。
//
// ブラウザ前提の mermaid を Node で動かすための最小の細工(jsdom は入れない):
//   1. window / document / addEventListener を最小スタブで用意する
//      (mermaid はトップレベルで window.addEventListener('load', ...) を呼ぶ)
//   2. DOMPurify を先に import して no-op のフックAPIを生やす
//      DOM が無い環境の DOMPurify は isSupported=false で addHook 等を定義せずに返るため、
//      mermaid が flowchart のパース中に呼ぶ DOMPurify.addHook で TypeError になる。
//      ESM のモジュールキャッシュを共有しているので、ここで生やせば mermaid 側にも効く。
// parse はレイアウト・描画を行わない(= サイズ計測の DOM を必要としない)ので、
// この程度のスタブで全図種を検証できる。

const g = globalThis;

const stubElement = () => ({
  style: {},
  setAttribute() {},
  appendChild() {},
  removeChild() {},
  remove() {},
  firstChild: null,
  innerHTML: '',
  getBoundingClientRect: () => ({ width: 0, height: 0, x: 0, y: 0 }),
});

g.window = g;
g.addEventListener = () => {};
g.removeEventListener = () => {};
g.document = {
  querySelector: () => null,
  querySelectorAll: () => [],
  getElementById: () => null,
  createElement: stubElement,
  createElementNS: stubElement,
  body: { appendChild() {}, removeChild() {} },
  documentElement: { style: {} },
  addEventListener: () => {},
};

const DOMPurify = (await import('dompurify')).default;
for (const name of ['addHook', 'removeHook', 'removeAllHooks', 'setConfig', 'clearConfig']) {
  if (typeof DOMPurify[name] !== 'function') DOMPurify[name] = () => {};
}
if (typeof DOMPurify.sanitize !== 'function') DOMPurify.sanitize = (html) => String(html);

const mermaid = (await import('mermaid')).default;
mermaid.initialize({ startOnLoad: false, suppressErrorRendering: true });

const stdin = await new Promise((resolve, reject) => {
  let buf = '';
  process.stdin.setEncoding('utf8');
  process.stdin.on('data', (chunk) => { buf += chunk; });
  process.stdin.on('end', () => resolve(buf));
  process.stdin.on('error', reject);
});

const { blocks } = JSON.parse(stdin);
const results = [];

for (const block of blocks) {
  try {
    const parsed = await mermaid.parse(block.source);
    results.push({ id: block.id, ok: true, diagramType: parsed?.diagramType ?? null });
  } catch (err) {
    results.push({ id: block.id, ok: false, error: String(err?.message ?? err) });
  }
}

process.stdout.write(JSON.stringify({ results }));
