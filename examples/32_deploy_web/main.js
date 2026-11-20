// Web デプロイ例: ./mdl_web.js の loadBundle だけでバンドルを読み、
// 同梱サンプルを推論して「予測クラス・正解率・µs/推論」を表示する最小ページ。
// (配布物の既定シェル demo.js は使わず、自前のページから公開APIを呼ぶ例)
import { loadBundle } from './mdl_web.js';

const $ = (id) => document.getElementById(id);
const params = new URL(location.href).searchParams;
// ?model=<url> / ?samples=<url> で読み込み先を差し替えられる
const MODEL_URL = params.get('model') ?? './model.bundle.json';
const SAMPLES_URL = params.get('samples') ?? './samples.json';
// 計測は「全サンプルを1周」を REPEAT 回。performance.now() は粒度が粗いので
// 1回ずつ測らず、全体の時間を推論回数で割って µs/推論 にする。
const REPEAT = 2000;

async function fetchJson(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url}: HTTP ${res.status}`);
  return res.json();
}

/** スコア(整数)の argmax = 予測クラス。 */
function argmax(scores) {
  let best = 0;
  for (let i = 1; i < scores.length; i++) if (scores[i] > scores[best]) best = i;
  return best;
}

async function run(engine) {
  $('status').textContent = `engine=${engine} で読み込み中…`;
  $('rows').innerHTML = '';
  try {
    const [doc, samplesDoc] = await Promise.all([fetchJson(MODEL_URL), fetchJson(SAMPLES_URL)]);
    // engine: 'auto' は wasm を試し、失敗したら notes に理由を残して js へ落ちる
    const bundle = await loadBundle(doc, { engine });
    const samples = samplesDoc.samples;

    // 1サンプルずつ推論して表に出す。bits は 0/1 の配列(長さ = 入力ビット幅)、batch=1。
    let correct = 0;
    let labeled = 0;
    samples.forEach((s, i) => {
      const { data } = bundle.scores(Uint8Array.from(s.bits), 1);
      const pred = argmax(data);
      const hasLabel = s.label !== null && s.label !== undefined;
      if (hasLabel) { labeled++; if (pred === s.label) correct++; }
      const tr = document.createElement('tr');
      const verdict = !hasLabel ? '' : pred === s.label ? '<span class="ok">一致</span>' : '<span class="ng">不一致</span>';
      tr.innerHTML = `<td>${i}</td><td><code>${s.bits.join('')}</code></td>`
        + `<td><code>[${Array.from(data).join(', ')}]</code></td><td>${pred}</td>`
        + `<td>${hasLabel ? s.label : '—'}</td><td>${verdict}</td>`;
      $('rows').appendChild(tr);
    });

    // 計測: ウォームアップ1周のあと REPEAT 周
    const inputs = samples.map((s) => Uint8Array.from(s.bits));
    for (const bits of inputs) bundle.scores(bits, 1);
    const t0 = performance.now();
    for (let r = 0; r < REPEAT; r++) for (const bits of inputs) bundle.scores(bits, 1);
    const us = ((performance.now() - t0) * 1000) / (REPEAT * inputs.length);

    const acc = labeled ? `${correct}/${labeled}(${((100 * correct) / labeled).toFixed(1)}%)` : '—';
    $('summary').textContent =
      `実エンジン: ${bundle.engine} / kind: ${bundle.kind} / 正解率: ${acc} / `
      + `${us.toFixed(2)} µs/推論(${REPEAT * inputs.length}回の平均)`;
    $('status').textContent = '完了';
    document.title = `完了 ${bundle.engine} ${us.toFixed(2)}us`;

    $('notes').innerHTML = '';
    for (const n of bundle.notes.length ? bundle.notes : ['(注記なし)']) {
      const li = document.createElement('li');
      li.textContent = n;
      $('notes').appendChild(li);
    }
  } catch (e) {
    $('status').textContent = `失敗: ${e.message}`;
    console.error(e);
  }
}

$('engine').value = params.get('engine') ?? 'auto';
$('engine').addEventListener('change', () => run($('engine').value));
run($('engine').value);
