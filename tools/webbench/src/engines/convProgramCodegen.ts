// conv2d 命令列(convProgram.ts)→ JS直線コード生成(codegen)。
//
// 動機: 命令列インタプリタ(bitplane.ts::BitplaneProgramConvKernel)は
// 「命令ごとに tapRowsWords 語のループを回す」形なので、命令1つあたり
//   switch分岐 + ループ立ち上げ + 2ロード/1ストア × n語
// を払う。実モデルの n(=HOut*wprOut、1平面のワード数)は **24〜512語**
// しかない(det: 288/96/24/24/24/24、seg: 128/32/16/32/128)ため、
// 語あたりの実効コストに占めるディスパッチ/ループ立ち上げの比率が大きい。
//
// このモジュールはループ構造を反転させる:
//   旧: for 命令 { for 語 { load,load,op,store } }
//   新: for 語 { 全命令をインライン(中間値はローカル変数=レジスタ) }
// 中間値のストア/ロードが丸ごと消え、残る記憶域アクセスは
// 「タップ平面の読み(命令列の外部入力)」と「出力平面への書き(oc)」だけになる。
//
// レジスタ圧の見積り: 命令列の同時生存数は生存区間解析の結果 nSlots に等しく、
// 実モデルでは **3〜6**(convProgram.ts が木を深さ優先順で辿るため)。
// つまり1語ぶんの本体で同時に生きるローカルは数個であり、命令数が1000を
// 超えてもスピルは起きない。にもかかわらず関数を CHUNK 分割するのは
// V8 の**最適化対象バイトコードサイズ上限**(巨大関数はTurboFan最適化が
// 見送られる)への保険で、dense版 codegen.ts の CHUNK=1500 と同じ理由。
// conv版は分割点をまたぐ値を「元の命令列のスロット」経由で受け渡す
// (= 生存区間解析の結果をそのまま再利用する)ので、分割コストは
// 「境界で生きている高々 nSlots 本の値のストア/ロード」だけで済む。
//
// 正しさの契約: **整数出力完全不変**。インタプリタと同じ
//   - 命令順(トポロジカル順)
//   - 参照解決(定数平面/タップ平面プール/スロット)
//   - 出力書き出し(oc ごとに HOut*wprOut 語 + 行末語への tailMask)
// を保つ。convProgramCodegen.test.ts が実5モデル+ランダム木で照合する。
//
// CSP: `new Function` は unsafe-eval を禁じた環境で throw する。呼び出し側
// (bitplane.ts)は throw を捕まえてインタプリタへ黙って戻すのではなく、
// 理由を note に残すこと(codegen.ts と同方針)。
import type { ConvProgramLayout } from './bitplane';
import { REF_KIND_CONST, REF_KIND_INSTR, REF_KIND_TAP, refIdx, refKind } from './convProgram';
import type { ConvRef } from './convProgram';

/** 生成されたチャンク関数。s=スクラッチ, o=出力ワード列, ooff=バッチ先頭。 */
export type ConvChunkFn = (s: Int32Array, o: Int32Array, ooff: number) => void;

export interface ConvCodegenOptions {
  /** 1関数あたりの最大**生成ステートメント数**(既定 DEFAULT_CHUNK)。
      命令数ではない: oc出力の書き出しが命令に張り付くため、OCの大きいノードは
      命令数が同じでも関数が数倍になる(det n5: 命令919 / oc書き1024)。 */
  chunkSize?: number;
  /** チャンク内で何回以上参照される外部ロードを語頭へ巻き上げるか(既定2)。 */
  hoistMinUses?: number;
}

export interface ConvCodegenStats {
  /** 生成した関数の本数(前置きチャンクを含む)。 */
  nChunks: number;
  /** 実効チャンクサイズ。 */
  chunkSize: number;
  /** 生成したJSソースの総文字数。 */
  sourceChars: number;
  /** new Function(パース+関数オブジェクト生成)の合計ms。 */
  compileMs: number;
  /** 巻き上げた外部ロードの本数(全チャンク合計)。 */
  hoisted: number;
  /** チャンク境界をまたぐためスロットへ書き戻す命令の数。 */
  slotStores: number;
  /** 命令数(参考)。 */
  nInstrs: number;
}

export interface CompiledConvProgram {
  fns: ConvChunkFn[];
  stats: ConvCodegenStats;
  /** 生成したJSソース(チャンクごと)。デバッグ/掃引用に保持する
      (実モデルで34〜128KB。モデルJSON本体に比べれば無視できる)。 */
  sources: string[];
}

/** 1関数あたりの既定ステートメント数。
    C4の掃引(5モデル×128/256/384/512/768)では 128〜768 は概ね横並び
    (±3%)で、256 が5モデル中3モデルで最良・残り2モデルも最良から2%以内
    だったのでこれを既定にする。**命令数**で切っていた版では det/detmono の
    最大ノード(命令919/1268・OC=1024)で 512以上にすると **6〜8倍の崖** が
    出た(1関数のバイトコードがV8の最適化対象サイズ上限を超え、TurboFanに
    上がらずIgnitionのままになる)。ステートメント数で切ればoc書き出しの
    多いノードも自動的に細かく割れるので、この崖は消える。 */
export const DEFAULT_CHUNK = 256;

/** n=2の16ゲート最短式(codegen.ts::gateExpr と同一の表)。
    オペランド式は `0` / `-1` / 識別子 / `s[K+w]` のいずれかで、
    単項マイナスはビット演算より強く束縛するので追加の括弧は不要。 */
function gateExpr(tt: number, a: string, b: string): string {
  switch (tt & 0xf) {
    case 0x0: return '0';
    case 0x1: return `~(${a}|${b})`;
    case 0x2: return `${a}&~${b}`;
    case 0x3: return `~${b}`;
    case 0x4: return `~${a}&${b}`;
    case 0x5: return `~${a}`;
    case 0x6: return `${a}^${b}`;
    case 0x7: return `~(${a}&${b})`;
    case 0x8: return `${a}&${b}`;
    case 0x9: return `~(${a}^${b})`;
    case 0xa: return a;
    case 0xb: return `${a}|~${b}`;
    case 0xc: return b;
    case 0xd: return `~${a}|${b}`;
    case 0xe: return `${a}|${b}`;
    default: return '-1';
  }
}

/** 命令jを最後に読む命令の添字(誰も読まないなら -1)。
    チャンク境界をまたぐかの判定に使う(oc出力はローカルから直接書くので
    ここには数えない)。 */
function lastInstrUse(L: ConvProgramLayout): Int32Array {
  const instrs = L.program.instrs;
  const last = new Int32Array(instrs.length).fill(-1);
  for (let j = 0; j < instrs.length; j++) {
    for (const r of [instrs[j].a, instrs[j].b]) {
      if (refKind(r) === REF_KIND_INSTR) last[refIdx(r)] = j;
    }
  }
  return last;
}

/** conv2d命令列をワード単位の直線JSコードへコンパイルする。
    `new Function` が使えない環境(CSP unsafe-eval禁止)では throw する。 */
export function compileConvProgram(
  L: ConvProgramLayout,
  opts: ConvCodegenOptions = {},
): CompiledConvProgram {
  const n = L.tapRowsWords;
  const planeWords = L.HOut * L.wprOut;
  if (planeWords !== n) {
    // tapWpr===wprOut は現行の幾何(高速路/strided路とも)で常に成り立つ。
    // 崩れたら黙って別物を生成せず、呼び出し側をインタプリタへ戻す。
    throw new Error(`convProgramCodegen: planeWords(${planeWords}) !== tapRowsWords(${n})`);
  }
  const chunkSize = Math.max(1, Math.floor(opts.chunkSize ?? DEFAULT_CHUNK));
  const hoistMin = Math.max(2, Math.floor(opts.hoistMinUses ?? 2));
  const prog = L.program;
  const instrs = prog.instrs;
  const nInstr = instrs.length;
  const last = lastInstrUse(L);

  // タップ/スロットの絶対ワードオフセット(instrFlat と同じ解決規則)。
  const tapOff = (t: number): number => L.tapWordBase[t];
  const slotOff = (j: number): number => L.regBase + instrs[j].slot * n;

  // oc の書き出し先を「生成元」で分類する。
  const ocPre: number[] = []; // 出力が定数/タップ直結(命令の前に書ける)
  const ocByInstr = new Map<number, number[]>();
  for (let oc = 0; oc < L.OC; oc++) {
    const r = prog.ocOut[oc];
    if (refKind(r) === REF_KIND_INSTR) {
      const j = refIdx(r);
      const arr = ocByInstr.get(j);
      if (arr) arr.push(oc); else ocByInstr.set(j, [oc]);
    } else {
      ocPre.push(oc);
    }
  }

  const sources: string[] = [];
  let hoisted = 0;
  let slotStores = 0;

  /** 参照 → 「外部ロードの絶対オフセット」または null(チャンク内ローカル/定数)。 */
  const externalOff = (r: ConvRef, lo: number): number | null => {
    const k = refKind(r);
    if (k === REF_KIND_CONST) return null;
    if (k === REF_KIND_TAP) return tapOff(refIdx(r));
    return refIdx(r) >= lo ? null : slotOff(refIdx(r));
  };

  /** チャンクを切る単位は「命令数」ではなく **生成ステートメント数**の見積り。
      内訳は 1(命令本体) + 1(巻き上げロード/スロット書き戻しの償却分) +
      その命令に張り付く oc 書き出しの本数。oc書き出しが命令に張り付くため、
      OCの大きいノード(det n5: 命令919 に対し oc書き1024)は命令数が同じでも
      関数が数倍に膨らむ。命令数で切ると、その大きい関数だけが V8 の
      最適化対象サイズ上限を超えて Ignition のまま取り残され、実測で
      6〜8倍の崖が出た(C4測定)。 */
  const stmtCost = (j: number): number => 2 + (ocByInstr.get(j)?.length ?? 0);

  const buildChunk = (lo: number, hi: number, pre: boolean): string => {
    // --- 巻き上げ対象(このチャンクで2回以上読む外部オフセット)を数える ---
    const useCount = new Map<number, number>();
    const bump = (off: number | null): void => {
      if (off !== null) useCount.set(off, (useCount.get(off) ?? 0) + 1);
    };
    for (let j = lo; j < hi; j++) {
      bump(externalOff(instrs[j].a, lo));
      bump(externalOff(instrs[j].b, lo));
    }
    if (pre) for (const oc of ocPre) bump(externalOff(prog.ocOut[oc], lo));
    const hoistName = new Map<number, string>();
    for (const [off, c] of useCount) {
      if (c >= hoistMin) hoistName.set(off, `h${hoistName.size}`);
    }
    hoisted += hoistName.size;

    const refExpr = (r: ConvRef): string => {
      const k = refKind(r);
      if (k === REF_KIND_CONST) return refIdx(r) === 0 ? '0' : '-1';
      const off = externalOff(r, lo);
      if (off === null) return `v${refIdx(r)}`;
      const h = hoistName.get(off);
      return h ?? `s[${off}+w]`;
    };

    const body: string[] = [];
    // 語頭の巻き上げロード(このチャンクではタップ/境界スロットへ書かないので
    // 「本体のどのストアより前に読む」ことが常に安全: 生存区間解析により
    // チャンク内のストア先スロットは、ここで読む値のスロットと重ならない)。
    for (const [off, name] of hoistName) body.push(`const ${name}=s[${off}+w];`);

    // 行末語へ掛ける tailMask の掛け方。wprOut===1 なら全語が行末なので
    // 定数を直に埋め込める(1回転しかしない内側ループを作らないため重要)。
    const maskSuffix = L.maskOut === -1 ? ''
      : (L.wprOut === 1 ? `&${L.maskOut}` : '&m');
    let writesOutput = false;
    const emitOcWrites = (ocs: number[], valExpr: string): void => {
      writesOutput = true;
      for (const oc of ocs) body.push(`o[q+${oc * planeWords}]=${valExpr}${maskSuffix};`);
    };

    if (pre && ocPre.length > 0) {
      // 出力が定数/タップ直結の oc(インタプリタのバケット0と同じ集合)。
      const byExpr = new Map<string, number[]>();
      for (const oc of ocPre) {
        const e = refExpr(prog.ocOut[oc]);
        const arr = byExpr.get(e);
        if (arr) arr.push(oc); else byExpr.set(e, [oc]);
      }
      for (const [e, ocs] of byExpr) emitOcWrites(ocs, e);
    }

    for (let j = lo; j < hi; j++) {
      const ins = instrs[j];
      body.push(`const v${j}=${gateExpr(ins.tt, refExpr(ins.a), refExpr(ins.b))};`);
      if (last[j] >= hi) {
        // チャンク境界をまたぐ値だけスロットへ書き戻す。
        body.push(`s[${slotOff(j)}+w]=v${j};`);
        slotStores++;
      }
      const ocs = ocByInstr.get(j);
      if (ocs) emitOcWrites(ocs, `v${j}`);
    }

    // 出力書き出しの基点 q=ooff+w は oc ごとに再計算せず1回だけ作る
    // (OC=256〜1024 のノードでは語あたり OC 回の加算がそのまま効く)。
    const inner = (writesOutput ? 'const q=ooff+w;' : '') + body.join('');
    // ループ構造: 行末語へ tailMask が要るチャンクだけ「行 × 行内語」の
    // 二重ループにする(maskOut===-1 や wprOut===1 のノードは単純ループ)。
    if (writesOutput && maskSuffix === '&m') {
      const wpr = L.wprOut;
      return `"use strict";for(let r=0;r<${n};r+=${wpr}){for(let k=0;k<${wpr};k++){`
        + `const w=r+k;const m=k===${wpr - 1}?${L.maskOut}:-1;${inner}}}`;
    }
    return `"use strict";for(let w=0;w<${n};w++){${inner}}`;
  };

  // 出力が定数/タップ直結の oc は命令に依存しないので独立した前置き関数にする。
  const ranges: Array<[number, number, boolean]> = [];
  if (ocPre.length > 0) ranges.push([0, 0, true]);
  let lo = 0;
  let cost = 0;
  for (let j = 0; j < nInstr; j++) {
    const c = stmtCost(j);
    if (j > lo && cost + c > chunkSize) {
      ranges.push([lo, j, false]);
      lo = j;
      cost = 0;
    }
    cost += c;
  }
  if (nInstr > lo) ranges.push([lo, nInstr, false]);
  for (const [a, b, pre] of ranges) sources.push(buildChunk(a, b, pre));

  const t0 = performance.now();
  const fns: ConvChunkFn[] = sources.map(
    // eslint-disable-next-line @typescript-eslint/no-implied-eval
    (src) => new Function('s', 'o', 'ooff', src) as ConvChunkFn,
  );
  const compileMs = performance.now() - t0;

  return {
    fns,
    stats: {
      nChunks: fns.length,
      chunkSize,
      sourceChars: sources.reduce((a, s) => a + s.length, 0),
      compileMs,
      hoisted,
      slotStores,
      nInstrs: nInstr,
    },
    sources,
  };
}
