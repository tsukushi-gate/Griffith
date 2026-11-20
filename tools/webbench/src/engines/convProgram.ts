// conv2d 重み共有木の lossless 命令列化(ゲート削減)。
//
// DLGN の conv2d ノードは「出力チャネル(oc)ごとの深さd・2入力ゲート二分木」
// (indices_flat[oc][leaf] + truth_tables oc-major)で表される。残差初期化の
// 影響で学習済みの木にはパススルー('A'=0xa=左入力素通し)ゲートが多数あり、
// さらに定数ゲート(0x0/0xf)や、oc をまたいだ重複サブ木が大量にある。
// 現行カーネルは全 oc を同じ深さのバランス木として「全ゲート」評価するため、
// これらは全て無駄なワード演算になる。
//
// このモジュールは木を「命令列(instruction program)」へ落とす純関数群:
//   1. 定数畳み込み  : 片入力が定数の2入力ゲートを16種×定数0/1×左右の表で簡約
//   2. 恒等/単項縮約 : 0xa→a、0xc→b、0x5→¬a、0x3→¬b(NOTは単項命令として保持)
//   3. NOT吸収       : 入力がNOT命令なら親の tt を入力反転側へ変換して取り込む
//   4. 可換正規化    : 可換ttは(a,b)を参照IDでソート、非可換は入替時のtt変換
//   5. CSE           : (tt,a,b) キーで oc内・oc間の両方を dedup(タップIDが
//                      共有なので oc 横断が自然に効く)
//   6. dead除去 + トポロジカル番号付け + レジスタ生存区間解析(スロット割当)
//
// 設計方針: **bitplane / wasm / webgpu のどれにも依存しない純粋な TS**。
// 入力は木の構造(タップID列 + 数値化truth table)だけ、出力は
// 「タップ参照・定数参照・命令参照」で書かれた命令列。ビットプレーンの
// 平面レイアウトやワードオフセットへの解決は呼び出し側(bitplane.ts /
// wasmBitplaneConv.ts)の責務。node からも素の .ts のまま import できる
// (実行時 import を持たない = 型のみ import)ことを維持すること
// (scripts/gate_stats.mjs が Node の type stripping で直接読む)。
//
// 正しさの契約: **整数出力完全不変**。convProgram.test.ts のランダム木
// property test と、実行時ロード時の verifyConvProgram(全数真理値表照合)
// の二重で守る。

// ============================================================
// 参照(Ref)エンコード: (idx << 2) | kind
// ============================================================

/** 定数参照(idx=0 なら全ビット0、idx=1 なら全ビット1)。 */
export const REF_KIND_CONST = 0;
/** タップ(木の葉 = 入力窓の1点)参照。idx は ConvProgram.taps の添字。 */
export const REF_KIND_TAP = 1;
/** 命令(先行する instrs の結果)参照。idx は ConvProgram.instrs の添字。 */
export const REF_KIND_INSTR = 2;

/** 命令のオペランド参照。非負整数(下位2bitがkind、上位がidx)。 */
export type ConvRef = number;

export function makeRef(kind: number, idx: number): ConvRef {
  return (idx << 2) | kind;
}
export function refKind(r: ConvRef): number {
  return r & 3;
}
export function refIdx(r: ConvRef): number {
  return r >>> 2;
}

export const REF_ZERO: ConvRef = makeRef(REF_KIND_CONST, 0);
export const REF_ONE: ConvRef = makeRef(REF_KIND_CONST, 1);

// ============================================================
// 2入力 truth table (tt) の代数
// ============================================================

/** tt のビット規約: bit index i = (b<<1)|a、値 = f(a,b)。
    packedConv.ts / bitplane.ts / assembly/index.ts の gate2 16式と同一
    (例: 0x8=a&b, 0xa=a, 0xc=b, 0x2=a&~b, 0xb=a|~b)。 */
export function ttEval(tt: number, a: number, b: number): number {
  return (tt >>> (((b & 1) << 1) | (a & 1))) & 1;
}

/** 4パターンの値から tt を組み立てる。 */
function ttFrom(f: (a: number, b: number) => number): number {
  let tt = 0;
  for (let b = 0; b < 2; b++) {
    for (let a = 0; a < 2; a++) {
      if (f(a, b)) tt |= 1 << ((b << 1) | a);
    }
  }
  return tt & 0xf;
}

function buildTable(map: (tt: number) => number): Uint8Array {
  const t = new Uint8Array(16);
  for (let tt = 0; tt < 16; tt++) t[tt] = map(tt);
  return t;
}

/** g(a,b) = f(b,a) の tt(オペランド入替)。bit1 と bit2 の交換。 */
export const TT_SWAP: Uint8Array = buildTable((tt) => ttFrom((a, b) => ttEval(tt, b, a)));
/** g(a,b) = f(~a,b) の tt(a入力の反転吸収)。 */
export const TT_NOT_A: Uint8Array = buildTable((tt) => ttFrom((a, b) => ttEval(tt, a ^ 1, b)));
/** g(a,b) = f(a,~b) の tt(b入力の反転吸収)。 */
export const TT_NOT_B: Uint8Array = buildTable((tt) => ttFrom((a, b) => ttEval(tt, a, b ^ 1)));

export function ttSwap(tt: number): number {
  return TT_SWAP[tt & 0xf];
}

/** tt が可換(f(a,b)===f(b,a))か。0x0,0x1,0x6,0x7,0x8,0x9,0xe,0xf。 */
export function ttIsCommutative(tt: number): boolean {
  return TT_SWAP[tt & 0xf] === (tt & 0xf);
}

// --- 単項簡約コード ---------------------------------------------------------
/** 結果は定数0。 */
export const RED_CONST0 = 0;
/** 結果は定数1。 */
export const RED_CONST1 = 1;
/** 結果は対象オペランドの素通し(alias)。 */
export const RED_ID = 2;
/** 結果は対象オペランドの否定(NOT単項命令)。 */
export const RED_NOT = 3;
/** 簡約不能(2入力のまま)。 */
export const RED_BINARY = 4;

/** (g0,g1) = (f(0),f(1)) から単項コードへ。 */
function redOf(g0: number, g1: number): number {
  if (g0 === 0 && g1 === 0) return RED_CONST0;
  if (g0 === 1 && g1 === 1) return RED_CONST1;
  return g0 === 0 ? RED_ID : RED_NOT;
}

/** 「b が定数 kb のとき、残る a についての簡約コード」表(16tt × kb∈{0,1})。
    添字 = tt*2 + kb。全16ttについて網羅(定数畳み込みの真実源)。 */
export const CONST_FOLD_B: Uint8Array = (() => {
  const t = new Uint8Array(16 * 2);
  for (let tt = 0; tt < 16; tt++) {
    for (let kb = 0; kb < 2; kb++) {
      t[tt * 2 + kb] = redOf(ttEval(tt, 0, kb), ttEval(tt, 1, kb));
    }
  }
  return t;
})();

/** 「a が定数 ka のとき、残る b についての簡約コード」表(16tt × ka∈{0,1})。
    添字 = tt*2 + ka。 */
export const CONST_FOLD_A: Uint8Array = (() => {
  const t = new Uint8Array(16 * 2);
  for (let tt = 0; tt < 16; tt++) {
    for (let ka = 0; ka < 2; ka++) {
      t[tt * 2 + ka] = redOf(ttEval(tt, ka, 0), ttEval(tt, ka, 1));
    }
  }
  return t;
})();

/** tt 自体が片方の入力に依存しない(退化)場合のコードとどちらの入力か。
    値: RED_* コード、対象入力は TT_DEGEN_ON_B[tt](0=a,1=b)。
    RED_CONST0/1 のときは入力に依存しない。 */
export const TT_DEGEN: Uint8Array = buildTable((tt) => {
  const dependsA = ttEval(tt, 0, 0) !== ttEval(tt, 1, 0) || ttEval(tt, 0, 1) !== ttEval(tt, 1, 1);
  const dependsB = ttEval(tt, 0, 0) !== ttEval(tt, 0, 1) || ttEval(tt, 1, 0) !== ttEval(tt, 1, 1);
  if (!dependsA && !dependsB) return ttEval(tt, 0, 0) ? RED_CONST1 : RED_CONST0;
  if (dependsA && dependsB) return RED_BINARY;
  if (dependsA) return ttEval(tt, 0, 0) ? RED_NOT : RED_ID; // f(a,*)
  return ttEval(tt, 0, 0) ? RED_NOT : RED_ID; // f(*,b)
});
/** TT_DEGEN が RED_ID/RED_NOT のとき、その対象が b 側なら 1、a 側なら 0。 */
export const TT_DEGEN_ON_B: Uint8Array = buildTable((tt) => {
  const dependsA = ttEval(tt, 0, 0) !== ttEval(tt, 1, 0) || ttEval(tt, 0, 1) !== ttEval(tt, 1, 1);
  return dependsA ? 0 : 1;
});

/** 同一参照が両入力に来たとき f(a,a) の簡約コード(g0=bit0, g1=bit3)。 */
export const SAME_FOLD: Uint8Array = buildTable((tt) => redOf(tt & 1, (tt >>> 3) & 1));

/** NOT を表す単項命令の tt(¬a、b は a と同一参照を置く)。 */
export const TT_NOT: number = 0x5;

// ============================================================
// 入出力の型
// ============================================================

/** 命令列化の入力(Conv2dParams からビットプレーン非依存の情報だけ抜いたもの)。 */
export interface ConvTreeSpec {
  /** 出力チャネル数。 */
  outChannels: number;
  /** 木の深さ(n_inputs=2 固定なので nLeaves = 2**treeDepth)。 */
  treeDepth: number;
  /** [OC][nLeaves] のタップID(indices_flat のまま。値の意味は呼び出し側の規約)。 */
  indicesFlat: ArrayLike<ArrayLike<number>>;
  /** [OC*nInternal] の数値化truth table(0..15、oc-major・レベル昇順)。 */
  truthTables: ArrayLike<number>;
}

export interface ConvInstr {
  /** 2入力truth table(0..15)。TT_NOT(0x5) かつ a===b は単項NOT命令。 */
  tt: number;
  a: ConvRef;
  b: ConvRef;
  /** 生存区間解析で割り当てた作業平面スロット(0..nSlots-1)。 */
  slot: number;
}

export interface ConvProgramStats {
  /** 縮約前のゲート数(OC * nInternal)。 */
  gatesBefore: number;
  /** 縮約後の命令数(dead除去後)。 */
  gatesAfter: number;
  /** 元ゲートのうち定数へ畳まれた数。 */
  constFolded: number;
  /** 元ゲートのうち入力素通し(恒等)になった数。 */
  aliasFolded: number;
  /** 元ゲートのうち単項NOTへ縮約された数。 */
  notFolded: number;
  /** 元ゲートのうち既存命令と一致(CSE)して消えた数。 */
  cseHits: number;
  /** 元ゲートのうち新規命令を発行した数。 */
  emitted: number;
  /** 発行された命令の総数(dead除去前)。 */
  instrsCreated: number;
  /** dead除去で消えた命令数。 */
  deadRemoved: number;
  /** 命令列中のNOT(単項)命令数。 */
  notInstrs: number;
  /** 木の葉に現れるユニークタップ数(未使用除去前)。 */
  tapsBefore: number;
  /** 実際に読まれるユニークタップ数(= 実体化するタップ平面の元)。 */
  nTaps: number;
  /** 作業平面スロット数(生存区間解析の結果)。 */
  nSlots: number;
  /** 出力が定数になった oc の数。 */
  ocConst: number;
  /** 出力がタップ直結になった oc の数。 */
  ocTap: number;
  /** ocOut のユニーク参照数(< OC なら重複出力チャネルがある)。 */
  ocUnique: number;
}

export interface ConvProgram {
  /** ユニークタップ(indices_flat の値、初出順)。 */
  taps: number[];
  /** トポロジカル順の命令列(instr j のオペランドは taps/定数/instr k<j のみ)。 */
  instrs: ConvInstr[];
  /** 作業平面スロット数。 */
  nSlots: number;
  /** 各 oc の出力参照。 */
  ocOut: ConvRef[];
  /** 出力書き出しスケジュール(CSR)。バケット0 = プログラム実行前
      (出力が定数/タップ直結の oc)、バケット j+1 = 命令 j の実行直後。
      長さ instrs.length+2。 */
  writeStart: Int32Array;
  /** writeStart が指す oc 番号の並び(長さ OC)。 */
  writeOc: Int32Array;
  stats: ConvProgramStats;
}

// ============================================================
// 命令列ビルダ
// ============================================================

interface DraftInstr {
  tt: number;
  a: ConvRef;
  b: ConvRef;
}

/** conv2d の重み共有木(全oc)を lossless に命令列へ縮約する。
    元のバランス木の評価順(レベル0: 葉(2g,2g+1) → 上位レベル)は
    bitplane.ts::evalConv2dBitplane / assembly/index.ts と厳密に同一。 */
export function buildConvProgram(spec: ConvTreeSpec): ConvProgram {
  const OC = spec.outChannels;
  const nLeaves = 1 << spec.treeDepth;
  const nInternal = nLeaves - 1; // n_inputs=2 固定: (nLeaves-1)/(2-1)

  const taps: number[] = [];
  const tapIdOf = new Map<number, number>();
  const draft: DraftInstr[] = [];
  const cse = new Map<string, number>();
  const ocOut: ConvRef[] = new Array<ConvRef>(OC);

  let constFolded = 0;
  let aliasFolded = 0;
  let notFolded = 0;
  let cseHits = 0;
  let emitted = 0;

  /** 新規/既存の2入力命令参照を返す(CSE)。newlyCreated を out パラメタで返す。 */
  let lastWasNew = false;
  const rawEmit = (tt: number, a: ConvRef, b: ConvRef): ConvRef => {
    const key = `${tt}:${a}:${b}`;
    const hit = cse.get(key);
    if (hit !== undefined) {
      lastWasNew = false;
      return makeRef(REF_KIND_INSTR, hit);
    }
    const idx = draft.length;
    draft.push({ tt, a, b });
    cse.set(key, idx);
    lastWasNew = true;
    return makeRef(REF_KIND_INSTR, idx);
  };

  /** x の否定参照(定数は即畳み、NOT命令の二重否定は消去)。 */
  const emitNot = (x: ConvRef): ConvRef => {
    if (refKind(x) === REF_KIND_CONST) {
      lastWasNew = false;
      return refIdx(x) === 0 ? REF_ONE : REF_ZERO;
    }
    if (refKind(x) === REF_KIND_INSTR) {
      const d = draft[refIdx(x)];
      if (d.tt === TT_NOT && d.a === d.b) {
        lastWasNew = false;
        return d.a; // ¬¬y = y
      }
    }
    return rawEmit(TT_NOT, x, x);
  };

  /** 単項簡約コードを参照へ適用する。 */
  const applyRed = (code: number, x: ConvRef): ConvRef => {
    if (code === RED_CONST0) return REF_ZERO;
    if (code === RED_CONST1) return REF_ONE;
    if (code === RED_ID) return x;
    return emitNot(x);
  };

  /** 1ゲート分の縮約。統計分類も行う。 */
  const emitGate = (tt0: number, a0: ConvRef, b0: ConvRef): ConvRef => {
    let tt = tt0 & 0xf;
    let a = a0;
    let b = b0;

    // (3) NOT吸収: オペランドがNOT命令ならその入力へ付け替えて tt を変換する
    // (NOTの二重否定は emitNot で消してあるので各辺1回で十分)。
    if (refKind(a) === REF_KIND_INSTR) {
      const d = draft[refIdx(a)];
      if (d.tt === TT_NOT && d.a === d.b) {
        a = d.a;
        tt = TT_NOT_A[tt];
      }
    }
    if (refKind(b) === REF_KIND_INSTR) {
      const d = draft[refIdx(b)];
      if (d.tt === TT_NOT && d.a === d.b) {
        b = d.a;
        tt = TT_NOT_B[tt];
      }
    }

    // (1) 定数畳み込み(両方定数 → 定数、片方定数 → 単項簡約表)
    const aConst = refKind(a) === REF_KIND_CONST;
    const bConst = refKind(b) === REF_KIND_CONST;
    if (aConst && bConst) {
      constFolded++;
      return ttEval(tt, refIdx(a), refIdx(b)) ? REF_ONE : REF_ZERO;
    }
    if (bConst) {
      const code = CONST_FOLD_B[tt * 2 + refIdx(b)];
      const r = applyRed(code, a);
      if (code === RED_CONST0 || code === RED_CONST1) constFolded++;
      else if (code === RED_ID) aliasFolded++;
      else notFolded++;
      return r;
    }
    if (aConst) {
      const code = CONST_FOLD_A[tt * 2 + refIdx(a)];
      const r = applyRed(code, b);
      if (code === RED_CONST0 || code === RED_CONST1) constFolded++;
      else if (code === RED_ID) aliasFolded++;
      else notFolded++;
      return r;
    }

    // (2) tt 自体の退化(0x0/0xf/0xa/0xc/0x5/0x3)
    const degen = TT_DEGEN[tt];
    if (degen !== RED_BINARY) {
      const target = TT_DEGEN_ON_B[tt] ? b : a;
      const r = applyRed(degen, target);
      if (degen === RED_CONST0 || degen === RED_CONST1) constFolded++;
      else if (degen === RED_ID) aliasFolded++;
      else notFolded++;
      return r;
    }

    // 同一参照が両入力に来た場合(タップ共有/CSEの結果)
    if (a === b) {
      const code = SAME_FOLD[tt];
      const r = applyRed(code, a);
      if (code === RED_CONST0 || code === RED_CONST1) constFolded++;
      else if (code === RED_ID) aliasFolded++;
      else notFolded++;
      return r;
    }

    // (4) 可換正規化: 可換ttは参照IDでソート、非可換は入替時に tt を変換
    if (b < a) {
      const t = a;
      a = b;
      b = t;
      tt = TT_SWAP[tt];
    }

    // (5) CSE
    const r = rawEmit(tt, a, b);
    if (lastWasNew) emitted++;
    else cseHits++;
    return r;
  };

  for (let oc = 0; oc < OC; oc++) {
    const row = spec.indicesFlat[oc];
    let cur: ConvRef[] = new Array<ConvRef>(nLeaves);
    for (let k = 0; k < nLeaves; k++) {
      const pp = row[k];
      let tid = tapIdOf.get(pp);
      if (tid === undefined) {
        tid = taps.length;
        taps.push(pp);
        tapIdOf.set(pp, tid);
      }
      cur[k] = makeRef(REF_KIND_TAP, tid);
    }
    let gOffset = 0;
    while (cur.length > 1) {
      const gOut = cur.length >> 1;
      const next: ConvRef[] = new Array<ConvRef>(gOut);
      for (let g = 0; g < gOut; g++) {
        const tt = spec.truthTables[oc * nInternal + gOffset + g] & 0xf;
        next[g] = emitGate(tt, cur[g << 1], cur[(g << 1) + 1]);
      }
      gOffset += gOut;
      cur = next;
    }
    ocOut[oc] = cur[0];
  }

  const instrsCreated = draft.length;

  // ---- (6) dead命令/未使用タップの除去 + トポロジカル番号付け ----
  // 縮約でゲートが消えると、その配下のタップも読まれなくなる(実測で
  // タップの6割超が未使用になる)。未使用タップを落とすと呼び出し側の
  // 「タップ平面の実体化」がそのまま減るので、命令削減と同じくらい効く。
  const live = new Uint8Array(instrsCreated);
  const tapUsed = new Uint8Array(taps.length);
  const stack: number[] = [];
  const pushRef = (r: ConvRef): void => {
    const kind = refKind(r);
    if (kind === REF_KIND_TAP) {
      tapUsed[refIdx(r)] = 1;
    } else if (kind === REF_KIND_INSTR && !live[refIdx(r)]) {
      live[refIdx(r)] = 1;
      stack.push(refIdx(r));
    }
  };
  for (let oc = 0; oc < OC; oc++) pushRef(ocOut[oc]);
  while (stack.length > 0) {
    const j = stack.pop() as number;
    pushRef(draft[j].a);
    pushRef(draft[j].b);
  }
  const tapRemap = new Int32Array(taps.length).fill(-1);
  const liveTaps: number[] = [];
  for (let t = 0; t < taps.length; t++) {
    if (!tapUsed[t]) continue;
    tapRemap[t] = liveTaps.length;
    liveTaps.push(taps[t]);
  }
  // draft は生成順=トポロジカル順(オペランドは必ず先行命令)なので、
  // liveな命令を昇順に詰め直せば順序は保たれる。
  const remap = new Int32Array(instrsCreated).fill(-1);
  const instrs: ConvInstr[] = [];
  for (let j = 0; j < instrsCreated; j++) {
    if (!live[j]) continue;
    remap[j] = instrs.length;
    const d = draft[j];
    instrs.push({ tt: d.tt, a: d.a, b: d.b, slot: -1 });
  }
  const fixRef = (r: ConvRef): ConvRef => {
    const kind = refKind(r);
    if (kind === REF_KIND_INSTR) return makeRef(REF_KIND_INSTR, remap[refIdx(r)]);
    if (kind === REF_KIND_TAP) return makeRef(REF_KIND_TAP, tapRemap[refIdx(r)]);
    return r;
  };
  for (const ins of instrs) {
    ins.a = fixRef(ins.a);
    ins.b = fixRef(ins.b);
    // 命令番号の詰め直しで参照IDの大小関係が変わりうるので、可換正規形
    // (a<b)を張り直す(意味は不変。CSEは既に終わっているので新たな重複は
    // 生じない: 同一ペアの命令は正規化前後で同じ向きになるため)。
    if (ins.b < ins.a) {
      const t = ins.a;
      ins.a = ins.b;
      ins.b = t;
      ins.tt = TT_SWAP[ins.tt];
    }
  }
  for (let oc = 0; oc < OC; oc++) ocOut[oc] = fixRef(ocOut[oc]);

  const n = instrs.length;

  // ---- 出力書き出しスケジュール(CSR) ----
  // 命令 j の直後に「出力が instr j である oc」を書き出す。こうすると
  // その値の生存区間が j で閉じ、作業平面スロット数が木の作業集合程度に
  // 収まる(全oc分を最後まで生かすとスロット数がOCまで膨らむ)。
  const bucketCount = new Int32Array(n + 2);
  for (let oc = 0; oc < OC; oc++) {
    const r = ocOut[oc];
    bucketCount[refKind(r) === REF_KIND_INSTR ? refIdx(r) + 1 : 0]++;
  }
  const writeStart = new Int32Array(n + 2);
  let acc = 0;
  for (let i = 0; i < n + 1; i++) {
    writeStart[i] = acc;
    acc += bucketCount[i];
  }
  writeStart[n + 1] = acc;
  const cursor = writeStart.slice(0, n + 1);
  const writeOc = new Int32Array(OC);
  for (let oc = 0; oc < OC; oc++) {
    const r = ocOut[oc];
    const bucket = refKind(r) === REF_KIND_INSTR ? refIdx(r) + 1 : 0;
    writeOc[cursor[bucket]++] = oc;
  }

  // ---- 生存区間解析(スロット割当) ----
  const lastUse = new Int32Array(n);
  for (let j = 0; j < n; j++) lastUse[j] = j; // 自身の実行直後(出力書き出し)まで
  for (let j = 0; j < n; j++) {
    const ins = instrs[j];
    if (refKind(ins.a) === REF_KIND_INSTR) lastUse[refIdx(ins.a)] = j;
    if (refKind(ins.b) === REF_KIND_INSTR) lastUse[refIdx(ins.b)] = j;
  }
  const free: number[] = [];
  let nSlots = 0;
  for (let j = 0; j < n; j++) {
    // dest を先に確保してからオペランドを解放する(dest がオペランドと
    // 同じスロットになることは無い = カーネル側でエイリアスを考えなくてよい)。
    const slot = free.length > 0 ? (free.pop() as number) : nSlots++;
    instrs[j].slot = slot;
    const ia = instrs[j].a;
    const ib = instrs[j].b;
    if (refKind(ia) === REF_KIND_INSTR && lastUse[refIdx(ia)] === j) {
      free.push(instrs[refIdx(ia)].slot);
    }
    if (refKind(ib) === REF_KIND_INSTR && ib !== ia && lastUse[refIdx(ib)] === j) {
      free.push(instrs[refIdx(ib)].slot);
    }
    // 消費者が無い命令(= oc 出力になるだけの木の根)も、その直後に出力へ
    // 書き出されるので次の命令から再利用してよい。これを忘れると木の根の
    // ぶんだけスロットが単調増加し(det n5 で472スロット)、作業平面が
    // キャッシュに乗らなくなる。
    if (lastUse[j] === j) free.push(slot);
  }

  let notInstrs = 0;
  for (const ins of instrs) if (ins.tt === TT_NOT && ins.a === ins.b) notInstrs++;
  let ocConst = 0;
  let ocTap = 0;
  const uniq = new Set<ConvRef>();
  for (let oc = 0; oc < OC; oc++) {
    uniq.add(ocOut[oc]);
    if (refKind(ocOut[oc]) === REF_KIND_CONST) ocConst++;
    else if (refKind(ocOut[oc]) === REF_KIND_TAP) ocTap++;
  }

  return {
    taps: liveTaps,
    instrs,
    nSlots,
    ocOut,
    writeStart,
    writeOc,
    stats: {
      gatesBefore: OC * nInternal,
      gatesAfter: n,
      constFolded,
      aliasFolded,
      notFolded,
      cseHits,
      emitted,
      instrsCreated,
      deadRemoved: instrsCreated - n,
      notInstrs,
      tapsBefore: taps.length,
      nTaps: liveTaps.length,
      nSlots,
      ocConst,
      ocTap,
      ocUnique: uniq.size,
    },
  };
}

// ============================================================
// ロード時セルフチェック(全数真理値表照合)
// ============================================================

/** 2^u パターン(u≤8)をビット並列で持つ語数。 */
function patternWords(u: number): number {
  return u <= 5 ? 1 : 1 << (u - 5);
}

/** 変数 i の「列パターン」(パターン p のビット = (p>>i)&1)を words へ書く。 */
function columnPattern(out: Int32Array, base: number, nw: number, i: number): void {
  if (i < 5) {
    // 0xaaaaaaaa, 0xcccccccc, 0xf0f0f0f0, 0xff00ff00, 0xffff0000
    let m = 0;
    for (let p = 0; p < 32; p++) if ((p >>> i) & 1) m |= 1 << p;
    for (let w = 0; w < nw; w++) out[base + w] = m;
  } else {
    for (let w = 0; w < nw; w++) out[base + w] = ((w >>> (i - 5)) & 1) ? -1 : 0;
  }
}

/** gate2 と同一の16式(ビット並列)。 */
function gate2Bits(tt: number, a: number, b: number): number {
  switch (tt & 0xf) {
    case 0x0: return 0;
    case 0x1: return ~(a | b);
    case 0x2: return a & ~b;
    case 0x3: return ~b;
    case 0x4: return ~a & b;
    case 0x5: return ~a;
    case 0x6: return a ^ b;
    case 0x7: return ~(a & b);
    case 0x8: return a & b;
    case 0x9: return ~(a ^ b);
    case 0xa: return a;
    case 0xb: return a | ~b;
    case 0xc: return b;
    case 0xd: return ~a | b;
    case 0xe: return a | b;
    default: return -1;
  }
}

/** 全数照合を行う上限(oc ごとのユニークタップ数 u)。Shannon分解のような
    深い木(tree_depth=9等)でも、木の葉が参照するユニーク値の数(=真に自由な
    変数の数)はずっと小さいのが普通(例: n=6 LUT分解なら u=6)。判定はこの
    u に基づく(木の深さ/葉数そのものには一切依存しない)。 */
const EXHAUSTIVE_MAX_U = 16;
/** u が上限を超えた oc のみ使うランダムサンプリング照合のワード数
    (1ワード=32パターンなので 2^17=131072 パターン相当)。 */
const SAMPLE_WORDS = 1 << 12;

/** 命令列が元のバランス木と(全数、または大域変数が多すぎる場合はサンプリングで)
    一致することを検証する(ロード時セルフチェック)。各 oc について「その木の
    ユニークタップ数 u」を数え、u≤EXHAUSTIVE_MAX_U なら 2^u 全パターンを
    ビット並列(32パターン/ワード)で評価して元の木と縮約後の命令列(oc出力の
    依存錐だけ)を突き合わせる。u が大きすぎる oc だけは決定的PRNGによる
    ランダムサンプリング照合に切り替える(黙って検証を諦めない)。不一致なら
    例外を投げる(呼び出し側は旧経路へフォールバックすること)。 */
export function verifyConvProgram(spec: ConvTreeSpec, prog: ConvProgram): void {
  const OC = spec.outChannels;
  const nLeaves = 1 << spec.treeDepth;
  const nInternal = nLeaves - 1;
  const tapIdOf = new Map<number, number>();
  for (let i = 0; i < prog.taps.length; i++) tapIdOf.set(prog.taps[i], i);

  const nInstr = prog.instrs.length;
  const tapSeen = new Int32Array(prog.taps.length).fill(-1);
  const inCone = new Uint8Array(nInstr);
  const coneList: number[] = [];
  const dfs: number[] = [];
  const varOfPp = new Map<number, number>();

  // 決定的PRNG(mulberry32)。u超過時のサンプリングのみに使う(CI再現性のため
  // 固定シード。落ちるべき不一致がシード次第で見逃されないよう、サンプル数は
  // 実運用の入力空間に対して十分大きく取る)。
  let rngA = 0x9e3779b9 ^ (nLeaves * 0x1000193);
  const nextRandWord = (): number => {
    rngA = (rngA + 0x6d2b79f5) | 0;
    let t = Math.imul(rngA ^ (rngA >>> 15), 1 | rngA);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return (t ^ (t >>> 14)) | 0;
  };

  for (let oc = 0; oc < OC; oc++) {
    const row = spec.indicesFlat[oc];
    // このocの葉に現れるユニークタップへパターン変数を割り当てる。命令列側で
    // 落ちたタップ(縮約により読まれなくなったもの)にも変数を与えるので、
    // 「元の木が本当にそのタップに依存していない」ことまで検証できる。
    varOfPp.clear();
    for (let k = 0; k < nLeaves; k++) {
      if (!varOfPp.has(row[k])) varOfPp.set(row[k], varOfPp.size);
    }
    const u = varOfPp.size;
    const exhaustive = u <= EXHAUSTIVE_MAX_U;
    const nw = exhaustive ? patternWords(u) : SAMPLE_WORDS;

    // このoc専用のバッファ(u/サンプリングどちらもnwは小さいので毎oc新規で十分。
    // このセルフチェックはロード時に1回だけ走る)。
    const leafCols = new Int32Array(u * nw);
    for (const [, v] of varOfPp) {
      if (exhaustive) {
        columnPattern(leafCols, v * nw, nw, v);
      } else {
        for (let w = 0; w < nw; w++) leafCols[v * nw + w] = nextRandWord();
      }
    }
    const tapVals = new Int32Array(prog.taps.length * nw);
    for (const [pp, v] of varOfPp) {
      const tid = tapIdOf.get(pp);
      if (tid !== undefined) {
        tapSeen[tid] = oc;
        for (let w = 0; w < nw; w++) tapVals[tid * nw + w] = leafCols[v * nw + w];
      }
    }

    const refWords = (r: ConvRef, out: Int32Array, w: number): number => {
      const kind = refKind(r);
      if (kind === REF_KIND_CONST) return refIdx(r) === 0 ? 0 : -1;
      if (kind === REF_KIND_TAP) return tapVals[refIdx(r) * nw + w];
      return out[refIdx(r) * nw + w];
    };

    // --- 元のバランス木を評価 ---
    let treeVals = new Int32Array(nLeaves * nw);
    for (let k = 0; k < nLeaves; k++) {
      const v = varOfPp.get(row[k]) as number;
      for (let w = 0; w < nw; w++) treeVals[k * nw + w] = leafCols[v * nw + w];
    }
    let nextVals = new Int32Array(nLeaves * nw);
    let len = nLeaves;
    let src = treeVals;
    let dst = nextVals;
    let gOffset = 0;
    while (len > 1) {
      const gOut = len >> 1;
      for (let g = 0; g < gOut; g++) {
        const tt = spec.truthTables[oc * nInternal + gOffset + g] & 0xf;
        for (let w = 0; w < nw; w++) {
          dst[g * nw + w] = gate2Bits(
            tt, src[(g << 1) * nw + w], src[((g << 1) + 1) * nw + w],
          );
        }
      }
      gOffset += gOut;
      len = gOut;
      const t = src;
      src = dst;
      dst = t;
    }
    void nextVals; // (treeVals/nextValsはsrc/dstのスワップ元として使うのみ)

    // --- 縮約後の命令列(出力の依存錐のみ)を評価 ---
    const instrVals = new Int32Array(nInstr * nw);
    coneList.length = 0;
    dfs.length = 0;
    const out0 = prog.ocOut[oc];
    if (refKind(out0) === REF_KIND_INSTR) {
      inCone[refIdx(out0)] = 1;
      dfs.push(refIdx(out0));
      coneList.push(refIdx(out0));
    }
    while (dfs.length > 0) {
      const j = dfs.pop() as number;
      for (const r of [prog.instrs[j].a, prog.instrs[j].b]) {
        if (refKind(r) === REF_KIND_INSTR && !inCone[refIdx(r)]) {
          inCone[refIdx(r)] = 1;
          dfs.push(refIdx(r));
          coneList.push(refIdx(r));
        }
        if (refKind(r) === REF_KIND_TAP && tapSeen[refIdx(r)] !== oc) {
          for (const c of coneList) inCone[c] = 0;
          throw new Error(
            `verifyConvProgram: oc=${oc} の依存錐に木の外のタップ ${prog.taps[refIdx(r)]} が現れた`,
          );
        }
      }
    }
    coneList.sort((p, q) => p - q); // トポロジカル順(生成順=昇順)
    for (const j of coneList) {
      const ins = prog.instrs[j];
      for (let w = 0; w < nw; w++) {
        instrVals[j * nw + w] = gate2Bits(
          ins.tt, refWords(ins.a, instrVals, w), refWords(ins.b, instrVals, w),
        );
      }
    }
    for (const j of coneList) inCone[j] = 0;

    // --- 突き合わせ(全数なら有効パターン数 2^u ぶんのビットのみ。
    //     サンプリングなら全ワード=全サンプル)。 ---
    for (let w = 0; w < nw; w++) {
      const validBits = !exhaustive || u >= 5 || nw > 1 ? -1 : ((1 << (1 << u)) - 1) | 0;
      const want = src[w] & validBits;
      const got = refWords(out0, instrVals, w) & validBits;
      if (want !== got) {
        throw new Error(
          `verifyConvProgram: oc=${oc} で真理値表不一致 (u=${u}, `
          + `${exhaustive ? '全数照合' : 'サンプリング照合'}, word ${w}: `
          + `期待 0x${(want >>> 0).toString(16)}, 実際 0x${(got >>> 0).toString(16)})`,
        );
      }
    }
  }
}
