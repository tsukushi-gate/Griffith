// deploykitの配布バンドルが持つ conv2d 命令列(`programs[node.id]`、
// griffith_logic/deploy/programs.py::program_to_dict の出力)を、TS側の
// ConvProgram(convProgram.ts)へ変換・構造検証する。
//
// 目的: バンドルには「Python側で縮約済みの命令列」が既に焼き込まれている
// ので、ブラウザ側で buildConvProgram(縮約パスの再実行)をする必要が無い。
// ここでは
//   1. dict → ConvProgram の変換(program_from_dict の逆写像)
//   2. 構造検証: bundle.py::_validate_program と同じ参照範囲/長さの検査に加え、
//      write_start/write_oc の CSR を oc_out から再計算して完全一致を要求し、
//      さらに(Pythonには無い)スロット生存区間の健全性検査を行う
//      (gate2RangeInto(bitplane.ts)が「dest は常にオペランドと非エイリアス」
//      を前提にしているため、これが崩れていないことをロード時に保証する)
//   3. verifyConvProgram(全数真理値表照合)との合わせ技で、バンドルの命令列を
//      信頼して良いかを判定する resolveConvProgram()
// を提供する。**bitplane / wasm / webgpu のどれにも依存しない純粋な TS**
// (import は './convProgram' のみ。convProgram.ts と同じ理由で node の型
// stripping で直接読める状態を保つ)。
//
// 正しさの契約: convProgramToDict∘convProgramFromDict は恒等(往復同一性)。
// 破損した dict は必ず construct 時点で例外になる(黙って壊れた命令列を
// 実行しない)。bundleProgram.test.ts が両方を見る。
import {
  REF_KIND_CONST, REF_KIND_INSTR, REF_KIND_TAP,
  buildConvProgram, refIdx, refKind, verifyConvProgram,
} from './convProgram';
import type { ConvInstr, ConvProgram, ConvProgramStats, ConvTreeSpec } from './convProgram';

/** griffith_logic/deploy/programs.py::program_to_dict のJSON形状。
    instrs は4要素/命令(tt,a,b,slot)のフラット配列、ref のエンコードは
    convProgram.ts と同一((idx<<2)|kind)。 */
export interface BundleProgramDict {
  taps: number[];
  n_instrs: number;
  instrs: number[];
  n_slots: number;
  oc_out: number[];
  write_start: number[];
  write_oc: number[];
  stats?: Record<string, number>;
}

/** ConvProgramStats の camelCase ↔ dataclass(export/optimize.py)の
    snake_case フィールド名対応(全16フィールド)。 */
function statsFromSnake(raw: Record<string, number> | undefined): ConvProgramStats {
  const r = raw ?? {};
  const get = (name: string): number => {
    const v = r[name];
    return typeof v === 'number' && Number.isFinite(v) ? Math.trunc(v) : 0;
  };
  return {
    gatesBefore: get('gates_before'),
    gatesAfter: get('gates_after'),
    constFolded: get('const_folded'),
    aliasFolded: get('alias_folded'),
    notFolded: get('not_folded'),
    cseHits: get('cse_hits'),
    emitted: get('emitted'),
    instrsCreated: get('instrs_created'),
    deadRemoved: get('dead_removed'),
    notInstrs: get('not_instrs'),
    tapsBefore: get('taps_before'),
    nTaps: get('n_taps'),
    nSlots: get('n_slots'),
    ocConst: get('oc_const'),
    ocTap: get('oc_tap'),
    ocUnique: get('oc_unique'),
  };
}

function statsToSnake(s: ConvProgramStats): Record<string, number> {
  return {
    gates_before: s.gatesBefore,
    gates_after: s.gatesAfter,
    const_folded: s.constFolded,
    alias_folded: s.aliasFolded,
    not_folded: s.notFolded,
    cse_hits: s.cseHits,
    emitted: s.emitted,
    instrs_created: s.instrsCreated,
    dead_removed: s.deadRemoved,
    not_instrs: s.notInstrs,
    taps_before: s.tapsBefore,
    n_taps: s.nTaps,
    n_slots: s.nSlots,
    oc_const: s.ocConst,
    oc_tap: s.ocTap,
    oc_unique: s.ocUnique,
  };
}

/** ref((idx<<2)|kind)の妥当性。命令参照は自分より前(トポ順)に限る
    (bundle.py::_validate_ref と同一ロジック)。 */
function validateRef(
  fail: (msg: string) => never,
  where: string,
  ref: number,
  nTaps: number,
  limit: number,
): void {
  const kind = refKind(ref);
  const idx = refIdx(ref);
  if (kind === REF_KIND_CONST) {
    if (idx !== 0 && idx !== 1) fail(`${where}: 定数参照 idx=${idx} が0/1でない`);
  } else if (kind === REF_KIND_TAP) {
    if (!(idx >= 0 && idx < nTaps)) fail(`${where}: tap参照 idx=${idx} が tap数 ${nTaps} の外`);
  } else if (kind === REF_KIND_INSTR) {
    if (!(idx >= 0 && idx < limit)) fail(`${where}: 命令参照 idx=${idx} が前方参照でない(limit=${limit})`);
  } else {
    fail(`${where}: 未知のref kind=${kind}`);
  }
}

/** program_to_dict の逆変換 + 構造検証(bundle.py::_validate_program のミラー
    + TS独自のCSR再計算照合・スロット生存区間検査)。不正な dict は必ず
    ここで例外になる(黙って壊れた命令列を実行しない)。 */
export function convProgramFromDict(d: BundleProgramDict, outChannels: number): ConvProgram {
  const fail = (msg: string): never => {
    throw new Error(`convProgramFromDict: ${msg}`);
  };

  const nInstrs = d.n_instrs;
  if (!Number.isInteger(nInstrs) || nInstrs < 0) fail(`n_instrsが不正な整数: ${d.n_instrs}`);
  if (d.instrs.length !== 4 * nInstrs) {
    fail(`instrs長 ${d.instrs.length} が 4*n_instrs=${4 * nInstrs} と違う`);
  }
  const nSlots = d.n_slots;
  const nTaps = d.taps.length;
  const slotLimit = Math.max(nSlots, 1);

  // ---- 命令ごとの範囲検査(tt/slot/オペランド参照) ----
  const instrs: ConvInstr[] = new Array(nInstrs);
  for (let j = 0; j < nInstrs; j++) {
    const tt = d.instrs[4 * j];
    const a = d.instrs[4 * j + 1];
    const b = d.instrs[4 * j + 2];
    const slot = d.instrs[4 * j + 3];
    if (!(tt >= 0 && tt <= 0xf)) fail(`命令${j}: tt=${tt} が0..15の外`);
    if (!(slot >= 0 && slot < slotLimit)) fail(`命令${j}: slot=${slot} が n_slots=${nSlots} の外`);
    validateRef(fail, `命令${j}.a`, a, nTaps, j);
    validateRef(fail, `命令${j}.b`, b, nTaps, j);
    instrs[j] = { tt, a, b, slot };
  }

  // ---- oc_out / write_oc / write_start の長さと参照範囲 ----
  if (d.oc_out.length !== outChannels) {
    fail(`oc_out長 ${d.oc_out.length} が out_channels=${outChannels} と違う`);
  }
  for (let oc = 0; oc < outChannels; oc++) {
    validateRef(fail, `oc_out[${oc}]`, d.oc_out[oc], nTaps, nInstrs);
  }
  if (d.write_oc.length !== outChannels) {
    fail(`write_oc長 ${d.write_oc.length} が out_channels=${outChannels} と違う`);
  }
  if (d.write_start.length !== nInstrs + 2) {
    fail(`write_start長 ${d.write_start.length} が n_instrs+2=${nInstrs + 2} と違う`);
  }
  const wocSorted = [...d.write_oc].sort((x, y) => x - y);
  for (let i = 0; i < outChannels; i++) {
    if (wocSorted[i] !== i) fail('write_ocが全出力チャネルの順列になっていない');
  }

  // ---- タップの重複禁止(dedup済みでなければならない) ----
  if (new Set(d.taps).size !== nTaps) fail('tapに重複がある(dedup済みでなければならない)');

  // ---- write_start/write_oc のCSRを oc_out から再計算し、完全一致を要求 ----
  // (convProgram.ts::buildConvProgram の該当ロジック(~494-512)と同一アルゴリズム。
  //  Pythonの_validate_programには無い検査: バンドルの書き出しスケジュールが
  //  「oc_outから決定的に導出されるものと一致する」ことまで保証する)
  const bucketCount = new Int32Array(nInstrs + 2);
  for (let oc = 0; oc < outChannels; oc++) {
    const r = d.oc_out[oc];
    bucketCount[refKind(r) === REF_KIND_INSTR ? refIdx(r) + 1 : 0]++;
  }
  const wantWriteStart = new Int32Array(nInstrs + 2);
  let acc = 0;
  for (let i = 0; i < nInstrs + 1; i++) {
    wantWriteStart[i] = acc;
    acc += bucketCount[i];
  }
  wantWriteStart[nInstrs + 1] = acc;
  for (let i = 0; i < wantWriteStart.length; i++) {
    if (d.write_start[i] !== wantWriteStart[i]) {
      fail(`write_start[${i}]=${d.write_start[i]} が再計算値 ${wantWriteStart[i]} と不一致`);
    }
  }
  const cursor = wantWriteStart.slice(0, nInstrs + 1);
  const wantWriteOc = new Int32Array(outChannels);
  for (let oc = 0; oc < outChannels; oc++) {
    const r = d.oc_out[oc];
    const bucket = refKind(r) === REF_KIND_INSTR ? refIdx(r) + 1 : 0;
    wantWriteOc[cursor[bucket]++] = oc;
  }
  for (let i = 0; i < outChannels; i++) {
    if (d.write_oc[i] !== wantWriteOc[i]) {
      fail(`write_oc[${i}]=${d.write_oc[i]} が再計算値 ${wantWriteOc[i]} と不一致`);
    }
  }

  // ---- スロット生存区間の健全性(Pythonの検証器には無い、TS側独自の検査) ----
  // gate2RangeInto(bitplane.ts)は「dest は常に現在生存中のオペランドと
  // 別スロット」を前提に区間演算を行う(エイリアス時のバグを検出しない)。
  // convProgram.ts の生存区間解析(~515-521)と同じ lastUse 計算を、oc_out
  // への参照も含めて行い(~444)、命令を先頭から辿って「dest slot が
  // まだ生きている先行命令に所有されていないか」を検査する。
  const lastUse = new Int32Array(nInstrs);
  for (let j = 0; j < nInstrs; j++) lastUse[j] = j; // 自身の実行直後(出力書き出し)まで
  for (let j = 0; j < nInstrs; j++) {
    const ins = instrs[j];
    if (refKind(ins.a) === REF_KIND_INSTR) {
      lastUse[refIdx(ins.a)] = Math.max(lastUse[refIdx(ins.a)], j);
    }
    if (refKind(ins.b) === REF_KIND_INSTR) {
      lastUse[refIdx(ins.b)] = Math.max(lastUse[refIdx(ins.b)], j);
    }
  }
  for (let oc = 0; oc < outChannels; oc++) {
    const r = d.oc_out[oc];
    if (refKind(r) === REF_KIND_INSTR) {
      lastUse[refIdx(r)] = Math.max(lastUse[refIdx(r)], refIdx(r));
    }
  }
  const owners = new Int32Array(slotLimit).fill(-1);
  for (let j = 0; j < nInstrs; j++) {
    const slot = instrs[j].slot;
    const owner = owners[slot];
    if (owner >= 0 && lastUse[owner] >= j) {
      fail(
        `命令${j}: dest slot=${slot} が命令${owner}(生存区間 lastUse=${lastUse[owner]})と`
        + 'エイリアスしている',
      );
    }
    owners[slot] = j;
  }

  return {
    taps: [...d.taps],
    instrs,
    nSlots,
    ocOut: [...d.oc_out],
    writeStart: Int32Array.from(d.write_start),
    writeOc: Int32Array.from(d.write_oc),
    stats: statsFromSnake(d.stats),
  };
}

/** convProgramFromDict の逆変換(program_to_dict のTS版・往復同一性テストで使う)。 */
export function convProgramToDict(p: ConvProgram): BundleProgramDict {
  const instrs: number[] = new Array(p.instrs.length * 4);
  for (let j = 0; j < p.instrs.length; j++) {
    const ins = p.instrs[j];
    instrs[4 * j] = ins.tt;
    instrs[4 * j + 1] = ins.a;
    instrs[4 * j + 2] = ins.b;
    instrs[4 * j + 3] = ins.slot;
  }
  return {
    taps: [...p.taps],
    n_instrs: p.instrs.length,
    instrs,
    n_slots: p.nSlots,
    oc_out: [...p.ocOut],
    write_start: Array.from(p.writeStart),
    write_oc: Array.from(p.writeOc),
    stats: statsToSnake(p.stats),
  };
}

/** conv2d命令列の出所。'bundle' = 配布バンドルの programs 節をそのまま採用、
    'built' = ブラウザ側で buildConvProgram を実行(バンドル未指定、または
    バンドルの命令列を採用できず再構築した場合)。 */
export type ProgramProvenance = 'bundle' | 'built';

/** spec→ConvProgram の唯一の解決口。バンドルの命令列(d)があればそれを
    デシリアライズ+構造検証+verifyConvProgram(全数真理値表照合)まで通して
    採用し、どこかで例外が出たら(strictでない限り)黙殺せず note に理由を
    残した上で buildConvProgram(spec) による再構築へフォールバックする。
    d が無ければ最初から再構築する(note は null = フォールバックではない)。 */
export function resolveConvProgram(
  spec: ConvTreeSpec,
  d: BundleProgramDict | null | undefined,
  opts: { strict?: boolean } = {},
): { program: ConvProgram; provenance: ProgramProvenance; note: string | null } {
  const buildFallback = (): ConvProgram => {
    const program = buildConvProgram(spec);
    verifyConvProgram(spec, program);
    return program;
  };

  if (d) {
    try {
      const program = convProgramFromDict(d, spec.outChannels);
      verifyConvProgram(spec, program);
      return { program, provenance: 'bundle', note: null };
    } catch (e) {
      if (opts.strict) throw e;
      const note = `programsバンドルの命令列を採用できず再構築: ${String(e)}`;
      return { program: buildFallback(), provenance: 'built', note };
    }
  }
  return { program: buildFallback(), provenance: 'built', note: null };
}
