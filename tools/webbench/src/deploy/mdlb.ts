// 配布バンドルのバイナリ形式 `.mdlb` — TS側デコーダ(読み込み専用)。
//
// `griffith_logic/deploy/mdlb.py` のバイト単位ミラー。フォーマットの
// 正準実装はPython側(`griffith deploy bundle -o *.mdlb` でのみ生成できる)で、
// このファイルはそれを読むだけ(webランタイムは読み込み専用ランタイム、
// というTrack Kの既存の非対称——webbackend.pyのdocstring参照——を
// そのまま踏襲する。ラウンドトリップの検証は、Pythonで生成した実バイナリを
// このデコーダで読んで元のJSONと突き合わせる形で行う
// (`tools/webbench/src/deploy/fixtures/bundle_golden.mdlb`、mdlb.test.ts)。
//
// セクション対応・エンコード規約は mdlb.py のモジュールdocstringが正。
// ここでは概要だけ: magic 'MDLB' + u32 version(LE)+ u32 n_sections + u32
// reserved(16B header)、続けて n_sections × 32B のセクションディレクトリ
// (tag 4B + reserved 4B + rel_offset u64 LE + length u64 LE + crc32 u32 LE
// + reserved 4B)、その後ろがペイロード領域(8バイトアライン)。
//
// 数値の扱い: 生バイト列のvarintデコードはbigintで行う(truth_tables は
// N<=6 LUTで最大64bit値になりうるため、JSのNumberの安全範囲53bitでは
// 原理的に不足しうる — mdlb.py参照)。他の値(長さ・個数・shape次元・
// ref・slot・conv2dのtt等)はすべて実運用で53bit未満に収まることが
// 保証されているので number へ変換する。変換時に安全範囲を超えていたら
// 例外にする(黙って精度を落とさない)。
import { makeRef, REF_KIND_INSTR, refIdx, refKind } from '../engines/convProgram';
import type { BundleProgramDict } from '../engines/bundleProgram';
import type { BundleDoc, PreprocDesc } from './bundleDoc';

// eslint-disable-next-line @typescript-eslint/no-explicit-any
export type JsonValue = any;

const MDLB_MAGIC = 'MDLB';
const MDLB_VERSION = 1;
const HEADER_SIZE = 16;
const DIR_ENTRY_SIZE = 32;
const SECTION_TAGS = ['IRMT', 'TTBL', 'IDXS', 'PROG', 'PPDS', 'LABL', 'RDOT', 'EXTR'] as const;

const T_NULL = 0;
const T_FALSE = 1;
const T_TRUE = 2;
const T_INT = 3;
const T_FLOAT = 4;
const T_STR = 5;
const T_LIST = 6;
const T_DICT = 7;
const T_B64BLOB = 8;

// ---------------------------------------------------------------------------
// CRC32(ISO-3309、zlib.crc32と同一。外部依存なし)
// ---------------------------------------------------------------------------

let crcTableCache: Uint32Array | null = null;

function crcTable(): Uint32Array {
  if (crcTableCache) return crcTableCache;
  const table = new Uint32Array(256);
  for (let n = 0; n < 256; n++) {
    let c = n;
    for (let k = 0; k < 8; k++) {
      c = (c & 1) !== 0 ? (0xedb88320 ^ (c >>> 1)) : (c >>> 1);
    }
    table[n] = c >>> 0;
  }
  crcTableCache = table;
  return table;
}

function crc32(bytes: Uint8Array): number {
  const table = crcTable();
  let crc = 0xffffffff;
  for (let i = 0; i < bytes.length; i++) {
    crc = table[(crc ^ bytes[i]) & 0xff] ^ (crc >>> 8);
  }
  return (crc ^ 0xffffffff) >>> 0;
}

// ---------------------------------------------------------------------------
// base64(Buffer/btoaに依存しない、Python base64.b64encodeと同一の手組み実装)
// ---------------------------------------------------------------------------

const B64_CHARS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/';

function bytesToBase64(bytes: Uint8Array): string {
  let out = '';
  let i = 0;
  for (; i + 3 <= bytes.length; i += 3) {
    const n = (bytes[i] << 16) | (bytes[i + 1] << 8) | bytes[i + 2];
    out += B64_CHARS[(n >> 18) & 0x3f] + B64_CHARS[(n >> 12) & 0x3f]
      + B64_CHARS[(n >> 6) & 0x3f] + B64_CHARS[n & 0x3f];
  }
  const rem = bytes.length - i;
  if (rem === 1) {
    const n = bytes[i] << 16;
    out += B64_CHARS[(n >> 18) & 0x3f] + B64_CHARS[(n >> 12) & 0x3f] + '==';
  } else if (rem === 2) {
    const n = (bytes[i] << 16) | (bytes[i + 1] << 8);
    out += B64_CHARS[(n >> 18) & 0x3f] + B64_CHARS[(n >> 12) & 0x3f]
      + B64_CHARS[(n >> 6) & 0x3f] + '=';
  }
  return out;
}

// ---------------------------------------------------------------------------
// バイト読み取り + varint(unsigned LEB128)/ zigzag
// ---------------------------------------------------------------------------

function numberFrom(v: bigint): number {
  if (v > BigInt(Number.MAX_SAFE_INTEGER) || v < BigInt(Number.MIN_SAFE_INTEGER)) {
    throw new Error(`mdlb: 値がNumberの安全範囲(53bit)を超える: ${v.toString()}`);
  }
  return Number(v);
}

class ByteReader {
  pos = 0;

  constructor(private readonly data: Uint8Array) {}

  readByte(): number {
    if (this.pos >= this.data.length) {
      throw new Error('mdlb: バッファ終端(データが切り詰められている)');
    }
    return this.data[this.pos++];
  }

  readBytes(n: number): Uint8Array {
    if (this.pos + n > this.data.length) {
      throw new Error('mdlb: バッファ終端(データが切り詰められている)');
    }
    const out = this.data.subarray(this.pos, this.pos + n);
    this.pos += n;
    return out;
  }

  readUVarintBig(): bigint {
    let result = 0n;
    let shift = 0n;
    for (;;) {
      const b = this.readByte();
      result |= BigInt(b & 0x7f) << shift;
      if ((b & 0x80) === 0) return result;
      shift += 7n;
    }
  }

  readUVarint(): number {
    return numberFrom(this.readUVarintBig());
  }

  readSVarint(): number {
    const z = this.readUVarintBig();
    const v = (z & 1n) === 0n ? (z >> 1n) : -((z + 1n) >> 1n);
    return numberFrom(v);
  }

  readStr(): string {
    const n = this.readUVarint();
    const bytes = this.readBytes(n);
    return new TextDecoder('utf-8').decode(bytes);
  }
}

function readUArray(r: ByteReader, n: number): number[] {
  const out: number[] = [];
  for (let i = 0; i < n; i++) out.push(r.readUVarint());
  return out;
}

function readSArray(r: ByteReader, n: number): number[] {
  const out: number[] = [];
  for (let i = 0; i < n; i++) out.push(r.readSVarint());
  return out;
}

function readDeltaSArray(r: ByteReader, n: number): number[] {
  const out: number[] = [];
  let prev = 0;
  for (let i = 0; i < n; i++) {
    prev += r.readSVarint();
    out.push(prev);
  }
  return out;
}

/** ニブル詰め(mode=0、0..15)/ varint配列(mode=1)の切り替え読み。
    bigint版はTT(最大64bit値になりうる)専用、number版はPROGRAM列
    (tt/slot、常に小さい)専用。 */
function readNibbleOrUArrayBig(r: ByteReader, n: number): bigint[] {
  const mode = r.readByte();
  if (mode === 0) {
    const nbytes = Math.floor((n + 1) / 2);
    const packed = r.readBytes(nbytes);
    const out: bigint[] = [];
    for (let i = 0; i < n; i++) {
      const byte = packed[Math.floor(i / 2)];
      const nib = (i % 2 === 0) ? (byte & 0xf) : ((byte >> 4) & 0xf);
      out.push(BigInt(nib));
    }
    return out;
  }
  const out: bigint[] = [];
  for (let i = 0; i < n; i++) out.push(r.readUVarintBig());
  return out;
}

function readNibbleOrUArray(r: ByteReader, n: number): number[] {
  return readNibbleOrUArrayBig(r, n).map((v) => numberFrom(v));
}

/** ref配列(命令オペランドa/b)の位置相対delta復元。mdlb.py::_read_ref_array
    と同じ規約: kind=INSTR(2)ならcodeの上位ビットは「現在位置jからの距離」。
    ref自体のbit layoutは (idx<<2)|kind で、既存TS実装(convProgram.ts::
    makeRef/refKind/refIdx)と単一のソースを共有する(新規に規約を
    作らない)。 */
function readRefArray(r: ByteReader, n: number): number[] {
  const out: number[] = [];
  for (let j = 0; j < n; j++) {
    const code = r.readUVarint();
    const kind = refKind(code);
    const payload = refIdx(code);
    const idx = kind === REF_KIND_INSTR ? (j - payload) : payload;
    out.push(makeRef(kind, idx));
  }
  return out;
}

// ---------------------------------------------------------------------------
// 汎用JSON値デコード(IR_META/EXTRA/LABELS/READOUT/statsが使う)
// ---------------------------------------------------------------------------

function readValue(r: ByteReader): JsonValue {
  const tag = r.readByte();
  switch (tag) {
    case T_NULL: return null;
    case T_FALSE: return false;
    case T_TRUE: return true;
    case T_INT: return r.readSVarint();
    case T_FLOAT: {
      const bytes = r.readBytes(8);
      const view = new DataView(bytes.buffer, bytes.byteOffset, 8);
      return view.getFloat64(0, true);
    }
    case T_STR: return r.readStr();
    case T_B64BLOB: {
      const n = r.readUVarint();
      const raw = r.readBytes(n);
      return bytesToBase64(raw);
    }
    case T_LIST: {
      const n = r.readUVarint();
      const out: JsonValue[] = [];
      for (let i = 0; i < n; i++) out.push(readValue(r));
      return out;
    }
    case T_DICT: {
      const n = r.readUVarint();
      const out: Record<string, JsonValue> = {};
      for (let i = 0; i < n; i++) {
        const k = r.readStr();
        out[k] = readValue(r);
      }
      return out;
    }
    default:
      throw new Error(`mdlb: 未知の型タグ ${tag}(データが壊れている)`);
  }
}

// ---------------------------------------------------------------------------
// ネストしたint配列(indices_flat等)のshape復元
// ---------------------------------------------------------------------------

function unflattenNestedInts(flat: number[], shape: number[]): JsonValue {
  let cursor = 0;
  function build(depth: number): JsonValue {
    if (depth === shape.length) {
      return flat[cursor++];
    }
    const out: JsonValue[] = [];
    for (let i = 0; i < shape[depth]; i++) out.push(build(depth + 1));
    return out;
  }
  return build(0);
}

// ---------------------------------------------------------------------------
// TT / INDICESセクションのエントリデコード
// ---------------------------------------------------------------------------

function decodeTtEntry(r: ByteReader): [string, string[]] {
  const nodeId = r.readStr();
  const n = r.readUVarint();
  const values = readNibbleOrUArrayBig(r, n);
  const hexlist = values.map((v) => `0x${v.toString(16)}`);
  return [nodeId, hexlist];
}

interface IdxEntry { nodeId: string; key: string; flat: number[]; shape: number[]; }

function decodeIdxEntry(r: ByteReader): IdxEntry {
  const nodeId = r.readStr();
  const key = r.readStr();
  const nDims = r.readUVarint();
  const shape: number[] = [];
  for (let i = 0; i < nDims; i++) shape.push(r.readUVarint());
  const nFlat = r.readUVarint();
  const flat = readUArray(r, nFlat);
  return {
    nodeId, key, flat, shape,
  };
}

function mergeIrFromMdlb(
  metaIr: Record<string, JsonValue>,
  ttEntries: [string, string[]][],
  idxEntries: IdxEntry[],
): Record<string, JsonValue> {
  const irDict: Record<string, JsonValue> = { ...metaIr };
  const nodesRaw = (irDict.nodes as JsonValue[] | undefined) ?? [];
  const nodes = nodesRaw.map((n) => ({ ...(n as Record<string, JsonValue>) }));
  const byId = new Map<string, number>();
  nodes.forEach((n, i) => {
    if (typeof n.id === 'string') byId.set(n.id, i);
  });

  const lookupOrThrow = (nodeId: string, section: string): number => {
    const i = byId.get(nodeId);
    if (i === undefined) {
      throw new Error(`mdlb: ${section}セクションのnode_id "${nodeId}" がIRに無い(データが壊れている)`);
    }
    return i;
  };

  for (const [nodeId, hexlist] of ttEntries) {
    const i = lookupOrThrow(nodeId, 'TT');
    const params = { ...((nodes[i].params as Record<string, JsonValue>) ?? {}) };
    params.truth_tables = hexlist;
    nodes[i] = { ...nodes[i], params };
  }
  for (const entry of idxEntries) {
    const i = lookupOrThrow(entry.nodeId, 'INDICES');
    const params = { ...((nodes[i].params as Record<string, JsonValue>) ?? {}) };
    params[entry.key] = unflattenNestedInts(entry.flat, entry.shape);
    nodes[i] = { ...nodes[i], params };
  }
  irDict.nodes = nodes;
  return irDict;
}

// ---------------------------------------------------------------------------
// PROGRAMセクション
// ---------------------------------------------------------------------------

function decodeProgram(r: ByteReader): [string, BundleProgramDict] {
  const nodeId = r.readStr();

  const nTaps = r.readUVarint();
  const taps = readUArray(r, nTaps);

  const nInstrs = r.readUVarint();
  const tts = readNibbleOrUArray(r, nInstrs);
  const aas = readRefArray(r, nInstrs);
  const bbs = readRefArray(r, nInstrs);
  const slots = readNibbleOrUArray(r, nInstrs);
  const instrs: number[] = [];
  for (let i = 0; i < nInstrs; i++) {
    instrs.push(tts[i], aas[i], bbs[i], slots[i]);
  }

  const nSlots = r.readUVarint();

  const nOc = r.readUVarint();
  const ocOut = readUArray(r, nOc);

  const nWs = r.readUVarint();
  const writeStart = readDeltaSArray(r, nWs);

  const nWoc = r.readUVarint();
  const writeOc = readUArray(r, nWoc);

  const stats = readValue(r) as Record<string, number>;

  const prog: BundleProgramDict = {
    taps,
    n_instrs: nInstrs,
    instrs,
    n_slots: nSlots,
    oc_out: ocOut,
    write_start: writeStart,
    write_oc: writeOc,
    stats,
  };
  return [nodeId, prog];
}

// ---------------------------------------------------------------------------
// PREPROC_DESCセクション
// ---------------------------------------------------------------------------

const PREPROC_DESC_SCALAR_KEYS = ['channels', 'image_channels', 'height', 'width', 'stride'] as const;

function decodePreprocDesc(r: ByteReader): PreprocDesc {
  const nDesc = r.readUVarint();
  const desc = readSArray(r, nDesc);
  const nQval = r.readUVarint();
  const qval = readSArray(r, nQval);
  const scalars: Record<string, number> = {};
  for (const key of PREPROC_DESC_SCALAR_KEYS) {
    scalars[key] = r.readSVarint();
  }
  const extra = readValue(r) as Record<string, JsonValue>;
  return {
    desc, qval, ...scalars, ...extra,
  } as unknown as PreprocDesc;
}

// ---------------------------------------------------------------------------
// コンテナ分解
// ---------------------------------------------------------------------------

function disassembleContainer(data: Uint8Array): Map<string, Uint8Array> {
  if (data.length < HEADER_SIZE) {
    throw new Error('mdlb: ヘッダ未満で切り詰められている');
  }
  const view = new DataView(data.buffer, data.byteOffset, data.byteLength);
  const magic = new TextDecoder('ascii').decode(data.subarray(0, 4));
  if (magic !== MDLB_MAGIC) {
    throw new Error(`mdlb: マジックが不正: ${JSON.stringify(magic)}(期待 "${MDLB_MAGIC}")`);
  }
  const version = view.getUint32(4, true);
  if (version !== MDLB_VERSION) {
    throw new Error(`mdlb: version=${version} は非対応(このリーダは${MDLB_VERSION}のみ)`);
  }
  const nSections = view.getUint32(8, true);
  const dirStart = HEADER_SIZE;
  const payloadStart = dirStart + DIR_ENTRY_SIZE * nSections;
  if (data.length < payloadStart) {
    throw new Error('mdlb: セクションディレクトリが切り詰められている');
  }

  const sections = new Map<string, Uint8Array>();
  for (let i = 0; i < nSections; i++) {
    const off = dirStart + i * DIR_ENTRY_SIZE;
    const tag = new TextDecoder('ascii').decode(data.subarray(off, off + 4));
    const relOffset = Number(view.getBigUint64(off + 8, true));
    const length = Number(view.getBigUint64(off + 16, true));
    const crc = view.getUint32(off + 24, true);
    const absOff = payloadStart + relOffset;
    if (absOff < payloadStart || absOff + length > data.length) {
      throw new Error(`mdlb: セクション"${tag}"のoffset/lengthがファイル範囲外(壊れている)`);
    }
    const blob = data.subarray(absOff, absOff + length);
    const gotCrc = crc32(blob);
    if (gotCrc !== crc) {
      throw new Error(`mdlb: セクション"${tag}"のチェックサム不一致(壊れている): `
        + `期待0x${crc.toString(16)} 実測0x${gotCrc.toString(16)}`);
    }
    if (sections.has(tag)) {
      throw new Error(`mdlb: セクションタグ"${tag}"が重複している`);
    }
    sections.set(tag, blob);
  }
  for (const tag of SECTION_TAGS) {
    if (!sections.has(tag)) throw new Error(`mdlb: 必須セクションが無い: ${tag}`);
  }
  return sections;
}

// ---------------------------------------------------------------------------
// 公開API
// ---------------------------------------------------------------------------

/** `.mdlb` バイナリ(fetch()の`response.arrayBuffer()`結果、または
    それをUint8Array化したもの)を配布バンドルdictへデコードする。
    `loadBundle`(lib.ts)へそのまま渡せる形(BundleDoc)を返す。 */
export function decodeMdlb(input: ArrayBuffer | Uint8Array): BundleDoc {
  const data = input instanceof Uint8Array ? input : new Uint8Array(input);
  const sections = disassembleContainer(data);

  const metaIr = readValue(new ByteReader(sections.get('IRMT')!)) as Record<string, JsonValue>;

  const ttReader = new ByteReader(sections.get('TTBL')!);
  const nTt = ttReader.readUVarint();
  const ttEntries: [string, string[]][] = [];
  for (let i = 0; i < nTt; i++) ttEntries.push(decodeTtEntry(ttReader));

  const idxReader = new ByteReader(sections.get('IDXS')!);
  const nIdx = idxReader.readUVarint();
  const idxEntries: IdxEntry[] = [];
  for (let i = 0; i < nIdx; i++) idxEntries.push(decodeIdxEntry(idxReader));

  const irDict = mergeIrFromMdlb(metaIr, ttEntries, idxEntries);

  const progReader = new ByteReader(sections.get('PROG')!);
  const hasPrograms = progReader.readByte() !== 0;
  let programs: Record<string, BundleProgramDict> | undefined;
  if (hasPrograms) {
    const nProg = progReader.readUVarint();
    programs = {};
    for (let i = 0; i < nProg; i++) {
      const [nodeId, prog] = decodeProgram(progReader);
      programs[nodeId] = prog;
    }
  }

  const ppdsReader = new ByteReader(sections.get('PPDS')!);
  const hasPreprocDesc = ppdsReader.readByte() !== 0;
  const preprocDesc = hasPreprocDesc ? decodePreprocDesc(ppdsReader) : undefined;

  const lablReader = new ByteReader(sections.get('LABL')!);
  const hasLabels = lablReader.readByte() !== 0;
  const labels = hasLabels ? (readValue(lablReader) as JsonValue[]) : undefined;

  const rdotReader = new ByteReader(sections.get('RDOT')!);
  const hasReadout = rdotReader.readByte() !== 0;
  const readout = hasReadout ? (readValue(rdotReader) as Record<string, JsonValue>) : undefined;

  const extra = readValue(new ByteReader(sections.get('EXTR')!)) as Record<string, JsonValue>;

  const bundle: Record<string, JsonValue> = { ...extra, ir: irDict };
  if (hasPrograms) bundle.programs = programs;
  if (hasPreprocDesc) bundle.preproc_desc = preprocDesc;
  if (hasLabels) bundle.labels = labels;
  if (hasReadout) bundle.readout = readout;

  return bundle as unknown as BundleDoc;
}
