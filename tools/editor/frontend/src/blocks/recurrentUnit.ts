// Recurrent時間展開(LSTM/GRUセルのGUI unroll配置)。
// セル1step分を既存IRノード型(lut_layer/concat/add/constant)の合成へ写す純関数と、
// unroll数TでT回インスタンス化し隠れ状態を直列接続する展開器。
//
// 要点(hard/eval modeでの厳密ブール等価):
//   - ゲート/候補(f/i/o/c̃・z/r/h̃) = LutStack(cat([x,h])) → lut_layer列(draft、build時学習)
//   - 状態更新(_mux/_and) = ブール演算: and=lut(tt=0x8) / not=lut(tt=0x1) / or=addノード
//     mux(sel,a,b) = (sel∧a) ∨ (¬sel∧b)  (selが排他なのでOR=XOR=正確)
//   - 入力は単一バス(幅 T*inWidth)。entry恒等lutで分配し、各stepのslice lut(tt=0xa)で x_t を取る
//   - 初期状態 h0/c0 は constant(bits=0)。hard等価のため{0,1}で初期化する
//   - 各ノードに meta.editor.recurrent = { cellType, unroll, step } の由来タグを付与
//
// 等価性検証(tests/test_editor_recurrent_n7.py)はこの代数をPython側で同型に再構成し、
// blocks/recurrent.py の scan_cell(hard) とビット一致を確認する。tt定数は下記を単一情報源とする。

import type { IdAllocator, SubgraphTemplate } from './templates';
import type { IREdgeJson, IRNodeJson } from '../ir/schema';

/** セル種別(blocks/recurrent.py の LogicLSTMCell/LogicGRUCell に対応)。 */
export type RecurrentCellType = 'lstm' | 'gru';

export const RECURRENT_CELL_TYPES: RecurrentCellType[] = ['lstm', 'gru'];

// --- 状態代数の真理値表定数(n_inputs=2、simulate規約 addr=in0*1+in1*2 準拠) ---
// AND: out=1 は addr=3(in0=in1=1)のみ → bit3 → 0b1000
export const AND_TT = '0x8';
// NOT: 両slotを同一入力に。addr∈{0,3}、out=¬in0 → addr0→1 → 0b0001
export const NOT_TT = '0x1';
// PASS-THROUGH(slot0素通し): out=in0 → addr1,addr3 → 0b1010(graphToIr.passThroughTT(2)と一致)
export const PASS_TT = '0xa';

// 相対配置(templates.tsと同じ規約: ドロップ位置基準で読める格子)。
const COL_X = 200;
const ROW_Y = 120;

/** 由来タグ付きの共通meta生成。 */
function recMeta(
  cellType: RecurrentCellType,
  unroll: number,
  step: number,
  col: number,
  row: number,
  detail?: string,
): Record<string, unknown> {
  return {
    editor: {
      position: { x: col * COL_X, y: row * ROW_Y },
      recurrent: { cellType, unroll, step },
      ...(detail ? { detail } : {}),
    },
  };
}

/** draft(学習可能)lut_layer: placeholder接続+pass-through真理値表(build時ランダム初期化)。 */
function draftLut(
  id: string,
  inWidth: number,
  outWidth: number,
  meta: Record<string, unknown>,
): IRNodeJson {
  // placeholderIndices/passThroughTT と同一規約(graphToIr)。循環importを避け直書き。
  const rows: number[][] = [];
  for (let j = 0; j < 2; j++) {
    const row: number[] = new Array<number>(outWidth);
    for (let o = 0; o < outWidth; o++) row[o] = (o * 2 + j) % inWidth;
    rows.push(row);
  }
  return {
    id,
    type: 'lut_layer',
    params: {
      in_width: inWidth,
      out_width: outWidth,
      n_inputs: 2,
      indices: rows,
      truth_tables: new Array<string>(outWidth).fill(PASS_TT),
    },
    train_info: {
      parameterization: 'softmax',
      connections: 'random',
      selection: 'soft',
      initialized: false,
    },
    meta,
  };
}

/** 固定ブール lut_layer(2W→W): 出力oは [busA[o], busB[o]] を n=2 で読み tt を適用する。 */
function fixedBinaryLut(
  id: string,
  width: number,
  tt: string,
  meta: Record<string, unknown>,
): IRNodeJson {
  const a: number[] = [];
  const b: number[] = [];
  for (let o = 0; o < width; o++) {
    a.push(o); // busA(slot0) は先頭 0..W-1
    b.push(width + o); // busB(slot1) は W..2W-1
  }
  return {
    id,
    type: 'lut_layer',
    params: {
      in_width: 2 * width,
      out_width: width,
      n_inputs: 2,
      indices: [a, b],
      truth_tables: new Array<string>(width).fill(tt),
    },
    train_info: { parameterization: 'softmax', connections: 'identity', selection: 'soft' },
    meta,
  };
}

/** NOT lut_layer(W→W): 両slotとも同一入力を読み ¬in0 を返す。 */
function notLut(id: string, width: number, meta: Record<string, unknown>): IRNodeJson {
  const idx = Array.from({ length: width }, (_, o) => o);
  return {
    id,
    type: 'lut_layer',
    params: {
      in_width: width,
      out_width: width,
      n_inputs: 2,
      indices: [idx, [...idx]],
      truth_tables: new Array<string>(width).fill(NOT_TT),
    },
    train_info: { parameterization: 'softmax', connections: 'identity', selection: 'soft' },
    meta,
  };
}

/** slice lut_layer(全入力→x_t): step t の inWidth ビットを素通しで取り出す。 */
function sliceLut(
  id: string,
  totalWidth: number,
  offset: number,
  outWidth: number,
  meta: Record<string, unknown>,
): IRNodeJson {
  const idx = Array.from({ length: outWidth }, (_, o) => offset + o);
  return {
    id,
    type: 'lut_layer',
    params: {
      in_width: totalWidth,
      out_width: outWidth,
      n_inputs: 2,
      indices: [idx, [...idx]],
      truth_tables: new Array<string>(outWidth).fill(PASS_TT),
    },
    train_info: { parameterization: 'softmax', connections: 'identity', selection: 'soft' },
    meta,
  };
}

/** concatノード(in_widths指定)。 */
function concatNode(
  id: string,
  inWidths: number[],
  meta: Record<string, unknown>,
): IRNodeJson {
  return {
    id,
    type: 'concat',
    params: { in_widths: inWidths },
    train_info: {},
    meta,
  };
}

/** add(OR)ノード。 */
function orAddNode(id: string, width: number, meta: Record<string, unknown>): IRNodeJson {
  return {
    id,
    type: 'add',
    params: { width, mode: 'or' },
    train_info: {},
    meta,
  };
}

/** constant(bits=0)ノード: 初期状態 h0/c0。 */
function zeroConstNode(id: string, width: number, meta: Record<string, unknown>): IRNodeJson {
  return {
    id,
    type: 'constant',
    params: { width, bits: new Array<number>(width).fill(0) },
    train_info: { from: 'recurrent_init_state' },
    meta,
  };
}

/** depth段の draft ゲートスタック(先頭は inWidth→hidden、以降 hidden→hidden)。 */
function gateStack(
  alloc: IdAllocator,
  name: string,
  inSrc: string,
  inWidth: number,
  hidden: number,
  depth: number,
  meta: (detail: string) => Record<string, unknown>,
): { nodes: IRNodeJson[]; edges: IREdgeJson[]; outId: string } {
  const nodes: IRNodeJson[] = [];
  const edges: IREdgeJson[] = [];
  let prevId = inSrc;
  let prevW = inWidth;
  for (let d = 0; d < depth; d++) {
    const id = alloc(`${name}${d}`);
    nodes.push(draftLut(id, prevW, hidden, meta(`${name} L${d}`)));
    edges.push({ src: prevId, dst: id });
    prevId = id;
    prevW = hidden;
  }
  return { nodes, edges, outId: prevId };
}

export interface RecurrentUnrollParams {
  cellType: RecurrentCellType;
  unroll: number; // 時間ステップ数 T
  inWidth: number; // 1ステップ入力幅
  hidden: number; // 隠れ状態幅
  depth: number; // ゲートLutStackの段数
}

/**
 * LSTM/GRUセルを unroll 回展開した SubgraphTemplate を生成する純関数。
 *   entry(恒等lut, 幅 T*inWidth) → 各stepの slice で x_t を取り、
 *   ゲート+状態代数で h_t/c_t を計算し次stepへ直列接続する。
 *   exit = 最終step の h。外部入力幅 = T*inWidth、外部出力幅 = hidden。
 */
export function unrollRecurrent(
  params: RecurrentUnrollParams,
  alloc: IdAllocator,
): SubgraphTemplate {
  const { cellType, unroll, inWidth, hidden, depth } = params;
  const total = unroll * inWidth;
  const nodes: IRNodeJson[] = [];
  const edges: IREdgeJson[] = [];
  const m = (step: number, col: number, row = 0, detail?: string) =>
    recMeta(cellType, unroll, step, col, row, detail);

  // 入力分配(恒等lut)。外部入力はここへ接続する(entry)。
  const entryId = alloc('rnn_in');
  nodes.push(sliceLut(entryId, total, 0, total, m(-1, 0, 0, `${cellType} 入力分配`)));
  // 恒等分配は sliceLut(offset=0,outWidth=total)で inWidth==total の素通し。

  // 初期状態(constant 0)。
  const h0 = alloc('h0');
  nodes.push(zeroConstNode(h0, hidden, m(-1, 0, 2, 'h0')));
  let hPrev = h0;
  let cPrev = '';
  if (cellType === 'lstm') {
    const c0 = alloc('c0');
    nodes.push(zeroConstNode(c0, hidden, m(-1, 0, 3, 'c0')));
    cPrev = c0;
  }

  for (let t = 0; t < unroll; t++) {
    const col = 1 + t * 6;
    // x_t スライス
    const xt = alloc(`x${t}`);
    nodes.push(sliceLut(xt, total, t * inWidth, inWidth, m(t, col, 0, `x[${t}]`)));
    edges.push({ src: entryId, dst: xt });

    // z = concat([x_t, h_prev])(slot0=x, slot1=h — torch cat([x,h])と同順)
    const zId = alloc(`z${t}`);
    nodes.push(concatNode(zId, [inWidth, hidden], m(t, col + 1, 0, 'cat[x,h]')));
    edges.push({ src: xt, dst: zId, dst_slot: 0 });
    edges.push({ src: hPrev, dst: zId, dst_slot: 1 });

    if (cellType === 'lstm') {
      // ゲート f/i/o と候補 c̃(すべて z 入力)
      const f = gateStack(alloc, `f${t}_`, zId, inWidth + hidden, hidden, depth, (d) =>
        m(t, col + 2, 0, d),
      );
      const i = gateStack(alloc, `i${t}_`, zId, inWidth + hidden, hidden, depth, (d) =>
        m(t, col + 2, 1, d),
      );
      const o = gateStack(alloc, `o${t}_`, zId, inWidth + hidden, hidden, depth, (d) =>
        m(t, col + 2, 2, d),
      );
      const cand = gateStack(alloc, `g${t}_`, zId, inWidth + hidden, hidden, depth, (d) =>
        m(t, col + 2, 3, d),
      );
      nodes.push(...f.nodes, ...i.nodes, ...o.nodes, ...cand.nodes);
      edges.push(...f.edges, ...i.edges, ...o.edges, ...cand.edges);

      // ic = and(i, cand)
      const icCat = alloc(`ic_cat${t}`);
      nodes.push(concatNode(icCat, [hidden, hidden], m(t, col + 3, 3, 'cat[i,c̃]')));
      edges.push({ src: i.outId, dst: icCat, dst_slot: 0 });
      edges.push({ src: cand.outId, dst: icCat, dst_slot: 1 });
      const ic = alloc(`ic${t}`);
      nodes.push(fixedBinaryLut(ic, hidden, AND_TT, m(t, col + 4, 3, 'and(i,c̃)')));
      edges.push({ src: icCat, dst: ic });

      // notf = not(f)
      const nf = alloc(`nf${t}`);
      nodes.push(notLut(nf, hidden, m(t, col + 3, 0, 'not(f)')));
      edges.push({ src: f.outId, dst: nf });

      // t1 = and(f, c_prev)
      const t1Cat = alloc(`t1cat${t}`);
      nodes.push(concatNode(t1Cat, [hidden, hidden], m(t, col + 3, 1, 'cat[f,c]')));
      edges.push({ src: f.outId, dst: t1Cat, dst_slot: 0 });
      edges.push({ src: cPrev, dst: t1Cat, dst_slot: 1 });
      const t1 = alloc(`t1_${t}`);
      nodes.push(fixedBinaryLut(t1, hidden, AND_TT, m(t, col + 4, 1, 'and(f,c)')));
      edges.push({ src: t1Cat, dst: t1 });

      // t2 = and(notf, ic)
      const t2Cat = alloc(`t2cat${t}`);
      nodes.push(concatNode(t2Cat, [hidden, hidden], m(t, col + 4, 2, 'cat[¬f,ic]')));
      edges.push({ src: nf, dst: t2Cat, dst_slot: 0 });
      edges.push({ src: ic, dst: t2Cat, dst_slot: 1 });
      const t2 = alloc(`t2_${t}`);
      nodes.push(fixedBinaryLut(t2, hidden, AND_TT, m(t, col + 5, 2, 'and(¬f,ic)')));
      edges.push({ src: t2Cat, dst: t2 });

      // c_t = or(t1, t2)
      const cNew = alloc(`c${t}`);
      nodes.push(orAddNode(cNew, hidden, m(t, col + 5, 1, 'c_t=mux')));
      edges.push({ src: t1, dst: cNew, dst_slot: 0 });
      edges.push({ src: t2, dst: cNew, dst_slot: 1 });

      // h_t = and(o, c_t)
      const hCat = alloc(`hcat${t}`);
      nodes.push(concatNode(hCat, [hidden, hidden], m(t, col + 5, 0, 'cat[o,c]')));
      edges.push({ src: o.outId, dst: hCat, dst_slot: 0 });
      edges.push({ src: cNew, dst: hCat, dst_slot: 1 });
      const hNew = alloc(`h${t}`);
      nodes.push(fixedBinaryLut(hNew, hidden, AND_TT, m(t, col + 6, 0, 'h_t=and(o,c)')));
      edges.push({ src: hCat, dst: hNew });

      hPrev = hNew;
      cPrev = cNew;
    } else {
      // GRU: z=update, r=reset(z入力)、候補 h̃ は cat([x, r*h])
      const zg = gateStack(alloc, `z${t}_`, zId, inWidth + hidden, hidden, depth, (d) =>
        m(t, col + 2, 0, d),
      );
      const rg = gateStack(alloc, `r${t}_`, zId, inWidth + hidden, hidden, depth, (d) =>
        m(t, col + 2, 1, d),
      );
      nodes.push(...zg.nodes, ...rg.nodes);
      edges.push(...zg.edges, ...rg.edges);

      // rh = and(r, h_prev)
      const rhCat = alloc(`rhcat${t}`);
      nodes.push(concatNode(rhCat, [hidden, hidden], m(t, col + 3, 1, 'cat[r,h]')));
      edges.push({ src: rg.outId, dst: rhCat, dst_slot: 0 });
      edges.push({ src: hPrev, dst: rhCat, dst_slot: 1 });
      const rh = alloc(`rh${t}`);
      nodes.push(fixedBinaryLut(rh, hidden, AND_TT, m(t, col + 4, 1, 'and(r,h)')));
      edges.push({ src: rhCat, dst: rh });

      // cand_in = concat([x_t, rh]) → h̃ = h_cand(cand_in)
      const candIn = alloc(`gin${t}`);
      nodes.push(concatNode(candIn, [inWidth, hidden], m(t, col + 4, 2, 'cat[x,rh]')));
      edges.push({ src: xt, dst: candIn, dst_slot: 0 });
      edges.push({ src: rh, dst: candIn, dst_slot: 1 });
      const cand = gateStack(alloc, `g${t}_`, candIn, inWidth + hidden, hidden, depth, (d) =>
        m(t, col + 5, 2, d),
      );
      nodes.push(...cand.nodes);
      edges.push(...cand.edges);

      // h_t = mux(z, h̃, h_prev) = (z∧h̃) ∨ (¬z∧h_prev)
      const nz = alloc(`nz${t}`);
      nodes.push(notLut(nz, hidden, m(t, col + 3, 0, 'not(z)')));
      edges.push({ src: zg.outId, dst: nz });

      const t1Cat = alloc(`t1cat${t}`);
      nodes.push(concatNode(t1Cat, [hidden, hidden], m(t, col + 6, 0, 'cat[z,h̃]')));
      edges.push({ src: zg.outId, dst: t1Cat, dst_slot: 0 });
      edges.push({ src: cand.outId, dst: t1Cat, dst_slot: 1 });
      const t1 = alloc(`t1_${t}`);
      nodes.push(fixedBinaryLut(t1, hidden, AND_TT, m(t, col + 7, 0, 'and(z,h̃)')));
      edges.push({ src: t1Cat, dst: t1 });

      const t2Cat = alloc(`t2cat${t}`);
      nodes.push(concatNode(t2Cat, [hidden, hidden], m(t, col + 6, 1, 'cat[¬z,h]')));
      edges.push({ src: nz, dst: t2Cat, dst_slot: 0 });
      edges.push({ src: hPrev, dst: t2Cat, dst_slot: 1 });
      const t2 = alloc(`t2_${t}`);
      nodes.push(fixedBinaryLut(t2, hidden, AND_TT, m(t, col + 7, 1, 'and(¬z,h)')));
      edges.push({ src: t2Cat, dst: t2 });

      const hNew = alloc(`h${t}`);
      nodes.push(orAddNode(hNew, hidden, m(t, col + 8, 0, 'h_t=mux')));
      edges.push({ src: t1, dst: hNew, dst_slot: 0 });
      edges.push({ src: t2, dst: hNew, dst_slot: 1 });

      hPrev = hNew;
    }
  }

  return { nodes, edges, entryId, exitId: hPrev };
}
