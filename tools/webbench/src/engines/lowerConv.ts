// conv2d / or_pool2d / upsample2d / permute ネイティブノードを lut_layer 列へ
// 展開するTSパス(upsample2d/permuteにも対応)。
// griffith_logic/export/lowering.py の _unroll_conv_tree / _or_pool_lut_node /
// _upsample2d_node_to_lut / _permute_node_to_lut / _specialize_zero /
// lower_to_primitives の忠実移植。ビット単位で ir/simulate.py と一致する
// (conv_golden.json 整数一致テストで固定)。
//
// upsample2d(nearest) / permute(SpatialGroupSumのチャネル並べ替え・thermo
// channel permute等)は conv2d と同型のネイティブノード化(小さい
// params、indexは持たない)。ここで lut_layer への展開(irflatエンジン用)を
// 行う。packed エンジンは展開せず packedConv.ts のネイティブ実行を使う。
import { ttParse } from '../shared';
import type {
  Conv2dParams,
  IREdgeJson,
  IRGraphJson,
  IRNodeJson,
  OrPool2dParams,
  PermuteParams,
  UpSample2dParams,
} from '../shared';

const IDENTITY_TT_1IN = '0x2'; // n_inputs=1 のパススルー(out = in)

function ttHex(v: bigint): string {
  return `0x${v.toString(16)}`;
}

/** スロットjを0固定へ畳み込む(padding処理の中核)。lowering._specialize_zero */
function specializeZero(tt: bigint, j: number, nInputs: number): bigint {
  const mask = ~(1 << j); // 32bit; k<64(nInputs<=6)なのでビットjクリアに十分
  let out = 0n;
  for (let k = 0; k < 1 << nInputs; k++) {
    const kk = k & mask;
    if ((tt >> BigInt(kk)) & 1n) out |= 1n << BigInt(k);
  }
  return out;
}

function lutNode(
  id: string,
  inWidth: number,
  outWidth: number,
  nInputs: number,
  indices: number[][],
  tts: string[],
  loweredFrom: string,
): IRNodeJson {
  return {
    id,
    type: 'lut_layer',
    params: {
      in_width: inWidth,
      out_width: outWidth,
      n_inputs: nInputs,
      indices,
      truth_tables: tts,
    },
    train_info: { lowered_from: loweredFrom },
    meta: {},
  };
}

function pair(v: number | number[]): [number, number] {
  return Array.isArray(v) ? [v[0], v[1]] : [v, v];
}

/** conv木を lut_layer 列へ展開する(lowering._unroll_conv_tree)。 */
function unrollConvTree(nodeId: string, p: Conv2dParams): IRNodeJson[] {
  const [C, H, W] = p.in_shape;
  const [kH, kW] = pair(p.kernel_size);
  const [sH, sW] = pair(p.stride);
  const [pH, pW] = pair(p.padding);
  const HOut = Math.floor((H + 2 * pH - kH) / sH) + 1;
  const WOut = Math.floor((W + 2 * pW - kW) / sW) + 1;
  const L = HOut * WOut;
  const OC = p.out_channels;
  const N = p.n_inputs;
  const d = p.tree_depth;
  const inWidth = C * H * W;
  const nLeaves = N ** d;
  const nInternal = N === 1 ? d : (nLeaves - 1) / (N - 1);

  // idxf[oc][leaf], ttsGates[oc][g] (oc-major, tree-level順)
  const idxf = p.indices_flat;
  const ttsGates: bigint[][] = [];
  for (let oc = 0; oc < OC; oc++) {
    const row: bigint[] = [];
    for (let g = 0; g < nInternal; g++) row.push(ttParse(p.truth_tables[oc * nInternal + g]));
    ttsGates.push(row);
  }

  const nodes: IRNodeJson[] = [];
  let nGates = nLeaves;
  let gateOffset = 0;
  let prevWidth = inWidth;

  for (let level = 0; level < d; level++) {
    const gOut = nGates / N;
    const widthOut = OC * L * gOut;
    const indices: number[][] = Array.from({ length: N }, () =>
      new Array<number>(widthOut).fill(0),
    );
    const tts: string[] = new Array(widthOut);

    for (let oc = 0; oc < OC; oc++) {
      for (let pos = 0; pos < L; pos++) {
        const oy = Math.floor(pos / WOut);
        const ox = pos % WOut;
        const baseOut = (oc * L + pos) * gOut;
        for (let g = 0; g < gOut; g++) {
          const outIdx = baseOut + g;
          let tt = ttsGates[oc][gateOffset + g];
          for (let j = 0; j < N; j++) {
            const child = g * N + j;
            let src: number;
            if (level === 0) {
              const pp = idxf[oc][child];
              const c = Math.floor(pp / (kH * kW));
              const rem = pp % (kH * kW);
              const dy = Math.floor(rem / kW);
              const dx = rem % kW;
              const y = oy * sH - pH + dy;
              const x = ox * sW - pW + dx;
              if (y >= 0 && y < H && x >= 0 && x < W) {
                src = c * H * W + y * W + x;
              } else {
                tt = specializeZero(tt, j, N);
                src = 0;
              }
            } else {
              src = (oc * L + pos) * nGates + child;
            }
            indices[j][outIdx] = src;
          }
          tts[outIdx] = ttHex(tt);
        }
      }
    }
    nodes.push(lutNode(`${nodeId}_lv${level}`, prevWidth, widthOut, N, indices, tts, 'LutConv2d'));
    prevWidth = widthOut;
    nGates = gOut;
    gateOffset += gOut;
  }
  return nodes;
}

/** 窓ORの lut_layer ノード(lowering._or_pool_lut_node)。 */
function orPoolLutNode(nodeId: string, p: OrPool2dParams): IRNodeJson {
  const [C, H, W] = p.in_shape;
  const k = p.kernel_size;
  const s = p.stride;
  const pad = p.padding ?? 0;
  const nIn = k * k;
  if (nIn > 6) {
    throw new Error(`${nodeId}: 窓${k}x${k}(${nIn}入力)のORは未対応(<=6入力)`);
  }
  const HOut = Math.floor((H + 2 * pad - k) / s) + 1;
  const WOut = Math.floor((W + 2 * pad - k) / s) + 1;
  const widthOut = C * HOut * WOut;
  const inWidth = C * H * W;
  const orTt = (1n << (1n << BigInt(nIn))) - 2n; // 全パターン1、パターン0のみ0

  const indices: number[][] = Array.from({ length: nIn }, () =>
    new Array<number>(widthOut).fill(0),
  );
  const tts: string[] = new Array(widthOut);
  for (let c = 0; c < C; c++) {
    for (let oy = 0; oy < HOut; oy++) {
      for (let ox = 0; ox < WOut; ox++) {
        const outIdx = c * HOut * WOut + oy * WOut + ox;
        let tt = orTt;
        for (let j = 0; j < nIn; j++) {
          const dy = Math.floor(j / k);
          const dx = j % k;
          const y = oy * s - pad + dy;
          const x = ox * s - pad + dx;
          if (y >= 0 && y < H && x >= 0 && x < W) {
            indices[j][outIdx] = c * H * W + y * W + x;
          } else {
            tt = specializeZero(tt, j, nIn);
            indices[j][outIdx] = 0;
          }
        }
        tts[outIdx] = ttHex(tt);
      }
    }
  }
  return lutNode(nodeId, inWidth, widthOut, nIn, indices, tts, 'OrPooling2d');
}

/** upsample2d(nearest)の恒等複製lut_layerノード(lowering._upsample2d_node_to_lut)。
    out[c,y,x] = in[c, y//scale, x//scale] のindexをJIT計算する。 */
function upsampleLutNode(nodeId: string, p: UpSample2dParams): IRNodeJson {
  const [C, H, W] = p.in_shape;
  const s = p.scale;
  const HOut = H * s;
  const WOut = W * s;
  const widthOut = C * HOut * WOut;
  const inWidth = C * H * W;

  const indices: number[][] = [new Array<number>(widthOut).fill(0)];
  for (let c = 0; c < C; c++) {
    for (let y = 0; y < HOut; y++) {
      for (let x = 0; x < WOut; x++) {
        const outIdx = (c * HOut + y) * WOut + x;
        indices[0][outIdx] = (c * H + Math.floor(y / s)) * W + Math.floor(x / s);
      }
    }
  }
  const tts = new Array<string>(widthOut).fill(IDENTITY_TT_1IN);
  return lutNode(nodeId, inWidth, widthOut, 1, indices, tts, 'UpSample2d');
}

/** permuteの恒等置換lut_layerノード(lowering._permute_node_to_lut)。
    入力をin_shape形状のreshape viewとみなし、permに従って軸並べ替えした
    row-major出力のindexをJIT計算する。 */
function permuteLutNode(nodeId: string, p: PermuteParams): IRNodeJson {
  const dims = p.in_shape;
  const perm = p.perm;
  const n = dims.length;
  const width = dims.reduce((a, b) => a * b, 1);

  // 入力側strides(row-major、末尾軸のstride=1)
  const inStrides = new Array<number>(n).fill(1);
  for (let i = n - 2; i >= 0; i--) inStrides[i] = inStrides[i + 1] * dims[i + 1];
  const outDims = perm.map((q) => dims[q]);
  const outStrides = new Array<number>(n).fill(1);
  for (let i = n - 2; i >= 0; i--) outStrides[i] = outStrides[i + 1] * outDims[i + 1];

  const indices: number[][] = [new Array<number>(width).fill(0)];
  for (let outIdx = 0; outIdx < width; outIdx++) {
    let rem = outIdx;
    let src = 0;
    for (let i = 0; i < n; i++) {
      const coord = Math.floor(rem / outStrides[i]);
      rem -= coord * outStrides[i];
      src += coord * inStrides[perm[i]];
    }
    indices[0][outIdx] = src;
  }
  const tts = new Array<string>(width).fill(IDENTITY_TT_1IN);
  return lutNode(nodeId, width, width, 1, indices, tts, 'Permute');
}

/** or_pool2d が「bitplaneエンジンのネイティブ2x2路(orPool2x2)」に載る
    パラメータかどうか(kernel=2,stride=2,padding=0のみ)。 */
function isNative2x2OrPool(p: OrPool2dParams): boolean {
  return p.kernel_size === 2 && p.stride === 2 && (p.padding ?? 0) === 0;
}

/** lowerToPrimitives のオプション。nativeConv=true のとき conv2d/upsample2d/
    permute を展開せずそのまま残す(packedエンジンが packedConv.ts の
    ネイティブ実装で実行する)。or_pool2d は既定では常に lut_layer へ展開
    する(実行の律速でないため共通経路)。nativeOrPool=true(bitplane
    エンジン専用)のときは、kernel=2,stride=2,padding=0のor_pool2dだけを
    展開せず残す(bitplaneGraph.ts::evalIrBitplaneがorPool2x2で処理する)。
    それ以外の窓/ストライド/パディングは nativeOrPool=true でも常に
    lut_layer へ展開する(正しさ優先: ネイティブ路が無い組合せはフォール
    バック先を確実に用意する)。 */
export interface LowerOptions {
  nativeConv?: boolean;
  nativeOrPool?: boolean;
}

/** conv2d / or_pool2d / upsample2d / permute を含むIRを lut_layer のみの
    IRへ展開する。純関数。lowering.lower_to_primitives と同一のid/エッジ規約。
    nativeConv=true のときは conv2d/upsample2d/permute を展開せず残す
    (packedエンジン用)。nativeOrPool=true のときはさらに2x2/pad0の
    or_pool2dも展開せず残す(bitplaneエンジン用、既定はfalse=現行動作)。 */
export function lowerToPrimitives(
  ir: IRGraphJson, opts: LowerOptions = {},
): IRGraphJson {
  const nativeConv = opts.nativeConv ?? false;
  const nativeOrPool = opts.nativeOrPool ?? false;
  const isNativeSpatial = (t: string) =>
    t === 'conv2d' || t === 'upsample2d' || t === 'permute';
  const keepsOrPoolNative = (n: IRNodeJson): boolean =>
    nativeOrPool && isNative2x2OrPool(n.params as unknown as OrPool2dParams);
  const hasLowerable = ir.nodes.some(
    (n) => (isNativeSpatial(n.type) && !nativeConv)
      || (n.type === 'or_pool2d' && !keepsOrPoolNative(n)),
  );
  if (!hasLowerable) return ir;

  const newNodes: IRNodeJson[] = [];
  const srcRemap = new Map<string, string>(); // 元id → 展開後の最終id
  const dstRemap = new Map<string, string>(); // 元id → 展開後の先頭id
  const chainEdges: IREdgeJson[] = [];

  for (const node of ir.nodes) {
    if (node.type === 'conv2d' && !nativeConv) {
      const chain = unrollConvTree(node.id, node.params as unknown as Conv2dParams);
      newNodes.push(...chain);
      dstRemap.set(node.id, chain[0].id);
      srcRemap.set(node.id, chain[chain.length - 1].id);
      for (let i = 0; i + 1 < chain.length; i++) {
        chainEdges.push({ src: chain[i].id, dst: chain[i + 1].id });
      }
    } else if (node.type === 'or_pool2d') {
      if (keepsOrPoolNative(node)) {
        newNodes.push(node);
      } else {
        newNodes.push(orPoolLutNode(node.id, node.params as unknown as OrPool2dParams));
      }
    } else if (node.type === 'upsample2d' && !nativeConv) {
      newNodes.push(upsampleLutNode(node.id, node.params as unknown as UpSample2dParams));
    } else if (node.type === 'permute' && !nativeConv) {
      newNodes.push(permuteLutNode(node.id, node.params as unknown as PermuteParams));
    } else {
      newNodes.push(node);
    }
  }

  const edges: IREdgeJson[] = ir.edges.map((e) => ({
    src: srcRemap.get(e.src) ?? e.src,
    dst: dstRemap.get(e.dst) ?? e.dst,
    dst_slot: e.dst_slot,
  }));
  edges.push(...chainEdges);

  return { ...ir, nodes: newNodes, edges };
}

export { IDENTITY_TT_1IN };
