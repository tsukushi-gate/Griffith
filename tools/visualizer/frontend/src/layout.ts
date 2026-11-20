/**
 * バイナリデータのデコードとレイアウト情報の管理。
 *
 * サーバーから受信したArrayBufferをTypedArrayに変換し、
 * deck.glレイヤーが消費できる形式にする。
 */

/** ノード1個あたりのfloat数: x, y, gate_id, confidence, r, g, b, a */
export const NODE_STRIDE = 8

/** エッジ1本あたりのfloat数: x1, y1, x2, y2 */
export const EDGE_STRIDE = 4

/** N=2用のゲート名マッピング */
export const GATE_NAMES: Record<number, string> = {
  [-1]: 'IN',
  0: 'FALSE', 1: 'AND', 2: 'A>B', 3: 'A',
  4: 'B>A', 5: 'B', 6: 'XOR', 7: 'OR',
  8: 'NOR', 9: 'XNOR', 10: '!B', 11: 'A≥B',
  12: '!A', 13: 'B≥A', 14: 'NAND', 15: 'TRUE',
}

export interface NetworkData {
  totalNodes: number
  totalEdges: number
  layerCount: number
  /** ノード配列: 各ノード8要素 [x, y, gateId, confidence, r, g, b, a] */
  nodes: Float32Array
  /** エッジ配列: 各エッジ4要素 [x1, y1, x2, y2] */
  edges: Float32Array
}

export interface NodeAccessor {
  position: [number, number]
  gateId: number
  confidence: number
  color: [number, number, number, number]
  index: number
}

/**
 * バイナリメッセージをデコードする。
 *
 * フォーマット:
 *   ヘッダ 12 bytes: [totalNodes: u32, totalEdges: u32, layerCount: u32]
 *   ノードデータ: float32[] × (totalNodes × 8)
 *   エッジデータ: float32[] × (totalEdges × 4)
 */
export function decodeNetworkData(buffer: ArrayBuffer): NetworkData {
  const header = new Uint32Array(buffer, 0, 3)
  const totalNodes = header[0]
  const totalEdges = header[1]
  const layerCount = header[2]

  const nodeOffset = 12  // 3 × uint32
  const nodeByteLength = totalNodes * NODE_STRIDE * 4
  const edgeOffset = nodeOffset + nodeByteLength

  const nodes = new Float32Array(buffer, nodeOffset, totalNodes * NODE_STRIDE)
  const edges = new Float32Array(buffer, edgeOffset, totalEdges * EDGE_STRIDE)

  return { totalNodes, totalEdges, layerCount, nodes, edges }
}

/** ノードデータを配列オブジェクトに展開する（deck.glのdata用） */
export function buildNodeArray(data: NetworkData): NodeAccessor[] {
  const { nodes, totalNodes } = data
  const result: NodeAccessor[] = new Array(totalNodes)

  for (let i = 0; i < totalNodes; i++) {
    const off = i * NODE_STRIDE
    result[i] = {
      position: [nodes[off], nodes[off + 1]],
      gateId: nodes[off + 2],
      confidence: nodes[off + 3],
      color: [nodes[off + 4], nodes[off + 5], nodes[off + 6], nodes[off + 7]],
      index: i,
    }
  }
  return result
}

/** エッジデータを配列オブジェクトに展開する */
export interface EdgeAccessor {
  source: [number, number]
  target: [number, number]
  index: number
}

export function buildEdgeArray(data: NetworkData): EdgeAccessor[] {
  const { edges, totalEdges } = data
  const result: EdgeAccessor[] = new Array(totalEdges)

  for (let i = 0; i < totalEdges; i++) {
    const off = i * EDGE_STRIDE
    result[i] = {
      source: [edges[off], edges[off + 1]],
      target: [edges[off + 2], edges[off + 3]],
      index: i,
    }
  }
  return result
}
