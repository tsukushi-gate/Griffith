/**
 * DDLGNビジュアライザーのエントリーポイント。
 *
 * deck.glのDeckインスタンスを生成し、WebSocket経由で受信した
 * ネットワークデータをリアルタイム描画する。
 */

import { Deck, OrthographicView, type OrthographicViewState } from '@deck.gl/core'

import { createNodeLayer, createEdgeLayer, createTextLayer } from './layers'
import { buildNodeArray, buildEdgeArray, GATE_NAMES, type NetworkData, type NodeAccessor } from './layout'
import { VisualizerSocket } from './websocket'

// --- 状態管理 ---
let currentNodes: NodeAccessor[] = []
let currentEdges: ReturnType<typeof buildEdgeArray> = []
let currentZoom = -6
let networkStats = { totalNodes: 0, totalEdges: 0, layerCount: 0 }

// --- UI要素 ---
const statsEl = document.getElementById('stats')!
const hoverEl = document.getElementById('hover-info')!

function updateStats(): void {
  const fmt = (n: number) => n.toLocaleString()
  statsEl.innerHTML = [
    `<span class="label">ノード:</span> <span class="value">${fmt(networkStats.totalNodes)}</span>`,
    `<span class="label">エッジ:</span> <span class="value">${fmt(networkStats.totalEdges)}</span>`,
    `<span class="label">レイヤー:</span> <span class="value">${networkStats.layerCount}</span>`,
    `<span class="label">ズーム:</span> <span class="value">${currentZoom.toFixed(1)}</span>`,
  ].join('<br>')
}

function updateHover(node?: NodeAccessor): void {
  if (!node) {
    hoverEl.textContent = ''
    return
  }
  const gateName = GATE_NAMES[node.gateId] ?? `Gate ${node.gateId}`
  const conf = (node.confidence * 100).toFixed(1)
  hoverEl.innerHTML = [
    `<b>${gateName}</b> (#${node.index})`,
    `確信度: ${conf}%`,
  ].join('<br>')
}

// --- deck.gl（OrthographicView: 2D座標系） ---
const ORTHO_VIEW = new OrthographicView({ id: 'ortho', flipY: false })

const VIEW_ID = 'ortho'

const deck = new Deck({
  canvas: 'deck-canvas',
  views: [ORTHO_VIEW],
  initialViewState: {
    [VIEW_ID]: {
      target: [240, 50, 0],
      zoom: -1,
      minZoom: -8,
      maxZoom: 10,
    } satisfies OrthographicViewState,
  },
  controller: true,
  onViewStateChange: (params) => {
    const vs = params.viewState
    currentZoom = ((vs as OrthographicViewState).zoom ?? 0) as number
    updateStats()
    updateLayers()
  },
  getTooltip: () => null,
  layers: [],
})

function updateLayers(): void {
  deck.setProps({
    layers: [
      createEdgeLayer(currentEdges, currentZoom),
      createNodeLayer(currentNodes, (info) => {
        updateHover(info.object as NodeAccessor | undefined)
      }),
      createTextLayer(currentNodes, currentZoom),
    ],
  })
}

// --- データ受信 ---
function onNetworkData(data: NetworkData): void {
  networkStats = {
    totalNodes: data.totalNodes,
    totalEdges: data.totalEdges,
    layerCount: data.layerCount,
  }

  currentNodes = buildNodeArray(data)
  currentEdges = buildEdgeArray(data)

  updateStats()
  updateLayers()

  // 初回表示: ネットワーク全体が見えるようにビューを調整
  if (data.totalNodes > 0) {
    let xMin = Infinity, xMax = -Infinity
    let yMin = Infinity, yMax = -Infinity
    for (const n of currentNodes) {
      if (n.position[0] < xMin) xMin = n.position[0]
      if (n.position[0] > xMax) xMax = n.position[0]
      if (n.position[1] < yMin) yMin = n.position[1]
      if (n.position[1] > yMax) yMax = n.position[1]
    }
    const cx = (xMin + xMax) / 2
    const cy = (yMin + yMax) / 2

    // ネットワーク全体が画面に収まるズームレベルを概算
    const span = Math.max(xMax - xMin, yMax - yMin, 1)
    const canvasSize = Math.min(window.innerWidth, window.innerHeight)
    const zoom = Math.log2(canvasSize / span) - 1

    const newViewState: OrthographicViewState = {
      target: [cx, cy, 0],
      zoom: Math.max(-8, Math.min(zoom, 5)),
      minZoom: -8,
      maxZoom: 10,
    }

    deck.setProps({ initialViewState: { [VIEW_ID]: newViewState } })
    currentZoom = newViewState.zoom as number
    updateStats()
  }
}

// --- WebSocket接続 ---
const wsUrl = `ws://${window.location.hostname}:8765`

const socket = new VisualizerSocket({
  url: wsUrl,
  onData: onNetworkData,
  onStatus: (status) => {
    const statusLine = `<span class="label">状態:</span> <span class="value">${status}</span>`
    if (statsEl.children.length === 0) {
      statsEl.innerHTML = statusLine
    }
  },
})

socket.connect()
updateStats()
