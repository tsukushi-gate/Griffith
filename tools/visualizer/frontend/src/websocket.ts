/**
 * WebSocket通信モジュール。
 *
 * バックエンドサーバーからバイナリデータを受信し、コールバックで通知する。
 */

import { decodeNetworkData, type NetworkData } from './layout'

export interface WSOptions {
  url: string
  onData: (data: NetworkData) => void
  onStatus: (status: string) => void
}

export class VisualizerSocket {
  private ws: WebSocket | null = null
  private options: WSOptions
  private reconnectTimer: ReturnType<typeof setTimeout> | null = null

  constructor(options: WSOptions) {
    this.options = options
  }

  connect(): void {
    this.options.onStatus('接続中...')

    const ws = new WebSocket(this.options.url)
    ws.binaryType = 'arraybuffer'

    ws.onopen = () => {
      this.options.onStatus('接続済み')
    }

    ws.onmessage = (event: MessageEvent) => {
      if (event.data instanceof ArrayBuffer) {
        console.log(`[WS] バイナリ受信: ${event.data.byteLength.toLocaleString()} bytes`)
        const data = decodeNetworkData(event.data)
        console.log(`[WS] デコード完了: ${data.totalNodes.toLocaleString()}ノード, ${data.totalEdges.toLocaleString()}エッジ`)
        this.options.onData(data)
      } else {
        console.log('[WS] テキスト受信:', event.data)
      }
    }

    ws.onclose = () => {
      this.options.onStatus('切断 — 再接続中...')
      this.ws = null
      this.scheduleReconnect()
    }

    ws.onerror = () => {
      this.options.onStatus('接続エラー')
      ws.close()
    }

    this.ws = ws
  }

  private scheduleReconnect(): void {
    if (this.reconnectTimer) return
    this.reconnectTimer = setTimeout(() => {
      this.reconnectTimer = null
      this.connect()
    }, 3000)
  }

  disconnect(): void {
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer)
      this.reconnectTimer = null
    }
    if (this.ws) {
      this.ws.close()
      this.ws = null
    }
  }
}
