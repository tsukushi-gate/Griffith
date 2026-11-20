// 描画中エラーで白画面にしない(原因をユーザーに見せる)。
// グローバルエラー/未処理 Promise 拒否もオーバーレイ表示する。
import { Component, useEffect, useState, type ReactNode } from 'react';

interface State {
  error: Error | null;
}

export class ErrorBoundary extends Component<{ children: ReactNode }, State> {
  state: State = { error: null };

  static getDerivedStateFromError(error: Error): State {
    return { error };
  }

  render() {
    if (this.state.error) {
      return (
        <div className="fatal-overlay">
          <h2>描画エラー</h2>
          <pre>{this.state.error.message}</pre>
          <pre className="stack">{this.state.error.stack}</pre>
          <button onClick={() => location.reload()}>リロード</button>
        </div>
      );
    }
    return this.props.children;
  }
}

export function GlobalErrorOverlay() {
  const [msg, setMsg] = useState<string | null>(null);
  useEffect(() => {
    const onError = (e: ErrorEvent) => {
      // ResizeObserver のループ通知はブラウザ既知の良性警告(ReactFlow×
      // ペインリサイズで発生、実害なし)— エラーオーバーレイに出さない
      if (/ResizeObserver loop/i.test(e.message)) return;
      setMsg(`${e.message}\n${e.filename}:${e.lineno}`);
    };
    const onReject = (e: PromiseRejectionEvent) =>
      setMsg(`Unhandled rejection: ${String(e.reason)}`);
    window.addEventListener('error', onError);
    window.addEventListener('unhandledrejection', onReject);
    return () => {
      window.removeEventListener('error', onError);
      window.removeEventListener('unhandledrejection', onReject);
    };
  }, []);
  if (!msg) return null;
  return (
    <div className="fatal-overlay">
      <h2>実行時エラー</h2>
      <pre>{msg}</pre>
      <button onClick={() => setMsg(null)}>閉じる</button>
    </div>
  );
}
