// 変換ウィザードの状態機械(進捗ポーリング)を、Reactから切り離した
// 純粋なasync関数として提供する。start→0.5sポーリング→done/error/cancelled を
// 単方向に駆動し、doneならgetCurrentIRでIRを取得してonIRロードコールバックを呼ぶ。
// UIに依存しないためvitestでfetchをモックして状態遷移を検証できる。

import {
  cancelConversion,
  getConversionStatus,
  getCurrentIR,
  startConversion,
  type ConvertStartRequest,
  type ConvertStatus,
} from '../api/rest';
import type { IRGraphJson } from '../ir/schema';

/** 変換phaseの日本語ラベル(backend ConversionSession.phase と対応)。 */
export const PHASE_LABELS: Record<string, string> = {
  preparing: '準備中',
  pruning: '枝刈り+微調整',
  quantizing: '量子化',
  lowering: '回路化',
  verifying: '検証',
  done: '完了',
  error: 'エラー',
  cancelled: 'キャンセル済み',
};

export function phaseLabel(phase: string): string {
  return PHASE_LABELS[phase] ?? phase;
}

/** phaseが終端(これ以上ポーリング不要)か。 */
export function isTerminalPhase(phase: string): boolean {
  return phase === 'done' || phase === 'error' || phase === 'cancelled';
}

export const POLL_INTERVAL_MS = 500;

export interface RunConversionCallbacks {
  /** 各ポーリングで最新statusを通知(進捗UI更新用)。 */
  onStatus: (status: ConvertStatus) => void;
  /** done時にIRをストアへ反映する(ckptインポートと同じ経路を呼び出し側で実装)。 */
  onIR: (ir: IRGraphJson) => void;
  /** 中断チェック(true=以降のポーリングを打ち切る。ダイアログclose等)。 */
  shouldAbort?: () => boolean;
  /** ポーリング間の待機(テストでfake timer不要にするため注入可能)。 */
  sleep?: (ms: number) => Promise<void>;
}

const defaultSleep = (ms: number) =>
  new Promise<void>((resolve) => setTimeout(resolve, ms));

export type RunConversionOutcome =
  | { kind: 'done' }
  | { kind: 'error'; message: string }
  | { kind: 'cancelled' }
  | { kind: 'aborted' };

/**
 * 変換を開始して終端phaseまでポーリングする。
 * - done: getCurrentIR()でIRを取得しonIR()を呼び、{kind:'done'}を返す
 * - error: {kind:'error', message}
 * - cancelled: {kind:'cancelled'}
 * - shouldAbort()がtrueになったら{kind:'aborted'}(以降のポーリングを止める)
 * 例外(startの409/422やネットワーク)はそのままthrowするので呼び出し側でcatchする。
 */
export async function runConversion(
  params: ConvertStartRequest,
  callbacks: RunConversionCallbacks,
): Promise<RunConversionOutcome & { convertId?: string }> {
  const sleep = callbacks.sleep ?? defaultSleep;
  const { convert_id: convertId } = await startConversion(params);

  for (;;) {
    if (callbacks.shouldAbort?.()) return { kind: 'aborted', convertId };

    const status = await getConversionStatus(convertId);
    callbacks.onStatus(status);

    if (status.phase === 'done') {
      const { ir } = await getCurrentIR();
      callbacks.onIR(ir);
      return { kind: 'done', convertId };
    }
    if (status.phase === 'error') {
      return { kind: 'error', message: status.error ?? '変換に失敗しました', convertId };
    }
    if (status.phase === 'cancelled') {
      return { kind: 'cancelled', convertId };
    }

    await sleep(POLL_INTERVAL_MS);
  }
}

/** 変換のキャンセル要求(失敗しても無視。UI側は最終phaseで判断する)。 */
export async function requestCancel(convertId: string): Promise<void> {
  try {
    await cancelConversion(convertId);
  } catch {
    // キャンセルAPIの失敗はポーリング側のcancelled/done検知に委ねる
  }
}
