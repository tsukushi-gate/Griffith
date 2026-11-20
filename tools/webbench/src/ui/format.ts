// 指標カード用の数値整形。数値と単位を分けて返すのは、カード上で
// 数字だけを大きく、単位を小さく描くため。欠損は modelMeta.MISSING で統一する。
import { MISSING } from '../modelMeta';

export interface BigValue {
  value: string;
  /** 単位(無いときは空文字)。 */
  unit: string;
}

const NONE: BigValue = { value: MISSING, unit: '' };

/** ミリ秒。1ms未満は小数2桁、それ以上は小数1桁(桁が踊らない範囲で情報を残す)。 */
export function fmtMs(v: number | null | undefined): BigValue {
  if (v === null || v === undefined || !Number.isFinite(v)) return NONE;
  return { value: v < 1 ? v.toFixed(2) : v.toFixed(1), unit: 'ms' };
}

/** バイト数。単位を分けて返す点以外は modelMeta.fmtBytes と同じ刻み。 */
export function fmtBytesBig(v: number | null | undefined): BigValue {
  if (v === null || v === undefined || !Number.isFinite(v)) return NONE;
  if (v >= 1024 * 1024) return { value: (v / (1024 * 1024)).toFixed(2), unit: 'MB' };
  if (v >= 1024) return { value: (v / 1024).toFixed(1), unit: 'KB' };
  return { value: String(Math.round(v)), unit: 'B' };
}

/** 精度(mAP / mIoU)。モデルJSONの実値をそのまま小数4桁で出す。 */
export function fmtMetricBig(v: number | null | undefined): BigValue {
  if (v === null || v === undefined || !Number.isFinite(v)) return NONE;
  return { value: v.toFixed(4), unit: '' };
}

/** 1行に収めるバイト表記(補足行用)。 */
export function fmtBytesInline(v: number | null | undefined): string {
  const b = fmtBytesBig(v);
  return b.unit === '' ? b.value : `${b.value} ${b.unit}`;
}
