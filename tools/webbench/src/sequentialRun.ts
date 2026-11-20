// side-by-side比較デモ(det/detmono/detreal/detface/seg)の「同時1本」実行ヘルパー。
//
// 動機: これらのページは複数ランタイム(DLGN / ONNX Runtime Web / TF.js / MediaPipe)の
// 推論を Promise.all で同時に走らせていたため、表示される「推論 X ms」がCPU/GPU資源の
// 競合で真の値より遅く出ていた。detbackends_main.ts / segbackends_main.ts の
// マトリクスベンチは既に for ループで逐次計測しており(資源競合を避け同一ページで
// 順に測る方法論)、本モジュールはその方針を5つのside-by-sideデモにも揃えるための
// 共通ロジックを1箇所に切り出したもの。DOM/document に一切触れないため node の
// vitest だけで「同時実行数が1であること」を機械的に検証できる。
export type SequentialOutcome<R> =
  | { ok: true; value: R }
  | { ok: false; error: unknown };

/**
 * items を先頭から1つずつ、前の runFn が完全に完了(成功/失敗いずれも)してから
 * 次を開始する形で実行する。
 *
 * - 同時に in-flight になる runFn 呼び出しは常に高々1つ(順序も items のまま)。
 * - ある列で runFn が例外を投げても(reject しても)、その場で catch して
 *   outcome化するだけで、残りの列の実行は止めない。
 * - onSettled は各列の処理が終わるたびに(次の列を開始する前に)呼ばれる。
 *   DOM更新のような副作用はここで行う想定(進捗が逐次見える)。onSettled 自体が
 *   投げた例外は上位へ伝播する(呼び出し側のバグとして扱う。runFn 側の失敗とは区別)。
 */
export async function runSequential<T, R>(
  items: readonly T[],
  runFn: (item: T, index: number) => Promise<R>,
  onSettled?: (item: T, index: number, outcome: SequentialOutcome<R>) => void | Promise<void>,
): Promise<SequentialOutcome<R>[]> {
  const outcomes: SequentialOutcome<R>[] = [];
  for (let i = 0; i < items.length; i++) {
    const item = items[i];
    let outcome: SequentialOutcome<R>;
    try {
      const value = await runFn(item, i);
      outcome = { ok: true, value };
    } catch (error) {
      outcome = { ok: false, error };
    }
    outcomes.push(outcome);
    if (onSettled) await onSettled(item, i, outcome);
  }
  return outcomes;
}
