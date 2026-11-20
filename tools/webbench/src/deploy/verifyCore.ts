// `md deploy verify-web` の出荷ゲート実装(node CLI本体は
// verify_main.ts、ここは環境非依存のコア)。frozen contract:
//   cases.json = {"mode": "bits"|"image", "cases": [{bits_b64?/image_b64?,
//                 expected: number[]}]}
//   bits_b64  = 入力幅(in_width)ビットの packed LSB-first バイト列(base64)
//               (conv.test.ts::b64ToBits / conv_golden.json と同一契約)
//   image_b64 = 生 uint8 CHW(image_channels×H×W)の base64
// 出力は整数厳密一致のみを ok とする(小数誤差の概念が無い: DLGNは整数演算)。
//
// strictPrograms は常に true(検証経路はフォールバック禁止。lib.ts::loadBundle
// 参照)。opts で渡された strictPrograms は上書きする — 検証の入口を1つに
// 保ち、「うっかりstrict指定を忘れて壊れたprogramsを見逃す」を構造的に防ぐ。
import { loadBundle } from './lib';
import type { LoadBundleOptions } from './lib';
import type { ImagePreprocDoc } from '../convpreproc';
import { encodeImageAuto } from '../convpreproc';

export interface VerifyCase {
  bits_b64?: string;
  image_b64?: string;
  expected: number[];
}

export interface VerifyCaseFile {
  mode: 'bits' | 'image';
  cases: VerifyCase[];
}

export interface VerifyReport {
  ok: boolean;
  engine: string;
  mode: string;
  outputs: number[][];
  maxAbsDiff: number;
  error?: string;
}

/** base64 → packed LSB-first ビット列(長さnBits)。
    conv.test.ts::b64ToBits / segdemo.ts::maskFromB64 と同じ atob 直読み流儀。 */
function decodeBitsB64(b64: string, nBits: number): Uint8Array {
  const raw = atob(b64);
  const bits = new Uint8Array(nBits);
  for (let i = 0; i < nBits; i++) {
    bits[i] = (raw.charCodeAt(i >> 3) >> (i & 7)) & 1;
  }
  return bits;
}

/** base64 → 生バイト列(そのまま)。 */
function decodeBytesB64(b64: string): Uint8Array {
  const raw = atob(b64);
  const out = new Uint8Array(raw.length);
  for (let i = 0; i < raw.length; i++) out[i] = raw.charCodeAt(i);
  return out;
}

/** CHW uint8([0,255]) → float([0,1])。convpreproc.ts::rgbaToNchw の
    `/255` 正規化と同じスケール(mean/stdはこの空間で較正されている)。 */
function chwBytesToFloat(bytes: Uint8Array): Float32Array {
  const out = new Float32Array(bytes.length);
  for (let i = 0; i < bytes.length; i++) out[i] = bytes[i] / 255;
  return out;
}

/** 1ケースを実行して整数スコアを返す(構造不正は例外、呼び出し側で
    Runtime error として扱う: verify_main.ts の catch-all)。 */
function runCase(
  bundle: Awaited<ReturnType<typeof loadBundle>>,
  mode: 'bits' | 'image',
  c: VerifyCase,
  inWidth: number,
  preprocDoc: ImagePreprocDoc,
  caseIndex: number,
): Int32Array {
  if (mode === 'bits') {
    if (c.bits_b64 === undefined) {
      throw new Error(`case[${caseIndex}]: mode=bits なのに bits_b64 が無い`);
    }
    const bits = decodeBitsB64(c.bits_b64, inWidth);
    return bundle.scores(bits, 1).data;
  }
  if (c.image_b64 === undefined) {
    throw new Error(`case[${caseIndex}]: mode=image なのに image_b64 が無い`);
  }
  const bytes = decodeBytesB64(c.image_b64);
  const expectedLen = preprocDoc.mean.length * preprocDoc.height * preprocDoc.width;
  if (bytes.length !== expectedLen) {
    throw new Error(
      `case[${caseIndex}]: image_b64のバイト数 ${bytes.length} が `
      + `image_channels×H×W=${expectedLen} と違う`,
    );
  }
  const nchw = chwBytesToFloat(bytes);
  const bits = encodeImageAuto(nchw, preprocDoc);
  return bundle.scores(bits, 1).data;
}

/** frozen contract 本体: バンドル+cases.json → 全ケース整数厳密一致検証。
    strictPrograms は常に true(programsが壊れていればここで例外)。 */
export async function verifyBundleCases(
  doc: unknown,
  file: VerifyCaseFile,
  opts: LoadBundleOptions = {},
): Promise<VerifyReport> {
  const bundle = await loadBundle(doc, { ...opts, strictPrograms: true });
  const inWidth = bundle.runtime.inWidth;
  const preprocDoc = bundle.doc.preproc as unknown as ImagePreprocDoc;

  const outputs: number[][] = [];
  let maxAbsDiff = 0;
  let firstError: string | null = null;

  for (let i = 0; i < file.cases.length; i++) {
    const c = file.cases[i];
    const data = runCase(bundle, file.mode, c, inWidth, preprocDoc, i);
    const out = Array.from(data);
    outputs.push(out);
    if (out.length !== c.expected.length) {
      firstError ??= `case[${i}]: 出力長 ${out.length} が expected長 ${c.expected.length} と違う`;
      continue;
    }
    for (let j = 0; j < out.length; j++) {
      const diff = Math.abs(out[j] - c.expected[j]);
      if (diff > maxAbsDiff) maxAbsDiff = diff;
      if (out[j] !== c.expected[j] && firstError === null) {
        firstError = `case[${i}] output[${j}]: expected=${c.expected[j]} actual=${out[j]}`;
      }
    }
  }

  // 同じケースを1回のバッチ(batch=ケース数)でも実行し、枚ごとの結果が
  // 1枚ずつの実行と同じ期待値に一致するかを見る。dense の lut_flat は 3 枚以上で
  // バッチ軸パック経路(lutFlat.ts / wasmLutFlat.ts)へ切り替わるので、1枚ずつの
  // 実行だけではその経路が出荷ゲートを素通りしてしまう。
  if (firstError === null && file.mode === 'bits' && file.cases.length >= 2) {
    const n = file.cases.length;
    const all = new Uint8Array(n * inWidth);
    for (let i = 0; i < n; i++) all.set(decodeBitsB64(file.cases[i].bits_b64!, inWidth), i * inWidth);
    const { data, width } = bundle.scores(all, n);
    for (let i = 0; i < n && firstError === null; i++) {
      const exp = file.cases[i].expected;
      if (width !== exp.length) {
        firstError = `batch=${n} case[${i}]: 出力幅 ${width} が expected長 ${exp.length} と違う`;
        break;
      }
      for (let j = 0; j < width; j++) {
        const v = data[i * width + j];
        const diff = Math.abs(v - exp[j]);
        if (diff > maxAbsDiff) maxAbsDiff = diff;
        if (v !== exp[j]) {
          firstError = `batch=${n} case[${i}] output[${j}]: expected=${exp[j]} actual=${v}`;
          break;
        }
      }
    }
  }

  if (firstError !== null) {
    return {
      ok: false, engine: bundle.engine, mode: file.mode, outputs, maxAbsDiff, error: firstError,
    };
  }
  return {
    ok: true, engine: bundle.engine, mode: file.mode, outputs, maxAbsDiff,
  };
}
