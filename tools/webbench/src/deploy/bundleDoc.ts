// 配布バンドル(Track K、griffith_logic/deploy/bundle.py)のTS側の型と
// バージョン検査。Wave1(bundleProgram.ts)は programs 節1つを扱ったが、
// こちらはバンドル全体(bundle_format/bundle_version/programs/preproc_desc + 既存の
// ModelDoc/ConvModelDoc相当のキー)を1つの型にまとめ、ロード時のバージョン
// ポリシーを検査する。
//
// bundle.py::validate_bundle(~116-153行)の互換ポリシーをミラーする
// (ONNXのopset運用に倣ったもの):
//   - bundle_format不一致 → 拒否(意味論が変わったとみなす)
//   - bundle_version の MAJOR 不一致 → 拒否
//   - MINOR がこのリーダより新しい → 警告のみ(追加は後方互換として扱う)
// さらに `griffith deploy codegen-web --slim` が書き出す「素のモデルJSON」
// (bundle_format を持たない)も受け付ける: この場合はバンドルではない旨を
// warningsに積んで doc をそのまま返す(programs/preproc_descが無い前提で
// 動く、resolveConvProgram が bundle未指定=再構築へ落ちるのと同じ扱い)。
import type {
  DetReadout, ImagePreproc, ModelDoc, Preproc, Readout,
} from '../types';
import type { BundleProgramDict } from '../engines/bundleProgram';

/** griffith_logic/deploy/preproc_desc.py::build_preproc_descriptors の出力
    (整数化済みの前処理記述子)。Webランタイム(js/wasm)自体は使わない
    (C/CUDA/Metalランタイム向けのデータ)が、バンドルの完全な型として持つ。 */
export interface PreprocDesc {
  desc: number[];
  qval: number[];
  channels: number;
  image_channels: number;
  height: number;
  width: number;
  stride: number;
  counts: number[];
}

/** 配布バンドル。ModelDoc(dense)/ConvModelDoc(画像)のどちらの
    preproc/readoutも受けるよう合併型にする(実際の判別は kind で行う。
    lib.ts::loadBundle 参照)。`about` / `demo_samples` はモデル固有かつ
    ModelDoc自体にも無い任意キーなので(segdemo.ts/detdemo.tsの既存流儀と
    同じ)、必要な側(demo_main.ts等)が個別にキャストしてアクセスする。 */
export type BundleDoc = Omit<ModelDoc, 'preproc' | 'readout'> & {
  preproc: Preproc | ImagePreproc;
  readout?: Readout | DetReadout;
  bundle_format: string;
  bundle_version: string;
  programs?: Record<string, BundleProgramDict> | null;
  preproc_desc?: PreprocDesc | null;
};

export const BUNDLE_FORMAT = 'magicdelta-bundle';
/** このリーダが対応する bundle_version(MAJOR.MINOR)。
    griffith_logic/deploy/bundle.py::BUNDLE_VERSION と同じ値を保つこと。 */
export const BUNDLE_VERSION = '1.0';

function versionPair(text: string): [number, number] {
  const parts = String(text).split('.');
  const major = Number(parts[0]);
  const minor = parts.length > 1 ? Number(parts[1]) : 0;
  if (!Number.isFinite(major) || !Number.isFinite(minor)) {
    throw new Error(`checkBundleDoc: バージョン文字列が不正: ${text}`);
  }
  return [major, minor];
}

/** モデルJSON(配布バンドル、または --slim の素のモデルdoc)を検査して
    BundleDoc として返す。
    - bundle_format が無い: バンドルではない(警告のみ、docはそのまま通す)
    - bundle_format が不正 / bundle_version の MAJOR 不一致: 例外
    - bundle_version の MINOR がこのリーダより新しい: 警告 */
export function checkBundleDoc(json: unknown): { doc: BundleDoc; warnings: string[] } {
  if (json === null || typeof json !== 'object' || Array.isArray(json)) {
    throw new Error('checkBundleDoc: モデルJSONがオブジェクトでない');
  }
  const d = json as Record<string, unknown>;
  const warnings: string[] = [];

  if (!('bundle_format' in d)) {
    warnings.push(
      'bundle_formatが無い(配布バンドルではなく素のモデルJSON、`griffith deploy codegen-web '
      + '--slim` 相当): programsが無いのでconv2dはブラウザ側でbuildConvProgramを再構築する',
    );
    return { doc: d as unknown as BundleDoc, warnings };
  }

  for (const key of ['bundle_format', 'bundle_version', 'ir'] as const) {
    if (!(key in d)) throw new Error(`checkBundleDoc: バンドルに必須キーがない: ${key}`);
  }
  if (d.bundle_format !== BUNDLE_FORMAT) {
    throw new Error(
      `checkBundleDoc: bundle_formatが不正: ${String(d.bundle_format)} (期待 ${BUNDLE_FORMAT})`,
    );
  }
  const [wantMajor, wantMinor] = versionPair(BUNDLE_VERSION);
  const [gotMajor, gotMinor] = versionPair(String(d.bundle_version));
  if (gotMajor !== wantMajor) {
    throw new Error(
      `checkBundleDoc: bundle_version ${String(d.bundle_version)} は非対応`
      + `(このリーダは ${wantMajor}.x のみ)`,
    );
  }
  if (gotMinor > wantMinor) {
    warnings.push(
      `bundle_version ${String(d.bundle_version)} はこのリーダ(${BUNDLE_VERSION})より新しい: `
      + '未知のキーは無視される',
    );
  }
  return { doc: d as unknown as BundleDoc, warnings };
}
