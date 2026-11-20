// i18n文言辞書(英語対応の足場)。外部ライブラリを足さない軽量機構。
// キー→{ ja, en } の単一情報源。新しい文言はこの表へ追記するだけで切替対象になる。
// 全訳は後続タスクで拡張する前提のため、ここは主要導線(トップタブ・AppHeaderの主要ボタン
// ・プレースホルダ)のみを収録した足場である。

export type Language = 'ja' | 'en';

/** 対応言語の並び(トグルUI・永続化バリデーションの単一情報源) */
export const LANGUAGES: readonly Language[] = ['ja', 'en'];

/** 既定言語。本プロジェクトは日本語UI前提(NNCの2バイト非対応制約は踏襲しない) */
export const DEFAULT_LANGUAGE: Language = 'ja';

/**
 * 文言辞書。値は各言語の文字列。`as const` によりキーはリテラル型として抽出され、
 * MessageKey のunion型を導出する(型安全: 未定義キーはコンパイルで弾かれる)。
 * トップタブのラベル(label)はNNC踏襲で日英ともに英大文字を維持し、
 * ツールチップ(hint)を各言語へローカライズする。
 * enは任意(足場のため未訳キーはjaへフォールバックする。全訳は後続タスク)。
 */
export const messages = {
  // トップタブのラベル(NNC同型・英大文字を日英で共有)
  'toptab.edit.label': { ja: 'EDIT', en: 'EDIT' },
  'toptab.training.label': { ja: 'TRAINING', en: 'TRAINING' },
  'toptab.evaluation.label': { ja: 'EVALUATION', en: 'EVALUATION' },
  'toptab.dataset.label': { ja: 'DATASET', en: 'DATASET' },
  'toptab.config.label': { ja: 'CONFIG', en: 'CONFIG' },
  // トップタブのツールチップ(ローカライズ対象)
  'toptab.edit.hint': { ja: '構造編集', en: 'Structure editing' },
  'toptab.training.hint': { ja: '学習監視', en: 'Training monitor' },
  'toptab.evaluation.hint': { ja: '評価', en: 'Evaluation' },
  'toptab.dataset.hint': { ja: 'データ管理', en: 'Dataset management' },
  'toptab.config.hint': { ja: 'プロジェクト設定', en: 'Project settings' },
  // AppHeaderの主要ボタン/導線
  'header.model': { ja: 'モデル', en: 'Model' },
  'header.build': { ja: 'build', en: 'build' },
  'header.build.hintNeedsBuild': {
    ja: '構造編集があります。訓練前にbuildが必要です',
    en: 'Structure edited. build is required before training',
  },
  'header.build.hintRebuild': { ja: 'モデルを再構築', en: 'Rebuild the model' },
  'header.trainStart': { ja: '学習開始', en: 'Start training' },
  'header.trainStop': { ja: '停止', en: 'Stop' },
  'header.undo': { ja: '元に戻す', en: 'Undo' },
  'header.redo': { ja: 'やり直す', en: 'Redo' },
  'header.togglePalette': { ja: 'パレット表示切替', en: 'Toggle palette' },
  'header.toggleBottom': { ja: '下部ドック表示切替', en: 'Toggle bottom dock' },
  'header.toggleInspector': { ja: 'インスペクタ表示切替', en: 'Toggle inspector' },
  'header.language': { ja: '表示言語', en: 'Display language' },
  // プレースホルダタブ(未実装タブの表示)
  'placeholder.plannedBadge': { ja: '実装予定', en: 'Planned' },
  // 未訳キーのフォールバック実証用(enを持たない=jaへフォールバックする足場の見本)。
  // 全訳完了までの間、辞書へキーだけ足して段階的に翻訳する運用を示す。
  'demo.untranslated': { ja: '未翻訳サンプル' },
} as const;

/** 辞書エントリの型: jaは必須、enは任意(未訳キーはjaへフォールバック) */
type MessageEntry = { ja: string; en?: string };

/** 辞書キーのunion型(型安全: t()の引数はこのキーに限定される) */
export type MessageKey = keyof typeof messages;

// ja必須の共通形(MessageEntry)として型付けした参照ビュー。
// enが無いエントリも安全に entry[language] で引けるようにする(union индекс回避)。
const DICT = messages as Record<MessageKey, MessageEntry>;

/**
 * 純関数の翻訳器。指定言語の文言を返し、未定義キー・未訳(en欠落)は既定言語(ja)へフォールバックする。
 * 既定言語にも無い場合はキー文字列そのものを返す(表示が空にならない安全策)。
 */
export function translate(key: MessageKey, language: Language): string {
  const entry = DICT[key];
  if (!entry) return key;
  return entry[language] ?? entry[DEFAULT_LANGUAGE] ?? key;
}
