// 軽量i18n(エディタ tools/editor/frontend/src/i18n/messages.ts と同方式:
// 外部ライブラリを足さないフラット辞書。en 省略時は ja へフォールバック)。
export type Language = 'ja' | 'en';
export const LANGUAGES: Language[] = ['ja', 'en'];
export const DEFAULT_LANGUAGE: Language = 'ja';

export const messages = {
  'header.brand': { ja: 'Griffith Viewer', en: 'Griffith Viewer' },
  'header.connected': { ja: '接続', en: 'Connected' },
  'header.disconnected': { ja: '切断', en: 'Disconnected' },
  'header.system': { ja: 'システム', en: 'System' },
  'header.theme': { ja: 'テーマ切替', en: 'Toggle theme' },

  'controls.step': { ja: 'STEP 1clk', en: 'STEP 1clk' },
  'controls.step10': { ja: '×10', en: '×10' },
  'controls.run': { ja: 'RUN', en: 'RUN' },
  'controls.stop': { ja: 'STOP', en: 'STOP' },
  'controls.reset': { ja: 'RESET', en: 'RESET' },
  'controls.autostep': {
    ja: '入力変更で自動 1clk(クリック=1クロック)',
    en: 'Auto-step 1clk on input change',
  },
  'input.title': { ja: '入力 x', en: 'Input x' },
  'input.rom': {
    ja: 'システムモード: 入力は ROM から供給されます。STEP でクロックを進めると画像が流れます。',
    en: 'System mode: inputs come from ROM. STEP the clock to stream images.',
  },
  'input.clear': { ja: 'クリア', en: 'Clear' },
  'input.painthint': {
    ja: '左ドラッグ=描く / 右ドラッグ=消す',
    en: 'Left drag = draw / right drag = erase',
  },

  'probe.pin': { ja: '画面に固定', en: 'Pin to screen' },
  'probe.unpin': { ja: '画面固定を解除', en: 'Unpin' },
  'probe.delete': { ja: 'プローブを削除', en: 'Delete probe' },
  'probe.cols': { ja: '幅', en: 'W' },
  'probe.depth': { ja: '段', en: 'Q' },
  'probe.until': { ja: '変化まで', en: 'until change' },
  'probe.until.hint': {
    ja: 'この値が変わるまでクロックを進める(上限1000)',
    en: 'Step clocks until this value changes (max 1000)',
  },
  'probe.expand': { ja: '展開', en: 'Expand' },
  'probe.collapse': { ja: '畳む', en: 'Collapse' },

  'canvas.hint': {
    ja: 'ノード右の ● から空きへドラッグ = プローブ追加(値表示部)',
    en: 'Drag from a node handle to empty canvas to add a probe',
  },
  'legend.logic': { ja: '論理', en: 'logic' },
  'legend.on': { ja: '点灯=1', en: 'on=1' },
  'legend.probe': { ja: 'プローブ', en: 'probe' },
  'legend.width': { ja: '太さ=ビット幅', en: 'width = bits' },

  'project.title': { ja: 'プロジェクト', en: 'Project' },
  'project.open': { ja: '開く…', en: 'Open…' },
  'project.opening': { ja: '切替中…', en: 'Opening…' },
  'project.opening.hint': {
    ja: 'ビルド中…(現在のプロジェクトは動作継続)',
    en: 'Building… (current project keeps running)',
  },
  'project.dialog.title': { ja: 'プロジェクトを開く', en: 'Open project' },
  'project.dialog.desc': {
    ja: 'イグザンプルから選ぶか、IR JSON のパスを指定します。',
    en: 'Pick an example or enter a path to an IR JSON.',
  },
  'project.tab.examples': { ja: 'イグザンプル', en: 'Examples' },
  'project.tab.file': { ja: 'ファイル', en: 'File' },
  'project.path.placeholder': {
    ja: 'IR のパス(例 D:\\...\\model.ir.json)',
    en: 'Path to IR (e.g. D:\\...\\model.ir.json)',
  },
  'project.recent': { ja: '最近使ったプロジェクト', en: 'Recent projects' },
  'project.notready': { ja: '未ビルド', en: 'not built' },
  'toast.project.opened': {
    ja: 'プロジェクトを切り替えました',
    en: 'Project switched',
  },

  'board.title': { ja: 'ボード', en: 'Board' },
  'board.detect': { ja: '認識', en: 'Detect' },
  'board.detecting': { ja: '認識中…', en: 'Detecting…' },
  'board.select.placeholder': { ja: 'ボードを選択', en: 'Select board' },
  'board.unverified': { ja: '実機未検証', en: 'unverified' },
  'board.fits.title': { ja: '搭載可能性', en: 'Fit' },
  'board.fits.yes': { ja: '有り', en: 'yes' },
  'board.fits.partial': { ja: 'やや有り(overlay)', en: 'partial (overlay)' },
  'board.fits.no': { ja: '無し', en: 'no' },
  'board.advice.unavailable': {
    ja: '搭載可能性は未計算(モデル読込後に更新)',
    en: 'Fit estimate unavailable (updates after model load)',
  },
  'board.detect.none': {
    ja: '候補なし。ボードの電源と USB 接続を確認(詳細は doctor)',
    en: 'No candidates. Check board power and USB (see doctor)',
  },
  'board.candidates.title': { ja: 'ボード候補の確認', en: 'Confirm board' },
  'board.candidates.desc': {
    ja: '複数の候補が見つかりました。基板のシルク印字(基板上の印刷文字)を見て選んでください。',
    en: 'Multiple candidates found. Check the silkscreen printing on the board and pick one.',
  },

  'toast.saved.error': { ja: 'レイアウト保存に失敗', en: 'Failed to save layout' },
  'toast.until.changed': { ja: '値が変化', en: 'value changed' },
  'toast.until.limit': { ja: '上限で打ち切り', en: 'hit step limit' },
  'toast.disconnected': {
    ja: 'シミュレーションとの接続が切れました',
    en: 'Simulation connection lost',
  },
} as const;

export type MessageKey = keyof typeof messages;

export function translate(key: MessageKey, language: Language): string {
  const entry = messages[key] as { ja: string; en?: string };
  return (language === 'en' ? entry.en : undefined) ?? entry.ja ?? key;
}
