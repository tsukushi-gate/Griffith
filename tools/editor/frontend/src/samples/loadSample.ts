// サンプルモデルの共通ローダ。CanvasEmptyStateとModelMenuで共用する。
// structuredCloneでモジュール定数のIRを守る(graphStore側の座標書き込み等による汚染防止)。

import { toast } from 'sonner';
import { importPythonModel, putCurrentIR } from '../api/rest';
import { useGraphStore } from '../store/graphStore';
import { useTrainingStore } from '../store/trainingStore';
import { useUIStore } from '../store/uiStore';
import type { SampleDef } from './sampleIr';

/** XORチュートリアルを起動するサンプルkey(オンボーディングのシナリオ1本) */
export const XOR_TUTORIAL_SAMPLE_KEY = 'xor';

/** サンプルIRをサーバへPUTしてエディタに読み込む。失敗時はトースト表示してfalse。 */
export async function loadSampleModel(sample: SampleDef): Promise<boolean> {
  try {
    let ir;
    let needsBuild = true;
    if (sample.source === 'python') {
      if (!sample.pythonImport) throw new Error('Python import設定がありません');
      const res = await importPythonModel(sample.pythonImport.path, {
        symbol: sample.pythonImport.symbol,
        mode: sample.pythonImport.mode,
      });
      ir = res.ir;
      needsBuild = false;
    } else {
      if (!sample.ir) throw new Error('IRサンプルがありません');
      ir = structuredClone(sample.ir);
      await putCurrentIR(ir);
    }
    useGraphStore.getState().setIR(ir, { needsBuild });
    const training = useTrainingStore.getState();
    training.resetAll();
    if (sample.trainingPreset) {
      training.applyTrainingPreset(sample.trainingPreset, sample.label);
    }
    // XORサンプルを開いたらチュートリアルガイドを表示する(オンボーディング)
    if (sample.key === XOR_TUTORIAL_SAMPLE_KEY) {
      useUIStore.getState().setXorGuideOpen(true);
    }
    toast.success(
      sample.source === 'python'
        ? sample.trainable === false
          ? `${sample.label} をPythonモデルから読み込みました(構造表示専用です)`
          : `${sample.label} をPythonモデルから読み込みました(そのまま学習できます)`
        : sample.trainable
        ? `${sample.label} を読み込みました(buildして学習を開始できます)`
        : sample.variant === 'reference'
          ? `${sample.label} を読み込みました(参照回路として詳細展開されます)`
          : `${sample.label} を読み込みました(小規模なので自動で詳細展開されます)`,
    );
    return true;
  } catch (e) {
    toast.error(e instanceof Error ? e.message : String(e));
    return false;
  }
}
