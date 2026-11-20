// (a) interpエンジン(WB-4): editorのevalCircuitをそのまま呼ぶ
// per-bitインタプリタ。正しさのアンカー(simulate.py移植の既検証実装)兼
// 低速側ベースライン。1サンプルずつ処理。
import { evalCircuit } from '../shared';
import type { ModelDoc, TestData } from '../types';
import { argmax, encodeThermometer, sampleView } from '../preproc';
import type { Engine } from './types';

export class InterpEngine implements Engine {
  readonly name = 'dlgn-interp';
  private model: ModelDoc;
  private inputId: string;
  private bitsBuf: Uint8Array;
  private floatBuf: Float32Array;

  constructor(model: ModelDoc) {
    this.model = model;
    this.inputId = model.ir.inputs[0].id;
    const w = model.ir.inputs[0].width;
    this.bitsBuf = new Uint8Array(w);
    this.floatBuf = new Float32Array(w);
  }

  /** 1サンプルのクラススコア(evalCircuit hard) */
  scoreOne(data: TestData, i: number): Float32Array {
    encodeThermometer(sampleView(data, i), this.model.preproc, this.bitsBuf);
    for (let j = 0; j < this.bitsBuf.length; j++) {
      this.floatBuf[j] = this.bitsBuf[j];
    }
    const result = evalCircuit(
      this.model.ir,
      { [this.inputId]: this.floatBuf },
      'hard',
    );
    return result.output;
  }

  async predict(data: TestData, from: number, count: number): Promise<Int32Array> {
    const preds = new Int32Array(count);
    for (let i = 0; i < count; i++) {
      preds[i] = argmax(this.scoreOne(data, from + i));
    }
    return preds;
  }

  scores(data: TestData, from: number, count: number): Int32Array[] {
    const out: Int32Array[] = [];
    for (let i = 0; i < count; i++) {
      out.push(Int32Array.from(this.scoreOne(data, from + i)));
    }
    return out;
  }
}
