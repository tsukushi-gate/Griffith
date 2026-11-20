// modelMeta.ts の単体テスト。**実際に配信しているモデルJSON**
// (public/models/*.json)に対して掛ける。fixtureを別に持つと、モデル差し替え時に
// fixtureだけ古くなって「陳腐化検出器が陳腐化する」ため。
import { existsSync, readFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';

import {
  MISSING,
  deriveModelMeta,
  fmtBytes,
  fmtCompact,
  fmtInt,
  formatRecipe,
  kFromModelName,
  metaFields,
  pickAccuracy,
} from './modelMeta';

const here = dirname(fileURLToPath(import.meta.url));
const MODELS = resolve(here, '..', 'public', 'models');

/** 各デモページが実際に配信しているDLGNモデル(定数は *_main.ts と一致させる)。 */
const SERVED = [
  { file: 'det_multimnist_lut_k256.json', page: 'det.html', metric: 'mAP', det: true },
  { file: 'det_multimnist_mono_lut_k256.json', page: 'detmono.html', metric: 'mAP', det: true },
  { file: 'det_fewclass_lut_k64.json', page: 'detreal.html', metric: 'mAP', det: true },
  { file: 'det_face_lut_k256.json', page: 'detface.html', metric: 'mAP', det: true },
  { file: 'seg_convdlgn_k128.json', page: 'seg.html', metric: 'mIoU', det: false },
] as const;

function load(file: string): unknown {
  return JSON.parse(readFileSync(resolve(MODELS, file), 'utf8'));
}

const haveModels = SERVED.every((m) => existsSync(resolve(MODELS, m.file)));

describe('整形ヘルパ', () => {
  it('欠損は MISSING を返す(嘘の既定値を出さない)', () => {
    expect(fmtInt(null)).toBe(MISSING);
    expect(fmtBytes(null)).toBe(MISSING);
    expect(fmtCompact(null)).toBe(MISSING);
  });

  it('桁区切りはロケール非依存', () => {
    expect(fmtInt(13104)).toBe('13,104');
    expect(fmtInt(53608448)).toBe('53,608,448');
    expect(fmtInt(999)).toBe('999');
  });

  it('短縮表記', () => {
    expect(fmtCompact(30818304)).toBe('30.8M');
    expect(fmtCompact(13104)).toBe('13.1k');
  });

  it('バイト表記', () => {
    expect(fmtBytes(36265)).toBe('35.4 KB');
    expect(fmtBytes(6363347)).toBe('6.07 MB');
  });
});

describe('kFromModelName', () => {
  it('モデル名から k を読む', () => {
    expect(kFromModelName('det_multimnist_lut_k256')).toBe(256);
    expect(kFromModelName('seg_convdlgn_k32')).toBe(32);
    // 末尾にサフィックスが付いていても読める(採用モデルの命名)
    expect(kFromModelName('det_multimnist_mono_lut_k256_w18')).toBe(256);
  });
  it('読めなければ null(推測しない)', () => {
    expect(kFromModelName('har_dlgn_dwn')).toBeNull();
    expect(kFromModelName(null)).toBeNull();
  });
});

describe('pickAccuracy', () => {
  it('指標キーから指標名を決める', () => {
    expect(pickAccuracy({ best_hard_map: 0.5 })).toEqual({
      key: 'best_hard_map', label: 'mAP', value: 0.5,
    });
    expect(pickAccuracy({ best_hard_miou: 0.5 })?.label).toBe('mIoU');
  });
  it('精度でないキー(ゲート数など)を精度にしない', () => {
    expect(pickAccuracy({ num_gates_weight_shared: 13104 })).toBeNull();
    expect(pickAccuracy({})).toBeNull();
    expect(pickAccuracy(undefined)).toBeNull();
  });
});

describe('deriveModelMeta(欠損耐性)', () => {
  it('空docでも例外を投げず全て null', () => {
    const m = deriveModelMeta({});
    expect(m.accuracy).toBeNull();
    expect(m.gatesUnrolled).toBeNull();
    expect(m.gatesProgram).toBeNull();
    expect(m.bundleVersion).toBeNull();
    expect(m.inputChannels).toBeNull();
    expect(m.scoreThreshold).toBeNull();
    expect(m.labels).toEqual([]);
    const f = metaFields(m);
    for (const v of Object.values(f)) expect(v).toBe(MISSING);
  });

  it('bundle_version と programs があれば bundleVersion/gatesProgram を導出する', () => {
    const m = deriveModelMeta({
      bundle_version: '1.0',
      programs: {
        n1: { stats: { gates_after: 120 } },
        n2: { stats: { gates_after: 30 } },
      },
    });
    expect(m.bundleVersion).toBe('1.0');
    expect(m.gatesProgram).toBe(150);
  });

  it('programsが空dict(dense、conv2d無し)なら gatesProgram=0(未計測のnullとは区別)', () => {
    const m = deriveModelMeta({ bundle_version: '1.0', programs: {} });
    expect(m.gatesProgram).toBe(0);
  });

  it('resources.bundle_json_bytes/bundle_gzip_bytesがあれば json_bytes/gzip_bytesより優先する', () => {
    const m = deriveModelMeta({
      resources: {
        json_bytes: 1000, gzip_bytes: 500,
        bundle_json_bytes: 1600, bundle_gzip_bytes: 900,
      },
    });
    expect(m.jsonBytes).toBe(1600);
    expect(m.gzipBytes).toBe(900);
  });

  it('bundle_*_bytes が無ければ従来の json_bytes/gzip_bytes へフォールバックする', () => {
    const m = deriveModelMeta({ resources: { json_bytes: 1000, gzip_bytes: 500 } });
    expect(m.jsonBytes).toBe(1000);
    expect(m.gzipBytes).toBe(500);
  });

  it('about が無ければ null(=ページ側の静的散文へフォールバック)', () => {
    expect(deriveModelMeta({}).about).toBeNull();
    expect(formatRecipe(null)).toBeNull();
  });

  it('about があれば recipe を1行に整形し、k は about を優先する', () => {
    const m = deriveModelMeta({
      ir: { metadata: { name: 'x_k64' } },
      about: { dataset_label: 'WIDER FACE', recipe: { k: 256, wiring: 'learnable' } },
    });
    expect(m.k).toBe(256); // 名前の k64 ではなく about.recipe.k
    expect(formatRecipe(m.about)).toBe('k=256 / 配線=learnable');
    expect(metaFields(m).datasetLabel).toBe('WIDER FACE');
  });
});

describe.skipIf(!haveModels)('配信中モデルJSONからの導出', () => {
  for (const s of SERVED) {
    describe(s.file, () => {
      const doc = load(s.file);
      const meta = deriveModelMeta(doc);

      it('精度が指標名つきで取れる', () => {
        expect(meta.accuracy).not.toBeNull();
        expect(meta.accuracy!.label).toBe(s.metric);
        expect(meta.accuracy!.value).toBeGreaterThan(0);
        expect(meta.accuracy!.value).toBeLessThanOrEqual(1);
      });

      it('ゲート数(重み共有 / 空間展開後)が両方取れる', () => {
        expect(meta.gatesWeightShared).toBeGreaterThan(0);
        expect(meta.gatesUnrolled).toBeGreaterThan(0);
        // 空間展開後のほうが必ず多い
        expect(meta.gatesUnrolled!).toBeGreaterThan(meta.gatesWeightShared!);
      });

      it('入力仕様(解像度・preproc種別・閾値数・総ch数)が取れる', () => {
        expect(meta.inputHeight).toBeGreaterThan(0);
        expect(meta.inputWidth).toBeGreaterThan(0);
        expect(meta.preprocKind).toMatch(/^image_/);
        expect(meta.nThresholds).toBeGreaterThan(0);
        expect(meta.inputChannels).toBeGreaterThan(0);
        // ir.inputs[0].width = ch * H * W が成り立つ(IRが真実源)
        expect(meta.inputBits).toBe(
          meta.inputChannels! * meta.inputHeight! * meta.inputWidth!,
        );
      });

      it('論理深さ・ノード構成・配信サイズが取れる', () => {
        expect(meta.logicDepth).toBeGreaterThan(0);
        expect(meta.nodeCounts).not.toBeNull();
        expect(Object.keys(meta.nodeCounts!).length).toBeGreaterThan(0);
        expect(meta.jsonBytes).toBeGreaterThan(0);
        expect(meta.gzipBytes).toBeGreaterThan(0);
        expect(meta.gzipBytes!).toBeLessThan(meta.jsonBytes!);
      });

      it('k がモデル名から取れる', () => {
        expect(meta.k).not.toBeNull();
      });

      // Web配信は配布バンドルではなく生モデルdocを配る、と実測を経て決めた。
      //
      // バンドルは簡約(ゲート木→命令列)を生成時に焼き込むが、実測すると
      // brotli後のサイズは 生doc 325KB に対しバンドル 614KB、木を剥がしても
      // 404KB で、いずれも生docより大きい。簡約後の命令列は冗長性が削れていて
      // 圧縮が効かない一方、生の木はShannon分解のパディングが規則的でよく縮む。
      // 節約できるのはロード時の buildConvProgram 105ms だけで、全数照合検証
      // 233ms はバンドルでも残る。配信サイズはこの方式の最大の売り(ONNX比28倍)
      // なので、+24〜34%を初回105msと引き換えにはしない。
      //
      // バンドルの本領は実行系にコンパイラを持てない C/CUDA/Metal 側にある。
      // 生docなら gatesProgram は null、gatesReduced はブラウザ側で数え直す。
      it('Web配信は生モデルdoc(バンドルではない)', () => {
        expect(meta.bundleVersion).toBeNull();
        expect(meta.gatesProgram).toBeNull();
        // 実行実体のゲート数は生docからでも導出できる(表示用)。
        expect(meta.gatesReduced).not.toBeNull();
        expect(meta.gatesReduced!).toBeGreaterThan(0);
        expect(meta.gatesReduced!).toBeLessThan(meta.gatesUnrolled!);
      });

      it('クラス数とラベルが整合する', () => {
        expect(meta.nClasses).toBeGreaterThan(0);
        expect(meta.labels.length).toBeGreaterThan(0);
      });

      if (s.det) {
        // 注: readout.threshold は CenterNet decode 側の閾値。デモUIの
        // スコア閾値スライダー(表示用フィルタ、実測較正値)とは別物。
        it('検出モデルは readout の閾値/stride/top_k を持つ', () => {
          expect(meta.outputStride).toBeGreaterThan(0);
          expect(meta.scoreThreshold).not.toBeNull();
          expect(meta.topK).toBeGreaterThan(0);
        });
        it('クラス数は labels と一致する', () => {
          expect(meta.nClasses).toBe(meta.labels.length);
        });
      }

      it('表示文字列に undefined / NaN / null が混入しない', () => {
        for (const [k, v] of Object.entries(metaFields(meta))) {
          expect(v, k).not.toMatch(/undefined|NaN|\bnull\b/);
          expect(v, k).not.toBe('');
        }
      });
    });
  }
});
