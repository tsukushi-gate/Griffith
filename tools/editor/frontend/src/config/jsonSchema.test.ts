// jsonSchema.ts純関数の単体テスト(調査で見つかったdefault_factory=dict回帰)。
// TrainConfigの expected_metrics / data.params のような「additionalProperties:true の
// 自由形式dict(default_factory=dict)」は pydantic JSON Schemaに "default" キーが出ない
// ため、buildSchemaDefaultsがnullでなく空dictを生成することを固定する
// (nullのままだとサーバ側TrainConfigのdict型検証で422になり、CONFIGタブの
// Global Config編集が最初の保存で必ず失敗する実害があった)。

import { describe, expect, it } from 'vitest';
import { buildSchemaDefaults, type JsonSchema } from './jsonSchema';

describe('buildSchemaDefaults: 自由形式dict(additionalProperties)のdefault', () => {
  const SCHEMA: JsonSchema = {
    type: 'object',
    properties: {
      // default_factory=dict相当: "default"キーを持たない自由形式object
      expected_metrics: { type: 'object', additionalProperties: { type: 'number' } },
      // additionalProperties: trueパターン(DataConfig.params相当)
      params: { type: 'object', additionalProperties: true },
      // 明示defaultがある通常フィールドは従来どおりdefaultを使う
      epochs: { type: 'integer', default: 3 },
    },
  };

  it('additionalPropertiesを持つ自由形式dictは空dict({})になる(nullにならない)', () => {
    const defaults = buildSchemaDefaults(SCHEMA, SCHEMA);
    expect(defaults.expected_metrics).toEqual({});
    expect(defaults.params).toEqual({});
  });

  it('明示defaultを持つフィールドは従来どおりdefault値になる', () => {
    const defaults = buildSchemaDefaults(SCHEMA, SCHEMA);
    expect(defaults.epochs).toBe(3);
  });
});

describe('buildSchemaDefaults: default_factory=list のリスト(FMC、5タブ横断E2E回帰)', () => {
  // RunConfig.wandb_tags = Field(default_factory=list) 相当: "default"キーを持たない配列
  const SCHEMA: JsonSchema = {
    type: 'object',
    properties: {
      wandb_tags: { type: 'array' },
      // 明示defaultを持つ配列はそのまま
      strides: { type: 'array', default: [1, 2] },
      // nullable配列(anyOf[array, null] + default null)は明示defaultが勝つ
      s_a: { anyOf: [{ type: 'array' }, { type: 'null' }], default: null },
    },
  };

  it('"default"の無い配列は空リスト([])になる(nullだと保存PUTが422で入力が戻る)', () => {
    const defaults = buildSchemaDefaults(SCHEMA, SCHEMA);
    expect(defaults.wandb_tags).toEqual([]);
  });

  it('明示defaultを持つ配列とnullable配列は従来どおり', () => {
    const defaults = buildSchemaDefaults(SCHEMA, SCHEMA);
    expect(defaults.strides).toEqual([1, 2]);
    expect(defaults.s_a).toBeNull();
  });
});

describe('buildSchemaDefaults: nullable objectの明示default優先(mainマージ回帰)', () => {
  it('anyOf[objectRef, null] + default:null はnullになる(不正オブジェクトを構築しない)', () => {
    const root = {
      type: 'object',
      properties: {
        distill: {
          anyOf: [{ $ref: '#/$defs/Distill' }, { type: 'null' }],
          default: null,
        },
      },
      $defs: {
        Distill: {
          type: 'object',
          properties: { teacher: { type: 'string' } },
          required: ['teacher'],
        },
      },
    } as never;
    expect(buildSchemaDefaults(root, root)).toEqual({ distill: null });
  });
});
