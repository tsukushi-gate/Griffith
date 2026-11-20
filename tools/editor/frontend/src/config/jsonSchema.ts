// JSON Schema(pydantic v2 model_json_schema形式)の型定義+純粋ヘルパー関数。
// SchemaForm.tsx(コンポーネント)からロジックを分離し、
// react-refresh/only-export-components(fast refresh制約)に適合させる。

// pydantic model_json_schema()が返すJSON Schemaの必要最小限の型
// (プロジェクト内では $defs を持つルートスキーマ or その部分木として扱う)。
export interface JsonSchema {
  type?: string;
  title?: string;
  description?: string;
  enum?: (string | number)[];
  properties?: Record<string, JsonSchema>;
  additionalProperties?: JsonSchema | boolean;
  $ref?: string;
  anyOf?: JsonSchema[];
  minimum?: number;
  maximum?: number;
  exclusiveMinimum?: number;
  exclusiveMaximum?: number;
  default?: unknown;
  $defs?: Record<string, JsonSchema>;
}

/** '#/$defs/XXX' 形式の$refをルートスキーマから解決する。 */
export function resolveRef(schema: JsonSchema, root: JsonSchema): JsonSchema {
  if (!schema.$ref) return schema;
  const m = /^#\/\$defs\/(.+)$/.exec(schema.$ref);
  const def = m ? root.$defs?.[m[1]] : undefined;
  return def ?? schema;
}

/** anyOf([{type:'string'},{type:'null'}]等)からnull以外の実体スキーマを取り出す(nullableサポート)。 */
export function unwrapNullable(schema: JsonSchema): { inner: JsonSchema; nullable: boolean } {
  if (!schema.anyOf) return { inner: schema, nullable: false };
  const nonNull = schema.anyOf.filter((s) => s.type !== 'null');
  const nullable = nonNull.length !== schema.anyOf.length;
  return { inner: nonNull[0] ?? schema, nullable };
}

/**
 * スキーマの default値から初期dictを機械生成する(新規エントリ初期値、
 * およびGlobal Config初回編集時の初期値に使う)。
 */
export function buildSchemaDefaults(schema: JsonSchema, root: JsonSchema): Record<string, unknown> {
  const resolved = resolveRef(schema, root);
  const out: Record<string, unknown> = {};
  for (const [key, rawFieldSchema] of Object.entries(resolved.properties ?? {})) {
    const fieldSchema = resolveRef(rawFieldSchema, root);
    const { inner } = unwrapNullable(fieldSchema);
    const innerResolved = resolveRef(inner, root);
    // 明示defaultを最優先する。nullable object(例: distill=anyOf[DistillConfig,null]
    // default null)でオブジェクト再帰を先にすると、required欠落の不正オブジェクト
    // ({teacher: null, ...})を構築してしまい保存PUTが常時422になる(mainマージで顕在化)。
    if (Object.hasOwn(fieldSchema, 'default')) {
      out[key] = fieldSchema.default;
    } else if (Object.hasOwn(inner, 'default')) {
      out[key] = inner.default;
    } else if (innerResolved.type === 'object' && innerResolved.properties) {
      out[key] = buildSchemaDefaults(innerResolved, root);
    } else if (innerResolved.type === 'object' && innerResolved.additionalProperties) {
      // default_factory=dict の自由形式dict(例: expected_metrics/data.params)は
      // pydantic JSON Schemaに"default"キーが出ないため、nullでなく空dictへ倒す
      // (null のままだとTrainConfig側のdict型検証(extra='forbid')が422になる)。
      out[key] = {};
    } else if (innerResolved.type === 'array') {
      // default_factory=list のリスト(例: run.wandb_tags)も同様に"default"が出ない。
      // null だと list 型検証で保存PUTが422になり、CONFIGタブの入力が既定値へ戻る
      // (5タブ横断E2Eの Epochs 30 → 3)。空リストへ倒す。
      out[key] = [];
    } else {
      out[key] = null;
    }
  }
  return out;
}

/** このスキーマ片が「サポート外→JSONテキストエリアfallback」対象かどうか。 */
export function isUnsupportedSchema(schema: JsonSchema): boolean {
  if (schema.type === 'array') return true;
  // additionalProperties:true の自由形式object(例: expected_metrics)はfallback
  if (schema.type === 'object' && !schema.properties && schema.additionalProperties) {
    return true;
  }
  return false;
}
