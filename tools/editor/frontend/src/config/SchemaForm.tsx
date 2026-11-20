// JSON Schema(pydantic v2 model_json_schema形式)からフォームを自動生成する汎用コンポーネント。
// TrainConfigのような$defs+$ref形式のネストスキーマを再帰的に描画し、
// 型→ウィジェット対応表に従う。未知型(array等)はJSONテキストエリアへfallbackする。
// 型定義+純粋ヘルパー関数は jsonSchema.ts に分離(fast refresh制約への対応)。

import { Fragment } from 'react';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Toggle } from '@/components/ui/toggle';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { JsonFallbackField } from './JsonFallbackField';
import { isUnsupportedSchema, resolveRef, unwrapNullable, type JsonSchema } from './jsonSchema';

export interface SchemaFormProps {
  /** $defsを含むルートスキーマ(ネスト時も常にルートを渡し、$refを解決できるようにする) */
  schema: JsonSchema;
  value: Record<string, unknown>;
  onChange: (next: Record<string, unknown>) => void;
  /** 再帰描画時の見出し表示レベル(トップレベルは省略) */
  depth?: number;
}

export function SchemaForm({ schema, value, onChange, depth = 0 }: SchemaFormProps) {
  // ルートスキーマ自身が$defsの解決基点になる(呼び出し側は常にルートを渡す想定)。
  return (
    <SchemaFormFields schema={schema} root={schema} value={value} onChange={onChange} depth={depth} />
  );
}

interface FieldsProps {
  schema: JsonSchema;
  root: JsonSchema;
  value: Record<string, unknown>;
  onChange: (next: Record<string, unknown>) => void;
  depth: number;
}

function SchemaFormFields({ schema, root, value, onChange, depth }: FieldsProps) {
  const resolved = resolveRef(schema, root);
  const properties = resolved.properties ?? {};
  const entries = Object.entries(properties);

  return (
    <div className="grid gap-3">
      {entries.map(([key, rawFieldSchema]) => {
        const fieldSchema = resolveRef(rawFieldSchema, root);
        const current = value?.[key];
        const setField = (v: unknown) => onChange({ ...value, [key]: v });
        return (
          <Fragment key={key}>
            <SchemaFormField
              name={key}
              schema={fieldSchema}
              root={root}
              value={current}
              onChange={setField}
              depth={depth}
            />
          </Fragment>
        );
      })}
    </div>
  );
}

interface FieldProps {
  name: string;
  schema: JsonSchema;
  root: JsonSchema;
  value: unknown;
  onChange: (v: unknown) => void;
  depth: number;
}

function SchemaFormField({ name, schema, root, value, onChange, depth }: FieldProps) {
  const { inner, nullable } = unwrapNullable(schema);
  const resolved = resolveRef(inner, root);
  // titleはanyOfでラップされた側(nullable_name等)に付くため、
  // アンラップ前のフィールドschema自体のtitleを優先する
  const label = schema.title ?? resolved.title ?? name;
  const fieldId = `schema-field-${name}-${depth}`;

  // nested object($ref経由含む) → 見出し付きで再帰描画
  if (resolved.type === 'object' && resolved.properties) {
    return (
      <div className="grid gap-2 rounded-md border p-2.5">
        <div className="text-[11px] font-semibold text-muted-foreground">{label}</div>
        <SchemaFormFields
          schema={resolved}
          root={root}
          value={(value as Record<string, unknown>) ?? {}}
          onChange={(next) => onChange(next)}
          depth={depth + 1}
        />
      </div>
    );
  }

  if (isUnsupportedSchema(resolved)) {
    return (
      <div className="grid gap-1.5">
        <Label htmlFor={fieldId}>{label}</Label>
        <JsonFallbackField id={fieldId} value={value} onChange={onChange} />
      </div>
    );
  }

  if (resolved.type === 'boolean') {
    const checked = Boolean(value);
    return (
      <div className="grid gap-1.5">
        <Label>{label}</Label>
        <Toggle
          variant="outline"
          size="sm"
          pressed={checked}
          onPressedChange={(v) => onChange(v)}
          className="w-full justify-start"
          aria-label={label}
        >
          {label}{checked ? '(on)' : '(off)'}
        </Toggle>
      </div>
    );
  }

  if (resolved.enum) {
    const stringValue = value === null || value === undefined ? '' : String(value);
    return (
      <div className="grid gap-1.5">
        <Label htmlFor={fieldId}>{label}</Label>
        <Select
          value={stringValue}
          onValueChange={(v) => {
            // enumの実体は文字列/数値どちらもありうるため元の型に合わせて復元する
            const match = resolved.enum?.find((e) => String(e) === v);
            onChange(match ?? v);
          }}
        >
          <SelectTrigger id={fieldId} size="sm" className="w-full">
            <SelectValue placeholder={`${label}を選択`} />
          </SelectTrigger>
          <SelectContent>
            {resolved.enum.map((opt) => (
              <SelectItem key={String(opt)} value={String(opt)}>
                {String(opt)}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>
    );
  }

  if (resolved.type === 'number' || resolved.type === 'integer') {
    const numValue = value === null || value === undefined ? '' : String(value);
    return (
      <div className="grid gap-1.5">
        <Label htmlFor={fieldId}>{label}</Label>
        <Input
          id={fieldId}
          type="number"
          step={resolved.type === 'integer' ? 1 : 'any'}
          min={resolved.minimum ?? resolved.exclusiveMinimum}
          max={resolved.maximum ?? resolved.exclusiveMaximum}
          value={numValue}
          onChange={(e) => {
            const raw = e.target.value;
            if (raw === '') {
              onChange(nullable ? null : '');
              return;
            }
            const parsed = resolved.type === 'integer' ? parseInt(raw, 10) : parseFloat(raw);
            onChange(Number.isNaN(parsed) ? raw : parsed);
          }}
        />
      </div>
    );
  }

  // 既定: string(nullable含む)
  const stringValue = value === null || value === undefined ? '' : String(value);
  return (
    <div className="grid gap-1.5">
      <Label htmlFor={fieldId}>{label}</Label>
      <Input
        id={fieldId}
        value={stringValue}
        placeholder={nullable ? '未指定' : undefined}
        onChange={(e) => {
          const raw = e.target.value;
          onChange(nullable && raw === '' ? null : raw);
        }}
      />
    </div>
  );
}
