// @vitest-environment jsdom
// SchemaFormの型→ウィジェット対応表の単体テスト。
// pydantic model_json_schema()形式($defs+$ref)を想定した小さなスキーマで検証する。

import { cleanup, fireEvent, render, screen } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { buildSchemaDefaults, type JsonSchema } from './jsonSchema';
import { SchemaForm } from './SchemaForm';

afterEach(() => {
  cleanup();
});

const SCHEMA: JsonSchema = {
  $defs: {
    Nested: {
      type: 'object',
      title: 'Nested',
      properties: {
        name: { type: 'string', title: 'Name', default: 'foo' },
      },
    },
  },
  type: 'object',
  properties: {
    label: { type: 'string', title: 'Label', default: 'hello' },
    count: { type: 'integer', title: 'Count', default: 3, minimum: 1 },
    enabled: { type: 'boolean', title: 'Enabled', default: false },
    kind: { type: 'string', enum: ['a', 'b'], title: 'Kind', default: 'a' },
    nullable_name: {
      anyOf: [{ type: 'string' }, { type: 'null' }],
      title: 'Nullable Name',
      default: null,
    },
    nested: { $ref: '#/$defs/Nested' },
    tags: { type: 'array', title: 'Tags', default: ['x', 'y'] },
  },
};

describe('SchemaForm', () => {
  it('string/number/boolean/enum/nested各フィールドをクラッシュせず描画する', () => {
    render(<SchemaForm schema={SCHEMA} value={buildSchemaDefaults(SCHEMA, SCHEMA)} onChange={() => {}} />);
    expect((screen.getByLabelText('Label') as HTMLInputElement).value).toBe('hello');
    expect((screen.getByLabelText('Count') as HTMLInputElement).value).toBe('3');
    expect(screen.getByText('Nested')).toBeTruthy();
    expect((screen.getByLabelText('Name') as HTMLInputElement).value).toBe('foo');
  });

  it('string入力の変更でonChangeが新しい値を伴って呼ばれる', () => {
    const onChange = vi.fn();
    render(<SchemaForm schema={SCHEMA} value={{ label: 'hello' }} onChange={onChange} />);
    fireEvent.change(screen.getByLabelText('Label'), { target: { value: 'world' } });
    expect(onChange).toHaveBeenCalledWith(expect.objectContaining({ label: 'world' }));
  });

  it('integer入力は数値へ変換してonChangeへ渡す', () => {
    const onChange = vi.fn();
    render(<SchemaForm schema={SCHEMA} value={{ count: 3 }} onChange={onChange} />);
    fireEvent.change(screen.getByLabelText('Count'), { target: { value: '7' } });
    expect(onChange).toHaveBeenCalledWith(expect.objectContaining({ count: 7 }));
  });

  it('nullable文字列は空入力でnullを渡す', () => {
    const onChange = vi.fn();
    render(
      <SchemaForm schema={SCHEMA} value={{ nullable_name: 'x' }} onChange={onChange} />,
    );
    fireEvent.change(screen.getByLabelText('Nullable Name'), { target: { value: '' } });
    expect(onChange).toHaveBeenCalledWith(expect.objectContaining({ nullable_name: null }));
  });

  it('未知型(array)はJSONテキストエリアへfallbackし、不正JSONはonChangeを呼ばない', () => {
    const onChange = vi.fn();
    render(<SchemaForm schema={SCHEMA} value={{ tags: ['x', 'y'] }} onChange={onChange} />);
    const textarea = screen.getByLabelText('Tags') as HTMLTextAreaElement;
    expect(textarea.value).toContain('"x"');

    fireEvent.change(textarea, { target: { value: 'not json' } });
    expect(onChange).not.toHaveBeenCalled();
    expect(screen.getByText(/JSONとして解釈できません/)).toBeTruthy();

    fireEvent.change(textarea, { target: { value: '["z"]' } });
    expect(onChange).toHaveBeenCalledWith(expect.objectContaining({ tags: ['z'] }));
  });

  it('ネストしたobjectの子フィールド変更が親のonChangeへ正しくマージされる', () => {
    const onChange = vi.fn();
    render(
      <SchemaForm
        schema={SCHEMA}
        value={{ label: 'hello', nested: { name: 'foo' } }}
        onChange={onChange}
      />,
    );
    fireEvent.change(screen.getByLabelText('Name'), { target: { value: 'bar' } });
    expect(onChange).toHaveBeenCalledWith(
      expect.objectContaining({ label: 'hello', nested: { name: 'bar' } }),
    );
  });
});

describe('buildSchemaDefaults', () => {
  it('スキーマのdefault値からネスト込みの初期dictを生成する', () => {
    const defaults = buildSchemaDefaults(SCHEMA, SCHEMA);
    expect(defaults.label).toBe('hello');
    expect(defaults.count).toBe(3);
    expect(defaults.enabled).toBe(false);
    expect(defaults.kind).toBe('a');
    expect(defaults.nullable_name).toBeNull();
    expect(defaults.nested).toEqual({ name: 'foo' });
    expect(defaults.tags).toEqual(['x', 'y']);
  });
});
