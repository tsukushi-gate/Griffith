// @vitest-environment jsdom
// NumberField: −/+ は1クリックで±1のみ(連続増減しない)/クリックで全選択
// (上書き入力=「1」に「8」と打って 18 にならない)/クランプ。
import { cleanup, fireEvent, render, screen } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';

import { NumberField } from './NumberField';

afterEach(cleanup);

describe('NumberField', () => {
  it('+ / − は1クリックにつき±1だけ(連続増減しない)', () => {
    const onCommit = vi.fn();
    render(<NumberField value={5} min={1} max={16} onCommit={onCommit} />);
    fireEvent.click(screen.getByTitle('+1'));
    expect(onCommit).toHaveBeenCalledTimes(1);
    expect(onCommit).toHaveBeenLastCalledWith(6);
    fireEvent.click(screen.getByTitle('-1'));
    expect(onCommit).toHaveBeenCalledTimes(2);
    expect(onCommit).toHaveBeenLastCalledWith(4);
  });

  it('クランプ(min/max を超えない)', () => {
    const onCommit = vi.fn();
    const { rerender } = render(
      <NumberField value={16} min={1} max={16} onCommit={onCommit} />);
    fireEvent.click(screen.getByTitle('+1'));
    expect(onCommit).toHaveBeenLastCalledWith(16);
    rerender(<NumberField value={1} min={1} max={16} onCommit={onCommit} />);
    fireEvent.click(screen.getByTitle('-1'));
    expect(onCommit).toHaveBeenLastCalledWith(1);
  });

  it('フォーカスで全選択され、上書き入力が commit される(1→8 が 18 にならない)', () => {
    const onCommit = vi.fn();
    render(<NumberField value={1} min={1} max={16} onCommit={onCommit} />);
    const input = screen.getByRole('textbox') as HTMLInputElement;
    fireEvent.focus(input);
    expect(input.selectionStart).toBe(0);
    expect(input.selectionEnd).toBe(input.value.length);
    // 全選択状態で「8」を打つ=値ごと置換
    fireEvent.change(input, { target: { value: '8' } });
    fireEvent.blur(input);
    expect(onCommit).toHaveBeenLastCalledWith(8);
  });

  it('空入力: placeholder あり=null(auto)/範囲外はクランプして commit', () => {
    const onCommit = vi.fn();
    render(<NumberField value={7} min={1} max={16} placeholder="auto"
                        onCommit={onCommit} />);
    const input = screen.getByRole('textbox') as HTMLInputElement;
    fireEvent.change(input, { target: { value: '' } });
    fireEvent.blur(input);
    expect(onCommit).toHaveBeenLastCalledWith(null);
    fireEvent.change(input, { target: { value: '99' } });
    fireEvent.blur(input);
    expect(onCommit).toHaveBeenLastCalledWith(16);
  });
});
