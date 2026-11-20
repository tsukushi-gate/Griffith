// サイドバーの操作パネル。各ページのHTMLに散らばっていた
// <select>/<input> を設定オブジェクトから描く。
//
// **要素IDは従来と同一にすること**。各ページの *_main.ts は
// document.getElementById('dlgnBackend') のようにIDで引いており、
// ここで描き忘れると起動時に null 参照で落ちる(ui/controls.test.ts で機械的に検査)。

export interface SelectOption {
  value: string;
  label: string;
  selected?: boolean;
  disabled?: boolean;
}

export type ControlSpec =
  | { kind: 'select'; id: string; label: string; options: SelectOption[]; note?: string }
  | {
    kind: 'range'; id: string; label: string;
    min: number; max: number; value: number; note?: string;
  }
  | { kind: 'checkbox'; id: string; label: string; checked?: boolean }
  | { kind: 'file'; id: string; label: string; accept?: string; disabled?: boolean }
  | { kind: 'button'; id: string; label: string }
  /** 表示/非表示をまとめて切り替えたい塊(webcam の #thrWrap)。 */
  | { kind: 'group'; id: string; hidden?: boolean; children: ControlSpec[] };

/** 描画される要素IDを全て列挙する(group自身のIDも含む)。 */
export function controlIds(specs: readonly ControlSpec[]): string[] {
  const out: string[] = [];
  for (const s of specs) {
    out.push(s.id);
    if (s.kind === 'group') out.push(...controlIds(s.children));
  }
  return out;
}

const LABEL_CLS = 'block text-xs font-medium text-muted';
const FIELD_CLS = 'mt-1 w-full rounded-md border border-line bg-card px-2 py-1.5 '
  + 'text-sm text-fg';

function labelled(text: string, field: HTMLElement, note?: string): HTMLElement {
  const wrap = document.createElement('label');
  wrap.className = 'block';
  const span = document.createElement('span');
  span.className = LABEL_CLS;
  span.textContent = text;
  wrap.append(span, field);
  if (note) {
    const n = document.createElement('span');
    n.className = 'mt-1 block text-xs text-muted';
    n.textContent = note;
    wrap.append(n);
  }
  return wrap;
}

function renderOne(spec: ControlSpec): HTMLElement {
  switch (spec.kind) {
    case 'select': {
      const sel = document.createElement('select');
      sel.id = spec.id;
      sel.className = FIELD_CLS;
      for (const o of spec.options) {
        const opt = document.createElement('option');
        opt.value = o.value;
        opt.textContent = o.label;
        if (o.disabled) opt.disabled = true;
        sel.append(opt);
      }
      // 追加後に value で初期選択を確定させる(append 途中の selected 設定は
      // ブラウザ側の選択アルゴリズムで上書きされうる)。
      const initial = spec.options.find((o) => o.selected);
      if (initial) sel.value = initial.value;
      return labelled(spec.label, sel, spec.note);
    }
    case 'range': {
      const input = document.createElement('input');
      input.type = 'range';
      input.id = spec.id;
      input.min = String(spec.min);
      input.max = String(spec.max);
      input.value = String(spec.value);
      input.className = 'mt-2 w-full accent-accent';
      return labelled(spec.label, input, spec.note);
    }
    case 'checkbox': {
      const wrap = document.createElement('label');
      wrap.className = 'flex items-center gap-2 text-xs font-medium text-muted';
      const input = document.createElement('input');
      input.type = 'checkbox';
      input.id = spec.id;
      input.className = 'accent-accent';
      if (spec.checked) input.checked = true;
      const span = document.createElement('span');
      span.textContent = spec.label;
      wrap.append(input, span);
      return wrap;
    }
    case 'file': {
      const input = document.createElement('input');
      input.type = 'file';
      input.id = spec.id;
      if (spec.accept) input.accept = spec.accept;
      if (spec.disabled) input.disabled = true;
      input.className = 'mt-1 w-full text-xs text-muted file:mr-2 file:rounded-md '
        + 'file:border file:border-line file:bg-card file:px-2 file:py-1 '
        + 'file:text-xs file:text-fg';
      return labelled(spec.label, input);
    }
    case 'button': {
      const btn = document.createElement('button');
      btn.type = 'button';
      btn.id = spec.id;
      btn.textContent = spec.label;
      btn.className = 'w-full rounded-md bg-accent px-3 py-2 text-sm font-medium '
        + 'text-white';
      return btn;
    }
    case 'group': {
      const div = document.createElement('div');
      div.id = spec.id;
      div.className = 'space-y-3';
      if (spec.hidden) div.style.display = 'none';
      div.append(renderControls(spec.children));
      return div;
    }
  }
}

export function renderControls(specs: readonly ControlSpec[]): DocumentFragment {
  const frag = document.createDocumentFragment();
  for (const s of specs) frag.append(renderOne(s));
  return frag;
}
