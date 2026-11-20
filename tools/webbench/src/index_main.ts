// ホーム画面。サイドバーと、各デモへのカードを描くだけ(推論はしない)。
import { renderShell } from './ui/shell';
import { PAGES } from './ui/pages';
import { navItems } from './ui/nav';

renderShell(PAGES.index);

const box = document.getElementById('tasks');
if (box) {
  for (const item of navItems('index')) {
    if (item.id === 'index') continue;
    const a = document.createElement('a');
    a.href = item.url;
    a.className = 'block rounded-xl border border-line bg-card p-4 hover:border-accent';
    const title = document.createElement('div');
    title.className = 'text-sm font-semibold';
    title.textContent = item.label;
    const hint = document.createElement('div');
    hint.className = 'mt-1 text-xs text-muted';
    hint.textContent = item.hint;
    a.append(title, hint);
    box.append(a);
  }
}
