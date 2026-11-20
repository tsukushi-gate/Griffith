#!/usr/bin/env python3
"""公開APIリファレンス(Starlight用Markdown)を自動生成する。

真実源は `griffith_logic/__init__.py` の `_SYMBOL_MODULES` dict。
このスクリプトはパッケージをimportせず、

1. `_SYMBOL_MODULES` をASTで読み(カテゴリ区切りは dict 内のコメント行)
2. 各シンボルの実体モジュールを griffe で静的解析し、
   docstring・シグネチャ・型注釈を抽出して

`docs-site/src/content/docs/reference/api/` へ、カテゴリごとの1ページ+索引ページを書き出す。

各シンボルの末尾には「このAPIを使っている例」(チュートリアルへの逆リンク)を付ける。
出典は `scripts/build_tutorials.py` の `collect_symbol_usage()` で、`examples/*.py` を
AST解析して import 経路から求める。中間ファイルを介さないので、シンボルを増やしても
example を増やしても、生成のたびに逆リンクが自動で追従する。

torchのimportを伴わないため、docs用の軽量環境で実行できる:

    uv run --no-project --with griffe python scripts/gen_api_docs.py

網羅検証(生成物を書かずにチェックのみ。CI向け):

    uv run --no-project --with griffe python scripts/gen_api_docs.py --check
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
import textwrap
from dataclasses import dataclass, field
from pathlib import Path

from griffe import GriffeLoader

# 同じ scripts/ にある build_tutorials は stdlib しか import しないので、
# torch 抜きの軽量環境からでも安全に読める(チュートリアルのレジストリが真実源)。
sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_tutorials import TUTORIALS, collect_symbol_usage  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
PACKAGE = 'griffith_logic'
INIT_PATH = REPO_ROOT / PACKAGE / '__init__.py'
OUT_DIR = REPO_ROOT / 'docs-site' / 'src' / 'content' / 'docs' / 'reference' / 'api'

# カテゴリコメント(dict内の `# ...` 行) → 出力ファイル名(slug)と見出し
CATEGORY_PAGES: dict[str, tuple[str, str]] = {
    'コア層': ('core', 'コア層'),
    '畳み込み・空間処理': ('conv', '畳み込み・空間処理'),
    'seq2seq': ('seq2seq', 'seq2seq(RDDLGN)'),
    'decoder-only 因果Transformer': ('logic-gpt', 'decoder-only 因果Transformer'),
    '論理関数データセット': ('datasets-logic', '論理関数データセット'),
    '視覚データセット中央ローダ': ('datasets-vision', '視覚データセット'),
    'UCI-HAR(スマホ加速度による行動認識)ローダ+温度計符号化': (
        'datasets-har', 'UCI-HAR(行動認識)データセット'),
    '設定管理+共通ランナー': ('training-config', '設定管理・共通ランナー'),
    '学習アルゴリズム(M5)': ('training-learning', '学習アルゴリズム'),
    'ハイブリッドAPI(M4)': ('hybrid', 'ハイブリッドAPI'),
    'エディタ連携(Python由来モデル登録)': ('editor', 'エディタ連携'),
    'DWN (Bacellar et al., ICML 2024)': ('dwn', 'DWN / LLViT'),
    'BinaryBrain互換': ('binarybrain', 'BinaryBrain互換'),
    '一般NN部品(A3)': ('blocks', '一般NN部品'),
    'Attention系': ('attention', 'Attention系'),
    '配線の受容野・被覆率(学習前に配線だけから決まる量)':
        ('wiring-stats', '配線の受容野・被覆率'),
}

KIND_LABELS = {
    'class': 'クラス',
    'function': '関数',
    'attribute': '定数',
    'module': 'モジュール',
    'alias': 'エイリアス',
}

# Google風セクション見出し → 出力する日本語見出し
PARAM_SECTIONS = {'Args', 'Arguments', 'Parameters', 'Attributes'}
SECTION_TITLES = {
    'Args': '引数',
    'Arguments': '引数',
    'Parameters': '引数',
    'Attributes': '属性',
    'Returns': '戻り値',
    'Return': '戻り値',
    'Yields': '生成値',
    'Raises': '例外',
    'Note': '補足',
    'Notes': '補足',
    'Example': '使用例',
    'Examples': '使用例',
    'References': '参考文献',
    'Warning': '注意',
    'Warnings': '注意',
    'See Also': '関連',
}
SECTION_RE = re.compile(r'^(' + '|'.join(map(re.escape, SECTION_TITLES)) + r')\s*:\s*$')
# `name (type): 説明` / `name: 説明`
ENTRY_RE = re.compile(r'^(\*{0,2}\w+)\s*(?:\(([^)]*)\))?\s*:\s*(.*)$')


@dataclass
class Symbol:
    """`_SYMBOL_MODULES` の1エントリ。"""

    name: str
    module: str  # '.layers.lut' 形式(相対)
    category: str

    @property
    def abs_module(self) -> str:
        return PACKAGE + self.module

    @property
    def qualname(self) -> str:
        return f'{self.abs_module}.{self.name}'


@dataclass
class Category:
    """カテゴリ(dict内のコメント区切り)1つ分。"""

    comment: str
    slug: str
    title: str
    order: int
    symbols: list[Symbol] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# 1. _SYMBOL_MODULES の読み取り(importせずAST+行走査)
# --------------------------------------------------------------------------- #


def parse_symbol_modules(init_path: Path = INIT_PATH) -> list[Category]:
    """`_SYMBOL_MODULES` をASTで読み、コメント区切りごとにカテゴリ化して返す。"""
    source = init_path.read_text(encoding='utf-8')
    dict_node = _find_symbol_modules_node(source, init_path)
    ast_pairs = {
        _const_str(k, init_path): _const_str(v, init_path)
        for k, v in zip(dict_node.keys, dict_node.values, strict=True)
    }

    # コメント(カテゴリ)は ast では取れないので、dict の行範囲だけ行走査する
    lines = source.splitlines()
    start, end = dict_node.lineno, dict_node.end_lineno or len(lines)
    categories: list[Category] = []
    current: Category | None = None
    seen: set[str] = set()

    for raw in lines[start:end]:
        stripped = raw.strip()
        if stripped.startswith('#'):
            comment = stripped.lstrip('#').strip()
            slug, title = CATEGORY_PAGES.get(
                comment, (f'section-{len(categories) + 1:02d}', comment)
            )
            current = Category(comment, slug, title, len(categories))
            categories.append(current)
            continue
        entry = re.match(r"^'([^']+)'\s*:\s*'([^']+)'\s*,?$", stripped)
        if not entry:
            continue
        name, module = entry.group(1), entry.group(2)
        if ast_pairs.get(name) != module:
            raise SystemExit(f'{init_path}: 行走査とASTの不一致: {name!r}')
        if current is None:
            raise SystemExit(f'{init_path}: {name!r} の前にカテゴリコメントがない')
        current.symbols.append(Symbol(name, module, current.comment))
        seen.add(name)

    missing = set(ast_pairs) - seen
    if missing:
        raise SystemExit(f'{init_path}: 行走査で拾えなかったシンボル: {sorted(missing)}')
    return categories


def _find_symbol_modules_node(source: str, path: Path) -> ast.Dict:
    tree = ast.parse(source, filename=str(path))
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == '_SYMBOL_MODULES':
                if not isinstance(node.value, ast.Dict):
                    raise SystemExit(f'{path}: _SYMBOL_MODULES が dict リテラルではない')
                return node.value
    raise SystemExit(f'{path}: _SYMBOL_MODULES が見つからない')


def _const_str(node: ast.expr, path: Path) -> str:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    raise SystemExit(f'{path}: _SYMBOL_MODULES に文字列以外の要素がある')


def detect_shim_modules() -> set[str]:
    """トップレベルの後方互換シム(モジュールdocstring先頭が「後方互換シム」)を列挙する。"""
    shims: set[str] = set()
    for py in sorted((REPO_ROOT / PACKAGE).glob('*.py')):
        if py.name == '__init__.py':
            continue
        try:
            doc = ast.get_docstring(ast.parse(py.read_text(encoding='utf-8'))) or ''
        except SyntaxError:
            continue
        if doc.lstrip().startswith('後方互換シム'):
            shims.add(f'.{py.stem}')
    return shims


# --------------------------------------------------------------------------- #
# 2. griffe による静的解析
# --------------------------------------------------------------------------- #


def load_objects(categories: list[Category]) -> dict[str, object]:
    """シンボル名 → griffeオブジェクト。解決できないものは SystemExit。"""
    loader = GriffeLoader(search_paths=[str(REPO_ROOT)], docstring_parser='google')
    modules: dict[str, object] = {}
    resolved: dict[str, object] = {}
    errors: list[str] = []

    for cat in categories:
        for sym in cat.symbols:
            mod = modules.get(sym.abs_module)
            if mod is None:
                try:
                    mod = loader.load(sym.abs_module)
                except Exception as exc:  # noqa: BLE001 — 収集して一括報告する
                    errors.append(f'{sym.abs_module}: {exc}')
                    continue
                modules[sym.abs_module] = mod
            try:
                resolved[sym.name] = mod[sym.name]  # type: ignore[index]
            except KeyError:
                errors.append(f'{sym.qualname}: モジュールに存在しない')

    if errors:
        raise SystemExit('griffeでの解決に失敗:\n  ' + '\n  '.join(errors))
    return resolved


# --------------------------------------------------------------------------- #
# 3. Markdown生成
# --------------------------------------------------------------------------- #


def escape_inline(text: str) -> str:
    """コードスパン外の `<` をエスケープする(Markdown/MDXのHTML誤認防止)。"""
    parts = re.split(r'(`+[^`]*`+)', text)
    return ''.join(p if p.startswith('`') else p.replace('<', '&lt;') for p in parts)


def escape_cell(text: str) -> str:
    """表セル用: パイプ・改行を無害化する。"""
    return escape_inline(text.replace('|', r'\|').replace('\n', ' ').strip())


def render_signature(name: str, obj) -> str:
    """`def f(...)` / `class C(...)` / `NAME: type` のコードブロック本文を作る。"""
    kind = obj.kind.value
    if kind == 'attribute':
        annotation = _expr(getattr(obj, 'annotation', None))
        return f'{name}: {annotation}' if annotation else name
    params = _parameters(obj, kind)
    head = f'class {name}' if kind == 'class' else f'def {name}'
    returns = '' if kind == 'class' else _return_suffix(obj)
    one_line = f'{head}({", ".join(params)}){returns}'
    if len(one_line) <= 88:
        return one_line
    body = ',\n'.join(f'    {p}' for p in params)
    return f'{head}(\n{body},\n){returns}'


def _parameters(obj, kind: str) -> list[str]:
    try:
        parameters = list(obj.parameters)
    except (AttributeError, KeyError):
        return []
    rendered: list[str] = []
    for param in parameters:
        if param.name in ('self', 'cls'):
            continue
        prefix = {'var_positional': '*', 'var_keyword': '**'}.get(param.kind.value, '')
        text = f'{prefix}{param.name}'
        annotation = _expr(param.annotation)
        if annotation:
            text += f': {annotation}'
        if param.default is not None:
            default = _expr(param.default)
            text += f' = {default}' if annotation else f'={default}'
        rendered.append(text)
    return rendered


def _return_suffix(obj) -> str:
    returns = _expr(getattr(obj, 'returns', None))
    return f' -> {returns}' if returns else ''


def _expr(node) -> str:
    if node is None:
        return ''
    text = str(node)
    return ' '.join(text.split())


def render_docstring(obj) -> str:
    """docstringを日本語のまま整形する(Google風セクションは定義リスト化)。"""
    doc = getattr(obj, 'docstring', None)
    if doc is None or not doc.value.strip():
        return '_(docstring未整備)_'

    lines = textwrap.dedent(doc.value).splitlines()
    out: list[str] = []
    in_fence = False
    index = 0
    while index < len(lines):
        line = lines[index]
        if line.lstrip().startswith('```'):
            in_fence = not in_fence
            out.append(line)
            index += 1
            continue
        match = None if in_fence else SECTION_RE.match(line.strip())
        if match and line[:1] not in (' ', '\t'):
            head = match.group(1)
            block, index = _collect_section(lines, index + 1)
            out.extend(_render_section(head, block))
            continue
        out.append(escape_inline(line) if not in_fence else line)
        index += 1
    return '\n'.join(out).strip()


def _collect_section(lines: list[str], start: int) -> tuple[list[str], int]:
    """インデントされたセクション本文を集める。"""
    block: list[str] = []
    index = start
    while index < len(lines):
        line = lines[index]
        if line.strip() and not line.startswith((' ', '\t')):
            break
        block.append(line)
        index += 1
    while block and not block[-1].strip():
        block.pop()
    return textwrap.dedent('\n'.join(block)).splitlines(), index


def _render_section(head: str, block: list[str]) -> list[str]:
    title = SECTION_TITLES[head]
    out = ['', f'**{title}**', '']
    if head in PARAM_SECTIONS:
        out.extend(_render_entries(block))
    else:
        out.extend(escape_inline(line) for line in block)
    out.append('')
    return out


def _render_entries(block: list[str]) -> list[str]:
    """`name (type): 説明` の並びを定義リスト風の箇条書きにする。"""
    items: list[list[str]] = []
    for line in block:
        if not line.strip():
            continue
        entry = ENTRY_RE.match(line) if not line.startswith((' ', '\t')) else None
        if entry:
            name, type_, desc = entry.group(1), entry.group(2), entry.group(3)
            term = f'`{name}`'
            if type_:
                term += f' (`{type_.strip()}`)'
            items.append([f'- {term} — {escape_inline(desc.strip())}'])
        elif items:
            items[-1].append(f'  {escape_inline(line.strip())}')
        else:
            items.append([escape_inline(line.strip())])
    return [chunk for item in items for chunk in item]


def summary_line(obj) -> str:
    doc = getattr(obj, 'docstring', None)
    if doc is None:
        return ''
    for line in doc.value.strip().splitlines():
        if line.strip():
            return line.strip()
    return ''


def anchor(name: str) -> str:
    """Starlight(github-slugger)と同じ見出しIDを再現する。"""
    return re.sub(r'[^a-z0-9_\-]', '', name.lower().replace(' ', '-'))


def frontmatter(title: str, description: str, order: int) -> str:
    return (
        '---\n'
        f"title: '{title}'\n"
        f"description: '{description}'\n"
        'sidebar:\n'
        f'  order: {order}\n'
        '---\n'
    )


GENERATED_NOTE = (
    ':::note[自動生成ページ]\n'
    'このページは `scripts/gen_api_docs.py` が `griffith_logic/__init__.py` の\n'
    '`_SYMBOL_MODULES` から生成しています。直接編集せず、docstringを修正してください。\n'
    ':::\n'
)


def render_examples(name: str, usage: dict[str, list[dict[str, str]]]) -> list[str]:
    """「このAPIを使っている例」ブロック(該当なしなら空リスト)。

    出典は `scripts/build_tutorials.py` の `collect_symbol_usage()`(examples をAST解析)。
    シンボルを増やしても example を増やしても、生成のたびに自動で追従する。
    """
    entries = usage.get(name)
    if not entries:
        return []
    items = [f'- [{e["title"]}]({e["url"]})' + (f' — `{e["source"]}`' if e['source'] else '')
             for e in entries]
    return ['', '**このAPIを使っている例**', '', *items, '']


def render_category_page(
    cat: Category, objects: dict[str, object], usage: dict[str, list[dict[str, str]]]
) -> str:
    names = ' / '.join(sym.name for sym in cat.symbols[:4])
    parts = [
        frontmatter(cat.title, f'{cat.title}の公開API({names} ほか)。', cat.order + 1),
        '',
        GENERATED_NOTE,
    ]
    for sym in cat.symbols:
        obj = objects[sym.name]
        kind = KIND_LABELS.get(obj.kind.value, obj.kind.value)
        parts.append(f'\n## {sym.name}\n')
        parts.append(f'実体: `{sym.qualname}`（{kind}）\n')
        parts.append('```python')
        parts.append(render_signature(sym.name, obj))
        parts.append('```\n')
        parts.append(render_docstring(obj))
        parts.extend(render_examples(sym.name, usage))
        parts.append('')
    return '\n'.join(parts).rstrip() + '\n'


def render_index_page(
    categories: list[Category],
    objects: dict[str, object],
    usage: dict[str, list[dict[str, str]]],
) -> str:
    total = sum(len(c.symbols) for c in categories)
    parts = [
        frontmatter(
            'Python API',
            f'`import griffith_logic` で公開されている{total}シンボルの自動生成リファレンス。',
            0,
        ),
        '',
        GENERATED_NOTE,
        '',
        f'`import griffith_logic` が公開するシンボルは **{total}個**、'
        f'カテゴリは **{len(categories)}区分** です。',
        'いずれも遅延インポート(PEP 562)で、属性アクセス時に実体モジュールが読み込まれます。',
        '',
        '```python',
        'import griffith_logic as ml',
        '',
        'layer = ml.LutLayer(in_dim=8, out_dim=16, n_inputs=2)',
        '```',
        '',
        '## 全シンボル一覧',
        '',
        '「使用例」は [チュートリアル](/tutorials/) のうち'
        'そのAPIを実際に import して動かしているページ数です'
        '(`examples/*.py` のAST解析による自動集計)。',
        '',
        '| 名前 | 種別 | 概要 | カテゴリ | 使用例 |',
        '|---|---|---|---|---|',
    ]
    for cat in categories:
        for sym in cat.symbols:
            obj = objects[sym.name]
            kind = KIND_LABELS.get(obj.kind.value, obj.kind.value)
            link = f'[`{sym.name}`]({cat.slug}/#{anchor(sym.name)})'
            examples = usage.get(sym.name, [])
            count = f'{len(examples)}本' if examples else '—'
            parts.append(
                f'| {link} | {kind} | {escape_cell(summary_line(obj))} '
                f'| [{cat.title}]({cat.slug}/) | {count} |'
            )
    parts += ['', '## カテゴリ', '', '| カテゴリ | シンボル数 | 収録シンボル |', '|---|---|---|']
    for cat in categories:
        members = ' / '.join(f'`{s.name}`' for s in cat.symbols)
        parts.append(f'| [{cat.title}]({cat.slug}/) | {len(cat.symbols)} | {members} |')
    parts += ['', '## 例から探す', '', '各チュートリアルが使っている公開APIの一覧(逆引き)。', '']
    parts += ['| チュートリアル | 使っている公開API |', '|---|---|']
    # 行の並びはチュートリアルのレジストリ順(= ギャラリー・サイドバーと同じ)にする。
    by_slug: dict[str, list[str]] = {}
    for name, entries in usage.items():
        for entry in entries:
            by_slug.setdefault(entry['slug'], []).append(name)
    for spec in TUTORIALS:
        names = by_slug.get(spec.slug)
        if not names:
            continue
        members = ' / '.join(f'`{n}`' for n in sorted(names))
        parts.append(f'| [{spec.title}](/tutorials/{spec.slug}/) | {members} |')
    parts.append('')
    return '\n'.join(parts)


def build_pages(
    categories: list[Category],
    objects: dict[str, object],
    usage: dict[str, list[dict[str, str]]],
) -> dict[str, str]:
    pages = {'index.md': render_index_page(categories, objects, usage)}
    for cat in categories:
        pages[f'{cat.slug}.md'] = render_category_page(cat, objects, usage)
    return pages


# --------------------------------------------------------------------------- #
# 4. 検証
# --------------------------------------------------------------------------- #


def verify(
    categories: list[Category],
    objects: dict[str, object],
    pages: dict[str, str],
    usage: dict[str, list[dict[str, str]]],
) -> int:
    """網羅検証。問題があれば件数を返す(0=OK)。"""
    problems: list[str] = []
    symbols = [s for c in categories for s in c.symbols]
    total = len(symbols)

    if len(set(s.name for s in symbols)) != total:
        problems.append('シンボル名が重複している')
    if len(objects) != total:
        problems.append(f'griffe解決数({len(objects)})がシンボル数({total})と一致しない')
    if len(pages) != len(categories) + 1:
        problems.append(f'ページ数({len(pages)})がカテゴリ数+1({len(categories) + 1})と一致しない')

    # 索引の表がシンボル数ぶんの行を持つか
    index_rows = [
        ln for ln in pages['index.md'].splitlines() if ln.startswith('| [`') and ln.endswith('|')
    ]
    if len(index_rows) != total:
        problems.append(f'索引の行数({len(index_rows)})がシンボル数({total})と一致しない')

    # 各シンボルの見出しがカテゴリページに1つだけあるか
    for cat in categories:
        body = pages[f'{cat.slug}.md']
        for sym in cat.symbols:
            if body.count(f'\n## {sym.name}\n') != 1:
                problems.append(f'{cat.slug}.md に見出し ## {sym.name} が1つでない')
            if f'`{sym.qualname}`' not in body:
                problems.append(f'{cat.slug}.md に実体表記 {sym.qualname} がない')

    # API→例の逆リンク: 集計されたシンボルがすべて公開シンボルであること(綴りズレ検出)と、
    # 逆リンクを持つシンボルのページに実際にブロックが出ていること。
    known = {s.name for s in symbols}
    stray = sorted(set(usage) - known)
    if stray:
        problems.append(f'例の集計に未知のシンボルが混ざっている: {stray}')
    linked = 0
    for cat in categories:
        body = pages[f'{cat.slug}.md']
        for sym in cat.symbols:
            if sym.name not in usage:
                continue
            linked += 1
            if '**このAPIを使っている例**' not in body:
                problems.append(f'{cat.slug}.md に使用例ブロックがない({sym.name})')

    # 後方互換シムを参照していないこと
    shims = detect_shim_modules()
    used_shims = sorted({s.module for s in symbols} & shims)
    if used_shims:
        problems.append(f'後方互換シムを参照している: {used_shims}')

    print(f'シンボル数: {total}')
    print(f'カテゴリ数: {len(categories)}  生成ページ数: {len(pages)} (カテゴリ + 索引)')
    print(f'索引の表の行数: {len(index_rows)}')
    print(f'検出した後方互換シム: {len(shims)}件 / うち参照: {len(used_shims)}件')
    print(f'使用例の逆リンクが付いたシンボル: {linked} / {total}')
    for cat in categories:
        print(f'  - {cat.slug}.md ({cat.title}): {len(cat.symbols)}シンボル')
    for problem in problems:
        print(f'NG: {problem}', file=sys.stderr)
    return len(problems)


def write_pages(pages: dict[str, str], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for filename, body in pages.items():
        (out_dir / filename).write_text(body, encoding='utf-8')


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--check',
        action='store_true',
        help='生成せず網羅検証だけ行う(CI用)。問題があれば非0で終了する',
    )
    parser.add_argument(
        '--out-dir', type=Path, default=OUT_DIR, help=f'出力先(既定: {OUT_DIR})'
    )
    args = parser.parse_args(argv)

    categories = parse_symbol_modules()
    objects = load_objects(categories)
    # 「どの example がどのAPIを使っているか」は examples/*.py のAST解析で毎回作り直す
    # (中間ファイルを介さないので、example を足しただけで逆リンクが増える)。
    usage = collect_symbol_usage()
    pages = build_pages(categories, objects, usage)

    failures = verify(categories, objects, pages, usage)
    if failures:
        return 1
    if args.check:
        print('OK: --check モードのため書き出しはしていない')
        return 0
    write_pages(pages, args.out_dir)
    print(f'書き出し: {args.out_dir}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
