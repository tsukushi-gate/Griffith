"""SVG内の<text>要素のバウンディングボックス重なりを検査する。

文字幅は保守的に見積もる(実レンダリングより広めに取る):
  全角(CJK/約物) = 1.00em, ASCII = 0.62em(等幅) / 0.56em(ゴシック)
transform=rotate(-90 cx cy) は幅と高さを入れ替えて扱う。
"""
import re
import sys
import unicodedata


def is_wide(ch: str) -> bool:
    return unicodedata.east_asian_width(ch) in ('W', 'F', 'A')


def text_width(s: str, size: float, mono: bool) -> float:
    narrow = 0.62 if mono else 0.56
    return sum(size * (1.0 if is_wide(c) else narrow) for c in s)


def parse(svg: str):
    """<text>要素を (x, y, size, anchor, mono, rotate, 本文) で返す。"""
    out = []
    # 親<g>のtext-anchorを拾う
    g_anchor = []
    g_tx = []
    for m in re.finditer(r'<g\b[^>]*>|</g>|<text\b[^>]*>.*?</text>', svg, re.S):
        tag = m.group(0)
        if tag.startswith('</g'):
            if g_anchor:
                g_anchor.pop()
            if g_tx:
                g_tx.pop()
        elif tag.startswith('<g'):
            a = re.search(r'text-anchor="([^"]+)"', tag)
            g_anchor.append(a.group(1) if a else (g_anchor[-1] if g_anchor else 'start'))
            tr = re.search(r'translate\(\s*([-\d.]+)[ ,]+([-\d.]+)\s*\)', tag)
            base = g_tx[-1] if g_tx else (0.0, 0.0)
            g_tx.append(
                (base[0] + float(tr.group(1)), base[1] + float(tr.group(2)))
                if tr else base
            )
        else:
            body = re.sub(r'<[^>]+>', '', tag)
            attrs = tag[:tag.index('>')]

            def at(name, default=None, attrs=attrs):
                mm = re.search(name + r'="([^"]+)"', attrs)
                return mm.group(1) if mm else default
            anchor = at('text-anchor') or (g_anchor[-1] if g_anchor else 'start')
            size = float(at('font-size', '12'))
            mono = 'Mono' in (at('font-family') or '')
            rot = at('transform')
            tx, ty = (g_tx[-1] if g_tx else (0.0, 0.0))
            out.append(dict(x=float(at('x', '0')) + tx, y=float(at('y', '0')) + ty, size=size,
                            anchor=anchor, mono=mono, rot=bool(rot), s=body.strip()))
    return out


def bbox(t):
    w = text_width(t['s'], t['size'], t['mono'])
    h = t['size'] * 1.25
    if t['rot']:  # rotate(-90 cx cy): cx,cy中心で90度回転 → 幅と高さが入れ替わる
        cx, cy = t['x'], t['y']
        return (cx - h / 2, cy - w / 2, cx + h / 2, cy + w / 2)
    x, y = t['x'], t['y']
    if t['anchor'] == 'middle':
        x0 = x - w / 2
    elif t['anchor'] == 'end':
        x0 = x - w
    else:
        x0 = x
    return (x0, y - t['size'] * 0.82, x0 + w, y + t['size'] * 0.28)


def overlap(a, b, pad=0.0):
    return not (a[2] <= b[0] + pad or b[2] <= a[0] + pad
                or a[3] <= b[1] + pad or b[3] <= a[1] + pad)


svg = open(sys.argv[1], encoding='utf-8').read()
texts = parse(svg)
boxes = [(t, bbox(t)) for t in texts]
print(f'text要素数: {len(boxes)}')
hits = 0
for i in range(len(boxes)):
    for j in range(i + 1, len(boxes)):
        if overlap(boxes[i][1], boxes[j][1]):
            hits += 1
            print(f'  重なり: {boxes[i][0]["s"][:34]!r} {tuple(round(v,1) for v in boxes[i][1])}')
            print(f'       vs {boxes[j][0]["s"][:34]!r} {tuple(round(v,1) for v in boxes[j][1])}')
vb = re.search(r'viewBox="0 0 (\d+) (\d+)"', svg)
W, H = float(vb.group(1)), float(vb.group(2))
oob = [t['s'][:34] for t, b in boxes if b[0] < -1 or b[1] < -1 or b[2] > W + 1 or b[3] > H + 1]
print(f'重なり件数: {hits}')
print(f'viewBox外へはみ出したラベル: {len(oob)} {oob}')
sys.exit(1 if (hits or oob) else 0)
