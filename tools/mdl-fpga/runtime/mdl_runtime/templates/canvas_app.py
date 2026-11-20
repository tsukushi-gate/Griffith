"""汎用 pygame テンプレ: 入力(手描き or ビットトグル)→ FPGA → 出力表示。

- 入力: semantics が bitmapNxM なら N×M 手描きグリッド(左ドラッグ=描く、
  右=消す)。それ以外は先頭 64bit のトグルグリッド
- 出力: class_scores なら棒グラフ+pred、それ以外はビットランプ
- キー: C=クリア / Esc=終了。通信はメインループ内で値変化時+定期に送信

改造の起点として最小限に書いてある(mdl_runtime の protocol/link を使えば
pygame 以外でも同じことができる。serve-stdio 参照)。
"""

from __future__ import annotations

import re
import time

from ..protocol import bits_of, decode_scores, value_of

CELL = 14          # 手描きグリッド1マスの px(トグルは拡大される)
FPS = 30
SEND_INTERVAL = 0.1

# 日本語を含む文字列の表示用(pygame 既定フォントは CJK グリフを持たない)
_JP_FONTS = ('meiryo', 'yu gothic ui', 'msgothic', 'noto sans cjk jp',
             'hiragino kaku gothic pron', 'malgun gothic')


def _font(size: int):
    import pygame
    for name in _JP_FONTS:
        path = pygame.font.match_font(name)
        if path:
            return pygame.font.Font(path, size)
    return pygame.font.SysFont(None, size)   # CJK 無し環境は既定にフォールバック


def _input_grid(contract: dict) -> tuple[int, int, bool]:
    """(cols, rows, is_bitmap)。bitmapNxM 以外はトグル(最大64bit)。"""
    inp = (contract.get('inputs') or [{}])[0]
    m = re.fullmatch(r'bitmap(\d+)x(\d+)', inp.get('semantics') or '')
    if m:
        return int(m.group(2)), int(m.group(1)), True
    w = min(contract['in_width'], 64)
    cols = min(w, 8)
    rows = (w + cols - 1) // cols
    return cols, rows, False


def run(contract: dict, link) -> None:
    import pygame
    pygame.init()          # display だけでなく font 等の全モジュールを初期化

    cols, rows, is_bitmap = _input_grid(contract)
    in_width = contract['in_width']
    out = (contract.get('outputs') or [{}])[0]
    k = out.get('k') if out.get('semantics') == 'class_scores' else None
    sb = out.get('score_bits', 1)

    # トグル(少ビット)はセルを拡大して操作しやすくする
    cell = CELL if is_bitmap else max(40, CELL)
    grid_w, grid_h = cols * cell, rows * cell
    panel_w = 280
    win_w = max(grid_w + panel_w + 40, 600)
    win_h = max(grid_h, 320) + 60
    win = pygame.display.set_mode((win_w, win_h))
    pygame.display.set_caption(f'mdl runtime — {contract["name"]}')
    font = _font(16)
    big = _font(48)
    clock = pygame.time.Clock()

    bits = [0] * in_width
    y_val: int | None = None
    err = ''
    last_sent = None
    last_t = 0.0

    def cell_at(pos):
        x, y = pos[0] - 10, pos[1] - 10
        if 0 <= x < grid_w and 0 <= y < grid_h:
            c, r = x // cell, y // cell
            i = r * cols + c
            return i if i < in_width else None
        return None

    running = True
    while running:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
            elif ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_ESCAPE:
                    running = False
                elif ev.key == pygame.K_c:
                    bits = [0] * in_width
            elif ev.type == pygame.MOUSEBUTTONDOWN and not is_bitmap:
                i = cell_at(ev.pos)
                if i is not None:
                    bits[i] ^= 1
        if is_bitmap and pygame.mouse.get_pressed()[0]:
            i = cell_at(pygame.mouse.get_pos())
            if i is not None:
                bits[i] = 1
        if is_bitmap and pygame.mouse.get_pressed()[2]:
            i = cell_at(pygame.mouse.get_pos())
            if i is not None:
                bits[i] = 0

        x_val = value_of(bits)
        now = time.time()
        if x_val != last_sent or now - last_t > SEND_INTERVAL:
            try:
                y_val = link.query(int(now * 1000) & 0xFF, x_val)
                err = ''
            except (TimeoutError, OSError) as e:
                if str(e) != err:
                    print(f'[GUI] {e}')       # 全文はコンソールへ
                err = str(e)
            last_sent, last_t = x_val, now

        win.fill((24, 24, 28))
        for i, b in enumerate(bits[:cols * rows]):
            r, c = divmod(i, cols)
            rect = (10 + c * cell, 10 + r * cell, cell - 1, cell - 1)
            win.fill((240, 240, 240) if b else (60, 60, 68), rect)
            if not is_bitmap and cell >= 32:   # トグルにはビット番号を表示
                lbl = font.render(str(i), True,
                                  (60, 60, 68) if b else (150, 150, 160))
                win.blit(lbl, (rect[0] + 4, rect[1] + 2))
        px = grid_w + 40
        if y_val is None:
            win.blit(font.render('FPGA 応答待ち…', True, (220, 200, 120)),
                     (px, 12))
            win.blit(font.render('flash 済みか・電源投入後に再 flash したかを'
                                 '確認(SRAM は電源断で消える)', True,
                                 (150, 150, 160)), (px, 36))
        elif k:
            scores = decode_scores(y_val, k, sb)
            top = max(range(k), key=lambda i: scores[i])
            smax = max(1, (1 << sb) - 1)
            for i, s in enumerate(scores):
                bar = int(180 * s / smax)
                col = (120, 200, 120) if i == top else (110, 110, 130)
                win.fill(col, (px + 30, 12 + i * 26, bar, 18))
                win.blit(font.render(f'{i}: {s}', True, (230, 230, 230)),
                         (px, 12 + i * 26))
            win.blit(big.render(str(top), True, (120, 220, 120)),
                     (px + 120, 14 + k * 26))
        elif y_val is not None:
            for i, b in enumerate(bits_of(y_val, contract['out_width'])[:64]):
                col = (120, 220, 120) if b else (60, 60, 68)
                win.fill(col, (px + (i % 8) * 22, 12 + (i // 8) * 22, 18, 18))
            win.blit(font.render(f'y = 0x{y_val:x}', True, (230, 230, 230)),
                     (px, 200))
        status = err or f'port={link.port} / C=クリア Esc=終了'
        if len(status) > 70:
            status = status[:70] + '…'
        win.blit(font.render(status, True,
                             (220, 120, 120) if err else (140, 140, 150)),
                 (10, win_h - 26))
        pygame.display.flip()
        clock.tick(FPS)
    pygame.quit()
