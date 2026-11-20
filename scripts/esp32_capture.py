"""ESP32 のUARTログを非対話で取得する(DTR/RTSでリセット → 読む)。

    uv run python scripts/esp32_capture.py --port /dev/cu.usbserial-XXXX \\
        --baud 115200 --seconds 30 --until DONE --out results/deploy/devices/esp32_xor.log

- 開いた直後に DTR/RTS で EN をパルスしてリセットし、起動ログから拾う
  (`--no-reset` で既に動いている出力をそのまま読む)
- `--until` に一致する行(部分一致)が来たら即終了。来なければ `--seconds` で打ち切る
- `RESULT {...}` 行を JSON として抽出し stdout に出す。`--out foo.log` なら
  `foo.json` に抽出結果も書く(RESULT が無ければ書かない)
- pyserial は `fpga` extra(`uv sync --extra fpga`)に入っている。venv に無い場合は
  ESP-IDF の Python 環境(~/.espressif/python_env/*)に自動で切り替えて再実行する

終了コード: 0=--until到達(RESULTがあれば抽出成功) / 1=タイムアウト / 2=ポートを開けない
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
import time
from pathlib import Path

try:
    import serial  # type: ignore
except ImportError:  # venv に pyserial が無い(extra未同期)場合
    if os.environ.get('ESP32_CAPTURE_REEXEC'):
        sys.exit('pyserial が見つからない: uv sync --extra fpga か ESP-IDF 環境が必要')
    cands = sorted(glob.glob(os.path.expanduser(
        '~/.espressif/python_env/*/bin/python')))
    if not cands:
        sys.exit('pyserial が見つからない: uv sync --extra fpga か ESP-IDF 環境が必要')
    os.environ['ESP32_CAPTURE_REEXEC'] = '1'
    os.execv(cands[-1], [cands[-1], *sys.argv])


ANSI_RE = re.compile(r'\x1b\[[0-9;]*m')  # ESPログの色コード


def reset_chip(ser) -> None:
    """EN を RTS で叩いてリセット(GPIO0 は DTR=False のままHIGH=通常起動)。"""
    ser.dtr = False
    ser.rts = True
    time.sleep(0.1)
    ser.rts = False
    time.sleep(0.05)


def capture(port: str, baud: int, seconds: float, until: str | None,
            reset: bool) -> tuple[str, bool]:
    ser = serial.Serial()
    ser.port = port
    ser.baudrate = baud
    ser.timeout = 0.2
    ser.dtr = False
    ser.rts = False
    ser.open()
    try:
        ser.reset_input_buffer()
        if reset:
            reset_chip(ser)
        buf = bytearray()
        deadline = time.monotonic() + seconds
        hit = False
        while time.monotonic() < deadline:
            chunk = ser.read(4096)
            if chunk:
                buf += chunk
                if until and until.encode() in bytes(buf).replace(b'\r', b''):
                    # until 行の末尾改行まで読み切る
                    t_end = time.monotonic() + 0.3
                    while time.monotonic() < t_end:
                        more = ser.read(4096)
                        if more:
                            buf += more
                    hit = True
                    break
        text = buf.decode('utf-8', errors='replace').replace('\r', '')
        return ANSI_RE.sub('', text), hit
    finally:
        ser.close()


def extract_results(text: str) -> list[dict]:
    out = []
    for line in text.splitlines():
        line = line.strip()
        if line.startswith('RESULT '):
            try:
                out.append(json.loads(line[len('RESULT '):]))
            except json.JSONDecodeError as e:
                print(f'WARN RESULT行のJSONが壊れている: {e}: {line[:120]}', file=sys.stderr)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--port', required=True, help='例 /dev/cu.usbserial-XXXX')
    ap.add_argument('--baud', type=int, default=115200)
    ap.add_argument('--seconds', type=float, default=30.0, help='打ち切り秒数')
    ap.add_argument('--until', default='DONE', help='この文字列を含む行で終了(空文字で無効)')
    ap.add_argument('--out', type=Path, default=None, help='生ログの保存先(.log)')
    ap.add_argument('--no-reset', action='store_true', help='リセットせずに読む')
    ap.add_argument('--quiet', action='store_true', help='生ログをstdoutに流さない')
    args = ap.parse_args(argv)

    try:
        text, hit = capture(args.port, args.baud, args.seconds,
                            args.until or None, reset=not args.no_reset)
    except serial.SerialException as e:
        print(f'ERROR ポートを開けない: {e}', file=sys.stderr)
        return 2

    if not args.quiet:
        print(text, end='' if text.endswith('\n') else '\n')
    results = extract_results(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding='utf-8')
        if results:
            args.out.with_suffix('.json').write_text(
                json.dumps(results[-1] if len(results) == 1 else results,
                           indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    for r in results:
        print('RESULT_JSON', json.dumps(r, ensure_ascii=False))
    if not hit and args.until:
        print(f'TIMEOUT {args.seconds}s 以内に {args.until!r} が来なかった', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
