"""プローブ(値表示部)の仕様・スライス・評価。

プローブは表示専用のサーバ側後処理で、回路(RTL)には一切影響しない。
値の真実は常に RTL 側(WS フレームの values)にあり、本モジュールは
「どのバスのどのビットを、どのレンダラで整形するか」だけを担う。

注意: viewer 配下では `from __future__ import annotations` を使わない
(FastAPI の WS 型解決が壊れる実測罠。server.py 参照)。
"""

import numpy as np
from pydantic import BaseModel, ConfigDict, Field


class ProbeError(Exception):
    """プローブ設定の不備(ユーザー向けメッセージを持つ)。"""


class ProbeSpec(BaseModel):
    """プローブ 1 個の仕様(永続化・API・評価で共有する契約)。"""

    model_config = ConfigDict(populate_by_name=True)

    id: str
    target: str                       # lowered ノード id | 'x' | 'y' | 'y_comb'
    bit_slice: list[int] | None = Field(default=None, alias='slice')  # [msb, lsb]
    renderer: str = 'auto'
    args: dict = Field(default_factory=dict)   # cols / order 等レンダラ固有
    pos: dict = Field(default_factory=lambda: {'x': 0.0, 'y': 0.0})
    pinned: bool = False
    label: str | None = None
    depth: int = 1                    # 表示キュー段数(過去値の保持数)
    scale: float = 1.0                # カード個別の拡大率(表示のみ)


def target_width(meta: dict, target: str) -> int:
    """プローブ対象のバス幅(meta 基準。ブリッジの hex は非パディングのため必須)。"""
    if target == 'x':
        return int(meta['in_width'])
    if target in ('y', 'y_comb'):
        return int(meta['out_width'])
    for n in meta['nodes']:
        if n['id'] == target:
            return int(n['width'])
    raise ProbeError(
        f'プローブ対象が見つからない: {target}(モデルを変更した場合、'
        '孤児化したプローブは削除してください)')


def target_meta(meta: dict, target: str) -> dict:
    for n in meta['nodes']:
        if n['id'] == target:
            return n
    return {}


def slice_bits(hex_value: str, width: int,
               bit_slice: list[int] | None) -> np.ndarray:
    """hex 値 → LSB-first の 0/1 配列(width 基準)→ スライス適用。

    bit_slice は [msb, lsb](Verilog の [7:0] 表記と同順)。None なら全ビット。
    """
    v = int(hex_value or '0', 16)
    bits = np.array([(v >> i) & 1 for i in range(width)], dtype=np.uint8)
    if bit_slice is None:
        return bits
    if len(bit_slice) != 2:
        raise ProbeError(f'スライスは [msb, lsb] の2要素: {bit_slice}')
    msb, lsb = int(bit_slice[0]), int(bit_slice[1])
    if not (0 <= lsb <= msb < width):
        raise ProbeError(
            f'スライス範囲外: [{msb}:{lsb}](バス幅 {width}。'
            f'0 <= lsb <= msb < {width} が必要)')
    return bits[lsb:msb + 1]


class ProbeRegistry:
    """アクティブなプローブ一覧+表示キュー履歴。

    変更はサーバのイベントループ上のハンドラからのみ行う(server.py と同じ
    単一スレッドモデル)ためロック不要。evaluate はスナップショットに対して行う。

    履歴はプローブごとに (cycle, スライス後 hex) を新しい順で保持する。
    クロックが進んだ時だけ積む(read/再表示では積まない)。リセット
    (cycle が戻った)で消す。target/slice が変わったプローブの履歴も消す。
    """

    def __init__(self) -> None:
        self._probes: dict[str, ProbeSpec] = {}
        self._history: dict[str, list[tuple[int, str]]] = {}
        self._last_cycle: dict[str, int] = {}

    def replace_all(self, probes: list[ProbeSpec]) -> None:
        old = self._probes
        self._probes = {p.id: p for p in probes}
        for pid, p in self._probes.items():
            prev = old.get(pid)
            if prev is None or prev.target != p.target \
                    or prev.bit_slice != p.bit_slice:
                self._history.pop(pid, None)
                self._last_cycle.pop(pid, None)
        for pid in list(self._history):
            if pid not in self._probes:
                self._history.pop(pid, None)
                self._last_cycle.pop(pid, None)

    def snapshot(self) -> list[ProbeSpec]:
        return list(self._probes.values())

    def record(self, probe_id: str, cycle: int, hex_value: str,
               depth: int) -> list[tuple[int, str]]:
        """履歴を更新して新しい順のリストを返す(時間が進んだ時のみ積む)。"""
        hist = self._history.get(probe_id, [])
        last = self._last_cycle.get(probe_id)
        if last is not None and cycle < last:
            hist = []                      # リセット(時間が戻った)
        if last is None or cycle != last or not hist:
            if not hist or hist[0][0] != cycle:
                hist = [(cycle, hex_value)] + hist
            else:
                hist = [(cycle, hex_value)] + hist[1:]   # 同 cycle は上書き
        self._last_cycle[probe_id] = cycle
        hist = hist[:max(1, depth)]
        self._history[probe_id] = hist
        return hist

    def __len__(self) -> int:
        return len(self._probes)


def _bits_to_hex(bits: np.ndarray) -> str:
    v = 0
    for i, b in enumerate(bits):
        v |= int(b) << i
    return f'{v:x}'


def evaluate_probes(registry: ProbeRegistry, values: dict, meta: dict,
                    cycle: int, hub) -> dict:
    """全アクティブプローブを評価して {probe_id: payload} を返す。

    1 個の失敗はそのプローブの {'kind':'error'} に閉じ込め、他へ波及させない。
    depth > 1 のプローブは表示キュー: 過去のスライス済み生値を保持し、
    毎回**現在のレンダラで再整形**して {'kind':'queue'} を返す
    (レンダラ変更が過去値にも遡って効く)。
    """
    out: dict[str, dict] = {}
    for spec in registry.snapshot():
        try:
            width = target_width(meta, spec.target)
            hex_value = values.get(spec.target)
            if hex_value is None:
                raise ProbeError(f'値が未取得: {spec.target}')
            bits = slice_bits(hex_value, width, spec.bit_slice)
            sliced_w = int(bits.shape[0])

            def ctx_for(cyc: int, _spec=spec, _w=sliced_w, _fw=width) -> dict:
                return {
                    'width': _w,
                    'full_width': _fw,
                    'cycle': cyc,
                    'target': _spec.target,
                    'slice': _spec.bit_slice,
                    'meta': meta,
                    'target_meta': target_meta(meta, _spec.target),
                    'args': _spec.args,
                }

            fn = hub.resolve(spec.renderer, ctx_for(cycle))

            def render(bits_arr: np.ndarray, cyc: int,
                       _fn=fn, _ctx=ctx_for, _name=spec.renderer) -> dict:
                payload = _fn(bits_arr, _ctx(cyc))
                if not isinstance(payload, dict) or 'kind' not in payload:
                    raise ProbeError(
                        f'レンダラ {_name} の返り値が不正'
                        "(dict で 'kind' キー必須)")
                return payload

            if spec.depth <= 1:
                out[spec.id] = render(bits, cycle)
                continue

            hist = registry.record(spec.id, cycle, _bits_to_hex(bits),
                                   spec.depth)
            items = []
            for cyc, hex_val in hist:      # 新しい順
                v = int(hex_val or '0', 16)
                arr = np.array([(v >> i) & 1 for i in range(sliced_w)],
                               dtype=np.uint8)
                items.append({'cycle': cyc, 'payload': render(arr, cyc)})
            out[spec.id] = {'kind': 'queue', 'depth': spec.depth,
                            'items': items}
        except Exception as e:  # noqa: BLE001 - プローブ単位で隔離
            out[spec.id] = {'kind': 'error',
                            'message': f'{type(e).__name__}: {e}'}
    return out
