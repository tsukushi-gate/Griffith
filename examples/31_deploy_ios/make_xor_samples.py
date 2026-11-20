"""XOR 用 samples.json を作る(4パターン全数)。

expected_scores は **IR simulate(numpy の素朴実装)** から作る。生成Cの出力を
ゴールデンにしない(検証対象自身から期待値を作らない鉄則)。

形式(ESP32 側 examples/30_deploy_esp32/ と共通):
  {"input_width": 2,
   "samples": [{"bits": "<hex>", "label": int|null, "expected_scores": [...]}]}
bits は hex 文字列、バイト内 LSB-first(入力ビット i はバイト i//8 の bit i%8)。

実行: uv run python examples/31_deploy_ios/make_xor_samples.py <xor_model.json> <samples.json>
"""

import json
import sys
from pathlib import Path

import numpy as np

from griffith_logic.ir.serialize import ir_from_dict
from griffith_logic.ir.simulate import simulate


def pack_hex(bits: list[int]) -> str:
    """0/1 列 → hex 文字列(バイト内 LSB-first)。"""
    out = bytearray((len(bits) + 7) // 8)
    for i, b in enumerate(bits):
        if b:
            out[i >> 3] |= 1 << (i & 7)
    return out.hex()


def main() -> None:
    model_json, out_path = Path(sys.argv[1]), Path(sys.argv[2])
    graph = ir_from_dict(json.loads(model_json.read_text(encoding='utf-8'))['ir'])
    patterns = [[0, 0], [0, 1], [1, 0], [1, 1]]
    scores = simulate(graph, np.array(patterns, dtype=np.int64))
    samples = [
        {'bits': pack_hex(p), 'label': p[0] ^ p[1],
         'expected_scores': [int(v) for v in scores[i]]}
        for i, p in enumerate(patterns)
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({'input_width': 2, 'samples': samples}),
                        encoding='utf-8')
    print(f'生成: {out_path} ({len(samples)} samples)')


if __name__ == '__main__':
    main()
