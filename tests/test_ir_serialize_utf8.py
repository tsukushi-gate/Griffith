"""IR の保存・読込が UTF-8 で行われることのテスト(CWO-11)。

Windows の既定コードページ(cp932)では、非 ASCII の metadata を持つ IR を
`save_ir` / `load_ir` が encoding 指定なしで開くと UnicodeEncodeError になる。
IR は JSON(= UTF-8)が仕様なので、ロケールに依らず往復できることを固定する。
(cp932 以外のロケールでは修正前も通るため、回帰の検出は cp932 環境に限る)
"""

import torch
import torch.nn as nn

from griffith_logic import GroupSum, LutLayer
from griffith_logic.export import export_ir
from griffith_logic.ir import load_ir, save_ir


def test_save_load_roundtrip_non_ascii_metadata(tmp_path):
    torch.manual_seed(0)
    model = nn.Sequential(
        LutLayer(16, 24, n_inputs=2, parameterization='softmax'),
        GroupSum(k=4, tau=2.5),
    )
    graph = export_ir(model, input_width=16)
    wiring = {'kind': 'rot', 'note': '層 0 のずらし量(右 1 画素・1 行下)'}
    graph.metadata['wiring'] = wiring

    path = tmp_path / 'utf8.ir.json'
    save_ir(graph, str(path))

    assert '層 0'.encode() in path.read_bytes()
    back = load_ir(str(path))
    assert back.metadata['wiring'] == wiring
