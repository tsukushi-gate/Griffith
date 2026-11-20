"""デプロイキット例(1/2): XOR学習 → IR抽出 → デプロイ用モデルJSON。

`griffith deploy bundle` が読む「webbench形式のモデルJSON」を書き出す。最小構成は
`{"ir": <IRのdict>}` だけでよい(griffith_logic/deploy/bundle.py の build_bundle)。

実行: uv run python examples/06_deploy_c/export_model.py
期待結果: out_examples/xor_model.json が生成される
"""

import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from griffith_logic import GroupSum, LutLayer
from griffith_logic.export import export_ir
from griffith_logic.ir import validate
from griffith_logic.ir.serialize import ir_to_dict

torch.manual_seed(0)
x = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
y = torch.tensor([0, 1, 1, 0])

model = nn.Sequential(
    LutLayer(2, 16, n_inputs=2),
    LutLayer(16, 16, n_inputs=2),
    GroupSum(k=2, tau=2.0),
)
opt = torch.optim.Adam(model.parameters(), lr=0.05)
for _ in range(500):
    opt.zero_grad()
    F.cross_entropy(model(x), y).backward()
    opt.step()

# 学習済みモデル → IR(全パラメータ化を真理値表へ万能離散化)
model.eval()
ir = export_ir(model, input_width=2, name='xor')
validate(ir, target='hdl')

out_dir = Path('out_examples')
out_dir.mkdir(exist_ok=True)
out_path = out_dir / 'xor_model.json'
doc = {'ir': ir_to_dict(ir)}
out_path.write_text(json.dumps(doc), encoding='utf-8')
print(f'生成: {out_path}')
print('次は: uv run griffith deploy bundle out_examples/xor_model.json '
      '-o out_examples/xor.bundle.json')
