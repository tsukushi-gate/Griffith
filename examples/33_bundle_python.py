"""Python API だけでバンドルを作り、Python と C で推論して整数一致を確かめる例。

`griffith deploy bundle` と同じ手順を、CLI を使わず Python から通す。
XOR を学習し、次の6段を1本のスクリプトで実行する。

    学習 → export_ir → モデルdoc → apply_compress → build_bundle → save/load
        → Python 推論(ir.simulate) と C 推論(generate_c + cc)を突き合わせる

注意: `build_bundle()` / `generate_c()` 自体は回路圧縮を自動では行わない。
CLI の `bundle`(既定 `--compress fast`)と `codegen-c`(既定 `--dense-flat auto`)が
やっている圧縮は、Python では `apply_compress` / `resolve_dense_flat` を自分で呼んで
同じ結果にする。このスクリプトはその書き方の実例になっている。

実行: uv run python examples/33_bundle_python.py
期待結果: 4入力すべてで Python 推論 = C 推論 = XOR の正解(Cコンパイラ cc/clang/gcc が必要)
"""

import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from griffith_logic import GroupSum, LutLayer
from griffith_logic.deploy import build_bundle, load_bundle, save_bundle
from griffith_logic.deploy.compress import (
    apply_compress,
    attach_compress_info,
    describe_compress,
)
from griffith_logic.deploy.flatten_dense import describe_dense_flat, resolve_dense_flat
from griffith_logic.deploy.verify import build_c_model, run_c_model
from griffith_logic.export import export_ir
from griffith_logic.ir import validate
from griffith_logic.ir.serialize import ir_from_dict, ir_to_dict
from griffith_logic.ir.simulate import simulate


def train_xor() -> tuple[nn.Module, torch.Tensor, torch.Tensor]:
    """XOR を学習する(examples/06_deploy_c/export_model.py と同じ構成)。"""
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
    return model, x, y


def main(work_dir: str | Path | None = None) -> dict:
    """全段を実行し、結果(各推論のスコア・サイズなど)を dict で返す。"""
    tmp = None
    if work_dir is None:
        tmp = tempfile.TemporaryDirectory(prefix='bundle_python_')
        work_dir = tmp.name
    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)
    try:
        # === 1. 学習 → IR 抽出 ===
        model, x, y = train_xor()
        model.eval()                      # 離散化(Boolean 回路に崩壊)
        ir = export_ir(model, input_width=2, name='xor')
        validate(ir, target='hdl')

        # === 2. モデル doc(webbench 形式)。最小は {'ir': ...} だけ ===
        doc = {'ir': ir_to_dict(ir), 'labels': ['xor=0', 'xor=1']}

        # === 3. 回路圧縮(CLI の `bundle --compress fast` と同じ) ===
        # build_bundle() は圧縮しない。CLI と同じ結果にするには自分で呼ぶ。
        doc, info, _ = apply_compress(doc, level='fast')
        doc = attach_compress_info(doc, info)     # about.compress に記録
        print(describe_compress(info))

        # === 4. バンドル化 → 書き出し → 読み戻し ===
        bundle = build_bundle(doc, verify=True)
        path = work / 'xor.bundle.json'
        save_bundle(bundle, path)
        bundle = load_bundle(path)                # 版・構造の検証つき
        print(f'バンドル: {path.name} ({path.stat().st_size:,}B)')

        # === 5. Python 推論(golden model = ir.simulate) ===
        graph = ir_from_dict(bundle['ir'])
        bits = x.numpy().astype(np.uint8)         # [N, 入力ビット幅] の 0/1
        py_scores = simulate(graph, bits).astype(np.int64)    # [N, クラス数]

        # === 6. C 推論(CLI の `codegen-c` 既定 `--dense-flat auto` と同じ) ===
        # 平坦化は生成時の判断なので、バンドルは変えず C 生成用の複製だけ平坦化する。
        c_bundle, decision = resolve_dense_flat(bundle, mode='auto')
        print(describe_dense_flat(decision))
        exe = build_c_model(c_bundle, work / 'c', dense_flat_decision=decision)
        n_out = int(graph.outputs[0].width)
        c_scores = np.stack([run_c_model(exe, row.tobytes(), n_out)
                             for row in bits])

        # === 突き合わせ ===
        expected = y.numpy()
        print('\n入力   Python scores  C scores   予測  正解')
        for i in range(len(bits)):
            pred = int(np.argmax(py_scores[i]))
            print(f'{tuple(bits[i].tolist())}  {py_scores[i].tolist()!s:<14} '
                  f'{c_scores[i].tolist()!s:<10} {pred:>4} {int(expected[i]):>5}')
        assert np.array_equal(py_scores, c_scores), 'Python と C の整数出力が違う'
        assert np.array_equal(py_scores.argmax(axis=1), expected), 'XOR を解けていない'
        print('\n一致: Python(simulate) = C、予測は XOR の正解どおり')
        return {'py_scores': py_scores, 'c_scores': c_scores,
                'compress': info, 'dense_flat': decision,
                'bundle_bytes': path.stat().st_size}
    finally:
        if tmp is not None:
            tmp.cleanup()


if __name__ == '__main__':
    main()
