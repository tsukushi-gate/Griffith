"""scripts/check_tutorial_drift.py のテスト(一時gitリポジトリ、ネットワーク不要)。"""

from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location(
    "check_tutorial_drift", Path(__file__).resolve().parent.parent / "scripts" / "check_tutorial_drift.py"
)
drift = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(drift)

MD = """---
title: demo
---

## 概要

```python
print(1)
```

```text title="出力"
loss 0.123
acc 0.797
```

![png](demo_files/demo_1_0.png)

おわり
"""


def _git(cwd, *args):
    subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@t", *args],
        cwd=cwd, check=True, capture_output=True,
    )


@pytest.fixture
def tdir(tmp_path):
    _git(tmp_path, "init", "-q")
    d = tmp_path / "tutorials"
    (d / "demo_files").mkdir(parents=True)
    (d / "demo.md").write_text(MD, encoding="utf-8")
    (d / "index.mdx").write_text("# idx\n", encoding="utf-8")
    (d / "demo_files" / "demo_1_0.png").write_bytes(b"png-a")
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-q", "-m", "init")
    return d


def _run(d):
    return drift.check(d, "HEAD")


def test_identical_ok(tdir):
    assert _run(tdir) == []


def test_output_numbers_ok(tdir):
    p = tdir / "demo.md"
    p.write_text(MD.replace("0.797", "0.876"), encoding="utf-8")
    assert _run(tdir) == []


def test_png_bytes_ok(tdir):
    (tdir / "demo_files" / "demo_1_0.png").write_bytes(b"png-b")
    assert _run(tdir) == []


def test_prose_change_drift(tdir):
    (tdir / "demo.md").write_text(MD.replace("おわり", "終わり"), encoding="utf-8")
    probs = _run(tdir)
    assert len(probs) == 1 and "demo.md" in probs[0] and "終わり" in probs[0]


def test_code_change_drift(tdir):
    (tdir / "demo.md").write_text(MD.replace("print(1)", "print(2)"), encoding="utf-8")
    assert _run(tdir)


def test_output_block_count_drift(tdir):
    extra = MD.replace("おわり", '```text title="出力"\nx\n```\n\nおわり')
    (tdir / "demo.md").write_text(extra, encoding="utf-8")
    assert _run(tdir)


def test_image_line_removed_drift(tdir):
    (tdir / "demo.md").write_text(MD.replace("![png](demo_files/demo_1_0.png)\n", ""), encoding="utf-8")
    assert _run(tdir)


def test_index_mdx_exact(tdir):
    (tdir / "index.mdx").write_text("# idx2\n", encoding="utf-8")
    assert _run(tdir)


def test_untracked_md_drift(tdir):
    (tdir / "new.md").write_text("x\n", encoding="utf-8")
    probs = _run(tdir)
    assert any("new.md" in p for p in probs)


def test_png_added_drift(tdir):
    (tdir / "demo_files" / "extra.png").write_bytes(b"x")
    probs = _run(tdir)
    assert any("extra.png" in p for p in probs)


def test_png_removed_drift(tdir):
    (tdir / "demo_files" / "demo_1_0.png").unlink()
    assert _run(tdir)


def test_main_exit_codes(tdir, capsys):
    assert drift.main(["--tutorials-dir", str(tdir)]) == 0
    (tdir / "demo.md").write_text(MD.replace("おわり", "x"), encoding="utf-8")
    assert drift.main(["--tutorials-dir", str(tdir)]) == 1
    assert "ドリフト" in capsys.readouterr().out


def test_stderr_streams_are_dropped_from_executed_notebook(tmp_path):
    import json

    bt_spec = importlib.util.spec_from_file_location(
        "build_tutorials", Path(__file__).resolve().parent.parent / "scripts" / "build_tutorials.py"
    )
    bt = importlib.util.module_from_spec(bt_spec)
    import sys

    sys.modules["build_tutorials"] = bt
    bt_spec.loader.exec_module(bt)
    nb = {
        "cells": [
            {
                "cell_type": "code",
                "outputs": [
                    {"output_type": "stream", "name": "stderr", "text": "10%|#\\r"},
                    {"output_type": "stream", "name": "stdout", "text": "ok\\n"},
                    {"output_type": "execute_result", "data": {}, "metadata": {}, "execution_count": 1},
                ],
            }
        ]
    }
    p = tmp_path / "n.ipynb"
    p.write_text(json.dumps(nb), encoding="utf-8")
    bt.strip_execute_results(p)
    outs = json.loads(p.read_text())["cells"][0]["outputs"]
    assert [o.get("name") for o in outs] == ["stdout"]
