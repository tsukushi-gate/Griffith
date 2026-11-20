#!/usr/bin/env python3
"""公開リファレンス「圧縮と最適化の一覧」(reference/deploy-optimizations.mdx)の図を生成する。

すべて**実データ**から描く(手で数値を置いた図は作らない):

- sweep の図 / 平坦化の図 / バッチ軸パックの図
    `examples/06_deploy_c/export_model.py` と同じ手順(seed 0)で XOR を学習して IR を作り、
    `ir.sweep.sweep` と `deploy.flatten_dense.flatten_dense_ir` をそのまま掛けた結果を描く
    (32 → 22 → 実効 3 ゲート)。
- conv の図
    顔検出 `det_face_lut_k256.json` の conv2d ノード `n6_lutconv2d` の出力チャネル 13 の木を、
    `export.optimize.build_conv_program`(命令列化)と `export.dc_resynth.resynth_tree`(再合成)
    に通した結果を描く。下限の棒グラフは同モデルの全 2,112 本の木の集計。
- Web / GPU
    Web の実測は `results/deploy/k79_web_lut_flat_20261011.json`、Metal の実測は
    `results/deploy/k79_metal_lut_flat_r2_20261011.json`(どちらも gitignore)。段分けの図は
    HAR k600/k2400/k6000 のモデル JSON から bundle(--compress fast)→ flatten_dense →
    flatlevels.levelize_lut_flat をその場で計算する(モデルが無ければ公開ページの値)。
- 実測の棒グラフ
    ESP32 は `results/deploy/devices/esp32_summary_20261007.json`、バッチ軸パックは
    `results/deploy/flat_bitpar_simd_20261010_2220.json`(gitignore)。ファイルが無い環境では、
    公開ページに載っている同じ値(下の *_FALLBACK)で描く。ファイルがあれば一致を確かめる。

実行:
    uv run python scripts/gen_deploy_opt_figures.py
    uv run python scripts/gen_deploy_opt_figures.py --face-model path/to/det_face_lut_k256.json

出力: docs-site/src/content/docs/reference/deploy-optimizations_files/*.png
(DPI は scripts/build_tutorials.py の FIGURE_DPI。図中の文字は英字のみ = docs-authoring §2)

wandb への記録: 図は公開ドキュメント用の再生成物で、実験の成果物ではない。資格情報が無い環境でも
動くよう、このスクリプトは wandb に書かない(元データの run は各 results/ の JSON 側に記録がある)。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Rectangle  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from build_tutorials import FIGURE_DPI  # noqa: E402

OUT_DIR = (
    ROOT / "docs-site" / "src" / "content" / "docs" / "reference" / "deploy-optimizations_files"
)
# webbench のモデル置き場(gitignore 済み。モデル JSON は各自が置く)。
# 別の場所にあるときは --models-dir / --face-model で指定する。
DEFAULT_MODELS_DIR = ROOT / "tools" / "webbench" / "public" / "models"
DEFAULT_FACE = DEFAULT_MODELS_DIR / "det_face_lut_k256.json"
ESP32_JSON = ROOT / "results" / "deploy" / "devices" / "esp32_summary_20261007.json"
BITPAR_JSON = ROOT / "results" / "deploy" / "flat_bitpar_simd_20261010_2220.json"
WEB_JSON = ROOT / "results" / "deploy" / "k79_web_lut_flat_20261011.json"
METAL_JSON = ROOT / "results" / "deploy" / "k79_metal_lut_flat_r2_20261011.json"
#: HAR k600/k2400/k6000 のモデル JSON を探す場所(main() が --models-dir で上書きする)
HAR_DIR = DEFAULT_MODELS_DIR

# 色の凡例(docs-authoring §1.1 / §4 が単一情報源)
ORANGE = "#f97316"
GREEN = "#10b981"
BLUE = "#38bdf8"
GREY = "#64748b"
YELLOW = "#eab308"
RED = "#ef4444"
INK = "#1e293b"

# 公開ページ(deploy-esp32 / deploy-optimizations)に載っている値。JSON が無い環境用。
ESP32_FALLBACK = {  # model: (raw, sweep, sweep+flat) mean µs
    "HAR k600": (410.873, 342.774, 290.662),
    "HAR k2400": (5785.5, 4163.7, 782.2),
    "HAR k6000": (26598.74, 21174.9, 12687.98),
    "MNIST": (725.248, 638.074, 324.681),
}
BITPAR_FALLBACK = {  # model: (flat 1-at-a-time, flat packed) µs/sample at n=32
    "HAR k600": (1.36234, 0.8024),
    "HAR k2400": (5.58988, 1.37274),
    "HAR k6000": (16.26803, 2.74399),
    "MNIST": (3.04968, 0.56023),
}

# 公開ページ(deploy-web / deploy-optimizations)に載っている値。JSON が無い環境用。
WEB_FALLBACK = {  # model: {batch: (lut_layer js, lut_flat js, lut_flat wasm)} µs/inference
    "HAR k600": {1: (24.65, 4.70, 3.31), 32: (21.05, 1.60, 0.94)},
    "HAR k2400": {1: (80.84, 19.40, 11.12), 32: (76.03, 7.48, 3.75)},
    "HAR k6000": {1: (248.34, 46.38, 27.64), 32: (243.11, 11.34, 6.04)},
}
WEB_SOURCE = {"HAR k600": "Node", "HAR k2400": "browser", "HAR k6000": "browser"}
METAL_BATCHES = (32, 128, 256, 1024, 4096)
METAL_FALLBACK = {  # model: [(dense_fuse, dense_packed, flat_fused, flat_packed)] k inf/s
    "HAR k600": [(87, 87, 84, 93), (264, 236, 290, 290), (539, 510, 513, 564),
                 (1009, 1213, 1017, 1238), (2868, 1720, 2402, 2955)],
    "HAR k2400": [(80, 82, 84, 73), (256, 253, 233, 223), (360, 389, 367, 449),
                  (1426, 922, 1219, 880), (1720, 1089, 1733, 1509)],
    "HAR k6000": [(61, 62, 61, 63), (153, 215, 143, 143), (514, 351, 210, 242),
                  (482, 1196, 854, 510), (529, 1034, 687, 584)],
}
LEVELS_FALLBACK = {  # model: (original widths, swept widths, flat level widths, n_signals)
    "HAR k600": ([600] * 4, [442, 458, 489, 600], [316, 289, 249, 117], 1640),
    "HAR k2400": ([2400] * 4, [2086, 2068, 2127, 2400], [1505, 1415, 1224, 742], 8135),
    "HAR k6000": ([6000] * 6, [4756, 4513, 4398, 4369, 4600, 6000],
                  [3705, 3114, 2865, 2556, 2018, 866], 19415),
}

# 2 入力ゲートの名前(IR の真理値表規約: bit index = (b << 1) | a)
GATE_NAME = {
    0x0: "0",
    0x1: "NOR",
    0x2: "a&~b",
    0x3: "~b",
    0x4: "~a&b",
    0x5: "~a",
    0x6: "XOR",
    0x7: "NAND",
    0x8: "AND",
    0x9: "XNOR",
    0xA: "a",
    0xB: "a|~b",
    0xC: "b",
    0xD: "~a|b",
    0xE: "OR",
    0xF: "1",
}


def tt_eval(tt: int, a: int, b: int) -> int:
    """2 入力真理値表を 1 ビットで評価する。"""
    return (int(tt) >> ((int(b) << 1) | int(a))) & 1


def tt_word(tt: int, a: int, b: int, mask: int = 0xFFFFFFFF) -> int:
    """2 入力真理値表を 32 ビットワード単位で評価する(バッチ軸パックの演算そのもの)。"""
    na, nb = ~a & mask, ~b & mask
    out = 0
    if tt & 1:
        out |= na & nb
    if tt & 2:
        out |= a & nb
    if tt & 4:
        out |= na & b
    if tt & 8:
        out |= a & b
    return out & mask


def style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": FIGURE_DPI,
            "savefig.dpi": FIGURE_DPI,
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "axes.edgecolor": "#94a3b8",
            "axes.labelcolor": INK,
            "text.color": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def save(fig, name: str) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / name
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    from PIL import Image

    with Image.open(path) as im:
        w, h = im.size
    assert w >= 1000, f"{name}: 幅 {w}px < 1000px"
    print(f"  wrote {path.relative_to(ROOT)} ({w}x{h})")
    return path


def box(ax, x, y, w, h, *, fc, ec, ls="-", lw=1.2, alpha=0.18, z=3):
    """半透明の塗り + 不透明の枠線(Mermaid と同じ配色規約)。"""
    ax.add_patch(
        FancyBboxPatch(
            (x - w / 2, y - h / 2),
            w,
            h,
            boxstyle="round,pad=0.02,rounding_size=0.08",
            fc=matplotlib.colors.to_rgba(fc, alpha),
            ec=ec,
            ls=ls,
            lw=lw,
            zorder=z,
        )
    )


# ============================================================
# XOR モデル(examples/06_deploy_c/export_model.py と同一手順)
# ============================================================


def build_xor():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    from griffith_logic import GroupSum, LutLayer
    from griffith_logic.export import export_ir
    from griffith_logic.ir import validate

    torch.manual_seed(0)
    x = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
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
    model.eval()
    ir = export_ir(model, input_width=2, name="xor")
    validate(ir, target="hdl")
    return ir


def lut_layers(graph):
    from griffith_logic.ir.serialize import tt_from_hex

    out = []
    for n in graph.nodes:
        if n.type == "lut_layer":
            idx = np.asarray(n.params["indices"], dtype=np.int64)
            out.append((idx, [int(t) for t in tt_from_hex(n.params["truth_tables"])]))
    return out


def eval_layers(layers, x):
    """層ごとの出力ビット列を返す(x は入力ビット列)。"""
    vals = [list(x)]
    for idx, tts in layers:
        prev = vals[-1]
        vals.append([tt_eval(tts[o], prev[idx[0, o]], prev[idx[1, o]]) for o in range(len(tts))])
    return vals


def classify_sweep(layers):
    """元の 2 層 XOR 回路の第 1 層ゲートを、sweep で何が起きるかで分類する。

    `ir/sweep.py` のパス(縮退スロットの束縛 → 定数吸収 → CSE → 未参照除去)を同じ規則で
    1 回ずつ当てたもの。結果は呼び出し側で本物の sweep の出力と突き合わせる。
    """
    from griffith_logic.ir.sweep import (
        _canonical_gate,
        _slot_used,
        _specialize_slot,
        _tie_slots,
        _used_slots,
    )

    (idx1, tts1), (idx2, tts2) = layers
    n = 2
    tied1 = []
    for o, tt in enumerate(tts1):
        if idx1[0, o] == idx1[1, o] and _slot_used(tt, 1, n):
            tt = _tie_slots(tt, 1, 0, n)
        tied1.append(tt)
    const = {o: tt & 1 for o, tt in enumerate(tied1) if not _used_slots(tt, n)}
    canon, dup_of = {}, {}
    for o, tt in enumerate(tied1):
        if o in const:
            continue
        key = _canonical_gate([int(v) for v in idx1[:, o]], tt, n)
        if key in canon:
            dup_of[o] = canon[key]
        else:
            canon[key] = o
    referenced = set()
    new_tts2 = []
    for o, tt in enumerate(tts2):
        w = [dup_of.get(int(v), int(v)) for v in idx2[:, o]]
        for j in range(n):
            if w[j] in const and _slot_used(tt, j, n):
                tt = _specialize_slot(tt, j, const[w[j]], n)
        if w[0] == w[1] and _slot_used(tt, 1, n):
            tt = _tie_slots(tt, 1, 0, n)
        for j in _used_slots(tt, n):
            referenced.add(w[j])
        new_tts2.append(tt)
    role = {}
    for o in range(len(tts1)):
        if o in const:
            same = " (a=b)" if idx1[0, o] == idx1[1, o] else ""
            role[o] = ("const", f"= {const[o]}{same}")
        elif o in dup_of:
            role[o] = ("cse", f"= g{dup_of[o]}")
        elif o not in referenced:
            role[o] = ("unref", "unused")
        else:
            role[o] = ("keep", "")
    return role, new_tts2


def reducer_outcomes(netlist):
    """`netlist_reduce.reduce_netlist` が各ゲートをどう畳んだか(const/wire/cse/new)を記録する。

    縮約器は結果を集計値でしか返さないので、モジュール内の `dict(...)` 呼び出しを
    記録付きの dict に差し替えて、カウンタの増分をゲート順に拾う(ライブラリは無改造)。
    """
    from griffith_logic.export import netlist_reduce as nr

    log: list[str] = []
    keys = {
        "const_folded": "const",
        "passthrough_folded": "wire",
        "cse_merged": "cse",
        "new_gates": "new",
    }

    class Tracking(dict):
        def __setitem__(self, k, v):
            if k in keys and "native_const_ops" in self:
                log.append(keys[k])
            super().__setitem__(k, v)

    orig = nr.__dict__.get("dict")
    nr.dict = Tracking
    try:
        res = nr.reduce_netlist(netlist)
    finally:
        if orig is None:
            del nr.dict
        else:
            nr.dict = orig
    total = sum(len(layer) for layer in netlist["layers"])
    assert len(log) == total, (len(log), total)
    return res, log


def prefix_refs(netlist):
    """各層のゲートが縮約後にどの参照(定数/入力/ゲート + 極性)になったかを返す。"""
    from griffith_logic.export.netlist_reduce import reduce_netlist

    refs = []
    for li in range(len(netlist["layers"])):
        w = len(netlist["layers"][li])
        sub = {
            "input_bits": netlist["input_bits"],
            "layers": netlist["layers"][: li + 1],
            "output": {"classes": 1, "group_size": w, "tau": 1.0},
        }
        refs.append(reduce_netlist(sub).final_refs)
    return refs


# ---------------- 層状回路の描画 ----------------

ROLE_STYLE = {
    "keep": dict(fc=GREEN, ec=GREEN, ls="-"),
    "pass": dict(fc=GREEN, ec=GREEN, ls=(0, (3, 2)), alpha=0.06),
    "const": dict(fc=GREY, ec=GREY, ls="-", alpha=0.35),
    "cse": dict(fc=YELLOW, ec="#a16207", ls="-", alpha=0.35),
    "unref": dict(fc="white", ec=RED, ls=(0, (3, 2)), alpha=0.0),
    "wire": dict(fc=BLUE, ec="#0284c7", ls=(0, (3, 2)), alpha=0.18),
    "new": dict(fc=GREEN, ec=GREEN, ls="-", alpha=0.25),
}


def draw_layered(ax, layers, x_bits, roles, notes, *, title, k=2, faded=None):
    """入力 → 層 → group_sum を描く。roles[(li, o)] で塗り分け、notes で注記。"""
    faded = faded or set()
    vals = eval_layers(layers, x_bits)
    xs = [0.0] + [1.75 * (i + 1) for i in range(len(layers))]
    xs.append(xs[-1] + 1.75)
    gw, gh = 1.2, 0.8
    max_w = max(len(t) for _, t in layers)

    def ypos(i, width):
        return (max_w - 1) / 2 - (i - (width - 1) / 2) * 1.0

    pos = {}
    for i in range(len(x_bits)):
        pos[(0, i)] = (xs[0], (max_w - 1) / 2 - (i - (len(x_bits) - 1) / 2) * 2.4)
    for li, (_idx, tts) in enumerate(layers, start=1):
        for o in range(len(tts)):
            pos[(li, o)] = (xs[li], ypos(o, len(tts)))
    # 配線
    for li, (idx, tts) in enumerate(layers, start=1):
        for o in range(len(tts)):
            x1, y1 = pos[(li, o)]
            for j in range(2):
                src = int(idx[j, o])
                if not _depends(tts[o], j):
                    continue
                if j == 1 and idx[0, o] == idx[1, o]:
                    continue
                x0, y0 = pos[(li - 1, src)]
                dead = (li, o) in faded or (li - 1, src) in faded
                ax.plot(
                    [x0 + gw / 2 if li > 1 else x0 + 0.32, x1 - gw / 2],
                    [y0, y1],
                    color="#cbd5e1" if dead else "#64748b",
                    lw=0.55 if dead else 0.7,
                    ls=(0, (2, 2)) if dead else "-",
                    zorder=1,
                )
    # 入力
    for i, b in enumerate(x_bits):
        x0, y0 = pos[(0, i)]
        box(ax, x0, y0, 0.62, 0.8, fc=BLUE, ec="#0284c7", alpha=0.22)
        ax.text(
            x0, y0 + 0.12, f"x{i}", ha="center", va="center", fontsize=11.2, weight="bold", zorder=5
        )
        ax.text(x0, y0 - 0.2, f"={b}", ha="center", va="center", fontsize=10.0, zorder=5)
    # ゲート
    for li, (_idx, tts) in enumerate(layers, start=1):
        for o, tt in enumerate(tts):
            x0, y0 = pos[(li, o)]
            role = roles.get((li, o), "keep")
            st = dict(ROLE_STYLE[role])
            alpha = st.pop("alpha", 0.18)
            box(ax, x0, y0, gw, gh, alpha=alpha, lw=1.1, **st)
            ax.text(
                x0 - gw / 2 + 0.06,
                y0 + 0.13,
                f"g{o}",
                ha="left",
                va="center",
                fontsize=7.8,
                color="#475569",
                zorder=5,
            )
            ax.text(
                x0 + 0.06,
                y0 + 0.13,
                GATE_NAME[tt],
                ha="center",
                va="center",
                fontsize=9.2,
                weight="bold",
                zorder=5,
            )
            note = notes.get((li, o), "")
            ax.text(
                x0,
                y0 - 0.15,
                note,
                ha="center",
                va="center",
                fontsize=7.5,
                color="#334155",
                zorder=5,
            )
            ax.text(
                x0 + gw / 2 - 0.05,
                y0 + 0.13,
                str(vals[li][o]),
                ha="right",
                va="center",
                fontsize=8.2,
                color="#0369a1",
                weight="bold",
                zorder=5,
            )
            if role == "unref":
                ax.plot(
                    [x0 - gw / 2 + 0.05, x0 + gw / 2 - 0.05],
                    [y0 - gh / 2 + 0.05, y0 + gh / 2 - 0.05],
                    color=RED,
                    lw=0.8,
                    alpha=0.6,
                    zorder=4,
                )
    # group_sum
    last = vals[-1]
    gsz = len(last) // k
    xg = xs[-1]
    for c in range(k):
        ys = [pos[(len(layers), o)][1] for o in range(c * gsz, (c + 1) * gsz)]
        top, bot = max(ys) + 0.35, min(ys) - 0.35
        for o in range(c * gsz, (c + 1) * gsz):
            x0, y0 = pos[(len(layers), o)]
            ax.plot(
                [x0 + gw / 2, xg - 0.55], [y0, (top + bot) / 2], color="#cbd5e1", lw=0.5, zorder=0
            )
        box(ax, xg, (top + bot) / 2, 1.0, 1.6, fc=BLUE, ec="#0284c7", alpha=0.18)
        ax.text(
            xg,
            (top + bot) / 2 + 0.35,
            f"class {c}",
            ha="center",
            va="center",
            fontsize=10.6,
            weight="bold",
            zorder=5,
        )
        ax.text(
            xg,
            (top + bot) / 2 - 0.15,
            f"score {sum(last[c * gsz : (c + 1) * gsz])}",
            ha="center",
            va="center",
            fontsize=10.6,
            zorder=5,
        )
    ax.text(
        xg,
        ypos(0, max_w) + 1.0,
        "GroupSum\n(k=2)",
        ha="center",
        va="bottom",
        fontsize=10.0,
        color="#475569",
    )
    for li in range(1, len(layers) + 1):
        ax.text(
            xs[li],
            ypos(0, max_w) + 1.0,
            f"LutLayer {li}\n({len(layers[li - 1][1])} gates)",
            ha="center",
            va="bottom",
            fontsize=10.0,
            color="#475569",
        )
    ax.text(
        xs[0],
        ypos(0, max_w) + 1.0,
        "input\nbits",
        ha="center",
        va="bottom",
        fontsize=10.0,
        color="#475569",
    )
    ax.set_xlim(-0.6, xg + 0.7)
    ax.set_ylim(ypos(max_w - 1, max_w) - 0.8, ypos(0, max_w) + 2.0)
    ax.set_title(title, loc="left")
    ax.axis("off")
    return vals


def _depends(tt: int, j: int) -> bool:
    for addr in range(4):
        if ((tt >> addr) & 1) != ((tt >> (addr ^ (1 << j))) & 1):
            return True
    return False


def legend_patches(ax, items, *, loc="upper center", ncol=None, anchor=(0.5, 0.0)):
    from matplotlib.patches import Patch

    handles = []
    for label, role in items:
        st = dict(ROLE_STYLE[role]) if role in ROLE_STYLE else role
        alpha = st.pop("alpha", 0.18)
        handles.append(
            Patch(
                facecolor=matplotlib.colors.to_rgba(st["fc"], max(alpha, 0.06)),
                edgecolor=st["ec"],
                linestyle=st.get("ls", "-"),
                label=label,
            )
        )
    ax.legend(
        handles=handles,
        loc=loc,
        ncol=ncol or min(len(items), 2),
        frameon=False,
        fontsize=10.5,
        bbox_to_anchor=anchor,
    )


# ============================================================
# 図: sweep(XOR 32 → 22)
# ============================================================


def fig_sweep(ir, x_example):
    from griffith_logic.ir.simulate import simulate
    from griffith_logic.ir.sweep import sweep

    layers = lut_layers(ir)
    swept, rep = sweep(ir)
    s_layers = lut_layers(swept)
    role, new_tts2 = classify_sweep(layers)
    # 本物の sweep と突き合わせる(図が sweep の結果そのものであることの検算)
    kept = [o for o, (r, _) in role.items() if r == "keep"]
    assert len(kept) == len(s_layers[0][1]), (kept, s_layers[0][1])
    assert new_tts2 == s_layers[1][1], (new_tts2, s_layers[1][1])
    assert rep["gates_before"] == 32 and rep["gates_after"] == 22, rep
    bits = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.uint8)
    assert np.array_equal(simulate(ir, bits), simulate(swept, bits))

    roles = {(1, o): r for o, (r, _) in role.items()}
    notes = {(1, o): t for o, (r, t) in role.items() if t}
    faded = {(1, o) for o, (r, _) in role.items() if r != "keep"}
    s_roles, s_notes = {}, {}
    from griffith_logic.ir.sweep import _used_slots

    for o, tt in enumerate(s_layers[1][1]):
        nu = len(_used_slots(tt, 2))
        if nu < 2:
            s_roles[(2, o)] = "pass"
            s_notes[(2, o)] = "1-input" if nu == 1 else "const"
    for o, tt in enumerate(s_layers[0][1]):
        if len(_used_slots(tt, 2)) < 2:
            s_roles[(1, o)] = "pass"
            s_notes[(1, o)] = "1-input"
    n_const = sum(1 for r, _ in role.values() if r == "const")
    n_cse = sum(1 for r, _ in role.values() if r == "cse")
    n_unref = sum(1 for r, _ in role.values() if r == "unref")

    fig, axes = plt.subplots(1, 2, figsize=(15, 8.2), gridspec_kw={"wspace": 0.04})
    vb = draw_layered(
        axes[0],
        layers,
        x_example,
        roles,
        notes,
        faded=faded,
        title=f"Before sweep: {rep['gates_before']} gates (trained XOR, seed 0)",
    )
    va = draw_layered(
        axes[1],
        s_layers,
        x_example,
        s_roles,
        s_notes,
        title=f"After sweep: {rep['gates_after']} gates (LutLayer 1: 16 -> {len(s_layers[0][1])})",
    )
    assert vb[-1] == va[-1]  # 最終層ビット列(=スコア)が一致
    items = [
        ("kept", "keep"),
        (f"constant -> folded into LutLayer 2, removed ({n_const})", "const"),
        (f"duplicate -> merged by CSE, removed ({n_cse})", "cse"),
    ]
    if n_unref:
        items.append((f"unreferenced -> removed ({n_unref})", "unref"))
    legend_patches(axes[0], items, ncol=1)
    legend_patches(
        axes[1],
        [
            ("2-input gate", "keep"),
            (
                f"1-input / constant gate left in the layer ({rep['passthrough_remaining']}+"
                f"{rep['const_remaining']}; dense-flat folds them)",
                "pass",
            ),
        ],
        ncol=1,
    )
    fig.suptitle(
        f"sweep on a real circuit: 32 -> 22 gates, same scores for x = ({x_example[0]}, {x_example[1]})"  # noqa: E501
        "   (blue digit = gate output for this input)",
        fontsize=14.4,
        weight="bold",
        y=0.995,
    )
    save(fig, "sweep-xor.png")
    return swept, rep, {"const": n_const, "cse": n_cse, "unref": n_unref}


# ============================================================
# 図: 平坦化(XOR 22 → 実効 3)+ ESP32 実測
# ============================================================


def sig_name(ref, in_w, new_index):
    kind, idx, pol = ref
    neg = "~" if pol else ""
    if kind == "c":
        return f"const {idx}"
    if kind == "i":
        return f"{neg}s{1 + idx}"
    if idx not in new_index:
        return f"{neg}(dropped)"
    return f"{neg}s{1 + in_w + new_index[idx]}"


def fig_dense_flat(swept, x_example):
    from griffith_logic.deploy.flatten_dense import flatten_dense_ir
    from griffith_logic.export.netlist_reduce import flat_netlist_from_ir
    from griffith_logic.ir.simulate import simulate

    layers = lut_layers(swept)
    flat_g, frep = flatten_dense_ir(swept)
    lf = next(n for n in flat_g.nodes if n.type == "lut_flat").params
    assert frep["gates_before"] == 22 and frep["gates_after"] == 3, frep
    netlist = flat_netlist_from_ir(swept)
    res, log = reducer_outcomes(netlist)
    refs = prefix_refs(netlist)
    in_w = netlist["input_bits"]
    new_index = {}
    for n in range(len(res.gates)):
        if res.reach[n]:
            new_index[n] = len(new_index)

    roles, notes = {}, {}
    k = 0
    for li, (_idx, tts) in enumerate(layers, start=1):
        for o in range(len(tts)):
            oc = log[k]
            k += 1
            ref = refs[li - 1][o]
            roles[(li, o)] = {"const": "const", "wire": "wire", "cse": "cse", "new": "new"}[oc]
            name = sig_name(ref, in_w, new_index)
            notes[(li, o)] = {
                "const": name,
                "wire": f"wire {name}",
                "cse": f"= {name}",
                "new": f"-> {name}",
            }[oc]

    fig = plt.figure(figsize=(16, 8.4))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.12, 1.0], wspace=0.05)
    ax0 = fig.add_subplot(gs[0])
    vals = draw_layered(
        ax0,
        layers,
        x_example,
        roles,
        notes,
        title=f"Layered lut_layer network after sweep: {frep['gates_before']} gates",
    )
    counts = {r: log.count(r) for r in ("const", "wire", "cse", "new")}
    legend_patches(
        ax0,
        [
            (f"real 2-input gate ({counts['new']})", "new"),
            (f"pass-through / NOT -> wire ({counts['wire']})", "wire"),
            (f"duplicate -> same signal ({counts['cse']})", "cse"),
            (f"constant ({counts['const']})", "const"),
        ],
    )

    # 右: 平坦ネットリスト(信号表 + 極性ビット付きの出力参照)
    ax = fig.add_subplot(gs[1])
    ax.set_title(
        f"Flat netlist (lut_flat): {len(lf['gates'])} effective gates + polarity bits", loc="left"
    )
    ax.axis("off")
    sig_vals = [0] + list(x_example)
    for tt, a, b in lf["gates"]:
        sig_vals.append(tt_eval(tt, sig_vals[a], sig_vals[b]))
    n_sig = len(sig_vals)
    sy = {s: 7.5 - s * 1.55 for s in range(n_sig)}
    sx = 1.2
    labels = ["const 0", "input x0", "input x1"]
    for tt, a, b in lf["gates"]:
        labels.append(
            f"{GATE_NAME[tt]}(s{a}, s{b})" if tt in GATE_NAME else f"tt{tt:#x}(s{a}, s{b})"
        )
    for s in range(n_sig):
        kind = "const" if s == 0 else ("in" if s <= in_w else "gate")
        fc, ec = {"const": (GREY, GREY), "in": (BLUE, "#0284c7"), "gate": (GREEN, GREEN)}[kind]
        box(ax, sx, sy[s], 2.1, 0.95, fc=fc, ec=ec, alpha=0.22)
        ax.text(
            sx - 0.95,
            sy[s] + 0.2,
            f"s{s}",
            ha="left",
            va="center",
            fontsize=11.2,
            weight="bold",
            zorder=5,
        )
        ax.text(
            sx + 0.15, sy[s] + 0.2, labels[s], ha="center", va="center", fontsize=10.0, zorder=5
        )
        ax.text(
            sx,
            sy[s] - 0.22,
            f"value {sig_vals[s]}",
            ha="center",
            va="center",
            fontsize=9.4,
            color="#0369a1",
            zorder=5,
        )
    for gi, (_tt, a, b) in enumerate(lf["gates"]):
        s = 1 + in_w + gi
        for src, dy in ((a, 0.15), (b, -0.15)):
            ax.annotate(
                "",
                xy=(sx - 1.05, sy[s] + dy),
                xytext=(sx - 1.05, sy[src] + dy),
                arrowprops=dict(
                    arrowstyle="-|>", color="#64748b", lw=0.7, connectionstyle="arc3,rad=0.45"
                ),
                zorder=2,
            )
    ax.text(
        sx,
        9.0,
        "signal table (topological order)",
        ha="center",
        va="center",
        fontsize=10.6,
        color="#475569",
    )
    # 出力参照
    outs = lf["outputs"]
    ox = 5.6
    oy = {o: 8.2 - o * 0.95 for o in range(len(outs))}
    out_bits = []
    for o, code in enumerate(outs):
        s, pol = code >> 1, code & 1
        bit = sig_vals[s] ^ pol
        out_bits.append(bit)
        ax.plot(
            [sx + 1.05, ox - 0.87],
            [sy[s], oy[o]],
            color=RED if pol else "#64748b",
            lw=0.75,
            alpha=0.85,
            zorder=1,
        )
        if pol:
            ax.add_patch(plt.Circle((ox - 0.93, oy[o]), 0.07, fc="white", ec=RED, lw=1.0, zorder=4))
        box(ax, ox, oy[o], 1.6, 0.7, fc=BLUE, ec="#0284c7", alpha=0.12)
        ref = ("~" if pol else "") + (f"s{s}" if not (s == 0) else "const")
        if s == 0:
            ref = "const"
        ax.text(
            ox - 0.74,
            oy[o],
            f"out{o}",
            ha="left",
            va="center",
            fontsize=8.5,
            color="#475569",
            zorder=5,
        )
        ax.text(ox + 0.74, oy[o], f"{ref}={bit}", ha="right", va="center", fontsize=8.5, zorder=5)
    assert out_bits == vals[-1], (out_bits, vals[-1])
    bits = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.uint8)
    assert np.array_equal(simulate(swept, bits), simulate(flat_g, bits))
    for c in range(2):
        lo, hi = c * 8, (c + 1) * 8
        ym = (oy[lo] + oy[hi - 1]) / 2
        ax.plot([ox + 0.85, ox + 1.05], [oy[lo] + 0.3, oy[lo] + 0.3], color="#94a3b8", lw=0.8)
        ax.plot([ox + 1.05, ox + 1.05], [oy[lo] + 0.3, oy[hi - 1] - 0.3], color="#94a3b8", lw=0.8)
        ax.plot(
            [ox + 0.85, ox + 1.05], [oy[hi - 1] - 0.3, oy[hi - 1] - 0.3], color="#94a3b8", lw=0.8
        )
        box(ax, ox + 1.9, ym, 1.3, 1.3, fc=BLUE, ec="#0284c7", alpha=0.18)
        ax.text(
            ox + 1.9,
            ym + 0.25,
            f"class {c}",
            ha="center",
            va="center",
            fontsize=10.6,
            weight="bold",
        )
        ax.text(
            ox + 1.9,
            ym - 0.2,
            f"score {sum(out_bits[lo:hi])}",
            ha="center",
            va="center",
            fontsize=10.6,
        )
    ax.text(
        ox,
        9.0,
        "16 output bits = (signal, polarity)",
        ha="center",
        va="center",
        fontsize=10.6,
        color="#475569",
    )
    ax.text(
        0.2,
        -7.3,
        "red line + bubble = inverted reference (polarity bit 1): a NOT costs no gate",
        fontsize=10.2,
        color=RED,
    )
    ax.set_xlim(-0.2, 8.5)
    ax.set_ylim(-7.8, 9.6)
    fig.suptitle(
        f"dense-flat on the same XOR circuit: {frep['gates_before']} -> {frep['gates_after']} gates "  # noqa: E501
        f"(original 32 -> 3), scores unchanged for x = ({x_example[0]}, {x_example[1]})",
        fontsize=14.4,
        weight="bold",
        y=0.995,
    )
    save(fig, "dense-flat-xor.png")
    return frep, counts


def load_esp32():
    data = dict(ESP32_FALLBACK)
    if ESP32_JSON.exists():
        d = json.loads(ESP32_JSON.read_text(encoding="utf-8"))
        name = {
            "har_dlgn_k600": "HAR k600",
            "har_dlgn_k2400": "HAR k2400",
            "har_dlgn_k6000": "HAR k6000",
            "mnist": "MNIST",
        }
        got = {}
        for m in d["models"]:
            if m["model"] in name and m.get("mean_us") is not None:
                got.setdefault(name[m["model"]], {})[m.get("variant")] = float(m["mean_us"])
        for k_, v in got.items():
            tup = (v["raw"], v["sweep"], v["flat"])
            assert np.allclose(tup, ESP32_FALLBACK[k_], rtol=1e-6), (k_, tup)
            data[k_] = tup
        print("  ESP32: read", ESP32_JSON.relative_to(ROOT))
    return data


def fig_esp32():
    data = load_esp32()
    gates = {
        "HAR k600": (2400, 1989, 971),
        "HAR k2400": (9600, 8681, 4886),
        "HAR k6000": (36000, 28636, 15124),
        "MNIST": (4000, 3615, 2210),
    }
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.4), gridspec_kw={"wspace": 0.32})
    cols = [GREY, matplotlib.colors.to_rgba(GREEN, 0.45), GREEN]
    labels = [
        "uncompressed\n(--compress none,\n--dense-flat off)",
        "sweep only",
        "default\n(sweep + dense-flat)",
    ]
    for ax, (model, vals) in zip(axes, data.items(), strict=False):
        unit, scale = ("ms", 1e-3) if max(vals) > 10000 else ("µs", 1.0)
        v = [x * scale for x in vals]
        bars = ax.bar(range(3), v, color=cols, edgecolor=[GREY, GREEN, GREEN], width=0.68)
        for i, (b, val) in enumerate(zip(bars, v, strict=False)):
            ax.text(
                b.get_x() + b.get_width() / 2,
                val + max(v) * 0.02,
                f"{val:,.2f}" if unit == "ms" else f"{val:,.1f}",
                ha="center",
                va="bottom",
                fontsize=10.6,
            )
            ax.text(
                b.get_x() + b.get_width() / 2,
                max(v) * 0.04,
                f"{gates[model][i]:,}\ngates",
                ha="center",
                va="bottom",
                fontsize=8.8,
                color="white" if i != 1 else INK,
            )
        ax.set_title(f"{model}   x{vals[0] / vals[2]:.2f} faster", fontsize=12.5)
        ax.set_ylabel(f"latency per inference ({unit})")
        ax.set_xticks(range(3), ["none", "sweep", "default"], fontsize=10.6)
        ax.set_ylim(0, max(v) * 1.15)
        ax.spines[["top", "right"]].set_visible(False)
    from matplotlib.patches import Patch

    fig.legend(
        handles=[
            Patch(fc=c, ec=e, label=lab.replace("\n", " "))
            for c, e, lab in zip(cols, [GREY, GREEN, GREEN], labels, strict=False)
        ],
        loc="lower center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, -0.08),
        fontsize=11.2,
    )
    fig.suptitle(
        "ESP32 (Xtensa LX6, 240 MHz) measured latency, same integer scores in every variant"
        "  (each panel has its own y axis)",
        fontsize=13.8,
        weight="bold",
        y=1.02,
    )
    save(fig, "esp32-latency.png")
    return data


# ============================================================
# 図: バッチ軸パック(XOR の lut_flat を 32 サンプル同時に)+ 実測
# ============================================================


def fig_batch_pack(swept):
    from griffith_logic.deploy.flatten_dense import flatten_dense_ir
    from griffith_logic.ir.simulate import simulate

    flat_g, _ = flatten_dense_ir(swept)
    lf = next(n for n in flat_g.nodes if n.type == "lut_flat").params
    rng = np.random.default_rng(7)
    X = rng.integers(0, 2, size=(32, 2), dtype=np.uint8)
    words = [0]
    for j in range(2):
        words.append(sum(int(X[i, j]) << i for i in range(32)))
    for tt, a, b in lf["gates"]:
        words.append(tt_word(tt, words[a], words[b]))
    out_words = []
    for code in lf["outputs"]:
        s, pol = code >> 1, code & 1
        out_words.append(words[s] ^ (0xFFFFFFFF if pol else 0))
    score1 = [sum((out_words[o] >> i) & 1 for o in range(8, 16)) for i in range(32)]
    score0 = [sum((out_words[o] >> i) & 1 for o in range(0, 8)) for i in range(32)]
    ref = simulate(swept, X)
    assert np.array_equal(ref[:, 0], score0) and np.array_equal(ref[:, 1], score1)

    fig = plt.figure(figsize=(16, 7.4))
    ax = fig.add_axes([0.02, 0.02, 0.96, 0.9])
    ax.axis("off")
    ax.set_xlim(-6.5, 34.5)
    ax.set_ylim(-10.9, 6.6)
    hl = 5  # 強調するサンプル

    # (1) サンプル順(1 枚ずつ)
    ax.text(
        -6.3,
        6.1,
        "1) sample-major input (mdl_infer reads one sample at a time)",
        fontsize=13.1,
        weight="bold",
    )
    for i in range(32):
        x0 = i
        for j in range(2):
            v = int(X[i, j])
            ax.add_patch(
                Rectangle(
                    (x0 + 0.06, 4.6 - j * 0.9),
                    0.88,
                    0.82,
                    fc=matplotlib.colors.to_rgba(BLUE, 0.55 if v else 0.08),
                    ec=ORANGE if i == hl else "#0284c7",
                    lw=1.6 if i == hl else 0.5,
                )
            )
            ax.text(x0 + 0.5, 4.6 - j * 0.9 + 0.41, str(v), ha="center", va="center", fontsize=9.4)
        ax.text(x0 + 0.5, 5.6, str(i), ha="center", va="center", fontsize=8.1, color="#475569")
    ax.text(-0.4, 5.6, "sample", ha="right", va="center", fontsize=9.4, color="#475569")
    ax.text(-0.4, 5.01, "x0", ha="right", va="center", fontsize=10.0)
    ax.text(-0.4, 4.11, "x1", ha="right", va="center", fontsize=10.0)
    ax.annotate(
        "", xy=(16, 2.55), xytext=(16, 3.55), arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.2)
    )
    ax.text(
        16.4,
        3.05,
        "transpose: bit j of sample i -> bit i of word j   (bit 0 = sample 0 is drawn on the left)",
        fontsize=11.2,
        va="center",
    )

    # (2) ビット順にパックしたワード
    ax.text(
        -6.3,
        2.2,
        "2) bit-major packed words (mdl_infer_batch_packed): one uint32 = one signal for 32 samples",  # noqa: E501
        fontsize=13.1,
        weight="bold",
    )
    rows = [("x0", words[1], BLUE, "input"), ("x1", words[2], BLUE, "input")]
    gate_rows = []
    for gi, (tt, a, b) in enumerate(lf["gates"]):
        s = 3 + gi
        nm = GATE_NAME.get(tt, f"tt{tt:#x}")
        gate_rows.append((f"s{s} = {nm}(s{a}, s{b})", words[s], GREEN, "gate"))
    y = 1.0

    def word_row(y, label, w, color, *, lw_lab=6.2):
        ax.text(-0.4, y + 0.41, label, ha="right", va="center", fontsize=10.2)
        for i in range(32):
            v = (w >> i) & 1
            ax.add_patch(
                Rectangle(
                    (i + 0.06, y),
                    0.88,
                    0.82,
                    fc=matplotlib.colors.to_rgba(color, 0.55 if v else 0.08),
                    ec=ORANGE if i == hl else "#94a3b8",
                    lw=1.6 if i == hl else 0.4,
                )
            )
            ax.text(i + 0.5, y + 0.41, str(v), ha="center", va="center", fontsize=9.0)
        ax.text(
            32.3, y + 0.41, f"0x{w:08X}", ha="left", va="center", fontsize=10.2, family="monospace"
        )

    for label, w, color, _ in rows:
        word_row(y, f"word {label}", w, color)
        y -= 1.0
    ax.text(
        -6.3,
        y - 0.15,
        "3) each gate of the lut_flat signal table = ONE 32-bit word operation for all 32 samples",
        fontsize=13.1,
        weight="bold",
    )
    y -= 1.35
    for label, w, color, _ in gate_rows:
        word_row(y, label, w, color)
        y -= 1.0
    # 出力
    y -= 0.35
    ax.text(
        -6.3,
        y + 0.1,
        "4) outputs = signal words with polarity (XOR with 0xFFFFFFFF if inverted), then popcount per class",  # noqa: E501
        fontsize=13.1,
        weight="bold",
    )
    y -= 1.25
    for c, sc in ((0, score0), (1, score1)):
        ax.text(-0.4, y + 0.41, f"class {c} score", ha="right", va="center", fontsize=10.2)
        for i in range(32):
            ax.add_patch(
                Rectangle(
                    (i + 0.06, y),
                    0.88,
                    0.82,
                    fc=matplotlib.colors.to_rgba(BLUE, 0.06 + 0.06 * sc[i]),
                    ec=ORANGE if i == hl else "#94a3b8",
                    lw=1.6 if i == hl else 0.4,
                )
            )
            ax.text(i + 0.5, y + 0.41, str(sc[i]), ha="center", va="center", fontsize=9.0)
        y -= 1.0
    ax.text(
        hl + 0.5,
        y + 0.55,
        f"sample {hl}: x=({X[hl, 0]},{X[hl, 1]}) -> scores ({score0[hl]}, {score1[hl]})",
        ha="left",
        va="top",
        fontsize=10.6,
        color=ORANGE,
        weight="bold",
    )
    ax.text(
        -6.3,
        -9.6,
        va="top",
        s=f"Real data: the XOR model's lut_flat ({len(lf['gates'])} gates), 32 random inputs (seed 7); "  # noqa: E501
        "scores equal ir.simulate for every sample.\nFor HAR (4,608 input bits) the same transpose makes "  # noqa: E501
        "4,608 words, and every one of the 15,124 gates (k6000) runs once per 32 windows.",
        fontsize=10.8,
        color="#334155",
    )
    fig.suptitle(
        "Batch packing: 32 samples share every instruction", fontsize=15.0, weight="bold", y=0.985
    )
    save(fig, "batch-pack.png")


def load_bitpar():
    data = dict(BITPAR_FALLBACK)
    if BITPAR_JSON.exists():
        d = json.loads(BITPAR_JSON.read_text(encoding="utf-8"))
        name = {
            "har_dlgn_k600_gumbel.json": "HAR k600",
            "har_dlgn_k2400_gumbel.json": "HAR k2400",
            "har_dlgn_k6000_gumbel.json": "HAR k6000",
            "mnist.bundle.json": "MNIST",
        }
        for m in d["models"]:
            r = next(x for x in m["rows"] if x["n"] == 32)
            tup = (float(r["flat-1_us"]), float(r["flat-pk_us"]))
            assert np.allclose(tup, BITPAR_FALLBACK[name[m["model"]]], rtol=1e-6), (m["model"], tup)
            data[name[m["model"]]] = tup
        print("  bitpar: read", BITPAR_JSON.relative_to(ROOT))
    return data


def fig_batch_bars():
    data = load_bitpar()
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.2), gridspec_kw={"wspace": 0.32})
    cols = [GREY, GREEN]
    for ax, (model, (one, pk)) in zip(axes, data.items(), strict=False):
        bars = ax.bar(
            [0, 1],
            [one, pk],
            color=[matplotlib.colors.to_rgba(GREY, 0.75), GREEN],
            edgecolor=cols,
            width=0.62,
        )
        for b, val in zip(bars, (one, pk), strict=False):
            ax.text(
                b.get_x() + b.get_width() / 2,
                val + one * 0.02,
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontsize=11.2,
            )
        ax.set_title(f"{model}   x{one / pk:.1f} throughput", fontsize=12.5)
        ax.set_xticks(
            [0, 1], ["mdl_infer\n(1 at a time)", "batch_packed\n(32 per call)"], fontsize=10.6
        )
        ax.set_ylabel("µs per sample (n = 32)")
        ax.set_ylim(0, one * 1.18)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(
        "Host (Apple M3) lut_flat, µs per sample for 32 samples: one at a time vs packed "
        "(each panel has its own y axis)",
        fontsize=13.8,
        weight="bold",
        y=1.03,
    )
    save(fig, "batch-throughput.png")
    return data


# ============================================================
# 図: conv の木 → 命令列 → 再合成(顔検出 n6_lutconv2d / oc 13)
# ============================================================


def tap_label(pp, params):
    kh, kw = params["kernel_size"]
    c, rem = divmod(int(pp), kh * kw)
    dy, dx = divmod(rem, kw)
    return f"ch{c} ({dy},{dx})"


def tree_roles(leaves, tts, depth):
    """木の各ゲートが命令列化で何になるか(instr / alias / NOT / const / dead)を求める。"""
    n_leaves = 1 << depth
    # ゲート番号: レベル昇順。子の参照を作る
    children = {}
    level_nodes = [[("leaf", k) for k in range(n_leaves)]]
    g = 0
    while len(level_nodes[-1]) > 1:
        cur = level_nodes[-1]
        nxt = []
        for i in range(len(cur) >> 1):
            children[g] = (cur[2 * i], cur[2 * i + 1])
            nxt.append(("gate", g))
            g += 1
        level_nodes.append(nxt)
    root = level_nodes[-1][0][1]
    # 使われるスロットを辿る(同一入力・退化 tt を考慮)
    used_gates = set()

    def leaf_id(node):
        return ("leaf", leaves[node[1]]) if node[0] == "leaf" else node

    def visit(gi):
        used_gates.add(gi)
        a, b = children[gi]
        tt = tts[gi]
        dep = [_depends(tt, 0), _depends(tt, 1)]
        if leaf_id(a) == leaf_id(b):
            dep = [dep[0] or dep[1], False]
        for node, d in zip((a, b), dep, strict=False):
            if d and node[0] == "gate":
                visit(node[1])

    visit(root)
    roles = {}
    for gi in range(g):
        a, b = children[gi]
        tt = tts[gi]
        dep = [_depends(tt, 0), _depends(tt, 1)]
        same = leaf_id(a) == leaf_id(b)
        if gi not in used_gates:
            roles[gi] = "const" if tt in (0x0, 0xF) else "dead"
        elif not any(dep):
            roles[gi] = "const"
        elif same or not all(dep):
            single = tt if not same else None
            roles[gi] = "alias" if (single in (0xA, 0xC)) else "not1"
        else:
            roles[gi] = "instr"
    return children, roles, root


def eval_tree_bits(leaves, tts, depth, assign):
    cur = [assign[pp] for pp in leaves]
    off = 0
    out_all = []
    while len(cur) > 1:
        nxt = [tt_eval(tts[off + i], cur[2 * i], cur[2 * i + 1]) for i in range(len(cur) >> 1)]
        out_all.extend(nxt)
        off += len(nxt)
        cur = nxt
    return cur[0], out_all


def eval_program(prog, assign):
    from griffith_logic.export.optimize import REF_KIND_CONST, REF_KIND_TAP, ref_idx, ref_kind

    vals = []

    def val(r):
        k = ref_kind(r)
        if k == REF_KIND_CONST:
            return ref_idx(r)
        if k == REF_KIND_TAP:
            return assign[prog.taps[ref_idx(r)]]
        return vals[ref_idx(r)]

    for ins in prog.instrs:
        vals.append(tt_eval(ins.tt, val(ins.a), val(ins.b)))
    return val(prog.oc_out[0]), vals


def ref_str(prog, r, tap_name):
    from griffith_logic.export.optimize import REF_KIND_CONST, REF_KIND_TAP, ref_idx, ref_kind

    k = ref_kind(r)
    if k == REF_KIND_CONST:
        return str(ref_idx(r))
    if k == REF_KIND_TAP:
        return tap_name[prog.taps[ref_idx(r)]]
    return f"i{ref_idx(r)}"


def draw_tree(ax, leaves, tts, depth, roles, tap_name, assign, title, params):
    n_leaves = 1 << depth
    _, gate_vals = eval_tree_bits(leaves, tts, depth, assign)
    pos = {}
    for k in range(n_leaves):
        pos[("leaf", k)] = (k * 1.25, 0.0)
    g = 0
    xs = [k * 1.25 for k in range(n_leaves)]
    lvl = 1
    while len(xs) > 1:
        nx = []
        for i in range(len(xs) >> 1):
            x = (xs[2 * i] + xs[2 * i + 1]) / 2
            pos[("gate", g)] = (x, lvl * 1.55)
            nx.append(x)
            g += 1
        xs = nx
        lvl += 1
    children = {}
    g = 0
    cur = [("leaf", k) for k in range(n_leaves)]
    while len(cur) > 1:
        nxt = []
        for i in range(len(cur) >> 1):
            children[g] = (cur[2 * i], cur[2 * i + 1])
            nxt.append(("gate", g))
            g += 1
        cur = nxt
    style_of = {
        "instr": dict(fc=GREEN, ec=GREEN, ls="-", alpha=0.25),
        "alias": dict(fc=BLUE, ec="#0284c7", ls=(0, (3, 2)), alpha=0.15),
        "not1": dict(fc=BLUE, ec="#0284c7", ls=(0, (3, 2)), alpha=0.15),
        "const": dict(fc=GREY, ec=GREY, ls=(0, (3, 2)), alpha=0.25),
        "dead": dict(fc="white", ec=RED, ls=(0, (3, 2)), alpha=0.0),
    }
    note_of = {"instr": "", "alias": "pass", "not1": "1-input", "const": "filler", "dead": "unused"}
    for gi, (a, b) in children.items():
        x1, y1 = pos[("gate", gi)]
        dep = [_depends(tts[gi], 0), _depends(tts[gi], 1)]
        for node, d in zip((a, b), dep, strict=False):
            x0, y0 = pos[node]
            ax.plot(
                [x0, x1],
                [y0 + 0.32, y1 - 0.3],
                color="#64748b" if d else "#cbd5e1",
                lw=0.9 if d else 0.6,
                ls="-" if d else (0, (2, 2)),
                zorder=1,
            )
    for k in range(n_leaves):
        x0, y0 = pos[("leaf", k)]
        pp = leaves[k]
        box(ax, x0, y0, 1.12, 0.78, fc=BLUE, ec="#0284c7", alpha=0.2)
        ax.text(
            x0,
            y0 + 0.17,
            tap_name[pp],
            ha="center",
            va="center",
            fontsize=10.0,
            weight="bold",
            zorder=5,
        )
        ax.text(
            x0,
            y0 - 0.15,
            tap_label(pp, params).replace(" ", "\n"),
            ha="center",
            va="center",
            fontsize=6.6,
            zorder=5,
            linespacing=0.9,
        )
        ax.text(
            x0 + 0.5,
            y0 + 0.2,
            str(assign[pp]),
            ha="right",
            va="center",
            fontsize=8.1,
            color="#0369a1",
            weight="bold",
            zorder=5,
        )
    for gi in children:
        x0, y0 = pos[("gate", gi)]
        r = roles[gi]
        st = dict(style_of[r])
        alpha = st.pop("alpha")
        box(ax, x0, y0, 1.15, 0.62, alpha=alpha, **st)
        ax.text(
            x0,
            y0 + 0.11,
            GATE_NAME[tts[gi]],
            ha="center",
            va="center",
            fontsize=10.6,
            weight="bold",
            zorder=5,
        )
        ax.text(
            x0,
            y0 - 0.16,
            note_of[r],
            ha="center",
            va="center",
            fontsize=7.5,
            color="#334155",
            zorder=5,
        )
        ax.text(
            x0 + 0.52,
            y0 - 0.17,
            str(gate_vals[gi]),
            ha="right",
            va="center",
            fontsize=8.1,
            color="#0369a1",
            weight="bold",
            zorder=5,
        )
    ax.set_xlim(-0.75, (n_leaves - 1) * 1.25 + 0.75)
    ax.set_ylim(-0.6, depth * 1.55 + 0.6)
    ax.set_title(title, loc="left", fontsize=11.5)
    ax.axis("off")


def fig_conv(face_path: Path):
    from griffith_logic.export.dc_resynth import (
        _solo_cost,
        essential_vars,
        resynth_tree,
        tree_function,
    )
    from griffith_logic.export.optimize import ConvTreeSpec, build_conv_program
    from griffith_logic.ir.serialize import ir_from_dict, tt_from_hex

    doc = json.loads(face_path.read_text(encoding="utf-8"))
    g = ir_from_dict(doc["ir"])
    node_id, oc = "n6_lutconv2d", 13
    convs = [n for n in g.nodes if n.type == "conv2d"]
    # 全体の数(ページの 14,784 → 1,279 → 1,077 を再計算して確かめる)
    tot_nom = tot_ins = tot_rs = 0
    per_node = []
    for n in convs:
        p = n.params
        d = int(p["tree_depth"])
        ni = (1 << d) - 1
        tts_all = tt_from_hex(p["truth_tables"])
        cats = {"skip": 0, "nogain": 0, "gain": 0, "limit": 0}
        for o in range(int(p["out_channels"])):
            row = [int(v) for v in p["indices_flat"][o]]
            tr = [int(t) & 0xF for t in tts_all[o * ni : (o + 1) * ni]]
            taps, onset = tree_function(row, tr, d)
            e = len(essential_vars(onset, len(taps)))
            _, _, before, after = resynth_tree(row, tr, d, max_cost=4)
            tot_nom += ni
            tot_ins += before
            tot_rs += after
            if before <= max(e - 1, 0):
                cats["skip"] += 1
            elif min(4, before - 1) < e - 1:
                cats["limit"] += 1
            elif after < before:
                cats["gain"] += 1
            else:
                cats["nogain"] += 1
        per_node.append((n.id, int(p["out_channels"]), cats))
    assert (tot_nom, tot_ins, tot_rs) == (14784, 1279, 1077), (tot_nom, tot_ins, tot_rs)

    node = next(n for n in convs if n.id == node_id)
    p = node.params
    d = int(p["tree_depth"])
    ni = (1 << d) - 1
    row = [int(v) for v in p["indices_flat"][oc]]
    tr = [int(t) & 0xF for t in tt_from_hex(p["truth_tables"])[oc * ni : (oc + 1) * ni]]
    taps, onset = tree_function(row, tr, d)
    ev = essential_vars(onset, len(taps))
    tap_name = {pp: f"t{i}" for i, pp in enumerate(taps)}
    prog = build_conv_program(
        ConvTreeSpec(out_channels=1, tree_depth=d, indices_flat=[row], truth_tables=tr)
    )
    leaves2, tts2, before, after = resynth_tree(row, tr, d, max_cost=4)
    prog2 = build_conv_program(
        ConvTreeSpec(out_channels=1, tree_depth=d, indices_flat=[leaves2], truth_tables=tts2)
    )
    assert before == len(prog.instrs) == _solo_cost(row, tr, d)
    assert after == len(prog2.instrs)
    _, roles1, _ = tree_roles(row, tr, d)
    _, roles2, _ = tree_roles(leaves2, tts2, d)
    assert sum(r == "instr" for r in roles1.values()) == before
    assert sum(r == "instr" for r in roles2.values()) == after

    # 全 16 パターンで 3 つの表現が一致することを確かめる
    table = []
    for pat in range(1 << len(taps)):
        assign = {pp: (pat >> i) & 1 for i, pp in enumerate(taps)}
        f0, _ = eval_tree_bits(row, tr, d, assign)
        f1, _ = eval_program(prog, assign)
        f2, _ = eval_tree_bits(leaves2, tts2, d, assign)
        assert f0 == f1 == f2 == (onset >> pat) & 1
        table.append((pat, f0, f1, f2))
    ex_pat = next(pat for pat, f0, _, _ in table if f0 == 1)
    assign = {pp: (ex_pat >> i) & 1 for i, pp in enumerate(taps)}

    fig = plt.figure(figsize=(17, 10.6))
    gs = fig.add_gridspec(
        2, 3, height_ratios=[1.0, 0.92], width_ratios=[1.15, 0.8, 1.15], hspace=0.28, wspace=0.08
    )
    ax_a = fig.add_subplot(gs[0, 0])
    draw_tree(
        ax_a,
        row,
        tr,
        d,
        roles1,
        tap_name,
        assign,
        f"(a) As trained: depth-{d} tree, {ni} nominal gates",
        p,
    )
    # (b) 命令列
    ax_b = fig.add_subplot(gs[0, 1])
    ax_b.axis("off")
    ax_b.set_title(f"(b) Instruction list: {len(prog.instrs)} gates", loc="left", fontsize=11.5)
    _, ivals = eval_program(prog, assign)
    yy = 0.86
    ax_b.text(
        0.02,
        0.95,
        "trivial gates folded, unused ones dropped",
        transform=ax_b.transAxes,
        fontsize=10.6,
        color="#475569",
    )
    for j, ins in enumerate(prog.instrs):
        txt = f"i{j} = {GATE_NAME[ins.tt]:<5}({ref_str(prog, ins.a, tap_name)}, {ref_str(prog, ins.b, tap_name)})"  # noqa: E501
        ax_b.add_patch(
            FancyBboxPatch(
                (0.03, yy - 0.045),
                0.9,
                0.085,
                transform=ax_b.transAxes,
                boxstyle="round,pad=0.005,rounding_size=0.01",
                fc=matplotlib.colors.to_rgba(GREEN, 0.18),
                ec=GREEN,
                lw=0.9,
            )
        )
        ax_b.text(
            0.06, yy, txt, transform=ax_b.transAxes, fontsize=11.2, family="monospace", va="center"
        )
        ax_b.text(
            0.9,
            yy,
            f"= {ivals[j]}",
            transform=ax_b.transAxes,
            fontsize=10.6,
            va="center",
            ha="right",
            color="#0369a1",
            weight="bold",
        )
        yy -= 0.11
    ax_b.text(
        0.06,
        yy,
        f"out = {ref_str(prog, prog.oc_out[0], tap_name)}",
        transform=ax_b.transAxes,
        fontsize=11.2,
        family="monospace",
        va="center",
    )
    yy -= 0.12
    ax_b.text(
        0.03,
        yy,
        "taps (channel, dy, dx):\n"
        + "\n".join(
            "  " + ",  ".join(f"{tap_name[pp]} = {tap_label(pp, p)}" for pp in taps[i : i + 2])
            for i in range(0, len(taps), 2)
        ),
        transform=ax_b.transAxes,
        fontsize=9.5,
        va="center",
        color="#334155",
        wrap=True,
    )
    # (c) 再合成
    ax_c = fig.add_subplot(gs[0, 2])
    draw_tree(
        ax_c,
        leaves2,
        tts2,
        d,
        roles2,
        tap_name,
        assign,
        f"(c) Resynthesized: {after} gates (fillers cost 0)",
        p,
    )
    # (d) 真理値表
    gsb = gs[1, :].subgridspec(1, 2, width_ratios=[0.85, 1.6], wspace=0.22)
    ax_d = fig.add_subplot(gsb[0])
    ax_d.axis("off")
    ax_d.set_title(
        f"(d) Same truth table on all 2^{len(taps)} = {len(table)} inputs",
        loc="left",
        fontsize=12.5,
    )
    hdr = [tap_name[pp] for pp in taps] + ["(a)", "(b)", "(c)"]
    ncol = len(hdr)
    cw = 1.0 / ncol
    for c, h in enumerate(hdr):
        ax_d.text(
            (c + 0.5) * cw,
            0.97,
            h,
            transform=ax_d.transAxes,
            ha="center",
            va="center",
            fontsize=10.0,
            weight="bold",
        )
    rh = 0.9 / len(table)
    for r, (pat, f0, f1, f2) in enumerate(table):
        y = 0.92 - (r + 0.5) * rh
        if f0:
            ax_d.add_patch(
                Rectangle(
                    (0, y - rh / 2),
                    1,
                    rh,
                    transform=ax_d.transAxes,
                    fc=matplotlib.colors.to_rgba(GREEN, 0.18),
                    ec="none",
                )
            )
        if pat == ex_pat:
            ax_d.add_patch(
                Rectangle(
                    (0, y - rh / 2), 1, rh, transform=ax_d.transAxes, fc="none", ec=ORANGE, lw=1.4
                )
            )
        cells = [(pat >> i) & 1 for i in range(len(taps))] + [f0, f1, f2]
        for c, v in enumerate(cells):
            ax_d.text(
                (c + 0.5) * cw,
                y,
                str(v),
                transform=ax_d.transAxes,
                ha="center",
                va="center",
                fontsize=9.5,
                weight="bold" if c >= len(taps) and v else "normal",
            )
    # (e) 下限
    ax_e = fig.add_subplot(gsb[1])
    names = [f"{nid.split('_')[0]} ({n_oc})" for nid, n_oc, _ in per_node]
    cats = [
        ("skip", "already at the lower bound -> search skipped", GREY),
        ("nogain", "searched, no smaller tree found", matplotlib.colors.to_rgba(GREEN, 0.4)),
        ("gain", "searched, fewer gates", GREEN),
        ("limit", "bound above search limit", YELLOW),
    ]
    left = np.zeros(len(per_node))
    for key, lab, col in cats:
        vals = np.array([c[key] / n_oc * 100 for _, n_oc, c in per_node])
        if not vals.any():
            continue
        ax_e.barh(
            range(len(per_node)),
            vals,
            left=left,
            color=col,
            edgecolor="white",
            label=lab,
            height=0.62,
        )
        for i, v in enumerate(vals):
            if v >= 6:
                ax_e.text(
                    left[i] + v / 2,
                    i,
                    f"{v:.0f}%",
                    ha="center",
                    va="center",
                    fontsize=9.4,
                    color="white" if key in ("skip", "gain") else INK,
                )
        left += vals
    ax_e.set_yticks(range(len(per_node)), names, fontsize=10.6)
    ax_e.invert_yaxis()
    ax_e.set_xlim(0, 100)
    ax_e.set_xlabel("share of trees in each conv2d node (%)   [node (number of trees)]")
    ax_e.spines[["top", "right"]].set_visible(False)
    ax_e.legend(
        loc="lower center", bbox_to_anchor=(0.5, -0.36), ncol=2, frameon=False, fontsize=10.4
    )
    ax_e.set_title(
        "(e) Lower bound: a function of e essential inputs needs >= e-1 two-input gates.\n"
        f"     Trees already at e-1 are never searched (this tree: e = {len(ev)}, "
        f"list {before} > {len(ev) - 1} -> searched, reached {after} = bound)",
        loc="left",
        fontsize=12.5,
    )
    fig.suptitle(
        f"conv2d tree of the face detector (det_face_lut_k256, node {node_id}, output channel {oc}):"  # noqa: E501
        f" {ni} -> {before} -> {after} gates.\nWhole model: 14,784 -> 1,279 -> 1,077",
        fontsize=14.4,
        weight="bold",
        y=1.035,
    )
    from matplotlib.patches import Patch

    fig.legend(
        handles=[
            Patch(
                fc=matplotlib.colors.to_rgba(BLUE, 0.2),
                ec="#0284c7",
                label="tap (input bit of the 3x3 window)",
            ),
            Patch(
                fc=matplotlib.colors.to_rgba(GREEN, 0.25),
                ec=GREEN,
                label="gate that becomes an instruction",
            ),
            Patch(
                fc=matplotlib.colors.to_rgba(BLUE, 0.15),
                ec="#0284c7",
                ls="--",
                label="pass-through (folded)",
            ),
            Patch(
                fc=matplotlib.colors.to_rgba(GREY, 0.25),
                ec=GREY,
                ls="--",
                label="constant filler (folded)",
            ),
            Patch(fc="white", ec=RED, ls="--", label="unused (removed)"),
        ],
        loc="upper center",
        ncol=5,
        frameon=False,
        fontsize=10.4,
        bbox_to_anchor=(0.5, 0.965),
    )
    save(fig, "conv-resynth.png")
    return {"tree": (ni, before, after), "e": len(ev), "per_node": per_node}



# ============================================================
# 図: Web— lut_layer と lut_flat(js / wasm)の実測
# ============================================================
def load_web():
    data = {m: dict(v) for m, v in WEB_FALLBACK.items()}
    if WEB_JSON.exists():
        d = json.loads(WEB_JSON.read_text(encoding="utf-8"))
        name = {"har_k600": "HAR k600", "har_k2400": "HAR k2400", "har_k6000": "HAR k6000"}
        src = {"HAR k600": d["node"], "HAR k2400": d["browser"], "HAR k6000": d["browser"]}
        for model, rows in src.items():
            for r in rows:
                if name.get(r["model"]) == model and r["batch"] in (1, 32):
                    tup = (r["lut_layer_js"], r["lut_flat_js"], r["lut_flat_wasm"])
                    assert np.allclose(tup, WEB_FALLBACK[model][r["batch"]], rtol=1e-3), (
                        model, r["batch"], tup)
                    data[model][r["batch"]] = tup
        print("  web: read", WEB_JSON.relative_to(ROOT))
    return data


def fig_web():
    data = load_web()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), gridspec_kw={"wspace": 0.28})
    cols = [matplotlib.colors.to_rgba(GREY, 0.75), matplotlib.colors.to_rgba(GREEN, 0.5), GREEN]
    edges = [GREY, GREEN, GREEN]
    names = ["lut_layer (legacy)", "lut_flat, js", "lut_flat, wasm"]
    width = 0.26
    for ax, (model, by_batch) in zip(axes, data.items(), strict=False):
        top = max(max(v) for v in by_batch.values())
        for gi, batch in enumerate((1, 32)):
            vals = by_batch[batch]
            for j, val in enumerate(vals):
                x = gi + (j - 1) * width
                ax.bar(x, val, width=width * 0.92, color=cols[j], edgecolor=edges[j])
                ax.text(x, val + top * 0.015, f"{val:.1f}" if val >= 10 else f"{val:.2f}",
                        ha="center", va="bottom", fontsize=9.4)
            ax.text(gi + width, max(vals[1], vals[2]) + top * 0.12,
                    f"x{vals[0] / vals[2]:.0f}", ha="center", fontsize=11.5,
                    color=GREEN, weight="bold")
        ax.set_xticks([0, 1], ["1 sample", "32 samples\n(µs per sample)"], fontsize=10.6)
        ax.set_ylabel("µs per inference")
        ax.set_ylim(0, top * 1.22)
        ax.set_title(f"{model}  ({WEB_SOURCE[model]})", fontsize=12.5)
        ax.spines[["top", "right"]].set_visible(False)
    from matplotlib.patches import Patch

    fig.legend(handles=[Patch(fc=c, ec=e, label=n) for c, e, n in zip(cols, edges, names,
                                                                     strict=False)],
               loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.07),
               fontsize=11.2)
    fig.suptitle(
        "Web runtime on Apple M3: same integer scores; browser = built-in Chrome 152, "
        "Node = v25 (each panel has its own y axis)",
        fontsize=13.6, weight="bold", y=1.03)
    save(fig, "web-latency.png")
    return data


# ============================================================
# 図: GPU の段分け— 層 → sweep → lut_flat の段
# ============================================================
def load_levels():
    """HAR の実モデルから元の層幅・sweep 後の層幅・平坦化後の段の幅を計算する。"""
    from griffith_logic.deploy import build_bundle
    from griffith_logic.deploy.compress import apply_compress
    from griffith_logic.deploy.flatlevels import levelize_lut_flat
    from griffith_logic.deploy.flatten_dense import flatten_dense_doc

    data = dict(LEVELS_FALLBACK)
    for model, k in (("HAR k600", 600), ("HAR k2400", 2400), ("HAR k6000", 6000)):
        path = HAR_DIR / f"har_dlgn_k{k}_gumbel.json"
        if not path.exists():
            continue
        doc = json.loads(path.read_text(encoding="utf-8"))
        orig = [int(n["params"]["out_width"]) for n in doc["ir"]["nodes"]
                if n["type"] == "lut_layer"]
        cdoc, _, _ = apply_compress(doc, level="fast")
        swept = [int(n["params"]["out_width"]) for n in cdoc["ir"]["nodes"]
                 if n["type"] == "lut_layer"]
        fdoc, _ = flatten_dense_doc(build_bundle(cdoc), n_verify=0)
        node = next(n for n in fdoc["ir"]["nodes"] if n["type"] == "lut_flat")
        lf = levelize_lut_flat(node["params"])
        got = (orig, swept, lf.level_widths, lf.n_signals)
        assert got == LEVELS_FALLBACK[model], (model, got)
        data[model] = got
        print(f"  levels: computed {model} from {path.name}")
    return data


def fig_gpu_levels():
    data = load_levels()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8),
                             gridspec_kw={"wspace": 0.26, "width_ratios": [4, 4, 6]})
    cols = [matplotlib.colors.to_rgba(GREY, 0.55), matplotlib.colors.to_rgba(GREEN, 0.4), GREEN]
    edges = [GREY, GREEN, GREEN]
    names = ["trained lut_layer (gates per layer)", "after sweep (shipped lut_layer)",
             "lut_flat level (one GPU launch or barrier each)"]
    width = 0.27
    for ax, (model, (orig, swept, levels, n_sig)) in zip(axes, data.items(), strict=False):
        depth = len(orig)
        for j, series in enumerate((orig, swept, levels)):
            xs = np.arange(depth) + (j - 1) * width
            ax.bar(xs, series, width=width * 0.92, color=cols[j], edgecolor=edges[j])
        ax.set_xticks(range(depth), [f"L{i + 1}" for i in range(depth)])
        ax.set_xlabel("layer / level")
        ax.set_ylabel("gates")
        tot = (sum(orig), sum(swept), sum(levels))
        ax.set_title(f"{model}: {tot[0]:,} -> {tot[1]:,} -> {tot[2]:,}", fontsize=12.2)
        fits = "fits" if n_sig <= 32768 else "exceeds"
        ax.text(0.02, 0.97, f"signal table {n_sig:,} B\n{fits} Metal 32 KB shared memory",
                transform=ax.transAxes, va="top", fontsize=9.8,
                bbox=dict(fc=matplotlib.colors.to_rgba(BLUE, 0.12), ec=BLUE, pad=4))
        ax.set_ylim(0, max(orig) * 1.32)
        ax.spines[["top", "right"]].set_visible(False)
    from matplotlib.patches import Patch

    fig.legend(handles=[Patch(fc=c, ec=e, label=n) for c, e, n in zip(cols, edges, names,
                                                                     strict=False)],
               loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.08),
               fontsize=11)
    fig.suptitle(
        "GPU lut_flat: gates re-ordered into ASAP levels; gates in a level never depend on "
        "each other (real HAR models, deploy/flatlevels.py)",
        fontsize=13.6, weight="bold", y=1.03)
    save(fig, "gpu-levels.png")
    return data


# ============================================================
# 図: Metal 実測— 平坦化しない最良に対する比
# ============================================================
def load_metal():
    data = {m: list(v) for m, v in METAL_FALLBACK.items()}
    if METAL_JSON.exists():
        rows = json.loads(METAL_JSON.read_text(encoding="utf-8"))["rows"]
        name = {"har_dlgn_k600_gumbel": "HAR k600", "har_dlgn_k2400_gumbel": "HAR k2400",
                "har_dlgn_k6000_gumbel": "HAR k6000"}
        keys = ("dense_fuse", "dense_packed", "dense_flat_fused", "dense_flat_packed")
        for m_id, model in name.items():
            got = []
            for b in METAL_BATCHES:
                got.append(tuple(round(rows[f"{m_id}::{k}_batch{b}"]
                                       ["throughput_per_s_median"] / 1000) for k in keys))
            assert all(abs(x - y) <= 1 for g, f in zip(got, METAL_FALLBACK[model],
                                                       strict=False)
                       for x, y in zip(g, f, strict=False)), (model, got)
            data[model] = got
        print("  metal: read", METAL_JSON.relative_to(ROOT))
    return data


def fig_gpu_metal():
    data = load_metal()
    fig, axes = plt.subplots(1, 2, figsize=(15, 4.8), gridspec_kw={"wspace": 0.22,
                                                                   "width_ratios": [5, 4]})
    ax = axes[0]
    marks = {"HAR k600": "o", "HAR k2400": "s", "HAR k6000": "^"}
    shades = {"HAR k600": 0.45, "HAR k2400": 0.7, "HAR k6000": 1.0}
    for model, rows in data.items():
        ratio = [max(r[2], r[3]) / max(r[0], r[1]) for r in rows]
        ax.plot(range(len(METAL_BATCHES)), ratio, marker=marks[model], lw=2,
                color=matplotlib.colors.to_rgba(GREEN, shades[model]),
                label=f"{model}  ({min(ratio):.2f}-{max(ratio):.2f})")
        if model == "HAR k6000":     # 線が重なるので、1 を割るモデルだけ値を書く
            for i, v in enumerate(ratio):
                ax.text(i, v - 0.075, f"{v:.2f}", ha="center", fontsize=9.4)
    ax.axhline(1.0, color=GREY, ls="--", lw=1.2)
    ax.axhspan(0, 1.0, color=matplotlib.colors.to_rgba(RED, 0.06))
    ax.text(len(METAL_BATCHES) - 1, 0.32, "slower than not flattening", ha="right",
            color=RED, fontsize=10.5)
    ax.set_xticks(range(len(METAL_BATCHES)), [str(b) for b in METAL_BATCHES])
    ax.set_xlabel("batch")
    ax.set_ylabel("best lut_flat / best lut_layer\n(dense_fuse or dense_packed)")
    ax.set_ylim(0.25, 1.45)
    ax.legend(frameon=False, loc="upper left")
    ax.set_title("Throughput ratio (7-set median)", fontsize=12.5)
    ax.spines[["top", "right"]].set_visible(False)

    ax = axes[1]
    rows = data["HAR k6000"]
    bi = METAL_BATCHES.index(1024)
    vals = rows[bi]
    labels = ["dense_fuse", "dense_packed", "lut_flat\nfused", "lut_flat\npacked"]
    cols = [matplotlib.colors.to_rgba(GREY, 0.55), matplotlib.colors.to_rgba(GREY, 0.85),
            matplotlib.colors.to_rgba(GREEN, 0.5), GREEN]
    bars = ax.bar(range(4), vals, color=cols, edgecolor=[GREY, GREY, GREEN, GREEN], width=0.62)
    for b, v in zip(bars, vals, strict=False):
        ax.text(b.get_x() + b.get_width() / 2, v + max(vals) * 0.02, f"{v:,}k", ha="center",
                fontsize=10.6)
    ax.set_xticks(range(4), labels, fontsize=10.4)
    ax.set_ylabel("k inferences / s")
    ax.set_ylim(0, max(vals) * 1.18)
    ax.set_title("HAR k6000, batch 1024: half the gates, still slower", fontsize=12.5)
    ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(
        "Metal (Apple M3) measured: lut_flat vs the best non-flattened kernel; batch=1 is "
        "0.32-0.40 ms for every variant",
        fontsize=13.6, weight="bold", y=1.03)
    save(fig, "gpu-metal.png")
    return data


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument(
        "--models-dir",
        type=Path,
        default=DEFAULT_MODELS_DIR,
        help="HAR の har_dlgn_k{600,2400,6000}_gumbel.json を探すディレクトリ"
        "(既定: tools/webbench/public/models。無ければ公開ページの値で描く)",
    )
    ap.add_argument(
        "--face-model",
        type=Path,
        default=None,
        help="det_face_lut_k256.json のパス(既定: <models-dir>/det_face_lut_k256.json)",
    )
    args = ap.parse_args()
    global HAR_DIR
    HAR_DIR = args.models_dir
    face_model = args.face_model or args.models_dir / DEFAULT_FACE.name
    style()
    print("XOR (examples/06_deploy_c と同じ手順) を学習")
    ir = build_xor()
    x_example = (0, 1)
    swept, rep, cnt = fig_sweep(ir, x_example)
    print(f"  sweep: {rep['gates_before']} -> {rep['gates_after']}  removed={cnt}")
    frep, counts = fig_dense_flat(swept, x_example)
    print(f"  dense-flat: {frep['gates_before']} -> {frep['gates_after']}  outcomes={counts}")
    fig_esp32()
    fig_batch_pack(swept)
    fig_batch_bars()
    fig_web()
    fig_gpu_levels()
    fig_gpu_metal()
    if face_model.exists():
        info = fig_conv(face_model)
        print(f"  conv tree: {info['tree']} e={info['e']}")
        for nid, n_oc, c in info["per_node"]:
            print(f"    {nid}: {n_oc} trees {c}")
    else:
        print(f"  skip conv figure: {face_model} が無い")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
