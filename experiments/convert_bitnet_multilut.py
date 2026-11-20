"""温度計部分和分解: 1ニューロン→複数LUT(fan-in≤6)でfan-inギャップを埋める。

段階検証(--stage):
    toy1   : 合成ternaryニューロン(G≤16、全2^G掃引可能)。
             予算-agreement曲線と厳密性assert(bound<margin→全点一致)
    mnist  : 自作STE-MLP(ternary bake)全層分解。arm A/B/C比較+
             rebuild_trainable→控えめpolish(lr=2e-3, soft固定)
    bitnet : 実BitNet FFN gate_proj 1層(GPU機B)。ニューロンサンプルに対し
             held-out a2(vs二値教師)/a3(vs実数教師)とLUT数の全腕比較

腕:
    A = 現行top-6転写(sparsify_from_mask+threshold_truth_tables、アンカー)
    B = popcount厳密加算木(exact_adder、agreement=1.0保証のコスト上限)
    C = 温度計部分和分解(本手法、DecomposeSpecのグリッド)

使用例:
    uv run python experiments/convert_bitnet_multilut.py --stage toy1 --no-wandb
    uv run python experiments/convert_bitnet_multilut.py --stage mnist --seed 0
    .venv/bin/python experiments/convert_bitnet_multilut.py --stage bitnet \
        --n-neurons 256 --calib-n 1024   # GPU機B
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from env_utils import load_dotenv  # noqa: E402
from wandb_utils import finish as wandb_finish  # noqa: E402
from wandb_utils import init_wandb, log_metrics  # noqa: E402

from griffith_logic.convert.decompose_thermo import (  # noqa: E402
    DecomposeSpec,
    decompose_mlp,
    decompose_threshold_neuron,
    eval_circuit,
    exact_adder_baseline_layer,
)
from griffith_logic.ir.simulate import simulate  # noqa: E402
from griffith_logic.ir.validate import validate  # noqa: E402

load_dotenv()
JST = timezone(timedelta(hours=9))


def _now() -> str:
    return datetime.now(JST).strftime('%H:%M:%S')


def _spec_grid(args) -> list[tuple[str, DecomposeSpec]]:
    """arm Cの設定グリッド(label, spec)。objective両方を張る。"""
    grid = []
    for obj, tag in (('max_err', ''), ('exp_err', 'e')):
        for b in args.bits_grid:
            grid.append((f'uni{b}{tag}', DecomposeSpec(
                alloc='uniform', bits_uniform=b, grouping=args.grouping,
                objective=obj, stage2=args.stage2, seed=args.seed)))
        for a in args.alpha_grid:
            grid.append((f'ada{a:g}{tag}', DecomposeSpec(
                alloc='adaptive', margin_alpha=a,
                margin_quantile=args.margin_quantile,
                grouping=args.grouping, objective=obj,
                stage2=args.stage2, seed=args.seed)))
    return grid


def _save(args, stage: str, payload: dict) -> Path:
    out_dir = Path(__file__).resolve().parents[1] / 'results' / 'multilut'
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(JST).strftime('%Y%m%d_%H%M%S')
    path = out_dir / f'multilut_{stage}_s{args.seed}_{ts}.json'
    with open(path, 'w') as f:
        json.dump({'args': vars(args), **payload}, f, indent=1,
                  ensure_ascii=False)
    print(f'[{_now()}] 保存: {path}')
    return path


# ============================================================
# stage: toy1(ゲート)
# ============================================================

def run_toy1(args) -> dict:
    rng = np.random.default_rng(args.seed)
    g = args.toy_g
    u = rng.choice([-1.0, 1.0], size=g)
    thr = 0.5
    X = np.array(list(itertools.product([0, 1], repeat=g)), dtype=np.int64)
    calib = X[rng.permutation(len(X))[:512]]
    y_true = (X @ u > thr).astype(np.int64)
    margins = np.abs(X @ u - thr)

    results = []
    # arm B: 厳密加算木(1行の層として)
    q_row = u.astype(np.int64)[None, :]
    gb = exact_adder_baseline_layer(q_row, np.array([thr]))
    validate(gb)
    agree_b = float((simulate(gb, X)[:, 0] == y_true).mean())
    assert agree_b == 1.0, f'baseline Bが厳密でない: {agree_b}'
    results.append({'arm': 'B', 'label': 'adder',
                    'agree': agree_b,
                    'n_luts': gb.metadata['decompose']['n_luts_total']})

    # arm A: top6(ternaryは同マグニチュードなので先頭6本)
    from griffith_logic.convert.transcribe import threshold_truth_tables
    top6 = np.argsort(-np.abs(u), kind='stable')[:6]
    dropped = np.setdiff1d(np.arange(g), top6)
    bshift = float((u[dropped] * calib[:, dropped].mean(0)).sum())
    tt = threshold_truth_tables(
        torch.as_tensor(u[top6][None, :]), torch.as_tensor([bshift - thr]))
    addr = (X[:, top6] * (1 << np.arange(6))).sum(1)
    pred_a = np.array([(int(tt[0]) >> int(a)) & 1 for a in addr])
    results.append({'arm': 'A', 'label': 'top6',
                    'agree': float((pred_a == y_true).mean()), 'n_luts': 1})

    # arm C: グリッド
    for label, spec in _spec_grid(args):
        circ = decompose_threshold_neuron(u, thr, calib, spec)
        y_hat = eval_circuit(circ, X)
        agree = float((y_hat == y_true).mean())
        bound = circ.meta['error_bound']
        # 安全マージン点は全点一致(構成的保証のassert)
        safe = margins > bound + 1e-9
        assert (y_hat[safe] == y_true[safe]).all(), f'{label}: 上界破れ'
        # 予算がcalib margin最小を下回るなら全点厳密のはず
        if bound < margins.min() - 1e-9:
            assert agree == 1.0, f'{label}: bound<margin_minなのに非厳密'
        results.append({'arm': 'C', 'label': label, 'agree': agree,
                        'n_luts': circ.meta['n_luts'],
                        'depth': circ.meta['depth'],
                        'error_bound': bound,
                        'budget_met': circ.meta['budget_met']})
        print(f"[{_now()}] toy1 {label:8s} agree={agree:.4f} "
              f"luts={circ.meta['n_luts']:3d} bound={bound:.2f}")

    print(f'[{_now()}]ゲート: 全assert通過(G={g})')
    return {'stage': 'toy1', 'G': g, 'results': results}


# ============================================================
# stage: mnist(ゲート)
# ============================================================

def _mnist_teacher(args, device):
    sys.path.insert(0, os.path.dirname(__file__))
    from convert_prune_transcribe_mnist import _flatten_binary
    from convert_ternary_to_dwn_mnist import _train_ste_teacher

    from griffith_logic.convert.binarize import quantize_ternary
    from griffith_logic.datasets.vision import load_mnist

    torch.manual_seed(args.seed)
    train_set, test_set = load_mnist()
    x_tr, y_tr = _flatten_binary(train_set, n=args.mnist_train_n)
    x_te, y_te = _flatten_binary(test_set, n=args.mnist_test_n)
    print(f'[{_now()}] STE teacher学習(hidden={args.hidden})')
    model = _train_ste_teacher(x_tr, y_tr, tuple(args.hidden),
                               args.teacher_epochs, 1e-3, 256, device)
    model = model.cpu()
    with torch.no_grad():
        for lin in model.linears[:-1]:
            q, scale, _ = quantize_ternary(
                lin.weight.detach().cpu().double(), per='channel')
            lin.weight.copy_((q.double() * scale).float())
    return model, (x_tr, y_tr), (x_te, y_te)


def _teacher_hard_pred(model, x: np.ndarray) -> np.ndarray:
    import torch.nn.functional as F

    from griffith_logic.convert.teacher_models import HeavisideSTE
    h = torch.as_tensor(x, dtype=torch.float64)
    with torch.no_grad():
        for lin in model.linears[:-1]:
            h = HeavisideSTE.apply(F.linear(
                h, lin.weight.double(),
                None if lin.bias is None else lin.bias.double()))
        fin = model.linears[-1]
        logits = F.linear(h, fin.weight.double(), fin.bias.double())
    return logits.argmax(-1).numpy()


def _polish(graph, x_tr, y_tr, x_te, y_te, args) -> dict:
    """控えめpolish(lr=2e-3, selection=soft固定, best-over-epochs)。"""
    import torch.nn.functional as F

    from griffith_logic.convert.recover import rebuild_trainable
    dwn = rebuild_trainable(graph, parameterization='independent',
                            selection='soft')
    xt = torch.as_tensor(x_tr, dtype=torch.float32)
    yt = torch.as_tensor(y_tr)
    xe = torch.as_tensor(x_te, dtype=torch.float32)

    def hard_acc() -> float:
        dwn.eval()
        preds = []
        with torch.no_grad():
            for i in range(0, len(xe), 1024):
                preds.append(dwn(xe[i:i + 1024]).argmax(-1))
        return float((torch.cat(preds).numpy() == y_te).mean())

    acc0 = hard_acc()
    opt = torch.optim.Adam(dwn.parameters(), lr=args.polish_lr)
    best = acc0
    for ep in range(args.polish_epochs):
        dwn.train()
        perm = torch.randperm(len(xt))
        for i in range(0, len(xt), 256):
            idx = perm[i:i + 256]
            opt.zero_grad()
            F.cross_entropy(dwn(xt[idx]), yt[idx]).backward()
            opt.step()
        cur = hard_acc()
        best = max(best, cur)
        print(f'[{_now()}]   polish ep{ep + 1} hard={cur:.4f} '
              f'(before={acc0:.4f} best={best:.4f})')
    return {'acc_before_polish': acc0, 'acc_best_polish': best}


def _simulate_argmax(graph, x: np.ndarray, batch: int = 256) -> np.ndarray:
    """simulateを小バッチで回す(巨大加算木グラフのOOM回避)。"""
    outs = []
    for i in range(0, len(x), batch):
        outs.append(simulate(graph, x[i:i + batch]).argmax(-1))
    return np.concatenate(outs)


def _assert_hidden_bits_exact(graph, model, x: np.ndarray) -> None:
    """分解IRの最終隠れ層ビットが参照 1[q·x>thr] 連鎖と全点一致をassert。"""
    from griffith_logic.convert.decompose_thermo import ternary_from_linear
    from griffith_logic.ir.simulate import simulate_trace

    # 最終LUTノード(finalを含むid)への入力srcが最終隠れ層の出力
    final_ids = [n.id for n in graph.nodes if 'final' in n.id]
    assert final_ids, 'finalノードが見つからない'
    src = next(e.src for e in graph.edges if e.dst == final_ids[0])
    trace = simulate_trace(graph, x)
    got = trace[src]

    h = np.asarray(x, dtype=np.int64)
    for lin in model.linears[:-1]:
        w = lin.weight.detach().cpu().double().numpy()
        b = (lin.bias.detach().cpu().double().numpy()
             if lin.bias is not None else np.zeros(w.shape[0]))
        q, thr = ternary_from_linear(w, b)
        h = (h @ q.T > thr[None, :]).astype(np.int64)
    assert (got == h).all(), '隠れ層ビットが参照と不一致(厳密性破れ)'


def run_mnist(args) -> dict:
    from griffith_logic.convert.binarize import quantize_ternary
    from griffith_logic.convert.transcribe import TranscribeSpec, transcribe_mlp

    device = ('cuda' if torch.cuda.is_available() else 'cpu')
    model, (x_tr, y_tr), (x_te, y_te) = _mnist_teacher(args, device)
    calib = x_tr[:args.calib_n]
    teacher_pred_te = _teacher_hard_pred(model, x_te)
    teacher_acc = float((teacher_pred_te == y_te).mean())
    print(f'[{_now()}] ternary teacher test hard acc = {teacher_acc:.4f}')

    results = []

    # arm A: 現行top-6転写(0マスク尊重)
    masks = [quantize_ternary(lin.weight.detach().cpu().double(),
                              per='channel')[2]
             for lin in model.linears[:-1]]
    ga = transcribe_mlp(model, calib_inputs=calib.astype(np.float64),
                        spec=TranscribeSpec(n_inputs=6), masks=masks)
    validate(ga)
    pred_a = _simulate_argmax(ga, x_te)
    n_luts_a = sum(n.params['out_width'] for n in ga.nodes
                   if n.type == 'lut_layer')
    results.append({
        'arm': 'A', 'label': 'top6',
        'test_acc': float((pred_a == y_te).mean()),
        'teacher_agree': float((pred_a == teacher_pred_te).mean()),
        'n_luts': int(n_luts_a),
    })
    print(f"[{_now()}] arm A  acc={results[-1]['test_acc']:.4f} "
          f"agree={results[-1]['teacher_agree']:.4f} luts={n_luts_a}")

    # arm B: 厳密加算木
    gb = decompose_mlp(model, calib, DecomposeSpec(), hidden_arm='adder')
    validate(gb)
    pred_b = _simulate_argmax(gb, x_te)
    res_b = {
        'arm': 'B', 'label': 'adder',
        'test_acc': float((pred_b == y_te).mean()),
        'teacher_agree': float((pred_b == teacher_pred_te).mean()),
        'n_luts': gb.metadata['decompose']['n_luts_total'],
    }
    # 厳密性assert: 隠れ層ビットが参照 1[q·x>thr] 連鎖と全点一致
    # (最終層argmaxは8bit量子化(全arm共通処理)で参照floatと僅差が出得る
    #  ため、arm Bの構成的保証の対象である隠れ層で厳密性を検証する)
    _assert_hidden_bits_exact(gb, model, x_te[:256])
    res_b['hidden_bit_exact'] = True
    results.append(res_b)
    print(f"[{_now()}] arm B  acc={res_b['test_acc']:.4f} "
          f"agree={res_b['teacher_agree']:.4f} (隠れ層bit厳密✓) "
          f"luts={res_b['n_luts']}")

    # arm C: グリッド(+代表設定のみpolish)
    for gi, (label, spec) in enumerate(_spec_grid(args)):
        gc = decompose_mlp(model, calib, spec, hidden_arm='thermo')
        validate(gc)
        pred_c = _simulate_argmax(gc, x_te)
        rec = {
            'arm': 'C', 'label': label,
            'test_acc': float((pred_c == y_te).mean()),
            'teacher_agree': float((pred_c == teacher_pred_te).mean()),
            'n_luts': gc.metadata['decompose']['n_luts_total'],
            'depth': max(m['depth'] for m in
                         gc.metadata['decompose']['layers']),
        }
        print(f"[{_now()}] arm C {label:8s} acc={rec['test_acc']:.4f} "
              f"agree={rec['teacher_agree']:.4f} luts={rec['n_luts']}")
        if gi == args.polish_grid_index and args.polish_epochs > 0:
            rec.update(_polish(gc, x_tr, y_tr, x_te, y_te, args))
        results.append(rec)

    return {'stage': 'mnist', 'teacher_acc': teacher_acc,
            'results': results}


# ============================================================
# stage: bitnet(ゲート)
# ============================================================

def _thermo_input_expand(u: np.ndarray, x_sup: np.ndarray,
                         fit_i: np.ndarray, bits: int
                         ) -> tuple[np.ndarray, float, np.ndarray]:
    """Phase 2.5: 支持次元の実活性を温度計bビット化し重み付き閾値形へ展開。

    x̂_i = c_i0 + Σ_j δ_ij·t_ij(δ=bin条件付き平均の差分、不偏復元)により
        1[Σ_i u_i·x_i > 0] ≈ 1[Σ_ij (u_i·δ_ij)·t_ij > −Σ_i u_i·c_i0]
    の右辺(重み付き閾値ニューロン、二値入力)を返す。分位点・復元は
    fit半のみで較正する(held-out健全性)。

    Returns:
        (w_expanded, thr0, T_bits (N, Σbits)):
            thr0 = −Σ u_i·c_i0(基準閾値。呼び手でT_opt再較正を推奨)
    """
    g = x_sup.shape[1]
    ws: list[float] = []
    cols: list[np.ndarray] = []
    offset = 0.0
    qs = [(k + 1) / (bits + 1) for k in range(bits)]
    for i in range(g):
        xf = x_sup[fit_i, i]
        taus = np.unique(np.quantile(xf, qs))
        if len(taus) == 0 or xf.std() == 0:
            offset += float(u[i]) * float(xf.mean())
            continue
        t = (x_sup[:, i][:, None] >= taus[None, :]).astype(np.int64)
        bins_f = t[fit_i].sum(axis=1)
        recon = np.empty(len(taus) + 1)
        prev = float(xf.min())
        for bidx in range(len(taus) + 1):
            m = bins_f == bidx
            prev = float(xf[m].mean()) if m.any() else prev
            recon[bidx] = prev
        deltas = np.diff(recon)
        keep = deltas > 1e-12  # 退化bin(同一平均)は情報なし→落とす
        offset += float(u[i]) * float(recon[0])
        ws.extend((float(u[i]) * deltas[keep]).tolist())
        if keep.any():
            cols.append(t[:, keep])
    T = (np.concatenate(cols, axis=1) if cols
         else np.zeros((x_sup.shape[0], 0), dtype=np.int64))
    return np.array(ws), -offset, T

def _load_bitnet_layer(args):
    """BitNet FFN gate_projの (q, scale, W, 実較正活性x) を返す。"""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from griffith_logic.convert.binarize import quantize_ternary

    repo = 'microsoft/bitnet-b1.58-2B-4T-bf16'
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'[{_now()}] {repo} をロード中(device={dev})')
    tok = AutoTokenizer.from_pretrained(repo)
    model = AutoModelForCausalLM.from_pretrained(
        repo, torch_dtype=torch.bfloat16).eval().to(dev)
    lin = model.model.layers[args.layer].mlp.gate_proj
    W = lin.weight.detach().float().cpu().double()
    q, scale, mask = quantize_ternary(W, per='tensor')
    in_dim = W.shape[1]

    text_path = Path(__file__).resolve().parents[1] / \
        'data-charlm' / 'tiny_shakespeare_input.txt'
    text = text_path.read_text()[:40000] if text_path.exists() else \
        'The quick brown fox jumps over the lazy dog. ' * 64
    feats: list[torch.Tensor] = []
    hook = lin.register_forward_hook(
        lambda mod, inp, out: feats.append(inp[0].detach().float().cpu()))
    ids = tok(text, return_tensors='pt').input_ids
    with torch.no_grad():
        for i in range(0, ids.shape[1], 256):
            chunk = ids[:, i:i + 256].to(dev)
            if chunk.shape[1] < 2:
                break
            model(chunk)
            if sum(f.reshape(-1, in_dim).shape[0] for f in feats) \
                    >= args.calib_n:
                break
    hook.remove()
    x = torch.cat([f.reshape(-1, in_dim) for f in feats])[:args.calib_n]
    print(f'[{_now()}] 較正活性 {tuple(x.shape)} 収集完了')
    del model
    if dev == 'cuda':
        torch.cuda.empty_cache()
    return q.numpy(), float(scale), x.double().numpy()


def run_bitnet(args) -> dict:
    from griffith_logic.analysis.neuron_bound import greedy_agreement_select
    sys.path.insert(0, os.path.dirname(__file__))
    from neuron_lut_lower_bound import _majority_fill_holdout, optimal_threshold_teacher

    q, scale, x = _load_bitnet_layer(args)
    xb = (x > 0).astype(np.int64)
    rng = np.random.default_rng(args.seed)
    rows = rng.permutation(q.shape[0])[:args.n_neurons]
    n = xb.shape[0]
    perm = rng.permutation(n)
    fit_i, val_i = perm[:n // 2], perm[n // 2:]

    grid = _spec_grid(args)
    per_neuron = []
    t0 = time.time()
    for i, r in enumerate(rows):
        sup = np.nonzero(q[r])[0]
        if len(sup) < 8:
            continue
        u = q[r][sup].astype(np.float64)
        y_real = (x @ (q[r] * scale) > 0).astype(np.int64)
        xs = xb[:, sup]
        s = xs @ u.astype(np.int64)
        # 二値教師: fit半で最適閾値
        t_opt, _, _ = optimal_threshold_teacher(s[fit_i], y_real[fit_i])
        y_bin = (s > t_opt).astype(np.int64)
        rec = {
            'neuron': int(r), 'G': int(len(sup)),
            'ceil_thresh_val': float(
                (y_bin[val_i] == y_real[val_i]).mean()),
            'arms': {},
        }

        # arm A: top-|W|6の閾値LUT(bias再較正、アンカー)
        order = np.argsort(-np.abs(q[r][sup].astype(np.float64)),
                           kind='stable')
        top6 = order[:6]
        dropped = order[6:]
        bshift = float((u[dropped] * xs[fit_i][:, dropped].mean(0)).sum())
        pred_a = (xs[val_i][:, top6] @ u[top6].astype(np.int64)
                  + bshift - t_opt > 0).astype(np.int64)
        rec['arms']['A_top6'] = {
            'a2': float((pred_a == y_bin[val_i]).mean()),
            'a3': float((pred_a == y_real[val_i]).mean()),
            'n_luts': 1,
        }
        # 参考: greedy6 majority-fill(P1の知見の追試)
        g6, _ = greedy_agreement_select(xs[fit_i], y_bin[fit_i], k=6)
        rec['arms']['greedy6_lut'] = {
            'a2': _majority_fill_holdout(
                xs[fit_i], y_bin[fit_i], xs[val_i], y_bin[val_i], g6),
            'a3': _majority_fill_holdout(
                xs[fit_i], y_real[fit_i], xs[val_i], y_real[val_i], g6),
            'n_luts': 1,
        }

        # arm B: 厳密加算木はa2=1.0が構成保証 → LUT数のみ算出
        # (BitNet規模のIR構築は重いので代表ニューロンのみ実測)
        if i < args.adder_verify_n:
            gb = exact_adder_baseline_layer(
                q[r][sup][None, :], np.array([t_opt]))
            pred_b = simulate(gb, xs[val_i])[:, 0]
            a2_b = float((pred_b == y_bin[val_i]).mean())
            assert a2_b == 1.0, f'baseline B非厳密: {a2_b}'
            rec['arms']['B_adder'] = {
                'a2': 1.0,
                'a3': rec['ceil_thresh_val'],
                'n_luts': gb.metadata['decompose']['n_luts_total'],
                'verified': True,
            }

        # arm C: 温度計分解グリッド(fitで構築、valで測定)
        for label, spec in grid:
            circ = decompose_threshold_neuron(u, float(t_opt),
                                              xs[fit_i], spec)
            pred_c = eval_circuit(circ, xs[val_i])
            rec['arms'][f'C_{label}'] = {
                'a2': float((pred_c == y_bin[val_i]).mean()),
                'a3': float((pred_c == y_real[val_i]).mean()),
                'n_luts': circ.meta['n_luts'],
                'depth': circ.meta['depth'],
                'error_bound': circ.meta['error_bound'],
                'budget_met': circ.meta['budget_met'],
            }

        # arm C2.5: 入力thermometer多bit化(ceil_thresh<0.95の突破)
        if args.input_bits > 0:
            w25, thr0, T25 = _thermo_input_expand(
                u, x[:, sup], fit_i, args.input_bits)
            s25 = T25 @ w25
            # 温度計復元後の閾値をfit半でy_realへ再較正
            t25, _, _ = optimal_threshold_teacher(
                np.round(s25[fit_i], 6), y_real[fit_i])
            y25 = (s25 > t25).astype(np.int64)
            rec['ceil25_val'] = float(
                (y25[val_i] == y_real[val_i]).mean())
            for label, spec in grid:
                if not label.endswith('e'):
                    continue  # C2.5はexp_err系のみ(P2初期結果より)
                circ = decompose_threshold_neuron(
                    w25, float(t25), T25[fit_i], spec)
                pred_c = eval_circuit(circ, T25[val_i])
                rec['arms'][f'C25_{label}'] = {
                    'a2': float((pred_c == y25[val_i]).mean()),
                    'a3': float((pred_c == y_real[val_i]).mean()),
                    'n_luts': circ.meta['n_luts'],
                    'depth': circ.meta['depth'],
                    'error_bound': circ.meta['error_bound'],
                    'budget_met': circ.meta['budget_met'],
                }
        per_neuron.append(rec)
        done = len(per_neuron)
        eta = datetime.now(JST) + timedelta(
            seconds=(time.time() - t0) / done * (len(rows) - i - 1))
        best_c = max((v['a3'] for k, v in rec['arms'].items()
                      if k.startswith('C_')), default=0)
        print(f"[{_now()}] neuron {r:5d} G={rec['G']} "
              f"ceil={rec['ceil_thresh_val']:.3f} "
              f"A={rec['arms']['A_top6']['a3']:.3f} "
              f"bestC(a3)={best_c:.3f} "
              f"[{done}, 終了予定 {eta.strftime('%H:%M:%S')}]")
        if not args.no_wandb:
            log_metrics({'neuron': int(r),
                         'ceil': rec['ceil_thresh_val'],
                         'A_a3': rec['arms']['A_top6']['a3'],
                         'bestC_a3': best_c}, step=i)

    # 腕別集計
    def _agg(key: str, metric: str):
        vals = [rec['arms'][key][metric] for rec in per_neuron
                if key in rec['arms']]
        return ({'mean': float(np.mean(vals)), 'std': float(np.std(vals)),
                 'min': float(np.min(vals))} if vals else None)

    arm_keys = sorted({k for rec in per_neuron for k in rec['arms']})
    summary = {
        'ceil_thresh_val': float(np.mean(
            [r['ceil_thresh_val'] for r in per_neuron])),
        'arms': {k: {'a2': _agg(k, 'a2'), 'a3': _agg(k, 'a3'),
                     'n_luts': _agg(k, 'n_luts')} for k in arm_keys},
    }
    print(json.dumps(summary, indent=1, ensure_ascii=False))
    return {'stage': 'bitnet', 'summary': summary,
            'per_neuron': per_neuron}


def main():
    p = argparse.ArgumentParser(description='温度計部分和分解')
    p.add_argument('--stage', choices=('toy1', 'mnist', 'bitnet'),
                   default='toy1')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--calib-n', type=int, default=1024)
    # arm Cグリッド
    p.add_argument('--bits-grid', type=int, nargs='+', default=[1, 2, 3])
    p.add_argument('--alpha-grid', type=float, nargs='+',
                   default=[0.5, 0.9])
    p.add_argument('--margin-quantile', type=float, default=0.05)
    p.add_argument('--grouping', default='balanced',
                   choices=('balanced', 'sorted', 'random'))
    p.add_argument('--stage2', default='recursive',
                   choices=('recursive', 'sat'))
    # toy1
    p.add_argument('--toy-g', type=int, default=14)
    # mnist
    p.add_argument('--hidden', type=int, nargs='+', default=[64])
    p.add_argument('--teacher-epochs', type=int, default=3)
    p.add_argument('--mnist-train-n', type=int, default=20000)
    p.add_argument('--mnist-test-n', type=int, default=5000)
    p.add_argument('--polish-epochs', type=int, default=5)
    p.add_argument('--polish-lr', type=float, default=2e-3)
    p.add_argument('--polish-grid-index', type=int, default=2,
                   help='polishを走らせるarm Cグリッドのindex')
    # bitnet
    p.add_argument('--layer', type=int, default=0)
    p.add_argument('--n-neurons', type=int, default=256)
    p.add_argument('--input-bits', type=int, default=3,
                   help='Phase 2.5: 入力活性の温度計ビット数(0=無効)')
    p.add_argument('--adder-verify-n', type=int, default=3,
                   help='baseline Bを実IR検証する代表ニューロン数')
    p.add_argument('--quick', action='store_true')
    p.add_argument('--no-wandb', action='store_true')
    args = p.parse_args()

    if args.quick:
        args.toy_g = min(args.toy_g, 12)
        args.n_neurons = min(args.n_neurons, 5)
        args.calib_n = min(args.calib_n, 256)
        args.teacher_epochs = 1
        args.mnist_train_n = 2000
        args.mnist_test_n = 1000
        args.polish_epochs = 1
        args.bits_grid = [1, 2]
        args.alpha_grid = [0.5]

    run_name = f'p2_{args.stage}_{args.grouping}_{args.stage2}_s{args.seed}'
    use_wandb = not args.no_wandb and init_wandb(
        run_name, vars(args),
        tags=['p2', 'thermo-decompose', args.stage],
        group='multilut-decomp', job_type='convert')
    args.no_wandb = not use_wandb

    t0 = time.time()
    payload = {'toy1': run_toy1, 'mnist': run_mnist,
               'bitnet': run_bitnet}[args.stage](args)
    _save(args, args.stage, payload)
    print(f'[{_now()}] 完了(総経過 {time.time() - t0:.0f}s)')
    if use_wandb:
        wandb_finish()


if __name__ == '__main__':
    main()
