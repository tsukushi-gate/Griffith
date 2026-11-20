"""ニューロン単位SAT下界マップ(multilut-decomposition Phase 1)。

問い: 「実LLMの1ニューロンは、fan-in≤6のLUT何個・深さ何段で表現可能か」

対象ニューロン y_real に対し、較正データN点の(入力ビット列, 出力ビット)
部分関数を収集し:
    1. agreement天井: ceil_thresh(最適閾値の二値入力ニューロン上限)と
       ceil_any(任意ブール関数の情報理論上限)。0.76には入力二値化
       損失が混入しているため、これをPhase 2の実現可能性判定に使う
    2. 3者比較: |W|上位6本 vs 貪欲6本 vs 厳密最良6本(majority-fill閉形式)
    3. 最小(深さd, LUT数k)のSAT探索(analysis.neuron_bound.min_circuit_search)
    4. don't-care率とマージン分布分位点(Phase 2適応ビット割当の入力)

対象(--target):
    synthetic     : ランダムternaryニューロン(support G可変)。conf_budget較正と
                    サニティ(G≤6はd=1でSAT必須)用
    mnist-ternary : 自作STE-MLP(MNIST)をternary量子化した隠れニューロン
                    (入力が最初から二値なので ceil=1.0 のサニティ)
    bitnet        : microsoft/bitnet-b1.58-2B-4T-bf16 のFFN gate_proj
                    (実較正活性、残存fan-in≈1563。GPU機B実行)

使用例:
    uv run python experiments/neuron_lut_lower_bound.py --target synthetic --quick --no-wandb
    uv run python experiments/neuron_lut_lower_bound.py --target mnist-ternary --n-neurons 20
    scripts/remote/run.sh GPU機B uv run python experiments/neuron_lut_lower_bound.py \
        --target bitnet --n-neurons 100 --calib-n 512
"""

from __future__ import annotations

import argparse
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

from griffith_logic.analysis.neuron_bound import (  # noqa: E402
    best_subset_select,
    consistency_ceiling,
    greedy_agreement_select,
    majority_fill_agreement,
    min_circuit_search,
)

load_dotenv()
JST = timezone(timedelta(hours=9))

MARGIN_QS = (0.0, 0.01, 0.05, 0.10, 0.25, 0.50)


def _now() -> str:
    return datetime.now(JST).strftime('%H:%M:%S')


# ============================================================
# ニューロン単位の解析コア(対象共通)
# ============================================================

def optimal_threshold_teacher(s: np.ndarray, y_real: np.ndarray
                              ) -> tuple[float, np.ndarray, float]:
    """整数部分和sの最良閾値Tを走査し (T_opt, y_bin, ceil_thresh) を返す。

    候補は unique(s) の半整数境界。二値入力の閾値ニューロン
    1[s > T] が y_real に到達できるagreementの上限を厳密に与える。
    """
    ys = np.asarray(y_real, dtype=np.int64)
    cands = np.unique(s).astype(np.float64)
    # 各uniqueレベルの下側境界 + 全1/全0になる両端
    thresholds = np.concatenate([cands - 0.5, [cands[-1] + 0.5]])
    best_t, best_a = thresholds[0], -1.0
    for t in thresholds:
        a = float(((s > t).astype(np.int64) == ys).mean())
        if a > best_a:
            best_t, best_a = float(t), a
    y_bin = (s > best_t).astype(np.int64)
    return best_t, y_bin, best_a


def _dc_rate(xb: np.ndarray, subset: np.ndarray) -> float:
    """6本部分集合LUTのdon't-care率 = 1 − 観測ユニークアドレス/2^k。"""
    k = len(subset)
    weights = np.left_shift(1, np.arange(k, dtype=np.int64))
    addr = xb[:, subset] @ weights
    return float(1.0 - len(np.unique(addr)) / (1 << k))


def _majority_fill_holdout(fit_x, fit_y, val_x, val_y,
                           subset: np.ndarray) -> float:
    """fit半で多数決LUTを構築し、val半で測るheld-out agreement。

    未観測アドレスはfit全体の多数決ラベルで埋める(don't care既定値)。
    """
    sub = np.asarray(subset, dtype=np.int64)
    k = len(sub)
    weights = np.left_shift(1, np.arange(k, dtype=np.int64))
    addr_f = fit_x[:, sub] @ weights
    cnt = np.bincount(addr_f, minlength=1 << k).astype(np.float64)
    ones = np.bincount(addr_f, weights=fit_y.astype(np.float64),
                       minlength=1 << k)
    default = 1 if fit_y.mean() > 0.5 else 0
    lut = np.where(cnt > 0, (ones * 2 > cnt).astype(np.int64), default)
    addr_v = val_x[:, sub] @ weights
    return float((lut[addr_v] == val_y).mean())


def analyze_neuron(
    xb_S: np.ndarray,
    y_real: np.ndarray,
    u: np.ndarray,
    w_rank: np.ndarray,
    args: argparse.Namespace,
) -> dict:
    """1ニューロンの下界マップ(天井/3者比較/最小(d,k)/マージン)。

    過学習を避けるため、選択・閾値・SAT合成はfit半で行い、agreementは
    val半(held-out)を主指標として報告する(*_fitはin-sample参考値)。

    Args:
        xb_S: (N, G) 支持集合に制限した二値入力
        y_real: (N,) 教師出力ビット(実数入力から計算した真値)
        u: (G,) ternary符号 ∈ {-1, +1}
        w_rank: (G,) |W|順位付け用の実数(master重み。ternary縮退時は同値)
    """
    n, g = xb_S.shape
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(n)
    n_fit = n // 2
    fit_i, val_i = perm[:n_fit], perm[n_fit:]
    s = xb_S @ u.astype(np.int64)

    # --- (1) agreement天井(T_optはfit半で選び、val半で測る) ---
    t_opt, _, ceil_thresh_fit = optimal_threshold_teacher(
        s[fit_i], y_real[fit_i])
    y_bin = (s > t_opt).astype(np.int64)
    ceil_thresh = float((y_bin[val_i] == y_real[val_i]).mean())
    ceil_thresh0 = float(((s[val_i] > 0).astype(np.int64)
                          == y_real[val_i]).mean())
    ceil_any = consistency_ceiling(xb_S, y_real)  # データ性質(全体)

    # --- (4) マージン分布(全体、閾値T_opt) ---
    margins = np.abs(s.astype(np.float64) - t_opt)
    margin_stats = {f'q{q:g}': float(np.quantile(margins, q))
                    for q in MARGIN_QS}
    margin_stats['mean'] = float(margins.mean())

    # --- (2) 3者比較(fit半で選択、val半のheld-out agreementが主指標) ---
    k = args.k
    fx, fy_bin, fy_real = xb_S[fit_i], y_bin[fit_i], y_real[fit_i]
    vx, vy_bin, vy_real = xb_S[val_i], y_bin[val_i], y_real[val_i]
    order_w = np.argsort(-np.abs(w_rank), kind='stable')
    topw = order_w[:k].astype(np.int64)
    m = min(args.cand_m, g)
    cand, _ = greedy_agreement_select(fx, fy_bin, k=m)
    greedy6 = cand[:k]
    best_rel, _ = best_subset_select(fx[:, cand], fy_bin, k=min(k, m))
    best6 = cand[best_rel]

    def _sel(sub: np.ndarray) -> dict:
        return {
            'cols': sub.tolist(),
            'agree_bin': _majority_fill_holdout(fx, fy_bin, vx, vy_bin,
                                                sub),
            'agree_real': _majority_fill_holdout(fx, fy_real, vx, vy_real,
                                                 sub),
            'agree_bin_fit': majority_fill_agreement(fx, fy_bin, sub),
            'agree_real_fit': majority_fill_agreement(fx, fy_real, sub),
            'dc_rate': _dc_rate(xb_S, sub),
        }

    selections = {'topw': _sel(topw), 'greedy': _sel(greedy6),
                  'best': _sel(best6)}
    # 意味論の再現: |W|top-kの閾値LUT(bias再較正)でのagreement(val半)
    dropped = order_w[k:]
    bias_shift = float((u[dropped] * fx[:, dropped].mean(axis=0)).sum())
    pred_thresh = (vx[:, topw] @ u[topw].astype(np.int64)
                   + bias_shift - t_opt > 0).astype(np.int64)
    selections['topw']['thresh_agree_real'] = float(
        (pred_thresh == vy_real).mean())
    selections['topw']['thresh_agree_bin'] = float(
        (pred_thresh == vy_bin).mean())

    # --- (3) 最小(d,k)のSAT探索(fit半・候補M本に制限した相対下界) ---
    ceil_cand = consistency_ceiling(fx[:, cand], fy_bin)
    sat_res = None
    sat_val_agree = None
    if not args.skip_sat:
        sat_res = min_circuit_search(
            fx[:, cand], fy_bin,
            arity=args.arity,
            depths=tuple(args.depths),
            width_grid=tuple(args.width_grid),
            conf_budget=args.conf_budget,
            wall_budget_s=args.neuron_wall_s,
            seed=args.seed,
        )
        # 最小SAT回路のwitnessをval半で評価(汎化の実測)
        wit = sat_res.pop('witness', None)
        topo = sat_res.pop('topology', None)
        if wit is not None and topo is not None:
            from griffith_logic.analysis.sat_capacity import witness_to_ir
            from griffith_logic.ir.simulate import simulate as ir_sim
            pred = ir_sim(witness_to_ir(wit, topo), vx[:, cand])[:, 0]
            sat_val_agree = float((pred == vy_bin).mean())

    return {
        'G': int(g), 'n_pos': int((u > 0).sum()), 'n_neg': int((u < 0).sum()),
        'T_opt': t_opt,
        'ceil_thresh': ceil_thresh, 'ceil_thresh0': ceil_thresh0,
        'ceil_thresh_fit': ceil_thresh_fit,
        'ceil_any': ceil_any,
        'margin': margin_stats,
        'selections': selections,
        'cand_cols': cand.tolist(),
        'ceil_within_cand': ceil_cand,
        'sat': sat_res,
        'sat_val_agree': sat_val_agree,
        'calib_n': int(n), 'fit_n': int(n_fit),
    }


# ============================================================
# 対象別: (xb_S, y_real, u, w_rank) の列を生成
# ============================================================

def gen_synthetic(args) -> list[dict]:
    """ランダムternaryニューロン。G≤6はd=1 SATのサニティ、大Gは較正pilot。"""
    rng = np.random.default_rng(args.seed)
    out = []
    for i in range(args.n_neurons):
        g = args.synthetic_g
        u = rng.choice([-1, 1], size=g).astype(np.int64)
        xb = rng.integers(0, 2, size=(args.calib_n, g), dtype=np.int64)
        thr = float(rng.integers(-max(1, g // 4), max(2, g // 4))) + 0.5
        y = (xb @ u > thr).astype(np.int64)
        out.append({'neuron': i, 'xb_S': xb, 'y_real': y, 'u': u,
                    'w_rank': rng.uniform(0.5, 1.5, size=g)})
    return out


def gen_mnist_ternary(args) -> list[dict]:
    """自作STE-MLPをternary量子化し、--layer 層のニューロンを抽出。

    入力が最初から二値なので y_real は二値入力で厳密に定まり、
    ceil_thresh==1.0 がサニティ条件になる。
    """
    sys.path.insert(0, os.path.dirname(__file__))
    from convert_prune_transcribe_mnist import _auto_device, _flatten_binary
    from convert_ternary_to_dwn_mnist import _train_ste_teacher

    from griffith_logic.convert.binarize import quantize_ternary
    from griffith_logic.datasets.vision import load_mnist

    device = _auto_device(args.device)
    torch.manual_seed(args.seed)
    train_set, _ = load_mnist()
    x_tr, y_tr = _flatten_binary(train_set, n=args.mnist_train_n)
    print(f'[{_now()}] MNIST teacher学習(hidden={args.hidden}, '
          f'epochs={args.teacher_epochs}, device={device})')
    model = _train_ste_teacher(x_tr, y_tr, tuple(args.hidden),
                               args.teacher_epochs, 1e-3, 256, device)

    # ternary量子化をbake(per-channel scale)
    hidden = list(model.linears[:-1])
    with torch.no_grad():
        for lin in hidden:
            # MPSはfloat64非対応なのでCPUへ移してから量子化する
            q, scale, _ = quantize_ternary(
                lin.weight.detach().cpu().double(), per='channel')
            lin.weight.copy_((q * scale).float().to(lin.weight.device))

    # --layer 層の入力活性(二値)を較正データで収集
    from griffith_logic.convert.teacher_models import HeavisideSTE
    h = torch.as_tensor(x_tr[:args.calib_n], dtype=torch.float32,
                        device=device)
    with torch.no_grad():
        for li in range(args.layer):
            lin = hidden[li]
            h = HeavisideSTE.apply(torch.nn.functional.linear(
                h, lin.weight, lin.bias))
    xb_layer = h.cpu().long().numpy()  # (N, in_dim) 0/1

    lin = hidden[args.layer]
    w = lin.weight.detach().cpu().double()
    b = lin.bias.detach().cpu().double()
    q, scale, mask = quantize_ternary(w, per='channel')
    rng = np.random.default_rng(args.seed)
    rows = rng.permutation(w.shape[0])[:args.n_neurons]
    out = []
    for r in rows:
        sup = np.nonzero(mask[r].numpy())[0]
        if len(sup) < 2:
            continue
        u = q[r].numpy()[sup].astype(np.int64)
        # y_real = 1[Wq·h + b > 0](Wq=q*scale。二値入力なので厳密)
        pre = xb_layer @ (q[r].numpy() * float(scale[r]))
        y = (pre + float(b[r]) > 0).astype(np.int64)
        out.append({'neuron': int(r), 'xb_S': xb_layer[:, sup],
                    'y_real': y, 'u': u,
                    'w_rank': w[r].numpy()[sup]})
    return out


def gen_bitnet(args) -> list[dict]:
    """実BitNet FFN gate_proj。実トークン較正活性(tiny_shakespeare)。

    transcribe_real_quant_llm.py(§5)と同じ意味論:
        y_real = 1[x·Wq_row > 0],  xb = (x > 0)
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from griffith_logic.convert.binarize import quantize_ternary

    repo = 'microsoft/bitnet-b1.58-2B-4T-bf16'
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'[{_now()}] {repo} をロード中(device={dev})')
    tok = AutoTokenizer.from_pretrained(repo)
    model = AutoModelForCausalLM.from_pretrained(
        repo, torch_dtype=torch.bfloat16).eval().to(dev)

    block = model.model.layers[args.layer]
    lin = block.mlp.gate_proj
    W = lin.weight.detach().float().cpu().double()
    q, scale, mask = quantize_ternary(W, per='tensor')
    Wq = (q * scale)
    in_dim = W.shape[1]

    # 較正: 多様な実テキスト(tiny_shakespeare)を流して層入力活性を収集
    text_path = Path(__file__).resolve().parents[1] / \
        'data-charlm' / 'tiny_shakespeare_input.txt'
    text = text_path.read_text()[:20000] if text_path.exists() else \
        'The quick brown fox jumps over the lazy dog. ' * 64
    feats: list[torch.Tensor] = []
    hook = lin.register_forward_hook(
        lambda mod, inp, out: feats.append(inp[0].detach().float().cpu()))
    ids = tok(text, return_tensors='pt').input_ids
    ctx = 256
    with torch.no_grad():
        for i in range(0, ids.shape[1], ctx):
            chunk = ids[:, i:i + ctx].to(dev)
            if chunk.shape[1] < 2:
                break
            model(chunk)
            if sum(f.reshape(-1, in_dim).shape[0] for f in feats) \
                    >= args.calib_n:
                break
    hook.remove()
    x = torch.cat([f.reshape(-1, in_dim) for f in feats])[:args.calib_n]
    x = x.double()
    print(f'[{_now()}] 較正活性 {tuple(x.shape)} 収集完了')
    xb = (x > 0).long().numpy()

    rng = np.random.default_rng(args.seed)
    rows = rng.permutation(W.shape[0])[:args.n_neurons]
    out = []
    for r in rows:
        sup = np.nonzero(mask[r].numpy())[0]
        if len(sup) < 2:
            continue
        u = q[r].numpy()[sup].astype(np.int64)
        y = (x.numpy() @ Wq[r].numpy() > 0).astype(np.int64)
        out.append({'neuron': int(r), 'xb_S': xb[:, sup], 'y_real': y,
                    'u': u, 'w_rank': W[r].numpy()[sup]})
    return out


GENERATORS = {
    'synthetic': gen_synthetic,
    'mnist-ternary': gen_mnist_ternary,
    'bitnet': gen_bitnet,
}


# ============================================================
# 集計
# ============================================================

def summarize(records: list[dict]) -> dict:
    """ニューロン横断の分布サマリ(レポート用)。"""
    def _vals(path):
        out = []
        for r in records:
            v = r
            for kk in path:
                v = v.get(kk) if isinstance(v, dict) else None
                if v is None:
                    break
            if v is not None:
                out.append(v)
        return out

    def _ms(vs):
        return ({'mean': float(np.mean(vs)), 'std': float(np.std(vs)),
                 'min': float(np.min(vs)), 'max': float(np.max(vs))}
                if vs else None)

    dk = {}
    for r in records:
        sat = r.get('sat')
        if not sat:
            continue
        best = sat.get('best')
        key = (f"d{best['depth']}k{best['total_luts']}" if best
               else 'unresolved')
        dk[key] = dk.get(key, 0) + 1

    return {
        'n_neurons': len(records),
        'ceil_thresh': _ms(_vals(['ceil_thresh'])),
        'ceil_thresh_fit': _ms(_vals(['ceil_thresh_fit'])),
        'ceil_thresh0': _ms(_vals(['ceil_thresh0'])),
        'ceil_any': _ms(_vals(['ceil_any'])),
        'sat_val_agree': _ms(_vals(['sat_val_agree'])),
        'sel_topw_bin': _ms(_vals(['selections', 'topw', 'agree_bin'])),
        'sel_greedy_bin': _ms(_vals(['selections', 'greedy', 'agree_bin'])),
        'sel_best_bin': _ms(_vals(['selections', 'best', 'agree_bin'])),
        'sel_topw_real': _ms(_vals(['selections', 'topw', 'agree_real'])),
        'sel_greedy_real': _ms(_vals(['selections', 'greedy',
                                      'agree_real'])),
        'sel_best_real': _ms(_vals(['selections', 'best', 'agree_real'])),
        'topw_thresh_real': _ms(_vals(['selections', 'topw',
                                       'thresh_agree_real'])),
        'dc_rate_best': _ms(_vals(['selections', 'best', 'dc_rate'])),
        'margin_q05': _ms(_vals(['margin', 'q0.05'])),
        'min_dk_histogram': dk,
    }


def main():
    p = argparse.ArgumentParser(description='ニューロン単位SAT下界マップ')
    p.add_argument('--target', choices=list(GENERATORS), default='synthetic')
    p.add_argument('--layer', type=int, default=0)
    p.add_argument('--n-neurons', type=int, default=100)
    p.add_argument('--calib-n', type=int, default=256)
    p.add_argument('--cand-m', type=int, default=16,
                   help='SAT/best6の候補入力本数(貪欲prefix)')
    p.add_argument('--k', type=int, default=6)
    p.add_argument('--arity', type=int, default=6, choices=(2, 4, 6))
    p.add_argument('--depths', type=int, nargs='+', default=[1, 2, 3])
    p.add_argument('--width-grid', type=int, nargs='+',
                   default=[1, 2, 4, 8, 16])
    p.add_argument('--conf-budget', type=int, default=200_000)
    p.add_argument('--neuron-wall-s', type=float, default=540.0,
                   help='SAT探索のニューロンあたり壁時計予算(秒)')
    p.add_argument('--skip-sat', action='store_true')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--device', default=None)
    # synthetic用
    p.add_argument('--synthetic-g', type=int, default=12)
    # mnist用
    p.add_argument('--hidden', type=int, nargs='+', default=[64])
    p.add_argument('--teacher-epochs', type=int, default=3)
    p.add_argument('--mnist-train-n', type=int, default=20000)
    p.add_argument('--quick', action='store_true')
    p.add_argument('--no-wandb', action='store_true')
    args = p.parse_args()

    if args.quick:
        args.n_neurons = min(args.n_neurons, 5)
        args.calib_n = min(args.calib_n, 64)
        args.teacher_epochs = 1
        args.mnist_train_n = 2000
        args.depths = [1, 2]
        args.width_grid = [1, 2, 4]
        args.conf_budget = 50_000

    run_name = (f'nlb_{args.target}_m{args.cand_m}_n{args.calib_n}'
                f'_s{args.seed}')
    use_wandb = not args.no_wandb and init_wandb(
        run_name, vars(args),
        tags=['p1', 'neuron-bound', args.target],
        group='multilut-decomp', job_type='analysis')

    t0 = time.time()
    neurons = GENERATORS[args.target](args)
    print(f'[{_now()}] 対象ニューロン {len(neurons)}個 (target={args.target})')

    records = []
    for i, nd in enumerate(neurons):
        t_n = time.time()
        rec = analyze_neuron(nd['xb_S'], nd['y_real'], nd['u'],
                             nd['w_rank'], args)
        rec['neuron'] = nd['neuron']
        rec['wall_s'] = time.time() - t_n
        records.append(rec)
        done = i + 1
        eta = datetime.now(JST) + timedelta(
            seconds=(time.time() - t0) / done * (len(neurons) - done))
        sat_s = (rec['sat']['best'] if rec.get('sat') else None)
        print(f"[{_now()}] neuron {nd['neuron']:5d} G={rec['G']:5d} "
              f"ceil={rec['ceil_thresh']:.3f} "
              f"top6={rec['selections']['topw']['agree_bin']:.3f} "
              f"greedy6={rec['selections']['greedy']['agree_bin']:.3f} "
              f"best6={rec['selections']['best']['agree_bin']:.3f} "
              f"min(d,k)={sat_s} {rec['wall_s']:.1f}s "
              f"[{done}/{len(neurons)}, 終了予定 "
              f"{eta.strftime('%H:%M:%S')}]")
        if use_wandb:
            log_metrics({
                'neuron': nd['neuron'], 'G': rec['G'],
                'ceil_thresh': rec['ceil_thresh'],
                'ceil_any': rec['ceil_any'],
                'topw_bin': rec['selections']['topw']['agree_bin'],
                'greedy_bin': rec['selections']['greedy']['agree_bin'],
                'best_bin': rec['selections']['best']['agree_bin'],
                'neuron_wall_s': rec['wall_s'],
            }, step=i)

    summary = summarize(records)
    print(f'\n[{_now()}] ==== サマリ ({args.target}) ====')
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    out_dir = Path(__file__).resolve().parents[1] / 'results' / 'multilut'
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(JST).strftime('%Y%m%d_%H%M%S')
    out_path = out_dir / f'neuron_lower_bound_{args.target}_{ts}.json'
    with open(out_path, 'w') as f:
        json.dump({'args': vars(args), 'summary': summary,
                   'records': records}, f, indent=1, ensure_ascii=False)
    print(f'[{_now()}] 保存: {out_path}  (総経過 {time.time() - t0:.0f}s)')

    if use_wandb:
        log_metrics({f'summary/{k}': v['mean']
                     for k, v in summary.items()
                     if isinstance(v, dict) and 'mean' in v})
        wandb_finish()


if __name__ == '__main__':
    main()
