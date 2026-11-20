"""勾配ガイド付きゲートflip修復: 大規模LUT網向けの離散修復。

repair_flip(dontcare_repair)の座標降下はO(ニューロン数×16×forward)で
大規模網(数万〜数十万ニューロン)には使えない。本モジュールは
**1回のbackwardで全ニューロンの「flip影響度」を同時に推定して
対象ニューロンをtop-Kに絞り**、絞ったニューロンだけ16ゲートの
厳密評価で最良ゲートを選ぶ:

    勾配 = どのニューロンを見るか(選択)
    厳密評価 = どのゲートにするか(採否)

原理: hard動作点(現在のargmaxゲート)における損失の真理値表方向の勾配
    g_tt[o,e] = Σ_batch grad_out[b,o] · basis[b,o,e]
から、一次予測利得 gain(o) = max_g g_tt[o]·(tt_cur − tt_g) を全ニューロン
一括計算(1 backward)。gain上位K件のみ、16ゲートを実際に差し替えて
厳密hard評価し、**(エラー数, 損失)の辞書式順序で改善する場合のみ採択**
(エラー修正を最優先し、マージン拡大のみのflipは同エラー数のとき許可)。

dense LutLayer(N=2, softmax 16ゲートモード)で構成される任意モデル
(dense DLGN / RDDLGN / attention系、conv網のdense尾部)に適用可能。

拡張: n_inputs∈{4,6}のindependent/warpパラメータ化にも対応。
n=2は全16真理値表を厳密評価(従来どおり)、n>2は2^2^n全列挙が不可能な
ため**現真理値表の単一エントリflip**(2^n候補)を評価する。hard表の
書き戻しはweights_from_truth_tables(任意n対応)。softmax N>2は
hard表を表現できないため対象外。wiring='learnable'とwh_degree_cap
付き層も対象外(patched forwardが固定indices前提のため/キャップ違反回避)。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import torch
import torch.nn as nn

from ..core.basis import _compute_basis_butterfly
from ..ir.weights import weights_from_truth_tables
from ..layers import LutLayer

# eval_metrics(model) -> (primary, secondary)。primary=エラー数/率、
# secondary=損失。辞書式で小さいほど良い
Metrics = tuple[float, float]


@dataclass
class GuidedFlipReport:
    metrics_before: Metrics
    metrics_after: Metrics
    n_proposed: int = 0
    n_accepted: int = 0
    accepted: list[dict] = field(default_factory=list)


def _lex_better(a: Metrics, b: Metrics, eps: float = 1e-12) -> bool:
    """a が b より辞書式に良い(小さい)か。"""
    if a[0] < b[0] - eps:
        return True
    if a[0] > b[0] + eps:
        return False
    return a[1] < b[1] - eps


def _repairable_layers(model: nn.Module) -> list[LutLayer]:
    layers = []
    for m in model.modules():
        if not isinstance(m, LutLayer):
            continue
        if getattr(m, 'wiring', 'fixed') != 'fixed':
            continue  # learnable配線はpatched forwardが未対応
        if getattr(m, 'wh_degree_cap', None) is not None:
            continue  # 度数キャップ層への任意TT書き戻しはキャップ違反
        if getattr(m, '_use_16gate', False):
            layers.append(m)
        elif m.parameterization in ('independent', 'warp'):
            layers.append(m)
        # softmax N>2はhard表(複数エントリ1)を表現できないため対象外
    assert layers, '修復可能なLutLayerが見つからない'
    return layers


@torch.no_grad()
def _neuron_tts(layer: LutLayer) -> torch.Tensor:
    """現在のhard真理値表 (out_dim, 2^n)(パラメータ化別のeval離散化)。"""
    if getattr(layer, '_use_16gate', False):
        return layer.gate_table[layer.weights.argmax(-1)]
    if layer.parameterization == 'independent':
        return (torch.sigmoid(layer.weights) > 0.5).to(layer.weights.dtype)
    if layer.parameterization == 'warp':
        from ..core.basis import _build_wh_entry_matrix
        h = _build_wh_entry_matrix(layer.n_inputs).to(
            device=layer.weights.device, dtype=layer.weights.dtype)
        return ((layer.weights @ h) > 0).to(layer.weights.dtype)
    raise ValueError(f'未対応のパラメータ化: {layer.parameterization}')


@torch.no_grad()
def _write_neuron_tt(layer: LutLayer, o: int, tt: int) -> None:
    param = ('softmax16g' if getattr(layer, '_use_16gate', False)
             else layer.parameterization)
    w = weights_from_truth_tables(param, layer.n_inputs, [tt])
    layer.weights[o].copy_(w[0].to(layer.weights.device, layer.weights.dtype))


class _GradCollector:
    """hard forwardの各LutLayerで g_tt[o,e] を収集する。

    eval経路(離散)のforwardをautograd化するため、離散TT行を
    requires_grad付き定数テンソルとして順伝播に挟む。
    """

    def __init__(self, layers: list[LutLayer]):
        self.layers = layers
        self.g_tt: dict[int, torch.Tensor | None] = {}
        self._w_hard: dict[int, torch.Tensor] = {}

    def __enter__(self):
        for li, layer in enumerate(self.layers):
            w_hard = _neuron_tts(layer).clone().requires_grad_(True)
            self._w_hard[li] = w_hard

            def make_fwd(layer=layer, w_hard=w_hard):
                def fwd(x):
                    gathered = x[:, layer.indices.T.reshape(-1)].reshape(
                        x.shape[0], layer.out_dim, layer.n_inputs)
                    basis = _compute_basis_butterfly(gathered, layer.n_inputs)
                    return torch.einsum('boe,oe->bo', basis, w_hard)
                return fwd

            layer._guided_fwd = make_fwd()
        return self

    def run_patched(self, fn: Callable[[], torch.Tensor]) -> torch.Tensor:
        """LutLayer.forwardを一時差し替えてfnを実行。"""
        originals = []
        for layer in self.layers:
            originals.append(layer.__dict__.get('forward'))
            layer.forward = layer._guided_fwd
        try:
            return fn()
        finally:
            for layer, f in zip(self.layers, originals, strict=True):
                if f is None:
                    layer.__dict__.pop('forward', None)
                else:
                    layer.forward = f

    def collect(self):
        for li, w in self._w_hard.items():
            self.g_tt[li] = w.grad.detach() if w.grad is not None else None

    def __exit__(self, *exc):
        for layer in self.layers:
            if hasattr(layer, '_guided_fwd'):
                del layer._guided_fwd
        return False


def guided_flip_repair(
    model: nn.Module,
    loss_on_batch: Callable[[nn.Module], torch.Tensor],
    eval_metrics: Callable[[nn.Module], Metrics],
    *,
    topk: int = 32,
    passes: int = 4,
    grad_batches: int = 1,
) -> GuidedFlipReport:
    """勾配ガイド付きflip修復。

    Args:
        model: LutLayer(16ゲートモード)を含むモデル
        loss_on_batch: model->スカラー損失(勾配収集用。訓練ミニバッチの
            hard損失。通常どおりmodel(x)を呼べばよい — hard-autograd経路への
            差し替えは本関数が行う)
        eval_metrics: model->(エラー数or率, 損失)。採否判定用の厳密hard評価
            (検証バッチ)。辞書式で小さいほど良い
        topk: 1パスで厳密評価するニューロン数
        passes: パス数(改善が無ければ早期終了)
        grad_batches: 勾配収集に使うloss_on_batch呼び出し回数

    Returns:
        GuidedFlipReport(metrics_before/after, 採択記録)
    """
    model.eval()
    layers = _repairable_layers(model)

    cur = eval_metrics(model)
    report = GuidedFlipReport(metrics_before=cur, metrics_after=cur)

    for _ in range(passes):
        # 1. 勾配収集(hard動作点、1 backward)
        with _GradCollector(layers) as col:
            for _ in range(grad_batches):
                loss = col.run_patched(lambda: loss_on_batch(model))
                loss.backward()
            col.collect()

            # 2. ニューロン選択: 一次予測利得の全ニューロン一括計算
            neuron_cands = []  # (gain, li, o)
            for li, layer in enumerate(layers):
                g = col.g_tt[li]
                if g is None:
                    continue
                cur_tt = _neuron_tts(layer).to(g.dtype)
                if layer.n_inputs == 2:
                    # 全16真理値表で一次スコア(16g/independent/warp共通:
                    # n=2のTT全列挙はどのパラメータ化でも16通り)
                    all_tts = torch.tensor(
                        [[(t >> k) & 1 for k in range(4)] for t in range(16)],
                        dtype=g.dtype, device=g.device)     # (16, 4)
                    scores = g @ all_tts.T                  # (out, 16)
                    cur_score = (g * cur_tt).sum(-1)
                    gain = cur_score - scores.min(-1).values  # (out,)
                else:
                    # n>2: 単一エントリflipの一次利得
                    # flip e: Δscore = g[o,e]·(1−2t_e) → 利得 = g·(2t−1)
                    gain = (g * (2.0 * cur_tt - 1.0)).max(-1).values
                k = min(topk, gain.numel())
                top = torch.topk(gain, k)
                for val, o in zip(top.values.tolist(), top.indices.tolist(), strict=True):
                    if val > 0:
                        neuron_cands.append((val, li, o))
        neuron_cands.sort(reverse=True)
        neuron_cands = neuron_cands[:topk]
        report.n_proposed += len(neuron_cands)

        # 3. 絞ったニューロンのみ16ゲート厳密評価(辞書式採否)
        improved = False
        for gain, li, o in neuron_cands:
            layer = layers[li]
            with torch.no_grad():
                old_w = layer.weights[o].clone()
            if layer.n_inputs == 2:
                cand_tts = list(range(16))
            else:
                # n>2: 現TTの単一エントリflip(2^n候補)
                row = _neuron_tts(layer)[o]
                cur_int = int(sum(int(b) << e for e, b in
                                  enumerate(row.tolist())))
                cand_tts = [cur_int ^ (1 << e)
                            for e in range(layer.n_entries)]
            best_tt, best_m = None, cur
            for cand in cand_tts:
                _write_neuron_tt(layer, o, cand)
                m = eval_metrics(model)
                if _lex_better(m, best_m):
                    best_tt, best_m = cand, m
            if best_tt is not None:
                _write_neuron_tt(layer, o, best_tt)
                cur = best_m
                report.n_accepted += 1
                report.accepted.append(
                    {'layer': li, 'neuron': o, 'tt': best_tt,
                     'gain_pred': gain, 'metrics': best_m})
                improved = True
            else:
                with torch.no_grad():
                    layer.weights[o].copy_(old_w)
        if not improved:
            break

    report.metrics_after = cur
    return report


def make_classification_metrics(x: torch.Tensor, y: torch.Tensor):
    """分類タスク用の標準eval_metrics: (エラー数, CE損失)。"""
    import torch.nn.functional as F

    @torch.no_grad()
    def eval_metrics(model: nn.Module) -> Metrics:
        model.eval()
        logits = model(x.round())
        err = float((logits.argmax(-1) != y).sum().item())
        return err, float(F.cross_entropy(logits, y).item())

    return eval_metrics
