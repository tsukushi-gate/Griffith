"""事前学習MLP→検証済みIR変換パイプラインの製品化(ROADMAP-v5)。

`experiments/convert_prune_transcribe_mnist.py` / `convert_int_lowering_mnist.py`
で確立した勝ちレシピ(mnist-intlower-v1)をライブラリ関数へ切り出す。エディタ
backendからも実験スクリプトからも同一の入口 `convert_pretrained_mlp`
を叩けるようにするのが目的。

パイプライン(各phaseはprogressコールバックで進捗0-1を報告する):
    'pruning'    — 反復magnitude枝刈り+STE fine-tune(段階fan-inスケジュール)
    'quantizing' — ニューロン別スケール整数量子化(必要なら短期QATで回復)
    'lowering'   — unary展開+popcount加算木で厳密lowering → IR
    'verifying'  — IR validate + 量子化torch参照 ⇔ simulate の全件argmax一致検証

厳密性の意味論:
    隠れ層は y_i=1[Σ_j q_ij x_j + b'_i > 0](量子化閾値ニューロン)、最終層は
    共有スケール整数スコアのargmax。lower_int_mlp はこの関数を厳密に回路化する
    ため、量子化torch参照(_quantized_int_forward_pred)とsimulateのargmaxは
    eval全件で一致しなければならない(不一致はloweringバグとして例外送出)。

既存 convert/*(prune / binarize / transcribe)関数は変更せず import 利用のみ。
実験スクリプトはこの関数を使わずとも従来通り動く(重複を許す設計)。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from griffith_logic.convert.binarize import (
    lower_int_mlp,
    quantize_threshold_layer,
)
from griffith_logic.convert.prune import (
    PruneSchedule,
    apply_fanin_pruning,
    max_fanin,
)
from griffith_logic.convert.transcribe import _quantize_final
from griffith_logic.ir import IRGraph
from griffith_logic.ir.serialize import ir_to_dict
from griffith_logic.ir.simulate import simulate
from griffith_logic.ir.validate import IRValidationError, validate

__all__ = [
    'ConvertPipelineSpec',
    'ProgressFn',
    'convert_pretrained_mlp',
]

# パイプラインが報告するphase名(この順に遷移する)。'preparing'/'done'/'error'は
# backend側(ConversionSession)の状態でありパイプライン本体は使わない。
PIPELINE_PHASES = ('pruning', 'quantizing', 'lowering', 'verifying')


@dataclass
class ConvertPipelineSpec:
    """mnist-intlower-v1系レシピの全パラメータ。

    Attributes:
        fanins: 段階fan-inスケジュール(反復枝刈りの各段の目標fan-in)。
            単段 (32,) が勝ちレシピ。最終値は加算木loweringのfan-inになる。
        epochs_per_stage: 各枝刈り段のfine-tuneエポック数。
        lr: 枝刈りfine-tuneの学習率(Adam)。
        weight_bits_hidden: 隠れ層のニューロン別スケール量子化ビット幅(勝ち=4)。
        weight_bits_final: 最終層の共有スケール量子化ビット幅(既定8)。
        qat_epochs: 量子化acc低下時のみ実行する短期QAT(量子化STE)のエポック数。
            0で無効。gate_accを下回った場合のみ発火する。
        qat_lr: QATの学習率(Adam)。
        gate_acc: QAT発火の判定閾値(量子化acc<gate_accならQATを試みる)。
            0.0で常にスキップ。
        batch_size: fine-tune/QATのバッチサイズ。
        seed: torch/numpyのシード。
    """

    fanins: tuple[int, ...] = (32,)
    epochs_per_stage: int = 4
    lr: float = 1e-3
    weight_bits_hidden: int = 4
    weight_bits_final: int = 8
    qat_epochs: int = 1
    qat_lr: float = 3e-4
    gate_acc: float = 0.955
    batch_size: int = 128
    seed: int = 0

    def __post_init__(self) -> None:
        if not self.fanins:
            raise ValueError('faninsは1段以上必要')
        if any(k < 1 for k in self.fanins):
            raise ValueError(f'faninsは全て1以上: {self.fanins}')
        if self.epochs_per_stage < 1:
            raise ValueError(f'epochs_per_stageは1以上: {self.epochs_per_stage}')
        if self.weight_bits_hidden < 1 or self.weight_bits_final < 1:
            raise ValueError('weight_bits_hidden/finalは1以上')
        if self.qat_epochs < 0:
            raise ValueError('qat_epochsは0以上')


# (phase名, progress0-1, metricsスナップショット) を受ける進捗コールバック。
# metricsは呼び出し時点までに判明した値の浅いコピー(呼び出し側でmutate禁止)。
ProgressFn = Callable[[str, float, dict], None]


# ============================================================
# 内部ヘルパ(experiments/convert_int_lowering_mnist.py から移設)
# ============================================================

def _hidden_linears_of(model: nn.Module) -> list[nn.Linear]:
    """転写器と同じ規約でnn.Linear列(最終層含む全て)を返す。"""
    linears = [m for m in model.children() if isinstance(m, nn.Linear)]
    if not linears:
        linears = [m for m in model.modules() if isinstance(m, nn.Linear)]
    if not linears:
        raise ValueError('変換対象のnn.Linearが見つからない')
    return linears


@torch.no_grad()
def _eval_acc(model: nn.Module, x: np.ndarray, y: np.ndarray, device: str,
              batch: int = 4000) -> float:
    """(N,in) 0/1 入力に対する素のtorch forward hard accuracy。"""
    model.eval()
    correct = 0
    for i in range(0, len(x), batch):
        xb = torch.as_tensor(x[i:i + batch], dtype=torch.float32, device=device)
        pred = model(xb).argmax(dim=-1).cpu().numpy()
        correct += int((pred == y[i:i + batch]).sum())
    return correct / len(x)


def _make_train_fn(x_tr: np.ndarray, y_tr: np.ndarray, device: str,
                   lr: float, batch_size: int) -> Callable[[nn.Module, int], dict]:
    """apply_fanin_pruning に渡す train_fn を作る(段ごとに新規Adam)。"""
    x_t = torch.as_tensor(x_tr, dtype=torch.float32)
    y_t = torch.as_tensor(y_tr, dtype=torch.int64)
    n = len(x_tr)
    loss_fn = nn.CrossEntropyLoss()

    def train_fn(model: nn.Module, epochs: int) -> dict:
        model.train()
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        logs = []
        for _ in range(epochs):
            perm = torch.randperm(n)
            ep_loss, nb = 0.0, 0
            for i in range(0, n, batch_size):
                idx = perm[i:i + batch_size]
                xb = x_t[idx].to(device)
                yb = y_t[idx].to(device)
                loss = loss_fn(model(xb), yb)
                opt.zero_grad()
                loss.backward()
                opt.step()
                ep_loss += loss.item()
                nb += 1
            logs.append(ep_loss / max(nb, 1))
        return {'epoch_losses': logs}

    return train_fn


@torch.no_grad()
def _quantized_int_forward_pred(model: nn.Module, x: np.ndarray,
                                bits_hidden: int, bits_final: int) -> np.ndarray:
    """lower_int_mlp が実装する意味論の argmax(CPU/float64、決定的)。

    隠れ: quantize_threshold_layer(bits_hidden) で q,β → 1[Σqx+β>0]。
    最終: _quantize_final(bits_final) の共有スケール整数スコア。
    素の model.forward は隠れ重みを量子化しないため、これが厳密性検証の
    比較基準(= lowering が表現する関数)になる。
    """
    linears = _hidden_linears_of(model)
    h = torch.as_tensor(x, dtype=torch.float64)
    for lin in linears[:-1]:
        w = lin.weight.detach().to(torch.float64)
        b = lin.bias.detach().to(torch.float64)
        q, beta = quantize_threshold_layer(w, b, bits_hidden)
        s = h @ q.to(torch.float64).T + beta.to(torch.float64)[None, :]
        h = (s > 0).to(torch.float64)
    fin = linears[-1]
    q, beta = _quantize_final(
        fin.weight.detach().to(torch.float64),
        fin.bias.detach().to(torch.float64), bits_final)
    scores = h.numpy() @ q.numpy().T.astype(np.float64) + beta.numpy()[None, :]
    return scores.argmax(axis=1)


def _quantized_int_acc(model: nn.Module, x: np.ndarray, y: np.ndarray,
                       bits_hidden: int, bits_final: int) -> float:
    pred = _quantized_int_forward_pred(model, x, bits_hidden, bits_final)
    return float((pred == y).sum()) / len(y)


class _QuantHiddenSTE(torch.autograd.Function):
    """隠れ重み量子化のSTE。forward=quantize_threshold_layer、backward=素通し。"""

    @staticmethod
    def forward(ctx, weight, bias, bits):  # noqa: ANN001
        q, beta = quantize_threshold_layer(weight.detach(), bias.detach(), bits)
        ctx.save_for_backward()
        return q.to(weight.dtype), beta.to(bias.dtype)

    @staticmethod
    def backward(ctx, grad_q, grad_beta):  # noqa: ANN001
        return grad_q, grad_beta, None


def _qat_finetune(model: nn.Module, x_tr: np.ndarray, y_tr: np.ndarray,
                  device: str, bits_hidden: int, epochs: int, lr: float,
                  batch_size: int, masks: list[torch.Tensor]) -> None:
    """量子化STEで短期QAT。隠れ層は量子化重みでforward、枝刈りマスクを維持する。

    modelはin-placeで更新される。masksは枝刈りfan-inを固定するための0/1マスク。
    """
    linears = _hidden_linears_of(model)
    hidden = linears[:-1]
    final = linears[-1]
    x_t = torch.as_tensor(x_tr, dtype=torch.float32, device=device)
    y_t = torch.as_tensor(y_tr, dtype=torch.int64, device=device)
    n = len(x_tr)
    loss_fn = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    def forward_qat(xb):  # noqa: ANN001
        h = xb.to(torch.float32)
        for lin in hidden:
            qw, qb = _QuantHiddenSTE.apply(lin.weight, lin.bias, bits_hidden)
            s = h @ qw.T + qb[None, :]
            h = (s > 0).to(torch.float32) + s - s.detach()  # 発火STE(値素通し)
        return final(h)

    model.train()
    for _ in range(epochs):
        perm = torch.randperm(n, device=device)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            logits = forward_qat(x_t[idx])
            loss = loss_fn(logits, y_t[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            with torch.no_grad():
                for lin, m in zip(hidden, masks, strict=True):
                    lin.weight.mul_(m)


def _circuit_stats(ir: IRGraph) -> dict[str, int]:
    lut_layers = [n for n in ir.nodes if n.type == 'lut_layer']
    total_outputs = sum(int(n.params['out_width']) for n in lut_layers)
    return {'n_lut_layers': len(lut_layers), 'total_lut_outputs': total_outputs}


def _simulate_pred(ir: IRGraph, x: np.ndarray, batch: int = 2000) -> np.ndarray:
    preds = []
    for i in range(0, len(x), batch):
        preds.append(simulate(ir, x[i:i + batch]).argmax(axis=1))
    return np.concatenate(preds)


# ============================================================
# データローダ吸い出し(loader → (N,in) 0/1 numpy)
# ============================================================

def _drain_loader(loader) -> tuple[np.ndarray, np.ndarray]:  # noqa: ANN001
    """DataLoader から (N, in) 0/1 int64 特徴量と (N,) ラベルを全件展開する。

    エディタ/実験の二値化MNISTローダは (B,1,28,28) or (B,784) の 0/1 画像と
    整数ラベルを供給する。ここでflattenし0/1を検証する(lowering前提を守る)。
    """
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    for xb, yb in loader:
        x = xb.reshape(xb.shape[0], -1).detach().cpu().numpy().astype(np.int64)
        y = yb.detach().cpu().numpy().reshape(-1).astype(np.int64)
        xs.append(x)
        ys.append(y)
    if not xs:
        raise ValueError('loaderが空(データが1件も供給されない)')
    x_all = np.concatenate(xs, axis=0)
    y_all = np.concatenate(ys, axis=0)
    uniq = set(np.unique(x_all).tolist())
    if not uniq.issubset({0, 1}):
        raise ValueError(
            f'入力が0/1ではない(binarize_threshold=0.5で二値化済みが前提): {sorted(uniq)[:5]}'
        )
    return x_all, y_all


# ============================================================
# 公開API
# ============================================================

def convert_pretrained_mlp(
    model: nn.Module,
    train_loader,  # noqa: ANN001
    eval_loader,  # noqa: ANN001
    spec: ConvertPipelineSpec,
    progress: ProgressFn | None = None,
    device: str = 'cpu',
) -> tuple[IRGraph, dict]:
    """事前学習MLP→(枝刈り+STE ft→量子化+QAT→加算木lowering)→検証済みIR。

    phase遷移: 'pruning'(段ごと)→'quantizing'→'lowering'→'verifying'。
    厳密性検証(量子化torch参照⇔simulate、eval全件argmax一致)込み。

    Args:
        model: nn.Linear列を持つMLP(隠れ活性は発火判定として吸収される。
            TeacherMLPSte 等のSTE-MLP、または素のReLU-MLPでもよい)。in-placeで
            枝刈り・QATされるため、呼び出し側で保持し続けたい場合はcopyを渡すこと。
        train_loader: 学習/較正データ(0/1二値化済み、(B,in)or(B,1,H,W))。
        eval_loader: 厳密性検証+hard acc計測用(全件展開する)。
        spec: レシピパラメータ。
        progress: (phase, progress0-1, metrics)を受けるコールバック。Noneで無効。
        device: 枝刈り/QATの計算デバイス('cpu'/'cuda'/'mps')。量子化参照・
            lowering・simulateは常にCPU/float64で決定的に行う。

    Returns:
        (ir, metrics):
            ir      検証済みIRGraph(lower_int_mlp出力)。
            metrics 各段のacc・最終hard_acc・回路統計・phase履歴を含むdict。

    Raises:
        RuntimeError: 厳密lowering違反(simulateと量子化参照のargmax不一致)。
        IRValidationError: IR検証失敗。
        ValueError: データが不正(0/1でない・空)、spec不正。
    """
    torch.manual_seed(spec.seed)
    np.random.seed(spec.seed)

    def _emit(phase: str, prog: float, metrics: dict) -> None:
        if progress is not None:
            progress(phase, float(max(0.0, min(1.0, prog))), dict(metrics))

    x_tr, y_tr = _drain_loader(train_loader)
    x_te, y_te = _drain_loader(eval_loader)

    metrics: dict = {
        'fanins': list(spec.fanins),
        'weight_bits_hidden': spec.weight_bits_hidden,
        'weight_bits_final': spec.weight_bits_final,
        'n_train': int(len(x_tr)),
        'n_eval': int(len(x_te)),
        'stages': [],
        'pruning_stage_acc': [],
        'quant_acc': None,
        'qat_acc': None,
        'qat_ran': False,
        'hard_acc': None,
        'exact': None,
    }
    bits_h = spec.weight_bits_hidden
    bits_f = spec.weight_bits_final

    # ------------------------------------------------------------------
    # phase 1: 枝刈り + STE fine-tune
    # ------------------------------------------------------------------
    model = model.to(device)
    sched = PruneSchedule(
        fanins=tuple(spec.fanins),
        epochs_per_stage=spec.epochs_per_stage,
        lr=spec.lr,
    )
    train_fn = _make_train_fn(x_tr, y_tr, device, sched.lr, spec.batch_size)
    n_stages = len(sched.fanins)
    stage_i = {'i': 0}
    _emit('pruning', 0.0, metrics)

    def _staged_train(m: nn.Module, epochs: int) -> dict:
        log = train_fn(m, epochs)
        stage_i['i'] += 1
        _emit('pruning', stage_i['i'] / n_stages, metrics)
        return log

    def eval_fn(m: nn.Module) -> float:
        return _eval_acc(m, x_te, y_te, device)

    hist = apply_fanin_pruning(model, sched, _staged_train, eval_fn)
    metrics['stages'] = hist['stages']
    metrics['pruning_stage_acc'] = [s['acc'] for s in hist['stages']]
    metrics['final_ft_acc'] = hist['final_acc']
    metrics['final_max_fanin'] = max_fanin(model)
    _emit('pruning', 1.0, metrics)

    # 枝刈りマスク(QATでfan-in維持)
    hidden = _hidden_linears_of(model)[:-1]
    masks = [(lin.weight.detach() != 0).to(lin.weight.dtype) for lin in hidden]

    # ------------------------------------------------------------------
    # phase 2: 量子化(+ 必要なら短期QAT)
    # ------------------------------------------------------------------
    _emit('quantizing', 0.0, metrics)
    model.to('cpu')
    quant_acc = _quantized_int_acc(model, x_te, y_te, bits_h, bits_f)
    metrics['quant_acc'] = quant_acc
    _emit('quantizing', 0.5, metrics)

    if quant_acc < spec.gate_acc and spec.qat_epochs > 0:
        masks_cpu = [m.to('cpu') for m in masks]
        _qat_finetune(model, x_tr, y_tr, 'cpu', bits_h,
                      spec.qat_epochs, spec.qat_lr, spec.batch_size, masks_cpu)
        qat_acc = _quantized_int_acc(model, x_te, y_te, bits_h, bits_f)
        metrics['qat_acc'] = qat_acc
        metrics['qat_ran'] = True
    _emit('quantizing', 1.0, metrics)

    # ------------------------------------------------------------------
    # phase 3: 加算木lowering(CPU/float64で決定的)
    # ------------------------------------------------------------------
    _emit('lowering', 0.0, metrics)
    model_cpu = model.to('cpu').eval()
    ir = lower_int_mlp(model_cpu, weight_bits_hidden=bits_h,
                       weight_bits_final=bits_f)
    stats = _circuit_stats(ir)
    metrics.update(stats)
    _emit('lowering', 1.0, metrics)

    # ------------------------------------------------------------------
    # phase 4: 検証(validate + 厳密性 + hard acc)
    # ------------------------------------------------------------------
    _emit('verifying', 0.0, metrics)
    errs = validate(ir)
    if errs:
        raise IRValidationError(f'IR検証失敗 {errs[:3]}')
    _emit('verifying', 0.4, metrics)

    ref_pred = _quantized_int_forward_pred(model_cpu, x_te, bits_h, bits_f)
    sim_pred = _simulate_pred(ir, x_te)
    n_mismatch = int((ref_pred != sim_pred).sum())
    if n_mismatch != 0:
        raise RuntimeError(
            f'厳密lowering違反 {n_mismatch}/{len(x_te)} 不一致。'
            f'バグなので停止(bits_hidden={bits_h}, bits_final={bits_f})'
        )
    hard_acc = float((sim_pred == y_te).sum()) / len(y_te)
    metrics['hard_acc'] = hard_acc
    metrics['exact'] = True
    metrics['gate_pass'] = bool(hard_acc >= spec.gate_acc)
    _emit('verifying', 1.0, metrics)

    return ir, metrics


# metrics に IR dict も欲しい呼び出し側(backend)向けヘルパ(convert/*非改変)。
def ir_to_json_dict(ir: IRGraph) -> dict:
    """IRGraph を IR JSON dict へ変換する薄いre-export(backend統合の便宜)。"""
    return ir_to_dict(ir)
