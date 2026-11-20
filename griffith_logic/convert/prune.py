"""C1' 枝刈り転写パイプラインの再利用部品(ROADMAP-v5'、D4でも使用)。

「密結合の事前学習MLPをnaiveに疎化転写すると崩壊する」ことが実証された。
根本原因はfan-in疎化で、
落とした接続のバイアス再較正項が残存接続の寄与を圧倒しニューロンが定数化する点にある。

本モジュールはpivot後の方針を実装する:
    事前学習重みを初期値に、fan-in制約を「反復magnitude枝刈り」で漸進的に課しながら
    STE活性で短期fine-tune → fan-in内に収まった重みは transcribe_mlp で厳密転写できる。

キーとなる意味論的整合:
    枝刈り後モデルの隠れ活性は y_i = 1[Σ_{j∈S_i} W_ij x_j + b_i > 0](S_i=残存接続)。
    transcribe_mlp(spec.n_inputs=最終fanin)は各行の |W| 上位k接続を選ぶが、枝刈り済み
    重みでは非ゼロが正確にk本なので上位k=残存接続と一致する。落とした接続は既にゼロ
    (再較正項もゼロ)なので、転写は fan-in疎化誤差ゼロ = 厳密(argmax全一致)になる。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class PruneSchedule:
    """反復magnitude枝刈りのスケジュール('のfan-in漸進スケジュール)。

    Attributes:
        fanins: 段階的fan-in目標。各段でこの本数まで枝刈りしてfine-tuneする。
            最終値が転写時のfan-in(transcribe_mlp spec.n_inputs と一致させる)。
        epochs_per_stage: 各段のfine-tuneエポック数。
        lr: fine-tuneの学習率(Adam想定)。
    """

    fanins: tuple[int, ...] = (32, 12, 6)
    epochs_per_stage: int = 2
    lr: float = 1e-3

    def __post_init__(self) -> None:
        if not self.fanins:
            raise ValueError('faninsは1段以上必要')
        if any(k < 1 for k in self.fanins):
            raise ValueError(f'faninsは全て1以上: {self.fanins}')
        if self.epochs_per_stage < 1:
            raise ValueError(f'epochs_per_stageは1以上: {self.epochs_per_stage}')


def fanin_mask(weight: torch.Tensor, k: int) -> torch.Tensor:
    """(out, in) 重みの各行で |W| 上位k本のみ1、他0の0/1マスクを返す。

    fan-in(=各出力ニューロンの入力接続数)をkに制限するための行方向マスク。
    タイ(同一|W|)がある場合の残す本数は torch.topk の順序に従い、常に厳密にk本
    (in以下ならk本、in未満のときは全接続=in本)を残す。

    Args:
        weight: (out, in) の重み(既存の枝刈りマスクが乗っていてもよい)
        k: 各行で残す接続数(in以下にクランプ)

    Returns:
        weight と同形・同dtype・同deviceの0/1マスク(各行の非ゼロ数<=k)
    """
    out_dim, in_dim = weight.shape
    k = min(k, in_dim)
    mask = torch.zeros_like(weight)
    # 各行の |W| 上位k本のindexを立てる。既にゼロの接続も含めtopkは常にk本選ぶが、
    # 反復枝刈りでは前段マスクを乗せた重みを渡すため単調にfan-inが縮む。
    topk_idx = weight.abs().topk(k, dim=1).indices        # (out, k)
    mask.scatter_(1, topk_idx, 1.0)
    return mask


def _hidden_linears(model: nn.Module) -> list[nn.Linear]:
    """転写器と同じ規約で隠れ層Linear(最終層を除く)を列挙する。

    transcribe_mlp と同一の探索(children優先、なければmodules全走査)で
    nn.Linear列を得て、最終層(=出力層)を除いた前段を返す。
    """
    linears = [m for m in model.children() if isinstance(m, nn.Linear)]
    if not linears:
        linears = [m for m in model.modules() if isinstance(m, nn.Linear)]
    if not linears:
        raise ValueError('枝刈り対象のnn.Linearが見つからない')
    return linears[:-1]


def max_fanin(model: nn.Module) -> int:
    """隠れ層Linear(最終層除く)の実効fan-in最大値(1行の非ゼロ数の最大)を返す。

    枝刈り後に「全ニューロンが目標fan-in以下に収まったか」を検証するために使う。
    隠れ層が無い(Linear1枚)モデルでは0を返す。
    """
    hidden = _hidden_linears(model)
    if not hidden:
        return 0
    best = 0
    with torch.no_grad():
        for lin in hidden:
            nnz_per_row = (lin.weight != 0).sum(dim=1)     # (out,)
            best = max(best, int(nnz_per_row.max().item()))
    return best


def _apply_masks(linears: list[nn.Linear], masks: list[torch.Tensor]) -> None:
    """各Linearの重みにマスクを乗算(in-place)。ゼロ化された接続を確定させる。"""
    with torch.no_grad():
        for lin, mask in zip(linears, masks, strict=True):
            lin.weight.mul_(mask)


def _register_grad_masks(
    linears: list[nn.Linear], masks: list[torch.Tensor]
) -> list:
    """勾配にマスクを掛けるhookを登録する。枝刈り接続をfine-tune中も0に固定する。

    backwardでマスク済み位置の勾配を0にすることで、optimizer stepでも
    momentum等由来の復活を防ぐ(forward前に再度マスクも掛ける二重防御)。
    Returns: 除去用のhandleリスト。
    """
    handles = []
    for lin, mask in zip(linears, masks, strict=True):
        # クロージャでmaskを束縛
        def _hook(grad, mask=mask):
            return grad * mask
        handles.append(lin.weight.register_hook(_hook))
    return handles


def apply_fanin_pruning(
    model: nn.Module,
    schedule: PruneSchedule,
    train_fn: Callable[[nn.Module, int], dict],
    eval_fn: Callable[[nn.Module], float],
) -> dict:
    """隠れ層Linearへ反復magnitude枝刈り+fine-tuneを適用する(C1'の中核)。

    各段(schedule.fanins の各値k)で:
        1. 現在の重みから fan-in マスクを計算(前段マスクの上に更に縮める)
        2. マスクを重みへ乗算 + 勾配hookで枝刈り位置を0固定
        3. train_fn(model, schedule.epochs_per_stage) で fine-tune(常にSTE活性)
        4. 段完了時に再度マスクを乗算(step由来の微小な復活を除去)
        5. eval_fn(model) で hard acc を記録

    weight_mask はモジュールにバッファとして保持せず、hook+乗算で確定させる方式。
    fine-tune後の重みは各行が正確にk本の非ゼロを持つ(max_fanin(model)<=最終k)。

    Args:
        model: STE活性の隠れ層を持つMLP(nn.Linear列)。枝刈りはin-placeで進む
        schedule: 段階fan-inスケジュール
        train_fn: (model, epochs) -> 学習ログdict。1段のfine-tuneを担う(STE活性・
            optimizer生成は呼び出し側。この関数内では毎epoch forward前提)
        eval_fn: (model) -> hard acc(float)

    Returns:
        {'stages': [{'fanin', 'max_fanin_after', 'acc', 'train_log'} ...],
         'final_fanin', 'final_acc'} の段階メトリクス履歴。
    """
    hidden = _hidden_linears(model)
    if not hidden:
        raise ValueError('枝刈り対象の隠れ層Linearが無い(Linear1枚のMLP)')

    history: dict = {'stages': []}
    handles: list = []
    for stage_i, k in enumerate(schedule.fanins):
        # 1-2. マスク計算・適用・勾配hook登録(前段hookは張り替える)
        for h in handles:
            h.remove()
        masks = [fanin_mask(lin.weight.detach(), k) for lin in hidden]
        _apply_masks(hidden, masks)
        handles = _register_grad_masks(hidden, masks)

        # 3. fine-tune(train_fn内でepochs分回す)
        train_log = train_fn(model, schedule.epochs_per_stage)

        # 4. step由来の復活を除去(hookで勾配は0だが数値の念押し)
        _apply_masks(hidden, masks)

        # 5. 評価
        acc = eval_fn(model)
        mf = max_fanin(model)
        history['stages'].append({
            'stage': stage_i,
            'fanin': k,
            'max_fanin_after': mf,
            'acc': acc,
            'train_log': train_log,
        })

    for h in handles:
        h.remove()

    history['final_fanin'] = schedule.fanins[-1]
    history['final_acc'] = history['stages'][-1]['acc'] if history['stages'] else None
    return history
