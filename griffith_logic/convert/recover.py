"""転写IRの回復学習。

C1直接転写(transcribe.py)やC2二値化lowering(binarize.py)が生成する
IRは、疎化・量子化・don't-care埋めに由来する近似誤差を持つ。C3は
`ir/weights.py` の真理値表→重み逆変換で転写IRを**学習可能な
微分可能DLGNへ再構築**し、少エポックの微調整で失われた精度を回復する。
先行研究のない本プロジェクトの新規性ポイント(変換パイプラインの Stage C)。

行程:
    rebuild_trainable — 転写IRを指定パラメータ化の学習可能モデルへ復元。
        真理値表がそのまま重み初期値になるため、ビルド直後(eval)の
        hard forwardは元IRの simulate とビット厳密に一致する。
    recover           — rebuild → 微調整(既定CrossEntropy、KD損失注入可) →
        離散化 → IR再抽出 → hard精度測定 までを一括実行する。

設計判断:
    - パラメータ化は independent / raw / warp のみ許可する。任意の真理値表を
      厳密表現できる方式に限る(ir/weights.py 参照。softmax(N>2)/iwp は
      単一mintermしか表現できず転写初期化に使えない)
    - build_torch(ir/build.py)は lut_layer の train_info.parameterization を
      見てレイヤを構築するため、転写IRの各lut_layerノードに指定パラメータ化を
      書き込んでから再構築する(既存機構の再利用)。転写IRは直列グラフなので
      build_torch は nn.Sequential を返し、そのまま export_ir で再抽出できる
    - group_sum の τ は転写時 1.0 固定でスコア振幅が大きく softmax が飽和し
      やすい。RecoverConfig.tau で soft 学習時のみ τ を上書きできる
      (Noneなら元IRの値を保つ)
    - build_torch は train_info.selection も読む(既定'soft')ため、
      rebuild_trainable/RecoverConfig に selection を追加し LutLayer の
      ゲート選択方式(SELECTIONS)を再構築時に指定できる。ただし
      selection='hard_st' は LutLayer 側の制約で parameterization が
      'softmax'/'warp' の場合のみ構築可能なため、RECOVER_PARAMETERIZATIONS
      (independent/raw/warp)との組み合わせでは実質 warp のみ有効。
      soft→hard_st の途中カリキュラム(hard_st_after)は
      RecoverConfig にフィールドのみ用意し、指定時は NotImplementedError
      とする(未実装のまま壊れた学習ループを提供しない。TODO参照)
"""

from __future__ import annotations

import copy
import itertools
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..export import export_ir
from ..ir.build import build_torch
from ..ir.schema import IRGraph
from ..ir.simulate import simulate
from ..layers.lut import LutLayer

# 任意の真理値表を厳密表現でき、転写初期化に使えるパラメータ化
RECOVER_PARAMETERIZATIONS = ('independent', 'raw', 'warp')


@dataclass
class RecoverConfig:
    """回復学習の設定。

    Attributes:
        parameterization: 再構築時のLUTパラメータ化
            ('independent' | 'raw' | 'warp')。任意真理値表を厳密表現できる
            方式のみ(RECOVER_PARAMETERIZATIONS)
        epochs: 微調整エポック数
        lr: Adam学習率
        batch_size: 未使用(train_loaderが供給。将来のインメモリ経路用に予約)
        device: 学習デバイス
        tau: soft学習時に上書きするgroup_sumのτ。Noneなら元IRの値を保つ
        selection: 再構築時のLutLayerゲート選択方式
            ('soft' | 'hard_st'、LutLayer.SELECTIONS)。既定'soft'は
            従来動作と同一。'hard_st'はLutLayer側の制約によりparameterization
            が'softmax'/'warp'の場合のみ構築可能(RECOVER_PARAMETERIZATIONSの
            中では'warp'のみ組み合わせ可)
        hard_st_after: soft→hard_stカリキュラムの切替タイミング(epochsに
            対する割合、0〜1)。Noneなら切替なし(既定、従来動作と同一)。
            TODO(未実装): LutLayer.selectionはforward内で参照されるだけの
            単純な属性で実行時変更に対応している(`layer.selection = ...`で
            切替可能)ため、rebuild後のモデルへ直接この属性代入を行う
            curriculumループをrecover()へ組み込むのは技術的に可能。
            本タスクでは範囲外としてフィールドのみ用意し、Noneの場合は
            何もしない(挙動不変)。指定した場合は現状NotImplementedErrorと
            する(壊れた実装を作らない)
    """

    parameterization: str = 'independent'
    epochs: int = 5
    lr: float = 0.01
    batch_size: int = 128
    device: str = 'cpu'
    tau: float | None = None
    selection: str = 'soft'
    hard_st_after: float | None = None

    def __post_init__(self):
        if self.parameterization not in RECOVER_PARAMETERIZATIONS:
            raise ValueError(
                f'parameterizationは{RECOVER_PARAMETERIZATIONS}のみ'
                f'(任意真理値表を厳密表現できる方式): {self.parameterization}'
            )
        if self.selection not in LutLayer.SELECTIONS:
            raise ValueError(
                f'selectionは{LutLayer.SELECTIONS}のみ: {self.selection}'
            )
        if self.hard_st_after is not None:
            raise NotImplementedError(
                'hard_st_afterによるsoft→hard_stカリキュラムは未実装(範囲外)。'
                'rebuild_trainable(..., selection=...)で固定selectionを'
                '指定してください'
            )


def rebuild_trainable(
    graph: IRGraph, parameterization: str = 'independent',
    selection: str = 'soft',
) -> nn.Module:
    """転写IRを指定パラメータ化の学習可能DLGNとして再構築する。

    各lut_layerノードのtrain_info.parameterization/selectionを指定値へ
    書き換えた上でbuild_torch(ir/build.py)へ渡す。真理値表がそのまま
    重み初期値になるため、ビルド直後(eval)のhard forwardは元IRのsimulateと
    ビット厳密に一致する(selectionの値によらない: LutLayer.forwardの
    eval分岐はself.trainingがFalseのときselectionを参照せず、常に
    parameterizationに基づく離散化を行うため)。

    Args:
        graph: 転写IR(入力'x'→lut_layer列→group_sum→'y')
        parameterization: 'independent' | 'raw' | 'warp'。任意の真理値表を
            厳密表現できる方式のみ(RECOVER_PARAMETERIZATIONS)
        selection: 'soft' | 'hard_st'(LutLayer.SELECTIONS)。既定'soft'は
            従来動作と同一。'hard_st'はLutLayer側の制約により
            parameterizationが'softmax'/'warp'の場合のみ構築可能
            (RECOVER_PARAMETERIZATIONSの中では'warp'のみ組み合わせ可。
            'independent'/'raw'を指定するとLutLayer.__init__のassertで
            エラーになる)

    Returns:
        学習可能なnn.Module(直列転写IRならnn.Sequential)

    Raises:
        ValueError: 表現不能なパラメータ化指定
    """
    if parameterization not in RECOVER_PARAMETERIZATIONS:
        raise ValueError(
            f'parameterizationは{RECOVER_PARAMETERIZATIONS}のみ: '
            f'{parameterization}'
        )
    if selection not in LutLayer.SELECTIONS:
        raise ValueError(
            f'selectionは{LutLayer.SELECTIONS}のみ: {selection}'
        )
    g = copy.deepcopy(graph)
    for node in g.nodes:
        if node.type == 'lut_layer':
            # softmax(N>2)/iwp初期化を避け、指定の厳密表現方式へ差し替える
            node.train_info = {
                **node.train_info,
                'parameterization': parameterization,
                'selection': selection,
                'initialized': True,
            }
    return build_torch(g)


def _set_group_sum_tau(model: nn.Module, tau: float | None) -> None:
    """モデル内のGroupSum層のτを上書きする(Noneなら何もしない)。"""
    if tau is None:
        return
    from ..group_sum import GroupSum
    for m in model.modules():
        if isinstance(m, GroupSum):
            m.tau = float(tau)


def _all_patterns(in_width: int) -> np.ndarray:
    """全2^in_width入力パターン(0/1)。厳密hard精度測定に使う(小規模前提)。"""
    return np.array(
        list(itertools.product([0, 1], repeat=in_width)), dtype=np.int64,
    )


@torch.no_grad()
def _hard_accuracy(graph: IRGraph, x: np.ndarray, y: np.ndarray) -> float:
    """IRのsimulate(hard eval)によるargmax分類精度。"""
    scores = simulate(graph, x)  # (B, k) 整数スコア
    pred = scores.argmax(axis=-1)
    return float((pred == y).mean())


@torch.no_grad()
def _soft_accuracy(model: nn.Module, loader, device: str) -> float:
    """微分可能モデル(soft forward)のargmax分類精度。"""
    model.eval()
    correct = 0
    total = 0
    for xb, yb in loader:
        xb = xb.to(device).float()
        yb = yb.to(device)
        out = model(xb)
        correct += int((out.argmax(-1) == yb).sum())
        total += int(yb.numel())
    model.train()
    return correct / total if total else 0.0


def _collect_eval_xy(eval_loader) -> tuple[np.ndarray, np.ndarray]:
    """eval_loaderを走査して(入力0/1配列, ラベル配列)を集める。"""
    xs, ys = [], []
    for xb, yb in eval_loader:
        xs.append(np.asarray(xb.detach().cpu().numpy()).astype(np.int64))
        ys.append(np.asarray(yb.detach().cpu().numpy()).astype(np.int64))
    x = np.concatenate(xs, axis=0)
    y = np.concatenate(ys, axis=0)
    return x, y


def recover(
    graph: IRGraph,
    train_loader,
    eval_loader,
    cfg: RecoverConfig,
    loss_fn=None,
) -> tuple[IRGraph, dict]:
    """回復学習の全行程を実行する。

    rebuild_trainable → 微調整 → 離散化(eval経由でexport_irが実施) →
    IR再抽出 → hard精度測定 の順で処理する。

    Args:
        graph: 転写IR(直列: 入力'x'→lut_layer列→group_sum→'y')
        train_loader: (x, y) を返すDataLoader(xは0/1、CrossEntropy用の
            クラスラベルy)
        eval_loader: hard/soft精度測定用のDataLoader
        cfg: 回復設定
        loss_fn: loss_fn(output, target)。省略時はCrossEntropy。
            将来KD損失(teacherロジット蒸留)を注入できる口

    Returns:
        (回復後IR, metrics):
            metrics = {
                'hard_acc_before': rebuild直後(=元IR)のhard精度,
                'hard_acc_after': 微調整+離散化後のhard精度,
                'soft_acc_before': 微調整前のsoft精度,
                'soft_acc_after': 微調整後のsoft精度,
                'epochs': 実行エポック数,
            }
    """
    device = cfg.device
    loss_fn = loss_fn or F.cross_entropy

    # eval入力・ラベルを先に集約(hard精度はsimulateで厳密測定)
    eval_x, eval_y = _collect_eval_xy(eval_loader)

    # rebuild直後のhard精度(=元転写IRのsimulate。忠実性により一致)
    hard_acc_before = _hard_accuracy(graph, eval_x, eval_y)

    model = rebuild_trainable(
        graph, cfg.parameterization, selection=cfg.selection,
    ).to(device)
    _set_group_sum_tau(model, cfg.tau)

    soft_acc_before = _soft_accuracy(model, eval_loader, device)

    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    model.train()
    for _ in range(cfg.epochs):
        for xb, yb in train_loader:
            xb = xb.to(device).float()
            yb = yb.to(device)
            opt.zero_grad()
            out = model(xb)
            loss = loss_fn(out, yb)
            loss.backward()
            opt.step()

    soft_acc_after = _soft_accuracy(model, eval_loader, device)

    # 離散化+IR再抽出: export_irがeval()モデルへtruth_tables_from_lut_layer
    # (export/discretize.py)を適用し、soft重みをhard真理値表へ離散化する。
    # τは元IRの意味論を保つため、上書き分を元値へ戻してから抽出する
    # (soft学習のτ上書きは離散化・hard評価に影響しない: hard評価はargmaxで
    #  τの正の定数倍に不変だが、再抽出IRのメタ情報を元IRと揃える)
    model.eval()
    if cfg.tau is not None:
        orig_tau = next(
            (float(n.params.get('tau', 1.0)) for n in graph.nodes
             if n.type == 'group_sum'), 1.0,
        )
        _set_group_sum_tau(model, orig_tau)

    in_width = graph.inputs[0].width
    recovered = export_ir(
        model, input_width=in_width, name='convert.recover',
        metadata={'source': 'convert.recover',
                  'parameterization': cfg.parameterization},
    )

    hard_acc_after = _hard_accuracy(recovered, eval_x, eval_y)

    metrics = {
        'hard_acc_before': hard_acc_before,
        'hard_acc_after': hard_acc_after,
        'soft_acc_before': soft_acc_before,
        'soft_acc_after': soft_acc_after,
        'epochs': cfg.epochs,
    }
    return recovered, metrics
