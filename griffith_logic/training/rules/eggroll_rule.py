"""EGGROLL学習ルール。

既存の EGGROLLOptimizer(進化戦略、backprop不要)を
HybridTrainer のルールプラグインとして接続する薄いラッパ。

eggroll注釈されたモジュールの2Dパラメータを requires_grad=False にし、
train_stepごとに「現在バッチで -loss を返すfitness閉包」を評価して
ES更新する。他ルール(bp/dfa)のパラメータはfitness評価中も固定値。
"""

import torch
import torch.nn as nn

from ...eggroll import EGGROLLConfig, EGGROLLOptimizer


class EggrollRule:
    def __init__(self, modules: list[tuple[str, nn.Module]]):
        params = []
        for name, module in modules:
            for p in module.parameters(recurse=True):
                if p.dim() != 2:
                    raise ValueError(
                        f'EGGROLL対象 {name} に2D以外のパラメータ: {tuple(p.shape)}'
                    )
                p.requires_grad_(False)  # autogradから除外(ESで直接更新)
                params.append(p)
        if not params:
            raise ValueError('EGGROLL: 対象パラメータが空')
        # 全モジュールで共通のcfg(最初の注釈を採用)
        spec = modules[0][1]._learning_spec
        self.cfg = spec.eggroll or EGGROLLConfig()
        self.optimizer = EGGROLLOptimizer(params, self.cfg)

    def step(self, model: nn.Module, loss_closure, t: int) -> dict:
        """loss_closure: 現在のパラメータ値でlossを返す。

        モデルのtrain/evalモードは呼び出し側の状態を尊重する
        (train=soft緩和のfitness、eval=hard離散のfitness)。
        """

        def fitness():
            with torch.no_grad():
                return -float(loss_closure())

        return self.optimizer.step(fitness, t)
