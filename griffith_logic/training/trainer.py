"""ハイブリッド学習トレーナ。

with_learning() で注釈された学習ルールをモジュール単位で解決し、
1つのtrain_stepの中で BP / DFA / EGGROLL / AP(Akiho Propagation) を協調実行する。

ルールごとの機構:
    bp      — 通常のautograd + optimizer(注釈なしのデフォルト)
    dfa     — フックで局所グラフを分離し、出力誤差の固定ランダム射影を
              param.gradに注入(rules/dfa.py)
    eggroll — 対象パラメータをautogradから外し、fitness閉包でES更新
              (rules/eggroll_rule.py)
    ap      — 全LutLayer+GroupSum構成のモデル専用。既存APTrainerに委譲
              (rules/ap.py)。他ルールとの混在は未対応(現時点)

保証: 全モジュールがbp(=注釈なし)のとき、train_stepは素のtorch訓練
ループ(forward→loss→backward→step)とbitwise一致する(L4テストで検証)。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..layers.lut import LutLayer
from .rules.dfa import DFARule
from .rules.eggroll_rule import EggrollRule
from .spec import get_learning_spec


class HybridTrainer:
    """学習ルール混在モデルのトレーナ。

    Args:
        model: with_learning注釈済み(または未注釈=全bp)のモデル
        optimizer: bp/dfaパラメータ用optimizer。None時はbp/dfaパラメータが
            あればAdamW(lr=1e-3)を自動生成
        loss_fn: (logits, target) -> スカラーloss
        clamp_raw: rawパラメータ化LutLayerのstep後clamp(0,1)(BinaryBrain意味論)
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer | None = None,
        loss_fn=F.cross_entropy,
        clamp_raw: bool = True,
    ):
        self.model = model
        self.loss_fn = loss_fn
        self.clamp_raw = clamp_raw
        self._t = 0

        # --- モジュール走査: ルール別グループ化(親の注釈を子へ継承) ---
        by_rule: dict[str, list[tuple[str, nn.Module]]] = {
            'ap': [], 'dfa': [], 'eggroll': [],
        }
        annotated_prefix: list[tuple[str, str]] = []  # (name, rule)
        for name, module in model.named_modules():
            spec = get_learning_spec(module)
            if spec is None:
                continue
            if any(name.startswith(p + '.') for p, _ in annotated_prefix):
                continue  # 注釈済みの親の内部は親のルールに従う
            annotated_prefix.append((name, spec.rule))
            if spec.rule != 'bp':
                by_rule[spec.rule].append((name, module))

        # --- AP: 専用経路(モデル全体をAPTrainerに委譲) ---
        self._ap = None
        if by_rule['ap']:
            if by_rule['dfa'] or by_rule['eggroll']:
                raise NotImplementedError(
                    'AP法と他ルール(dfa/eggroll)の混在は未対応(現時点)'
                )
            from .rules.ap import APTrainer
            spec = by_rule['ap'][0][1]._learning_spec
            self._ap = APTrainer(
                model,
                lr=spec.lr if spec.lr is not None else 0.1,
                update_mode=spec.update_mode,
            )
            return  # AP経路ではoptimizer等は不要

        # --- EGGROLL: 対象paramをautogradから除外(bp収集より先) ---
        self._eggroll = EggrollRule(by_rule['eggroll']) if by_rule['eggroll'] else None

        # --- DFA: フック設置 ---
        self._dfa = DFARule(by_rule['dfa']) if by_rule['dfa'] else None

        # --- BP: 残りの学習可能パラメータ ---
        if optimizer is not None:
            self.optimizer = optimizer
        else:
            trainable = [p for p in model.parameters() if p.requires_grad]
            self.optimizer = (
                torch.optim.AdamW(trainable, lr=1e-3) if trainable else None
            )

        # raw層のclamp対象
        self._raw_layers = [
            m for m in model.modules()
            if isinstance(m, LutLayer) and m.parameterization == 'raw'
        ] if clamp_raw else []

    # ------------------------------------------------------------------

    def _output_error(self, logits: torch.Tensor, loss: torch.Tensor,
                      target: torch.Tensor) -> torch.Tensor:
        """DFA用の出力誤差 e = ∂L/∂logits(backward前に呼ぶこと)。

        logitsが大域グラフに乗っていればautogradで厳密に計算(loss_fn任意)。
        全層DFA等でlogitsが定数の場合はcross_entropy(mean)の解析式
        (softmax - onehot)/batch にフォールバックする。
        """
        if logits.requires_grad:
            (e,) = torch.autograd.grad(loss, logits, retain_graph=True)
            return e.detach()
        probs = F.softmax(logits, dim=-1)
        onehot = F.one_hot(target, num_classes=logits.shape[-1]).to(probs.dtype)
        return (probs - onehot) / logits.shape[0]

    def train_step(self, x: torch.Tensor, target: torch.Tensor) -> dict:
        """1バッチの訓練ステップ。Returns: {'loss', 'accuracy', ...}"""
        # --- AP専用経路 ---
        if self._ap is not None:
            return self._ap.train_step(x, target)

        self.model.train()
        if self._dfa is not None:
            self._dfa.begin_step()

        logits = self.model(x)
        loss = self.loss_fn(logits, target)

        # DFA誤差はグラフ解放前に取得
        e = None
        if self._dfa is not None:
            e = self._output_error(logits, loss, target)

        if self.optimizer is not None:
            self.optimizer.zero_grad(set_to_none=True)

        # BP: 大域backward(bpパラメータへ)
        if loss.requires_grad:
            loss.backward()

        # DFA: 出力誤差を射影してparam.gradに注入
        if self._dfa is not None:
            self._dfa.inject(e)

        if self.optimizer is not None:
            self.optimizer.step()

        stats = {}
        # EGGROLL: 現在バッチのfitnessでES更新
        if self._eggroll is not None:
            def loss_closure():
                return self.loss_fn(self.model(x), target)
            es_stats = self._eggroll.step(self.model, loss_closure, self._t)
            stats.update({f'eggroll/{k}': v for k, v in es_stats.items()})

        # raw層のBinaryBrain意味論: [0,1]クリップ
        for layer in self._raw_layers:
            layer.weights.data.clamp_(0.0, 1.0)

        self._t += 1
        with torch.no_grad():
            acc = (logits.argmax(dim=-1) == target).float().mean().item()
        stats.update({'loss': float(loss.detach()), 'accuracy': acc})
        return stats

    def close(self):
        """フック解除(モデルを素の状態に戻す)"""
        if getattr(self, '_dfa', None) is not None:
            self._dfa.remove()
