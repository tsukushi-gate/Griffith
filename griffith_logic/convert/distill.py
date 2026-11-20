"""C4 知識蒸留(KD)損失+packed-target蒸留タスク(ROADMAP-v5 Track D)。

二重温度(τ_gs × T)設計 — 本モジュールの中核的注意点
================================================================
studentロジットはGroupSum終端の出力である。GroupSumは
`score = (グループ内の和) / τ_gs`(griffith_logic/group_sum.py)であり、
すでにモデル側温度 τ_gs による軟化が掛かっている。KD側でさらに温度 T で
`softmax(s / T)`を取るため、studentの実効的な軟化係数は **τ_gs × T** となる
(二重温度)。

このため本モジュールでは τ_gs(モデル側=レシピで固定される定数)と
T(KD側=スイープする探索軸)を**実装上も明示的に分離**する。すなわち
kd_kl_loss / make_kd_loss が受け取る`temperature`は KD側 T のみを指し、
τ_gs には一切触れない(τ_gs はモデル定義=GroupSum.tau が保持する)。
利用者が実効温度を揃えたい場合は「T をスイープしつつ τ_gs をレシピ固定」
という運用で分離管理する。

teacherロジットのスケールは student(=τ_gs×T で軟化済み)と無関係なので、
温度だけでは両者の鋭さが揃わない。第一処方として
**Logit Standardization**(サンプルごとのZ-score標準化、arXiv:2403.01427)を
オプションで両ロジットに適用し、スケール/シフト依存を消してから比較する。
Conv-DLGN(NeurIPS 2024)の経験則「teacher使用時は温度を √2 倍する」は
温度初期値の参考値として付記する(本実装は強制しない)。

KL が不調(teacherが過度に鋭い/勾配消失)な場合の fallback として、
中心化ロジットの MSE 損失(kd_mse_loss)も提供する。

packed-target 方式
------------------
TaskSpec.loss_fn(output, target) の契約(griffith_logic/training/tasks.py)を
変えずに runner/エディタの学習ループを無改造で KD に流用するため、
teacherロジットを target に相乗りさせる:

    target = [B, 1 + C]  (float)
        列0     : ラベル(float格納。損失側で long 化)
        列1..C  : teacherロジット

これにより loss_fn/metric_fn は packed target を内部でスライスするだけで済み、
既存の (output, target) 2引数契約を保ったまま KD が動く。

参照論文:
    Hinton et al. 2015, "Distilling the Knowledge in a Neural Network"
        (arXiv:1503.02531) — soft項の T^2 勾配スケール補正
    Sun et al. 2024, "Logit Standardization in Knowledge Distillation"
        (arXiv:2403.01427)
    Conv-DLGN (NeurIPS 2024) — teacher温度 √2 倍の経験則
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from griffith_logic.training.tasks import TaskSpec, _classification_acc

__all__ = [
    'LogitExitHead',
    'dist_loss_seq',
    'kd_kl_loss',
    'kd_kl_loss_seq',
    'kd_mse_loss',
    'make_dense_kd_loss',
    'make_distill_task',
    'make_exit_kd_loss',
    'make_kd_loss',
    'make_seq_kd_loss',
    'make_seq_kd_loss_v2',
    'pack_targets',
    'resolve_schedule',
    'standardize_logits',
]


def kd_kl_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float = 4.0,
) -> torch.Tensor:
    """KL(softmax(t/T) || softmax(s/T)) * T^2 のバッチ平均。

    T は KD側温度のみ(モデル側 τ_gs とは分離。二重温度 τ_gs×T のうち T 成分)。
    T^2 補正は Hinton 2015 (arXiv:1503.02531) の soft項勾配スケール補正:
    softmax(z/T) の勾配が 1/T スケールするため、hard損失と勾配大きさを揃えるべく
    KL に T^2 を掛ける。

    Args:
        student_logits: [B, C] studentロジット(GroupSum出力)。
        teacher_logits: [B, C] teacherロジット。
        temperature: KD温度 T(>0)。

    Returns:
        スカラー(バッチ平均)。s==t のとき 0、常に非負。
    """
    t = float(temperature)
    log_p_student = F.log_softmax(student_logits / t, dim=-1)
    log_p_teacher = F.log_softmax(teacher_logits / t, dim=-1)
    # KL(teacher || student) = Σ p_t (log p_t - log p_s)。log_target=Trueで数値安定。
    kl = F.kl_div(
        log_p_student, log_p_teacher, reduction='batchmean', log_target=True,
    )
    return kl * (t * t)


def kd_kl_loss_seq(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float = 1.0,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """位置ごとのKL(softmax(t/T) || softmax(s/T)) * T^2 を、マスク対象位置で平均する。

    kd_kl_loss([B,C])のシーケンス版。系列モデル(RDDLGN/CharLM蒸留)は
    [B, S, V]で各時刻ごとにロジットを持つため、時刻方向を独立サンプルとして
    扱いつつ、pad位置等を`mask`で平均から除外できるようにする。数式・T^2補正の
    根拠はkd_kl_lossと同一(Hinton 2015のsoft項勾配スケール補正)。

    Args:
        student_logits: [B, S, V] studentロジット。
        teacher_logits: [B, S, V] teacherロジット。
        temperature: KD温度 T(>0)。
        mask: [B, S] のbool/float。Trueまたは非0の位置のみ平均に含める。
            Noneなら全位置を対象にする(全て有効なマスクを渡した場合と等価)。

    Returns:
        スカラー(マスク対象位置での平均)。

    Raises:
        ValueError: maskが全てFalse(有効位置が1つもない)の場合。
    """
    t = float(temperature)
    log_p_student = F.log_softmax(student_logits / t, dim=-1)
    log_p_teacher = F.log_softmax(teacher_logits / t, dim=-1)
    # reduction='none'で位置ごとのKLを保持してから、マスク対象のみ平均する。
    kl = F.kl_div(
        log_p_student, log_p_teacher, reduction='none', log_target=True,
    ).sum(dim=-1)  # [B, S]
    kl = kl * (t * t)

    if mask is None:
        return kl.mean()

    mask = mask.to(dtype=kl.dtype, device=kl.device)
    n_valid = mask.sum()
    if n_valid.item() == 0:
        raise ValueError('mask が全て無効(True/非0の位置なし)です。KD対象位置がありません。')
    return (kl * mask).sum() / n_valid


def kd_mse_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
) -> torch.Tensor:
    """中心化ロジットの MSE(KL fallback)。

    softmax はロジットの定数シフトに不変なので、蒸留の観点では絶対値ではなく
    相対的な形が重要。各サンプルで平均を引いて中心化することで**シフト不変**に
    してから MSE を取る(スケール差は温度/標準化で扱う)。

    Args:
        student_logits: [B, C]。
        teacher_logits: [B, C]。

    Returns:
        スカラー(全要素平均)。s==t のとき 0。
    """
    s = student_logits - student_logits.mean(dim=-1, keepdim=True)
    t = teacher_logits - teacher_logits.mean(dim=-1, keepdim=True)
    return F.mse_loss(s, t)


def standardize_logits(logits: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """サンプルごと Z-score 標準化(Logit Standardization KD, arXiv:2403.01427)。

    各サンプル(最終次元)について平均 0・分散 1 に正規化する。teacher/student の
    ロジットスケールが無関係でも、標準化後は温度非依存に形だけを比較できる。
    アフィン不変: standardize(a*x + b) == standardize(x)(a>0)。

    Args:
        logits: [B, C]。
        eps: 0除算防止(母集団標準偏差に加算)。

    Returns:
        [B, C] 標準化済みロジット。
    """
    mean = logits.mean(dim=-1, keepdim=True)
    # 母標準偏差(unbiased=False)。標準化KDの定義に合わせる。
    std = logits.std(dim=-1, keepdim=True, unbiased=False)
    return (logits - mean) / (std + eps)


def make_kd_loss(
    alpha: float,
    temperature: float,
    standardize: bool = False,
    kind: str = 'kl',
) -> Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
    """KD損失クロージャを構築する。

    返す loss(student_logits, labels, teacher_logits) =
        alpha * CE(student_logits, labels) + (1 - alpha) * KD(student, teacher)

    standardize=True の場合、KD項に入れる**両ロジット**を standardize_logits して
    から比較する(CE項の student_logits には適用しない)。

    Args:
        alpha: hard(CE)項の重み。alpha=1 で純CE、alpha=0 で純KD。
        temperature: KD温度 T(kind='kl' のとき使用。二重温度の T 成分)。
        standardize: KD項の両ロジットを Z-score 標準化するか。
        kind: 'kl'(既定, kd_kl_loss)または 'mse'(kd_mse_loss)。

    Returns:
        loss(student_logits[B,C], labels[B], teacher_logits[B,C]) -> スカラー。
    """
    if kind not in ('kl', 'mse'):
        raise ValueError(f"kind は 'kl' か 'mse': {kind}")

    a = float(alpha)

    def loss_fn(
        student_logits: torch.Tensor,
        labels: torch.Tensor,
        teacher_logits: torch.Tensor,
    ) -> torch.Tensor:
        ce = F.cross_entropy(student_logits, labels.long())
        s = student_logits
        t = teacher_logits
        if standardize:
            s = standardize_logits(s)
            t = standardize_logits(t)
        if kind == 'kl':
            kd = kd_kl_loss(s, t, temperature)
        else:
            kd = kd_mse_loss(s, t)
        return a * ce + (1.0 - a) * kd

    return loss_fn


def make_seq_kd_loss(
    alpha: float = 0.25,
    temperature: float = 4.0,
    standardize: bool = True,
    pad_id: int = 0,
    kind: str = 'kl',
) -> Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
    """系列KD損失クロージャを構築する(make_kd_lossのシーケンス版)。

    返す loss(student_logits[B,S,V], target_ids[B,S], teacher_logits[B,S,V]) =
        alpha * CE(flatten, ignore_index=pad_id) + (1 - alpha) * KD(student, teacher)

    hard項(CE)は`target_ids != pad_id`をignore_indexで自動的に除外する
    (F.cross_entropyの標準機構、ラベル平滑化なし)。KD項は同じpad位置を
    `mask=target_ids != pad_id`としてkd_kl_loss_seqに渡し、pad位置の
    ロジットを平均から除外する(pad位置は本来無意味な次トークン予測のため)。

    standardize=True の場合、KD項の**両ロジット**にstandardize_logitsを
    適用してから比較する(standardize_logitsはlast dim正規化のため
    [B,S,V]でもそのまま[B,S,V]を保って動作する)。

    Args:
        alpha: hard(CE)項の重み。alpha=1で純CE、alpha=0で純KD。
        temperature: KD温度T(kind='kl'のとき使用)。
        standardize: KD項の両ロジットをZ-score標準化するか。
        pad_id: パディングトークンID(CE/KD双方でこのIDの位置を除外する)。
        kind: 'kl'(既定、kd_kl_loss_seq)または'mse'(kd_mse_loss、flatten適用)。

    Returns:
        loss(student_logits[B,S,V], target_ids[B,S], teacher_logits[B,S,V]) -> スカラー。

    参照論文: kd_kl_loss/make_kd_lossのdocstring/モジュールdocstring参照
        (Hinton et al. 2015、Sun et al. 2024 Logit Standardization arXiv:2403.01427)。
    """
    if kind not in ('kl', 'mse'):
        raise ValueError(f"kind は 'kl' か 'mse': {kind}")

    a = float(alpha)

    def loss_fn(
        student_logits: torch.Tensor,
        target_ids: torch.Tensor,
        teacher_logits: torch.Tensor,
    ) -> torch.Tensor:
        vocab = student_logits.shape[-1]
        ce = F.cross_entropy(
            student_logits.reshape(-1, vocab), target_ids.reshape(-1).long(),
            ignore_index=pad_id,
        )

        s = student_logits
        t = teacher_logits
        if standardize:
            s = standardize_logits(s)
            t = standardize_logits(t)

        mask = target_ids != pad_id
        if kind == 'kl':
            kd = kd_kl_loss_seq(s, t, temperature, mask=mask)
        else:
            mask_flat = mask.reshape(-1)
            kd = kd_mse_loss(
                s.reshape(-1, vocab)[mask_flat], t.reshape(-1, vocab)[mask_flat],
            )
        return a * ce + (1.0 - a) * kd

    return loss_fn


# ============================================================
# W3-seg: 密予測(per-pixel)KD損失
# ============================================================

def make_dense_kd_loss(
    alpha: float = 0.25,
    temperature: float = 2.0,
    standardize: bool = True,
    ignore_index: int = 255,
    pseudo_labels: bool = False,
    kind: str = 'kl',
    hard_loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
) -> Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
    """密予測(セグメンテーション)向けの per-pixel KD損失クロージャを構築する(W3-seg)。

    分類版 make_kd_loss([B,C]) を [B,C,H,W] ロジットへ拡張したもの。式・T^2補正・
    最適α=0.25・standardize必須の意味は画像分類版と**完全同一**で、各ピクセルを
    独立サンプルとして扱う。GroupSum終端 τ_gs と KD温度 T の二重温度対策として
    standardize=True(クラス次元 Z-score、per-pixel)を既定onにする(§2-2)。

    ★C=2(バイナリセグ)でのstandardize退化に注意:
        2要素のZ-scoreは (x1,x2) → (±1,∓1)(符号のみ)に潰れ、teacherロジット差の
        大きさ(確信度=dark knowledge)が完全に消える。KD項は温度に依らず
        「teacher argmaxの±1固定ターゲット」=ノイズ入りハードラベルと化し、
        境界画素のteacher誤りを高確信で注入してしまう(W3-seg w3r実測:
        standardize=onのKD 3腕がscratch比-6.5pt)。**C=2ではstandardize=Falseを
        推奨**(tests/l1_math/test_dense_kd_loss.py の退化実証テスト参照)。
        C が十分大きい多クラスでは分類版と同じく standardize=on が有効。

    返す loss(student_logits[B,C,H,W], labels[B,H,W], teacher_logits[B,C,H,W]) =
        alpha * CE + (1 - alpha) * KD

    hard項:
        - pseudo_labels=False(既定): labels をGTとして CE(ignore_index対応)。
        - pseudo_labels=True: labels を無視し teacher argmax(疑似ラベル)をGT化する
          (ラベルフリー変換、§2-4)。疑似ラベルは全ピクセルで定義されるため
          ignore マスクは適用しない。
        - hard_loss_fn が指定されると、上記で決定したターゲット(labels または
          teacher argmax)に対して `F.cross_entropy` の代わりに
          `hard_loss_fn(student_logits, target)` を呼ぶ(VOC 21クラス
          背景collapse対策の不均衡損失=focal/dice/weighted_ce差し替え口。
          experiments/seg_losses.py の build_seg_loss が返す関数を想定)。
          hard_loss_fn は自身のIGNORE_INDEXマスキングに責任を持つ(既存の
          F.cross_entropy(..., ignore_index=ignore_index)と同じ契約)。
          Noneなら従来通り F.cross_entropy を使う(後方互換)。

    KD項:
        クラス次元を最後に移し([B,H,W,C])、standardize後に per-pixel KL を取る
        (kind='mse' は中心化per-pixel MSE)。**ignore_index のピクセルは KD からも
        除外する**(pseudo_labels=True のときは全ピクセル有効)。

    Args:
        alpha: hard(CE)項の重み。alpha=1 で純CE、alpha=0 で純KD。
        temperature: KD温度 T(kind='kl' のとき使用。二重温度の T 成分)。
        standardize: KD項の両ロジットを(クラス次元で)Z-score標準化するか。
        ignore_index: CE/KD双方で除外するマスクラベル(pseudo_labels=False時のみ有効)。
        pseudo_labels: hard項のGTを teacher argmax に置換する(ラベルフリー)。
        kind: 'kl'(既定, per-pixel KL)または 'mse'(中心化per-pixel MSE)。
        hard_loss_fn: hard項の損失関数を差し替える。
            `hard_loss_fn(student_logits[B,C,H,W], target[B,H,W]) -> スカラー`。
            Noneで従来の F.cross_entropy(ignore_index対応)。

    Returns:
        loss(student_logits[B,C,H,W], labels[B,H,W], teacher_logits[B,C,H,W]) -> スカラー。

    参照論文: make_kd_loss/kd_kl_loss/モジュールdocstring(Hinton et al. 2015、
        Sun et al. 2024 Logit Standardization arXiv:2403.01427)。
    """
    if kind not in ('kl', 'mse'):
        raise ValueError(f"kind は 'kl' か 'mse': {kind}")

    a = float(alpha)

    def loss_fn(
        student_logits: torch.Tensor,
        labels: torch.Tensor,
        teacher_logits: torch.Tensor,
    ) -> torch.Tensor:
        # --- hard項ターゲットと有効マスクの決定 ---
        if pseudo_labels:
            with torch.no_grad():
                ce_target = teacher_logits.argmax(dim=1)  # [B,H,W]
            if hard_loss_fn is not None:
                ce = hard_loss_fn(student_logits, ce_target.long())
            else:
                ce = F.cross_entropy(student_logits, ce_target.long())
            valid = torch.ones_like(ce_target, dtype=torch.bool)  # 全ピクセル有効
        else:
            if hard_loss_fn is not None:
                ce = hard_loss_fn(student_logits, labels.long())
            else:
                ce = F.cross_entropy(
                    student_logits, labels.long(), ignore_index=ignore_index,
                )
            valid = labels != ignore_index  # [B,H,W]

        # --- KD項: クラス次元を最後に移して per-pixel 処理 ---
        s = student_logits.permute(0, 2, 3, 1)  # [B,H,W,C]
        t = teacher_logits.permute(0, 2, 3, 1)
        if standardize:
            s = standardize_logits(s)
            t = standardize_logits(t)

        if kind == 'kl':
            temp = float(temperature)
            log_p_s = F.log_softmax(s / temp, dim=-1)
            log_p_t = F.log_softmax(t / temp, dim=-1)
            kd_map = F.kl_div(
                log_p_s, log_p_t, reduction='none', log_target=True,
            ).sum(dim=-1) * (temp * temp)  # [B,H,W]
        else:
            # 中心化per-pixel MSE(シフト不変。s==t で0)。
            s_c = s - s.mean(dim=-1, keepdim=True)
            t_c = t - t.mean(dim=-1, keepdim=True)
            kd_map = ((s_c - t_c) ** 2).mean(dim=-1)  # [B,H,W]

        vf = valid.to(kd_map.dtype)
        n_valid = vf.sum().clamp(min=1)
        kd = (kd_map * vf).sum() / n_valid
        return a * ce + (1.0 - a) * kd

    return loss_fn


# ============================================================
# スケジュール(alpha/temperatureの動的化)
# ============================================================

_SCHEDULE_RE = re.compile(
    r'^(?P<kind>const|linear|cosine):(?P<rest>.+)$',
)


def resolve_schedule(spec: float | int | str) -> Callable[[int], float]:
    """スカラー or 文字列スケジュール仕様を `step -> float` の関数へ解決する。

    spec が float/int の場合はconst関数化(常にその値を返す)。

    文字列スペックの形式:
        'const:V'                    — 常にV
        'linear:START:END:DURATION'  — [0, DURATION]でSTART→END線形補間、
                                        DURATION以降はENDに固定
        'cosine:START:END:DURATION'  — [0, DURATION]でSTART→ENDをcosine補間
                                        (cosine annealing、DURATION以降はEND固定)

    DURATION<=0 は不正(ValueError)。step<0はstep=0として扱う(呼び出し側は
    通常0以上のみ渡す想定だが、防御的にクランプする)。

    Args:
        spec: float/int(定数)、または上記形式の文字列。

    Returns:
        Callable[[int], float]: step(int) -> スケジュール値。

    Raises:
        ValueError: 文字列形式が不正な場合(未知kind、パラメータ数不一致、
            数値変換失敗、duration<=0)。
    """
    if isinstance(spec, int | float) and not isinstance(spec, bool):
        value = float(spec)
        return lambda step: value

    if not isinstance(spec, str):
        raise ValueError(f'spec は float/int/str のいずれか: {type(spec)!r}')

    m = _SCHEDULE_RE.match(spec.strip())
    if m is None:
        raise ValueError(
            f"不正なスケジュール形式: {spec!r}。"
            " 'const:V' / 'linear:START:END:DURATION' / 'cosine:START:END:DURATION' のいずれか。"
        )

    kind = m.group('kind')
    parts = m.group('rest').split(':')

    if kind == 'const':
        if len(parts) != 1:
            raise ValueError(f"'const:V' はパラメータ1個: {spec!r}")
        try:
            value = float(parts[0])
        except ValueError as e:
            raise ValueError(f'const の値が数値でない: {spec!r}') from e
        return lambda step: value

    # linear / cosine は共に START:END:DURATION の3パラメータ
    if len(parts) != 3:
        raise ValueError(f"'{kind}:START:END:DURATION' はパラメータ3個: {spec!r}")
    try:
        start, end, duration = float(parts[0]), float(parts[1]), float(parts[2])
    except ValueError as e:
        raise ValueError(f'{kind} のパラメータが数値でない: {spec!r}') from e
    if duration <= 0:
        raise ValueError(f'duration は正: {spec!r}')

    if kind == 'linear':
        def _linear(step: int) -> float:
            s = max(0, step)
            if s >= duration:
                return end
            progress = s / duration
            return start + (end - start) * progress
        return _linear

    # cosine
    def _cosine(step: int) -> float:
        s = max(0, step)
        if s >= duration:
            return end
        progress = s / duration
        cosine_factor = 0.5 * (1.0 - math.cos(math.pi * progress))
        return start + (end - start) * cosine_factor
    return _cosine


# ============================================================
# DIST風相関損失(arXiv:2205.10536)
# ============================================================

def _pearson_corr_rows(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """[N, D]の行方向(dim=-1)Pearson相関を返す([N])。

    アフィン変換(a*x+b, a>0)に不変(平均を引いて標準偏差で割るため)。
    """
    a = a - a.mean(dim=-1, keepdim=True)
    b = b - b.mean(dim=-1, keepdim=True)
    numerator = (a * b).sum(dim=-1)
    denom = a.norm(dim=-1) * b.norm(dim=-1) + eps
    return numerator / denom


def dist_loss_seq(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    mask: torch.Tensor | None = None,
    beta: float = 1.0,
    gamma: float = 1.0,
    temperature: float = 1.0,
) -> torch.Tensor:
    """DIST風の相関蒸留損失(Huang et al. 2022, arXiv:2205.10536)のシーケンス版。

    KLのような分布の完全一致要求を、Pearson相関(=順位・形状の一致)に緩和する。
    Pearson相関は**アフィン不変**(corr(a*x+b, y) == corr(x, y)、a>0)なので、
    standardize_logits によるZ-score標準化は数学的に不要(相関を取る時点で
    平均・分散差は自動的に吸収される)。この点がKL系(kd_kl_loss_seq)と異なる
    本損失の核心的な設計思想である。

    手順:
        1. mask有効位置をflattenして[N, V]化(N=有効位置数、V=vocab)。
        2. 両ロジットをtemperatureで割ってからsoftmax(確率化)。
        3. inter-class: サンプル(行)ごとにteacher/student確率ベクトルの
           Pearson相関を取り、`1 - mean(corr)` をbeta倍。
        4. intra-class: クラス(列)ごとに全サンプルにわたるteacher/student
           確率のPearson相関を取り、`1 - mean(corr)` をgamma倍。
        5. 上記の和を返す。

    Pearson相関はその直接の入力(本実装ではsoftmax確率ベクトル)に対して
    アフィン変換 `a*x+b`(a>0)不変(平均を引いて標準偏差で割るため)なので、
    standardize_logits によるZ-score標準化を確率側に重ねて適用する必要はない
    (相関を取る時点でスケール/オフセット差は自動的に吸収される)。
    ただし本実装はDIST公式実装(github.com/hunto/DIST_KD)に合わせて
    **softmax後**の確率に相関を取るため、この不変性が及ぶのは
    「softmax出力(確率)へのアフィン変換」であり、「softmax前のロジットへの
    任意のアフィン変換」ではない点に注意(logitへの加法シフトはsoftmaxの
    シフト不変性でそのまま吸収されるが、logitへの乗算スケールは実効温度を
    変えるため一般には不変でない)。

    Args:
        student_logits: [B, S, V] studentロジット。
        teacher_logits: [B, S, V] teacherロジット。
        mask: [B, S] のbool/float。Trueまたは非0の位置のみ対象にする。
            Noneなら全位置を対象にする。
        beta: inter-class相関損失の重み。
        gamma: intra-class相関損失の重み。
        temperature: softmax前の温度(KD温度T)。
        eps: (内部)相関のゼロ割防止。

    Returns:
        スカラー。student==teacherのとき0近傍(相関1に漸近)。

    Raises:
        ValueError: maskが全てFalse(有効位置が1つもない)の場合
            (kd_kl_loss_seqと同じ規約)。
    """
    b, s, v = student_logits.shape
    student_flat = student_logits.reshape(b * s, v)
    teacher_flat = teacher_logits.reshape(b * s, v)

    if mask is None:
        valid = torch.ones(b * s, dtype=torch.bool, device=student_logits.device)
    else:
        valid = mask.reshape(-1).to(dtype=torch.bool, device=student_logits.device)

    n_valid = int(valid.sum().item())
    if n_valid == 0:
        raise ValueError('mask が全て無効(True/非0の位置なし)です。KD対象位置がありません。')

    student_valid = student_flat[valid]  # [N, V]
    teacher_valid = teacher_flat[valid]  # [N, V]

    t = float(temperature)
    p_student = F.softmax(student_valid / t, dim=-1)  # [N, V]
    p_teacher = F.softmax(teacher_valid / t, dim=-1)  # [N, V]

    # inter-class: サンプル(行)方向の相関。
    inter_corr = _pearson_corr_rows(p_student, p_teacher)  # [N]
    inter_loss = 1.0 - inter_corr.mean()

    # intra-class: クラス(列)方向の相関。転置して行方向相関の関数を再利用する。
    intra_corr = _pearson_corr_rows(p_student.t(), p_teacher.t())  # [V]
    intra_loss = 1.0 - intra_corr.mean()

    return beta * inter_loss + gamma * intra_loss


# ============================================================
# KD損失v2(スケジュール対応・confidence-alpha・hard-agree補助損失)
# ============================================================

def make_seq_kd_loss_v2(
    alpha: float | str = 0.25,
    temperature: float | str = 4.0,
    standardize: bool = True,
    pad_id: int = 0,
    kind: str = 'kl',
    alpha_mode: str = 'fixed',
    hard_agree_weight: float = 0.0,
    dist_beta: float = 1.0,
    dist_gamma: float = 1.0,
) -> Callable[[torch.Tensor, torch.Tensor, torch.Tensor, int], torch.Tensor]:
    """系列KD損失v2クロージャを構築する(make_seq_kd_lossの拡張版)。

    既存 make_seq_kd_loss は無改造(後方互換専用に維持)。本関数は以下を追加する:

    1. alpha/temperature をスケジュール対応にする(resolve_schedule経由。
       float/intを渡せば従来通り定数、文字列 'linear:...'/'cosine:...' で
       ステップ依存にできる)。
    2. alpha_mode='confidence': 位置ごとにteacherの確信度
       `conf = max_c softmax(teacher_logits/T)[c]` を計算し、
       `alpha_pos = alpha_base + (1 - alpha_base) * (1 - conf)` で
       位置別のCE/KD混合比を決める(teacherが自信を持てない位置ほど
       alpha_posが大きくなりCE=hardラベル寄りになる)。
    3. hard_agree_weight>0: 補助損失として
       `CE(student_logits, teacher_logits.argmax(-1))` をmask付きで加算する
       (teacher argmaxとの一致を直接促す)。
    4. kind='dist': dist_loss_seq(Pearson相関損失、DIST arXiv:2205.10536)を
       KD項に使う。standardizeはPearson相関がアフィン不変のため'dist'には
       効果が無い(適用してもしなくても結果は同じ、docstring上の注意のみ)。

    返す loss_fn のシグネチャ:
        loss_fn(student_logits[B,S,V], target_ids[B,S], teacher_logits[B,S,V],
                step=0) -> スカラー

    Args:
        alpha: hard(CE)項の基準重み。float/int または schedule spec文字列。
        temperature: KD温度T。float/int または schedule spec文字列。
        standardize: KD項(kind='kl')の両ロジットをZ-score標準化するか。
            kind='dist'では数学的に無効(Pearson相関がアフィン不変のため)。
        pad_id: パディングトークンID。
        kind: 'kl'(既定、kd_kl_loss_seq)または'dist'(dist_loss_seq)。
        alpha_mode: 'fixed'(既定、alphaをそのまま使う)または
            'confidence'(teacher確信度に応じて位置別に調整)。
        hard_agree_weight: teacher argmaxとの一致を促す補助CE損失の重み
            (既定0.0=無効)。
        dist_beta: kind='dist'時のinter-class相関重み(dist_loss_seq参照)。
        dist_gamma: kind='dist'時のintra-class相関重み(dist_loss_seq参照)。

    Returns:
        loss_fn(student_logits, target_ids, teacher_logits, step=0) -> スカラー。

    参照論文: モジュールdocstring、DIST(arXiv:2205.10536)。
    """
    if kind not in ('kl', 'dist'):
        raise ValueError(f"kind は 'kl' か 'dist': {kind}")
    if alpha_mode not in ('fixed', 'confidence'):
        raise ValueError(f"alpha_mode は 'fixed' か 'confidence': {alpha_mode}")

    alpha_fn = resolve_schedule(alpha)
    temperature_fn = resolve_schedule(temperature)

    def loss_fn(
        student_logits: torch.Tensor,
        target_ids: torch.Tensor,
        teacher_logits: torch.Tensor,
        step: int = 0,
    ) -> torch.Tensor:
        vocab = student_logits.shape[-1]
        alpha_base = alpha_fn(step)
        t = temperature_fn(step)

        mask = target_ids != pad_id
        mask_flat = mask.reshape(-1)
        if not bool(mask_flat.any()):
            raise ValueError('mask が全て無効(pad_id以外の位置なし)です。KD対象位置がありません。')

        # --- hard(CE)項: 位置ごとのCEを保持(confidence-alpha合成に使うため) ---
        ce_per_pos = F.cross_entropy(
            student_logits.reshape(-1, vocab), target_ids.reshape(-1).long(),
            ignore_index=pad_id, reduction='none',
        ).reshape(target_ids.shape)  # [B, S]

        # --- KD項の入力ロジット(standardizeはkind='kl'のみ意味を持つ) ---
        s = student_logits
        te = teacher_logits
        if standardize and kind == 'kl':
            s = standardize_logits(s)
            te = standardize_logits(te)

        if alpha_mode == 'fixed':
            # 従来どおりスカラーalphaでCE/KDを合成する。
            ce = (ce_per_pos.reshape(-1) * mask_flat).sum() / mask_flat.sum().clamp(min=1)
            if kind == 'kl':
                kd = kd_kl_loss_seq(s, te, t, mask=mask)
            else:
                kd = dist_loss_seq(
                    s, te, mask=mask, beta=dist_beta, gamma=dist_gamma, temperature=t,
                )
            main_loss = alpha_base * ce + (1.0 - alpha_base) * kd
        else:
            # confidence: 位置ごとのalpha場を作り、CE項/KD項ともに位置別に重み付ける。
            with torch.no_grad():
                teacher_probs = F.softmax(teacher_logits / t, dim=-1)
                conf = teacher_probs.max(dim=-1).values  # [B, S]
            alpha_pos = alpha_base + (1.0 - alpha_base) * (1.0 - conf)  # [B, S]

            if kind == 'kl':
                t_scaled = float(t)
                log_p_student = F.log_softmax(s / t_scaled, dim=-1)
                log_p_teacher = F.log_softmax(te / t_scaled, dim=-1)
                kd_per_pos = F.kl_div(
                    log_p_student, log_p_teacher, reduction='none', log_target=True,
                ).sum(dim=-1) * (t_scaled * t_scaled)  # [B, S]
            else:
                # dist_loss_seqは位置別出力を返さないため、位置ごとのペア相関を
                # 直接計算する(inter-classのみ; intra-classはバッチ全体でしか
                # 定義できずconfidence重み付けと相性が悪いため、confidence-alpha
                # モードではinter-classのみをdist_gamma無しで使う設計とする)。
                t_scaled = float(t)
                p_student = F.softmax(s / t_scaled, dim=-1)
                p_teacher = F.softmax(te / t_scaled, dim=-1)
                b, seqlen, _v = p_student.shape
                corr = _pearson_corr_rows(
                    p_student.reshape(b * seqlen, _v), p_teacher.reshape(b * seqlen, _v),
                ).reshape(b, seqlen)
                # [B, S] (intra-classはスカラー項として別途加算)
                kd_per_pos = dist_beta * (1.0 - corr)

            mask_f = mask.to(dtype=ce_per_pos.dtype)
            n_valid = mask_f.sum().clamp(min=1)
            weighted = (alpha_pos * ce_per_pos + (1.0 - alpha_pos) * kd_per_pos) * mask_f
            main_loss = weighted.sum() / n_valid

            if kind == 'dist' and dist_gamma != 0.0:
                # intra-class相関はサンプル横断でしか定義できないため、confidence
                # モードでもスカラー項として(全体一律で)追加する。
                intra_corr = _pearson_corr_rows(
                    p_student.reshape(b * seqlen, _v)[mask_flat].t(),
                    p_teacher.reshape(b * seqlen, _v)[mask_flat].t(),
                )
                main_loss = main_loss + dist_gamma * (1.0 - intra_corr.mean())

        total = main_loss

        if hard_agree_weight > 0.0:
            with torch.no_grad():
                teacher_argmax = teacher_logits.argmax(dim=-1)  # [B, S]
                # pad位置はteacher_argmaxが無意味なので、ignore_index相当に
                # pad_idへ置き換えて自動除外する(target_ids同様の規約)。
                teacher_argmax = torch.where(
                    mask, teacher_argmax, torch.full_like(teacher_argmax, pad_id),
                )
            hard_agree_ce = F.cross_entropy(
                student_logits.reshape(-1, vocab), teacher_argmax.reshape(-1),
                ignore_index=pad_id,
            )
            total = total + hard_agree_weight * hard_agree_ce

        return total

    return loss_fn


def pack_targets(labels: torch.Tensor, teacher_logits: torch.Tensor) -> torch.Tensor:
    """[B]int + [B,C]float → [B, 1+C]float の packed target を作る。

    列0 = ラベル(float格納)、列1.. = teacherロジット。teacherロジットの dtype に
    合わせて返す(損失側でラベルを long 化して使う)。

    Args:
        labels: [B] 整数ラベル。
        teacher_logits: [B, C] teacherロジット。

    Returns:
        [B, 1+C] float テンソル。
    """
    if labels.ndim != 1:
        raise ValueError(f'labels は [B] の1次元: {tuple(labels.shape)}')
    if teacher_logits.ndim != 2 or teacher_logits.shape[0] != labels.shape[0]:
        raise ValueError(
            f'teacher_logits は [B, C] かつ B一致: '
            f'{tuple(teacher_logits.shape)} vs labels {tuple(labels.shape)}'
        )
    label_col = labels.to(teacher_logits.dtype).unsqueeze(-1)  # [B, 1]
    return torch.cat([label_col, teacher_logits], dim=-1)


def make_distill_task(
    num_classes: int,
    *,
    alpha: float = 0.5,
    temperature: float = 4.0,
    standardize: bool = False,
    kind: str = 'kl',
) -> TaskSpec:
    """packed-target 蒸留タスクの TaskSpec を構築する。

    training/tasks.py の TaskSpec を直接構築する(make_task を経由しないため
    tasks.py の LOSS_NAMES 変更は不要)。kind(TaskSpec.kind)は 'classification'、
    loss_name は 'kd'。loss_fn/metric_fn は packed target([B, 1+C])をスライスする:

        列0     -> ラベル labels[B]
        列1..C  -> teacherロジット[B, C]

    - loss_fn(output, target) は make_kd_loss(...) をスライス後に呼ぶ。
    - metric_fn(output, target) はラベルに対する分類精度
      (tasks._classification_acc と同値)。

    Args:
        num_classes: クラス数 C(teacherロジット幅の検証に使用)。
        alpha: CE項の重み(make_kd_loss参照)。
        temperature: KD温度 T(二重温度の T 成分。τ_gs はモデル側 GroupSum.tau)。
        standardize: KD項の両ロジットを標準化するか。
        kind: 'kl' または 'mse'。

    Returns:
        TaskSpec(kind='classification', loss_name='kd')。
    """
    kd = make_kd_loss(
        alpha=alpha, temperature=temperature,
        standardize=standardize, kind=kind,
    )

    def _split(target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        labels = target[..., 0].long()
        teacher_logits = target[..., 1:1 + num_classes]
        return labels, teacher_logits

    def loss_fn(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        labels, teacher_logits = _split(target)
        return kd(output, labels, teacher_logits)

    def metric_fn(output: torch.Tensor, target: torch.Tensor) -> float:
        labels = target[..., 0].long()
        return _classification_acc(output, labels)

    return TaskSpec(
        kind='classification',
        loss_name='kd',
        loss_fn=loss_fn,
        metric_fn=metric_fn,
    )


# ============================================================
# exit-head蒸留(OFA-KD式ロジット空間投影)
# ============================================================

class LogitExitHead(nn.Module):
    """中間活性 → vocabロジットへの線形投影(OFA-KD式ロジット空間投影)。

    クロスアーキKDの事前調査の結論: teacher hidden
    statesを直接模倣するfeature-matchingはDLGN(離散論理ゲート)には効果が
    薄く(層別二値蒸留がスクラッチ以下)、student中間特徴をteacherと
    同じ「ロジット空間」に投影してからKLで比較する方式(OFA-KD, 2310.19444の
    exit-head相当)で足りる。

    **訓練時のみのfloat足場である**: この nn.Linear は通常の連続値線形層で
    あり、論理ゲート化(LUT変換)を一切要求しない。推論時・変換時には
    このモジュールごと破棄され、最終出力(GroupSum経由のロジット)にも
    一切寄与しない。あくまで訓練中に中間層へ勾配を注入するための補助枝。
    """

    def __init__(self, hidden_dim: int, vocab_size: int):
        """
        Args:
            hidden_dim: 中間活性の特徴次元(最終軸のサイズ)。
            vocab_size: 投影先のロジット次元(teacherロジット幅と一致させる)。
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.proj = nn.Linear(hidden_dim, vocab_size)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden: [..., hidden_dim] 中間活性(例: [B, hidden_dim] または
                [B, S, hidden_dim])。

        Returns:
            [..., vocab_size] ロジット(teacherロジットと同じ形状規約で
            kd_kl_loss_seq等にそのまま渡せる)。
        """
        return self.proj(hidden)


def make_exit_kd_loss(
    exit_heads: list[nn.Module],
    weights: list[float],
    temperature: float = 1.0,
    pad_id: int | None = None,
) -> Callable[
    [list[torch.Tensor], torch.Tensor, torch.Tensor | None], torch.Tensor,
]:
    """複数exit headをteacherロジットへ多対一蒸留するヘルパ。

    各exit headの出力ロジット(shapeはteacher_logitsに揃うようbroadcastまたは
    投影済みである前提)を、同一のteacher最終ロジットに対して個別に
    kd_kl_loss_seqでKD損失を取り、`Σ_l w_l · KL_l` を返す(多対一マッチング:
    複数の中間層がいずれも同じteacher最終出力を模倣対象にする、OFA-KDの
    exit-head設計と同じ考え方)。

    マスク規約は既存(kd_kl_loss_seq/make_seq_kd_loss)と同一:
    `mask`(または`target_ids != pad_id`から導出したmask)がTrue/非0の位置のみ
    平均に含める。

    Args:
        exit_heads: 各中間活性に対応する LogitExitHead(または同じ
            forward(hidden)->logits契約を満たす任意のnn.Module)のリスト。
            長さは hidden_states のリストと一致させる(呼び出し側の責務)。
        weights: 各exit headのKD損失重み w_l。exit_heads と同じ長さ。
        temperature: KD温度T(kd_kl_loss_seqにそのまま渡す。二重温度の
            T成分。モデル側τ_gsとは無関係、distill.pyモジュールdocstring参照)。
        pad_id: mask導出用のpad ID。maskを直接渡す場合はNoneのままでよい
            (返す損失関数がmask引数を直接受け取れるため)。

    Returns:
        loss_fn(hidden_states, teacher_logits, mask=None) -> スカラー
            hidden_states: exit_heads と同じ長さの中間活性リスト
                (各要素は対応するexit_headのforward入力形状と一致する必要がある)。
            teacher_logits: [B, S, V](または[B, V])teacher最終ロジット
                (exit_headの出力と同じ形状に揃っている前提)。
            mask: [B, S]のbool/float。Noneかつpad_idが与えられていれば
                呼び出し側でtarget_idsからmaskを作る想定(本関数はmaskを
                そのままkd_kl_loss_seqへ渡すのみ)。

    Raises:
        ValueError: exit_heads と weights の長さが一致しない場合。
    """
    if len(exit_heads) != len(weights):
        raise ValueError(
            f'exit_heads({len(exit_heads)})と weights({len(weights)})の'
            '長さが一致しません。'
        )

    t = float(temperature)

    def loss_fn(
        hidden_states: list[torch.Tensor],
        teacher_logits: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if len(hidden_states) != len(exit_heads):
            raise ValueError(
                f'hidden_states({len(hidden_states)})と'
                f' exit_heads({len(exit_heads)})の長さが一致しません。'
            )

        total = None
        for head, weight, hidden in zip(exit_heads, weights, hidden_states, strict=True):
            exit_logits = head(hidden)
            kd = kd_kl_loss_seq(exit_logits, teacher_logits, temperature=t, mask=mask)
            term = weight * kd
            total = term if total is None else total + term
        assert total is not None  # exit_heads/weightsが空でない限り必ずセットされる
        return total

    return loss_fn
