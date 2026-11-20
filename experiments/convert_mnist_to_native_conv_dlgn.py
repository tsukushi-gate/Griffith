"""コンパクトnative Conv-DLGN獲得 — CNN構造転写(wiring+gate-init)で
MNIST HARD精度がscratch/IO-KDを超えるか(算術の Conv版)。

`convert_arith_to_native_dlgn.py`(算術、arm A/B/E/E+KD)と同じ問いを
画像convへ拡張する: **コンパクトなnative Conv-DLGN**
(difflogic方式、`LutConv2d(n_inputs=2, parameterization='softmax16g')`、
少チャンネル数、ゲート選択を勾配で自由学習)を、MNIST分類タスクで
「CNN構造転写(wiring+gate-init)→soft磨き」経由でscratchより高いhard精度に
到達できるか。

背景: `convert_transcribe_mnist_conv.py`(Conv PoC)は同じ問いを
**warp(非native、真理値表数固定で自由な任意関数を表現できる大きい回路)**
パラメータ化で検証し、構造転写+soft磨き(arm C)がscratch/IO-KDを上回ることを
示した。本スクリプトはそれを**native softmax16g**(16ゲート自由選択、
warpより小さい回路)に置き換えて同じ問いを再検証する。

## 追補: 交絡の除去(fair init) ★本改修

初回計測(fair init前)ではarm Eがscratch(A)を上回れなかったが、比較には
**交絡**があった。`restore_lut_conv2d`(ビット厳密転写)は真理値表を
DEFAULT_LOGIT_M=10.0の飽和ロジット(one-hot)で書き込むため、(1) softmaxが
ほぼ完全に離散化し勾配がほぼ消える、(2) `LutConv2d._residual_init`が
仕込む「ゲートA(パススルー)への事前バイアス」を完全に上書きする
(深いtree convで活性が段ごとに87%減衰する問題への対策が消える)。
つまりarm Eは磨き開始時点でscratch(Aは`residual_init`の恩恵付きで
ランダム初期化)より不利な出発点(soft acc実測0.10-0.16=chance)から
磨いていた。scratchが持つ利点を転写armだけ剥奪した状態での比較は
「構造転写が効くか」の答えにならない。

修正: `griffith_logic/ir/weights.py`に`weights_from_truth_tables_fair_init`
(+ `restore_lut_conv2d_fair_init`)を**追加**(既存の`weights_from_truth_tables`
/`conv_weights_from_truth_tables`/`restore_lut_conv2d`は無改造、既存の
bit-exact roundtripテストに影響なし)。転写ゲートのロジットに中庸な正値
`--fair-transcribed-logit`(既定3.0、飽和logit_m=10.0よりずっと小さい)を
加算し、ゲートA(パススルー、真理値表[0,1,0,1])のロジットへ
`--fair-residual-bias`(既定5.0、`_residual_init`のz3既定と同じ)を**加算**する。
結果、初期状態は「転写関数へ強くバイアスされているが飽和しておらず
(勾配が流れる)、かつ残差パススルー事前分布も残る」softmax混合になる。
意図的にbit-exactではない(`assert_conv_rebuild_bit_exact`は純粋rebuild経路
(`rebuild_conv_as_native16g`+`restore_lut_conv2d`)専用に残し、fair init経路
には適用しない)。

## 追補: arm E_wire(配線のみ転写、ゲートはrandom/residual-init)

構造転写のリフトが「配線(wiring)の事前分布」由来か「関数保存(gate-init)」
由来かを切り分けるアブレーション。転写された`indices_flat`(受容野内の
どのリーフを見るか)だけをコピーし、ゲート真理値表は転写せず
scratch同様のランダム初期化(+residual_init)のままにする。
RNN PoCと同型の分解(wiring-prior vs function-preservation)を画像convへ適用。

## native softmax16g conv rebuild(達成した経路)

`transcribe_conv2d`(convert/transcribe.py)は多数決構築のため
`train_info.parameterization='independent'`固定で真理値表を返す
(任意真理値表になり得るため)。native softmax16gで使うには、
conv2d IRNodeの`train_info.parameterization`を`'independent'`から
`'softmax16g'`へ書き換えてから`build_torch`に通す(`ir/build.py::_build_conv2d`が
train_infoからparameterizationを読む)。

この経路が機能するのは、LutConv2dの`n_inputs=2`のとき真理値表が
必ず4bit(16通り)であり、softmax16gが**全16通りの2入力Boolean関数を
表現できる**ため(`core/basis.py::_build_gate_table_n2`の全単射)。
つまりindependent→softmax16gの書き換えは**常に無損失**(bit-exact)である
(算術のrebuild_as_native16gと同じ理屈をconvへ適用したもの)。

ただし`ir/weights.py::conv_weights_from_truth_tables`は従来
'softmax'/'independent'/'iwp'のみ対応でsoftmax16gが未対応だった
(欠落はrestore_lut_conv2dの内部依存として発覚)。本タスクで
**追加**した(`weights_from_truth_tables`のis_16gate分岐を再利用する1行、
既存パラメータ化の挙動は変更なし。回帰テスト
`tests/test_ir_build.py::TestConvWeightsRoundtrip`で担保)。

## アーム(すべてnative Conv-DLGN、hard acc = hard_acc_via_export)

- **A(scratch)**: `build_conv_dlgn`(softmax16g既定、residual_init込み)を
  ラベルCEで学習。下限対照(`convert_transcribe_mnist_conv.run_arm_a`を再利用)。
- **B(IO-KDのみ)**: 同じnative archをteacher soft-logit KDで学習
  (`convert_transcribe_mnist_conv.run_arm_b`を再利用)。
- **E(本命: wiring+gate-init→soft磨き、fair init)**: `SmallTeacherCNN`
  (STE版、`TeacherConvSTE`)の第1conv層を`transcribe_conv2d(n_inputs=2)`で
  近似構造転写→native softmax16gへ**fair init**で復元
  (`restore_lut_conv2d_fair_init`、非飽和+残差事前分布を保持)→
  soft磨き(低LR CE fine-tune、Conv PoCの知見を継承しつつ
  polish-epochs/lrを較正)。
- **E_wire(アブレーション: wiring転写のみ)**: 転写`indices_flat`のみコピーし、
  ゲート真理値表は転写せずscratch同様のランダム初期化+residual_initのまま
  にする。E − E_wire がgate-init(関数保存)由来のリフト、
  E_wire − A がwiring事前分布単独のリフトを示す。
- **E+KD**: fair init(E)の初期化からKD磨き(CE+KL混合)。

## 報告指標

native Conv-DLGN HARD acc(A/B/E/E_wire/E+KD)、lift(E-A)、lift(E_wire-A)、
lift(E-E_wire)、lift(E-B)、lift(E+KD-E)、**Eのfair init直後soft acc**
(chanceでないことの確認)、ゲート数(native回路)、転写近似agreement
(較正データ上)。絶対精度はPoC規模(小学習件数・少epoch)で頭打ちである点は
正直に明記する。fair init(E/E+KD)はbit-exactではない(意図的、上記参照)。
純粋rebuild経路の厳密性assert(`assert_conv_rebuild_bit_exact`)は
E_wireの配線コピーの妥当性確認にのみ引き続き使う。

使用例:
    # 煙実行(数十秒、wandb無効、1 seed・小データ・少epoch)
    uv run python experiments/convert_mnist_to_native_conv_dlgn.py --quick --no-wandb

    # 本実行
    uv run python experiments/convert_mnist_to_native_conv_dlgn.py --seeds 3 --no-wandb
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone

JST = timezone(timedelta(hours=9))

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
# convert_transcribe_mnist_conv.py(Conv PoC、並行タスクの対象外ファイル)を
# 無改造でimportして再利用する(taskの明示指示: 対象ファイルを改変せずimport/copy)
from convert_transcribe_mnist_conv import (
    IN_SHAPE,
    N_CLASSES,
    _auto_device,
    _flatten_binary_images,
    build_conv_dlgn,
    hard_acc_via_export,
    make_teacher_logits,
    run_arm_a,
    run_arm_b,
    soft_acc,
    train_teacher,
)
from env_utils import load_dotenv
from wandb_utils import finish as wandb_finish
from wandb_utils import init_wandb, log_metrics

from griffith_logic.conv import LutConv2d
from griffith_logic.convert.distill import make_kd_loss
from griffith_logic.convert.teacher_models import HeavisideSTE
from griffith_logic.convert.transcribe import transcribe_conv2d
from griffith_logic.datasets.vision import load_mnist
from griffith_logic.export import export_ir
from griffith_logic.export.estimate import estimate_resources
from griffith_logic.ir.schema import IRGraph, IRNode
from griffith_logic.ir.simulate import simulate
from griffith_logic.ir.validate import IRValidationError, validate

load_dotenv()

NATIVE_N_INPUTS = 2  # compact native Conv-DLGNはdifflogic方式、N=2専用


# ============================================================
# STE teacher(隠れ活性を二値化してtranscribe誤差を切り分ける、同じ発想)
# ============================================================

class TeacherConvSTE(nn.Module):
    """Conv(out_ch)-FCの最小teacher。第1conv出力をHeavisideSTEで二値化する。

    `SmallTeacherCNN`(teacher_models.py)は2conv+BatchNorm+MaxPoolの標準構成
    だが、本タスクは「第1conv層のみをtranscribe_conv2dで転写」するPoC
    (convert_transcribe_mnist_conv.pyと同じ設計)なので、第2conv以降を
    持たない最小構成にしてteacher学習を軽量化する(--quickで数秒に収める)。
    BatchNormを避ける(transcribe_conv2dはnn.Conv2d単体を対象とし、後続の
    BatchNormを畳み込めない前提のため、teacher自体をBatchNorm無しにして
    conv1の重み・biasだけで閾値関数が完結するようにする)。

    Args:
        num_classes: 出力クラス数
        in_channels: 入力チャンネル数
        conv_out_ch: 第1(唯一の)conv出力チャンネル数
        hidden: 分類head隠れ幅
    """

    def __init__(self, num_classes: int = 10, in_channels: int = 1,
                conv_out_ch: int = 16, hidden: int = 64) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, conv_out_ch, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)  # 28x28 -> 14x14
        flat_dim = conv_out_ch * 14 * 14
        self.fc1 = nn.Linear(flat_dim, hidden)
        self.fc2 = nn.Linear(hidden, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = HeavisideSTE.apply(self.conv1(x))
        h = self.pool(h)
        h = h.flatten(1)
        h = F.relu(self.fc1(h))
        return self.fc2(h)


# ============================================================
# native softmax16g rebuild(算術のrebuild_as_native16gと同じ理屈をconvへ適用)
# ============================================================

def rebuild_conv_as_native16g(node: IRNode) -> IRNode:
    """conv2d IRNode(train_info.parameterization='independent')を
    softmax16gへ書き換えたコピーを返す(training-free)。

    n_inputs=2のconv内部ゲートは必ず4bit真理値表(16通り)であり、
    softmax16gは全16通りの2入力Boolean関数を表現できる(全単射、
    core/basis.py::_build_gate_table_n2)。よってこの書き換えは
    常に無損失(bit-exact)である。

    Args:
        node: transcribe_conv2dが返したconv2d IRNode(n_inputs=2前提)

    Returns:
        train_info.parameterization='softmax16g'の同一params(deepcopy)のnode

    Raises:
        ValueError: node.params['n_inputs'] != 2
    """
    n_in = int(node.params['n_inputs'])
    if n_in != NATIVE_N_INPUTS:
        raise ValueError(
            f'native16g復元はn_inputs={NATIVE_N_INPUTS}専用: n_inputs={n_in}'
        )
    node2 = copy.deepcopy(node)
    node2.train_info = {
        **node2.train_info,
        'parameterization': 'softmax16g',
        'initialized': True,
    }
    return node2


def assert_conv_rebuild_bit_exact(
    node_independent: IRNode, node_softmax16g: IRNode, x_np: np.ndarray,
) -> None:
    """独立表現convと復元後softmax16g convのIR simulate出力がargmax一致することを確認する。

   算術のassert_native16g_bit_exactと同じ役割: 理論上bit-exactの
    はずの変換が実際に壊れていないことを、閾値を緩めず例外で停止するassertで
    担保する。ここではフルモデルを組む前段のconv単体IR(1ノード)同士を比較する
    ため、'independent'/'softmax16g'それぞれを最小限のグラフに包んでsimulateする。
    """
    def _wrap(node: IRNode) -> IRGraph:
        from griffith_logic.ir.schema import IREdge, IRPort
        width = int(np.prod(node.params['in_shape']))
        out_ch = int(node.params['out_channels'])
        h, w = node.params['in_shape'][1], node.params['in_shape'][2]
        out_w = out_ch * h * w  # stride=1, padding=同一のため空間サイズ不変
        return IRGraph(
            metadata={'name': 'conv_probe'},
            inputs=[IRPort(id='x', width=width)],
            outputs=[IRPort(id='y', width=out_w)],
            nodes=[node],
            edges=[IREdge(src='x', dst=node.id), IREdge(src=node.id, dst='y')],
        )

    x_flat = x_np.reshape(x_np.shape[0], -1).astype(np.int64)
    y_indep = simulate(_wrap(node_independent), x_flat)
    y_16g = simulate(_wrap(node_softmax16g), x_flat)
    n_mismatch = int((y_indep != y_16g).sum())
    if n_mismatch != 0:
        raise RuntimeError(
            f'厳密性assert違反: independent→softmax16g復元後のconv出力が'
            f'{n_mismatch}要素不一致。理論上N=2真理値表→16ゲートは無損失の'
            'はずなのでバグ(閾値は緩めない)。'
        )


def native_conv_gate_count(model: nn.Sequential) -> int:
    """native Conv-DLGNモデル(LutConv2d+LutLayer)の総ゲート数。"""
    ir = export_ir(model, input_shape=IN_SHAPE, name='gate_count_probe')
    est = estimate_resources(ir)
    return int(est['total_luts'])


# ============================================================
# arm E / E+KD: CNN構造転写(wiring+gate-init native16g) → soft磨き(→KD磨き)
# ============================================================

def build_native_model_from_conv_node(
    conv_node: IRNode, *, head_width: int, device: str, seed: int,
    fair_init: bool = False,
    fair_transcribed_logit: float = 3.0, fair_residual_bias: float = 5.0,
) -> nn.Sequential:
    """転写(rebuild後)conv2d IRNodeから、OrPool→Flatten→LutLayer→GroupSumを
    ランダム初期化で継ぎ足したフルnative Conv-DLGNを構築する。

    build_conv_dlgn(scratch, softmax16g既定、residual_init込み)と同型の
    nn.Sequentialを作り、先頭のLutConv2dだけを転写node由来の重み・接続で
    上書きする(OrPool以降は転写対象外・scratch初期化のまま)。build_torch経由
    (NCHWAdapter)は使わない(build_conv_dlgnの生LutConv2dは4D NCHW入力を
    直接受け、IRのflatten規約(NCHWAdapter前提)とは入出力形状の扱いが異なる
    ため、直接restore_lut_conv2d(_fair_init)で書き戻すほうが両者の入出力
    契約に忠実)。

    Args:
        fair_init: Trueなら`restore_lut_conv2d_fair_init`
            (非飽和+残差事前分布保持、bit-exactではない)を使う。Falseなら
            従来の`restore_lut_conv2d`(飽和one-hot、bit-exact、
            E_wireの厳密性assert用途に残す)。
    """
    from griffith_logic.ir.weights import (
        restore_lut_conv2d,
        restore_lut_conv2d_fair_init,
    )

    torch.manual_seed(seed)
    out_ch = int(conv_node.params['out_channels'])
    tree_depth = int(conv_node.params['tree_depth'])
    model = build_conv_dlgn(
        out_ch=out_ch, tree_depth=tree_depth, n_inputs=NATIVE_N_INPUTS,
        head_width=head_width, device=device, seed=seed,
        conv_parameterization='softmax16g',
    )
    conv_module = model[0]
    assert isinstance(conv_module, LutConv2d)
    if fair_init:
        restore_lut_conv2d_fair_init(
            conv_module, conv_node.params['indices_flat'],
            conv_node.params['truth_tables'],
            transcribed_logit=fair_transcribed_logit,
            residual_bias=fair_residual_bias,
        )
    else:
        restore_lut_conv2d(
            conv_module, conv_node.params['indices_flat'],
            conv_node.params['truth_tables'],
        )
    return model


def build_native_model_wire_only(
    conv_node: IRNode, *, head_width: int, device: str, seed: int,
) -> nn.Sequential:
    """arm E_wire: 転写`indices_flat`(配線)のみコピーし、ゲート真理値表は
    転写せずscratch同様のランダム初期化+residual_initのままにする。

    build_conv_dlgnがLutConv2d構築時に既定でresidual_init(ゲートAへの
    事前バイアス)を適用する(`LutConv2d.__init__`のresidual_init=True既定)。
    ここでは重みには一切触れず、`indices_flat`バッファだけを転写node由来の
    配線で上書きする(wiring事前分布のみを転写、関数保存なし)。
    """
    torch.manual_seed(seed)
    out_ch = int(conv_node.params['out_channels'])
    tree_depth = int(conv_node.params['tree_depth'])
    model = build_conv_dlgn(
        out_ch=out_ch, tree_depth=tree_depth, n_inputs=NATIVE_N_INPUTS,
        head_width=head_width, device=device, seed=seed,
        conv_parameterization='softmax16g',
    )
    conv_module = model[0]
    assert isinstance(conv_module, LutConv2d)
    idx = torch.as_tensor(
        np.asarray(conv_node.params['indices_flat'], dtype=np.int64)
    )
    if tuple(idx.shape) != (conv_module.out_channels, conv_module.n_leaves):
        raise ValueError(
            f'indices_flat形状 {tuple(idx.shape)} != '
            f'({conv_module.out_channels}, {conv_module.n_leaves})'
        )
    with torch.no_grad():
        conv_module.indices_flat.copy_(idx.to(conv_module.indices_flat.device))
    return model


def polish_native_conv(
    model: nn.Sequential, x_tr: torch.Tensor, y_tr: torch.Tensor,
    x_ev_np: np.ndarray, y_ev_np: np.ndarray, *, device: str,
    epochs: int, lr: float, batch_size: int,
    teacher_logits_tr: torch.Tensor | None = None,
    kd_alpha: float = 0.5, kd_temperature: float = 4.0,
    kd_standardize: bool = True,
) -> dict:
    """native softmax16g Conv-DLGNを低LR CE(またはKD)でfine-tuneし、
    磨き前後のsoft/hard accを返す(Conv PoCのrun_arm_cと同じ磨き方)。
    """
    model = model.to(device)
    soft_before = soft_acc(model, x_tr, y_tr, device)
    hard_before = hard_acc_via_export(model, x_ev_np, y_ev_np)

    if teacher_logits_tr is not None:
        loss_fn = make_kd_loss(
            alpha=kd_alpha, temperature=kd_temperature,
            standardize=kd_standardize, kind='kl',
        )
        ds = TensorDataset(x_tr, y_tr, teacher_logits_tr)
    else:
        loss_fn = None
        ds = TensorDataset(x_tr, y_tr)

    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for _ in range(epochs):
        for batch in loader:
            if teacher_logits_tr is not None:
                xb, yb, tb = batch
                tb = tb.to(device)
            else:
                xb, yb = batch
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            out = model(xb)
            loss = loss_fn(out, yb, tb) if teacher_logits_tr is not None \
                else F.cross_entropy(out, yb)
            loss.backward()
            opt.step()

    soft_after = soft_acc(model, x_tr, y_tr, device)
    hard_after = hard_acc_via_export(model, x_ev_np, y_ev_np)
    return {
        'soft_acc_before': soft_before, 'hard_acc_before': hard_before,
        'soft_acc_after': soft_after, 'hard_acc_after': hard_after,
        'gap_after': soft_after - hard_after,
    }


def build_transcribed_conv_node(
    teacher_conv1: nn.Conv2d, calib_x_np: np.ndarray, *,
    tree_depth: int, seed: int,
) -> tuple[IRNode, dict]:
    """teacherの第1conv層をtranscribe_conv2d(n_inputs=2)で転写する。

    Returns:
        (node, approx_info): nodeはtrain_info.parameterization='independent'
        (transcribe_conv2dの既定契約のまま)。呼び出し側がarm Eでのみ
        rebuild_conv_as_native16gへ渡す。
    """
    calib_t = torch.as_tensor(calib_x_np, dtype=torch.float32)
    node, approx_info = transcribe_conv2d(
        teacher_conv1, calib_inputs=calib_t, tree_depth=tree_depth,
        n_inputs=NATIVE_N_INPUTS, seed=seed,
    )
    return node, approx_info


def run_arm_e(
    conv_node_independent: IRNode, x_all, y_all, x_all_t, y_all_t, *,
    head_width: int, args, device,
    teacher_logits_all: torch.Tensor | None = None,
) -> dict:
    """arm E(fair init: wiring+gate-init native16g→soft磨き)またはE+KD
    (teacher_logits_all指定時)。

    conv_node_independentから毎回新規にmodelを復元する(E本体とE+KDが
    同じ初期化から独立に出発できるよう呼び出し側でnodeを共有し、
    本関数内でrebuildする)。

    厳密性assert(`assert_conv_rebuild_bit_exact`)は「independent真理値表→
    softmax16g真理値表」の変換自体(rebuild_conv_as_native16g)が無損失で
    あることのみを確認する(purely representational)。その後の
    fair init(`build_native_model_from_conv_node(fair_init=True)`)は
    意図的に非飽和ロジットへ変換するため、fair init後のmodel自体は
    conv_node_independentとbit-exactではない(README/PROGRESS参照)。
    """
    node_16g = rebuild_conv_as_native16g(conv_node_independent)
    assert_conv_rebuild_bit_exact(conv_node_independent, node_16g, x_all)
    model = build_native_model_from_conv_node(
        node_16g, head_width=head_width, device=device, seed=args.seed_for_rebuild,
        fair_init=True,
        fair_transcribed_logit=args.fair_transcribed_logit,
        fair_residual_bias=args.fair_residual_bias,
    )
    res = polish_native_conv(
        model, x_all_t, y_all_t, x_all, y_all, device=device,
        epochs=args.polish_epochs, lr=args.polish_lr,
        batch_size=args.batch_size,
        teacher_logits_tr=teacher_logits_all,
        kd_alpha=args.alpha, kd_temperature=args.temperature,
        kd_standardize=args.standardize,
    )
    res['exactness_ok'] = True  # rebuild段(独立→16g真理値表)の厳密性のみ保証
    return res


def run_arm_e_wire(
    conv_node_independent: IRNode, x_all, y_all, x_all_t, y_all_t, *,
    head_width: int, args, device,
) -> dict:
    """arm E_wire(アブレーション): 配線(indices_flat)のみ転写、ゲートは
    scratch同様random+residual_init。wiring事前分布単独の寄与を見る。

    conv_node_independentはrun_arm_eと同じ転写結果を共有する(E/E_wireが
    同じ配線から出発する比較にするため、呼び出し側で共通のnodeを渡す)。
    """
    node_16g = rebuild_conv_as_native16g(conv_node_independent)
    assert_conv_rebuild_bit_exact(conv_node_independent, node_16g, x_all)
    model = build_native_model_wire_only(
        node_16g, head_width=head_width, device=device, seed=args.seed_for_rebuild,
    )
    res = polish_native_conv(
        model, x_all_t, y_all_t, x_all, y_all, device=device,
        epochs=args.polish_epochs, lr=args.polish_lr,
        batch_size=args.batch_size,
        teacher_logits_tr=None,
        kd_alpha=args.alpha, kd_temperature=args.temperature,
        kd_standardize=args.standardize,
    )
    return res


# ============================================================
# 1 seedの全アーム実行
# ============================================================

def run_one_seed(seed: int, args, device: str,
                 x_tr_all, y_tr_all, x_te, y_te) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    args.seed_for_rebuild = seed + 3000

    n_tr = args.train_n
    x_tr = x_tr_all[:n_tr]
    y_tr = y_tr_all[:n_tr]
    calib_x = x_tr_all[:args.calib_n]

    x_tr_t = torch.as_tensor(x_tr, dtype=torch.float32)
    y_tr_t = torch.as_tensor(y_tr, dtype=torch.int64)

    result: dict = {'seed': seed}

    # ---- teacher(TeacherConvSTE)学習: 第1conv出力を二値化するSTE teacher ----
    torch.manual_seed(seed)
    teacher = TeacherConvSTE(
        num_classes=N_CLASSES, in_channels=1, conv_out_ch=args.out_ch,
        hidden=args.teacher_hidden,
    ).to(device)
    teacher_train_acc = train_teacher(
        teacher, x_tr_t, y_tr_t, epochs=args.teacher_epochs,
        lr=args.teacher_lr, device=device, batch_size=args.batch_size,
    )
    with torch.no_grad():
        teacher.eval()
        pred = teacher(torch.as_tensor(x_te, dtype=torch.float32).to(device))
        teacher_test_acc = float((pred.argmax(-1).cpu().numpy() == y_te).mean())
    print(f'  [seed={seed}] teacher train_acc={teacher_train_acc:.4f} '
          f'test_acc={teacher_test_acc:.4f}')
    result['teacher_train_acc'] = teacher_train_acc
    result['teacher_test_acc'] = teacher_test_acc

    out_ch = teacher.conv1.out_channels
    teacher_conv1_cpu = copy.deepcopy(teacher.conv1).to('cpu')

    # ---- transcribe_conv2d(n_inputs=2、training-free近似構造転写) ----
    t0 = time.time()
    conv_node, approx_info = build_transcribed_conv_node(
        teacher_conv1_cpu, calib_x, tree_depth=args.tree_depth, seed=seed,
    )
    print(f'  [seed={seed}] transcribe_conv2d 近似agreement='
          f'{approx_info["approx_agreement"]:.4f} '
          f'(n_leaves={approx_info["n_leaves"]}, '
          f'receptive_field={approx_info["receptive_field"]}) '
          f'({time.time() - t0:.1f}s)')
    result['conv_approx_agreement'] = approx_info['approx_agreement']
    result['conv_approx_per_channel'] = approx_info['per_channel_agreement'].tolist()
    result['conv_n_leaves'] = approx_info['n_leaves']
    result['conv_receptive_field'] = approx_info['receptive_field']

    # ---- teacherロジット(arm B/E+KD用) ----
    teacher_logits_tr = make_teacher_logits(teacher, x_tr_t, device).cpu()

    # ---- arm A: scratch(native softmax16g) ----
    t0 = time.time()
    res_a = run_arm_a(x_tr_t, y_tr_t, x_tr, y_tr, out_ch=out_ch, args=args,
                      device=device, seed=seed + 1000)
    res_a['seconds'] = round(time.time() - t0, 2)
    print(f"  [seed={seed}] A(scratch native): soft={res_a['soft_acc']:.4f} "
          f"hard={res_a['hard_acc']:.4f} ({res_a['seconds']}s)")
    result['A'] = res_a

    # ---- arm B: IO-KDのみ(native softmax16g) ----
    t0 = time.time()
    res_b = run_arm_b(x_tr_t, y_tr_t, teacher_logits_tr, x_tr, y_tr,
                      out_ch=out_ch, args=args, device=device, seed=seed + 2000)
    res_b['seconds'] = round(time.time() - t0, 2)
    print(f"  [seed={seed}] B(IO-KD native): soft={res_b['soft_acc']:.4f} "
          f"hard={res_b['hard_acc']:.4f} ({res_b['seconds']}s)")
    result['B'] = res_b

    # ---- arm E: fair init(wiring+gate-init native16g、非飽和+残差保持) → soft磨き ----
    t0 = time.time()
    res_e = run_arm_e(
        conv_node, x_tr, y_tr, x_tr_t, y_tr_t, head_width=args.head_width,
        args=args, device=device, teacher_logits_all=None,
    )
    res_e['seconds'] = round(time.time() - t0, 2)
    print(f"  [seed={seed}] E(fair init native+soft磨き, 本命): "
          f"soft {res_e['soft_acc_before']:.4f}->{res_e['soft_acc_after']:.4f} "
          f"hard {res_e['hard_acc_before']:.4f}->{res_e['hard_acc_after']:.4f} "
          f"({res_e['seconds']}s)")
    result['E'] = res_e

    # ---- arm E_wire: wiring転写のみ(ゲートはrandom+residual_init) → soft磨き ----
    t0 = time.time()
    res_ewire = run_arm_e_wire(
        conv_node, x_tr, y_tr, x_tr_t, y_tr_t, head_width=args.head_width,
        args=args, device=device,
    )
    res_ewire['seconds'] = round(time.time() - t0, 2)
    print(f"  [seed={seed}] E_wire(wiring転写のみ+soft磨き, アブレーション): "
          f"soft {res_ewire['soft_acc_before']:.4f}->{res_ewire['soft_acc_after']:.4f} "
          f"hard {res_ewire['hard_acc_before']:.4f}->{res_ewire['hard_acc_after']:.4f} "
          f"({res_ewire['seconds']}s)")
    result['E_wire'] = res_ewire

    # ---- arm E+KD: 同じfair初期化からKD磨き ----
    t0 = time.time()
    res_ekd = run_arm_e(
        conv_node, x_tr, y_tr, x_tr_t, y_tr_t, head_width=args.head_width,
        args=args, device=device, teacher_logits_all=teacher_logits_tr,
    )
    res_ekd['seconds'] = round(time.time() - t0, 2)
    print(f"  [seed={seed}] E+KD(fair init native+KD磨き): "
          f"soft {res_ekd['soft_acc_before']:.4f}->{res_ekd['soft_acc_after']:.4f} "
          f"hard {res_ekd['hard_acc_before']:.4f}->{res_ekd['hard_acc_after']:.4f} "
          f"({res_ekd['seconds']}s)")
    result['E_KD'] = res_ekd

    # ---- ゲート数(native A/B/E/E_wire回路、いずれも同型なので1回構築して測る) ----
    gate_probe_model = build_conv_dlgn(
        out_ch=out_ch, tree_depth=args.tree_depth, n_inputs=NATIVE_N_INPUTS,
        head_width=args.head_width, device='cpu', seed=0,
        conv_parameterization='softmax16g',
    )
    result['gates_native'] = native_conv_gate_count(gate_probe_model)

    # ---- 派生指標 ----
    hard_a = res_a['hard_acc']
    hard_b = res_b['hard_acc']
    hard_e = res_e['hard_acc_after']
    hard_ewire = res_ewire['hard_acc_after']
    hard_ekd = res_ekd['hard_acc_after']
    result['lift_E_vs_A'] = hard_e - hard_a
    result['lift_E_vs_B'] = hard_e - hard_b
    result['lift_Ewire_vs_A'] = hard_ewire - hard_a
    result['lift_E_vs_Ewire'] = hard_e - hard_ewire
    result['lift_EKD_vs_E'] = hard_ekd - hard_e
    # Eのfair init直後(磨き前)soft acc: chanceでないことの直接的な確認
    result['E_init_soft_acc'] = res_e['soft_acc_before']

    return result


# ============================================================
# 集計・表示
# ============================================================

def _mean_std(vals: list[float]) -> tuple[float, float]:
    arr = np.asarray(vals, dtype=np.float64)
    return float(arr.mean()), float(arr.std())


def summarize(results: list[dict], args) -> dict:
    print('\n' + '=' * 100)
    print('追補: fair init — CNN構造転写(E)/配線のみ(E_wire)/IO-KD(B)/'
          'init+KD(E+KD) vs scratch(A) — MNIST結果')
    print('=' * 100)

    def get_hard(arm, r):
        if arm in ('A', 'B'):
            return r[arm]['hard_acc']
        return r[arm]['hard_acc_after']

    def get_soft(arm, r):
        if arm in ('A', 'B'):
            return r[arm]['soft_acc']
        return r[arm]['soft_acc_after']

    arms = ['A', 'B', 'E', 'E_wire', 'E_KD']
    labels = {
        'A': 'A: scratch native',
        'B': 'B: IO-KDのみ native',
        'E': 'E: CNN構造転写(fair init: wiring+gate-init)native + soft磨き(本命)',
        'E_wire': 'E_wire: 配線のみ転写(ゲートrandom+residual_init)native + soft磨き',
        'E_KD': 'E+KD: CNN構造転写(fair init)native + KD磨き',
    }

    print('| arm | soft_acc(mean±std) | hard_acc(mean±std) |')
    print('|---|---|---|')
    summary: dict = {'arms': {}}
    for arm in arms:
        softs = [get_soft(arm, r) for r in results]
        hards = [get_hard(arm, r) for r in results]
        sm, ss = _mean_std(softs)
        hm, hs = _mean_std(hards)
        print(f'| {labels[arm]} | {sm:.4f}±{ss:.4f} | {hm:.4f}±{hs:.4f} |')
        summary['arms'][arm] = {
            'soft_acc_mean': sm, 'soft_acc_std': ss,
            'hard_acc_mean': hm, 'hard_acc_std': hs,
        }

    gates = [r['gates_native'] for r in results]
    gm, gs = _mean_std([float(g) for g in gates])
    print(f'\nnativeゲート数(mean±std): {gm:.0f}±{gs:.0f}')
    summary['gates_native_mean'] = gm

    agreements = [r['conv_approx_agreement'] for r in results]
    am, as_ = _mean_std(agreements)
    print(f'転写conv近似agreement(較正データ, mean±std): {am:.4f}±{as_:.4f}')
    summary['conv_approx_agreement_mean'] = am
    summary['conv_approx_agreement_std'] = as_

    e_init_soft = [r['E_init_soft_acc'] for r in results]
    eism, eiss = _mean_std(e_init_soft)
    print(f'E fair init直後(磨き前)soft_acc(mean±std): {eism:.4f}±{eiss:.4f} '
          f'(chance={1.0 / N_CLASSES:.4f})')
    summary['E_init_soft_acc_mean'] = eism
    summary['E_init_soft_acc_std'] = eiss

    lift_e_a = [r['lift_E_vs_A'] for r in results]
    lift_e_b = [r['lift_E_vs_B'] for r in results]
    lift_ewire_a = [r['lift_Ewire_vs_A'] for r in results]
    lift_e_ewire = [r['lift_E_vs_Ewire'] for r in results]
    lift_ekd_e = [r['lift_EKD_vs_E'] for r in results]

    leam, leas = _mean_std(lift_e_a)
    lebm, lebs = _mean_std(lift_e_b)
    lewam, lewas = _mean_std(lift_ewire_a)
    leewm, leews = _mean_std(lift_e_ewire)
    lkem, lkes = _mean_std(lift_ekd_e)

    print('\n--- 派生指標(mean±std over seeds) ---')
    print(f'lift(E - A)          : {leam:+.4f}±{leas:.4f}')
    print(f'lift(E - B)          : {lebm:+.4f}±{lebs:.4f}')
    print(f'lift(E_wire - A)     : {lewam:+.4f}±{lewas:.4f}  (wiring事前分布単独)')
    print(f'lift(E - E_wire)     : {leewm:+.4f}±{leews:.4f}  (gate-init/関数保存の寄与)')
    print(f'lift(E+KD - E)       : {lkem:+.4f}±{lkes:.4f}')

    summary['lift_E_vs_A_mean'] = leam
    summary['lift_E_vs_B_mean'] = lebm
    summary['lift_Ewire_vs_A_mean'] = lewam
    summary['lift_E_vs_Ewire_mean'] = leewm
    summary['lift_EKD_vs_E_mean'] = lkem

    print('\n--- 仮説判定 ---')
    print(f'H(CNN構造転写(E) > scratch(A)): lift(E-A)={leam:+.4f} '
          f"→ {'支持' if leam > 0 else '不支持'}")
    print(f'H(CNN構造転写(E) > IO-KD(B)): lift(E-B)={lebm:+.4f} '
          f"→ {'支持' if lebm > 0 else '不支持'}")
    print(f'H(配線のみ(E_wire) > scratch(A)): lift(E_wire-A)={lewam:+.4f} '
          f"→ {'支持' if lewam > 0 else '不支持'}")
    print(f'H(gate-init追加(E) > 配線のみ(E_wire)): lift(E-E_wire)={leewm:+.4f} '
          f"→ {'支持' if leewm > 0 else '不支持'}")
    print(f'H(init+KD(E+KD) > init単独(E)): lift(E+KD-E)={lkem:+.4f} '
          f"→ {'支持' if lkem > 0 else '不支持'}")

    exactness_all = all(r['E']['exactness_ok'] for r in results)
    print(f'\nrebuild段(independent→softmax16g真理値表)bit-exact厳密性assert'
          f'(全seed): {"OK" if exactness_all else "NG"}')
    print('注意: fair init(E/E+KD)後のmodel自体は非飽和ロジットへ意図的に'
          '変換しており、bit-exactではない(残差事前分布を保つためのトレードオフ)。')
    summary['exactness_ok'] = exactness_all
    print('\n注意: 絶対精度はPoC規模(小学習件数・少epoch・小teacher)で頭打ち。'
          'transcribe_conv2dは近似転写(read-once木が受容野全体でなくサンプル'
          'リーフのみを見るため)。agreementは実測値であり厳密性0の主張ではない。')

    return summary


# ============================================================
# メイン
# ============================================================

def _strip_extra(r: dict) -> dict:
    return {k: v for k, v in r.items() if k not in ('conv_approx_per_channel',)}


def main():
    parser = argparse.ArgumentParser(
        description='コンパクトnative Conv-DLGN獲得(CNN構造転写) vs scratch/IO-KD、'
                    'MNIST')
    parser.add_argument('--seeds', type=int, default=2)
    parser.add_argument('--train-n', type=int, default=4000,
                        help='arm A/B/E学習件数(共通、既定4000)')
    parser.add_argument('--calib-n', type=int, default=2000,
                        help='transcribe_conv2d較正件数')
    parser.add_argument('--eval-n', type=int, default=2000,
                        help='test評価件数')
    parser.add_argument('--epochs', type=int, default=5,
                        help='arm A/B(scratch native/IO-KD native)の学習エポック数')
    parser.add_argument('--teacher-epochs', type=int, default=3)
    parser.add_argument('--teacher-lr', type=float, default=1e-3)
    parser.add_argument('--teacher-hidden', type=int, default=64,
                        help='TeacherConvSTEのFC隠れ幅')
    parser.add_argument('--lr', type=float, default=0.01, help='arm A/BのAdam学習率')
    parser.add_argument('-bs', '--batch-size', type=int, default=128)
    parser.add_argument('--out-ch', type=int, default=16,
                        help='conv出力チャンネル数(TeacherConvSTE.conv1と'
                             'native Conv-DLGN共通、既定16= Conv PoCの32より'
                             '小さいcompact構成)')
    parser.add_argument('--tree-depth', type=int, default=3,
                        help='LutConv2d/transcribe_conv2dの木の深さ'
                             '(既定3、n_leaves=2^3=8)')
    parser.add_argument('--n-inputs', type=int, default=NATIVE_N_INPUTS,
                        choices=(NATIVE_N_INPUTS,),
                        help='compact native Conv-DLGNのLUT fan-in(常に2固定)')
    parser.add_argument('--head-width', type=int, default=256,
                        help='未使用(build_conv_dlgnの予約引数を継承)')
    parser.add_argument('--alpha', type=float, default=0.5, help='KD損失のCE項重み')
    parser.add_argument('--temperature', type=float, default=4.0, help='KD温度T')
    parser.add_argument('--standardize', action='store_true', default=True,
                        help='KD項の両ロジットをZ-score標準化する(既定ON)')
    # arm E/E+KD 磨き。旧既定(20ep/lr=0.3)は飽和init(logit_m=10)を前提に
    # 較正されたものだったが、fair init導入でchance初期値問題は解消したため、
    # 磨き予算そのものを再較正する(40-60epに拡大、--polish-lrはrobustness
    # チェック対象としてCLIから振れるようにする)。
    parser.add_argument('--polish-epochs', type=int, default=40)
    parser.add_argument('--polish-lr', type=float, default=0.3)
    # fair init(arm E/E+KD)のハイパラ: 転写ゲートへの中庸ロジットと
    # ゲートA(パススルー)への残差バイアス(両方ともDEFAULT_LOGIT_M=10.0より
    # 十分小さく、softmaxが飽和しないようにする)
    parser.add_argument('--fair-transcribed-logit', type=float, default=3.0,
                        help='fair init: 転写ゲートのロジットへ加算する値'
                             '(既定3.0、飽和logit_m=10.0より小)')
    parser.add_argument('--fair-residual-bias', type=float, default=5.0,
                        help='fair init: ゲートA(パススルー)のロジットへ'
                             '加算する値(既定5.0、LutConv2d._residual_initの'
                             'z3既定と同じ)')
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--quick', action='store_true',
                        help='煙実行(1 seed・小データ・少epoch、<90s目標)')
    parser.add_argument('--no-wandb', action='store_true')
    parser.add_argument('--out-dir', type=str,
                        default='outputs/mnist_to_native_conv_dlgn')
    args = parser.parse_args()

    device = _auto_device(args.device)
    if args.no_wandb:
        os.environ['WANDB_MODE'] = 'disabled'

    if args.quick:
        args.seeds = 1
        args.train_n = 500
        args.calib_n = 300
        args.eval_n = 300
        args.epochs = 1
        args.teacher_epochs = 1
        args.teacher_hidden = 32
        args.tree_depth = 2
        args.out_ch = 8
        args.polish_epochs = 3
        args.polish_lr = 0.3

    print(f'デバイス: {device}')
    print(f'seeds={args.seeds} train_n={args.train_n} calib_n={args.calib_n} '
          f'eval_n={args.eval_n}')
    print(f'native Conv-DLGN(arm A/B/E/E_wire共通): out_ch={args.out_ch} '
          f'tree_depth={args.tree_depth} n_inputs={NATIVE_N_INPUTS} '
          f'(n_leaves={NATIVE_N_INPUTS ** args.tree_depth}) parameterization=softmax16g')
    print(f'teacher(TeacherConvSTE): conv_out_ch={args.out_ch} '
          f'hidden={args.teacher_hidden} epochs={args.teacher_epochs} lr={args.teacher_lr}')
    print(f'arm A/B: epochs={args.epochs} lr={args.lr}')
    print(f'arm E/E_wire/E+KD(polish): epochs={args.polish_epochs} lr={args.polish_lr}')
    print(f'arm E fair init: transcribed_logit={args.fair_transcribed_logit} '
          f'residual_bias={args.fair_residual_bias}')

    print('\nMNIST(binarize_threshold=0.5)読み込み中...')
    train_set, test_set = load_mnist(binarize_threshold=0.5)
    x_tr_all, y_tr_all = _flatten_binary_images(
        train_set, n=max(args.train_n, args.calib_n) + 1000)
    x_te, y_te = _flatten_binary_images(test_set, n=args.eval_n)
    print(f'train_pool={len(x_tr_all)} test={len(x_te)}')

    use_wandb = init_wandb(
        run_name=f'mnist_to_native_conv_dlgn_d{args.tree_depth}',
        config_dict=vars(args),
        device=device,
        tags=['D15', 'native-dlgn', 'conv', 'structural-transfer', 'kd', 'fair-init'],
        group='convert-mnist-to-native-conv-dlgn',
    )

    started = time.time()
    results = []
    for seed in range(args.seeds):
        print(f'\n########## seed {seed + 1}/{args.seeds} ##########')
        r = run_one_seed(seed, args, device, x_tr_all, y_tr_all, x_te, y_te)
        results.append(r)
        if use_wandb:
            log_metrics({
                f'seed{seed}/teacher_test_acc': r['teacher_test_acc'],
                f'seed{seed}/conv_approx_agreement': r['conv_approx_agreement'],
                f'seed{seed}/A_hard_acc': r['A']['hard_acc'],
                f'seed{seed}/B_hard_acc': r['B']['hard_acc'],
                f'seed{seed}/E_hard_acc': r['E']['hard_acc_after'],
                f'seed{seed}/E_init_soft_acc': r['E_init_soft_acc'],
                f'seed{seed}/Ewire_hard_acc': r['E_wire']['hard_acc_after'],
                f'seed{seed}/EKD_hard_acc': r['E_KD']['hard_acc_after'],
                f'seed{seed}/gates_native': r['gates_native'],
                f'seed{seed}/lift_E_vs_A': r['lift_E_vs_A'],
                f'seed{seed}/lift_E_vs_B': r['lift_E_vs_B'],
                f'seed{seed}/lift_Ewire_vs_A': r['lift_Ewire_vs_A'],
                f'seed{seed}/lift_E_vs_Ewire': r['lift_E_vs_Ewire'],
                f'seed{seed}/lift_EKD_vs_E': r['lift_EKD_vs_E'],
            })

    summary = summarize(results, args)
    if use_wandb:
        log_metrics({f'summary/{k}': v for k, v in summary.items()
                     if isinstance(v, int | float)})
        wandb_finish()

    os.makedirs(args.out_dir, exist_ok=True)
    ts = datetime.now(JST).strftime('%Y%m%d_%H%M%S')
    out_path = os.path.join(args.out_dir, f'result_{ts}.json')
    dump = {'args': vars(args), 'summary': summary,
            'per_seed': [_strip_extra(r) for r in results]}
    with open(out_path, 'w') as f:
        json.dump(dump, f, indent=2, default=str)
    print(f'\n結果を保存: {out_path}')

    total = str(timedelta(seconds=int(time.time() - started)))
    print(f'完了(総時間 {total})')


if __name__ == '__main__':
    main()
