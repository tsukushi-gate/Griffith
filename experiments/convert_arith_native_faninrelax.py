"""fan-in≤2制約の緩和がnative DLGN獲得のボトルネックを解消するか
(ROADMAP-v5 Track D、`convert_arith_to_native_dlgn.py`の続き)。

確認したボトルネック: compact native DLGN(difflogic方式、
`LutLayer(n_inputs=2, parameterization='softmax')`の16ゲートセル)を
teacherから獲得するには、teacherをfan-in<=2まで枝刈りする必要があり、
この枝刈りがteacher精度を大きく損なう(算術oracle損失+0.52程度)。結果、
native arm E(wiring+gate-init→soft磨き)はhard 0.585止まりで、non-native
warp(n_inputs=6の構造転写、structural_transfer arm C)のhard 0.988に
遠く及ばない。

本タスクは「fan-in<=2制約そのものを緩和すれば、nativeのままwarp天井へ
近づけるか」を検証する。緩和の軸は2つ:
    (a) 配線を学習可能にする(DWN方式のLearnableMapping、知見:
        parity-in-noiseで配線学習が0.64→1.0の停滞解消を示した実績を
        native DLGN獲得へ転用)
    (b) native LUTのfan-in自体をN=2からN=4/6へ上げる(independent
        パラメータ化なら任意真理値表を厳密表現できるため、fan-in枝刈りの
        近似誤差(oracle損失)自体が縮む)

## アーム(算術n=4加算、native DLGNのhard accを測定。すべて`convert_arith_to_native_dlgn.py`
  のteacher/データセット/評価ヘルパを無改造で再利用)

- **E(基準、そのまま再掲)**: n_inputs=2固定配線、fan-in<=2枝刈り→
  `rebuild_as_native16g`→soft磨き。実測~0.585の参照点。
- **E_dwn(配線緩和)**: Eと全く同じ転写元IR(fan-in<=2枝刈り済みteacher、
  n_inputs=2)を使うが、復元後のnative16g DLGNの配線を`wiring='learnable'`
  (DWN LearnableMapping)へ差し替える。ゲート初期値は転写と同じ
  (softmax16g argmax one-hot)、配線初期値は転写indices由来のargmax
  one-hot(`restore_dwn_layer`と同じ規約)にして、E同一の初期条件から
  「配線も学習対象にする」ことの純効果を測る。磨き中に配線が動けるので、
  fan-in<=2枝刈りで失われた接続情報の一部を勾配で再発見できるか検証する。
- **E_n4 / E_n6(fan-in緩和)**: teacherをn_inputs=4/6まで(2でなく)枝刈りし
  transcribe(近似誤差が小さいはず)、`rebuild_trainable(parameterization=
  'independent')`でnative(warpでない、任意真理値表を直接保持する)DLGNへ
  復元→soft磨き。ゲート数はn_inputsに対し指数(2^n_inputs)で増えるため、
  fan-in緩和のコストをゲート数で定量化する。
- **A(参照、再掲)**: scratch native(n_inputs=2)。下限対照。
- **warp参照(再掲、非native)**: `run_arm_c`(structural_transfer,
  fanin=6, selection='soft')。0.988程度が天井。

## 派生指標

- lift(E_dwn - E): 配線緩和の純効果
- lift(E_n4 - E), lift(E_n6 - E): fan-in緩和の純効果(n_inputs別)
- gap(E_n6 - warp): fan-in=6 native が warp天井にどれだけ迫れるか
- gate_ratio(E_n4|E_n6 / warp): fan-in緩和のコンパクトさコスト
- fan-in={2,4,6}それぞれのoracle損失(teacher-after-prune精度)

## 使用例

    # 煙実行(数十秒、wandb無効、1 seed・小width・少epoch)
    uv run python experiments/convert_arith_native_faninrelax.py --quick --no-wandb

    # 本実行(既定nbits=4、3 seeds)
    uv run python experiments/convert_arith_native_faninrelax.py --nbits 4 --seeds 3 --no-wandb
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
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
# の機構を無改造でimport(task指示: base実験ファイルを改変せず再利用する)
from convert_arith_structural_transfer import (
    ArithTeacherMlp,
    arith_classify_dataset,
    assert_transcribe_exact,
    build_transcribed_ir,
    ensure_teacher_fanin,
    hard_acc_via_export,
    make_teacher_logits,
    num_classes_for,
    run_arm_a,
    run_arm_c,
    train_teacher,
)
from convert_arith_to_native_dlgn import (
    NATIVE_N_INPUTS,
    assert_native16g_bit_exact,
    hard_acc_via_eval_forward,
    native_gate_count,
    polish_native,
    rebuild_as_native16g,
    run_arm_e,
)
from env_utils import load_dotenv
from wandb_utils import finish as wandb_finish
from wandb_utils import init_wandb, log_metrics

from griffith_logic.convert.recover import RECOVER_PARAMETERIZATIONS, rebuild_trainable
from griffith_logic.export.estimate import estimate_resources
from griffith_logic.ir.schema import IRGraph
from griffith_logic.ir.simulate import simulate

load_dotenv()

EPS = 1e-8
RELAX_N_INPUTS = (4, 6)  # fan-in緩和で試すnative n_inputs(LutLayer対応上限は6)


# ============================================================
# E_dwn: 固定配線native16gを学習可能配線(DWN LearnableMapping)へ差し替える
# ============================================================

def rebuild_as_native16g_learnable_wiring(graph: IRGraph) -> torch.nn.Module:
    """n_inputs=2の転写IRをwiring='learnable'なsoftmax16g native DLGNへ復元する。

    まず`rebuild_as_native16g`(固定配線、bit-exact)でモデルを構築し、次に
    各LutLayerを同形状・同ゲート重みのwiring='learnable'版へ差し替える。
    LearnableMapping.weightsは、固定配線が復元したindicesと同じ接続を
    argmax one-hotで再現するよう初期化する(`ir/weights.py::restore_dwn_layer`
    がDWNLayerのmapping='learnable'に対して行うのと同じ規約:
    flat[u*n_inputs+j] = indices[j,u] の位置に1.0を置く)。

    これにより「E(固定配線)とE_dwn(学習配線)が全く同じ初期条件(ゲートも
    配線も)から出発し、磨き中に配線が動けるかどうかだけが違う」という
    比較対象を作る。

    Args:
        graph: n_inputs=2で転写されたIR(直列: 入力→lut_layer列→group_sum)。

    Returns:
        wiring='learnable'なLutLayer列を持つnative softmax16g DLGN
        (nn.Sequential)。
    """
    from griffith_logic.layers.lut import LutLayer

    fixed_model = rebuild_as_native16g(graph)

    new_layers = []
    for m in fixed_model:
        if isinstance(m, LutLayer):
            learnable = LutLayer(
                m.in_dim, m.out_dim, n_inputs=m.n_inputs, device=m.weights.device,
                parameterization='softmax', selection=m.selection, wiring='learnable',
            )
            with torch.no_grad():
                # ゲート重み(16ロジット)はfixed_modelの復元値をそのまま複製
                learnable.weights.copy_(m.weights)
                # 配線初期値: fixed_modelのindices(n_inputs, out_dim)をargmax
                # one-hotとしてlearnable_mapping.weightsへ書き込む
                # (restore_dwn_layerのmapping='learnable'分岐と同じ規約)
                w = torch.zeros_like(learnable.learnable_mapping.weights)
                flat = m.indices.T.reshape(-1)  # flat[u*n_inputs+j] = indices[j,u]
                w[flat, torch.arange(flat.numel(), device=flat.device)] = 1.0
                learnable.learnable_mapping.weights.copy_(w)
            new_layers.append(learnable)
        else:
            new_layers.append(copy.deepcopy(m))
    return torch.nn.Sequential(*new_layers)


def assert_native16g_learnable_wiring_bit_exact(
    fixed_model: torch.nn.Module, learnable_model: torch.nn.Module,
    x_np: np.ndarray,
) -> None:
    """rebuild_as_native16g_learnable_wiring直後のargmaxがfixed版と全パターン一致することを確認する。

    配線初期値がfixed版のindicesと厳密に同じargmax one-hotなら、
    LearnableMappingFunction.forwardのハード選択(argmax(W))はfixed版が
    使うindicesと完全一致するため、この一致は理論上常に成立するはず。
    破れたらバグなので閾値を緩めずSTOPする。
    """
    fixed_model.eval()
    learnable_model.eval()
    with torch.no_grad():
        x_t = torch.as_tensor(x_np, dtype=torch.float32)
        y_fixed = fixed_model(x_t).numpy()
        y_learnable = learnable_model(x_t).numpy()
    n_mismatch = int((y_fixed.argmax(axis=-1) != y_learnable.argmax(axis=-1)).sum())
    if n_mismatch != 0:
        raise RuntimeError(
            f'厳密性assert違反: rebuild_as_native16g_learnable_wiring直後のargmaxが'
            f'固定配線版と{n_mismatch}/{len(x_np)}件不一致。配線初期化ロジックの'
            'バグの可能性が高い(閾値は緩めない)。'
        )


def run_arm_e_dwn(
    native_ir, x_all, y_all, x_all_t, y_all_t, *, in_width, args, device,
) -> dict:
    """arm E_dwn(配線緩和): native16g+wiring='learnable'→soft磨き。

    hard acc測定はexport_ir経由の`hard_acc_via_export`が使えない
    (`LutLayer(wiring='learnable')`は`export/discretize.py::
    truth_tables_from_composition`のprobe機構がfixed配線専用の実装のため
    現状crashする。既存コードは無改造の方針のためここでは回避し、
    `hard_acc_via_eval_forward`(eval()forwardのargmaxを直接取る、
    arm Fと同じ設計)で代替する。eval forwardはwiring非依存の正しい
    離散化パスなので、hard精度としての妥当性はexport経由と同等)。
    """
    fixed_model = rebuild_as_native16g(native_ir)
    model = rebuild_as_native16g_learnable_wiring(native_ir)
    assert_native16g_learnable_wiring_bit_exact(fixed_model, model, x_all)
    model = model.to(device)

    soft_acc_before = _soft_acc(model, x_all_t, y_all_t, device)
    hard_acc_before = hard_acc_via_eval_forward(model, x_all, y_all, device)

    loader = DataLoader(TensorDataset(x_all_t, y_all_t), batch_size=args.batch_size,
                        shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=args.polish_lr)
    model.train()
    for _ in range(args.polish_epochs):
        for xb, yb in loader:
            xb, yb = xb.to(device).float(), yb.to(device)
            opt.zero_grad()
            loss = F.cross_entropy(model(xb), yb)
            loss.backward()
            opt.step()

    soft_acc_after = _soft_acc(model, x_all_t, y_all_t, device)
    hard_acc_after = hard_acc_via_eval_forward(model, x_all, y_all, device)
    return {
        'soft_acc_before': soft_acc_before, 'hard_acc_before': hard_acc_before,
        'soft_acc_after': soft_acc_after, 'hard_acc_after': hard_acc_after,
        'gap_after': soft_acc_after - hard_acc_after,
        'exactness_ok': True,
    }


@torch.no_grad()
def _soft_acc(model, x, y, device) -> float:
    model.eval()
    out = model(x.to(device).float())
    return float((out.argmax(-1).cpu() == y).float().mean())


# ============================================================
# E_n4 / E_n6: fan-in緩和(native n_inputs=4/6, independentパラメータ化)
# ============================================================

def run_arm_e_relaxed_fanin(
    relaxed_ir, x_all, y_all, x_all_t, y_all_t, *, in_width, args, device,
) -> dict:
    """arm E_n{4,6}: fan-in緩和native(independentパラメータ化)→soft磨き。

    `rebuild_trainable(parameterization='independent')`
    (`convert/recover.py`)がn_inputs=4/6でも任意真理値表をbit-exactに
    復元することを利用する(softmax(N>2)やiwpと違いindependentは
    popcount制約なしで転写直後と一致する、`ir/weights.py`のdocstring参照)。
    """
    if args.relax_parameterization not in RECOVER_PARAMETERIZATIONS:
        raise ValueError(
            f'--relax-parameterizationは{RECOVER_PARAMETERIZATIONS}のいずれか'
        )
    model = rebuild_trainable(
        relaxed_ir, parameterization=args.relax_parameterization, selection='soft',
    )
    model = model.to(device)

    # bit-exactness: rebuild直後(eval)のargmaxが転写IRのsimulateと一致すること
    model.eval()
    with torch.no_grad():
        y_model = model(x_all_t.to(device)).cpu().numpy()
    y_sim = simulate(relaxed_ir, x_all)
    n_mismatch = int((y_model.argmax(axis=-1) != y_sim.argmax(axis=-1)).sum())
    if n_mismatch != 0:
        raise RuntimeError(
            f'厳密性assert違反: rebuild_trainable(parameterization='
            f"'{args.relax_parameterization}')直後のargmaxが転写IR argmaxと"
            f'{n_mismatch}/{len(x_all)}件不一致(independentは任意真理値表を'
            '厳密表現できるはずなのでバグ、閾値は緩めない)。'
        )

    res = polish_native(
        model, x_all_t, y_all_t, x_all, y_all, in_width=in_width,
        device=device, epochs=args.polish_epochs, lr=args.polish_lr,
        batch_size=args.batch_size,
    )
    res['exactness_ok'] = True
    return res


# ============================================================
# 1 seedの全アーム実行
# ============================================================

def run_one_seed(seed: int, args, device: str) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)

    n = args.nbits
    task = args.task
    in_width = 2 * n
    k_classes = num_classes_for(n, task)
    x_all, y_all = arith_classify_dataset(n, task)
    x_all_t = torch.as_tensor(x_all, dtype=torch.float32)
    y_all_t = torch.as_tensor(y_all, dtype=torch.int64)

    print(f'  [seed={seed}] task={task} n={n} in_width={in_width} '
          f'k_classes={k_classes} n_patterns={len(x_all)}')

    result: dict = {'seed': seed, 'n': n, 'k_classes': k_classes}

    # ---- teacher学習(暗記、fan-in=2/4/6それぞれに独立に枝刈りするため
    #      枝刈り前の同一初期化teacherを都度作り直す) ----
    def fresh_teacher() -> ArithTeacherMlp:
        torch.manual_seed(seed)
        t = ArithTeacherMlp(in_width, args.teacher_hidden, k_classes).to(device)
        train_teacher(
            t, x_all_t, y_all_t, epochs=args.teacher_epochs,
            lr=args.teacher_lr, device=device, batch_size=args.batch_size,
        )
        return t

    teacher_before = fresh_teacher()
    teacher_acc_before_prune = float(
        (teacher_before.eval()(x_all_t.to(device)).argmax(-1).cpu()
         == y_all_t).float().mean()
    )
    print(f'  [seed={seed}] teacher train hard_acc(枝刈り前)='
          f'{teacher_acc_before_prune:.4f}')
    result['teacher_acc_before_prune'] = teacher_acc_before_prune

    # ---- fan-in=2枝刈り(E/E_dwn共通の転写元、同一条件) ----
    teacher2, _ = ensure_teacher_fanin(
        fresh_teacher(), x_all, y_all, n_inputs=NATIVE_N_INPUTS,
        epochs_per_stage=args.teacher_prune_epochs, lr=args.teacher_lr,
        device=device,
    )
    teacher_acc_after_prune2 = float(
        (teacher2.to(device).eval()(x_all_t.to(device)).argmax(-1).cpu()
         == y_all_t).float().mean()
    )
    print(f'  [seed={seed}] teacher fan-in<=2枝刈り後 hard_acc='
          f'{teacher_acc_after_prune2:.4f}')
    result['teacher_acc_after_prune_fanin2'] = teacher_acc_after_prune2
    result['fanin2_oracle_loss'] = (
        teacher_acc_before_prune - teacher_acc_after_prune2
    )

    ir2, teacher2_cpu = build_transcribed_ir(
        teacher2, x_all, n_inputs=NATIVE_N_INPUTS, weight_bits=args.weight_bits,
        device=device,
    )
    assert_transcribe_exact(ir2, teacher2_cpu, x_all, args.weight_bits)
    gates_e = native_gate_count(ir2)
    result['gates_E'] = gates_e
    print(f'  [seed={seed}] fan-in<=2転写 厳密性assert: OK  gates(native E)={gates_e}')

    # ---- arm A: scratch(compact native, n_inputs=2、参照) ----
    t0 = time.time()
    res_a = run_arm_a(
        x_all_t, y_all_t, x_all, y_all, in_width=in_width, k_classes=k_classes,
        args=args, device=device, seed=seed + 1000,
    )
    res_a['seconds'] = round(time.time() - t0, 2)
    print(f"  [seed={seed}] A(scratch native): soft={res_a['soft_acc']:.4f} "
          f"hard={res_a['hard_acc']:.4f} ({res_a['seconds']}s)")
    result['A'] = res_a

    # ---- arm E: 固定配線native16g(基準、再掲) ----
    t0 = time.time()
    res_e = run_arm_e(
        ir2, x_all, y_all, x_all_t, y_all_t, in_width=in_width, args=args,
        device=device, teacher_logits_all=None,
    )
    res_e['seconds'] = round(time.time() - t0, 2)
    print(f"  [seed={seed}] E(fan-in<=2, 固定配線, 基準): "
          f"soft {res_e['soft_acc_before']:.4f}->{res_e['soft_acc_after']:.4f} "
          f"hard {res_e['hard_acc_before']:.4f}->{res_e['hard_acc_after']:.4f} "
          f"({res_e['seconds']}s)")
    result['E'] = res_e

    # ---- arm E_dwn: 学習可能配線(配線緩和) ----
    t0 = time.time()
    res_edwn = run_arm_e_dwn(
        ir2, x_all, y_all, x_all_t, y_all_t, in_width=in_width, args=args,
        device=device,
    )
    res_edwn['seconds'] = round(time.time() - t0, 2)
    print(f"  [seed={seed}] E_dwn(fan-in<=2, 学習配線): "
          f"soft {res_edwn['soft_acc_before']:.4f}->{res_edwn['soft_acc_after']:.4f} "
          f"hard {res_edwn['hard_acc_before']:.4f}->{res_edwn['hard_acc_after']:.4f} "
          f"({res_edwn['seconds']}s)")
    result['E_dwn'] = res_edwn

    # ---- arm E_n4 / E_n6: fan-in緩和(native n_inputs=4/6) ----
    for relax_n in args.relax_n_inputs:
        teacher_r, _ = ensure_teacher_fanin(
            fresh_teacher(), x_all, y_all, n_inputs=relax_n,
            epochs_per_stage=args.teacher_prune_epochs, lr=args.teacher_lr,
            device=device,
        )
        teacher_acc_after_prune_r = float(
            (teacher_r.to(device).eval()(x_all_t.to(device)).argmax(-1).cpu()
             == y_all_t).float().mean()
        )
        print(f'  [seed={seed}] teacher fan-in<={relax_n}枝刈り後 hard_acc='
              f'{teacher_acc_after_prune_r:.4f}')
        result[f'teacher_acc_after_prune_fanin{relax_n}'] = teacher_acc_after_prune_r
        result[f'fanin{relax_n}_oracle_loss'] = (
            teacher_acc_before_prune - teacher_acc_after_prune_r
        )

        ir_r, teacher_r_cpu = build_transcribed_ir(
            teacher_r, x_all, n_inputs=relax_n, weight_bits=args.weight_bits,
            device=device,
        )
        assert_transcribe_exact(ir_r, teacher_r_cpu, x_all, args.weight_bits)
        gates_r = native_gate_count(ir_r)
        result[f'gates_E_n{relax_n}'] = gates_r
        print(f'  [seed={seed}] fan-in<={relax_n}転写 厳密性assert: OK  '
              f'gates(native E_n{relax_n})={gates_r}')

        t0 = time.time()
        res_r = run_arm_e_relaxed_fanin(
            ir_r, x_all, y_all, x_all_t, y_all_t, in_width=in_width, args=args,
            device=device,
        )
        res_r['seconds'] = round(time.time() - t0, 2)
        print(f"  [seed={seed}] E_n{relax_n}(fan-in<={relax_n}, native "
              f"{args.relax_parameterization}): "
              f"soft {res_r['soft_acc_before']:.4f}->{res_r['soft_acc_after']:.4f} "
              f"hard {res_r['hard_acc_before']:.4f}->{res_r['hard_acc_after']:.4f} "
              f"({res_r['seconds']}s)")
        result[f'E_n{relax_n}'] = res_r

    # ---- (参照) warp: 構造転写+soft磨き(非native、本命の再利用) ----
    t0 = time.time()
    teacher_for_c = fresh_teacher()
    teacher_for_c, _ = ensure_teacher_fanin(
        teacher_for_c, x_all, y_all, n_inputs=args.fanin_warp,
        epochs_per_stage=args.teacher_prune_epochs, lr=args.teacher_lr,
        device=device,
    )
    ir_c, teacher_c_cpu = build_transcribed_ir(
        teacher_for_c, x_all, n_inputs=args.fanin_warp, weight_bits=args.weight_bits,
        device=device,
    )
    assert_transcribe_exact(ir_c, teacher_c_cpu, x_all, args.weight_bits)
    teacher_logits_c = make_teacher_logits(teacher_c_cpu, x_all_t.cpu(), 'cpu')
    gates_warp = native_gate_count(ir_c)
    res_warp = run_arm_c(
        ir_c, x_all, y_all, x_all, y_all, selection='soft', args=args,
        device=device, teacher_logits_tr=teacher_logits_c,
    )
    res_warp['seconds'] = round(time.time() - t0, 2)
    res_warp['gates_warp'] = gates_warp
    print(f"  [seed={seed}] (参照)warp(構造転写+soft磨き, 非native, "
          f"fanin={args.fanin_warp}): "
          f"hard {res_warp['hard_acc_before']:.4f}->{res_warp['hard_acc_after']:.4f} "
          f"gates={gates_warp} ({res_warp['seconds']}s)")
    result['warp_ref'] = {k: v for k, v in res_warp.items() if k != 'recovered_ir'}

    # ---- 派生指標 ----
    hard_a = res_a['hard_acc']
    hard_e = res_e['hard_acc_after']
    hard_edwn = res_edwn['hard_acc_after']
    hard_warp = res_warp['hard_acc_after']
    result['lift_E_vs_A'] = hard_e - hard_a
    result['lift_Edwn_vs_E'] = hard_edwn - hard_e
    result['gap_E_vs_warp'] = hard_e - hard_warp
    result['gap_Edwn_vs_warp'] = hard_edwn - hard_warp
    for relax_n in args.relax_n_inputs:
        hard_r = result[f'E_n{relax_n}']['hard_acc_after']
        result[f'lift_En{relax_n}_vs_E'] = hard_r - hard_e
        result[f'gap_En{relax_n}_vs_warp'] = hard_r - hard_warp
        result[f'gate_ratio_En{relax_n}_over_warp'] = (
            result[f'gates_E_n{relax_n}'] / max(gates_warp, 1)
        )

    return result


# ============================================================
# 集計・表示
# ============================================================

def _mean_std(vals: list[float]) -> tuple[float, float]:
    arr = np.asarray(vals, dtype=np.float64)
    return float(arr.mean()), float(arr.std())


def summarize(results: list[dict], args) -> dict:
    print('\n' + '=' * 104)
    print('fan-in≤2緩和(配線学習E_dwn / native fan-in拡大E_n4,n6) '
          'vs 固定配線native(E)/scratch(A)/warp参照 — 結果')
    print('=' * 104)

    arms = ['A', 'E', 'E_dwn'] + [f'E_n{n}' for n in args.relax_n_inputs] + ['warp_ref']
    labels = {
        'A': 'A: scratch native(下限)',
        'E': 'E: fan-in<=2固定配線native(基準)',
        'E_dwn': 'E_dwn: fan-in<=2学習配線native(配線緩和)',
        'warp_ref': '(参照)warp: 構造転写+soft磨き(非native、天井)',
    }
    for n in args.relax_n_inputs:
        labels[f'E_n{n}'] = f'E_n{n}: fan-in<={n} native {args.relax_parameterization}(fan-in緩和)'

    def get(arm, r, key):
        if arm == 'A':
            return r['A'][key if key != 'hard_acc_after' else 'hard_acc'] \
                if key != 'soft_acc_after' else r['A']['soft_acc']
        if arm == 'warp_ref':
            return r['warp_ref'][key]
        return r[arm][key]

    print('| arm | soft_acc(mean±std) | hard_acc(mean±std) |')
    print('|---|---|---|')
    summary: dict = {'arms': {}}
    for arm in arms:
        if arm == 'A':
            softs = [r['A']['soft_acc'] for r in results]
            hards = [r['A']['hard_acc'] for r in results]
        else:
            softs = [get(arm, r, 'soft_acc_after') for r in results]
            hards = [get(arm, r, 'hard_acc_after') for r in results]
        sm, ss = _mean_std(softs)
        hm, hs = _mean_std(hards)
        print(f'| {labels[arm]} | {sm:.4f}±{ss:.4f} | {hm:.4f}±{hs:.4f} |')
        summary['arms'][arm] = {
            'soft_acc_mean': sm, 'soft_acc_std': ss,
            'hard_acc_mean': hm, 'hard_acc_std': hs,
        }

    print('\n--- ゲート数(コンパクトさ) ---')
    gates_e = [float(r['gates_E']) for r in results]
    gates_warp = [float(r['warp_ref']['gates_warp']) for r in results]
    gem, ges = _mean_std(gates_e)
    gwm, gws = _mean_std(gates_warp)
    print(f'gates E(native fan-in<=2)      = {gem:.0f}±{ges:.0f}')
    print(f'gates warp(非native, 参照)      = {gwm:.0f}±{gws:.0f}')
    summary['gates_E_mean'] = gem
    summary['gates_warp_mean'] = gwm
    for n in args.relax_n_inputs:
        gates_r = [float(r[f'gates_E_n{n}']) for r in results]
        grm, grs = _mean_std(gates_r)
        ratio = grm / max(gwm, 1)
        print(f'gates E_n{n}(native fan-in<={n})  = {grm:.0f}±{grs:.0f}  '
              f'(/warp比={ratio:.4f})')
        summary[f'gates_E_n{n}_mean'] = grm
        summary[f'gate_ratio_En{n}_over_warp_mean'] = ratio

    print('\n--- fan-in別 oracle損失(teacher-before - teacher-after-prune) ---')
    fanin2_loss = [r['fanin2_oracle_loss'] for r in results]
    flm, fls = _mean_std(fanin2_loss)
    print(f'fan-in<=2: {flm:+.4f}±{fls:.4f}')
    summary['fanin2_oracle_loss_mean'] = flm
    for n in args.relax_n_inputs:
        loss_r = [r[f'fanin{n}_oracle_loss'] for r in results]
        lrm, lrs = _mean_std(loss_r)
        print(f'fan-in<={n}: {lrm:+.4f}±{lrs:.4f}')
        summary[f'fanin{n}_oracle_loss_mean'] = lrm

    print('\n--- 派生指標(mean±std over seeds) ---')
    lift_e_a = [r['lift_E_vs_A'] for r in results]
    lift_edwn_e = [r['lift_Edwn_vs_E'] for r in results]
    gap_e_warp = [r['gap_E_vs_warp'] for r in results]
    gap_edwn_warp = [r['gap_Edwn_vs_warp'] for r in results]
    leam, leas = _mean_std(lift_e_a)
    ldem, ldes = _mean_std(lift_edwn_e)
    gewm, gews = _mean_std(gap_e_warp)
    gdwm, gdws = _mean_std(gap_edwn_warp)
    print(f'lift(E - A)          : {leam:+.4f}±{leas:.4f}')
    print(f'lift(E_dwn - E)      : {ldem:+.4f}±{ldes:.4f}  (配線緩和の純効果)')
    print(f'gap(E - warp)        : {gewm:+.4f}±{gews:.4f}')
    print(f'gap(E_dwn - warp)    : {gdwm:+.4f}±{gdws:.4f}')
    summary['lift_E_vs_A_mean'] = leam
    summary['lift_Edwn_vs_E_mean'] = ldem
    summary['gap_E_vs_warp_mean'] = gewm
    summary['gap_Edwn_vs_warp_mean'] = gdwm

    for n in args.relax_n_inputs:
        lift_r = [r[f'lift_En{n}_vs_E'] for r in results]
        gap_r = [r[f'gap_En{n}_vs_warp'] for r in results]
        lrm, lrs = _mean_std(lift_r)
        grm, grs = _mean_std(gap_r)
        print(f'lift(E_n{n} - E)      : {lrm:+.4f}±{lrs:.4f}  (fan-in={n}緩和の純効果)')
        print(f'gap(E_n{n} - warp)    : {grm:+.4f}±{grs:.4f}')
        summary[f'lift_En{n}_vs_E_mean'] = lrm
        summary[f'gap_En{n}_vs_warp_mean'] = grm

    print('\n--- 仮説判定 ---')
    print(f'H(配線緩和E_dwnがE超え): lift(E_dwn-E)={ldem:+.4f} '
          f"→ {'支持' if ldem > 0 else '不支持'}")
    for n in args.relax_n_inputs:
        lrm = summary[f'lift_En{n}_vs_E_mean']
        print(f'H(fan-in緩和E_n{n}がE超え): lift(E_n{n}-E)={lrm:+.4f} '
              f"→ {'支持' if lrm > 0 else '不支持'}")
    best_arm, best_hard = max(
        [('E_dwn', ldem + summary['arms']['E']['hard_acc_mean'])]
        + [(f'E_n{n}', summary['arms'][f'E_n{n}']['hard_acc_mean'])
           for n in args.relax_n_inputs],
        key=lambda kv: kv[1],
    )
    print(f'\n最良の緩和アーム: {best_arm} (hard_acc≈{best_hard:.4f}) '
          f"vs warp天井={summary['arms']['warp_ref']['hard_acc_mean']:.4f}")
    summary['best_relax_arm'] = best_arm
    summary['best_relax_hard_acc'] = best_hard

    ge_90 = summary['arms']['warp_ref']['hard_acc_mean'] >= 0
    reaches_09 = {
        arm: summary['arms'][arm]['hard_acc_mean'] >= 0.9
        for arm in ('E_dwn',) + tuple(f'E_n{n}' for n in args.relax_n_inputs)
    }
    print(f"\n>=0.9 hard到達アーム: {[a for a, v in reaches_09.items() if v] or 'なし'}")
    summary['reaches_0.9'] = reaches_09

    exactness_all = all(
        r['E']['exactness_ok'] and r['E_dwn']['exactness_ok']
        and all(r[f'E_n{n}']['exactness_ok'] for n in args.relax_n_inputs)
        for r in results
    )
    print(f'\n全アーム厳密性assert(全seed): {"OK" if exactness_all else "NG"}')
    summary['exactness_ok'] = exactness_all

    return summary


# ============================================================
# メイン
# ============================================================

def _auto_device(arg: str | None) -> str:
    if arg is not None:
        return arg
    if torch.cuda.is_available():
        return 'cuda'
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


def _strip_ir(r: dict) -> dict:
    r2 = dict(r)
    if 'warp_ref' in r2 and isinstance(r2['warp_ref'], dict):
        r2['warp_ref'] = {k: v for k, v in r2['warp_ref'].items()
                          if k != 'recovered_ir'}
    return r2


def main():
    parser = argparse.ArgumentParser(
        description='fan-in≤2緩和(配線学習/native fan-in拡大) vs '
                    '固定配線native/scratch/warp参照、算術分類testbed')
    parser.add_argument('--task', type=str, default='adder',
                        choices=('adder', 'multiplier'))
    parser.add_argument('--nbits', type=int, default=4)
    parser.add_argument('--seeds', type=int, default=3)
    parser.add_argument('--epochs', type=int, default=200,
                        help='arm A(scratch native DLGN)の学習エポック数')
    parser.add_argument('--teacher-epochs', type=int, default=300)
    parser.add_argument('--teacher-prune-epochs', type=int, default=300)
    parser.add_argument('--teacher-lr', type=float, default=1e-2)
    parser.add_argument('--teacher-hidden', type=int, nargs='+', default=[64, 32])
    parser.add_argument('--lr', type=float, default=5e-2, help='arm AのAdam学習率')
    parser.add_argument('-bs', '--batch-size', type=int, default=64)
    parser.add_argument('--width', type=int, default=128,
                        help='arm A compact native DLGN隠れ層幅')
    parser.add_argument('--depth', type=int, default=3,
                        help='arm A compact native DLGN隠れ層数')
    parser.add_argument('--n-inputs', type=int, default=NATIVE_N_INPUTS,
                        choices=(NATIVE_N_INPUTS,),
                        help='arm A compact native DLGNのLUT fan-in(常に2固定)')
    parser.add_argument('--group-width', type=int, default=4)
    parser.add_argument('--weight-bits', type=int, default=8)
    parser.add_argument('--fanin-warp', type=int, default=6, choices=(2, 4, 6),
                        help='(参照)warp armのtranscribe fan-in(既定6)')
    parser.add_argument('--relax-n-inputs', type=int, nargs='+',
                        default=list(RELAX_N_INPUTS), choices=(4, 6),
                        help='fan-in緩和で試すnative n_inputs(既定[4, 6])')
    parser.add_argument('--relax-parameterization', type=str, default='independent',
                        choices=RECOVER_PARAMETERIZATIONS,
                        help="E_n4/E_n6のLUTパラメータ化(既定'independent'、"
                             '任意真理値表を厳密表現できnativeにコンパクトなまま'
                             '残る。warpは基底変換で表現力は同じだが本実験の狙いは'
                             "warpでない素のnativeなのでindependentを既定にする)")
    # arm E/E_dwn/E_n4/E_n6 磨き(arm E既定を踏襲、飽和初期化対策で
    # lr=1.0・epochs=60が必要だった実測に基づく)
    parser.add_argument('--polish-epochs', type=int, default=60)
    parser.add_argument('--polish-lr', type=float, default=1.0)
    # (参照)warp armのrecoverハイパラ(run_arm_c/RecoverConfigが読む)
    parser.add_argument('--recover-epochs', type=int, default=30)
    parser.add_argument('--recover-lr', type=float, default=0.05)
    parser.add_argument('--recover-tau', type=float, default=4.0)
    parser.add_argument('--polish-selection', type=str, default='soft',
                        choices=('soft', 'hard_st'))
    parser.add_argument('--aux', type=str, default='none', choices=('none', 'rkd'))
    parser.add_argument('--rkd-beta', type=float, default=1.0)
    parser.add_argument('--polish-kd', action='store_true',
                        help='(参照)warp armをKD磨きにする(既定false=CE磨き)')
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--temperature', type=float, default=4.0)
    parser.add_argument('--standardize', action='store_true', default=True)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--quick', action='store_true',
                        help='煙実行(1 seed・小width・少epoch、<120s目標)')
    parser.add_argument('--no-wandb', action='store_true')
    parser.add_argument('--out-dir', type=str,
                        default='outputs/arith_native_faninrelax')
    args = parser.parse_args()

    device = _auto_device(args.device)
    if args.no_wandb:
        os.environ['WANDB_MODE'] = 'disabled'

    if args.quick:
        args.nbits = min(args.nbits, 3)
        args.seeds = 1
        args.epochs = 60
        args.teacher_epochs = 150
        args.teacher_prune_epochs = 30
        args.teacher_hidden = [16]
        args.teacher_lr = 3e-2
        args.width = 32
        args.depth = 2
        args.group_width = 2
        args.polish_epochs = 10
        args.polish_lr = 1.0
        args.recover_epochs = 5
        args.batch_size = 32
        args.fanin_warp = 6
        args.relax_n_inputs = [4]  # 煙実行はn=4のみ(n=6は2^6=64エントリで重い)

    k_classes = num_classes_for(args.nbits, args.task)
    print(f'デバイス: {device}')
    print(f'task={args.task} nbits={args.nbits} (入力={2 * args.nbits}bit, '
          f'K={k_classes}クラス, n_patterns={1 << (2 * args.nbits)}) '
          f'seeds={args.seeds}')
    print(f'compact native DLGN(arm A/E系): n_inputs(E)={NATIVE_N_INPUTS} '
          f'relax_n_inputs={args.relax_n_inputs} '
          f'relax_parameterization={args.relax_parameterization} '
          f'width={args.width} depth={args.depth} group_width={args.group_width}')
    print(f'teacher: hidden={args.teacher_hidden} epochs={args.teacher_epochs} '
          f'lr={args.teacher_lr}')
    print(f'(参照)warp: fanin={args.fanin_warp} weight_bits={args.weight_bits}')

    use_wandb = init_wandb(
        run_name=f'arith_native_faninrelax_{args.task}_n{args.nbits}',
        config_dict=vars(args), device=device,
        tags=['D13', 'native-dlgn', 'fanin-relax', 'dwn-wiring',
              'n-inputs-relax', args.task],
        group='convert-arith-native-faninrelax',
    )

    started = time.time()
    results = []
    for seed in range(args.seeds):
        print(f'\n########## seed {seed + 1}/{args.seeds} ##########')
        r = run_one_seed(seed, args, device)
        results.append(r)
        if use_wandb:
            log_dict = {
                f'seed{seed}/teacher_acc_before_prune': r['teacher_acc_before_prune'],
                f'seed{seed}/fanin2_oracle_loss': r['fanin2_oracle_loss'],
                f'seed{seed}/A_hard_acc': r['A']['hard_acc'],
                f'seed{seed}/E_hard_acc': r['E']['hard_acc_after'],
                f'seed{seed}/Edwn_hard_acc': r['E_dwn']['hard_acc_after'],
                f'seed{seed}/warp_hard_acc': r['warp_ref']['hard_acc_after'],
                f'seed{seed}/gates_E': r['gates_E'],
                f'seed{seed}/gates_warp': r['warp_ref']['gates_warp'],
                f'seed{seed}/lift_E_vs_A': r['lift_E_vs_A'],
                f'seed{seed}/lift_Edwn_vs_E': r['lift_Edwn_vs_E'],
                f'seed{seed}/gap_E_vs_warp': r['gap_E_vs_warp'],
                f'seed{seed}/gap_Edwn_vs_warp': r['gap_Edwn_vs_warp'],
            }
            for n in args.relax_n_inputs:
                log_dict[f'seed{seed}/En{n}_hard_acc'] = r[f'E_n{n}']['hard_acc_after']
                log_dict[f'seed{seed}/gates_En{n}'] = r[f'gates_E_n{n}']
                log_dict[f'seed{seed}/lift_En{n}_vs_E'] = r[f'lift_En{n}_vs_E']
                log_dict[f'seed{seed}/gap_En{n}_vs_warp'] = r[f'gap_En{n}_vs_warp']
            log_metrics(log_dict)

    summary = summarize(results, args)
    if use_wandb:
        log_metrics({f'summary/{k}': v for k, v in summary.items()
                     if isinstance(v, int | float)})
        wandb_finish()

    os.makedirs(args.out_dir, exist_ok=True)
    ts = datetime.now(JST).strftime('%Y%m%d_%H%M%S')
    out_path = os.path.join(
        args.out_dir, f'result_{args.task}_n{args.nbits}_{ts}.json',
    )
    dump = {
        'args': vars(args), 'summary': summary,
        'per_seed': [_strip_ir(r) for r in results],
    }
    with open(out_path, 'w') as f:
        json.dump(dump, f, indent=2, default=str)
    print(f'\n結果を保存: {out_path}')

    total = str(timedelta(seconds=int(time.time() - started)))
    print(f'完了(総時間 {total})')


if __name__ == '__main__':
    main()
