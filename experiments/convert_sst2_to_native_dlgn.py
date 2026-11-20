"""コンパクトnative DLGN(difflogic方式)獲得 — SST-2感情分類でのwiring+
gate-init(構造転写)/KD/scratchのテキスト版検証(ROADMAP-v5 Track D、テキスト移植)。

`convert_arith_to_native_dlgn.py`(開発記録の採番で、算術分類testbed。
ファイル自体のdocstringは番号衝突により旧称"D12"を名乗るが、開発記録の
確定履歴に従いここでは呼ぶ)のアーム構成
(A scratch-native / B IO-KD-native / E init-native+soft磨き / E+KD)を、
**そのままの機構で**テキスト分類(SST-2二値感情分類)へ移植する。改変対象は
「データセット+teacher」のみで、native DLGN構築・rebuild・polish・KDの
機構はから無改造でimportして再利用する(taskの明示指示)。

## なぜbinary-BoW STE-MLP teacherを新設するか

`transcribe_mlp`はnn.Linear列+0/1入力を前提とする(教師の隠れ層
意味論 y=1[Wx+b>0] が転写の唯一のground truth)。既存のSST-2 teacher
(train_kd_sst2.pyのDistilBERT等)は連続値埋め込み+Attentionで、この前提を
満たさない。そこで「二値bag-of-words入力→STE-MLP」という、transcribe_mlp
が扱える形の弱いteacherをゼロから用意する(taskの明示指示: 新規構築)。
弱いteacher=scratch native DLGNとの差が小さくなり得るが、これは
model-conversion-kdスキル§0の「teacher差が小さい領域では変換価値ゼロ」
という予測を検証する側の条件でもある(恣意的に有利な設定を選ばない)。

## アーム(すべてnative DLGN、`hard_acc_via_export`/`simulate`でhard acc測定)

- **A(scratch)**: `LutLayer(n_inputs=2, softmax)`列+GroupSumをラベルCEで
  学習(`convert_arith_structural_transfer.build_dlgn`/`run_arm_a`を
  n_inputs=2で再利用)。下限対照。
- **B(IO-KDのみ)**: 同じcompact native archをteacher soft-logit KD+CEで
  学習(`run_arm_b`を再利用)。「KD単体、構造転写なし」を測る。
- **E(本命: wiring+gate-init→soft磨き)**: binary-BoW STE teacherを
  `gentle_fanin_schedule`でfan-in=2まで枝刈り→`transcribe_mlp(n_inputs=2)`
  で厳密転写→`rebuild_as_native16g`(から無改造import)でnative
  softmax16g DLGNへ復元。**復元直後がtranscribe直後のIRとbit-exactで
  あることをassert**(理論上必ず成立するはず、破れたら停止)。その後
  soft磨き(低LR CE fine-tune)。
- **E+KD**: Eの初期化から、磨きをKD(CE+KL混合)で行う。
- **(F層別蒸留は対象外、taskで明示された必須アームのみを実装。オプション
  扱いのFはコスト対効果が低いため見送り、その旨をVERDICTに明記する)**

## 使用例

    # 煙実行(数十秒、wandb無効、1 seed・小vocab・少epoch)
    uv run python experiments/convert_sst2_to_native_dlgn.py --quick --no-wandb

    # 本実行(既定vocab-size=256、3 seeds)
    uv run python experiments/convert_sst2_to_native_dlgn.py --seeds 3 --no-wandb
"""

from __future__ import annotations

import argparse
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
sys.path.insert(0, os.path.dirname(__file__))

# の機構を無改造でimport(taskの明示指示: convert_arith_to_native_dlgn.py/
# convert_arith_structural_transfer.py を改変せず、importして再利用する)
from convert_arith_structural_transfer import (
    assert_transcribe_exact,
    build_dlgn,
    ensure_teacher_fanin,
    hard_acc_via_export,
    make_teacher_logits,
    soft_hard_acc_dlgn,
)
from convert_arith_to_native_dlgn import (
    assert_native16g_bit_exact,
    native_gate_count,
    polish_native,
    rebuild_as_native16g,
)
from data.binary_bow import build_bow_vocab, texts_to_binary_bow
from data.sst2 import download_and_tokenize
from env_utils import load_dotenv
from torch import nn
from wandb_utils import finish as wandb_finish
from wandb_utils import init_wandb, log_metrics

from griffith_logic.convert.distill import make_kd_loss
from griffith_logic.convert.recover import _hard_accuracy
from griffith_logic.convert.teacher_models import HeavisideSTE
from griffith_logic.convert.transcribe import TranscribeSpec, transcribe_mlp
from griffith_logic.ir.validate import validate

load_dotenv()

EPS = 1e-8
NATIVE_N_INPUTS = 2  # compact native DLGN(difflogic方式)はN=2専用
N_CLASSES = 2  # SST-2: 0=negative, 1=positive


# ============================================================
# データ: binary-BoW SST-2(V語彙、{0,1}^V特徴量)
# ============================================================

def load_binary_bow_sst2(
    vocab_size: int, seq_len: int, data_dir: str, train_limit: int | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """SST-2を生テキストからロードし、binary-BoW特徴量へ変換する。

    語彙はWordTokenizer由来のトークンID化データセット(`download_and_tokenize`)
    の`.texts`から構築する(既存のSST-2ダウンロード/キャッシュ機構をそのまま
    再利用し、テキストのトークンID化自体は使わずtextsのみ流用する)。

    Returns:
        (x_train, y_train, x_val, y_val, vocab): x_*は{0,1}^V(int64)、
        y_*はint64ラベル、vocabは長さVの語彙リスト。
    """
    train_set, val_set, _tok = download_and_tokenize(
        vocab_size=8000, seq_len=seq_len, data_dir=data_dir,
        train_limit=train_limit,
    )
    vocab = build_bow_vocab(train_set.texts, vocab_size=vocab_size)
    x_train = texts_to_binary_bow(train_set.texts, vocab)
    x_val = texts_to_binary_bow(val_set.texts, vocab)
    y_train = train_set.labels.numpy().astype(np.int64)
    y_val = val_set.labels.numpy().astype(np.int64)
    return x_train, y_train, x_val, y_val, vocab


# ============================================================
# teacher: binary-BoW STE-MLP(構造転写元、arm E/E+KD)
# ============================================================

class BowTeacherSte(nn.Module):
    """binary-BoW入力の小さなSTE隠れ活性MLP(transcribe_mlp対応)。

    `ArithTeacherMlp`(convert_arith_structural_transfer.py)と同一設計
    (Linear列をself.linearsに登録、隠れ活性y=1[Wx+b>0]、最終層は非二値化)を、
    入力がbinary-BoW(0/1、幅V)である点のみ変えてテキスト用に複製する。
    convert_arith_structural_transfer.pyは改変しない(task指示)ため、
    ここに専用クラスとして定義する。
    """

    def __init__(self, in_width: int, hidden: tuple[int, ...], num_classes: int) -> None:
        super().__init__()
        self.linears = nn.ModuleList()
        prev = in_width
        for h in hidden:
            self.linears.append(nn.Linear(prev, h))
            prev = h
        self.linears.append(nn.Linear(prev, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x.float()
        for lin in self.linears[:-1]:
            h = HeavisideSTE.apply(lin(h))
        return self.linears[-1](h)


def train_teacher_bow(
    teacher: nn.Module, x_tr: torch.Tensor, y_tr: torch.Tensor,
    x_val: torch.Tensor, y_val: torch.Tensor, *,
    epochs: int, lr: float, device: str, batch_size: int,
) -> tuple[float, float]:
    """binary-BoW teacherを訓練する。(train hard_acc, val hard_acc)を返す。"""
    teacher.to(device).train()
    opt = torch.optim.Adam(teacher.parameters(), lr=lr)
    loader = DataLoader(TensorDataset(x_tr, y_tr), batch_size=batch_size, shuffle=True)
    for _ in range(epochs):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            out = teacher(xb.float())
            loss = F.cross_entropy(out, yb)
            loss.backward()
            opt.step()
    teacher.eval()
    with torch.no_grad():
        pred_tr = teacher(x_tr.to(device).float()).argmax(-1).cpu()
        pred_val = teacher(x_val.to(device).float()).argmax(-1).cpu()
    train_acc = float((pred_tr == y_tr).float().mean())
    val_acc = float((pred_val == y_val).float().mean())
    return train_acc, val_acc


# ============================================================
# arm A/B: run_arm_a/run_arm_b の薄い再実装(soft_acc/hard_accをval評価する版)
# ============================================================
# convert_arith_structural_transfer.run_arm_a/run_arm_bは学習データそのものを
# eval対象にする(算術=全パターン列挙で汎化を測らない設計)。テキストは
# train/valが分かれるため、学習ループ自体はbuild_dlgn+同じ最適化コードを
# 再利用しつつ、評価だけ独立したval集合に対して行う専用版を用意する
# (run_arm_a/bは改変しない: task指示によりconvert_arith_structural_transfer.py
# は無改造)。

def run_arm_a_sst2(x_tr, y_tr, x_val_np, y_val_np, *, in_width, args, device, seed):
    model = build_dlgn(
        in_width, N_CLASSES, width=args.width, depth=args.depth,
        n_inputs=NATIVE_N_INPUTS, group_width=args.group_width,
        device=device, parameterization='softmax', seed=seed,
    )
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loader = DataLoader(TensorDataset(x_tr, y_tr), batch_size=args.batch_size, shuffle=True)
    model.train()
    for _ in range(args.epochs):
        for xb, yb in loader:
            xb, yb = xb.to(device).float(), yb.to(device)
            opt.zero_grad()
            loss = F.cross_entropy(model(xb), yb)
            loss.backward()
            opt.step()
    soft_acc = soft_hard_acc_dlgn(model, x_val_np_t(x_val_np), y_val_np_t(y_val_np), device)
    hard_acc = hard_acc_via_export(model, x_val_np, y_val_np, in_width)
    return {'soft_acc': soft_acc, 'hard_acc': hard_acc, 'gap': soft_acc - hard_acc}


def run_arm_b_sst2(x_tr, y_tr, teacher_logits_tr, x_val_np, y_val_np, *,
                   in_width, args, device, seed):
    model = build_dlgn(
        in_width, N_CLASSES, width=args.width, depth=args.depth,
        n_inputs=NATIVE_N_INPUTS, group_width=args.group_width,
        device=device, parameterization='softmax', seed=seed,
    )
    loss_fn = make_kd_loss(
        alpha=args.alpha, temperature=args.temperature,
        standardize=args.standardize, kind='kl',
    )
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loader = DataLoader(
        TensorDataset(x_tr, y_tr, teacher_logits_tr), batch_size=args.batch_size,
        shuffle=True,
    )
    model.train()
    for _ in range(args.epochs):
        for xb, yb, tb in loader:
            xb, yb, tb = xb.to(device).float(), yb.to(device), tb.to(device)
            opt.zero_grad()
            out = model(xb)
            loss = loss_fn(out, yb, tb)
            loss.backward()
            opt.step()
    soft_acc = soft_hard_acc_dlgn(model, x_val_np_t(x_val_np), y_val_np_t(y_val_np), device)
    hard_acc = hard_acc_via_export(model, x_val_np, y_val_np, in_width)
    return {'soft_acc': soft_acc, 'hard_acc': hard_acc, 'gap': soft_acc - hard_acc}


def x_val_np_t(x_np: np.ndarray) -> torch.Tensor:
    return torch.as_tensor(x_np, dtype=torch.float32)


def y_val_np_t(y_np: np.ndarray) -> torch.Tensor:
    return torch.as_tensor(y_np, dtype=torch.int64)


# ============================================================
# arm E/E+KD: wiring+gate-init(native16g) → soft磨き(→ KD磨き)、val評価版
# ============================================================

def run_arm_e_sst2(
    native_ir, x_tr, y_tr, x_val_np, y_val_np, *, in_width, args, device,
    teacher_logits_tr: torch.Tensor | None = None,
) -> dict:
    """arm E(wiring+gate-init→soft磨き)またはE+KD(teacher_logits_tr指定時)。

    polish_native/rebuild_as_native16gをそのまま呼ぶが、
    train集合そのものをeval対象にする(算術=全パターン)ため、ここではvalへ
    分離評価するようpolish_nativeの引数(x_ev_np, y_ev_np)にval集合を渡す
    薄いラッパーとして実装する(polish_native自体は無改造)。
    """
    model = rebuild_as_native16g(native_ir)
    # bit-exact assertは較正データ(=train入力)で行う(転写・復元はtrain上で
    # 定義された関数の保存を主張するものであり、val入力での一致は無意味な
    # 主張ではないがtranscribe仕様上train較正データが規約通りの対象)
    calib_x = x_tr.numpy().astype(np.int64)
    assert_native16g_bit_exact(native_ir, model, calib_x)
    res = polish_native(
        model, x_tr, y_tr, x_val_np, y_val_np, in_width=in_width,
        device=device, epochs=args.polish_epochs, lr=args.polish_lr,
        batch_size=args.batch_size,
        teacher_logits_tr=teacher_logits_tr,
        kd_alpha=args.alpha, kd_temperature=args.temperature,
        kd_standardize=args.standardize,
    )
    res['exactness_ok'] = True
    return res


# ============================================================
# 1 seedの全アーム実行
# ============================================================

def run_one_seed(seed: int, args, device: str) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)

    x_train_np, y_train_np, x_val_np, y_val_np, vocab = load_binary_bow_sst2(
        args.vocab_size, args.seq_len, args.data_dir, args.train_limit,
    )
    v = len(vocab)
    x_tr_t = torch.as_tensor(x_train_np, dtype=torch.float32)
    y_tr_t = torch.as_tensor(y_train_np, dtype=torch.int64)
    x_val_t = torch.as_tensor(x_val_np, dtype=torch.float32)
    y_val_t = torch.as_tensor(y_val_np, dtype=torch.int64)

    print(f'  [seed={seed}] vocab_size={v} n_train={len(x_train_np)} '
          f'n_val={len(x_val_np)} pos_rate_train={y_train_np.mean():.4f} '
          f'pos_rate_val={y_val_np.mean():.4f}')

    result: dict = {'seed': seed, 'vocab_size': v,
                    'n_train': len(x_train_np), 'n_val': len(x_val_np)}

    # ---- teacher学習(binary-BoW STE-MLP) ----
    torch.manual_seed(seed)
    teacher = BowTeacherSte(v, tuple(args.teacher_hidden), N_CLASSES).to(device)
    teacher_train_acc, teacher_val_acc = train_teacher_bow(
        teacher, x_tr_t, y_tr_t, x_val_t, y_val_t,
        epochs=args.teacher_epochs, lr=args.teacher_lr, device=device,
        batch_size=args.batch_size,
    )
    print(f'  [seed={seed}] teacher(枝刈り前) train_acc={teacher_train_acc:.4f} '
          f'val_acc={teacher_val_acc:.4f}')
    result['teacher_train_acc_before_prune'] = teacher_train_acc
    result['teacher_val_acc_before_prune'] = teacher_val_acc

    # ---- fan-in=2まで枝刈り(compact native DLGNはN=2専用) ----
    teacher, prune_history = ensure_teacher_fanin(
        teacher, x_train_np, y_train_np, n_inputs=NATIVE_N_INPUTS,
        epochs_per_stage=args.teacher_prune_epochs, lr=args.teacher_lr,
        device=device,
    )
    teacher.to(device).eval()
    with torch.no_grad():
        teacher_train_acc_after = float(
            (teacher(x_tr_t.to(device)).argmax(-1).cpu() == y_tr_t).float().mean()
        )
        teacher_val_acc_after = float(
            (teacher(x_val_t.to(device)).argmax(-1).cpu() == y_val_t).float().mean()
        )
    print(f'  [seed={seed}] teacher fan-in=2枝刈り後 train_acc='
          f'{teacher_train_acc_after:.4f} val_acc={teacher_val_acc_after:.4f}')
    result['teacher_train_acc_after_prune'] = teacher_train_acc_after
    result['teacher_val_acc_after_prune'] = teacher_val_acc_after
    result['fanin2_oracle_loss'] = teacher_train_acc - teacher_train_acc_after
    result['prune_stages'] = prune_history.get('stages', [])

    # ---- transcribe(n_inputs=2、training-free) + 厳密性assert ----
    teacher_cpu = teacher.to('cpu').eval()
    spec = TranscribeSpec(n_inputs=NATIVE_N_INPUTS, weight_bits=args.weight_bits, isf=False)
    ir2 = transcribe_mlp(teacher_cpu, x_train_np, spec)
    errs = validate(ir2)
    if errs:
        raise RuntimeError(f'転写IR検証失敗: {errs[:3]}')
    assert_transcribe_exact(ir2, teacher_cpu, x_train_np, args.weight_bits)
    print(f'  [seed={seed}] fan-in=2転写 厳密性assert: OK')

    transcribe_train_hard_acc = _hard_accuracy(ir2, x_train_np, y_train_np)
    transcribe_val_hard_acc = _hard_accuracy(ir2, x_val_np, y_val_np)
    result['transcribe_train_hard_acc'] = transcribe_train_hard_acc
    result['transcribe_val_hard_acc'] = transcribe_val_hard_acc
    gates_e = native_gate_count(ir2)
    result['gates_E'] = gates_e
    print(f'  [seed={seed}] 転写直後(磨き前, oracle参照) '
          f'train_hard_acc={transcribe_train_hard_acc:.4f} '
          f'val_hard_acc={transcribe_val_hard_acc:.4f} gates(native E)={gates_e}')

    # teacherロジット(arm B/E+KD用。fan-in枝刈り後teacherのロジット)
    teacher_logits_tr = make_teacher_logits(teacher_cpu, x_tr_t.cpu(), 'cpu')

    in_width = v

    # ---- arm A: scratch(compact native, n_inputs=2) ----
    t0 = time.time()
    res_a = run_arm_a_sst2(
        x_tr_t, y_tr_t, x_val_np, y_val_np, in_width=in_width, args=args,
        device=device, seed=seed + 1000,
    )
    res_a['seconds'] = round(time.time() - t0, 2)
    print(f"  [seed={seed}] A(scratch native): soft={res_a['soft_acc']:.4f} "
          f"hard={res_a['hard_acc']:.4f} ({res_a['seconds']}s)")
    result['A'] = res_a

    # ---- arm B: IO-KDのみ(compact native, n_inputs=2) ----
    t0 = time.time()
    res_b = run_arm_b_sst2(
        x_tr_t, y_tr_t, teacher_logits_tr, x_val_np, y_val_np,
        in_width=in_width, args=args, device=device, seed=seed + 2000,
    )
    res_b['seconds'] = round(time.time() - t0, 2)
    print(f"  [seed={seed}] B(IO-KD native): soft={res_b['soft_acc']:.4f} "
          f"hard={res_b['hard_acc']:.4f} ({res_b['seconds']}s)")
    result['B'] = res_b

    # ---- arm E: wiring+gate-init(native16g) → soft磨き ----
    t0 = time.time()
    res_e = run_arm_e_sst2(
        ir2, x_tr_t, y_tr_t, x_val_np, y_val_np, in_width=in_width, args=args,
        device=device, teacher_logits_tr=None,
    )
    res_e['seconds'] = round(time.time() - t0, 2)
    print(f"  [seed={seed}] E(init native+soft磨き, 本命): "
          f"soft {res_e['soft_acc_before']:.4f}->{res_e['soft_acc_after']:.4f} "
          f"hard {res_e['hard_acc_before']:.4f}->{res_e['hard_acc_after']:.4f} "
          f"({res_e['seconds']}s)")
    result['E'] = res_e

    # ---- arm E+KD: 同じ初期化からKD磨き ----
    t0 = time.time()
    res_ekd = run_arm_e_sst2(
        ir2, x_tr_t, y_tr_t, x_val_np, y_val_np, in_width=in_width, args=args,
        device=device, teacher_logits_tr=teacher_logits_tr,
    )
    res_ekd['seconds'] = round(time.time() - t0, 2)
    print(f"  [seed={seed}] E+KD(init native+KD磨き): "
          f"soft {res_ekd['soft_acc_before']:.4f}->{res_ekd['soft_acc_after']:.4f} "
          f"hard {res_ekd['hard_acc_before']:.4f}->{res_ekd['hard_acc_after']:.4f} "
          f"({res_ekd['seconds']}s)")
    result['E_KD'] = res_ekd

    # ---- 派生指標 ----
    hard_a = res_a['hard_acc']
    hard_b = res_b['hard_acc']
    hard_e = res_e['hard_acc_after']
    hard_ekd = res_ekd['hard_acc_after']
    result['lift_E_vs_A'] = hard_e - hard_a
    result['lift_E_vs_B'] = hard_e - hard_b
    result['lift_EKD_vs_E'] = hard_ekd - hard_e
    result['teacher_recovery_E'] = hard_e / max(teacher_val_acc_after, EPS)

    return result


# ============================================================
# 集計・表示
# ============================================================

def _mean_std(vals: list[float]) -> tuple[float, float]:
    arr = np.asarray(vals, dtype=np.float64)
    return float(arr.mean()), float(arr.std())


def summarize(results: list[dict], args) -> dict:
    print('\n' + '=' * 100)
    print('SST-2 binary-BoW native DLGN獲得 — wiring+gate-init(E)/IO-KD(B)/'
          'init+KD(E+KD) vs scratch(A) — 結果')
    print('=' * 100)

    def get_hard(arm, r):
        if arm == 'A':
            return r['A']['hard_acc']
        if arm == 'B':
            return r['B']['hard_acc']
        if arm == 'E':
            return r['E']['hard_acc_after']
        if arm == 'E_KD':
            return r['E_KD']['hard_acc_after']
        raise ValueError(arm)

    def get_soft(arm, r):
        if arm == 'A':
            return r['A']['soft_acc']
        if arm == 'B':
            return r['B']['soft_acc']
        if arm == 'E':
            return r['E']['soft_acc_after']
        if arm == 'E_KD':
            return r['E_KD']['soft_acc_after']
        raise ValueError(arm)

    arms = ['A', 'B', 'E', 'E_KD']
    labels = {
        'A': 'A: scratch native',
        'B': 'B: IO-KDのみ native',
        'E': 'E: wiring+gate-init native + soft磨き(本命)',
        'E_KD': 'E+KD: wiring+gate-init native + KD磨き',
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

    teacher_val_accs = [r['teacher_val_acc_after_prune'] for r in results]
    tm, ts = _mean_std(teacher_val_accs)
    gates_e = [float(r['gates_E']) for r in results]
    gem, ges = _mean_std(gates_e)
    print(f'\nteacher(fan-in=2枝刈り後) val_acc={tm:.4f}±{ts:.4f}')
    print(f'native Eゲート数={gem:.0f}±{ges:.0f}')
    summary['teacher_val_acc_mean'] = tm
    summary['gates_E_mean'] = gem

    fanin2_loss = [r['fanin2_oracle_loss'] for r in results]
    flm, fls = _mean_std(fanin2_loss)
    print(f'\nfan-in≤2 oracle損失(teacher_train_before - teacher_train_after_prune): '
          f'{flm:+.4f}±{fls:.4f}')
    summary['fanin2_oracle_loss_mean'] = flm
    summary['fanin2_oracle_loss_std'] = fls

    lift_e_a = [r['lift_E_vs_A'] for r in results]
    lift_e_b = [r['lift_E_vs_B'] for r in results]
    lift_ekd_e = [r['lift_EKD_vs_E'] for r in results]
    recov_e = [r['teacher_recovery_E'] for r in results]

    leam, leas = _mean_std(lift_e_a)
    lebm, lebs = _mean_std(lift_e_b)
    lkem, lkes = _mean_std(lift_ekd_e)
    rem, res_ = _mean_std(recov_e)

    print('\n--- 派生指標(mean±std over seeds) ---')
    print(f'lift(E - A)          : {leam:+.4f}±{leas:.4f}')
    print(f'lift(E - B)          : {lebm:+.4f}±{lebs:.4f}')
    print(f'lift(E+KD - E)       : {lkem:+.4f}±{lkes:.4f}')
    print(f'回復率(E hard / teacher val_acc): {rem:.4f}±{res_:.4f}')

    summary['lift_E_vs_A_mean'] = leam
    summary['lift_E_vs_B_mean'] = lebm
    summary['lift_EKD_vs_E_mean'] = lkem
    summary['teacher_recovery_E_mean'] = rem

    print('\n--- 仮説判定 ---')
    print(f'H(wiring+gate-init(E) > scratch(A)): lift(E-A)={leam:+.4f} '
          f"→ {'支持' if leam > 0 else '不支持'}")
    print(f'H(wiring+gate-init(E) > IO-KD(B)): lift(E-B)={lebm:+.4f} '
          f"→ {'支持' if lebm > 0 else '不支持'}")
    print(f'H(init+KD(E+KD) > init単独(E)): lift(E+KD-E)={lkem:+.4f} '
          f"→ {'支持' if lkem > 0 else '不支持'}")

    exactness_all = all(r['E']['exactness_ok'] for r in results)
    print(f'\nnative16g復元bit-exact厳密性assert(全seed): '
          f'{"OK" if exactness_all else "NG"}')
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


def main():
    parser = argparse.ArgumentParser(
        description='SST-2 binary-BoW native DLGN獲得(wiring+gate-init/KD) '
                    'vs scratch')
    parser.add_argument('--vocab-size', type=int, default=256,
                        help='binary-BoW語彙サイズV(既定256)')
    parser.add_argument('--seq-len', type=int, default=64,
                        help='download_and_tokenizeの内部seq_len'
                             '(binary-BoW自体は使わないが既存キャッシュ機構が要求)')
    parser.add_argument('--data-dir', type=str, default='./data-sst2')
    parser.add_argument('--train-limit', type=int, default=None,
                        help='訓練データを先頭N件に決定的サブサンプル(デバッグ用)')
    parser.add_argument('--seeds', type=int, default=3)
    parser.add_argument('--epochs', type=int, default=200,
                        help='arm A/B(scratch native DLGN)の学習エポック数')
    parser.add_argument('--teacher-epochs', type=int, default=100)
    parser.add_argument('--teacher-prune-epochs', type=int, default=100)
    parser.add_argument('--teacher-lr', type=float, default=1e-2)
    parser.add_argument('--teacher-hidden', type=int, nargs='+', default=[64, 32])
    parser.add_argument('--lr', type=float, default=5e-2, help='arm A/BのAdam学習率')
    parser.add_argument('-bs', '--batch-size', type=int, default=64)
    parser.add_argument('--width', type=int, default=128,
                        help='arm A/B compact native DLGN隠れ層幅')
    parser.add_argument('--depth', type=int, default=3,
                        help='arm A/B compact native DLGN隠れ層数')
    parser.add_argument('--group-width', type=int, default=4)
    parser.add_argument('--weight-bits', type=int, default=8)
    # arm E/E+KD 磨き(同じ既定値: native16g復元直後の飽和初期化に対し
    # lr=0.05だと事実上no-opになるため、lr>=0.5相当が必要という実測知見を踏襲)
    parser.add_argument('--polish-epochs', type=int, default=60)
    parser.add_argument('--polish-lr', type=float, default=1.0)
    # KD共通(arm B/E+KD)
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--temperature', type=float, default=2.0,
                        help='KD温度T(既定2.0。model-conversion-kdスキル§8.3-1: '
                             '二値分類C=2はT=2がT=4を圧倒するテキスト実測知見)')
    parser.add_argument('--standardize', action='store_true', default=True)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--quick', action='store_true',
                        help='煙実行(1 seed・小vocab・少epoch・train-limit、'
                             '<120s目標)')
    parser.add_argument('--no-wandb', action='store_true')
    parser.add_argument('--out-dir', type=str,
                        default='outputs/sst2_to_native_dlgn')
    args = parser.parse_args()

    device = _auto_device(args.device)
    if args.no_wandb:
        os.environ['WANDB_MODE'] = 'disabled'

    if args.quick:
        args.vocab_size = 64
        args.train_limit = 2000
        args.seeds = 1
        args.epochs = 40
        args.teacher_epochs = 30
        args.teacher_prune_epochs = 20
        args.teacher_hidden = [16]
        args.teacher_lr = 3e-2
        args.width = 32
        args.depth = 2
        args.group_width = 2
        args.polish_epochs = 10
        args.polish_lr = 1.0
        args.batch_size = 32

    print(f'デバイス: {device}')
    print(f'vocab_size={args.vocab_size} seq_len={args.seq_len} '
          f'train_limit={args.train_limit} seeds={args.seeds}')
    print(f'compact native DLGN(arm A/B/E): n_inputs={NATIVE_N_INPUTS} '
          f'width={args.width} depth={args.depth} group_width={args.group_width}')
    print(f'teacher(binary-BoW STE-MLP): hidden={args.teacher_hidden} '
          f'epochs={args.teacher_epochs} lr={args.teacher_lr}')
    print(f'KD: alpha={args.alpha} T={args.temperature} standardize={args.standardize}')

    use_wandb = init_wandb(
        run_name=f'sst2_to_native_dlgn_v{args.vocab_size}',
        config_dict=vars(args), device=device,
        tags=['D13', 'native-dlgn', 'structural-transfer', 'kd', 'sst2', 'text'],
        group='convert-sst2-to-native-dlgn',
    )

    started = time.time()
    results = []
    for seed in range(args.seeds):
        print(f'\n########## seed {seed + 1}/{args.seeds} ##########')
        r = run_one_seed(seed, args, device)
        results.append(r)
        if use_wandb:
            log_metrics({
                f'seed{seed}/teacher_val_acc_before_prune': r['teacher_val_acc_before_prune'],
                f'seed{seed}/teacher_val_acc_after_prune': r['teacher_val_acc_after_prune'],
                f'seed{seed}/fanin2_oracle_loss': r['fanin2_oracle_loss'],
                f'seed{seed}/A_hard_acc': r['A']['hard_acc'],
                f'seed{seed}/B_hard_acc': r['B']['hard_acc'],
                f'seed{seed}/E_hard_acc': r['E']['hard_acc_after'],
                f'seed{seed}/EKD_hard_acc': r['E_KD']['hard_acc_after'],
                f'seed{seed}/gates_E': r['gates_E'],
                f'seed{seed}/lift_E_vs_A': r['lift_E_vs_A'],
                f'seed{seed}/lift_E_vs_B': r['lift_E_vs_B'],
                f'seed{seed}/lift_EKD_vs_E': r['lift_EKD_vs_E'],
                f'seed{seed}/teacher_recovery_E': r['teacher_recovery_E'],
            })

    summary = summarize(results, args)
    if use_wandb:
        log_metrics({f'summary/{k}': v for k, v in summary.items()
                     if isinstance(v, int | float)})
        wandb_finish()

    os.makedirs(args.out_dir, exist_ok=True)
    ts = datetime.now(JST).strftime('%Y%m%d_%H%M%S')
    out_path = os.path.join(args.out_dir, f'result_v{args.vocab_size}_{ts}.json')
    import json
    dump = {'args': vars(args), 'summary': summary, 'per_seed': results}
    with open(out_path, 'w') as f:
        json.dump(dump, f, indent=2, default=str)
    print(f'\n結果を保存: {out_path}')

    total = str(timedelta(seconds=int(time.time() - started)))
    print(f'完了(総時間 {total})')


if __name__ == '__main__':
    main()
