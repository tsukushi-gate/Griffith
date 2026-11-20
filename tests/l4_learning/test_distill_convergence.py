"""L4: 知識蒸留(KD)の学習収束+runner統合スモーク。

合成・数秒。実データセットのDLは行わない(teacherは解析重みの凍結MLP、
runnerスモークはload_raw/resolve_teacherをmonkeypatchして合成データで回す)。

検証:
  1. 解析重みXOR凍結teacher + α=0(純KD、ラベル不使用)→ DLGN student hard acc=100%
  2. α=0.5 + parity3 でも hard acc=100%
  3. runner統合スモーク: DistillConfig付きTrainConfigで run_training が完走し
     hard_acc が記録される(合成データ、wandb無効)
  4. キャッシュ: 同一keyで2回目は teacher.predict_logits が呼ばれない、
     params変更でkey変化、PackedDistillDataset の形状/dtype契約
  5. 既存backcompat(distill省略=従来動作)は test_training_runner.py / test_backcompat.py 側
"""

import itertools
import os
import sys

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from griffith_logic import GroupSum, LutLayer
from griffith_logic.convert.cache import (
    PackedDistillDataset,
    get_or_compute_logits,
    teacher_cache_key,
)
from griffith_logic.convert.distill import make_kd_loss
from griffith_logic.convert.teacher import TorchTeacher

# L4=学習収束テストは全てlearning(CI除外、ローカルmake testでは実行)
pytestmark = pytest.mark.learning

# ======================================================================
# 解析重み凍結teacher(XOR/parityを厳密に計算する2層float MLP)
# ======================================================================

class _ParityMLP(nn.Module):
    """nビットパリティを厳密に2クラスロジットで出す解析重みMLP(学習しない)。

    hidden_k = Σ ReLU(Σ x - k + 0.5)  (k=1..n) は「立っているビット数」を
    数える(Σx>=k のとき 1)。パリティ = Σ (-1)^(k+1) count_k の偶奇。
    ここでは count(=Σx) から parity を作る決定的な構成を使う:
        s = Σ_{k odd} relu(Σx - k + .5) - Σ_{k even} relu(Σx - k + .5) の差分列で
    偶奇を表現し、クラス0(even)/クラス1(odd)ロジットへ写す。
    """

    def __init__(self, n: int, scale: float = 12.0):
        super().__init__()
        self.n = n
        self.scale = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, n] in {0,1}
        count = x.sum(dim=-1)  # [B]
        # 立っているビット数の偶奇(parity)。整数countの偶奇を鋭いロジットで表現。
        parity = (count.round().long() % 2)  # [B] 0/1
        logit_odd = (parity.float() * 2.0 - 1.0) * self.scale  # +scale if odd else -scale
        logits = torch.stack([-logit_odd, logit_odd], dim=-1)  # [B, 2]
        return logits


def _bits(n: int) -> torch.Tensor:
    return torch.tensor(list(itertools.product([0.0, 1.0], repeat=n)))


def _parity_labels(x: torch.Tensor) -> torch.Tensor:
    return (x.sum(dim=-1).long() % 2)


def _make_teacher(n: int) -> TorchTeacher:
    return TorchTeacher(
        _ParityMLP(n), num_classes=2, input_size=(n,),
        fingerprint=f'parity{n}-analytic',
    )


def _dlgn_student(in_dim: int, k: int = 24, layers: int = 2, seed: int = 0):
    torch.manual_seed(seed)
    mods: list[nn.Module] = []
    d = in_dim
    for _ in range(layers):
        mods.append(LutLayer(d, k, n_inputs=2, parameterization='softmax'))
        d = k
    mods.append(GroupSum(k=2, tau=max(1.0, (k // 2) ** 0.5)))
    return nn.Sequential(*mods)


def _train_kd(model, x, labels, teacher_logits, loss_fn, *, lr=0.05,
              max_steps=1500, check_every=50):
    """KD損失で soft訓練し、hard eval(入力round+weight argmax)で100%到達で早期終了。"""
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for step in range(max_steps):
        model.train()
        opt.zero_grad()
        loss = loss_fn(model(x), labels, teacher_logits)
        loss.backward()
        opt.step()
        if (step + 1) % check_every == 0:
            acc = _hard_acc(model, x, labels)
            if acc == 1.0:
                return 1.0
    return _hard_acc(model, x, labels)


def _hard_acc(model, x, labels) -> float:
    model.eval()
    with torch.no_grad():
        pred = model(x.round()).argmax(dim=-1)
    return (pred == labels).float().mean().item()


# ======================================================================
# 1/2. KD収束(解析teacher)
# ======================================================================

class TestKDConvergence:
    def test_xor_pure_kd_alpha0(self):
        """XOR: α=0(純KD、ラベル不使用)で DLGN student が hard acc=100%"""
        x = _bits(2)
        labels = _parity_labels(x)  # XOR = parity2
        teacher = _make_teacher(2)
        teacher_logits = teacher.predict_logits(x)
        loss_fn = make_kd_loss(alpha=0.0, temperature=2.0)
        acc = _train_kd(_dlgn_student(2, k=16), x, labels, teacher_logits, loss_fn)
        assert acc == 1.0

    def test_parity3_alpha_half(self):
        """parity3: α=0.5(CE+KD)で hard acc=100%"""
        x = _bits(3)
        labels = _parity_labels(x)
        teacher = _make_teacher(3)
        teacher_logits = teacher.predict_logits(x)
        loss_fn = make_kd_loss(alpha=0.5, temperature=2.0)
        acc = _train_kd(_dlgn_student(3, k=24), x, labels, teacher_logits, loss_fn)
        assert acc == 1.0


# ======================================================================
# 3. runner統合スモーク(合成データ、load_raw/resolve_teacherをmonkeypatch)
# ======================================================================

class _SyntheticStudentDataset(torch.utils.data.Dataset):
    """二値化済みstudent入力+ラベル([in_dim] float, int)を返す合成データセット。"""

    def __init__(self, x: torch.Tensor, labels: torch.Tensor):
        self.x = x
        self.labels = labels

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, i):
        return self.x[i], int(self.labels[i].item())


class TestRunnerDistillSmoke:
    def test_run_training_with_distill(self, tmp_path, monkeypatch):
        """DistillConfig付きTrainConfigで run_training が完走し hard_acc が出る。"""
        import griffith_logic.convert.teacher as teacher_mod
        import griffith_logic.datasets.vision as vision_mod
        from griffith_logic.datasets.vision import (
            VISION_DATASETS,
            VisionDatasetSpec,
        )
        from griffith_logic.training.config import load_config
        from griffith_logic.training.runner import run_training

        n = 3
        x = _bits(n)
        labels = _parity_labels(x)
        # student(二値化済み)ローダ: VISION_DATASETSに合成データセットを登録
        student_ds = _SyntheticStudentDataset(x, labels)

        def _fake_load(**params):
            return student_ds, student_ds

        monkeypatch.setitem(
            VISION_DATASETS, 'synth-parity3',
            VisionDatasetSpec('synth-parity3', in_width=n, classes=2, load=_fake_load),
        )
        # load_raw(未二値化)も同じx(合成なので二値化=生入力)を返す
        monkeypatch.setattr(
            vision_mod, 'load_raw',
            lambda name, data_root=None, download=True: (student_ds, student_ds),
        )
        # teacher解決を解析teacherに差し替え
        monkeypatch.setattr(
            teacher_mod, 'resolve_teacher',
            lambda source, device=None: _make_teacher(n),
        )

        cfg = load_config(None, [
            'data.name=synth-parity3',
            'data.batch_size=4',
            'model.kind=dlgn',
            'model.k=24',
            'model.layers=2',
            'model.tau=3.0',
            'optim.lr=0.05',
            'run.epochs=6',
            'run.seed=0',
            'run.device=cpu',
            'run.wandb=false',
            'distill.teacher=analytic:parity3',
            'distill.alpha=0.5',
            'distill.temperature=2.0',
            'distill.cache=false',
        ])
        result = run_training(cfg, out_dir=str(tmp_path / 'run'))
        assert 'hard_acc' in result
        assert 0.0 <= result['hard_acc'] <= 1.0
        assert result['steps'] > 0

    def test_distill_rejects_binary_vector(self, tmp_path):
        """蒸留はclassification専用: logic(binary_vector)ではValueError。"""
        from griffith_logic.training.config import load_config
        from griffith_logic.training.runner import run_training

        cfg = load_config(None, [
            'data.name=logic:xor',
            'model.kind=logic_mlp',
            'run.device=cpu',
            'distill.teacher=analytic:xor',
        ])
        with pytest.raises(ValueError, match='classification'):
            run_training(cfg, out_dir=str(tmp_path / 'run'))


# ======================================================================
# 4. キャッシュ + PackedDistillDataset契約
# ======================================================================

class _CountingTeacher:
    """predict_logits の呼び出し回数を数えるteacher(キャッシュ検証用)。"""

    num_classes = 2
    input_size = (3,)
    fingerprint = 'counting-fp'

    def __init__(self):
        self.calls = 0

    def predict_logits(self, batch):
        self.calls += 1
        return torch.zeros(batch.shape[0], self.num_classes)


class TestCache:
    def test_second_call_uses_cache(self, tmp_path):
        """同一keyで2回目は teacher.predict_logits が呼ばれない(ロードで済む)。"""
        x = _bits(3)
        ds = _SyntheticStudentDataset(x, _parity_labels(x))
        teacher = _CountingTeacher()
        root = str(tmp_path / 'cache')
        key = teacher_cache_key(teacher.fingerprint, 'synth', {}, 'train', len(ds))

        first = get_or_compute_logits(teacher, ds, key=key, root=root)
        calls_after_first = teacher.calls
        assert calls_after_first > 0

        second = get_or_compute_logits(teacher, ds, key=key, root=root)
        assert teacher.calls == calls_after_first  # 増えていない=キャッシュ利用
        assert torch.equal(first, second)

    def test_key_changes_on_params(self):
        """params(や fingerprint)が変わると key が変わる。"""
        base = teacher_cache_key('fp', 'mnist', {'t': 'none'}, 'train', 100)
        assert base != teacher_cache_key('fp', 'mnist', {'t': 'mnist'}, 'train', 100)
        assert base != teacher_cache_key('fp2', 'mnist', {'t': 'none'}, 'train', 100)
        assert base != teacher_cache_key('fp', 'mnist', {'t': 'none'}, 'test', 100)
        assert base != teacher_cache_key('fp', 'mnist', {'t': 'none'}, 'train', 200)
        # 同一内容(dict順序非依存)は同一key
        assert teacher_cache_key('fp', 'd', {'a': 1, 'b': 2}, 's', 1) == \
            teacher_cache_key('fp', 'd', {'b': 2, 'a': 1}, 's', 1)

    def test_packed_dataset_shape_dtype(self):
        """PackedDistillDataset: packed target が [1+C] float、列0=ラベル、列1..=ロジット。"""
        x = _bits(3)
        labels = _parity_labels(x)
        ds = _SyntheticStudentDataset(x, labels)
        logits = torch.randn(len(ds), 2)
        packed_ds = PackedDistillDataset(ds, logits)
        assert len(packed_ds) == len(ds)

        xi, target = packed_ds[2]
        assert torch.equal(xi, x[2])
        assert target.shape == (1 + 2,)
        assert target.dtype == torch.float32
        assert int(target[0].item()) == int(labels[2].item())
        assert torch.allclose(target[1:], logits[2])

    def test_packed_dataset_length_mismatch_raises(self):
        """行数不一致(インデックス整列崩れ)は ValueError。"""
        x = _bits(3)
        ds = _SyntheticStudentDataset(x, _parity_labels(x))
        with pytest.raises(ValueError, match='不一致|整列'):
            PackedDistillDataset(ds, torch.randn(len(ds) - 1, 2))

    def test_packed_dataset_pseudo_labels(self):
        """pseudo_labels=True: 列0が正解でなくteacherロジットのargmaxになる。

        完全ラベルフリー変換の要: 正解ラベルが存在しなくてもhard項が
        疑似ラベルを錨に機能する。正解と疑似が食い違うサンプルで検証。
        """
        x = _bits(3)
        labels = _parity_labels(x)
        ds = _SyntheticStudentDataset(x, labels)
        # 全サンプルで正解と逆のクラスをteacherが最大にするロジット
        wrong = 1 - labels.long()
        logits = torch.full((len(ds), 2), -1.0)
        logits[torch.arange(len(ds)), wrong] = 1.0

        packed_true = PackedDistillDataset(ds, logits)
        packed_pseudo = PackedDistillDataset(ds, logits, pseudo_labels=True)
        for i in range(len(ds)):
            _, t_true = packed_true[i]
            _, t_pseudo = packed_pseudo[i]
            assert int(t_true[0].item()) == int(labels[i].item())
            assert int(t_pseudo[0].item()) == int(wrong[i].item())
            assert torch.allclose(t_true[1:], t_pseudo[1:])
