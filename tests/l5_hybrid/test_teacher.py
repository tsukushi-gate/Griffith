"""L5: convert/teacher.py — Teacher統一抽象の構造検証。

ネットワークアクセス禁止(torchvisionの重みDLもしない)。合成モデル・
一時ファイルのみで検証する。
"""

import builtins
import os
import sys

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from griffith_logic.convert.teacher import (
    TEACHER_TRANSFORMS,
    UNSAFE_CKPT_ENV,
    OnnxTeacher,
    TorchTeacher,
    resolve_teacher,
)


def _tiny_mlp(in_dim=4, hidden=8, out_dim=3):
    return nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.ReLU(),
        nn.Linear(hidden, out_dim),
    )


# ckpt: の読み込みは weights_only=True 既定になったため、
# metadata['model_class'] に入るこのテスト用ファクトリを明示的に許可する
# (本番のteacherクラスは convert/teacher.py が teacher_models を登録する)。
torch.serialization.add_safe_globals([_tiny_mlp])


def _unlisted_factory():
    """許可リストに登録しないモジュールレベルのcallable(拒否経路の検証用)"""
    return _tiny_mlp()


class TestTorchTeacher:
    def test_predict_logits_shape(self, seed):
        model = _tiny_mlp()
        teacher = TorchTeacher(model, num_classes=3, input_size=(4,))
        batch = torch.randn(5, 4)
        logits = teacher.predict_logits(batch)
        assert logits.shape == (5, 3)

    def test_predict_logits_no_grad(self, seed):
        model = _tiny_mlp()
        teacher = TorchTeacher(model, num_classes=3, input_size=(4,))
        batch = torch.randn(5, 4, requires_grad=True)
        logits = teacher.predict_logits(batch)
        assert not logits.requires_grad

    def test_eval_mode_fixed(self, seed):
        model = _tiny_mlp()
        model.train()
        TorchTeacher(model, num_classes=3, input_size=(4,))
        assert not model.training

    def test_transform_applied(self, seed):
        """transformに渡した関数が実際にpredict_logits内で適用されること"""
        calls = []

        def spy_transform(x):
            calls.append(x.clone())
            return x

        model = _tiny_mlp()
        teacher = TorchTeacher(
            model, num_classes=3, input_size=(4,), transform=spy_transform
        )
        batch = torch.randn(2, 4)
        teacher.predict_logits(batch)
        assert len(calls) == 1
        assert torch.equal(calls[0], batch)

    def test_transform_changes_output(self, seed):
        """恒等以外のtransformでロジットが変わること(実際に適用されている証拠)"""
        model = _tiny_mlp()
        teacher_plain = TorchTeacher(model, num_classes=3, input_size=(4,))
        teacher_scaled = TorchTeacher(
            model, num_classes=3, input_size=(4,), transform=lambda x: x * 10.0
        )
        batch = torch.randn(2, 4)
        out_plain = teacher_plain.predict_logits(batch)
        out_scaled = teacher_scaled.predict_logits(batch)
        assert not torch.allclose(out_plain, out_scaled)

    def test_fingerprint_stable_for_same_model(self, seed):
        model = _tiny_mlp()
        t1 = TorchTeacher(model, num_classes=3, input_size=(4,))
        t2 = TorchTeacher(model, num_classes=3, input_size=(4,))
        assert t1.fingerprint == t2.fingerprint

    def test_fingerprint_changes_with_weights(self, seed):
        model_a = _tiny_mlp()
        model_b = _tiny_mlp()
        # 明示的に重みを変える(seed固定でも別インスタンスなら異なるが、念のため直接改変)
        with torch.no_grad():
            for p in model_b.parameters():
                p.add_(1.0)
        t_a = TorchTeacher(model_a, num_classes=3, input_size=(4,))
        t_b = TorchTeacher(model_b, num_classes=3, input_size=(4,))
        assert t_a.fingerprint != t_b.fingerprint

    def test_fingerprint_changes_with_last_layer_only(self, seed):
        """先頭以外(最終層のみ)の重み変更でも必ず変わる。

        ロジットキャッシュの陳腐化防止に必須の性質。ヘッドのみ
        fine-tuneしたモデルは先頭層が同一なので、全パラメータを
        ハッシュしないと衝突する(過去の欠陥の回帰テスト)。
        """
        model_a = _tiny_mlp()
        model_b = _tiny_mlp()
        with torch.no_grad():
            for pa, pb in zip(model_a.parameters(), model_b.parameters(),
                              strict=True):
                pb.copy_(pa)
            last = list(model_b.parameters())[-1]
            last.add_(1.0)
        t_a = TorchTeacher(model_a, num_classes=3, input_size=(4,))
        t_b = TorchTeacher(model_b, num_classes=3, input_size=(4,))
        assert t_a.fingerprint != t_b.fingerprint

    def test_fingerprint_explicit_override(self, seed):
        model = _tiny_mlp()
        teacher = TorchTeacher(
            model, num_classes=3, input_size=(4,), fingerprint='my-fixed-id'
        )
        assert teacher.fingerprint == 'my-fixed-id'

    def test_satisfies_teacher_protocol_attrs(self, seed):
        model = _tiny_mlp()
        teacher = TorchTeacher(model, num_classes=3, input_size=(4,))
        assert teacher.num_classes == 3
        assert teacher.input_size == (4,)
        assert isinstance(teacher.fingerprint, str) and teacher.fingerprint


class TestResolveTeacherCkpt:
    def _make_ckpt(self, tmp_path, *, with_metadata=True, missing_key=None):
        model = _tiny_mlp()
        metadata = {
            'model_class': _tiny_mlp,
            'model_kwargs': {},
            'num_classes': 3,
            'input_size': (4,),
        }
        if missing_key is not None:
            metadata.pop(missing_key, None)
        data = {
            'model_state_dict': model.state_dict(),
            'metadata': metadata if with_metadata else {},
        }
        path = tmp_path / 'ckpt.pt'
        torch.save(data, path)
        return path, model

    def test_roundtrip_predict_logits_matches_original(self, tmp_path, seed):
        path, model = self._make_ckpt(tmp_path)
        teacher = resolve_teacher(f'ckpt:{path}')
        batch = torch.randn(4, 4)
        model.eval()
        with torch.no_grad():
            expected = model(batch)
        actual = teacher.predict_logits(batch)
        assert torch.allclose(expected, actual)
        assert teacher.num_classes == 3
        assert teacher.input_size == (4,)

    def test_missing_model_class_raises_value_error(self, tmp_path, seed):
        path, _ = self._make_ckpt(tmp_path, missing_key='model_class')
        with pytest.raises(ValueError, match='model_class'):
            resolve_teacher(f'ckpt:{path}')

    def test_missing_num_classes_raises_value_error(self, tmp_path, seed):
        path, _ = self._make_ckpt(tmp_path, missing_key='num_classes')
        with pytest.raises(ValueError, match='num_classes'):
            resolve_teacher(f'ckpt:{path}')

    def test_nonexistent_path_raises_value_error(self):
        with pytest.raises(ValueError, match='存在しません'):
            resolve_teacher('ckpt:/no/such/path/here.pt')

    def test_non_dict_checkpoint_raises_value_error(self, tmp_path):
        path = tmp_path / 'bad.pt'
        torch.save([1, 2, 3], path)
        with pytest.raises(ValueError, match='非対応形式'):
            resolve_teacher(f'ckpt:{path}')

    def test_unsafe_pickle_rejected_by_default(self, tmp_path, monkeypatch):
        """許可リスト外の型を含むckptは既定(weights_only=True)で拒否する(B6)"""
        monkeypatch.delenv(UNSAFE_CKPT_ENV, raising=False)

        model = _tiny_mlp()
        data = {
            'model_state_dict': model.state_dict(),
            'metadata': {
                'model_class': _tiny_mlp, 'model_kwargs': {},
                'num_classes': 3, 'input_size': (4,),
                'extra': _unlisted_factory,
            },
        }
        path = tmp_path / 'unsafe.pt'
        torch.save(data, path)
        with pytest.raises(ValueError, match='weights_only'):
            resolve_teacher(f'ckpt:{path}')

    def test_unsafe_pickle_allowed_with_env_optin(self, tmp_path, monkeypatch):
        """環境変数でのオプトイン時のみ警告付きでpickle読込にフォールバックする"""
        monkeypatch.setenv(UNSAFE_CKPT_ENV, '1')

        model = _tiny_mlp()
        data = {
            'model_state_dict': model.state_dict(),
            'metadata': {
                'model_class': _tiny_mlp, 'model_kwargs': {},
                'num_classes': 3, 'input_size': (4,),
                'extra': _unlisted_factory,
            },
        }
        path = tmp_path / 'unsafe2.pt'
        torch.save(data, path)
        with pytest.warns(RuntimeWarning, match=UNSAFE_CKPT_ENV):
            teacher = resolve_teacher(f'ckpt:{path}')
        assert teacher.num_classes == 3


class TestResolveTeacherErrors:
    def test_invalid_source_no_colon(self):
        with pytest.raises(ValueError, match='scheme:value'):
            resolve_teacher('not-a-valid-source')

    def test_unknown_scheme(self):
        with pytest.raises(ValueError, match='未知のteacherスキーム'):
            resolve_teacher('bogus:whatever')

    def test_hf_unsupported_repo_raises_value_error(self):
        """限定サポート外のrepo_idはネットに触れずValueError(汎用AutoModel非対応)。"""
        with pytest.raises(ValueError, match='hf:'):
            resolve_teacher('hf:some/unsupported-repo')


class TestResolveTeacherHf:
    """'hf:' 分岐(edadaltocg/resnet18_cifar10限定)のネット不要テスト。

    huggingface-hub の hf_hub_download を monkeypatch し、ローカルで合成した
    state_dict を返すことで、①CIFAR resnet18 stem 改変モデルへのロード経路と
    ②huggingface-hub 未導入時のImportError分岐を検証する。実DLは実験側で行う。
    """

    _REPO = 'edadaltocg/resnet18_cifar10'

    def _fake_state_dict(self):
        """CIFAR化resnet18の正しいkey構造を持つランダムstate_dict。"""
        from griffith_logic.convert.teacher import _build_cifar_resnet18

        return _build_cifar_resnet18(num_classes=10).state_dict()

    def test_loads_via_monkeypatched_hub(self, monkeypatch, tmp_path):
        """hf_hub_download を偽装し、state_dictロード経路が通ること。"""
        huggingface_hub = pytest.importorskip('huggingface_hub')
        sd = self._fake_state_dict()
        bin_path = tmp_path / 'pytorch_model.bin'
        torch.save(sd, bin_path)

        def fake_download(repo_id, filename, *args, **kwargs):
            assert repo_id == self._REPO
            assert filename == 'pytorch_model.bin'
            return str(bin_path)

        monkeypatch.setattr(huggingface_hub, 'hf_hub_download', fake_download)

        teacher = resolve_teacher(f'hf:{self._REPO}')
        assert teacher.num_classes == 10
        assert teacher.input_size == (3, 32, 32)
        assert teacher.transform is not None  # 恒等transformが素で入る(要上書き規約)
        # (3,32,32)入力で(B,10)ロジットが出る(stem改変で32x32が保たれる)
        logits = teacher.predict_logits(torch.rand(2, 3, 32, 32))
        assert logits.shape == (2, 10)

    def test_import_error_when_hub_missing(self, monkeypatch):
        """huggingface-hub未導入環境では導入方法を含むImportError。"""
        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == 'huggingface_hub':
                raise ImportError("No module named 'huggingface_hub'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, '__import__', fake_import)
        with pytest.raises(ImportError, match='uv sync --extra teachers'):
            resolve_teacher(f'hf:{self._REPO}')


class TestOnnxTeacher:
    def test_import_error_message_when_onnxruntime_missing(self, monkeypatch):
        """onnxruntime未導入環境ではImportErrorに導入方法が含まれること。

        onnxruntimeが実際に入っている環境でも、importをmonkeypatchして
        未導入分岐を強制的に検証する。
        """
        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == 'onnxruntime':
                raise ImportError("No module named 'onnxruntime'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, '__import__', fake_import)
        with pytest.raises(ImportError, match='uv sync --extra teachers'):
            OnnxTeacher('dummy.onnx', num_classes=10, input_size=(3, 224, 224))

    def test_resolve_teacher_onnx_scheme_import_error(self, monkeypatch):
        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == 'onnxruntime':
                raise ImportError("No module named 'onnxruntime'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, '__import__', fake_import)
        with pytest.raises(ImportError, match='uv sync --extra teachers'):
            resolve_teacher('onnx:dummy.onnx')


class TestTeacherTransforms:
    def test_keys(self):
        assert set(TEACHER_TRANSFORMS) == {
            'none', 'mnist', 'cifar10', 'cifar10-resnet18-hf', 'binarize05',
        }

    def test_none_is_identity(self):
        x = torch.randn(2, 1, 4, 4)
        assert torch.equal(TEACHER_TRANSFORMS['none'](x), x)

    def test_binarize05_thresholds_at_half(self):
        x = torch.tensor([[[[0.0, 0.4, 0.5, 0.6, 1.0]]]])
        out = TEACHER_TRANSFORMS['binarize05'](x)
        expected = torch.tensor([[[[0.0, 0.0, 0.0, 1.0, 1.0]]]])
        assert torch.equal(out, expected)
        assert out.dtype == x.dtype

    def test_mnist_shape_and_normalization(self):
        x = torch.zeros(2, 1, 4, 4)
        out = TEACHER_TRANSFORMS['mnist'](x)
        assert out.shape == x.shape
        expected = (0.0 - 0.1307) / 0.3081
        assert torch.allclose(out, torch.full_like(out, expected), atol=1e-6)

    def test_cifar10_shape_and_channels(self):
        x = torch.zeros(2, 3, 8, 8)
        out = TEACHER_TRANSFORMS['cifar10'](x)
        assert out.shape == x.shape
        means = (0.4914, 0.4822, 0.4465)
        stds = (0.2470, 0.2435, 0.2616)
        for c in range(3):
            expected = (0.0 - means[c]) / stds[c]
            assert torch.allclose(
                out[:, c], torch.full_like(out[:, c], expected), atol=1e-6
            )
