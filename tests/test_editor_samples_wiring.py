"""エディタのサンプル: 構造化配線の段別パラメータ例 3 本。

サンプルの構造はフロントと共有する仕様 JSON
`tools/editor/frontend/src/samples/wiringSamples.json` にあり、フロント(`sampleIr.ts`)と
本テストが同じ規則で draft の IR に展開する(indices / truth_tables は placeholder、
build 時に train_info.connections + wiring から `make_connections` で再生成)。

- 3 本の IR が validate を通り、build_torch が各層に張った indices が
  `core.connections.make_connections` の出力と bitwise 一致する
- MNIST の 2 本は `configs/recipes/fmc_w1000_{rot,bf4}.yaml` の runner モデルと層ごとに同一配線
- 「2bit 加算器 rot 配線 学習」はエディタの TrainingSession(論理プリセット lr 0.08 / batch 16 /
  60 epoch / bce / adam)で全 16 行が exact 一致になる(learning。事前実験 fmc-sample-rot16 の
  4 seed と同じ判定)
- MNIST の 2 本は 1 epoch の smoke(slow。ローカル data-mnist が無ければ skip、閾値は置かない)
- 論理データセット adder2〜mul4 に表示名がある
"""

import json
import time
from pathlib import Path

import pytest
import torch

from griffith_logic import GroupSum, LutLayer
from griffith_logic.core.connections import make_connections
from griffith_logic.datasets import LOGIC_DATASETS
from griffith_logic.ir import build_torch, ir_from_dict, validate
from griffith_logic.training.config import load_config
from griffith_logic.training.runner import ResolvedData, _build_model
from tools.editor.backend.datasets import DATASET_LABELS, list_datasets
from tools.editor.backend.training import TrainingSession

REPO_ROOT = Path(__file__).resolve().parents[1]
SPEC_PATH = REPO_ROOT / 'tools' / 'editor' / 'frontend' / 'src' / 'samples' / 'wiringSamples.json'
SPECS: dict = json.loads(SPEC_PATH.read_text(encoding='utf-8'))
SAMPLE_KEYS = ('adder2_rot16', 'mnist_rot1000', 'mnist_bf4_1000')
MNIST_KEYS = ('mnist_rot1000', 'mnist_bf4_1000')

# エディタの論理プリセット(sampleIr.ts の LOGIC_TRAINING_PRESET と同値)
LOGIC_PRESET = dict(lr=0.08, batch_size=16, epochs=60, method='bp', loss='bce',
                    optimizer_name='adam')
# エディタの MNIST 1000 用プリセット(sampleIr.ts の MNIST_W1000_TRAINING_PRESET と同値、epoch 以外)
MNIST_PRESET = dict(lr=0.01, batch_size=100, method='bp', loss='ce', optimizer_name='adam')


def _draft_lut(node_id: str, in_width: int, layer: dict) -> dict:
    """sampleIr.ts の draftWiredLut と同じ展開(placeholder + 構造化配線の指定)。"""
    out_w, n = layer['out_width'], layer['n_inputs']
    return {
        'id': node_id, 'type': 'lut_layer',
        'params': {
            'in_width': in_width, 'out_width': out_w, 'n_inputs': n,
            'indices': [[0] * out_w for _ in range(n)],
            'truth_tables': ['0x0'] * out_w,
        },
        'train_info': {
            'parameterization': 'softmax', 'selection': 'soft', 'initialized': False,
            'connections': layer['connections'], 'wiring': dict(layer['wiring']),
        },
        'meta': {},
    }


def sample_ir(key: str) -> dict:
    """仕様 JSON の 1 件を IR(dict)に展開する(sampleIr.ts の wiredSampleGraph と同形)。"""
    spec = SPECS[key]
    nodes = []
    width = spec['in_width']
    for i, layer in enumerate(spec['layers']):
        nodes.append(_draft_lut(f'l{i}_lut', width, layer))
        width = layer['out_width']
    if spec['group_sum']:
        nodes.append({
            'id': 'group_sum', 'type': 'group_sum',
            'params': {'in_width': width, 'k': spec['group_sum']['k'],
                       'tau': spec['group_sum']['tau']},
            'train_info': {}, 'meta': {},
        })
    ids = [n['id'] for n in nodes]
    edges = [{'src': 'x', 'dst': ids[0]}]
    edges += [{'src': a, 'dst': b} for a, b in zip(ids, ids[1:])]
    edges += [{'src': ids[-1], 'dst': 'y'}]
    return {
        'format': 'magicdelta-ir', 'version': '0.1.0',
        'metadata': {'name': f'sample_{key}'},
        'inputs': [{'id': 'x', 'width': spec['in_width'], 'shape': None,
                    'encoding': spec['input_encoding'], 'semantics': None}],
        'outputs': [{'id': 'y', 'width': spec['out_width'], 'shape': None,
                     'encoding': None, 'semantics': spec['output_semantics']}],
        'nodes': nodes, 'edges': edges,
    }


def _lut_layers(model: torch.nn.Module) -> list[LutLayer]:
    return [m for m in model if isinstance(m, LutLayer)]


def _exact_match(model: torch.nn.Module, dataset_key: str) -> tuple[float, float]:
    """全列挙を hard(eval)で流し、(行の exact match 率, ビット正解率)を返す。"""
    x, y = LOGIC_DATASETS[dataset_key]().dataset.tensors
    was = model.training
    model.eval()
    with torch.no_grad():
        out = model(x.to(torch.float32))
    if was:
        model.train()
    pred = (out >= 0.5).to(y.dtype)
    return (pred == y).all(dim=1).float().mean().item(), (pred == y).float().mean().item()


@pytest.fixture(autouse=True)
def _session_roots(tmp_path, monkeypatch):
    import tools.editor.backend.training as training_mod
    monkeypatch.setattr(training_mod, 'HISTORY_ROOT', tmp_path / 'hist')
    monkeypatch.setattr(training_mod, 'RUNS_ROOT', tmp_path / 'runs')


# --- 仕様 JSON の形 ---

class TestSpec:
    def test_keys(self):
        assert tuple(SPECS) == SAMPLE_KEYS

    @pytest.mark.parametrize('key', SAMPLE_KEYS)
    def test_widths_chain(self, key):
        spec = SPECS[key]
        width = spec['in_width']
        for layer in spec['layers']:
            if layer['connections'] == 'rot_n':
                assert len(layer['wiring']['shifts']) == layer['n_inputs']
            width = layer['out_width']
        if spec['group_sum']:
            assert width % spec['group_sum']['k'] == 0
            assert spec['group_sum']['k'] == spec['out_width']
        else:
            assert width == spec['out_width']

    def test_adder2_shape(self):
        """(a): 4 → rot(0,2) → 16 → rot(0,1)/(0,2)/(0,4)/(0,8) → rot(0,8, step 5) → 3"""
        spec = SPECS['adder2_rot16']
        assert spec['dataset'] == 'logic:adder2'
        assert [l['out_width'] for l in spec['layers']] == [16, 16, 16, 16, 16, 3]
        assert [l['wiring']['s_b'] for l in spec['layers']] == [2, 1, 2, 4, 8, 8]
        assert spec['layers'][-1]['wiring']['step'] == 5


# --- validate → build_torch → 生成器との一致 ---

class TestBuild:
    @pytest.mark.parametrize('key', SAMPLE_KEYS)
    def test_validate_and_indices_match_generator(self, key):
        spec = SPECS[key]
        graph = ir_from_dict(sample_ir(key))
        validate(graph)
        core = build_torch(graph)
        luts = _lut_layers(core)
        assert len(luts) == len(spec['layers'])
        width = spec['in_width']
        for layer, lut in zip(spec['layers'], luts):
            want = make_connections(
                layer['connections'], width, layer['out_width'], layer['n_inputs'],
                **layer['wiring'])
            assert torch.equal(lut.indices, want)
            assert lut.n_inputs == layer['n_inputs']
            width = layer['out_width']
        if spec['group_sum']:
            assert isinstance(core[-1], GroupSum)

    @pytest.mark.parametrize('key', SAMPLE_KEYS)
    def test_draft_forward_backward(self, key):
        core = build_torch(ir_from_dict(sample_ir(key)))
        core.train()
        x = torch.randint(0, 2, (4, SPECS[key]['in_width'])).to(torch.float32)
        loss = core(x).sum()
        loss.backward()
        assert torch.isfinite(loss)

    @pytest.mark.parametrize('key', MNIST_KEYS)
    def test_mnist_sample_equals_recipe_model(self, key):
        """サンプルは事前実験のレシピと同じ配線・幅・tau"""
        spec = SPECS[key]
        cfg = load_config(str(REPO_ROOT / spec['recipe']))
        data = ResolvedData(name='mnist', kind='classification', in_width=784, out_width=10,
                            train_loader=None, eval_loader=None)
        ref = _build_model(cfg.model, data, 'cpu')
        core = build_torch(ir_from_dict(sample_ir(key)))
        ref_luts, luts = _lut_layers(ref), _lut_layers(core)
        assert len(ref_luts) == len(luts) == cfg.model.layers == 6
        for a, b in zip(ref_luts, luts):
            assert torch.equal(a.indices, b.indices)
            assert a.n_inputs == b.n_inputs == cfg.model.n_inputs
        assert spec['group_sum'] == {'k': 10, 'tau': cfg.model.tau}
        assert cfg.model.k == spec['layers'][0]['out_width'] == 1000


# --- データセットの表示名 ---

class TestDatasetLabels:
    def test_logic_arith_datasets_have_labels(self):
        by_name = {d['name']: d for d in list_datasets()}
        for key in ('adder2', 'adder3', 'adder4', 'mul2', 'mul3', 'mul4'):
            assert key in DATASET_LABELS
            assert by_name[f'logic:{key}']['label'] == DATASET_LABELS[key] != key


# --- 学習(エディタの TrainingSession をそのまま使う) ---

@pytest.mark.learning
class TestAdder2Rot16Learns:
    def test_exact_match_all_rows_within_60_epochs(self, tmp_path):
        """事前実験 fmc-sample-rot16(lyc0ohc2 / xv9y2uf5 / kggnwr56 / 0oem00z2)と同じ判定"""
        torch.manual_seed(0)
        ir = sample_ir('adder2_rot16')
        graph = ir_from_dict(ir)
        validate(graph)
        model = build_torch(graph)
        sess = TrainingSession(
            model, dataset='logic:adder2', ir_snapshot=ir, use_wandb=False,
            history_root=tmp_path / 'hist', runs_root=tmp_path / 'runs', **LOGIC_PRESET)
        sess._run()  # 同期実行(スレッドを使わない)
        assert sess.state == 'finished', sess.error
        rows, bits = _exact_match(model, 'adder2')
        assert bits == 1.0
        assert rows == 1.0


@pytest.mark.slow
@pytest.mark.skipif(not (REPO_ROOT / 'data-mnist').is_dir(), reason='ローカル data-mnist が無い')
class TestMnistSamplesSmoke:
    @pytest.mark.parametrize('key', MNIST_KEYS)
    def test_one_epoch(self, key, tmp_path):
        pytest.importorskip('torchvision')
        torch.manual_seed(0)
        # エディタの build と同じく、画像入力 (B,1,28,28) のために Flatten を前置する
        model = torch.nn.Sequential(torch.nn.Flatten(), *build_torch(ir_from_dict(sample_ir(key))))
        t0 = time.time()
        sess = TrainingSession(
            model, dataset='mnist', epochs=1, use_wandb=False,
            history_root=tmp_path / 'hist', runs_root=tmp_path / 'runs', **MNIST_PRESET)
        sess._run()
        secs = time.time() - t0
        assert sess.state == 'finished', sess.error
        assert sess.last_val_acc is not None
        print(f'{key}: 1 epoch {secs:.0f}s, hard val acc {sess.last_val_acc:.4f}')
