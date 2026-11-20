"""モデルテンプレートの腐敗防止CI。

同梱テンプレが将来のIRスキーマ/ビルダ/APIドリフトで壊れないことを固定する:
    - 全テンプレの全ネットワークが validate() 通過(常時)
    - 全 train_config が TrainConfig として妥当
    - verification 水準ごとの追加保証:
        'build'    → build_torch + forward 有限
        'transfer' → enc ノードの重み転写(network_transfer)が機能
        'smoke'    → MNIST部分集合で数十step学習し hard acc が下限超え
        'validate' → validate のみ(blackbox=build/学習不可)
    - REST(GET /api/templates, POST /api/templates/{id}/instantiate)

hard acc 下限は seed=0 固定での実測較正値(流儀。下記 SMOKE_FLOORS の
docstring 参照)。実装に合わせて緩めない(実測を下回ったら回帰=修正対象)。

本テストは griffith_logic 本体を一切変更せず、validate/build/simulate を
読み取りでのみ利用する。
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))

from griffith_logic.ir import build_torch, ir_from_dict, validate  # noqa: E402
from tools.editor.backend import templates as T  # noqa: E402
from tools.editor.backend.network_transfer import transfer_weights  # noqa: E402
from tools.editor.backend.project import validate_train_config  # noqa: E402

# ---------------------------------------------------------------------------
# 学習スモークの hard acc 下限(seed=0 固定・MNIST部分集合3000件/40〜150step)。
# 実測(seed 0/1/7): logreg 0.21〜0.30 / mlp 0.28〜0.34 / lenet 0.28〜0.29。
# 下限は最小実測からさらに余裕を取り、random(0.10)を明確に上回る値に固定する。
# ---------------------------------------------------------------------------
SMOKE_FLOORS = {'logreg': 0.15, 'mlp': 0.18, 'lenet': 0.15}
SMOKE_STEPS = {'logreg': (60, 0.05, 128), 'mlp': (60, 0.03, 128),
               'lenet': (150, 0.1, 128)}


# ---------------------------------------------------------------------------
# フィクスチャ的ヘルパ
# ---------------------------------------------------------------------------

def _mnist_subset(n_train: int = 3000, n_eval: int = 1000, bs: int = 128,
                  seed: int = 0):
    """MNIST部分集合のtrain/evalローダ。ローカルデータ不在時はskipへ誘導。"""
    try:
        import torchvision
        from torch.utils.data import DataLoader, Subset

        from tools.editor.backend.datasets import MNIST_ROOT

        tf = torchvision.transforms.ToTensor()
        ds = torchvision.datasets.MNIST(
            MNIST_ROOT, train=True, download=True, transform=tf,
        )
    except Exception as e:  # データ未取得/torchvision無し → スモークはskip
        pytest.skip(f'MNIST未取得のため学習スモークをスキップ: {e}')

    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(ds), generator=g)
    from torch.utils.data import DataLoader, Subset
    tr = Subset(ds, perm[:n_train].tolist())
    ev = Subset(ds, perm[n_train:n_train + n_eval].tolist())
    return (DataLoader(tr, batch_size=bs, shuffle=True, drop_last=True),
            DataLoader(ev, batch_size=256))


def _train_hard_acc(graph: dict, steps: int, lr: float, bs: int,
                    seed: int = 0) -> float:
    """graph を build して数十step学習し hard acc(eval mode + 入力round)を返す。"""
    torch.manual_seed(seed)
    model = build_torch(ir_from_dict(graph))
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss()
    train_loader, eval_loader = _mnist_subset(bs=bs, seed=seed)

    model.train()
    it = iter(train_loader)
    for _ in range(steps):
        try:
            x, y = next(it)
        except StopIteration:
            it = iter(train_loader)
            x, y = next(it)
        x = x.reshape(x.shape[0], -1)
        opt.zero_grad()
        loss_fn(model(x), y).backward()
        opt.step()

    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in eval_loader:
            x = x.reshape(x.shape[0], -1).round()
            correct += (model(x).argmax(1) == y).sum().item()
            total += y.shape[0]
    return correct / total


def _ids(collection):
    return [t.id for t in collection]


# ---------------------------------------------------------------------------
# 登録簿の健全性
# ---------------------------------------------------------------------------

def test_template_ids_unique_and_known_verification():
    ids = _ids(T.TEMPLATES)
    assert len(ids) == len(set(ids)), f'テンプレIDが重複: {ids}'
    known = {'smoke', 'build', 'transfer', 'validate'}
    for t in T.TEMPLATES:
        assert t.verification in known, f'{t.id}: 未知の検証水準 {t.verification}'


def test_first_and_second_wave_present():
    """(第1弾3種)+(第2弾3種)が揃っている。"""
    ids = set(_ids(T.TEMPLATES))
    assert {'logreg', 'mlp', 'lenet'} <= ids  #
    assert {'autoencoder', 'lstm', 'transformer'} <= ids  #


# ---------------------------------------------------------------------------
# 全テンプレ: validate + train_config 妥当性(常時=腐敗防止の核)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('template', T.TEMPLATES, ids=_ids(T.TEMPLATES))
def test_all_networks_validate(template):
    built = template.builder()
    assert built['networks'], f'{template.id}: networksが空'
    for name, ir in built['networks'].items():
        graph = ir_from_dict(ir)
        errs = validate(graph)
        assert errs == [], f'{template.id}/{name}: validate失敗 {errs}'


@pytest.mark.parametrize('template', T.TEMPLATES, ids=_ids(T.TEMPLATES))
def test_all_train_configs_valid(template):
    built = template.builder()
    for name, cfg in built.get('train_configs', {}).items():
        validate_train_config(name, cfg)  # 不正ならValueError


# ---------------------------------------------------------------------------
# verification='build'/'transfer'/'smoke': build_torch + forward 有限
# ---------------------------------------------------------------------------

_BUILDABLE = [t for t in T.TEMPLATES if t.verification in ('build', 'transfer', 'smoke')]


@pytest.mark.parametrize('template', _BUILDABLE, ids=_ids(_BUILDABLE))
def test_buildable_templates_forward_finite(template):
    built = template.builder()
    for name, ir in built['networks'].items():
        graph = ir_from_dict(ir)
        model = build_torch(graph)
        width = ir['inputs'][0]['width']
        x = torch.randint(0, 2, (4, width)).float()
        with torch.no_grad():
            y = model(x)
        assert y.shape[0] == 4
        assert torch.isfinite(y).all(), f'{template.id}/{name}: 非有限出力'


# ---------------------------------------------------------------------------
# verification='validate': blackbox は build 不可(意図どおり raise)
# ---------------------------------------------------------------------------

def test_transformer_blackbox_validate_only():
    built = T.TEMPLATES_BY_ID['transformer'].builder()
    graph = ir_from_dict(built['networks']['Main'])
    assert validate(graph) == []
    with pytest.raises(Exception):  # noqa: B017 — blackboxはbuild不可
        build_torch(graph)


# ---------------------------------------------------------------------------
# LL-ViT複合テンプレ(パッチ埋込+PE+Transformer×N+分類ヘッド)
# ---------------------------------------------------------------------------

def test_llvit_template_registered():
    """llvit が登録簿にあり verification=validate(blackbox含むため)。"""
    t = T.TEMPLATES_BY_ID['llvit']
    assert t.verification == 'validate'
    assert t.builder is T._build_llvit


def test_llvit_structure_and_widths():
    """構成ノード(constant/blackbox/lut_layer/group_sum)の存在と幅整合を固定する。"""
    built = T.TEMPLATES_BY_ID['llvit'].builder()
    graph = built['networks']['Main']
    by_type: dict[str, list[dict]] = {}
    for n in graph['nodes']:
        by_type.setdefault(n['type'], []).append(n)

    # 各構成ノード型が存在する
    assert len(by_type['constant']) == 1                      # PE定数線
    assert len(by_type['concat']) == 1                        # PE連結
    assert len(by_type['blackbox']) == T.LLVIT_DEPTH          # Transformer×N
    assert len(by_type['lut_layer']) == 2                     # patch_embed + head
    assert len(by_type['group_sum']) == 1                     # 分類

    token_width = T.LLVIT_SEQ_LEN * T.LLVIT_D_MODEL
    embed_width = T.LLVIT_SEQ_LEN * T.LLVIT_D_EMBED
    pos_width = T.LLVIT_SEQ_LEN * T.LLVIT_POS_BITS

    # PE: 埋込幅 + 位置幅 = トークン幅(= Transformer入出力幅)
    concat = by_type['concat'][0]
    assert concat['params']['in_widths'] == [embed_width, pos_width]
    const = by_type['constant'][0]
    assert const['params']['width'] == pos_width
    # 全 Transformer blackbox は幅不変 token_width
    for bb in by_type['blackbox']:
        assert bb['params']['torch_class'] == 'LogicTransformerBlock'
        assert bb['params']['in_width'] == token_width
        assert bb['params']['out_width'] == token_width
    # 分類ヘッド: token_width 入力 → GroupSum(10)
    head = next(n for n in by_type['lut_layer'] if n['id'] == 'head')
    assert head['params']['in_width'] == token_width
    assert by_type['group_sum'][0]['params']['k'] == 10


def test_llvit_validate_passes_build_blocked():
    """validate 通過(構造妥当)かつ blackbox を含むため build は不可。"""
    built = T.TEMPLATES_BY_ID['llvit'].builder()
    graph = ir_from_dict(built['networks']['Main'])
    assert validate(graph) == []
    with pytest.raises(Exception):  # noqa: B017 — blackboxはbuild不可
        build_torch(graph)


# ---------------------------------------------------------------------------
# verification='transfer': AutoEncoder の enc 重み転写が機能
# ---------------------------------------------------------------------------

def test_autoencoder_weight_transfer():
    built = T.TEMPLATES_BY_ID['autoencoder'].builder()
    train_ir = built['networks']['Train']
    runtime_ir = built['networks']['Runtime']
    result = transfer_weights(train_ir, runtime_ir)
    assert result['transferred'] == ['enc'], result
    assert result['skipped'] == []
    # 転写後の推論ネットが build + forward 可能
    model = build_torch(ir_from_dict(result['target_ir']))
    x = torch.randint(0, 2, (4, T.AE_IN_WIDTH)).float()
    with torch.no_grad():
        y = model(x)
    assert tuple(y.shape) == (4, T.AE_CODE_WIDTH)


# ---------------------------------------------------------------------------
# verification='smoke': MNIST部分集合で学習し hard acc 下限超え
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('template_id', list(SMOKE_FLOORS), ids=list(SMOKE_FLOORS))
@pytest.mark.learning
def test_smoke_hard_acc_floor(template_id):
    built = T.TEMPLATES_BY_ID[template_id].builder()
    steps, lr, bs = SMOKE_STEPS[template_id]
    acc = _train_hard_acc(built['networks']['Main'], steps, lr, bs, seed=0)
    floor = SMOKE_FLOORS[template_id]
    assert acc >= floor, f'{template_id}: hard_acc={acc:.3f} < 下限{floor}'


# ---------------------------------------------------------------------------
# REST(GET一覧 / POST instantiate)
# ---------------------------------------------------------------------------

def _client(tmp_path, monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    monkeypatch.setattr(T, 'PROJECTS_ROOT', tmp_path / 'editor_projects')
    app = FastAPI()
    app.include_router(T.configure(object()))
    return TestClient(app, base_url='http://localhost:8765')


def test_rest_list_templates(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)
    resp = client.get('/api/templates')
    assert resp.status_code == 200
    body = resp.json()
    ids = {t['id'] for t in body['templates']}
    assert {'logreg', 'mlp', 'lenet', 'autoencoder', 'lstm', 'transformer'} <= ids
    for item in body['templates']:
        assert item['verification'] in {'smoke', 'build', 'transfer', 'validate'}


def test_rest_instantiate_saves_project(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)
    resp = client.post('/api/templates/lenet/instantiate', json={'name': 'マイCNN'})
    assert resp.status_code == 200
    project = resp.json()
    assert project['name'] == 'マイCNN'
    assert 'Main' in project['networks']
    assert project['id']
    # 保存ファイルが実在する
    saved = tmp_path / 'editor_projects' / f'{project["id"]}.json'
    assert saved.exists()


def test_rest_instantiate_unknown_404(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)
    resp = client.post('/api/templates/nope/instantiate', json={})
    assert resp.status_code == 404
