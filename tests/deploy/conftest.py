"""tests/deploy 共通フィクスチャ(Track K)。

scripts/export_webbench.py はパッケージ外なのでファイルパスから直接import
する(tests/test_webbench_export.py と同じ方式)。合成IRビルダ
(build_seg_ir / build_dense_ir / build_concat_ir)をテストで再利用する。
"""

import copy
import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
RUNTIME_C = _REPO_ROOT / 'runtime' / 'c'


def _load_export_webbench():
    if 'export_webbench' in sys.modules:
        return sys.modules['export_webbench']
    spec = importlib.util.spec_from_file_location(
        'export_webbench', _REPO_ROOT / 'scripts' / 'export_webbench.py',
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules['export_webbench'] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope='session')
def webbench():
    """scripts/export_webbench.py モジュール(合成IRビルダ群)。"""
    return _load_export_webbench()


def randomize_conv_truth_tables(graph, seed: int = 0):
    """conv2dノードの真理値表を一様ランダムな2入力関数に差し替える(その場変更)。

    未学習のLutConv2dは残差初期化のパススルーが支配的で、命令列化すると
    ゲートが**全て畳み込まれて消える**(n_instrs=0)。それはそれで実在の
    ケースだが、命令列の実行経路を踏まないので、テストの主役には
    ランダム化した木を使う(レポートのalias支配の観察と同じ理由)。
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    for node in graph.nodes:
        if node.type != 'conv2d':
            continue
        n = len(node.params['truth_tables'])
        node.params['truth_tables'] = [
            f'0x{int(v):x}' for v in rng.integers(0, 16, size=n)
        ]
    return graph


@pytest.fixture(scope='session')
def seg_graph_passthrough(webbench):
    """未学習のまま(パススルー支配 → 命令列が空になる)セグIR。"""
    return webbench.build_seg_ir(
        seed=3, in_shape=(6, 16, 16), oc1=8, oc2=6, k=3, depth=2,
        pool_k=2, up=True, pad=1, name='deploy_seg_passthrough',
    )


@pytest.fixture(scope='session')
def _seg_doc_cached(webbench):
    graph = randomize_conv_truth_tables(
        webbench.build_seg_ir(
            seed=3, in_shape=(6, 16, 16), oc1=8, oc2=6, k=3, depth=2,
            pool_k=2, up=True, pad=1, name='deploy_seg',
        ),
        seed=11,
    )
    return webbench.build_model_doc(
        graph,
        preproc={'kind': 'image_thermometer', 'rgb_channels': 3,
                 'n_thresholds': 2, 'height': 16, 'width': 16,
                 'mean': [0.0, 0.0, 0.0], 'std': [1.0, 1.0, 1.0],
                 'thresholds': [[0.33, 0.66]] * 3,
                 'bit_order': 'rgb,thr,y,x'},
        labels=['a', 'b', 'c'],
    )


@pytest.fixture
def seg_doc(_seg_doc_cached):
    """conv2d/or_pool2d/upsample2d/group_sum を含む小さなセグIRのモデルdoc。

    真理値表はランダム化済み(命令列が実際に残るようにするため)。
    build_bundle が返すバンドルは入力docと入れ子構造を**共有する**ので、
    改竄テストがフィクスチャを汚さないようテストごとにdeepcopyを渡す。
    """
    return copy.deepcopy(_seg_doc_cached)


@pytest.fixture(scope='session')
def _dense_doc_cached(webbench):
    graph = webbench.build_dense_ir(seed=4, in_width=24, widths=[32, 24], k=3,
                                    name='deploy_dense')
    return webbench.build_model_doc(
        graph,
        preproc={'kind': 'thermometer', 'input_kind': 'bits'},
        labels=['a', 'b', 'c'],
    )


@pytest.fixture
def dense_doc(_dense_doc_cached):
    """conv2dを含まない(lut_layer+group_sumのみの)モデルdoc。"""
    return copy.deepcopy(_dense_doc_cached)


# ---------------------------------------------------------------------------
# Cランタイムのビルド/実行ヘルパ
# ---------------------------------------------------------------------------
def _find_cc() -> str | None:
    for name in ('cc', 'clang', 'gcc'):
        path = shutil.which(name)
        if path:
            return path
    return None


CC = _find_cc()


@pytest.fixture(scope='session')
def cc() -> str:
    """Cコンパイラのパス(無い環境ではskip)。"""
    if CC is None:
        pytest.skip('Cコンパイラ(cc/clang/gcc)が無い環境')
    return CC


CFLAGS = ('-std=c99', '-O2', '-Wall', '-Wextra', '-Werror')


@pytest.fixture(scope='session')
def runtime_objects(cc, tmp_path_factory):
    """runtime/c のソースをセッション中1回だけオブジェクトへコンパイルする。

    テストごとにランタイム全体を再コンパイルすると、テスト数×コンパイル時間が
    高速スイートの時間予算を食う(実測で3倍以上の差)。
    """
    out_dir = tmp_path_factory.mktemp('mdl_obj')
    objects: dict[str, str] = {}
    for rel in ('src/mdl_kernels.c',):
        src = RUNTIME_C / rel
        obj = out_dir / (Path(rel).stem + '.o')
        cmd = [cc, *CFLAGS, f'-I{RUNTIME_C / "include"}', '-c', str(src),
               '-o', str(obj)]
        build = subprocess.run(cmd, capture_output=True, text=True)
        if build.returncode != 0:
            raise AssertionError(f'{rel} のコンパイルに失敗:\n{build.stderr}')
        objects[rel] = str(obj)
    return objects


@pytest.fixture
def build_and_run(cc, runtime_objects, tmp_path):
    """Cソースを runtime/c と一緒にビルドして実行し、stdoutをbytesで返す。

    テストごとにデータを配列リテラルとして焼いたmainを生成する方式なので、
    ファイルI/Oもパーサもテスト対象に持ち込まずに済む。
    """
    counter = {'n': 0}

    def run(source: str, *, sources: tuple[str, ...] = ('src/mdl_kernels.c',),
            stdin: bytes = b'', args: tuple[str, ...] = ()) -> bytes:
        counter['n'] += 1
        main_c = tmp_path / f'main{counter["n"]}.c'
        main_c.write_text(source, encoding='utf-8')
        exe = tmp_path / f'a{counter["n"]}.out'
        cmd = [
            cc, *CFLAGS, f'-I{RUNTIME_C / "include"}',
            str(main_c), *[runtime_objects[s] for s in sources],
            '-o', str(exe),
        ]
        build = subprocess.run(cmd, capture_output=True, text=True)
        if build.returncode != 0:
            raise AssertionError(
                f'Cのビルドに失敗:\n{build.stderr}\n--- source ---\n{source}')
        proc = subprocess.run([str(exe), *args], input=stdin,
                              capture_output=True)
        if proc.returncode != 0:
            raise AssertionError(
                f'C実行が失敗 (exit={proc.returncode}):\n'
                f'{proc.stderr.decode(errors="replace")}')
        return proc.stdout

    return run


def c_u32_array(name: str, values) -> str:
    """uint32_t配列のCリテラルを作る(空配列は1要素のダミーにする)。"""
    items = [f'{int(v) & 0xffffffff}u' for v in values] or ['0u']
    return f'static const uint32_t {name}[] = {{{",".join(items)}}};'


def c_i32_array(name: str, values) -> str:
    """int32_t配列のCリテラル(空配列は1要素のダミー)。"""
    items = [str(int(v)) for v in values] or ['0']
    return f'static const int32_t {name}[] = {{{",".join(items)}}};'


def c_u16_array(name: str, values) -> str:
    """uint16_t配列のCリテラル(空配列は1要素のダミー、添字u16化用)。"""
    items = [str(int(v) & 0xffff) for v in values] or ['0']
    return f'static const uint16_t {name}[] = {{{",".join(items)}}};'


# ---------------------------------------------------------------------------
# RDDLGN用の小さな固定モデル
# ---------------------------------------------------------------------------
def build_rddlgn_tiny_classifier(seed: int = 0, n_classes: int = 3):
    """RDDLGNConfig.tiny()ベースの分類器を作って返す(.eval()済み)。

    tests/deploy/test_rddlgn_*.py 共通(抽出・plan・memory・C検証)。
    毎回同じ乱数シードから作るので、テスト間でモデルの重みは決定的。
    """
    import torch

    from griffith_logic.models.rddlgn import RDDLGNClassifier, RDDLGNConfig

    torch.manual_seed(seed)
    cfg = RDDLGNConfig.tiny()
    model = RDDLGNClassifier(cfg, n_classes=n_classes)
    model.eval()
    return model, cfg


def rddlgn_classifier_doc(model, seq_len: int, *, labels=None) -> dict:
    """RDDLGNClassifierのIRを配布バンドル入力形式のdocへ変換する。"""
    from griffith_logic.export.rddlgn_ir import export_rddlgn_classifier_ir
    from griffith_logic.ir.serialize import ir_to_dict
    from griffith_logic.ir.validate import validate

    graph = export_rddlgn_classifier_ir(model, seq_len)
    validate(graph)
    return {'ir': ir_to_dict(graph), 'preproc': {},
           'labels': list(labels) if labels else []}


@pytest.fixture
def rddlgn_tiny():
    """(model, config)。RDDLGNConfig.tiny()、n_classes=3、乱数シード固定。"""
    return build_rddlgn_tiny_classifier(seed=0)


@pytest.fixture
def rddlgn_tiny_doc(rddlgn_tiny):
    model, cfg = rddlgn_tiny
    return rddlgn_classifier_doc(model, cfg.seq_len, labels=['a', 'b', 'c'])


@pytest.fixture
def rddlgn_tiny_bundle(rddlgn_tiny_doc):
    from griffith_logic.deploy.bundle import build_bundle
    return build_bundle(rddlgn_tiny_doc)


# ---------------------------------------------------------------------------
# RDDLGN seq2seq(教師強制)用の小さな固定モデル
# ---------------------------------------------------------------------------
def build_rddlgn_tiny_seq2seq(seed: int = 0):
    """RDDLGNConfig.tiny()ベースのRDDLGN(エンコーダ+デコーダ全体)を作って返す
    (.eval()済み)。

    tests/deploy/test_rddlgn_seq2seq_*.py / test_c_rddlgn_seq2seq*.py 共通。
    build_rddlgn_tiny_classifierと同じ乱数規約(seedから決定的に再構成)。
    src_len/tgt_lenはともにcfg.seq_lenを使う(RDDLGNConfig.tinyは単一の
    seq_lenしか持たないため。教師強制なのでsrc/tgtの長さが同じでも
    一般性は失わない)。
    """
    import torch

    from griffith_logic.models.rddlgn import RDDLGN, RDDLGNConfig

    torch.manual_seed(seed)
    cfg = RDDLGNConfig.tiny()
    model = RDDLGN(cfg)
    model.eval()
    return model, cfg


def rddlgn_seq2seq_doc(model, src_len: int, tgt_len: int, *,
                       labels=None) -> dict:
    """RDDLGN(seq2seq全体)のIRを配布バンドル入力形式のdocへ変換する。"""
    from griffith_logic.export.rddlgn_ir import export_rddlgn_ir
    from griffith_logic.ir.serialize import ir_to_dict
    from griffith_logic.ir.validate import validate

    graph = export_rddlgn_ir(model, src_len, tgt_len)
    assert validate(graph) == []
    return {'ir': ir_to_dict(graph), 'preproc': {},
           'labels': list(labels) if labels else []}


@pytest.fixture
def rddlgn_seq2seq_tiny():
    """(model, config)。RDDLGNConfig.tiny()、乱数シード固定。"""
    return build_rddlgn_tiny_seq2seq(seed=0)


@pytest.fixture
def rddlgn_seq2seq_tiny_doc(rddlgn_seq2seq_tiny):
    model, cfg = rddlgn_seq2seq_tiny
    return rddlgn_seq2seq_doc(model, cfg.seq_len, cfg.seq_len)


@pytest.fixture
def rddlgn_seq2seq_tiny_bundle(rddlgn_seq2seq_tiny_doc):
    from griffith_logic.deploy.bundle import build_bundle
    return build_bundle(rddlgn_seq2seq_tiny_doc)


# ---------------------------------------------------------------------------
# RDDLGN自己回帰生成用: embed_table付きバンドル
# ---------------------------------------------------------------------------
@pytest.fixture
def rddlgn_seq2seq_tiny_bundle_gen(rddlgn_seq2seq_tiny, rddlgn_seq2seq_tiny_doc):
    """embed_table節付きのバンドル(griffith deploy codegen-c ではなく
    generate_rddlgn_c で使う。tests/deploy/test_c_rddlgn_greedy.py 用)。"""
    from griffith_logic.deploy.bundle import build_bundle
    from griffith_logic.export.rddlgn_ir import extract_embed_table

    model, _cfg = rddlgn_seq2seq_tiny
    doc = copy.deepcopy(rddlgn_seq2seq_tiny_doc)
    doc['embed_table'] = extract_embed_table(model.src_embedding)
    return build_bundle(doc)
