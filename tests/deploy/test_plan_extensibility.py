"""プラン層の拡張性を「守られているか自動で分かる」形に固定する。

目的は挙動を変えることではなく、**今後モデルが大規模化しても・
タスクが変わっても・RDDLGN のような別のNN構造が来ても、最適化とバックエンドが
増殖しない**基盤にすること。設計目標そのものはテストできないので、
「その目標が壊れたときに落ちる性質」を4つ固定する。

    1. 新しいIRノード型を足すのに触るのは登録1箇所(3バックエンド無改造)
    2. 未対応は「ノード型・バックエンド・理由」を明示して落ちる
    3. 前処理の符号化コードは Python が単一情報源(C/GPU が追従)
    4. 最適化が乗る土台(op列)の不変条件がノード型に依存しない
"""

import copy
import re

import pytest

from griffith_logic.deploy import build_bundle
from griffith_logic.deploy import plan as planmod
from griffith_logic.deploy.backends import (
    BACKENDS,
    can_run,
    capabilities,
    supported_node_types,
    unsupported_node_types,
)
from griffith_logic.deploy.plan import (
    KIND_BITPLANE,
    KIND_FLAT,
    KIND_SCORES,
    KIND_SCRATCH,
    UnsupportedNode,
    build_plan,
)
from griffith_logic.deploy.preproc_desc import (
    PREPROC_DESC_STRIDE,
    PREPROC_OPS,
)

from .conftest import RUNTIME_C

RUNTIME = RUNTIME_C.parent


@pytest.fixture
def seg_bundle(seg_doc):
    return build_bundle(seg_doc)


class TestNewNodeTypeIsOneRegistration:
    """基準1: 新しいIRノード型は**登録1箇所**で3バックエンドに載る。

    ここで「載る」の意味を強くしてある: 別名ノード型で生成したソースが、
    元のノード型で生成したものと**完全に一致する**こと。どれか1つの
    バックエンドに種別分岐が残っていれば、その差分として現れる。
    """

    ALIAS = 'test_pool_alias'

    @pytest.fixture(autouse=True)
    def _register_alias(self):
        @planmod.register_node_kind(self.ALIAS)
        def _plan_alias(b, node, ins):
            return planmod.NODE_KINDS['or_pool2d'](b, node, ins)

        yield
        planmod.NODE_KINDS.pop(self.ALIAS, None)

    def _rename(self, bundle: dict) -> dict:
        """IR中の or_pool2d を別名ノード型へ差し替える。"""
        out = copy.deepcopy(bundle)
        renamed = 0
        for node in out['ir']['nodes']:
            if node['type'] == 'or_pool2d':
                node['type'] = self.ALIAS
                renamed += 1
        assert renamed > 0, 'テスト対象の or_pool2d がモデルに無い'
        return out

    def test_alias_node_type_is_recognized_by_plan(self, seg_bundle):
        base = build_plan(seg_bundle)
        alias = build_plan(self._rename(seg_bundle))
        assert [op.kind for op in alias.ops] == [op.kind for op in base.ops]
        assert [op.attrs.get('op') for op in alias.ops] == \
               [op.attrs.get('op') for op in base.ops]

    # id を 'cuda' そのものにしない: tests/conftest.py が keywords に
    # 'cuda' を含むテストをGPU非搭載機で自動skipするが、CUDA
    # **コード生成**にGPUは要らないので、ここは常に走らせたい。
    # webはここに含めない: webはIRをそのままバンドルへ実体として詰めて
    # 配布する(プラン歩行の走査結果からコードを合成する経路が無い)ので、
    # エイリアス型で出力バイト列が変わるのは正当(そもそもIR側の型名を
    # 見ていない)。この一致テストはプラン歩行を持つバックエンドの回帰検出用。
    @pytest.mark.parametrize('backend', [
        'c', pytest.param('cuda', id='cuda_src'), 'metal'])
    def test_all_backends_handle_alias_without_modification(
            self, seg_bundle, backend):
        generate = BACKENDS[backend][1]
        base = generate(seg_bundle, name='m')
        alias = generate(self._rename(seg_bundle), name='m')
        assert alias == base, (
            f'{backend} バックエンドが別名ノード型で違うコードを吐いた'
            '(どこかにノード型の分岐が残っている)')

    def test_registration_is_the_only_touch_point(self):
        """別名を外せば、同じモデルが再び未対応になる(登録が効いている証明)。"""
        planmod.NODE_KINDS.pop(self.ALIAS)
        with pytest.raises(UnsupportedNode):
            build_plan(self._rename(build_bundle_stub()))


class TestRecurrentAliasNodeTypeIsOneRegistration:
    """基準1をrecurrentにも適用する(CUDA/Metalは追加)。

    recurrentはbody(nested IRGraph)を同一_PlanBuilderで再帰planし、
    Cバックエンドが静的ステップ関数(mdl_step_<id>)へ、GPUバックエンドが
    ホストのfor(t)ループへ切り出すという、or_pool2dより複雑な経路を通る。
    それでも「別名ノード型で生成したソースが元と完全一致」が成り立つこと、
    つまりどこにも`node.type == 'recurrent'`という分岐が残っていないことを
    3バックエンドすべてで固定する。
    """

    ALIAS = 'test_recurrent_alias'

    @pytest.fixture(autouse=True)
    def _register_alias(self):
        @planmod.register_node_kind(self.ALIAS)
        def _plan_alias(b, node, ins):
            return planmod.NODE_KINDS['recurrent'](b, node, ins)

        yield
        planmod.NODE_KINDS.pop(self.ALIAS, None)

    def _rename(self, bundle: dict) -> dict:
        """IR中のトップレベルrecurrentノードを別名ノード型へ差し替える。

        body内のscan_var/lut_layer/concatはrecurrentではないのでそのまま
        (別名にするのは「recurrentノード自身」が正しく登録1箇所で
        扱われるか、という基準1の主張そのもの)。
        """
        out = copy.deepcopy(bundle)
        renamed = 0
        for node in out['ir']['nodes']:
            if node['type'] == 'recurrent':
                node['type'] = self.ALIAS
                renamed += 1
        assert renamed > 0, 'テスト対象のrecurrentがモデルに無い'
        return out

    def test_alias_node_type_is_recognized_by_plan(self, rddlgn_tiny_bundle):
        base = build_plan(rddlgn_tiny_bundle)
        alias = build_plan(self._rename(rddlgn_tiny_bundle))
        assert [op.kind for op in alias.ops] == [op.kind for op in base.ops]
        assert [op.scope for op in alias.ops] == [op.scope for op in base.ops]
        assert [op.attrs for op in alias.ops] == [op.attrs for op in base.ops]

    # id を 'cuda' そのものにしない: tests/conftest.py が keywords に
    # 'cuda' を含むテストをGPU非搭載機で自動skipするが、CUDA
    # **コード生成**にGPUは要らないので、ここは常に走らせたい。
    @pytest.mark.parametrize('backend', [
        'c', pytest.param('cuda', id='cuda_src'), 'metal'])
    def test_all_backends_handle_alias_without_modification(
            self, rddlgn_tiny_bundle, backend):
        generate = BACKENDS[backend][1]
        kwargs = {'validate': False} if backend == 'c' else {}
        base = generate(rddlgn_tiny_bundle, name='m', **kwargs)
        alias = generate(self._rename(rddlgn_tiny_bundle), name='m', **kwargs)
        assert alias == base, (
            f'{backend} バックエンドが別名ノード型で違うコードを吐いた'
            '(どこかにrecurrentの種別分岐が残っている)'
        )

    def test_registration_is_the_only_touch_point(self, rddlgn_tiny_bundle):
        """別名を外せば、同じモデルが再び未対応になる(登録が効いている証明)。"""
        renamed = self._rename(rddlgn_tiny_bundle)
        planmod.NODE_KINDS.pop(self.ALIAS)
        with pytest.raises(UnsupportedNode):
            build_plan(renamed)


def build_bundle_stub() -> dict:
    """最小のIR: or_pool2d 1つだけ(登録解除の確認用)。"""
    return {
        'ir': {
            'format': 'magicdelta-ir',
            'version': '0.2.0',
            'inputs': [{'id': 'x', 'width': 2 * 4 * 4}],
            'outputs': [{'id': 'y', 'width': 2 * 2 * 2}],
            'nodes': [{'id': 'p', 'type': 'or_pool2d',
                       'params': {'in_shape': [2, 4, 4], 'kernel_size': 2}}],
            'edges': [{'src': 'x', 'dst': 'p'}, {'src': 'p', 'dst': 'y'}],
        },
    }


class TestUnsupportedIsExplained:
    """基準2: 未対応は「ノード型・バックエンド・理由」を伴って落ちる。

    以前は同じ趣旨の例外が3バックエンドに51箇所散っていて、
    「何が対応で何が未対応か」をコードから読み取れなかった。
    """

    def test_every_ir_node_type_is_supported_or_explained(self):
        """IRの全ノード型が「対応」か「理由つき非対応」のどちらかである。

        新しいノード型をIRに足したのにデプロイ側で放置すると落ちる。
        """
        from griffith_logic.ir.schema import NODE_TYPES

        known = supported_node_types() | set(unsupported_node_types())
        missing = sorted(set(NODE_TYPES) - known)
        assert not missing, (
            f'デプロイ側で対応も非対応宣言もされていないノード型: {missing}。'
            'plan.py で register_node_kind するか declare_unsupported すること')

    def test_unsupported_reason_is_not_empty(self):
        for node_type, reason in unsupported_node_types().items():
            assert len(reason) > 10, f'{node_type} の非対応理由が薄い'

    def test_error_names_node_and_type(self, seg_bundle):
        bad = copy.deepcopy(seg_bundle)
        for node in bad['ir']['nodes']:
            if node['type'] == 'or_pool2d':
                node['params']['kernel_size'] = 3
                target = node['id']
                break
        with pytest.raises(UnsupportedNode) as exc:
            build_plan(bad)
        assert exc.value.node_id == target
        assert exc.value.node_type == 'or_pool2d'
        assert '未対応' in str(exc.value)
        assert target in str(exc.value), 'どのノードか特定できない'

    def test_backend_error_names_the_backend(self):
        """バックエンド固有の非対応には、どのバックエンドかが載る。"""
        exc = UnsupportedNode('理由', node_id='n1', node_type='conv2d',
                              backend='metal')
        text = str(exc)
        assert 'metal' in text and 'n1' in text and 'conv2d' in text

    # id を 'cuda' そのものにしない: tests/conftest.py が keywords に
    # 'cuda' を含むテストをGPU非搭載機で自動skipするが、CUDA
    # **コード生成**にGPUは要らないので、ここは常に走らせたい。
    @pytest.mark.parametrize('backend', [
        'c', pytest.param('cuda', id='cuda_src'), 'metal', 'web'])
    def test_can_run_reports_ok_for_real_model(self, seg_bundle, backend):
        ok, reason = can_run(seg_bundle, backend)
        assert ok, reason

    # id を 'cuda' そのものにしない: tests/conftest.py が keywords に
    # 'cuda' を含むテストをGPU非搭載機で自動skipするが、CUDA
    # **コード生成**にGPUは要らないので、ここは常に走らせたい。
    @pytest.mark.parametrize('backend', [
        'c', pytest.param('cuda', id='cuda_src'), 'metal', 'web'])
    def test_capabilities_cover_the_plan(self, seg_bundle, backend):
        """申告した能力が、実モデルのop列を実際に覆っている。"""
        cap = capabilities(backend)
        plan = build_plan(seg_bundle)
        for op in plan.ops:
            assert op.kind in cap['ops'], f'{backend}: {op.kind} が申告漏れ'
            sub = op.attrs.get('op')
            if sub is not None:
                assert sub in cap['sub_ops'], f'{backend}: {sub} が申告漏れ'


class TestPreprocOpCodesHaveOneSource:
    """基準3: 前処理の符号化コードは Python が単一情報源。

    以前は Python / C / GPU に3系統手書きされていて、入力符号化を1つ足すのに
    3箇所直す必要があった(しかもずれても静かに壊れる)。
    """

    def _defines(self, path, prefix: str) -> dict[str, int]:
        text = path.read_text(encoding='utf-8')
        found = re.findall(rf'#define {re.escape(prefix)}(\w+)\s+(\d+)', text)
        return {name: int(value) for name, value in found}

    def test_c_header_matches_python(self):
        c = self._defines(RUNTIME / 'c' / 'include' / 'mdl_kernels.h',
                          'MDL_PREPROC_OP_')
        assert c == PREPROC_OPS, (
            'C の MDL_PREPROC_OP_* が deploy/preproc_desc.py::PREPROC_OPS と'
            'ずれている(前処理opの単一情報源はPython側)')

    def test_gpu_header_matches_python(self):
        gpu = self._defines(RUNTIME / 'gpu' / 'mdl_gpu_kernels.h', 'MDL_G_OP_')
        assert gpu == PREPROC_OPS, (
            'GPU の MDL_G_OP_* が PREPROC_OPS とずれている')

    def test_descriptor_stride_matches(self):
        header = (RUNTIME / 'c' / 'include' / 'mdl_kernels.h').read_text(
            encoding='utf-8')
        m = re.search(r'#define MDL_PREPROC_STRIDE\s+(\d+)', header)
        assert m, 'MDL_PREPROC_STRIDE が見つからない'
        assert int(m.group(1)) == PREPROC_DESC_STRIDE, (
            '記述子のストライドがPython側とずれている'
            '(チャネルの読み出し位置が全部ずれる)')


class TestPlanInvariants:
    """基準4: 最適化が乗る土台の不変条件が、ノード型に依存しない。

   (メモリ計画)・(バッチ)・(命令列融合)はどれも
    「op列の各値がいつ生きているか」を根拠に動く。その根拠が成り立つことを
    **op列だけを見て**確認する — ノード型の知識を一切使わない。
    """

    @pytest.fixture(params=['seg', 'dense'])
    def any_plan(self, request, seg_doc, dense_doc):
        doc = seg_doc if request.param == 'seg' else dense_doc
        return build_plan(build_bundle(doc))

    def test_single_assignment(self, any_plan):
        """1つの値を書くopは高々1つ(生存区間解析の前提)。"""
        written = [op.output for op in any_plan.ops]
        assert len(written) == len(set(written)), (
            '同じ値を2回書いている(生存区間が定義できない)')

    def test_defined_before_use(self, any_plan):
        """opの入力は、それより前のopの出力かグラフ入力である。"""
        available = {any_plan.input_value}
        if any_plan.input_bitplane is not None:
            available.add(any_plan.input_bitplane)
        for i, op in enumerate(any_plan.ops):
            for vid in op.inputs:
                assert vid in available, (
                    f'op[{i}] {op.kind} が未定義の値 v{vid} を読んでいる')
            available.add(op.output)

    def test_output_is_scores(self, any_plan):
        assert any_plan.value(any_plan.output_value).kind == KIND_SCORES
        assert any_plan.value(any_plan.output_value).width == \
               any_plan.output_width

    def test_every_value_has_a_size(self, any_plan):
        """全ての値がバイト数を答えられる(メモリ計画の前提)。"""
        for v in any_plan.values:
            assert v.nbytes() > 0, f'v{v.id}({v.kind}) のサイズが0'
            if v.kind == KIND_BITPLANE:
                assert v.words == v.c * v.h * v.wpr

    def test_all_values_reachable(self, any_plan):
        """使われない値を作っていない(死んだバッファが残らない)。"""
        used = {any_plan.output_value}
        for op in any_plan.ops:
            used.update(op.inputs)
            used.update(op.attrs.get('scratch', ()))
            used.add(op.output)
        used.add(any_plan.input_value)
        orphans = [v.id for v in any_plan.values if v.id not in used]
        assert not orphans, f'どのopからも参照されない値: {orphans}'

    def test_kinds_are_from_the_closed_set(self, any_plan):
        allowed = {KIND_BITPLANE, KIND_FLAT, KIND_SCORES, KIND_SCRATCH}
        for v in any_plan.values:
            assert v.kind in allowed, f'未知の値種別 {v.kind}'

    def test_ops_carry_their_origin(self, any_plan):
        """opは由来ノードを覚えている(未対応の説明・プロファイルに要る)。"""
        for op in any_plan.ops:
            if op.kind in ('pack', 'unpack', 'to_scores'):
                continue          # 値の変換はノード由来でない
            assert op.node_id, f'{op.kind} が由来ノードを持たない'


class TestMetalKernelNamesCannotDrift:
    """Metalのカーネル登録漏れは実行時まで気づけないので、静的に固定する。

    以前はパイプラインを作るカーネル名が `mdl_metal.mm` に手書きで並んでいて、
    `.metal` にカーネルを足しても登録を忘れると**黙って動かない**状態だった。
    いまは `functionNames` から作るので取りこぼしようが無いが、逆向き
    (呼び出しているのに定義が無い)はまだ静的にしか見つからない。
    """

    def _text(self, rel: str) -> str:
        return (RUNTIME / 'metal' / rel).read_text(encoding='utf-8')

    def test_dispatched_kernels_are_defined(self):
        defined = set(re.findall(r'kernel\s+void\s+(mdl_k_\w+)',
                                 self._text('mdl_metal_entry.metal')))
        dispatched = set(re.findall(r'dispatch\(@"(mdl_k_\w+)"',
                                    self._text('mdl_metal.mm')))
        assert dispatched, 'dispatch 呼び出しが1つも見つからない(正規表現の腐り)'
        missing = sorted(dispatched - defined)
        assert not missing, (
            f'ホストが呼ぶのに .metal に定義が無いカーネル: {missing}')

    def test_pipelines_are_built_from_the_library(self):
        """カーネル名の手書きリストが復活していないこと。"""
        src = self._text('mdl_metal.mm')
        assert '[g.library functionNames]' in src, (
            'パイプラインをライブラリの関数一覧から作っていない'
            '(手書きリストは登録漏れを実行時まで隠す)')
