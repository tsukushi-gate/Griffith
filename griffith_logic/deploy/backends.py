"""バックエンドのレジストリと能力申告。

ONNX Runtime の実行プロバイダは「自分が実行できるノード」をランタイムへ申告し、
残りは既定プロバイダへ落ちる。同じ考え方で、各バックエンドが**実行できる op
種別**を申告する。おかげで

    ・「このモデルはどのバックエンドで動くか」を生成前に答えられる
    ・未対応の理由が(ノード型・バックエンド・理由)の形で1箇所に出る

新しいバックエンドを足すときは、`GpuBackend` か `CBackend` を継承して
op → コード生成の表を埋め、ここに1行登録する。
"""

from .cbackend import CBackend, generate_c
from .cudabackend import CudaBackend, generate_cuda
from .metalbackend import MetalBackend, generate_metal
from .plan import NODE_KINDS, UNSUPPORTED_NODE_TYPES, UnsupportedNode, build_plan
from .webbackend import WebBackend, generate_web

#: バックエンド名 → (クラス, バンドル→ソース の関数)
BACKENDS: dict[str, tuple] = {}


def register_backend(name: str, cls, generate) -> None:
    BACKENDS[name] = (cls, generate)


register_backend('c', CBackend, generate_c)
register_backend('cuda', CudaBackend, generate_cuda)
register_backend('metal', MetalBackend, generate_metal)
register_backend('web', WebBackend, generate_web)


def capabilities(name: str) -> dict:
    """バックエンドが実行できる op 種別と下位種別。

    op種別は**コード生成表からそのまま導く**ので、実装と申告がずれない。
    """
    cls = BACKENDS[name][0]
    return {
        'ops': frozenset(cls.EMITTERS),
        'sub_ops': frozenset(getattr(cls, 'SUB_OPS', ())),
    }


def supported_node_types() -> frozenset:
    """プラン層が扱えるIRノード型。"""
    return frozenset(NODE_KINDS)


def unsupported_node_types() -> dict[str, str]:
    """扱えないIRノード型と、その理由。"""
    return dict(UNSUPPORTED_NODE_TYPES)


def can_run(bundle: dict, name: str) -> tuple[bool, str]:
    """このバンドルを指定バックエンドで生成できるか。

    Returns:
        (可否, 駄目な理由)。可のときの理由は空文字列
    """
    try:
        plan = build_plan(bundle)
    except UnsupportedNode as exc:
        return False, str(exc)
    cap = capabilities(name)
    for op in plan.ops:
        if op.kind not in cap['ops']:
            return False, str(UnsupportedNode(
                f'op種別 {op.kind} は未対応', node_id=op.node_id,
                node_type=op.node_type, backend=name))
        sub = op.attrs.get('op')
        if sub is not None and sub not in cap['sub_ops']:
            return False, str(UnsupportedNode(
                f'{op.kind} の {sub} は未対応', node_id=op.node_id,
                node_type=op.node_type, backend=name))
    return True, ''
