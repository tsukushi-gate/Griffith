"""IR の読込・検証(mdl-fpga convert)。

blackbox ノード(Attention 等の実数部)の検出・明示拒否はここで行う。
エラーは3分類(環境/設計/ツール内部)のうち「設計」に該当し、
メッセージは「原因・次の一手」を必ず含める(構想書 §9)。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from griffith_logic.ir.schema import IRGraph
from griffith_logic.ir.serialize import load_ir, save_ir
from griffith_logic.ir.validate import validate


@dataclass
class ConvertResult:
    ok: bool
    ir_path: str
    messages: list[tuple[str, str]] = field(default_factory=list)  # (level, text)
    graph: IRGraph | None = None
    out_path: str | None = None

    def render(self) -> str:
        lines = [f'convert: {self.ir_path}']
        for level, text in self.messages:
            lines.append(f'  [{level}] {text}')
        lines.append(f'結果: {"OK(HDL化可能なIR)" if self.ok else "NG"}')
        if self.out_path:
            lines.append(f'保存: {self.out_path}')
        return '\n'.join(lines)


def load_graph(path: str | Path) -> IRGraph:
    """IR JSON を読み込む(失敗はそのまま例外。CLI 側で整形)。"""
    return load_ir(str(path))


def convert(path: str | Path, out_path: str | Path | None = None) -> ConvertResult:
    res = ConvertResult(ok=False, ir_path=str(path))

    if not Path(path).exists():
        res.messages.append(('環境', f'ファイルが存在しない: {path}。パスを確認'))
        return res

    try:
        graph = load_ir(str(path))
    except Exception as e:  # noqa: BLE001 - 設計エラーとして整形して返す
        res.messages.append(
            ('設計', f'IR として読めない({type(e).__name__}: {e})。'
                     'magicdelta-ir v0.2.0 の JSON か確認。'
                     'checkpoint からは `griffith export-hdl` 等で IR を抽出してから渡す'))
        return res
    res.graph = graph

    ver = getattr(graph, 'version', '?')
    res.messages.append(('info', f'format/version: magicdelta-ir {ver}'))

    # blackbox = 実数ノード(ハイブリッドモデルの非論理部)→ 明示拒否+誘導
    blackboxes = [n for n in graph.nodes if n.type == 'blackbox']
    if blackboxes:
        names = ', '.join(
            f'{n.id}({n.params.get("torch_class", "?")})' for n in blackboxes[:5])
        res.messages.append(
            ('設計', f'blackbox ノードを検出: {names}'
                     f'{" ほか" if len(blackboxes) > 5 else ""}。'
                     '実数演算を含むモデルは v1 の HDL 化対象外。'
                     '論理ゲート部のみの IR を抽出し直すか、float 汎用 NN は '
                     'hls4ml 等の実数系フローを検討'))
        return res

    try:
        validate(graph, target='generic')
    except Exception as e:  # noqa: BLE001
        res.messages.append(('設計', f'IR 検証エラー: {e}'))
        return res

    # 展開対象のノード型は export/lowering.LOWERABLE_TYPES が単一情報源(複製しない)
    from griffith_logic.export.lowering import LOWERABLE_TYPES
    native = [n.id for n in graph.nodes if n.type in LOWERABLE_TYPES]
    if native:
        res.messages.append(
            ('info', f'展開対象ノード {len(native)} 個を検出 → '
                     'HDL 生成時に lut_layer 列へ自動展開する(lowering)'))

    # HDL 化可能性は lowering 後の形で確認する
    try:
        from griffith_logic.export.lowering import lower_to_primitives
        lowered = lower_to_primitives(graph)
        validate(lowered, target='hdl')
    except Exception as e:  # noqa: BLE001
        res.messages.append(
            ('設計', f'HDL 化不可: {e}。'
                     '再帰モデルの1ステップ IR・add(mode=sum) 等は v1 対象外'))
        return res

    n_luts = sum(n.params.get('out_width', 0)
                 for n in lowered.nodes if n.type == 'lut_layer')
    res.messages.append(('info', f'lowering 後の LUT ゲート総数: {n_luts}'))

    res.ok = True
    if out_path is not None:
        save_ir(graph, str(out_path))
        res.out_path = str(out_path)
    return res
