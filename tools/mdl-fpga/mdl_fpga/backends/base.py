"""合成バックエンドの共通契約(ベンダ中立)。

FlowResult / flow_summary.json が「合成結果」の唯一の受け渡し形。
バックエンド(Quartus / gowin-oss / 将来の vivado 等)はこの契約だけを満たす。
build.py はバックエンドの具象を知らない(backends.get_backend で解決)。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable

from ..toolchain import Toolchain


@dataclass
class FlowResult:
    """合成フロー結果(旧 synth.FlowResult を逐語移設。契約はそのまま)。

    sof フィールド名は歴史的経緯で維持(意味は「主ビットストリームのパス」。
    Quartus=.sof / gowin-oss=.fs)。新規コードは bitstream プロパティを使う。
    """
    ok: bool
    projdir: str
    name: str
    stages: list[dict] = field(default_factory=list)
    summary: dict = field(default_factory=dict)
    sof: str | None = None
    diagnosis: str | None = None

    @property
    def bitstream(self) -> str | None:
        return self.sof

    def render(self) -> str:
        lines = [f'synth: {"OK" if self.ok else "FAIL"}({self.projdir})']
        for s in self.stages:
            lines.append(f'  {s["stage"]}: rc={s["rc"]} {s["seconds"]:.1f}s')
        if self.summary:
            if 'lut' in self.summary:
                lut = self.summary.get('lut')
                lut_a = self.summary.get('lut_avail')
                fmax = self.summary.get('fmax_mhz')
                lines.append(f'  LUT {lut}/{lut_a} / Fmax {fmax}MHz')
            else:
                alm = self.summary.get('alm')
                alm_a = self.summary.get('alm_avail')
                # overlay は分周した clk_core が主クロック(clk50 は入力段のみ)
                fmax = (self.summary.get('fmax_clk50_mhz')
                        or self.summary.get('fmax_clk_core_mhz'))
                slack = (self.summary.get('setup_slack_clk50_ns')
                         if 'setup_slack_clk_core_ns' not in self.summary
                         else self.summary.get('setup_slack_clk_core_ns'))
                lines.append(
                    f'  ALM {alm}/{alm_a} / Fmax {fmax}MHz / slack {slack}ns')
        if self.sof:
            lines.append(f'  SOF: {self.sof}')
        if self.diagnosis:
            lines.append(f'  診断: {self.diagnosis}')
        return '\n'.join(lines)


def write_summary(projdir: Path, res: FlowResult) -> None:
    """flow_summary.json を書く(全バックエンド共通の出力契約)。"""
    doc = {'name': res.name, 'ok': res.ok, 'stages': res.stages,
           **res.summary}
    if res.sof:
        doc['sof'] = res.sof
    if res.diagnosis:
        doc['diagnosis'] = res.diagnosis
    (Path(projdir) / 'flow_summary.json').write_text(
        json.dumps(doc, ensure_ascii=False, indent=2), encoding='utf-8')


@runtime_checkable
class SynthBackend(Protocol):
    """合成バックエンドの契約。

    実装は状態を持たないシングルトンで良い(レジストリはクラスを保持し
    get_backend がインスタンス化する)。
    """
    name: str                        # レジストリキー('quartus' 等)
    bitstream_ext: str               # 主ビットストリーム拡張子('.sof' / '.fs')
    project_subdir: str              # workdir 直下のプロジェクトディレクトリ名
    convert_formats: tuple[str, ...]  # bundle が試す変換形式(無変換は ())

    def available(self, tc: Toolchain) -> tuple[bool, str]:
        """(使えるか, 不足時の初心者向けメッセージ=doctor 誘導)。"""
        ...

    def write_project_files(self, projdir: str | Path, *, name: str,
                            board: dict, sources: list[str],
                            clock_hz: int, groups: tuple,
                            uart: dict | None, clock_div: int = 1) -> None:
        """プロジェクトファイル(qsf/sdc・cst/ys 等)を生成する。

        clock_div > 1 は「トップが内部で clk_core を /N 分周する」の意で、
        制約ファイルに派生クロックを宣言する必要があるバックエンドが使う
        (対応しないバックエンドは 1 以外を拒否してよい)。
        """
        ...

    def run_flow(self, projdir: str | Path, name: str,
                 tc: Toolchain | None = None, *, board: dict,
                 clock_hz: int, clock_div: int = 1) -> FlowResult:
        """合成フローを実行し flow_summary.json を書いて FlowResult を返す。"""
        ...

    def convert_bitstream(self, bitstream: str | Path, fmt: str,
                          tc: Toolchain | None = None) -> Path:
        """主ビットストリームを他形式へ変換(非対応形式は例外で言い切る)。"""
        ...

    def bitstream_globs(self, name_hint: str = '*') -> tuple[str, ...]:
        """workdir からの主ビットストリーム探索パターン(bundle 用)。"""
        ...
