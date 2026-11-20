"""packed LUT proxy: yosys+ABCテクノロジマッピングによるLUT数見積り(SM3)。

実EDA(Quartus/Vivado/Gowin)のコンパイルなしで、合成段のLUTパッキング結果を
数秒で近似する。配置配線は行わないためタイミングは対象外(SM6が担当)。

パイプライン:
    IR → export/verilog.py → yosys「read_verilog; synth -flatten -lut K」
       → ltp(論理深度) → stat -json($lutセル数)

指標の分離(構造メトリクスの設計方針):
    - 掃除率: sweep(SM2)のsweep_ratio(EDAの定数伝播・恒等除去に相当する分)
    - パッキング率 gates_per_lut = swept_gates / lut_count
    yosysへは**元のIR**を渡す(実EDAフローと同じく掃除も合成側にさせる)。

較正: Quartus実測(2入力ゲート約4個/LUT6、共有なし理論上限5個)と一度
突き合わせ、差分は開発記録に記録して以後は相対値で運用する。

yosysの解決順: 引数 > 環境変数 MDL_YOSYS > PATH(shutil.which)。
未検出は明示エラー(黙ってスキップしない)。oss-cad-suiteの自動導入は
tools/mdl-fpga の `eda install` を参照(マージ後に共通化予定)。
"""

import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

DEFAULT_K = 6  # Intel/Xilinx=6、Gowin GW2A=4

_LTP_RE = re.compile(r'length=(\d+)')


def find_yosys(explicit: str | None = None) -> str:
    """yosys実行ファイルを解決する。見つからなければRuntimeError。"""
    candidates = [explicit, os.environ.get('MDL_YOSYS'), shutil.which('yosys')]
    for c in candidates:
        if c and Path(c).exists():
            return str(c)
    raise RuntimeError(
        'yosysが見つかりません。PATHに追加するか環境変数 MDL_YOSYS で'
        '実行ファイルを指定してください(oss-cad-suite導入は '
        'tools/mdl-fpga の eda install を参照)。'
    )


def _yosys_env(yosys_bin: str) -> dict | None:
    """oss-cad-suite配置(bin/の隣にlib/)ならPATHへbin+libを前置した環境を返す。

    Windowsのoss-cad-suiteはyosys.exeが lib/ のDLL(libreadline8.dll等)に
    依存し、単体起動ではロードに失敗する。mdl-fpgaの `_oss_env()` と同じ
    PATH罠対策。通常配置(PATH導入済み等)ではNone=環境を変えない。
    """
    bin_dir = Path(yosys_bin).resolve().parent
    lib_dir = bin_dir.parent / 'lib'
    if not lib_dir.is_dir():
        return None
    env = os.environ.copy()
    env['PATH'] = os.pathsep.join(
        [str(bin_dir), str(lib_dir), env.get('PATH', '')]
    )
    return env


def parse_stat_json(text: str) -> dict:
    """`stat -json` の出力からLUTセル数・総セル数を抽出する(純パーサ)。"""
    data = json.loads(text)
    lut_count = 0
    cell_count = 0
    for mod in data.get('modules', {}).values():
        by_type = mod.get('num_cells_by_type', {})
        lut_count += int(by_type.get('$lut', 0))
        cell_count += int(mod.get('num_cells', 0))
    return {'lut_count': lut_count, 'cell_count': cell_count}


def parse_ltp_depth(log: str) -> int | None:
    """yosys `ltp` のログから最長トポロジカルパス長を抽出する(純パーサ)。"""
    m = _LTP_RE.search(log)
    return int(m.group(1)) if m else None


def yosys_lut_map(
    verilog_path: str,
    k: int = DEFAULT_K,
    top: str = 'logic_core',
    yosys: str | None = None,
    timeout: float = 600.0,
) -> dict:
    """Verilogファイルをyosys+ABCでK入力LUTへマップし統計を返す。

    Returns:
        {'lut_count', 'cell_count', 'depth', 'k'}
        depth = LUT段数の最長トポロジカルパス(ltp。取れなければNone)
    """
    yosys_bin = find_yosys(yosys)
    vpath = Path(verilog_path).resolve()
    stat_path = vpath.parent / f'{vpath.stem}_stat.json'
    script = '; '.join([
        f'read_verilog {vpath.as_posix()}',
        f'synth -top {top} -flatten -lut {k}',
        'ltp -noff',
        f'tee -o {stat_path.as_posix()} stat -json',
    ])
    proc = subprocess.run(
        [yosys_bin, '-p', script],
        capture_output=True, text=True, timeout=timeout,
        env=_yosys_env(yosys_bin),
    )
    if proc.returncode != 0:
        tail = (proc.stdout + proc.stderr)[-2000:]
        raise RuntimeError(f'yosys失敗(exit={proc.returncode}):\n{tail}')
    stats = parse_stat_json(stat_path.read_text(encoding='utf-8'))
    stats['depth'] = parse_ltp_depth(proc.stdout)
    stats['k'] = int(k)
    return stats


def packed_lut_stats(
    graph,
    k: int = DEFAULT_K,
    top: str = 'logic_core',
    yosys: str | None = None,
    workdir: str | None = None,
    timeout: float = 600.0,
) -> dict:
    """IRのpacked LUT数と、掃除率/パッキング率の分離レポートを返す。

    Args:
        graph: IRGraph(元のIR。yosysへは無最適化のまま渡す)
        k: LUT入力数(6=Intel/Xilinx、4=Gowin GW2A)
        workdir: 中間Verilog/JSONの出力先。省略時は一時ディレクトリ
            (呼び出し側で成果物を残したい場合のみ指定する)

    Returns:
        {'lut_count', 'cell_count', 'depth', 'k',
         'gates_per_lut',   # sweep後ゲート数 / LUT数(パッキング率)
         'raw_gates_per_lut',  # sweep前ゲート数 / LUT数(参考値)
         'sweep': sweepレポート(掃除率sweep_ratio等、SM2)}
    """
    from ..export.verilog import to_verilog
    from ..ir.sweep import sweep

    _, sweep_report = sweep(graph)

    def _run(dirpath: str) -> dict:
        vpath = Path(dirpath) / f'{top}.v'
        vpath.write_text(to_verilog(graph, module_name=top), encoding='utf-8')
        return yosys_lut_map(str(vpath), k=k, top=top, yosys=yosys,
                             timeout=timeout)

    if workdir is not None:
        Path(workdir).mkdir(parents=True, exist_ok=True)
        stats = _run(workdir)
    else:
        with tempfile.TemporaryDirectory(prefix='mdl_lut_pack_') as tmp:
            stats = _run(tmp)

    lut = stats['lut_count']
    stats['gates_per_lut'] = (
        sweep_report['swept_gates'] / lut if lut else None
    )
    stats['raw_gates_per_lut'] = (
        sweep_report['gates_before'] / lut if lut else None
    )
    stats['sweep'] = sweep_report
    return stats
