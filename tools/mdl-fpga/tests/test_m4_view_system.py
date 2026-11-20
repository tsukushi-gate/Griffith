"""E5: ビューア システムモード(SimSession(system_dir)+パス径路ブリッジ)。"""

import importlib.util
import json
import shutil

import numpy as np
import pytest
from conftest import build_groupsum_model, requires_iverilog

from griffith_logic.ir.serialize import ir_from_dict
from griffith_logic.ir.simulate import simulate

from mdl_fpga.hdlgen.system import (
    generate_system,
    generate_system_view_wrapper,
    to_bcd,
    write_system_vectors,
)
from mdl_fpga.hdlgen.wrapper import HDL_DIR
from mdl_fpga.viewer.probes import target_width

requires_cocotb = pytest.mark.skipif(
    importlib.util.find_spec('cocotb') is None,
    reason='cocotb 未導入(uv sync --extra fpga)',
)

N_IMAGES = 4


def _make_system_dir(tmp_path):
    graph = ir_from_dict(build_groupsum_model())
    rng = np.random.default_rng(11)
    images = rng.integers(0, 2, size=(N_IMAGES, 4), dtype=np.uint8)
    preds = np.argmax(simulate(graph, images.astype(np.int64)), axis=1)
    labels = preds.copy()
    labels[0] = 1 - labels[0]   # 1枚だけ不正解に

    gen = generate_system(graph, n_images=N_IMAGES)
    sysdir = tmp_path / 'system'
    sysdir.mkdir()
    (sysdir / 'top.v').write_text(gen.top_verilog, encoding='utf-8')
    (sysdir / 'sys_rom.v').write_text(gen.rom_verilog, encoding='utf-8')
    (sysdir / 'mdl_core_pipe.v').write_text(gen.core.verilog, encoding='utf-8')
    (sysdir / 'top_view.v').write_text(generate_system_view_wrapper(),
                                       encoding='utf-8')
    shutil.copyfile(HDL_DIR / 'seg7.v', sysdir / 'seg7.v')
    (sysdir / 'system_meta.json').write_text(
        json.dumps(gen.system_meta, ensure_ascii=False), encoding='utf-8')
    golden = write_system_vectors(graph, images, labels, sysdir,
                                  gen.label_bits)
    return sysdir, golden, gen


def test_meta_merge_shape(groupsum_path, tmp_path):
    """メタ合成(シム起動なし): path 付与・system ノード・set_x 禁止。"""
    from mdl_fpga.hdlgen.core import generate_core
    from mdl_fpga.ir_io import load_graph
    from mdl_fpga.viewer.session import build_viewer_meta

    sysdir, _, gen = _make_system_dir(tmp_path)
    system_meta = json.loads(
        (sysdir / 'system_meta.json').read_text(encoding='utf-8'))
    graph = load_graph(groupsum_path)
    core = generate_core(graph, pipelined=True)
    meta = build_viewer_meta(graph, core, system=system_meta)

    assert meta['mode'] == 'system' and meta['allow_set_x'] is False
    assert meta['latency'] == gen.latency
    by_id = {n['id']: n for n in meta['nodes']}
    assert by_id['gates']['path'] == 'u_sys.u_core.w_gates'
    assert by_id['img_idx']['path'] == 'u_sys.img_idx'
    assert by_id['fsm']['width'] == 3
    # プローブ解決はシステム信号にもそのまま効く
    assert target_width(meta, 'correct') == 16
    assert target_width(meta, 'img_idx') == by_id['img_idx']['width']
    # エッジ合成(rom系 + コア系の両方)
    pairs = {(e['src'], e['dst']) for e in meta['edges']}
    assert ('img_idx', 'x') in pairs and ('x', 'gates') in pairs


@requires_iverilog
@requires_cocotb
def test_system_session_end_to_end(groupsum_path, tmp_path):
    """システム全体を1クロックずつ進め、完走時のカウンタが golden と一致。"""
    from mdl_fpga.viewer.session import SimSession

    sysdir, golden, gen = _make_system_dir(tmp_path)
    session = SimSession(model=str(groupsum_path),
                         workdir=tmp_path / 'view', system_dir=sysdir)
    try:
        meta = session.start()
        assert meta['mode'] == 'system'
        # リスタートは KEY 2FF 同期のため反映に数クロックかかる(実機仕様)
        session.reset()
        r = session.step(3)
        assert int(r['values']['img_idx'], 16) == 0
        assert int(r['values']['correct'], 16) == 0

        # set_x はシステムモードでは拒否される
        rj = session.set_x('f')
        assert rj['ok'] is False and 'ROM' in rj['error']

        # 完走まで step(N枚 × (latency+4) + 余裕)
        steps = N_IMAGES * (gen.latency + 6) + 10
        r = session.step(steps)
        assert int(r['values']['fsm'], 16) == 6   # S_DONE
        assert int(r['values']['correct'], 16) == to_bcd(golden['correct'], 4)
        assert int(r['values']['img_idx'], 16) == N_IMAGES - 1
        # コア層の値も読めている(path 経由)
        assert 'gates' in r['values'] and 'gs' in r['values']

        # リセット(=KEY[0]リスタート)でカウンタが戻る
        session.reset()
        r = session.step(3)
        assert int(r['values']['correct'], 16) == 0
        assert int(r['values']['fsm'], 16) != 6
    finally:
        session.close()
