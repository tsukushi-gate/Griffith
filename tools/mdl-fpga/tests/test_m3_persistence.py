"""レイアウト永続化(サイドカー JSON)。"""

from mdl_fpga.viewer.persistence import (
    ViewerConfig,
    load_config,
    save_config,
    sidecar_path,
)
from mdl_fpga.viewer.probes import ProbeSpec


def test_sidecar_path_variants():
    assert sidecar_path('a/b/model.ir.json').name == 'model.viewer.json'
    assert sidecar_path('model.json').name == 'model.viewer.json'
    assert sidecar_path('model.bin').name == 'model.bin.viewer.json'


def test_roundtrip_with_slice_alias(tmp_path):
    path = tmp_path / 'm.viewer.json'
    cfg = ViewerConfig(
        probes=[ProbeSpec(id='p1', target='gates', slice=[7, 0],
                          renderer='bars', pinned=True,
                          pos={'x': 10.5, 'y': -3.0})],
        positions={'gates': {'x': 1.0, 'y': 2.0}},
        expanded=['gates'],
        viewport={'x': 0, 'y': 0, 'zoom': 1.5},
    )
    save_config(path, cfg)
    text = path.read_text(encoding='utf-8')
    assert '"slice"' in text            # by_alias で保存される
    loaded = load_config(path)
    assert loaded == cfg
    assert loaded.probes[0].bit_slice == [7, 0]
    assert not path.with_suffix('.json.tmp').exists()


def test_missing_and_corrupt_files(tmp_path):
    assert load_config(tmp_path / 'nope.viewer.json') == ViewerConfig()
    bad = tmp_path / 'bad.viewer.json'
    bad.write_text('{invalid json', encoding='utf-8')
    assert load_config(bad) == ViewerConfig()
    bad.write_text('{"probes": "not-a-list"}', encoding='utf-8')
    assert load_config(bad) == ViewerConfig()
