"""griffith benchのテスト。

logic_xorレシピを対象に、--skip-trainベンチのJSONスキーマと
決定的な値(ゲート数)を検証する。学習込み・チェックポイント指定は@slow。
"""

import json
from pathlib import Path

import pytest

from griffith_logic.cli import main

REPO_ROOT = Path(__file__).resolve().parents[1]
RECIPE = REPO_ROOT / 'configs' / 'recipes' / 'logic_xor.yaml'

# logic_xorレシピのゲート数: LutLayer(2,16)=16 + LutLayer(16,1)=1
XOR_RECIPE_GATES = 17


def _load_single_result(out_dir: Path) -> dict:
    files = sorted(out_dir.glob('*.json'))
    assert len(files) == 1, f'結果JSONが1件のはず: {files}'
    return json.loads(files[0].read_text(encoding='utf-8'))


def _assert_schema(data: dict) -> None:
    """固定スキーマの検証"""
    assert set(data) == {'name', 'timestamp', 'config', 'metrics', 'env'}
    assert set(data['metrics']) == {
        'hard_acc', 'total_gates', 'latency_ms_b1', 'latency_ms_b128',
    }
    assert set(data['env']) == {'python', 'torch', 'platform'}
    assert isinstance(data['config'], dict)
    assert data['metrics']['latency_ms_b1'] > 0
    assert data['metrics']['latency_ms_b128'] > 0


def test_bench_skip_train_schema(tmp_path):
    """--skip-train: 妥当なJSON(スキーマ・レイテンシ>0・ゲート数)を出力する"""
    rc = main([
        'bench', str(RECIPE), '--skip-train', '--out-dir', str(tmp_path),
    ])
    assert rc == 0

    data = _load_single_result(tmp_path)
    _assert_schema(data)
    assert data['name'] == 'logic_xor'
    assert data['metrics']['hard_acc'] is None  # 未学習のため計測しない
    assert data['metrics']['total_gates'] == XOR_RECIPE_GATES


def test_bench_missing_source(tmp_path):
    """存在しないパスはエラー終了(exit code 1)"""
    rc = main(['bench', str(tmp_path / 'nai.yaml'), '--skip-train'])
    assert rc == 1


@pytest.mark.slow
def test_bench_with_training(tmp_path):
    """学習込みベンチ: hard_acc=1.0(レシピのexpected_metricsと同水準)。

    続けて生成されたbestチェックポイント指定のベンチも検証する。
    """
    out_dir = tmp_path / 'recipe'
    rc = main(['bench', str(RECIPE), '--out-dir', str(out_dir)])
    assert rc == 0

    data = _load_single_result(out_dir)
    _assert_schema(data)
    assert data['metrics']['hard_acc'] == 1.0
    assert data['metrics']['total_gates'] == XOR_RECIPE_GATES

    # チェックポイント指定(学習out_dirはベンチout_dir配下に作られる)
    ckpt = out_dir / 'train_logic_xor' / 'checkpoints' / 'best.pt'
    assert ckpt.is_file()
    ckpt_out = tmp_path / 'ckpt'
    rc = main([
        'bench', str(ckpt), '--out-dir', str(ckpt_out), '--name', 'xor_ckpt',
    ])
    assert rc == 0

    ckpt_data = _load_single_result(ckpt_out)
    _assert_schema(ckpt_data)
    assert ckpt_data['name'] == 'xor_ckpt'
    assert ckpt_data['metrics']['hard_acc'] == 1.0
    assert ckpt_data['metrics']['total_gates'] == XOR_RECIPE_GATES
