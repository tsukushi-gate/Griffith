"""Modal実行基盤: 既存のexperiments/train_*.pyを無改造でクラウドGPU実行する。

RunPodの「/workspaceに全部置く」構成をModal Volumeで再現し、
os.chdirするだけで data-*/checkpoints/logs の相対パスが全て
永続Volumeに載る。訓練スクリプト側の変更は不要。

使い方:
    # ローカル(従来どおり、無変更)
    uv run python experiments/train.py --dataset mnist -e 20

    # Modal(同じスクリプト・同じ引数)
    uv run modal run scripts/modal_app.py --script train.py --args "--dataset mnist -e 20"

    # GPU指定・デタッチ実行(長時間訓練)。--background は必ず --detach とセット
    # (detachなしだとentrypoint終了でephemeralアプリごとジョブが落ちる)
    uv run modal run --detach scripts/modal_app.py --background \\
        --gpu H100 --script train_cnn.py --args "--dataset cifar10 -k 64 -e 100"

    # LL-ViT CIFAR-10再現(既定GPUはA10G。full runはA100推奨)
    # --timeout の上限は Modal サーバ側で 86400秒(24h)。超える値は InvalidError で拒否される
    uv run modal run --detach scripts/modal_app.py --background \\
        --gpu A100-80GB --timeout 86400 \\
        --script train_llvit.py --args "--config configs/recipes/llvit_cifar10_repro.yaml"

    # GPU上でのテスト実行(CUDA parity等)
    uv run modal run scripts/modal_app.py::pytest_cuda

    # griffith CLIからの利用(train_config / sweep_agent を内部でspawnする)
    uv run griffith launch --grid "optim.lr=[0.001,0.01]" \
        configs/recipes/dlgn_mnist_quick.yaml --modal
    uv run griffith sweep configs/sweeps/example_mnist.yaml --count 4 --modal

セットアップ手順: https://docs.griffith.magicdelta.jp/guides/cloud-gpu/
"""

import os

import modal

app = modal.App('magicdelta-logic')

# @app.function デコレータ引数のため modal run 時にローカルで評価される
GPU = os.environ.get('MODAL_GPU', 'A10G')
TIMEOUT = int(os.environ.get('MODAL_TIMEOUT', str(6 * 3600)))

# nvcc入りCUDAイメージ(griffith_logic/cuda/lut_kernels.cu のJITコンパイルに必要)
# uv_sync(): pyproject.toml + uv.lock から依存解決(Linux解決 = cu128版torch)。
# ソースコードは add_local_dir でマウント(非copy)するため、
# コード変更のたびにイメージ再ビルドは走らない
# イメージのenv。CUDAカーネルJITのキャッシュはコンテナローカル(/tmp)に置く:
# Volume上のninjaロック競合による無限ハングを回避(MODAL_KERNEL_ISSUE.md参照)。
# トレードオフ: コールドスタート毎にカーネル再コンパイル(~2-3分/回)。
_image_env = {
    'TZ': 'Asia/Tokyo',
    'TORCH_EXTENSIONS_DIR': '/tmp/torch_ext',
    # 論文水準モード等のテスト用フラグをローカル環境から透過
    'DLGN_FULL': os.environ.get('DLGN_FULL', '0'),
    'LLVIT_FULL': os.environ.get('LLVIT_FULL', '0'),
    'WANDB_PROJECT': os.environ.get('WANDB_PROJECT', 'griffith-logic'),
    # 重みArtifactの継続アップロード方針(off/best/all)をローカルから透過
    'WANDB_ARTIFACTS': os.environ.get('WANDB_ARTIFACTS', 'best'),
    'MODAL_REQUESTED_GPU': GPU,
}
# TORCH_CUDA_ARCH_LISTがローカルで設定されていれば透過(nvccプローブのハング回避)。
# デフォルトは設定しない(GPUタイプ可変のため。未指定時はnvccの自動プローブに任せる)。
_arch_list = os.environ.get('TORCH_CUDA_ARCH_LIST')
if _arch_list:
    _image_env['TORCH_CUDA_ARCH_LIST'] = _arch_list
# PYTORCH_CUDA_ALLOC_CONF(expandable_segments等)をローカルから透過。
# 断片化によるOOM回避に使うため、コンテナ側で設定できる必要がある。
_alloc_conf = os.environ.get('PYTORCH_CUDA_ALLOC_CONF')
if _alloc_conf:
    _image_env['PYTORCH_CUDA_ALLOC_CONF'] = _alloc_conf

image = (
    modal.Image.from_registry(
        'nvidia/cuda:12.8.1-devel-ubuntu24.04', add_python='3.11',
    )
    .apt_install('iverilog', 'ghdl')  # @hdl/@hdl_vhdl テスト用シミュレータ
    # teachers: hf:公開teacher(D6)
    # wmt:/RDDLGNのWMT'14翻訳runに必要(datasets=データ取得, sacrebleu=BLEU)。
    # 無いとBLEUが無言スキップされ、キャッシュ未生成のModalではデータ取得自体が失敗する。
    .uv_sync(extras=['dev', 'cu128', 'teachers', 'wmt'])
    # envは_image_envへ集約(JITキャッシュ/tmp化=根本対応を維持)
    .env(_image_env)
    .add_local_dir('griffith_logic', '/root/griffith_logic')
    .add_local_dir('experiments', '/root/experiments')
    .add_local_dir('configs', '/root/configs')
    .add_local_dir('tests', '/root/tests')
    .add_local_file('pyproject.toml', '/root/pyproject.toml')
)

# Volume 名は MD_MODAL_VOLUME で上書きできる(共有アカウントでの衝突回避用)
workspace = modal.Volume.from_name(
    os.environ.get('MD_MODAL_VOLUME', 'griffith-workspace'), create_if_missing=True)


@app.function(
    image=image,
    gpu=GPU,
    timeout=TIMEOUT,
    volumes={'/workspace': workspace},
    secrets=[modal.Secret.from_name('wandb')],
)
def train(script: str, argv: list[str], requested_gpu: str | None = None):
    """experiments/配下のスクリプトをrunpyで実行する汎用ランナー"""
    import runpy
    import sys

    # ローカルの `python experiments/train.py` と同じsys.path構成を再現する
    # (runpy.run_pathはスクリプトのディレクトリをsys.pathに追加しない)
    sys.path.insert(0, '/root')              # griffith_logic インポート用
    sys.path.insert(0, '/root/experiments')  # checkpoint_utils等の同階層import用
    os.chdir('/workspace')       # 相対パス(data-*/checkpoints/logs)をVolumeへ
    if not os.path.exists('/workspace/configs'):
        os.symlink('/root/configs', '/workspace/configs')
    if requested_gpu:
        os.environ['MODAL_GPU'] = requested_gpu
        os.environ['MODAL_REQUESTED_GPU'] = requested_gpu
    sys.argv = [script] + argv
    print(
        f'[modal] GPU={os.environ.get("MODAL_REQUESTED_GPU", GPU)} '
        f'script={script} argv={argv}'
    )
    try:
        runpy.run_path(f'/root/experiments/{script}', run_name='__main__')
    except SystemExit as e:
        # argparse --help 等の正常終了(code 0/None)は許容
        if e.code not in (0, None):
            raise
    workspace.commit()


@app.function(
    image=image,
    gpu=GPU,
    timeout=TIMEOUT,
    volumes={'/workspace': workspace},
    secrets=[modal.Secret.from_name('wandb')],
)
def train_config(config_yaml: str, overrides: list[str]) -> dict:
    """TrainConfig(YAML本文+ドット記法オーバーライド)でrun_trainingを実行する。

    `griffith launch --modal` が組合せごとに .spawn() する。設定はローカルファイルが
    リモートに存在しないためYAML本文で受け取る。出力は /workspace/outputs に
    永続化され、`griffith collect` で回収できる。
    """
    import sys
    import tempfile

    sys.path.insert(0, '/root')
    os.chdir('/workspace')  # outputs/ 等の相対パスをVolumeへ

    from griffith_logic.training.config import load_config
    from griffith_logic.training.runner import run_training

    path = None
    if config_yaml.strip():
        with tempfile.NamedTemporaryFile(
            'w', suffix='.yaml', delete=False, encoding='utf-8'
        ) as f:
            f.write(config_yaml)
            path = f.name
    cfg = load_config(path, overrides)
    print(f'[modal] GPU={GPU} train_config overrides={overrides}')
    result = run_training(cfg)
    workspace.commit()
    return result


@app.function(
    image=image,
    gpu=GPU,
    timeout=TIMEOUT,
    volumes={'/workspace': workspace},
    secrets=[modal.Secret.from_name('wandb')],
)
def sweep_agent(sweep_id: str, count: int | None = None):
    """wandb sweepのagentをGPUコンテナで実行する。

    `griffith sweep --modal` が並列数ぶん .spawn() する。1試行の内容は
    griffith_logic/training/sweep.py の sweep_train_once
    (sweepパラメータ→TrainConfig→run_training)。
    """
    import sys

    sys.path.insert(0, '/root')
    os.chdir('/workspace')  # outputs/ 等の相対パスをVolumeへ

    from griffith_logic.training.sweep import run_sweep_agent

    print(f'[modal] GPU={GPU} sweep_agent sweep_id={sweep_id} count={count}')
    run_sweep_agent(sweep_id, count=count)
    workspace.commit()


@app.function(
    image=image,
    gpu=GPU,
    timeout=1800,
    volumes={'/workspace': workspace},
)
def pytest_cuda(marker: str = 'cuda', path: str = 'tests/'):
    """GPU必須テスト(@cuda等)をModal上で実行する"""
    import subprocess
    import sys

    os.chdir('/root')
    result = subprocess.run(
        [sys.executable, '-m', 'pytest', path, '-m', marker, '-v'],
    )
    if result.returncode not in (0, 5):  # 5 = テスト0件
        raise RuntimeError(f'pytest failed: exit={result.returncode}')


def _preflight_ok() -> bool:
    """投入前にModal同時実行数をチェックする(scripts/modal_preflight.py)。

    取得失敗・スクリプト欠落時はfail-openでTrue(投入をブロックしない)。
    上限超過時のみFalse。MD_SKIP_PREFLIGHT=1 での無効化はpreflight_check内で処理。
    """
    import importlib.util
    from pathlib import Path

    path = Path(__file__).with_name('modal_preflight.py')
    if not path.is_file():
        return True
    try:
        spec = importlib.util.spec_from_file_location('_md_preflight', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return bool(module.preflight_check())
    except Exception:  # noqa: BLE001 — preflightの失敗で投入自体は止めない
        return True


@app.local_entrypoint()
def main(
    script: str = 'train.py',
    args: str = '',
    gpu: str = '',
    timeout: int = 0,
    background: bool = False,
):
    # GPU同時実行limit(10)超過を防ぐ。上限超過なら投入せず中止(緊急時 MD_SKIP_PREFLIGHT=1)
    if not _preflight_ok():
        raise SystemExit(1)
    selected_gpu = gpu or GPU
    selected_timeout = timeout or TIMEOUT
    runner = train
    if gpu or timeout:
        runner = train.with_options(gpu=selected_gpu, timeout=selected_timeout)
    argv = args.split() if args else []
    # 同期呼び出し(.remote)は禁止: detach時にジョブごとキャンセルされ得る(Modal警告)。
    # 常に .spawn() し、フォアグラウンド時のみ .get() で完了を待つ。
    call = runner.spawn(script, argv, selected_gpu)
    print(
        '[modal] spawned train call '
        f'object_id={getattr(call, "object_id", "")} '
        f'local_uuid={getattr(call, "local_uuid", "")}'
    )
    if not background:
        call.get()
