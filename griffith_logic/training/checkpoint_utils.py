"""チェックポイント保存・読み込みユーティリティ(パッケージ内正本)。

このモジュールは `experiments/` からパッケージ内(`griffith_logic.training`)へ
移設した。`experiments/checkpoint_utils.py` は後方互換の再exportシムとして残っている。

セキュリティ: `torch.load` は既定で ``weights_only=True``
(テンソル・基本型のみを復元する安全モード)を使う。pickle の任意オブジェクト復元が
必要な古いチェックポイントを読むときだけ ``allow_unsafe=True`` を明示するか、
環境変数 ``GRIFFITH_ALLOW_UNSAFE_CKPT=1`` を設定してオプトインする。
"""

import glob
import os
import re
from datetime import datetime, timedelta, timezone

import torch

JST = timezone(timedelta(hours=9))

# 環境変数によるオプトイン(呼び出し側を書き換えられない実験スクリプト向け)
UNSAFE_CKPT_ENV = 'GRIFFITH_ALLOW_UNSAFE_CKPT'


def _weights_only(allow_unsafe: bool) -> bool:
    """`torch.load(weights_only=...)` に渡す値を決める。

    既定は安全側(True)。`allow_unsafe=True` か環境変数
    ``GRIFFITH_ALLOW_UNSAFE_CKPT=1`` の明示オプトイン時のみ False にする。
    """
    if allow_unsafe:
        return False
    return os.environ.get(UNSAFE_CKPT_ENV, '') not in ('1', 'true', 'True')


def torch_load(path: str, map_location='cpu', *, allow_unsafe: bool = False):
    """`torch.load` の安全既定ラッパ(weights_only=True)。

    weights_only=True で読めない古い形式(numpy配列やdataclassを直接pickleした
    metadata等)の場合は、原因と対処(``allow_unsafe=True`` /
    ``GRIFFITH_ALLOW_UNSAFE_CKPT=1``)を添えて再raiseする。
    """
    weights_only = _weights_only(allow_unsafe)
    try:
        return torch.load(path, map_location=map_location, weights_only=weights_only)
    except OSError:
        raise  # ファイル欠落・読み取り不可は呼び出し側のハンドリングに委ねる
    except Exception as e:
        if not weights_only:
            raise
        raise RuntimeError(
            f'チェックポイントを安全モード(weights_only=True)で読めませんでした: {path}\n'
            f'  原因: {e}\n'
            f'  信頼できるファイルであれば allow_unsafe=True を渡すか、'
            f'環境変数 {UNSAFE_CKPT_ENV}=1 を設定してください'
            f'(pickleの任意コード実行を許可します)。'
        ) from e

# setup_checkpoint_dir が付与する run_dir 末尾のタイムスタンプ（YYYYmmdd_HHMMSS）
_RUN_DIR_TIMESTAMP_RE = re.compile(r'^\d{8}_\d{6}$')


def _restore_dynamic_buffers(model: torch.nn.Module, state_dict: dict) -> None:
    """Restore buffers that are absent from a fresh module state_dict.

    ``ThermometerEncoder`` registers ``thresholds`` as ``None`` before fitting,
    so fresh models do not expose those keys to ``state_dict()``.  Checkpoints
    from fitted LL-ViT runs do include them; install them before strict loading.
    """
    modules = dict(model.named_modules())
    for key, value in state_dict.items():
        if not key.endswith('.thresholds') or not torch.is_tensor(value):
            continue
        module_name = key.rsplit('.', 1)[0]
        module = modules.get(module_name)
        if module is not None and hasattr(module, 'thresholds'):
            module.thresholds = value.detach().clone()


def setup_checkpoint_dir(prefix: str, base_dir: str = 'checkpoints') -> tuple[str, str, str]:
    """チェックポイントディレクトリを作成。

    Args:
        prefix: ディレクトリ名のプレフィックス（例: 'rddlgn_copy'）
        base_dir: ベースディレクトリ

    Returns:
        (run_dir, latest_path, best_path)
    """
    timestamp = datetime.now(JST).strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join(base_dir, f'{prefix}_{timestamp}')
    os.makedirs(run_dir, exist_ok=True)
    latest_path = os.path.join(run_dir, 'latest.pt')
    best_path = os.path.join(run_dir, 'best.pt')
    return run_dir, latest_path, best_path


def save_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    step: int,
    best_metric: float,
    metadata: dict,
    extra_state: dict | None = None,
) -> None:
    """チェックポイントを保存。

    Args:
        path: 保存先パス
        model: モデル
        optimizer: オプティマイザ
        scheduler: スケジューラ（Noneの場合あり）
        step: 現在のステップ数
        best_metric: ベストメトリクス値
        metadata: スクリプト固有の情報（args, config等）
    """
    data = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'step': step,
        'best_metric': best_metric,
        'metadata': metadata,
    }
    if extra_state:
        data['extra_state'] = extra_state
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(data, path)


def load_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler=None,
    device: str = 'cpu',
    *,
    allow_unsafe: bool = False,
) -> dict:
    """チェックポイントを読み込み、state_dictを復元。

    Args:
        path: チェックポイントファイルパス
        model: モデル（state_dictを読み込む先）
        optimizer: オプティマイザ（Noneなら復元しない）
        scheduler: スケジューラ（Noneなら復元しない）
        device: デバイス
        allow_unsafe: Trueならpickleの任意オブジェクト復元を許可する
            (既定はweights_only=Trueの安全モード)

    Returns:
        {'step': int, 'best_metric': float, 'metadata': dict}
    """
    data = torch_load(path, map_location=device, allow_unsafe=allow_unsafe)
    _restore_dynamic_buffers(model, data['model_state_dict'])
    model.load_state_dict(data['model_state_dict'])
    if optimizer is not None and 'optimizer_state_dict' in data:
        optimizer.load_state_dict(data['optimizer_state_dict'])
    if scheduler is not None and data.get('scheduler_state_dict') is not None:
        scheduler.load_state_dict(data['scheduler_state_dict'])
    return {
        'step': data['step'],
        'best_metric': data['best_metric'],
        'metadata': data.get('metadata', {}),
        'extra_state': data.get('extra_state', {}),
    }


def _remap_key(
    key: str, key_map: dict[str, str] | list[tuple[str, str]] | None,
) -> str:
    """モデル側キー名 → ckpt側キー名へprefix変換する(最初にマッチした規則を使う)。

    key_map の各エントリは (model側prefix, ckpt側prefix) のペア。
    keyがmodel側prefixで前方一致すれば、その部分をckpt側prefixへ置換する。
    どの規則にもマッチしなければkeyをそのまま返す(素通し、W13a-4のCIFAR
    事前学習エンコーダのようにキー名がそもそも一致している場合はkey_map不要)。
    """
    if not key_map:
        return key
    items = key_map.items() if isinstance(key_map, dict) else key_map
    for model_prefix, ckpt_prefix in items:
        if key.startswith(model_prefix):
            return ckpt_prefix + key[len(model_prefix):]
    return key


def load_partial_state_dict(
    model: torch.nn.Module,
    ckpt_state: dict,
    key_map: dict[str, str] | list[tuple[str, str]] | None = None,
    strict_shapes: bool = True,
) -> dict:
    """部分ロード: キー名リマップ + 形状一致キーのみロードする(W13a-3)。

    完全一致state_dict専用のload_checkpointと異なり、異構造間の重み移植
    (例: CIFAR事前学習エンコーダ → det/segバックボーンの一部)を想定する。
    モデル側state_dictの各キーについて、key_mapでprefix変換した名前を
    ckpt_stateから探し、形状が一致すればロードする。一致しないキー
    (ckpt側に無い/形状不一致)はスキップし、ロード率をログとして返す。

    Args:
        model: ロード先モデル(state_dict()のキー集合が「求める全体」)
        ckpt_state: チェックポイントのstate_dict(通常
            torch.load(...)['model_state_dict'])
        key_map: {model側prefix: ckpt側prefix} のdict、または
            [(model側prefix, ckpt側prefix), ...] のリスト。
            Noneならリマップなし(キー名がそのまま一致する前提)
        strict_shapes: True(既定)なら形状不一致キーは常にスキップする
            (将来reshapeロード等を許容する拡張点として引数を残すが、
            現状Falseでも同じ挙動＝形状不一致の自動reshapeは行わない)

    Returns:
        {
            'loaded_keys': list[str]（ロードできたモデル側キー名）,
            'skipped_missing': list[str]（ckpt側に対応キーが無かったもの）,
            'skipped_shape_mismatch': list[tuple[str, tuple, tuple]]
                （キー名, モデル側shape, ckpt側shape）,
            'total_keys': int（モデル側state_dictの全キー数）,
            'load_ratio': float（loaded_keys数 / total_keys、0件ならNoneでなく0.0）,
        }
    """
    model_state = model.state_dict()
    loaded_keys: list[str] = []
    skipped_missing: list[str] = []
    skipped_shape_mismatch: list[tuple[str, tuple, tuple]] = []
    new_state = {}

    for key, tensor in model_state.items():
        src_key = _remap_key(key, key_map)
        if src_key not in ckpt_state:
            skipped_missing.append(key)
            continue
        ckpt_tensor = ckpt_state[src_key]
        model_shape = tuple(tensor.shape)
        ckpt_shape = tuple(ckpt_tensor.shape)
        if model_shape != ckpt_shape:
            skipped_shape_mismatch.append((key, model_shape, ckpt_shape))
            continue
        new_state[key] = ckpt_tensor
        loaded_keys.append(key)

    model.load_state_dict(new_state, strict=False)

    total_keys = len(model_state)
    return {
        'loaded_keys': loaded_keys,
        'skipped_missing': skipped_missing,
        'skipped_shape_mismatch': skipped_shape_mismatch,
        'total_keys': total_keys,
        'load_ratio': (len(loaded_keys) / total_keys) if total_keys else 0.0,
    }


def checkpoint_paths(run_dir: str) -> tuple[str, str]:
    """既存 run_dir に対する (latest_path, best_path) を返す。"""
    return os.path.join(run_dir, 'latest.pt'), os.path.join(run_dir, 'best.pt')


def _normalize(value):
    """metadata比較用にJSON的な等価性へ正規化する（tuple/listの差を吸収）。"""
    if isinstance(value, list | tuple):
        return [_normalize(v) for v in value]
    if isinstance(value, dict):
        return {k: _normalize(v) for k, v in value.items()}
    return value


def _metadata_mismatch(
    stored: dict,
    expected: dict,
    top_keys: tuple[str, ...],
    args_keys: tuple[str, ...],
) -> str | None:
    """metadataの不一致理由を返す（一致していればNone）。"""
    for key in top_keys:
        if _normalize(stored.get(key)) != _normalize(expected.get(key)):
            return f'{key} 不一致'
    stored_args = stored.get('args') or {}
    expected_args = expected.get('args') or {}
    if args_keys and not isinstance(stored_args, dict):
        return 'args がdictでない'
    for key in args_keys:
        if key not in expected_args:
            continue
        if _normalize(stored_args.get(key)) != _normalize(expected_args.get(key)):
            return f'args.{key} 不一致'
    return None


def find_resumable_checkpoint(
    prefix: str,
    base_dir: str = 'checkpoints',
    *,
    expected_metadata: dict,
    num_iterations: int | None = None,
    match_keys: tuple[str, ...] = ('task', 'config'),
    args_match_keys: tuple[str, ...] = (),
    ckpt_name: str = 'latest.pt',
    log=print,
    allow_unsafe: bool = False,
) -> dict | None:
    """同一 prefix の過去 run_dir を走査し、再開可能なチェックポイントを探す。

    Modal などの自動リトライでは ``setup_checkpoint_dir`` が**新しいタイムスタンプ付き
    run_dir を作る**ため、「自分の run_dir 内の latest.pt を見る」だけでは前試行の
    チェックポイントを見つけられない。本関数は ``{base_dir}/{prefix}_*`` を横断走査し、
    metadata が現在の引数と整合し、かつ未完了（step < num_iterations）のものだけを
    候補として、**最も step が進んだもの**を返す。

    Args:
        prefix: run_dir のプレフィックス（setup_checkpoint_dir と同じもの）
        base_dir: チェックポイントのベースディレクトリ
        expected_metadata: 現在の設定から作った metadata（save_checkpoint に渡すもの）
        num_iterations: 総ステップ数。step >= これ の run は完了済みとして除外
        match_keys: metadata トップレベルで一致必須のキー
        args_match_keys: metadata['args'] 内で一致必須のキー
        ckpt_name: 探すファイル名
        log: ログ出力関数
        allow_unsafe: Trueならpickleの任意オブジェクト復元を許可する

    Returns:
        {'path', 'run_dir', 'step', 'best_metric', 'metadata'} または None
    """
    pattern = os.path.join(base_dir, f'{prefix}_*')
    best: dict | None = None
    for run_dir in sorted(glob.glob(pattern)):
        name = os.path.basename(run_dir)
        suffix = name[len(prefix) + 1:]
        if not _RUN_DIR_TIMESTAMP_RE.match(suffix):
            continue  # 別prefix（prefixが前方一致しただけ）を誤って拾わない
        path = os.path.join(run_dir, ckpt_name)
        if not os.path.isfile(path):
            continue
        try:
            data = torch_load(path, map_location='cpu', allow_unsafe=allow_unsafe)
        except Exception as exc:  # 壊れたckptは黙って無視せず理由を出す
            log(f'自動再開: {path} を読めないためスキップ ({exc})')
            continue
        step = int(data.get('step', 0))
        stored = data.get('metadata') or {}
        del data  # 大きなstate_dictを保持しない
        if step <= 0:
            log(f'自動再開: {path} は step={step} のためスキップ')
            continue
        if num_iterations is not None and step >= num_iterations:
            log(f'自動再開: {path} は完了済み (step={step} >= '
                f'num_iterations={num_iterations}) のためスキップ')
            continue
        reason = _metadata_mismatch(stored, expected_metadata, match_keys, args_match_keys)
        if reason is not None:
            log(f'自動再開: {path} は設定不一致({reason})のためスキップ')
            continue
        if best is None or step > best['step']:
            best = {
                'path': path,
                'run_dir': run_dir,
                'step': step,
                'best_metric': None,
                'metadata': stored,
            }
    return best
