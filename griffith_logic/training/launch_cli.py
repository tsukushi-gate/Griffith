"""実験オーケストレーション。

griffith launch --grid "optim.lr=[0.001,0.01] model.k=[64,256]" [config.yaml]
    グリッド全組合せ(デカルト積)を展開して投入する。
    --dry-run: 一覧表示のみ / --local: このプロセスで順次実行 /
    --modal: scripts/modal_app.py の train_config を並列spawnして全完了待ち

griffith collect [--dest outputs/collected]
    Modal Volume(既定 `griffith-workspace`。`MD_MODAL_VOLUME` で上書き)の outputs/ 配下から、
    ローカル未取得のrunディレクトリをmodal CLI(subprocess)で回収する。
"""

from __future__ import annotations

import importlib.util
import itertools
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import yaml

# リポジトリルートの明示指定用環境変数
REPO_ROOT_ENV = 'GRIFFITH_REPO_ROOT'

# パッケージの2階層上。ソースツリーから使う場合はこれがrepo直下になるが、
# pip install(site-packages)配下では意味を持たないためフォールバック扱いにする。
_PACKAGE_PARENT = Path(__file__).resolve().parents[2]

# 後方互換: 旧来 REPO_ROOT を直接参照していた箇所のための定数
REPO_ROOT = _PACKAGE_PARENT

# griffith launch / griffith runpod は scripts/ を必要とする。
# pip install だけの環境には scripts/ が無いため、統一したガイダンスを出す。
SCRIPTS_REQUIRED_GUIDANCE = (
    '`griffith launch` はリポジトリを clone した環境で `scripts/` が必要です'
    '(pip インストールのみでは使えません)。\n'
    '  1. git clone してリポジトリ直下で実行する\n'
    f'  2. もしくは環境変数 {REPO_ROOT_ENV} にリポジトリのパスを設定する'
)


def resolve_repo_root(marker: str) -> Path | None:
    """`marker`(リポジトリ相対パス)を含むリポジトリルートを解決する。

    探索順は
      1. 環境変数 ``GRIFFITH_REPO_ROOT``
      2. カレントディレクトリ
      3. パッケージの2階層上(ソースツリーから実行している場合のみ有効)
    で、最初に ``<root>/<marker>`` が存在したものを返す。無ければ None。
    """
    candidates: list[Path] = []
    env_root = os.environ.get(REPO_ROOT_ENV)
    if env_root:
        candidates.append(Path(env_root))
    candidates.append(Path.cwd())
    candidates.append(_PACKAGE_PARENT)
    for root in candidates:
        if (root / marker).exists():
            return root
    return None


def _load_repo_script(marker: str, module_name: str):
    """リポジトリ内のスクリプトをimportlibでロードする(scripts/はパッケージ外)。

    Raises:
        RuntimeError: 見つからない場合(clone が必要な旨のガイダンス付き)
    """
    root = resolve_repo_root(marker)
    if root is None:
        raise RuntimeError(f'{marker} が見つかりません。\n{SCRIPTS_REQUIRED_GUIDANCE}')
    path = root / marker
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# Volume 名は MD_MODAL_VOLUME で上書きできる(scripts/modal_app.py と同じ変数名)
MODAL_VOLUME = os.environ.get('MD_MODAL_VOLUME', 'griffith-workspace')
MODAL_REMOTE_OUTPUTS = 'outputs'
DEFAULT_COLLECT_DEST = 'outputs/collected'

# グリッド指定のトークン: key=[v1,v2,...](リスト=掃引) or key=value(固定)
_GRID_TOKEN_RE = re.compile(r'([A-Za-z0-9_.]+)=(\[[^\]]*\]|\S+)')

MODAL_GUIDANCE = """\
Modalが未設定のため実行できません。以下を確認してください:
  1. uv sync --extra cpu --extra dev --extra cloud  (modalをインストール)
  2. uv run modal setup                             (初回認証)
  3. 詳細: docs/guides/modal.md"""


def parse_grid(spec: str) -> tuple[list[tuple[str, list[str]]], list[str]]:
    """グリッド指定文字列を解析する。

    Args:
        spec: 'optim.lr=[0.001,0.01] model.k=[64,256] run.epochs=3' 形式。
              リスト値は掃引軸、スカラー値は全組合せ共通の固定オーバーライド。

    Returns:
        (掃引軸のリスト[(ドットパス, 値文字列リスト)], 固定オーバーライドのリスト)

    Raises:
        ValueError: 書式不正・掃引軸が空
    """
    consumed = _GRID_TOKEN_RE.sub('', spec).strip()
    if consumed:
        raise ValueError(
            f'--grid の書式が不正です: {consumed!r}'
            '(例: --grid "optim.lr=[0.001,0.01] model.k=[64,256]")'
        )

    axes: list[tuple[str, list[str]]] = []
    fixed: list[str] = []
    for match in _GRID_TOKEN_RE.finditer(spec):
        key, raw = match.group(1), match.group(2)
        if raw.startswith('['):
            values = [v.strip() for v in raw[1:-1].split(',') if v.strip()]
            if not values:
                raise ValueError(f'グリッド軸 {key} の値リストが空です: {raw}')
            axes.append((key, values))
        else:
            fixed.append(f'{key}={raw}')
    if not axes:
        raise ValueError('--grid にリスト値(key=[v1,v2,...])の掃引軸が1つ以上必要です')
    return axes, fixed


def expand_grid(
    axes: list[tuple[str, list[str]]], fixed: list[str],
    base_run_name: str | None = None,
) -> list[dict]:
    """掃引軸のデカルト積を展開する。

    Returns:
        [{'name': 'lr0.01_k64', 'overrides': ['optim.lr=0.01', ...]}, ...]
        nameはグリッド値から生成(ドットパス末尾+値)し、run.run_nameへ反映される。
    """
    combos = []
    keys = [key for key, _ in axes]
    for values in itertools.product(*(vals for _, vals in axes)):
        suffix = '_'.join(
            f'{key.split(".")[-1]}{value}'
            for key, value in zip(keys, values, strict=False)
        )
        name = f'{base_run_name}_{suffix}' if base_run_name else suffix
        overrides = [
            f'{key}={value}' for key, value in zip(keys, values, strict=False)
        ]
        overrides += list(fixed)
        overrides.append(f'run.run_name={name}')
        combos.append({'name': name, 'overrides': overrides})
    return combos


def _base_run_name(config_path: str | None) -> str | None:
    """設定YAMLのrun.run_nameを読む(グリッド名のプレフィックスに使う)。"""
    if config_path is None:
        return None
    try:
        with open(config_path, encoding='utf-8') as f:
            doc = yaml.safe_load(f) or {}
        return (doc.get('run') or {}).get('run_name')
    except OSError:
        return None


def build_grid_combos(grid_spec: str, config_path: str | None) -> list[dict]:
    """--gridと設定YAMLからグリッド組合せ一覧を構築する。"""
    axes, fixed = parse_grid(grid_spec)
    return expand_grid(axes, fixed, base_run_name=_base_run_name(config_path))


def print_combos(combos: list[dict]) -> None:
    """組合せ一覧をmarkdown表で表示する(--dry-run)。"""
    print(f'グリッド展開: {len(combos)}件')
    print('| # | run名 | オーバーライド |')
    print('|---|-------|----------------|')
    for i, combo in enumerate(combos, 1):
        ov = ' '.join(o for o in combo['overrides'] if not o.startswith('run.run_name='))
        print(f'| {i} | {combo["name"]} | `{ov}` |')


def launch_local(combos: list[dict], config_path: str | None) -> list[dict]:
    """グリッドをこのプロセスで順次実行する(--local)。"""
    from .config import load_config
    from .runner import run_training

    results = []
    for i, combo in enumerate(combos, 1):
        print(f'=== [{i}/{len(combos)}] {combo["name"]} ===')
        cfg = load_config(config_path, combo['overrides'])
        result = run_training(cfg)
        results.append({'name': combo['name'], **result})
    return results


def load_modal_app():
    """scripts/modal_app.py をimportlibでロードする(scripts/はパッケージ外)。

    Raises:
        RuntimeError: modal未導入・ファイル欠落(呼び出し側でガイダンス表示)
    """
    try:
        import modal  # noqa: F401
    except ImportError as e:
        raise RuntimeError(MODAL_GUIDANCE) from e
    return _load_repo_script('scripts/modal_app.py', '_md_modal_app')


def preflight_ok() -> bool:
    """Modal同時実行数のpreflightを実行する(scripts/modal_preflight.py)。

    取得失敗・スクリプト欠落時はfail-openでTrue(投入をブロックしない)。
    上限超過時のみFalse。MD_SKIP_PREFLIGHT=1 での無効化はpreflight_check内で処理。
    """
    try:
        module = _load_repo_script('scripts/modal_preflight.py', '_md_modal_preflight')
        return bool(module.preflight_check())
    except Exception:  # noqa: BLE001 — preflightの失敗で投入自体は止めない(fail-open)
        return True


def launch_modal(combos: list[dict], config_path: str | None) -> list[dict]:
    """グリッドをModalへ並列spawnし全完了を待つ(--modal)。

    設定はYAML本文を渡す(リモートにローカルファイルは存在しないため)。

    Raises:
        RuntimeError: Modal未導入・未認証等(呼び出し側がガイダンス表示)
    """
    mod = load_modal_app()  # modal未導入はここでRuntimeError(ガイダンス付き)
    import modal

    # GPU同時実行limit(10)超過を防ぐ。上限超過ならspawnせず中止(緊急時 MD_SKIP_PREFLIGHT=1)
    if not preflight_ok():
        raise RuntimeError(
            'Modal同時実行数のpreflightで中止しました。`modal app list` で確認し、'
            '不要なジョブを1件ずつ停止してから再投入してください'
            '(緊急時は MD_SKIP_PREFLIGHT=1)。'
        )

    config_yaml = ''
    if config_path is not None:
        config_yaml = Path(config_path).read_text(encoding='utf-8')

    results = []
    try:
        with modal.enable_output(), mod.app.run():
            calls = [
                mod.train_config.spawn(config_yaml, combo['overrides'])
                for combo in combos
            ]
            print(f'Modalへ{len(calls)}件をspawnしました。全完了を待ちます...')
            for combo, call in zip(combos, calls, strict=False):
                try:
                    result = call.get()
                    results.append({'name': combo['name'], **(result or {})})
                    print(f'完了: {combo["name"]}')
                except Exception as e:  # noqa: BLE001 — 1件の失敗で全体を止めない
                    results.append({'name': combo['name'], 'error': str(e)})
                    print(f'失敗: {combo["name"]}: {e}', file=sys.stderr)
    except Exception as e:  # noqa: BLE001 — 認証・接続等のModal側エラーを案内へ変換
        raise RuntimeError(f'Modal実行に失敗しました: {e}\n{MODAL_GUIDANCE}') from e
    return results


# ============================================================
# RunPod backend(griffith launch --backend runpod)
# ============================================================

RUNPOD_GUIDANCE = """\
RunPodが未設定のため実行できません。以下を確認してください:
  1. uv sync --extra cpu --extra dev --extra cloud   (runpodをインストール)
  2. .env に RUNPOD_API_KEY / MD_RUNPOD_VOLUME_ID / MD_RUNPOD_DC を設定
  3. 初回のみ scripts/runpod_bootstrap.py で deploy key 登録+環境ウォーム
  4. 詳細: .env.example の RunPod 節"""


def load_runpod_app():
    """scripts/runpod_app.py をimportlibでロードする(scripts/はパッケージ外)。

    Raises:
        RuntimeError: runpod未導入・ファイル欠落(呼び出し側でガイダンス表示)
    """
    try:
        import runpod  # noqa: F401
    except ImportError as e:
        raise RuntimeError(RUNPOD_GUIDANCE) from e
    return _load_repo_script('scripts/runpod_app.py', '_md_runpod_app')


def runpod_preflight_ok() -> bool:
    """RunPod同時pod数のpreflightを実行する(scripts/runpod_preflight.py)。

    取得失敗・スクリプト欠落時はfail-openでTrue(投入をブロックしない)。
    上限超過時のみFalse。MD_SKIP_PREFLIGHT=1 での無効化はpreflight_check内で処理。
    """
    try:
        module = _load_repo_script('scripts/runpod_preflight.py', '_md_runpod_preflight')
        return bool(module.preflight_check())
    except Exception:  # noqa: BLE001 — preflightの失敗で投入自体は止めない(fail-open)
        return True


def _git_root() -> Path:
    """git操作の作業ディレクトリ。GRIFFITH_REPO_ROOT→cwd→パッケージ2階層上の順。"""
    return resolve_repo_root('.git') or Path.cwd()


def _current_commit_sha_checked() -> str:
    """HEADのSHAを返す。未push(リモートブランチに含まれない)ならRuntimeError。

    RunPodは投入時SHAをpod内で checkout するため、未pushだと再現不能になる。
    """
    head = subprocess.run(
        ['git', 'rev-parse', 'HEAD'],
        capture_output=True, text=True, cwd=_git_root(),
    )
    sha = head.stdout.strip()
    if head.returncode != 0 or not sha:
        raise RuntimeError('HEADのSHAを取得できませんでした(gitリポジトリ外?)')

    contains = subprocess.run(
        ['git', 'branch', '-r', '--contains', sha],
        capture_output=True, text=True, cwd=_git_root(),
    )
    if not contains.stdout.strip():
        raise RuntimeError(
            f'HEAD({sha[:8]})が未pushです。RunPodは投入時SHAをpod内でcheckoutするため、'
            '先に git push してください。'
        )
    return sha


def launch_runpod(combos: list[dict], config_path: str | None) -> list[dict]:
    """グリッドをRunPodへ fire-and-forget 投入する(griffith launch --backend runpod)。

    Modalと違い完了待ちしない(各podが学習後に自己終了する)。投入時HEAD SHAを
    pod内で checkout するため、未pushだと再現不能 → 事前にSHAを検証する。
    1件のsubmitが失敗しても残りは続行する。

    Raises:
        RuntimeError: runpod未導入・preflight中止・未push等(呼び出し側がガイダンス表示)
    """
    mod = load_runpod_app()  # runpod未導入はここでRuntimeError(ガイダンス付き)

    # 稼働pod数の上限チェック(緊急時 MD_SKIP_PREFLIGHT=1)
    if not runpod_preflight_ok():
        raise RuntimeError(
            'RunPod同時pod数のpreflightで中止しました。`griffith runpod ls` で確認し、'
            '不要なpodを1件ずつ停止してから再投入してください'
            '(緊急時は MD_SKIP_PREFLIGHT=1)。'
        )

    commit_sha = _current_commit_sha_checked()  # 未pushはRuntimeError

    config_yaml = ''
    if config_path is not None:
        config_yaml = Path(config_path).read_text(encoding='utf-8')

    results = []
    for combo in combos:
        try:
            pod_id = mod.submit_config(
                config_yaml, combo['overrides'], commit_sha=commit_sha,
            )
            results.append({'name': combo['name'], 'pod_id': pod_id})
            print(f'投入: {combo["name"]} → pod {pod_id}')
        except Exception as e:  # noqa: BLE001 — 1件の失敗で全体を止めない
            results.append({'name': combo['name'], 'error': str(e)})
            print(f'失敗: {combo["name"]}: {e}', file=sys.stderr)

    print(
        '\nRunPodへの投入が完了しました。各podは学習後に自己終了します。\n'
        '  結果はwandbで確認してください(WANDB_PROJECT)。\n'
        '  稼働中podの一覧・停止: griffith runpod ls / griffith runpod stop <pod_id>'
    )
    return results


def print_runpod_summary(results: list[dict]) -> None:
    """RunPod投入結果のサマリ表を表示する(pod_id列。hard_acc等は追えないため省く)。"""
    print('--- 投入pod一覧 ---')
    print('| run名 | pod_id | 状態 |')
    print('|-------|--------|------|')
    for r in results:
        if 'error' in r:
            print(f'| {r["name"]} | - | error |')
            continue
        print(f'| {r["name"]} | {r.get("pod_id", "-")} | 投入済 |')


def print_launch_summary(results: list[dict]) -> None:
    """launch結果のサマリ表を表示する。"""
    print('--- launch結果 ---')
    print('| run名 | hard_acc | best | expected |')
    print('|-------|----------|------|----------|')
    for r in results:
        if 'error' in r:
            print(f'| {r["name"]} | error | - | - |')
            continue
        acc = r.get('hard_acc')
        best = r.get('best_hard_acc')
        expected = r.get('expected_metrics')
        verdict = '-' if expected is None else ('OK' if expected.get('ok') else '未達')
        acc_s = '-' if acc is None else f'{acc:.4f}'
        best_s = '-' if best is None else f'{best:.4f}'
        print(f'| {r["name"]} | {acc_s} | {best_s} | {verdict} |')


# ============================================================
# griffith collect: Modal Volumeからoutputs/を回収
# ============================================================

def _run_modal_cli(args: list[str]) -> subprocess.CompletedProcess:
    """modal CLIをsubprocessで呼ぶ(`uv run modal ...`)。テストで差し替え可能。"""
    return subprocess.run(
        ['uv', 'run', 'modal', *args], capture_output=True, text=True,
    )


def list_remote_run_dirs() -> list[str]:
    """Modal Volumeの outputs/ 直下のエントリ名一覧を返す。

    Raises:
        RuntimeError: modal CLI失敗(未認証・Volume/パス無し等)
    """
    proc = _run_modal_cli(
        ['volume', 'ls', MODAL_VOLUME, MODAL_REMOTE_OUTPUTS, '--json']
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f'modal volume ls に失敗しました:\n{proc.stderr.strip()}\n{MODAL_GUIDANCE}'
        )
    try:
        entries = json.loads(proc.stdout)
    except json.JSONDecodeError as e:
        raise RuntimeError(
            f'modal volume ls の出力を解釈できません: {proc.stdout[:200]}'
        ) from e

    names = []
    for entry in entries:
        # modal CLIのJSONキーはバージョンで大文字/小文字が揺れるため両対応
        raw = entry.get('Filename') or entry.get('filename') or entry.get('path') or ''
        name = raw.rsplit('/', 1)[-1]
        if name:
            names.append(name)
    return names


def select_new_dirs(remote_names: list[str], dest: Path) -> list[str]:
    """ローカル未取得のrunディレクトリ名を選ぶ(既存名はスキップ)。"""
    existing = {p.name for p in dest.iterdir()} if dest.is_dir() else set()
    return [name for name in remote_names if name not in existing]


def collect(dest: str = DEFAULT_COLLECT_DEST) -> int:
    """Modal Volumeから新しいrunディレクトリを回収する(griffith collect)。"""
    dest_path = Path(dest)
    try:
        remote = list_remote_run_dirs()
    except RuntimeError as e:
        print(str(e), file=sys.stderr)
        return 1

    new_dirs = select_new_dirs(remote, dest_path)
    if not new_dirs:
        print(f'新しいrunディレクトリはありません(リモート{len(remote)}件は取得済み)')
        return 0

    dest_path.mkdir(parents=True, exist_ok=True)
    failed = 0
    for name in new_dirs:
        proc = _run_modal_cli([
            'volume', 'get', MODAL_VOLUME,
            f'{MODAL_REMOTE_OUTPUTS}/{name}', str(dest_path / name),
        ])
        if proc.returncode == 0:
            print(f'回収: {name} → {dest_path / name}')
        else:
            failed += 1
            print(f'失敗: {name}: {proc.stderr.strip()}', file=sys.stderr)
    print(f'完了: {len(new_dirs) - failed}/{len(new_dirs)}件を回収')
    return 1 if failed else 0
