"""wandb 共通ヘルパー(パッケージ内正本)。

全訓練スクリプトで共有するwandb初期化・ログ関数。

このモジュールは `experiments/` からパッケージ内(`griffith_logic.training`)へ
移設した。`experiments/wandb_utils.py` は後方互換の再exportシムとして残っている。
wandb未インストール・未設定でもimportは成功し、各関数は何もしないで返る。
"""

import atexit
import json
import os
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

import torch

JST = timezone(timedelta(hours=9))

# 実験台帳: finish_wandb が1run=1行を自動追記する。cwd基準の相対パス。
# 台帳はリポジトリにコミットされた開発環境の資産で、pip インストール版には存在しない。
# ファイルが無い環境では finish_wandb からの自動追記を黙ってスキップする(_ledger_enabled)。
LEDGER_PATH = 'docs/experiments-ledger.md'
# 台帳の final 列に採用する主要メトリクスの優先順(先に見つかった1つを採用)
_LEDGER_METRIC_PRIORITY = ('val_loss', 'best_val_loss', 'hard_acc', 'eval_acc', 'loss')

# finishレコードを二重に書かないためのガード(run_idごと1回のみ)。
# atexit と明示 finish() の両方から呼ばれても1回に収束させる。
_FINISHED_RUN_IDS: set[str] = set()


class _LedgerSkipped(Exception):
    """台帳の書き込み先が無いため追記をスキップした(内部シグナル)。"""

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


def should_use_wandb() -> bool:
    """wandbが使用可能かチェック。"""
    if not HAS_WANDB:
        return False
    mode = os.environ.get('WANDB_MODE', 'online')
    if mode == 'disabled':
        return False
    # offline モードは API キー不要でローカルに記録できる(スモーク・オフライン検証用)。
    # これを許可しないと offline 実行時に wandb.run が生成されず、
    # jsonl の finish レコードや台帳追記まで丸ごと欠落してしまう。
    if mode == 'offline':
        return True
    return bool(os.environ.get('WANDB_API_KEY'))


def normalize_tags(
    model: str | None = None,
    dataset: str | None = None,
    seed: int | str | None = None,
    series: str | None = None,
) -> list[str]:
    """taxonomy規約に沿った `key:value` 形式のタグリストを返す。

    順序は series → model → dataset → seed。None の項目は省略する。
    """
    tags: list[str] = []
    if series is not None:
        tags.append(f'series:{series}')
    if model is not None:
        tags.append(f'model:{model}')
    if dataset is not None:
        tags.append(f'dataset:{dataset}')
    if seed is not None:
        tags.append(f'seed:{seed}')
    return tags


def add_experiment_args(parser) -> None:
    """実験メタデータ用の共通argparse引数を追加する。

    --notes: この実験で何を・なぜ変えたか(wandb notes へ)
    --hypothesis: 検証したい仮説(config の experiment/hypothesis へ)
    """
    parser.add_argument(
        '--notes', type=str, default=None,
        help='この実験で何を・なぜ変えたか(wandb notesへ記録)',
    )
    parser.add_argument(
        '--hypothesis', type=str, default=None,
        help='検証したい仮説(config experiment/hypothesis と台帳へ記録)',
    )


def get_device_info(device: str) -> dict:
    """デバイス情報を辞書で返す。"""
    info = {'device': device}
    if device == 'cuda' and torch.cuda.is_available():
        info['gpu_name'] = torch.cuda.get_device_name(0)
        info['gpu_memory_gb'] = round(
            torch.cuda.get_device_properties(0).total_memory / (1024 ** 3), 2
        )
        info['cuda_version'] = torch.version.cuda or 'N/A'
    return info


def count_logic_gates(model) -> dict[str, int | str]:
    """モデル内のロジックゲート数をLUTタイプ別に集計する。

    LutConv2d の num_gates（ツリー内部ノード）、
    LutLayer の num_neurons、DWNLayer の n_luts を合算する。
    LUTタイプ（LUT2, LUT4, LUT6）も記録する。
    """
    from collections import Counter

    from griffith_logic.lut_layer import LutLayer
    try:
        from griffith_logic.conv import LutConv2d
    except ImportError:
        LutConv2d = None
    try:
        from griffith_logic.layers.dwn import DWNLayer
    except ImportError:
        DWNLayer = None

    conv_gates = 0
    dense_gates = 0
    dwn_gates = 0
    lut_type_counts: Counter[int, int] = Counter()

    for m in model.modules():
        if LutConv2d is not None and isinstance(m, LutConv2d):
            n = m.num_gates
            conv_gates += n
            lut_type_counts[m.n_inputs] += n
        elif isinstance(m, LutLayer):
            n = m.num_neurons
            dense_gates += n
            lut_type_counts[m.n_inputs] += n
        elif DWNLayer is not None and isinstance(m, DWNLayer):
            n = m.n_luts
            dwn_gates += n
            lut_type_counts[m.n_inputs] += n

    total = conv_gates + dense_gates + dwn_gates
    result: dict[str, int | str] = {'gates/total': total}
    if conv_gates > 0:
        result['gates/conv'] = conv_gates
    if dense_gates > 0:
        result['gates/dense'] = dense_gates
    if dwn_gates > 0:
        result['gates/dwn'] = dwn_gates

    # LUTタイプ別の内訳（例: gates/LUT2 = 2880）
    for n_inputs, count in sorted(lut_type_counts.items()):
        result[f'gates/LUT{n_inputs}'] = count

    # LUTタイプの要約（例: "LUT2" や "LUT2+LUT4"）
    if lut_type_counts:
        type_labels = [f'LUT{n}' for n in sorted(lut_type_counts)]
        result['gates/lut_type'] = '+'.join(type_labels)

    return result


def get_model_param_info(model) -> dict:
    """モデルのパラメータ情報とロジックゲート数を辞書で返す。

    model.count_parameters() があればそれを使い、なければ総パラメータ数のみ。
    ロジックゲート数（LutConv2d/LutLayer）も自動集計する。
    """
    info = {}
    if hasattr(model, 'count_parameters'):
        counts = model.count_parameters()
        for name, count in counts.items():
            info[f'params/{name}'] = count
    else:
        total = sum(p.numel() for p in model.parameters())
        info['params/total'] = total

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    info['params/trainable'] = trainable

    # ロジックゲート数
    try:
        info.update(count_logic_gates(model))
    except Exception:
        pass

    return info


def _git_short_hash() -> str:
    """現在のgit短縮ハッシュを返す。取得失敗時は空文字。"""
    try:
        return subprocess.check_output(
            ['git', 'rev-parse', '--short', 'HEAD'],
            stderr=subprocess.DEVNULL, text=True,
        ).strip()
    except Exception:
        return ''


def init_wandb(
    run_name: str,
    config_dict: dict,
    model=None,
    device: str = 'cpu',
    tags: list[str] | None = None,
    group: str | None = None,
    job_type: str = 'train',
    resume_run_id: str | None = None,
    notes: str | None = None,
    hypothesis: str | None = None,
) -> bool:
    """wandbを初期化する。

    run_nameにはタイムスタンプ(MMdd_HHmm)が自動付加される。

    Args:
        run_name: 実行名（ハイパーパラメータ部分）
        config_dict: ハイパーパラメータ辞書（argsから生成）
        model: モデル（パラメータ数を記録）
        device: デバイス文字列
        tags: wandbタグ
        group: runのグループ名（例: 'cnn', 'detection'）
        job_type: ジョブ種別（デフォルト: 'train'）
        resume_run_id: 指定時は同一wandb runへ再開する(id指定+resume='allow')。
            この場合run名にタイムスタンプを新規付加せず既存runの名前を維持する
        notes: 変更内容の自然言語メモ。wandb notes には「notes + git=<hash>」を結合して渡す
        hypothesis: 検証したい仮説。config の `experiment/hypothesis` として記録する

    Returns:
        wandbが有効化されたかどうか
    """
    if not should_use_wandb():
        print('wandb: 無効 (WANDB_API_KEYが未設定またはwandb未インストール)')
        return False

    project = os.environ.get('WANDB_PROJECT', 'griffith-logic')
    entity = os.environ.get('WANDB_ENTITY') or None

    if resume_run_id:
        # 同一runへ再開: タイムスタンプを新規付加せず既存run名を維持する
        full_name = run_name
    else:
        # run_nameにタイムスタンプを付加して一意にする
        ts = datetime.now(JST).strftime('%m%d_%H%M')
        full_name = f'{run_name}_{ts}'

    # configにデバイス・モデル情報を追加
    full_config = dict(config_dict)
    full_config.update(get_device_info(device))
    if model is not None:
        full_config.update(get_model_param_info(model))

    # hypothesis(仮説)はconfigに experiment/hypothesis として記録する
    if hypothesis:
        full_config['experiment/hypothesis'] = hypothesis

    # wandb notes = ユーザーnotes + git=<hash> を結合(どちらか一方でも可)
    git_hash = _git_short_hash()
    git_note = f'git={git_hash}' if git_hash else ''
    if notes:
        wandb_notes = f'{notes} | {git_note}' if git_note else notes
    else:
        wandb_notes = git_note or None

    init_kwargs = dict(
        project=project,
        entity=entity,
        config=full_config,
        tags=tags,
        group=group,
        job_type=job_type,
        notes=wandb_notes,
    )
    if resume_run_id:
        init_kwargs['id'] = resume_run_id
        init_kwargs['resume'] = 'allow'
    else:
        init_kwargs['name'] = full_name

    wandb.init(**init_kwargs)
    resumed_note = ' (resumed)' if resume_run_id else ''
    print(f'wandb: 有効 (project={project}, name={full_name}, group={group}){resumed_note}')

    # run IDを機械可読形式で出力（Claude Codeが解析可能）
    print(f'WANDB_RUN_ID={wandb.run.id}')
    print(f'WANDB_RUN_URL={wandb.run.url}')

    # 最新のrun IDをファイルに保存（常に上書き）
    try:
        run_id_path = os.path.join(os.getcwd(), '.wandb_last_run_id')
        with open(run_id_path, 'w') as f:
            f.write(wandb.run.id + '\n')
    except OSError:
        pass

    # 全runの情報をJSONLに蓄積
    try:
        run_log_path = os.path.join(os.getcwd(), '.wandb_runs.jsonl')
        run_entry = {
            'run_id': wandb.run.id,
            'run_name': full_name,
            'group': group,
            'job_type': job_type,
            'tags': tags or [],
            'notes': notes,
            'hypothesis': hypothesis,
            'started_at': datetime.now(JST).isoformat(),
        }
        if resume_run_id:
            run_entry['resumed'] = True
        with open(run_log_path, 'a') as f:
            f.write(json.dumps(run_entry, ensure_ascii=False) + '\n')
    except OSError:
        pass

    # finishレコード欠落バグ対策: 明示 finish() の呼び忘れ・例外・
    # KeyboardInterrupt・Modal --detach のプロセス終了などで finish_wandb が
    # 呼ばれないと init レコードだけ残り finish が欠落する。atexit で保険をかけ、
    # 異常終了でも finish レコード(+台帳)を確実に残す(二重書きは _FINISHED_RUN_IDS でガード)。
    atexit.register(finish_wandb)

    return True


def log_metrics(metrics: dict, step: int | None = None) -> None:
    """wandbにメトリクスをログする。wandb未使用時は何もしない。"""
    if HAS_WANDB and wandb.run is not None:
        wandb.log(metrics, step=step)


# ============================================================
# train loss の定期ログ(SKILL.md §11)
# ============================================================

# train loss を wandb へ記録する壁時計間隔の上限(秒)。
# 「評価チェックポイントでしか何も記録されない」実装を禁じる全PJ共通ルール。
LOSS_LOG_INTERVAL_SEC = 300.0


class LossLogPacer:
    """train loss を壁時計で一定間隔以内に必ず1点 wandb へ出すためのペーサ。

    重い評価(BLEU / hard eval 等)の間隔とは**独立**に使う。step間隔は最初の
    `warmup_steps` step の実測 秒/step から自動決定し、ハードコードしない
    (モデル規模やデバイスが変わると固定値では壁時計間隔が数十分に伸びるため)。

    使い方::

        pacer = LossLogPacer(every_steps=args.log_every_steps)
        ...
        if pacer.should_log(done_steps, elapsed, is_last=...):
            if pacer.resolved_now:
                print(pacer.describe())
            log_metrics({'loss/main': ..., **progress_metrics(...)}, step=...)
    """

    def __init__(self, interval_sec: float = LOSS_LOG_INTERVAL_SEC,
                 every_steps: int | None = None, warmup_steps: int = 10) -> None:
        self.interval_sec = interval_sec
        self.warmup_steps = max(2, warmup_steps)
        self.every_steps = max(1, every_steps) if every_steps else None
        self.sec_per_step: float | None = None
        self.resolved_now = False
        self._first_elapsed: float | None = None
        self._next = self.every_steps if self.every_steps else self.warmup_steps

    def should_log(self, done_steps: int, elapsed: float, is_last: bool = False) -> bool:
        """このstepでログすべきかを返す。

        Args:
            done_steps: このrunで消化したstep数(1始まり。resume分は含めない)
            elapsed: 訓練開始からの経過秒
            is_last: 最終stepなら True(ウォームアップ未達でも間隔を確定させる)
        """
        self.resolved_now = False
        if done_steps <= 0:
            return False
        if done_steps == 1:
            # 1step目は初期化・カーネルコンパイルが乗るので平均から除外する
            self._first_elapsed = elapsed
        if self.every_steps is None and (done_steps >= self.warmup_steps or is_last):
            self.sec_per_step = ((elapsed - (self._first_elapsed or 0.0))
                                 / max(done_steps - 1, 1))
            self.every_steps = max(
                1, int(self.interval_sec / max(self.sec_per_step, 1e-9)))
            self._next = done_steps
            self.resolved_now = True
        if self.every_steps is None or done_steps < self._next:
            return False
        self._next = done_steps + self.every_steps
        return True

    def describe(self) -> str:
        """決定した記録間隔を人間可読な1行で返す(起動ログ用)。"""
        if self.sec_per_step is None:
            return f'wandbのloss記録間隔: {self.every_steps} step (明示指定)'
        return (f'wandbのloss記録間隔: {self.every_steps} step '
                f'(実測 {self.sec_per_step:.4f} s/step → 約'
                f'{self.every_steps * self.sec_per_step:.0f} 秒ごと, '
                f'上限 {int(self.interval_sec)} 秒)')


def progress_metrics(done_steps: int, current_step: int, total_steps: int,
                     elapsed: float) -> dict:
    """進捗率・ETA・スループットの共通メトリクス。

    キーは既存の指標体系(`loss/*`, `acc/*`, `bleu/*`, `mem/*` 等)と衝突しない
    `progress/*` `throughput/*` 名前空間に置く。

    Args:
        done_steps: このrunで消化したstep数(スループットの分子)
        current_step: 通し step 番号(resume分を含む。進捗率の分子)
        total_steps: 総step数
        elapsed: 訓練開始からの経過秒
    """
    steps_per_sec = done_steps / max(elapsed, 1e-9)
    remaining = max(total_steps - current_step, 0)
    return {
        'progress/fraction': current_step / max(total_steps, 1),
        'progress/elapsed_sec': elapsed,
        'progress/eta_sec': remaining / max(steps_per_sec, 1e-9),
        'throughput/it_per_sec': steps_per_sec,
    }


def _numeric_summary(summary) -> dict:
    """run.summary から数値メトリクスのみを抽出する(内部キー `_*` は除外)。"""
    try:
        return {
            k: v for k, v in dict(summary).items()
            if isinstance(v, int | float) and not isinstance(v, bool)
            and not str(k).startswith('_')
        }
    except Exception:
        return {}


def _pick_ledger_metric(metrics: dict) -> str:
    """台帳の final 列用に主要メトリクスを1つ選び `key=value` で返す。無ければ '-'。"""
    for key in _LEDGER_METRIC_PRIORITY:
        v = metrics.get(key)
        if isinstance(v, int | float) and not isinstance(v, bool):
            return f'{key}={v:.4f}' if isinstance(v, float) else f'{key}={v}'
    return '-'


LEDGER_HEADER = (
    '# 実験台帳\n'
    '\n'
    '1 run = 1 行。`finish_wandb`'
    '(griffith_logic/training/wandb_utils.py)が訓練終了時に自動追記する。\n'
    '判定列は空欄で出力され、人間または experiment-report スキルが後から埋める。\n'
    '\n'
    '| 日付 | run ID | group | job_type | hypothesis | notes | final | 判定 |\n'
    '|------|--------|-------|----------|------------|-------|-------|------|\n'
)


def _ledger_escape(s) -> str:
    """台帳セル用にパイプ・改行をエスケープする。空値は '-'。"""
    text = str(s) if s not in (None, '') else '-'
    return text.replace('|', '\\|').replace('\n', ' ').strip()


def format_ledger_row(
    date: str, run_id: str, group, job_type, hypothesis, notes,
    final_metric: str, verdict: str = '',
) -> str:
    """台帳の1行(markdown表)を組み立てる。判定列は既定で空欄。"""
    cells = [
        date, run_id, _ledger_escape(group), _ledger_escape(job_type),
        _ledger_escape(hypothesis), _ledger_escape(notes),
        _ledger_escape(final_metric), verdict,
    ]
    return '| ' + ' | '.join(cells) + ' |'


def append_ledger_row(
    path: str, date: str, run_id: str, group, job_type,
    hypothesis, notes, final_metric: str, verdict: str = '',
) -> None:
    """台帳へ1行追記する。ファイルが無ければヘッダ付きで新規作成する。"""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if not p.exists():
        p.write_text(LEDGER_HEADER, encoding='utf-8')
    row = format_ledger_row(
        date, run_id, group, job_type, hypothesis, notes, final_metric, verdict,
    )
    with open(p, 'a', encoding='utf-8') as f:
        f.write(row + '\n')


def _ledger_enabled(path: str = LEDGER_PATH) -> bool:
    """台帳へ追記してよいかを返す。

    台帳(`docs/experiments-ledger.md`)はリポジトリにコミットされている資産で、
    公開版(pip install)には存在しない。ファイルが無い環境では黙ってスキップし、
    実験と無関係な場所に docs/ を作らない(警告も出さない)。
    """
    return Path(path).exists()


def finish_wandb(status: str = 'finished', final_metrics: dict | None = None) -> None:
    """wandb runを明示終了する。Modalのatexit待ちタイムアウトを避ける。

    wandb.finish() の前に .wandb_runs.jsonl(init_wandbと同じcwd直下)へ
    終了レコードを追記し、オフライン時でも終了状態・成果物URLをローカルに残す。
    さらに実験台帳へ1行追記する。

    明示 finish() と atexit の両経路から呼ばれるため、run_id 単位で1回だけ
    finish レコード・台帳追記を行う(_FINISHED_RUN_IDS でガード)。

    Args:
        status: 終了状態('finished' / 'failed' 等)
        final_metrics: 最終メトリクス辞書(省略時は run.summary の数値のみ)
    """
    if not (HAS_WANDB and wandb.run is not None):
        return

    run_id = wandb.run.id
    # 既に finish 済みなら記録は二重書きせず、wandb.finish() だけ冪等に呼ぶ
    if run_id in _FINISHED_RUN_IDS:
        wandb.finish()
        return

    try:
        url = wandb.run.url  # オフラインモードでは取得できない
    except Exception:
        url = None
    if final_metrics is None:
        # final_metrics は run.summary の「数値のみ」を採用する
        final_metrics = _numeric_summary(wandb.run.summary)
    # 台帳用に run 側のメタデータを取得(init との整合を保つ)
    group = getattr(wandb.run, 'group', None)
    job_type = getattr(wandb.run, 'job_type', None)
    try:
        hypothesis = dict(wandb.run.config).get('experiment/hypothesis', '')
    except Exception:
        hypothesis = ''
    notes = getattr(wandb.run, 'notes', '') or ''

    finished_at = datetime.now(JST).isoformat()

    # (1) finishレコードをJSONLへ確実に追記
    try:
        finish_entry = {
            'run_id': run_id,
            'event': 'finish',
            'status': status,
            'url': url,
            'final_metrics': final_metrics,
            'finished_at': finished_at,
        }
        run_log_path = os.path.join(os.getcwd(), '.wandb_runs.jsonl')
        with open(run_log_path, 'a') as f:
            # default=strでnumpy等の非JSON値も安全化
            f.write(json.dumps(finish_entry, ensure_ascii=False, default=str) + '\n')
    except OSError:
        pass

    # (2) 実験台帳へ1行追記(失敗は握り潰さずwarning。訓練/終了処理は落とさない)
    # 台帳を置く場所が無い環境(pip install版)では黙ってスキップする
    try:
        if not _ledger_enabled():
            raise _LedgerSkipped
        append_ledger_row(
            LEDGER_PATH,
            date=datetime.now(JST).strftime('%Y-%m-%d'),
            run_id=run_id, group=group, job_type=job_type,
            hypothesis=hypothesis, notes=notes,
            final_metric=_pick_ledger_metric(final_metrics),
        )
    except _LedgerSkipped:
        pass
    except Exception as e:  # noqa: BLE001 — 台帳追記失敗で訓練を落とさない
        print(f'警告: 実験台帳への追記に失敗しました({LEDGER_PATH}): {e}')

    _FINISHED_RUN_IDS.add(run_id)
    wandb.finish()


def log_images(key: str, images: list, captions: list[str] | None = None,
               step: int | None = None) -> None:
    """PIL画像のリストをwandbにログする。

    Args:
        key: ログキー名（例: 'detection/hard'）
        images: PIL.Image.Imageのリスト
        captions: 各画像のキャプション
        step: wandbステップ
    """
    if not (HAS_WANDB and wandb.run is not None):
        return
    wandb_images = []
    for i, img in enumerate(images):
        caption = captions[i] if captions and i < len(captions) else None
        wandb_images.append(wandb.Image(img, caption=caption))
    wandb.log({key: wandb_images}, step=step)


def log_model_checkpoint(path: str, name: str,
                         metadata: dict | None = None) -> None:
    """チェックポイントをwandb Artifactとして保存する。

    Args:
        path: チェックポイントファイルパス
        name: Artifact名（例: 'detection-centernet-best'）
        metadata: Artifactに付与するメタデータ
    """
    if not (HAS_WANDB and wandb.run is not None):
        return
    artifact = wandb.Artifact(name, type='model', metadata=metadata)
    artifact.add_file(path)
    wandb.log_artifact(artifact)


# ============================================================
# Artifacts運用
# ============================================================

def artifact_policy() -> str:
    """Artifactアップロードポリシー。環境変数 WANDB_ARTIFACTS で制御。

    off  — アップロードしない
    best — 訓練終了時に best.pt を1回のみ（デフォルト）
    all  — 訓練中の定期保存(latest)もすべてアップロード
    """
    policy = os.environ.get('WANDB_ARTIFACTS', 'best').lower()
    if policy not in ('off', 'best', 'all'):
        print(f'wandb: 不明なWANDB_ARTIFACTS={policy} → bestとして扱う')
        return 'best'
    return policy


def _sanitize_artifact_name(name: str) -> str:
    """wandb Artifact名に使えない文字を'-'に置換する。"""
    import re
    return re.sub(r'[^a-zA-Z0-9_.\-]', '-', name)


def log_artifact(
    paths: str | list[str],
    name: str,
    type: str = 'model',
    aliases: list[str] | None = None,
    metadata: dict | None = None,
) -> None:
    """任意ファイル群をwandb Artifactとしてログする。

    git hashを自動でmetadataに付与する。wandb未使用時は何もしない。

    Args:
        paths: ファイルパス（単一または複数）
        name: Artifact名
        type: Artifact種別（'model' / 'model-ir' 等）
        aliases: 付与するエイリアス（例: ['best', 'latest']）
        metadata: メタデータ辞書
    """
    if not (HAS_WANDB and wandb.run is not None):
        return
    if isinstance(paths, str):
        paths = [paths]
    meta = dict(metadata or {})
    git_hash = _git_short_hash()
    if git_hash and 'git_hash' not in meta:
        meta['git_hash'] = git_hash
    artifact = wandb.Artifact(_sanitize_artifact_name(name), type=type, metadata=meta)
    for p in paths:
        artifact.add_file(p)
    wandb.log_artifact(artifact, aliases=aliases)
    print(f'wandb: Artifact記録 name={name} type={type} aliases={aliases}')


def log_checkpoint_artifact(
    path: str,
    name: str,
    stage: str = 'final',
    metadata: dict | None = None,
    model=None,
) -> None:
    """チェックポイントをポリシーに従ってArtifact化する。

    Args:
        path: チェックポイントファイルパス
        name: Artifact名（例: run名。'-ckpt'が自動付加される）
        stage: 'eval'（訓練中の定期保存）| 'final'（訓練終了時のbest）
        metadata: メタデータ（argsなど）
        model: モデル（ゲート数・パラメータ数を自動でmetadataに追加）
    """
    policy = artifact_policy()
    if policy == 'off':
        return
    if stage == 'eval' and policy != 'all':
        return
    if not os.path.exists(path):
        return

    meta = dict(metadata or {})
    if model is not None:
        try:
            meta.update(count_logic_gates(model))
        except Exception:
            pass
    aliases = ['latest'] if stage == 'eval' else ['best', 'latest']
    log_artifact(path, f'{name}-ckpt', type='model', aliases=aliases, metadata=meta)


def use_checkpoint(artifact_ref: str, download_dir: str = 'checkpoints') -> str:
    """wandb Artifactからチェックポイントをダウンロードしてパスを返す。

    Modal等の別環境で --resume する用途。

    Args:
        artifact_ref: 'entity/project/name:alias' 形式の参照
        download_dir: ダウンロード先ディレクトリ

    Returns:
        ダウンロードした .pt ファイルのパス
    """
    if not HAS_WANDB:
        raise RuntimeError('wandbがインストールされていない')
    if wandb.run is not None:
        artifact = wandb.run.use_artifact(artifact_ref)
    else:
        artifact = wandb.Api().artifact(artifact_ref)
    root = artifact.download(root=download_dir)
    pts = [f for f in os.listdir(root) if f.endswith('.pt')]
    if not pts:
        raise FileNotFoundError(f'Artifact内に.ptが存在しない: {artifact_ref}')
    path = os.path.join(root, sorted(pts)[0])
    print(f'wandb: Artifact取得 {artifact_ref} → {path}')
    return path


def log_ir_artifact(
    ir_json_path: str,
    sidecar_paths: list[str] | tuple = (),
    name: str | None = None,
    metadata: dict | None = None,
) -> None:
    """IR(中間表現)JSONとサイドカーファイルをArtifact化する（M7連携用）。"""
    if artifact_policy() == 'off':
        return
    base = name or os.path.splitext(os.path.basename(ir_json_path))[0]
    log_artifact(
        [ir_json_path, *sidecar_paths],
        f'{base}-ir', type='model-ir', aliases=['latest'], metadata=metadata,
    )


def finish() -> None:
    """wandbのrunを終了する(finish_wandbへ委譲して一本化)。"""
    finish_wandb()
