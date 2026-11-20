"""wandb sweep対応。

sweep.yaml形式:
    method: bayes | grid | random
    metric: {name: hard_acc, goal: maximize}
    parameters:
        <TrainConfigのドットパス>: {values: [...] | min/max: ...}
    base_config: <TrainConfig YAMLのパス | null>

base_configはsweep作成時にJSON化して定数パラメータ(_base_config)として
sweepへ埋め込む。これによりModal等のリモートagentがローカルのファイルに
依存せず同一設定で試行できる。

agentの1試行 = sweepが選んだドットパスパラメータ → TrainConfig → run_training。
run_training内のwandb初期化は無効化し(二重init防止)、最終メトリクスを
agent側でsweep runへログする。
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone

import yaml

from .config import TrainConfig, dump_config, load_config

JST = timezone(timedelta(hours=9))

# base設定をsweepへ埋め込む定数パラメータ名(ドットを含まない=オーバーライド対象外)
BASE_CONFIG_PARAM = '_base_config'

SWEEP_METHODS = ('bayes', 'grid', 'random')

WANDB_GUIDANCE = """\
wandbが未設定のためsweepを実行できません。以下を確認してください:
  1. WANDB_API_KEY を設定する(https://wandb.ai/authorize で取得)
       export WANDB_API_KEY=<your-key>
  2. 必要なら WANDB_PROJECT / WANDB_ENTITY も設定する(既定: griffith-logic)
  3. Modal実行時はシークレット `wandb` に登録済み(docs/guides/modal.md参照)"""


def wandb_ready() -> bool:
    """wandbが利用可能か(ライブラリ+APIキー)を実通信なしで判定する。"""
    if not os.environ.get('WANDB_API_KEY'):
        return False
    try:
        import wandb  # noqa: F401
    except ImportError:
        return False
    return True


def wandb_project() -> str:
    return os.environ.get('WANDB_PROJECT', 'griffith-logic')


def wandb_entity() -> str | None:
    return os.environ.get('WANDB_ENTITY') or None


def load_sweep_spec(path: str) -> dict:
    """sweep.yamlを読み込み検証する(wandb通信なしの純粋関数)。

    Raises:
        ValueError: method/metric/parametersの欠落・不正
    """
    with open(path, encoding='utf-8') as f:
        spec = yaml.safe_load(f)
    if not isinstance(spec, dict):
        raise ValueError(f'sweep YAMLのルートはマッピングが必要です: {path}')

    method = spec.get('method')
    if method not in SWEEP_METHODS:
        raise ValueError(f'methodは {SWEEP_METHODS} のいずれかが必要です: {method!r}')

    metric = spec.get('metric')
    if not isinstance(metric, dict) or 'name' not in metric:
        raise ValueError(f'metricは {{name, goal}} 形式が必要です: {metric!r}')

    parameters = spec.get('parameters')
    if not isinstance(parameters, dict) or not parameters:
        raise ValueError('parametersに1件以上のドットパスキーが必要です')
    for key in parameters:
        if '.' not in key:
            raise ValueError(
                f'parametersのキーはTrainConfigのドットパスが必要です: {key!r}'
                '(例: optim.lr, model.k)'
            )
    return spec


def build_sweep_config(spec: dict) -> dict:
    """sweep spec → wandb.sweep() に渡す設定dictを構築する。

    base_configはロード・検証のうえJSON文字列の定数パラメータとして埋め込む。
    values指定のパラメータはbase設定に対して事前検証し、タイポを
    sweep作成前に検出する。
    """
    base_path = spec.get('base_config')
    base_dict: dict = {}
    if base_path:
        base_dict = dump_config(load_config(str(base_path)))

    parameters = dict(spec['parameters'])

    # values列挙のパラメータはドットパスの妥当性を事前検証する
    for key, param in parameters.items():
        if isinstance(param, dict) and 'values' in param:
            for value in param['values']:
                tree = json.loads(json.dumps(base_dict)) if base_dict else {}
                _assign_by_dots(tree, key, value)
                TrainConfig.model_validate(tree)

    config = {
        'method': spec['method'],
        'metric': dict(spec['metric']),
        'parameters': parameters,
    }
    if spec.get('name'):
        config['name'] = str(spec['name'])
    if base_dict:
        config['parameters'][BASE_CONFIG_PARAM] = {
            'value': json.dumps(base_dict, ensure_ascii=False),
        }
    return config


def create_sweep(spec: dict) -> str:
    """wandb sweepを作成しsweep_idを返す(要WANDB_API_KEY)。"""
    import wandb

    return wandb.sweep(
        build_sweep_config(spec), project=wandb_project(), entity=wandb_entity(),
    )


def _assign_by_dots(tree: dict, dotted_key: str, value) -> None:
    """型付きの値をドットパスでネスト辞書へ書き込む(config._set_by_dotsの型付き版)。"""
    keys = dotted_key.split('.')
    node = tree
    for key in keys[:-1]:
        node = node.setdefault(key, {})
    node[keys[-1]] = value


def config_from_sweep_params(params: dict) -> TrainConfig:
    """sweepが選んだパラメータ辞書からTrainConfigを構築する。

    _base_config(JSON)をベースに、ドットパスキーを上書きする。
    ドットを含まないキー(wandb内部キー等)は無視する。
    """
    base_json = params.get(BASE_CONFIG_PARAM)
    tree: dict = json.loads(base_json) if base_json else {}
    for key, value in params.items():
        if key == BASE_CONFIG_PARAM or '.' not in key:
            continue
        _assign_by_dots(tree, key, value)
    return TrainConfig.model_validate(tree)


def _append_runs_jsonl(run_id: str, run_name: str) -> None:
    """sweep runを.wandb_runs.jsonlへ追記する(wandb_utils.init_wandbと同形式)。"""
    try:
        entry = {
            'run_id': run_id,
            'run_name': run_name,
            'group': 'sweep',
            'tags': ['md-sweep'],
            'started_at': datetime.now(JST).isoformat(),
        }
        with open(os.path.join(os.getcwd(), '.wandb_runs.jsonl'), 'a',
                  encoding='utf-8') as f:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    except OSError:
        pass


def sweep_train_once() -> None:
    """wandb agentから呼ばれる1試行。

    wandb.init()でsweepが選んだパラメータを受け取り、TrainConfigを構築して
    run_trainingを実行する。ランナー内のwandbは無効化し(agentのrunと
    二重にならないように)、最終メトリクスをこの関数がsweep runへログする。
    """
    import wandb

    from .runner import run_training

    run = wandb.init()
    print(f'WANDB_RUN_ID={run.id}')
    _append_runs_jsonl(run.id, run.name or run.id)

    cfg = config_from_sweep_params(dict(run.config))
    cfg.run.wandb = False  # 二重init防止(このsweep runへ直接ログする)
    result = run_training(cfg)

    final = {k: v for k, v in result.items() if isinstance(v, int | float)}
    wandb.log(final)
    wandb.finish()


def run_sweep_agent(sweep_id: str, count: int | None = None) -> None:
    """sweep agentを実行する(このプロセスでcount試行、要WANDB_API_KEY)。"""
    import wandb

    wandb.agent(
        sweep_id, function=sweep_train_once, count=count,
        project=wandb_project(), entity=wandb_entity(),
    )
