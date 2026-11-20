"""NASオーケストレータ。

構造自動探索の本体。本体ライブラリ(``griffith_logic/``)は
``run_training``(training/runner.py)・``estimate_resources``
(export/estimate.py)・``export_ir``/``ir_to_dict``(ir)を**呼び出すだけ**で、
一切変更しない。

設計判断(重要):
    - ``run_training`` は ``TrainConfig``(=uniformなModelConfigスタック)からしか
      学習できない(runner.py L288: bp固定、_build_modelはk一定・n_inputs単一の
      層列を組む)。従って本オーケストレータの探索遺伝子(genome)は
      「一様な構造記述子」= (layers, width, n_inputs) とする。各trialは
        1. genome → TrainConfig → run_training で学習+hard eval → hard_acc
        2. 同じgenomeから素の(未学習)アーキテクチャを組み → export_ir →
           estimate_resources で回路面積(total_luts / logic_depth / wire等)
      を得る。estimate_resourcesは重みに依存せず構造のみで決まるため、
      未学習モデルからの抽出で面積指標は正しい。
    - tools/nas/mutate.pyの任意IR変異は非一様IRを生み ``run_training``
      では学習できない(uniform ModelConfigに写せない)。よって本探索の変異は
      genome記述子レベル(layers±1 / width近傍 / n_inputs選択)で行い、常に
      ModelConfigへ写せる範囲に閉じる(設計裁定5=bp/runner固定の帰結)。
      mutate.pyは将来 build_torch 経由で非一様IRを学習する探索のために温存する。
    - 決定性: 各trialの乱数は ``random.Random`` を (seed, trial_id) から
      毎回作り直す(実行中の連続ストリームを持たない)。親選択は永続化済みの
      台帳(JSONL)のみに依存する。これにより「中断→再開」が
      無中断実行と完全に同一系列になる(trial_idは連番、Date/timeは不使用)。
    - 台帳: outputs(既定)/nas/<search_id>/trials.jsonl に1行1trial、
      各trialのIRは trial_<id>.ir.json に保存する。再開は台帳の行数から
      次のtrial_idを決め、重複なく継続する。
"""

from __future__ import annotations

import json
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
import torch.nn as nn

from griffith_logic.export import estimate_resources, export_ir
from griffith_logic.group_sum import GroupSum
from griffith_logic.ir import ir_to_dict
from griffith_logic.layers.lut import LutLayer
from griffith_logic.training.config import TrainConfig
from griffith_logic.training.runner import run_training

# ------------------------------------------------------------------
# 探索空間・設定・trialのデータ構造
# ------------------------------------------------------------------


@dataclass
class SearchSpace:
    """一様構造の探索範囲(全てModelConfigへ写せる範囲)。"""

    layers_choices: tuple[int, ...] = (1, 2, 3)
    width_choices: tuple[int, ...] = (4, 8, 16)
    n_inputs_choices: tuple[int, ...] = (2, 4)


@dataclass
class SearchConfig:
    """探索全体の設定。"""

    search_id: str
    dataset: str = 'logic:xor'
    model_kind: str = 'logic_mlp'  # 'logic_mlp'(binary_vector) | 'dlgn'(classification)
    max_trials: int = 12
    time_budget_sec: float | None = None
    epochs: int = 5
    batch_size: int = 32
    lr: float = 0.02
    seed: int = 0
    n_random: int = 4  # 序盤のランダム探索trial数(以降は進化)
    space: SearchSpace = field(default_factory=SearchSpace)
    out_root: str = 'outputs/nas'
    # 親選択戦略: 'evolution'(既定・従来のパレートランダム選択)|
    # 'nsga2'(tools/nas/nsga2.py の非劣ソート+混雑度)。既定値は従来の
    # 挙動・乱数消費列を一切変えないためゴールデンに影響しない。
    strategy: str = 'evolution'


@dataclass
class Trial:
    """1trialの記録(台帳の1行)。"""

    trial_id: int
    genome: dict
    state: str  # 'done' | 'failed'
    acc: float
    best_acc: float
    total_luts: int
    logic_depth: int
    wire_count: int
    group_sums: int
    gates: int
    num_params: int
    partial: bool
    parent_id: int | None
    steps: int
    seed: int
    error: str | None = None


# ------------------------------------------------------------------
# genome ⇄ 学習設定 / アーキテクチャ
# ------------------------------------------------------------------


def _trial_rng(base_seed: int, trial_id: int) -> random.Random:
    """(seed, trial_id) から決定的にRandomを生成する(連続ストリームを持たない)。"""
    return random.Random(base_seed * 1_000_003 + trial_id * 97 + 1)


def _trial_seed(base_seed: int, trial_id: int) -> int:
    """trialの学習seed(run_training / torch初期化に渡す決定的seed)。"""
    return (base_seed * 7919 + trial_id * 31 + 13) % (2**31 - 1)


def random_genome(rng: random.Random, space: SearchSpace) -> dict:
    """探索空間から一様構造genomeをランダムに1つサンプルする。"""
    return {
        'layers': rng.choice(space.layers_choices),
        'width': rng.choice(space.width_choices),
        'n_inputs': rng.choice(space.n_inputs_choices),
    }


def _neighbor(rng: random.Random, choices: tuple[int, ...], current: int) -> int:
    """choices上でcurrentの近傍(前後1つ、端は内側)へ動かす。"""
    if current not in choices or len(choices) == 1:
        return rng.choice(choices)
    i = choices.index(current)
    if i == 0:
        j = 1
    elif i == len(choices) - 1:
        j = i - 1
    else:
        j = i + rng.choice((-1, 1))
    return choices[j]


def mutate_genome(rng: random.Random, parent: dict, space: SearchSpace) -> dict:
    """親genomeの1軸を近傍へ変異させる(常にuniform=ModelConfig写像可能)。

    layers / width / n_inputs のいずれか1つを近傍値へ動かす。
    """
    child = dict(parent)
    axis = rng.choice(('layers', 'width', 'n_inputs'))
    if axis == 'layers':
        child['layers'] = _neighbor(rng, space.layers_choices, parent['layers'])
    elif axis == 'width':
        child['width'] = _neighbor(rng, space.width_choices, parent['width'])
    else:
        child['n_inputs'] = _neighbor(rng, space.n_inputs_choices, parent['n_inputs'])
    return child


def genome_to_config(genome: dict, cfg: SearchConfig, trial_id: int) -> TrainConfig:
    """genome+探索設定から run_training 用の TrainConfig を組み立てる。

    runner.py/config.py 規約:
        - logic:* データセット + model.kind='logic_mlp' は binary_vector 用
          (GroupSumなし、最終層out_width=データ出力幅)。
        - vision + model.kind='dlgn' は classification 用(GroupSum終端、
          k はクラス数の倍数)。
    """
    return TrainConfig.model_validate({
        'data': {'name': cfg.dataset, 'batch_size': cfg.batch_size},
        'model': {
            'kind': cfg.model_kind,
            'k': genome['width'],
            'layers': genome['layers'],
            'n_inputs': genome['n_inputs'],
        },
        'optim': {'name': 'adam', 'lr': cfg.lr},
        'run': {
            'seed': _trial_seed(cfg.seed, trial_id),
            'epochs': cfg.epochs,
            'run_name': f'{cfg.search_id}_trial{trial_id}',
        },
    })


def _dataset_widths(dataset: str) -> tuple[int, int]:
    """データセット名から (in_width, out_width) を解決する(estimate用の構造復元)。"""
    if dataset.startswith('logic:'):
        from griffith_logic.datasets.logic import LOGIC_DATASETS

        key = dataset.split(':', 1)[1]
        spec = LOGIC_DATASETS[key]()
        return spec.n_inputs, spec.n_outputs

    from griffith_logic.datasets.vision import VISION_DATASETS

    spec = VISION_DATASETS[dataset]
    return spec.in_width, spec.classes


def build_arch_model(genome: dict, cfg: SearchConfig, trial_seed: int) -> tuple[nn.Module, int]:
    """genomeから素のアーキテクチャ(未学習)を組む。runner._build_modelと同型。

    Returns:
        (model, in_width)。in_width は export_ir に渡す入力幅。
    """
    in_width, out_width = _dataset_widths(cfg.dataset)
    torch.manual_seed(trial_seed)  # 抽出IR(真理値表)の再現性のため固定
    n_inputs = genome['n_inputs']
    k = genome['width']
    layers_n = genome['layers']
    modules: list[nn.Module] = [nn.Flatten()]

    if cfg.model_kind == 'dlgn':
        d = in_width
        for _ in range(layers_n):
            modules.append(LutLayer(d, k, n_inputs=n_inputs, connections='random'))
            d = k
        modules.append(GroupSum(k=out_width, tau=1.0))
    else:  # logic_mlp
        d = in_width
        for _ in range(layers_n - 1):
            modules.append(LutLayer(d, k, n_inputs=n_inputs, connections='random'))
            d = k
        modules.append(LutLayer(d, out_width, n_inputs=n_inputs, connections='random'))

    model = nn.Sequential(*modules)
    model.eval()
    return model, in_width


def estimate_genome(genome: dict, cfg: SearchConfig, trial_seed: int) -> tuple[dict, dict]:
    """genomeのアーキテクチャを組んで IR抽出 → estimate_resources する。

    Returns:
        (estimate_dict, ir_json)。estimate_dictは total_luts/logic_depth/
        wire_count/group_sums/node_counts 等 + partial。ir_jsonは ir_to_dict。
    """
    model, in_width = build_arch_model(genome, cfg, trial_seed)
    graph = export_ir(model, input_width=in_width, name=f'nas_{genome}')
    est = estimate_resources(graph)
    partial = any(n.type == 'blackbox' for n in graph.nodes)
    num_params = sum(
        (1 << int(n.params['n_inputs'])) * int(n.params['out_width'])
        for n in graph.nodes if n.type == 'lut_layer'
    )
    est = dict(est)
    est['partial'] = partial
    est['num_params'] = num_params
    est['gates'] = sum(est['node_counts'].values())
    return est, ir_to_dict(graph)


# ------------------------------------------------------------------
# パレート(非劣解)判定 — partial:true は除外(設計裁定6)
# ------------------------------------------------------------------


def _dominates(a: Trial, b: Trial) -> bool:
    """aがbを支配するか(acc高い方が良い / total_luts小さい方が良い)。"""
    better_or_eq = a.acc >= b.acc and a.total_luts <= b.total_luts
    strictly = a.acc > b.acc or a.total_luts < b.total_luts
    return better_or_eq and strictly


def pareto_front(trials: list[Trial]) -> list[int]:
    """非劣解のtrial_id集合を返す(state=='done'かつpartial:false のみ対象)。"""
    cands = [t for t in trials if t.state == 'done' and not t.partial]
    front: list[int] = []
    for t in cands:
        if not any(_dominates(o, t) for o in cands if o.trial_id != t.trial_id):
            front.append(t.trial_id)
    return sorted(front)


# ------------------------------------------------------------------
# 台帳の永続化(JSONL)
# ------------------------------------------------------------------


def _search_dir(cfg: SearchConfig) -> Path:
    return Path(cfg.out_root) / cfg.search_id


def _ledger_path(cfg: SearchConfig) -> Path:
    return _search_dir(cfg) / 'trials.jsonl'


def load_ledger(cfg: SearchConfig) -> list[Trial]:
    """台帳を読み込む(再開用)。存在しなければ空。"""
    path = _ledger_path(cfg)
    if not path.exists():
        return []
    trials: list[Trial] = []
    for line in path.read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if not line:
            continue
        trials.append(Trial(**json.loads(line)))
    return trials


def _append_ledger(cfg: SearchConfig, trial: Trial) -> None:
    path = _ledger_path(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as f:
        f.write(json.dumps(asdict(trial), ensure_ascii=False) + '\n')


def _write_trial_ir(cfg: SearchConfig, trial_id: int, ir_json: dict) -> None:
    path = _search_dir(cfg) / f'trial_{trial_id}.ir.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(ir_json, ensure_ascii=False), encoding='utf-8')


def load_trial_ir(cfg: SearchConfig, trial_id: int) -> dict | None:
    """指定trialのIR JSONを読む(EDITで開く用)。無ければNone。"""
    path = _search_dir(cfg) / f'trial_{trial_id}.ir.json'
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding='utf-8'))


# ------------------------------------------------------------------
# 1 trial の実行
# ------------------------------------------------------------------


def _select_parent(rng: random.Random, ledger: list[Trial]) -> int | None:
    """現時点のパレートフロントから親を1つ選ぶ(決定的=rngとledgerのみ依存)。"""
    front = pareto_front(ledger)
    if not front:
        done = [t.trial_id for t in ledger if t.state == 'done']
        if not done:
            return None
        return rng.choice(sorted(done))
    return rng.choice(front)


def next_genome(cfg: SearchConfig, ledger: list[Trial], trial_id: int) -> tuple[dict, int | None]:
    """次trialのgenomeと親IDを決める(序盤=random、以降=親から変異)。

    cfg.strategy=='nsga2' のときのみ親選択に非劣ソート+混雑度
    (tools/nas/nsga2.py)を用いる。既定(strategy=='evolution')は従来の
    単純パレートランダム選択のまま(設計裁定: ゴールデン不変)。
    """
    rng = _trial_rng(cfg.seed, trial_id)
    if trial_id < cfg.n_random:
        return random_genome(rng, cfg.space), None
    if cfg.strategy == 'nsga2':
        from tools.nas.nsga2 import select_parent as _nsga2_select_parent
        parent_id = _nsga2_select_parent(rng, ledger)
    else:
        parent_id = _select_parent(rng, ledger)
    if parent_id is None:
        return random_genome(rng, cfg.space), None
    parent = next(t for t in ledger if t.trial_id == parent_id)
    return mutate_genome(rng, parent.genome, cfg.space), parent_id


def run_trial(cfg: SearchConfig, genome: dict, trial_id: int,
              parent_id: int | None) -> Trial:
    """1trialを実行する: 学習+hard eval → 面積見積り → Trialを返す。"""
    trial_seed = _trial_seed(cfg.seed, trial_id)
    trial_out = str(_search_dir(cfg) / f'trial_{trial_id}')

    # 面積見積り(学習に依存しないので先に計算し、IRを保存)
    est, ir_json = estimate_genome(genome, cfg, trial_seed)
    _write_trial_ir(cfg, trial_id, ir_json)

    try:
        train_cfg = genome_to_config(genome, cfg, trial_id)
        result = run_training(train_cfg, out_dir=trial_out)
        acc = float(result.get('hard_acc', 0.0))
        best = float(result.get('best_hard_acc', acc))
        steps = int(result.get('steps', 0))
        state = 'done'
        error = None
    except Exception as e:  # 学習失敗はfailedで記録(探索は継続)
        acc = 0.0
        best = 0.0
        steps = 0
        state = 'failed'
        error = f'{type(e).__name__}: {e}'

    return Trial(
        trial_id=trial_id,
        genome=genome,
        state=state,
        acc=acc,
        best_acc=best,
        total_luts=int(est['total_luts']),
        logic_depth=int(est['logic_depth']),
        wire_count=int(est['wire_count']),
        group_sums=int(est['group_sums']),
        gates=int(est['gates']),
        num_params=int(est['num_params']),
        partial=bool(est['partial']),
        parent_id=parent_id,
        steps=steps,
        seed=trial_seed,
        error=error,
    )


# ------------------------------------------------------------------
# 探索ループ本体
# ------------------------------------------------------------------


def run_search(cfg: SearchConfig, *, stop_check=None, on_trial=None) -> list[Trial]:
    """探索を実行する(中断→再開対応)。

    Args:
        cfg: 探索設定
        stop_check: 呼ぶとTrueで停止するcallable(バックグラウンド停止用、任意)
        on_trial: 各trial完了時に呼ぶcallable(trial)。進捗通知用(任意)

    Returns:
        全trialの台帳(既存分+今回追加分)。
    """
    ledger = load_ledger(cfg)  # 再開: 既存台帳から継続
    start = time.time()

    while len(ledger) < cfg.max_trials:
        if stop_check is not None and stop_check():
            break
        if cfg.time_budget_sec is not None and (time.time() - start) >= cfg.time_budget_sec:
            break

        trial_id = len(ledger)  # 連番(Date/time不使用)
        genome, parent_id = next_genome(cfg, ledger, trial_id)
        trial = run_trial(cfg, genome, trial_id, parent_id)
        _append_ledger(cfg, trial)
        ledger.append(trial)
        if on_trial is not None:
            on_trial(trial)

    return ledger
