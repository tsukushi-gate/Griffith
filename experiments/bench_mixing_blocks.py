"""の mixing ブロック内訳を実測する計測ハーネス。

**最適化は行わない**(計測専用)。bench_lut_kernel.py は出力ヘッド相当
(LutStack 400k/400k/480k, b=1024)だけを測っており、全体 1.47 s/it のうち
約 21.6% しかカバーしていない。残り約78%(embedding / token-mixing /
channel-mixing / GroupSum / optimizer 等)は未プロファイルであり、本スクリプトが
その内訳を初めて実測する。

■ 代理モデルについて(重要)
    LogicSeqMixer は feature/lut-kernel-opt ブランチに存在しないため、
    ``experiments/tsmix1_proxy.py`` に mixer worktree の
    ``griffith_logic/models/logic_seq_mixer.py`` を**逐語コピー**して用いる
    (相対import 3行のみ絶対importへ書き換え)。依存部品
    (models/rddlgn.py の LutStack/LogicEmbedding、group_sum.py、
     blocks/attention.py)は両ブランチで byte 単位同一であることを diff で確認済み。
    構造の一致は count_parameters()=40,849,536 と count_gates()=1,528,952 が
    実 run(wandb 524l1cnd)の公称値と厳密一致することで検証している
    (``--verify-structure`` が既定で毎回照合し、ズレたら中断する)。
    **実 run との差分はデータのみ**(WMT'14 の実トークンでなく乱数トークン)で、
    形状・語彙・バッチ・損失・optimizer は train_logic_seq_mixer.py と同一。
    LutLayer 実装だけは本ブランチ(=最適化対象)のものを意図的に使う。

■ 計測方法
    1. ブロック内訳: CUDA Event を forward hook / tensor grad hook に仕込み、
       **forward と backward を部品ごとに個別計測**する。モデルが直鎖構造なので
       「gradが出力に届いた時刻 → gradが入力に届いた時刻」がその部品のbackward。
       measured の合計と全体との差は 'unattributed' として必ず明示する
       (差を隠さないための検算枠)。
    2. op内訳: torch.profiler。分母は**カーネルレベルのエントリのみ**
       (bench_lut_kernel.py の _is_device_level_event と同一規則)。op レベル行
       (aten::copy_ 等)と実体カーネル行を両方足すと**ちょうど2倍**になる
       二重計上バグを避けるため。profiler が印字する "Self CUDA time total" と
       自前の分母が一致するかを毎回検算・出力する。
    3. Amdahl: 実測占有率のみから「そこをX倍にしたときの全体速度比」を機械計算する
       (主観の順位付けを混ぜない)。

使用例:
    uv run python experiments/bench_mixing_blocks.py --scale tiny --no-wandb  # ローカル確認
    uv run modal run --detach scripts/modal_app.py --background \\
        --script bench_mixing_blocks.py --args "--scale paper"

出力:
    results/kernel2/mixing_<device>_<YYYYmmdd_HHMMSS>.json  (+ wandb Artifact)
"""

import argparse
import json
import os
import re
import statistics
import sys
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone

import torch
import torch.nn.functional as F
from torch.autograd import DeviceType
from torch.profiler import ProfilerActivity, profile

# experiments/ 直下のユーティリティをimportできるようにする(Modal実行でも同様)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data import TOKEN_BOS, TOKEN_EOS, TOKEN_PAD  # noqa: E402
from tsmix1_proxy import LogicSeqMixer, LogicSeqMixerConfig  # noqa: E402

from griffith_logic.cuda import cuda_available  # noqa: E402

JST = timezone(timedelta(hours=9))

# ---------------------------------------------------------------------------
# 計測パラメータ(すべて定数として明示。数値の推測をしないため)
# ---------------------------------------------------------------------------

# 実 run の学習設定(出典: experiments/train_logic_seq_mixer.py(mixer 実験ブランチ時点の版))。
BATCH_SIZE = 64
LR = 0.05
WEIGHT_DECAY = 0.001
BETAS = (0.9, 0.999)
EPS = 1e-8
LABEL_SMOOTHING = 0.1
# emb_reg の重みは step<1000 で 0.0(線形ランプ開始前)。1 step 計測なので step=0 相当。
EMB_REG_WEIGHT = 0.0

# 公称構造値(wandb run 524l1cnd / logic_seq_mixer.py の docstring)。
# 代理モデルが実モデルと厳密一致していることの検証に使う。ズレたら計測を中断する。
EXPECTED_PARAMS = 40_849_536
EXPECTED_GATES = 1_528_952

WARMUP = 3
ITERS = 10
PROFILE_STEPS = 3
PROFILE_TOP_K = 25

TINY_WARMUP = 2
TINY_ITERS = 5
TINY_PROFILE_STEPS = 2
TINY_BATCH = 8

# profiler から構造化して抜き出す演算(名前の部分一致)。
# bench_lut_kernel.py の分類(copy_/lut_*)に、mixing ブロックで効きそうな一般opを足してある。
PROFILE_OP_PATTERNS = {
    'copy_': 'aten::copy_',
    'lut_forward_kernel': 'lut_forward_kernel',
    'lut_backward_w_kernel': 'lut_backward_w_kernel',
    'lut_backward_x_kernel': 'lut_backward_x_kernel',
    'sum': 'aten::sum',
    'mm': 'aten::mm',
    'bmm': 'aten::bmm',
    'cat': 'aten::cat',
    'index_select': 'aten::index_select',
    'gather': 'aten::gather',
    'softmax': 'softmax',
    'sigmoid': 'sigmoid',
    'mul': 'aten::mul',
    'add': 'aten::add',
    'elementwise_kernel': 'elementwise_kernel',
    'reduce_kernel': 'reduce_kernel',
    'adam': 'adam',
}

# ブロック種別への分類。計測対象モジュール名(下記 iter_leaf_targets のラベル)の
# 接頭辞で決める。単一情報源にするためここだけを見ればよい。
CATEGORY_RULES = (
    ('embedding', 'embedding'),
    ('embed_proj', 'embed_proj'),
    ('context_proj', 'context_proj'),
    ('token_mix', 'token_mix'),
    ('channel_mix', 'channel_mix'),
    ('head', 'head'),
    ('group_sum', 'group_sum'),
    ('cross', 'cross'),
)

# Amdahl則で提示する高速化倍率(∞=そのブロックを完全にゼロ化した理想上限)
AMDAHL_SPEEDUPS = (2.0, float('inf'))


def categorize(label: str) -> str:
    """計測ラベルをブロック種別へ分類する(未知は 'other')。"""
    for category, token in CATEGORY_RULES:
        if token in label:
            return category
    return 'other'


# ---------------------------------------------------------------------------
# デバイス / 同期(bench_lut_kernel.py と同じ規約)
# ---------------------------------------------------------------------------

def detect_device() -> str:
    """cuda → mps → cpu の順で利用可能なデバイスを返す。"""
    if torch.cuda.is_available():
        return 'cuda'
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


def sync(device: str) -> None:
    """デバイスの非同期実行を待つ(計測の正確性に必須)。"""
    if device == 'cuda':
        torch.cuda.synchronize()
    elif device == 'mps':
        torch.mps.synchronize()


def reset_peak_memory(device: str) -> None:
    if device == 'cuda':
        torch.cuda.reset_peak_memory_stats()


def peak_memory_bytes(device: str) -> int | None:
    if device == 'cuda':
        return int(torch.cuda.max_memory_allocated())
    if device == 'mps' and hasattr(torch.mps, 'driver_allocated_memory'):
        return int(torch.mps.driver_allocated_memory())
    return None


def env_report(device: str) -> dict:
    try:
        import triton
        triton_version = getattr(triton, '__version__', 'unknown')
    except Exception:
        triton_version = None

    info = {
        'device': device,
        'torch_version': torch.__version__,
        'triton_version': triton_version,
        'cuda_is_available': torch.cuda.is_available(),
        'griffith_cuda_kernel_available': cuda_available(),
    }
    if device == 'cuda':
        info['gpu_name'] = torch.cuda.get_device_name(0)
    print(f'device={device}')
    print(f'  torch={info["torch_version"]}  triton={triton_version or "未インストール"}')
    print(f'  griffith cuda_available()={info["griffith_cuda_kernel_available"]}')
    if 'gpu_name' in info:
        print(f'  GPU: {info["gpu_name"]}')
    return info


def fmt_eta(elapsed: float, done: int, total: int) -> str:
    remaining = (elapsed / done) * (total - done) if done else 0.0
    eta = datetime.now(JST) + timedelta(seconds=remaining)
    return (f'[{elapsed:.1f}s<{remaining:.1f}s, '
            f'終了予定 {eta.strftime("%H:%M:%S")}]')


# ---------------------------------------------------------------------------
# モデル / 1 train step(実 run の train_step と同一手順)
# ---------------------------------------------------------------------------

def build_config(scale: str) -> LogicSeqMixerConfig:
    if scale == 'paper':
        return LogicSeqMixerConfig.from_paper()
    if scale == 'tiny':
        return LogicSeqMixerConfig.tiny()
    raise ValueError(f'未知のscale: {scale}')


def verify_structure(model: LogicSeqMixer, scale: str, strict: bool) -> dict:
    """代理モデルが公称構造値と一致するか照合する。

    paper構成でズレていたら「代理が実モデルと違う」ことを意味するので、
    数値を出す前に落とす(創作した数値を残さないため)。
    """
    params = model.count_parameters()
    gates = model.count_gates()
    result = {
        'params_total': params['total'],
        'gates_total': gates['total'],
        'params_breakdown': params,
        'gates_breakdown': gates,
        'expected_params_total': EXPECTED_PARAMS if scale == 'paper' else None,
        'expected_gates_total': EXPECTED_GATES if scale == 'paper' else None,
    }
    if scale != 'paper':
        result['match'] = None
        print(f'構造検証: scale={scale} は公称値照合の対象外 '
              f'(params={params["total"]:,} gates={gates["total"]:,})')
        return result

    ok = (params['total'] == EXPECTED_PARAMS and gates['total'] == EXPECTED_GATES)
    result['match'] = ok
    print(f'構造検証: params={params["total"]:,} (期待{EXPECTED_PARAMS:,}) / '
          f'gates={gates["total"]:,} (期待{EXPECTED_GATES:,}) → '
          f'{"一致" if ok else "不一致"}')
    if not ok and strict:
        raise SystemExit(
            '代理モデルの構造が公称値と一致しない。計測を中断する'
            '(--no-verify-structure で強行可能だが、その結果は実モデルの代理'
            'として使えない)。'
        )
    return result


def make_batch(config: LogicSeqMixerConfig, batch: int, device: str):
    """実 run と同一形状の1バッチを乱数で作る。

    train_logic_seq_mixer.py の prepare_batch と同じく tgt_full=[BOS, tgt, EOS]
    (長さ seq_len+2)を作り、tgt_input=tgt_full[:, :-1] / label=tgt_full[:, 1:]
    にする。**中身だけが WMT'14 の実トークンでなく乱数**で、形状は厳密に同じ。
    特殊トークン(PAD/BOS/EOS)と衝突しないよう本文は 3 以上からサンプルする。
    """
    n_special = max(TOKEN_PAD, TOKEN_BOS, TOKEN_EOS) + 1
    src = torch.randint(n_special, config.vocab_size, (batch, config.seq_len),
                        device=device)
    body = torch.randint(n_special, config.vocab_size, (batch, config.seq_len),
                         device=device)
    bos = torch.full((batch, 1), TOKEN_BOS, dtype=torch.long, device=device)
    eos = torch.full((batch, 1), TOKEN_EOS, dtype=torch.long, device=device)
    tgt_full = torch.cat([bos, body, eos], dim=1)
    return src, tgt_full


def make_step(model, optimizer, src, tgt_full, vocab_size, timer=None):
    """1 train step。実 run の train_step(label_smoothing=0.1)と同一。

    timer が渡された場合は forward / backward / optimizer の3区間へ
    CUDA Event のマークを打つ(区間の入れ子は許容。合計時に混ぜない)。
    """
    tgt_input = tgt_full[:, :-1]
    tgt_label = tgt_full[:, 1:]

    def step():
        model.train()
        optimizer.zero_grad()
        if timer:
            timer.mark('__forward__', 'start')
        logits, emb_loss = model(src, tgt_input)
        main_loss = F.cross_entropy(
            logits.reshape(-1, vocab_size),
            tgt_label.reshape(-1),
            label_smoothing=LABEL_SMOOTHING,
            ignore_index=TOKEN_PAD,
        )
        total_loss = main_loss + EMB_REG_WEIGHT * emb_loss
        if timer:
            timer.mark('__forward__', 'end')
            timer.mark('__backward__', 'start')
        total_loss.backward()
        if timer:
            timer.mark('__backward__', 'end')
            timer.mark('__optimizer__', 'start')
        optimizer.step()
        if timer:
            timer.mark('__optimizer__', 'end')

    return step


def timed_median(step_fn, device: str, warmup: int, iters: int) -> dict:
    """step_fn を warmup 回空回し後 iters 回計測し、中央値等(秒)を返す。"""
    for _ in range(warmup):
        step_fn()
    sync(device)

    samples = []
    for _ in range(iters):
        sync(device)
        t0 = time.perf_counter()
        step_fn()
        sync(device)
        samples.append(time.perf_counter() - t0)

    return {
        'median_s': statistics.median(samples),
        'mean_s': statistics.fmean(samples),
        'min_s': min(samples),
        'max_s': max(samples),
        'stdev_s': statistics.stdev(samples) if len(samples) > 1 else 0.0,
        'iters': iters,
        'warmup': warmup,
    }


# ---------------------------------------------------------------------------
# ブロック内訳: CUDA Event による forward/backward の部品別計測
# ---------------------------------------------------------------------------

class SegmentTimer:
    """CUDA Event で「区間の開始/終了」をマークし、後で ms を取り出す。

    直鎖モデルなので、部品Mについて
      forward : forward_pre_hook(start) 〜 forward_hook(end)
      backward: Mの**出力**にgradが届いた時刻(start) 〜 Mの**入力**にgradが届いた
                時刻(end)
    が、それぞれMのforward/backwardの実時間になる。tensor.register_hook は
    「そのテンソルのgradが計算された直後」に呼ばれるため、この対応が成り立つ。
    """

    def __init__(self, device: str):
        self.device = device
        self.enabled = True
        # cuda では CUDA Event(非同期実行を正しく計測できる唯一の手段)。
        # cpu/mps では Event が無いので、同期+perf_counter で代用する。
        # **代用モードは配線の検証用**であり、GPU計測の代わりにはならない
        # (mode を JSON に必ず残して取り違えを防ぐ)。
        self.mode = 'cuda_event' if device == 'cuda' else 'wallclock_sync'
        self.marks: list[tuple[str, str, object]] = []

    def mark(self, label: str, phase: str) -> None:
        if self.mode == 'cuda_event':
            ev = torch.cuda.Event(enable_timing=True)
            ev.record()
            self.marks.append((label, phase, ev))
        else:
            sync(self.device)
            self.marks.append((label, phase, time.perf_counter()))

    def reset(self) -> None:
        self.marks = []

    def _delta_ms(self, start, end) -> float:
        if self.mode == 'cuda_event':
            return start.elapsed_time(end)
        return (end - start) * 1000.0

    def collect(self) -> dict[str, float]:
        """(label, phase) の対から区間長(ms)を集計して返す。

        同一labelが複数回現れる(ブロックが複数段ある)場合は合計する。
        start/end が対にならなかったものは無視し、件数を warnings で返せるよう
        別途カウントする。
        """
        sync(self.device)
        pending: dict[str, object] = {}
        totals: dict[str, float] = defaultdict(float)
        counts: dict[str, int] = defaultdict(int)
        unmatched = 0
        for label, phase, ev in self.marks:
            if phase == 'start':
                pending[label] = ev
            else:
                start_ev = pending.pop(label, None)
                if start_ev is None:
                    unmatched += 1
                    continue
                totals[label] += self._delta_ms(start_ev, ev)
                counts[label] += 1
        # 対にならなかったラベルを名指しで残す。件数だけだと「何が測れていないか」が
        # 分からず、内訳の欠落を見落とすため。
        # 既知の未対応: 埋め込みは入力が整数トークンでgradが流れないため
        # backward の 'end' が発火しない(= backward時間は unattributed_backward 側)。
        self.last_unmatched = unmatched + len(pending)
        self.last_unmatched_labels = sorted(pending)
        self.last_counts = dict(counts)
        return dict(totals)


def iter_leaf_targets(model: LogicSeqMixer) -> list[tuple[str, torch.nn.Module]]:
    """内訳を取る「葉」部品の (ラベル, モジュール) 一覧。

    ブロックの中身(channel_mix_in / token_mix / channel_mix_out)まで割る。
    ラベルは categorize() が種別へ落とせる語を必ず含める。
    """
    targets: list[tuple[str, torch.nn.Module]] = []
    targets.append(('src_embedding', model.src_embedding))
    if model.tgt_embedding is not model.src_embedding:
        targets.append(('tgt_embedding', model.tgt_embedding))

    targets.append(('enc_embed_proj', model.encoder.embed_proj))
    for i, block in enumerate(model.encoder.blocks):
        targets.append((f'enc{i}_channel_mix_in', block.channel_mix_in))
        targets.append((f'enc{i}_token_mix', block.token_mix))
        targets.append((f'enc{i}_channel_mix_out', block.channel_mix_out))

    dec = model.decoder
    targets.append(('dec_embed_proj', dec.embed_proj))
    targets.append(('dec_context_proj', dec.context_proj))
    for i, block in enumerate(dec.blocks):
        targets.append((f'dec{i}_channel_mix_in', block.channel_mix_in))
        targets.append((f'dec{i}_token_mix', block.token_mix))
        targets.append((f'dec{i}_channel_mix_out', block.channel_mix_out))
    if dec.cross_attn is not None:
        targets.append(('dec_cross_attn', dec.cross_attn))
    if dec.cross_out is not None:
        targets.append(('dec_cross_out', dec.cross_out))
    targets.append(('dec_head', dec.head))
    targets.append(('dec_group_sum', dec.group_sum))
    return targets


def _first_grad_tensor(obj):
    """入出力から「gradが流れる最初のTensor」を取り出す(無ければNone)。"""
    if isinstance(obj, torch.Tensor):
        return obj if obj.requires_grad else None
    if isinstance(obj, tuple | list):
        for item in obj:
            found = _first_grad_tensor(item)
            if found is not None:
                return found
    return None


def install_segment_hooks(model: LogicSeqMixer, timer: SegmentTimer) -> tuple[list, list]:
    """葉部品へ forward/backward 計測フックを仕掛ける。

    Returns:
        (handles, labels) — handles は remove() 用、labels は計測対象ラベル一覧。
    """
    handles = []
    labels = []
    for label, module in iter_leaf_targets(model):
        labels.append(label)

        def pre_hook(mod, inputs, label=label):
            timer.mark(f'{label}/fwd', 'start')
            tensor = _first_grad_tensor(inputs)
            if tensor is not None:
                # 入力にgradが届く = この部品のbackwardが終わった時刻
                tensor.register_hook(
                    lambda g, label=label: (timer.mark(f'{label}/bwd', 'end'), g)[1]
                )
            return None

        def post_hook(mod, inputs, output, label=label):
            timer.mark(f'{label}/fwd', 'end')
            tensor = _first_grad_tensor(output)
            if tensor is not None:
                # 出力にgradが届く = この部品のbackwardが始まる時刻
                tensor.register_hook(
                    lambda g, label=label: (timer.mark(f'{label}/bwd', 'start'), g)[1]
                )
            return None

        handles.append(module.register_forward_pre_hook(pre_hook))
        handles.append(module.register_forward_hook(post_hook))
    return handles, labels


def measure_blocks(model, optimizer, src, tgt_full, config, device: str,
                   warmup: int, iters: int, uninstrumented_step_s: float) -> dict:
    """フック計測を iters 回まわし、部品別 forward/backward の中央値(秒)を返す。"""
    timer = SegmentTimer(device)
    handles, labels = install_segment_hooks(model, timer)
    step = make_step(model, optimizer, src, tgt_full, config.vocab_size, timer=timer)

    try:
        for _ in range(warmup):
            timer.reset()
            step()
        sync(device)

        per_iter: list[dict[str, float]] = []
        wall: list[float] = []
        unmatched_total = 0
        unmatched_labels: set[str] = set()
        for _ in range(iters):
            timer.reset()
            sync(device)
            t0 = time.perf_counter()
            step()
            sync(device)
            wall.append(time.perf_counter() - t0)
            per_iter.append(timer.collect())
            unmatched_total += getattr(timer, 'last_unmatched', 0)
            unmatched_labels.update(getattr(timer, 'last_unmatched_labels', []))
    finally:
        for h in handles:
            h.remove()

    # ラベルごとに iters 分の中央値(ms→s)
    keys = sorted({k for d in per_iter for k in d})
    medians = {
        k: statistics.median([d.get(k, 0.0) for d in per_iter]) / 1000.0
        for k in keys
    }

    total_fwd = medians.get('__forward__', 0.0)
    total_bwd = medians.get('__backward__', 0.0)
    total_opt = medians.get('__optimizer__', 0.0)
    instrumented_wall = statistics.median(wall)

    rows = []
    for label in labels:
        fwd = medians.get(f'{label}/fwd', 0.0)
        bwd = medians.get(f'{label}/bwd', 0.0)
        rows.append({
            'label': label,
            'category': categorize(label),
            'forward_s': fwd,
            'backward_s': bwd,
            'total_s': fwd + bwd,
            'backward_measured': f'{label}/bwd' in medians,
        })

    measured_fwd = sum(r['forward_s'] for r in rows)
    measured_bwd = sum(r['backward_s'] for r in rows)

    # 種別ごとに集約
    by_category: dict[str, dict] = {}
    for r in rows:
        c = by_category.setdefault(
            r['category'], {'forward_s': 0.0, 'backward_s': 0.0, 'labels': []})
        c['forward_s'] += r['forward_s']
        c['backward_s'] += r['backward_s']
        c['labels'].append(r['label'])
    for c in by_category.values():
        c['total_s'] = c['forward_s'] + c['backward_s']

    # 帰属できなかった分を必ず明示する(loss/cat/chunk/soft_and/soft_xor/reshape/
    # embedding backward など、葉部品の外側で起きている処理)。
    by_category['unattributed_forward'] = {
        'forward_s': max(total_fwd - measured_fwd, 0.0),
        'backward_s': 0.0,
        'total_s': max(total_fwd - measured_fwd, 0.0),
        'labels': ['(loss前のcat/chunk/soft_and/soft_xor/reshape等、葉部品の外側)'],
    }
    by_category['unattributed_backward'] = {
        'forward_s': 0.0,
        'backward_s': max(total_bwd - measured_bwd, 0.0),
        'total_s': max(total_bwd - measured_bwd, 0.0),
        'labels': ['(embedding backward・elementwiseのbackward等)'],
    }
    by_category['optimizer'] = {
        'forward_s': 0.0, 'backward_s': total_opt, 'total_s': total_opt,
        'labels': ['AdamW.step()'],
    }

    attributed_total = sum(c['total_s'] for c in by_category.values())
    # 検算用の分母は「実測した3区間の合計」= forward + backward + optimizer
    step_total = total_fwd + total_bwd + total_opt
    for c in by_category.values():
        c['share_of_step'] = (c['total_s'] / step_total) if step_total else None

    return {
        'supported': True,
        'timer_mode': timer.mode,
        'iters': iters,
        'warmup': warmup,
        'rows': rows,
        'by_category': by_category,
        'totals': {
            'forward_s': total_fwd,
            'backward_s': total_bwd,
            'optimizer_s': total_opt,
            'event_step_total_s': step_total,
            'measured_forward_s': measured_fwd,
            'measured_backward_s': measured_bwd,
            'attributed_total_s': attributed_total,
            'instrumented_wall_s': instrumented_wall,
            'uninstrumented_wall_s': uninstrumented_step_s,
            # フック計測のオーバーヘッド(この分だけ内訳側は水増しされている)
            'instrumentation_overhead_s': instrumented_wall - uninstrumented_step_s,
            'instrumentation_overhead_ratio': (
                (instrumented_wall / uninstrumented_step_s)
                if uninstrumented_step_s else None
            ),
            'event_vs_wall_ratio': (
                (step_total / instrumented_wall) if instrumented_wall else None
            ),
            'unmatched_marks': unmatched_total,
            'unmatched_labels': sorted(unmatched_labels),
        },
    }


# ---------------------------------------------------------------------------
# profiler(分母 = カーネルレベルのみ。bench_lut_kernel.py と同一規則)
# ---------------------------------------------------------------------------

def _is_device_level_event(ev) -> bool:
    """そのエントリが「カーネルレベル」(デバイス側)かどうか。

    torch/autograd/profiler_util.py の _build_table が "Self CUDA time total" を
    合計するときと同じ規則。ATen op レベル(aten::copy_ 等)は device_type=CPU なので
    False になる。**両方を足すとちょうど2倍の二重計上になる**(実際に踏んだ)。
    実装は experiments/bench_lut_kernel.py の同名関数と同一。
    """
    dev_type = getattr(ev, 'device_type', None)
    if dev_type is None:
        return False
    if dev_type == DeviceType.CPU:
        return bool(getattr(ev, 'is_legacy', False))
    device_levels = [DeviceType.CUDA]
    for name in ('PrivateUse1', 'MTIA', 'XPU'):
        member = getattr(DeviceType, name, None)
        if member is not None:
            device_levels.append(member)
    return dev_type in device_levels and not getattr(ev, 'is_user_annotation', False)


def _parse_table_totals(table: str) -> dict:
    """profilerが印字した "Self CPU/CUDA time total" をusに直して返す(検証用)。"""
    units = {'us': 1.0, 'ms': 1e3, 's': 1e6}
    out: dict[str, float | None] = {
        'table_self_cpu_total_us': None,
        'table_self_device_total_us': None,
    }
    pattern = re.compile(
        r'Self\s+(?P<what>[A-Za-z0-9_]+)\s+time total:\s*'
        r'(?P<val>[0-9.]+)(?P<unit>us|ms|s)\b'
    )
    for m in pattern.finditer(table or ''):
        value = float(m.group('val')) * units[m.group('unit')]
        if m.group('what').upper() == 'CPU':
            out['table_self_cpu_total_us'] = value
        else:
            out['table_self_device_total_us'] = value
    return out


def profile_step(step_fn, device: str, steps: int) -> dict:
    """torch.profilerで数stepを取り、self time上位のテーブルと配列を返す。"""
    activities = [ProfilerActivity.CPU]
    use_cuda = device == 'cuda'
    if use_cuda:
        activities.append(ProfilerActivity.CUDA)

    step_fn()  # プロファイル対象外のwarmup
    sync(device)

    with profile(activities=activities) as prof:
        for _ in range(steps):
            step_fn()
        sync(device)

    sort_key = 'self_cuda_time_total' if use_cuda else 'self_cpu_time_total'
    try:
        table = prof.key_averages().table(sort_by=sort_key, row_limit=PROFILE_TOP_K)
    except Exception as exc:
        print(f'  警告: profiler table(sort_by={sort_key}) 失敗: {exc}')
        table = prof.key_averages().table(row_limit=PROFILE_TOP_K)

    rows = []
    for ev in prof.key_averages():
        rows.append({
            'name': ev.key,
            'count': int(ev.count),
            'device_type': str(ev.device_type),
            'is_kernel_level': _is_device_level_event(ev),
            'self_cpu_time_total_us': float(ev.self_cpu_time_total),
            'cpu_time_total_us': float(ev.cpu_time_total),
            'self_device_time_total_us': float(
                getattr(ev, 'self_device_time_total', 0.0) or 0.0
            ),
        })
    key = 'self_device_time_total_us' if use_cuda else 'self_cpu_time_total_us'
    rows.sort(key=lambda r: r[key], reverse=True)

    self_cpu_total = sum(r['self_cpu_time_total_us'] for r in rows)
    device_kernel_total = sum(
        r['self_device_time_total_us'] for r in rows if r['is_kernel_level']
    )
    device_op_total = sum(
        r['self_device_time_total_us'] for r in rows if not r['is_kernel_level']
    )
    device_naive_sum = device_kernel_total + device_op_total

    if device_kernel_total > 0:
        device_denominator = device_kernel_total
        denominator_source = 'kernel_level'
    elif device_op_total > 0:
        device_denominator = device_op_total
        denominator_source = 'op_level_fallback'
    else:
        device_denominator = 0.0
        denominator_source = 'none'

    breakdown = {}
    for label, pattern in PROFILE_OP_PATTERNS.items():
        hits = [r for r in rows if pattern in r['name']]
        kernel_hits = [r for r in hits if r['is_kernel_level']]
        op_hits = [r for r in hits if not r['is_kernel_level']]
        dev_kernel = sum(r['self_device_time_total_us'] for r in kernel_hits)
        dev_op = sum(r['self_device_time_total_us'] for r in op_hits)
        both = bool(kernel_hits) and bool(op_hits)
        dev_us = max(dev_kernel, dev_op) if both else dev_kernel + dev_op
        cpu_us = sum(r['self_cpu_time_total_us'] for r in hits)
        count = (max(sum(r['count'] for r in kernel_hits),
                     sum(r['count'] for r in op_hits))
                 if both else sum(r['count'] for r in hits))
        breakdown[label] = {
            'count': count,
            'self_cpu_time_total_us': cpu_us,
            'self_device_time_total_us': dev_us,
            'self_device_time_kernel_level_us': dev_kernel,
            'self_device_time_op_level_us': dev_op,
            'matched_both_levels': both,
            'device_share': (
                (dev_us / device_denominator) if device_denominator else None
            ),
            'cpu_share': (cpu_us / self_cpu_total) if self_cpu_total else None,
            'matched_names': [
                {'name': r['name'], 'is_kernel_level': r['is_kernel_level'],
                 'count': r['count'],
                 'self_device_time_total_us': r['self_device_time_total_us']}
                for r in hits
            ],
        }

    # 分類できたカーネル時間の合計(重複を避けるためカーネルレベルのみ、名前で一意化)
    matched_kernel_names = set()
    for label, pattern in PROFILE_OP_PATTERNS.items():
        for r in rows:
            if r['is_kernel_level'] and pattern in r['name']:
                matched_kernel_names.add(r['name'])
    classified_us = sum(
        r['self_device_time_total_us'] for r in rows
        if r['is_kernel_level'] and r['name'] in matched_kernel_names
    )
    unclassified_rows = [
        r for r in rows
        if r['is_kernel_level'] and r['name'] not in matched_kernel_names
        and r['self_device_time_total_us'] > 0
    ]

    result = {
        'steps': steps, 'sort_by': sort_key, 'table': table,
        'self_cpu_time_total_us': self_cpu_total,
        'self_device_time_total_us': device_denominator,
        'self_device_time_denominator_us': device_denominator,
        'self_device_time_denominator_source': denominator_source,
        'self_device_time_kernel_level_us': device_kernel_total,
        'self_device_time_op_level_us': device_op_total,
        'self_device_time_naive_sum_us': device_naive_sum,
        'naive_sum_over_denominator': (
            (device_naive_sum / device_denominator) if device_denominator else None
        ),
        'classified_kernel_time_us': classified_us,
        'unclassified_kernel_time_us': device_kernel_total - classified_us,
        'unclassified_top': [
            {'name': r['name'], 'count': r['count'],
             'self_device_time_total_us': r['self_device_time_total_us']}
            for r in unclassified_rows[:PROFILE_TOP_K]
        ],
        'op_breakdown': breakdown,
        'top_ops': rows[:PROFILE_TOP_K],
    }
    result.update(_parse_table_totals(table))
    return result


def print_profile_consistency(prof_res: dict) -> None:
    """算出した分母がprofilerの印字合計と一致しているかを検算表示する。"""
    denom = prof_res['self_device_time_denominator_us']
    table_dev = prof_res.get('table_self_device_total_us')
    table_cpu = prof_res.get('table_self_cpu_total_us')
    naive = prof_res['self_device_time_naive_sum_us']
    ratio = prof_res['naive_sum_over_denominator']
    ratio_s = f'{ratio:.3f}' if ratio is not None else 'n/a'
    dev_match = ('n/a' if not (denom and table_dev)
                 else ('一致' if abs(denom - table_dev) <= 0.01 * table_dev
                       else f'不一致(差{denom - table_dev:.1f}us)'))
    cpu_match = ('n/a' if not table_cpu
                 else ('一致'
                       if abs(prof_res['self_cpu_time_total_us'] - table_cpu)
                       <= 0.01 * table_cpu else '不一致'))
    print(f'  検算: 分母({prof_res["self_device_time_denominator_source"]})='
          f'{denom / 1000:.3f}ms / profiler印字='
          f'{(table_dev / 1000) if table_dev else float("nan"):.3f}ms → {dev_match}'
          f' | naive合計={naive / 1000:.3f}ms (naive/分母={ratio_s})'
          f' | CPU合計={prof_res["self_cpu_time_total_us"] / 1000:.3f}ms vs 印字='
          f'{(table_cpu / 1000) if table_cpu else float("nan"):.3f}ms → {cpu_match}')


# ---------------------------------------------------------------------------
# Amdahl 優先リスト(実測占有率のみから機械計算。主観の順位付けをしない)
# ---------------------------------------------------------------------------

def amdahl_speedup(share: float, factor: float) -> float:
    """占有率 share の部分を factor 倍にしたときの全体速度比。

    S = 1 / ((1 - share) + share / factor)。factor=inf なら 1/(1-share)。
    """
    if factor == float('inf'):
        return float('inf') if share >= 1.0 else 1.0 / (1.0 - share)
    return 1.0 / ((1.0 - share) + share / factor)


def build_priority_list(block_res: dict, prof_res: dict) -> list[dict]:
    """ブロック内訳と op 内訳から、実測占有率つきの優先リストを作る。

    **順位は実測占有率の降順のみで決める**(「速そう」という主観を混ぜない)。
    難易度は所感として文字列で添えるが、順位には一切使わない。
    """
    entries: list[dict] = []

    if block_res.get('supported'):
        step_total = block_res['totals']['event_step_total_s']
        for name, cat in block_res['by_category'].items():
            if not step_total or cat['total_s'] <= 0:
                continue
            entries.append({
                'kind': 'block',
                'target': name,
                'measured_s': cat['total_s'],
                'share': cat['total_s'] / step_total,
                'evidence': (
                    f'CUDA Event実測 fwd={cat["forward_s"]:.6f}s + '
                    f'bwd={cat["backward_s"]:.6f}s / '
                    f'step合計={step_total:.6f}s'
                ),
            })

    denom = prof_res.get('self_device_time_denominator_us') or 0.0
    if denom:
        for label, stats in prof_res['op_breakdown'].items():
            if stats['self_device_time_total_us'] <= 0:
                continue
            entries.append({
                'kind': 'op',
                'target': label,
                'measured_s': stats['self_device_time_total_us'] / 1e6,
                'share': stats['device_share'],
                'evidence': (
                    f'profiler self CUDA {stats["self_device_time_total_us"] / 1000:.3f}ms '
                    f'/ 分母(kernel_level) {denom / 1000:.3f}ms, '
                    f'{stats["count"]}回'
                ),
            })

    for e in entries:
        share = e['share'] or 0.0
        e['amdahl'] = {
            ('x2' if f == 2.0 else 'zeroed'): amdahl_speedup(share, f)
            for f in AMDAHL_SPEEDUPS
        }
    entries.sort(key=lambda e: e['share'] or 0.0, reverse=True)
    return entries


# ---------------------------------------------------------------------------
# 表示 / 保存
# ---------------------------------------------------------------------------

def print_block_table(block_res: dict) -> None:
    if not block_res.get('supported'):
        print(f'\n=== ブロック内訳: 計測不可 ({block_res.get("reason")}) ===')
        return
    t = block_res['totals']
    mode = block_res.get('timer_mode')
    if mode != 'cuda_event':
        print(f'\n※ 計測モード={mode}(CUDA Eventではない)。'
              'これは配線検証用の代用計測であり、GPU計測の代わりにはならない。')
    print(f'\n=== ブロック種別ごとの内訳 ({mode}実測, 中央値) ===')
    header = (f'{"category":>24} {"fwd s":>12} {"bwd s":>12} {"total s":>12} '
              f'{"share%":>8}')
    print(header)
    print('-' * len(header))
    ordered = sorted(block_res['by_category'].items(),
                     key=lambda kv: kv[1]['total_s'], reverse=True)
    for name, c in ordered:
        share = c['share_of_step']
        print(f'{name:>24} {c["forward_s"]:>12.6f} {c["backward_s"]:>12.6f} '
              f'{c["total_s"]:>12.6f} '
              f'{(share * 100 if share is not None else float("nan")):>8.2f}')
    print('-' * len(header))
    print(f'{"合計(event3区間)":>24} {t["forward_s"]:>12.6f} '
          f'{t["backward_s"]:>12.6f} {t["event_step_total_s"]:>12.6f} '
          f'{100.0:>8.2f}')
    print(f'  検算: event合計={t["event_step_total_s"]:.6f}s / '
          f'計装ありwall={t["instrumented_wall_s"]:.6f}s '
          f'(比={t["event_vs_wall_ratio"]:.3f}) | '
          f'計装なしwall={t["uninstrumented_wall_s"]:.6f}s '
          f'(計装オーバーヘッド={t["instrumentation_overhead_s"]:+.6f}s) | '
          f'未対応マーク={t["unmatched_marks"]}件 '
          f'{t["unmatched_labels"] or ""}')

    print('\n=== 部品ごとの内訳(葉レベル) ===')
    header = (f'{"label":>24} {"category":>14} {"fwd s":>12} {"bwd s":>12} '
              f'{"total s":>12}')
    print(header)
    print('-' * len(header))
    for r in sorted(block_res['rows'], key=lambda r: r['total_s'], reverse=True):
        print(f'{r["label"]:>24} {r["category"]:>14} {r["forward_s"]:>12.6f} '
              f'{r["backward_s"]:>12.6f} {r["total_s"]:>12.6f}')


def print_op_table(prof_res: dict) -> None:
    denom = prof_res['self_device_time_denominator_us']
    print('\n=== op内訳 (profiler self CUDA, 分母=カーネルレベル合計) ===')
    header = (f'{"op":>22} {"calls":>10} {"self GPU ms":>13} {"GPU share%":>11}')
    print(header)
    print('-' * len(header))
    ordered = sorted(prof_res['op_breakdown'].items(),
                     key=lambda kv: kv[1]['self_device_time_total_us'], reverse=True)
    for label, s in ordered:
        share = s['device_share']
        print(f'{label:>22} {s["count"]:>10} '
              f'{s["self_device_time_total_us"] / 1000:>13.3f} '
              f'{(share * 100 if share is not None else float("nan")):>11.2f}')
    print('-' * len(header))
    print(f'{"分類済み計":>22} {"-":>10} '
          f'{prof_res["classified_kernel_time_us"] / 1000:>13.3f} '
          f'{(prof_res["classified_kernel_time_us"] / denom * 100) if denom else float("nan"):>11.2f}')
    print(f'{"未分類":>22} {"-":>10} '
          f'{prof_res["unclassified_kernel_time_us"] / 1000:>13.3f} '
          f'{(prof_res["unclassified_kernel_time_us"] / denom * 100) if denom else float("nan"):>11.2f}')
    if prof_res['unclassified_top']:
        print('  未分類の上位カーネル:')
        for r in prof_res['unclassified_top'][:8]:
            print(f'    {r["self_device_time_total_us"] / 1000:>10.3f}ms '
                  f'x{r["count"]:<6} {r["name"][:90]}')


def print_priority_table(entries: list[dict]) -> None:
    print('\n=== Amdahl上限つき優先リスト(実測占有率の降順。主観の順位付けなし) ===')
    header = (f'{"kind":>6} {"target":>24} {"実測s":>11} {"占有率%":>9} '
              f'{"x2で全体":>10} {"ゼロ化で全体":>13}')
    print(header)
    print('-' * len(header))
    for e in entries:
        a = e['amdahl']
        zeroed = a['zeroed']
        zeroed_s = '∞' if zeroed == float('inf') else f'{zeroed:.3f}x'
        print(f'{e["kind"]:>6} {e["target"]:>24} {e["measured_s"]:>11.6f} '
              f'{(e["share"] or 0) * 100:>9.2f} {a["x2"]:>9.3f}x {zeroed_s:>13}')
    print('  注: kind=block と kind=op は**別々の分母**(block=1 train stepのevent合計 / '
          'op=profilerのカーネルレベル合計)。混ぜて足さないこと。')


def save_json(payload: dict, device: str, timestamp: str) -> str:
    out_dir = os.path.join('results', 'kernel2')
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f'mixing_{device}_{timestamp}.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return path


def flat_metrics(payload: dict) -> dict:
    """wandbのsummary向けにフラットな指標辞書を作る。"""
    metrics: dict[str, float] = {}
    metrics['step/median_s'] = payload['train_step']['median_s']
    metrics['step/mean_s'] = payload['train_step']['mean_s']
    if payload.get('peak_memory_bytes'):
        metrics['peak_memory_bytes'] = payload['peak_memory_bytes']

    block = payload['blocks']
    if block.get('supported'):
        for name, c in block['by_category'].items():
            metrics[f'block/{name}/total_s'] = c['total_s']
            if c['share_of_step'] is not None:
                metrics[f'block/{name}/share'] = c['share_of_step']
        for k, v in block['totals'].items():
            if isinstance(v, int | float):
                metrics[f'block/totals/{k}'] = v

    prof = payload['profile']
    metrics['profile/self_gpu_ms'] = prof['self_device_time_total_us'] / 1000
    metrics['profile/self_cpu_ms'] = prof['self_cpu_time_total_us'] / 1000
    metrics['profile/naive_sum_over_denominator'] = (
        prof['naive_sum_over_denominator'] or 0.0
    )
    for label, s in prof['op_breakdown'].items():
        metrics[f'profile/op/{label}_gpu_ms'] = s['self_device_time_total_us'] / 1000
        if s['device_share'] is not None:
            metrics[f'profile/op/{label}_gpu_share'] = s['device_share']
    return metrics


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description='mixingブロックの内訳計測(計測専用)',
    )
    parser.add_argument('--scale', type=str, default='paper',
                        choices=('paper', 'tiny'),
                        help="'paper'=実構成(既定) / 'tiny'=ローカル確認用")
    parser.add_argument('--batch', type=int, default=None,
                        help=f'バッチサイズ(既定: paper={BATCH_SIZE}, tiny={TINY_BATCH})')
    parser.add_argument('--no-wandb', action='store_true', help='wandbを無効化')
    parser.add_argument('--no-verify-structure', action='store_true',
                        help='代理モデルの構造照合に失敗しても続行する(非推奨)')
    parser.add_argument('--skip-profile', action='store_true',
                        help='torch.profilerによるop内訳をスキップ')
    parser.add_argument('--tag', type=str, default='', help='結果ラベル')
    parser.add_argument('--device', type=str, default=None,
                        help='デバイス(既定は自動検出 cuda→mps→cpu)')
    parser.add_argument('--seed', type=int, default=0, help='乱数シード')
    try:
        from wandb_utils import add_experiment_args
        add_experiment_args(parser)
    except Exception:
        parser.add_argument('--notes', type=str, default=None)
        parser.add_argument('--hypothesis', type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    started = datetime.now(JST)
    print(f'開始: {started.strftime("%Y-%m-%d %H:%M:%S")} JST')

    device = args.device or detect_device()
    env = env_report(device)

    is_tiny = args.scale == 'tiny'
    batch = args.batch or (TINY_BATCH if is_tiny else BATCH_SIZE)
    warmup = TINY_WARMUP if is_tiny else WARMUP
    iters = TINY_ITERS if is_tiny else ITERS
    profile_steps = TINY_PROFILE_STEPS if is_tiny else PROFILE_STEPS

    config = build_config(args.scale)
    print(f'\nモデル構築中 (scale={args.scale}, batch={batch}) ...')
    t0 = time.perf_counter()
    model = LogicSeqMixer(config, device=device).to(device)
    print(f'  構築完了 {time.perf_counter() - t0:.1f}s')

    structure = verify_structure(model, args.scale, strict=not args.no_verify_structure)

    src, tgt_full = make_batch(config, batch, device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY,
        betas=BETAS, eps=EPS,
    )
    print(f'入力形状: src={tuple(src.shape)} tgt_full={tuple(tgt_full.shape)} '
          f'(tgt_input={tuple(tgt_full[:, :-1].shape)})')

    # 1) 素の 1 train step(計装なし。これが全体の基準値)
    print(f'\n[1/3] 1 train step の実測 (warmup={warmup}, iters={iters}) ...')
    plain_step = make_step(model, optimizer, src, tgt_full, config.vocab_size)
    reset_peak_memory(device)
    step_stats = timed_median(plain_step, device, warmup, iters)
    peak = peak_memory_bytes(device)
    print(f'  1 train step = {step_stats["median_s"]:.6f} s/it (中央値, '
          f'min={step_stats["min_s"]:.6f} max={step_stats["max_s"]:.6f}) '
          f'peak={(peak / 1024 ** 3) if peak else float("nan"):.2f}GiB')

    # 2) ブロック内訳(CUDA Eventフック計測)
    print(f'\n[2/3] ブロック内訳の実測 (CUDA Event, iters={iters}) ...')
    block_res = measure_blocks(
        model, optimizer, src, tgt_full, config, device, warmup, iters,
        step_stats['median_s'],
    )
    print_block_table(block_res)

    # 3) op内訳(profiler)
    prof_res: dict = {}
    if args.skip_profile:
        print('\n[3/3] profiler: --skip-profile によりスキップ')
    else:
        print(f'\n[3/3] op内訳の実測 (torch.profiler, {profile_steps} steps) ...')
        prof_res = profile_step(plain_step, device, profile_steps)
        print(f'--- profiler table (sort_by={prof_res["sort_by"]}) ---')
        print(prof_res['table'])
        print_profile_consistency(prof_res)
        print_op_table(prof_res)

    priority = build_priority_list(block_res, prof_res) if prof_res or block_res else []
    if priority:
        print_priority_table(priority)

    timestamp = started.strftime('%Y%m%d_%H%M%S')
    payload = {
        'task': 'mixing-blocks-bench',
        'tag': args.tag,
        'proxy_model': {
            'kind': 'verbatim_copy',
            'source': ('griffith_logic/models/logic_seq_mixer.py'
                       '(mixer 実験ブランチ時点の版)'),
            'module': 'experiments/tsmix1_proxy.py',
            'note': ('相対import3行のみ絶対importへ書換。依存部品(rddlgn.py / '
                     'group_sum.py / blocks/attention.py)は両ブランチでbyte同一。'
                     'layers/lut.pyのみ本ブランチ(最適化対象)の実装を使用。'
                     'データはWMT\'14実トークンでなく乱数トークン。'),
        },
        'started_at_jst': started.strftime('%Y-%m-%d %H:%M:%S'),
        'finished_at_jst': datetime.now(JST).strftime('%Y-%m-%d %H:%M:%S'),
        'env': env,
        'config': {
            'scale': args.scale,
            'batch': batch,
            'warmup': warmup,
            'iters': iters,
            'profile_steps': profile_steps,
            'lr': LR,
            'weight_decay': WEIGHT_DECAY,
            'label_smoothing': LABEL_SMOOTHING,
            'emb_reg_weight': EMB_REG_WEIGHT,
            'seed': args.seed,
            'model_config': {k: (list(v) if isinstance(v, tuple) else v)
                             for k, v in vars(config).items()},
        },
        'structure': structure,
        'train_step': step_stats,
        'peak_memory_bytes': peak,
        'blocks': block_res,
        'profile': {k: v for k, v in prof_res.items() if k != 'table'},
        'priority': priority,
    }

    json_path = save_json(payload, device, timestamp)
    print(f'\n結果JSON: {os.path.abspath(json_path)}')

    if not args.no_wandb:
        log_to_wandb(args, payload, device, json_path)

    finished = datetime.now(JST)
    print(f'終了: {finished.strftime("%Y-%m-%d %H:%M:%S")} JST '
          f'(所要 {(finished - started).total_seconds():.1f}s)')


def log_to_wandb(args, payload: dict, device: str, json_path: str) -> None:
    """wandbへ指標とJSON Artifactを記録する。失敗時は警告を出して続行。"""
    try:
        from wandb_utils import (
            finish,
            init_wandb,
            log_artifact,
            log_metrics,
            normalize_tags,
        )
    except Exception as exc:
        print(f'警告: wandb_utilsをimportできずwandb記録をスキップ: {exc}')
        return

    try:
        suffix = f'_{args.tag}' if args.tag else ''
        ok = init_wandb(
            run_name=f'mixingbench_{device}_{args.scale}{suffix}',
            config_dict={**payload['config'], 'tag': args.tag, 'device': device},
            device=device,
            tags=['bench', 'kernel-2', device] + normalize_tags(
                model='logic_seq_mixer_proxy', seed=args.seed, series='mixing-blocks-bench',
            ),
            group='kernel-2',
            job_type='eval',
            notes=getattr(args, 'notes', None),
            hypothesis=getattr(args, 'hypothesis', None),
        )
        if not ok:
            print('警告: wandbが無効のため記録をスキップ(結果はJSONに保存済み)')
            return
        log_metrics(flat_metrics(payload))
        log_artifact(
            json_path, name=f'kernel2-mixing-{device}-{args.scale}{suffix}',
            type='benchmark',
            metadata={'task': 'mixing-blocks-bench', 'device': device, 'scale': args.scale},
        )
        finish()
    except Exception as exc:
        print(f'警告: wandb記録に失敗(結果はJSONに保存済み): {exc}')


if __name__ == '__main__':
    main()
