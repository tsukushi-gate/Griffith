"""前段NN + 後段DLGN のハイブリッドモデル訓練スクリプト(HYBRID-1)。

BYPASS系列(R1〜R5)で、**DLGNの弱点は入力層の疎結合そのもの**であることが
分かった(1ゲートあたり2本しか配線できないため入力次元が増えるほど取りこぼす。
R5の直接証拠: mnist784 0.8683 < mnist20x20 0.8994)。固定配線のバイパス・残差・
深層化はいずれもこのボトルネックを解消できなかった。

そこで本スクリプトは分業仮説を検証する:

    「圧縮」は密結合の普通のNNが得意 / 「拡散(組合せ論理の展開)」は疎結合DLGNが得意

    入力(784等) --[前段NN: Linear+ReLU...+Linear+Sigmoid]--> 細い実数ベクトル
                --[Binarize(0.5, STE)]--> **ビット界面(0/1)**
                --[後段DLGN: LutLayer列 + GroupSum]--> クラスロジット

ビット界面から先は完全に二値なので、後段DLGNはそのままHDLに落とせる
(前段NNだけが実数演算として残る = ハードウェア化の対象は界面から後ろ)。

## ポート埋め(--expand-fill)

DLGN層 i のポート数 2·k_i が前層出力数 P を超えるとき、P本の線では全ポートを
埋められない。その余りをどう埋めるかで3方式を比較する:

    none    — 前層のみを見る対照。全入力被覆(unique)のオフセットペアリング配線
    bypass  — まず前層の全出力を1回ずつ使い、**余りは二値化界面(DLGN入力)と
              既出のDLGN中間層出力**から引く(層をまたぐショートカット)
    fanout2 — まず前層の全出力を1回ずつ、余りは前層出力の2周目3周目…を均等に
              (1ゲートあたりのfan-outを最大均等化。「1ゲートから2本の線」)

圧縮層(2·k_i < P)では全被覆が原理的に不可能なので、どの方式でもランダム配線
(get_random_connections)に落ちる。

## 後段LUTの入力数N(--n-inputs)

N=2(既定)がdifflogic互換の16ゲートsoftmax。N=4/6はBinaryBrainのLUT4/LUT6と
同じ表現力で、ポート数は N·k_i、1LUTあたりのテーブルは 2^N エントリになる
(N=6は64エントリ。N=2のsoftmaxはdifflogic互換の16ゲートモード=16パラメータ
なので、**1LUTあたりのパラメータ数は4倍**であって16倍ではない)。ポート埋めの計算は
すべて N·k で一般化してあるので expand-fill はNに依らず使える。
N=4/6でsoftmaxの学習が不安定なときは --parameterization warp を使う。

## LUT6を活かす構造(HYBRID-4)

HYBRID-3 で「LUT6がLUT2に勝てない」ことが分かった(560ゲートで両者0.9703)。
一般的知見(LUT6 > LUT2)に反するため、配線とアーキテクチャ側を疑って
3つのスイッチを足した。DWN(arXiv:2410.11112)とBinaryBrainを参考にしている。

    --connections-mode balanced — 均等被覆配線。ゲート内は全相異、前層の各出力の
        使用回数は max-min <= 1。N·k_i が前層幅の倍数になる幅(6の倍数系)を選べば
        厳密均等になる。従来の unique/fanout2/bypass はいずれも取りこぼしがあった
    --interlayer-binarize — 各LUT層出力に BatchNorm→Binarize を挿入(BinaryBrain流)。
        層をまたぐ値を毎回0/1へ畳み直す
    --dlgn-layer0-wiring learnable — 第1層だけ配線を勾配で学習(DWN流)

## 評価

前段NNは常に実数(float)で動かし、hard/soft の切替は**ビット界面から後ろ**にのみ
適用する(model.train(mode) で LutLayer の soft混合 / hard argmax を切替える)。
train.evaluate は入力そのものを round() してしまい前段NNの入力を壊すため、
本スクリプトは round しない専用の evaluate_hybrid を使う。
Binarize は train/eval どちらでも前向きの値が閾値二値化で同一(違いは勾配だけ)
なので、界面の値は hard/soft で不変であり比較は公平。

使用例:
    # スモークテスト
    uv run python experiments/train_hybrid_front.py --dataset mnist \
        --front-widths "64" --dlgn-widths "80,160,320" --expand-fill bypass \
        --num-iterations 50 --eval-freq 25

    # 本走(MNIST)
    uv run python experiments/train_hybrid_front.py --dataset mnist \
        --front-widths "256,64" --dlgn-widths "80,160,320" --expand-fill bypass \
        --num-iterations 10000 --eval-freq 1000 --lr 0.03 --tau 10 \
        --wandb-group hybrid-front-dlgn --series HYBRID-1 \
        --run-name m-front2-expand-bypass

    # MLP対照(後段DLGNなし)
    uv run python experiments/train_hybrid_front.py --dataset mnist \
        --front-widths "256,64" --dlgn-widths "" --run-name m-mlp-control
"""

import argparse
import os
import random
import sys
import time
from datetime import datetime, timedelta, timezone

JST = timezone(timedelta(hours=9))

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from checkpoint_utils import save_checkpoint, setup_checkpoint_dir
from env_utils import load_dotenv

# train.py のデータローダ・ロガーを再利用する(evaluate は round() が
# 前段NNの入力を壊すため再利用せず、本ファイルの evaluate_hybrid を使う)
from train import DATASET_INFO, load_dataset, load_n, setup_logger
from wandb_utils import (
    add_experiment_args,
    init_wandb,
    log_checkpoint_artifact,
    log_images,
    log_metrics,
    normalize_tags,
)
from wandb_utils import (
    finish as wandb_finish,
)

from griffith_logic import Binarize, GroupSum, LutLayer
from griffith_logic.analysis.wiring_viz import (
    GROUP_SUM_NAME,
    INPUT_NAME,
    INTERFACE_NAME,
    LayerNode,
    WiringEdge,
    WiringSpec,
    count_wires,
)
from griffith_logic.core.connections import (
    get_random_connections,
    get_unique_connections,
)

load_dotenv()

EXPAND_FILLS = ('none', 'bypass', 'fanout2')

# 後段DLGNの配線生成方式。'legacy'=従来(expand_fillに従う) /
# 'balanced'=均等被覆(HYBRID-4。抜け漏れゼロ・fan-out均一・ゲート内全相異)
CONNECTIONS_MODES = ('legacy', 'balanced')

# 配線は初期化時に固定する(BYPASS系列と同じ方針。通常配線の学習はしない)
FIXED_WIRING = 'fixed'

# CIFAR-10のfloat入力(前段NNへ生の実数を渡す)。train.py の
# cifar-*-thresholds は ThresholdTransform で二値化済みなので前段NNには使えない
CIFAR10_RAW = 'cifar10-raw'
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)

HYBRID_DATASET_INFO = {
    **DATASET_INFO,
    CIFAR10_RAW: {'in_dim': 3 * 32 * 32, 'num_classes': 10},
}


# ============================================================
# データセット
# ============================================================

def _rebuild_loader(loader, args, *, shuffle: bool):
    """既存DataLoaderのdatasetから num_workers 指定のloaderを作り直す。

    train.load_dataset は num_workers=2 固定だが、**macOSのfork型workerとMPSの
    組合せで最初のバッチ取得がデッドロックする**(worker全滅・CPU0%のまま停止)
    実績があるため、--num-workers 0(既定)で作り直せるようにする。
    train.py 自体は他スクリプトと共有のため変更しない。
    """
    return torch.utils.data.DataLoader(
        loader.dataset, batch_size=args.batch_size, shuffle=shuffle,
        pin_memory=(args.device == 'cuda'), drop_last=True,
        num_workers=args.num_workers,
    )


def load_dataset_hybrid(args):
    """cifar10-raw を追加した上で、それ以外は train.load_dataset に委譲する。"""
    if args.dataset != CIFAR10_RAW:
        train_loader, test_loader = load_dataset(args)
        if args.num_workers != 2:
            train_loader = _rebuild_loader(train_loader, args, shuffle=True)
            test_loader = _rebuild_loader(test_loader, args, shuffle=False)
        return train_loader, test_loader

    transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    train_set = torchvision.datasets.CIFAR10(
        './data-cifar', train=True, download=True, transform=transforms,
    )
    test_set = torchvision.datasets.CIFAR10(
        './data-cifar', train=False, transform=transforms,
    )
    use_pin_memory = args.device == 'cuda'
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        pin_memory=use_pin_memory, drop_last=True,
        num_workers=args.num_workers,
    )
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=args.batch_size, shuffle=False,
        pin_memory=use_pin_memory, drop_last=True,
        num_workers=args.num_workers,
    )
    return train_loader, test_loader


def parse_widths(spec: str) -> list[int]:
    """カンマ区切りの幅指定をintリストにする(空文字は空リスト)。"""
    spec = spec.strip()
    if not spec:
        return []
    widths = [int(w) for w in spec.split(',') if w.strip()]
    assert all(w > 0 for w in widths), f'幅は正の整数: {spec}'
    return widths


# ============================================================
# 配線生成(純関数 + 自己検証assert)
# ============================================================

def _fix_self_pairs_n2(conn: torch.Tensor, rng: torch.Generator,
                       max_tries: int = 200) -> torch.Tensor:
    """N=2専用: 両slotが同じ入力を指すゲート(自己ペア)をslot1の交換で解消する。

    **HYBRID-1の基準値(m-front2-wide 0.9714 等)を再現するため、乱数消費まで
    含めて当時の実装のまま凍結している**。厳密には dup と partner の
    インデックスが重なると「入れ替え」が置換にならず入力の多重集合が
    わずかに崩れるが、N=2では arange 由来の初期配線に自己ペアがほとんど
    無く実害が観測されていないため、比較可能性を優先して変更しない。
    N>2 では自己ペアが桁違いに増えてこの崩れが致命的になるので、
    `_fix_duplicate_ports` の厳密な置換版を使う。
    """
    k = conn.shape[1]
    for _ in range(max_tries):
        dup = (conn[0] == conn[1]).nonzero().flatten()
        if dup.numel() == 0:
            return conn
        partner = torch.randint(0, k, (dup.numel(),), generator=rng,
                                dtype=torch.int64)
        tmp = conn[1, dup].clone()
        conn[1, dup] = conn[1, partner]
        conn[1, partner] = tmp
    n_dup = int((conn[0] == conn[1]).sum())
    if n_dup:
        print(f'警告: 自己ペアを{n_dup}ゲート解消できませんでした(無害だが'
              f'そのゲートは実質1入力になります)')
    return conn


def _fix_duplicate_ports(conn: torch.Tensor, rng: torch.Generator,
                         max_tries: int = 200) -> torch.Tensor:
    """同一ゲート内で同じ入力を指すポート(重複ポート)を解消する(N入力一般)。

    slot ごとに「重複しているゲートの位置」+「同数のランダムな他ゲート位置」を
    集めて**その位置集合の中で値を厳密に置換(randperm)する**。置換なので
    各入力のfan-out数(多重集合)は完全に不変であり、fanout2の均等性や
    bypassの全被覆を壊さない。
    """
    n_inputs, k = conn.shape
    for slot in range(1, n_inputs):
        for _ in range(max_tries):
            # slot が それより手前のいずれかのslotと重複しているゲート
            dup = (conn[:slot] == conn[slot]).any(dim=0).nonzero().flatten()
            if dup.numel() == 0:
                break
            others = torch.randperm(k, generator=rng)[:dup.numel()]
            # unique で位置集合を重複なしにしてから置換(置換性の担保に必須)
            sel = torch.unique(torch.cat([dup, others]))
            perm = torch.randperm(sel.numel(), generator=rng)
            conn[slot, sel] = conn[slot, sel[perm]].clone()
    n_dup = 0
    for slot in range(1, n_inputs):
        n_dup += int((conn[:slot] == conn[slot]).any(dim=0).sum())
    if n_dup:
        print(f'警告: 重複ポートを{n_dup}箇所解消できませんでした(無害だが'
              f'そのゲートは実質fan-inが減ります)')
    return conn


def _fix_self_pairs(conn: torch.Tensor, rng: torch.Generator,
                    max_tries: int = 200) -> torch.Tensor:
    """重複ポート解消のディスパッチ(N=2は凍結版、N>2は厳密置換版)。"""
    if conn.shape[0] == 2:
        return _fix_self_pairs_n2(conn, rng, max_tries)
    return _fix_duplicate_ports(conn, rng, max_tries)


def build_fanout2_connections(prev_width: int, k: int, rng: torch.Generator,
                              n_inputs: int = 2) -> torch.Tensor:
    """前層出力を均等に周回させてポートを埋める配線 (n_inputs, k) を返す。

    ポート数 2k >= 前層幅 P のとき、arange(P) を ceil(2k/P) 周だけ並べて
    先頭 2k 個を取る。これで各前層出力のfan-outは floor(2k/P) か ceil(2k/P) に
    収まる(最大均等)。**割当自体は決定的**で、乱数はポート↔ゲートの
    ペアリング(どのゲートに配るか)にのみ使う。
    """
    ports = n_inputs * k
    assert ports >= prev_width, (ports, prev_width)
    reps = -(-ports // prev_width)  # ceil
    src = torch.arange(prev_width, dtype=torch.int64).repeat(reps)[:ports]
    perm = torch.randperm(ports, generator=rng)
    conn = src[perm].reshape(n_inputs, k).contiguous()
    conn = _fix_self_pairs(conn, rng)

    cnt = torch.bincount(conn.flatten(), minlength=prev_width)
    assert tuple(conn.shape) == (n_inputs, k), conn.shape
    assert int(cnt.min()) >= ports // prev_width, 'fan-outの下限割れ'
    assert int(cnt.max()) <= reps, 'fan-outの上限超過'
    assert int(cnt.min()) >= 1, '前層出力に未使用のものがある'
    return conn


def build_bypass_connections(prev_width: int, aux_width: int, k: int,
                             rng: torch.Generator,
                             n_inputs: int = 2) -> torch.Tensor:
    """前層を全被覆し、余りを補助プール(界面+既出中間層)から引く配線を返す。

    入力プールは concat([prev(P), aux(A)]) で prev が 0..P-1、aux が P..P+A-1。
    aux_width=0(=DLGN第0層。前層が界面そのもの)のときは補助プールが無いので
    余りを prev からランダムに引く。
    """
    ports = n_inputs * k
    assert ports >= prev_width, (ports, prev_width)
    pool_width = prev_width + aux_width
    rest = ports - prev_width
    base = torch.arange(prev_width, dtype=torch.int64)
    if rest == 0:
        src = base
    elif aux_width > 0:
        src = torch.cat([
            base,
            prev_width + torch.randint(0, aux_width, (rest,), generator=rng,
                                       dtype=torch.int64),
        ])
    else:
        src = torch.cat([
            base,
            torch.randint(0, prev_width, (rest,), generator=rng,
                          dtype=torch.int64),
        ])
    perm = torch.randperm(ports, generator=rng)
    conn = src[perm].reshape(n_inputs, k).contiguous()
    conn = _fix_self_pairs(conn, rng)

    cnt = torch.bincount(conn.flatten(), minlength=pool_width)
    assert tuple(conn.shape) == (n_inputs, k), conn.shape
    assert int(conn.min()) >= 0 and int(conn.max()) < pool_width, '配線が範囲外'
    assert int(cnt[:prev_width].min()) >= 1, '前層出力に未使用のものがある'
    if aux_width > 0:
        assert int(cnt[prev_width:].sum()) == rest, '補助プールへの本数が不一致'
    return conn


def build_balanced_connections(prev_width: int, k: int, rng: torch.Generator,
                               n_inputs: int = 2) -> torch.Tensor:
    """均等被覆配線(HYBRID-4)。抜け漏れゼロ・fan-out均一・ゲート内全相異。

    HYBRID-3 で「LUT6がLUT2に勝てない」原因として配線が疑われた。既存方式には
    いずれも取りこぼしがある:

        unique   — 全入力を1回は使うが、以降のfan-outは偏る
        fanout2  — fan-outは均等だが、**ゲート内の重複を許す**(N=6では
                   `get_unique_connections` 由来の重複が実測4.75%あった)
        bypass   — 前層は全被覆だが余りは補助プールからの乱択で偏る

    本方式は2つの制約を同時に満たす:

        制約1: 各ゲートの N 本の入力がすべて相異なるソース
               (同じ線を2本引くのは 2^N のLUT容量を捨てるのと同じ)
        制約2: 前層の各出力の使用回数が floor/ceil(N·k / prev_width) に均等
               (max-min <= 1。使われない出力もfan-outの偏りも無い)

    N·k が prev_width の倍数になる幅(6の倍数系)を選べば全出力が**厳密に
    同じ回数**使われる。余りが出る場合は余剰分を乱択した出力に1回ずつ配る。
    """
    ports = n_inputs * k
    assert prev_width >= n_inputs, (
        f'均等被覆にはゲート内を相異なる入力で埋める必要があるため '
        f'prev_width({prev_width}) >= n_inputs({n_inputs}) が必須'
    )
    base_reps = ports // prev_width
    remainder = ports % prev_width

    src = torch.arange(prev_width, dtype=torch.int64).repeat(base_reps)
    if remainder:
        # 余りは重複なしで選んだ出力へ1回ずつ(これで max-min <= 1 が保たれる)
        extra = torch.randperm(prev_width, generator=rng)[:remainder]
        src = torch.cat([src, extra])
    assert src.numel() == ports, (src.numel(), ports)

    perm = torch.randperm(ports, generator=rng)
    conn = src[perm].reshape(n_inputs, k).contiguous()
    # ゲート内重複を厳密置換で解消(多重集合=使用回数は不変なので制約2を壊さない)
    conn = _fix_duplicate_ports(conn, rng)

    # --- 自己検証 ---
    assert tuple(conn.shape) == (n_inputs, k), conn.shape
    n_unique = torch.tensor([len(set(conn[:, g].tolist())) for g in range(k)])
    assert int(n_unique.min()) == n_inputs, (
        f'ゲート内に重複入力が残っている(最小 {int(n_unique.min())} < {n_inputs})'
    )
    cnt = torch.bincount(conn.flatten(), minlength=prev_width)
    assert int(cnt.min()) >= 1, '使われない前層出力がある'
    assert int(cnt.max()) - int(cnt.min()) <= 1, (
        f'fan-outが均等でない(max={int(cnt.max())}, min={int(cnt.min())})'
    )
    return conn


def build_layer_connections(expand_fill: str, prev_width: int, aux_width: int,
                            k: int, rng: torch.Generator, n_inputs: int = 2,
                            connections_mode: str = 'legacy'):
    """DLGN1層分の配線を返す。(connections, 使う入力プール幅) のタプル。

    connections が文字列のときは LutLayer 側の既定生成に委ねる。

    connections_mode='balanced' の挙動(HYBRID-4b で bypass 併用に対応):

        expand_fill='bypass' かつ補助プールがあり、ポート数がプール全体を
        覆えるなら **[前層 + 界面 + 既出中間層] のプール全体を均等被覆**する。
        HYBRID-4 の要因分解で効果量が最大だった「層をまたぐショートカット」
        (+0.93pt)と、次点の「均等被覆」(+0.49pt)を合成した配線。
        従来のbypassは前層を1回ずつ covering した後の**余りを乱択**で補助
        プールへ配るため偏りが残っていたが、こちらは前層も補助プールも
        まとめて floor/ceil 均等になる。

        それ以外(expand_fill='none' 等 / 補助プール無し / ポート不足)は
        従来どおり前層のみを均等被覆する。
    """
    ports = n_inputs * k
    if connections_mode == 'balanced':
        pool_width = prev_width + aux_width
        if expand_fill == 'bypass' and aux_width > 0 and ports >= pool_width:
            return (build_balanced_connections(pool_width, k, rng, n_inputs),
                    pool_width)
        return build_balanced_connections(prev_width, k, rng, n_inputs), prev_width
    if ports < prev_width:
        # 圧縮層: 全被覆は原理的に不可能なので、方式に依らずランダム配線
        return get_random_connections(prev_width, k, n_inputs), prev_width
    if expand_fill == 'bypass':
        return (build_bypass_connections(prev_width, aux_width, k, rng,
                                         n_inputs),
                prev_width + aux_width)
    if expand_fill == 'fanout2':
        return build_fanout2_connections(prev_width, k, rng, n_inputs), prev_width
    # none: 前層のみを見る対照。difflogic流のオフセットペアリングで全入力被覆
    return get_unique_connections(prev_width, k, n_inputs), prev_width


# ============================================================
# モデル
# ============================================================

class HybridFrontDLGN(torch.nn.Module):
    """前段NN(密結合・圧縮担当) + 二値化界面 + 後段DLGN(疎結合・拡散担当)。

    front_widths が空なら前段NLなし(入力をそのまま二値化 = ほぼ素のDLGN)。
    dlgn_widths が空なら後段DLGNなし(界面から Linear で直接クラス数へ = MLP対照)。
    """

    def __init__(
        self,
        in_dim: int,
        num_classes: int,
        front_widths: list[int],
        dlgn_widths: list[int],
        n_inputs: int = 2,
        tau: float = 10.0,
        expand_fill: str = 'none',
        parameterization: str = 'softmax',
        connections_mode: str = 'legacy',
        interlayer_binarize: bool = False,
        interlayer_binarize_last: bool = True,
        dlgn_layer0_wiring: str = 'fixed',
        device: str = 'cpu',
        grad_factor: float = 1.0,
        seed: int = 0,
        binarize_threshold: float = 0.5,
    ):
        super().__init__()
        assert expand_fill in EXPAND_FILLS, expand_fill
        assert connections_mode in CONNECTIONS_MODES, connections_mode
        assert dlgn_layer0_wiring in LutLayer.WIRINGS, dlgn_layer0_wiring
        assert front_widths or dlgn_widths, (
            '--front-widths と --dlgn-widths の両方が空です'
        )
        if dlgn_widths:
            assert dlgn_widths[-1] % num_classes == 0, (
                f'後段DLGNの最終幅({dlgn_widths[-1]})はクラス数({num_classes})で'
                f'割り切れる必要があります'
            )

        self.flatten = torch.nn.Flatten()
        self.in_dim = in_dim
        self.front_widths = front_widths
        self.dlgn_widths = dlgn_widths
        self.expand_fill = expand_fill
        self.connections_mode = connections_mode
        self.interlayer_binarize = interlayer_binarize
        self.interlayer_binarize_last = interlayer_binarize_last
        self.dlgn_layer0_wiring = dlgn_layer0_wiring

        # --- 前段NN: Linear+ReLU の繰り返し、最終層のみ Linear+Sigmoid ---
        if front_widths:
            layers: list[torch.nn.Module] = []
            prev = in_dim
            for j, w in enumerate(front_widths):
                layers.append(torch.nn.Linear(prev, w))
                is_last = j == len(front_widths) - 1
                layers.append(torch.nn.Sigmoid() if is_last
                              else torch.nn.ReLU())
                prev = w
            self.front = torch.nn.Sequential(*layers)
            self.interface_dim = front_widths[-1]
        else:
            self.front = None
            self.interface_dim = in_dim

        # --- 二値化界面(STE)。前向きの値は train/eval どちらでも0/1 ---
        self.binarize = Binarize(threshold=binarize_threshold)

        # --- 後段DLGN ---
        llkw = dict(
            device=device,
            grad_factor=grad_factor,
            parameterization=parameterization,
            selection='soft',
        )
        rng = torch.Generator().manual_seed(seed + 12345)
        self.dlgn = torch.nn.ModuleList()
        # 各層が「前層に加えて補助プールを連結するか」= aux_width>0 か
        self.aux_widths: list[int] = []
        prev_width = self.interface_dim
        for i, k in enumerate(dlgn_widths):
            # 補助プール = [界面] + [o_0 .. o_{i-2}](o_{i-1}=前層は prev 側にある)
            aux_w = (0 if i == 0
                     else self.interface_dim + sum(dlgn_widths[:max(0, i - 1)]))
            conn, lut_in = build_layer_connections(
                expand_fill, prev_width, aux_w, k, rng, n_inputs,
                connections_mode=connections_mode,
            )
            self.aux_widths.append(lut_in - prev_width)
            # DWN流(arXiv:2410.11112 §3.2): 第1層だけ配線を勾配で学習する。
            # 界面ビット→最初のLUT層が最も情報を取りこぼす箇所なので、
            # そこだけ配線自体を学習対象にする(隠れ層はfixedのまま)
            wiring = (dlgn_layer0_wiring if i == 0 else FIXED_WIRING)
            self.dlgn.append(LutLayer(in_dim=lut_in, out_dim=k,
                                      n_inputs=n_inputs, connections=conn,
                                      wiring=wiring, **llkw))
            prev_width = k

        # --- 層間再二値化(BinaryBrain流) ---
        # BinaryBrainのDifferentiableLutは LUT→BatchNorm→Binarize を1ブロックと
        # する(`griffith_logic/differentiable_lut_block.py`)。DLGN側は素の
        # LUT出力(soft混合の実数)をそのまま次層へ渡すので、層をまたぐたびに
        # 「0/1でない中間値」が積み重なる。ここを毎層0/1へ畳み直す。
        # BN初期値 gamma=0.3 / beta=0.5 はBinaryBrainの既定に合わせる。
        # **momentumだけは0.1(PyTorch既定のEMA)に変える**: BinaryBrainの
        # momentum=1.0 は running統計が「最後の1バッチ」だけで決まり、
        # bs=128ではeval時の閾値が偶然のバッチに振り回されるため。
        if interlayer_binarize:
            assert dlgn_widths, '--interlayer-binarize は後段DLGNが必要です'
            self.inter_bn = torch.nn.ModuleList([
                torch.nn.BatchNorm1d(k, momentum=0.1, device=device)
                for k in dlgn_widths
            ])
            for bn in self.inter_bn:
                torch.nn.init.constant_(bn.weight, 0.3)
                torch.nn.init.constant_(bn.bias, 0.5)
            self.inter_binarize = Binarize(
                threshold=0.5, hardtanh_min=0.0, hardtanh_max=1.0,
            )
        else:
            self.inter_bn = None
            self.inter_binarize = None
        self.group_sum = GroupSum(num_classes, tau=tau) if dlgn_widths else None
        # MLP対照(後段なし): 界面ビットから直接クラスへ線形読み出し
        self.head = (None if dlgn_widths
                     else torch.nn.Linear(self.interface_dim, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.flatten(x)
        z = self.front(x) if self.front is not None else x
        bits = self.binarize(z)
        if self.head is not None:
            return self.head(bits)

        h = bits
        acts: list[torch.Tensor] = []  # o_0, o_1, ...
        for i, layer in enumerate(self.dlgn):
            if self.aux_widths[i] > 0:
                # プール = [h(=o_{i-1}), 界面, o_0, ..., o_{i-2}]
                inp = torch.cat([h, bits, *acts[:max(0, i - 1)]], dim=-1)
            else:
                inp = h
            h = layer(inp)
            if self.inter_bn is not None and (
                self.interlayer_binarize_last or i < len(self.dlgn) - 1
            ):
                # BinaryBrain流: LUT出力を毎層 BatchNorm→Binarize で0/1へ畳み直す
                h = self.inter_binarize(self.inter_bn[i](h))
            acts.append(h)
        return self.group_sum(h)

    def wiring_spec(self) -> WiringSpec:
        """配線可視化(VIZ-2)用の構造記述を返す。

        `griffith_logic.analysis.wiring_viz.extract_wiring_spec` は
        このメソッドがあれば自動推論せずこちらへ委譲する。ハイブリッドは
        前段が密結合NN・後段が疎結合DLGNで配線の意味が違うため、自動推論では
        表現できない(前段には indices が無い)。

        ノードは描画順(左→右)に並べる:
            input → front0(NN) → ... → interface → dlgn0 → ... → group_sum
        前段の Linear は個別配線を持たない全結合なので `detail=None` とし
        (全配線図では帯 + 本数ラベルで表示される)、後段DLGNのエッジだけ
        `LutLayer.indices` から実配線を集計する。
        """
        layers: list[LayerNode] = [LayerNode(INPUT_NAME, int(self.in_dim))]
        edges: list[WiringEdge] = []

        # --- 前段NN(密結合) ---
        prev_name, prev_width = INPUT_NAME, int(self.in_dim)
        for j, w in enumerate(self.front_widths):
            name = f'front{j}(NN)'
            layers.append(LayerNode(name, int(w)))
            edges.append(WiringEdge(prev_name, name, prev_width * int(w)))
            prev_name, prev_width = name, int(w)

        # --- 二値化界面(要素ごとの Binarize なので1対1) ---
        layers.append(LayerNode(INTERFACE_NAME, int(self.interface_dim)))
        edges.append(WiringEdge(prev_name, INTERFACE_NAME,
                                int(self.interface_dim)))

        if self.head is not None:
            # MLP対照(後段DLGNなし): 界面から全結合で直接クラスへ
            out_dim = int(self.head.out_features)
            layers.append(LayerNode('head(NN)', out_dim))
            edges.append(WiringEdge(INTERFACE_NAME, 'head(NN)',
                                    int(self.interface_dim) * out_dim))
            return WiringSpec(layers=layers, edges=edges)

        # --- 後段DLGN ---
        prev_name, prev_width = INTERFACE_NAME, int(self.interface_dim)
        for i, layer in enumerate(self.dlgn):
            dst = f'dlgn{i}'
            layers.append(LayerNode(dst, int(layer.out_dim)))
            # プール = [前層(prev_width), 界面, o_0, ..., o_{i-2}]
            segments = [(0, prev_width, prev_name)]
            if self.aux_widths[i] > 0:
                cursor = prev_width
                segments.append((cursor, int(self.interface_dim),
                                 INTERFACE_NAME))
                cursor += int(self.interface_dim)
                for j in range(max(0, i - 1)):
                    w = int(self.dlgn_widths[j])
                    segments.append((cursor, w, f'dlgn{j}'))
                    cursor += w
                assert cursor == prev_width + self.aux_widths[i], (
                    f'{dst}: プール分解の合計({cursor})が'
                    f'実際の幅({prev_width + self.aux_widths[i]})と不一致'
                )
            for offset, width, src in segments:
                n_wires, detail = count_wires(layer.indices, offset, width)
                if n_wires:
                    edges.append(WiringEdge(src, dst, n_wires, detail))
            prev_name, prev_width = dst, int(layer.out_dim)

        # --- GroupSum(クラスごとに連続ゲートを合計) ---
        k = int(self.group_sum.k)
        per_class = prev_width // k
        detail = torch.arange(prev_width, dtype=torch.int64).reshape(
            k, per_class).T.contiguous()
        layers.append(LayerNode(GROUP_SUM_NAME, k))
        edges.append(WiringEdge(prev_name, GROUP_SUM_NAME, prev_width, detail))
        return WiringSpec(layers=layers, edges=edges)


def build_model(args):
    info = HYBRID_DATASET_INFO[args.dataset]
    front_widths = parse_widths(args.front_widths)
    dlgn_widths = parse_widths(args.dlgn_widths)
    model = HybridFrontDLGN(
        in_dim=info['in_dim'],
        num_classes=info['num_classes'],
        front_widths=front_widths,
        dlgn_widths=dlgn_widths,
        n_inputs=args.n_inputs,
        tau=args.tau,
        expand_fill=args.expand_fill,
        parameterization=args.parameterization,
        connections_mode=args.connections_mode,
        interlayer_binarize=args.interlayer_binarize,
        interlayer_binarize_last=args.interlayer_binarize_last,
        dlgn_layer0_wiring=args.dlgn_layer0_wiring,
        device=args.device,
        grad_factor=args.grad_factor,
        seed=args.seed,
    ).to(args.device)

    total_gates = sum(
        m.num_neurons for m in model.modules() if isinstance(m, LutLayer)
    )
    front_params = (sum(p.numel() for p in model.front.parameters())
                    if model.front is not None else 0)
    head_params = (sum(p.numel() for p in model.head.parameters())
                   if model.head is not None else 0)
    total_params = sum(p.numel() for p in model.parameters())

    print(f'前段NN: {front_widths or "なし"} '
          f'(パラメータ {front_params:,})')
    print(f'二値化界面: {model.interface_dim} bit (threshold=0.5, STE)')
    lut_params = sum(
        m.weights.numel() for m in model.modules() if isinstance(m, LutLayer)
    )
    # LUT1個あたりのテーブルサイズ(softmax N=2は16ゲートモードなので16)
    entries_per_lut = (lut_params // total_gates) if total_gates else 0

    if args.connections_mode == 'balanced':
        fill_desc = ('balanced+bypass(プール全体を均等被覆)'
                     if args.expand_fill == 'bypass'
                     else 'balanced(前層を均等被覆)')
    else:
        fill_desc = f'expand-fill={args.expand_fill}'
    print(f'後段DLGN: {dlgn_widths or "なし(Linear直結のMLP対照)"} '
          f'(N={args.n_inputs}入力LUT, {args.parameterization}, {fill_desc}'
          f'{", 層間再二値化" if args.interlayer_binarize else ""}'
          f'{", 第1層配線学習" if args.dlgn_layer0_wiring == "learnable" else ""})')
    for i in range(len(dlgn_widths)):
        prev = model.interface_dim if i == 0 else dlgn_widths[i - 1]
        ports = args.n_inputs * dlgn_widths[i]
        kind = ('圧縮' if ports < prev else '拡張' if ports > prev else '等幅')
        aux = model.aux_widths[i]
        extra = ''
        if args.connections_mode == 'balanced':
            # 均等被覆の分母は「実際に使ったプール幅」= prev + aux
            pool = prev + aux
            reps = ports / pool
            extra = f', プール{pool}の各要素を{reps:.3g}回使用'
            if ports % pool == 0:
                extra += '(厳密均等)'
        print(f'  層{i}: 前層{prev} -> {dlgn_widths[i]}ゲート '
              f'(ポート{ports}, {kind}, プール幅{prev + aux}'
              f'{f", 補助プール+{aux}" if aux else ""}{extra})')
    print(f'総ロジックゲート(LUT)数: {total_gates:,}')
    print(f'後段LUTパラメータ数: {lut_params:,} '
          f'(1LUTあたり{entries_per_lut}エントリ)')
    print(f'NNパラメータ数(前段+読み出し): {front_params + head_params:,}')
    print(f'総パラメータ数: {total_params:,}')
    extra = {'total_luts': total_gates, 'lut_params': lut_params,
             'entries_per_lut': entries_per_lut}
    return model, total_gates, total_params, front_params + head_params, extra


# ============================================================
# 評価(入力をround()しない = 前段NNには実数を渡す)
# ============================================================

def evaluate_hybrid(model, loader, device, soft: bool, desc=None) -> float:
    """hard/soft を LutLayer の train フラグだけで切替えて精度を測る。

    train.evaluate との違いは **x.round() をしない**こと。前段NNの入力は
    実数のまま渡し、二値化は Binarize 層が界面で行う(前向きの値は
    train/eval どちらでも閾値二値化なので hard/soft で界面は不変)。
    """
    orig_mode = model.training
    model.train(mode=soft)
    correct = 0
    total = 0
    it = tqdm(loader, desc=desc, leave=False) if desc else loader
    with torch.no_grad():
        for x, y in it:
            x = x.to(torch.float32).to(device)
            y = y.to(device)
            output = model(x)
            pred = output.argmax(dim=-1)
            correct += (pred == y).sum().item()
            total += y.shape[0]
            if desc and isinstance(it, tqdm):
                it.set_postfix(acc=f'{correct / total:.4f}')
    model.train(mode=orig_mode)
    return correct / total if total > 0 else 0.0


def verify_interface_is_binary(model, loader, device) -> None:
    """界面が厳密に0/1であることを1バッチで検証する(hard/soft両モード)。"""
    x, _ = next(iter(loader))
    x = x.to(torch.float32).to(device)
    orig_mode = model.training
    for mode in (True, False):
        model.train(mode=mode)
        with torch.no_grad():
            z = model.front(model.flatten(x)) if model.front is not None \
                else model.flatten(x)
            bits = model.binarize(z)
        uniq = torch.unique(bits)
        assert bool(((uniq == 0) | (uniq == 1)).all()), (
            f'界面が0/1でない(train={mode}): {uniq[:8]}'
        )
    model.train(mode=orig_mode)
    print('界面の0/1性: OK (train/eval 両モードで検証)')


def verify_interlayer_bits(model, loader, device) -> None:
    """層間再二値化が有効なとき、各LutLayer出力が0/1であることを検証する。

    eval時はBatchNormがrunning統計で固定されるため、Binarizeの前向きが
    純粋な閾値二値化になり0/1が保たれる — これを実測で確かめる。
    """
    if model.inter_bn is None:
        return
    x, _ = next(iter(loader))
    x = x.to(torch.float32).to(device)
    orig_mode = model.training
    for mode in (True, False):
        model.train(mode=mode)
        with torch.no_grad():
            z = model.front(model.flatten(x)) if model.front is not None \
                else model.flatten(x)
            bits = model.binarize(z)
            h = bits
            acts: list[torch.Tensor] = []
            for i, layer in enumerate(model.dlgn):
                if model.aux_widths[i] > 0:
                    inp = torch.cat([h, bits, *acts[:max(0, i - 1)]], dim=-1)
                else:
                    inp = h
                h = layer(inp)
                is_last = i == len(model.dlgn) - 1
                if model.interlayer_binarize_last or not is_last:
                    h = model.inter_binarize(model.inter_bn[i](h))
                    uniq = torch.unique(h)
                    assert bool(((uniq == 0) | (uniq == 1)).all()), (
                        f'層{i}の出力が0/1でない(train={mode}): {uniq[:8]}'
                    )
                acts.append(h)
    model.train(mode=orig_mode)
    n_bin = len(model.dlgn) - (0 if model.interlayer_binarize_last else 1)
    print(f'層間再二値化の0/1性: OK ({n_bin}層 × train/eval 両モード'
          f'{"" if model.interlayer_binarize_last else "、最終層は非二値化"})')


# ============================================================
# 学習曲線の図
# ============================================================

def plot_curves(history: dict, title: str, out_path: str) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].plot(history['loss_iters'], history['loss'], lw=0.8, color='tab:blue')
    axes[0].set_xlabel('iteration')
    axes[0].set_ylabel('train loss')
    axes[0].set_title('train loss')
    axes[0].grid(alpha=0.3)

    ev = history['eval_iters']
    axes[1].plot(ev, history['train_acc'], 'o-', label='train acc (hard)')
    axes[1].plot(ev, history['test_acc'], 's-', label='test acc (hard)')
    axes[1].plot(ev, history['test_acc_soft'], '^--', label='test acc (soft)')
    axes[1].set_xlabel('iteration')
    axes[1].set_ylabel('accuracy')
    axes[1].set_ylim(0, 1)
    axes[1].set_title('accuracy')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.suptitle(title)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


# ============================================================
# メイン
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='前段NN+後段DLGNハイブリッドの訓練スクリプト(HYBRID-1)'
    )
    parser.add_argument('--dataset', type=str, default='mnist',
                        choices=list(HYBRID_DATASET_INFO.keys()),
                        help='データセット (default: mnist)')
    parser.add_argument('--front-widths', type=str, default='256,64',
                        help='前段NNの各層幅(カンマ区切り、空なら前段なし)。'
                             '最終層のみSigmoid、それ以外はReLU')
    parser.add_argument('--dlgn-widths', type=str, default='80,160,320',
                        help='後段DLGNの各層幅(カンマ区切り、空なら後段なし='
                             'Linear直結のMLP対照)。最終幅はクラス数の倍数')
    parser.add_argument('--expand-fill', type=str, default='none',
                        choices=list(EXPAND_FILLS),
                        help='拡張層(2k>前層幅)の余ポートの埋め方 '
                             '(default: none)')
    parser.add_argument('--n-inputs', type=int, default=2, choices=[2, 4, 6],
                        help='後段LUTの入力数N (default: 2)。1個あたりのLUT'
                             'エントリ数は2^N。ただしN=2のsoftmaxはdifflogic'
                             '互換の16ゲートモード(16パラメータ)なので、'
                             'N=6(64エントリ)は1LUTあたり**4倍**のパラメータ')
    parser.add_argument('--connections-mode', type=str, default='legacy',
                        choices=list(CONNECTIONS_MODES),
                        help='後段DLGNの配線生成方式 (default: legacy)。'
                             'balanced=均等被覆(抜け漏れゼロ・fan-out均一・'
                             'ゲート内全相異)。**--expand-fill は無視される**')
    parser.add_argument('--interlayer-binarize', action='store_true',
                        help='BinaryBrain流に各LutLayer出力へ '
                             'BatchNorm1d→Binarize(STE) を挿入し、層をまたぐ'
                             '値を毎回0/1へ畳み直す(既定Off)')
    parser.add_argument('--no-binarize-last', dest='interlayer_binarize_last',
                        action='store_false',
                        help='--interlayer-binarize 時に**最終LUT層だけ**'
                             '再二値化しない。BinaryBrainの最終段は'
                             'BinaryToReal(フレーム平均)でありGroupSumの直前を'
                             '0/1に潰しはしないので、こちらが本家に近い')
    parser.add_argument('--dlgn-layer0-wiring', type=str, default='fixed',
                        choices=list(LutLayer.WIRINGS),
                        help='後段DLGN第1層の配線方式 (default: fixed)。'
                             'learnable=DWN流に配線を勾配で学習(隠れ層は'
                             '常にfixed)')
    parser.add_argument('--parameterization', type=str, default='softmax',
                        choices=['softmax', 'independent', 'warp'],
                        help='後段LUTのパラメータ化 (default: softmax)。'
                             '**N=6ではsoftmaxは出力が1/64に張り付き勾配が'
                             'N=2の1/446になる**(ΣW=1とΣbasis=1の二重simplex。'
                             'experiments/diagnose_lut_output_scale.py で実測)。'
                             'independent(sigmoid)なら出力≈0.5に戻るので'
                             'N=4/6ではこちらを使う。warpも選べる')
    parser.add_argument('--tau', type=float, default=10.0,
                        help='GroupSum温度 (default: 10)')
    parser.add_argument('--lr', type=float, default=0.03,
                        help='学習率 (default: 0.03)')
    parser.add_argument('--dlgn-lr', type=float, default=None,
                        help='後段DLGNのみに使う学習率 (default: None=--lrと同じ)。'
                             'LutLayerの基底は入力の積なので連続値入力では'
                             '基底値が0.5^Nに減衰し、LUT6はLUT2の1/16の実効lrで'
                             '訓練されてしまう。N=6なら--lrの16倍を目安に指定する')
    parser.add_argument('--front-lr', type=float, default=None,
                        help='前段NNのみに使う学習率 (default: None=--lrと同じ)。'
                             '前段の学習が不安定なとき1e-3等に分離する')
    parser.add_argument('-bs', '--batch-size', type=int, default=128,
                        help='バッチサイズ (default: 128)')
    parser.add_argument('-ni', '--num-iterations', type=int, default=10000,
                        help='反復回数 (default: 10000)')
    parser.add_argument('-ef', '--eval-freq', type=int, default=1000,
                        help='評価頻度[イテレーション] (default: 1000)')
    parser.add_argument('--grad-factor', type=float, default=1.0,
                        help='勾配スケーリング係数 (default: 1.0)')
    parser.add_argument('--num-workers', type=int, default=0,
                        help='DataLoaderのworker数 (default: 0)。'
                             'macOSのfork型worker×MPSで初回バッチ取得が'
                             'デッドロックする実績があるため既定0')
    parser.add_argument('--seed', type=int, default=0,
                        help='ランダムシード (default: 0)')
    parser.add_argument('--device', type=str, default=None,
                        help='デバイス (default: None=cpu)')
    parser.add_argument('--wandb-group', type=str, default='hybrid-front-dlgn',
                        help='wandb group (default: hybrid-front-dlgn)')
    parser.add_argument('--wandb-tags', type=str, default='hybrid-1',
                        help='wandbタグ(カンマ区切り。default: hybrid-1)')
    parser.add_argument('--series', type=str, default='HYBRID-1',
                        help='taxonomyのseriesタグ (default: HYBRID-1)')
    parser.add_argument('--run-name', type=str, default=None,
                        help='run名(未指定なら構成から自動生成)')
    parser.add_argument('--log-dir', type=str, default='logs',
                        help='ログ出力先 (default: logs)')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints',
                        help='チェックポイント保存先 (default: checkpoints)')
    parser.add_argument('--results-dir', type=str,
                        default='results/hybrid_front',
                        help='図の保存先 (default: results/hybrid_front)')
    add_experiment_args(parser)

    args = parser.parse_args()

    if args.device is None:
        args.device = 'cpu'
    print(f'デバイス: {args.device}')

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    train_loader, test_loader = load_dataset_hybrid(args)
    print(f'データセット: {args.dataset}')
    print(f'訓練: {len(train_loader.dataset):,}サンプル, '
          f'テスト: {len(test_loader.dataset):,}サンプル')

    model, total_gates, total_params, nn_params, extra = build_model(args)
    verify_interface_is_binary(model, test_loader, args.device)
    verify_interlayer_bits(model, test_loader, args.device)

    loss_fn = torch.nn.CrossEntropyLoss()
    # 学習率のパラメータ群分離。--dlgn-lr は LUT6-1 で追加(下記の勾配減衰補正用)。
    #
    # LutLayerの基底は入力の積なので、**連続値入力(≈0.5)ではN入力LUTの基底値が
    # 0.5^N に減衰する**(AP 法の解析より)。更新量は
    # `error × basis × lr` なので、LUT6はLUT2に対して基底値が 0.5^6/0.5^2 = 1/16
    # になり、**同じlrでは実効的に1/16の学習率で訓練していた**ことになる。
    # PJ内の既知の対処は学習率のN依存スケーリング(experiments-catalog.md の
    # AP法レシピ: LUT2 lr0.1 / LUT4 0.4 / LUT6 1.6 = 4倍ずつ)。
    # ハイブリッドの後段DLGNは**前層のsoft出力(連続値)**を受けるため隠れ層で
    # この減衰をまともに受ける。--dlgn-lr で後段だけlrを上げて公平にする。
    param_groups = []
    used_ids: set[int] = set()
    if args.front_lr is not None and model.front is not None:
        front_params = list(model.front.parameters())
        param_groups.append({'params': front_params, 'lr': args.front_lr})
        used_ids |= {id(p) for p in front_params}
    if args.dlgn_lr is not None and len(model.dlgn) > 0:
        dlgn_params = [p for p in model.dlgn.parameters() if id(p) not in used_ids]
        param_groups.append({'params': dlgn_params, 'lr': args.dlgn_lr})
        used_ids |= {id(p) for p in dlgn_params}
    if param_groups:
        rest = [p for p in model.parameters() if id(p) not in used_ids]
        if rest:
            param_groups.append({'params': rest, 'lr': args.lr})
        optimizer = torch.optim.Adam(param_groups)
        desc = []
        if args.front_lr is not None and model.front is not None:
            desc.append(f'前段NN={args.front_lr}')
        if args.dlgn_lr is not None and len(model.dlgn) > 0:
            desc.append(f'後段DLGN={args.dlgn_lr}')
        desc.append(f'その他={args.lr}')
        print(f'学習率を分離: {" / ".join(desc)}')
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    num_iterations = args.num_iterations
    eval_freq = args.eval_freq
    batches_per_epoch = len(train_loader)

    front_tag = args.front_widths.replace(',', '-') or 'nofront'
    dlgn_tag = args.dlgn_widths.replace(',', '-') or 'nodlgn'
    mode_tag = f'f{front_tag}_d{dlgn_tag}_{args.expand_fill}'
    run_name = args.run_name or (
        f'{args.dataset}_{mode_tag}_i{num_iterations}'
    )

    log, log_file = setup_logger('train_hybrid_front', args.dataset, args.log_dir)
    log(f'START run={run_name} dataset={args.dataset} '
        f'front_widths={args.front_widths} dlgn_widths={args.dlgn_widths} '
        f'expand_fill={args.expand_fill} interface_dim={model.interface_dim} '
        f'n_inputs={args.n_inputs} param={args.parameterization} '
        f'conn_mode={args.connections_mode} '
        f'interlayer_bin={args.interlayer_binarize} '
        f'layer0_wiring={args.dlgn_layer0_wiring} '
        f'tau={args.tau} lr={args.lr} '
        f'front_lr={args.front_lr} dlgn_lr={args.dlgn_lr} '
        f'bs={args.batch_size} '
        f'iterations={num_iterations} grad_factor={args.grad_factor} '
        f'seed={args.seed} device={args.device} '
        f'gates={total_gates} lut_params={extra["lut_params"]} '
        f'entries_per_lut={extra["entries_per_lut"]} '
        f'nn_params={nn_params} params={total_params}')

    run_dir, latest_path, best_path = setup_checkpoint_dir(
        f'hybrid_front_{args.dataset}', base_dir=args.checkpoint_dir,
    )
    ckpt_metadata = {'args': vars(args)}
    print(f'チェックポイント保存先: {run_dir}/')

    use_wandb = init_wandb(
        run_name=run_name,
        config_dict={**vars(args), 'total_gates': total_gates,
                     'total_params': total_params,
                     'nn_params': nn_params,
                     'interface_dim': model.interface_dim,
                     **extra},
        model=model,
        device=args.device,
        tags=args.wandb_tags.split(',') + [args.dataset, args.expand_fill]
        + normalize_tags(model='hybrid-front-dlgn', dataset=args.dataset,
                         seed=args.seed, series=args.series),
        group=args.wandb_group,
        job_type='train',
        notes=args.notes,
        hypothesis=args.hypothesis,
    )

    history = {
        'loss_iters': [], 'loss': [],
        'eval_iters': [], 'train_acc': [], 'test_acc': [], 'test_acc_soft': [],
    }
    best_test_acc = 0.0

    print(f'\n訓練開始: {num_iterations:,}イテレーション '
          f'({num_iterations / batches_per_epoch:.1f}エポック相当)')
    print('=' * 60)
    train_start = time.time()

    pbar = tqdm(
        enumerate(load_n(train_loader, num_iterations)),
        desc='train', total=num_iterations,
    )
    for i, (x, y) in pbar:
        x = x.to(torch.float32).to(args.device)
        y = y.to(args.device)

        model.train()
        output = model(x)
        loss_t = loss_fn(output, y)
        optimizer.zero_grad()
        loss_t.backward()
        optimizer.step()
        loss = loss_t.item()

        history['loss_iters'].append(i + 1)
        history['loss'].append(loss)

        elapsed_t = time.time() - train_start
        remaining_sec = elapsed_t / (i + 1) * (num_iterations - (i + 1))
        eta_str = (datetime.now(JST)
                   + timedelta(seconds=remaining_sec)).strftime('%H:%M:%S')
        pbar.set_postfix(loss=f'{loss:.4f}', eta=eta_str)

        if use_wandb:
            log_metrics({'iteration': i + 1, 'loss': loss})

        if (i + 1) % eval_freq == 0 or (i + 1) == num_iterations:
            train_acc = evaluate_hybrid(
                model, train_loader, args.device, soft=False,
                desc=f'train(hard) {i+1}/{num_iterations}')
            test_acc = evaluate_hybrid(
                model, test_loader, args.device, soft=False,
                desc=f'eval(hard) {i+1}/{num_iterations}')
            test_acc_soft = evaluate_hybrid(
                model, test_loader, args.device, soft=True,
                desc=f'eval(soft) {i+1}/{num_iterations}')

            is_best = test_acc > best_test_acc
            if is_best:
                best_test_acc = test_acc

            save_checkpoint(latest_path, model, optimizer, None,
                            i + 1, best_test_acc, ckpt_metadata)
            if is_best:
                save_checkpoint(best_path, model, optimizer, None,
                                i + 1, best_test_acc, ckpt_metadata)

            history['eval_iters'].append(i + 1)
            history['train_acc'].append(train_acc)
            history['test_acc'].append(test_acc)
            history['test_acc_soft'].append(test_acc_soft)

            elapsed = time.time() - train_start
            remaining = elapsed / (i + 1) * (num_iterations - (i + 1))
            eta = datetime.now(JST) + timedelta(seconds=remaining)
            elapsed_str = str(timedelta(seconds=int(elapsed)))
            remain_str = str(timedelta(seconds=int(remaining)))

            print(
                f'\n[iter {i+1:,}/{num_iterations:,}] loss={loss:.4f}  '
                f'train_acc={train_acc:.4f}  test_acc={test_acc:.4f}  '
                f'test_acc(soft)={test_acc_soft:.4f}'
                f'{"  ★ BEST" if is_best else ""}'
                f'  [{elapsed_str}<{remain_str}, 終了予定 {eta.strftime("%H:%M:%S")}]'
            )
            log(f'ITER {i+1}/{num_iterations} loss={loss:.4f} '
                f'train_acc={train_acc:.4f} test_acc={test_acc:.4f} '
                f'test_acc_soft={test_acc_soft:.4f} best={str(is_best).lower()} '
                f'elapsed={elapsed_str} eta={remain_str}')
            if use_wandb:
                log_metrics({
                    'iteration': i + 1,
                    'train_acc': train_acc,
                    'test_acc': test_acc,
                    'test_acc_soft': test_acc_soft,
                    'best_test_acc': best_test_acc,
                })

    total_sec = time.time() - train_start
    total_time = str(timedelta(seconds=int(total_sec)))
    final_test_acc = history['test_acc'][-1] if history['test_acc'] else 0.0
    final_test_acc_soft = (history['test_acc_soft'][-1]
                           if history['test_acc_soft'] else 0.0)
    print(f'\n最終hard test acc: {final_test_acc:.4f} / '
          f'best: {best_test_acc:.4f} (総訓練時間: {total_time})')
    log(f'DONE final_test_acc={final_test_acc:.4f} '
        f'final_test_acc_soft={final_test_acc_soft:.4f} '
        f'best_test_acc={best_test_acc:.4f} total_time={total_time}')
    log_file.close()

    fig_path = os.path.join(args.results_dir, f'{run_name}_curves.png')
    plot_curves(
        history,
        f'{run_name} (gates={total_gates:,}, nn_params={nn_params:,}, '
        f'iface={model.interface_dim}bit)',
        fig_path,
    )
    print(f'学習曲線: {fig_path}')

    if use_wandb:
        try:
            from PIL import Image
            with Image.open(fig_path) as im:
                log_images('curves', [im.convert('RGB')], captions=[run_name])
        except Exception as e:  # noqa: BLE001 — 図のログ失敗で訓練結果を落とさない
            print(f'警告: 学習曲線のwandbログに失敗: {e}')

        import wandb
        wandb.run.summary['final_test_acc'] = final_test_acc
        wandb.run.summary['final_test_acc_soft'] = final_test_acc_soft
        wandb.run.summary['best_test_acc'] = best_test_acc
        wandb.run.summary['total_gates'] = total_gates
        wandb.run.summary['nn_params'] = nn_params
        wandb.run.summary['total_params'] = total_params
        wandb.run.summary['interface_dim'] = model.interface_dim
        wandb.run.summary['total_time_sec'] = round(total_sec, 1)
        wandb.run.summary['expand_fill'] = args.expand_fill
        for key, value in extra.items():
            wandb.run.summary[key] = value

        log_checkpoint_artifact(
            best_path, run_name, stage='final',
            metadata={'args': vars(args), 'best_test_acc': best_test_acc},
            model=model,
        )
        wandb_finish()


if __name__ == '__main__':
    main()
