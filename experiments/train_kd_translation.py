"""
翻訳DLGN byte-level ja→en 系列KD訓練スクリプト。

目的はsoftではなくhard(離散eval=実回路)性能。生成タスクのsoft KDは冗長
(知見)なので、soft-logit KDは使わずCEのみで学習する:
    - data-freeアーム: teacher(既存MT等)の疑似訳(byte化済み)への純CE(alpha=1.0相当)
    - goldアーム: gold訳(byte化済み)への純CE(alpha=1.0相当)
どちらのアームも「teacherロジット」は不要で、損失は同一の
`cross_entropy(logits, target_ids, ignore_index=BYTE_PAD_ID)`(PAD除外)。
--supervision はwandbタグ/run名のためのラベルであり、損失関数自体は分岐しない
(コーパスの中身がteacher疑似訳かgold訳かだけが違う)。

参照元: train_kd_charlm_rddlgn.py のseq2seqアーム(train_step_seq2seq_scratch系,
evaluate_bpc/evaluate_teacher_agreementのsoft/hard切替パターン, _backward_and_step,
get_emb_reg_weight, wandb/checkpoint流儀)を踏襲する。

byte語彙: griffith_logic.convert.byte_vocab(BYTE_VOCAB_SIZE=259)。
Dataset: experiments/data/byte_pair_corpus.py の BytePairDataset/collate_pairs。
モデル: griffith_logic.DLGNAttentionTranslator / TranslatorConfig。

使用例:
    # スモーク(合成トイコーパス、CPU、--size s)
    WANDB_MODE=offline uv run python experiments/train_kd_translation.py \\
        --corpus data-translation/toy_train.pt \\
        --eval-corpus data-translation/toy_eval.pt \\
        --supervision datafree --size s \\
        -ni 30 -ef 10 -bs 8 --device cpu

    # フルスケール(GPU、Modal経由)
    uv run modal run scripts/modal_app.py --script train_kd_translation.py \\
        --args "--corpus data-translation/train_datafree.pt \\
                --eval-corpus data-translation/eval_gold.pt \\
                --supervision datafree --size m -ni 20000 -ef 1000"
"""

import argparse
import math
import os
import random
import sys
import time
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timedelta, timezone

JST = timezone(timedelta(hours=9))

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))

from checkpoint_utils import load_checkpoint, save_checkpoint, setup_checkpoint_dir
from data.byte_pair_corpus import BytePairDataset, collate_pairs
from env_utils import load_dotenv
from wandb_utils import (
    finish as wandb_finish,
)
from wandb_utils import (
    init_wandb,
    log_checkpoint_artifact,
    log_metrics,
)

from griffith_logic import DLGNAttentionTranslator, TranslatorConfig
from griffith_logic.convert.byte_vocab import (
    BYTE_BOS_ID,
    BYTE_EOS_ID,
    BYTE_PAD_ID,
    BYTE_VOCAB_SIZE,
    decode_bytes,
)
from griffith_logic.group_sum import GroupSum

load_dotenv()


# ============================================================
# 損失: 純CE(soft-logit KDは使わない。=生成タスクではsoft冗長)
# ============================================================

def make_pure_ce_seq_loss(pad_id: int = BYTE_PAD_ID):
    """純CE系列損失(teacher_logits不要)。

    train_kd_charlm_rddlgn.py の make_pure_ce_seq_loss と数値的に同一
    (ignore_index=pad_idのCEのみ)。data-free/goldどちらのアームも
    このCEのみを使う(teacherロジットへのKL/MSEは一切計算しない)。

    Returns:
        loss(student_logits[B,S,V], target_ids[B,S]) -> スカラー。
    """
    def loss_fn(student_logits: torch.Tensor, target_ids: torch.Tensor) -> torch.Tensor:
        vocab = student_logits.shape[-1]
        return F.cross_entropy(
            student_logits.reshape(-1, vocab), target_ids.reshape(-1).long(),
            ignore_index=pad_id,
        )
    return loss_fn


# ============================================================
# ランプ・訓練ステップ(train_kd_charlm_rddlgn.pyから踏襲)
# ============================================================

def get_emb_reg_weight(step: int, ramp_start: int = 1000, ramp_end: int = 100000,
                       max_weight: float = 0.1) -> float:
    """二値化正則化の重みを線形ランプで計算(train_rddlgn.py/train_kd_charlm_rddlgn.pyと同一)。"""
    if step < ramp_start:
        return 0.0
    if step >= ramp_end:
        return max_weight
    return max_weight * (step - ramp_start) / (ramp_end - ramp_start)


# Spectral正則化係数。mainでargs.spectral_regから設定される。
_SPECTRAL_REG = 0.0


def _backward_and_step(model, optimizer, scheduler, total_loss, grad_clip: float = 0.0):
    """backward+optimizer.step(+任意でgrad clip/scheduler)。train_kd_charlm_rddlgn.pyと同一契約。"""
    if _SPECTRAL_REG > 0:
        from griffith_logic.regularization import spectral_penalty
        total_loss = total_loss + _SPECTRAL_REG * spectral_penalty(model)
    optimizer.zero_grad()
    total_loss.backward()
    if grad_clip > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()
    if scheduler is not None:
        scheduler.step()


def train_step_translation(
    model, src, tgt_input, tgt_target, optimizer, step, loss_fn_ce, device, emb_reg_ramp,
    scheduler=None, grad_clip=0.0,
):
    """1ステップ訓練(純CE、data-free/gold共通)。

    train_kd_charlm_rddlgn.py の train_step_seq2seq_scratch と同型
    (teacher呼び出し無し、model(src, tgt_input) -> (logits, emb_reg_loss)契約)。
    """
    model.train()

    logits, emb_loss = model(src, tgt_input)  # (B, tgt_len, BYTE_VOCAB_SIZE)

    main_loss = loss_fn_ce(logits, tgt_target)

    emb_weight = get_emb_reg_weight(step, *emb_reg_ramp)
    total_loss = main_loss + emb_weight * emb_loss

    _backward_and_step(model, optimizer, scheduler, total_loss, grad_clip)

    return total_loss.item(), main_loss.item(), emb_loss.item()


# ============================================================
# chrF(文字n-gram F_beta、簡潔な自前実装。外部依存を足さない)
# ============================================================

def _char_ngrams(text: str, n: int) -> Counter:
    if len(text) < n:
        return Counter()
    return Counter(text[i:i + n] for i in range(len(text) - n + 1))


def chrf_score(hyp: str, ref: str, max_n: int = 6, beta: float = 2.0) -> float:
    """文字n-gram(1..max_n)のF_beta(既定beta=2、chrF標準値)。

    hyp/refどちらも空文字列の場合は1.0(完全一致とみなす)、片方だけ空なら0.0。
    """
    if len(hyp) == 0 and len(ref) == 0:
        return 1.0
    if len(hyp) == 0 or len(ref) == 0:
        return 0.0

    precisions = []
    recalls = []
    for n in range(1, max_n + 1):
        hyp_ngrams = _char_ngrams(hyp, n)
        ref_ngrams = _char_ngrams(ref, n)
        if not hyp_ngrams and not ref_ngrams:
            continue
        overlap = sum((hyp_ngrams & ref_ngrams).values())
        hyp_total = sum(hyp_ngrams.values())
        ref_total = sum(ref_ngrams.values())
        precisions.append(overlap / hyp_total if hyp_total > 0 else 0.0)
        recalls.append(overlap / ref_total if ref_total > 0 else 0.0)

    if not precisions:
        return 0.0

    p = sum(precisions) / len(precisions)
    r = sum(recalls) / len(recalls)
    if p + r == 0:
        return 0.0
    beta2 = beta * beta
    return (1 + beta2) * p * r / (beta2 * p + r)


# ============================================================
# データ準備
# ============================================================

def resolve_path(path: str) -> str:
    if not os.path.exists(path):
        raise FileNotFoundError(f'該当ファイルなし: {path}')
    return path


# ============================================================
# 評価: soft/hard両方のBPC・teacher-forced acc・greedy自己回帰chrF
# ============================================================

@torch.no_grad()
def evaluate_teacher_forced(model, loader, device, use_soft: bool):
    """held-out評価コーパスでのteacher-forced NLL/ln2(BPC)とnext-byte accuracy。

    use_soft: Trueならtrain mode(sigmoid連続=soft)、Falseならeval mode
        (Heaviside離散化後=hard、論理ゲート実機性能相当)。
        train_kd_charlm_rddlgn.py の evaluate_bpc/evaluate_teacher_agreement と
        同じ「model.train(mode=use_soft)」切替パターンを踏襲する。
    """
    was_training = model.training
    model.train(mode=use_soft)

    total_nll = 0.0
    total_correct = 0
    total_count = 0

    for src, tgt_input, tgt_target in loader:
        src = src.to(device)
        tgt_input = tgt_input.to(device)
        tgt_target = tgt_target.to(device)

        logits, _ = model(src, tgt_input)  # (B, tgt_len, V)
        mask = tgt_target != BYTE_PAD_ID

        log_probs = torch.log_softmax(logits, dim=-1)
        nll = -log_probs.gather(2, tgt_target.clamp(min=0).unsqueeze(-1)).squeeze(-1)
        nll = nll * mask
        total_nll += nll.sum().item()

        pred = logits.argmax(dim=-1)
        total_correct += ((pred == tgt_target) & mask).sum().item()
        total_count += mask.sum().item()

    model.train(mode=was_training)

    bpc = (total_nll / total_count) / math.log(2) if total_count > 0 else float('inf')
    acc = total_correct / total_count if total_count > 0 else 0.0
    return bpc, acc


@torch.no_grad()
def generate_greedy(model, src: torch.Tensor, tgt_len: int, device, use_soft: bool) -> torch.Tensor:
    """srcをencode→decoderを[BOS]から1バイトずつgreedy生成する。

    Args:
        src: (B, src_len) long
    Returns:
        (B, tgt_len) long のtoken列(BYTE_EOS_ID到達後はPADで埋める、ただし
        本実装は簡潔さのためtgt_len分すべて生成しEOS以降はそのまま出力する。
        chrf計算側でEOSより前のみを文字列化する)。
    """
    was_training = model.training
    model.train(mode=use_soft)

    b = src.shape[0]
    gen_ids = torch.full((b, 1), BYTE_BOS_ID, dtype=torch.long, device=device)
    for _t in range(tgt_len):
        tgt_input = torch.cat([
            gen_ids,
            torch.full((b, tgt_len - gen_ids.shape[1]), BYTE_PAD_ID, dtype=torch.long, device=device),
        ], dim=1)[:, :tgt_len]
        logits, _ = model(src, tgt_input)  # (B, tgt_len, V)
        step_idx = gen_ids.shape[1] - 1
        next_id = logits[:, step_idx, :].argmax(dim=-1, keepdim=True)  # (B, 1)
        gen_ids = torch.cat([gen_ids, next_id], dim=1)

    model.train(mode=was_training)
    return gen_ids[:, 1:1 + tgt_len]  # BOSを除いた生成分(最大tgt_len)


def _ids_to_text(ids: list) -> str:
    """生成ID列をEOSで打ち切ってからdecode_bytesで文字列化する。"""
    if BYTE_EOS_ID in ids:
        ids = ids[:ids.index(BYTE_EOS_ID)]
    return decode_bytes(ids)


@torch.no_grad()
def evaluate_chrf(model, loader, device, tgt_len, use_soft: bool, max_batches: int | None = None):
    """greedy自己回帰デコード英訳 vs gold のchrF平均。"""
    scores = []
    for i, (src, _tgt_input, tgt_target) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        src = src.to(device)
        gen = generate_greedy(model, src, tgt_len, device, use_soft=use_soft)
        gen = gen.cpu().tolist()
        gold = tgt_target.cpu().tolist()
        for g, ref in zip(gen, gold):
            hyp_text = _ids_to_text(g)
            ref_text = _ids_to_text(ref)
            scores.append(chrf_score(hyp_text, ref_text))
    return sum(scores) / len(scores) if scores else 0.0


@torch.no_grad()
def compute_anchors(eval_set, tgt_len: int):
    """比較アンカー: uniform BPC=log2(256)、gold unigram BPC、copy-source chrF。"""
    uniform_bpc = math.log2(256)

    # gold tgt の unigram(byte値0..255のみ)分布からBPCを計算する。
    counts = Counter()
    total = 0
    for tgt in eval_set.tgt:
        for b in tgt:
            if 0 <= b < 256:
                counts[b] += 1
                total += 1
    if total == 0:
        unigram_bpc = uniform_bpc
    else:
        entropy_bits = 0.0
        for c in counts.values():
            p = c / total
            entropy_bits -= p * math.log2(p)
        unigram_bpc = entropy_bits

    # copy-source chrF: srcをそのまま出力した場合のgoldとのchrF(srcはja、goldはen
    # なのでミスマッチが基本、参考アンカーとして低い値になる想定)。
    copy_scores = []
    for src, tgt in zip(eval_set.src, eval_set.tgt):
        hyp_text = _ids_to_text(list(src[:tgt_len]))
        ref_text = _ids_to_text(list(tgt[:tgt_len]))
        copy_scores.append(chrf_score(hyp_text, ref_text))
    copy_source_chrf = sum(copy_scores) / len(copy_scores) if copy_scores else 0.0

    return uniform_bpc, unigram_bpc, copy_source_chrf


# ============================================================
# モデル構築(3段プリセット: s / m / l)
#
# LutLayerのパラメータ数はout_dim*2^n_inputsで決まりin_dimに依存しない
# (n_inputs=2固定なら out_dim*4)。層幅を統一スケールkでスケーリングした
# 際の実測パラメータ数(uv run python内で確認済み、count_parameters()参照):
#   k=0.3 (s preset) -> 約485k
#   k=1.2 (m preset) -> 約1.9M
#   k=3.0 (l preset) -> 約4.8M (tgt_len=96のrecurrent decoderは1ステップごとに
#       LutStack全体を再計算するため、活性メモリはtransformerより重い。
#       11GB級GPUではbatch-sizeを32以下に落とすことを推奨)
# ============================================================

_SIZE_SCALE = {
    's': 0.3,
    'm': 1.2,
    'l': 3.0,
}


def build_config_translation(size: str, src_len: int, tgt_len: int,
                             selection: str = 'soft',
                             wiring: str = 'fixed', lm_tau: float = 0.001,
                             n_inputs: int = 2,
                             parameterization: str = 'softmax') -> TranslatorConfig:
    """sizeプリセット(s/m/l)からTranslatorConfigを構築する。

    vocab_size=259(byte語彙)はfrom_paperのvocab16000より2桁近く軽いため、
    浮いたパラメータ予算をencoder/decoder/attention層(n/key/l/p/qp/m)に
    振り直す設計(from_paper/tinyの group_factor 調整パターンを踏襲)。
    m_layer_sizes[-1]はvocab_size(259)の倍数に丸める。
    """
    if size not in _SIZE_SCALE:
        raise ValueError(f'size は {list(_SIZE_SCALE)} のいずれか: {size!r}')

    k = _SIZE_SCALE[size]
    vocab_size = BYTE_VOCAB_SIZE

    d_emb = max(48, int(80 * math.sqrt(k)))
    n_layer_sizes = [max(64, int(1500 * k))] * 2
    key_encoder_sizes = [max(32, int(750 * k)), max(16, int(375 * k))]
    l_layer_sizes = [max(64, int(1500 * k))] * 2
    p_layer_sizes = [max(64, int(3000 * k)), max(32, int(1500 * k))]
    query_producer_sizes = [max(32, int(750 * k)), max(16, int(375 * k))]

    m_hidden = max(128, int(4000 * k))
    m_last_factor = max(1, int(300 * k))
    m_layer_sizes = [m_hidden, m_hidden, vocab_size * m_last_factor]

    return TranslatorConfig(
        vocab_size=vocab_size,
        d_emb=d_emb,
        src_len=src_len,
        tgt_len=tgt_len,
        n_layer_sizes=n_layer_sizes,
        key_encoder_sizes=key_encoder_sizes,
        l_layer_sizes=l_layer_sizes,
        p_layer_sizes=p_layer_sizes,
        query_producer_sizes=query_producer_sizes,
        m_layer_sizes=m_layer_sizes,
        selection=selection,
        wiring=wiring,
        lm_tau=lm_tau,
        n_inputs=n_inputs,
        parameterization=parameterization,
    )


# ============================================================
# ログ
# ============================================================

def setup_logger(supervision, log_dir='logs'):
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now(JST).strftime('%Y%m%d_%H%M%S')
    log_path = os.path.join(log_dir, f'kd_translation_{supervision}_{timestamp}.log')
    log_file = open(log_path, 'a', buffering=1)
    print(f'ログファイル: {log_path}')

    def log(msg):
        ts = datetime.now(JST).strftime('%Y-%m-%d %H:%M:%S')
        log_file.write(f'[{ts}] {msg}\n')

    return log, log_file


def load_n(loader, n):
    i = 0
    while i < n:
        for batch in loader:
            yield batch
            i += 1
            if i == n:
                return


# ============================================================
# メイン
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='翻訳DLGN byte-level ja->en 系列KD訓練(純CE)')

    parser.add_argument('--corpus', type=str, required=True, help='訓練コーパス.pt(byte_pair_corpus形式)')
    parser.add_argument('--eval-corpus', type=str, required=True, help='held-out gold評価コーパス.pt')
    parser.add_argument('--supervision', type=str, default='datafree', choices=['datafree', 'gold'],
                        help="datafree(既定)=teacher疑似訳への純CE。gold=gold訳への純CE。"
                             "損失関数自体は分岐しない(どちらもCEのみ)。wandbタグ/run名にのみ反映。")

    parser.add_argument('--size', type=str, default='m', choices=['s', 'm', 'l'])
    parser.add_argument('--selection', type=str, default='soft',
                        choices=['soft', 'hard_st', 'gumbel_st'],
                        help="訓練時ゲート選択。soft=連続ブレンド(既定)。"
                             "hard_st=argmax forward+STE backward(CAGE)でhard経路を訓練し"
                             "soft-hard gapを縮める(hard性能レバー)。")
    parser.add_argument('--spectral-reg', type=float, default=0.0,
                        help='Spectral正則化係数λ(DWN §3.4。0で無効)')
    parser.add_argument('--wiring', type=str, default='fixed', choices=['fixed', 'learnable'],
                        help="入力配線。fixed=random固定(既定)。learnable=DWN配線学習"
                             "(各LutStackの第1層のみ、BP専用)。「停滞は容量でなく配線」検証。")
    parser.add_argument('--lm-tau', type=float, default=0.001,
                        help="learnable配線時のLearnableMapping softmax温度(既定0.001)。")
    parser.add_argument('--n-inputs', type=int, default=2,
                        help="LUTのfan-in(1ゲートが読む入力数)。既定2(16ゲート)。"
                             "算術のhard律速と判明→翻訳の必要fan-inを測る掃引軸。")
    parser.add_argument('--parameterization', type=str, default='softmax',
                        choices=['softmax', 'independent', 'warp'],
                        help="LUTパラメータ化。softmax(既定,n=2の16ゲート)/independent"
                             "(任意真理値表,n>2のnative,式)/warp。"
                             "注: hard_stはsoftmax/warpのみ対応(independentは非対応)。")
    parser.add_argument('--src-len', type=int, default=96)
    parser.add_argument('--tgt-len', type=int, default=96)

    parser.add_argument('--lr', type=float, default=0.05)
    parser.add_argument('--weight-decay', type=float, default=0.001)
    parser.add_argument('-bs', '--batch-size', type=int, default=64)
    parser.add_argument('-ni', '--num-iters', type=int, default=20000)
    parser.add_argument('-ef', '--eval-freq', type=int, default=1000)
    parser.add_argument('--eval-max-batches', type=int, default=None,
                        help='chrF評価(自己回帰生成、重い)に使う最大バッチ数。Noneなら全評価コーパス。')

    parser.add_argument('--emb-reg-ramp-start', type=int, default=1000)
    parser.add_argument('--emb-reg-ramp-end', type=int, default=100000)
    parser.add_argument('--emb-reg-max-weight', type=float, default=0.1)

    parser.add_argument('--grad-clip', type=float, default=1.0,
                        help='>0のときclip_grad_norm_を適用(既定1.0)。')
    parser.add_argument('--warmup', type=int, default=1000, help='線形warmupステップ数(0でwarmup無効)。')
    parser.add_argument('--min-lr-ratio', type=float, default=0.1,
                        help='warmup後のcosine減衰の下限(lr*ratio)。')

    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints')
    parser.add_argument('--init-ckpt', type=str, default=None,
                        help='初期重みを読むcheckpoint(.pt)。soft事前学習→hard_st finetuneの'
                             'カリキュラム用(optimizer/schedulerは復元せずweightsのみ)。')

    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--log-dir', type=str, default='logs')
    parser.add_argument('--no-wandb', action='store_true')
    parser.add_argument('--wandb-group', type=str, default='kd-translation')
    parser.add_argument('--wandb-tags', type=str, default=None,
                        help='追加タグ(カンマ区切り)。既定タグ(kd,translation,jaen,dlgn-attn,supervision)に追加する。')

    args = parser.parse_args()

    # Spectral正則化の一元設定(_backward_and_stepが参照)
    global _SPECTRAL_REG
    _SPECTRAL_REG = getattr(args, 'spectral_reg', 0.0)

    if args.device == 'cuda' and not torch.cuda.is_available():
        print('CUDA不可、CPUにフォールバック')
        args.device = 'cpu'
    print(f'デバイス: {args.device}')

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    corpus_path = resolve_path(args.corpus)
    eval_corpus_path = resolve_path(args.eval_corpus)
    print(f'訓練コーパス: {corpus_path}')
    print(f'評価コーパス: {eval_corpus_path}')

    train_set = BytePairDataset(corpus_path, src_len=args.src_len, tgt_len=args.tgt_len)
    eval_set = BytePairDataset(eval_corpus_path, src_len=args.src_len, tgt_len=args.tgt_len)
    print(f'訓練サンプル: {len(train_set):,}  評価サンプル: {len(eval_set):,}')

    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, drop_last=True,
        collate_fn=collate_pairs,
    )
    eval_loader = DataLoader(
        eval_set, batch_size=args.batch_size, shuffle=False, drop_last=True,
        collate_fn=collate_pairs,
    )

    # モデル構築
    config = build_config_translation(args.size, args.src_len, args.tgt_len, args.selection,
                                      args.wiring, args.lm_tau, args.n_inputs, args.parameterization)
    model = DLGNAttentionTranslator(config, device=args.device).to(args.device)
    config_dict = asdict(config)

    counts = model.count_parameters()
    print('\nパラメータ数:')
    for name, count in counts.items():
        print(f'  {name}: {count:,}')

    # soft→hard_st カリキュラム: 事前学習重みをロード(weightsのみ、optimizerは新規)
    if args.init_ckpt:
        info = load_checkpoint(resolve_path(args.init_ckpt), model, device=args.device)
        print(f'init-ckpt ロード: {args.init_ckpt} '
              f'(step={info["step"]}, best_metric={info["best_metric"]:.4f})')

    loss_fn_ce = make_pure_ce_seq_loss(pad_id=BYTE_PAD_ID)
    emb_reg_ramp = (args.emb_reg_ramp_start, args.emb_reg_ramp_end, args.emb_reg_max_weight)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
        betas=(0.9, 0.999), eps=1e-8,
    )

    # warmup-cosine LR(train_kd_charlm_rddlgn.pyのlr-schedule=warmup-cosineと同型)
    warmup_iters = args.warmup
    total_iters = args.num_iters
    min_lr_ratio = args.min_lr_ratio

    def lr_lambda(current_step: int) -> float:
        if warmup_iters > 0 and current_step < warmup_iters:
            return (current_step + 1) / max(1, warmup_iters)
        progress = (current_step - warmup_iters) / max(1, total_iters - warmup_iters)
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    log, log_file = setup_logger(args.supervision, args.log_dir)
    log(f'START supervision={args.supervision} size={args.size} corpus={corpus_path} '
        f'bs={args.batch_size} ni={args.num_iters} device={args.device} params={counts["total"]:,}')

    # selection を prefix に含める(soft/hard_st を同時起動しても checkpoint dir が
    # 衝突しないように。setup_checkpoint_dir は秒精度のtimestampなので、同一
    # (supervision,size)を同秒に2本起動すると同名dirに衝突しうる)。
    ckpt_prefix = (f'kd_translation_{args.supervision}_{args.size}_{args.selection}'
                   f'_{args.wiring}_n{args.n_inputs}{args.parameterization[:4]}')
    run_dir, latest_path, best_path = setup_checkpoint_dir(ckpt_prefix, base_dir=args.checkpoint_dir)
    # best(soft基準)とは別に、hard byte BPC最小のbest_hard.ptを保存する
    # (このrunの主目的=hard性能。train_kd_charlm_rddlgn.pyのbest_hard.pt流儀を踏襲)。
    best_hard_path = os.path.join(os.path.dirname(best_path), 'best_hard.pt')
    ckpt_metadata = {'args': vars(args), 'config': config_dict}
    print(f'チェックポイント保存先: {run_dir}/')

    uniform_bpc, unigram_bpc, copy_source_chrf = compute_anchors(eval_set, args.tgt_len)
    print(f'アンカー: uniform_BPC={uniform_bpc:.4f}  gold_unigram_BPC={unigram_bpc:.4f}  '
          f'copy_source_chrF={copy_source_chrf:.4f}')

    run_name = (
        f'kd_translation_{args.supervision}_{args.size}_bs{args.batch_size}'
        f'_lr{args.lr}_ni{args.num_iters}'
    )
    wandb_config = {
        **vars(args), **config_dict,
        'uniform_bpc': uniform_bpc,
        'gold_unigram_bpc': unigram_bpc,
        'copy_source_chrf': copy_source_chrf,
        'n_train_pairs': len(train_set),
        'n_eval_pairs': len(eval_set),
    }
    use_wandb = False
    if not args.no_wandb:
        tags = ['kd', 'translation', 'jaen', args.supervision, 'dlgn-attn', args.size,
                args.selection, f'wiring-{args.wiring}',
                f'nin{args.n_inputs}', f'param-{args.parameterization}']
        if args.wandb_tags:
            tags += [t.strip() for t in args.wandb_tags.split(',') if t.strip()]
        use_wandb = init_wandb(
            run_name=run_name,
            config_dict=wandb_config,
            model=model,
            device=args.device,
            tags=tags,
            group=args.wandb_group,
        )

    print(f'\n訓練開始: {args.num_iters:,} イテレーション')
    print('=' * 60)

    train_start = time.time()
    best_bpc_soft = float('inf')
    best_bpc_hard = float('inf')

    pbar = tqdm(
        enumerate(load_n(train_loader, args.num_iters)),
        desc='train', total=args.num_iters,
    )

    for step, (src, tgt_input, tgt_target) in pbar:
        src = src.to(args.device)
        tgt_input = tgt_input.to(args.device)
        tgt_target = tgt_target.to(args.device)

        total_loss, main_loss, emb_loss = train_step_translation(
            model, src, tgt_input, tgt_target, optimizer, step, loss_fn_ce,
            args.device, emb_reg_ramp, scheduler=scheduler, grad_clip=args.grad_clip,
        )

        elapsed = time.time() - train_start
        sec_per_iter = elapsed / (step + 1)
        remaining_sec = sec_per_iter * (args.num_iters - (step + 1))
        eta_str = (datetime.now(JST) + timedelta(seconds=remaining_sec)).strftime('%H:%M:%S')
        pbar.set_postfix(loss=f'{total_loss:.4f}', main=f'{main_loss:.4f}', eta=eta_str)

        if (step + 1) % args.eval_freq == 0 or step == args.num_iters - 1:
            bpc_soft, acc_soft = evaluate_teacher_forced(model, eval_loader, args.device, use_soft=True)
            bpc_hard, acc_hard = evaluate_teacher_forced(model, eval_loader, args.device, use_soft=False)
            chrf_soft = evaluate_chrf(
                model, eval_loader, args.device, args.tgt_len, use_soft=True,
                max_batches=args.eval_max_batches,
            )
            chrf_hard = evaluate_chrf(
                model, eval_loader, args.device, args.tgt_len, use_soft=False,
                max_batches=args.eval_max_batches,
            )

            is_best_soft = bpc_soft < best_bpc_soft
            if is_best_soft:
                best_bpc_soft = bpc_soft

            # best checkpointはhard byte BPC最小で選ぶ(目的はhard性能)。
            is_best_hard = bpc_hard < best_bpc_hard
            if is_best_hard:
                best_bpc_hard = bpc_hard

            saved_step = step + 1
            save_checkpoint(latest_path, model, optimizer, scheduler, saved_step, best_bpc_hard, ckpt_metadata)
            if is_best_soft:
                save_checkpoint(best_path, model, optimizer, scheduler, saved_step, best_bpc_soft, ckpt_metadata)
            if is_best_hard:
                save_checkpoint(
                    best_hard_path, model, optimizer, scheduler, saved_step, best_bpc_hard, ckpt_metadata,
                )

            elapsed = time.time() - train_start
            sec_per_iter = elapsed / (step + 1)
            remaining = sec_per_iter * (args.num_iters - (step + 1))
            eta = datetime.now(JST) + timedelta(seconds=remaining)
            elapsed_str = str(timedelta(seconds=int(elapsed)))
            remain_str = str(timedelta(seconds=int(remaining)))
            eta_str = eta.strftime('%H:%M:%S')
            emb_w = get_emb_reg_weight(step, *emb_reg_ramp)
            current_lr = optimizer.param_groups[0]['lr']

            print(
                f'\n[iter {step+1:,}/{args.num_iters:,}] '
                f'loss={total_loss:.4f} main={main_loss:.4f} emb={emb_loss:.4f}(w={emb_w:.4f})  '
                f'BPC(soft)={bpc_soft:.4f} BPC(hard)={bpc_hard:.4f}  '
                f'acc(soft)={acc_soft:.4f} acc(hard)={acc_hard:.4f}  '
                f'chrF(soft)={chrf_soft:.4f} chrF(hard)={chrf_hard:.4f}  lr={current_lr:.6f}'
                f'{"  * BEST" if is_best_soft else ""}'
                f'{"  * BEST(hard)" if is_best_hard else ""}'
                f'  [{elapsed_str}<{remain_str}, 終了予定 {eta_str}]'
            )

            # サンプル3件: (ja入力, soft英訳, hard英訳, gold)
            n_samples = min(3, src.shape[0])
            sample_src = src[:n_samples]
            gen_soft = generate_greedy(model, sample_src, args.tgt_len, args.device, use_soft=True).cpu().tolist()
            gen_hard = generate_greedy(model, sample_src, args.tgt_len, args.device, use_soft=False).cpu().tolist()
            print('  サンプル:')
            for i in range(n_samples):
                ja_text = _ids_to_text(sample_src[i].cpu().tolist())
                soft_text = _ids_to_text(gen_soft[i])
                hard_text = _ids_to_text(gen_hard[i])
                gold_text = _ids_to_text(tgt_target[i].cpu().tolist())
                print(f'    ja  : {ja_text!r}')
                print(f'    soft: {soft_text!r}')
                print(f'    hard: {hard_text!r}')
                print(f'    gold: {gold_text!r}')

            log(f'ITER {step+1}/{args.num_iters} loss={total_loss:.4f} main={main_loss:.4f} '
                f'emb={emb_loss:.4f} bpc_soft={bpc_soft:.4f} bpc_hard={bpc_hard:.4f} '
                f'acc_soft={acc_soft:.4f} acc_hard={acc_hard:.4f} '
                f'chrf_soft={chrf_soft:.4f} chrf_hard={chrf_hard:.4f} '
                f'best_soft={str(is_best_soft).lower()} best_hard={str(is_best_hard).lower()}')

            if use_wandb:
                log_metrics({
                    'iteration': step + 1,
                    'loss/total': total_loss,
                    'loss/main': main_loss,
                    'loss/emb_reg': emb_loss,
                    'loss/emb_reg_weight': emb_w,
                    'bpc/soft': bpc_soft,
                    'bpc/hard': bpc_hard,
                    'acc/soft': acc_soft,
                    'acc/hard': acc_hard,
                    'chrf/soft': chrf_soft,
                    'chrf/hard': chrf_hard,
                    'best_bpc_soft': best_bpc_soft,
                    'best_bpc_hard': best_bpc_hard,
                    'anchor/uniform_bpc': uniform_bpc,
                    'anchor/gold_unigram_bpc': unigram_bpc,
                    'anchor/copy_source_chrf': copy_source_chrf,
                    'lr': current_lr,
                }, step=step + 1)

    total_time = str(timedelta(seconds=int(time.time() - train_start)))
    print(f'\n最終BPC(best soft): {best_bpc_soft:.4f}  最終BPC(best hard): {best_bpc_hard:.4f}  (総訓練時間: {total_time})')
    log(f'DONE best_bpc_soft={best_bpc_soft:.4f} best_bpc_hard={best_bpc_hard:.4f} total_time={total_time}')
    log_file.close()

    if use_wandb:
        log_checkpoint_artifact(
            best_hard_path, run_name, stage='final',
            metadata={'args': vars(args), 'best_bpc_hard': best_bpc_hard, 'best_bpc_soft': best_bpc_soft},
            model=model,
        )
        wandb_finish()


if __name__ == '__main__':
    main()
