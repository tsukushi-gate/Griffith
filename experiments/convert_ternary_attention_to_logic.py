"""量子化(1bit/1.58bit) attention/FFNブロック → LogicAttention+FFN(DLGN)転写。

量子化起点の転写(experiments/convert_ternary_to_dwn_mnist.py)の「量子化起点→転写は崖回避」を、
transformerブロック(causal self-attention + FFN)へ拡張する。土台は
experiments/convert_attention_ffn_charlm.py(char-LM FFN構造転写PoC)。

正直な範囲宣言(重要):
    softmax/argmax は論理プリミティブ(lut_layer/concat/add)で表現できない
    (blocks/attention.py docstring)。よって attention 核(XNOR類似度 + softmax/
    argmax)は既存の LogicAttention(soft訓練→hard推論)機構をそのまま使い、
    転写対象は **QKV射影(q/k/v_proj = LutStack)と FFN(c_fc→GELU→c_proj)** に限る。
    「転写の厳密性」は QKV射影・FFN の較正データ上 agreement で報告し、softmax
    部分は hard attention 評価精度(HARD BPC / teacher agreement)で報告する。

アーム(4本、同一パイプライン。QKV射影+FFN線形層を arm別に量子化):
    float   : 枝刈りなしfloat重みを閾値二値化転写(sparsify_fanin top-F。基準)
    bin     : 1bit sign 量子化(0マスクなし=密、binarize_weights)+QAT回復
    ternary : 1.58bit absmean 量子化(0マスク=疎、quantize_ternary)+QAT回復。
              sparsify_from_mask で 0マスク(枝刈り済み構造)を尊重して転写
    scratch : ランダム初期化LogicGPT(転写なし)を同容量・同学習ループでpolish(下限)

パイプライン(1 arm):
    1. teacher の各 block について c_attn(→q/k/v 3分割)と MLP(c_fc/c_proj)の
       線形重みを arm別に量子化(枝刈り0保持)+QAT(STE forward, 系列CE/KDで回復)。
    2. 量子化重みを bake し、
         - QKV射影 → transcribe_linear_to_lutlayer(単層 閾値二値化LUT) →
           student attn.{q,k,v}_proj.layers[0] へ restore_lut_layer
         - FFN → transcribe_ffn_block(既存)→ block.mlp へ init_logic_mlp_from_ffn
    3. LogicAttention は mode='soft' で訓練、'hard' で評価(soft→hard推論)。
    4. soft polish(純CE または --polish-kd で系列KD混合)。soft最良で重み選択
       (memory: soft-best-not-hard-best、評価は全部soft側で選び hard は実体報告)。

指標:
    - source hard(量子化teacherの nanoGPT自体の hard BPC。attention核は実数のまま
      なので「量子化QKV/FFNを bake した teacher」の教師強制BPC)
    - QKV射影 agreement / FFN agreement(較正データ上、転写の厳密性代理)
    - soft polish後の best BPC(soft)で重み選択 → その時点の hard BPC / teacher agreement
    - arm比較: ternary−scratch(一次)、ternary−bin(0マスク純効果)、ternary−float

使用例:
    # 煙実行(CPU数分、wandb無効)
    uv run python experiments/convert_ternary_attention_to_logic.py --quick --no-wandb

    # 本実行(GPU推奨)
    uv run python experiments/convert_ternary_attention_to_logic.py \\
        --corpus data-charlm/corpus_XXXX.pt --eval-corpus data-charlm/*_eval.pt -ni 20000 -ef 1000
"""

from __future__ import annotations

import argparse
import copy
import math
import os
import random
import sys
import time
from dataclasses import asdict
from datetime import datetime, timedelta, timezone

JST = timezone(timedelta(hours=9))

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from data.char_corpus import (
    PAD_ID,
    STUDENT_VOCAB_SIZE,
    TeacherCorpusDataset,
    load_tiny_shakespeare_val,
)
from env_utils import load_dotenv
from train_kd_charlm_rddlgn import (
    collate_raw_window,
    evaluate_bpc,
    evaluate_teacher_agreement,
    evaluate_teacher_bpc,
    generate_sample,
    make_pure_ce_seq_loss,
    resolve_glob,
    train_step_decoder_only,
    train_step_decoder_only_scratch,
)
from wandb_utils import finish as wandb_finish
from wandb_utils import init_wandb, log_metrics

from griffith_logic.convert.binarize import binarize_weights, quantize_ternary
from griffith_logic.convert.distill import make_seq_kd_loss
from griffith_logic.convert.text_teacher import resolve_text_teacher
from griffith_logic.convert.transcribe import (
    sparsify_fanin,
    sparsify_from_mask,
    threshold_truth_tables,
)
from griffith_logic.convert.transcribe_seq import FfnTranscribeSpec, init_logic_mlp_from_ffn
from griffith_logic.encoding.thermometer import ThermometerEncoder
from griffith_logic.hybrid.policy import LogicSpec
from griffith_logic.ir.serialize import tt_to_hex
from griffith_logic.ir.weights import restore_lut_layer
from griffith_logic.layers.logic_linear import LogicMLP
from griffith_logic.models.logic_gpt import LogicGPT, LogicGPTConfig

load_dotenv()

ARMS = ('float', 'bin', 'ternary', 'scratch')


# ============================================================
# 量子化ユーティリティ(convert_ternary_to_dwn_mnist と同一 STE 意味論)
# ============================================================

def _quantize_weight(w: torch.Tensor, arm: str, mask: torch.Tensor,
                     eps: float = 1e-5) -> torch.Tensor:
    """arm別に量子化した実効重みを返す(bake用。枝刈りmask外は0)。

    bin: sign × alpha(mask内平均|W|)。ternary: absmean三値化(per-channel scale)。
    float: 恒等(maskは全True)。scale/alphaは mask内平均で計算する
    (枝刈り0が平均を薄めない、convert_ternary_to_dwn_mnist._quantize_ste_weightと同式)。
    """
    m = mask.to(w.dtype)
    if arm == 'bin':
        alpha = (w.detach().abs() * m).sum() / m.sum().clamp_min(1.0)
        q = torch.sign(w) * alpha
    elif arm == 'ternary':
        denom = m.sum(dim=1, keepdim=True).clamp_min(1.0)
        scale = ((w.detach().abs() * m).sum(dim=1, keepdim=True) / denom).clamp_min(eps)
        q = torch.clamp(torch.round(w / scale), -1.0, 1.0) * scale
    else:  # float
        q = w
    return q * m


def _quantize_ste(w: torch.Tensor, arm: str, mask: torch.Tensor) -> torch.Tensor:
    """STE量子化: forward=量子化重み、backward=恒等(shadow重みへ素通し)。"""
    q = _quantize_weight(w, arm, mask)
    return w + (q - w).detach()


def _linear_mask(w: torch.Tensor, arm: str) -> torch.Tensor:
    """arm別の枝刈りマスク。ternaryのみ 0マスク(absmean三値化の0)、他は全True。"""
    if arm == 'ternary':
        _, _, mask = quantize_ternary(w.detach(), per='channel')
        return mask
    return torch.ones_like(w, dtype=torch.bool)


# ============================================================
# teacher量子化ブロックの構築(c_attn split + QAT)
# ============================================================

class _QuantAttnBlock:
    """1 teacher block の量子化対象線形層(q/k/v/FFN)を束ねるハンドル。

    teacher._CausalSelfAttention.c_attn(n_embd→3*n_embd)を q/k/v の3つの
    (d_model×d_model)線形へ分割し、FFN(c_fc/c_proj)と合わせて arm別に量子化・
    QAT・bakeする。teacher本体(nanoGPT)のc_attn.weightを直接書き換える
    (QAT後の量子化決定値で bake)ことで、量子化QKV/FFNを反映した teacher の
    hard BPCを source指標として測れる。
    """

    def __init__(self, teacher_block, d_model: int, arm: str, device: str):
        self.blk = teacher_block
        self.d = d_model
        self.arm = arm
        # c_attn.weight: (3*d, d)。行 [0:d]=q, [d:2d]=k, [2d:3d]=v。
        w = teacher_block.attn.c_attn.weight.detach()
        self.masks = {
            'q': _linear_mask(w[:d_model], arm).to(device),
            'k': _linear_mask(w[d_model:2 * d_model], arm).to(device),
            'v': _linear_mask(w[2 * d_model:], arm).to(device),
        }
        # FFN のマスク
        self.masks['c_fc'] = _linear_mask(teacher_block.mlp.c_fc.weight.detach(), arm).to(device)
        self.masks['c_proj'] = _linear_mask(teacher_block.mlp.c_proj.weight.detach(), arm).to(device)


def _bake_quant_into_teacher(qblocks: list[_QuantAttnBlock], arm: str) -> None:
    """各 teacher block の c_attn / c_fc / c_proj の重みを量子化決定値で bake する。"""
    if arm in ('float', 'scratch'):
        return
    for qb in qblocks:
        d = qb.d
        with torch.no_grad():
            w = qb.blk.attn.c_attn.weight
            for name, sl in (('q', slice(0, d)), ('k', slice(d, 2 * d)), ('v', slice(2 * d, 3 * d))):
                w[sl].copy_(_quantize_weight(w[sl], arm, qb.masks[name]).detach())
            fc = qb.blk.mlp.c_fc.weight
            fc.copy_(_quantize_weight(fc, arm, qb.masks['c_fc']).detach())
            pr = qb.blk.mlp.c_proj.weight
            pr.copy_(_quantize_weight(pr, arm, qb.masks['c_proj']).detach())


def _qat_finetune_teacher(teacher, qblocks, arm, windows, device, *, epochs, lr, batch_size):
    """量子化aware fine-tune: 毎forwardで c_attn/FFN重みをSTE量子化しつつ、
    teacher(nanoGPT)を系列CEで回復する。backwardはshadow(float)重みへ流す。

    teacherの他パラメータ(embedding/LN/lm_head)も一緒に更新して整合を取る。
    枝刈り0はstepごとに再ゼロして維持する(shadow復活防止)。
    """
    if arm in ('float', 'scratch') or epochs <= 0:
        return
    model = teacher.model
    model.train()
    params = list(model.parameters())
    opt = torch.optim.Adam(params, lr=lr)
    n = len(windows)
    d = qblocks[0].d
    for _ep in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            batch = windows[idx].to(device)  # (B, 64)
            # STE量子化重みを計算して qb._wq_* に保持し、forward がそれを使う
            # (weightはleafパラメータのため直接置換せず、量子化値を別途持ち回る)。
            for qb in qblocks:
                w = qb.blk.attn.c_attn.weight
                wq = w.clone()
                for name, sl in (('q', slice(0, d)), ('k', slice(d, 2 * d)),
                                 ('v', slice(2 * d, 3 * d))):
                    wq[sl] = _quantize_ste(w[sl], arm, qb.masks[name])
                qb._wq_attn = wq
                qb._wq_fc = _quantize_ste(qb.blk.mlp.c_fc.weight, arm, qb.masks['c_fc'])
                qb._wq_proj = _quantize_ste(qb.blk.mlp.c_proj.weight, arm, qb.masks['c_proj'])

            logits = _teacher_forward_with_quant(model, batch, qblocks, d)
            # 系列CE: 位置iはbatch[i+1]を予測
            target = batch[:, 1:].reshape(-1)
            loss = F.cross_entropy(logits[:, :-1, :].reshape(-1, logits.shape[-1]), target)
            opt.zero_grad()
            loss.backward()
            opt.step()
            with torch.no_grad():  # 枝刈り0維持
                for qb in qblocks:
                    w = qb.blk.attn.c_attn.weight
                    for name, sl in (('q', slice(0, d)), ('k', slice(d, 2 * d)),
                                     ('v', slice(2 * d, 3 * d))):
                        w[sl].mul_(qb.masks[name].to(w.dtype))
                    qb.blk.mlp.c_fc.weight.mul_(qb.masks['c_fc'].to(w.dtype))
                    qb.blk.mlp.c_proj.weight.mul_(qb.masks['c_proj'].to(w.dtype))


def _teacher_forward_with_quant(model, idx, qblocks, d):
    """nanoGPT forward を、c_attn/FFN の重みをSTE量子化値(qb._wq_*)で差し替えて実行する。

    _MinGPT.forward の構造を手動で辿り、attention/FFN の線形演算だけ量子化重みを使う。
    F.scaled_dot_product_attention(is_causal=True)はteacher原実装に合わせる。
    """
    b, t = idx.shape
    device = idx.device
    pos = torch.arange(t, dtype=torch.long, device=device)
    x = model.transformer.wte(idx) + model.transformer.wpe(pos)
    qb_by_block = {id(qb.blk): qb for qb in qblocks}
    for block in model.transformer.h:
        qb = qb_by_block.get(id(block))
        # attention(量子化 c_attn)
        h = block.ln_1(x)
        if qb is not None:
            qkv = F.linear(h, qb._wq_attn, block.attn.c_attn.bias)
        else:
            qkv = block.attn.c_attn(h)
        q, k, v = qkv.split(block.attn.n_embd, dim=2)
        nh = block.attn.n_head
        c = block.attn.n_embd
        q = q.view(b, t, nh, c // nh).transpose(1, 2)
        k = k.view(b, t, nh, c // nh).transpose(1, 2)
        v = v.view(b, t, nh, c // nh).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(b, t, c)
        x = x + block.attn.c_proj(y)
        # FFN(量子化 c_fc/c_proj)
        hf = block.ln_2(x)
        if qb is not None:
            mid = F.gelu(F.linear(hf, qb._wq_fc, block.mlp.c_fc.bias))
            x = x + F.linear(mid, qb._wq_proj, block.mlp.c_proj.bias)
        else:
            x = x + block.mlp(hf)
    x = model.transformer.ln_f(x)
    return model.lm_head(x)


# ============================================================
# QKV射影の単層 閾値二値化LUT転写(FFN hidden層の意味論を単層に適用)
# ============================================================

def transcribe_linear_to_lutlayer(
    lin_weight: torch.Tensor,
    lin_bias: torch.Tensor,
    lut_layer,
    calib_bits: np.ndarray,
    *,
    n_inputs: int,
    mask: torch.Tensor | None = None,
) -> float:
    """量子化済み線形層(d_in→d_out)を単層 閾値二値化LUT へ転写し、LutLayerへ書き込む。

    y_i = 1[Σ_j W_ij x_j + b_i > 0] を fan-in疎化(mask尊重 or |W|上位)して
    真理値表化する(transcribe.transcribe_mlp の隠れ層意味論と同一)。QKV射影は
    class scoreでなく残差論理ビットなので transcribe_ffn_block と同じ「符号情報のみ
    保持する近似」であり、agreementを honest に返す(0を主張しない)。

    Args:
        lin_weight: (out, in) の量子化済み実効重み(mask外=0前提)
        lin_bias: (out,) バイアス(None不可、呼び出し側でゼロ埋め)
        lut_layer: 書き込み先 LutLayer(in/out/n_inputs が一致していること)
        calib_bits: (B, in) の 0/1 較正データ(post-LN 二値化 hidden)
        n_inputs: LUT fan-in(2/4/6)
        mask: (out, in) bool 残存接続マスク(ternary)。指定時 sparsify_from_mask

    Returns:
        較正データ上の agreement(閾値ゲート厳密値 vs fan-in疎化後の一致率)。
    """
    w = lin_weight.detach().to(torch.float64).cpu()
    b = lin_bias.detach().to(torch.float64).cpu()
    h = np.asarray(calib_bits).astype(np.int64)
    assert set(np.unique(h)).issubset({0, 1}), '較正データは0/1のみ'

    if mask is not None:
        indices, w_sel, bshift = sparsify_from_mask(w, mask.cpu(), n_inputs, calib_inputs=h)
    else:
        indices, w_sel, bshift = sparsify_fanin(w, n_inputs, calib_inputs=h)
    b_eff = b + bshift

    k_actual = indices.shape[0]
    if k_actual < n_inputs:
        pad = n_inputs - k_actual
        indices = np.vstack([indices, np.zeros((pad, indices.shape[1]), dtype=np.int64)])
        w_sel = torch.cat([w_sel, torch.zeros(w_sel.shape[0], pad, dtype=w_sel.dtype)], dim=1)
    tts = threshold_truth_tables(w_sel, b_eff)

    # agreement: 全接続 閾値ゲート(exact) vs 転写(fan-in疎化)
    ref = ((h.astype(np.float64) @ w.numpy().T + b.numpy()[None, :]) > 0).astype(np.int64)
    got = _simulate_lut(h, indices, tts)
    agreement = float((got == ref).mean())

    restore_lut_layer(lut_layer, indices, tt_to_hex(tts))
    return agreement


def _simulate_lut(x: np.ndarray, indices: np.ndarray, tts: list[int]) -> np.ndarray:
    """(B,in) 0/1 配列に IR lut_layer を適用(transcribe_seq._simulate_lut_layer と同規約)。"""
    n = indices.shape[0]
    gathered = x[:, indices]
    weights = (1 << np.arange(n, dtype=np.int64))[None, :, None]
    addr = (gathered * weights).sum(axis=1)
    tt_arr = np.asarray(tts, dtype=np.uint64)
    out = (tt_arr[None, :] >> addr.astype(np.uint64)) & np.uint64(1)
    return out.astype(np.int64)


def transcribe_linear_thermometer(
    lin_weight: torch.Tensor,
    lin_bias: torch.Tensor,
    lut_layer,
    calib_bits: np.ndarray,
    *,
    n_inputs: int,
    bits: int,
    mask: torch.Tensor | None = None,
    thermometer_mode: str = 'distributive',
) -> tuple[float, dict]:
    """量子化済み線形層(d_in→d_out)を **N-bit thermometer** LUT へ転写する。

    1bit sign 転写(transcribe_linear_to_lutlayer)が magnitude を捨てて頭打ちに
    なる課題への直接対処。teacher射影の pre-activation z_i(x)=Σ_j W_ij x_j + b_i を
    昇順しきい値 t_{i,1}<...<t_{i,N} で thermometer 符号化する:

        bit(i, p) = 1[z_i(x) > t_{i,p}] = 1[Σ_j W_ij x_j + (b_i − t_{i,p}) > 0]

    各 bitplane p は「同一重み・バイアスを t_{i,p} だけシフトした閾値ゲート」なので、
    transcribe_linear_to_lutlayer と同じ fan-in疎化 + threshold_truth_tables で
    厳密転写できる(bit=1 は t=0 のみ = sign と同一 → N=1で1bitベースラインを再現)。

    出力レイアウト(feature-major、ThermometerEncoder.forward と同規約):
        列 i*bits + p が neuron i の bitplane p。LogicAttention は q/k で同一手順を
        使うため、q と k の列対応が保たれ XNOR類似度が dim毎に整合する。

    しきい値は teacher の実数 pre-activation を calib データ上で ThermometerEncoder
    に fit して得る(distributive=経験分位点で magnitude 分布に密に配置)。

    Args:
        lin_weight: (out, in) 量子化済み実効重み
        lin_bias: (out,) バイアス
        lut_layer: 書き込み先 LutLayer(out_dim == out*bits であること)
        calib_bits: (B, in) の 0/1 較正データ(post-LN 二値化 hidden)
        n_inputs: LUT fan-in(2/4/6)
        bits: thermometer ビット数 N(1 で sign ベースラインと一致)
        mask: (out, in) bool 残存接続マスク(ternary)。指定時 sparsify_from_mask
        thermometer_mode: 'distributive'(経験分位点) | 'uniform'(等間隔)

    Returns:
        (agreement, info):
            agreement: 較正データ上の per-bit 一致率(全接続 exact thermometer vs
                fan-in疎化後の転写)。1bit sign agreement と同列に比較できる。
            info: {'thresholds_mean': しきい値の平均, 'similarity_fidelity': ...}
                similarity_fidelity は「転写bit列のXNOR自己類似度」が
                「teacher実数 pre-activation の内積類似度」をどれだけ順位保存するか
                (spearman-free の Pearson相関、magnitude回復の直接指標)。
    """
    w = lin_weight.detach().to(torch.float64).cpu()
    b = lin_bias.detach().to(torch.float64).cpu()
    h = np.asarray(calib_bits).astype(np.int64)
    assert set(np.unique(h)).issubset({0, 1}), '較正データは0/1のみ'
    out_dim, _in_dim = w.shape

    # teacher の実数 pre-activation を calib 上で計算 → thermometer しきい値を fit。
    z = torch.as_tensor(h, dtype=torch.float64) @ w.T + b[None, :]  # (B, out)
    if bits == 1 and thermometer_mode == 'sign':
        thresholds = torch.zeros(1, out_dim, dtype=torch.float64)
    else:
        enc = ThermometerEncoder(bits, mode=thermometer_mode).fit(z)
        thresholds = enc.thresholds.to(torch.float64)  # (bits, out)

    # fan-in疎化は bitplane 間で共有(同一重み)。1回だけ計算して各planeで使い回す。
    if mask is not None:
        indices, w_sel, bshift = sparsify_from_mask(w, mask.cpu(), n_inputs, calib_inputs=h)
    else:
        indices, w_sel, bshift = sparsify_fanin(w, n_inputs, calib_inputs=h)
    k_actual = indices.shape[0]
    if k_actual < n_inputs:
        pad = n_inputs - k_actual
        indices = np.vstack([indices, np.zeros((pad, indices.shape[1]), dtype=np.int64)])
        w_sel = torch.cat([w_sel, torch.zeros(w_sel.shape[0], pad, dtype=w_sel.dtype)], dim=1)

    # 各 neuron i の bitplane p を feature-major で並べる。
    full_indices = np.zeros((n_inputs, out_dim * bits), dtype=np.int64)
    full_tts: list[int] = []
    ref = np.zeros((h.shape[0], out_dim * bits), dtype=np.int64)
    for p in range(bits):
        # bias を t_{i,p} だけシフトした閾値ゲート
        b_eff_p = b + bshift - thresholds[p]  # (out,)
        tts_p = threshold_truth_tables(w_sel, b_eff_p)
        for i in range(out_dim):
            col = i * bits + p
            full_indices[:, col] = indices[:, i]
            full_tts.append(tts_p[i])
        # exact thermometer 参照(全接続 pre-activation としきい値)
        ref_p = (z.numpy() > thresholds[p].numpy()[None, :]).astype(np.int64)  # (B, out)
        for i in range(out_dim):
            ref[:, i * bits + p] = ref_p[:, i]

    got = _simulate_lut(h, full_indices, full_tts)  # (B, out*bits)
    agreement = float((got == ref).mean())

    # magnitude回復の直接指標: 転写bit列の XNOR自己類似度 vs teacher内積類似度の相関。
    similarity_fidelity = _xnor_similarity_fidelity(got, z.numpy())

    restore_lut_layer(lut_layer, full_indices, tt_to_hex(full_tts))
    return agreement, {
        'thresholds_mean': float(thresholds.mean()),
        'similarity_fidelity': similarity_fidelity,
    }


def _xnor_similarity_fidelity(bits_bt: np.ndarray, z_real: np.ndarray,
                              max_rows: int = 128) -> float:
    """転写bit列の XNOR類似度が teacher の実数内積類似度をどれだけ保存するかの Pearson相関。

    magnitude回復の直接指標(per-bit agreement は中間しきい値planeを罰し極端plane
    を trivial に報酬するため magnitude gain を捉えられない。これはそれを補う)。

    bits_bt: (B, out*bits) 転写thermometer bit列。z_real: (B, out) teacher pre-activation。
    先頭 max_rows サンプルの **全対** (i<j) について XNOR score(popcount一致)と
    teacher内積を並べ、Pearson相関を返す(1に近いほど magnitude順位が保存)。
    全対・決定的なのでランダム対より安定する。
    """
    b = min(bits_bt.shape[0], max_rows)
    if b < 4:
        return float('nan')
    q = bits_bt[:b].astype(np.float64)
    z = z_real[:b].astype(np.float64)
    # 全対 (i<j) の上三角。XNOR score = Σ_d[q_d q'_d + (1-q_d)(1-q'_d)]
    iu, ju = np.triu_indices(b, k=1)
    xnor = (q[iu] * q[ju] + (1 - q[iu]) * (1 - q[ju])).sum(axis=1)
    dot = (z[iu] * z[ju]).sum(axis=1)
    if xnor.std() < 1e-9 or dot.std() < 1e-9:
        return float('nan')
    return float(np.corrcoef(xnor, dot)[0, 1])


# ============================================================
# モデル構築(FFN転写と1:1対応する identity-decoder な mlp へ差し替え)
# ============================================================

def build_logicgpt_config(args, teacher_n_embd: int) -> LogicGPTConfig:
    config = LogicGPTConfig(
        vocab_size=STUDENT_VOCAB_SIZE,
        d_model=teacher_n_embd,
        n_layers=args.gpt_layers,
        n_heads=args.gpt_heads,
        block_size=63,
        mlp_ratio=args.gpt_mlp_ratio,
        pos_width=max(6, math.ceil(math.log2(64))),
        head_group_factor=args.head_group_factor,
        tau=1.0,
        n_inputs=args.n_inputs,
        parameterization='softmax',
        connections='random',
        merge='xor',
        norm='layernorm',
    )
    if config.d_model < config.pos_width:
        config.pos_width = config.d_model
    return config


def _setup_attention_projections(model: LogicGPT, config: LogicGPTConfig,
                                 qkv_bits: int, vproj_bits: int = 1) -> None:
    """各 block の attention の q/k/v 射影を独立・多bit幅の LutStack へ差し替える。

    (1) share_qk 分離: 既定 share_qk=True では q_proj is k_proj。q/k を独立転写する
        ため別インスタンスにする。
    (2) q/k thermometer 多bit化: 出力幅を d_model → d_model*qkv_bits にする
        (各 teacher射影 neuron を qkv_bits 本の thermometer bitplane で表現)。
        XNOR類似度 score=Σ_d[q_d k_d+(1-q_d)(1-k_d)] は q/k の幅が広いほど magnitude を
        popcount として保持できる。
    (3) v 多bit化(vproj_bits>1): v_proj を 2層 LutStack にする:
            層0: d_model → d_model*vproj_bits(thermometer)
            層1: d_model*vproj_bits → d_model(**圧縮LUT**、残差幅へ戻す)
        層1は転写せず polish で学習する(良init不要な純圧縮)。attention出力は
        d_model幅のまま残差ストリームへ merge できる。vproj_bits=1 なら従来通り
        単層 d_model の 1bit sign v。
        LogicAttention.attend は reshape(...,-1) で出力幅を推論するため d_qk/d_v を
        更新すれば n_heads 分割も整合する(幅 % n_heads == 0 が必要)。

    全アーム(float/bin/ternary/scratch)に同じ差し替えを適用し、attention容量を
    アーム間で揃える。qkv_bits=1 かつ vproj_bits=1 は 1bit sign ベースラインと同一構造。
    """
    from griffith_logic.models.rddlgn import LutStack

    qk_width = config.d_model * qkv_bits
    assert qk_width % config.n_heads == 0, (
        f'd_model*qkv_bits({qk_width}) は n_heads({config.n_heads}) で割り切れる必要がある'
    )
    assert config.d_model % config.n_heads == 0

    def _new_stack(layer_sizes):
        return LutStack(
            config.d_model, layer_sizes,
            n_inputs=config.n_inputs, parameterization=config.parameterization,
            connections=config.connections, selection=config.selection,
        )

    v_sizes = [config.d_model] if vproj_bits <= 1 \
        else [config.d_model * vproj_bits, config.d_model]
    for blk in model.blocks:
        attn = blk.attn
        attn.k_proj = _new_stack([qk_width])
        attn.q_proj = _new_stack([qk_width])
        attn.v_proj = _new_stack(v_sizes)  # vproj_bits>1 なら thermometer+圧縮の2層
        attn.d_qk = qk_width
        attn.d_v = config.d_model  # attend後の出力は常に d_model(残差幅整合)


def replace_block_mlps_for_transcription(model: LogicGPT, config: LogicGPTConfig) -> None:
    """各 block.mlp を transcribe_ffn_block と1:1対応する
    (width=d_model, decoder='identity', depth=2)の LogicMLP へ差し替える
    (convert_attention_ffn_charlm と同一。全アームで構造を揃える)。"""
    for blk in model.blocks:
        blk.mlp = LogicMLP(
            config.d_model,
            spec=LogicSpec(
                encoder='identity', decoder='identity', width=config.d_model,
                n_inputs=config.n_inputs, parameterization=config.parameterization,
                connections=config.connections, depth=2,
            ),
        )


# ============================================================
# teacher量子化ブロックからの転写初期化(QKV射影 + FFN)
# ============================================================

def collect_calib_bits(teacher, calib_windows: torch.Tensor, device):
    """各 block の post-LN hidden(attention入力 ln_1(x)、FFN入力 ln_2(x))を
    閾値二値化(>0)して 0/1 較正データにする。

    QKV射影の入力は ln_1(x)、FFN の入力は ln_2(x)。transcribe.py の隠れ層意味論
    y=1[Wx+b>0] と同じ二値化規約を teacher活性そのものへ適用する。
    """
    teacher.model.eval()
    with torch.no_grad():
        idx = calib_windows.to(device)
        _b, t = idx.shape
        pos = torch.arange(t, dtype=torch.long, device=device)
        x = teacher.model.transformer.wte(idx) + teacher.model.transformer.wpe(pos)
        attn_in, ffn_in = [], []
        for block in teacher.model.transformer.h:
            ln1 = block.ln_1(x)
            attn_in.append((ln1 > 0).to(torch.float64).reshape(-1, ln1.shape[-1]).cpu().numpy())
            x = x + block.attn(ln1)
            ln2 = block.ln_2(x)
            ffn_in.append((ln2 > 0).to(torch.float64).reshape(-1, ln2.shape[-1]).cpu().numpy())
            x = x + block.mlp(ln2)
    return attn_in, ffn_in


def init_from_teacher(model, teacher, qblocks, arm, calib_windows, args, device):
    """各 block の QKV射影 + FFN を量子化teacherから転写初期化する。

    Returns: per-block info 辞書のリスト(agreement群)。
    """
    d = qblocks[0].d
    attn_in, ffn_in = collect_calib_bits(teacher, calib_windows, device)
    infos = []
    n_blocks = min(len(model.blocks), len(teacher.model.transformer.h))
    for i in range(n_blocks):
        blk = model.blocks[i]
        tblk = teacher.model.transformer.h[i]
        qb = qblocks[i]
        w = tblk.attn.c_attn.weight  # (3d, d) 量子化bake済み
        bias = tblk.attn.c_attn.bias
        if bias is None:
            bias = torch.zeros(3 * d, dtype=w.dtype, device=w.device)
        calib = attn_in[i]

        mask_q = qb.masks['q'] if arm == 'ternary' else None
        mask_k = qb.masks['k'] if arm == 'ternary' else None
        mask_v = qb.masks['v'] if arm == 'ternary' else None

        bits = args.qkv_bits
        vbits = args.vproj_bits
        # v: vproj_bits=1 は 1bit sign(単層 d_model)。>1 は thermometer(層0=
        # d_model*vbits)を転写し、層1(圧縮 d_model*vbits→d_model)は polish で学習。
        if vbits <= 1:
            ag_v = transcribe_linear_to_lutlayer(
                w[2 * d:], bias[2 * d:], blk.attn.v_proj.layers[0], calib,
                n_inputs=args.n_inputs, mask=mask_v)
            fid_v = float('nan')
        else:
            ag_v, ivv = transcribe_linear_thermometer(
                w[2 * d:], bias[2 * d:], blk.attn.v_proj.layers[0], calib,
                n_inputs=args.n_inputs, bits=vbits, mask=mask_v,
                thermometer_mode=args.thermometer_mode)
            fid_v = ivv['similarity_fidelity']
        if bits <= 1:
            ag_q = transcribe_linear_to_lutlayer(
                w[:d], bias[:d], blk.attn.q_proj.layers[0], calib,
                n_inputs=args.n_inputs, mask=mask_q)
            ag_k = transcribe_linear_to_lutlayer(
                w[d:2 * d], bias[d:2 * d], blk.attn.k_proj.layers[0], calib,
                n_inputs=args.n_inputs, mask=mask_k)
            fid_q = fid_k = float('nan')
        else:
            ag_q, iq = transcribe_linear_thermometer(
                w[:d], bias[:d], blk.attn.q_proj.layers[0], calib,
                n_inputs=args.n_inputs, bits=bits, mask=mask_q,
                thermometer_mode=args.thermometer_mode)
            ag_k, ik = transcribe_linear_thermometer(
                w[d:2 * d], bias[d:2 * d], blk.attn.k_proj.layers[0], calib,
                n_inputs=args.n_inputs, bits=bits, mask=mask_k,
                thermometer_mode=args.thermometer_mode)
            fid_q, fid_k = iq['similarity_fidelity'], ik['similarity_fidelity']

        # FFN 転写(既存 init_logic_mlp_from_ffn)。ternaryのmaskは transcribe_ffn_block が
        # 未対応(unit部分選択のみ)なので、量子化bake済み重みの符号を閾値二値化する
        # (0マスクは |W|=0 として sparsify_fanin が自然に選外にする)。
        spec = FfnTranscribeSpec(
            n_inputs=args.n_inputs,
            hidden_width=blk.mlp.luts[0].out_dim,
            out_width=blk.mlp.luts[1].out_dim,
            unit_select='l2_norm',
        )
        # transcribe_ffn_block は teacher重み @ calib(CPU)を計算するため CPU で行う
        # (CUDA上の tblk.mlp をそのまま渡すと device 不一致になる。student blk.mlp は
        # restore_lut_layer が layer.weights.device へコピーするので device のままでよい)。
        tmlp_cpu = copy.deepcopy(tblk.mlp).to('cpu')
        ffn_info = init_logic_mlp_from_ffn(blk.mlp, tmlp_cpu, ffn_in[i], spec)
        infos.append({
            'block': i, 'qkv_agree_q': ag_q, 'qkv_agree_k': ag_k, 'qkv_agree_v': ag_v,
            'qkv_fid_q': fid_q, 'qkv_fid_k': fid_k, 'qkv_fid_v': fid_v,
            'ffn_hidden_agree': ffn_info['hidden_agreement'],
            'ffn_output_agree': ffn_info['output_agreement'],
        })
        fid_str = ('' if math.isnan(fid_q)
                   else f' XNOR-sim-fidelity q/k/v={fid_q:.3f}/{fid_k:.3f}/{fid_v:.3f}')
        print(f'  block {i}: QKV agree q/k/v={ag_q:.4f}/{ag_k:.4f}/{ag_v:.4f}{fid_str}  '
              f'FFN hidden/out={ffn_info["hidden_agreement"]:.4f}/'
              f'{ffn_info["output_agreement"]:.4f}')
    return infos


# ============================================================
# 1 arm の学習ループ
# ============================================================

def run_arm(arm, args, base_teacher, train_loader, eval_loader, val_windows, device,
            config, calib_windows, qat_windows):
    torch.manual_seed(args.seed + {'float': 0, 'bin': 1000, 'ternary': 2000, 'scratch': 3000}[arm])
    model = LogicGPT(config).to(device)
    _setup_attention_projections(model, config, args.qkv_bits, args.vproj_bits)
    replace_block_mlps_for_transcription(model, config)
    model.to(device)

    transcribe_infos = None
    src_bpc_hard = float('nan')
    if arm != 'scratch':
        teacher = copy.deepcopy(base_teacher)
        teacher.model.to(device)
        d = config.d_model
        qblocks = [_QuantAttnBlock(b, d, arm, device) for b in teacher.model.transformer.h]
        if arm in ('bin', 'ternary'):
            print(f'  [arm {arm}] full-corpus QAT(epochs={args.qat_epochs}, '
                  f'windows={len(qat_windows)})...')
            _qat_finetune_teacher(
                teacher, qblocks, arm, qat_windows, device,
                epochs=args.qat_epochs, lr=args.qat_lr, batch_size=args.batch_size)
        _bake_quant_into_teacher(qblocks, arm)
        src_bpc_hard = evaluate_teacher_bpc(teacher, val_windows, device)
        print(f'  [arm {arm}] 量子化source(bake後) teacher BPC={src_bpc_hard:.4f}')
        print(f'  [arm {arm}] teacher QKV/FFN転写中(blocks={config.n_layers})...')
        transcribe_infos = init_from_teacher(model, teacher, qblocks, arm, calib_windows, args, device)
        del teacher

    use_kd = args.polish_kd
    if use_kd:
        _kd_raw = make_seq_kd_loss(
            alpha=args.distill_alpha, temperature=args.distill_temperature,
            standardize=args.distill_standardize, pad_id=PAD_ID)

        def loss_fn_kd(student_logits, target_ids, teacher_logits, step=0):
            return _kd_raw(student_logits, target_ids, teacher_logits)
        loss_fn_ce = None
    else:
        loss_fn_kd = None
        loss_fn_ce = make_pure_ce_seq_loss(pad_id=PAD_ID)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
        betas=(0.9, 0.999), eps=1e-8)
    emb_reg_ramp = (args.emb_reg_ramp_start, args.emb_reg_ramp_end, args.emb_reg_max_weight)

    best_bpc_soft = float('inf')
    best_hard_at_best_soft = float('nan')
    best_agree_at_best_soft = float('nan')
    best_bpc_hard = float('inf')
    history = []

    def load_n(loader, n):
        i = 0
        while i < n:
            for batch in loader:
                yield batch
                i += 1
                if i == n:
                    return

    kd_teacher = base_teacher if use_kd else None
    pbar = tqdm(enumerate(load_n(train_loader, args.num_iterations)),
                desc=f'arm {arm}', total=args.num_iterations)
    t0 = time.time()
    for step, window in pbar:
        if use_kd:
            total_loss, _m, _e, _ = train_step_decoder_only(
                model, kd_teacher, window, optimizer, step, loss_fn_kd, device, emb_reg_ramp)
        else:
            total_loss, _m, _e = train_step_decoder_only_scratch(
                model, window, optimizer, step, loss_fn_ce, device, emb_reg_ramp)
        pbar.set_postfix(loss=f'{total_loss:.4f}')

        if (step + 1) % args.eval_freq == 0 or step == args.num_iterations - 1:
            bpc_soft = evaluate_bpc(model, 'logicgpt', val_windows, device, use_soft=True)
            bpc_hard = evaluate_bpc(model, 'logicgpt', val_windows, device, use_soft=False)
            agree_hard = evaluate_teacher_agreement(
                model, 'logicgpt', eval_loader, base_teacher, device, use_soft=False)
            best_bpc_hard = min(best_bpc_hard, bpc_hard)
            # soft最良で重み選択(memory: soft-best-not-hard-best)。その時点のhard実体を保持。
            if bpc_soft < best_bpc_soft:
                best_bpc_soft = bpc_soft
                best_hard_at_best_soft = bpc_hard
                best_agree_at_best_soft = agree_hard
            history.append({'step': step + 1, 'bpc_soft': bpc_soft, 'bpc_hard': bpc_hard,
                            'agree_hard': agree_hard})
            print(f'\n  [arm {arm}][iter {step+1}/{args.num_iterations}] loss={total_loss:.4f} '
                  f'BPC(soft)={bpc_soft:.4f} BPC(hard)={bpc_hard:.4f} agree(hard)={agree_hard:.4f}')
            if args.use_wandb:
                log_metrics({
                    f'arm_{arm}/loss': total_loss, f'arm_{arm}/bpc_soft': bpc_soft,
                    f'arm_{arm}/bpc_hard': bpc_hard, f'arm_{arm}/agree_hard': agree_hard,
                }, step=step + 1)

    seconds = time.time() - t0
    return {
        'arm': arm,
        'source_bpc_hard': src_bpc_hard,
        'best_bpc_soft': best_bpc_soft,
        'hard_at_best_soft': best_hard_at_best_soft,
        'agree_at_best_soft': best_agree_at_best_soft,
        'best_bpc_hard': best_bpc_hard,
        'final_bpc_soft': history[-1]['bpc_soft'] if history else float('nan'),
        'final_bpc_hard': history[-1]['bpc_hard'] if history else float('nan'),
        'final_agree_hard': history[-1]['agree_hard'] if history else float('nan'),
        'seconds': round(seconds, 2),
        'history': history,
        'transcribe_infos': transcribe_infos,
    }


# ============================================================
# メイン
# ============================================================

def _auto_device(arg):
    if arg is not None:
        return arg
    if torch.cuda.is_available():
        return 'cuda'
    return 'cpu'


def main():
    parser = argparse.ArgumentParser(
        description='量子化QKV/FFN → LogicAttention+FFN(DLGN)転写 vs bin/float/scratch(char-LM HARD BPC)')
    parser.add_argument('--teacher', type=str,
                        default='hf-nanogpt-char:sosier/nanoGPT-shakespeare-char-tied-weights')
    parser.add_argument('--corpus', type=str,
                        default='data-charlm/corpus__MinGPT-_500x64_T1.0_s42.pt')
    parser.add_argument('--eval-corpus', type=str,
                        default='data-charlm/corpus__MinGPT-_500x64_T1.0_s42_eval.pt')

    parser.add_argument('--gpt-layers', type=int, default=1)
    parser.add_argument('--gpt-heads', type=int, default=1)
    parser.add_argument('--gpt-mlp-ratio', type=int, default=1)
    parser.add_argument('--head-group-factor', type=int, default=4)
    parser.add_argument('--n-inputs', type=int, default=2, choices=[2, 4, 6])
    parser.add_argument('--qkv-bits', type=int, default=1,
                        help='q/k射影の thermometer ビット数 N(1=sign 1bitベースライン、'
                             '2/3/4 で magnitude を多bit保持)')
    parser.add_argument('--vproj-bits', type=int, default=1,
                        help='v射影の thermometer ビット数(>1で d_model*N thermometer→'
                             'd_model 圧縮LUT。1=従来の1bit sign v)')
    parser.add_argument('--thermometer-mode', type=str, default='distributive',
                        choices=['distributive', 'uniform'],
                        help='thermometer しきい値の決め方(distributive=経験分位点)')

    parser.add_argument('--distill-alpha', type=float, default=0.25)
    parser.add_argument('--distill-temperature', type=float, default=4.0)
    parser.add_argument('--distill-standardize', dest='distill_standardize',
                        action='store_true', default=True)
    parser.add_argument('--no-distill-standardize', dest='distill_standardize',
                        action='store_false')
    parser.add_argument('--polish-kd', action='store_true',
                        help='soft polish に系列KDを混ぜる(既定off=純CE)')

    parser.add_argument('--qat-epochs', type=int, default=2,
                        help='量子化aware fine-tune(STE)のepoch数(0で無効=PTQ)')
    parser.add_argument('--qat-lr', type=float, default=1e-3)

    parser.add_argument('--lr', type=float, default=0.05)
    parser.add_argument('--weight-decay', type=float, default=0.001)
    parser.add_argument('-bs', '--batch-size', type=int, default=16)
    parser.add_argument('-ni', '--num-iterations', type=int, default=200)
    parser.add_argument('-ef', '--eval-freq', type=int, default=50)
    parser.add_argument('--eval-max-chars', type=int, default=20_000)
    parser.add_argument('--emb-reg-ramp-start', type=int, default=50)
    parser.add_argument('--emb-reg-ramp-end', type=int, default=2000)
    parser.add_argument('--emb-reg-max-weight', type=float, default=0.1)
    parser.add_argument('--n-calib-windows', type=int, default=64,
                        help='転写の較正データに使う train window 数')
    parser.add_argument('--n-qat-windows', type=int, default=512,
                        help='full-corpus QAT に使う train window 数(較正とは別。'
                             '大きいほど量子化source回復が安定)')

    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--quick', action='store_true')
    parser.add_argument('--no-wandb', action='store_true')
    args = parser.parse_args()

    if args.quick:
        args.num_iterations = min(args.num_iterations, 40)
        args.eval_freq = min(args.eval_freq, 20)
        args.eval_max_chars = min(args.eval_max_chars, 3000)
        args.batch_size = min(args.batch_size, 8)
        args.n_calib_windows = min(args.n_calib_windows, 24)
        args.n_qat_windows = min(args.n_qat_windows, 48)
        args.qat_epochs = min(args.qat_epochs, 1)
        args.emb_reg_ramp_start = 20
        args.emb_reg_ramp_end = 200
        if args.device is None:
            args.device = 'cpu'

    args.device = _auto_device(args.device)
    print(f'デバイス: {args.device}')
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    print('teacherをロード中...')
    teacher = resolve_text_teacher(args.teacher, device=args.device)
    teacher_n_embd = teacher.cfg['n_embd']
    print(f'  teacher fingerprint={teacher.fingerprint}  n_embd={teacher_n_embd}  '
          f'n_layer={teacher.cfg["n_layer"]}')

    config = build_logicgpt_config(args, teacher_n_embd)
    print(f'LogicGPT config: d_model={config.d_model} n_layers={config.n_layers} '
          f'n_heads={config.n_heads} n_inputs={config.n_inputs}')

    corpus_path = resolve_glob(args.corpus)
    eval_corpus_path = resolve_glob(args.eval_corpus)
    print(f'訓練コーパス: {corpus_path}')
    print(f'評価コーパス: {eval_corpus_path}')

    train_set = TeacherCorpusDataset(corpus_path, mode='decoder_only')
    eval_set = TeacherCorpusDataset(eval_corpus_path, mode='decoder_only')
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              drop_last=True, collate_fn=collate_raw_window)
    eval_loader = DataLoader(eval_set, batch_size=min(args.batch_size, len(eval_set)),
                             shuffle=False, drop_last=False, collate_fn=collate_raw_window)

    calib_windows = torch.stack(
        [train_set[i] for i in range(min(args.n_calib_windows, len(train_set)))])
    # full-corpus QAT 用の別 window 集合(較正より大きく取る。train全体からサンプル)。
    n_qat = min(args.n_qat_windows, len(train_set))
    qat_windows = torch.stack([train_set[i] for i in range(n_qat)])
    print(f'較正windows={len(calib_windows)}  QAT windows={len(qat_windows)}  '
          f'(train_set={len(train_set)})')
    val_windows = load_tiny_shakespeare_val(max_chars=args.eval_max_chars)

    print('teacher自身(float)のBPCを計算中...')
    teacher_bpc = evaluate_teacher_bpc(teacher, val_windows, args.device)
    print(f'  teacher BPC = {teacher_bpc:.4f}  (uniform anchor = {math.log2(65):.4f})')

    args.use_wandb = False
    if not args.no_wandb:
        args.use_wandb = init_wandb(
            run_name=(f'ternary_attn_d{config.d_model}_L{config.n_layers}'
                      f'_qkv{args.qkv_bits}_v{args.vproj_bits}_ni{args.num_iterations}'),
            config_dict={**vars(args), **asdict(config), 'teacher_bpc': teacher_bpc},
            device=args.device,
            tags=['convert', 'ternary-dwn', 'attention', 'quantized-qkv-ffn', 'charlm', 'logicgpt'],
            group='ternary-dwn')

    results = {}
    for arm in ARMS:
        print(f'\n{"#" * 20} arm {arm} {"#" * 20}')
        results[arm] = run_arm(arm, args, teacher, train_loader, eval_loader,
                               val_windows, args.device, config, calib_windows,
                               qat_windows)

    # ---- 結果表 ----
    print('\n' + '=' * 104)
    print(f'量子化QKV/FFN → LogicAttention+FFN 転写 — 結果'
          f'(L={config.n_layers} q/k-bits={args.qkv_bits} v-bits={args.vproj_bits}, '
          f'HARD BPC 小さいほど良い)')
    print('=' * 104)
    print('| arm | source BPC(hard) | best BPC(soft) | hard@best-soft | agree@best-soft | '
          'best BPC(hard) | sec |')
    print('|' + '---|' * 7)
    for arm in ARMS:
        r = results[arm]
        print(f"| {arm} | {r['source_bpc_hard']:.4f} | {r['best_bpc_soft']:.4f} | "
              f"{r['hard_at_best_soft']:.4f} | {r['agree_at_best_soft']:.4f} | "
              f"{r['best_bpc_hard']:.4f} | {r['seconds']:.1f} |")
    print(f'\nteacher BPC(float) = {teacher_bpc:.4f}  (uniform anchor = {math.log2(65):.4f})')

    # soft最良時のhard実体で arm比較(memory準拠)
    def h(arm):
        return results[arm]['hard_at_best_soft']
    lift_vs_scratch = h('scratch') - h('ternary')
    lift_vs_bin = h('bin') - h('ternary')
    lift_vs_float = h('float') - h('ternary')
    print('\n--- 派生指標(hard@best-soft のBPC差。lift = 相手 − ternary、正なら ternary が改善) ---')
    print(f'lift(ternary vs scratch) : {lift_vs_scratch:+.4f}  '
          f"→ {'支持(転写起点>scratch)' if lift_vs_scratch > 0 else '不支持'}")
    print(f'lift(ternary vs bin)     : {lift_vs_bin:+.4f}  (0マスクの純効果)')
    print(f'lift(ternary vs float)   : {lift_vs_float:+.4f}')

    print(f'\n--- QKV転写忠実度(thermometer bits={args.qkv_bits}。1bit比較: '
          f'前回 QKV~0.67-0.74) ---')
    for arm in ('float', 'bin', 'ternary'):
        infos = results[arm]['transcribe_infos']
        if infos:
            aq = np.mean([i['qkv_agree_q'] for i in infos])
            ak = np.mean([i['qkv_agree_k'] for i in infos])
            av = np.mean([i['qkv_agree_v'] for i in infos])
            qkv = np.mean([aq, ak, av])
            ffn = np.mean([i['ffn_output_agree'] for i in infos if i['ffn_output_agree'] is not None])
            fid_vals = [i['qkv_fid_q'] for i in infos] + [i['qkv_fid_k'] for i in infos] \
                + [i['qkv_fid_v'] for i in infos]
            fid_vals = [v for v in fid_vals if not (isinstance(v, float) and math.isnan(v))]
            fid = np.mean(fid_vals) if fid_vals else float('nan')
            fid_str = '' if math.isnan(fid) else f'  XNOR-sim-fidelity={fid:.3f}'
            print(f'[{arm}] agree q/k/v={aq:.4f}/{ak:.4f}/{av:.4f} (平均{qkv:.4f})  '
                  f'FFN出力={ffn:.4f}{fid_str}')

    if args.use_wandb:
        log_metrics({
            'summary/teacher_bpc': teacher_bpc,
            'summary/ternary_source_bpc': results['ternary']['source_bpc_hard'],
            'summary/ternary_hard_at_best_soft': h('ternary'),
            'summary/bin_hard_at_best_soft': h('bin'),
            'summary/float_hard_at_best_soft': h('float'),
            'summary/scratch_hard_at_best_soft': h('scratch'),
            'summary/lift_vs_scratch': lift_vs_scratch,
            'summary/lift_vs_bin': lift_vs_bin,
            'summary/lift_vs_float': lift_vs_float,
        })
        wandb_finish()


if __name__ == '__main__':
    main()
