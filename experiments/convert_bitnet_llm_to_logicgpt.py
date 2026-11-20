"""1bit/1.58bit量子化(BitNet-style)多層char-LM → LogicGPT(多層)丸ごと転写。

このスクリプトは前タスク experiments/convert_ternary_attention_to_logic.py(単層 attention+FFN
ブロックの量子化転写)を **多層 nanoGPT系 char-LM teacher → 多層 LogicGPT** へ拡張する。

正直な範囲宣言(重要):
    * 公開HFの BitNet b1.58-2B / Bonsai-PrunedPhi-1.8B は2B級で 12GB GPU・本環境
      (CPU/MPS スモーク)に載らない(サイズ制約)。→ **小規模 BitNet-style char-LM
      (多層 nanoGPT系)をローカルで学習し、全線形層(QKV射影・FFN)を ternary量子化
      (BitNet b1.58 absmean)+QATで回復して teacher とし、それを LogicGPT(多層)へ
      丸ごと転写**する feasibility を作る。HF実重みのロードはサイズ制約を明示して
      **未実施**(_load_hf_bitnet_note を参照。コードとしては入口を残しコメントで理由を記す)。
    * softmax/argmax は論理プリミティブで表現できない(blocks/attention.py docstring)。
      よって attention 核は既存 LogicAttention(soft訓練→hard推論)をそのまま使い、
      転写対象は **埋め込み(sign整合)/ QKV射影 / FFN / (head は random+polish)** に限る。

パイプライン(1 arm、多層):
    1. ローカル teacher: 小さな multi-layer nanoGPT char-LM(shakespeare 65-vocab)を
       tiny_shakespeare でローカル学習(--quick では極短)。既存 _MinGPT を流用。
    2. 全ブロックの c_attn(→q/k/v)と FFN(c_fc/c_proj)を arm別に量子化(BitNet
       absmean ternary、0マスク保持)+QAT(STE forward、系列CEで回復)。
    3. bake し、各ブロックの QKV射影→単層閾値二値化LUT、FFN→2層LUTへ転写。
       埋め込みは teacher wte を sign整合して LogicEmbedding へ転写。head は
       構造不一致(GroupSum head)のため random+polish(下限、honest)。
    4. LogicAttention は soft訓練/hard推論。soft polish(純CE、--polish-kd で系列KD)。
       soft最良で重み選択(memory: soft-best-not-hard-best)。

アーム(4本):
    float   : 枝刈りなしfloat重みを閾値二値化転写(基準)
    bin     : 1bit sign 量子化(密)+QAT回復
    ternary : 1.58bit absmean 量子化(0マスク=疎)+QAT回復(本命)
    scratch : ランダム初期化LogicGPT(転写なし)を同容量・同学習でpolish(下限)

指標:
    teacher BPC(float) / 量子化source BPC(hard) / QKV・FFN・埋め込み block別 agreement /
    soft polish後 best BPC(soft)で重み選択 → その時点の hard BPC / teacher agreement /
    lift(ternary−scratch, ternary−bin, ternary−float)。
    --sweep-layers で層数を振り、多層の誤差蓄積(hard BPC 劣化・agreement 減衰)を測る。

使用例:
    uv run python experiments/convert_bitnet_llm_to_logicgpt.py --quick --no-wandb
    uv run python experiments/convert_bitnet_llm_to_logicgpt.py \\
        --sweep-layers 1,2 -ni 8000 -ef 500 --teacher-steps 3000
"""

from __future__ import annotations

import argparse
import copy
import math
import os
import random
import sys
import time
from datetime import datetime, timedelta, timezone

JST = timezone(timedelta(hours=9))

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from data.char_corpus import (  # noqa: E402
    PAD_ID,
    STUDENT_VOCAB_SIZE,
    TeacherCorpusDataset,
    load_tiny_shakespeare_val,
)
from env_utils import load_dotenv  # noqa: E402
from train_kd_charlm_rddlgn import (  # noqa: E402
    collate_raw_window,
    evaluate_bpc,
    evaluate_teacher_agreement,
    evaluate_teacher_bpc,
    make_pure_ce_seq_loss,
    resolve_glob,
    train_step_decoder_only,
    train_step_decoder_only_scratch,
)
from wandb_utils import finish as wandb_finish  # noqa: E402
from wandb_utils import init_wandb, log_metrics  # noqa: E402

from griffith_logic.convert.binarize import quantize_ternary  # noqa: E402
from griffith_logic.convert.distill import make_seq_kd_loss  # noqa: E402
from griffith_logic.convert.text_teacher import (  # noqa: E402
    _SHAKESPEARE_ITOS,
    _SHAKESPEARE_STOI,
    SHAKESPEARE_CHAR_VOCAB,
    _auto_fingerprint,
    _MinGPT,
    encode_shakespeare_char,
)
from griffith_logic.convert.transcribe import (  # noqa: E402
    _final_layer_nodes,
    sparsify_fanin,
    sparsify_from_mask,
    threshold_truth_tables,
)
from griffith_logic.convert.transcribe_seq import (  # noqa: E402
    FfnTranscribeSpec,
    init_logic_mlp_from_ffn,
)
from griffith_logic.group_sum import GroupSum  # noqa: E402
from griffith_logic.hybrid.policy import LogicSpec  # noqa: E402
from griffith_logic.ir.serialize import tt_to_hex  # noqa: E402
from griffith_logic.ir.weights import restore_lut_layer  # noqa: E402
from griffith_logic.layers.logic_linear import LogicMLP  # noqa: E402
from griffith_logic.models.logic_gpt import LogicGPT, LogicGPTConfig  # noqa: E402
from griffith_logic.models.rddlgn import LutStack  # noqa: E402

load_dotenv()

ARMS = ('float', 'bin', 'ternary', 'scratch')


def _now_jst() -> str:
    return datetime.now(JST).strftime('%Y-%m-%d %H:%M:%S JST')


# ============================================================
# HF実重み(BitNet b1.58-2B / Bonsai)ロードの入口(サイズ制約で未実施)
# ============================================================

def _load_hf_bitnet_note() -> None:
    """HF公開 BitNet/Bonsai 実重みのロードは本環境では未実施(意図的)。

    理由(サイズ制約):
        - microsoft/bitnet-b1.58-2B-4T / HF1BitLLM/Llama3-8B-1.58 等は 1.5B〜8B級。
          bfloat16 で 3〜16GB、本スモーク環境(CPU/MPS、12GB GPU上限)に載らない。
        - Bonsai-PrunedPhi-1.8B も同様に 1.8B 級で本環境外。
        - これらは AutoModelForCausalLM + trust_remote_code(BitLinear カスタム層)を
          要し、char-LM 語彙とも不整合(subword tokenizer)。転写パイプラインは
          「線形層を absmean ternary 量子化 → 閾値二値化LUT」という構造にのみ依存する
          ため、小規模 char-LM でも feasibility は同型で示せる。

    実重みを通す場合の入口(将来、GPUで):
        from transformers import AutoModelForCausalLM
        m = AutoModelForCausalLM.from_pretrained(
            'microsoft/bitnet-b1.58-2B-4T', trust_remote_code=True)
        # 各 decoder layer の self_attn.{q,k,v}_proj / mlp.{gate,up,down}_proj の
        # BitLinear 重みは既に {-1,0,+1}×scale。quantize_ternary で mask を復元し、
        # 本スクリプトの transcribe_linear_to_lutlayer / init_logic_mlp_from_ffn を
        # そのまま適用できる(語彙・block_size を LogicGPTConfig に合わせる)。
    """
    return None


# ============================================================
# ローカル multi-layer char-LM teacher(BitNet-style 量子化の起点)
# ============================================================

class LocalCharLMTeacher:
    """tiny_shakespeare でローカル学習する小型 multi-layer nanoGPT char-LM teacher。

    NanoGPTCharTeacher と同じインターフェース(`.model`(_MinGPT), `.cfg`,
    `.fingerprint`, `next_token_logits`)を満たすので、既存の量子化・転写・評価
    パイプラインへそのまま渡せる。HF ネットワークに依存しない(重み制約回避)。

    shakespeare 65-vocab を使うため、生成される logits は data-charlm コーパス /
    load_tiny_shakespeare_val の window と直接整列する。
    """

    def __init__(self, *, n_layer: int, n_embd: int, n_head: int,
                 block_size: int = 64, device: str = 'cpu'):
        cfg = {
            'n_layer': int(n_layer),
            'n_head': int(n_head),
            'n_embd': int(n_embd),
            'block_size': int(block_size),
            'vocab_size': 65,
            'bias': True,
        }
        self.cfg = cfg
        self.device = torch.device(device)
        self.model = _MinGPT(cfg).to(self.device)
        self.vocab_size = 65
        self.block_size = int(block_size)
        self.stoi = dict(_SHAKESPEARE_STOI)
        self.itos = dict(_SHAKESPEARE_ITOS)
        self.fingerprint = _auto_fingerprint(self.model)

    def train_local(self, *, steps: int, lr: float, batch_size: int,
                    seq_len: int, seed: int = 0) -> float:
        """tiny_shakespeare の train prefix でローカル学習する(data-free 主張は
        student 側のみ。teacher 自体はテキストで学習してよい)。

        val とは disjoint な先頭側テキストを使う(末尾は評価予約)。
        Returns: 最終学習損失(nats/char)。
        """
        cache = os.path.join(os.path.dirname(__file__), '..', 'data-charlm',
                             'tiny_shakespeare_input.txt')
        with open(cache, encoding='utf-8') as f:
            text = f.read()
        vocab = set(SHAKESPEARE_CHAR_VOCAB)
        # 末尾 20k 文字は評価に予約(load_tiny_shakespeare_val の既定 max_chars と整合的
        # に disjoint を保つ。ここでは安全側に先頭側のみ使う)。
        train_text = ''.join(ch for ch in text[:-20_000] if ch in vocab)
        ids = torch.tensor(encode_shakespeare_char(train_text), dtype=torch.long)

        self.model.train()
        opt = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=0.0)
        g = torch.Generator().manual_seed(seed)
        n = ids.shape[0]
        last = float('nan')
        t0 = time.time()
        for step in range(steps):
            starts = torch.randint(0, n - seq_len - 1, (batch_size,), generator=g)
            batch = torch.stack([ids[s:s + seq_len + 1] for s in starts]).to(self.device)
            inp, tgt = batch[:, :-1], batch[:, 1:]
            logits = self.model(inp)
            loss = F.cross_entropy(logits.reshape(-1, 65), tgt.reshape(-1))
            opt.zero_grad()
            loss.backward()
            opt.step()
            last = float(loss.item())
            if steps >= 10 and (step + 1) % max(1, steps // 5) == 0:
                el = time.time() - t0
                eta = el / (step + 1) * (steps - step - 1)
                print(f'    teacher学習 [{step+1}/{steps}] loss={last:.4f} '
                      f'({last / math.log(2):.3f} bpc) ETA={eta:.0f}s')
        self.model.eval()
        self.fingerprint = _auto_fingerprint(self.model)
        return last

    @torch.no_grad()
    def next_token_logits(self, tokens: torch.Tensor) -> torch.Tensor:
        x = tokens.to(self.device)
        return self.model(x).detach().to('cpu').float()


# ============================================================
# 量子化ユーティリティ(BitNet b1.58 absmean / STE。前タスクと同一意味論)
# ============================================================

def _quantize_weight(w: torch.Tensor, arm: str, mask: torch.Tensor,
                     eps: float = 1e-5) -> torch.Tensor:
    """arm別に量子化した実効重みを返す(bake用。枝刈りmask外は0)。

    bin: sign × alpha(mask内平均|W|)。ternary: absmean三値化(per-channel scale)。
    float: 恒等。scale/alpha は mask 内平均で計算(枝刈り0が平均を薄めない)。
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


class _QuantAttnBlock:
    """1 teacher block の量子化対象線形層(q/k/v/FFN)を束ねるハンドル。"""

    def __init__(self, teacher_block, d_model: int, arm: str, device: str):
        self.blk = teacher_block
        self.d = d_model
        self.arm = arm
        w = teacher_block.attn.c_attn.weight.detach()  # (3d, d)
        self.masks = {
            'q': _linear_mask(w[:d_model], arm).to(device),
            'k': _linear_mask(w[d_model:2 * d_model], arm).to(device),
            'v': _linear_mask(w[2 * d_model:], arm).to(device),
            'c_fc': _linear_mask(teacher_block.mlp.c_fc.weight.detach(), arm).to(device),
            'c_proj': _linear_mask(teacher_block.mlp.c_proj.weight.detach(), arm).to(device),
        }


def _bake_quant_into_teacher(qblocks: list[_QuantAttnBlock], arm: str) -> None:
    """各 teacher block の c_attn / c_fc / c_proj を量子化決定値で bake する。"""
    if arm in ('float', 'scratch'):
        return
    for qb in qblocks:
        d = qb.d
        with torch.no_grad():
            w = qb.blk.attn.c_attn.weight
            for name, sl in (('q', slice(0, d)), ('k', slice(d, 2 * d)),
                             ('v', slice(2 * d, 3 * d))):
                w[sl].copy_(_quantize_weight(w[sl], arm, qb.masks[name]).detach())
            qb.blk.mlp.c_fc.weight.copy_(
                _quantize_weight(qb.blk.mlp.c_fc.weight, arm, qb.masks['c_fc']).detach())
            qb.blk.mlp.c_proj.weight.copy_(
                _quantize_weight(qb.blk.mlp.c_proj.weight, arm, qb.masks['c_proj']).detach())


def _teacher_forward_with_quant(model, idx, qblocks, d):
    """nanoGPT forward を c_attn/FFN の重みを STE量子化値(qb._wq_*)で差し替えて実行。"""
    b, t = idx.shape
    device = idx.device
    pos = torch.arange(t, dtype=torch.long, device=device)
    x = model.transformer.wte(idx) + model.transformer.wpe(pos)
    qb_by_block = {id(qb.blk): qb for qb in qblocks}
    for block in model.transformer.h:
        qb = qb_by_block.get(id(block))
        h = block.ln_1(x)
        if qb is not None:
            qkv = F.linear(h, qb._wq_attn, block.attn.c_attn.bias)
        else:
            qkv = block.attn.c_attn(h)
        q, k, v = qkv.split(block.attn.n_embd, dim=2)
        nh, c = block.attn.n_head, block.attn.n_embd
        q = q.view(b, t, nh, c // nh).transpose(1, 2)
        k = k.view(b, t, nh, c // nh).transpose(1, 2)
        v = v.view(b, t, nh, c // nh).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(b, t, c)
        x = x + block.attn.c_proj(y)
        hf = block.ln_2(x)
        if qb is not None:
            mid = F.gelu(F.linear(hf, qb._wq_fc, block.mlp.c_fc.bias))
            x = x + F.linear(mid, qb._wq_proj, block.mlp.c_proj.bias)
        else:
            x = x + block.mlp(hf)
    x = model.transformer.ln_f(x)
    return model.lm_head(x)


def _qat_finetune_teacher(teacher, qblocks, arm, windows, device, *, epochs, lr, batch_size):
    """量子化aware fine-tune: 毎forwardで c_attn/FFN重みをSTE量子化しつつ系列CEで回復。"""
    if arm in ('float', 'scratch') or epochs <= 0:
        return
    model = teacher.model
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n = len(windows)
    d = qblocks[0].d
    for _ep in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            batch = windows[idx].to(device)
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
            target = batch[:, 1:].reshape(-1)
            loss = F.cross_entropy(
                logits[:, :-1, :].reshape(-1, logits.shape[-1]), target)
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
    model.eval()


# ============================================================
# 転写: QKV射影の単層閾値二値化LUT
# ============================================================

def _simulate_lut(x: np.ndarray, indices: np.ndarray, tts: list[int]) -> np.ndarray:
    """(B,in) 0/1 配列に IR lut_layer を適用。"""
    n = indices.shape[0]
    gathered = x[:, indices]
    weights = (1 << np.arange(n, dtype=np.int64))[None, :, None]
    addr = (gathered * weights).sum(axis=1)
    tt_arr = np.asarray(tts, dtype=np.uint64)
    out = (tt_arr[None, :] >> addr.astype(np.uint64)) & np.uint64(1)
    return out.astype(np.int64)


def transcribe_linear_to_lutlayer(
    lin_weight: torch.Tensor,
    lin_bias: torch.Tensor,
    lut_layer,
    calib_bits: np.ndarray,
    *,
    n_inputs: int,
    mask: torch.Tensor | None = None,
) -> float:
    """量子化済み線形層(d_in→d_out)を単層閾値二値化LUTへ転写し LutLayer へ書き込む。

    y_i = 1[Σ_j W_ij x_j + b_i > 0] を fan-in疎化(mask尊重 or |W|上位)して真理値表化。
    Returns: 較正データ上の agreement(閾値ゲート厳密値 vs fan-in疎化後の一致率)。
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

    ref = ((h.astype(np.float64) @ w.numpy().T + b.numpy()[None, :]) > 0).astype(np.int64)
    got = _simulate_lut(h, indices, tts)
    agreement = float((got == ref).mean())

    restore_lut_layer(lut_layer, indices, tt_to_hex(tts))
    return agreement


# ============================================================
# 転写: 埋め込み(teacher wte を sign整合で LogicEmbedding へ)
# ============================================================

def transcribe_embedding(model: LogicGPT, teacher, *, margin: float = 4.0) -> float:
    """teacher の wte(vocab×n_embd)の符号を LogicEmbedding へ転写する。

    LogicEmbedding は sigmoid(e)>0.5 ⟺ e>0 で {0,1} 化するので、teacher の
    wte 行の各成分の符号ビット b_ij = 1[wte_ij > 0] を student embedding.weight へ
    ±margin で焼く(sigmoid(+margin)>0.5、sigmoid(-margin)<0.5)。

    ただし LogicEmbedding は d_model 幅、teacher wte も n_embd=d_model 幅で1:1整合。
    student vocab は 67(PAD/BOS 追加)なので teacher の 65 行のみ転写し、残り2行は
    そのまま(random init)。

    Returns: teacher wte 符号ビットと student 二値化埋め込みの一致率(整合の代理指標)。
    """
    wte = teacher.model.transformer.wte.weight.detach().cpu()  # (65, d_model)
    v_t, d = wte.shape
    signs = (wte > 0).to(torch.float32)  # (65, d) 0/1
    with torch.no_grad():
        emb = model.embedding.embedding.weight  # (67, d_model)
        emb[:v_t].copy_(((signs * 2.0 - 1.0) * margin).to(emb.dtype).to(emb.device))
    # student側の二値化(sigmoid>0.5 ⟺ weight>0)
    got = (emb[:v_t].detach().cpu() > 0).to(torch.float32)
    return float((got == signs).float().mean().item())


# ============================================================
# 転写: head(teacher lm_head → LutStack最終層 + GroupSum、argmax厳密保存)
# ============================================================

@torch.no_grad()
def student_final_hidden_bits(model: LogicGPT, calib_windows: torch.Tensor,
                              device) -> np.ndarray:
    """student の hard forward で最終ブロック出力(二値hidden)を較正データとして収集する。

    head の入力は各位置の最終ブロック出力 x[B,S,d_model](全経路[0,1]論理域)。
    hard 推論(eval mode = Heaviside)で走らせて {0,1} 化した (B*S, d_model) を返す。
    domain整合: teacher float hidden でなく **student自身が実際に head へ渡す二値hidden**
    を較正に使う(conv/RNN 転写と同じ方針。teacher/student の分布差を吸収)。
    """
    was_training = model.training
    model.eval()
    # 訓練の input と同じく先頭 block_size トークンのみ使う(pos 表は block_size 幅)。
    idx = calib_windows[:, :model.config.block_size].to(device)
    b, s = idx.shape
    x = model.embedding(idx)
    pos = model.pos[:s].unsqueeze(0).to(x.dtype)
    x = x + pos - 2.0 * x * pos
    for blk in model.blocks:
        x = blk(x)
    bits = (x.reshape(b * s, model.config.d_model) > 0.5).to(torch.int64).cpu().numpy()
    model.train(mode=was_training)
    return bits


def _fit_domain_head(calib_bits: np.ndarray, teacher_logits: np.ndarray,
                     *, ridge: float = 1.0) -> tuple[torch.Tensor, torch.Tensor]:
    """student二値hidden → teacher logits を最小二乗(ridge)で近似する線形head を解く。

    domain整合(重要): teacher lm_head の重みは teacher の float `ln_f` hidden 上で
    定義されており、student が実際に head へ渡す二値hidden とは分布が違う。そのまま
    転写すると argmax がずれる(head_agreement 低下)。ここで student二値hidden h と
    teacher logits z の対を較正データとして W,b を解き直す(z ≈ W h + b)ことで、
    転写後の argmax が teacher の next-char 予測に整合する head を得る。

    Args:
        calib_bits: (B, d) の 0/1 student二値hidden
        teacher_logits: (B, vocab_t) の teacher next-char logits(同じ較正位置)
        ridge: L2正則化(H^T H + ridge I の条件数改善)

    Returns:
        (W (vocab_t, d) float64, b (vocab_t,) float64)  — closed-form ridge解
    """
    h = torch.as_tensor(np.asarray(calib_bits), dtype=torch.float64)   # (B, d)
    z = torch.as_tensor(np.asarray(teacher_logits), dtype=torch.float64)  # (B, vocab_t)
    b_n, d = h.shape
    hb = torch.cat([h, torch.ones(b_n, 1, dtype=torch.float64)], dim=1)  # (B, d+1) bias列
    a = hb.T @ hb + ridge * torch.eye(d + 1, dtype=torch.float64)
    sol = torch.linalg.solve(a, hb.T @ z)  # (d+1, vocab_t)
    w = sol[:d].T.contiguous()             # (vocab_t, d)
    b = sol[d].contiguous()                # (vocab_t,)
    return w, b


def transcribe_head(model: LogicGPT, teacher, calib_bits: np.ndarray,
                    config: LogicGPTConfig, *, weight_bits: int = 1,
                    mask: torch.Tensor | None = None,
                    teacher_logits: np.ndarray | None = None,
                    ridge: float = 1.0) -> tuple[float, int]:
    """teacher lm_head(logits射影 Linear)を LogicGPT head(LutStack→GroupSum)へ転写する。

    _final_layer_nodes(argmax厳密保存構成)で線形head を「整数量子化 score_c =
    Σ q_cj h_j + β_c」の unary展開 LUT + GroupSum に落とす。score_c は各クラス c の
    g ビットの popcount(+定数K、全クラス共通)で表現され、GroupSum(k=vocab) の
    クラス別和が score_c + K を復元 → argmax(=次文字予測)が保存される。

    転写元の線形head:
        teacher_logits!=None(既定・domain整合): student二値hidden → teacher logits を
            ridge最小二乗で解いた W,b(_fit_domain_head)。teacher/student の hidden
            分布差を吸収し、student が実際に head へ渡す二値hidden 上で argmax が
            teacher next-char 予測に整合する。
        teacher_logits=None(素朴): teacher lm_head 重みをそのまま量子化転写
            (分布差を無視。ablation/比較用)。

    student head の構造(head_group_factor=g、n_inputs=2、tau=1.0)を転写結果に
    合わせて再構築してから restore する。teacher vocab=65 < student vocab=67 の
    差分(PAD/BOS)は score=0 の空クラスとしてゼロ埋めする(実65クラスargmax保存)。

    Returns:
        (head_agreement, g):
            head_agreement = 転写head の argmax vs teacher lm_head float argmax
                (student二値hidden 上、次文字予測の一致率)
            g = クラス群幅(head_group_factor に相当)
    """
    lm = teacher.model.lm_head  # (vocab_t, d_model) tied、bias無し
    w_lm = lm.weight.detach().to(torch.float64).cpu()  # (65, d_model)
    v_t, d = w_lm.shape
    v_s = config.vocab_size  # 67
    bias_lm = (lm.bias.detach().to(torch.float64).cpu() if lm.bias is not None
               else torch.zeros(v_t, dtype=torch.float64))

    if teacher_logits is not None:
        # domain整合: student二値hidden → teacher logits を解き直す
        w_t, bias_t = _fit_domain_head(calib_bits, teacher_logits, ridge=ridge)
    else:
        w_t, bias_t = w_lm, bias_lm

    # ternary の 0マスクを反映(接続外の重みを0に落としてから量子化)
    if mask is not None:
        w_t = w_t * mask.to(torch.float64).cpu()

    # student vocab へゼロ埋め(PAD/BOS の 2クラスは全ゼロ = score 0)
    w_full = torch.zeros(v_s, d, dtype=torch.float64)
    w_full[:v_t] = w_t
    b_full = torch.zeros(v_s, dtype=torch.float64)
    b_full[:v_t] = bias_t

    lut_node, gs_node, g = _final_layer_nodes(
        w_full, b_full, weight_bits, node_id='head_final')
    indices = lut_node.params['indices']         # (2, v_s*g)
    tts_hex = lut_node.params['truth_tables']

    # student head を転写形状(width=v_s*g, n_inputs=2, softmax)へ再構築
    head_width = v_s * g
    new_head = LutStack(
        d, [head_width], n_inputs=2,
        parameterization=config.parameterization,
        connections=config.connections, selection=config.selection,
    ).to(model.pos.device)
    restore_lut_layer(new_head.layers[0], indices, tts_hex)
    model.head = new_head
    model.group_sum = GroupSum(v_s, tau=1.0).to(model.pos.device)

    # ---- head_agreement: 転写head argmax vs teacher lm_head float argmax ----
    h = np.asarray(calib_bits).astype(np.int64)  # (B, d_model) 0/1
    # 転写head の score(GroupSum後)を素朴 simulate
    n = indices.shape[0]
    gathered = h[:, indices]                                    # (B, n, v_s*g)
    weights = (1 << np.arange(n, dtype=np.int64))[None, :, None]
    addr = (gathered * weights).sum(axis=1)                     # (B, v_s*g)
    tt_arr = np.asarray([int(t, 16) for t in tts_hex], dtype=np.uint64)
    lut_out = ((tt_arr[None, :] >> addr.astype(np.uint64)) & np.uint64(1)).astype(np.int64)
    score_student = lut_out.reshape(h.shape[0], v_s, g).sum(axis=2)  # (B, v_s)
    argmax_student = score_student[:, :v_t].argmax(axis=1)          # 実65クラスのみ

    # (a) 転写fidelity: 転写LUT head argmax vs 転写元 float head(w_t)argmax
    #     (整数量子化argmax保存が student二値hidden 上で成立するか)
    fid_score = h.astype(np.float64) @ w_t.numpy().T + bias_t.numpy()[None, :]  # (B, v_t)
    transcribe_fid = float((argmax_student == fid_score.argmax(axis=1)).mean())
    # (b) teacher整合: 転写LUT head argmax vs teacher が同じ student二値hidden で出す
    #     next-char argmax(w_lm)。coordinator が求める head_agreement 本指標。
    ref_score = h.astype(np.float64) @ w_lm.numpy().T + bias_lm.numpy()[None, :]
    head_agreement = float((argmax_student == ref_score.argmax(axis=1)).mean())
    print(f'    (head転写fidelity vs 転写元float head = {transcribe_fid:.4f})')
    return head_agreement, g


# ============================================================
# モデル構築(FFN/QKV を転写と1:1対応する構造へ差し替え)
# ============================================================

def build_logicgpt_config(args, teacher_n_embd: int, n_layers: int) -> LogicGPTConfig:
    config = LogicGPTConfig(
        vocab_size=STUDENT_VOCAB_SIZE,
        d_model=teacher_n_embd,
        n_layers=n_layers,
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


def _split_qk(model: LogicGPT, config: LogicGPTConfig) -> None:
    """各 block の attention の q_proj を k_proj と別インスタンスへ分離する
    (既定 share_qk=True では q_proj is k_proj。q/k 独立転写のため)。"""
    for blk in model.blocks:
        attn = blk.attn
        if attn.q_proj is attn.k_proj:
            attn.q_proj = LutStack(
                config.d_model, [config.d_model],
                n_inputs=config.n_inputs, parameterization=config.parameterization,
                connections=config.connections, selection=config.selection,
            )


def replace_block_mlps(model: LogicGPT, config: LogicGPTConfig) -> None:
    """各 block.mlp を transcribe_ffn_block と1:1対応する
    (width=d_model, decoder='identity', depth=2)の LogicMLP へ差し替える。"""
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
# 較正データ収集 + 多層転写初期化
# ============================================================

def collect_calib_bits(teacher, calib_windows: torch.Tensor, device):
    """各 block の post-LN hidden(ln_1(x)/ln_2(x))を >0 二値化して 0/1 較正データに。"""
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


def init_from_teacher(model, teacher, qblocks, arm, calib_windows, args, device,
                      *, head_transfer=True):
    """各 block の QKV射影 + FFN + 埋め込み + (任意で)head を量子化teacherから転写初期化する。

    順序(重要): QKV/FFN/embedding を先に転写 → student の hard forward で最終ブロック
    出力(二値hidden)を較正データにして head を転写する(domain整合)。

    Returns: (per-block info 群のリスト, 埋め込み agreement, head agreement, g)。
    head 転写しない場合は head_agree=nan, g=head_group_factor。

    device注意: 転写(sparsify_fanin/transcribe_ffn_block 等)は numpy較正(CPU)と
    teacher重みを混在演算するため、CUDA では device mismatch する(transcribe.py の
    weight @ x_mean)。転写フェーズは teacher/student を CPU に移して実行し、呼び出し側
    が student を device へ戻す(calib は CPU numpy、hard forward も CPU で厳密同値)。
    """
    # 転写フェーズは CPU 固定(共有 transcribe パスの device 非対応を回避)。
    model.to('cpu')
    teacher.model.to('cpu')
    calib_windows = calib_windows.cpu()
    tdev = 'cpu'
    d = qblocks[0].d
    # qblocks の mask も CPU へ揃える(sparsify_from_mask の mask.cpu() 前提)。
    for qb in qblocks:
        qb.masks = {k: v.cpu() for k, v in qb.masks.items()}
    attn_in, ffn_in = collect_calib_bits(teacher, calib_windows, tdev)
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
        ag_q = transcribe_linear_to_lutlayer(
            w[:d], bias[:d], blk.attn.q_proj.layers[0], calib,
            n_inputs=args.n_inputs, mask=mask_q)
        ag_k = transcribe_linear_to_lutlayer(
            w[d:2 * d], bias[d:2 * d], blk.attn.k_proj.layers[0], calib,
            n_inputs=args.n_inputs, mask=mask_k)
        ag_v = transcribe_linear_to_lutlayer(
            w[2 * d:], bias[2 * d:], blk.attn.v_proj.layers[0], calib,
            n_inputs=args.n_inputs, mask=mask_v)

        spec = FfnTranscribeSpec(
            n_inputs=args.n_inputs,
            hidden_width=blk.mlp.luts[0].out_dim,
            out_width=blk.mlp.luts[1].out_dim,
            unit_select='l2_norm',
        )
        ffn_info = init_logic_mlp_from_ffn(blk.mlp, tblk.mlp, ffn_in[i], spec)
        infos.append({
            'block': i, 'qkv_agree_q': ag_q, 'qkv_agree_k': ag_k, 'qkv_agree_v': ag_v,
            'ffn_hidden_agree': ffn_info['hidden_agreement'],
            'ffn_output_agree': ffn_info['output_agreement'],
        })
        print(f'  block {i}: QKV agree q/k/v={ag_q:.4f}/{ag_k:.4f}/{ag_v:.4f}  '
              f'FFN hidden/out={ffn_info["hidden_agreement"]:.4f}/'
              f'{ffn_info["output_agreement"]:.4f}')

    emb_agree = transcribe_embedding(model, teacher)
    print(f'  embedding sign-align agreement={emb_agree:.4f}')

    head_agree = float('nan')
    g = model.config.head_group_factor
    if head_transfer:
        # student の hard forward で最終ブロック出力(二値hidden)を較正データに。
        # embedding/QKV/FFN 転写済みの state で走らせる(domain整合)。CPU 固定。
        head_calib = student_final_hidden_bits(model, calib_windows, tdev)
        teacher_logits = None
        if args.head_fit:
            # 同じ較正 window/位置(先頭 block_size)の teacher logits を集める。
            with torch.no_grad():
                bs = model.config.block_size
                tl = teacher.next_token_logits(calib_windows[:, :bs])  # (B, bs, 65)
                teacher_logits = tl.reshape(-1, tl.shape[-1]).cpu().numpy()
        head_agree, g = transcribe_head(
            model, teacher, head_calib, model.config,
            weight_bits=args.head_weight_bits, teacher_logits=teacher_logits,
            ridge=args.head_ridge)
        mode = 'domain-fit' if args.head_fit else 'naive-lm_head'
        print(f'  head transcribe({mode}): argmax agreement={head_agree:.4f}  '
              f'(group_width g={g}, weight_bits={args.head_weight_bits})')
    else:
        print('  head: 転写なし(random+polish、ablation)')
    return infos, emb_agree, head_agree, g


# ============================================================
# 1 arm の学習ループ
# ============================================================

def run_arm(arm, args, base_teacher, train_loader, eval_loader, val_windows, device,
            config, calib_windows, *, head_transfer_override=None):
    # head_transfer_override: ablation で arm単位に head転写 on/off を上書きする。
    head_transfer = args.head_transfer if head_transfer_override is None else head_transfer_override
    torch.manual_seed(args.seed + {'float': 0, 'bin': 1000, 'ternary': 2000, 'scratch': 3000}[arm])
    model = LogicGPT(config).to(device)
    _split_qk(model, config)
    replace_block_mlps(model, config)
    model.to(device)

    transcribe_infos = None
    emb_agree = float('nan')
    head_agree = float('nan')
    src_bpc_hard = float('nan')
    if arm != 'scratch':
        teacher = copy.deepcopy(base_teacher)
        teacher.model.to(device)
        d = config.d_model
        qblocks = [_QuantAttnBlock(b, d, arm, device) for b in teacher.model.transformer.h]
        if arm in ('bin', 'ternary'):
            print(f'  [arm {arm}] QAT(epochs={args.qat_epochs})...')
            _qat_finetune_teacher(
                teacher, qblocks, arm, calib_windows, device,
                epochs=args.qat_epochs, lr=args.qat_lr, batch_size=args.batch_size)
        _bake_quant_into_teacher(qblocks, arm)
        src_bpc_hard = evaluate_teacher_bpc(teacher, val_windows, device)
        print(f'  [arm {arm}] 量子化source(bake後) teacher BPC={src_bpc_hard:.4f}')
        print(f'  [arm {arm}] teacher QKV/FFN/emb/head転写中(blocks={config.n_layers})...')
        transcribe_infos, emb_agree, head_agree, _g = init_from_teacher(
            model, teacher, qblocks, arm, calib_windows, args, device,
            head_transfer=head_transfer)
        model.to(device)  # 転写フェーズは CPU 固定だったので device へ戻す
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
                    f'L{config.n_layers}/arm_{arm}/loss': total_loss,
                    f'L{config.n_layers}/arm_{arm}/bpc_soft': bpc_soft,
                    f'L{config.n_layers}/arm_{arm}/bpc_hard': bpc_hard,
                    f'L{config.n_layers}/arm_{arm}/agree_hard': agree_hard,
                }, step=step + 1)

    seconds = time.time() - t0
    return {
        'arm': arm,
        'n_layers': config.n_layers,
        'source_bpc_hard': src_bpc_hard,
        'emb_agree': emb_agree,
        'head_agree': head_agree,
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
# 1 層数分の全アーム実行
# ============================================================

def run_layer_count(n_layers, args, teacher, train_loader, eval_loader,
                    val_windows, device, calib_windows):
    config = build_logicgpt_config(args, teacher.cfg['n_embd'], n_layers)
    print(f'\n{"=" * 30} n_layers={n_layers} {"=" * 30}')
    print(f'LogicGPT config: d_model={config.d_model} n_layers={config.n_layers} '
          f'n_heads={config.n_heads} n_inputs={config.n_inputs}')
    results = {}
    for arm in ARMS:
        print(f'\n{"#" * 12} [L{n_layers}] arm {arm} {"#" * 12}')
        # ablation 時は ternary arm を head転写ON に強制(nohead との対照を成立させる)。
        override = True if (args.head_ablation and arm == 'ternary') else None
        results[arm] = run_arm(arm, args, teacher, train_loader, eval_loader,
                               val_windows, device, config, calib_windows,
                               head_transfer_override=override)
    if args.head_ablation:
        # ablation: ternary arm を head転写なしでも走らせて head転写の効果を分離。
        print(f'\n{"#" * 12} [L{n_layers}] arm ternary_nohead (ablation) {"#" * 12}')
        results['ternary_nohead'] = run_arm(
            'ternary', args, teacher, train_loader, eval_loader,
            val_windows, device, config, calib_windows, head_transfer_override=False)
    return results


# ============================================================
# メイン
# ============================================================

def _auto_device(arg):
    if arg is not None:
        return arg
    if torch.cuda.is_available():
        return 'cuda'
    if getattr(torch.backends, 'mps', None) is not None and torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


def main():
    parser = argparse.ArgumentParser(
        description='BitNet-style 量子化多層char-LM → LogicGPT(多層)丸ごと転写')
    parser.add_argument('--teacher', type=str, default='local',
                        help="'local'(ローカル学習の小型多層char-LM。既定・唯一実装)。"
                             'HF BitNet/Bonsai 実重みはサイズ制約で未対応'
                             '(_load_hf_bitnet_note 参照)')
    parser.add_argument('--corpus', type=str,
                        default='data-charlm/corpus__MinGPT-_500x64_T1.0_s42.pt')
    parser.add_argument('--eval-corpus', type=str,
                        default='data-charlm/corpus__MinGPT-_500x64_T1.0_s42_eval.pt')

    # ローカル teacher 構成
    parser.add_argument('--teacher-layers', type=int, default=2)
    parser.add_argument('--teacher-embd', type=int, default=64)
    parser.add_argument('--teacher-heads', type=int, default=2)
    parser.add_argument('--teacher-steps', type=int, default=1500)
    parser.add_argument('--teacher-lr', type=float, default=3e-3)

    # LogicGPT 生徒 / 層数スイープ
    parser.add_argument('--sweep-layers', type=str, default='1,2',
                        help='比較する student 層数(カンマ区切り、例 "1,2")')
    parser.add_argument('--gpt-heads', type=int, default=1)
    parser.add_argument('--gpt-mlp-ratio', type=int, default=1)
    parser.add_argument('--head-group-factor', type=int, default=4)
    parser.add_argument('--n-inputs', type=int, default=2, choices=[2, 4, 6])

    # head 転写(teacher lm_head → LutStack最終層 + GroupSum、argmax厳密保存)。
    # 実測結論(report参照): head転写は hard BPC を下げない(むしろ悪化 Δ−0.10〜−0.63)。
    # 転写head の argmax agreement は wb=4 で ~0.78 と高いが、frozen head 構造が
    # polish するblockと co-adapt できず soft/hard gap を生むため。→ 既定は転写OFF
    # (random head + polish が優位、前タスクの結果と整合)。--head-transfer で opt-in。
    parser.add_argument('--head-transfer', dest='head_transfer',
                        action='store_true', default=False,
                        help='teacher lm_head を head へ転写(既定off。実測で hard BPC 改善せず)')
    parser.add_argument('--no-head-transfer', dest='head_transfer',
                        action='store_false',
                        help='head 転写を無効化(random+polish、既定)')
    parser.add_argument('--head-weight-bits', type=int, default=4,
                        help='head 整数量子化ビット幅(既定4。1=sign は argmax fidelity が'
                             '0.3程度に落ちるため不可。4で teacher整合 ~0.9、g~120)')
    # domain-fit(student二値hidden→teacher logits の ridge解を転写)は g 爆発+
    # sign量子化で head を破壊するため既定off(スモークで実証、report参照)。
    # 既定は teacher lm_head 重みを直接量子化転写(小さい g、argmax保存)。
    parser.add_argument('--head-fit', dest='head_fit', action='store_true', default=False,
                        help='domain整合: student二値hidden→teacher logits の ridge解を転写'
                             '(既定off。sign量子化で g 爆発・head破壊のため比較用)')
    parser.add_argument('--no-head-fit', dest='head_fit', action='store_false')
    parser.add_argument('--head-ridge', type=float, default=1.0,
                        help='domain-fit の L2正則化係数')
    parser.add_argument('--head-ablation', action='store_true',
                        help='ternary arm を head転写あり/なし両方走らせて比較')

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
    parser.add_argument('-ni', '--num-iterations', type=int, default=300)
    parser.add_argument('-ef', '--eval-freq', type=int, default=100)
    parser.add_argument('--eval-max-chars', type=int, default=20_000)
    parser.add_argument('--emb-reg-ramp-start', type=int, default=50)
    parser.add_argument('--emb-reg-ramp-end', type=int, default=2000)
    parser.add_argument('--emb-reg-max-weight', type=float, default=0.1)
    parser.add_argument('--n-calib-windows', type=int, default=64)

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
        args.qat_epochs = min(args.qat_epochs, 1)
        args.teacher_steps = min(args.teacher_steps, 200)
        args.teacher_embd = min(args.teacher_embd, 32)
        args.emb_reg_ramp_start = 20
        args.emb_reg_ramp_end = 200
        if args.device is None:
            args.device = 'cpu'

    args.device = _auto_device(args.device)
    print(f'開始 {_now_jst()}  デバイス: {args.device}')
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    sweep = [int(x) for x in args.sweep_layers.split(',') if x.strip()]

    # ---- ローカル teacher(多層 char-LM)を学習 ----
    _load_hf_bitnet_note()  # HF BitNet/Bonsai 実重みはサイズ制約で未実施(docstring 参照)
    print('注意: HF公開 BitNet b1.58-2B / Bonsai-PrunedPhi-1.8B は 1.8〜2B級で本環境'
          '(CPU/MPS スモーク)に載らないため未使用。ローカル小型多層 char-LM を'
          'ternary量子化して同型 feasibility を示す(_load_hf_bitnet_note 参照)。')
    if args.teacher != 'local':
        raise NotImplementedError(
            f'--teacher {args.teacher!r} は未対応。HF BitNet/Bonsai 実重みは'
            'サイズ制約で未実施(_load_hf_bitnet_note 参照)。--teacher local を使う。')
    print(f'ローカル teacher を学習中(layers={args.teacher_layers} '
          f'n_embd={args.teacher_embd} heads={args.teacher_heads} '
          f'steps={args.teacher_steps})...')
    teacher = LocalCharLMTeacher(
        n_layer=args.teacher_layers, n_embd=args.teacher_embd,
        n_head=args.teacher_heads, block_size=64, device=args.device)
    teacher.train_local(
        steps=args.teacher_steps, lr=args.teacher_lr,
        batch_size=max(8, args.batch_size), seq_len=63, seed=args.seed)
    print(f'  teacher fingerprint={teacher.fingerprint}  n_embd={teacher.cfg["n_embd"]}  '
          f'n_layer={teacher.cfg["n_layer"]}')

    # ---- データ ----
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
    val_windows = load_tiny_shakespeare_val(max_chars=args.eval_max_chars)

    print('teacher自身(float)のBPCを計算中...')
    teacher_bpc = evaluate_teacher_bpc(teacher, val_windows, args.device)
    print(f'  teacher BPC = {teacher_bpc:.4f}  (uniform anchor = {math.log2(65):.4f})')

    args.use_wandb = False
    if not args.no_wandb:
        args.use_wandb = init_wandb(
            run_name=f'bitnet_llm_logicgpt_L{args.sweep_layers}_ni{args.num_iterations}',
            config_dict={**vars(args), 'teacher_bpc': teacher_bpc},
            device=args.device,
            tags=['convert', 'ternary-dwn', 'bitnet', 'logicgpt', 'multilayer', 'charlm'],
            group='ternary-dwn')

    # ---- 層数スイープ ----
    all_results = {}
    for n_layers in sweep:
        all_results[n_layers] = run_layer_count(
            n_layers, args, teacher, train_loader, eval_loader,
            val_windows, args.device, calib_windows)

    # ---- 結果表 ----
    print('\n' + '=' * 112)
    print('BitNet-style 量子化多層char-LM → LogicGPT(多層) 転写 — 結果'
          '(HARD BPC、小さいほど良い)')
    print('=' * 112)
    arm_order = list(ARMS) + (['ternary_nohead'] if args.head_ablation else [])
    print('| L | arm | source BPC(hard) | emb_agree | head_agree | best BPC(soft) | '
          'hard@best-soft | agree@best-soft | best BPC(hard) | sec |')
    print('|' + '---|' * 10)
    for n_layers in sweep:
        for arm in arm_order:
            r = all_results[n_layers].get(arm)
            if r is None:
                continue
            print(f"| {n_layers} | {arm} | {r['source_bpc_hard']:.4f} | {r['emb_agree']:.4f} | "
                  f"{r['head_agree']:.4f} | {r['best_bpc_soft']:.4f} | {r['hard_at_best_soft']:.4f} | "
                  f"{r['agree_at_best_soft']:.4f} | {r['best_bpc_hard']:.4f} | {r['seconds']:.1f} |")
    print(f'\nteacher BPC(float) = {teacher_bpc:.4f}  (uniform anchor = {math.log2(65):.4f})')

    # ---- head転写 ablation(あり=ternary vs なし=ternary_nohead) ----
    if args.head_ablation:
        print('\n--- head転写 ablation(ternary、あり vs なし) ---')
        print('| L | head転写あり hard | head_agree | head転写なし hard | Δhard(なし−あり) |')
        print('|' + '---|' * 5)
        for n_layers in sweep:
            r_on = all_results[n_layers]['ternary']
            r_off = all_results[n_layers].get('ternary_nohead')
            if r_off is None:
                continue
            d_hard = r_off['hard_at_best_soft'] - r_on['hard_at_best_soft']
            print(f"| {n_layers} | {r_on['hard_at_best_soft']:.4f} | {r_on['head_agree']:.4f} | "
                  f"{r_off['hard_at_best_soft']:.4f} | {d_hard:+.4f} |")
        print('  Δhard>0 = head転写ありの方が hard BPC が低い(head転写が効いた)')

    # ---- 派生指標: lift と多層の誤差蓄積 ----
    print('\n--- lift(hard@best-soft、相手 − ternary、正なら ternary が改善) ---')
    print('| L | lift vs scratch | lift vs bin | lift vs float |')
    print('|' + '---|' * 4)
    for n_layers in sweep:
        def h(arm, nl=n_layers):
            return all_results[nl][arm]['hard_at_best_soft']
        lv_s = h('scratch') - h('ternary')
        lv_b = h('bin') - h('ternary')
        lv_f = h('float') - h('ternary')
        print(f'| {n_layers} | {lv_s:+.4f} | {lv_b:+.4f} | {lv_f:+.4f} |')

    print('\n--- 多層の誤差蓄積(ternary arm、block別 QKV/FFN agreement) ---')
    for n_layers in sweep:
        infos = all_results[n_layers]['ternary']['transcribe_infos']
        if not infos:
            continue
        print(f'[L{n_layers}] ternary block別:')
        for info in infos:
            qkv = np.mean([info['qkv_agree_q'], info['qkv_agree_k'], info['qkv_agree_v']])
            print(f'    block {info["block"]}: QKV={qkv:.4f}  '
                  f'FFN_hidden={info["ffn_hidden_agree"]:.4f}  '
                  f'FFN_out={info["ffn_output_agree"]:.4f}')

    print('\n--- 層数による hard BPC(ternary、多層で誤差が積むか) ---')
    for n_layers in sweep:
        r = all_results[n_layers]['ternary']
        print(f'  L={n_layers}: ternary source(hard)={r["source_bpc_hard"]:.4f}  '
              f'hard@best-soft={r["hard_at_best_soft"]:.4f}  '
              f'scratch hard={all_results[n_layers]["scratch"]["hard_at_best_soft"]:.4f}')

    if args.use_wandb:
        summary = {'summary/teacher_bpc': teacher_bpc}
        for n_layers in sweep:
            for arm in ARMS:
                r = all_results[n_layers][arm]
                summary[f'summary/L{n_layers}_{arm}_hard'] = r['hard_at_best_soft']
                summary[f'summary/L{n_layers}_{arm}_source'] = r['source_bpc_hard']
            summary[f'summary/L{n_layers}_lift_vs_scratch'] = (
                all_results[n_layers]['scratch']['hard_at_best_soft']
                - all_results[n_layers]['ternary']['hard_at_best_soft'])
        log_metrics(summary)
        wandb_finish()

    print(f'\n完了 {_now_jst()}')


if __name__ == '__main__':
    main()
