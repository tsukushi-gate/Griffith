"""テキスト蒸留(ROADMAP-v5 Track D7)向けteacher抽象。

convert/teacher.py の `Teacher`(画像分類teacher)と同じ設計思想を、
テキスト系の2タスクへ拡張する:

    TextClassifierTeacher — 文分類(SST-2等)のsoft-label供給元。
        `predict_logits(texts) -> (B, C)` で convert/cache.py の
        `get_or_compute_logits` にそのまま渡せる(画像teacherと同じ
        インデックス整列キャッシュ機構を再利用できる)。
    CharLMTeacher        — 文字レベル言語モデル(次トークン分布/自己回帰生成)。
        シーケンス蒸留(distill.kd_kl_loss_seq)の教師ロジット供給元。

teacher.py 本体は無改造(この2抽象はteacher.pyのTeacherと独立)。
fingerprintの生成は teacher.py の `_auto_fingerprint` をそのまま再利用し、
同一の「重みが変われば必ず変わる」不変条件を保つ。
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Protocol, runtime_checkable

import torch
import torch.nn as nn
import torch.nn.functional as F

from .teacher import _auto_fingerprint

__all__ = [
    'CharLMTeacher',
    'HFMarianMTTeacher',
    'HFSequenceClassifierTeacher',
    'NanoGPTCharTeacher',
    'SHAKESPEARE_CHAR_VOCAB',
    'TextClassifierTeacher',
    'decode_shakespeare_char',
    'encode_shakespeare_char',
    'resolve_text_teacher',
]


@runtime_checkable
class TextClassifierTeacher(Protocol):
    """テキスト分類teacherの統一プロトコル(SST-2等の文分類蒸留用)。

    convert/cache.py の `get_or_compute_logits` は `teacher.predict_logits(x)`
    (x = DataLoaderのデフォルトcollateで束ねたバッチ)を呼ぶだけなので、
    `(text_str, label)` を返すDatasetをそのままラップしたDataLoaderから来る
    `x`(文字列のtuple/list)を受け付けられることが必須。
    """

    num_classes: int
    fingerprint: str  # キャッシュキー用の安定識別子(重みが変われば変わる)

    def predict_logits(self, texts: Sequence[str]) -> torch.Tensor:
        """B個の文字列から(B, num_classes)のロジットを返す。"""
        ...


class HFSequenceClassifierTeacher:
    """HuggingFace `AutoModelForSequenceClassification` のラップ。

    eval()固定+no_grad推論。トークナイズは `padding=True, truncation=True,
    max_length=max_length` で可変長バッチをそのまま処理する。ラベル順序は
    モデルごとに異なりうるため決め打ちせず、`model.config` から
    `num_labels`/`id2label` を読み取る(呼び出し側で順序を仮定しないこと)。

    Args:
        repo_id: HuggingFace Hub のモデルrepo
            (例 'distilbert-base-uncased-finetuned-sst-2-english')。
        device: 推論デバイス。Noneならモデル既定(通常cpu)。
        max_length: トークナイズ時の最大長(超過分はtruncation)。
    """

    def __init__(
        self,
        repo_id: str,
        *,
        device: str | torch.device | None = None,
        max_length: int = 128,
    ) -> None:
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as e:
            raise ImportError(
                'HFSequenceClassifierTeacherには transformers が必要です。'
                "`uv sync --extra text` で導入してください。"
            ) from e

        self.repo_id = repo_id
        self.max_length = max_length
        self.device = torch.device(device) if device is not None else torch.device('cpu')

        self.tokenizer = AutoTokenizer.from_pretrained(repo_id)
        self.model = AutoModelForSequenceClassification.from_pretrained(repo_id)
        self.model.to(self.device)
        self.model.eval()

        # ラベル順序はモデル(config)依存なので決め打ちしない。
        self.num_classes = int(self.model.config.num_labels)
        self.id2label = dict(self.model.config.id2label)

        self.fingerprint = _auto_fingerprint(self.model)

    @torch.no_grad()
    def predict_logits(self, texts: Sequence[str]) -> torch.Tensor:
        """B個の文字列から(B, num_classes)のロジットを返す(CPU、float)。"""
        texts = list(texts)
        encoded = self.tokenizer(
            texts, padding=True, truncation=True, max_length=self.max_length,
            return_tensors='pt',
        )
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        out = self.model(**encoded)
        return out.logits.detach().to('cpu').float()


@runtime_checkable
class CharLMTeacher(Protocol):
    """文字レベル言語モデルteacherの統一プロトコル(シーケンス蒸留用)。

    `next_token_logits` はシーケンス蒸留(distill.kd_kl_loss_seq)への
    teacherロジット供給に、`generate` はデータ拡張・定性評価に使う。
    """

    vocab_size: int
    stoi: dict[str, int]
    itos: dict[int, str]
    fingerprint: str  # キャッシュキー用の安定識別子(重みが変われば変わる)

    def next_token_logits(self, tokens: torch.Tensor) -> torch.Tensor:
        """(B, S)整数トークン列から(B, S, V)の次トークンロジットを返す。"""
        ...

    def generate(
        self,
        n_seqs: int,
        length: int,
        *,
        temperature: float = 1.0,
        seed: int | None = None,
        device: str | torch.device | None = None,
        prompt: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """自己回帰サンプリングで(n_seqs, length)のトークン列を生成する。"""
        ...


# ---------------------------------------------------------------------------
# nanoGPT (karpathy形式) 最小実装 — sosier/nanoGPT-shakespeare-char-tied-weights用
# ---------------------------------------------------------------------------
#
# Vendorする理由: nanoGPTリポジトリ本体(model.py)への依存やtrust_remote_code
# 実行を避け、この変換パイプライン内で完結させるため(spike検証済み)。
# repo実測(spike): config.json + pytorch_model.bin(生state_dict、40テンソル、
# karpathy命名 transformer.wte.weight / transformer.h.{i}.attn.c_attn.weight /
# lm_head.weight、`_orig_mod.`プレフィックスなし、bias=Falseでbiasキーなし)。


class _LayerNorm(nn.Module):
    """bias有無を選べるLayerNorm(karpathy nanoGPT流。F.layer_normはbias=Noneを許容)。"""

    def __init__(self, ndim: int, bias: bool) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)


class _CausalSelfAttention(nn.Module):
    """マルチヘッド因果的self-attention(F.scaled_dot_product_attention使用)。"""

    def __init__(self, cfg: dict) -> None:
        super().__init__()
        assert cfg['n_embd'] % cfg['n_head'] == 0
        self.c_attn = nn.Linear(cfg['n_embd'], 3 * cfg['n_embd'], bias=cfg['bias'])
        self.c_proj = nn.Linear(cfg['n_embd'], cfg['n_embd'], bias=cfg['bias'])
        self.n_head = cfg['n_head']
        self.n_embd = cfg['n_embd']

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(b, t, self.n_head, c // self.n_head).transpose(1, 2)
        q = q.view(b, t, self.n_head, c // self.n_head).transpose(1, 2)
        v = v.view(b, t, self.n_head, c // self.n_head).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(b, t, c)
        return self.c_proj(y)


class _MLP(nn.Module):
    """GELU付きフィードフォワード(4倍拡張、karpathy nanoGPT標準)。"""

    def __init__(self, cfg: dict) -> None:
        super().__init__()
        self.c_fc = nn.Linear(cfg['n_embd'], 4 * cfg['n_embd'], bias=cfg['bias'])
        self.c_proj = nn.Linear(4 * cfg['n_embd'], cfg['n_embd'], bias=cfg['bias'])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.c_proj(F.gelu(self.c_fc(x)))


class _Block(nn.Module):
    """Pre-LN Transformerブロック(attn + MLP、残差接続)。"""

    def __init__(self, cfg: dict) -> None:
        super().__init__()
        self.ln_1 = _LayerNorm(cfg['n_embd'], cfg['bias'])
        self.attn = _CausalSelfAttention(cfg)
        self.ln_2 = _LayerNorm(cfg['n_embd'], cfg['bias'])
        self.mlp = _MLP(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class _MinGPT(nn.Module):
    """karpathy nanoGPT互換の最小GPT実装(lm_headとwteを重み共有)。

    cfgキー: n_layer, n_head, n_embd, block_size, vocab_size, bias(bool)。
    state_dictキー名がkarpathy形式(transformer.wte.weight等)と一致するため、
    sosier/nanoGPT-shakespeare-char-tied-weights等のcheckpointをそのまま
    `load_state_dict(strict=True)` できる。
    """

    def __init__(self, cfg: dict) -> None:
        super().__init__()
        self.cfg = cfg
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(cfg['vocab_size'], cfg['n_embd']),
            wpe=nn.Embedding(cfg['block_size'], cfg['n_embd']),
            h=nn.ModuleList([_Block(cfg) for _ in range(cfg['n_layer'])]),
            ln_f=_LayerNorm(cfg['n_embd'], cfg['bias']),
        ))
        self.lm_head = nn.Linear(cfg['n_embd'], cfg['vocab_size'], bias=False)
        self.transformer.wte.weight = self.lm_head.weight  # weight tying

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        b, t = idx.size()
        assert t <= self.cfg['block_size'], (
            f'シーケンス長({t})がblock_size({self.cfg["block_size"]})を超えています'
        )
        pos = torch.arange(t, dtype=torch.long, device=idx.device)
        x = self.transformer.wte(idx) + self.transformer.wpe(pos)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        return self.lm_head(x)


# nanoGPT shakespeare_char の標準語彙(tiny_shakespeareの文字集合をsorted)。
# 検証済み: sosier/nanoGPT-shakespeare-char-tied-weights のconfig.json
# (vocab_size=65)+著者のtoken-count JSON(shakespeare_char_train_token_counts.json)
# と突き合わせ済み。改行 + ` !$&',-.3:;?` + A-Z + a-z の65文字。
SHAKESPEARE_CHAR_VOCAB = "\n !$&',-.3:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
assert len(SHAKESPEARE_CHAR_VOCAB) == 65, len(SHAKESPEARE_CHAR_VOCAB)

_SHAKESPEARE_STOI = {ch: i for i, ch in enumerate(SHAKESPEARE_CHAR_VOCAB)}
_SHAKESPEARE_ITOS = dict(enumerate(SHAKESPEARE_CHAR_VOCAB))


def encode_shakespeare_char(text: str) -> list[int]:
    """SHAKESPEARE_CHAR_VOCAB に基づき文字列をトークンID列へ変換する。"""
    return [_SHAKESPEARE_STOI[ch] for ch in text]


def decode_shakespeare_char(tokens: Sequence[int]) -> str:
    """SHAKESPEARE_CHAR_VOCAB に基づきトークンID列を文字列へ変換する。"""
    return ''.join(_SHAKESPEARE_ITOS[int(i)] for i in tokens)


class NanoGPTCharTeacher:
    """`sosier/nanoGPT-shakespeare-char-tied-weights` 形式のnanoGPT文字LMラップ。

    huggingface-hubを遅延importしてconfig.json/pytorch_model.binを取得し、
    `_MinGPT` に重みをロードする。**karpathy形式(汎用AutoModel/trust_remote_codeは
    使わない)にのみ対応**。`_orig_mod.`プレフィックス(torch.compile済みcheckpoint)
    は防御的に除去する。重み共有(wte.weight = lm_head.weight)はロード前に
    張り直してから `load_state_dict(strict=True)` する(tied checkpointは
    lm_head.weightキーを含まない場合があるためstrict一致に必要)。

    Args:
        repo_id: HuggingFace Hub のモデルrepo。
        device: 推論デバイス。Noneならcpu。
    """

    def __init__(
        self,
        repo_id: str,
        *,
        device: str | torch.device | None = None,
    ) -> None:
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as e:
            raise ImportError(
                'NanoGPTCharTeacherには huggingface-hub が必要です。'
                "`uv sync --extra text` で導入してください。"
            ) from e

        cfg_path = hf_hub_download(repo_id, 'config.json')
        bin_path = hf_hub_download(repo_id, 'pytorch_model.bin')
        with open(cfg_path, encoding='utf-8') as f:
            hf_cfg = json.load(f)
        cfg = {
            k: hf_cfg[k]
            for k in ('n_layer', 'n_head', 'n_embd', 'block_size', 'vocab_size', 'bias')
        }

        # weights_only=True必須(pickle実行なし。ckpt:スキームと同じ安全側の既定)。
        state_dict = torch.load(bin_path, map_location='cpu', weights_only=True)
        # torch.compile済みcheckpoint等の防御的プレフィックス除去(この repo には無いが
        # 他のkarpathy形式ソースを将来通す際の安全策)。
        state_dict = {k.removeprefix('_orig_mod.'): v for k, v in state_dict.items()}

        self.device = torch.device(device) if device is not None else torch.device('cpu')
        self.model = _MinGPT(cfg)
        # tie前にload_state_dict(strict=True)するため、ロード前に重み共有を張る
        # (tied checkpointがlm_head.weightキーを省く場合でも一致させる)。
        self.model.transformer.wte.weight = self.model.lm_head.weight
        self.model.load_state_dict(state_dict, strict=True)
        self.model.to(self.device)
        self.model.eval()

        self.cfg = cfg
        self.vocab_size = int(cfg['vocab_size'])
        self.block_size = int(cfg['block_size'])
        self.stoi = dict(_SHAKESPEARE_STOI)
        self.itos = dict(_SHAKESPEARE_ITOS)
        self.fingerprint = _auto_fingerprint(self.model)

    @torch.no_grad()
    def next_token_logits(self, tokens: torch.Tensor) -> torch.Tensor:
        """(B, S)トークン列から(B, S, V)の次トークンロジットを返す(CPU、float)。"""
        x = tokens.to(self.device)
        logits = self.model(x)
        return logits.detach().to('cpu').float()

    @torch.no_grad()
    def generate(
        self,
        n_seqs: int,
        length: int,
        *,
        temperature: float = 1.0,
        seed: int | None = None,
        device: str | torch.device | None = None,
        prompt: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """自己回帰サンプリングで(n_seqs, length)のトークン列を生成する。

        block_sizeを超えた文脈は直近block_size分にクロップする(karpathy
        nanoGPT generate()と同じ規約)。

        Args:
            n_seqs: 生成するシーケンス本数。
            length: 生成後の総トークン長(promptを含む)。
            temperature: サンプリング温度(>0)。
            seed: 指定するとこの呼び出し専用のGeneratorで決定的にサンプリングする。
            device: 生成デバイス。Noneならteacher既定device。
            prompt: (n_seqs, p)または(1, p)の初期トークン列。Noneなら
                トークン0から開始する。

        Returns:
            (n_seqs, length) の整数トークンID列。
        """
        gen_device = torch.device(device) if device is not None else self.device
        generator = None
        if seed is not None:
            gen_kind = gen_device.type if gen_device.type != 'mps' else 'cpu'
            generator = torch.Generator(device=gen_kind)
            generator.manual_seed(int(seed))

        if prompt is None:
            idx = torch.zeros(n_seqs, 1, dtype=torch.long, device=gen_device)
        else:
            idx = prompt.to(gen_device)
            if idx.shape[0] == 1 and n_seqs > 1:
                idx = idx.expand(n_seqs, -1).contiguous()

        model = self.model.to(gen_device)
        n_new = length - idx.shape[1]
        for _ in range(max(0, n_new)):
            idx_cond = idx[:, -self.block_size:]
            logits = model(idx_cond)[:, -1, :] / float(temperature)
            probs = F.softmax(logits, dim=-1)
            if generator is not None:
                idx_next = torch.multinomial(probs, num_samples=1, generator=generator)
            else:
                idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx[:, :length]


class HFMarianMTTeacher:
    """HF の翻訳モデル(MarianMT / opus-mt-*)を text->text の翻訳 teacher として wrap。

    系列KD方針: teacher は**訳文テキスト**を返すだけ。学生は byte-level なので、
    呼び出し側が teacher の英文出力を `byte_vocab.encode_bytes` で byte 化して CE
    ターゲットにする。teacher の subword tokenizer は学生から完全に隠蔽され、
    logit-level KD は行わない(生成タスクでは soft KL 冗長=)。

    このクラスは主に **コーパス生成時**(data-free の疑似訳生成)に使う。訓練ループは
    生成済み byte 対を消費するだけで teacher を呼ばない。

    Args:
        repo_id: HF Hub の翻訳モデル(例 'Helsinki-NLP/opus-mt-ja-en')。
        device: 推論デバイス。Noneなら cpu。
        num_beams: greedy=1 / beam探索。既定4(opus-mt標準)。
        max_new_tokens: 生成上限。
    """

    def __init__(
        self,
        repo_id: str,
        *,
        device: str | torch.device | None = None,
        num_beams: int = 4,
        max_new_tokens: int = 128,
    ) -> None:
        try:
            from transformers import MarianMTModel, MarianTokenizer
        except ImportError as e:
            raise ImportError(
                'HFMarianMTTeacher には transformers + sentencepiece が必要です。'
                '`uv sync --extra text` で導入してください。'
            ) from e

        self.repo_id = repo_id
        self.device = torch.device(device) if device is not None else torch.device('cpu')
        self.num_beams = num_beams
        self.max_new_tokens = max_new_tokens
        self.tokenizer = MarianTokenizer.from_pretrained(repo_id)
        self.model = MarianMTModel.from_pretrained(repo_id)
        self.model.to(self.device)
        self.model.eval()
        self.fingerprint = _auto_fingerprint(self.model)

    @torch.no_grad()
    def translate(self, texts: Sequence[str], *, batch_size: int = 32) -> list[str]:
        """source言語の文字列リストを target言語の訳文リストへ(決定的 beam探索)。"""
        results: list[str] = []
        texts = list(texts)
        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            enc = self.tokenizer(
                batch, return_tensors='pt', padding=True, truncation=True
            ).to(self.device)
            gen = self.model.generate(
                **enc,
                max_new_tokens=self.max_new_tokens,
                num_beams=self.num_beams,
                do_sample=False,
            )
            results.extend(self.tokenizer.batch_decode(gen, skip_special_tokens=True))
        return results


def resolve_text_teacher(
    source: str, *, device: str | torch.device | None = None
) -> TextClassifierTeacher | CharLMTeacher | HFMarianMTTeacher:
    """文字列指定からテキストteacherを解決する統一入口(convert/teacher.pyのresolve_teacherに対応)。

    対応スキーム:
        'hf-seqcls:<repo_id>'      — HFSequenceClassifierTeacher
            (AutoModelForSequenceClassification、SST-2等の文分類teacher)
        'hf-nanogpt-char:<repo_id>' — NanoGPTCharTeacher
            (karpathy nanoGPT形式の文字レベル言語モデル。汎用AutoModelはスコープ外)

    Args:
        source: 'scheme:value' 形式の文字列。
        device: 推論デバイス。

    Returns:
        TextClassifierTeacher または CharLMTeacher プロトコルを満たすインスタンス。

    Raises:
        ValueError: sourceの形式が不正、またはスキームが未知。
        ImportError: 対応schemeに必要なパッケージ(transformers/huggingface-hub)未導入。
    """
    if ':' not in source:
        raise ValueError(
            f"sourceは'scheme:value'形式が必要です(例 'hf-seqcls:<repo_id>'): {source!r}"
        )
    scheme, _, value = source.partition(':')

    if scheme == 'hf-seqcls':
        return HFSequenceClassifierTeacher(value, device=device)
    if scheme == 'hf-nanogpt-char':
        return NanoGPTCharTeacher(value, device=device)
    if scheme == 'hf-mt':
        return HFMarianMTTeacher(value, device=device)
    raise ValueError(
        f'未知のテキストteacherスキーム: {scheme!r}'
        "(対応: 'hf-seqcls:' / 'hf-nanogpt-char:' / 'hf-mt:')"
    )
