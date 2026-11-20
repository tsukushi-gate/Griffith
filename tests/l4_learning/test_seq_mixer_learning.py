"""L4: LogicSeqMixerのトイ収束テスト。

固定する性質(いずれも model.eval() 下の**hard(離散化後)トークン精度 = 1.0**。
soft精度では判定しない = 離散化が効いた状態での正しさを主張するため):

    1. 自己コピー(`tgt_in = labels`): 埋め込み→因果Toeplitz混合→出力ヘッドの
       学習経路の疎通確認。encoder→decoder経路は通らない。
    2. 右シフトteacher forcing(`experiments/train_logic_seq_mixer.py` の
       `prepare_batch` と同じ意味論: `tgt_full=[BOS,tgt,EOS]`、入力
       `tgt_full[:, :-1]`、ラベル `tgt_full[:, 1:]`、損失は `ignore_index=PAD`)
       による**本番と同じseq2seq経路**の収束。experiments/ には依存せず、
       意味論だけをここに再実装している。

合格ラインは 1.0 固定。到達しない場合でも閾値は下げない(実測値を
アサーションメッセージに出して失敗させ、実装側の課題として報告する)。

閾値の実測較正(CPU/float32、tiny構成 selection='soft'・gumbel=False):
    右シフトseq2seqは `context_mode` に強く依存する。V=12/L=5/N=48/600step/lr=0.05 で
        - context_mode='gather', cross_attention='none':
          seed 0/1/2/3/4/42 の**6/6で hard=1.0000**(400stepでは seed3 のみ 0.9896)
        - context_mode='broadcast': arm(i) hard 0.3333〜0.5729 /
          arm(ii) hard 0.3594〜0.6250 で頭打ち(soft も 0.38〜0.71 で上がらない)。
          contextが `enc_states[:, -1]` のみで位置ごとのsrc情報を運べないため。
    よって本テストは gather 構成に対して 1.0 を主張する(broadcast は未達として
    実験側で扱う)。
"""

import os
import sys
from dataclasses import replace

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from griffith_logic.models.logic_seq_mixer import (
    LogicSeqMixer,
    LogicSeqMixerConfig,
)

# トイ課題のサイズ(高速スイート基準に収めるための制約。tiny構成の容量
# — d_channels=48・出力ヘッド m_layer_sizes=(128, vocab*3) — で hard acc=1.0 に
# 到達する最小規模を実測で選んだ: V=8/T=6/N=16/600step で 1.0000)
_MAX_VOCAB = 8
_MAX_SEQ_LEN = 6
_N_SAMPLES = 16
_N_STEPS = 600
_LR = 0.05
_EMB_REG_WEIGHT = 0.1

# 右シフトteacher forcingのトイ課題(experiments/data/synthetic.py と同じ特殊トークン割当)
_PAD, _BOS, _EOS = 0, 1, 2
_FIRST_CONTENT_TOKEN = 3
_S2S_VOCAB = 12      # 内容トークンは [3, 12)
_S2S_LENGTH = 5      # tgt_full長 = 5+2 = 7、デコーダ入力長 = 6 (<= tiny.seq_len=8)
_S2S_SAMPLES = 48
_S2S_STEPS = 600


def _make_copy_task(vocab_used: int, seq_len: int, n_samples: int, generator):
    """コピータスク(tgt = src)のteacher forcing用データを作る。

    デコーダ入力もsrc(=tgt)そのものを与える自己コピー形式にする。
    アーム(i)のcontextは「エンコーダ最終位置のブロードキャスト」のみで
    位置ごとのsrc情報を運べないため、1トークン右シフト形式(tgt_in[t]=tgt[t-1])
    はtiny構成の容量では解けない(実測 hard acc≈0.39)。ここで測りたいのは
    「埋め込み→因果Toeplitz混合→出力ヘッドの学習経路が離散化後も機能するか」
    なので、その経路だけを切り出す自己コピー形式を採る。

    Returns:
        src: (N, T) 入力系列(内容トークンは [1, vocab_used))
        tgt_in: (N, T) デコーダ入力(= labels)
        labels: (N, T) 正解(= src)
    """
    src = torch.randint(1, vocab_used, (n_samples, seq_len), generator=generator)
    labels = src.clone()
    tgt_in = labels.clone()
    return src, tgt_in, labels


def _make_shifted_copy_task(vocab_used: int, length: int, n_samples: int,
                            generator):
    """右シフトteacher forcingのコピータスク(tgt = src)。

    ``experiments/train_logic_seq_mixer.py`` の ``prepare_batch`` と同じ意味論:
    ``tgt_full = [BOS, tgt, EOS]`` を作り、デコーダ入力は ``tgt_full[:, :-1]``、
    ラベルは ``tgt_full[:, 1:]``。位置tのラベルはデコーダ入力に含まれないため、
    src(=encoder出力)を参照しないと解けない**本番と同じseq2seq経路**になる。

    Returns:
        src: (N, L) 入力系列(内容トークンは [3, vocab_used))
        tgt_in: (N, L+1) デコーダ入力 [BOS, tgt_0..tgt_{L-1}]
        labels: (N, L+1) 正解 [tgt_0..tgt_{L-1}, EOS]
    """
    src = torch.randint(_FIRST_CONTENT_TOKEN, vocab_used, (n_samples, length),
                        generator=generator)
    batch = src.shape[0]
    bos = torch.full((batch, 1), _BOS, dtype=torch.long)
    eos = torch.full((batch, 1), _EOS, dtype=torch.long)
    tgt_full = torch.cat([bos, src.clone(), eos], dim=1)
    return src, tgt_full[:, :-1], tgt_full[:, 1:]


def _hard_token_accuracy(model, src, tgt_in, labels,
                         ignore_index: int | None = None) -> float:
    """model.eval()(離散化)でのトークン精度(ignore_indexの位置は除外)。"""
    was_training = model.training
    model.eval()
    with torch.no_grad():
        logits, _ = model(src, tgt_in)
        correct = logits.argmax(dim=-1) == labels
        if ignore_index is None:
            acc = correct.double().mean().item()
        else:
            mask = labels != ignore_index
            acc = (correct & mask).sum().item() / int(mask.sum())
    if was_training:
        model.train()
    return acc


def _train(model, src, tgt_in, labels, steps: int,
           ignore_index: int | None = None) -> None:
    """Adam + CrossEntropy(+埋め込み二値化正則化)で学習する。"""
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=_LR)
    criterion = (nn.CrossEntropyLoss() if ignore_index is None
                 else nn.CrossEntropyLoss(ignore_index=ignore_index))
    for _ in range(steps):
        optimizer.zero_grad()
        logits, emb_reg = model(src, tgt_in)
        loss = criterion(
            logits.reshape(-1, logits.shape[-1]), labels.reshape(-1),
        ) + _EMB_REG_WEIGHT * emb_reg
        loss.backward()
        optimizer.step()


@pytest.mark.learning
class TestLogicSeqMixerLearning:
    def test_learns_copy_to_hard_acc_100(self, seed):
        """tiny構成でコピータスクを学習し、hardトークン精度=1.0に到達する。"""
        base = LogicSeqMixerConfig.tiny()
        cfg = replace(base, selection='soft', gumbel=False)
        vocab_used = min(cfg.vocab_size, _MAX_VOCAB)
        seq_len = min(cfg.seq_len, _MAX_SEQ_LEN)
        assert vocab_used > 1, f'語彙が小さすぎる: vocab_size={cfg.vocab_size}'

        generator = torch.Generator().manual_seed(seed)
        src, tgt_in, labels = _make_copy_task(
            vocab_used, seq_len, _N_SAMPLES, generator)

        torch.manual_seed(seed)
        model = LogicSeqMixer(cfg)
        _train(model, src, tgt_in, labels, _N_STEPS)

        acc = _hard_token_accuracy(model, src, tgt_in, labels)
        assert acc == 1.0, (
            f'コピータスクが離散化後に解けていない: hard token acc={acc:.4f} '
            f'(vocab_used={vocab_used}, seq_len={seq_len}, steps={_N_STEPS})'
        )

    def test_learns_shifted_copy_seq2seq_to_hard_acc_100(self, seed):
        """右シフトteacher forcingのseq2seqコピーでhardトークン精度=1.0に到達する。

        本番(`experiments/train_logic_seq_mixer.py`)と同じ
        「[BOS,tgt,EOS] を作り 入力[:-1] / ラベル[1:]、損失はignore_index=PAD」
        の意味論で、**encoder→decoder経路**が離散化後も機能することを固定する。
        context_mode='gather'(位置対応の静的context)構成に対する主張である
        (broadcast構成の未達実測はモジュールdocstring参照)。
        """
        cfg = replace(
            LogicSeqMixerConfig.tiny(),
            selection='soft', gumbel=False,
            context_mode='gather', cross_attention='none',
        )
        assert _S2S_VOCAB <= cfg.vocab_size, '課題語彙がモデル語彙を超えている'
        assert _S2S_LENGTH + 1 <= cfg.seq_len, \
            f'デコーダ入力長 {_S2S_LENGTH + 1} が seq_len={cfg.seq_len} を超える'

        generator = torch.Generator().manual_seed(seed)
        src, tgt_in, labels = _make_shifted_copy_task(
            _S2S_VOCAB, _S2S_LENGTH, _S2S_SAMPLES, generator)
        assert not torch.equal(tgt_in, labels), \
            'デコーダ入力とラベルが同一(右シフトになっていない)'

        torch.manual_seed(seed)
        model = LogicSeqMixer(cfg)
        _train(model, src, tgt_in, labels, _S2S_STEPS, ignore_index=_PAD)

        acc = _hard_token_accuracy(model, src, tgt_in, labels,
                                   ignore_index=_PAD)
        assert acc == 1.0, (
            f'右シフトseq2seqコピーが離散化後に解けていない: '
            f'hard token acc={acc:.4f} '
            f'(V={_S2S_VOCAB}, L={_S2S_LENGTH}, N={_S2S_SAMPLES}, '
            f'steps={_S2S_STEPS}, context_mode={cfg.context_mode}, '
            f'cross_attention={cfg.cross_attention})'
        )


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
