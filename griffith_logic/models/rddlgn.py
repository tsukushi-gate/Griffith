"""
RDDLGN (Recurrent Deep Differentiable Logic Gate Networks) の実装。

Bührer et al., 2025 (arXiv:2508.06097) に基づく。
微分可能論理ゲートによるseq2seqエンコーダ・デコーダモデル。
"""

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn

from ..group_sum import GroupSum
from ..lut_layer import LutLayer

# ---------------------------------------------------------------------------
# 設定
# ---------------------------------------------------------------------------

@dataclass
class RDDLGNConfig:
    """RDDLGNモデルの全ハイパーパラメータ。"""

    vocab_size: int = 16000
    d_emb: int = 1024
    seq_len: int = 16

    # 各レイヤーグループの出力次元リスト（入力次元は連結から自動計算）
    n_layer_sizes: list = field(default_factory=lambda: [12000, 12000])
    k_layer_sizes: list = field(default_factory=lambda: [54000, 32000])
    l_layer_sizes: list = field(default_factory=lambda: [12000, 12000])
    p_layer_sizes: list = field(default_factory=lambda: [64000, 48000])
    m_layer_sizes: list = field(default_factory=lambda: [400000, 400000, 480000])

    # LutLayer設定
    n_inputs: int = 2
    parameterization: str = 'softmax'
    connections: str = 'random'
    grad_factor: float = 1.0
    selection: str = 'soft'  # 'soft' | 'hard_st' | 'gumbel_st'

    # 出力設定
    tau: float = 2.0

    # WARP設定（parameterization='warp'時のみ有効）
    tau_warp: float = 1.0   # WARPの温度パラメータ
    gumbel: bool = True     # Gumbel-Sigmoid smoothing

    # 隠れ状態初期化: 'gaussian'(σ(N(0,scale²))), 'zero'(0.5固定), 'one'(1.0固定)
    init_mode: str = 'zero'
    init_scale: float = 0.5

    # 語彙共有（src/tgt同一のEmbedding）
    shared_vocab: bool = True

    @classmethod
    def from_paper(cls, scale: float = 1.0) -> 'RDDLGNConfig':
        """論文Table 1の設定を返す。scaleで全レイヤーサイズを均等にスケール。"""

        def _scale(sizes: list[int]) -> list[int]:
            return [max(4, int(s * scale)) for s in sizes]

        vocab_size = max(4, int(16000 * scale)) if scale < 1.0 else 16000
        d_emb = max(4, int(1024 * scale)) if scale < 1.0 else 1024

        # M最終層はvocab_sizeで割り切れる必要がある
        m_sizes = _scale([400000, 400000, 480000])
        # 最終M層をvocab_sizeの倍数に調整
        group_factor = max(1, m_sizes[-1] // vocab_size)
        m_sizes[-1] = vocab_size * group_factor

        return cls(
            vocab_size=vocab_size,
            d_emb=d_emb,
            n_layer_sizes=_scale([12000, 12000]),
            k_layer_sizes=_scale([54000, 32000]),
            l_layer_sizes=_scale([12000, 12000]),
            p_layer_sizes=_scale([64000, 48000]),
            m_layer_sizes=m_sizes,
        )

    @classmethod
    def tiny(cls) -> 'RDDLGNConfig':
        """デバッグ・テスト用の小構成（copyタスク向け推奨サイズ）。"""
        vocab_size = 50
        group_factor = 20
        return cls(
            vocab_size=vocab_size,
            d_emb=128,
            seq_len=4,
            n_layer_sizes=[512, 512],
            k_layer_sizes=[1024, 1024],
            l_layer_sizes=[512, 512],
            p_layer_sizes=[1024, 768],
            m_layer_sizes=[2000, 2000, vocab_size * group_factor],
            tau=1.0,
        )


# ---------------------------------------------------------------------------
# LogicEmbedding
# ---------------------------------------------------------------------------

class LogicEmbedding(nn.Module):
    """
    トークン埋め込み + sigmoid活性化。

    訓練時: Embedding → sigmoid（連続[0,1]）
    推論時: Embedding → Heaviside（離散{0,1}）
    """

    def __init__(self, vocab_size: int, d_emb: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_emb)
        self._last_output: torch.Tensor | None = None

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tokens: (batch, seq_len) 整数トークンID

        Returns:
            (batch, seq_len, d_emb) [0,1]値テンソル
        """
        e = self.embedding(tokens)
        if self.training:
            x = torch.sigmoid(e)
        else:
            # Collapse: Heaviside（閾値0.5）
            x = (torch.sigmoid(e) > 0.5).to(e.dtype)
        self._last_output = x
        return x

    def binary_reg_loss(self) -> torch.Tensor:
        """二値化正則化損失: mean(x * (1 - x))。0/1に近いほど小さくなる。"""
        assert self._last_output is not None, 'forwardを先に呼んでください'
        x = self._last_output
        return (x * (1.0 - x)).mean()


# ---------------------------------------------------------------------------
# LutStack
# ---------------------------------------------------------------------------

class LutStack(nn.Module):
    """
    LutLayerの連鎖。N/K/L/P/M各レイヤーグループの構築に使う。

    in_dim → layer_sizes[0] → layer_sizes[1] → ... → layer_sizes[-1]
    """

    def __init__(
        self,
        in_dim: int,
        layer_sizes: list[int],
        n_inputs: int = 2,
        device: str = 'cpu',
        grad_factor: float = 1.0,
        parameterization: str = 'softmax',
        connections: str = 'random',
        tau_warp: float = 1.0,
        gumbel: bool = True,
        selection: str = 'soft',
        wiring: str = 'fixed',
        lm_tau: float = 0.001,
    ):
        super().__init__()
        self.layers = nn.ModuleList()
        prev = in_dim
        for i, size in enumerate(layer_sizes):
            # 学習可能配線(DWN)は第1層のみ(DWNClassifierの mapping if i==0 else 'random'
            # と同じ定石)。LearnableMapping重み= in_dim×(out_dim×n_inputs)で大きいため、
            # 全層learnableは非現実的。
            layer_wiring = wiring if i == 0 else 'fixed'
            layer = LutLayer(
                in_dim=prev,
                out_dim=size,
                n_inputs=n_inputs,
                device=device,
                grad_factor=grad_factor,
                parameterization=parameterization,
                connections=connections,
                selection=selection,
                wiring=layer_wiring,
                lm_tau=lm_tau,
            )
            if parameterization == 'warp':
                layer.tau_warp = tau_warp
                layer.gumbel = gumbel
            self.layers.append(layer)
            prev = size
        self.out_dim = layer_sizes[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

def _init_hidden(batch: int, dim: int, device: torch.device,
                  mode: str = 'zero', scale: float = 0.5,
                  training: bool = True) -> torch.Tensor:
    """リカレント隠れ状態の初期化。

    Args:
        mode: 'gaussian' — σ(N(0, scale²))
              'zero' — 0.5固定（sigmoid(0)=0.5、情報的にニュートラル）
              'one' — 1.0固定
        training: True（既定、後方互換）なら訓練時と同じ連続値をそのまま返す。
            False（hard eval）の場合、超立方体の頂点（{0.0, 1.0}）に丸める。
            丸めないと K/P-layer の再帰状態が全タイムステップにわたって分数値の
            ままになり、「.eval()すれば完全なBoolean回路として振る舞う」という
            RDDLGN論文(collapse)およびデプロイキットのビット厳密等価の前提が
            崩れる。

            二値化規則は `x > 0.5 → 1.0, else → 0.0`（タイブレークは0.5→0）とし、
            リポジトリ全体の既存の閾値規則に揃えた:
              - griffith_logic/binarize.py の BinarizeFunction.forward
                (19行) / Binarize.forward の eval経路 (56行) —
                BinaryBrain DifferentiableLutN.h 準拠の元祖実装。
              - griffith_logic/layers/lut.py の重み離散化
                (411, 429, 439行、いずれも `> 0.5`)。
              - 本ファイル LogicEmbedding.forward の eval経路 (131行、`> 0.5`)。
            いずれも strict `>` であり、ちょうど閾値に一致する値は0側に倒れる。
            この規則により init_mode='zero'（ちょうど0.5）はhard evalで0.0に
            倒れる（0.5は閾値そのものと一致し「>」で偽になるため）。
    """
    if mode == 'gaussian':
        x = torch.sigmoid(torch.randn(batch, dim, device=device) * scale)
    elif mode == 'one':
        x = torch.ones(batch, dim, device=device)
    else:
        # 'zero' — 0.5はLUTにとって最も情報量が少ない中立値
        x = torch.full((batch, dim), 0.5, device=device)

    if not training:
        # hard eval: リポジトリ全体の二値化規則（上記docstring参照）で丸める。
        x = (x > 0.5).to(x.dtype)

    return x


class RDDLGNEncoder(nn.Module):
    """
    RDDLGNエンコーダ。

    N-layers: フィードフォワード表現学習（全タイムステップをバッチ化）
    K-layers: リカレント時間方向処理（タイムステップをループ）
    """

    def __init__(self, config: RDDLGNConfig, device: str = 'cpu'):
        super().__init__()
        lut_kwargs = dict(
            n_inputs=config.n_inputs,
            device=device,
            grad_factor=config.grad_factor,
            parameterization=config.parameterization,
            connections=config.connections,
            tau_warp=config.tau_warp,
            gumbel=config.gumbel,
            selection=config.selection,
        )

        # N-layers: d_emb → n_layer_sizes
        self.n_layers = LutStack(config.d_emb, config.n_layer_sizes, **lut_kwargs)

        # K-layers: (N出力 + K出力) → k_layer_sizes
        n_out = config.n_layer_sizes[-1]
        k_out = config.k_layer_sizes[-1]
        self.k_layers = LutStack(n_out + k_out, config.k_layer_sizes, **lut_kwargs)

        self.k_out_dim = k_out
        self.init_mode = config.init_mode
        self.init_scale = config.init_scale

    def forward(self, embedded_seq: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embedded_seq: (batch, seq_len, d_emb) sigmoid後の埋め込み

        Returns:
            context: (batch, k_out_dim) エンコーダコンテキストベクトル
        """
        batch, seq_len, _ = embedded_seq.shape
        device = embedded_seq.device

        # N-layers: 全タイムステップをバッチ化して処理
        h = embedded_seq.reshape(batch * seq_len, -1)
        h = self.n_layers(h)
        h = h.reshape(batch, seq_len, -1)

        # K-layers: タイムステップを逐次処理
        k_prev = _init_hidden(
            batch, self.k_out_dim, device, self.init_mode, self.init_scale,
            training=self.training,
        )
        for t in range(seq_len):
            k_input = torch.cat([h[:, t, :], k_prev], dim=-1)
            k_prev = self.k_layers(k_input)

        return k_prev


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------

class RDDLGNDecoder(nn.Module):
    """
    RDDLGNデコーダ。

    L-layers: フィードフォワードターゲット処理（バッチ化）
    P-layers: 自己回帰リカレント（タイムステップをループ）
    M-layers: 出力生成 → GroupSum → logits
    """

    def __init__(self, config: RDDLGNConfig, device: str = 'cpu'):
        super().__init__()
        lut_kwargs = dict(
            n_inputs=config.n_inputs,
            device=device,
            grad_factor=config.grad_factor,
            parameterization=config.parameterization,
            connections=config.connections,
            tau_warp=config.tau_warp,
            gumbel=config.gumbel,
            selection=config.selection,
        )

        context_dim = config.k_layer_sizes[-1]
        l_out = config.l_layer_sizes[-1]
        p_out = config.p_layer_sizes[-1]

        # L-layers: d_emb → l_layer_sizes
        self.l_layers = LutStack(config.d_emb, config.l_layer_sizes, **lut_kwargs)

        # P-layers: (P出力 + context + L出力) → p_layer_sizes
        p_in = p_out + context_dim + l_out
        self.p_layers = LutStack(p_in, config.p_layer_sizes, **lut_kwargs)

        # M-layers: (P出力 + context + L出力) → m_layer_sizes
        m_in = p_out + context_dim + l_out
        self.m_layers = LutStack(m_in, config.m_layer_sizes, **lut_kwargs)

        # GroupSum: M最終出力 → vocab_size個のlogits
        self.group_sum = GroupSum(k=config.vocab_size, tau=config.tau)

        self.p_out_dim = p_out
        self.init_mode = config.init_mode
        self.init_scale = config.init_scale

    def forward(
        self,
        target_embedded: torch.Tensor,
        context: torch.Tensor,
        return_hidden: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        """
        Args:
            target_embedded: (batch, tgt_len, d_emb) sigmoid後のターゲット埋め込み
            context: (batch, context_dim) エンコーダ出力
            return_hidden: True の場合、中間活性リストも返す(exit-head蒸留用)。

        Returns:
            return_hidden=False(既定): logits (batch, tgt_len, vocab_size) のみ。
            return_hidden=True: (logits, hidden_states) のタプル。
                hidden_states は長さ tgt_len の要素からなるリストで、各要素は
                P-layer出力 p_t: (batch, p_out_dim)([0,1]論理域、GroupSum前の
                自己回帰リカレント状態)。M-layerはGroupSum直前でvocabに
                投影されるM出力そのものより、時間方向の情報を集約する
                P-layerの方がexit-head蒸留の中間表現として自然なため選択。
        """
        batch, tgt_len, _ = target_embedded.shape
        device = target_embedded.device

        # L-layers: 全タイムステップをバッチ化
        l = target_embedded.reshape(batch * tgt_len, -1)
        l = self.l_layers(l)
        l = l.reshape(batch, tgt_len, -1)

        # P-layers: タイムステップを逐次処理
        p_prev = _init_hidden(
            batch, self.p_out_dim, device, self.init_mode, self.init_scale,
            training=self.training,
        )

        logits_list = []
        hidden_list: list[torch.Tensor] = []
        for t in range(tgt_len):
            l_t = l[:, t, :]

            # P-layer: [p_prev; context; l_t] → p_t
            p_input = torch.cat([p_prev, context, l_t], dim=-1)
            p_t = self.p_layers(p_input)
            p_prev = p_t
            if return_hidden:
                hidden_list.append(p_t)

            # M-layer: [p_t; context; l_t] → logits
            m_input = torch.cat([p_t, context, l_t], dim=-1)
            m_t = self.m_layers(m_input)
            logits_t = self.group_sum(m_t)
            logits_list.append(logits_t)

        logits = torch.stack(logits_list, dim=1)
        if return_hidden:
            return logits, hidden_list
        return logits


# ---------------------------------------------------------------------------
# RDDLGN（トップレベルモデル）
# ---------------------------------------------------------------------------

class RDDLGN(nn.Module):
    """
    Recurrent Deep Differentiable Logic Gate Network。

    seq2seqエンコーダ・デコーダモデル。全結合層を微分可能論理ゲート（LUT）で置換。
    """

    def __init__(self, config: RDDLGNConfig, device: str = 'cpu'):
        super().__init__()
        self.config = config

        # Embedding
        self.src_embedding = LogicEmbedding(config.vocab_size, config.d_emb)
        if config.shared_vocab:
            self.tgt_embedding = self.src_embedding
        else:
            self.tgt_embedding = LogicEmbedding(config.vocab_size, config.d_emb)

        # Encoder / Decoder
        self.encoder = RDDLGNEncoder(config, device=device)
        self.decoder = RDDLGNDecoder(config, device=device)

    def forward(
        self,
        src_tokens: torch.Tensor,
        tgt_tokens: torch.Tensor,
        return_hidden: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
        """
        Args:
            src_tokens: (batch, src_len) ソーストークンID
            tgt_tokens: (batch, tgt_len) ターゲットトークンID（teacher forcing用、シフト済み）
            return_hidden: True の場合、中間活性リストも追加で返す
                (exit-head蒸留用。既定Falseでは既存の2-tuple契約を完全維持する)。

        Returns:
            return_hidden=False(既定): (logits, emb_reg_loss) の2-tuple(既存契約と同一)。
                logits: (batch, tgt_len, vocab_size)
                emb_reg_loss: スカラー、二値化正則化損失
            return_hidden=True: (logits, emb_reg_loss, hidden_states) の3-tuple。
                hidden_states は長さ2のリスト([0,1]論理域の中間活性、
                exit head蒸留は各要素を個別のexit head入力として使う想定):
                    [0] encoder context: (batch, context_dim)
                        — エンコーダが集約した固定長の文脈ベクトル。
                    [1] decoder P-layer出力(時間方向にstack済み):
                        (batch, tgt_len, p_out_dim)
                        — デコーダの自己回帰リカレント状態(M-layer/GroupSum
                        直前)。RDDLGNDecoder.forward(return_hidden=True)の
                        hidden_states(P-layer出力のリスト)をstackしたもの。
        """
        # Embedding
        src_emb = self.src_embedding(src_tokens)
        tgt_emb = self.tgt_embedding(tgt_tokens)

        # Encode
        context = self.encoder(src_emb)

        # Decode
        if return_hidden:
            logits, decoder_hidden = self.decoder(tgt_emb, context, return_hidden=True)
        else:
            logits = self.decoder(tgt_emb, context)

        # 二値化正則化損失
        emb_reg_loss = self.src_embedding.binary_reg_loss()
        if not self.config.shared_vocab:
            emb_reg_loss = (emb_reg_loss + self.tgt_embedding.binary_reg_loss()) / 2.0

        if return_hidden:
            p_stack = torch.stack(decoder_hidden, dim=1)  # (batch, tgt_len, p_out_dim)
            hidden_states = [context, p_stack]
            return logits, emb_reg_loss, hidden_states

        return logits, emb_reg_loss

    def generate_greedy(
        self, src_tokens: torch.Tensor, *, bos_token: int, eos_token: int,
        max_new_tokens: int,
    ) -> list[int]:
        """自己回帰貪欲デコード(PyTorch側リファレンス実装)。

        教師強制(forward)と違い、1トークンずつ: 前ステップの出力(argmax)を
        次ステップの入力として埋め込み直す。C/simulate側との3者一致の
        基準の1つ(他は export/rddlgn_ir.py::simulate_rddlgn_greedy)。
        argmaxのタイブレークは「最初の最大値」(strict `>`)を手動で実装し、
        runtime/c の `mdl_argmax` / numpy `np.argmax` と規約を揃える
        (torch.argmaxの同点処理はバージョン依存で信頼しないため)。

        Args:
            src_tokens: (src_len,) 整数トークンID(バッチ次元なし、1系列)
            bos_token / eos_token: 開始・終了トークンID
            max_new_tokens: 生成する最大トークン数(EOSで早期終了)

        Returns:
            生成トークン列(EOSに達したらそれを含めて打ち切る)
        """
        assert not self.training, 'hard eval(.eval())で呼ぶこと'
        device = src_tokens.device

        with torch.no_grad():
            src_emb = self.src_embedding(src_tokens.unsqueeze(0))
            context = self.encoder(src_emb)

            p_prev = _init_hidden(
                1, self.decoder.p_out_dim, device, self.decoder.init_mode,
                self.decoder.init_scale, training=False)

            tokens: list[int] = []
            prev_token = int(bos_token)
            for _ in range(int(max_new_tokens)):
                tok = torch.tensor([[prev_token]], device=device,
                                   dtype=torch.long)
                l_t = self.tgt_embedding(tok)[:, 0, :]
                l_t = self.decoder.l_layers(l_t)

                p_input = torch.cat([p_prev, context, l_t], dim=-1)
                p_t = self.decoder.p_layers(p_input)
                p_prev = p_t

                m_input = torch.cat([p_t, context, l_t], dim=-1)
                m_t = self.decoder.m_layers(m_input)
                logits = self.decoder.group_sum(m_t)[0]

                token = 0
                best = logits[0]
                for i in range(1, logits.shape[0]):
                    if logits[i] > best:
                        best = logits[i]
                        token = i
                token = int(token)
                tokens.append(token)
                if token == int(eos_token):
                    break
                prev_token = token

        return tokens

    def count_parameters(self) -> dict[str, int]:
        """パラメータ数の内訳を返す。"""
        counts = {}
        counts['embedding'] = sum(
            p.numel() for p in self.src_embedding.parameters()
        )
        if not self.config.shared_vocab:
            counts['embedding'] += sum(
                p.numel() for p in self.tgt_embedding.parameters()
            )

        counts['n_layers'] = sum(
            p.numel() for p in self.encoder.n_layers.parameters()
        )
        counts['k_layers'] = sum(
            p.numel() for p in self.encoder.k_layers.parameters()
        )
        counts['l_layers'] = sum(
            p.numel() for p in self.decoder.l_layers.parameters()
        )
        counts['p_layers'] = sum(
            p.numel() for p in self.decoder.p_layers.parameters()
        )
        counts['m_layers'] = sum(
            p.numel() for p in self.decoder.m_layers.parameters()
        )
        counts['total'] = sum(counts.values())
        return counts


# ---------------------------------------------------------------------------
# RDDLGNClassifier（テキスト分類向け）
# ---------------------------------------------------------------------------

class RDDLGNClassifier(nn.Module):
    """
    RDDLGNエンコーダ + 分類ヘッド（SST-2等のテキスト分類蒸留、ROADMAP-v5 Track D7）。

    LogicEmbedding → RDDLGNEncoder（系列をリカレントに畳み込みcontextベクトル化）
    → LutStack（分類ヘッド） → GroupSum(k=n_classes) の構成。
    seq2seqデコーダ（RDDLGNDecoder）は使わず、エンコーダのcontext出力のみを使う点が
    RDDLGN本体との違い。
    """

    def __init__(
        self,
        config: RDDLGNConfig,
        n_classes: int,
        head_sizes: list[int] | None = None,
        device: str = 'cpu',
    ):
        super().__init__()
        self.config = config
        self.n_classes = n_classes

        if head_sizes is None:
            head_sizes = [1024, 512]
        head_sizes = list(head_sizes)
        # 最終層はn_classesで割り切れる必要がある（RDDLGNConfig.from_paperの
        # group_factor調整と同じ考え方: 最終サイズをn_classesの倍数に丸める）。
        group_factor = max(1, head_sizes[-1] // n_classes)
        head_sizes[-1] = n_classes * group_factor

        self.embedding = LogicEmbedding(config.vocab_size, config.d_emb)
        self.encoder = RDDLGNEncoder(config, device=device)

        context_dim = config.k_layer_sizes[-1]
        self.head = LutStack(
            context_dim,
            head_sizes,
            n_inputs=config.n_inputs,
            device=device,
            grad_factor=config.grad_factor,
            parameterization=config.parameterization,
            connections=config.connections,
            tau_warp=config.tau_warp,
            gumbel=config.gumbel,
            selection=config.selection,
        )

        # train_cnn.pyのtau規約: 未指定時 tau = max(1.0, sqrt(head出力/クラス数))
        head_out = head_sizes[-1]
        tau = max(1.0, math.sqrt(head_out / n_classes))
        self.group_sum = GroupSum(k=n_classes, tau=tau)

    def forward(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            tokens: (batch, seq_len) 整数トークンID

        Returns:
            logits: (batch, n_classes)
            emb_reg_loss: スカラー、二値化正則化損失（RDDLGN.forwardと同じ契約）
        """
        embedded = self.embedding(tokens)
        context = self.encoder(embedded)
        h = self.head(context)
        logits = self.group_sum(h)

        emb_reg_loss = self.embedding.binary_reg_loss()

        return logits, emb_reg_loss
