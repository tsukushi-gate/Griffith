"""エディタ用 LL-ViT synthetic debug サンプル。

LL-ViT方針に合わせ、self-attentionは通常のtoken mixerとして残し、
channel mixer(MLP)だけをLUTベースのLogicMLPへ置換する。
実データセット/論文構成ではなく、import・学習・監視の疎通確認用toy。
"""

from __future__ import annotations

import torch
import torch.nn as nn

from griffith_logic import LogicSpec, editor_model, replace_mlp_with_lut

SEQ = 4
TOKEN_DIM = 8
DIM = 16
N_CLASSES = 4


class TinyViTBlock(nn.Module):
    def __init__(self, dim: int = DIM):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, 2, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim),
        )

    def forward(self, x):
        h = self.norm1(x)
        attn, _ = self.attn(h, h, h)
        x = x + attn
        return x + self.mlp(self.norm2(x))


class TinyViT(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Linear(TOKEN_DIM, DIM)
        self.pos = nn.Parameter(torch.randn(1, SEQ, DIM) * 0.02)
        self.blocks = nn.Sequential(TinyViTBlock(), TinyViTBlock())
        self.head = nn.Linear(DIM, N_CLASSES)

    def forward(self, x):
        h = self.embed(x) + self.pos
        h = self.blocks(h)
        return self.head(h.mean(dim=1))


@editor_model(
    id='llvit_synthetic_debug',
    name='LL-ViT Synthetic Debug',
    task='classification',
    input_spec={
        'shape': [SEQ, TOKEN_DIM],
        'width': SEQ * TOKEN_DIM,
        'encoding': 'real_tokens',
    },
    output_spec={
        'width': N_CLASSES,
        'semantics': 'class_scores',
    },
    datasets=['llvit_synthetic_debug'],
    training_presets={
        'default': {
            'dataset': 'llvit_synthetic_debug',
            'lr': 0.001,
            'logic_lr': 0.02,
            'ann_lr': 0.001,
            'batch_size': 32,
            'epochs': 8,
            'method': 'bp',
            'loss': 'ce',
            'optimizer': 'adam',
        },
    },
    capabilities={
        'python_executable': True,
        'trainable': True,
        'ir_buildable': False,
        'gate_expandable': False,
        'hdl_exportable': False,
    },
    stages=[
        {
            'id': 'embed',
            'label': 'Patch/Token Embed',
            'torch_class': 'nn.Linear',
            'in_width': SEQ * TOKEN_DIM,
            'out_width': SEQ * DIM,
            'level': 'L2',
            'detail': '実数tokenをViT次元へ射影',
        },
        {
            'id': 'token_mixer',
            'label': 'Self-Attention Token Mixer',
            'torch_class': 'nn.MultiheadAttention',
            'in_width': SEQ * DIM,
            'out_width': SEQ * DIM,
            'level': 'L2',
            'detail': 'LL-ViT方針: Attentionは実行可能blackboxとして保持',
        },
        {
            'id': 'lut_channel_mixer',
            'label': 'LUT Channel Mixer',
            'torch_class': 'LogicMLP',
            'in_width': SEQ * DIM,
            'out_width': SEQ * DIM,
            'level': 'L1',
            'detail': 'MLP/channel mixerを学習可能LUTへ置換',
        },
        {
            'id': 'head',
            'label': 'Classifier Head',
            'torch_class': 'nn.Linear',
            'in_width': SEQ * DIM,
            'out_width': N_CLASSES,
            'level': 'L2',
            'detail': 'mean pooling後の4クラス分類',
        },
    ],
    description='通常self-attention + LogicMLP channel mixerの合成debugモデル',
)
def build_model():
    torch.manual_seed(23)
    model = TinyViT()
    hybrid, _ = replace_mlp_with_lut(
        model,
        spec=LogicSpec(n_inputs=2, depth=1, neurons_per_output=4),
    )
    return hybrid
