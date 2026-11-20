"""LL-ViT I-ViT-T CIFAR-10 structure preview.

This sample keeps self-attention as an executable Python token mixer and
shows the LL channel mixer hierarchy.  Paper reproduction training lives in
``experiments/train_llvit.py``; this editor sample is intentionally not routed
through the local lightweight trainer.
"""

from __future__ import annotations

import torch

from griffith_logic import editor_model
from griffith_logic.models.llvit import (
    LLViTConfig as ReproLLViTConfig,
    build_llvit_cifar10 as build_repro_llvit_cifar10,
)

IMG_SIZE = 224
PATCH_SIZE = 16
IN_CHANS = 3
DIM = 192
DEPTH = 12
HEADS = 3
TOKEN_COUNT = (IMG_SIZE // PATCH_SIZE) ** 2 + 1
MLP_HIDDEN = DIM * 4
THERMO_BITS = 8
N_CLASSES = 10


CAP_EXEC_ONLY = {
    'python_executable': True,
    'trainable': False,
    'ir_buildable': False,
    'gate_expandable': False,
    'hdl_exportable': False,
}

CAP_CONTAINER = {
    'python_executable': True,
    'trainable': False,
    'ir_buildable': False,
    'gate_expandable': True,
    'hdl_exportable': False,
}

CAP_GATE = {
    'python_executable': True,
    'trainable': False,
    'ir_buildable': True,
    'gate_expandable': True,
    'hdl_exportable': True,
}


def _shape(tokens: int = TOKEN_COUNT, dim: int = DIM) -> dict:
    return {'tokens': tokens, 'dim': dim, 'flat_width': tokens * dim}


def _hnode(
    id: str,
    label: str,
    level: str,
    kind: str,
    *,
    parent: str | None = None,
    detail: str | None = None,
    shape: dict | None = None,
    capabilities: dict | None = None,
    lowering: dict | None = None,
    paper_note: str | None = None,
) -> dict:
    return {
        'id': id,
        'label': label,
        'level': level,
        'kind': kind,
        'parent': parent,
        'detail': detail,
        'shape': shape,
        'capabilities': capabilities,
        'lowering': lowering,
        'paper_note': paper_note,
    }


def _hierarchy() -> list[dict]:
    nodes = [
        _hnode(
            'llvit_ivitt_cifar10',
            'LL-ViT I-ViT-T CIFAR-10',
            'L3',
            'model',
            detail='I-ViT-T baseline shape with LL channel mixers',
            shape={'input': [3, 224, 224], 'tokens': TOKEN_COUNT, 'dim': DIM, 'classes': 10},
            capabilities=CAP_CONTAINER,
        ),
        _hnode(
            'patch_embed',
            'Patch Embed 16x16',
            'L2',
            'conv_patch_embed',
            parent='llvit_ivitt_cifar10',
            detail='3x224x224 image -> 196 patch tokens at D=192',
            shape={'input': [3, 224, 224], 'patch': 16, 'tokens': 196, 'dim': DIM},
            capabilities=CAP_EXEC_ONLY,
        ),
        _hnode(
            'encoder_stack',
            'Transformer Encoder x12',
            'L2',
            'encoder_stack',
            parent='llvit_ivitt_cifar10',
            detail='12 I-ViT-T encoder blocks; inspect a block to reach the LUT channel mixer',
            shape=_shape(),
            capabilities=CAP_CONTAINER,
        ),
        _hnode(
            'norm_head',
            'LayerNorm + Classifier Head',
            'L2',
            'classifier',
            parent='llvit_ivitt_cifar10',
            detail='CLS token projection to CIFAR-10 logits',
            shape={'input_dim': DIM, 'classes': N_CLASSES},
            capabilities=CAP_EXEC_ONLY,
        ),
    ]
    for i in range(DEPTH):
        block = f'encoder_{i}'
        cm = f'{block}.channel_mixer'
        nodes.extend([
            _hnode(
                block,
                f'Encoder {i}',
                'L2',
                'encoder_block',
                parent='encoder_stack',
                detail='LayerNorm + MHA token mixer + LL channel mixer',
                shape=_shape(),
                capabilities=CAP_CONTAINER,
            ),
            _hnode(
                f'{block}.token_mixer',
                'MHA Token Mixer',
                'L1',
                'mha_token_mixer',
                parent=block,
                detail='Self-attention is kept as executable Python blackbox in phase 1',
                shape=_shape(),
                capabilities=CAP_EXEC_ONLY,
                paper_note='Token mixer remains attention; channel mixer is LUT-based.',
            ),
            _hnode(
                cm,
                'LL Channel Mixer',
                'L1',
                'll_channel_mixer',
                parent=block,
                detail='Original MLP GELU(zW1)W2 is replaced by thermometer -> DWN -> DWN -> conditional sum',
                shape={'input_dim': DIM, 'hidden_lut': MLP_HIDDEN, 'output_dim': DIM},
                capabilities=CAP_CONTAINER,
                paper_note='Two DWN/LUT layers: (768, 192), 8-bit thermometer, LUT6, int4 encoded value.',
            ),
            _hnode(
                f'{cm}.thermometer',
                'Thermometer Encode 8bit',
                'L1',
                'thermometer_encode',
                parent=cm,
                detail='D=192 real channels -> 1536 thermometer bits',
                shape={'input_dim': DIM, 'bits': THERMO_BITS, 'out_width': DIM * THERMO_BITS},
                capabilities=CAP_EXEC_ONLY,
            ),
            _hnode(
                f'{cm}.lut0',
                'LUT Layer 0 (768)',
                'L0',
                'lut_layer',
                parent=cm,
                detail='First channel-mixer DWN layer, 1536 -> 768',
                shape={'in_width': DIM * THERMO_BITS, 'out_width': MLP_HIDDEN, 'n_inputs': 6},
                capabilities=CAP_GATE,
                lowering={'kind': 'lut_layer'},
            ),
            _hnode(
                f'{cm}.lut1',
                'LUT Layer 1 (192)',
                'L0',
                'lut_layer',
                parent=cm,
                detail='Second channel-mixer DWN layer, 768 -> 192',
                shape={'in_width': MLP_HIDDEN, 'out_width': DIM, 'n_inputs': 6},
                capabilities=CAP_GATE,
                lowering={'kind': 'lut_layer'},
            ),
            _hnode(
                f'{cm}.conditional_sum',
                'Conditional Sum',
                'L1',
                'conditional_sum',
                parent=cm,
                detail='Encoded LUT outputs are accumulated as int4-like channel values',
                shape={'dim': DIM, 'encoded_value': 'int4'},
                capabilities=CAP_EXEC_ONLY,
            ),
        ])
    return nodes


def _lowering() -> dict[str, dict]:
    out: dict[str, dict] = {}
    for i in range(DEPTH):
        prefix = f'encoder_{i}.channel_mixer'
        out[f'{prefix}.lut0'] = {
            'kind': 'lut_layer',
            'label': f'Encoder {i} LUT Layer 0',
            'in_width': DIM * THERMO_BITS,
            'out_width': MLP_HIDDEN,
            'n_inputs': 6,
            'parameterization': 'dwn_official',
            'selection': 'hard_ste',
            'truth_table_pattern': [0x6, 0x8, 0xE, 0xA],
            'detail': 'Virtual L0 view for the first LL-ViT channel-mixer DWN layer.',
            'aggregate': {
                'display': 'deck_l0',
                'range_expandable': True,
                'gate_search': True,
                'top_n_confidence': 32,
            },
        }
        out[f'{prefix}.lut1'] = {
            'kind': 'lut_layer',
            'label': f'Encoder {i} LUT Layer 1',
            'in_width': MLP_HIDDEN,
            'out_width': DIM,
            'n_inputs': 6,
            'parameterization': 'dwn_official',
            'selection': 'hard_ste',
            'truth_table_pattern': [0xA, 0x6, 0x8, 0xE],
            'detail': 'Virtual L0 view for the second LL-ViT channel-mixer DWN layer.',
            'aggregate': {
                'display': 'deck_l0',
                'range_expandable': True,
                'gate_search': True,
                'top_n_confidence': 32,
            },
        }
    return out


@editor_model(
    id='llvit_ivitt_cifar10',
    name='LL-ViT I-ViT-T CIFAR-10 Structure Preview',
    task='classification',
    input_spec={
        'shape': [3, 224, 224],
        'width': 3 * 224 * 224,
        'encoding': 'rgb_224_deit',
    },
    output_spec={'width': N_CLASSES, 'semantics': 'class_scores'},
    datasets=['llvit:cifar10'],
    training_presets={
        'default': {
            'dataset': 'llvit:cifar10',
            'lr': 0.0001,
            'logic_lr': 0.02,
            'ann_lr': 0.0001,
            'batch_size': 1,
            'epochs': 1,
            'method': 'bp',
            'loss': 'ce',
            'optimizer': 'adam',
        },
    },
    capabilities=CAP_CONTAINER,
    stages=[
        {
            'id': 'patch_embed',
            'hierarchy_id': 'patch_embed',
            'label': 'Patch Embed 16x16',
            'torch_class': 'PatchEmbed',
            'in_width': 3 * 224 * 224,
            'out_width': TOKEN_COUNT * DIM,
            'level': 'L2',
            'detail': '3x224x224 image -> 196 patch tokens + CLS token',
            'capabilities': CAP_EXEC_ONLY,
        },
        {
            'id': 'encoder_stack',
            'hierarchy_id': 'encoder_stack',
            'label': 'Transformer Encoder x12',
            'torch_class': 'LLViTEncoderBlock',
            'in_width': TOKEN_COUNT * DIM,
            'out_width': TOKEN_COUNT * DIM,
            'level': 'L2',
            'detail': 'MHA token mixer + LUT channel mixer',
            'capabilities': CAP_CONTAINER,
        },
        {
            'id': 'norm_head',
            'hierarchy_id': 'norm_head',
            'label': 'LayerNorm + Classifier',
            'torch_class': 'nn.LayerNorm + nn.Linear',
            'in_width': TOKEN_COUNT * DIM,
            'out_width': N_CLASSES,
            'level': 'L2',
            'detail': 'CLS token -> CIFAR-10 logits',
            'capabilities': CAP_EXEC_ONLY,
        },
    ],
    hierarchy=_hierarchy(),
    lowering=_lowering(),
    paper_ref={
        'title': 'LL-ViT',
        'url': 'https://arxiv.org/pdf/2511.00812',
        'baseline': 'I-ViT-T',
        'datasets': ['CIFAR-10', 'CIFAR-100', 'Flowers-102', 'Tiny-ImageNet'],
        'config': {
            'image_size': 224,
            'patch_size': 16,
            'dim': DIM,
            'encoders': DEPTH,
            'token_count': TOKEN_COUNT,
            'channel_mixer_luts': [MLP_HIDDEN, DIM],
            'thermometer_bits': THERMO_BITS,
            'lut_inputs': 6,
            'lut_mapping': 'learnable',
            'lut_efd': 'official',
            'encoded_value': 'int4',
        },
        'reported_cifar10': {'params': '2.5M', 'gmacs': 0.65, 'accuracy': 95.5},
    },
    dataset_presets={
        'default': {
            'dataset': 'llvit:cifar10',
            'input_spec': {'shape': [3, 224, 224], 'width': 3 * 224 * 224},
            'output_spec': {'width': N_CLASSES},
        },
        'ci_fake': {
            'dataset': 'llvit:cifar10',
            'dataset_params': {'fake': True, 'fake_size': 8},
        },
    },
    description='Structure preview only. Use experiments/train_llvit.py or Modal for paper reproduction training.',
)
def build_model():
    torch.manual_seed(41)
    # The editor keeps this import light enough for structure inspection.  Full
    # paper reproduction uses learnable mapping via experiments/train_llvit.py.
    cfg = ReproLLViTConfig(lut_mapping='random')
    return build_repro_llvit_cifar10(cfg)
