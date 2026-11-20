"""事前学習MLP→IR変換のプリセット定義(ROADMAP-v5)。

各プリセットは ConvertPipelineSpec の「勝ち値」+データセット指定+teacher省略時の
説明をまとめる。GET /api/convert/presets が一覧を返し、ConvertWizardが
選択肢として提示する。overrides でユーザが個別値を上書きできる。

第1号 mnist-intlower-v1 は(整数閾値ニューロン加算木lowering)の勝ちレシピ。
各パラメータの値は事前の実験でゲート基準を通過したrunで確定したもの。
"""

from __future__ import annotations

from dataclasses import asdict

from griffith_logic.convert.pipeline import ConvertPipelineSpec

# spec のうちプリセット/overridesで受け付けるキー(ConvertPipelineSpecのフィールド)。
# fanins は list で受け取り tuple へ正規化する(JSON互換のため)。
_SPEC_FIELDS = set(asdict(ConvertPipelineSpec()).keys())


# 各プリセット: spec(ConvertPipelineSpecのkwargs)+メタ情報。
# 'dataset' はエディタのデータセット解決キー、'teacher_hint' はteacher省略時の
# 既定/説明(ウィザードのプレースホルダ)。
CONVERT_PRESETS: dict[str, dict] = {
    'mnist-intlower-v1': {
        'label': 'MNIST 整数閾値 加算木lowering v1',
        'description': (
            '事前学習MLP(784-300-100-10)→ fan-in≤32枝刈り+STE fine-tune '
            '→ int4ニューロン別スケール量子化(+必要ならQAT)→ unary展開'
            '+popcount加算木で厳密lowering。teacher-2%(hard acc≈0.955)達成の勝ちレシピ'
            '。'
        ),
        'dataset': 'mnist',
        'teacher_hint': (
            "teacher省略時は 'ckpt:checkpoints/.../best.pt' 形式の事前学習MLP "
            'チェックポイントを指定する(TeacherMLP/TeacherMLPSte互換の784-300-100-10)。'
        ),
        'spec': {
            'fanins': [32],
            'epochs_per_stage': 4,
            'lr': 1e-3,
            'weight_bits_hidden': 4,
            'weight_bits_final': 8,
            'qat_epochs': 1,
            'qat_lr': 3e-4,
            'gate_acc': 0.955,
            'batch_size': 128,
            'seed': 0,
        },
    },
    'fashion-intlower-v1': {
        'label': 'Fashion-MNIST 整数閾値 加算木lowering v1',
        'description': (
            'mnist-intlower-v1のFashion-MNIST転移版(\'検証済み)。'
            'タスク難度が高いためfan-in予算を48へ拡大+隠れ層6bit量子化が必要。'
            'hard acc 0.8196(teacher 0.8391の-2%ゲートを+0.05ptの僅差で通過。'
            'シード依存で揺れうる点に注意。堅牢化はfanins=[64]か[64,48]を検討)。'
            '回路規模1.80M LUT出力/77層。'
        ),
        'dataset': 'fashion',  # NOTE: backendのデータセット解決はfashion未対応(mnistのみ)。対応後に有効化
        'teacher_hint': (
            "'ckpt:checkpoints/teacher_mlp_float_fashion_mnist_.../best.pt' 形式の"
            'Fashion-MNIST事前学習MLP(784-300-100-10)を指定する。'
        ),
        'spec': {
            'fanins': [48],
            'epochs_per_stage': 4,
            'lr': 1e-3,
            'weight_bits_hidden': 6,
            'weight_bits_final': 8,
            'qat_epochs': 1,
            'qat_lr': 3e-4,
            'gate_acc': 0.8191,  # teacher 0.8391 - 2%
            'batch_size': 128,
            'seed': 0,
        },
    },
}


def list_presets() -> list[dict]:
    """GET /api/convert/presets 用のプリセット一覧(name/label/説明/dataset/spec)。"""
    out = []
    for name, p in CONVERT_PRESETS.items():
        out.append({
            'name': name,
            'label': p['label'],
            'description': p['description'],
            'dataset': p['dataset'],
            'teacher_hint': p['teacher_hint'],
            'spec': dict(p['spec']),
        })
    return out


def resolve_spec(preset: str, overrides: dict | None = None) -> ConvertPipelineSpec:
    """プリセット名+overrides から ConvertPipelineSpec を構築する。

    overrides はプリセット spec の上に浅くマージする。ConvertPipelineSpec に
    無いキーが overrides に含まれれば ValueError(不正キーの早期検出)。

    Raises:
        ValueError: 未知のプリセット名 / overrides に不正キー / spec構築失敗
    """
    if preset not in CONVERT_PRESETS:
        raise ValueError(
            f'未知のプリセット: {preset}(対応: {sorted(CONVERT_PRESETS)})'
        )
    merged = dict(CONVERT_PRESETS[preset]['spec'])
    if overrides:
        unknown = set(overrides) - _SPEC_FIELDS
        if unknown:
            raise ValueError(
                f'overridesに不正キー: {sorted(unknown)}'
                f'(対応: {sorted(_SPEC_FIELDS)})'
            )
        merged.update(overrides)
    if 'fanins' in merged and merged['fanins'] is not None:
        merged['fanins'] = tuple(merged['fanins'])
    try:
        return ConvertPipelineSpec(**merged)
    except (TypeError, ValueError) as e:
        raise ValueError(f'spec構築失敗: {e}') from e


def preset_dataset(preset: str) -> str:
    """プリセットの既定データセットキーを返す。"""
    if preset not in CONVERT_PRESETS:
        raise ValueError(f'未知のプリセット: {preset}')
    return CONVERT_PRESETS[preset]['dataset']
