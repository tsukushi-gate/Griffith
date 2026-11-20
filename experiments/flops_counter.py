"""
DLGN / BinaryBrain モデルの理論FLOPS計測ユーティリティ。

LUT forward計算のFLOPS:
  butterfly基底: N × 2^N 乗算 + N × 2^N 加算 = 2N × 2^N
  重み付き和:    2^N 乗算 + (2^N - 1) 加算 ≒ 2 × 2^N
  合計:          (2N + 2) × 2^N FLOPs/ゲート/サンプル
"""


def lut_flops_per_gate(n_inputs: int) -> int:
    """1ゲート1サンプルあたりのLUT forward FLOPS"""
    n_entries = 2 ** n_inputs
    return (2 * n_inputs + 2) * n_entries


def count_flops_dlgn(total_gates: int, n_inputs: int, k: int,
                     batch_size: int = 1) -> dict:
    """
    DLGNモデルのforward FLOPS計算。

    Args:
        total_gates: 総ロジックゲート数
        n_inputs: LUTの入力数
        k: 層あたりニューロン数（GroupSum計算用）
        batch_size: バッチサイズ

    Returns:
        各コンポーネントのFLOPSと合計
    """
    flops_per_gate = lut_flops_per_gate(n_inputs)

    lut_flops = total_gates * flops_per_gate * batch_size
    # GroupSum: kニューロンの合計 + τで割る
    groupsum_flops = k * batch_size

    total = lut_flops + groupsum_flops
    per_sample = total // batch_size

    return {
        'lut_flops': lut_flops,
        'groupsum_flops': groupsum_flops,
        'total_flops': total,
        'flops_per_sample': per_sample,
        'flops_per_gate': flops_per_gate,
    }


def count_flops_binarybrain(total_gates: int, n_inputs: int,
                             in_dim: int = 784,
                             frame_modulation_size: int = 15,
                             batch_size: int = 1) -> dict:
    """
    BinaryBrainモデルのforward FLOPS計算。

    frame_modulation_sizeでバッチが膨張するため、
    LUT/BN/Binarize全てのFLOPSがframe_modulation_size倍になる。

    Args:
        total_gates: 総ロジックゲート数
        n_inputs: LUTの入力数
        in_dim: 入力次元数
        frame_modulation_size: バイナリ変調フレーム数
        batch_size: 元のバッチサイズ

    Returns:
        各コンポーネントのFLOPSと合計
    """
    M = frame_modulation_size
    effective_batch = batch_size * M
    flops_per_gate = lut_flops_per_gate(n_inputs)

    # RealToBinary: 比較のみ（in_dim × M × batch_size）
    rtb_flops = in_dim * M * batch_size

    # LUT: バッチが15倍
    lut_flops = total_gates * flops_per_gate * effective_batch

    # Unbinarize: 比較+加算 = 2 FLOPs/要素、各層の入力次元分
    # 近似: 総ゲート数分（各ゲートのN入力で実行されるがゲート数で近似）
    unbinarize_flops = 2 * total_gates * n_inputs * effective_batch // total_gates * total_gates
    # 簡略化: unbinarizeは入力要素ごとに2 FLOPs
    unbinarize_flops = 2 * total_gates * effective_batch

    # BatchNorm: (x - mean) * rstd * gamma + beta = 4 FLOPs/要素
    bn_flops = 4 * total_gates * effective_batch

    # Binarize: 比較 = 1 FLOP/要素
    binarize_flops = 1 * total_gates * effective_batch

    # BinaryToReal: reshape + mean = M加算 + 1除算 ≒ M FLOPs/出力要素
    btr_flops = 10 * M * batch_size

    total = rtb_flops + lut_flops + unbinarize_flops + bn_flops + binarize_flops + btr_flops
    per_sample = total // batch_size

    return {
        'rtb_flops': rtb_flops,
        'lut_flops': lut_flops,
        'unbinarize_flops': unbinarize_flops,
        'bn_flops': bn_flops,
        'binarize_flops': binarize_flops,
        'btr_flops': btr_flops,
        'total_flops': total,
        'flops_per_sample': per_sample,
        'flops_per_gate': flops_per_gate,
        'frame_modulation_size': M,
    }


def format_flops(flops: int) -> str:
    """FLOPSを人間が読みやすい形式に変換"""
    if flops >= 1e9:
        return f'{flops / 1e9:.2f} GFLOPS'
    elif flops >= 1e6:
        return f'{flops / 1e6:.2f} MFLOPS'
    elif flops >= 1e3:
        return f'{flops / 1e3:.2f} KFLOPS'
    return f'{flops} FLOPS'
