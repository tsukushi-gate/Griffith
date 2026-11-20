"""
CUDAカーネルのJITコンパイルラッパー。

CUDA対応環境では torch.utils.cpp_extension.load でカーネルをコンパイルし、
非対応環境では None を返してPythonフォールバックに任せる。
"""

import os

_module = None
_load_attempted = False


def get_cuda_module():
    """CUDAモジュールを取得。非対応環境ではNoneを返す。"""
    global _module, _load_attempted

    if _load_attempted:
        return _module

    _load_attempted = True

    try:
        import torch
        if not torch.cuda.is_available():
            print("[griffith CUDA] CUDA未対応: Pythonフォールバックを使用")
            return None

        from torch.utils.cpp_extension import load

        _dir = os.path.dirname(os.path.abspath(__file__))
        sources = [
            os.path.join(_dir, 'lut_bindings.cpp'),
            os.path.join(_dir, 'lut_kernels.cu'),
        ]

        # 拡張名にソース内容のハッシュを含め、内容ベースでキャッシュを無効化する。
        # ninjaのmtime比較はマウント環境(Modal等)でタイムスタンプが保存されず
        # 古いバイナリを使い続けることがあるため
        import hashlib
        h = hashlib.sha256()
        for src in sources:
            with open(src, 'rb') as f:
                h.update(f.read())
        name = f'griffith_lut_cuda_{h.hexdigest()[:12]}'

        print("[griffith CUDA] カーネルをコンパイル中...")
        _module = load(
            name=name,
            sources=sources,
            extra_cuda_cflags=['-O3', '-lineinfo'],
            verbose=False,
        )
        print("[griffith CUDA] コンパイル完了 — CUDAカーネルを使用")

    except Exception as e:
        print(f"[griffith CUDA] コンパイル失敗 — Pythonフォールバック: {e}")
        _module = None

    return _module


def cuda_available() -> bool:
    """CUDAカーネルが利用可能かどうか。"""
    return get_cuda_module() is not None
