"""Webランタイムの等価性ガード(K36-Py)。

`ir/simulate.py`(numpyのgolden model)を基準に、`runtime/web/dist/verify.mjs`
(TS製、Node上で動く)を実際に起動して**整数厳密**に突き合わせる。
`deploy/verify.py`(Cランタイム版)と同じ教訓に従い、ゴールデンは
常に検証対象自身(バンドル)からではなく `ir/simulate.py` から作る。

Web版で特に注意する点:
    - 画像前処理の**Python側の基準**は、バンドルに焼き込まれた
      `preproc_desc` をそのまま使わず、`preproc` 宣言から
      `build_preproc_descriptors` で**作り直した**記述子を使う
      (`preproc_desc` 自体が壊れていても両側が同じように壊れて
      一致してしまう、というC版と同じ穴を避ける)
    - `verify.mjs` はTS側で画像から二値ビットへの符号化を**独自に**行う
      (Pythonの `encode_bits` を経由しない)。生画像を渡してTS側の
      符号化結果(が生む出力)とPython側の記述子実行結果が一致するかを
      突き合わせるのがこの経路の狙い。ここが一致しないのはTS側の
      符号化実装がPython側の記述子と食い違っている、という重大な
      シグナルなので絶対に握りつぶさない
    - `verify.mjs` の `ok` フラグは信用の起点にしない。子プロセスの
      `outputs` を必ずPython側で独立に `np.array_equal` で突き合わせ
      (二重化)、`ok=False` は追加のシグナルとして扱う
"""

import base64
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from ..ir.serialize import ir_from_dict
from ..ir.simulate import simulate
from .bundle import save_bundle
from .preproc_desc import SUPPORTED_KINDS, build_preproc_descriptors, encode_bits
from .webbackend import RUNTIME_WEB

__all__ = ['RUNTIME_WEB', 'find_node', 'verify_web_matches_simulate']


def find_node() -> str | None:
    return shutil.which('node')


def _bits_cases_doc(graph, n_samples: int, seed: int) -> tuple[dict, np.ndarray]:
    """ビット列モードの cases.json とゴールデン出力。"""
    in_width = int(graph.inputs[0].width)
    rng = np.random.default_rng(seed)
    bits = rng.integers(0, 2, size=(n_samples, in_width), dtype=np.uint8)
    expected = simulate(graph, bits)
    cases = []
    for i in range(n_samples):
        packed = np.packbits(bits[i], bitorder='little').tobytes()
        cases.append({
            'bits_b64': base64.b64encode(packed).decode('ascii'),
            'expected': expected[i].astype(np.int64).tolist(),
        })
    return {'mode': 'bits', 'cases': cases}, expected


def _image_cases_doc(graph, bundle: dict, n_samples: int, seed: int) -> tuple[dict, np.ndarray]:
    """画像モードの cases.json とゴールデン出力。

    基準は **preproc宣言から作り直した記述子**(バンドル内の preproc_desc
    は使わない、build_c_model/verify_c_matches_simulate と同じ理由)。
    """
    preproc = bundle.get('preproc') or {}
    if preproc.get('kind') not in SUPPORTED_KINDS:
        raise ValueError(
            f"preproc_desc があるのに preproc.kind={preproc.get('kind')!r} "
            'から記述子を作り直せない(検証の基準を作れない)')
    plan = build_preproc_descriptors(preproc)
    rng = np.random.default_rng(seed)
    images = rng.integers(
        0, 256,
        size=(n_samples, plan['image_channels'], plan['height'], plan['width']),
        dtype=np.uint8)
    bits = encode_bits(plan, images)
    expected = simulate(graph, bits)
    cases = []
    for i in range(n_samples):
        cases.append({
            'image_b64': base64.b64encode(images[i].tobytes()).decode('ascii'),
            'expected': expected[i].astype(np.int64).tolist(),
        })
    return {'mode': 'image', 'cases': cases}, expected


def _run_verify_mjs(node_bin: str, verify_mjs: Path, bundle_path: Path,
                    cases_path: Path, *, engine: str, mode: str,
                    timeout: float = 300.0) -> tuple[dict, int]:
    cmd = [node_bin, str(verify_mjs), str(bundle_path), str(cases_path),
           '--engine', engine]
    try:
        proc = subprocess.run(cmd, capture_output=True, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f'verify.mjs がタイムアウトした(経路={mode}, engine={engine}, '
            f'{timeout}秒): {exc}') from exc
    out = proc.stdout.decode('utf-8', errors='replace').strip()
    err = proc.stderr.decode('utf-8', errors='replace').strip()
    if not out:
        raise RuntimeError(
            f'verify.mjs が標準出力を返さなかった(経路={mode}, '
            f'exit={proc.returncode}): {err}')
    # 仕様は「1行のJSON」だが、念のため最終行を取る(空行/ログ混入に強くする)。
    line = out.splitlines()[-1]
    try:
        result = json.loads(line)
    except ValueError as exc:
        raise RuntimeError(
            f'verify.mjs の出力がJSONでない(経路={mode}, '
            f'exit={proc.returncode}): {out}\n{err}') from exc
    return result, proc.returncode


def _check_case_result(result: dict, expected: np.ndarray, returncode: int,
                       *, mode: str, n_out: int) -> None:
    """child(verify.mjs)の結果をゴールデンと**独立に**突き合わせる(二重化)。

    `result['ok']` は最後の追加チェックとして使うだけで、一致判定の主体は
    常に `np.array_equal` によるPython側の再比較。
    """
    want = np.asarray(expected, dtype=np.int64)
    outputs = result.get('outputs')
    if outputs is None or len(outputs) != want.shape[0]:
        raise ValueError(
            f'web ⇔ simulate が一致しない(経路={mode}): verify.mjs の出力件数が'
            f"期待({want.shape[0]}件)と違う: outputs={outputs!r} "
            f"ok={result.get('ok')} error={result.get('error')!r} "
            f'exit={returncode}')
    got = np.asarray(outputs, dtype=np.int64)
    if got.shape != want.shape:
        raise ValueError(
            f'web ⇔ simulate が一致しない(経路={mode}): 出力形状 {got.shape} が'
            f"期待 {want.shape} と違う(ok={result.get('ok')} "
            f"error={result.get('error')!r})")
    if not np.array_equal(got, want):
        bad_rows = np.nonzero(np.any(got != want, axis=1))[0]
        i = int(bad_rows[0])
        row_bad = np.nonzero(got[i] != want[i])[0]
        raise ValueError(
            f'web ⇔ simulate が一致しない(サンプル{i}、経路={mode}): '
            f'{len(row_bad)}/{n_out} 要素が不一致。'
            f'最初の不一致 index={int(row_bad[0])} '
            f'web={int(got[i, row_bad[0]])} simulate={int(want[i, row_bad[0]])}')
    if not result.get('ok', False) or returncode != 0:
        raise ValueError(
            f'web ⇔ simulate が一致しない(経路={mode}): Python側の再比較は'
            f'一致したが verify.mjs 自身は ok=False を報告した(exit={returncode}, '
            f"error={result.get('error')!r})。TS側検証ロジックとの食い違いを"
            '無視できない')


def verify_web_matches_simulate(bundle: dict, *, n_samples: int = 3,
                                seed: int = 0, engine: str = 'js',
                                node: str | None = None, work_dir=None,
                                verbose: bool = True,
                                dense_flat: str | None = None,
                                dense_flat_min_reduction: float | None = None
                                ) -> dict:
    """Webランタイム(Node上のverify.mjs) ⇔ ir/simulate を整数厳密に突き合わせる。

    ビット列モードは常に検証する。バンドルが `preproc_desc` を持つ場合は
    追加で画像モードも検証する(生画像を渡し、TS側の符号化とPython側の
    記述子実行がそれぞれ独立に導いた `simulate` 結果と一致するかを見る)。

    ``dense_flat``: ``'auto'|'on'|'off'`` を渡すと `codegen-web` と同じ
    判定(`resolve_dense_flat(target='web')`)で、Web へ渡すバンドルだけを
    メモリ上で平坦化する。**ゴールデンは常に元バンドルの IR の simulate**
    なので、TS 側 `lut_flat` 実行器の lossless 性がこの関門で実証される。
    None(既定)は ``bundle`` をそのまま渡す。

    Returns:
        {'n_samples', 'n_out', 'mode': 'bits'|'bits+image', 'engine',
         'max_abs_diff': 0, 'dense_flat': decision|None}

    Raises:
        ValueError: 1サンプルでも不一致があった場合(経路名を含む)
        RuntimeError: node/verify.mjs が見つからない、または実行に失敗した場合
    """
    node_bin = node or find_node()
    if node_bin is None:
        raise RuntimeError(
            'node が見つからない(`md deploy verify-web` には Node.js が必要)')
    verify_mjs = RUNTIME_WEB / 'dist' / 'verify.mjs'
    if not verify_mjs.is_file():
        raise RuntimeError(
            'runtime/web/dist/verify.mjs が無い。先に `make web-runtime`'
            '(= npm --prefix tools/webbench run build:runtime)でビルドすること。'
        )

    graph = ir_from_dict(bundle['ir'])      # ゴールデン(元バンドルのIR)
    n_out = int(graph.outputs[0].width)

    web_bundle, flat_decision = bundle, None
    if dense_flat is not None:
        from .flatten_dense import describe_dense_flat, resolve_dense_flat
        web_bundle, flat_decision = resolve_dense_flat(
            bundle, mode=dense_flat, min_reduction=dense_flat_min_reduction,
            target='web')
        if verbose:
            print('  ' + describe_dense_flat(flat_decision))

    tmp = None
    if work_dir is None:
        tmp = tempfile.TemporaryDirectory(prefix='mdl_verify_web_')
        work_dir = tmp.name
    try:
        work = Path(work_dir)
        work.mkdir(parents=True, exist_ok=True)
        bundle_path = save_bundle(web_bundle, work / 'model.bundle.json')

        modes: list[str] = []

        bits_doc, bits_expected = _bits_cases_doc(graph, n_samples, seed)
        bits_cases_path = work / 'cases_bits.json'
        bits_cases_path.write_text(json.dumps(bits_doc), encoding='utf-8')
        result, rc = _run_verify_mjs(
            node_bin, verify_mjs, bundle_path, bits_cases_path,
            engine=engine, mode='bits')
        _check_case_result(result, bits_expected, rc, mode='bits', n_out=n_out)
        if verbose:
            print(f'  経路=bits: 一致({n_out}要素 × {n_samples}サンプル、'
                  f'engine={engine})')
        modes.append('bits')

        if bundle.get('preproc_desc') is not None:
            image_doc, image_expected = _image_cases_doc(
                graph, bundle, n_samples, seed)
            image_cases_path = work / 'cases_image.json'
            image_cases_path.write_text(json.dumps(image_doc), encoding='utf-8')
            result, rc = _run_verify_mjs(
                node_bin, verify_mjs, bundle_path, image_cases_path,
                engine=engine, mode='image')
            _check_case_result(result, image_expected, rc, mode='image',
                               n_out=n_out)
            if verbose:
                print(f'  経路=image: 一致({n_out}要素 × {n_samples}サンプル、'
                      f'engine={engine})')
            modes.append('image')
    finally:
        if tmp is not None:
            tmp.cleanup()

    return {'n_samples': n_samples, 'n_out': n_out, 'mode': '+'.join(modes),
            'engine': engine, 'max_abs_diff': 0, 'dense_flat': flat_decision}
