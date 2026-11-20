"""バンドル → Web(ブラウザ)配布物(プラン層に相乗り)。

C/CUDA/Metalは「バンドル → 自己完結したソース」を**生成**するが、Webは
実行体を最初から持っている(`runtime/web` = `tools/webbench` がビルドする
TypeScript製のWASM/WebGPUエンジン群)。したがってこのバックエンドが
やることは他の3つと非対称:

    - `WebBackend` は**申告だけ**する薄いクラス。EMITTERSはTSエンジンが
      実行できるプラン層のop語彙(C/CUDA/Metalと同一の粒度)。Pythonは
      コードを1行も生成しない
    - `generate_web` はソースコード生成ではなく、**バンドルそのものを
      配布用JSONとして書き出す**(TSは自前でパースしてグラフを組む)

申告(EMITTERS)と実装(TS側が実際に何を実行できるか)が一致しているかは、
このファイルだけでは保証できない — Python↔TSをまたぐ突き合わせは
`md deploy verify-web`(別ウェーブで実装予定)が担う。TSエンジンは実際には
add / binarize / constant のような下位演算も個別に実行できるが、
`can_run` が見るのはプラン層のop語彙(gate_program / plane_op…)なので、
ここでの申告もその粒度に保守的に合わせる(過大宣言をしない)。

web はバンドルの `programs`(命令列化済みconv2d)を**そのまま**消費する。
C/CUDA/Metalと同様、簡約(ゲート木→命令列への圧縮)はバンドル生成時
(`md deploy bundle`)に済ませてあり、ブラウザ側では二度と行わない。
"""

import json
from pathlib import Path

__all__ = ['WebBackend', 'generate_web', 'RUNTIME_WEB']

#: runtime/web の在り処。テストからmonkeypatchできるようモジュール変数にする。
RUNTIME_WEB = Path(__file__).resolve().parents[2] / 'runtime' / 'web'


class WebBackend:
    """Webランタイム(TS)が実行できるop種別の申告。

    `capabilities('web')` / `can_run(bundle, 'web')` はこのクラスの
    `EMITTERS` / `SUB_OPS` からそのまま能力を導く(backends.pyの規約)。
    実処理(コード生成)を持たない点がCBackend/GpuBackendと違う唯一の点。
    """

    name = 'web'

    #: プラン層のop語彙。CBackend.EMITTERS のキー集合と同一。preprocはplan.opsに現れない
    #: (plan.preprocに乗る別経路)ので、ここにも含めない。
    EMITTERS = frozenset({
        'pack', 'unpack', 'gate_program', 'plane_op', 'reduce_scores',
        'lut_dense', 'copy_scores', 'to_scores',
        # 縮約済み dense 平坦ネットリスト(engines/lutFlat.ts)
        'lut_flat',
    })

    #: plane_op / reduce_scores の下位種別。CBackend.SUB_OPS と同一
    #: (cbackend.py:59-62)。
    SUB_OPS = frozenset({
        'or_pool2x2', 'upsample2d_x2', 'permute_channels', 'concat_planes',
        'spatial_group_sum', 'group_sum',
    })


def generate_web(bundle: dict, name: str = 'model', *,
                 slim: bool = False) -> dict[str, str]:
    """バンドル → Web配布物(コンパクトJSON。TSがそのまま読み込む)。

    既定(slim=False)は配布バンドルをそのまま書き出す。`save_bundle` と
    同じ整形(`ensure_ascii=False, separators=(',',':')`)を使うので、
    バイト数は `md deploy bundle` の出力と一致する。

    Args:
        slim: Trueなら配布バンドルではなく、bundle_format / bundle_version /
            programs / preproc_desc を持たない**素のwebbenchモデルdoc**を
            書き出す。programsを落とすということは、conv2dの命令列化
            (簡約)をブラウザ側でIRから組み直す前提になる、ということ。
            つまり**エクスポータ側で完結させていた簡約保証を放棄する**
            選択であり、それはもはや妥当な配布バンドルではない
            (`load_bundle`/`validate_bundle`には通せない)。通常配布は
            False を使い、slimはIRを直接触るデバッグ/実験用途に限る。

    Returns:
        {ファイル名: 中身(JSON文字列)}。slim時は `{name}.json`、
        通常時は `{name}.bundle.json`。
    """
    if slim:
        doc = {k: v for k, v in bundle.items()
               if k not in ('bundle_format', 'bundle_version', 'programs',
                            'preproc_desc')}
        payload = json.dumps(doc, ensure_ascii=False, separators=(',', ':'))
        return {f'{name}.json': payload}
    payload = json.dumps(bundle, ensure_ascii=False, separators=(',', ':'))
    return {f'{name}.bundle.json': payload}
