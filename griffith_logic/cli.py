"""griffith 統一CLI(AI駆動実験ループ)。

サブコマンド:
    griffith train [config.yaml] [key=value ...]   学習(共通ランナー、--strictで期待値照合)
        --from-project <project.json> エディタプロジェクトから学習設定を解決
    griffith eval <checkpoint> [--dataset <name>]  hard eval(離散化精度)
    griffith export-hdl <ir.json> --lang ... -o    IR→Verilog/VHDL
    griffith deploy bundle|info                    多環境デプロイ用の配布バンドル(K1)
    griffith editor                                エディタバックエンド起動
    griffith sweep <sweep.yaml>                    wandb sweep作成+agent実行
    griffith launch --grid "..." [config.yaml] グリッド実験の一括投入
        --backend modal|runpod               投入先クラウドの選択
    griffith runpod ls|stop|gpus                   RunPod稼働pod一覧・停止・GPU型ID一覧
    griffith collect [--dest DIR]                  Modal Volumeから成果物回収
    griffith runs list / compare                   run一覧・比較表
    griffith report <run_id...>                    wandbレポート/markdown生成
    griffith bench <recipe.yaml|checkpoint> ベンチマーク(--skip-train対応)
    griffith --version                             バージョン表示

`griffith editor` / `griffith launch` はリポジトリの `tools/` `scripts/` を必要とする。
pip インストールのみの環境では利用できず、その旨のガイダンスを表示して終了する
(リポジトリのパスは環境変数 GRIFFITH_REPO_ROOT でも指定できる)。

コマンド名は `griffith`。旧名 `md` は後方互換エイリアスとして残している。

依存を増やさないためargparseサブコマンドで実装する。
テストからは main(argv) を直接呼べる(戻り値=exit code)。
実装はディスパッチのみとし、ロジックは training/ 配下の各モジュールに置く。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Sequence

EDITOR_DEFAULT_PORT = 8765


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog='griffith',
        description=(
            'Griffith 統一CLI。微分可能論理ゲートNNの学習・評価・'
            'HDLエクスポート・エディタ起動をひとつの入口から実行する。'
            '(旧名 `md` も後方互換エイリアスとして同じCLIを起動する)'
        ),
    )
    from . import __version__

    parser.add_argument(
        '--version', action='version', version=f'griffith-logic {__version__}',
        help='バージョンを表示して終了する',
    )
    sub = parser.add_subparsers(dest='command', metavar='サブコマンド')

    # ---- griffith train ----
    train_p = sub.add_parser(
        'train',
        help='学習を実行する(設定YAML+ドット記法オーバーライド)',
        description=(
            '共通ランナーで学習を実行する。設定はYAMLファイルと '
            'key=value 形式のドット記法オーバーライドで指定する。\n'
            '例: griffith train experiments/configs/mnist.yaml model.k=8000 run.epochs=5'
        ),
    )
    train_p.add_argument(
        'config', nargs='?', default=None,
        help='設定YAMLのパス(省略時は既定値。key=value形式ならオーバーライド扱い)',
    )
    train_p.add_argument(
        'overrides', nargs='*',
        help='ドット記法の上書き(例: model.k=8000 run.epochs=5 data.name=logic:xor)',
    )
    train_p.add_argument(
        '--from-run', metavar='DIR',
        help='既存ランのディレクトリ(中のconfig.yamlを読んで再実行する)',
    )
    train_p.add_argument(
        '--from-project', metavar='PATH',
        help=(
            'エディタプロジェクトJSON(outputs/editor_projects/*.json、形式)'
            'から学習設定を解決して再現実行する'
        ),
    )
    train_p.add_argument(
        '--network', metavar='NAME', default=None,
        help='--from-project時に使うネットワーク名(省略時は先頭ネットワーク)',
    )
    train_p.add_argument(
        '--config', dest='project_config_name', metavar='NAME', default=None,
        help='--from-project時に使う学習設定名(省略時はglobal)',
    )
    train_p.add_argument(
        '--optimizer', metavar='NAME', default=None,
        help='--from-project時に適用するオプティマイザ定義名(省略時は適用しない)',
    )
    train_p.add_argument(
        '--out-dir', metavar='DIR',
        help='出力先ディレクトリ(省略時は outputs/<timestamp>_<run名>)',
    )
    train_p.add_argument(
        '--strict', action='store_true',
        help='expected_metrics未達のとき非0で終了する',
    )

    # ---- griffith eval ----
    eval_p = sub.add_parser(
        'eval',
        help='チェックポイントをhard eval(離散化評価)する',
        description=(
            'griffith train が保存したチェックポイントを読み込み、離散化した回路の '
            '精度(hard acc)を出力する。モデル構成はチェックポイント内の設定から復元する。'
        ),
    )
    eval_p.add_argument('checkpoint', help='チェックポイント(.pt)のパス')
    eval_p.add_argument(
        '--dataset', default=None,
        help='評価データセット名(省略時はチェックポイントの学習時設定)',
    )
    eval_p.add_argument('--batch-size', type=int, default=None, help='評価バッチサイズ')
    eval_p.add_argument('--device', default=None, help='デバイス(省略時は自動検出)')

    # ---- griffith export-hdl ----
    hdl_p = sub.add_parser(
        'export-hdl',
        help='IR(JSON)からVerilog/VHDLを生成する',
        description='IR(中間表現)JSONを検証し、Verilog/VHDLへトランスパイルする。',
    )
    hdl_p.add_argument('source', help='IR JSONファイルのパス')
    hdl_p.add_argument(
        '--lang', choices=['verilog', 'vhdl'], default='verilog',
        help='生成するHDL言語(default: verilog)',
    )
    hdl_p.add_argument(
        '-o', '--output', default=None,
        help='出力パス(省略時は入力と同じ場所に .v / .vhdl)',
    )

    # ---- griffith deploy (K1) ----
    deploy_p = sub.add_parser(
        'deploy',
        help='多環境デプロイ用の配布バンドルを作る/検証する',
        description=(
            '学習済みモデルを全ターゲット共通の配布バンドル(モデル+前処理+'
            'readout+最適化済み命令列)へまとめる。最適化はここで済ませ、'
            'ターゲット側ランタイムは実行するだけにする。'
        ),
    )
    deploy_sub = deploy_p.add_subparsers(
        dest='deploy_command',
        metavar=('bundle|codegen-c|codegen-cuda|codegen-metal|codegen-web|'
                 'verify-c|verify-metal|verify-web|info'),
    )
    deploy_bundle_p = deploy_sub.add_parser(
        'bundle', help='学習済み.pt/モデルJSON → 配布バンドルv1(conv2d命令列を焼き込む)',
    )
    deploy_bundle_p.add_argument(
        'source',
        help='学習済みチェックポイント(.pt/.pth、griffith trainの出力)'
             'またはモデルJSON(webbench形式)のパス',
    )
    deploy_bundle_p.add_argument(
        '--config', default=None, metavar='YAML',
        help='.pt入力時のモデル構成(学習時の設定ファイル)。省略時は'
             'チェックポイント内に保存された設定を使う',
    )
    deploy_bundle_p.add_argument(
        '-o', '--output', default=None,
        help='出力パス(省略時は入力と同じ場所に .bundle.json)',
    )
    deploy_bundle_p.add_argument(
        '--gzip', action='store_true', help='gzip版も書き出して配信サイズを測る',
    )
    deploy_bundle_p.add_argument(
        '--format', choices=['json', 'mdlb'], default=None,
        help='出力形式(既定: 出力パスの拡張子から自動判定。.mdlbならバイナリ'
             '形式、それ以外はJSON。-o省略時のデフォルト出力名にも影響する。'
             'バイナリ形式の仕様は docs/reference/deploy-bundle.md 参照)',
    )
    deploy_bundle_p.add_argument(
        '--no-verify', action='store_true',
        help='命令列の全数等価性検証を省略する(デバッグ用。通常は使わない)',
    )
    deploy_bundle_p.add_argument(
        '--compress', choices=['none', 'fast', 'max'], default='fast',
        help='回路圧縮(lossless、精度不変)の自動適用レベル(default: fast)。'
             'モデルの種類によらず同じ意味: fast=数秒〜十数秒で終わる削減を全部'
             '(dense は sweep、conv2d は木の再合成〈探索上限4・軽量検証〉)、'
             'max=fast + 時間をかける探索と ir/simulate 照合(分オーダー)、'
             'none=IR を書き換えない。conv2d 木の命令列化はどの level でも行う。'
             '結果は about.compress に記録し info が表示する。'
             '個別フラグ --sweep/--no-sweep/--resynth は --compress より優先',
    )
    deploy_bundle_p.add_argument(
        '--resynth', choices=['lossless'], default=None,
        help='conv2d木の再合成を強制する(lossless=精度不変)。現在は既定の '
             '--compress fast でも軽量検証で自動適用されるので、通常は不要。'
             'こちらは ir/simulate による照合も行う(実モデルで分オーダー)',
    )
    deploy_bundle_p.add_argument(
        '--resynth-max-cost', type=int, default=None, metavar='N',
        help='再合成の探索ゲート数上限(default: --compress fast は 4、'
             'max と --resynth lossless は 5)',
    )
    deploy_bundle_p.add_argument(
        '--resynth-verify-samples', type=int, default=8, metavar='N',
        help='再合成の検証に使うランダム入力サンプル数(default: 8)',
    )
    sweep_grp = deploy_bundle_p.add_mutually_exclusive_group()
    sweep_grp.add_argument(
        '--sweep', action='store_true',
        help='バンドル化の前に dense(lut_layer列)IR を sweep して定数・重複・'
             '未参照ゲートを除去する(lossless=任意入力で出力不変。整数一致検証'
             '込み。conv2dを含むIRは拒否するので conv は --resynth を使う)。'
             '--compress none でも強制的に適用する(既定の fast では自動適用)',
    )
    sweep_grp.add_argument(
        '--no-sweep', action='store_true',
        help='--compress fast/max の自動 sweep を行わない',
    )
    deploy_bundle_p.add_argument(
        '--sweep-verify-samples', type=int, default=8, metavar='N',
        help='sweep の検証に使うランダム入力サンプル数(default: 8。'
             'demo_samples があれば実データでも追加検証する)',
    )
    deploy_bundle_p.add_argument(
        '--dense-flat', action='store_true',
        help='dense(lut_layer列+group_sum)の連続する2入力lut_layerを、定数・'
             'パススルー・NOT・CSE・未参照を畳み込んだ平坦ネットリスト(lut_flatノード)'
             'へ置換する(lossless。変換前後のsimulate整数一致検証込み。'
             '焼き込んだバンドルは C/CUDA/Metal/Web のどれでも実行できる。'
             '通常は付けなくてよい: codegen-c/codegen-web は生成時に自動判定する'
             '(--dense-flat auto)。--resynth/--sweepと併用可、'
             '適用順は resynth → sweep → dense-flat)',
    )
    deploy_bundle_p.add_argument(
        '--dense-flat-verify-samples', type=int, default=16, metavar='N',
        help='--dense-flat の検証に使うランダム入力サンプル数(default: 16。'
             'demo_samples があれば実データでも検証する)',
    )
    deploy_codegen_p = deploy_sub.add_parser(
        'codegen-c', help='バンドル → 自己完結したCソース(組み込み/モバイル向け)',
    )
    deploy_codegen_p.add_argument('source', help='バンドルJSONのパス')
    deploy_codegen_p.add_argument(
        '-o', '--output', required=True, metavar='DIR', help='出力ディレクトリ',
    )
    deploy_codegen_p.add_argument(
        '--name', default='model', help='生成ファイル名の基底(default: model)',
    )
    deploy_codegen_p.add_argument(
        '--no-runtime', action='store_true',
        help='runtime/c のカーネルをコピーしない(既にリンク済みの場合)',
    )
    deploy_codegen_p.add_argument(
        '--fuse', action='store_true',
        help='ゲート命令列を直線コードへ展開する(実測で conv 系モデルが'
             '2〜9%%速い代わりに生成Cが1.2〜2.3倍になる)',
    )
    deploy_codegen_p.add_argument(
        '--dense-batch', type=int, default=None, metavar='N',
        help='dense専用モデルにバッチ軸パック経路(mdl_infer_batch_packed、'
             '1語=32枚)を生やす。N は1回に渡せる最大枚数、0 で生やさない。'
             '既定は平坦化(lut_flat)を採用したときだけ N=32 で自動生成'
             '(表を共有するので flash はほぼ増えない。実測で 32 枚 1.7〜5.9 倍の'
             'スループット)。conv を含むモデルでは無視される',
    )
    deploy_codegen_p.add_argument(
        '--dense-flat', choices=['auto', 'on', 'off'], default='auto',
        help='dense の lut_layer 列を縮約済みの平坦ネットリスト(lut_flat)で'
             '実行するか(default: auto)。バンドルは変えず C 生成時にメモリ上で'
             '平坦化する。auto=試行して実効ゲート削減率が閾値以上なら採用、'
             'on=常に採用、off=従来のbit-sliceカーネル。採否と削減率は標準出力と'
             'model.h の先頭コメント・MDL_DENSE_FLAT マクロに出す。'
             '作業RAM(MDL_CTX_BYTES)は増えるので RAM 優先なら off',
    )
    deploy_codegen_p.add_argument(
        '--dense-flat-min-reduction', type=float, default=None, metavar='R',
        help='--dense-flat auto の採用閾値(実効ゲート削減率、default: 0.25 '
             '= plan.DENSE_FLAT_AUTO_MIN_REDUCTION。ホスト実測の損益分岐は約26%%)',
    )

    deploy_cuda_p = deploy_sub.add_parser(
        'codegen-cuda', help='バンドル → CUDAソース(NVIDIA GPU向け)',
    )
    deploy_cuda_p.add_argument('source', help='バンドルJSONのパス')
    deploy_cuda_p.add_argument(
        '--dense-flat', choices=['auto', 'on', 'off'], default='off',
        help='dense の 2入力lut_layer列を縮約済み平坦ネットリスト(lut_flat)へ'
             '置き換えて生成するか(default: off = 置き換えない。opt-in)。'
             'auto = 実効ゲート削減率が閾値(0.25)以上なら採用、on = 常に採用。'
             'Metal 実測では大きいモデル(HAR k6000、batch 128 以上)で平坦化しない'
             '最良(dense_fuse/dense_packed)の 0.47〜0.71 倍に落ちたため既定は off。'
             '--dense-fuse/--dense-packed を明示したときは平坦化しない',
    )
    deploy_cuda_p.add_argument(
        '--dense-flat-min-reduction', type=float, default=None, metavar='R',
        help='--dense-flat auto の採用閾値(実効ゲート削減率、default: 0.25)',
    )
    deploy_cuda_p.add_argument(
        '-o', '--output', required=True, metavar='DIR', help='出力ディレクトリ',
    )
    deploy_cuda_p.add_argument(
        '--name', default='model_cuda',
        help='生成ファイル名の基底(default: model_cuda)',
    )
    deploy_cuda_p.add_argument(
        '--no-runtime', action='store_true',
        help='runtime/cuda と runtime/gpu をコピーしない',
    )
    deploy_cuda_p.add_argument(
        '--batch', type=int, default=1, metavar='N',
        help='同時に流せる最大枚数(default: 1)。デバイスメモリはこの倍数に'
             'なるが、conv のスレッド数が枚数倍になるのでGPUでは大きいほど'
             '速い。実行時の枚数は 1..N で可変',
    )
    deploy_cuda_p.add_argument(
        '--n-streams', type=int, default=1, metavar='N', dest='n_streams',
        help='ストリーム・パイプライン化(Metal版ミラー、'
             'default: 1=従来どおり)。2以上でバッファ集合をN本複製し、'
             'mdl_cuda_infer_batch_submit/_collect(commitしてから待たずに'
             '次を積める。CUDAはcudaStream_t+cudaEventベース)を追加生成'
             'する。デバイスメモリは batch*N 倍になる。も自動ONに'
             'しない(opt-inのまま、モデル依存で悪化する実測があるため)',
    )
    deploy_cuda_p.add_argument(
        '--conv-engine', choices=['split', 'combined'], default=None,
        dest='conv_engine',
        help='conv2dの実行方式を明示指定する(既定: 未指定。自動選択'
             '=auto ON時はmax_slots_per_oc>8のノードがあればcombinedへ'
             '自動フォールバックする、実測)',
    )
    deploy_cuda_p.add_argument(
        '--conv-codegen', action='store_true', dest='conv_codegen',
        help='conv2dノードごとにtt即値+スロット静的展開のモデル特化'
             'CUDAカーネルを生成する(既定OFF=インタプリタ'
             'mdl_k_conv_exec_split のまま。も自動ONにしない、'
             'e2e改善がほぼ無い割にコードサイズが1.78〜6.65倍になる実測'
             'のため)',
    )
    deploy_cuda_p.add_argument(
        '--dense-fuse', action='store_true', dest='dense_fuse',
        help='連続するlut_dense層+末尾group_sumを1カーネルへ融合'
             'する(GEM Boomerang型、活性を共有メモリへ)。共有メモリ上限は'
             'CUDAのstatic __shared__既定上限48KB(Metal版の32KBより緩い)。'
             '自動選択(既定ON)でbatch>=32かつ融合可能な鎖があれば'
             '自動的にも有効になる ─ このフラグは明示指定・auto無効時'
             '(--no-auto)向け',
    )
    deploy_cuda_p.add_argument(
        '--dense-packed', action='store_true', dest='dense_packed',
        help='dense(lut_dense列+末尾group_sumのみのモデル)のバッチ軸'
             'を32サンプル/語のビットプレーンへ詰め替える(既定OFF=1'
             'スレッド1ゲートのまま)。batch>=32・非画像入力・全主経路opが'
             'lut_dense/group_sumのみのモデルにのみ適用可(満たさなければ'
             'ValueError)。--dense-fuse/--n-streamsとの併用は未対応。'
             '自動選択(既定ON)ではbatch>=128かつdense_fuseが'
             '自動ONにならないときだけ自動的にも有効になる',
    )
    deploy_cuda_p.add_argument(
        '--no-auto', action='store_true', dest='no_auto',
        help='最適化の自動選択(conv_engine/dense_fuse/dense_packedの'
             '実測判定表、既定ON)を無効化する。無効化すると、明示指定'
             'しなかった引数は旧版の既定値(conv_engine=split・'
             'dense_fuse/dense_packed=OFF)のまま生成する。判定表の詳細は'
             'docs/guides/deploy-optimization.md参照',
    )

    deploy_metal_p = deploy_sub.add_parser(
        'codegen-metal', help='バンドル → Metalソース(Apple GPU向け)',
    )
    deploy_metal_p.add_argument('source', help='バンドルJSONのパス')
    deploy_metal_p.add_argument(
        '--dense-flat', choices=['auto', 'on', 'off'], default='off',
        help='dense の 2入力lut_layer列を縮約済み平坦ネットリスト(lut_flat)へ'
             '置き換えて生成するか(default: off = 置き換えない。opt-in)。'
             'auto = 実効ゲート削減率が閾値(0.25)以上なら採用、on = 常に採用。'
             'Metal 実測では大きいモデル(HAR k6000、batch 128 以上)で平坦化しない'
             '最良(dense_fuse/dense_packed)の 0.47〜0.71 倍に落ちたため既定は off。'
             '--dense-fuse/--dense-packed を明示したときは平坦化しない',
    )
    deploy_metal_p.add_argument(
        '--dense-flat-min-reduction', type=float, default=None, metavar='R',
        help='--dense-flat auto の採用閾値(実効ゲート削減率、default: 0.25)',
    )
    deploy_metal_p.add_argument(
        '-o', '--output', required=True, metavar='DIR', help='出力ディレクトリ',
    )
    deploy_metal_p.add_argument(
        '--name', default='model_metal',
        help='生成ファイル名の基底(default: model_metal)',
    )
    deploy_metal_p.add_argument(
        '--no-runtime', action='store_true',
        help='runtime/metal をコピーしない',
    )
    deploy_metal_p.add_argument(
        '--batch', type=int, default=1, metavar='N',
        help='同時に流せる最大枚数(default: 1)。デバイスメモリはこの倍数に'
             'なるが、conv のスレッド数が枚数倍になるのでGPUでは大きいほど'
             '速い。実行時の枚数は 1..N で可変',
    )
    deploy_metal_p.add_argument(
        '--n-streams', type=int, default=1, metavar='N', dest='n_streams',
        help='ストリーム・パイプライン化(default: 1=従来どおり)。'
             '2以上でバッファ集合をN本複製し、'
             'mdl_metal_infer_batch_submit/_collect(commitしてから待たずに'
             '次を積める)を追加生成する。デバイスメモリは batch*N 倍になる。'
             'も自動ONにしない(opt-inのまま、モデル依存で悪化する'
             '実測があるため)',
    )
    deploy_metal_p.add_argument(
        '--conv-engine', choices=['split', 'combined'], default=None,
        dest='conv_engine',
        help='conv2dの実行方式を明示指定する(既定: 未指定。自動選択'
             '=auto ON時はmax_slots_per_oc>8のノードがあればcombinedへ'
             '自動フォールバックする、実測)',
    )
    deploy_metal_p.add_argument(
        '--conv-codegen', action='store_true', dest='conv_codegen',
        help='conv2dノードごとにtt即値+スロット静的展開のモデル特化'
             'MSLを生成する(既定OFF=インタプリタ mdl_k_conv_exec_split の'
             'まま。も自動ONにしない、e2e改善がほぼ無い割にコード'
             'サイズが1.78〜6.65倍になる実測のため)',
    )
    deploy_metal_p.add_argument(
        '--dense-fuse', action='store_true', dest='dense_fuse',
        help='連続するlut_dense層+末尾group_sumを1カーネルへ融合する'
             '(GEM Boomerang型、活性をthreadgroupメモリへ)。実測'
             '(har_dlgn_k6000_gumbel、Apple M3): batch=1で0.81x(悪化)、'
             'batch=32で1.91x、batch=128で2.51x(最良)。自動選択'
             '(既定ON)でbatch>=32かつ融合可能な鎖があれば自動的にも'
             '有効になる ─ このフラグは明示指定・auto無効時(--no-auto)'
             '向け',
    )
    deploy_metal_p.add_argument(
        '--dense-packed', action='store_true', dest='dense_packed',
        help='dense(lut_dense列+末尾group_sumのみのモデル)のバッチ軸'
             'を32サンプル/語のビットプレーンへ詰め替える(既定OFF=1'
             'スレッド1ゲートのまま、バッチはただの平坦展開)。ゲート'
             'あたりの添字・tt読みが32サンプルで償却され算術強度が約32倍'
             'になる。batch>=32・非画像入力・全主経路opがlut_dense/'
             'group_sumのみのモデルにのみ適用可(満たさなければ'
             'ValueError)。--dense-fuse/--n-streamsとの併用は未対応。'
             '自動選択(既定ON)ではbatch>=128かつdense_fuseが'
             '自動ONにならないときだけ自動的にも有効になる',
    )
    deploy_metal_p.add_argument(
        '--no-auto', action='store_true', dest='no_auto',
        help='最適化の自動選択(conv_engine/dense_fuse/dense_packedの'
             '実測判定表、既定ON)を無効化する。無効化すると、明示指定'
             'しなかった引数は旧版の既定値(conv_engine=split・'
             'dense_fuse/dense_packed=OFF)のまま生成する。判定表の詳細は'
             'docs/guides/deploy-optimization.md参照',
    )

    deploy_web_p = deploy_sub.add_parser(
        'codegen-web', help='バンドル → Web(ブラウザ)配布物(+ ランタイムのコピー)',
    )
    deploy_web_p.add_argument('source', help='バンドルJSONのパス')
    deploy_web_p.add_argument(
        '-o', '--output', required=True, metavar='DIR', help='出力ディレクトリ',
    )
    deploy_web_p.add_argument(
        '--name', default='model', help='生成ファイル名の基底(default: model)',
    )
    deploy_web_p.add_argument(
        '--no-runtime', action='store_true',
        help='runtime/web/dist をコピーしない(既に配信済みの場合)',
    )
    deploy_web_p.add_argument(
        '--slim', action='store_true',
        help='配布バンドルでなくwebbenchモデルdocそのものを書き出す'
             '(ブラウザ側でIRから命令列を組み直す前提になる。エクスポータ側の'
             '簡約保証を放棄する選択なので、通常は使わない)',
    )
    deploy_web_p.add_argument(
        '--gzip', action='store_true', help='gzip版も書き出して配信サイズを測る',
    )
    deploy_web_p.add_argument(
        '--dense-flat', choices=['auto', 'on', 'off'], default='auto',
        help='dense の 2入力lut_layer列を縮約済み平坦ネットリスト'
             '(lut_flat)へ置き換えて配る(default: auto = 列があれば採用。'
             'Web は lut_flat が1ゲートあたりも速いので C と違い閾値 0)。'
             'off で従来の lut_layer 列のまま配る',
    )
    deploy_web_p.add_argument(
        '--dense-flat-min-reduction', type=float, default=None, metavar='R',
        help='--dense-flat auto の採用閾値(実効ゲート削減率、default: 0)',
    )

    deploy_verify_p = deploy_sub.add_parser(
        'verify-c', help='生成CをビルドしてIR simulateとの整数一致を検証する',
    )
    deploy_verify_p.add_argument('source', help='バンドルJSONのパス')
    deploy_verify_p.add_argument(
        '-n', '--samples', type=int, default=3, help='検証サンプル数(default: 3)',
    )
    deploy_verify_p.add_argument(
        '--seed', type=int, default=0, help='乱数シード(default: 0)',
    )
    deploy_verify_p.add_argument(
        '--dense-flat', choices=['auto', 'on', 'off'], default='auto',
        help='codegen-c と同じ dense-flat 判定で C を生成して検証する'
             '(default: auto)。ゴールデンは元バンドルIRの simulate なので、'
             '採用した平坦化が lossless であることの実証になる',
    )
    deploy_verify_p.add_argument(
        '--dense-flat-min-reduction', type=float, default=None, metavar='R',
        help='--dense-flat auto の採用閾値(default: 0.25)',
    )
    deploy_verify_p.add_argument(
        '--dense-batch', type=int, default=None, metavar='N',
        help='codegen-c と同じバッチ軸パック入口(mdl_infer_batch_packed)を生成して'
             'それも照合する(default: codegen-c と同じ = 平坦化を採用したときだけ 32、'
             '0 で生成しない)。入口があれば 1 枚ずつの mdl_infer に加えて、'
             '語境界を跨ぐ枚数(33 枚以上)でパック入口も simulate と整数照合する',
    )

    deploy_verify_metal_p = deploy_sub.add_parser(
        'verify-metal',
        help='生成Metal(Apple GPU)をビルドしてIR simulateとの整数一致を検証する',
    )
    deploy_verify_metal_p.add_argument('source', help='バンドルJSONのパス')
    deploy_verify_metal_p.add_argument(
        '--dense-flat', choices=['auto', 'on', 'off'], default='off',
        help='codegen-metal と同じ dense-flat 判定で生成して検証する'
             '(default: off = codegen-metal の既定と同じく平坦化しない)。'
             'auto = 実効ゲート削減率 0.25 以上なら採用、on = 常に採用。'
             'ゴールデンは元バンドルIRの simulate。--dense-fuse を明示したときは平坦化しない',
    )
    deploy_verify_metal_p.add_argument(
        '--dense-flat-min-reduction', type=float, default=None, metavar='R',
        help='--dense-flat auto の採用閾値(実効ゲート削減率、default: 0.25)',
    )
    deploy_verify_metal_p.add_argument(
        '-n', '--samples', type=int, default=3, help='検証サンプル数(default: 3)',
    )
    deploy_verify_metal_p.add_argument(
        '--seed', type=int, default=0, help='乱数シード(default: 0)',
    )
    deploy_verify_metal_p.add_argument(
        '--conv-codegen', action='store_true', dest='conv_codegen',
        help='モデル特化MSL生成(--conv-codegen)を検証する',
    )
    deploy_verify_metal_p.add_argument(
        '--dense-fuse', action='store_true', dest='dense_fuse',
        help='dense多層融合カーネル生成(--dense-fuse)を検証する',
    )

    deploy_verify_web_p = deploy_sub.add_parser(
        'verify-web',
        help='Webランタイム(Node上のverify.mjs)をIR simulateと整数一致で検証する',
    )
    deploy_verify_web_p.add_argument('source', help='バンドルJSONのパス')
    deploy_verify_web_p.add_argument(
        '-n', '--samples', type=int, default=3, help='検証サンプル数(default: 3)',
    )
    deploy_verify_web_p.add_argument(
        '--seed', type=int, default=0, help='乱数シード(default: 0)',
    )
    deploy_verify_web_p.add_argument(
        '--engine', choices=['js', 'wasm'], default='js',
        help='Webランタイムのエンジン(default: js)',
    )
    deploy_verify_web_p.add_argument(
        '--dense-flat', choices=['auto', 'on', 'off'], default='auto',
        help='codegen-web と同じ dense-flat 判定で Web へ渡すバンドルを作って'
             '検証する(default: auto)。ゴールデンは元バンドルIRの simulate',
    )
    deploy_verify_web_p.add_argument(
        '--dense-flat-min-reduction', type=float, default=None, metavar='R',
        help='--dense-flat auto の採用閾値(default: 0)',
    )

    deploy_info_p = deploy_sub.add_parser(
        'info', help='バンドルを検証して内容を要約表示する',
    )
    deploy_info_p.add_argument('source', help='バンドルJSON(.json/.json.gz)のパス')
    deploy_info_p.add_argument(
        '--strict', action='store_true',
        help='命令列をIRから再構築して完全一致まで確認する',
    )

    # ---- griffith editor ----
    editor_p = sub.add_parser(
        'editor',
        help='ビジュアルエディタのバックエンドを起動する',
        description=(
            'エディタバックエンド(FastAPI)をuvicornで起動する。'
            'フロントエンドは起動しないので、開発時は make editor を使うこと。'
        ),
    )
    editor_p.add_argument('--host', default='127.0.0.1', help='バインドするホスト')
    editor_p.add_argument(
        '--port', type=int, default=EDITOR_DEFAULT_PORT,
        help=f'ポート番号(default: {EDITOR_DEFAULT_PORT})',
    )

    # ---- griffith sweep ----
    sweep_p = sub.add_parser(
        'sweep',
        help='wandb sweepを作成してagentを実行する',
        description=(
            'sweep.yaml({method, metric, parameters, base_config})から'
            'wandb sweepを作成し、agentを実行する。\n'
            'クラウド実行は Modal 専用(--modal)。RunPodには非対応。\n'
            '例: griffith sweep configs/sweeps/example_mnist.yaml --count 4 --local'
        ),
    )
    sweep_p.add_argument('spec', help='sweep YAMLのパス')
    sweep_p.add_argument(
        '--count', type=int, default=None,
        help='実行する試行数(省略時はgrid全件/無制限)',
    )
    sweep_where = sweep_p.add_mutually_exclusive_group()
    sweep_where.add_argument(
        '--local', action='store_true', help='このプロセスでagentを実行(既定)',
    )
    sweep_where.add_argument(
        '--modal', action='store_true', help='Modal上でagentを実行(GPU=MODAL_GPU)',
    )
    sweep_p.add_argument(
        '--parallel', type=int, default=1, metavar='N',
        help='--modal時のagent並列数(default: 1。試行数は--countをN分割)',
    )

    # ---- griffith launch ----
    launch_p = sub.add_parser(
        'launch',
        help='グリッド実験を一括投入する(デカルト積展開)',
        description=(
            '--gridのリスト値を全組合せ展開して学習を投入する。\n'
            '例: griffith launch --grid "optim.lr=[0.001,0.01] model.k=[64,256]" '
            'configs/recipes/dlgn_mnist_quick.yaml --modal'
        ),
    )
    launch_p.add_argument(
        '--grid', required=True, metavar='SPEC',
        help='掃引指定(例: "optim.lr=[0.001,0.01] model.k=[64,256]"。'
             'スカラー値は固定オーバーライド)',
    )
    launch_p.add_argument(
        'config', nargs='?', default=None, help='ベース設定YAML(省略可)',
    )
    launch_p.add_argument(
        '--backend', choices=['local', 'modal', 'runpod'], default='local',
        help='投入先バックエンド(default: local。クラウド投入は明示指定)。'
             '--local/--modalが優先される',
    )
    launch_where = launch_p.add_mutually_exclusive_group()
    launch_where.add_argument(
        '--local', action='store_true', help='このプロセスで順次実行(既定)',
    )
    launch_where.add_argument(
        '--modal', action='store_true', help='Modalへ並列spawnして全完了を待つ',
    )
    launch_p.add_argument(
        '--dry-run', action='store_true', help='展開結果の一覧表示のみ(実行しない)',
    )

    # ---- griffith collect ----
    collect_p = sub.add_parser(
        'collect',
        help='Modal Volumeからoutputs/のrunディレクトリを回収する',
        description=(
            'Modal Volume(既定 griffith-workspace)の outputs/ 配下から、'
            'ローカル未取得のrunディレクトリをダウンロードする。'
        ),
    )
    collect_p.add_argument(
        '--dest', default='outputs/collected', metavar='DIR',
        help='回収先ディレクトリ(default: outputs/collected)',
    )

    # ---- griffith runpod ----
    runpod_p = sub.add_parser(
        'runpod',
        help='RunPod稼働podの一覧・停止(griffith launch --backend runpod の補助)',
        description=(
            'RunPod Secure Cloudの稼働pod一覧・停止・gpu_type_id一覧を扱う。'
            'ロジックは scripts/runpod_app.py 側。'
        ),
    )
    runpod_sub = runpod_p.add_subparsers(
        dest='runpod_command', metavar='ls|stop|gpus',
    )
    runpod_sub.add_parser('ls', help='稼働pod一覧(uptime・概算課金額)')
    runpod_stop_p = runpod_sub.add_parser('stop', help='podを1件停止する')
    runpod_stop_p.add_argument('pod_id', help='停止するpod ID(この1件のみ停止)')
    runpod_sub.add_parser('gpus', help='gpu_type_id一覧(GPU_TYPE_MAP実測検証用)')

    # ---- griffith runs ----
    runs_p = sub.add_parser(
        'runs',
        help='wandb runの一覧・比較',
        description='`.wandb_runs.jsonl`とwandb公開APIでrunを一覧・比較する。',
    )
    runs_sub = runs_p.add_subparsers(dest='runs_command', metavar='list|compare')
    runs_list_p = runs_sub.add_parser(
        'list', help='最近のrun一覧をmarkdown表で表示する',
    )
    runs_list_p.add_argument(
        '-n', type=int, default=20, help='表示件数(default: 20)',
    )
    runs_cmp_p = runs_sub.add_parser(
        'compare', help='run同士のsummary/configを比較する',
    )
    runs_cmp_p.add_argument('run_ids', nargs='+', help='比較するwandb run ID')
    runs_cmp_p.add_argument(
        '--metrics', default=None,
        help='比較するメトリクス(カンマ区切り。省略時は全summaryキー)',
    )
    runs_cmp_p.add_argument(
        '--format', choices=['md', 'csv'], default='md', help='出力形式',
    )

    # ---- griffith report ----
    report_p = sub.add_parser(
        'report',
        help='実験レポートを生成する(wandb/markdownフォールバック)',
        description=(
            'wandb-workspaces利用時はwandbレポート(学習曲線+config差分+結論欄)を'
            '生成しURL表示。不可なら reports/auto/ へmarkdownを生成する。'
        ),
    )
    report_p.add_argument(
        'run_ids', nargs='*', help='対象のwandb run ID(--group指定時は不要)',
    )
    report_p.add_argument(
        '--group', default=None, metavar='G',
        help='グループ横断のデータパックを reports/auto/ へ生成する',
    )
    report_p.add_argument('--title', default=None, help='レポートタイトル')
    report_p.add_argument(
        '--dest', default=None, metavar='DIR',
        help='markdownフォールバックの出力先(default: reports/auto)',
    )

    # ---- griffith bench ----
    bench_p = sub.add_parser(
        'bench',
        help='ベンチマーク(hard acc×ゲート数×forwardレイテンシ)を計測する',
        description=(
            'レシピYAMLまたはチェックポイントを対象に、hard eval精度・'
            'ゲート数・CPU forwardレイテンシ(batch=1/128)を計測し、'
            'benchmarks/results/ へJSONで保存する。\n'
            '例: griffith bench configs/recipes/logic_xor.yaml --skip-train'
        ),
    )
    bench_p.add_argument(
        'source', help='レシピYAML(.yaml/.yml)またはチェックポイント(.pt)のパス',
    )
    bench_p.add_argument(
        '--skip-train', action='store_true',
        help='レシピ指定時に学習せずランダム初期化で計測(レイテンシ/ゲート数のみ)',
    )
    bench_p.add_argument(
        '--out-dir', default=None, metavar='DIR',
        help='結果JSONの出力先(default: benchmarks/results)',
    )
    bench_p.add_argument(
        '--name', default=None,
        help='結果のname(省略時はsourceのファイル名から導出)',
    )

    return parser


def _load_project_train_config(
    project_path: str,
    *,
    network: str | None,
    config_name: str | None,
    optimizer: str | None,
):
    """エディタプロジェクトJSON(outputs/editor_projects/*.json)から
    指定ネットワーク+学習設定(+任意でオプティマイザ定義)を解決しTrainConfigを構築する。

    プロジェクトのスキーマは tools/editor/backend/project.py の ProjectSchema
    だが、console script(griffith)実行時はcwdがsys.pathへ入らず tools/ 配下を
    importできない(pip install した環境には tools/ 自体が無い)。
    ここではJSONを直接読み、形状だけ前提にして辞書アクセスする(読み取り専用参照)。

    ネットワーク(IR)は現状の共通ランナー(run_training)がTrainConfig.model
    (ModelConfig)からのみモデルを構築する仕様のため、モデル構築には使わない。
    --networkでは指定ネットワークがプロジェクト内に存在しIRとして妥当である
    ことのみ検証する(IRの層構成をModelConfigへ逆変換する経路は持たない)。
    """
    try:
        with open(project_path, encoding='utf-8') as f:
            doc = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f'プロジェクトJSONの形式が不正です: {project_path}: {e}') from e

    if not isinstance(doc, dict):
        raise ValueError(f'プロジェクトJSONのルートはマッピングが必要です: {project_path}')

    networks = doc.get('networks') or {}
    train_configs = doc.get('train_configs') or {}
    optimizers = doc.get('optimizers') or {}

    if network is not None:
        if network not in networks:
            raise ValueError(
                f"プロジェクトにネットワーク '{network}' がありません"
                f"(存在するネットワーク: {sorted(networks)})"
            )
        net_name = network
    elif networks:
        net_name = next(iter(networks))
    else:
        raise ValueError(f'プロジェクトにネットワークが1件もありません: {project_path}')

    from .ir import IRValidationError, ir_from_dict
    from .ir import validate as ir_validate

    try:
        graph = ir_from_dict(networks[net_name])
        ir_validate(graph)
    except (ValueError, KeyError, TypeError, AssertionError, IRValidationError) as e:
        raise ValueError(f"ネットワーク '{net_name}' のIRが不正です: {e}") from e

    if config_name is not None:
        if config_name not in train_configs:
            raise ValueError(
                f"プロジェクトに学習設定 '{config_name}' がありません"
                f"(存在する設定: {sorted(train_configs)})"
            )
        cfg_name = config_name
    elif 'global' in train_configs:
        cfg_name = 'global'
    elif train_configs:
        cfg_name = next(iter(train_configs))
    else:
        raise ValueError(f'プロジェクトに学習設定が1件もありません: {project_path}')

    from .training.config import OptimConfig, TrainConfig

    cfg = TrainConfig.model_validate(train_configs[cfg_name])

    if optimizer is not None:
        if optimizer not in optimizers:
            raise ValueError(
                f"プロジェクトにオプティマイザ定義 '{optimizer}' がありません"
                f"(存在する定義: {sorted(optimizers)})"
            )
        opt_doc = optimizers[optimizer] or {}
        if not isinstance(opt_doc, dict):
            raise ValueError(
                f"オプティマイザ定義 '{optimizer}' はマッピングである必要があります"
            )
        merged_optim = {**cfg.optim.model_dump(), **opt_doc}
        cfg = cfg.model_copy(update={'optim': OptimConfig.model_validate(merged_optim)})

    return cfg


def _cmd_train(args: argparse.Namespace) -> int:
    from .training.config import TrainConfig, dump_config, load_config
    from .training.runner import run_training

    overrides = list(args.overrides)
    config_path = args.config
    # 位置引数configが key=value 形式ならオーバーライドとして扱う
    if config_path is not None and '=' in config_path:
        overrides.insert(0, config_path)
        config_path = None

    sources = [
        name for name, value in (
            ('config', config_path),
            ('--from-run', args.from_run),
            ('--from-project', args.from_project),
        ) if value is not None
    ]
    if len(sources) > 1:
        print(f'エラー: {" と ".join(sources)} は同時に指定できません',
              file=sys.stderr)
        return 2
    if args.from_project is None and (
        args.network is not None
        or args.project_config_name is not None
        or args.optimizer is not None
    ):
        print('エラー: --network/--config/--optimizer は --from-project と'
              '同時に指定してください', file=sys.stderr)
        return 2

    if args.from_project is not None:
        cfg = _load_project_train_config(
            args.from_project, network=args.network,
            config_name=args.project_config_name, optimizer=args.optimizer,
        )
        if overrides:
            from .training.config import _set_by_dots

            data = dump_config(cfg)
            for item in overrides:
                if '=' not in item:
                    raise ValueError(
                        f'オーバーライドは key=value 形式で指定してください: {item!r}'
                    )
                key, _, value = item.partition('=')
                _set_by_dots(data, key.strip(), value)
            cfg = TrainConfig.model_validate(data)
    else:
        if args.from_run is not None:
            config_path = os.path.join(args.from_run, 'config.yaml')
        cfg = load_config(config_path, overrides)

    result = run_training(cfg, out_dir=args.out_dir)

    print('--- 結果 ---')
    print(f'hard_acc={result["hard_acc"]:.4f}  '
          f'best_hard_acc={result["best_hard_acc"]:.4f}  '
          f'steps={result["steps"]}')
    print(f'出力: {result["out_dir"]}')
    print(f'再実行: griffith train {result["config_path"]}')

    # --strict時はexpected_metrics未達で非0 exit
    if args.strict:
        verdict = result.get('expected_metrics')
        if verdict is None:
            print('注意: --strict指定だがexpected_metricsが設定されていません')
        elif not verdict['ok']:
            print('エラー: expected_metrics未達(--strict)', file=sys.stderr)
            return 1
    return 0


def _cmd_eval(args: argparse.Namespace) -> int:
    from .training.runner import evaluate_checkpoint

    result = evaluate_checkpoint(
        args.checkpoint,
        dataset=args.dataset,
        batch_size=args.batch_size,
        device=args.device,
    )
    extra = ''.join(
        f'  {k}={v:.4f}' for k, v in result.items()
        if k.startswith('hard_') and k != 'hard_acc'
    )
    print(f'dataset={result["dataset"]}  step={result["step"]}')
    print(f'hard_acc={result["hard_acc"]:.4f}{extra}')
    return 0


def _cmd_export_hdl(args: argparse.Namespace) -> int:
    from .export.verilog import write_verilog
    from .export.vhdl import write_vhdl
    from .ir import load_ir, validate

    ir = load_ir(args.source)
    for warning in validate(ir, target='hdl'):
        print(f'警告: {warning}')

    stem = os.path.splitext(args.source)[0]
    if args.lang == 'verilog':
        out_path = args.output or f'{stem}.v'
        path = write_verilog(ir, out_path)
    else:
        out_path = args.output or f'{stem}.vhdl'
        path = write_vhdl(ir, out_path)
    print(f'{args.lang}: {path}')
    return 0


EDITOR_BACKEND_MARKER = 'tools/editor/backend/main.py'

EDITOR_NOT_AVAILABLE = (
    'ビジュアルエディタを使うには、リポジトリを clone した環境で '
    '`uv sync --extra editor` が必要です。\n'
    f'  期待する場所: <リポジトリ直下>/{EDITOR_BACKEND_MARKER}\n'
    '  リポジトリ直下で `griffith editor` を実行するか、環境変数 '
    'GRIFFITH_REPO_ROOT にリポジトリのパスを設定してください。'
)
def _cmd_deploy(args: argparse.Namespace) -> int:
    from .deploy.cli import run

    return run(args)


def _cmd_editor(args: argparse.Namespace) -> int:
    from .training.launch_cli import resolve_repo_root

    # 以前は app_dir=os.getcwd() を無条件でuvicornへ渡していたため、
    # 任意のディレクトリで実行すると そこの tools/ が import されうる(cwd汚染)。
    # エディタのバックエンドが実在するリポジトリルートを解決できたときだけ起動する。
    repo_root = resolve_repo_root(EDITOR_BACKEND_MARKER)
    if repo_root is None:
        print(EDITOR_NOT_AVAILABLE, file=sys.stderr)
        return 1

    try:
        import uvicorn
    except ImportError:
        print('エラー: uvicornが未導入です。uv sync --extra editor を実行してください',
              file=sys.stderr)
        return 1

    print('エディタバックエンドを起動します(フロントエンドは起動しません)。')
    print('フロントエンド込みの開発起動は make editor を使ってください。')
    print(f'リポジトリルート: {repo_root}')
    uvicorn.run(
        'tools.editor.backend.main:app',
        host=args.host, port=args.port, app_dir=str(repo_root),
    )
    return 0


def _cmd_sweep(args: argparse.Namespace) -> int:
    from .training import sweep as sweep_mod

    if not sweep_mod.wandb_ready():
        print(sweep_mod.WANDB_GUIDANCE)
        return 1

    spec = sweep_mod.load_sweep_spec(args.spec)
    sweep_id = sweep_mod.create_sweep(spec)
    print(f'SWEEP_ID={sweep_id}')

    if args.modal:
        from .training.launch_cli import MODAL_GUIDANCE, load_modal_app

        parallel = max(1, args.parallel)
        per_agent = args.count
        if per_agent is not None:
            per_agent = max(1, -(-args.count // parallel))  # ceil(count/parallel)
        try:
            import modal

            mod = load_modal_app()
            with modal.enable_output(), mod.app.run():
                calls = [
                    mod.sweep_agent.spawn(sweep_id, per_agent)
                    for _ in range(parallel)
                ]
                print(f'Modalへsweep agentを{parallel}並列でspawnしました'
                      f'(agentあたり試行数: {per_agent or "無制限"})')
                for call in calls:
                    call.get()
        except RuntimeError as e:
            print(str(e), file=sys.stderr)
            return 1
        except Exception as e:  # noqa: BLE001 — modal認証エラー等
            print(f'Modal実行に失敗しました: {e}\n{MODAL_GUIDANCE}', file=sys.stderr)
            return 1
        return 0

    sweep_mod.run_sweep_agent(sweep_id, count=args.count)
    return 0


def _resolve_launch_backend(args: argparse.Namespace) -> str:
    """投入先を解決する。優先順位: --local → --modal → --backend(既定local)。"""
    if args.local:
        return 'local'
    if args.modal:
        return 'modal'
    return args.backend


def _cmd_launch(args: argparse.Namespace) -> int:
    from .training import launch_cli

    combos = launch_cli.build_grid_combos(args.grid, args.config)
    launch_cli.print_combos(combos)
    if args.dry_run:
        return 0

    backend = _resolve_launch_backend(args)
    if backend == 'runpod':
        try:
            results = launch_cli.launch_runpod(combos, args.config)
        except RuntimeError as e:
            print(str(e), file=sys.stderr)
            return 1
        launch_cli.print_runpod_summary(results)
        return 1 if any('error' in r for r in results) else 0

    if backend == 'modal':
        try:
            results = launch_cli.launch_modal(combos, args.config)
        except RuntimeError as e:
            print(str(e), file=sys.stderr)
            return 1
    else:
        results = launch_cli.launch_local(combos, args.config)
    launch_cli.print_launch_summary(results)
    return 1 if any('error' in r for r in results) else 0


def _cmd_runpod(args: argparse.Namespace) -> int:
    from .training.launch_cli import load_runpod_app

    try:
        mod = load_runpod_app()  # runpod未導入・APIキー無し等はRuntimeError
        if args.runpod_command == 'ls':
            return mod.cmd_ls()
        if args.runpod_command == 'stop':
            return mod.cmd_stop(args.pod_id)
        if args.runpod_command == 'gpus':
            return mod.cmd_gpus()
    except RuntimeError as e:
        print(str(e), file=sys.stderr)
        return 1
    print('エラー: runpod のサブコマンド(ls/stop/gpus)を指定してください',
          file=sys.stderr)
    return 2


def _cmd_collect(args: argparse.Namespace) -> int:
    from .training.launch_cli import collect

    return collect(dest=args.dest)


def _cmd_runs(args: argparse.Namespace) -> int:
    from .training import runs_cli

    if args.runs_command == 'list':
        return runs_cli.cmd_list(n=args.n)
    if args.runs_command == 'compare':
        metrics = (
            [m.strip() for m in args.metrics.split(',') if m.strip()]
            if args.metrics else None
        )
        return runs_cli.cmd_compare(args.run_ids, metrics=metrics, fmt=args.format)
    print('エラー: runs のサブコマンド(list/compare)を指定してください',
          file=sys.stderr)
    return 2


def _cmd_report(args: argparse.Namespace) -> int:
    from .training import report_cli

    dest = args.dest or report_cli.DEFAULT_DEST
    # --group 指定時はグループ横断のデータパックを生成する
    if args.group:
        return report_cli.cmd_report_group(args.group, dest=dest)
    if not args.run_ids:
        print('エラー: run ID か --group のどちらかを指定してください',
              file=sys.stderr)
        return 2
    return report_cli.cmd_report(args.run_ids, title=args.title, dest=dest)


def _cmd_bench(args: argparse.Namespace) -> int:
    from .training import bench_cli

    return bench_cli.cmd_bench(
        args.source,
        skip_train=args.skip_train,
        out_dir=args.out_dir or bench_cli.DEFAULT_OUT_DIR,
        name=args.name,
    )


_COMMANDS = {
    'train': _cmd_train,
    'eval': _cmd_eval,
    'export-hdl': _cmd_export_hdl,
    'deploy': _cmd_deploy,
    'editor': _cmd_editor,
    'sweep': _cmd_sweep,
    'launch': _cmd_launch,
    'runpod': _cmd_runpod,
    'collect': _cmd_collect,
    'runs': _cmd_runs,
    'report': _cmd_report,
    'bench': _cmd_bench,
}


def main(argv: Sequence[str] | None = None) -> int:
    """mdコマンドのエントリポイント(戻り値=exit code)"""
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 2

    try:
        return _COMMANDS[args.command](args)
    except KeyboardInterrupt:
        print('中断しました', file=sys.stderr)
        return 130
    # pydantic.ValidationErrorはValueErrorのサブクラスなのでここに含まれる
    except (ValueError, NotImplementedError, FileNotFoundError) as e:
        print(f'エラー: {e}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
