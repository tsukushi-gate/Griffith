"""`griffith deploy` サブコマンドの実装。

argparse のパーサ定義は griffith_logic/cli.py 側にあり(重いimportを
避けるため)、ここには実処理だけを置く。
"""

import json
import sys
from pathlib import Path

from .bundle import (
    BUNDLE_VERSION,
    attach_bundle_sizes,
    build_bundle,
    load_bundle,
    save_bundle,
    validate_bundle,
)
from .from_checkpoint import doc_from_checkpoint, is_checkpoint_path


def _resolve_cli_auto_args(args):
    """`--no-auto`/`--conv-engine`/`--dense-fuse`/`--dense-packed`の
    生CLI引数から、`generate_metal`/`generate_cuda`へ渡す
    `(conv_engine, dense_fuse, dense_packed, auto)`を組み立てる。

    `--dense-fuse`/`--dense-packed`は`store_true`のboolフラグなので
    「未指定」と「明示的にFalse」を区別できない。そこでCLIでは
    「フラグを付けたらTrue(明示ON、autoの値によらず常にON)、付けな
    ければNone(auto有効時は自動選択に委ねる、auto無効=--no-auto時は
    旧来の既定Falseのまま)」という素直な写像にする(明示的にOFFへ
    固定したいだけなら`--no-auto`でauto自体を切ればよい)。
    """
    no_auto = getattr(args, 'no_auto', False)
    conv_engine = getattr(args, 'conv_engine', None)
    dense_fuse_flag = getattr(args, 'dense_fuse', False)
    dense_packed_flag = getattr(args, 'dense_packed', False)
    if no_auto:
        auto = 'off'
        dense_fuse = dense_fuse_flag
        dense_packed = dense_packed_flag
        if conv_engine is None:
            conv_engine = 'split'
    else:
        auto = 'measured'
        dense_fuse = True if dense_fuse_flag else None
        dense_packed = True if dense_packed_flag else None
    return conv_engine, dense_fuse, dense_packed, auto


def run(args) -> int:
    handlers = {
        'bundle': _cmd_bundle,
        'info': _cmd_info,
        'codegen-c': _cmd_codegen_c,
        'codegen-cuda': _cmd_codegen_cuda,
        'codegen-metal': _cmd_codegen_metal,
        'codegen-web': _cmd_codegen_web,
        'verify-c': _cmd_verify_c,
        'verify-metal': _cmd_verify_metal,
        'verify-web': _cmd_verify_web,
    }
    handler = handlers.get(getattr(args, 'deploy_command', None))
    if handler is None:
        print('griffith deploy <bundle|info> を指定してください')
        return 2
    return handler(args)


def _cmd_bundle(args) -> int:
    """モデルdoc(webbench形式)または学習済み .pt → 配布バンドルv1。

    .pt/.pth はチェックポイント内(または --config)の設定でモデルを再構築し、
    離散化したIRに変換してからバンドル化する(from_checkpoint.py)。

    lossless な回路圧縮を build_bundle の**前**に自動適用する
    (`--compress none|fast|max`、既定 fast = dense に sweep。`max` は
    conv2d 木の再合成も。個別の `--sweep`/`--no-sweep`/`--resynth` が優先。
    詳細は `deploy/compress.py`)。結果は `about.compress` に残る。

    `--resynth lossless`(または `--compress max`)の再合成は、build_bundle の
    **前**にconv2d木を再合成する。再合成後のIRがそのままバンドルの真実源になるので、
    build_bundle(verify=True)/validate_bundle(strict=True) は再合成後の
    programs と整合したまま通る(検証対象自身から真実源を作り直す、
    という抜け道を作らない)。
    """
    from .compress import (
        apply_compress,
        attach_compress_info,
        describe_compress,
    )

    src = Path(args.source)
    if is_checkpoint_path(src):
        doc = doc_from_checkpoint(src, config=getattr(args, 'config', None))
    else:
        if getattr(args, 'config', None):
            print('--config は .pt/.pth 入力のときだけ使う(無視します)')
        doc = json.loads(src.read_text(encoding='utf-8'))
    # 回路圧縮を既定で自動適用する(--compress fast)。個別フラグ
    # (--sweep/--no-sweep/--resynth)は --compress より優先。属性が無い
    # Namespace(テスト・他スクリプト)も CLI 既定と同じ fast とみなす。
    level = getattr(args, 'compress', None) or 'fast'
    if getattr(args, 'sweep', False):
        sweep_opt = True
    elif getattr(args, 'no_sweep', False):
        sweep_opt = False
    else:
        sweep_opt = None
    try:
        doc, compress_info, summaries = apply_compress(
            doc, level=level, sweep=sweep_opt,
            resynth=getattr(args, 'resynth', None),
            sweep_verify_samples=getattr(args, 'sweep_verify_samples', 8),
            resynth_max_cost=args.resynth_max_cost,
            resynth_verify_samples=args.resynth_verify_samples)
    except ValueError as exc:
        print(f'griffith deploy bundle: {exc}', file=sys.stderr)
        return 2
    resynth_summary = summaries.get('resynth')
    if resynth_summary is not None:
        print(f"resynth(lossless): ゲート数 "
              f"{resynth_summary['gates_before']:,} → "
              f"{resynth_summary['gates_after']:,} "
              f"(木{resynth_summary['n_trees']}本中"
              f"{resynth_summary['n_trees_changed']}本変更、"
              f"ノード{resynth_summary['nodes_changed']}個)")
    sweep_summary = summaries.get('sweep')
    if sweep_summary is not None:
        print(f"sweep(lossless): ゲート数 "
              f"{sweep_summary['gates_before']:,} → "
              f"{sweep_summary['gates_after']:,} "
              f"(EDA換算 {sweep_summary['swept_gates']:,}、"
              f"定数吸収{sweep_summary['constants_absorbed']}・"
              f"CSE統合{sweep_summary['cse_merged']}・"
              f"除去{sweep_summary['gates_removed']}、"
              f"検証 ランダム{sweep_summary['verify_samples']}件"
              f"+実データ{sweep_summary['verify_demo_samples']}件で一致)")
    if getattr(args, 'dense_flat', False):
        from .flatten_dense import flatten_dense_doc

        doc, flat_report = flatten_dense_doc(
            doc, n_verify=getattr(args, 'dense_flat_verify_samples', 16))
        if flat_report['chains']:
            print(_dense_flat_line(flat_report))
            compress_info['applied'].append('dense-flat')
            compress_info.setdefault('gates', {})['dense-flat'] = [
                int(flat_report['gates_before']), int(flat_report['gates_after'])]
        else:
            print('dense-flat: 対象の2入力lut_layer列が無い(変換なし)')
            compress_info['skipped'].append({
                'pass': 'dense-flat',
                'reason': '対象の2入力lut_layer列が無い'})
    doc = attach_compress_info(doc, compress_info)
    bundle = build_bundle(doc, verify=not args.no_verify)
    if (bundle.get('about') or {}).get('compress'):
        # conv2d 木の命令列化(programs 節。どの level でも必ず行う変換)の
        # 削減も、他の圧縮と同じ形式で記録・表示する
        before, after = _program_gates(bundle)
        if before:
            comp = dict(bundle['about']['compress'])
            comp['always'] = [{'pass': 'conv-program', 'gates': [before, after]}]
            bundle['about'] = {**bundle['about'], 'compress': comp}
        print(describe_compress(bundle['about']['compress']))
    # resources.bundle_json_bytes/bundle_gzip_bytes を、これから
    # save_bundle で書き出す実バイト数として焼き込む(build_bundleの直後
    # = --resynth lossless で置き換わった doc を使った場合も含めて正しく
    # 測るため)。webbenchのUIヘッダはこれを配信サイズとして表示する。
    # (この2フィールドはJSON書き出し基準の見積りのまま。.mdlb出力時も
    # 焼き込む値自体は変えない — 「JSON配信サイズの見積り」という既存の
    # 意味を保つため。.mdlb自体の実サイズは後述のprintで別途表示する)
    attach_bundle_sizes(bundle)
    fmt = args.format
    if args.output:
        out = Path(args.output)
    else:
        out = src.with_suffix('.bundle.mdlb' if fmt == 'mdlb' else '.bundle.json')
    save_bundle(bundle, out, gzip_too=args.gzip, format=fmt)
    size = out.stat().st_size
    n_conv = len(bundle.get('programs', {}))
    fmt_label = out.suffix.lstrip('.')
    print(f'bundle v{BUNDLE_VERSION} ({fmt_label}): {out} '
          f'({size:,}B, conv2d {n_conv}ノード)')
    if args.gzip:
        gz = Path(str(out) + '.gz')
        print(f'  gzip: {gz} ({gz.stat().st_size:,}B)')
    _print_gate_summary(bundle)
    _print_sweep_summary(bundle)
    return 0


def _cmd_info(args) -> int:
    """バンドルの検証と要約表示。"""
    bundle = load_bundle(args.source, validate=False)
    warnings = validate_bundle(bundle, strict=args.strict)
    for w in warnings:
        print(f'警告: {w}')
    ir = bundle['ir']
    node_types: dict[str, int] = {}
    for node in ir['nodes']:
        node_types[node['type']] = node_types.get(node['type'], 0) + 1
    print(f"bundle_version={bundle['bundle_version']}  ir_version={ir.get('version')}")
    print('ノード: ' + ', '.join(f'{t}×{c}' for t, c in sorted(node_types.items())))
    if bundle.get('labels'):
        print(f"labels: {len(bundle['labels'])}クラス")
    compress = (bundle.get('about') or {}).get('compress')
    if compress:
        from .compress import describe_compress

        print(describe_compress(compress))
    dense_flat = (ir.get('metadata') or {}).get('dense_flat')
    if dense_flat:
        print(_dense_flat_line(dense_flat))
    _print_dense_flat_outlook(bundle)
    _print_gate_summary(bundle)
    _print_sweep_summary(bundle)
    print('検証: OK' + ('(strict: IRから再構築して一致確認済み)' if args.strict else ''))
    return 0


def _cmd_codegen_c(args) -> int:
    """バンドル → 自己完結したCソース(+ ランタイムのコピー)。"""
    import shutil

    from .cbackend import generate_c
    from .flatten_dense import describe_dense_flat, resolve_dense_flat
    from .plan import DENSE_BATCH_AUTO

    bundle = load_bundle(args.source, validate=True)
    dense_batch = getattr(args, 'dense_batch', None)
    # バンドル(真実源)は変えず、C 生成の段でメモリ上だけ平坦化する。
    c_bundle, flat_decision = resolve_dense_flat(
        bundle, mode=getattr(args, 'dense_flat', None) or 'auto',
        min_reduction=getattr(args, 'dense_flat_min_reduction', None))
    print(describe_dense_flat(flat_decision))
    if dense_batch is None:
        # 平坦化を採用した dense には既定でバッチ軸パック入口を生やす
        # (表を共有するので flash はほぼ増えない)。採用しなかったときは
        # パック経路がゲートごとの生の表を足すので、明示指定のときだけ。
        dense_batch = DENSE_BATCH_AUTO if flat_decision['applied'] else 0
        if dense_batch:
            print(f'dense-batch(auto): mdl_infer_batch_packed を生成'
                  f'(最大 {dense_batch} 枚/回。--dense-batch 0 で抑止)')
    files = generate_c(c_bundle, name=args.name, validate=False,
                       fuse=getattr(args, 'fuse', False),
                       dense_batch=dense_batch,
                       dense_flat_decision=flat_decision)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    for filename, text in files.items():
        (out_dir / filename).write_text(text, encoding='utf-8')
        print(f'  {out_dir / filename} ({len(text):,}B)')
    if not args.no_runtime:
        runtime = Path(__file__).resolve().parents[2] / 'runtime' / 'c'
        for rel in ('include/mdl_kernels.h', 'include/mdl_simd.h',
                    'src/mdl_kernels.c'):
            shutil.copy2(runtime / rel, out_dir / Path(rel).name)
            print(f'  {out_dir / Path(rel).name} (ランタイム)')
        # ホスト側ドライバ用(stdin/stdout を Windows でもバイナリで扱う MDL_STDIO_BINARY)
        shutil.copy2(runtime / 'include/mdl_stdio.h', out_dir / 'mdl_stdio.h')
        print(f'  {out_dir / "mdl_stdio.h"} (任意: ホストドライバ用)')
        # マルチコアヘルパは任意。コアは pthread を参照しないので、
        # ベアメタルではこの2本をリンクしなければよい。
        for rel in ('include/mdl_threads.h', 'src/mdl_threads.c'):
            shutil.copy2(runtime / rel, out_dir / Path(rel).name)
            print(f'  {out_dir / Path(rel).name} (任意: マルチコア)')
    print(f'ビルド例: cc -std=c99 -O2 -I. {args.name}.c mdl_kernels.c '
          'your_main.c -o app')
    print('  マルチコアで回すなら: ... mdl_threads.c -lpthread')
    return 0


def _gpu_dense_flat(bundle: dict, args, dense_fuse, dense_packed):
    """codegen-cuda/metal でも C と同じく生成直前に平坦化を判定する。

    GPU の既定は ``off``(opt-in)。Metal 実測で大きいモデルが平坦化しない最良より
    遅くなる(HAR k6000・batch 128 以上で 0.47〜0.71 倍)ため。``--dense-flat
    auto|on`` で明示したときだけ平坦化を試みる。
    `--dense-fuse` / `--dense-packed` を明示したときは、それらが lut_dense 列を
    前提にするので平坦化しない(明示が勝つ)。
    """
    from .flatten_dense import _decision, describe_dense_flat, resolve_dense_flat

    mode = getattr(args, 'dense_flat', None) or 'off'
    if dense_fuse is True or dense_packed is True:
        dec = _decision(mode, 0.0, reason='--dense-fuse/--dense-packed を明示したので平坦化しない')
        print(describe_dense_flat(dec))
        return bundle
    bundle, dec = resolve_dense_flat(
        bundle, mode=mode,
        min_reduction=getattr(args, 'dense_flat_min_reduction', None),
        target='gpu')
    print(describe_dense_flat(dec))
    return bundle


def _cmd_codegen_cuda(args) -> int:
    """バンドル → CUDAソース(+ 共有カーネルとランタイムのコピー)。"""
    import shutil

    from .cudabackend import generate_cuda

    bundle = load_bundle(args.source, validate=True)
    n_streams = getattr(args, 'n_streams', 1)
    conv_codegen = getattr(args, 'conv_codegen', False)
    conv_engine, dense_fuse, dense_packed, auto = _resolve_cli_auto_args(args)
    bundle = _gpu_dense_flat(bundle, args, dense_fuse, dense_packed)
    files = generate_cuda(bundle, name=args.name, batch=args.batch,
                          conv_engine=conv_engine, n_streams=n_streams,
                          conv_codegen=conv_codegen, dense_fuse=dense_fuse,
                          dense_packed=dense_packed, auto=auto)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    for filename, text in files.items():
        (out_dir / filename).write_text(text, encoding='utf-8')
        print(f'  {out_dir / filename} ({len(text):,}B)')
    if not args.no_runtime:
        root = Path(__file__).resolve().parents[2] / 'runtime'
        for rel in ('cuda/mdl_cuda.h', 'cuda/mdl_cuda.cu',
                    'gpu/mdl_gpu_kernels.h'):
            shutil.copy2(root / rel, out_dir / Path(rel).name)
            print(f'  {out_dir / Path(rel).name} (ランタイム)')
    print(f'ビルド例: nvcc -O2 -arch=sm_89 -I. {args.name}.cu mdl_cuda.cu '
          'your_main.cu -o app')
    if n_streams > 1:
        print(f'  n_streams={n_streams}: mdl_cuda_infer_batch_submit/_collect'
              '(画像入力があれば image 版も)を追加生成(パイプライン化)')
    if conv_codegen:
        print('  --conv-codegen: conv2dノードごとにtt即値+スロット静的展開の'
              'モデル特化カーネルを生成')
    if dense_fuse:
        print('  --dense-fuse: dense多層融合カーネルを明示指定('
              'Metal版と対称のソース。共有メモリ上限48KB)')
    if dense_packed:
        print('  --dense-packed: denseのバッチ軸ビットプレーン化を明示指定'
              '(Metal版と対称のソース)')
    # conv_engine/dense_fuse/dense_packedの自動選択(auto既定ON)の
    # 結果と理由は generate_cuda() 自身が1行サマリを標準出力へ出す
    # (--no-auto指定時は出ない)。
    return 0


def _cmd_codegen_metal(args) -> int:
    """バンドル → Metal(Apple GPU)ソース。"""
    import shutil

    from .metalbackend import generate_metal

    bundle = load_bundle(args.source, validate=True)
    n_streams = getattr(args, 'n_streams', 1)
    conv_codegen = getattr(args, 'conv_codegen', False)
    conv_engine, dense_fuse, dense_packed, auto = _resolve_cli_auto_args(args)
    bundle = _gpu_dense_flat(bundle, args, dense_fuse, dense_packed)
    files = generate_metal(bundle, name=args.name, batch=args.batch,
                           conv_engine=conv_engine, n_streams=n_streams,
                           conv_codegen=conv_codegen, dense_fuse=dense_fuse,
                           dense_packed=dense_packed, auto=auto)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    for filename, text in files.items():
        (out_dir / filename).write_text(text, encoding='utf-8')
        print(f'  {out_dir / filename} ({len(text):,}B)')
    if not args.no_runtime:
        root = Path(__file__).resolve().parents[2] / 'runtime'
        for rel in ('metal/mdl_metal.h', 'metal/mdl_metal.mm'):
            shutil.copy2(root / rel, out_dir / Path(rel).name)
            print(f'  {out_dir / Path(rel).name} (ランタイム)')
    print(f'ビルド例: clang++ -std=c++17 -O2 -fobjc-arc -I. '
          f'-framework Metal -framework Foundation '
          f'{args.name}.mm mdl_metal.mm your_main.mm -o app')
    print('  (シェーダは実行時にコンパイルするので metal ツールチェーン不要)')
    if n_streams > 1:
        print(f'  n_streams={n_streams}: mdl_metal_infer_batch_submit/_collect'
              '(画像入力があれば image 版も)を追加生成(パイプライン化)')
    if conv_codegen:
        print('  --conv-codegen: conv2dノードごとにtt即値+スロット静的展開の'
              'モデル特化カーネルを生成')
    if dense_fuse:
        print('  --dense-fuse: dense多層融合カーネルを明示指定'
              '(GEM Boomerang型、活性をthreadgroupメモリへ)')
    if dense_packed:
        print('  --dense-packed: denseのバッチ軸ビットプレーン化を明示指定'
              '(32サンプル/語、batch>=32必須)')
    # conv_engine/dense_fuse/dense_packedの自動選択(auto既定ON)の
    # 結果と理由は generate_metal() 自身が1行サマリを標準出力へ出す
    # (--no-auto指定時は出ない)。
    return 0


def _cmd_codegen_web(args) -> int:
    """バンドル → Web(ブラウザ)配布物(+ ランタイムのコピー)。

    C/CUDA/Metalと違い「ソース生成」ではなく「バンドルJSONの配置」が
    出力の主体になる(実行体はTS製でここでは生成しない)。
    """
    import gzip
    import shutil

    from .flatten_dense import describe_dense_flat, resolve_dense_flat
    from .webbackend import RUNTIME_WEB, generate_web

    bundle = load_bundle(args.source, validate=True)
    # codegen-c と同じく、配る直前に dense-flat(lut_flat)の採否を決める
    # (Web は閾値 0 = 2入力lut_layer列があれば採用。plan.DENSE_FLAT_AUTO_MIN_REDUCTION_WEB)
    bundle, flat_decision = resolve_dense_flat(
        bundle, mode=getattr(args, 'dense_flat', None) or 'auto',
        min_reduction=getattr(args, 'dense_flat_min_reduction', None),
        target='web')
    print(describe_dense_flat(flat_decision))
    files = generate_web(bundle, name=args.name, slim=args.slim)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    for filename, text in files.items():
        path = out_dir / filename
        path.write_text(text, encoding='utf-8')
        print(f'  {path} ({len(text):,}B)')
        if args.gzip:
            gz = Path(str(path) + '.gz')
            gz.write_bytes(gzip.compress(text.encode('utf-8'), 6))
            print(f'  {gz} ({gz.stat().st_size:,}B)')
    if not args.no_runtime:
        dist = RUNTIME_WEB / 'dist'
        if not dist.is_dir():
            raise SystemExit(
                'runtime/web/dist が無い。先に `make web-runtime`'
                '(= npm --prefix tools/webbench run build:runtime)でビルドすること。'
            )
        for src in dist.rglob('*'):
            if src.is_dir():
                continue
            rel = src.relative_to(dist)
            dst = out_dir / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            print(f'  {dst} (ランタイム)')
    print(f'配信例: python -m http.server -d {out_dir} '
          '(単一スレッド実行なのでCOOP/COEPヘッダは不要)')
    _print_gate_summary(bundle)
    return 0


def _cmd_verify_c(args) -> int:
    """C実行 ⇔ ir/simulate の整数厳密一致を確かめる(出荷前の関門)。"""
    from .verify import verify_c_matches_simulate

    bundle = load_bundle(args.source, validate=True)
    print(f'C等価性検証: {args.source}(サンプル{args.samples}件)')
    result = verify_c_matches_simulate(
        bundle, n_samples=args.samples, seed=args.seed,
        dense_flat=getattr(args, 'dense_flat', None) or 'auto',
        dense_flat_min_reduction=getattr(args, 'dense_flat_min_reduction', None),
        dense_batch=getattr(args, 'dense_batch', None))
    flat = result.get('dense_flat') or {}
    via = ' + dense-flat(lut_flat)' if flat.get('applied') else ''
    n_packed = result.get('batch_packed_samples', 0)
    packed = f'、mdl_infer_batch_packed {n_packed}枚' if n_packed else ''
    print(f'一致: 経路={result["path"]}{via} 出力{result["n_out"]}要素 × '
          f'{result["n_samples"]}サンプル{packed}(整数厳密、ゴールデンは元バンドルIRのsimulate)')
    return 0


def _cmd_verify_metal(args) -> int:
    """Metal(Apple GPU)実行 ⇔ ir/simulate の整数厳密一致を確かめる(出荷前の関門)。

    macOS + Apple GPU + clang++ が無い環境ではその場でエラーにする
    (`verify-c` と同じ「その場で失敗させる」方針。CIでは事前に
    `metalverify.has_metal()` 相当の判定でジョブごとskipすること)。
    """
    from .metalverify import has_metal, verify_metal_matches_simulate

    if not has_metal():
        print('griffith deploy verify-metal: macOS + Metal(clang++)のある環境が必要')
        return 2
    bundle = load_bundle(args.source, validate=True)
    conv_codegen = getattr(args, 'conv_codegen', False)
    dense_fuse = getattr(args, 'dense_fuse', False)
    print(f'Metal等価性検証: {args.source}(サンプル{args.samples}件、'
          f'conv_codegen={conv_codegen}, dense_fuse={dense_fuse})')
    result = verify_metal_matches_simulate(
        bundle, n_samples=args.samples, seed=args.seed,
        conv_codegen=conv_codegen, dense_fuse=dense_fuse,
        dense_flat=getattr(args, 'dense_flat', None) or 'off',
        dense_flat_min_reduction=getattr(args, 'dense_flat_min_reduction', None))
    flat = result.get('dense_flat') or {}
    via = ' + dense-flat(lut_flat)' if flat.get('applied') else ''
    print(f'一致: 経路={result["path"]}{via} 出力{result["n_out"]}要素 × '
          f'{result["n_samples"]}サンプル(整数厳密、ゴールデンは元バンドルIRのsimulate)')
    return 0


def _cmd_verify_web(args) -> int:
    """Webランタイム ⇔ ir/simulate の整数厳密一致を確かめる(出荷前の関門)。"""
    from .webverify import verify_web_matches_simulate

    bundle = load_bundle(args.source, validate=True)
    print(f'Web等価性検証: {args.source}(サンプル{args.samples}件、'
          f'engine={args.engine})')
    result = verify_web_matches_simulate(
        bundle, n_samples=args.samples, seed=args.seed, engine=args.engine,
        dense_flat=getattr(args, 'dense_flat', None) or 'auto',
        dense_flat_min_reduction=getattr(args, 'dense_flat_min_reduction', None))
    flat = result.get('dense_flat') or {}
    via = ' + dense-flat(lut_flat)' if flat.get('applied') else ''
    print(f'一致: 経路={result["mode"]}{via} 出力{result["n_out"]}要素 × '
          f'{result["n_samples"]}サンプル(整数厳密、engine={result["engine"]}、'
          'ゴールデンは元バンドルIRのsimulate)')
    return 0


def _print_dense_flat_outlook(bundle: dict) -> None:
    """codegen-c 既定(--dense-flat auto)で何が起きるかの見込みを1行で出す。

    平坦化を**試行だけ**して(検証サンプル0、IRは書き換えない)削減率から
    採否を予告する。バンドルに焼き込み済みなら何もしない(上の行で出る)。
    試行に失敗しても info 全体は失敗させない(見込み行を出さないだけ)。
    """
    from .flatten_dense import describe_dense_flat, resolve_dense_flat

    if (bundle['ir'].get('metadata') or {}).get('dense_flat'):
        return
    try:
        _, decision = resolve_dense_flat(bundle, mode='auto', n_verify=0)
    except Exception:                              # noqa: BLE001
        return
    if decision['gates_before'] == 0:
        return
    print('codegen-c 既定の見込み: ' + describe_dense_flat(decision))


def _print_sweep_summary(bundle: dict) -> None:
    """`--sweep`で焼き込まれた about.sweep があれば前後ゲート数を表示する。"""
    sw = (bundle.get('about') or {}).get('sweep')
    if not sw:
        return
    before = int(sw['gates_before'])
    after = int(sw['gates_after'])
    swept = int(sw['swept_gates'])
    pct = 100.0 * (before - after) / before if before else 0.0
    print(f'ゲート数(sweep {sw.get("mode", "lossless")}): {before:,} → {after:,} '
          f'({pct:.1f}%削減、EDA換算 {swept:,})')


def _dense_flat_line(report: dict) -> str:
    """`flatten_dense` の report(ir.metadata.dense_flat)を1行にまとめる。"""
    before = int(report['gates_before'])
    after = int(report['gates_after'])
    cut = 100.0 * (before - after) / before if before else 0.0
    return (f'dense-flat(lossless): ゲート数 {before:,} → {after:,} '
            f'({cut:.1f}%削減、定数{report["constants"]:,} / '
            f'パススルー・NOT{report["passthrough"]:,} / CSE{report["cse"]:,}、'
            f'検証: ランダム{report.get("verify_random", 0)}本'
            f'+実データ{report.get("verify_demo", 0)}本で整数一致)')


def _program_gates(bundle: dict) -> tuple[int, int]:
    """conv2d 木の命令列化前後のゲート数(programs 節が無ければ (0, 0))。"""
    programs = bundle.get('programs') or {}
    before = sum(p['stats']['gates_before'] for p in programs.values())
    after = sum(p['stats']['gates_after'] for p in programs.values())
    return before, after


def _print_gate_summary(bundle: dict) -> None:
    before, after = _program_gates(bundle)
    if before == 0:
        return
    print(f'ゲート数(conv2d): {before:,} → {after:,} '
          f'({100.0 * (before - after) / before:.1f}%削減、命令列化済み)')
