"""Cランタイムの移植性を契約として固定する(契約として再構成)。

主張していること:
    - **命令セット別の配布は不要**。同じC99ソースが MCU からデスクトップまで
      そのまま通る(SIMD組み込み関数もアーキ固有コードも使っていない)
    - ホスト環境(libc)なしでもビルドできる。libc依存は memcpy/memset だけで、
      `-DMDL_NO_LIBC` で自前実装に切り替わる。C標準がフリースタンディング処理系に
      要求するヘッダに <string.h> は**含まれない**ので、これが無いとベアメタルで
      そもそもコンパイルできない
    - ワード演算はビット位置で定義してあり、バイト順に依存しない
      (uint8_t* と uint32_t* を相互にキャストしている箇所が無いこと)
    - アーキ固有コード(NEON/SSE2/AVX2の組み込み関数・検出マクロ)は
      `mdl_simd.h` の1ファイルに隔離されており、対応の無いアーキでは
      **ビット同一のスカラー実装**へ自動フォールバックする
      (`TestSIMDContract`。「コアにアーキ固有コードが漏れていないか」
      「SIMD経路とスカラー経路が実モデルでビット完全一致するか」を
      機械的なテストとして固定した——それ以前はこの2点を裏付けるテストが
      無く、README/platform-support.mdの記述が手動ベンチの記憶に依存していた)

クロスコンパイル(`-target`)は clang 専用の構文なので、clang(`cc` が clang か
PATH 上の `clang`)がある環境でのみ検証する。gcc しか無い環境と、clang に
バックエンドが無いターゲットは skip(コード側の問題ではないため)。
"""

import re
import shutil
import subprocess

import numpy as np
import pytest

from griffith_logic.deploy import build_bundle
from griffith_logic.deploy.verify import build_c_model, run_c_model
from griffith_logic.ir.serialize import ir_from_dict

from .conftest import RUNTIME_C

KERNEL_C = RUNTIME_C / 'src' / 'mdl_kernels.c'
KERNEL_H = RUNTIME_C / 'include' / 'mdl_kernels.h'
RDDLGN_C = RUNTIME_C / 'src' / 'mdl_rddlgn.c'
RDDLGN_H = RUNTIME_C / 'include' / 'mdl_rddlgn.h'
SIMD_H = RUNTIME_C / 'include' / 'mdl_simd.h'

# 「コア」= mdl_simd.h(アーキ固有コードの隔離先として設計上そう作られている)
# 以外の runtime/c ヘッダ・ソース。mdl_threads.c/.h はPOSIX専用の**任意**並列
# ヘルパで、対象はISAではなくOS(pthread)なので対象外(runtime/c/README.md
# 「スレッド安全性とマルチコア」節のとおり)。
CORE_FILES = (KERNEL_H, KERNEL_C, RDDLGN_H, RDDLGN_C)

# MCU(Cortex-M0/M3/M4/M33)〜64bit、Windows、WASMまで。ローカルclangに
# 無いものは自動skip(リモートLinux機(Ubuntu、gcc 9.4.0のみ・clang/
# クロスツールチェーン無し)も確認したが、そちらにも無いので同じくskip)。
CROSS_TARGETS = [
    'thumbv6m-none-eabi',       # Cortex-M0/M0+
    'armv7m-none-eabi',         # Cortex-M3/M4
    'armv7a-none-eabi',         # 32bit ARM application profile(Cortex-A系)
    'thumbv7em-none-eabi',      # Cortex-M4F/M7(DSP拡張コア)
    'armv8m.main-none-eabi',    # Cortex-M33
    'aarch64-none-elf',         # 64bit ARM
    'aarch64_be-none-elf',      # ビッグエンディアン
    'i386-unknown-linux-gnu',   # 32bit x86
    'x86_64-unknown-linux-gnu',
    # Windows: freestanding + -DMDL_NO_LIBC なので Windows SDK 無しでも
    # オブジェクト生成まで通る(確認)。実行検証環境は無い。
    'x86_64-pc-windows-msvc',   # Windows(MSVC ABI/COFF)
    'x86_64-w64-windows-gnu',   # Windows(MinGW ABI/COFF)
    # 以下は確認時点でローカルclang(Apple clang 21)にもバックエンドが無く、
    # 確認できたリモート機(Ubuntu 20.04、gcc 9.4.0のみ・clang/リンカ含め
    # クロスツールチェーン皆無)にも無いため、現状は自動skipになる。
    # 標準C99の範囲しか使っていないのでビルドできるはずだが実績はまだ無い
    # (標準準拠につき理論上可・未検証。docs/reference/platform-support.md)。
    'riscv32-unknown-elf',        # RISC-V 32bit組込み
    'riscv64-unknown-elf',        # RISC-V 64bit組込み
    'riscv64-unknown-linux-gnu',  # RISC-V Linux
    'mipsel-unknown-linux-gnu',   # MIPS(リトルエンディアン)Linux
    'wasm32-unknown-unknown',     # freestanding WASM(C版。webbenchのTS/AS製ランタイムとは別経路)
]


def _compile_kernel(cc, tmp_path, *, target=None, extra=()):
    cmd = [cc, '-std=c99', '-Os', '-ffreestanding', '-DMDL_NO_LIBC',
           '-c', f'-I{RUNTIME_C / "include"}', *extra]
    if target:
        cmd += ['-target', target]
    obj = tmp_path / f'k_{target or "host"}.o'
    cmd += [str(KERNEL_C), '-o', str(obj)]
    return subprocess.run(cmd, capture_output=True, text=True), obj


class TestFreestanding:
    def test_builds_without_libc_headers(self, cc, tmp_path):
        """-DMDL_NO_LIBC + -ffreestanding でホストヘッダ無しに通る。"""
        proc, obj = _compile_kernel(cc, tmp_path,
                                    extra=('-Wall', '-Wextra', '-Werror'))
        assert proc.returncode == 0, proc.stderr
        assert obj.stat().st_size > 0

    def test_only_stdint_and_stddef_are_included(self):
        """ヘッダはフリースタンディングでも必ずある2つだけを使う。"""
        text = KERNEL_H.read_text(encoding='utf-8')
        includes = {line.split('<')[1].split('>')[0]
                    for line in text.splitlines()
                    if line.strip().startswith('#include <')}
        assert includes <= {'stdint.h', 'stddef.h'}, (
            f'ホスト専用ヘッダを使っている: {includes}')

    def test_no_byte_order_dependent_casts(self):
        """uint8_t* と uint32_t* を相互キャストしていない(バイト順非依存)。"""
        source = KERNEL_C.read_text(encoding='utf-8')
        for bad in ('(uint32_t *)', '(uint32_t*)',
                    '(const uint32_t *)', '(const uint32_t*)'):
            assert bad not in source, (
                f'{bad} へのキャストがある: バイト順依存の恐れ')

    def test_no_heap_or_io_in_runtime(self):
        """実行時のヒープ確保・I/O・浮動小数点ライブラリを使わない。"""
        source = KERNEL_C.read_text(encoding='utf-8')
        for bad in ('malloc', 'calloc', 'free(', 'printf', 'fopen',
                    'sqrt', 'pow(', 'double ', 'float '):
            assert bad not in source, f'ランタイムに {bad} がある'

    def test_freestanding_build_gives_same_numbers(self, cc, tmp_path,
                                                   seg_doc):
        """libc版とフリースタンディング版で出力が1ビットも変わらない。"""
        bundle = build_bundle(seg_doc)
        graph = ir_from_dict(bundle['ir'])
        n_out = int(graph.outputs[0].width)
        plan = bundle['preproc_desc']
        rng = np.random.default_rng(5)
        image = rng.integers(
            0, 256,
            size=(plan['image_channels'], plan['height'], plan['width']),
            dtype=np.uint8)

        host = build_c_model(bundle, tmp_path / 'host', cc=cc, image_entry=True)
        want = run_c_model(host, image.tobytes(), n_out)

        # フリースタンディング版を手で組み立てる(build_c_model は既定フラグのため)
        work = tmp_path / 'free'
        free_exe = build_c_model(bundle, work, cc=cc, image_entry=True)
        assert free_exe.exists()
        rebuild = subprocess.run(
            [cc, '-std=c99', '-Os', '-DMDL_NO_LIBC',
             f'-I{work}', f'-I{RUNTIME_C / "include"}',
             str(work / 'model.c'), str(KERNEL_C), str(work / 'driver.c'),
             '-o', str(work / 'app_free')],
            capture_output=True, text=True)
        assert rebuild.returncode == 0, rebuild.stderr
        got = run_c_model(work / 'app_free', image.tobytes(), n_out)
        assert np.array_equal(got, want)


def _is_clang(compiler: str) -> bool:
    r = subprocess.run([compiler, '--version'], capture_output=True, text=True)
    return 'clang' in (r.stdout + r.stderr).lower()


@pytest.fixture(scope='module')
def clang(cc):
    """`-target` を受け付けるコンパイラ(clang)。gcc しか無い環境では skip。"""
    if _is_clang(cc):
        return cc
    found = shutil.which('clang')
    if found:
        return found
    pytest.skip('クロスコンパイル(-target)は clang 専用。gcc しか無い環境では検証できない')


class TestCrossCompile:
    @pytest.mark.parametrize('target', CROSS_TARGETS)
    def test_compiles_for_target(self, clang, tmp_path, target):
        """MCUから64bitまで、同一ソースがそのまま通る(命令セット別配布は不要)。"""
        proc, obj = _compile_kernel(clang, tmp_path, target=target)
        if proc.returncode != 0 and 'unable to create target' in proc.stderr:
            pytest.skip(f'ローカルclangに {target} のバックエンドが無い')
        assert proc.returncode == 0, proc.stderr
        assert obj.stat().st_size > 0

    def test_generated_model_also_cross_compiles(self, cc, clang, tmp_path,
                                                 seg_doc):
        """生成されたモデルコード側もアーキ非依存。"""
        target = 'armv7m-none-eabi'
        work = tmp_path / 'gen'
        build_c_model(build_bundle(seg_doc), work, cc=cc, image_entry=True)
        proc = subprocess.run(
            [clang, '-std=c99', '-Os', '-ffreestanding', '-DMDL_NO_LIBC',
             '-target', target, '-c', f'-I{work}',
             f'-I{RUNTIME_C / "include"}',
             str(work / 'model.c'), '-o', str(tmp_path / 'model_arm.o')],
            capture_output=True, text=True)
        if proc.returncode != 0 and 'unable to create target' in proc.stderr:
            pytest.skip(f'ローカルclangに {target} のバックエンドが無い')
        assert proc.returncode == 0, proc.stderr


class TestSIMDContract:
    """移植性契約の中核(監査で裏取りした主張を機械的に固定する)。

    「C99コンパイラがあるターゲットなら(SIMDはスカラーへフォールバックして)
    動く」の内訳:
        1. アーキ固有コードは mdl_simd.h の1ファイルに隔離されている
        2. コアはバイト順・アライメントを仮定するポインタキャストを持たない
           (ワード演算はビット位置で定義してあり、メモリ上の型再解釈をしない)
        3. SIMD経路とスカラー経路(`-DMDL_NO_SIMD`)は実モデルでビット完全一致

    旧版はこの3点のうち(3)を機械的に検証するテストが無かった
    (README/platform-support.mdの「テストが両経路の一致を固定している」という
    記述が、実際には以前のリモート機での手動ベンチ結果
    `results/deploy/cpu_x86_k57_ab.json` を指しており、リポジトリのpytestには
    対応するテストが存在しなかった)。
    """

    _ARCH_MACRO_RE = re.compile(
        r'__ARM_NEON|__SSE2__|__AVX2__|__aarch64__|_M_X64|_M_IX86'
        r'|__x86_64|__asm__|\basm\b|__builtin_(neon|ia32)')
    _BYTE_CAST_RE = re.compile(
        r'\(\s*(const\s+)?'
        r'(uint8_t|uint16_t|uint32_t|int16_t|int32_t)\s*\*\s*\)')

    @staticmethod
    def _grep_offenders(paths, pattern):
        offenders = []
        for path in paths:
            for lineno, line in enumerate(
                    path.read_text(encoding='utf-8').splitlines(), 1):
                if pattern.search(line):
                    offenders.append(f'{path.name}:{lineno}: {line.strip()}')
        return offenders

    def test_arch_macros_isolated_to_simd_header(self):
        """アーキ検出マクロ・インラインasmは mdl_simd.h の外に1件も無い
        (Part1-1監査の恒久化)。mdl_simd.h 自身にはあってよい(隔離先そのもの)。
        """
        offenders = self._grep_offenders(CORE_FILES, self._ARCH_MACRO_RE)
        assert not offenders, (
            'mdl_simd.h の外にアーキ固有コードがある(隔離が破れている):\n'
            + '\n'.join(offenders))
        # 隔離先そのものには存在すること(検出パターンの健全性チェック)。
        assert self._ARCH_MACRO_RE.search(SIMD_H.read_text(encoding='utf-8'))

    def test_no_byte_reinterpret_casts_in_core(self):
        """uint8_t*/uint16_t*/int16_t*/int32_t*/uint32_t* への相互キャストが
        コアに無い(Part1-2監査の恒久化)。ビットプレーンのワード演算はビット
        位置(`x>>5`、`x&31`)で定義してあり、メモリ上の型再解釈をしないので、
        バイト順にもアライメントにも依存しない設計であるはずという主張を固定する。
        """
        offenders = self._grep_offenders(CORE_FILES, self._BYTE_CAST_RE)
        assert not offenders, (
            '型キャストがある(バイト順・アライメント仮定を持ち込む恐れ):\n'
            + '\n'.join(offenders))

    def test_scalar_forced_build_matches_host_default_build(
            self, cc, tmp_path, seg_doc):
        """`-DMDL_NO_SIMD` のスカラー強制ビルドと、このホストの既定ビルド
        (NEON/SSE2/AVX2いずれかが自動選択される)が実モデルでビット完全一致する。

        README.md/platform-support.mdが主張する「SIMD経路とスカラー経路は
        整数出力が完全に同一」を、以前の手動ベンチ(リモートXeon機、
        `-mavx2`/`-mno-avx2`のA/B)ではなく、このリポジトリのpytestとして
        再現可能な形で固定する。ホストのSIMD経路(このMacならNEON、
        典型的なx86 CIならSSE2)は `mdl_pack_planes`(`mdl_mask32`)・
        前処理(`mdl_ge_mask16`)・`mdl_lut_layer_slice`経由で実際に叩かれる
        (seg_doc は image_thermometer 前処理付きの畳み込みモデル)。
        """
        bundle = build_bundle(seg_doc)
        graph = ir_from_dict(bundle['ir'])
        n_out = int(graph.outputs[0].width)
        plan = bundle['preproc_desc']
        rng = np.random.default_rng(7)
        image = rng.integers(
            0, 256,
            size=(plan['image_channels'], plan['height'], plan['width']),
            dtype=np.uint8)

        default_exe = build_c_model(bundle, tmp_path / 'default', cc=cc,
                                    image_entry=True)
        default_out = run_c_model(default_exe, image.tobytes(), n_out)

        # build_c_model は既定フラグ固定(SIMD自動選択)なので、
        # -DMDL_NO_SIMD 版は model.c/driver.c を使い回して手で組む。
        work = tmp_path / 'scalar'
        build_c_model(bundle, work, cc=cc, image_entry=True)
        scalar_exe = work / 'app_scalar'
        build = subprocess.run(
            [cc, '-std=c99', '-O2', '-DMDL_NO_SIMD',
             f'-I{work}', f'-I{RUNTIME_C / "include"}',
             str(work / 'model.c'), str(KERNEL_C), str(work / 'driver.c'),
             '-o', str(scalar_exe)],
            capture_output=True, text=True)
        assert build.returncode == 0, build.stderr
        scalar_out = run_c_model(scalar_exe, image.tobytes(), n_out)

        assert np.array_equal(scalar_out, default_out)

    @pytest.mark.parametrize('doc_fixture', ['seg_doc', 'dense_doc'])
    def test_generated_model_has_no_arch_specific_code(self, doc_fixture,
                                                        request):
        """cbackend が吐く model.c/model.h 側にもアーキ固有コードが無い
        (コア以外の生成物も「配布物は命令セット別に分かれない」契約を守る)。

        バイト再解釈キャストの不在は**ここでは検査しない**(監査で判明:
        生成コードは `ctx->work`(コンテキストの`uint32_t`アリーナ)を
        `(uint8_t *)`/`(int32_t *)` へ型付け直すキャストを、バッファ種別
        (KIND_FLAT の0/1バイト列、group_sum等のint32スコア)ごとに常に生成する
        —— `griffith_logic/deploy/cbackend.py::_declare_buffers`。これは
        「共有アリーナに対する型付きビュー」であって「意味のある多バイト値を
        別の型で再解釈する」ことはしない(各バッファはその生存区間内で常に
        同じ型のビューだけを通してアクセスされ、KIND_FLATは1バイト=1個の
        0/1値なのでバイト順の入る余地が無い)。dense_doc で実際にこの形の
        キャストが出ることを確認済みで、コア側の
        `test_no_byte_reinterpret_casts_in_core`(mdl_kernels.c/.h、
        mdl_rddlgn.c/.h)のような例外ゼロの機械チェックにはできないため、
        生成コードについてはアーキ固有コードの不在だけを恒久チェックにする。
        """
        from griffith_logic.deploy.cbackend import generate_c

        doc = request.getfixturevalue(doc_fixture)
        bundle = build_bundle(doc)
        files = generate_c(bundle, name='model', validate=False)
        arch_offenders = []
        for filename, text in files.items():
            for lineno, line in enumerate(text.splitlines(), 1):
                if self._ARCH_MACRO_RE.search(line):
                    arch_offenders.append(f'{filename}:{lineno}: {line.strip()}')
        assert not arch_offenders, (
            '生成コードにアーキ固有コードがある:\n' + '\n'.join(arch_offenders))


@pytest.mark.slow
class TestSanitizers:
    def test_no_undefined_behaviour_on_real_graph(self, cc, tmp_path, seg_doc):
        """UBSan/ASan で実モデルを1回通す(符号・シフト・境界の検査)。"""
        bundle = build_bundle(seg_doc)
        plan = bundle['preproc_desc']
        work = tmp_path / 'san'
        build_c_model(bundle, work, cc=cc, image_entry=True)
        build = subprocess.run(
            [cc, '-std=c99', '-O1', '-g',
             '-fsanitize=address,undefined', '-fno-sanitize-recover=all',
             f'-I{work}', f'-I{RUNTIME_C / "include"}',
             str(work / 'model.c'), str(KERNEL_C), str(work / 'driver.c'),
             '-o', str(work / 'app_san')],
            capture_output=True, text=True)
        if build.returncode != 0:
            pytest.skip(f'サニタイザ付きビルド不可: {build.stderr[:200]}')
        rng = np.random.default_rng(6)
        image = rng.integers(
            0, 256,
            size=(plan['image_channels'], plan['height'], plan['width']),
            dtype=np.uint8)
        proc = subprocess.run([str(work / 'app_san')],
                              input=image.tobytes(), capture_output=True)
        assert proc.returncode == 0, proc.stderr.decode(errors='replace')
        assert b'runtime error' not in proc.stderr
