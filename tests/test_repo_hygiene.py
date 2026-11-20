"""リポジトリ衛生: プライベートなホスト情報(user@IP等・私有マシンの愛称)を
追跡ファイルに載せないことの機械的強制。

背景: 開発で使うリモートGPU機のSSH宛先(ユーザー名@Tailscale IP)が
スクリプトのdocstring例やdocsに混入し、公開リポジトリへ露出しかけた
(にユーザー指摘で除去)。続けて裸のマシン愛称もdocs/skills等
から除去し(GPU機A〜Eの匿名ラベルへ置換、スペックは保持)、さらに
にブランチ履歴全体からも書き換えで除去した。例示は
`<user>@<gpu-host>` のようなプレースホルダで書くこと。

★設計上の要点: 禁止対象の具体値をこのファイルに**平文で書くと、
このファイル自身が機密の掲載になる**(実際、履歴書き換えの際に平文の
禁止リストが巻き添えで置換され、匿名ラベルを禁止する壊れたテストに
なった)。そのため具体値は必ず`_lit()`の断片連結で組み立て、平文が
blobに現れないようにする。汎用パターン(CGNAT帯user@IP等)は平文でよい。

愛称のうち1語だけは「カナリア学習」(小規模お試し学習の技法用語)と
衝突するため、裸の単語では禁止せずSSH宛先/コマンド形(`<名前>@`・
`ssh <名前>`)のみを禁止する。他の4語は本リポジトリで技術用語と
衝突しないことをgrepで確認済み(英単語の内部に偶然含まれる並び——
群れを表す英単語や「知識」の英語などの内側——は\b境界で除外される)。
"""
import re
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


def _lit(*parts: str) -> str:
    """機密文字列を断片連結で組み立てる(平文をこのblobに残さないため)。"""
    return ''.join(parts)


_NAME_A = _lit('human', 'ophilic')   # リモートGPU機A(x86+4070 Ti)の旧愛称
_NAME_B = _lit('ki', 'wi')           # GPU機B(RTX 4090)の旧愛称
_NAME_C = _lit('cr', 'ow')           # GPU機C(RTX 8000×2)の旧愛称
_NAME_D = _lit('can', 'ary')        # GPU機D。技法用語と衝突するため形を限定
_NAME_E = _lit('o', 'wl')            # GPU機Eの旧愛称
_JUMP = _lit('ura', 'nus')           # SSHジャンプホストの旧愛称
_IP_A = _lit(r'100\.80\.', r'247\.83')   # GPU機AのTailscale IP
_IP_JUMP = _lit(r'172\.16\.', r'0\.66')  # ジャンプホスト側の内部IP(RFC1918)

#: 禁止パターン(見つかったら混入。プレースホルダ化すること)
FORBIDDEN = [
    re.compile(_IP_A),
    re.compile(_IP_JUMP),
    re.compile(rf'\b{_NAME_A}@'),
    # user@CGNAT帯IP(100.64.0.0/10、Tailscale等)の一般形
    re.compile(r'\b[a-z][a-z0-9_-]{1,31}@100\.'
               r'(6[4-9]|[7-9]\d|1[01]\d|12[0-7])\.\d{1,3}\.\d{1,3}'),
    # 私有GPU機の裸の愛称(匿名化済み。\bで部分一致を除外)
    re.compile(rf'\b{_NAME_A}\b'),
    re.compile(rf'\b{_NAME_B}\b'),
    re.compile(rf'\b{_NAME_C}\b'),
    re.compile(rf'\b{_NAME_E}\b'),
    re.compile(rf'\b{_JUMP}\b'),
    # 技法用語と衝突する1語はSSH宛先/コマンド形のみ禁止
    re.compile(rf'\b{_NAME_D}@'),
    re.compile(rf'\bssh {_NAME_D}\b'),
]

TEXT_SUFFIXES = {'.py', '.md', '.sh', '.c', '.h', '.cu', '.mm', '.metal',
                 '.ts', '.tsx', '.js', '.mjs', '.json', '.toml', '.yml',
                 '.yaml', '.txt', '.cfg', '.ini'}


def _tracked_files() -> list[Path]:
    # 公開ツリーの組み立て結果(git 管理外のディレクトリ)でも実行されうるため、
    # git が使えない場所では検査を skip する(検査を緩めるのではなく、
    # 「追跡ファイル」という概念が存在しない環境を除外するだけ)。
    proc = subprocess.run(['git', 'ls-files'], cwd=REPO, capture_output=True,
                          text=True)
    if proc.returncode != 0:
        pytest.skip('git 管理下でないため追跡ファイルを列挙できない')
    return [REPO / line for line in proc.stdout.splitlines()
            if Path(line).suffix in TEXT_SUFFIXES]


def test_no_private_host_identifiers_in_tracked_files():
    hits = []
    for path in _tracked_files():
        if path.name == 'test_repo_hygiene.py':
            continue  # 断片連結とはいえ、このファイル自身は検査対象外
        try:
            text = path.read_text(encoding='utf-8')
        except (UnicodeDecodeError, FileNotFoundError):
            continue
        for pat in FORBIDDEN:
            for m in pat.finditer(text):
                line = text.count('\n', 0, m.start()) + 1
                hits.append(f'{path.relative_to(REPO)}:{line}: {m.group(0)}')
    assert not hits, (
        'プライベートなホスト情報が追跡ファイルに混入している。'
        'プレースホルダ(<user>@<gpu-host>等)へ置換すること:\n'
        + '\n'.join(hits))


def test_forbidden_literals_are_not_plaintext_in_this_file():
    """このファイル自身に機密の平文が現れていないこと(断片連結の強制)。"""
    src = Path(__file__).read_text(encoding='utf-8')
    for name in (_NAME_A, _NAME_B, _NAME_C, _NAME_D, _NAME_E, _JUMP):
        assert name not in src, f'平文 {name!r} を書かず_lit()で組み立てること'
