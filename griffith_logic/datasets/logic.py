"""論理関数データセット生成器。

エディタのサンプル回路(XOR/パリティ/半加算器/全加算器/MUX)を
そのまま学習できる TensorDataset をコアライブラリとして提供する。

真理値表の規約: butterfly基底順。入力パターン k のビット j = (k >> j) & 1
(ビジュアルエディタのサンプルIR・LutLayerの基底計算と同じ規約)。
"""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from functools import partial

import torch
from torch.utils.data import DataLoader, TensorDataset

# 真理値表の指定方法(truth_table_dataset の table 引数):
#   (a) 関数        fn(入力ビットのtuple) -> 出力ビット列(intも可)
#   (b) リスト      出力ビット列の列(butterflyパターン順、長さ2^n_inputs)
#   (c) マッピング  {入力ビットのtuple(またはパターン番号k): 出力ビット列}
TruthTable = (
    Callable[[tuple[int, ...]], 'Sequence[int] | int']
    | Sequence['Sequence[int] | int']
    | Mapping
)


def _normalize_output_bits(value, n_outputs: int, context: str) -> tuple[int, ...]:
    """出力ビット列を{0,1}のtupleに正規化する(int/boolの単独値も許容)"""
    if isinstance(value, int | bool) and not isinstance(value, torch.Tensor):
        bits = (int(value),)
    else:
        bits = tuple(int(v) for v in value)
    if len(bits) != n_outputs:
        raise ValueError(
            f'{context}: 出力ビット数({len(bits)})がn_outputs({n_outputs})と一致しません'
        )
    if any(b not in (0, 1) for b in bits):
        raise ValueError(f'{context}: 出力ビットは0/1のみ有効です: {bits}')
    return bits


def _resolve_outputs(
    table: TruthTable,
    input_rows: list[tuple[int, ...]],
    n_outputs: int,
) -> list[tuple[int, ...]]:
    """table(関数/リスト/マッピング)から全パターンの出力ビット列を得る"""
    n_patterns = len(input_rows)
    if callable(table):
        return [
            _normalize_output_bits(table(bits), n_outputs, f'パターンk={k}')
            for k, bits in enumerate(input_rows)
        ]
    if isinstance(table, Mapping):
        rows = []
        for k, bits in enumerate(input_rows):
            if bits in table:
                value = table[bits]
            elif k in table:
                value = table[k]
            else:
                raise KeyError(f'真理値表にパターンk={k}(入力{bits})がありません')
            rows.append(_normalize_output_bits(value, n_outputs, f'パターンk={k}'))
        return rows
    if isinstance(table, Sequence) and not isinstance(table, str | bytes):
        if len(table) != n_patterns:
            raise ValueError(
                f'真理値表の長さ({len(table)})が2^n_inputs({n_patterns})と一致しません'
            )
        return [
            _normalize_output_bits(value, n_outputs, f'パターンk={k}')
            for k, value in enumerate(table)
        ]
    raise TypeError(f'tableは関数/リスト/マッピングのいずれかを指定してください: {type(table)}')


def truth_table_dataset(
    table: TruthTable,
    n_inputs: int,
    n_outputs: int,
    *,
    repeat: int = 1,
) -> TensorDataset:
    """真理値表から2^n_inputs全列挙のTensorDatasetを生成する。

    Args:
        table: 真理値表。関数 fn(入力ビットのtuple)->出力ビット列、
            出力ビット列のリスト(butterflyパターン順: パターンkのビットj=(k>>j)&1)、
            または {入力ビットのtuple|パターン番号k: 出力ビット列} のマッピング
        n_inputs: 入力ビット数
        n_outputs: 出力ビット数
        repeat: 全パターンの複製回数(バッチ学習でのepoch調整用)

    Returns:
        TensorDataset(X, Y)。X: (2^n_inputs·repeat, n_inputs) float32 {0,1}、
        Y: (2^n_inputs·repeat, n_outputs) float32 {0,1}
    """
    if n_inputs < 1:
        raise ValueError(f'n_inputsは1以上を指定してください: {n_inputs}')
    if n_outputs < 1:
        raise ValueError(f'n_outputsは1以上を指定してください: {n_outputs}')
    if repeat < 1:
        raise ValueError(f'repeatは1以上を指定してください: {repeat}')

    n_patterns = 1 << n_inputs
    input_rows = [
        tuple((k >> j) & 1 for j in range(n_inputs)) for k in range(n_patterns)
    ]
    output_rows = _resolve_outputs(table, input_rows, n_outputs)

    x = torch.tensor(input_rows, dtype=torch.float32)
    y = torch.tensor(output_rows, dtype=torch.float32)
    if repeat > 1:
        x = x.repeat(repeat, 1)
        y = y.repeat(repeat, 1)
    return TensorDataset(x, y)


@dataclass(frozen=True)
class LogicDatasetSpec:
    """論理関数データセットの仕様(エディタバックエンド等からレジストリ的に引く単位)"""

    name: str
    n_inputs: int
    n_outputs: int
    dataset: TensorDataset

    def loader(self, batch_size: int | None = None, *, shuffle: bool = False) -> DataLoader:
        """DataLoaderを生成する。batch_size省略時は全パターン一括"""
        if batch_size is None:
            batch_size = len(self.dataset)
        return DataLoader(self.dataset, batch_size=batch_size, shuffle=shuffle)


def xor2(*, repeat: int = 1) -> LogicDatasetSpec:
    """2入力XOR: (a,b) → a⊕b"""
    return LogicDatasetSpec(
        name='xor',
        n_inputs=2,
        n_outputs=1,
        dataset=truth_table_dataset(lambda b: b[0] ^ b[1], 2, 1, repeat=repeat),
    )


def parity(n: int, *, repeat: int = 1) -> LogicDatasetSpec:
    """n入力パリティ: 全ビットのXOR(立っているビット数の偶奇)"""
    return LogicDatasetSpec(
        name=f'parity{n}',
        n_inputs=n,
        n_outputs=1,
        dataset=truth_table_dataset(lambda b: sum(b) % 2, n, 1, repeat=repeat),
    )


def majority3(*, repeat: int = 1) -> LogicDatasetSpec:
    """3入力多数決: 2bit以上が1なら1。"""
    return LogicDatasetSpec(
        name='majority3',
        n_inputs=3,
        n_outputs=1,
        dataset=truth_table_dataset(lambda b: int(sum(b) >= 2), 3, 1, repeat=repeat),
    )


def half_adder(*, repeat: int = 1) -> LogicDatasetSpec:
    """半加算器: (a,b) → (sum=a⊕b, carry=a∧b)"""
    return LogicDatasetSpec(
        name='half_adder',
        n_inputs=2,
        n_outputs=2,
        dataset=truth_table_dataset(
            lambda b: (b[0] ^ b[1], b[0] & b[1]), 2, 2, repeat=repeat
        ),
    )


def full_adder(*, repeat: int = 1) -> LogicDatasetSpec:
    """全加算器: (a,b,cin) → (sum, carry)"""

    def fn(b: tuple[int, ...]) -> tuple[int, int]:
        total = b[0] + b[1] + b[2]
        return total % 2, total // 2

    return LogicDatasetSpec(
        name='full_adder',
        n_inputs=3,
        n_outputs=2,
        dataset=truth_table_dataset(fn, 3, 2, repeat=repeat),
    )


def nbit_adder(n: int, *, repeat: int = 1) -> LogicDatasetSpec:
    """nビットripple-carry加算器: (a, b) → a+b(桁上げ込み)

    入力=2n bit。LSB-first(下位ビット先)で a を n bit、続けて b を n bit
    連結する: 入力ビット列 = (a_0, a_1, ..., a_{n-1}, b_0, ..., b_{n-1})。
    （butterfly規約のパターン番号kのビットj=(k>>j)&1と同じ並びを流用し、
    a はビット0..n-1、b はビットn..2n-1に対応させる。）

    出力=n+1 bit。LSB-firstで和 a+b の下位nビットを並べ、最後の1ビット
    (出力ビットn)が最終桁からの繰り上がり(carry-out)。すなわち
    出力を同じLSB-first規約で整数に戻すと a+b にちょうど一致する
    (n+1ビット幅なので桁あふれしない)。

    真理値表はPythonの素朴な整数加算(a+b)で計算する
    (`core/reference.py` 同様、素朴さを優先しhaltする最適化はしない)。

    Args:
        n: 各オペランドのビット幅(n>=1)
        repeat: 全パターンの複製回数

    Returns:
        LogicDatasetSpec(name=f'adder{n}', n_inputs=2n, n_outputs=n+1)
    """
    if n < 1:
        raise ValueError(f'nは1以上を指定してください: {n}')

    def fn(b: tuple[int, ...]) -> tuple[int, ...]:
        a_val = sum(b[j] << j for j in range(n))
        b_val = sum(b[n + j] << j for j in range(n))
        total = a_val + b_val  # 素朴な整数加算(ground truth)
        return tuple((total >> j) & 1 for j in range(n + 1))

    return LogicDatasetSpec(
        name=f'adder{n}',
        n_inputs=2 * n,
        n_outputs=n + 1,
        dataset=truth_table_dataset(fn, 2 * n, n + 1, repeat=repeat),
    )


def nbit_multiplier(n: int, *, repeat: int = 1) -> LogicDatasetSpec:
    """nビット×nビット符号なし整数乗算器: (a, b) → a*b

    入力=2n bit。LSB-first(下位ビット先)で a を n bit、続けて b を n bit
    連結する: 入力ビット列 = (a_0, a_1, ..., a_{n-1}, b_0, ..., b_{n-1})
    (nbit_adder と同じ規約)。

    出力=2n bit。LSB-firstで積 a*b を並べる。a, b は最大 2^n - 1 なので
    a*b の最大値は (2^n - 1)^2 < 2^(2n) - 1(n>=1で常に成立)であり、
    2n bit幅に桁あふれなく収まる。すなわち出力を同じLSB-first規約で
    整数に戻すと a*b にちょうど一致する。

    真理値表はPythonの素朴な整数乗算(a*b)で計算する
    (`core/reference.py` 同様、素朴さを優先し最適化はしない)。

    Args:
        n: 各オペランドのビット幅(n>=1)
        repeat: 全パターンの複製回数

    Returns:
        LogicDatasetSpec(name=f'mul{n}', n_inputs=2n, n_outputs=2n)
    """
    if n < 1:
        raise ValueError(f'nは1以上を指定してください: {n}')

    def fn(b: tuple[int, ...]) -> tuple[int, ...]:
        a_val = sum(b[j] << j for j in range(n))
        b_val = sum(b[n + j] << j for j in range(n))
        total = a_val * b_val  # 素朴な整数乗算(ground truth)
        return tuple((total >> j) & 1 for j in range(2 * n))

    return LogicDatasetSpec(
        name=f'mul{n}',
        n_inputs=2 * n,
        n_outputs=2 * n,
        dataset=truth_table_dataset(fn, 2 * n, 2 * n, repeat=repeat),
    )


def mux(n_sel: int = 1, *, repeat: int = 1) -> LogicDatasetSpec:
    """2^n_sel:1マルチプレクサ: 入力=[データ(2^n_sel bit), セレクト(n_sel bit)]

    エディタのSAMPLE_MUX21と同じ入力順(データが先、セレクトが後)。
    出力 = データビット[セレクト値]。
    """
    n_data = 1 << n_sel

    def fn(b: tuple[int, ...]) -> int:
        sel = sum(b[n_data + j] << j for j in range(n_sel))
        return b[sel]

    return LogicDatasetSpec(
        name=f'mux{n_data}',
        n_inputs=n_data + n_sel,
        n_outputs=1,
        dataset=truth_table_dataset(fn, n_data + n_sel, 1, repeat=repeat),
    )


def kv_attention2(*, repeat: int = 1) -> LogicDatasetSpec:
    """1bit query と2組の(key,value)によるlookup toy。

    入力順は (q, k0, v0, k1, v1)。
    出力は (q == k0 and v0) or (q == k1 and v1)。
    本物のAttentionではなく、query/key一致でvalueを読む教材タスク。
    """

    def fn(b: tuple[int, ...]) -> int:
        q, k0, v0, k1, v1 = b
        return int(((q == k0) and v0) or ((q == k1) and v1))

    return LogicDatasetSpec(
        name='kv_attention2',
        n_inputs=5,
        n_outputs=1,
        dataset=truth_table_dataset(fn, 5, 1, repeat=repeat),
    )


def seq101(*, repeat: int = 1) -> LogicDatasetSpec:
    """3stepの系列で 1→0→1 を検出するRNN風教材タスク。"""
    return LogicDatasetSpec(
        name='seq101',
        n_inputs=3,
        n_outputs=1,
        dataset=truth_table_dataset(lambda b: int(b == (1, 0, 1)), 3, 1, repeat=repeat),
    )


def random_logic(
    n_inputs: int, n_outputs: int, seed: int, *, repeat: int = 1
) -> LogicDatasetSpec:
    """seed固定のランダム真理値表(同一seedで常に同一のデータセット)"""
    generator = torch.Generator().manual_seed(seed)
    table = torch.randint(
        0, 2, (1 << n_inputs, n_outputs), generator=generator
    ).tolist()
    return LogicDatasetSpec(
        name=f'random{n_inputs}x{n_outputs}_seed{seed}',
        n_inputs=n_inputs,
        n_outputs=n_outputs,
        dataset=truth_table_dataset(table, n_inputs, n_outputs, repeat=repeat),
    )


# エディタバックエンド等から名前で引くレジストリ(全てdeterministic)。
# 値はLogicDatasetSpecを返すファクトリ(repeat等のキーワード引数も透過する)
LOGIC_DATASETS: dict[str, Callable[..., LogicDatasetSpec]] = {
    'xor': xor2,
    'parity3': partial(parity, 3),
    'parity4': partial(parity, 4),
    'majority3': majority3,
    'half_adder': half_adder,
    'full_adder': full_adder,
    'mux2': partial(mux, 1),
    'kv_attention2': kv_attention2,
    'seq101': seq101,
    'adder2': partial(nbit_adder, 2),
    'adder3': partial(nbit_adder, 3),
    'adder4': partial(nbit_adder, 4),
    'mul2': partial(nbit_multiplier, 2),
    'mul3': partial(nbit_multiplier, 3),
    'mul4': partial(nbit_multiplier, 4),
}
