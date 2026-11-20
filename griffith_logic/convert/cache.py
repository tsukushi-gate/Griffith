"""teacherロジットのキャッシュ+packed-target蒸留データセット(ROADMAP-v5 Track D)。

蒸留(C4)では毎エポック teacher を推論すると重い(かつ decodeが決定的なら
無駄)。そこで teacher ロジットを1度だけ計算して `data-teacher-cache/` に
torch.save し、以降はロードで済ませる(内容ベースのキャッシュキー)。

キャッシュキーは teacher.fingerprint(重みが変われば必ず変わる、
convert/teacher.py 参照)+ データセット同定情報から作る。fingerprint が
効いているので、同じデータセットでも別モデルのロジットとは衝突しない。

不変条件(**重要**、`get_or_compute_logits` / `PackedDistillDataset` の前提)
================================================================================
ロジットは「datasetのインデックス i → teacherロジット[i]」の**インデックス
整列**で student 側と対応付ける。この整列が健全なのは、student/teacher が
包む基底データセットの `__getitem__(i)` が**決定的**(同じ i なら常に同じ
サンプル)である場合に限る。すなわち:

    - datasets/vision.py の変換は ToTensor/Binarize 等の決定的変換のみで、
      ランダムな augmentation(RandomCrop/RandomFlip 等)を含まないこと。

将来 augmentation を導入する場合、キャッシュされたロジットは学習中に変動する
入力と整合しなくなる。その場合はキャッシュを捨て、学習ループ内で毎バッチ
オンラインに teacher.predict_logits を呼ぶ方式へ切り替えること。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from .teacher import Teacher

__all__ = [
    'PackedDistillDataset',
    'get_or_compute_logits',
    'teacher_cache_key',
]


def teacher_cache_key(
    teacher_fingerprint: str,
    dataset_name: str,
    params: dict,
    split: str,
    n: int,
) -> str:
    """teacherロジットキャッシュの安定キーを作る。

    sha256(fingerprint | dataset | sorted(params) | split | n) の先頭16桁。
    paramsは順序非依存にするため sorted(JSON) で正規化する(同じ内容の
    dictは同じキー)。fingerprintが重み依存なので、重みが変われば
    キーも変わり自動的にキャッシュが無効化される。

    Args:
        teacher_fingerprint: Teacher.fingerprint(重み依存の安定識別子)。
        dataset_name: データセット名(例 'mnist')。
        params: ローダへ渡すパラメータ(binarize_threshold等)。
        split: 'train' / 'test' 等の分割名。
        n: サンプル数(データセット長)。

    Returns:
        16桁の16進文字列(ファイル名として安全)。
    """
    params_norm = json.dumps(params, sort_keys=True, ensure_ascii=False, default=str)
    payload = '|'.join([
        teacher_fingerprint, dataset_name, params_norm, split, str(int(n))
    ])
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()[:16]


def get_or_compute_logits(
    teacher: Teacher,
    dataset: Dataset,
    *,
    key: str,
    batch_size: int = 512,
    device: str | torch.device | None = None,
    root: str = 'data-teacher-cache',
) -> torch.Tensor:
    """teacherロジット [N, C] を計算(またはキャッシュからロード)する。

    `root/<key>.pt` が存在すればそれをロードして返す。無ければ dataset 全体を
    バッチ推論してロジットを組み立て、torch.save してから返す。

    不変条件(モジュール docstring 参照): dataset の `__getitem__` は決定的で
    augmentation を含まないこと。この前提により、返るロジットはインデックス
    整列(logits[i] が dataset[i] に対応)として student と対応付けられる。
    将来 augmentation を入れる場合はキャッシュを使わずオンライン教師推論に
    切り替えること。

    Args:
        teacher: Teacherプロトコル実装(predict_logits/num_classes)。
        dataset: 推論対象データセット。`(x, y)` を返す前提(y は無視)。
        key: teacher_cache_key で作ったキー(ファイル名に使う)。
        batch_size: 推論バッチサイズ。
        device: teacher推論に載せるデバイス(TorchTeacher側で移動)。
        root: キャッシュディレクトリ(既定 data-teacher-cache/、.gitignore済み)。

    Returns:
        [N, num_classes] のロジット(CPU、float)。dataset のインデックス順。
    """
    cache_dir = Path(root)
    cache_path = cache_dir / f'{key}.pt'
    if cache_path.is_file():
        # 自前で書いたテンソルのみのキャッシュ。weights_only=Trueを明示
        return torch.load(cache_path, map_location='cpu', weights_only=True)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    chunks: list[torch.Tensor] = []
    for batch in loader:
        x = batch[0] if isinstance(batch, list | tuple) else batch
        logits = teacher.predict_logits(x)
        chunks.append(logits.detach().to('cpu'))
    all_logits = torch.cat(chunks, dim=0) if chunks else torch.empty(
        0, teacher.num_classes
    )

    cache_dir.mkdir(parents=True, exist_ok=True)
    torch.save(all_logits, cache_path)
    return all_logits


class PackedDistillDataset(Dataset):
    """student入力 + packed-target([1+C]) を返す蒸留用データセット。

    student側の(既に二値化済みの)データセットをそのまま包み、
    事前計算した teacher ロジットをインデックス整列で対応付ける。
    `__getitem__(i)` は `(student_x_i, packed_target_i)` を返す:

        packed_target = [1 + C] float
            列0     : ラベル(convert/distill.pack_targets と同じ、float格納)
            列1..C  : teacherロジット

    こうすることで既存の (output, target) 2引数損失契約(training/tasks.py)を
    保ったまま、runner の学習ループ本体を無改造で KD に流用できる。

    不変条件(モジュール docstring 参照): student_dataset[i] と
    teacher_logits[i] が同一サンプルを指すこと。student_dataset の変換が
    決定的(augmentationなし)であることが前提。

    Args:
        student_dataset: `(x, label)` を返す既存の(二値化済み)データセット。
        teacher_logits: [N, C] の teacher ロジット(get_or_compute_logits出力)。
        pseudo_labels: Trueなら列0のラベルを**正解でなくteacherロジットのargmax**
            (疑似ラベル)にする。完全ラベルフリー変換用: 正解ラベルが一切
            存在しない状況(LLM変換の実態)でも、α>0のhard項が疑似ラベルを錨に
            機能する。α=0純KDの後半劣化(D6実測: best 0.405→final 0.284)への対処。
            注意: metric_fn(分類acc)も疑似ラベル基準になるため、真の精度は
            _hard_eval(素のeval loader+正解ラベル)側で測ること。
    """

    def __init__(self, student_dataset: Dataset, teacher_logits: torch.Tensor,
                 *, pseudo_labels: bool = False) -> None:
        n_student = len(student_dataset)  # type: ignore[arg-type]
        if teacher_logits.ndim != 2:
            raise ValueError(
                f'teacher_logits は [N, C] の2次元が必要: {tuple(teacher_logits.shape)}'
            )
        if teacher_logits.shape[0] != n_student:
            raise ValueError(
                f'teacher_logits の行数({teacher_logits.shape[0]})と'
                f'student_dataset 長({n_student})が不一致'
                '(インデックス整列が崩れています)'
            )
        self.student_dataset = student_dataset
        self.teacher_logits = teacher_logits.float()
        self.pseudo_labels = pseudo_labels

    def __len__(self) -> int:
        return len(self.student_dataset)  # type: ignore[arg-type]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        x, label = self.student_dataset[index]
        logits = self.teacher_logits[index]  # [C]
        if self.pseudo_labels:
            label = int(logits.argmax().item())  # 正解不使用(完全ラベルフリー)
        label_col = torch.tensor([float(label)], dtype=logits.dtype)  # [1]
        packed = torch.cat([label_col, logits], dim=0)  # [1 + C]
        return x, packed
