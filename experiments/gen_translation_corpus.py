"""翻訳DLGN(ROADMAP-v5)向け対訳バイトコーパス生成スクリプト。

ja→en 翻訳のトイ対訳コーパスを2アーム分作る:

    data-freeアーム: ja文を入力に、teacher(opus-mt-ja-en)が en 疑似訳を生成し、
        (ja, en_pseudo) 対を保存する。実データの en 訳文は一切参照しない
        (teacherの出力のみが学生の教師信号)。
    goldアーム: HFデータセットの (ja, en_gold) 対をそのまま保存する。

held-out evalセット(gold ja-en)は両アーム共通で、学習データと disjoint に切り出す。

対訳データ取得は `Helsinki-NLP/tatoeba`(lang1='en', lang2='ja')をまず試し、
ロードできなければ `Helsinki-NLP/opus-100`('en-ja' config)にフォールバックする
(実測: tatoebaは`datasets`最新版でdataset scriptが禁止されロード不可、
opus-100は動作する。使用したdataset名はmetaに記録する)。

保存フォーマット(BytePairDatasetが読む契約、変更禁止):
    {'src': list[list[int]], 'tgt': list[list[int]], 'meta': dict}
    src = ja生バイト列のリスト(encode_bytes出力)、tgt = en生バイト列のリスト。
    特殊トークンは含まない(生バイトのみ)。

使用例:
    # スモーク(少量、CPU)
    uv run python experiments/gen_translation_corpus.py \\
        --n-train 40 --n-eval 10 --arm both --device cpu

    # 本実行
    uv run python experiments/gen_translation_corpus.py \\
        --n-train 4000 --n-eval 500 --arm both --device cpu
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from datetime import datetime, timedelta, timezone

JST = timezone(timedelta(hours=9))

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from griffith_logic.convert.byte_vocab import encode_bytes  # noqa: E402
from griffith_logic.convert.text_teacher import resolve_text_teacher  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description='翻訳DLGN向け対訳バイトコーパス生成'
    )
    parser.add_argument('--n-train', type=int, default=4000, help='学習側の文ペア数')
    parser.add_argument('--n-eval', type=int, default=500, help='held-out eval の文ペア数')
    parser.add_argument(
        '--max-ja-chars', type=int, default=24, help='ja側の短文フィルタ(文字数上限)'
    )
    parser.add_argument(
        '--max-en-chars', type=int, default=96, help='en側の短文フィルタ(文字数上限)'
    )
    parser.add_argument(
        '--min-ja-chars', type=int, default=3, help='ja側の下限(極短フラグメント除去)'
    )
    parser.add_argument(
        '--min-en-chars', type=int, default=3, help='en側の下限(極短フラグメント除去)'
    )
    parser.add_argument(
        '--teacher', type=str, default='hf-mt:Helsinki-NLP/opus-mt-ja-en',
        help='resolve_text_teacher に渡すスキーム文字列(data-freeアームの疑似訳生成用)',
    )
    parser.add_argument(
        '--arm', type=str, default='both', choices=['datafree', 'gold', 'both'],
        help='生成するアーム',
    )
    parser.add_argument('--out-dir', type=str, default='data-translation',
                        help='出力ディレクトリ(相対パス、Modal無改造規約)')
    parser.add_argument('--device', type=str, default='cpu', help="teacher推論デバイス")
    parser.add_argument('--seed', type=int, default=42, help='乱数シード')
    parser.add_argument('--batch-size', type=int, default=32, help='teacher.translateのバッチサイズ')
    return parser.parse_args()


# ---------------------------------------------------------------------------
# 対訳データ取得
# ---------------------------------------------------------------------------

def load_parallel_pairs() -> tuple[list[tuple[str, str]], str]:
    """ja-en対訳ペアの (ja, en) リストと、実際に使ったdataset名を返す。

    まず Helsinki-NLP/tatoeba(lang1='en', lang2='ja')を試し、ロードできなければ
    Helsinki-NLP/opus-100('en-ja')にフォールアックする。全splitを結合して返す
    (train/eval分割はこのスクリプト側でseed固定シャッフル+disjoint切り出しを行う)。
    """
    from datasets import load_dataset

    try:
        ds = load_dataset('Helsinki-NLP/tatoeba', lang1='en', lang2='ja')
        dataset_name = 'Helsinki-NLP/tatoeba(en-ja)'
        pairs: list[tuple[str, str]] = []
        for split in ds:
            for row in ds[split]:
                t = row.get('translation', row)
                ja = t.get('ja')
                en = t.get('en')
                if ja and en:
                    pairs.append((ja, en))
        print(f'[data] {dataset_name} をロード: {len(pairs)}件')
        return pairs, dataset_name
    except Exception as e:
        print(f'[data] tatoeba ロード失敗({e!r})、opus-100 にフォールバック')

    ds = load_dataset('Helsinki-NLP/opus-100', 'en-ja')
    dataset_name = 'Helsinki-NLP/opus-100(en-ja)'
    pairs = []
    for split in ds:
        for row in ds[split]:
            t = row['translation']
            ja = t.get('ja')
            en = t.get('en')
            if ja and en:
                pairs.append((ja, en))
    print(f'[data] {dataset_name} をロード: {len(pairs)}件')
    return pairs, dataset_name


def filter_and_dedup(
    pairs: list[tuple[str, str]],
    max_ja_chars: int,
    max_en_chars: int,
    min_ja_chars: int = 3,
    min_en_chars: int = 3,
) -> list[tuple[str, str]]:
    """短文フィルタ + 空/極短除去 + **ja-source単位の重複除去**。

    opus-100 は同一 ja 源が別々の(しばしばノイズな)en 訳と重複することがある。
    ja をキーに最初の出現だけ残すことで、(a) train/eval を ja 源で disjoint にでき、
    (b) 1つの ja に複数のノイズ訳がぶら下がる汚染を減らせる。
    """
    seen_ja: set[str] = set()
    out: list[tuple[str, str]] = []
    for ja, en in pairs:
        ja = ja.strip()
        en = en.strip()
        if not ja or not en:
            continue
        if not (min_ja_chars <= len(ja) <= max_ja_chars):
            continue
        if not (min_en_chars <= len(en) <= max_en_chars):
            continue
        if ja in seen_ja:
            continue
        seen_ja.add(ja)
        out.append((ja, en))
    return out


def main():
    args = parse_args()
    rng = random.Random(args.seed)

    os.makedirs(args.out_dir, exist_ok=True)

    print('=== 対訳データロード ===')
    raw_pairs, dataset_name = load_parallel_pairs()

    print('=== 短文フィルタ + ja源dedup ===')
    filtered = filter_and_dedup(
        raw_pairs, args.max_ja_chars, args.max_en_chars,
        args.min_ja_chars, args.min_en_chars,
    )
    print(
        f'フィルタ後: {len(filtered)}件 '
        f'(ja {args.min_ja_chars}-{args.max_ja_chars}文字, '
        f'en {args.min_en_chars}-{args.max_en_chars}文字, ja源unique)'
    )

    n_needed = args.n_train + args.n_eval
    if len(filtered) < n_needed:
        raise ValueError(
            f'フィルタ後のペア数({len(filtered)})が要求件数(train={args.n_train}+'
            f'eval={args.n_eval}={n_needed})に足りません。max-ja-chars/max-en-charsを'
            '緩めてください。'
        )

    # seed固定シャッフル -> train/evalをdisjointに分割。
    indices = list(range(len(filtered)))
    rng.shuffle(indices)
    train_indices = indices[:args.n_train]
    eval_indices = indices[args.n_train:args.n_train + args.n_eval]
    assert set(train_indices).isdisjoint(eval_indices)

    train_pairs = [filtered[i] for i in train_indices]
    eval_pairs = [filtered[i] for i in eval_indices]

    train_ja = [ja for ja, _ in train_pairs]
    train_en_gold = [en for _, en in train_pairs]
    eval_ja = [ja for ja, _ in eval_pairs]
    eval_en_gold = [en for _, en in eval_pairs]

    created_jst = datetime.now(JST).isoformat()
    base_meta = {
        'dataset': dataset_name,
        'max_ja_chars': args.max_ja_chars,
        'max_en_chars': args.max_en_chars,
        'seed': args.seed,
        'created_jst': created_jst,
    }

    # --- eval(gold, 両アーム共通) --------------------------------------
    eval_path = os.path.join(args.out_dir, 'jaen_eval.pt')
    eval_data = {
        'src': [encode_bytes(ja) for ja in eval_ja],
        'tgt': [encode_bytes(en) for en in eval_en_gold],
        'meta': dict(base_meta, arm='eval', n=len(eval_pairs)),
    }
    torch.save(eval_data, eval_path)
    print(f'保存: {eval_path} ({len(eval_pairs)}件)')

    # --- goldアーム -------------------------------------------------------
    gold_path = os.path.join(args.out_dir, 'jaen_gold_train.pt')
    if args.arm in ('gold', 'both'):
        gold_data = {
            'src': [encode_bytes(ja) for ja in train_ja],
            'tgt': [encode_bytes(en) for en in train_en_gold],
            'meta': dict(base_meta, arm='gold', n=len(train_pairs)),
        }
        torch.save(gold_data, gold_path)
        print(f'保存: {gold_path} ({len(train_pairs)}件)')

    # --- data-freeアーム ----------------------------------------------
    train_en_pseudo: list[str] | None = None
    datafree_path = os.path.join(args.out_dir, 'jaen_datafree_train.pt')
    if args.arm in ('datafree', 'both'):
        print(f'=== teacherロード: {args.teacher} ===')
        teacher = resolve_text_teacher(args.teacher, device=args.device)
        print('=== teacherで疑似訳生成中(data-freeアーム) ===')
        train_en_pseudo = teacher.translate(train_ja, batch_size=args.batch_size)
        assert len(train_en_pseudo) == len(train_ja)
        datafree_data = {
            'src': [encode_bytes(ja) for ja in train_ja],
            'tgt': [encode_bytes(en) for en in train_en_pseudo],
            'meta': dict(
                base_meta, arm='datafree', n=len(train_pairs), teacher=args.teacher,
            ),
        }
        torch.save(datafree_data, datafree_path)
        print(f'保存: {datafree_path} ({len(train_pairs)}件)')

    # --- サニティ確認: サンプル5件を (ja, teacher_en, gold_en) で並べて表示 ---
    print('\n=== サンプル(先頭5件): ja / teacher_en / gold_en ===')
    n_show = min(5, len(train_pairs))
    for i in range(n_show):
        ja = train_ja[i]
        gold_en = train_en_gold[i]
        teacher_en = train_en_pseudo[i] if train_en_pseudo is not None else '(data-free未生成)'
        print(f'[{i}] ja      : {ja}')
        print(f'    teacher : {teacher_en}')
        print(f'    gold    : {gold_en}')
        print()


if __name__ == '__main__':
    main()
