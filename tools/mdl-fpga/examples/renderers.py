"""ユーザー定義レンダラの実例(full_adder.ir.json 用)。

モデルと同じディレクトリの renderers.py は mdl-fpga view が自動で読み込み、
ファイルを書き換えると**実行中でも自動リロード**される。
プローブのレンダラ選択に、ここの公開関数(先頭 _ 以外)が現れる。

シグネチャ: def 関数名(bits, ctx) -> dict
  bits: LSB-first の 0/1 numpy 配列(スライス適用済み)
  ctx:  width / cycle / target / slice / meta / target_meta / args
返り値(payload)の種類は tools/mdl-fpga/mdl_fpga/viewer/renderers.py 冒頭を参照
(error / text / bits / bars / image)。
"""


def carry_sum(bits, ctx):
    """4クラススコア(carry*2+sum)から桁上がりと和を復元して表示する。"""
    score_bits = ctx['meta'].get('score_bits') or 5
    k = ctx['meta'].get('k') or 4
    scores = []
    for c in range(k):
        seg = bits[c * score_bits:(c + 1) * score_bits]
        scores.append(sum(int(b) << i for i, b in enumerate(seg)))
    cls = max(range(k), key=lambda c: scores[c])
    return {'kind': 'text',
            'text': f'carry={cls // 2} sum={cls % 2}(class {cls})'}


def adder_inputs(bits, ctx):
    """入力 x(a, b, cin)を式の形で表示する。"""
    a, b, cin = (int(bits[i]) if i < len(bits) else 0 for i in range(3))
    return {'kind': 'text', 'text': f'{a} + {b} + cin{cin} = {a + b + cin}'}
