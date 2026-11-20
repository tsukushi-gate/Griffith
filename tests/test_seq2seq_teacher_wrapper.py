"""HFMarianMTTeacher の translate() 契約をオフラインで固定(HF DL不要)。

fake な Marian互換モデル/tokenizer を注入し、translate(texts) が
  - バッチ分割して回す、
  - tokenizer.batch_decode の出力(訳文文字列リスト)をそのまま返す、
ことを確認する。実モデルの翻訳品質は tests 外(GPU機E前の実地確認)で担保する。
"""

from griffith_logic.convert.text_teacher import HFMarianMTTeacher


class _FakeTok:
    def __call__(self, batch, **kw):
        class _Enc(dict):
            def to(self, _dev):
                return self
        return _Enc(input_ids=list(range(len(batch))), _batch=batch)

    def batch_decode(self, gen, **kw):
        # gen は _FakeModel.generate が返した「元テキスト」リスト。恒等訳を返す。
        return [f'EN:{t}' for t in gen]


class _FakeModel:
    def to(self, _dev):
        return self

    def eval(self):
        return self

    def generate(self, **enc):
        return enc['_batch']  # batch_decode がここから訳文を作る


def _make_teacher(**kw) -> HFMarianMTTeacher:
    teacher = HFMarianMTTeacher.__new__(HFMarianMTTeacher)
    import torch
    teacher.device = torch.device('cpu')
    teacher.tokenizer = _FakeTok()
    teacher.model = _FakeModel()
    teacher.num_beams = kw.get('num_beams', 4)
    teacher.max_new_tokens = kw.get('max_new_tokens', 128)
    return teacher


def test_translate_returns_one_string_per_input():
    teacher = _make_teacher()
    out = teacher.translate(['こんにちは', '私は学生です', '犬が好き'], batch_size=2)
    assert out == ['EN:こんにちは', 'EN:私は学生です', 'EN:犬が好き']


def test_translate_batches_cover_all_inputs():
    teacher = _make_teacher()
    texts = [f's{i}' for i in range(7)]
    out = teacher.translate(texts, batch_size=3)
    assert len(out) == 7
    assert out[-1] == 'EN:s6'
