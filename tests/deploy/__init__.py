"""tests/deploy をパッケージにする(Track K)。

これが無いと pytest は `tests/deploy` を sys.path へ直接差し込み、
このディレクトリの `conftest.py` がトップレベルモジュール `conftest` として
`tests/conftest.py` を覆い隠す。他の層のテストは `from conftest import
TOL_F64_EQ` のようにルートのconftestをモジュール名でimportしているため、
覆い隠すと収集時にImportErrorで全滅する。
"""
