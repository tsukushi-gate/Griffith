"""
.env ファイルローダー（標準ライブラリのみ）。

プロジェクトルートの .env を読み、os.environ に未設定のキーのみセットする。
"""

import os


def load_dotenv(env_path: str | None = None) -> None:
    """`.env` ファイルを読み込んで環境変数にセットする。

    既に設定済みの環境変数は上書きしない。
    ファイルが存在しない場合は何もしない。
    """
    if env_path is None:
        # experiments/ の親ディレクトリ（プロジェクトルート）を探す
        env_path = os.path.join(os.path.dirname(__file__), '..', '.env')

    env_path = os.path.abspath(env_path)
    if not os.path.isfile(env_path):
        return

    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if '=' not in line:
                continue
            key, _, value = line.partition('=')
            key = key.strip()
            value = value.strip()
            # 既存の環境変数は上書きしない
            if key and key not in os.environ:
                os.environ[key] = value
