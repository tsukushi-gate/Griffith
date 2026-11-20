""".mdlb vs bundle.json のサイズ比較(raw/gzip)。時間計測は一切しない。

実モデル(tools/webbench/public/models/bundles/*.bundle.json)を対象に、
既存のbundle.json(gzip後)を基準として.mdlbのraw/gzip後サイズを測り、
`results/deploy/mdlb_size_k58.json` へ書き出す。

Usage:
    uv run python scripts/bench_deploy_mdlb_size.py
"""

import gzip
import json
from datetime import datetime
from pathlib import Path

from griffith_logic.deploy.mdlb import encode_bundle_to_mdlb

ROOT = Path(__file__).resolve().parents[1]
BUNDLES_DIR = ROOT / 'tools' / 'webbench' / 'public' / 'models' / 'bundles'
OUT_PATH = ROOT / 'results' / 'deploy' / 'mdlb_size_k58.json'


def main() -> None:
    rows = []
    for path in sorted(BUNDLES_DIR.glob('*.bundle.json')):
        raw_json = path.read_bytes()
        bundle = json.loads(raw_json)
        canonical = json.loads(json.dumps(bundle, ensure_ascii=False))
        mdlb_bytes = encode_bundle_to_mdlb(canonical)

        json_bytes = len(raw_json)
        json_gz = len(gzip.compress(raw_json, 6))
        mdlb_raw = len(mdlb_bytes)
        mdlb_gz = len(gzip.compress(mdlb_bytes, 6))

        rows.append({
            'model': path.stem.removesuffix('.bundle'),
            'json_bytes': json_bytes,
            'json_gz_bytes': json_gz,
            'mdlb_bytes': mdlb_raw,
            'mdlb_gz_bytes': mdlb_gz,
            'raw_reduction_ratio': round(json_bytes / mdlb_raw, 3),
            'gz_reduction_ratio': round(json_gz / mdlb_gz, 3),
        })
        print(f'{path.stem}: json={json_bytes:,}B json.gz={json_gz:,}B '
              f'mdlb={mdlb_raw:,}B mdlb.gz={mdlb_gz:,}B '
              f'(raw {json_bytes / mdlb_raw:.2f}x, gz {json_gz / mdlb_gz:.2f}x)')

    total_json_gz = sum(r['json_gz_bytes'] for r in rows)
    total_mdlb_gz = sum(r['mdlb_gz_bytes'] for r in rows)
    summary = {
        'total_json_gz_bytes': total_json_gz,
        'total_mdlb_gz_bytes': total_mdlb_gz,
        'total_gz_reduction_ratio': round(total_json_gz / total_mdlb_gz, 3),
        'target_gz_reduction_ratio': 2.0,
    }
    print(f'\n合計: json.gz={total_json_gz:,}B mdlb.gz={total_mdlb_gz:,}B '
          f'({total_json_gz / total_mdlb_gz:.2f}x)')

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps({
        'rows': rows,
        'summary': summary,
        'when': datetime.now().astimezone().isoformat(timespec='seconds'),
        'note': ('性能(時間)ベンチではない。サイズ計測のみ。'
                'gzip level=6、griffith deploy bundle --gzip / save_bundle と同一。'),
    }, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(f'\n書き出し: {OUT_PATH}')


if __name__ == '__main__':
    main()
