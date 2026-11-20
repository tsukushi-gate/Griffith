#!/usr/bin/env python
"""ONNX Runtime CPU上での同規模GRU seq2seqベースライン計測。

RDDLGNデプロイキット(C99ランタイム)との公平比較のため、パラメータ数を
合わせた従来型GRU seq2seq(Embedding→GRU encoder→GRU decoder(自己回帰)→
Linear head)をONNX Runtime CPUで測る。

公平性の教訓(ONNX 公平比較レポート、必読):
  - 配信サイズは gzip後(transferSize相当)で揃える。非圧縮サイズだけを
    見ると2.7〜3倍以上過大になり得る。
  - fp32同士で比較する(int8は速度/起動を大きく損なうため混ぜない)。
  - 精度は比較しない(GRUとRDDLGNはアーキ族が違うので無意味。ここでは
    測定すらしない)。
  - ONNX Runtime本体(dylib/pybind/wheel)のサイズは主比較(配信サイズ)に
    含めない。付録として記録するのみ。

このスクリプトはこのworktreeのpyproject.tomlにonnxruntime/onnxを追加
しない(汚さない)。実行は必ず `--with` 経由:

    uv run --with onnxruntime --with onnx python scripts/bench_onnx_rnn.py
    uv run --with onnxruntime --with onnx python scripts/bench_onnx_rnn.py --scales copy_tiny
    uv run --with onnxruntime --with onnx python scripts/bench_onnx_rnn.py --reps 30

torch(埋め込み/GRU/エクスポート用)は通常の環境(uv sync --extra dev --extra cpu)
に既にあるので `--with torch` は不要。

メモリ計測(ピークRSS)は本プロセスの中で測ると torch import 分が混入して
不当に大きくなるため、`--mem-probe <scale>` モードで onnxruntime のみを
importする別プロセスを `/usr/bin/time -l` 経由で起動し、そこだけを測る
(このモードは本スクリプトが自分自身をsubprocessで再帰呼び出しする):

    /usr/bin/time -l uv run --with onnxruntime python scripts/bench_onnx_rnn.py \\
        --mem-probe copy_tiny --build-dir results/deploy/build_bench_onnx

結果は results/deploy/onnx_baseline.json に書き出す。
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import platform
import statistics
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT_DIR = REPO / 'results' / 'deploy'
DEFAULT_BUILD_DIR = OUT_DIR / 'build_bench_onnx'

# ---------------------------------------------------------------------------
# スケール定義: RDDLGN側の対応モデル(デプロイキット設計メモ, results/
# deploy/rddlgn_bench.json)にパラメータ数を±10%以内で合わせる。
#
# アーキテクチャは全スケール共通: Embedding(共有, encoder/decoderで同じ
# 重み) -> GRU encoder(1層) -> GRU decoder(1層, 自己回帰) -> Linear head。
# emb/hiddenだけをスケールごとに変えてRDDLGN側のパラメータ数に合わせた
# (探索は本ファイル末尾のコメント参照。emb=RDDLGNのd_embとは独立に選定)。
# ---------------------------------------------------------------------------
SCALES = {
    'copy_tiny': dict(
        vocab=50, seq_len=4, emb=192, hidden=96, num_layers=1,
        rddlgn_desc='copy_tiny (vocab=50, seq_len=4)',
        rddlgn_params=180_608,
    ),
    'charlm': dict(
        vocab=67, seq_len=32, emb=128, hidden=384, num_layers=1,
        rddlgn_desc='char-LM (vocab=67, seq_len=32)',
        rddlgn_params=1_223_993,
    ),
    'paper_gru_17_9m': dict(
        vocab=16000, seq_len=16, emb=512, hidden=448, num_layers=1,
        rddlgn_desc=(
            'paper相当 (vocab=16000, seq_len=16) の参考点。RDDLGN学習時'
            'soft paramsは40.8Mだが、ここではRDDLGNのcollapsed後(離散化'
            '済みLUT表現)params 17.91Mに合わせた。埋め込み込み'
        ),
        rddlgn_params=17_910_000,
    ),
}

BATCH_SIZES_AGG = [1, 2, 4, 8]


# ---------------------------------------------------------------------------
# モデル定義(torch) — mem-probeモードではこのブロックに触れない(torch未import)
# ---------------------------------------------------------------------------
def _build_torch_model(vocab: int, emb: int, hidden: int, num_layers: int):
    import torch.nn as nn

    class Seq2SeqGRU(nn.Module):
        """Embedding共有(shared_vocab, RDDLGNのshared_vocab=Trueに倣う)の
        GRU seq2seq。encode()とdecode_step()を分けてexportする。
        """

        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(vocab, emb)
            self.encoder = nn.GRU(emb, hidden, num_layers=num_layers, batch_first=True)
            self.decoder = nn.GRU(emb, hidden, num_layers=num_layers, batch_first=True)
            self.head = nn.Linear(hidden, vocab)

        def encode(self, src_ids):
            x = self.embedding(src_ids)
            _, h = self.encoder(x)
            return h

        def decode_step(self, prev_token, hidden):
            x = self.embedding(prev_token)  # (B,1,emb)
            out, h_new = self.decoder(x, hidden)
            logits = self.head(out[:, -1, :])
            return logits, h_new

    class EncoderWrapper(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, src_ids):
            return self.model.encode(src_ids)

    class DecoderStepWrapper(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, prev_token, hidden):
            return self.model.decode_step(prev_token, hidden)

    model = Seq2SeqGRU().eval()
    return model, EncoderWrapper(model).eval(), DecoderStepWrapper(model).eval()


def _param_count(model) -> int:
    return sum(p.numel() for p in model.parameters())


def export_scale(name: str, cfg: dict, build_dir: Path) -> dict:
    """torchモデルを構築し、パラメータ数を検証しつつencoder/decoder-stepの
    2グラフをONNXへexportする。opset既定(=legacy TorchScript exporterの
    既定opset)。torch 2.9+で既定がdynamo exporterに変わったため、GRUの
    後方互換のため明示的にdynamo=Falseの legacy exporterを使う(要onnx)。
    """
    import torch

    vocab, seq_len = cfg['vocab'], cfg['seq_len']
    emb, hidden, num_layers = cfg['emb'], cfg['hidden'], cfg['num_layers']

    torch.manual_seed(0)
    model, enc_wrap, dec_wrap = _build_torch_model(vocab, emb, hidden, num_layers)

    total_params = _param_count(model)
    rddlgn_params = cfg['rddlgn_params']
    rel_diff = (total_params - rddlgn_params) / rddlgn_params

    scale_dir = build_dir / name
    scale_dir.mkdir(parents=True, exist_ok=True)
    enc_path = scale_dir / 'encoder.onnx'
    dec_path = scale_dir / 'decoder_step.onnx'

    dummy_src = torch.randint(0, vocab, (1, seq_len), dtype=torch.long)
    dummy_prev = torch.randint(0, vocab, (1, 1), dtype=torch.long)
    dummy_hidden = torch.zeros(num_layers, 1, hidden)

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        torch.onnx.export(
            enc_wrap, (dummy_src,), str(enc_path),
            input_names=['src_ids'], output_names=['hidden'],
            dynamic_axes={'src_ids': {0: 'batch', 1: 'seq'}, 'hidden': {1: 'batch'}},
            opset_version=17, dynamo=False,
        )
        torch.onnx.export(
            dec_wrap, (dummy_prev, dummy_hidden), str(dec_path),
            input_names=['prev_token', 'hidden_in'],
            output_names=['logits', 'hidden_out'],
            dynamic_axes={
                'prev_token': {0: 'batch'}, 'hidden_in': {1: 'batch'},
                'logits': {0: 'batch'}, 'hidden_out': {1: 'batch'},
            },
            opset_version=17, dynamo=False,
        )

    enc_bytes = enc_path.stat().st_size
    dec_bytes = dec_path.stat().st_size
    enc_gz = len(gzip.compress(enc_path.read_bytes(), compresslevel=9))
    dec_gz = len(gzip.compress(dec_path.read_bytes(), compresslevel=9))

    embed_bytes_fp32 = vocab * emb * 4  # このモデルの埋め込みはencoder/decoder
    # 両方のonnxグラフに独立したinitializerとして複製される(torchのtied
    # embeddingを別々のONNXグラフへexportした際の既知の実務上の制約。
    # HuggingFace optimumのencoder/decoder分割exportでも同様に複製される)。
    # 「重複込みの実測サイズ」と「重複を1つ分だけ除いた参考値」の両方を残す。

    eager_bench = _bench_eager_torch(model, vocab, seq_len)

    return dict(
        name=name,
        rddlgn_desc=cfg['rddlgn_desc'],
        vocab_size=vocab,
        seq_len=seq_len,
        emb_dim=emb,
        hidden_dim=hidden,
        num_layers=num_layers,
        params=dict(
            total=total_params,
            rddlgn_params=rddlgn_params,
            rel_diff_pct=100.0 * rel_diff,
            within_10pct=abs(rel_diff) <= 0.10,
        ),
        sizes=dict(
            encoder_onnx_bytes=enc_bytes,
            decoder_onnx_bytes=dec_bytes,
            combined_raw_bytes=enc_bytes + dec_bytes,
            encoder_onnx_gz_bytes=enc_gz,
            decoder_onnx_gz_bytes=dec_gz,
            combined_gz_bytes=enc_gz + dec_gz,
            embed_table_fp32_bytes=embed_bytes_fp32,
            combined_gz_bytes_dedup_one_embed_copy=max(
                0, enc_gz + dec_gz - embed_bytes_fp32),
            note=(
                '共有embeddingをencoder.onnxとdecoder_step.onnxの両方へ'
                '独立export(実務上の標準的な分割exportの制約)。'
                'combined_*_bytesは重複込みの実測値、'
                'combined_gz_bytes_dedup_one_embed_copyは埋め込み1個分を'
                '差し引いた参考値(fp32生バイト換算、粗い近似)。'
            ),
        ),
        eager_torch=eager_bench,
        paths=dict(encoder=str(enc_path), decoder=str(dec_path)),
    )


def _bench_eager_torch(model, vocab, seq_len, reps=20, warmup=5):
    """PyTorch eagerでの1点測定(ORTの妥当性確認用)。
    torch.no_grad()、fp32、モデルのデフォルトスレッド設定のまま。
    """
    import torch

    torch_threads = torch.get_num_threads()
    src = torch.randint(0, vocab, (1, seq_len), dtype=torch.long)
    prev = torch.randint(0, vocab, (1, 1), dtype=torch.long)

    with torch.no_grad():
        for _ in range(warmup):
            h = model.encode(src)
            model.decode_step(prev, h)

        encode_times = []
        for _ in range(reps):
            t0 = time.perf_counter()
            h = model.encode(src)
            encode_times.append((time.perf_counter() - t0) * 1000.0)

        decode_times = []
        for _ in range(reps):
            t0 = time.perf_counter()
            model.decode_step(prev, h)
            decode_times.append((time.perf_counter() - t0) * 1000.0)

    return dict(
        torch_threads=torch_threads,
        encode_ms=_stats(encode_times),
        decode_step_ms=_stats(decode_times),
        reps=reps,
    )


# ---------------------------------------------------------------------------
# 統計ヘルパ
# ---------------------------------------------------------------------------
def _stats(values: list[float]) -> dict:
    return dict(
        median=statistics.median(values),
        mean=statistics.mean(values),
        stdev=statistics.stdev(values) if len(values) > 1 else 0.0,
        min=min(values),
        max=max(values),
        n=len(values),
    )


# ---------------------------------------------------------------------------
# ORT計測(mem-probeモードでも呼べるよう torch非依存)
# ---------------------------------------------------------------------------
def _make_session(path: str, threads: str):
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    if threads == 'threads1':
        so.intra_op_num_threads = 1
        so.inter_op_num_threads = 1
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    # threads == 'default' の場合は何も設定しない(ORT既定に任せる)
    return ort.InferenceSession(path, sess_options=so, providers=['CPUExecutionProvider'])


def bench_ort_scale(name: str, cfg: dict, build_dir: Path, threads: str,
                     reps: int, warmup: int) -> dict:
    import numpy as np

    vocab, seq_len = cfg['vocab'], cfg['seq_len']
    scale_dir = build_dir / name
    enc_sess = _make_session(str(scale_dir / 'encoder.onnx'), threads)
    dec_sess = _make_session(str(scale_dir / 'decoder_step.onnx'), threads)

    rng = np.random.default_rng(0)
    src1 = rng.integers(0, vocab, size=(1, seq_len)).astype(np.int64)

    def run_encode(batch_src):
        return enc_sess.run(None, {'src_ids': batch_src})[0]

    def run_decode(prev_tok, hidden_state):
        return dec_sess.run(None, {'prev_token': prev_tok, 'hidden_in': hidden_state})

    # --- warmup ---
    for _ in range(warmup):
        h = run_encode(src1)
        prev = np.zeros((1, 1), dtype=np.int64)
        run_decode(prev, h)

    # --- encode_ms ---
    encode_times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        h1 = run_encode(src1)
        encode_times.append((time.perf_counter() - t0) * 1000.0)

    # --- decode_step_ms (単一セッション、batch=1、固定hidden再利用) ---
    prev1 = np.zeros((1, 1), dtype=np.int64)
    decode_times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        run_decode(prev1, h1)
        decode_times.append((time.perf_counter() - t0) * 1000.0)

    # --- seq_e2e_ms (greedy, フル長, batch=1, 早期終了なし) ---
    def greedy_once():
        h = run_encode(src1)
        prev = np.zeros((1, 1), dtype=np.int64)
        for _ in range(seq_len):
            logits, h = run_decode(prev, h)
            prev = logits.argmax(axis=-1).reshape(1, 1).astype(np.int64)
        return h

    for _ in range(warmup):
        greedy_once()
    e2e_times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        greedy_once()
        e2e_times.append((time.perf_counter() - t0) * 1000.0)

    decode_median_ms = statistics.median(decode_times)
    tokens_per_s = 1000.0 / decode_median_ms if decode_median_ms > 0 else float('inf')

    # --- バッチ並列スループット(N系列をlockstepでgreedy生成、早期終了なし) ---
    tokens_per_s_agg = {}
    agg_wall_ms = {}
    for n in BATCH_SIZES_AGG:
        srcN = rng.integers(0, vocab, size=(n, seq_len)).astype(np.int64)

        def batched_greedy(n=n, srcN=srcN):
            h = run_encode(srcN)
            prev = np.zeros((n, 1), dtype=np.int64)
            for _ in range(seq_len):
                logits, h = run_decode(prev, h)
                prev = logits.argmax(axis=-1).reshape(n, 1).astype(np.int64)
            return h

        for _ in range(max(3, warmup // 2)):
            batched_greedy()
        n_reps = max(5, reps // 2)
        wall_times = []
        for _ in range(n_reps):
            t0 = time.perf_counter()
            batched_greedy()
            wall_times.append(time.perf_counter() - t0)
        wall_median_s = statistics.median(wall_times)
        agg_wall_ms[str(n)] = wall_median_s * 1000.0
        tokens_per_s_agg[str(n)] = (n * seq_len) / wall_median_s

    return dict(
        threads=threads,
        reps=reps,
        encode_ms=_stats(encode_times),
        decode_step_ms=_stats(decode_times),
        seq_e2e_ms=_stats(e2e_times),
        gen_tokens=seq_len,
        tokens_per_s=tokens_per_s,
        tokens_per_s_agg=tokens_per_s_agg,
        agg_wall_ms_median=agg_wall_ms,
        batch_sizes=BATCH_SIZES_AGG,
    )


# ---------------------------------------------------------------------------
# メモリ計測(mem-probeサブプロセス。torchをimportしない)
# ---------------------------------------------------------------------------
def run_mem_probe(scale: str, build_dir: Path, threads: str) -> None:
    """--mem-probe モード本体。onnxruntime+numpyのみimportし、1回encode+
    1回フルgreedy生成を行って終了する。呼び出し元が/usr/bin/time -lで
    包んでピークRSSを取る。標準出力はJSON1行のみ(検証用)。
    """
    import numpy as np

    cfg = SCALES[scale]
    vocab, seq_len = cfg['vocab'], cfg['seq_len']
    scale_dir = build_dir / scale
    enc_sess = _make_session(str(scale_dir / 'encoder.onnx'), threads)
    dec_sess = _make_session(str(scale_dir / 'decoder_step.onnx'), threads)

    rng = np.random.default_rng(0)
    src = rng.integers(0, vocab, size=(1, seq_len)).astype(np.int64)
    h = enc_sess.run(None, {'src_ids': src})[0]
    prev = np.zeros((1, 1), dtype=np.int64)
    for _ in range(seq_len):
        logits, h = dec_sess.run(None, {'prev_token': prev, 'hidden_in': h})
        prev = logits.argmax(axis=-1).reshape(1, 1).astype(np.int64)
    print(json.dumps({'scale': scale, 'ok': True, 'final_hidden_shape': list(h.shape)}))


def measure_peak_rss(scale: str, build_dir: Path, threads: str) -> dict:
    """このスクリプト自身を --mem-probe で再帰起動し、/usr/bin/time -lで
    ピークRSSを取る。呼び出しは常に `uv run --with onnxruntime`(torch不要)
    で行い、torch importによるRSS膨張を避ける。
    """
    cmd = [
        '/usr/bin/time', '-l',
        'uv', 'run', '--with', 'onnxruntime', 'python', str(Path(__file__).resolve()),
        '--mem-probe', scale, '--build-dir', str(build_dir), '--threads', threads,
    ]
    proc = subprocess.run(cmd, cwd=str(REPO), capture_output=True, text=True)
    stderr = proc.stderr
    max_rss_bytes = None
    peak_footprint_bytes = None
    for line in stderr.splitlines():
        line = line.strip()
        if line.endswith('maximum resident set size'):
            max_rss_bytes = int(line.split()[0])
        elif line.endswith('peak memory footprint'):
            peak_footprint_bytes = int(line.split()[0])
    return dict(
        ok=proc.returncode == 0,
        max_rss_kb=(max_rss_bytes / 1024.0) if max_rss_bytes is not None else None,
        peak_memory_footprint_kb=(
            peak_footprint_bytes / 1024.0 if peak_footprint_bytes is not None else None),
        stdout_tail=proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else '',
        raw_stderr_tail='\n'.join(stderr.splitlines()[-16:]),
    )


# ---------------------------------------------------------------------------
# ONNX Runtime本体の参考サイズ(付録。主比較には含めない)
# ---------------------------------------------------------------------------
def measure_ort_runtime_reference_sizes() -> dict:
    import onnxruntime as ort

    capi_dir = Path(ort.__file__).resolve().parent / 'capi'
    lib_files = {}
    for f in sorted(capi_dir.glob('*')):
        if f.suffix in ('.dylib', '.so'):
            raw = f.stat().st_size
            gz = len(gzip.compress(f.read_bytes(), compresslevel=9))
            lib_files[f.name] = dict(raw_bytes=raw, gz_bytes=gz)

    wheel_info = None
    try:
        url = f'https://pypi.org/pypi/onnxruntime/{ort.__version__}/json'
        with urllib.request.urlopen(url, timeout=8) as r:
            d = json.load(r)
        py_tag = f'cp{sys.version_info.major}{sys.version_info.minor}'
        mach = platform.machine()
        for u in d.get('urls', []):
            fn = u['filename']
            if py_tag in fn and 'macosx' in fn and mach in fn:
                wheel_info = dict(filename=fn, size_bytes=u['size'])
                break
    except Exception as e:  # noqa: BLE001 - ネットワーク不可時はNoneのまま
        wheel_info = dict(error=str(e))

    return dict(
        ort_version=ort.__version__,
        note=(
            'ONNX Runtime本体(CPU実行に必要な.dylib/.so一式、または'
            'PyPI wheelダウンロードサイズ)。主比較(モデル配信サイズ)には'
            '含めない付録値。RDDLGN側のCランタイムはmodel.c 1個(数十KB'
            'オーダー)にリンクされるため、この規模のランタイム分離コストは'
            'RDDLGN側には存在しない、という非対称性の参考情報。'
        ),
        capi_lib_files=lib_files,
        capi_lib_raw_total_bytes=sum(v['raw_bytes'] for v in lib_files.values()),
        capi_lib_gz_total_bytes=sum(v['gz_bytes'] for v in lib_files.values()),
        pypi_wheel_cp311_macos_arm64=wheel_info,
    )


def _get_uptime() -> str:
    try:
        out = subprocess.run(['uptime'], capture_output=True, text=True, timeout=5)
        return out.stdout.strip()
    except Exception as e:  # noqa: BLE001
        return f'<uptime失敗: {e}>'


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--scales', default=','.join(SCALES.keys()),
                     help='カンマ区切りのスケール名(既定: 全部)')
    ap.add_argument('--reps', type=int, default=25, help='計測反復回数(既定25、20+推奨)')
    ap.add_argument('--warmup', type=int, default=5)
    ap.add_argument('--build-dir', default=str(DEFAULT_BUILD_DIR))
    ap.add_argument('--out', default=str(OUT_DIR / 'onnx_baseline.json'))
    ap.add_argument('--skip-mem', action='store_true',
                     help='ピークRSS計測(サブプロセス起動)を飛ばす')
    ap.add_argument('--mem-probe', default=None, metavar='SCALE',
                     help=argparse.SUPPRESS)  # 内部用: 自分自身を再帰呼び出しするモード
    ap.add_argument('--threads', default='default', choices=['default', 'threads1'],
                     help='--mem-probe専用: セッションのスレッド設定')
    args = ap.parse_args()

    build_dir = Path(args.build_dir)

    if args.mem_probe is not None:
        run_mem_probe(args.mem_probe, build_dir, args.threads)
        return

    scale_names = args.scales.split(',')
    build_dir.mkdir(parents=True, exist_ok=True)

    print(f'[bench_onnx_rnn] python={sys.version.split()[0]} platform={platform.platform()}')
    import onnxruntime as ort
    import torch
    print(f'[bench_onnx_rnn] torch={torch.__version__} onnxruntime={ort.__version__}')
    print(f'[bench_onnx_rnn] cpu_count={os.cpu_count()} '
          f'torch_threads_default={torch.get_num_threads()}')

    results = {}
    for name in scale_names:
        cfg = SCALES[name]
        print(f'--- {name} ---')
        exported = export_scale(name, cfg, build_dir)
        print(f'  params: total={exported["params"]["total"]:,} '
              f'target={exported["params"]["rddlgn_params"]:,} '
              f'diff={exported["params"]["rel_diff_pct"]:+.2f}%')

        ort_results = {}
        for threads in ['default', 'threads1']:
            print(f'  ORT threads={threads} ...')
            ort_results[threads] = bench_ort_scale(
                name, cfg, build_dir, threads, args.reps, args.warmup)
            print(f'    encode_ms(median)={ort_results[threads]["encode_ms"]["median"]:.4f} '
                  f'decode_step_ms(median)={ort_results[threads]["decode_step_ms"]["median"]:.4f} '
                  f'seq_e2e_ms(median)={ort_results[threads]["seq_e2e_ms"]["median"]:.4f} '
                  f'tokens_per_s={ort_results[threads]["tokens_per_s"]:.1f}')

        mem = {}
        if not args.skip_mem:
            for threads in ['default', 'threads1']:
                print(f'  mem-probe threads={threads} ...')
                mem[threads] = measure_peak_rss(name, build_dir, threads)
                print(f'    max_rss_kb={mem[threads]["max_rss_kb"]}')

        results[name] = dict(
            **exported,
            ort=ort_results,
            mem_peak_rss=mem,
        )

    runtime_ref = measure_ort_runtime_reference_sizes()
    print(f'[bench_onnx_rnn] ORT capi lib raw={runtime_ref["capi_lib_raw_total_bytes"]:,}B '
          f'gz={runtime_ref["capi_lib_gz_total_bytes"]:,}B')

    out = dict(
        meta=dict(
            date=time.strftime('%Y-%m-%d'),
            platform=platform.platform(),
            machine=platform.machine(),
            processor=platform.processor(),
            cpu_count=os.cpu_count(),
            python_version=sys.version,
            torch_version=torch.__version__,
            onnxruntime_version=ort.__version__,
            onnx_export_mode='legacy TorchScript exporter (dynamo=False), opset_version=17',
            reps=args.reps,
            warmup=args.warmup,
            precision='fp32 (両モデルとも量子化なし)',
            note_fairness=(
                '公平性の教訓(ONNX 公平比較レポート)に'
                '従う: 配信サイズはgzip後で比較、fp32同士、精度比較はしない。'
                'ここではORT側の数値のみを計測する。RDDLGN側との突合は親タスク'
                'が行う。'
            ),
            background_load_note=(
                'ベンチ実行中に本セッションが意図的に起動した重い並行ジョブは'
                'なし。ただし共有マシン上の一般的なOS/GUIプロセス以外の負荷を'
                '独立監視・固定してはいない(uptimeのload averageのみ参考記録)。'
            ),
            uptime_at_run=_get_uptime(),
        ),
        scales=results,
        onnxruntime_reference_sizes=runtime_ref,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False) + '\n')
    print(f'[bench_onnx_rnn] wrote {out_path}')


if __name__ == '__main__':
    main()


# ---------------------------------------------------------------------------
# 付記: emb/hidden探索メモ(再現用)
# ---------------------------------------------------------------------------
# 各スケールのGRU総パラメータ数(embedding + encoder GRU + decoder GRU + head)は
#   embed = vocab*emb
#   gru(1層) = 3*hidden*(in+hidden) + 3*hidden*2   (weight_ih+weight_hh+bias*2)
#   total = embed + 2*gru(emb->hidden) + hidden*vocab + vocab
# という式になる。(vocab, target_params)ごとに emb,hiddenを
# {16,24,32,...,1024}×{16,...,2048}の格子で総当たりし、
# |total-target|/target が最小かつ"丸い"組を選んだ:
#   copy_tiny: vocab=50,  target=180,608   -> emb=192, hidden=96  (+0.49%)
#   charlm:    vocab=67,  target=1,223,993 -> emb=128, hidden=384 (-0.44%)
#   paper:     vocab=16000, target=17,910,000(collapsed後) -> emb=512, hidden=448 (+0.29%)
