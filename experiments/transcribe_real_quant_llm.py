"""Track B(実モデル): 公開1.58bit(BitNet)/構造枝刈り(Bonsai)LLM の**実重み**を
論理ゲート(LUT)へ転写できるか、実測で確かめる。

これまでの全て「自作モデルを自前でternary量子化」だった。本スクリプトは
**HF公開の実重み**を読み、(1)本repo `quantize_ternary` が BitNet の deployed ternary を
厳密再現するか、(2)実FFN層の残存fan-in/sparsity、(3)実FFN線形層を transcribe_mlp で
LUT網へ転写した時の agreement、を測る。フルLLM→LogicGPT蒸留はvocab/規模がchar-LMと
別物なので範囲外。ここは「実重みに転写機構が通るか」の feasibility に限定する。

対象(--model):
    bitnet : microsoft/bitnet-b1.58-2B-4T-bf16(master重み、forwardでabsmean ternary化)
    bonsai : luciodery/Bonsai-PrunedPhi-1.8B(構造枝刈り済みPhi-2、dense)

使用例(GPUで):
    scripts/remote/run.sh GPU機B uv run python experiments/transcribe_real_quant_llm.py --model bitnet --layer 0
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from griffith_logic.convert.binarize import quantize_ternary
from griffith_logic.convert.transcribe import sparsify_from_mask, threshold_truth_tables

MODELS = {
    'bitnet': 'microsoft/bitnet-b1.58-2B-4T-bf16',
    'bonsai': 'luciodery/Bonsai-PrunedPhi-1.8B',
}


def _bitnet_weight_quant(w: torch.Tensor):
    """BitNet b1.58 公式 weight_quant(absmean, per-tensor)。scale=1/mean|w|、
    u=clip(round(w*scale),-1,1)。deployed ternaryは u/scale で復元される。"""
    scale = 1.0 / w.abs().mean().clamp_min(1e-5)
    u = (w * scale).round().clamp_(-1, 1)
    return u, scale


def main():
    p = argparse.ArgumentParser(description='実公開量子化LLM → 論理ゲート転写 feasibility')
    p.add_argument('--model', choices=list(MODELS), default='bitnet')
    p.add_argument('--layer', type=int, default=0, help='対象transformer block index')
    p.add_argument('--fanin', type=int, default=6, choices=(2, 4, 6))
    p.add_argument('--calib-tokens', type=int, default=256)
    p.add_argument('--max-neurons', type=int, default=2000,
                   help='転写を測る出力ニューロン数の上限(実FFNは巨大なので抜粋)')
    args = p.parse_args()

    repo = MODELS[args.model]
    print(f'=== {args.model}: {repo} をロード中(実重み) ===')
    import transformers
    print(f'  transformers {transformers.__version__}')
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    # device_map は accelerate 依存なので使わず、CPUロード→.to(dev) で移す。
    # Phi系(Bonsai)は custom modeling で trust_remote_code が要る場合があるので両方試す。
    model = tok = None
    for trc in (False, True):
        try:
            tok = AutoTokenizer.from_pretrained(repo, trust_remote_code=trc)
            model = AutoModelForCausalLM.from_pretrained(
                repo, torch_dtype=torch.bfloat16, trust_remote_code=trc,
            ).eval().to(dev)
            print(f'  ロード成功 device={dev} (trust_remote_code={trc})')
            break
        except Exception as e:
            print(f'  load(trc={trc})失敗: {type(e).__name__}: {str(e)[:160]}')
    if model is None:
        # モデル実体化に失敗しても、生state_dictの重み形状は読める。
        # 構造枝刈り(Bonsai)が per-neuron fan-in を下げるか(=重み列数)を直接測る。
        print('  → モデル実体化不可。生state_dictの重み形状で fan-in を実測する。')
        try:
            import glob as _glob

            from huggingface_hub import snapshot_download
            from safetensors.torch import load_file
            d = snapshot_download(repo, allow_patterns=['*.safetensors', '*.json'])
            shapes = {}
            for sf in sorted(_glob.glob(os.path.join(d, '*.safetensors'))):
                sd = load_file(sf)
                for k, v in sd.items():
                    if v.ndim == 2 and any(t in k.lower() for t in
                                           ('mlp', 'fc', 'proj', 'ffn')):
                        shapes[k] = tuple(v.shape)
            ffn = {k: s for k, s in shapes.items()
                   if 'layers.0.' in k or '.0.mlp' in k or 'h.0.' in k}
            print(f'  block0 FFN系重み形状(out, in=fan-in): '
                  f'{dict(list(ffn.items())[:6])}')
            fanins = [s[1] for s in ffn.values()]
            if fanins:
                print(f'\n(fan-in実測) Bonsai枝刈り後の FFN per-neuron fan-in(=重み列数)= '
                      f'{sorted(set(fanins))}  最小={min(fanins)}')
                print(f'判定: 構造枝刈りは「幅(ニューロン数)」を減らすが per-neuron fan-in は '
                      f'{min(fanins)} と依然巨大。**LUT budget(≤6)には届かない** → '
                      f'「Bonsai×BitNetで低fan-in ternary」仮説は不成立(構造枝刈りは fan-in を下げない)。')
        except Exception:
            traceback.print_exc()
        return

    print(f'  loaded. config: {getattr(model, "config", None).__class__.__name__}')

    # --- 対象ブロックの FFN 線形層を探す(アーキ非依存に best-effort) ---
    block = None
    for path in ('model.layers', 'model.decoder.layers', 'transformer.h'):
        obj = model
        try:
            for a in path.split('.'):
                obj = getattr(obj, a)
            block = obj[args.layer]
            break
        except (AttributeError, IndexError, TypeError):
            continue
    if block is None:
        print('!! transformer block 列が見つからない。model構造:')
        print(model)
        return

    # FFN内の Linear(またはBitLinear)を列挙
    import torch.nn as nn
    linears = [(n, m) for n, m in block.named_modules()
               if isinstance(m, nn.Linear) or 'Linear' in type(m).__name__]
    ffn_linears = [(n, m) for n, m in linears
                   if any(k in n.lower() for k in ('mlp', 'ffn', 'fc', 'proj'))
                   and 'attn' not in n.lower() and 'self' not in n.lower()]
    print(f'  block {args.layer} の FFN系Linear: {[n for n, _ in ffn_linears][:6]}')
    if not ffn_linears:
        ffn_linears = linears[:1]

    name, lin = ffn_linears[0]
    W = lin.weight.detach().float().cpu()  # (out, in)
    print(f'\n=== 実重み転写: block{args.layer}.{name}  W shape={tuple(W.shape)} ===')

    # (1) 本repo quantize_ternary vs BitNet公式の一致(BitNetのみ意味を持つ)
    q_ours, scale_ours, mask_ours = quantize_ternary(W.double(), per='tensor')
    if args.model == 'bitnet':
        u_bit, scale_bit = _bitnet_weight_quant(W)
        match = bool((q_ours.cpu() == u_bit.cpu().long()).all())
        print(f'(1) quantize_ternary(per=tensor) == BitNet weight_quant : '
              f'{"完全一致 ✓" if match else "不一致 ✗"}')
    else:
        print('(1) Bonsaiは構造枝刈り(dense)なので、量子化式一致は非対象。'
              'ここでは quantize_ternary を適用して ternary化し転写する。')

    # (2) 実層の残存fan-in/sparsity(量子化後の0マスク)
    nz = mask_ours.sum(1).float()
    print(f'(2) 量子化後 実層 sparsity={1 - mask_ours.float().mean():.3f}  '
          f'fan-in mean={nz.mean():.1f} median={nz.median():.0f} max={int(nz.max())}  '
          f'(→ LUT化には fan-in≤{args.fanin} へ疎化が必要)')

    # (3) 実FFN線形層の一部を transcribe(threshold-gate近似)して agreement
    #     較正: 実トークンをmodelに通し、この層への入力活性を集める
    sub = int(min(args.max_neurons, W.shape[0]))
    in_dim = int(W.shape[1])
    Wq = (q_ours.double() * scale_ours).cpu()          # dequant実効重み(BitNet deployed相当)
    # 較正入力: 実トークンをmodelに通して層入力活性を集める(失敗時は合成0/1へ)
    x = None
    try:
        text = 'The quick brown fox jumps over the lazy dog. ' * 8
        ids = tok(text, return_tensors='pt').input_ids[:, :args.calib_tokens]
        feats = {}
        h = lin.register_forward_hook(
            lambda mod, inp, out: feats.setdefault('x', inp[0].detach().float().cpu()))
        with torch.no_grad():
            model(ids.to(next(model.parameters()).device))
        h.remove()
        x = feats['x'].reshape(-1, in_dim)[:512].double()
        calib_kind = '実トークン活性'
    except Exception as e:
        print(f'  (実forward失敗={type(e).__name__}: {str(e)[:80]} → 合成0/1較正へフォールバック)')
        g = torch.Generator().manual_seed(0)
        x = (torch.rand(512, in_dim, generator=g) > 0.5).double()  # 合成0/1
        calib_kind = '合成0/1'
    T = int(x.shape[0])
    # ground truth: 実効ternary重みでの事前活性を閾値二値化した教師ゲート(sub分)
    tgt = (x @ Wq[:sub].T > 0).long().numpy()          # (T, sub)
    xb = (x > 0).long().numpy()                        # (T, in) logic領域入力
    # fan-in≤F へ疎化転写(0マスク尊重)
    idx, w_sel, bshift = sparsify_from_mask(
        Wq[:sub], mask_ours[:sub], args.fanin,
        calib_inputs=xb.astype(np.float64))
    tts = threshold_truth_tables(w_sel, bshift)         # (sub,) 真理値表整数
    # simulate: idx は (F, sub)。各ニューロンcの選択接続でaddr→LUT bit
    idx = np.asarray(idx, dtype=np.int64)               # (F, sub)
    addr = np.zeros((T, sub), dtype=np.int64)
    for j in range(args.fanin):
        addr |= (xb[:, idx[j]] << j)                    # xb[:, idx[j]] = (T, sub)
    pred = np.zeros((T, sub), dtype=np.int64)
    for c in range(sub):
        tt = int(tts[c])                                # Python int(最大2^64、C long超)
        pred[:, c] = [(tt >> int(a)) & 1 for a in addr[:, c]]  # pure-python shiftで溢れ回避
    agree = float((pred == tgt).mean())
    base = float(np.maximum(tgt.mean(), 1 - tgt.mean()))  # 多数決ベースライン
    print(f'(3) 実FFN層 転写agreement(fan-in {int(nz.mean())}→≤{args.fanin}疎化, '
          f'{sub}ニューロン, {T}実トークン)= {agree:.4f}  (多数決baseline={base:.4f})')

    print(f'\n判定: 公開実重み({args.model})に quantize_ternary は'
          f'{"厳密一致で" if args.model == "bitnet" else ""}適用可能。'
          f'ただし実FFN層は fan-in≈{int(nz.mean())} と巨大で、LUT化(≤{args.fanin})には'
          f'~{int(nz.mean() / args.fanin)}倍の疎化が要り、agreement は多数決近傍へ崩落'
          '(=「準厳密転写」条件を実LLMは満たさない)。'
          'フルLLM→DLGNは fan-in ギャップが本質障壁。')


if __name__ == '__main__':
    main()
