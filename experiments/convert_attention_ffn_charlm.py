"""Attention→Attention-DLGN構造転写PoC(D-attn-ffn): FFNのみをtranscribeでLogicGPTへ移植する。

(experiments/convert_arith_structural_transfer.py)が算術回路で実証した
「構造転写(配線+ゲート初期化)+soft磨きはIO-KD/scratchを圧倒する」を、
公開transformer teacher(nanoGPT文字LM)のFFNブロックへ拡張する**部分**構造転写。

対象範囲(重要、正直な範囲宣言):
    teacherのFFN(`c_fc: Linear → GELU → c_proj: Linear`)は
    `convert/transcribe_seq.py::transcribe_ffn_block`で転写可能
    (2層Linear+活性化=transcribe.pyの隠れ層意味論がそのまま使える)。
    一方attention(softmax + q/k/v射影)は文献知見(Launay 2020: attentionの
    完全BP-free化は未達)通り論理ゲート化・転写の対象外とし、LogicGPTの
    `LogicAttention`(実数q/k/v射影+softmax、既定のBP学習)をそのまま使う。
    つまり本実験は「モデル全体の構造転写」ではなく「FFNサブモジュールのみの
    部分構造転写」であり、その旨をレポートでも明示する。

次元整合の制約(実装上のブロッカー、正直に記録する):
    `transcribe_ffn_block`は teacher FFNの`in_features`/`out_features`を
    そのまま転写先へ引き継ぐ(次元射影を行わない、モジュールdocstring参照)。
    公開teacher(`sosier/nanoGPT-shakespeare-char-tied-weights`)は
    `n_embd=384`であり、LogicGPTの`d_model`を384に一致させる必要がある
    (`d_model`が異なると`extract_ffn_linears`後の`sparsify_fanin`が
    無意味な次元不一致になる)。中間(GELU)幅は教師1536に対し、
    `FfnTranscribeSpec.hidden_width`のユニット部分選択(L2ノルム上位選択)で
    LogicGPT側の`LogicMLP`幅(`spec.width`)へ縮約する(情報損失を伴う近似、
    honestに`info['output_agreement']`で報告)。

LogicMLPへの書き込み(次の理由でblockのmlpを差し替える、実装ノート):
    `LogicTransformerBlock.__init__`は`LogicMLP`のspecを内部で固定
    (`width=d_model*mlp_ratio`, `decoder='groupsum'`)しており、この
    groupsum(group_size=mlp_ratio>1)は複数の生LUTビットを平均するため
    「1出力=1真理値表ビット」という transcribe_ffn_block の前提と噛み合わない
    (docs参照: 平均は非可逆で教師出力への1:1対応が崩れる)。そのため
    本実験では各blockの`mlp`属性を、`spec.width=d_model, decoder='identity'`
    (group_size=1、生LUTビットをそのまま出力する)の`LogicMLP`へ**置き換える**
    (transformer.py/logic_gpt.py自体は無改造、モデル構築後の属性差し替えのみ)。

アーム(reduced、3本。--quickでCPU数分完走):
    A(scratch)   ランダム初期化LogicGPT(attentionはBP既定)を純CE(seq)で学習。
    B(IO-KDのみ) 同じランダム初期化LogicGPTを、教師ロジットとの系列KD
                 (make_seq_kd_loss)で学習。構造転写なし。
    C(FFN構造転写+soft磨き) 各blockのmlpをtranscribe_ffn_block+
                 init_logic_mlp_from_ffnで教師FFNから初期化した上で、
                 A/Bと同じ学習ループ(--polish-kdでKD混合)を続ける
                 (recover()と同じ「良い初期値からの軽いsoft磨き」を
                 LogicGPT全体の学習ループへ適用したもの)。

評価: HARD BPC(eval mode = Heaviside離散化後、論理ゲート実機性能相当)を
主指標として報告する。lift(C-A)/lift(C-B)/teacher BPCを併記する。

使用例:
    # 煙実行(数分、CPU、既定コーパス流用)
    uv run python experiments/convert_attention_ffn_charlm.py --quick --no-wandb

    # 本実行(GPU推奨。--quick省略時は d_model=384 のLogicGPTでBPが重くなるため)
    uv run python experiments/convert_attention_ffn_charlm.py \\
        --corpus data-charlm/corpus_XXXXXXXX.pt --eval-corpus data-charlm/*_eval.pt \\
        -ni 20000 -ef 1000
"""

from __future__ import annotations

import argparse
import math
import os
import random
import sys
import time
from dataclasses import asdict
from datetime import datetime, timedelta, timezone

JST = timezone(timedelta(hours=9))

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from data.char_corpus import (
    PAD_ID,
    STUDENT_VOCAB_SIZE,
    TeacherCorpusDataset,
    load_tiny_shakespeare_val,
)
from env_utils import load_dotenv
from train_kd_charlm_rddlgn import (
    collate_raw_window,
    evaluate_bpc,
    evaluate_teacher_agreement,
    evaluate_teacher_bpc,
    generate_sample,
    make_pure_ce_seq_loss,
    resolve_glob,
    train_step_decoder_only,
    train_step_decoder_only_scratch,
)
from wandb_utils import finish as wandb_finish
from wandb_utils import init_wandb, log_metrics

from griffith_logic.convert.distill import make_seq_kd_loss
from griffith_logic.convert.text_teacher import resolve_text_teacher
from griffith_logic.convert.transcribe_seq import FfnTranscribeSpec, init_logic_mlp_from_ffn
from griffith_logic.hybrid.policy import LogicSpec
from griffith_logic.layers.logic_linear import LogicMLP
from griffith_logic.models.logic_gpt import LogicGPT, LogicGPTConfig

load_dotenv()

ARMS = ('A', 'B', 'C')


# ============================================================
# モデル構築(FFN転写に対応するd_model=teacher n_embdを強制)
# ============================================================

def build_logicgpt_config(args, teacher_n_embd: int) -> LogicGPTConfig:
    """LogicGPTConfigをteacherのn_embdへ一致させて構築する(転写の次元整合)。

    block_size=63固定(train_step_decoder_onlyのinput/targetオフセット規約、
    train_kd_charlm_rddlgn.pyのbuild_config_logicgptと同じ)。
    """
    config = LogicGPTConfig(
        vocab_size=STUDENT_VOCAB_SIZE,
        d_model=teacher_n_embd,
        n_layers=args.gpt_layers,
        n_heads=args.gpt_heads,
        block_size=63,
        mlp_ratio=args.gpt_mlp_ratio,
        pos_width=max(6, math.ceil(math.log2(64))),
        head_group_factor=args.head_group_factor,
        tau=1.0,
        n_inputs=args.n_inputs,
        parameterization='softmax',
        connections='random',
        merge='xor',
        norm='layernorm',
    )
    if config.d_model < config.pos_width:
        config.pos_width = config.d_model
    return config


def replace_block_mlps_for_transcription(model: LogicGPT, config: LogicGPTConfig) -> None:
    """各LogicTransformerBlockのmlpを、transcribe_ffn_block出力と1:1対応する
    shape(width=d_model, decoder='identity', depth=2)のLogicMLPへ置き換える。

    モジュールdocstring「LogicMLPへの書き込み」参照: 既定のLogicTransformerBlock
    spec(width=d_model*mlp_ratio, decoder='groupsum')はgroup_size>1の平均化を
    伴うため、transcribe_ffn_blockの「1出力=1真理値表ビット」前提と噛み合わない。
    ここではblock.mlpのみをin-placeで差し替える(transformer.py/logic_gpt.pyは
    無改造。全arm(A/B/C)で同じ差し替えを行い、mlp容量をアーム間で揃える
    ことでC以外のアームにも公平な比較にする)。
    """
    for blk in model.blocks:
        blk.mlp = LogicMLP(
            config.d_model,
            spec=LogicSpec(
                encoder='identity', decoder='identity', width=config.d_model,
                n_inputs=config.n_inputs, parameterization=config.parameterization,
                connections=config.connections, depth=2,
            ),
        )


def init_from_teacher_ffn(
    model: LogicGPT, teacher, calib_windows: torch.Tensor, args,
) -> list[dict]:
    """各LogicTransformerBlock.mlpを対応するteacher blockのFFNから転写初期化する。

    較正データはcalib_windowsの生トークンをteacher embeddingへ通した後の
    各blockのFFN入力(post-LN隠れ状態)を二値化して使うのが理想だが、
    teacher内部のGELU入力は実数(±非有界)であり、LogicMLPの入力契約は
    [0,1]の論理ビットである。transcribe_ffn_blockの較正はfan-in疎化の
    バイアス再較正とagreement測定のみに使う0/1データを要求するため、
    ここではteacherのblock入力をteacher自身の閾値二値化(>0)で0/1化した
    ものを較正データとして使う(transcribe.pyの隠れ層意味論 y=1[Wx+b>0] と
    同じ二値化規約をteacher活性そのものにも適用する)。

    Returns:
        各blockのinit_logic_mlp_from_ffn info辞書のリスト(層別近似誤差)。
    """
    teacher.model.eval()
    device = next(teacher.model.parameters()).device
    with torch.no_grad():
        idx = calib_windows.to(device)
        _b, t = idx.shape
        pos = torch.arange(t, dtype=torch.long, device=device)
        x = teacher.model.transformer.wte(idx) + teacher.model.transformer.wpe(pos)
        block_mlp_inputs = []
        for block in teacher.model.transformer.h:
            x = x + block.attn(block.ln_1(x))
            ln2 = block.ln_2(x)
            # block.mlpの実際の入力はln_2(x)。閾値二値化(>0)して較正データ
            # (transcribe.pyの隠れ層意味論 y=1[Wx+b>0] と同じ規約)にする。
            block_mlp_inputs.append((ln2 > 0).to(torch.float64).reshape(-1, ln2.shape[-1]))
            x = x + block.mlp(ln2)

    infos = []
    n_blocks = min(len(model.blocks), len(teacher.model.transformer.h))
    for i in range(n_blocks):
        calib_bits = block_mlp_inputs[i].cpu().numpy()
        spec = FfnTranscribeSpec(
            n_inputs=args.n_inputs,
            hidden_width=model.blocks[i].mlp.luts[0].out_dim,
            out_width=model.blocks[i].mlp.luts[1].out_dim,
            unit_select='l2_norm',
        )
        info = init_logic_mlp_from_ffn(
            model.blocks[i].mlp, teacher.model.transformer.h[i].mlp, calib_bits, spec,
        )
        info['block'] = i
        infos.append(info)
        print(
            f'  block {i}: hidden_agreement={info["hidden_agreement"]:.4f} '
            f'output_agreement={info["output_agreement"]:.4f} '
            f'(full_hidden={info["full_hidden_width"]} -> {info["transcribed_hidden_width"]})'
        )
    return infos


# ============================================================
# 1アームの学習ループ
# ============================================================

def run_arm(
    arm: str, args, teacher, train_loader, eval_loader, val_windows, device,
    config: LogicGPTConfig, calib_windows: torch.Tensor,
) -> dict:
    torch.manual_seed(args.seed + {'A': 0, 'B': 1000, 'C': 2000}[arm])
    model = LogicGPT(config).to(device)
    replace_block_mlps_for_transcription(model, config)
    model.to(device)

    transcribe_infos = None
    if arm == 'C':
        print(f'  [arm C] teacher FFN転写を実行中(blocks={config.n_layers})...')
        transcribe_infos = init_from_teacher_ffn(model, teacher, calib_windows, args)

    use_kd = arm in ('B',) or (arm == 'C' and args.polish_kd)
    if use_kd:
        _loss_fn_kd_raw = make_seq_kd_loss(
            alpha=args.distill_alpha, temperature=args.distill_temperature,
            standardize=args.distill_standardize, pad_id=PAD_ID,
        )

        # train_step_decoder_only(train_kd_charlm_rddlgn.py)はloss_fnを
        # step=キーワード付きで呼ぶ契約(main()のv1ラッパと同じパターン)。
        # make_seq_kd_lossの生クロージャはstepを受け取らないため、ここで吸収する。
        def loss_fn_kd(student_logits, target_ids, teacher_logits, step=0):
            return _loss_fn_kd_raw(student_logits, target_ids, teacher_logits)

        loss_fn_ce = None
    else:
        loss_fn_kd = None
        loss_fn_ce = make_pure_ce_seq_loss(pad_id=PAD_ID)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
        betas=(0.9, 0.999), eps=1e-8,
    )
    emb_reg_ramp = (args.emb_reg_ramp_start, args.emb_reg_ramp_end, args.emb_reg_max_weight)

    best_bpc_hard = float('inf')
    best_bpc_soft = float('inf')
    history = []

    def load_n(loader, n):
        i = 0
        while i < n:
            for batch in loader:
                yield batch
                i += 1
                if i == n:
                    return

    pbar = tqdm(
        enumerate(load_n(train_loader, args.num_iterations)),
        desc=f'arm {arm}', total=args.num_iterations,
    )
    t0 = time.time()
    for step, window in pbar:
        if use_kd:
            total_loss, main_loss, emb_loss, _ = train_step_decoder_only(
                model, teacher, window, optimizer, step, loss_fn_kd, device, emb_reg_ramp,
            )
        else:
            total_loss, main_loss, emb_loss = train_step_decoder_only_scratch(
                model, window, optimizer, step, loss_fn_ce, device, emb_reg_ramp,
            )
        pbar.set_postfix(loss=f'{total_loss:.4f}')

        if (step + 1) % args.eval_freq == 0 or step == args.num_iterations - 1:
            bpc_soft = evaluate_bpc(model, 'logicgpt', val_windows, device, use_soft=True)
            bpc_hard = evaluate_bpc(model, 'logicgpt', val_windows, device, use_soft=False)
            agree_hard = evaluate_teacher_agreement(
                model, 'logicgpt', eval_loader, teacher, device, use_soft=False,
            )
            best_bpc_soft = min(best_bpc_soft, bpc_soft)
            best_bpc_hard = min(best_bpc_hard, bpc_hard)
            history.append({
                'step': step + 1, 'bpc_soft': bpc_soft, 'bpc_hard': bpc_hard,
                'agree_hard': agree_hard,
            })
            print(
                f'\n  [arm {arm}][iter {step+1}/{args.num_iterations}] '
                f'loss={total_loss:.4f} BPC(soft)={bpc_soft:.4f} BPC(hard)={bpc_hard:.4f} '
                f'agree(hard)={agree_hard:.4f}'
            )
            if args.use_wandb:
                log_metrics({
                    f'arm_{arm}/loss': total_loss,
                    f'arm_{arm}/bpc_soft': bpc_soft,
                    f'arm_{arm}/bpc_hard': bpc_hard,
                    f'arm_{arm}/agree_hard': agree_hard,
                    f'arm_{arm}/best_bpc_hard': best_bpc_hard,
                }, step=step + 1)

    seconds = time.time() - t0
    sample_text = generate_sample(model, 'logicgpt', teacher, device, length=200)
    print(f'  [arm {arm}] 生成サンプル: {sample_text[:200]!r}')

    return {
        'arm': arm,
        'best_bpc_soft': best_bpc_soft,
        'best_bpc_hard': best_bpc_hard,
        'final_bpc_soft': history[-1]['bpc_soft'] if history else float('nan'),
        'final_bpc_hard': history[-1]['bpc_hard'] if history else float('nan'),
        'final_agree_hard': history[-1]['agree_hard'] if history else float('nan'),
        'seconds': round(seconds, 2),
        'history': history,
        'transcribe_infos': transcribe_infos,
        'sample_text': sample_text,
    }


# ============================================================
# メイン
# ============================================================

def _auto_device(arg: str | None) -> str:
    if arg is not None:
        return arg
    if torch.cuda.is_available():
        return 'cuda'
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


def main():
    parser = argparse.ArgumentParser(
        description='Attention→Attention-DLGN構造転写PoC: teacher FFN構造転写 vs IO-KD/scratch(char-LM HARD BPC)'
    )
    parser.add_argument('--teacher', type=str,
                        default='hf-nanogpt-char:sosier/nanoGPT-shakespeare-char-tied-weights')
    parser.add_argument('--corpus', type=str,
                        default='data-charlm/corpus__MinGPT-_500x64_T1.0_s42.pt')
    parser.add_argument('--eval-corpus', type=str,
                        default='data-charlm/corpus__MinGPT-_500x64_T1.0_s42_eval.pt')

    parser.add_argument('--gpt-layers', type=int, default=1)
    parser.add_argument('--gpt-heads', type=int, default=1)
    parser.add_argument('--gpt-mlp-ratio', type=int, default=1,
                        help='未使用のmlp内蔵spec(参考値のみ。実際のmlp幅は'
                             'replace_block_mlps_for_transcriptionでd_model幅に固定)')
    parser.add_argument('--head-group-factor', type=int, default=4)
    parser.add_argument('--n-inputs', type=int, default=2, choices=[2, 4, 6])

    parser.add_argument('--distill-alpha', type=float, default=0.25)
    parser.add_argument('--distill-temperature', type=float, default=4.0)
    parser.add_argument('--distill-standardize', dest='distill_standardize',
                        action='store_true', default=True)
    parser.add_argument('--no-distill-standardize', dest='distill_standardize',
                        action='store_false')
    parser.add_argument('--polish-kd', action='store_true',
                        help='arm CのソフトポリッシュにKD損失を混ぜる(既定off=純CE)')

    parser.add_argument('--lr', type=float, default=0.05)
    parser.add_argument('--weight-decay', type=float, default=0.001)
    parser.add_argument('-bs', '--batch-size', type=int, default=16)
    parser.add_argument('-ni', '--num-iterations', type=int, default=200)
    parser.add_argument('-ef', '--eval-freq', type=int, default=50)
    parser.add_argument('--eval-max-chars', type=int, default=20_000)
    parser.add_argument('--emb-reg-ramp-start', type=int, default=50)
    parser.add_argument('--emb-reg-ramp-end', type=int, default=2000)
    parser.add_argument('--emb-reg-max-weight', type=float, default=0.1)
    parser.add_argument('--n-calib-windows', type=int, default=64,
                        help='teacher FFN転写の較正に使うwindow数(訓練コーパスから採取)')

    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--quick', action='store_true',
                        help='煙実行(数分、CPU、少iteration)')
    parser.add_argument('--no-wandb', action='store_true')
    args = parser.parse_args()

    if args.quick:
        args.num_iterations = min(args.num_iterations, 60)
        args.eval_freq = min(args.eval_freq, 20)
        args.eval_max_chars = min(args.eval_max_chars, 4000)
        args.batch_size = min(args.batch_size, 8)
        args.n_calib_windows = min(args.n_calib_windows, 32)
        args.emb_reg_ramp_start = 20
        args.emb_reg_ramp_end = 200
        if args.device is None:
            # --quickは「CPUで数分完走」が要件(タスク指示)。加えて
            # text_teacher.NanoGPTCharTeacher.generate()のtorch.multinomialは
            # generatorがmps deviceのtensorに対しcpu-generatorだと
            # RuntimeErrorになる既知の制約があり(mps未対応、train_kd_charlm_
            # rddlgn.pyのgenerate_sample経由で顕在化)、既存コード
            # (train_kd_charlm_rddlgn.py)を無改造で使うためquick既定はcpuに固定する。
            args.device = 'cpu'

    args.device = _auto_device(args.device)
    print(f'デバイス: {args.device}')
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    print('teacherをロード中...')
    teacher = resolve_text_teacher(args.teacher, device=args.device)
    teacher_n_embd = teacher.cfg['n_embd']
    print(f'  teacher fingerprint={teacher.fingerprint}  n_embd={teacher_n_embd}  '
          f'n_layer={teacher.cfg["n_layer"]}')

    config = build_logicgpt_config(args, teacher_n_embd)
    print(f'LogicGPT config: d_model={config.d_model} n_layers={config.n_layers} '
          f'n_heads={config.n_heads} n_inputs={config.n_inputs}')

    corpus_path = resolve_glob(args.corpus)
    eval_corpus_path = resolve_glob(args.eval_corpus)
    print(f'訓練コーパス: {corpus_path}')
    print(f'評価コーパス: {eval_corpus_path}')

    train_set = TeacherCorpusDataset(corpus_path, mode='decoder_only')
    eval_set = TeacherCorpusDataset(eval_corpus_path, mode='decoder_only')
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, drop_last=True,
        collate_fn=collate_raw_window,
    )
    eval_loader = DataLoader(
        eval_set, batch_size=min(args.batch_size, len(eval_set)), shuffle=False,
        drop_last=False, collate_fn=collate_raw_window,
    )

    calib_windows = torch.stack([train_set[i] for i in range(min(args.n_calib_windows, len(train_set)))])

    val_windows = load_tiny_shakespeare_val(max_chars=args.eval_max_chars)

    print('teacher自身のBPCを計算中...')
    teacher_bpc = evaluate_teacher_bpc(teacher, val_windows, args.device)
    print(f'  teacher BPC = {teacher_bpc:.4f}  (uniform anchor = {math.log2(65):.4f})')

    args.use_wandb = False
    if not args.no_wandb:
        args.use_wandb = init_wandb(
            run_name=f'convert_attention_ffn_charlm_d{config.d_model}_ni{args.num_iterations}',
            config_dict={**vars(args), **asdict(config), 'teacher_bpc': teacher_bpc},
            device=args.device,
            tags=['convert', 'attention-ffn-transfer', 'charlm', 'logicgpt'],
            group='convert-attention-ffn-charlm',
        )

    results = {}
    for arm in ARMS:
        print(f'\n{"#" * 20} arm {arm} {"#" * 20}')
        results[arm] = run_arm(
            arm, args, teacher, train_loader, eval_loader, val_windows, args.device,
            config, calib_windows,
        )

    print('\n' + '=' * 90)
    print('結果: HARD BPC比較(小さいほど良い)')
    print('=' * 90)
    print('| arm | best BPC(hard) | final BPC(hard) | best BPC(soft) | agree(hard) | seconds |')
    print('|' + '---|' * 6)
    labels = {
        'A': 'A: scratch',
        'B': 'B: IO-KDのみ',
        'C': f'C: FFN構造転写+soft磨き{"+KD" if args.polish_kd else ""}',
    }
    for arm in ARMS:
        r = results[arm]
        print(
            f"| {labels[arm]} | {r['best_bpc_hard']:.4f} | {r['final_bpc_hard']:.4f} | "
            f"{r['best_bpc_soft']:.4f} | {r['final_agree_hard']:.4f} | {r['seconds']:.1f} |"
        )
    print(f'\nteacher BPC = {teacher_bpc:.4f}  (uniform anchor = {math.log2(65):.4f})')

    lift_vs_scratch = results['A']['best_bpc_hard'] - results['C']['best_bpc_hard']
    lift_vs_iokd = results['B']['best_bpc_hard'] - results['C']['best_bpc_hard']
    print('\n--- 派生指標(BPCは小さいほど良いのでlift = 相手のBPC - Cのbpc、正なら改善) ---')
    print(f'lift(C - A, hard BPC改善量) : {lift_vs_scratch:+.4f}')
    print(f'lift(C - B, hard BPC改善量) : {lift_vs_iokd:+.4f}')

    if results['C']['transcribe_infos']:
        agreements = [i['output_agreement'] for i in results['C']['transcribe_infos']]
        print(f"\nFFN転写近似誤差(較正データ上のoutput agreement, blocks平均): "
              f"{np.mean(agreements):.4f} (per-block: {[f'{a:.4f}' for a in agreements]})")

    if args.use_wandb:
        log_metrics({
            'summary/teacher_bpc': teacher_bpc,
            'summary/A_best_bpc_hard': results['A']['best_bpc_hard'],
            'summary/B_best_bpc_hard': results['B']['best_bpc_hard'],
            'summary/C_best_bpc_hard': results['C']['best_bpc_hard'],
            'summary/lift_vs_scratch': lift_vs_scratch,
            'summary/lift_vs_iokd': lift_vs_iokd,
        })
        wandb_finish()


if __name__ == '__main__':
    main()
