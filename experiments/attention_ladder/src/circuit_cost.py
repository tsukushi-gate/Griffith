"""attention梯子 L4: 回路規模の見積りアダプタ(A8)。

「梯子」の最終段(L4)では、collapse後の回路がどれだけの規模になるかを見積もる。
本モジュールは2つの経路を合成する:

  (1) 射影LUTコスト(実測): q/k/v 射影スタックの `LutLayer` 列だけを nn.Sequential
      として取り出し、既存の `export_ir → estimate_resources` に通して LUT総数 /
      論理深さ / 配線数を得る(実値)。attention本体(スコア・選択器)はIR対象外
      (ENV_NOTES §4: BLACKBOX)なので、ここには含めない。

  (2) attention機構の解析見積り(推測/近似): XNOR-popcountスコアと4案(C/S/B/A)の
      選択器を、比較器・MUX・加算器・BRAMビットの解析式でカウントする。式と仮定は
      各値に併記する。**解析式に基づく推定であり、実合成値ではない(推測/近似)**。

  (3) 残存FP箇所の列挙(トランスパイラ要件定義): collapse後も浮動小数点が残る箇所を
      モデル走査で列挙し、「なぜFPか / 整数化・比較器化で除去可能か」を1行で付す。

コア実装(griffith_logic/)は読み取りのみ。追加は experiments/attention_ladder/ 配下。

実行:
    uv run python experiments/attention_ladder/src/circuit_cost.py
"""

from __future__ import annotations

import math
import os
import sys

# ------------------------------------------------------------
# import規約: src/ と experiments/ をパス追加(harness/train と同じ流儀)
# ------------------------------------------------------------
_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
_EXP_DIR = os.path.abspath(os.path.join(_SRC_DIR, '..'))              # attention_ladder/
_REPO_ROOT = os.path.abspath(os.path.join(_EXP_DIR, '..', '..'))      # リポジトリのルート
for _p in (_EXP_DIR, _SRC_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import yaml  # noqa: E402
from ladder_attention import LadderAttention  # noqa: E402
from models import build_model  # noqa: E402
from torch import nn  # noqa: E402

from griffith_logic.export import estimate_resources, export_ir  # noqa: E402

# ------------------------------------------------------------
# 定数
# ------------------------------------------------------------
# sMNIST(行系列化)のmeta。data.py::load_smnist と一致させる(seq_len=28, 28bit/行)
_SMNIST_META = {
    'task': 'classification',
    'seq_len': 28,
    'input_dim': 28,
    'num_classes': 10,
}

# 処理対象のL3 4案(C/S/B/A)。相対パスはリポジトリルート基準で解決する
_L3_CONFIGS = (
    'experiments/attention_ladder/configs/l3c_smnist.yaml',
    'experiments/attention_ladder/configs/l3s_smnist.yaml',
    'experiments/attention_ladder/configs/l3b_smnist.yaml',
    'experiments/attention_ladder/configs/l3a_smnist.yaml',
)

# norm(内部名) → (案ラベル, 選択器の説明)
_NORM_LABEL = {
    'mux_gumbel': ('C', 'argmax(1トークン選択, Gumbel-ST)'),
    'softmax_int': ('S', 'softmax(FP, 固定論理化不可)'),
    'topk_ste': ('B', 'top-k + ビット多数決'),
    'threshold': ('A', '閾値ゲート s_i≥θ'),
}

# 出力Markdownの保存先(リポジトリルート基準)
_RESULT_MD = os.path.join(
    'experiments', 'attention_ladder', 'results', 'circuit_cost.md',
)


def _resolve_path(path: str) -> str:
    """cwd相対で見つからなければリポジトリルート基準で再解決する。"""
    if os.path.isfile(path):
        return path
    alt = os.path.join(_REPO_ROOT, path)
    if os.path.isfile(alt):
        return alt
    raise FileNotFoundError(f'ファイルが見つかりません: {path} (alt: {alt})')


def _tree_depth(n: int) -> int:
    """n入力の二分木(トーナメント/加算木)深さ = ceil(log2(n))。n<=1で0。"""
    return 0 if n <= 1 else math.ceil(math.log2(n))


# ============================================================
# (1) 射影LUTコスト(既存API: export_ir → estimate_resources)
# ============================================================

def _projection_lut_cost(model: nn.Module) -> dict:
    """全 LadderAttention の q/k/v 射影スタックのLUTコストを実測合算する。

    各射影スタックは `Binarize + LutLayer×proj_depth`。**LutLayer列だけ**を
    (`_LogicProjection.luts` は既にそのnn.Sequential) `export_ir(input_width=
    d_model)` に通す。Binarizeは比較器として (2) で別途カウントするためIRには
    含めない(意図的な分離であり抽出失敗ではない)。

    Returns:
        {'total_luts', 'logic_depth', 'wire_count', 'lut_counts',
         'per_proj': [...], 'n_projections', 'notes', 'ok'}
    """
    attns = [m for m in model.modules() if isinstance(m, LadderAttention)]
    total_luts = 0
    total_wire = 0
    max_depth = 0
    lut_counts: dict[int, int] = {}
    per_proj: list[dict] = []
    notes: list[str] = []
    ok = True

    for layer_idx, attn in enumerate(attns):
        d_model = attn.d_model
        for name in ('q_proj', 'k_proj', 'v_proj'):
            proj = getattr(attn, name)
            seq = proj.luts  # LutLayer列のnn.Sequential(Binarizeは含まない)
            tag = f'layer{layer_idx}.{name}'
            try:
                graph = export_ir(seq, input_width=d_model)
                est = estimate_resources(graph)
            except Exception as exc:  # noqa: BLE001 - 握り潰さず記録する
                ok = False
                notes.append(
                    f'{tag}: export_ir/estimate_resources 失敗 → '
                    f'このスタックはLUTコスト未計上(近似): {type(exc).__name__}: {exc}'
                )
                per_proj.append({'tag': tag, 'error': str(exc)})
                continue
            total_luts += est['total_luts']
            total_wire += est['wire_count']
            max_depth = max(max_depth, est['logic_depth'])
            for k, v in est['lut_counts'].items():
                lut_counts[k] = lut_counts.get(k, 0) + v
            per_proj.append({
                'tag': tag,
                'total_luts': est['total_luts'],
                'logic_depth': est['logic_depth'],
                'wire_count': est['wire_count'],
                'lut_counts': est['lut_counts'],
            })

    return {
        'total_luts': total_luts,
        'logic_depth': max_depth,
        'wire_count': total_wire,
        'lut_counts': lut_counts,
        'per_proj': per_proj,
        'n_projections': len(per_proj),
        'notes': notes,
        'ok': ok,
    }


# ============================================================
# (2) attention機構の解析見積り(推測/近似)
# ============================================================

def _score_cost(shape: dict) -> dict:
    """XNOR-popcountスコアのゲートコスト(1 attention層あたり)。

    per(head, query, key): d_k個のXNORゲート + popcount加算木(d_k-1加算器,
    深さ ceil(log2 d_k))。全体は H·Sq·Sk 個のスコアを持つ。
    """
    h, sq, sk, dk = shape['H'], shape['Sq'], shape['Sk'], shape['dk']
    n_scores_q = h * sk            # 1クエリ位置あたりのスコア数(全head×全key)
    n_scores_all = h * sq * sk     # 全クエリ位置
    return {
        # XNORゲート
        'xnor_q': n_scores_q * dk,
        'xnor_all': n_scores_all * dk,
        # popcount加算器(d_k-1/スコア)
        'popcount_adders_q': n_scores_q * (dk - 1),
        'popcount_adders_all': n_scores_all * (dk - 1),
        # スコア論理深さ: XNOR(1) + popcount加算木(ceil(log2 dk))
        'score_depth': 1 + _tree_depth(dk),
    }


def _selector_cost(norm: str, shape: dict, topk: int) -> dict:
    """選択器(norm別)の比較器/MUX/加算器コスト(1 attention層あたり)。

    数はすべて「per(head, query)を H·Sq(全S)/ H(1クエリ)で展開」した値。
    S案(softmax)は固定論理回路化不可のため None を返しFP扱いにする。
    """
    h, sq, sk, dv = shape['H'], shape['Sq'], shape['Sk'], shape['dv']
    per_q = h            # 1クエリ位置あたりの (head,query) 個数
    per_all = h * sq     # 全クエリ位置

    if norm == 'mux_gumbel':  # C案: argmax over Sk → 1トークン選択
        # 比較器: argmaxトーナメント = (Sk-1)/(head,query)
        cmp_unit = sk - 1
        # MUX(2:1)表現: v_argmax選択 = d_v bit × (Sk-1)個 の 2:1 MUX/(head,query)
        mux_unit = dv * (sk - 1)
        # 別表現: 選択線 d_v · ceil(log2 Sk)(log段MUX)/(head,query)
        mux_log_unit = dv * _tree_depth(sk)
        adder_unit = 0
        # 論理深さ: argmax木(ceil log2 Sk) + MUX木(ceil log2 Sk)
        depth = _tree_depth(sk) + _tree_depth(sk)
        note = 'argmax比較器+MUX選択。score_scaleは単調変換なので比較前に除去可(整数化)'
        return _selector_pack(
            per_q, per_all, cmp_unit, mux_unit, adder_unit, depth, note,
            extra={'mux_log_unit': mux_log_unit,
                   'mux_log_q': mux_log_unit * per_q,
                   'mux_log_all': mux_log_unit * per_all},
        )

    if norm == 'topk_ste':  # B案: hard top-k + ビット多数決
        k = min(topk, sk)
        # 比較器: top-k比較 ≈ k·Sk + 多数決の閾値比較 d_v個/(head,query)
        cmp_unit = k * sk + dv
        mux_unit = 0  # 選択はマスクAND(選択k個のvを合算)。専用MUXは数えない
        # 加算器: 多数決 = 選択k個のvビット和(d_v·(k-1)加算器 ≈ d_v·k)/(head,query)
        adder_unit = dv * k
        # 論理深さ: top-k逐次(k·log2 Sk 近似) + 多数決加算木(log2 k) + 閾値比較(1)
        depth = k * _tree_depth(sk) + _tree_depth(k) + 1
        note = f'top-k(k={k})比較+多数決。選択はマスクAND(MUXなし)。深さは逐次top-kの近似'
        return _selector_pack(
            per_q, per_all, cmp_unit, mux_unit, adder_unit, depth, note,
        )

    if norm == 'threshold':  # A案: a_i=[s_i≥θ] 閾値ゲート
        # 比較器: s_i≥θ を Sk個/(head,query)
        cmp_unit = sk
        mux_unit = 0
        # 加算器: 加重和(選択vの合算)= d_v·(Sk-1)/(head,query)
        adder_unit = dv * (sk - 1)
        # 論理深さ: 比較(1) + 加算木(ceil log2 Sk)。※正規化の除算は純論理でない(注記)
        depth = 1 + _tree_depth(sk)
        note = ('閾値比較+加算集約。正規化は選択数での除算が残る(純論理でない/近似)。'
                'score_scaleはθへ吸収し整数化可')
        pack = _selector_pack(
            per_q, per_all, cmp_unit, mux_unit, adder_unit, depth, note,
        )
        pack['needs_division'] = True
        return pack

    if norm == 'softmax_int':  # S案: FP softmax
        return {
            'is_fp': True,
            'comparators_q': None, 'comparators_all': None,
            'mux_q': None, 'mux_all': None,
            'adders_q': None, 'adders_all': None,
            'selector_depth': None,
            'note': 'FP softmax(整数スコアに exp/除算+温度学習)。固定論理回路化不可',
        }

    raise NotImplementedError(f"norm='{norm}' は解析見積り未対応")


def _selector_pack(
    per_q: int,
    per_all: int,
    cmp_unit: int,
    mux_unit: int,
    adder_unit: int,
    depth: int,
    note: str,
    extra: dict | None = None,
) -> dict:
    """per(head,query)単位量を 1クエリ/全S に展開してdict化する。"""
    pack = {
        'is_fp': False,
        'comparators_q': cmp_unit * per_q,
        'comparators_all': cmp_unit * per_all,
        'mux_q': mux_unit * per_q,
        'mux_all': mux_unit * per_all,
        'adders_q': adder_unit * per_q,
        'adders_all': adder_unit * per_all,
        'selector_depth': depth,
        'note': note,
    }
    if extra:
        pack.update(extra)
    return pack


def _attention_analytic_cost(shape: dict, norm: str, topk: int) -> dict:
    """1 attention層の解析見積り(スコア+選択器+Binarize+BRAM+深さ)。

    すべて解析式に基づく**推定(推測/近似)**であり実合成値ではない。
    """
    h, sq, sk, dk, dv, d_model = (
        shape['H'], shape['Sq'], shape['Sk'],
        shape['dk'], shape['dv'], shape['d_model'],
    )
    score = _score_cost(shape)
    sel = _selector_cost(norm, shape, topk)

    # 入力二値化(Binarize): 射影入力 d_model を各1比較器。q/k/vは同一入力x
    # (self-attention)なので共有すれば d_model、共有しなければ 3·d_model
    binarize_shared = d_model
    binarize_unshared = 3 * d_model

    # BRAM(value格納 = KVキャッシュのV部): H·Sk·d_v bit/token = v_bits·Sk
    bram_bits = h * sk * dv

    is_fp = sel['is_fp']

    # 比較器/MUX/加算器 総計(全S)。比較器 = Binarize + 選択器比較器
    if is_fp:
        comparators_all = binarize_shared              # 選択器はFP(比較器なし)
        mux_all = 0
        adders_all = score['popcount_adders_all']      # 選択器加算器はFP
        total_gates_all = (
            score['xnor_all'] + comparators_all + mux_all + adders_all
        )
        selector_depth = None
        total_depth = None
        comparators_q = binarize_shared
        mux_q = 0
        adders_q = score['popcount_adders_q']
        total_gates_q = score['xnor_q'] + comparators_q + mux_q + adders_q
    else:
        comparators_all = binarize_shared + sel['comparators_all']
        mux_all = sel['mux_all']
        adders_all = score['popcount_adders_all'] + sel['adders_all']
        total_gates_all = (
            score['xnor_all'] + comparators_all + mux_all + adders_all
        )
        comparators_q = binarize_shared + sel['comparators_q']
        mux_q = sel['mux_q']
        adders_q = score['popcount_adders_q'] + sel['adders_q']
        total_gates_q = score['xnor_q'] + comparators_q + mux_q + adders_q
        selector_depth = sel['selector_depth']
        total_depth = score['score_depth'] + selector_depth

    return {
        'shape': shape,
        'norm': norm,
        'is_fp_selector': is_fp,
        'score': score,
        'selector': sel,
        'binarize_shared': binarize_shared,
        'binarize_unshared': binarize_unshared,
        'bram_bits': bram_bits,
        # 全S位置(1 attention層)
        'xnor_all': score['xnor_all'],
        'popcount_adders_all': score['popcount_adders_all'],
        'comparators_all': comparators_all,
        'mux_all': mux_all,
        'adders_all': adders_all,
        'total_gates_all': total_gates_all,
        # 1トークン=1クエリ位置(1 attention層)
        'xnor_q': score['xnor_q'],
        'popcount_adders_q': score['popcount_adders_q'],
        'comparators_q': comparators_q,
        'mux_q': mux_q,
        'adders_q': adders_q,
        'total_gates_q': total_gates_q,
        # 論理深さ(推定)
        'score_depth': score['score_depth'],
        'selector_depth': selector_depth,
        'total_depth': total_depth,
    }


# ============================================================
# (3) 残存FP箇所の列挙(トランスパイラ要件定義)
# ============================================================

def _fp_residuals(model: nn.Module, norm: str) -> list[dict]:
    """collapse後も浮動小数点が残る箇所を列挙する(L4要件)。

    各項目: {'location', 'count', 'why_fp', 'removable'}。
    設計上FP固定の埋め込み/正規化/ヘッドと、選択器依存(score_scale/softmax)を分ける。
    """
    residuals: list[dict] = []

    # --- モデル走査で実在モジュールを数える ---
    n_linear = sum(1 for m in model.modules() if isinstance(m, nn.Linear))
    n_layernorm = sum(1 for m in model.modules() if isinstance(m, nn.LayerNorm))
    n_embedding = sum(1 for m in model.modules() if isinstance(m, nn.Embedding))
    attns = [m for m in model.modules() if isinstance(m, LadderAttention)]

    # 入力射影 Linear(input_dim→d_model)
    residuals.append({
        'location': '入力射影 nn.Linear(28→128)',
        'count': 1,
        'why_fp': '実数MNIST行→d_modelの線形写像。乗加算がFP',
        'removable': '設計上FP固定(変数をattentionに限定)。整数量子化/二値射影に置換は可',
    })
    # 位置埋め込み Embedding
    residuals.append({
        'location': '位置埋め込み nn.Embedding(seq_len→d_model)',
        'count': sum(1 for m in model.modules() if isinstance(m, nn.Embedding)),
        'why_fp': '学習FPベクトルのテーブル参照',
        'removable': '設計上FP固定。Gray-code等の固定{0,1}符号なら比較器/定数化で除去可',
    })
    # LayerNorm×5
    residuals.append({
        'location': f'LayerNorm ×{n_layernorm}(pre-norm×4 + final×1)',
        'count': n_layernorm,
        'why_fp': '特徴横断の平均/分散正規化(√・除算)。特徴間依存でLUT折り畳み不能',
        'removable': '設計上FP固定。固定小数点/近似正規化への置換は要検討(現状FP残存)',
    })
    # 分類ヘッド Linear(d_model→num_classes)
    residuals.append({
        'location': '分類ヘッド nn.Linear(128→10)',
        'count': 1,
        'why_fp': 'd_model→クラスロジットの線形写像。乗加算がFP',
        'removable': '設計上FP固定。GroupSum等の整数集約に置換は可',
    })

    # --- 射影encoderの確認: sigmoidでなくBinarize(比較器)か ---
    enc_ok = all(
        type(a.q_proj.binarize).__name__ == 'Binarize'
        for a in attns
    ) if attns else False
    residuals.append({
        'location': '射影encoder(q/k/v入力の実数→{0,1})',
        'count': len(attns) * 3,
        'why_fp': ('Binarize(閾値0の比較器)であり sigmoid ではない' if enc_ok
                   else '要確認(Binarizeでない可能性)'),
        'removable': ('FP残存なし: 比較器で二値化済み(閾値0のsign比較)。'
                      'ただし入力自体はpre-norm(LayerNorm)出力のためLayerNormはFP'
                      if enc_ok else '要確認'),
    })

    # --- score_scale(FPスカラ): 選択器により除去可否が変わる ---
    has_score_scale = any(hasattr(a, 'score_scale') for a in attns)
    if has_score_scale:
        if norm in ('mux_gumbel', 'topk_ste'):
            removable = ('除去可: argmax/top-kは正の単調変換に不変。'
                         '比較前にscaleを落とし整数popcountで比較できる')
        elif norm == 'threshold':
            removable = ('除去可: s·scale≥θ ⇔ s≥θ/scale。'
                         'scaleを閾値θへ吸収し整数化できる')
        else:  # softmax_int(S案)
            removable = ('除去不可: softmax(scale·s/temp)はscaleが分布形状に影響。'
                         'FP温度として残る')
        residuals.append({
            'location': 'score_scale(学習FPスカラ, xnor_popcountスコア)',
            'count': len(attns),
            'why_fp': 'popcount整数スコアに乗じる学習FP係数',
            'removable': removable,
        })

    # --- 選択器固有のFP(S案のsoftmax温度 / A案の除算) ---
    if norm == 'softmax_int':
        residuals.append({
            'location': 'S案 softmax(softmax_temp学習FP温度)',
            'count': len(attns),
            'why_fp': 'exp・除算・FP温度による正規化。固定論理回路化不可',
            'removable': '除去不可(FP softmaxが本質)。C/B/A案なら選択器で置換され消える',
        })
    if norm == 'threshold':
        residuals.append({
            'location': 'A案 正規化の除算(加重和 / 選択数)',
            'count': len(attns),
            'why_fp': '選択トークン数での割り算。純論理でない',
            'removable': '固定小数点除算/逆数LUT/非正規化(和のまま)への変更で除去可(近似)',
        })

    # --- FFN(LogicMLP)の実数境界(参考: 梯子共通、attention段外) ---
    # ffn='logic' は encoder=sigmoid / decoder=groupsum_centered のハイブリッド境界。
    # attention梯子L4の対象外だが、モデル全体でFPが残る点として記録する。
    residuals.append({
        'location': 'FFN LogicMLP encoder=sigmoid / decoder=groupsum_centered(参考)',
        'count': len(attns) * 1,  # ブロックごと1 FFN(=attention層数と同数)
        'why_fp': ('sigmoid(実数→[0,1]確率)とgroupsum(整数和/τのFPスケール)が'
                   '実数⇔論理の境界に残る。attention段の変数ではないが全体では残存'),
        'removable': 'encoderをBinarize/thermometer(比較器)へ、decoderを整数GroupSumへ置換で除去可',
    })

    residuals.append({
        'location': f'(参考) 実在モジュール数: Linear={n_linear}, '
                    f'LayerNorm={n_layernorm}, Embedding={n_embedding}',
        'count': 0,
        'why_fp': 'モデル走査による実測(内訳確認用)',
        'removable': '-',
    })

    return residuals


# ============================================================
# 公開API
# ============================================================

def estimate_ladder_circuit(config_path: str) -> dict:
    """1つのL3 configから回路コスト見積り一式を計算して返す。

    Returns:
        {'stage','norm','norm_label','shape','projection','attention',
         'fp_residuals','config_path'}
    """
    resolved = _resolve_path(config_path)
    with open(resolved, encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    model_cfg = cfg['model']
    stage = cfg.get('stage', '?')
    norm = str(model_cfg.get('norm', 'softmax'))
    topk = int(model_cfg.get('topk', 4))

    # CPUでモデル構築(sMNIST meta)。eval不要(export_irが内部でdeepcopy+eval)
    model = build_model(model_cfg, _SMNIST_META)

    # 射影LUTコスト(実測)
    projection = _projection_lut_cost(model)

    # 形状(先頭 LadderAttention と meta から確定)
    attns = [m for m in model.modules() if isinstance(m, LadderAttention)]
    if not attns:
        raise RuntimeError('LadderAttention が見つからない(attention=logic_proj必須)')
    a0 = attns[0]
    seq_len = int(_SMNIST_META['seq_len'])
    shape = {
        'n_layers': len(attns),
        'H': a0.n_heads,
        'Sq': seq_len,
        'Sk': seq_len,
        'dk': a0.dk,
        'dv': a0.v_bits // a0.n_heads,
        'qk_bits': a0.qk_bits,
        'v_bits': a0.v_bits,
        'd_model': a0.d_model,
    }

    # attention機構の解析見積り(1 attention層)
    attention = _attention_analytic_cost(shape, norm, topk)

    # 残存FP列挙
    fp = _fp_residuals(model, norm)

    label, sel_desc = _NORM_LABEL.get(norm, ('?', norm))
    return {
        'config_path': config_path,
        'stage': stage,
        'norm': norm,
        'norm_label': label,
        'selector_desc': sel_desc,
        'shape': shape,
        'projection': projection,
        'attention': attention,
        'fp_residuals': fp,
    }


# ============================================================
# Markdown出力
# ============================================================

def _fmt(n) -> str:
    """整数を桁区切りで整形。None/FPは記号化。"""
    if n is None:
        return 'FP(不可)'
    if isinstance(n, float):
        return f'{n:g}'
    return f'{n:,}'


def _render_markdown(results: list[dict]) -> str:
    """4案の見積りをMarkdown比較表に整形する。"""
    lines: list[str] = []
    lines.append('# attention梯子 L4: 回路コスト見積り比較(C/S/B/A)')
    lines.append('')
    lines.append('自動生成: `experiments/attention_ladder/src/circuit_cost.py`')
    lines.append('')
    lines.append('- **射影LUT数** = 既存API `export_ir → estimate_resources`（実測値）。')
    lines.append('- **attention機構（比較器/MUX/加算器/BRAM/深さ）= 解析式に基づく'
                 '推定（推測/近似）。実合成値ではない。**')
    lines.append('- attention系はIR抽出対象外（BLACKBOX, ENV_NOTES §4）のため'
                 '解析アダプタで別途カウント。')
    lines.append('')

    # 形状(全案共通)
    sh = results[0]['shape']
    lines.append('## 形状パラメータ（sMNIST, 全案共通）')
    lines.append('')
    lines.append(
        f"n_layers={sh['n_layers']}, H(n_heads)={sh['H']}, "
        f"S=Sq=Sk={sh['Sq']}, d_model={sh['d_model']}, "
        f"qk_bits={sh['qk_bits']}→d_k={sh['dk']}/head, "
        f"v_bits={sh['v_bits']}→d_v={sh['dv']}/head"
    )
    lines.append('')

    # 表1: 射影LUTコスト(実測)
    lines.append('## 表1: 射影LUTコスト（実測: export_ir→estimate_resources）')
    lines.append('')
    lines.append('q/k/v射影の `LutLayer` 列のみをIR化。Binarizeは含まず（表2で比較器計上）。'
                 'モデル総計 = 全 LadderAttention の q+k+v 合算。')
    lines.append('')
    lines.append('| 案 | stage | 射影LUT総数 | 論理深さ | 配線数 | LUT内訳 | 抽出 |')
    lines.append('|---|---|--:|--:|--:|---|---|')
    for r in results:
        p = r['projection']
        lut_break = ', '.join(f'N{k}:{v}' for k, v in sorted(p['lut_counts'].items()))
        ok = 'OK' if p['ok'] else '一部失敗(近似)'
        lines.append(
            f"| {r['norm_label']} | {r['stage']} | {_fmt(p['total_luts'])} | "
            f"{p['logic_depth']} | {_fmt(p['wire_count'])} | {lut_break} | {ok} |"
        )
    lines.append('')
    # 射影の失敗ノート
    proj_notes = [n for r in results for n in r['projection']['notes']]
    if proj_notes:
        lines.append('射影抽出の注記:')
        for n in dict.fromkeys(proj_notes):  # 重複排除
            lines.append(f'- {n}')
        lines.append('')

    # 表2: attention機構(全S位置, 1 attention層あたり)
    lines.append('## 表2: attention機構の解析見積り（推測/近似・1 attention層・全S位置）')
    lines.append('')
    lines.append('比較器 = Binarize入力二値化(共有 d_model) + 選択器比較器。'
                 '加算器 = popcount + 選択器加算器。総ゲート = XNOR+比較器+MUX+加算器。')
    lines.append('')
    lines.append('| 案 | 選択器 | XNOR | 比較器 | MUX(2:1) | 加算器 | 総ゲート | '
                 'BRAM[bit] | 論理深さ |')
    lines.append('|---|---|--:|--:|--:|--:|--:|--:|--:|')
    for r in results:
        a = r['attention']
        depth = a['total_depth']
        depth_s = _fmt(depth) if depth is not None else 'FP(不可)'
        lines.append(
            f"| {r['norm_label']} | {r['selector_desc']} | "
            f"{_fmt(a['xnor_all'])} | {_fmt(a['comparators_all'])} | "
            f"{_fmt(a['mux_all'])} | {_fmt(a['adders_all'])} | "
            f"{_fmt(a['total_gates_all'])} | {_fmt(a['bram_bits'])} | {depth_s} |"
        )
    lines.append('')

    # 表3: attention機構(1トークン=1クエリ位置)
    lines.append('## 表3: attention機構（推測/近似・1 attention層・1トークン=1クエリ位置）')
    lines.append('')
    lines.append('| 案 | XNOR | 比較器 | MUX(2:1) | 加算器 | 総ゲート |')
    lines.append('|---|--:|--:|--:|--:|--:|')
    for r in results:
        a = r['attention']
        lines.append(
            f"| {r['norm_label']} | {_fmt(a['xnor_q'])} | {_fmt(a['comparators_q'])} | "
            f"{_fmt(a['mux_q'])} | {_fmt(a['adders_q'])} | {_fmt(a['total_gates_q'])} |"
        )
    lines.append('')

    # 表4: モデル総計(×n_layers, 射影LUT + attention全S)
    lines.append('## 表4: モデル総計（射影LUT実測 + attention解析×n_layers）')
    lines.append('')
    lines.append(f"attention機構は 1層値 × n_layers(={sh['n_layers']}) で全モデル換算。"
                 'BRAMも×n_layers。射影LUTは既に全層合算済み。')
    lines.append('')
    lines.append('| 案 | 射影LUT(実測) | XNOR | 比較器 | MUX | 加算器 | '
                 'attention総ゲート | BRAM[bit] |')
    lines.append('|---|--:|--:|--:|--:|--:|--:|--:|')
    nl = sh['n_layers']
    for r in results:
        a = r['attention']
        p = r['projection']
        lines.append(
            f"| {r['norm_label']} | {_fmt(p['total_luts'])} | "
            f"{_fmt(a['xnor_all'] * nl)} | {_fmt(a['comparators_all'] * nl)} | "
            f"{_fmt(a['mux_all'] * nl)} | {_fmt(a['adders_all'] * nl)} | "
            f"{_fmt(a['total_gates_all'] * nl)} | {_fmt(a['bram_bits'] * nl)} |"
        )
    lines.append('')

    # C案のMUX別表現
    c = next((r for r in results if r['norm'] == 'mux_gumbel'), None)
    if c is not None:
        sel = c['attention']['selector']
        lines.append('### C案 MUXの別表現')
        lines.append('')
        lines.append(
            f"- 2:1 MUX表現（表2/3で採用）: 全S {_fmt(c['attention']['mux_all'])} 個 "
            f"= H·d_v·(Sk-1)。"
        )
        lines.append(
            f"- log段選択表現: 全S {_fmt(sel.get('mux_log_all'))} 選択線 "
            f"= H·d_v·ceil(log2 Sk)。"
        )
        lines.append('')

    # 仮定・近似
    lines.append('## 仮定・近似（すべて推測/近似）')
    lines.append('')
    lines.append('- XNORスコア: per(head,query,key)に d_k XNOR + popcount加算木'
                 '（d_k-1加算器, 深さ ceil(log2 d_k)）。総XNOR=H·Sq·Sk·d_k。')
    lines.append('- C(argmax): 比較器=(Sk-1)/(head,query), '
                 'MUX=d_v·(Sk-1)/(head,query)。深さ=argmax木+MUX木=2·ceil(log2 Sk)。')
    lines.append('- B(top-k): 比較器≈k·Sk+d_v(多数決), 加算器=d_v·k。'
                 'MUXなし（マスクAND選択）。深さは逐次top-kの近似。')
    lines.append('- A(threshold): 比較器=Sk, 加算器=d_v·(Sk-1)。'
                 '正規化の除算が純論理でなく残る（近似）。')
    lines.append('- S(softmax): 固定論理回路化不可（FP exp/除算/温度）。比較器/MUXなし。')
    lines.append('- Binarize比較器: 射影入力 d_model を共有入力とみなし d_model/層'
                 '（q/k/v非共有なら 3·d_model/層）。')
    lines.append('- BRAM: value格納 H·Sk·d_v bit/token（KVキャッシュV部）。')
    lines.append('- 論理深さは二分木深さ ceil(log2 n) の近似。配線・面積・FF段は未計上。')
    lines.append('')

    # FP残存リスト(全案分)
    lines.append('## 残存FP箇所（トランスパイラ要件定義）')
    lines.append('')
    for r in results:
        lines.append(f"### {r['norm_label']}案（stage {r['stage']}, norm={r['norm']}）")
        lines.append('')
        lines.append('| 箇所 | 個数 | なぜFPか | 整数化/比較器化で除去可能か |')
        lines.append('|---|--:|---|---|')
        for item in r['fp_residuals']:
            cnt = item['count'] if item['count'] else '-'
            lines.append(
                f"| {item['location']} | {cnt} | {item['why_fp']} | {item['removable']} |"
            )
        lines.append('')

    return '\n'.join(lines)


def main() -> None:
    results = [estimate_ladder_circuit(p) for p in _L3_CONFIGS]
    md = _render_markdown(results)

    # stdout出力
    print(md)

    # ファイル出力(リポジトリルート基準)
    out_path = _RESULT_MD
    out_dir = os.path.dirname(out_path)
    if not os.path.isdir(out_dir):
        out_dir = os.path.join(_REPO_ROOT, out_dir)
        out_path = os.path.join(_REPO_ROOT, _RESULT_MD)
    os.makedirs(out_dir, exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(md + '\n')
    print(f'\n[written] {out_path}', file=sys.stderr)


if __name__ == '__main__':
    main()
