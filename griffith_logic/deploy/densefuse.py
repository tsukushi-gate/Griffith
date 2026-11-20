"""dense多層融合カーネルの生成(GEM(DAC2025) Boomerang型)。

`plan.py::fuse_dense_layers` が検出した「連続する lut_dense 列 + 末尾の
group_sum」(`dense_fuse` op)を、1スレッドグループ(CUDAなら1スレッド
ブロック)=1サンプルの単一カーネルへ落とす。活性はthreadgroup/sharedメモリの
ping-pong2本に置き、層間はブロック内バリアで区切る。真理値表・添字表は
従来どおりdevice/constantバッファのまま(共有メモリに置くのは活性だけ)。

Metal/CUDAでカーネルの「殻」(引数属性・共有メモリ宣言・バリア呼び出し・
スレッド/ブロックの現在位置の取り方)は違うが、層本体の計算式(真理値表引き)
は完全に同じ文字列を共有する(`_layer_lines`)。両バックエンドとも
ローカル変数名を `tid_`(スレッド内位置)/`tgs_`(グループの全スレッド数)/
`bid_`(グループ番号=バッチ内サンプル番号)に揃えてあるので、殻の違いは
「この3つの値をどう取得するか」だけに閉じ込められる。

最小実験(har_dlgn_k6000_gumbel、Metal実機、GPU busy時間、5セット
中央値、`ir.simulate`との整数厳密一致を確認済み):

    batch=1    : 0.81x(悪化。1グループしか立たず占有率が下がるため)
    batch=32   : 1.91x
    batch=128  : 2.51x(最良)
    batch=256  : 2.30x

判定基準(batch=1のGPU時間 または 最良バッチのスループットで1.3倍以上)を
最良バッチ側で満たしたため採用。batch=1の悪化は既知のトレードオフとして
この関数のdocstringに残す。
"""

from .plan import lut_dense_indices_fit_u16

#: 生成カーネル名の接頭辞(convcodegen.py と同じ流儀)。
#: **"mdl_k_"で始めること** ─ `mdl_metal_init`(runtime/metal/mdl_metal.mm)は
#: ライブラリ内の関数のうち "mdl_k_" 接頭辞のものだけを自動でパイプライン
#: 登録するため(CODEGEN_KERNEL_PREFIXも同じ制約下にある)。
CODEGEN_KERNEL_PREFIX = 'mdl_k_densefuse_'


def act_buf_bytes(layers: list[dict]) -> int:
    """ping-pong活性バッファ1本ぶんのバイト数。`plan.dense_fuse_act_bytes`
    と同じ計算(バックエンド側は `op.attrs['layers']` から独立に呼べるよう、
    ここにも同じ関数を持つ ─ 値は必ず一致する、両者ともテストで固定)。"""
    widths = [int(layers[0]['in_width'])]
    widths += [int(layer['out_width']) for layer in layers]
    return max(widths)


def idx_ctype_metal(layer: dict) -> str:
    """この層の添字バッファのMSL型('ushort'か'int')。"""
    return 'ushort' if lut_dense_indices_fit_u16(layer['in_width']) else 'int'


def idx_ctype_cuda(layer: dict) -> str:
    """この層の添字バッファのCUDA(C++)型。"""
    return ('unsigned short' if lut_dense_indices_fit_u16(layer['in_width'])
           else 'int')


def _layer_lines(layers: list[dict], k: int, gsz: int) -> list[str]:
    """層本体の計算式(バックエンド非依存)。

    ローカル変数 `tid_`(スレッド内位置)/`tgs_`(グループの総スレッド数)/
    `bid_`(バッチ内サンプル番号)は呼び出し側が用意しておくこと。
    `{BARRIER}` プレースホルダだけを呼び出し側が文字列置換(`.replace()`、
    本文にCのブレース `{`/`}` を含むので `.format()` は使えない)で埋める
    (Metal: `threadgroup_barrier(mem_flags::mem_threadgroup);` /
     CUDA: `__syncthreads();`)。
    """
    in_w0 = int(layers[0]['in_width'])
    lines = [
        f'for (int i = tid_; i < {in_w0}; i += tgs_) '
        f'bufA[i] = x0[(long)bid_ * {in_w0} + i];',
        '{BARRIER}',
    ]
    cur, nxt = 'bufA', 'bufB'
    for i, layer in enumerate(layers):
        ln = i + 1
        ow = int(layer['out_width'])
        lines.append(f'/* layer {ln}: in_width={layer["in_width"]} '
                    f'out_width={ow} */')
        lines.append(f'for (int o = tid_; o < {ow}; o += tgs_) {{')
        lines.append(f'  unsigned int addr = ((unsigned int){cur}[idx{ln}[o]])'
                    f' | (((unsigned int){cur}[idx{ln}[{ow} + o]]) << 1u);')
        lines.append(f'  {nxt}[o] = (unsigned char)((tt{ln}[o] >> addr) & 1u);')
        lines.append('}')
        lines.append('{BARRIER}')
        cur, nxt = nxt, cur
    # curが最終層の出力(widthはlayers[-1]['out_width'])。
    lines.append(f'if (tid_ < {k}) {{')
    lines.append('  int s = 0;')
    lines.append(f'  for (int i = 0; i < {gsz}; i++) '
                f's += (int){cur}[tid_ * {gsz} + i];')
    lines.append(f'  out[(long)bid_ * {k} + tid_] = s;')
    lines.append('}')
    return lines


def build_metal_kernel(op_attrs: dict, kernel_name: str) -> str:
    """融合カーネルのMSLソース(1関数)。

    バッファ割付け: buffer(0)=入力(1サンプルの生のuchar列)、buffer(1)=
    group_sum出力(int)、buffer(2+2i)/buffer(3+2i)=i層目(0始まり)の
    添字/真理値表。threadgroup(0)/(1)=ping-pong活性。
    """
    layers = op_attrs['layers']
    k, gsz = int(op_attrs['k']), int(op_attrs['gsz'])
    args = ['device const uchar *x0 [[buffer(0)]]',
           'device int *out [[buffer(1)]]']
    for i, layer in enumerate(layers):
        ln = i + 1
        b = 2 + 2 * i
        ct = idx_ctype_metal(layer)
        args.append(f'constant {ct} *idx{ln} [[buffer({b})]]')
        args.append(f'constant uint *tt{ln} [[buffer({b + 1})]]')
    args += [
        'threadgroup uchar *bufA [[threadgroup(0)]]',
        'threadgroup uchar *bufB [[threadgroup(1)]]',
        'uint tid_ [[thread_position_in_threadgroup]]',
        'uint tgs_ [[threads_per_threadgroup]]',
        'uint bid_ [[threadgroup_position_in_grid]]',
    ]
    sig = ',\n    '.join(args)
    body = '\n  '.join(_layer_lines(layers, k, gsz)).replace(
        '{BARRIER}', 'threadgroup_barrier(mem_flags::mem_threadgroup);')
    return f'kernel void {kernel_name}(\n    {sig}) {{\n  {body}\n}}\n'


#: CUDA側スレッドブロック幅(要実測チューニング。バッチ=グリッド側
#: なので、これは「1サンプルぶんの並列度」。har系のout_widthは600〜6000
#: なので1024で全層をカバーしきれる幅もあれば複数周ループする幅もある
#: ─ どちらでも`_layer_lines`のstride-loopが正しく処理する)。
CUDA_BLOCK_THREADS = 1024


def build_cuda_kernel(op_attrs: dict, kernel_name: str) -> str:
    """融合カーネルのCUDA(`__global__`)ソース(1関数)。

    CUDAはAOTコンパイルなので、Metalと違い実行時の名前引きディスパッチが
    要らない ─ 生成した.cu内でこの関数をそのまま `<<<batch,
    CUDA_BLOCK_THREADS>>>` で起動できる(conv codegenと同じ設計)。

    **実行未検証**(このMacにCUDA実機/nvccが無いため。タスク指示
    どおり「ソース対称実装、実行検証は後日」)。
    """
    layers = op_attrs['layers']
    k, gsz = int(op_attrs['k']), int(op_attrs['gsz'])
    act_bytes = act_buf_bytes(layers)
    params = ['const unsigned char *x0', 'int *out']
    for i, layer in enumerate(layers):
        ln = i + 1
        ct = idx_ctype_cuda(layer)
        params.append(f'const {ct} *idx{ln}')
        params.append(f'const unsigned int *tt{ln}')
    sig = ',\n    '.join(params)
    body = '\n  '.join(_layer_lines(layers, k, gsz)).replace(
        '{BARRIER}', '__syncthreads();')
    return f'''__global__ void {kernel_name}(
    {sig}) {{
  __shared__ unsigned char bufA[{act_bytes}];
  __shared__ unsigned char bufB[{act_bytes}];
  const int tid_ = (int)threadIdx.x, tgs_ = (int)blockDim.x;
  const int bid_ = (int)blockIdx.x;
  {body}
}}
'''
