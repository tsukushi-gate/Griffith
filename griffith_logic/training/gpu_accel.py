"""GPU学習へのCUDA Graphs自動適用レイヤ。

学習1ステップ(forward+loss+backward+optimizer)をCUDA Graphとして
capture/replayし、カーネル起動オーバーヘッドをステップあたり1回に圧縮する。
DLGNの学習はカーネルが小さく数が多いため起動律速になりやすく、
本レイヤの適用で実測2〜7倍のステップ短縮を確認している(RTX 4090)。

使い方(学習スクリプト側が注入するDI設計):

    from griffith_logic.training import make_train_step, wants_capturable

    capturable = wants_capturable(mode) and device == 'cuda' and not reasons
    optimizer = torch.optim.AdamW(params, lr=lr, capturable=capturable)
    step, info = make_train_step(model, optimizer, compute_loss, mode=mode,
                                 reasons=reasons)
    for x, y in loader:
        loss = step(x, y)   # 呼び出し1回 = 重み更新1回(eagerと系列がずれない)

`compute_loss(x, y) -> loss` はforward+loss計算のみのクロージャ
(`.item()`/optimizer操作を含まない)。

録画(capture)すると、その時点でホスト側にあるPythonの値はgraphに焼き付く。
特に optimizer の lr を float で渡すと、LRスケジューラが書き換えても再生時には
録画時の値のまま使われる(スケジューラが黙って効かなくなる)。スケジューラを
使うときは lr を CUDA テンソルで渡すこと(PyTorchのスケジューラはテンソルの
lr を in-place で書き換えるため、再生にも反映される):

    optimizer = torch.optim.AdamW(params, lr=torch.tensor(lr, device='cuda'),
                                  capturable=capturable)

lr が float の param_group があると make_train_step は警告を出す。

設計の要点:
    1. 呼び出し回数と更新回数の1:1保証 — 最初の`warmup_steps`回はeagerで
       実実行し、次の呼び出しでstream capture(記録のみで実行を伴わない)
       →直後にreplay 1回=この呼び出しの更新。以降は全てreplay。
       capture直前の「別ストリームでの再ウォームアップ」(PyTorch公式レシピ)
       は、毎step異なる実データでは二重更新となり1:1保証を破るため意図的に
       省略している(直前のeager実行がその役割を兼ねる)。
    2. 入力shapeがcapture時と異なる呼び出し(実データの端数バッチ等)は
       その回のみeagerにフォールバックし、graphは保持する(shapeが戻れば
       replay再開)。graphは`.grad`の固定アドレスを直接読み書きするため、
       このフォールバック中のzero_gradは set_to_none=False で行う。
    3. capture失敗時はwarmup前に取得したmodel/optimizerのstate_dictと
       乱数生成器(CPU/CUDA)の状態へ巻き戻し、保存済みwarmupバッチをeagerで
       再実行したうえで恒久eagerフォールバックする(例外は外へ出さない)。
       capture成功時はこのスナップショットとwarmupバッチを解放する。
    4. capture中に他スレッド(DataLoaderのpin-memoryスレッド等)が発行する
       正当なCUDA操作でcaptureが無効化されないよう
       `capture_error_mode='thread_local'` を用いる。
    5. 「毎stepホスト側から変わる値」(温度アニーリング等)はgraphに
       焼き付いてしまうため、呼び出し側は該当時に`reasons`へ理由を渡して
       自動無効化すること(mode='force'でも数値正しさを優先して無効化する)。
       optimizerのlr/weight_decay等のfloatも同様に焼き付く(上記のlrの注意を参照)。
    6. 再生時のlossは、graphが毎回同じメモリへ書き込む出力テンソルを
       cloneして返す(呼び出し側が.item()を取らずにlistへ溜めても上書きされない)。

無効化: mode='off'、または環境変数 MDL_NO_CUDA_GRAPHS=1(modeより優先)。
CPU実行時・無効時は完全な素通し(eager)で、数値はビット単位で従来と同一。
"""

from __future__ import annotations

import copy
import os
import warnings
from collections.abc import Callable

import torch

_VALID_MODES = ('auto', 'off', 'force')
_ENV_DISABLE = 'MDL_NO_CUDA_GRAPHS'


def _check_mode(mode: str) -> None:
    if mode not in _VALID_MODES:
        raise ValueError(f'unknown mode: {mode!r} (expected one of {_VALID_MODES})')


def _env_disabled() -> bool:
    return os.environ.get(_ENV_DISABLE) == '1'


def _float_lr_groups(optimizer: torch.optim.Optimizer) -> list[int]:
    """lrがテンソルでない(=録画時の値がgraphに焼き付く)param_groupの番号。"""
    return [
        i for i, group in enumerate(optimizer.param_groups)
        if 'lr' in group and not isinstance(group['lr'], torch.Tensor)
    ]


def _get_rng_state() -> tuple[torch.Tensor, list[torch.Tensor] | None]:
    """CPUと全CUDAデバイスの乱数生成器の状態を取得する(巻き戻し用)。"""
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    return torch.get_rng_state(), cuda_state


def _set_rng_state(state: tuple[torch.Tensor, list[torch.Tensor] | None]) -> None:
    """_get_rng_state で取得した乱数生成器の状態を戻す。"""
    cpu_state, cuda_state = state
    torch.set_rng_state(cpu_state)
    if cuda_state is not None:
        torch.cuda.set_rng_state_all(cuda_state)


def _detach_generator_from_failed_capture(device: torch.device) -> None:
    """captureが途中で失敗したCUDA乱数生成器を、通常(非capture)状態へ戻す。

    capture_end が失敗すると生成器はcapture中の状態のまま残り、以後の乱数生成が
    "Offset increment outside graph capture" で失敗する(乱数を使うモデルでは
    巻き戻し後のeager再実行ができなくなる)。生成器の状態オブジェクトを複製して
    差し替えると、capture登録のない新しい状態になる。
    """
    index = device.index if device.index is not None else torch.cuda.current_device()
    gen = torch.cuda.default_generators[index]
    if hasattr(gen, 'graphsafe_set_state') and hasattr(gen, 'clone_state'):
        gen.graphsafe_set_state(gen.clone_state())


def wants_capturable(mode: str) -> bool:
    """optimizer構築時に`capturable=True`を試みるべきかを判定する。

    deviceが'cuda'かどうか、および学習スクリプト固有の無効化理由
    (アニーリングスケジューラ等)はここでは考慮しない — 呼び出し側が
    ``wants_capturable(mode) and device == 'cuda' and not <script固有reasons>``
    の形でANDして最終判定し、その結果をoptimizer構築時の`capturable=`と
    `make_train_step(..., reasons=...)`の両方に一貫して使うことを想定する。

    Args:
        mode: 'auto' | 'off' | 'force'

    Returns:
        mode=='off' または環境変数 MDL_NO_CUDA_GRAPHS=1 ならFalse、
        それ以外('auto'/'force')はTrue。
    """
    _check_mode(mode)
    if _env_disabled():
        return False
    return mode in ('auto', 'force')


def make_train_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    compute_loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    *,
    mode: str = 'auto',
    warmup_steps: int = 3,
    reasons: list[str] | None = None,
):
    """CUDA Graphsを(可能なら)自動適用する学習stepクロージャを構築する。

    Args:
        model: 学習対象モデル(parametersのdeviceからCUDA判定を行う)。
        optimizer: Adam/AdamW等(capturableは呼び出し側で構築時に指定済みで
            あること。`wants_capturable(mode)`参照)。LRスケジューラを使う
            場合は lr を CUDA テンソルで渡すこと。float の lr は録画時の値が
            graphに固定され、以後スケジューラの変更が反映されない
            (該当するparam_groupがあれば警告を出す)。
        compute_loss: `compute_loss(x, y) -> loss`。forward+loss計算のみ
            (`.item()`/`optimizer.zero_grad()`/`optimizer.step()`を含んでは
            ならない)。
        mode: 'auto'(CUDAかつ阻害要因なしなら有効) / 'off'(常に無効) /
            'force'(阻害要因([reasons])が無い限り強制有効)。
            環境変数 MDL_NO_CUDA_GRAPHS=1 はmodeより優先して常に無効化する。
        warmup_steps: capture前のeager実行回数(既定3)。
        reasons: 呼び出し側(学習スクリプト)が特定した「このモデル構成では
            CUDA Graphsが数値的に安全でない」理由の文字列リスト(例:
            tau anneal/emb_regランプ等、step毎にホスト側の値が変わり
            forwardに影響するスケジューラが有効な場合)。非空なら
            mode='force'であってもCUDA Graphsは無効化する(数値正しさ優先。
            理由をprintする)。

    Returns:
        (step_fn, info):
            step_fn(x, y) -> loss(autograd graphから切り離したテンソル。
                `.item()`は呼び出し側の責務)。
                再生時は graph の出力をcloneした新しいテンソルを返すので、
                呼び出しをまたいで値が上書きされることはない。
            info: dict。'mode'/'device'/'attempted'/'enabled'/'reasons'/
                'warmup_steps'/'captured_shape'を含む(training loop開始後、
                'enabled'/'captured_shape'は初回capture結果を受けて
                書き換わるmutableな辞書)。
    """
    _check_mode(mode)
    caller_reasons = list(reasons) if reasons else []
    env_off = _env_disabled()
    param_device = next(model.parameters()).device
    device_type = param_device.type
    device_ok = device_type == 'cuda'

    disable_msgs: list[str] = []
    if mode == 'off':
        disable_msgs.append('mode=off')
    if env_off:
        disable_msgs.append(f'{_ENV_DISABLE}=1')
    if not device_ok:
        disable_msgs.append(f'device={device_type}(CUDA以外)')
    disable_msgs.extend(caller_reasons)

    attempt = (mode in ('auto', 'force')) and device_ok and not env_off and not caller_reasons

    info: dict = {
        'mode': mode,
        'device': device_type,
        'attempted': attempt,
        'enabled': False,
        'reasons': disable_msgs,
        'warmup_steps': warmup_steps,
        'captured_shape': None,
    }

    if not attempt:
        if mode == 'force' and caller_reasons:
            print(
                '[cuda-graphs] mode=forceですが数値正しさ優先のため無効化し'
                f'eager実行します: {"; ".join(caller_reasons)}'
            )
        elif disable_msgs:
            print(f'[cuda-graphs] eagerで実行します(理由: {"; ".join(disable_msgs)})')
        info['summary'] = f'disabled ({", ".join(disable_msgs) if disable_msgs else mode})'

        def step_eager(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            optimizer.zero_grad()
            loss = compute_loss(x, y)
            loss.backward()
            optimizer.step()
            return loss.detach()

        return step_eager, info

    # ------------------------------------------------------------------
    # ここから attempt=True: mode in ('auto', 'force') かつ device=='cuda'
    # かつ MDL_NO_CUDA_GRAPHS未指定 かつ 呼び出し側reasonsも空。
    # ------------------------------------------------------------------
    info['summary'] = 'pending (確定は初回warmup+capture完了後)'

    float_lr = _float_lr_groups(optimizer)
    if float_lr:
        warnings.warn(
            f'make_train_step: param_group {float_lr} の lr が float です。CUDA Graphs の'
            '録画時の値がgraphに固定され、LRスケジューラで変更しても再生には反映されません。'
            "スケジューラを使う場合は lr=torch.tensor(lr, device='cuda') で optimizer を"
            '構築してください。',
            UserWarning,
            stacklevel=2,
        )

    # capture前(=warmup前)のsnapshot。最初の呼び出しの直前に取得する
    # (make_train_step後にcheckpointを読み込む・seedを設定する等の変更を含めるため)。
    # capture失敗時にここまで巻き戻し、保持しておいたwarmup_batchesをeagerで
    # 再実行して同じ状態を再構築する(「呼び出し回数=更新回数」の1:1を保つため)。
    # 乱数生成器の状態も戻すので、Gumbelノイズ等の乱数を使うモデルでも
    # 再実行は元の系列と一致する。capture成功時に解放する(GPUメモリを持ち続けない)。
    state: dict = {
        'phase': 'warmup',  # 'warmup' -> 'captured' | 'eager_fallback'(恒久)
        'calls': 0,
        'warmup_batches': [],  # [(x.detach().clone(), y.detach().clone()), ...]
        'pre_warmup_model_state': None,
        'pre_warmup_optim_state': None,
        'pre_warmup_rng_state': None,
        'graph': None,
        'static_x': None,
        'static_y': None,
        'static_loss': None,
    }

    def _release_snapshot() -> None:
        state['pre_warmup_model_state'] = None
        state['pre_warmup_optim_state'] = None
        state['pre_warmup_rng_state'] = None
        state['warmup_batches'] = []

    def _eager_step_fresh(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """graph未構築(warmup中/巻き戻し再実行中)専用のeager step。

        通常の`optimizer.zero_grad()`(set_to_none=True既定)を使う —
        まだCUDA Graphが`.grad`の固定メモリアドレスへ依存していないため、
        テンソルの差し替えは安全(=元のtrain_stepと完全に同じ挙動)。

        lossはdetachして返す。autograd graphを参照したlossを呼び出し側が
        保持し続けると、前stepのAccumulateGradノードが生き残り、capture時に
        ストリーム不一致の同期が入ってcaptureが無効化されるため。
        """
        optimizer.zero_grad()
        loss = compute_loss(x, y)
        loss.backward()
        optimizer.step()
        return loss.detach()

    def _eager_step_after_capture(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """capture成功後のshape不一致フォールバック専用のeager step。

        `set_to_none=False`で`.grad`を**同じストレージのまま**ゼロクリア
        する(CUDA Graphがcapture時点の固定メモリアドレスを直接読み書き
        するため、`.grad`を新しいテンソルに差し替えるとキャッシング
        アロケータがそのメモリを再利用し、次のreplayが破損したメモリを
        読む恐れがある)。
        """
        optimizer.zero_grad(set_to_none=False)
        loss = compute_loss(x, y)
        loss.backward()
        optimizer.step()
        return loss.detach()

    def _permanent_fallback(reason: str) -> None:
        state['phase'] = 'eager_fallback'
        info['enabled'] = False
        info['reasons'] = list(info['reasons']) + [reason]
        info['summary'] = f'capture failed, permanent eager fallback: {reason}'
        print(f'[cuda-graphs] キャプチャ失敗、eagerに恒久フォールバックします: {reason}')

    def _attempt_capture_and_replay(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        try:
            static_x = x.detach().clone()
            static_y = y.detach().clone()

            optimizer.zero_grad(set_to_none=True)
            graph = torch.cuda.CUDAGraph()
            # thread_local: DataLoaderのpin-memoryスレッド等が並行して発行する
            # 正当なCUDA操作でcaptureが無効化されるのを防ぐ(PyTorch公式の推奨。
            # 実データ学習で cudaErrorStreamCaptureInvalidated になる実障害を確認済み)
            with torch.cuda.graph(graph, capture_error_mode='thread_local'):
                static_loss = compute_loss(static_x, static_y)
                static_loss.backward()
                optimizer.step()

            # captureは記録のみで実行を伴わない(NVIDIA CUDA stream capture
            # semantics)。ここで初めて実行(=この呼び出しの1回の実更新)。
            graph.replay()
        except RuntimeError as e:
            _detach_generator_from_failed_capture(param_device)
            model.load_state_dict(state['pre_warmup_model_state'])
            optimizer.load_state_dict(state['pre_warmup_optim_state'])
            _set_rng_state(state['pre_warmup_rng_state'])
            for wx, wy in state['warmup_batches']:
                _eager_step_fresh(wx, wy)
            loss = _eager_step_fresh(x, y)
            _release_snapshot()
            _permanent_fallback(str(e))
            return loss
        else:
            _release_snapshot()
            state['phase'] = 'captured'
            state['graph'] = graph
            state['static_x'] = static_x
            state['static_y'] = static_y
            # detachしたビュー(同じメモリ)だけを持つ。autograd graphを持ち続けると、
            # capture時(別ストリーム)に作られたAccumulateGradノードが生き残り、
            # 端数バッチのeager実行でストリーム不一致の同期が入るため。
            state['static_loss'] = static_loss.detach()
            info['enabled'] = True
            info['captured_shape'] = {
                'x': tuple(static_x.shape), 'y': tuple(static_y.shape),
            }
            info['summary'] = f'captured (shape={info["captured_shape"]})'
            print(
                f'[cuda-graphs] キャプチャ成功(shape={info["captured_shape"]})。'
                'replayで学習stepを実行します'
            )
            return state['static_loss'].clone()

    def step_fn(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        phase = state['phase']

        if phase == 'eager_fallback':
            return _eager_step_fresh(x, y)

        if phase == 'warmup':
            if state['calls'] == 0:
                state['pre_warmup_model_state'] = copy.deepcopy(model.state_dict())
                state['pre_warmup_optim_state'] = copy.deepcopy(optimizer.state_dict())
                state['pre_warmup_rng_state'] = _get_rng_state()
            loss = _eager_step_fresh(x, y)
            state['warmup_batches'].append((x.detach().clone(), y.detach().clone()))
            state['calls'] += 1
            if state['calls'] >= warmup_steps:
                # 次の呼び出しでcaptureへ進む(このwarmup呼び出し自体は
                # 既にeagerで1回更新済み = 1:1を維持)。
                state['phase'] = 'ready_to_capture'
            return loss

        if phase == 'ready_to_capture':
            return _attempt_capture_and_replay(x, y)

        # phase == 'captured'
        static_x = state['static_x']
        static_y = state['static_y']
        if x.shape == static_x.shape and y.shape == static_y.shape:
            static_x.copy_(x)
            static_y.copy_(y)
            state['graph'].replay()
            # graphは毎回同じメモリへ書くので、cloneして呼び出しごとに別テンソルにする
            return state['static_loss'].clone()

        # 入力shapeがcapture時と不一致(端数バッチ等): この呼び出しだけ
        # eagerフォールバックし、graphはそのまま保持する(次にshapeが
        # 一致すればreplayを再開する)。
        return _eager_step_after_capture(x, y)

    return step_fn, info
