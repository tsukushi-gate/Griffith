/* griffith-logic — 複数コアで回すための**任意**ヘルパ
 *
 * これは本体ではない。`mdl_kernels.c` と生成モデルはスレッドライブラリを
 * 一切参照しないので、ベアメタルでは**このファイルをリンクしなければよい**。
 * 「依存ゼロ・ヒープ確保ゼロ・ベアメタル動作」はコア側の性質として保たれる。
 *
 * 並列化の単位は**画像1枚**。DLGNの1推論はビットプレーンを次々に書き換える
 * 直線処理で、層の中に切れ目が無い(ゲート命令列は前の命令の出力を次が読む)。
 * 1枚を複数コアで分割するより、**コアごとに独立したコンテキストを持たせて
 * 別々の画像を流す**ほうが素直に線形に伸びる。
 *
 * `mdl_run_pool`/`mdl_range_parallel` は内部で**常駐ワーカプール**を
 * 使う(実測でpthread_createの起床に19〜53µsかかる、残件)。
 * 呼び出しのたびにスレッドを作っていたのを、初回呼び出しでlazy生成した
 * ワーカがcondvarで待機する形に変えた。**シグネチャと戻り値の意味は
 * 完全に不変**で、既存の呼び出し側コードは一切変更不要。挙動の違いは:
 *
 *   - プールはプロセス内で**1つ**(シングルトン)。同じスレッド数(n_ctx /
 *     n_threads)の呼び出しが続く限りスレッドを再利用し、数が変わったら
 *     一旦joinしてから作り直す(bench_deploy.pyのようにN=1→2→4→8と
 *     掃くベンチでは、遷移のたびに1回だけ作り直しが起きる)
 *   - `mdl_run_pool`/`mdl_range_parallel` の呼び出しは**プロセス全体で
 *     直列化される**(同時に2箇所から呼ぶと、後から来た方は前の呼び出しが
 *     終わるまでブロックする)。単一の呼び出しスレッドが逐次に
 *     `mdl_preproc_parallel` → `mdl_run_pool` のように呼ぶ既存の使い方
 *     (`docs`のサンプル、生成コードのdriver.c)は影響を受けない。
 *     **ただし**、このプールのワーカスレッド自身の中から
 *     `mdl_run_pool`/`mdl_range_parallel` を再入すると(例:
 *     `mdl_run_pool` のコンテキスト単位並列の中で、その推論関数が
 *     さらに `mdl_preproc_parallel` を呼ぶような入れ子構成)、
 *     直列化ロックを呼び出しスレッド自身が握ったまま待つ形になり
 *     **デッドロックする**。現状のコード生成・テスト・ベンチはこの入れ子を
 *     一切行わない(前処理の並列化と推論の並列化は別々にしか呼ばれない)
 *   - プロセス終了時に明示的な後始末をしなくても壊れない
 *     (`mdl_thread_pool_shutdown` は**任意**。呼ばなければ常駐ワーカは
 *     condvar待ちのままプロセス終了に巻き込まれる — detachedスレッドが
 *     プロセス終了時に残っても良い、という前提を踏襲している)
 *
 * 使い方:
 *     uint32_t work[N][MDL_CTX_BYTES / 4];
 *     mdl_ctx  ctx[N];
 *     void    *ctxp[N];
 *     for (i = 0; i < N; i++) {
 *       mdl_ctx_init(&ctx[i], work[i], MDL_CTX_BYTES);
 *       ctxp[i] = &ctx[i];
 *     }
 *     mdl_run_pool(mdl_infer_ctx_v, ctxp, N,
 *                  inputs, MDL_INPUT_WIDTH,
 *                  scores, MDL_OUTPUT_WIDTH * sizeof(int32_t), n_images);
 *
 * ビルド: `cc ... mdl_threads.c -lpthread`(POSIX)。
 *
 * このファイル(`mdl_threads.c`)は **pthread専用でWindows非対応**。
 * `mdl_kernels.c` + 生成モデル本体はlibc最小依存でWindows向けにも
 * オブジェクト生成まで通ることを確認済み(`tests/deploy/test_c_portability.py`
 * の `x86_64-pc-windows-msvc` / `x86_64-w64-windows-gnu`)だが、それは
 * **単一コンテキストでの単発推論**の話であり、この並列ヘルパは別物。
 * Windowsで並列に呼びたい場合は、今のところ:
 *   (a) 呼び出し側で独自にスレッドを立てる(`mdl_infer_ctx` はコンテキストが
 *       別なら再入可能なので、Win32 `CreateThread` / `std::thread` から
 *       直接呼んで構わない。このファイルは経由しなくてよい)
 *   (b) 将来 win32 実装を足すなら、`CreateThread` + `WaitForMultipleObjects`
 *       でこのヘッダの関数シグネチャ(`mdl_run_pool` 等)をそのまま実装する
 *       `mdl_threads_win32.c` を追加する形が自然(公開APIは変えずに済む)。
 * (b) は実行検証環境が無いため未実装(設計メモのみ)。
 */
#ifndef MDL_THREADS_H
#define MDL_THREADS_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/** 1枚ぶんの推論。生成モデルの `mdl_infer_ctx_v` /
 *  `mdl_infer_image_ctx_v` をそのまま渡せる形にしてある
 *  (関数ポインタのキャストは未定義動作なので、void* 版を用意している)。 */
typedef int (*mdl_infer_fn)(void *ctx, const uint8_t *input, int32_t *scores);

/** n_items 枚を n_ctx 個のコンテキストで分担して処理する。
 *
 *  各ワーカは自分専用のコンテキストしか触らないので、モデルの重みだけが
 *  共有される(読み取り専用)。**結果は逐次実行と完全に一致する**
 *  — 枚ごとに独立で、順序にも依存しないため。
 *
 *  @param fn        1枚ぶんの推論関数
 *  @param ctxs      コンテキストの配列(要素数 n_ctx)。スレッド数もこれで決まる
 *  @param inputs    入力を並べた領域。i枚目は inputs + i * in_stride
 *  @param in_stride 1枚ぶんの入力バイト数
 *  @param scores    出力を並べた領域。i枚目は scores + i * out_stride バイト
 *  @param out_stride 1枚ぶんの出力バイト数
 *  @param n_items   枚数
 *  @return 0 で成功。どれか1枚でも fn が非0を返したら最初の非0値
 *          (スレッド生成に失敗したときは、その場から逐次実行へ落とす)
 */
int mdl_run_pool(mdl_infer_fn fn, void **ctxs, int n_ctx,
                 const uint8_t *inputs, size_t in_stride,
                 int32_t *scores, size_t out_stride, int n_items);

/** 使えるコア数の目安(取得できなければ1)。コンテキストを何本作るかの参考に。 */
int mdl_hw_threads(void);

/** 「index i(0..n-1) を start=own, stride=n_workers のストライドで担当する」
 *  形の並列化を行う関数。呼ばれ方は `fn(ctx, start, stride)`(自分の担当は
 *  start, start+stride, start+2*stride, ... < n)。 */
typedef int (*mdl_index_fn)(void *ctx, int start, int stride);

/** **1つの仕事をNコアへストライド分割**する汎用プリミティブ(* 元は前処理チャネル並列専用だった`mdl_preproc_parallel`のロジックを
 *  汎用化して切り出したもの)。
 *
 *  `mdl_run_pool` が「N枚をNコアへ」(枚ごとに独立)なのに対し、こちらは
 *  「1つの仕事をNコアへ」割る。ストライド分割にしてあるのは、費用が
 *  偏っている系列(例: 前処理チャネルは種別ごとに費用が18倍違う)でも、
 *  連続ブロック分割だと最後のワーカへ高価な種別が集中するのを避けるため
 *  (飛び飛びに割れば均等にばらける)。
 *
 *  @param fn        1ストライドぶんの仕事
 *  @param ctx       fnへそのまま渡す不透明なコンテキスト
 *  @param n         仕事の総数(例: チャネル数、系列長)
 *  @param n_threads 分割数の上限
 *  @return 0 で成功。どれかのワーカが非0を返したら最初の非0値
 */
int mdl_range_parallel(mdl_index_fn fn, void *ctx, int n, int n_threads);

/** 前処理の担当チャネル(start から stride 飛ばし)を処理する関数。
 *  生成モデルの `mdl_preproc_range_v` をそのまま渡せる。 */
typedef int (*mdl_range_fn)(void *ctx, const uint8_t *image, int start,
                            int stride);

/** **1枚のレイテンシを詰める**ための並列化。前処理のチャネルを n_threads へ
 *  等分して同時に走らせる(`mdl_range_parallel`の薄いラッパ)。
 *
 *  `mdl_run_pool` が「N枚をNコアへ」なのに対し、こちらは「1枚をNコアへ」。
 *  画像が大きいモデルでは推論時間の大半が前処理で(detface は実測で76%)、
 *  しかもチャネル同士は完全に独立なので、ここだけ割るのが一番効く。
 *  出力バッファは1つでよい(チャネル範囲が重ならない)。
 *
 *      mdl_preproc_parallel(mdl_preproc_range_v, &ctx, image,
 *                           MDL_PREPROC_CHANNELS, mdl_hw_threads());
 *      mdl_run_ctx(&ctx, scores);
 */
int mdl_preproc_parallel(mdl_range_fn fn, void *ctx, const uint8_t *image,
                         int channels, int n_threads);

/** 常駐ワーカプール(`mdl_run_pool`/`mdl_range_parallel`が内部で使う
 *  シングルトン)を明示的に畳む。**任意**——呼ばなくてもプロセス終了時に
 *  壊れることはない(常駐スレッドはcondvar待ちのままプロセス終了に
 *  巻き込まれる)。長寿命プロセスでスレッド数を頻繁に変えたあと、
 *  待機中のワーカを能動的に片付けたい場合や、valgrind等で
 *  「アプリ終了時に生きているスレッド」を消しておきたい場合に使う。
 *  次に `mdl_run_pool` 等を呼べば、通常どおりプールが再生成される。
 *  戻り値は常に0。 */
int mdl_thread_pool_shutdown(void);

#ifdef __cplusplus
}
#endif

#endif /* MDL_THREADS_H */
