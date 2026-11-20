/* デモアプリ共通のモデル記述子。Swift(bridging header 経由)から使う。
 *
 * 各モデルは Models/<name>/<name>_unit.c(gen_model.py が生成)が
 * `const gd_model gd_model_<name>` を1つ公開し、Models/gd_registry.c が一覧にする。
 * model.c の mdl_* シンボルは unit 内で <name>_mdl_* に置換済みなので、
 * 何個モデルを同梱してもリンク時に衝突しない。 */
#ifndef GD_MODEL_H
#define GD_MODEL_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct gd_model {
  const char *name;
  int input_width;    /* MDL_INPUT_WIDTH: 推論器が受ける入力ビット数(1バイト1ビット) */
  int output_width;   /* MDL_OUTPUT_WIDTH: スコア数 */
  int ctx_bytes;      /* MDL_CTX_BYTES: 作業RAM(呼び出し側が確保) */
  int num_labels;     /* MDL_NUM_LABELS(無ければ 0) */
  int num_gates;      /* ゲート数(LUT段の出力幅合計 + conv2d 縮約後) */
  int input_kind;     /* 0=bits(1バイト1ビットの0/1配列) / 1=image_u8(u8 NCHW 画像) */
  int input_bytes;    /* ctx_infer に渡す入力のバイト数(bits: input_width / image: C*H*W) */
  size_t state_size;  /* sizeof(mdl_ctx) */
  int (*ctx_init)(void *state, void *work, size_t bytes);
  int (*ctx_infer)(void *state, const uint8_t *bits, int32_t *scores);
} gd_model;

int gd_model_count(void);
const gd_model *gd_model_at(int i);

/* 同じ入力を何度も推論して 1 推論あたりの ns を返す(最小値採用)。
 * 経過が target_ns を超えるまで回数を倍々にして決め、決めた回数で 3 回測って最小を返す。
 * 時計は clock_gettime_nsec_np(CLOCK_UPTIME_RAW)。scores には最後の推論結果が残る。
 * 失敗時は負値。 */
double gd_time_inference(const gd_model *m, void *state, const uint8_t *bits,
                         int32_t *scores, uint64_t target_ns, uint64_t *reps_out);

#ifdef __cplusplus
}
#endif

#endif
