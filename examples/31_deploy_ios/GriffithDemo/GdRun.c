/* 推論の時間計測ヘルパ。推論本体は生成Cそのまま(モデル関数ポインタ経由)。 */
#include <time.h>

#include "GdModel.h"

static uint64_t gd_run_reps(const gd_model *m, void *state, const uint8_t *bits,
                            int32_t *scores, uint64_t reps, int *rc) {
  const uint64_t t0 = clock_gettime_nsec_np(CLOCK_UPTIME_RAW);
  for (uint64_t i = 0; i < reps; i++) {
    if (m->ctx_infer(state, bits, scores) != 0) { *rc = -1; break; }
  }
  return clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - t0;
}

double gd_time_inference(const gd_model *m, void *state, const uint8_t *bits,
                         int32_t *scores, uint64_t target_ns, uint64_t *reps_out) {
  int rc = 0;
  uint64_t reps = 1;
  uint64_t el;
  (void)gd_run_reps(m, state, bits, scores, 1, &rc);   /* ウォームアップ */
  for (;;) {
    el = gd_run_reps(m, state, bits, scores, reps, &rc);
    if (rc != 0) return -1.0;
    if (el >= target_ns || reps >= (1ull << 22)) break;
    reps <<= 1;
  }
  uint64_t best = el;
  for (int k = 0; k < 2; k++) {
    el = gd_run_reps(m, state, bits, scores, reps, &rc);
    if (el < best) best = el;
  }
  if (rc != 0) return -1.0;
  if (reps_out) *reps_out = reps;
  return (double)best / (double)reps;
}
