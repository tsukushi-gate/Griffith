/* ESP32 実機デモ: デプロイキット(codegen-c)の生成モデルを同梱サンプルで推論し、
 * 精度とレイテンシをUARTへ出す。ホストでも同じソースがビルド・実行できる。
 *
 *   実機: idf.py build / flash(README.md)
 *   ホスト: cc -std=c99 -O2 -DHOST_BUILD -DMDL_NO_DEFAULT_CTX -Imain -Imain/model \
 *             main/main.c main/model/model.c main/model/mdl_kernels.c -o host_demo
 *
 * 出力は2種類。
 *   SAMPLE i pred=.. exp=.. ns=.. h=XXXXXXXX [scores=..]  サンプルごと(整数一致の照合用)
 *   RESULT {...}                                          機械可読の要約(1行JSON)
 * 最後に DONE を出して止まる(HOST_BUILD は return 0)。
 *
 * ヒープ確保なし: 作業RAMは static uint32_t work[MDL_CTX_BYTES/4]。
 * 浮動小数点なし(時間・精度とも整数の固定小数で出力する)。
 */
/* ホストの -std=c99 では glibc が clock_gettime/CLOCK_MONOTONIC を隠すので POSIX を明示する
 * (macOS の libc は既定で見せるため、Linux の gcc でだけ落ちていた) */
#if !defined(ESP_PLATFORM) && !defined(_POSIX_C_SOURCE)
#define _POSIX_C_SOURCE 199309L
#endif
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "mdl_kernels.h"
#include "model.h"
#include "samples.h"

#if SAMPLE_INPUT_WIDTH != MDL_INPUT_WIDTH || SAMPLE_OUTPUT_WIDTH != MDL_OUTPUT_WIDTH
#error "samples.h と model.h の入出力幅が一致しない: gen_model.py を同じバンドルで再実行すること"
#endif

#ifdef ESP_PLATFORM
#include "esp_timer.h"
#include "esp_system.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "sdkconfig.h"
#define PLATFORM_NAME "esp32"
static uint64_t now_ns(void) { return (uint64_t)esp_timer_get_time() * 1000u; }
static void yield_task(void) { vTaskDelay(1); }
#else
#include <time.h>
#define PLATFORM_NAME "host"
static uint64_t now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000u + (uint64_t)ts.tv_nsec;
}
static void yield_task(void) {}
#endif

/* 1サンプルあたりの計測は「合計時間 ≥ MIN_WINDOW_NS になるまで繰り返す」 */
#define MIN_WINDOW_NS 5000000u /* 5 ms */
#define MAX_REPS 5000u

static uint32_t work[MDL_CTX_BYTES / 4]; /* 4バイト整列の static 作業RAM */
static mdl_ctx ctx;
#if !SAMPLE_INPUT_IS_IMAGE
static uint8_t input_bits[MDL_INPUT_WIDTH]; /* 0/1 に展開した入力 */
#endif
static int32_t scores[MDL_OUTPUT_WIDTH];
static uint32_t sample_ns[SAMPLE_COUNT]; /* 1推論あたり ns */
static uint32_t sorted_ns[SAMPLE_COUNT];

#if SAMPLE_INPUT_IS_IMAGE
/* 画像入力モデル: SAMPLE_BITS は u8 NCHW 画像そのもの。展開バッファは持たない */
#define unpack_input(i) ((void)(i))
#define INFER(c, s) mdl_infer_image_ctx((c), SAMPLE_BITS[i], (s))
#else
static void unpack_input(int i) {
    for (int b = 0; b < MDL_INPUT_WIDTH; b++)
        input_bits[b] = (SAMPLE_BITS[i][b >> 3] >> (b & 7)) & 1u;
}
#define INFER(c, s) mdl_infer_ctx((c), input_bits, (s))
#endif

static uint32_t fnv1a(uint32_t h, const int32_t *v, int n) {
    for (int i = 0; i < n; i++) {
        uint32_t x = (uint32_t)v[i];
        for (int k = 0; k < 4; k++) {
            h ^= (x >> (8 * k)) & 0xffu;
            h *= 16777619u;
        }
    }
    return h;
}

/* 固定小数点: ns → "us.mmm" の整数2つ */
#define US_INT(ns) ((unsigned)((ns) / 1000u))
#define US_FRAC(ns) ((unsigned)((ns) % 1000u))

static void sort_u32(uint32_t *a, int n) { /* 挿入ソート(Nは小さい) */
    for (int i = 1; i < n; i++) {
        uint32_t v = a[i];
        int j = i - 1;
        while (j >= 0 && a[j] > v) { a[j + 1] = a[j]; j--; }
        a[j + 1] = v;
    }
}

static int run_demo(void) {
    int n_labeled = 0, correct = 0, score_checked = 0, score_mismatch = 0;
    uint32_t total_hash = 2166136261u;
    uint64_t sum_ns = 0;

    if (mdl_ctx_init(&ctx, work, sizeof(work)) != 0) {
        printf("ERROR mdl_ctx_init failed\n");
        return 1;
    }

    printf("== model=%s platform=%s samples=%d input=%d output=%d ctx=%dB gates=%d ==\n",
           MODEL_NAME, PLATFORM_NAME, SAMPLE_COUNT, MDL_INPUT_WIDTH,
           MDL_OUTPUT_WIDTH, MDL_CTX_BYTES, MODEL_GATES);

    for (int i = 0; i < SAMPLE_COUNT; i++) {
        unpack_input(i);

        /* 1回目: 結果の取得 + 反復回数の見積り */
        uint64_t t0 = now_ns();
        int rc = INFER(&ctx, scores);
        uint64_t one = now_ns() - t0;
        if (rc != 0) {
            printf("ERROR mdl_infer_ctx rc=%d sample=%d\n", rc, i);
            return 1;
        }
        uint32_t reps = (one == 0) ? MAX_REPS : (uint32_t)(MIN_WINDOW_NS / one);
        if (reps < 1u) reps = 1u;
        if (reps > MAX_REPS) reps = MAX_REPS;

        uint64_t t1 = now_ns();
        for (uint32_t r = 0; r < reps; r++) INFER(&ctx, scores);
        uint64_t per = (now_ns() - t1) / reps;
        sample_ns[i] = (uint32_t)per;
        sum_ns += per;

        int pred = -1;
        int want = SAMPLE_EXPECTED[i];
        if (want >= 0) {
            pred = (int)mdl_argmax(scores, MDL_OUTPUT_WIDTH);
            n_labeled++;
            if (pred == want) correct++;
        }
        uint32_t h = fnv1a(2166136261u, scores, MDL_OUTPUT_WIDTH);
        total_hash = fnv1a(total_hash, scores, MDL_OUTPUT_WIDTH);
#if SAMPLE_HAS_SCORES
        score_checked++;
        if (memcmp(scores, SAMPLE_EXPECTED_SCORES[i], sizeof(scores)) != 0) score_mismatch++;
#endif
        printf("SAMPLE %d pred=%d exp=%d ns=%u reps=%u h=%08x", i, pred, want,
               (unsigned)per, (unsigned)reps, (unsigned)h);
        if (MDL_OUTPUT_WIDTH <= 32) {
            printf(" scores=");
            for (int k = 0; k < MDL_OUTPUT_WIDTH; k++)
                printf(k ? ",%d" : "%d", (int)scores[k]);
        }
        printf("\n");
        yield_task();
    }

    uint32_t *sorted = sorted_ns;
    memcpy(sorted, sample_ns, sizeof(sorted_ns));
    sort_u32(sorted, SAMPLE_COUNT);
    uint32_t mean_ns = (uint32_t)(sum_ns / SAMPLE_COUNT);
    uint32_t med_ns = (SAMPLE_COUNT & 1)
        ? sorted[SAMPLE_COUNT / 2]
        : (sorted[SAMPLE_COUNT / 2 - 1] + sorted[SAMPLE_COUNT / 2]) / 2u;
    uint32_t min_ns = sorted[0], max_ns = sorted[SAMPLE_COUNT - 1];
    unsigned acc_pm = n_labeled ? (unsigned)((correct * 10000u) / (unsigned)n_labeled) : 0u;

    printf("+-----------+------------+\n");
    printf("| accuracy  | %d/%d (%u.%02u%%)\n", correct, n_labeled, acc_pm / 100, acc_pm % 100);
    printf("| mean us   | %u.%03u\n", US_INT(mean_ns), US_FRAC(mean_ns));
    printf("| median us | %u.%03u\n", US_INT(med_ns), US_FRAC(med_ns));
    printf("| min us    | %u.%03u\n", US_INT(min_ns), US_FRAC(min_ns));
    printf("| max us    | %u.%03u\n", US_INT(max_ns), US_FRAC(max_ns));
    printf("| ctx bytes | %d\n", MDL_CTX_BYTES);
    printf("+-----------+------------+\n");

    printf("RESULT {\"model\":\"%s\",\"platform\":\"%s\",\"n\":%d,\"n_labeled\":%d,"
           "\"correct\":%d,\"acc\":%u.%04u,\"score_checked\":%d,\"score_mismatch\":%d,"
           "\"mean_us\":%u.%03u,\"median_us\":%u.%03u,\"min_us\":%u.%03u,\"max_us\":%u.%03u,"
           "\"ctx_bytes\":%d,\"input_width\":%d,\"output_width\":%d,\"gates\":%d,"
           "\"score_hash\":\"%08x\"",
           MODEL_NAME, PLATFORM_NAME, SAMPLE_COUNT, n_labeled, correct,
           acc_pm / 10000, acc_pm % 10000, score_checked, score_mismatch,
           US_INT(mean_ns), US_FRAC(mean_ns), US_INT(med_ns), US_FRAC(med_ns),
           US_INT(min_ns), US_FRAC(min_ns), US_INT(max_ns), US_FRAC(max_ns),
           MDL_CTX_BYTES, MDL_INPUT_WIDTH, MDL_OUTPUT_WIDTH, MODEL_GATES,
           (unsigned)total_hash);
#ifdef ESP_PLATFORM
    printf(",\"cpu_mhz\":%d,\"free_heap\":%u,\"min_free_stack_bytes\":%u",
           CONFIG_ESP_DEFAULT_CPU_FREQ_MHZ, (unsigned)esp_get_free_heap_size(),
           (unsigned)uxTaskGetStackHighWaterMark(NULL));
#endif
    printf("}\n");
    printf("DONE\n");
#if SAMPLE_HAS_SCORES
    /* 参照スコアがあるときは整数一致だけを合否にする(実モデルの精度は100%でない) */
    return score_mismatch == 0 ? 0 : 2;
#else
    return correct == n_labeled ? 0 : 2;
#endif
}

#ifdef ESP_PLATFORM
void app_main(void) {
    /* 電源投入直後のUART取りこぼし防止に少し待つ */
    vTaskDelay(pdMS_TO_TICKS(500));
    run_demo();
    /* 要約後は停止。キャプチャ側が再リセットしたときだけ再実行される */
    for (;;) vTaskDelay(pdMS_TO_TICKS(10000));
}
#else
int main(void) {
    return run_demo();
}
#endif
