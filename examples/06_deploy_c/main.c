/* デプロイキット例(2/2): 生成された model.h / model.c を使うCドライバ。
 *
 * XOR(2入力2出力)の4パターンを推論し、期待値と突き合わせる。
 * runtime/c/README.md の「API」節にある mdl_infer() をそのまま呼ぶだけで、
 * ヒープ確保・浮動小数点は一切使わない。
 *
 * ビルド(examples/06_deploy_c/README.md の手順どおりコード生成した後):
 *   cc -std=c99 -O2 -I out_examples/xor_c \
 *      out_examples/xor_c/model.c out_examples/xor_c/mdl_kernels.c \
 *      examples/06_deploy_c/main.c -o out_examples/xor_c/xor_demo
 *   ./out_examples/xor_c/xor_demo
 */
#include <stdio.h>

#include "model.h"

static int argmax2(const int32_t *scores) {
    return scores[1] > scores[0] ? 1 : 0;
}

int main(void) {
    uint8_t inputs[4][MDL_INPUT_WIDTH] = {
        {0, 0}, {0, 1}, {1, 0}, {1, 1},
    };
    int expected[4] = {0, 1, 1, 0};
    int32_t scores[MDL_OUTPUT_WIDTH];
    int all_ok = 1;

    for (int i = 0; i < 4; i++) {
        /* 既定コンテキスト版(単一スレッド向け)。作業RAMは自分で持ちたい
         * 場合は mdl_ctx_init() + mdl_infer_ctx() を使う(README参照)。 */
        mdl_infer(inputs[i], scores);
        int pred = argmax2(scores);
        int ok = (pred == expected[i]);
        all_ok &= ok;
        printf("xor(%d,%d) = %d (scores=%d,%d) expected=%d %s\n",
               inputs[i][0], inputs[i][1], pred, scores[0], scores[1],
               expected[i], ok ? "OK" : "NG");
    }
    if (!all_ok) {
        fprintf(stderr, "不一致あり\n");
        return 1;
    }
    printf("全4パターン一致\n");
    return 0;
}
