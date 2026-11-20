/* griffith-logic — 移植可能なSIMD薄層
 *
 * 方針:
 *   - **配布物は分けない**。同じソースがコンパイル時に NEON / AVX2 / SSE2 /
 *     スカラーへ振り分けられる。アーキ固有コードは全部このヘッダの中に
 *     閉じ込める
 *   - **整数出力は経路によらず完全に同一**。SIMD版とスカラー版が1ビットでも
 *     違ってはならない(テストで両方を突き合わせる)
 *   - 手で無効化できる: -DMDL_NO_SIMD(全経路をスカラーへ)、
 *     -DMDL_NO_AVX2 または -mno-avx2(AVX2だけ止めてSSE2へ、A/B比較用)
 *
 * 提供するもの:
 *   MDL_HAS_SIMD          1 ならベクタ経路が使える
 *   mdl_ge_mask16(p, k)   u8 16個を読み「>= k」の16ビットマスクを返す(符号なし比較)
 *   mdl_lut_mask16(p, l)  u8 16個を読み 256bitLUT を引いた16ビットマスクを返す
 *   mdl_mask32(p)         u8 32個を読み「!= 0」の32ビットマスクを返す
 *
 *(x86 AVX2対応): ビットスライスLUT層(mdl_lut_layer_slice/packed)と
 * バッチパック(mdl_pack_batch)の勝ち筋は、どちらも `mdl_mask32` —
 * 32バイトの0/1列を32ビットへ詰めるだけの操作だった。SSE2はmovemaskが
 * 16バイトぶんしか無いので `mdl_ne_zero_mask16` を2回叩いて32ビットへ
 * 合成していたが、AVX2の `_mm256_movemask_epi8` は32バイト→32ビットを
 * 1命令で返す。ここが最有力の伸びしろなので、AVX2ではmdl_mask32を
 * 2回合成ではなく直接実装へ差し替える(下のMDL_SIMD_AVX2ブロック)。
 *
 * 一方、`mdl_pair_mask16`/`mdl_tbl256`(256エントリ表のギャザー、EDGE/
 * CHROMA/SATの前処理が使う)はNEON専用のままで、SSE2には実装を足して
 * いない(SSE2はレジスタ幅16バイトしか無く、下のAVX2版をそのまま
 * 半幅に落としても回数が倍になるだけで割に合わない)。
 * 同じ理由で `mdl_lut_mask16`(256ビットLUT参照)もスカラーのまま
 * (ヘッダ既存コメントの通り、単調な階段に畳めない稀なチャネルにしか
 * 落ちない)。
 *
 *(前処理のAVX2化): det_faceのCPUレーンはプロファイルで前処理が
 * 支配項(実測 x86 Xeon W-1270P で e2e の72%程度、うちEDGE/CHROMA/SATの
 * `mdl_tbl256`ギャザーだけで前処理時間の約72%=EDGE 52%+CHROMA 12%+
 * SAT 8%、CURVは今回スコープ外で未着手のまま残る)。ここは前段の
 *(前半/後半)時点ではSSE2側ゼロ実装(=スカラー)のまま見送られていた。
 * 「256エントリ ÷ 16(pshufbはレーン内16エントリしか引けない)」という
 * 上のNEON比較メモの見積り通り、AVX2は16回のシャッフル+範囲比較+ORが
 * 要る(`mdl_tbl256_avx2_get`)。**ただしNEONとの比較ではなく現行の
 * x86スカラーとの比較でだけ勝てばよい**ので、実装してA/B実測した:
 *
 *   - 候補A(採用): `_mm256_shuffle_epi8`を16チャンク回し、各チャンクで
 *     `(b - 16k) & 0xF0 == 0` の範囲マスクを掛けてOR合成
 *     (`mdl_tbl256_avx2_get`)。**落とし穴**: `vpshufb`は添字バイトの
 *     bit7が立っていると無条件に0を返す仕様があり(NEONの
 *     `vqtbl4q_u8`のような「表サイズを超えたら0」ではなく「bit7で0」)、
 *     添字をそのまま渡すと b>=128 のチャンクが軒並み消える
 *     (256要素中128要素が壊れる、実測で全ブロックの約半数が不一致)。
 *     シャッフルへ渡す添字は事前に下位4bitへマスクし(`b & 0x0F`)、
 *     範囲判定は別変数(マスクしていない `diff`)で行うことで解決。
 *     単体マイクロベンチ(1Mピクセル、饱和あり/なし・診断的な単調表
 *     3種、ベストオブ7)でスカラー比 **2.2〜3.25倍**、全パターンで
 *     スカラーとビット完全一致。
 *   - 候補B(不採用): `_mm256_i32gather_epi32`によるdword単位ギャザー
 *     (バイト表をint32へ展開し、32添字を4回のgatherで捌く)。
 *     同じマイクロベンチで**スカラー比0.47〜0.67倍(スカラーより遅い)**
 *     — Comet Lake(Skylakeクライアント系列)のgatherは各要素を
 *     ほぼ逐次的にしか処理できず、周辺のパック/アンパック(32バイト→
 *     8要素ずつ4回、結果をbyteへ再パック)のオーバーヘッドも乗るため。
 *     さらに実装time点で早期飽和パターンに対して不一致も残っていた
 *     (未デバッグのまま破棄——速度で負けている時点で正しさを詰める
 *     価値が無い)。**この負の結果は再訪しない**(vpgatherは今回の
 *     ワークロード=1バイトあたり1要素の細粒度ギャザーには向かない)。
 *
 * 候補Aを`mdl_tbl256_avx2`/`mdl_tbl256_avx2_get`/`mdl_pair_mask32_avx2`
 * として採用し、`MDL_HAS_PAIR_GATHER_AVX2`で存在を通知する。既存の
 * `MDL_HAS_PAIR_GATHER`(NEON専用)とは別名にしてあるので、
 * `mdl_kernels.c`側のNEON分岐(`pair_word`のuint8x16_t直書き部分・
 * SATのmax/min部分)はテキストレベルで無改変のまま、AVX2は
 * `#elif`で並記した別ブロックとして追加する。AVX2はレジスタ幅が
 * NEONの2倍(32バイト/命令)なので、`pair_word`側のAVX2ループは
 * 32画素ぶんを1回で処理する(NEONは16画素×2回)。
 *
 * AVX-512(vpternlogd等でgate2_rangeを1命令に畳む、等)は検証機
 * (Xeon W-1270P)にAVX-512が無いため実装しない。将来 AVX-512 が使える
 * 環境が手に入ったら: `mdl_mask32`相当を64バイト単位(`_mm512_movepi8_mask`
 * は無いのでcmpで直接kレジスタを作る)に広げるより先に、
 * gate2_range(mdl_kernels.c)のワードループをvpternlogdで1命令化する方が
 * 見込みが大きい(2入力ゲート16種は3変数の論理式なのでimm8=1命令に
 * 畳める組み合わせが多い)。
 *
 *(後半): 上のAVX-512メモが見込んだ通り、`-mavx2`ビルドでconv系
 * (seg_convdlgn_k32)が再現性ある0.79倍に悪化する主因は gate2_range
 * (mdl_kernels.c、16分岐のワードループ、conv実行時間の支配項)への
 * gccの生半可な自動ベクタ化だった。ここでは明示AVX2ベクタ化で潰す:
 * 「u32ワード列に対する2入力論理演算」の一次プリミティブ(AND/OR/XOR/
 * NOT/ANDNOT の5つ、`mdl_gate2_vec_*`)だけをこのヘッダに置き、
 * 16種のゲート式そのもの(どの組み合わせがどのttか、という分岐構造)は
 * mdl_kernels.c側に残す — mdl_gate2()のスカラー版と対称な構造を保つため
 * (アーキ固有コードはintrinsics止まりで、ゲート意味論はここに漏らさない)。
 * この5プリミティブ+`MDL_GATE2_VEC_WIDTH`+`mdl_gate2_vec`型+
 * `MDL_HAS_GATE2_VEC`を実装すれば、将来NEONへ同じインタフェースを
 * 追加してgate2_range側は無改造のまま横展開できる設計だが、**今回は
 * NEON側を実装しない**(既存NEON実測に影響を与えないため。ヘルパが
 * 無いアーキは`MDL_HAS_GATE2_VEC`が定義されずgate2_rangeは現行の
 * スカラーループへ自動フォールバックする)。
 */
#ifndef MDL_SIMD_H
#define MDL_SIMD_H

#include <stdint.h>

#if !defined(MDL_NO_SIMD)
#  if defined(__ARM_NEON) || defined(__ARM_NEON__)
#    if defined(__aarch64__)
#      define MDL_SIMD_NEON 1
#    endif
#  elif defined(__AVX2__) && !defined(MDL_NO_AVX2)
     /* AVX2はSSE2の上位互換(同一プロセスでSSE2命令も出せる)なので、
        下のSSE2ブロックをそのまま使い回し、mdl_mask32だけを直接実装で
        上書きする(MDL_SIMD_AVX2はMDL_SIMD_SSE2に加えて立てる)。 */
#    define MDL_SIMD_AVX2 1
#    define MDL_SIMD_SSE2 1
#  elif defined(__SSE2__) || (defined(_M_IX86_FP) && _M_IX86_FP >= 2) \
        || defined(_M_X64)
#    define MDL_SIMD_SSE2 1
#  endif
#endif

#if defined(MDL_SIMD_NEON) || defined(MDL_SIMD_SSE2)
#  define MDL_HAS_SIMD 1
#else
#  define MDL_HAS_SIMD 0
#endif

#if defined(MDL_SIMD_NEON)
#include <arm_neon.h>

/* 各レーンが 0x00/0xFF のベクタ → 16ビットのマスク。
 * NEON には movemask が無いのでビット位置を掛けて水平加算する。 */
static inline uint32_t mdl_movemask_u8(uint8x16_t v) {
  static const uint8_t kBits[16] = {1, 2, 4, 8, 16, 32, 64, 128,
                                    1, 2, 4, 8, 16, 32, 64, 128};
  const uint8x16_t masked = vandq_u8(v, vld1q_u8(kBits));
  return (uint32_t)vaddv_u8(vget_low_u8(masked))
       | ((uint32_t)vaddv_u8(vget_high_u8(masked)) << 8);
}

/** p[0..15] >= k(符号なし)の16ビットマスク。 */
static inline uint32_t mdl_ge_mask16(const uint8_t *p, uint8_t k) {
  return mdl_movemask_u8(vcgeq_u8(vld1q_u8(p), vdupq_n_u8(k)));
}

/** p[0..15] != 0 の16ビットマスク(0/1バイト列 → ビット詰め)。 */
static inline uint32_t mdl_ne_zero_mask16(const uint8_t *p) {
  return mdl_movemask_u8(vtstq_u8(vld1q_u8(p), vld1q_u8(p)));
}

/* ---- 256エントリ u8 表のギャザー ----
 * EDGE/CHROMA/SAT は「a >= thr[b]」という形に還元済みで(I(k)の単調性)、
 * 残る費用は **thr[b] の256エントリ表引き**だった(実測 EDGE は
 * thermometer の8.7倍/チャネル)。NEON の `vqtbl4q_u8` は64バイト表しか
 * 引けないので、256エントリを64バイト×4に割り、添字が範囲外なら0を返す
 * 性質を使ってORで合成する(u8の減算が巡回するので範囲外判定が要らない)。 */
typedef struct { uint8x16x4_t q[4]; } mdl_tbl256;

static inline void mdl_tbl256_load(mdl_tbl256 *t, const uint8_t *table) {
  int k;
  for (k = 0; k < 4; k++) t->q[k] = vld1q_u8_x4(table + k * 64);
}

static inline uint8x16_t mdl_tbl256_get(const mdl_tbl256 *t, uint8x16_t b) {
  uint8x16_t r = vqtbl4q_u8(t->q[0], b);
  r = vorrq_u8(r, vqtbl4q_u8(t->q[1], vsubq_u8(b, vdupq_n_u8(64))));
  r = vorrq_u8(r, vqtbl4q_u8(t->q[2], vsubq_u8(b, vdupq_n_u8(128))));
  r = vorrq_u8(r, vqtbl4q_u8(t->q[3], vsubq_u8(b, vdupq_n_u8(192))));
  return r;
}

/** 16画素ぶんの「pa[i] >= thr[pb[i]]」(dir_le なら "<")の16ビットマスク。
 *
 *  thr は [0,256] の値をとるが、**b について単調非減少**である
 *  (`build_pair_threshold` の2ポインタ構成による)。よって thr[b]==256 は
 *  b >= sat_from でまとめて表せ、表そのものは u8 に収まる。
 *  `sat_from` が 256 のときは飽和が起きない(u8では255を渡し、
 *  b<=255 なので比較は常に偽になる)。 */
static inline uint32_t mdl_pair_mask16(const uint8_t *pa, const uint8_t *pb,
                                       const mdl_tbl256 *thr8,
                                       uint8x16_t sat_from, int dir_le) {
  const uint8x16_t b = vld1q_u8(pb);
  const uint8x16_t a = vld1q_u8(pa);
  const uint8x16_t th = mdl_tbl256_get(thr8, b);
  const uint8x16_t sat = vcgeq_u8(b, sat_from);   /* thr[b] == 256 の画素 */
  const uint8x16_t ge = vcgeq_u8(a, th);
  /* dir_le: a < thr。飽和側(thr=256)は常に真。 */
  return mdl_movemask_u8(dir_le ? vorrq_u8(vmvnq_u8(ge), sat)
                                : vbicq_u8(ge, sat));
}

#define MDL_HAS_PAIR_GATHER 1

#elif defined(MDL_SIMD_SSE2)
#include <emmintrin.h>

/** p[0..15] >= k(符号なし)の16ビットマスク。
 *  SSE2 の比較は符号付きしか無いので max_epu8 との一致で符号なし比較を作る。 */
static inline uint32_t mdl_ge_mask16(const uint8_t *p, uint8_t k) {
  const __m128i a = _mm_loadu_si128((const __m128i *)(const void *)p);
  const __m128i kv = _mm_set1_epi8((char)k);
  return (uint32_t)_mm_movemask_epi8(_mm_cmpeq_epi8(_mm_max_epu8(a, kv), a));
}

/** p[0..15] != 0 の16ビットマスク(0/1バイト列 → ビット詰め)。 */
static inline uint32_t mdl_ne_zero_mask16(const uint8_t *p) {
  const __m128i a = _mm_loadu_si128((const __m128i *)(const void *)p);
  const __m128i eq0 = _mm_cmpeq_epi8(a, _mm_setzero_si128());
  return (uint32_t)_mm_movemask_epi8(eq0) ^ 0xffffu;   /* 反転して != 0 に */
}

#if defined(MDL_SIMD_AVX2)
#include <immintrin.h>

/** p[0..31] != 0 の32ビットマスクを1命令ぶんで作る(勝ち筋)。
 *  SSE2版 mdl_ne_zero_mask16 と同じ作り(0との一致を取ってから反転)を
 *  32バイトへそのまま広げただけなので、値は
 *  `mdl_ne_zero_mask16(p) | (mdl_ne_zero_mask16(p+16) << 16)`
 *  と1ビットも変わらない(下のmdl_mask32の既定実装と等価)。
 *  `_mm256_movemask_epi8` はレーン0〜31の符号ビットをそのまま32bit整数の
 *  bit0〜31へ詰めるので、SSE2の16bit版と違って上位ビットのマスクは不要。 */
static inline uint32_t mdl_mask32_avx2(const uint8_t *p) {
  const __m256i a = _mm256_loadu_si256((const __m256i *)(const void *)p);
  const __m256i eq0 = _mm256_cmpeq_epi8(a, _mm256_setzero_si256());
  return (uint32_t)_mm256_movemask_epi8(eq0) ^ 0xffffffffu;
}
/* 下のmdl_mask32本体(全経路共通の場所)に「直接実装がある」ことを伝える。
   NEON/SSE2/スカラーは触らず、AVX2のときだけ2回合成をこちらへ差し替える
   (関数定義そのものを上書きするので、ここではマクロにしない — マクロに
   すると下の `static inline uint32_t mdl_mask32(...)` という定義文自体が
   展開されて壊れる)。 */
#define MDL_MASK32_HAS_DIRECT 1

/* ---- gate2_range(mdl_kernels.c)向け: u32ワード列の2入力論理演算 ----
 * AVX2は256bit=uint32_t 8個ぶんを1命令で読み書きできる。ここに置くのは
 * AND/OR/XOR/NOT/ANDNOTの5つの一次プリミティブだけで、16種のゲート式
 * (どの組み合わせがどのttか)はmdl_kernels.c側の分岐構造がそのまま
 * 呼び出す形にする(ヘッダ冒頭の後半メモ参照)。*/
#define MDL_GATE2_VEC_WIDTH 8   /* 1ベクタで処理する uint32_t の個数 */

typedef __m256i mdl_gate2_vec;

static inline mdl_gate2_vec mdl_gate2_vec_load(const uint32_t *p) {
  return _mm256_loadu_si256((const __m256i *)(const void *)p);
}

static inline void mdl_gate2_vec_store(uint32_t *p, mdl_gate2_vec v) {
  _mm256_storeu_si256((__m256i *)(void *)p, v);
}

static inline mdl_gate2_vec mdl_gate2_vec_and(mdl_gate2_vec x, mdl_gate2_vec y) {
  return _mm256_and_si256(x, y);
}

static inline mdl_gate2_vec mdl_gate2_vec_or(mdl_gate2_vec x, mdl_gate2_vec y) {
  return _mm256_or_si256(x, y);
}

static inline mdl_gate2_vec mdl_gate2_vec_xor(mdl_gate2_vec x, mdl_gate2_vec y) {
  return _mm256_xor_si256(x, y);
}

static inline mdl_gate2_vec mdl_gate2_vec_not(mdl_gate2_vec x) {
  return _mm256_xor_si256(x, _mm256_set1_epi32(-1));
}

/* ~x & y(AVX2の andnot は第1引数を反転してANDするという定義そのまま)。 */
static inline mdl_gate2_vec mdl_gate2_vec_andnot(mdl_gate2_vec x, mdl_gate2_vec y) {
  return _mm256_andnot_si256(x, y);
}

#define MDL_HAS_GATE2_VEC 1

/* ---- 256エントリ u8 表のギャザー(AVX2) ----
 * ヘッダ冒頭のメモ参照。NEON版(`mdl_tbl256`)と同じ「チャンク分割+
 * 範囲マスク+OR合成」だが、レーン内16エントリしか引けない`vpshufb`の
 * 制約で256/16=16チャンク要る(NEONの`vqtbl4q_u8`は64エントリ×4=4回)。 */
typedef struct { __m256i t[16]; } mdl_tbl256_avx2;

static inline void mdl_tbl256_avx2_load(mdl_tbl256_avx2 *t, const uint8_t *table) {
  int k;
  for (k = 0; k < 16; k++) {
    const __m128i lane =
        _mm_loadu_si128((const __m128i *)(const void *)(table + k * 16));
    /* 同じ16バイトチャンクを上下レーンへ複製し、32バイト全体を
       1回の`_mm256_shuffle_epi8`で(レーンごとに独立に)引けるようにする。 */
    t->t[k] = _mm256_broadcastsi128_si256(lane);
  }
}

static inline __m256i mdl_tbl256_avx2_get(const mdl_tbl256_avx2 *t, __m256i b) {
  /* `vpshufb`は添字バイトのbit7が立っていると無条件に0を返す
   * (NEONの`vqtbl4q_u8`のような「表サイズを超えたら0」ではない)。
   * b はそのまま渡さず下位4bitへマスクする(範囲判定は別途 diff で行う)。 */
  const __m256i lo4 = _mm256_and_si256(b, _mm256_set1_epi8(0x0f));
  const __m256i step = _mm256_set1_epi8(16);
  const __m256i hi_mask = _mm256_set1_epi8((char)0xf0);
  __m256i acc = _mm256_setzero_si256();
  __m256i diff = b;
  int k;
  for (k = 0; k < 16; k++) {
    const __m256i shuffled = _mm256_shuffle_epi8(t->t[k], lo4);
    /* diff==(b - 16k) の上位4bitが全て0 ⟺ b が [16k, 16k+16) の範囲内。 */
    const __m256i in_range =
        _mm256_cmpeq_epi8(_mm256_and_si256(diff, hi_mask), _mm256_setzero_si256());
    acc = _mm256_or_si256(acc, _mm256_and_si256(in_range, shuffled));
    diff = _mm256_sub_epi8(diff, step);
  }
  return acc;
}

/** 32画素ぶんの「pa[i] >= thr[pb[i]]」(dir_le なら "<")の32ビットマスク。
 *  NEON版`mdl_pair_mask16`の32レーン版(AVX2は256bit=32バイト/命令なので
 *  1回で32画素処理できる。movemaskも`_mm256_movemask_epi8`が32bit分を
 *  直接返すので、NEON側のようなビット位置合成は不要)。 */
static inline uint32_t mdl_pair_mask32_avx2(const uint8_t *pa, const uint8_t *pb,
                                            const mdl_tbl256_avx2 *thr8,
                                            __m256i sat_from, int dir_le) {
  const __m256i b = _mm256_loadu_si256((const __m256i *)(const void *)pb);
  const __m256i a = _mm256_loadu_si256((const __m256i *)(const void *)pa);
  const __m256i th = mdl_tbl256_avx2_get(thr8, b);
  const __m256i sat = _mm256_cmpeq_epi8(_mm256_max_epu8(b, sat_from), b); /* b>=sat_from(符号なし) */
  const __m256i ge = _mm256_cmpeq_epi8(_mm256_max_epu8(a, th), a);       /* a>=th(符号なし) */
  const __m256i result = dir_le
      ? _mm256_or_si256(_mm256_xor_si256(ge, _mm256_set1_epi8(-1)), sat) /* ~ge | sat */
      : _mm256_andnot_si256(sat, ge);                                    /* ge & ~sat */
  return (uint32_t)_mm256_movemask_epi8(result);
}

/** 32画素ぶんの「a >= thr[b]」(dir_le なら "<")の32ビットマスク。
 *  飽和が一度も起きない表(`sat_from > 255`)専用: sat_from を u8 に
 *  収められないので、上のmdl_pair_mask32_avx2は使えない
 *  (`mdl_kernels.c`側`pair_word`のNEON分岐と対称な特殊ケース)。 */
static inline uint32_t mdl_pair_mask32_nosat_avx2(const uint8_t *pa, const uint8_t *pb,
                                                   const mdl_tbl256_avx2 *thr8,
                                                   int dir_le) {
  const __m256i b = _mm256_loadu_si256((const __m256i *)(const void *)pb);
  const __m256i a = _mm256_loadu_si256((const __m256i *)(const void *)pa);
  const __m256i th = mdl_tbl256_avx2_get(thr8, b);
  const __m256i ge = _mm256_cmpeq_epi8(_mm256_max_epu8(a, th), a);
  const __m256i result = dir_le ? _mm256_xor_si256(ge, _mm256_set1_epi8(-1)) : ge;
  return (uint32_t)_mm256_movemask_epi8(result);
}

#define MDL_HAS_PAIR_GATHER_AVX2 1

#endif /* MDL_SIMD_AVX2 */

#else

/** スカラー版(SIMDが無い環境。値はベクタ版と完全に同一)。 */
static inline uint32_t mdl_ge_mask16(const uint8_t *p, uint8_t k) {
  uint32_t m = 0u;
  int i;
  for (i = 0; i < 16; i++) {
    if (p[i] >= k) m |= (uint32_t)1 << i;
  }
  return m;
}

/** p[0..15] != 0 の16ビットマスク(値はベクタ版と完全に同一)。 */
static inline uint32_t mdl_ne_zero_mask16(const uint8_t *p) {
  uint32_t m = 0u;
  int i;
  for (i = 0; i < 16; i++) {
    if (p[i] != 0) m |= (uint32_t)1 << i;
  }
  return m;
}

#endif

/** 256ビットLUT(8ワード)を引いた16ビットマスク。
 *  テーブル参照はベクタ化しにくい(NEONのgatherは16要素までで256entryに届かない)
 *  ので全経路でスカラー。実モデルでは単調な thermometer がほぼ全て
 *  「k >= k0」へ畳まれるため、この経路に落ちるチャネルはまれ。 */
static inline uint32_t mdl_lut_mask16(const uint8_t *p, const int32_t *lut) {
  uint32_t m = 0u;
  int i;
  for (i = 0; i < 16; i++) {
    const uint32_t k = p[i];
    m |= (((uint32_t)lut[k >> 5] >> (k & 31u)) & 1u) << i;
  }
  return m;
}

/** 0/1 のバイト32個を1ワードへ詰める(LSBが先頭)。
 *  16ビットぶんは各アーキの movemask 相当へ落ちる(NEON/SSE2/スカラー)。
 *  AVX2だけは32バイトを1命令で処理する直接実装があるので、そちらを使う
 *  (`MDL_MASK32_HAS_DIRECT`、上のAVX2ブロック参照)。値はどちらの経路でも
 *  1ビットも変わらない。 */
#if defined(MDL_MASK32_HAS_DIRECT)
static inline uint32_t mdl_mask32(const uint8_t *p) {
  return mdl_mask32_avx2(p);
}
#else
static inline uint32_t mdl_mask32(const uint8_t *p) {
  return mdl_ne_zero_mask16(p) | (mdl_ne_zero_mask16(p + 16) << 16);
}
#endif

#endif /* MDL_SIMD_H */
