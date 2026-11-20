/* mdl_stdio.h — ホスト側ドライバ用の任意ヘルパ: stdin/stdout をバイナリモードにする。
 *
 * ランタイム本体(mdl_kernels.c / mdl_rddlgn.c / 生成モデル)は stdio を一切使わない。
 * これは「生のバイト列を stdin から読み、整数をそのまま stdout に書く」ホスト
 * プログラム(`griffith deploy verify-c` が生成するドライバ、tests/deploy の main)
 * のためのもので、フリースタンディング契約(mdl_kernels.h は <stdint.h>/<stddef.h>
 * のみ)の対象外。
 *
 * Windows の C ランタイムは stdin/stdout を既定でテキストモードで開くため、
 * 出力の 0x0A が 0x0D 0x0A に化け、入力では 0x0D 0x0A が 0x0A に縮み 0x1A が EOF
 * 扱いになる。POSIX にはテキスト/バイナリの区別が無いので何もしない。
 *
 * 使い方: main の先頭(最初の fread/fwrite より前)で MDL_STDIO_BINARY(); を呼ぶ。
 */
#ifndef MDL_STDIO_H
#define MDL_STDIO_H

#if defined(_WIN32)
#include <fcntl.h>
#include <io.h>
#include <stdio.h>
#define MDL_STDIO_BINARY()                       \
  do {                                           \
    _setmode(_fileno(stdin), _O_BINARY);         \
    _setmode(_fileno(stdout), _O_BINARY);        \
  } while (0)
#else
#define MDL_STDIO_BINARY() ((void)0)
#endif

#endif /* MDL_STDIO_H */
