/* griffith-logic — 複数コアで回すための任意ヘルパ(常駐スレッドプール)。
 *
 * ここだけが pthread に依存する。**コア(mdl_kernels.c と生成モデル)は
 * このファイルを知らない**ので、ベアメタルではリンクしなければよい。
 *
 * 仕事の取り方は「共有カウンタから1枚ずつ奪う」動的分配。静的に等分すると、
 * 画像によって命令列の実行時間が揺れたとき(早期の write_bucket など)に
 * 最も遅いワーカを待つことになる。1枚が数百µs〜数msなので、
 * ロック1回ぶんの費用は無視できる。
 *
 * ----: 常駐ワーカプール ----------------------------------------------
 *
 * 背景: 以前は `mdl_run_pool` / `mdl_range_parallel` が呼ばれるたびに
 * pthread_create していた。1推論が数百µs〜数msの用途では無視できる比率
 * だが、pthread_create の起床コスト自体が実測19〜53µs(残件)なので、
 * バッチが小さい・呼び出し頻度が高い用途では支配項になりうる。
 *
 * ここでは「タスクを条件変数で待つ常駐スレッド」のプールに置き換える。
 * 設計判断:
 *
 *   - **プロセス内シングルトン**(`g_pool`、static)。初回呼び出しで
 *     lazy生成する。単体テスト・生成コードのどちらも呼び出し元は
 *     1スレッドから逐次に `mdl_run_pool` / `mdl_range_parallel` /
 *     `mdl_preproc_parallel` を呼ぶ構成しか無い(bench_deploy.py の
 *     driver.c 参照: 前処理の並列化・推論の並列化は同じmain()内で
 *     順番に呼ばれるだけで、互いの中から再入することは無い)ので、
 *     プール自体は「1回に1呼び出し」を前提にしてよい。呼び出し全体
 *     (プールの起こし/作り直し判断 〜 ワーカの完了待ち)を
 *     `g_call_lock` で直列化することで、複数スレッドから同時に
 *     呼ばれても(遅くはなるが)壊れないようにしてある。
 *     **ただしワーカスレッド自身の中からの再入は直列化ロックの
 *     自己デッドロックになる**(mdl_threads.h の説明を参照) —
 *     既存の呼び出しパターンはこれを行わない。
 *   - **同じスレッド数が続く限り再利用**する。呼び出しごとのスレッド数
 *     (n_ctx-1 本、または n_threads-1 本)が変わったら、既存のワーカを
 *     join してから作り直す(「同一Nなら再利用。Nが変わったら作り直して
 *     よい」という設計指示のとおり。ベンチはN∈{1,2,4,8}のように離散
 *     少数の値を掃くだけなので、遷移のたびの作り直しコストは無視できる)。
 *   - `mdl_run_pool` の「1本目は呼び出しスレッド自身が担当する」・
 *     `mdl_range_parallel` の「1本目は呼び出しスレッド自身が担当する」
 *     という既存の性質はそのまま維持する。n==1(スレッド0本でよい)の
 *     ときは常駐プールに一切触らない(pthread_create/joinをまったく
 *     行わない、以前と同じ経路)。`mdl_run_pool`側のジョブ用mutex
 *     (`mdl_pool.lock`、常駐プールの内部ロックとは別物)は以前から
 *     n値に関係なく毎回init/destroyしており、そこは変えていない。
 *   - スレッド生成が(リソース枯渇等で)**部分的に失敗**した場合の
 *     フォールバックも既存のまま踏襲する: `mdl_run_pool` は動的配分
 *     (共有カウンタ)なので、起動できた本数だけがワーカとして働き、
 *     残りの仕事は自動的に他のワーカ(または呼び出しスレッド)が拾う。
 *     `mdl_range_parallel` は静的ストライド分担なので、起動できなかった
 *     担当分を呼び出しスレッドが逐次で拾う(でないと該当インデックスが
 *     抜け落ちる)。
 *   - **teardown用に `mdl_thread_pool_shutdown()` を追加**した(任意)。
 *     既存の呼び出し側は変更不要 — 呼ばなくても壊れない。プロセス終了時に
 *     常駐スレッドがcondvar待ちのまま残るのは許容する(detached scheduling
 *     と同じ扱いで、OSがプロセス終了時に後始末する。pthread_joinしない
 *     まま main() を抜けても、Linux/macOSどちらでも安全に終了する)。
 *   - **コア(mdl_kernels.c)はこのファイルを一切参照しない**という
 *     ベアメタル不変条件は変わらない(このファイルはリンクopt-inのまま)。
 */

#include "mdl_threads.h"

#include <pthread.h>
#include <unistd.h>

enum { MDL_MAX_THREADS = 64 };

/* ------------------------------------------------------------------ */
/* 常駐ワーカプール本体                                                 */
/* ------------------------------------------------------------------ */

/** 1ワーカスレッドに割り当てるタスク。`arg` はそのワーカ専用の引数
 *  (呼び出し側が用意する、`mdl_pool_dispatch` の `argv[slot]`)。 */
typedef void (*mdl_task_fn)(void *arg);

typedef struct {
  pthread_t tid;
  int slot;      /* 0..n_workers-1。argv[slot] が自分の引数 */
} mdl_pool_slot;

typedef struct {
  pthread_mutex_t lock;   /* wake/done の同期専用(g_call_lockとは別) */
  pthread_cond_t wake;
  pthread_cond_t done;
  int n_workers;          /* 常駐ワーカ本数(呼び出しスレッド自身は含まない) */
  int generation;         /* 増分でワーカを起こす。1回のdispatchにつき+1 */
  int pending;             /* 今の世代でまだ完了していないワーカ数 */
  int shutdown;
  int initialized;
  mdl_task_fn task;
  void *const *argv;      /* argv[0..n_workers-1] が各ワーカの引数 */
  mdl_pool_slot slots[MDL_MAX_THREADS];
} mdl_thread_pool;

/* ゼロ初期化(BSS)= 未初期化状態。pthread_mutex_t/cond_t は
   `initialized` が立つまで一切触らない(pthread_mutex_init前提)。 */
static mdl_thread_pool g_pool;

/* プール全体(生成・破棄・dispatch)を直列化する管理ロック。
   g_pool.lock はワーカとの起床/完了同期専用で、こちらは
   「1回に1呼び出し」を保証するための別ロック。 */
static pthread_mutex_t g_call_lock = PTHREAD_MUTEX_INITIALIZER;

static void *mdl_pool_worker_main(void *arg_) {
  mdl_pool_slot *me = (mdl_pool_slot *)arg_;
  mdl_thread_pool *p = &g_pool;
  int seen = 0;

  pthread_mutex_lock(&p->lock);
  for (;;) {
    while (p->generation == seen && !p->shutdown) {
      pthread_cond_wait(&p->wake, &p->lock);
    }
    if (p->shutdown) break;
    seen = p->generation;
    {
      mdl_task_fn task = p->task;
      void *arg = (void *)p->argv[me->slot];
      pthread_mutex_unlock(&p->lock);
      task(arg);
      pthread_mutex_lock(&p->lock);
    }
    p->pending--;
    if (p->pending == 0) pthread_cond_signal(&p->done);
  }
  pthread_mutex_unlock(&p->lock);
  return NULL;
}

/* n_workers 本の常駐ワーカを用意する(既に同数なら何もしない。
   違えば既存プールをjoinしてから作り直す)。呼び出し元は
   g_call_lock を保持していること。n_workers<=0 は「畳むだけ」。
   戻り値は実際に起動できた本数(<=n_workers。生成失敗時は途中で止める
   既存フォールバック方針を踏襲)。 */
static int mdl_pool_ensure(int n_workers) {
  int i;

  if (n_workers > MDL_MAX_THREADS) n_workers = MDL_MAX_THREADS;
  if (g_pool.initialized && g_pool.n_workers == n_workers) {
    return g_pool.n_workers;
  }

  if (g_pool.initialized) {
    pthread_mutex_lock(&g_pool.lock);
    g_pool.shutdown = 1;
    g_pool.generation++;
    pthread_cond_broadcast(&g_pool.wake);
    pthread_mutex_unlock(&g_pool.lock);
    for (i = 0; i < g_pool.n_workers; i++) {
      pthread_join(g_pool.slots[i].tid, NULL);
    }
    pthread_cond_destroy(&g_pool.wake);
    pthread_cond_destroy(&g_pool.done);
    pthread_mutex_destroy(&g_pool.lock);
    g_pool.initialized = 0;
    g_pool.n_workers = 0;
  }

  if (n_workers <= 0) return 0;

  if (pthread_mutex_init(&g_pool.lock, NULL) != 0) return 0;
  if (pthread_cond_init(&g_pool.wake, NULL) != 0) {
    pthread_mutex_destroy(&g_pool.lock);
    return 0;
  }
  if (pthread_cond_init(&g_pool.done, NULL) != 0) {
    pthread_cond_destroy(&g_pool.wake);
    pthread_mutex_destroy(&g_pool.lock);
    return 0;
  }
  g_pool.generation = 0;
  g_pool.pending = 0;
  g_pool.shutdown = 0;
  g_pool.task = NULL;
  g_pool.argv = NULL;

  for (i = 0; i < n_workers; i++) {
    g_pool.slots[i].slot = i;
    if (pthread_create(&g_pool.slots[i].tid, NULL, mdl_pool_worker_main,
                       &g_pool.slots[i]) != 0) {
      break;   /* 起動できた分だけで我慢する(既存の生成失敗フォールバック) */
    }
  }
  g_pool.n_workers = i;
  g_pool.initialized = 1;
  return g_pool.n_workers;
}

/* usable本の常駐ワーカを起こし、それぞれ argv[0..usable-1] を渡して task を
   実行させる。**待たずに即戻る**(kick/joinの2段に分けてあるのが肝心 —
   呼び出しスレッド自身もslot0の仕事を持っているので、ここでワーカを
   起こしたらすぐ戻り、呼び出し側がその間に自分の仕事をすることで
   ワーカと並行に走らせる。1関数でブロックまでしてしまうと、ワーカの
   完了を待ってから呼び出しスレッドが自分の分を始める形になり、
   並列化が名目だけになる)。usable<=0 なら何もしない。
   呼び出し元は g_call_lock を保持していること。 */
static void mdl_pool_kick(int usable, mdl_task_fn task, void *const *argv) {
  if (usable <= 0) return;
  pthread_mutex_lock(&g_pool.lock);
  g_pool.task = task;
  g_pool.argv = argv;
  g_pool.pending = usable;
  g_pool.generation++;
  pthread_cond_broadcast(&g_pool.wake);
  pthread_mutex_unlock(&g_pool.lock);
}

/* mdl_pool_kick で起こしたワーカ全員の完了を待つ。usable<=0(kickが
   何もしていない)なら即戻る。 */
static void mdl_pool_join(int usable) {
  if (usable <= 0) return;
  pthread_mutex_lock(&g_pool.lock);
  while (g_pool.pending > 0) {
    pthread_cond_wait(&g_pool.done, &g_pool.lock);
  }
  pthread_mutex_unlock(&g_pool.lock);
}

int mdl_thread_pool_shutdown(void) {
  pthread_mutex_lock(&g_call_lock);
  mdl_pool_ensure(0);
  pthread_mutex_unlock(&g_call_lock);
  return 0;
}

/* ------------------------------------------------------------------ */
/* mdl_run_pool: N枚をNコンテキストへ(動的配分)                        */
/* ------------------------------------------------------------------ */

typedef struct {
  mdl_infer_fn fn;
  void **ctxs;
  const uint8_t *inputs;
  size_t in_stride;
  int32_t *scores;
  size_t out_stride;
  int n_items;
  /* 共有状態(このジョブ専用。常駐プールのlockとは別物) */
  pthread_mutex_t lock;
  int next;
  int err;
} mdl_pool;

/* ワーカ引数は (ジョブ, 自分のコンテキスト番号)。 */
typedef struct { mdl_pool *pool; int slot; } mdl_arg;

static void mdl_worker(mdl_arg *arg) {
  mdl_pool *p = arg->pool;
  void *ctx = p->ctxs[arg->slot];
  for (;;) {
    int i;
    pthread_mutex_lock(&p->lock);
    i = p->next < p->n_items ? p->next++ : -1;
    pthread_mutex_unlock(&p->lock);
    if (i < 0) break;
    {
      int rc = p->fn(ctx, p->inputs + (size_t)i * p->in_stride,
                     (int32_t *)((char *)p->scores +
                                 (size_t)i * p->out_stride));
      if (rc != 0) {
        pthread_mutex_lock(&p->lock);
        if (p->err == 0) p->err = rc;
        p->next = p->n_items;      /* 以降は打ち切る */
        pthread_mutex_unlock(&p->lock);
        break;
      }
    }
  }
}

/* 常駐プールへ渡す void(*)(void*) 版(mdl_task_fn への薄いアダプタ)。 */
static void mdl_worker_task(void *arg) { mdl_worker((mdl_arg *)arg); }

int mdl_run_pool(mdl_infer_fn fn, void **ctxs, int n_ctx,
                 const uint8_t *inputs, size_t in_stride,
                 int32_t *scores, size_t out_stride, int n_items) {
  mdl_pool job;
  mdl_arg args[MDL_MAX_THREADS];
  int n, rc;

  if (fn == NULL || ctxs == NULL || n_ctx < 1 || n_items < 0) return -1;
  if (n_items == 0) return 0;
  n = n_ctx < MDL_MAX_THREADS ? n_ctx : MDL_MAX_THREADS;
  if (n > n_items) n = n_items;      /* 枚数より多くスレッドを立てない */

  job.fn = fn;
  job.ctxs = ctxs;
  job.inputs = inputs;
  job.in_stride = in_stride;
  job.scores = scores;
  job.out_stride = out_stride;
  job.n_items = n_items;
  job.next = 0;
  job.err = 0;
  args[0].pool = &job;
  args[0].slot = 0;

  /* job.lock は mdl_worker が n==1 でも(無contentionで)使うので、
     常駐プールに触るかどうかに関わらず必ず初期化する
     (以前のバージョンもn値に関係なく常にpthread_mutex_initしていた)。 */
  if (pthread_mutex_init(&job.lock, NULL) != 0) return -1;

  if (n == 1) {
    /* 唯一のコンテキストは呼び出しスレッド自身が担う。常駐プールには
       触らない(以前と同じ「スレッド生成を一切しない」経路)。 */
    mdl_worker(&args[0]);
    rc = job.err;
    pthread_mutex_destroy(&job.lock);
    return rc;
  }

  {
    void *dispatch_argv[MDL_MAX_THREADS];
    int usable, i;
    pthread_mutex_lock(&g_call_lock);
    usable = mdl_pool_ensure(n - 1);
    for (i = 0; i <= usable; i++) {
      args[i].pool = &job;
      args[i].slot = i;
    }
    for (i = 0; i < usable; i++) dispatch_argv[i] = &args[i + 1];
    mdl_pool_kick(usable, mdl_worker_task, dispatch_argv);
    mdl_worker(&args[0]);            /* 呼び出しスレッド自身はslot0を担当
                                         (常駐ワーカと並行に走る) */
    mdl_pool_join(usable);
    pthread_mutex_unlock(&g_call_lock);
  }
  rc = job.err;
  pthread_mutex_destroy(&job.lock);
  return rc;
}

int mdl_hw_threads(void) {
#ifdef _SC_NPROCESSORS_ONLN
  long n = sysconf(_SC_NPROCESSORS_ONLN);
  return n > 0 ? (int)n : 1;
#else
  return 1;
#endif
}


/* ------------------------------------------------------------------ */
/* 1つの仕事をNコアへストライド分割(mdl_preproc_parallelの */
/* ロジックを汎用化して切り出したもの。以降mdl_preproc_parallelは薄い  */
/* ラッパになる)                                                       */
/* ------------------------------------------------------------------ */

typedef struct {
  mdl_index_fn fn;
  void *ctx;
  int lo, hi;
  int rc;
} mdl_range_arg;

static void mdl_range_worker(mdl_range_arg *a) {
  a->rc = a->fn(a->ctx, a->lo, a->hi);
}

/* 常駐プールへ渡す void(*)(void*) 版。 */
static void mdl_range_worker_task(void *arg) {
  mdl_range_worker((mdl_range_arg *)arg);
}

int mdl_range_parallel(mdl_index_fn fn, void *ctx, int n, int n_threads) {
  mdl_range_arg args[MDL_MAX_THREADS];
  int nt, i, rc = 0;

  if (fn == NULL || n <= 0) return -1;
  nt = n_threads < 1 ? 1 : n_threads;
  if (nt > MDL_MAX_THREADS) nt = MDL_MAX_THREADS;
  if (nt > n) nt = n;

  for (i = 0; i < nt; i++) {
    args[i].fn = fn;
    args[i].ctx = ctx;
    /* 費用が偏っている系列(例: 前処理チャネルは種別ごとに費用が18倍
       違う)でも、連続ブロック分割だと最後のワーカへ高価な種別が集中する。
       飛び飛び(ストライド)に割れば均等にばらける。 */
    args[i].lo = i;
    args[i].hi = nt;
    args[i].rc = 0;
  }

  if (nt == 1) {
    mdl_range_worker(&args[0]);        /* 常駐プールにも触らない */
    return args[0].rc;
  }

  {
    void *dispatch_argv[MDL_MAX_THREADS];
    int usable;
    pthread_mutex_lock(&g_call_lock);
    usable = mdl_pool_ensure(nt - 1);
    for (i = 0; i < usable; i++) dispatch_argv[i] = &args[i + 1];
    mdl_pool_kick(usable, mdl_range_worker_task, dispatch_argv);
    mdl_range_worker(&args[0]);        /* 1本目は呼び出しスレッド自身
                                           (常駐ワーカと並行に走る) */
    /* プール生成が部分的に失敗した分(usable < nt-1)は、静的ストライド
       分担なので担当者がいないと該当インデックスが抜け落ちる。
       run_poolの動的分配と違い、ここは呼び出しスレッドで逐次に拾う
       (既存方針の踏襲)。ワーカと並行に走らせてよい(それぞれ独立した
       ストライド範囲を担当するので競合しない)。 */
    for (i = usable + 1; i < nt; i++) mdl_range_worker(&args[i]);
    mdl_pool_join(usable);
    pthread_mutex_unlock(&g_call_lock);
  }
  for (i = 0; i < nt; i++) if (args[i].rc != 0 && rc == 0) rc = args[i].rc;
  return rc;
}


/* ------------------------------------------------------------------ */
/* 1枚を割る: 前処理のチャネル並列(mdl_range_parallelの薄いラッパ)     */
/* ------------------------------------------------------------------ */

typedef struct {
  mdl_range_fn fn;
  void *ctx;
  const uint8_t *image;
} mdl_preproc_adapt;

static int mdl_preproc_call(void *adapt_, int start, int stride) {
  mdl_preproc_adapt *a = (mdl_preproc_adapt *)adapt_;
  return a->fn(a->ctx, a->image, start, stride);
}

int mdl_preproc_parallel(mdl_range_fn fn, void *ctx, const uint8_t *image,
                         int channels, int n_threads) {
  mdl_preproc_adapt adapt;
  if (fn == NULL || channels <= 0) return -1;
  adapt.fn = fn;
  adapt.ctx = ctx;
  adapt.image = image;
  return mdl_range_parallel(mdl_preproc_call, &adapt, channels, n_threads);
}
