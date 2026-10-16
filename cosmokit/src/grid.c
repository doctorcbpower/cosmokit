/* grid.c - context (mesh geometry, MPI slabs, FFT plans), grids, basic ops */
#include "ck_internal.h"

static char g_err[512] = "";

int ck_set_error(int code, const char *fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  vsnprintf(g_err, sizeof g_err, fmt, ap);
  va_end(ap);
  return code;
}

const char *ck_last_error(void) { return g_err; }

const char *ck_strerror(int code) {
  switch (code) {
    case CK_OK: return "ok";
    case CK_ERR_ARG: return "invalid argument";
    case CK_ERR_ALLOC: return "allocation failed";
    case CK_ERR_MPI: return "MPI error";
    case CK_ERR_UNSUPPORTED: return "unsupported in this configuration";
    case CK_ERR_IO: return "I/O error";
    case CK_ERR_STATE: return "grid is in the wrong space (real/Fourier)";
    default: return "unknown error";
  }
}

int ck_has_mpi(void) {
#ifdef CK_ENABLE_MPI
  return 1;
#else
  return 0;
#endif
}
int ck_has_openmp(void) {
#ifdef _OPENMP
  return 1;
#else
  return 0;
#endif
}
const char *ck_version(void) { return CK_VERSION; }

/* ---------------------------------------------------------- reductions */
int ck_allreduce_sum_d(const ck_context *c, double *v, int n) {
#ifdef CK_ENABLE_MPI
  if (c->use_mpi)
    if (MPI_Allreduce(MPI_IN_PLACE, v, n, MPI_DOUBLE, MPI_SUM, c->comm) != MPI_SUCCESS)
      CK_FAIL(CK_ERR_MPI, "MPI_Allreduce failed");
#else
  (void)c; (void)v; (void)n;
#endif
  return CK_OK;
}
int ck_allreduce_sum_i64(const ck_context *c, int64_t *v, int n) {
#ifdef CK_ENABLE_MPI
  if (c->use_mpi)
    if (MPI_Allreduce(MPI_IN_PLACE, v, n, MPI_INT64_T, MPI_SUM, c->comm) != MPI_SUCCESS)
      CK_FAIL(CK_ERR_MPI, "MPI_Allreduce failed");
#else
  (void)c; (void)v; (void)n;
#endif
  return CK_OK;
}
int ck_allreduce_max_i64(const ck_context *c, int64_t *v, int n) {
#ifdef CK_ENABLE_MPI
  if (c->use_mpi)
    if (MPI_Allreduce(MPI_IN_PLACE, v, n, MPI_INT64_T, MPI_MAX, c->comm) != MPI_SUCCESS)
      CK_FAIL(CK_ERR_MPI, "MPI_Allreduce failed");
#else
  (void)c; (void)v; (void)n;
#endif
  return CK_OK;
}

/* -------------------------------------------------------------- context */
static int fftw_ready = 0;

ck_context *ck_context_create(int nmesh, double box, int nthreads, int comm_f) {
  if (nmesh < 4 || !(box > 0.0)) {
    ck_set_error(CK_ERR_ARG, "need nmesh>=4 and box>0");
    return NULL;
  }
  ck_context *c = calloc(1, sizeof *c);
  if (!c) { ck_set_error(CK_ERR_ALLOC, "context"); return NULL; }
  c->n = nmesh;
  c->box = box;
  c->rank = 0;
  c->size = 1;

#ifdef _OPENMP
  if (nthreads > 0) omp_set_num_threads(nthreads);
  c->nthreads = omp_get_max_threads();
#else
  c->nthreads = 1;
  (void)nthreads;
#endif

#ifdef CK_ENABLE_MPI
  if (comm_f >= 0) {
    int flag = 0;
    MPI_Initialized(&flag);
    if (!flag) {
      ck_set_error(CK_ERR_MPI, "MPI_Init must be called before creating an MPI context");
      free(c);
      return NULL;
    }
    c->comm = MPI_Comm_f2c((MPI_Fint)comm_f);
    MPI_Comm_rank(c->comm, &c->rank);
    MPI_Comm_size(c->comm, &c->size);
    c->use_mpi = c->size > 1;
  }
#else
  (void)comm_f;
#endif

  if (!fftw_ready) {
#ifdef CK_ENABLE_OPENMP
    fftw_init_threads();
#endif
#ifdef CK_ENABLE_MPI
    fftw_mpi_init();
#endif
    fftw_ready = 1;
  }
#ifdef CK_ENABLE_OPENMP
  fftw_plan_with_nthreads(c->nthreads);
#endif

  const ptrdiff_t N = nmesh;
  c->nzh = (size_t)(N / 2 + 1);
  c->pitch = 2 * c->nzh;
  c->plane = (size_t)N * c->pitch;
  c->slab_start = calloc((size_t)c->size + 1, sizeof(ptrdiff_t));
  if (!c->slab_start) { ck_set_error(CK_ERR_ALLOC, "slab table"); free(c); return NULL; }

#ifdef CK_ENABLE_MPI
  if (c->use_mpi) {
    c->alloc_local = fftw_mpi_local_size_3d(N, N, N / 2 + 1, c->comm, &c->local_n0,
                                            &c->local_0_start);
    ptrdiff_t mine[2] = {c->local_0_start, c->local_n0};
    ptrdiff_t *all = malloc(2 * (size_t)c->size * sizeof(ptrdiff_t));
    MPI_Allgather(mine, 2, MPI_AINT, all, 2, MPI_AINT, c->comm);
    int ok = 1;
    for (int r = 0; r < c->size; r++) {
      c->slab_start[r] = all[2 * r];
      if (all[2 * r + 1] < 1) ok = 0;
      if (r > 0 && all[2 * r] != all[2 * (r - 1)] + all[2 * (r - 1) + 1]) ok = 0;
    }
    c->slab_start[c->size] = N;
    free(all);
    if (!ok) {
      ck_set_error(CK_ERR_ARG, "nmesh=%d too small for %d MPI ranks (every rank needs >=1 x-plane)", nmesh, c->size);
      free(c->slab_start);
      free(c);
      return NULL;
    }
  } else
#endif
  {
    c->local_n0 = N;
    c->local_0_start = 0;
    c->alloc_local = N * N * (N / 2 + 1);
    c->slab_start[0] = 0;
    c->slab_start[1] = N;
  }

  c->pad = ((c->plane + 7) / 8) * 8;
  size_t need = (size_t)(c->local_n0 + 1) * c->plane;
  size_t fftn = 2 * (size_t)c->alloc_local;
  c->total = c->pad + (need > fftn ? need : fftn);

  /* plans (FFTW_ESTIMATE does not touch the arrays) */
  double *scratch = fftw_malloc(c->total * sizeof(double));
  if (!scratch) { ck_set_error(CK_ERR_ALLOC, "plan scratch"); free(c->slab_start); free(c); return NULL; }
  double *d = scratch + c->pad;
  fftw_complex *cd = (fftw_complex *)d;
#ifdef CK_ENABLE_MPI
  if (c->use_mpi) {
    c->fwd = fftw_mpi_plan_dft_r2c_3d(N, N, N, d, cd, c->comm, FFTW_ESTIMATE);
    c->inv = fftw_mpi_plan_dft_c2r_3d(N, N, N, cd, d, c->comm, FFTW_ESTIMATE);
  } else
#endif
  {
    c->fwd = fftw_plan_dft_r2c_3d(nmesh, nmesh, nmesh, d, cd, FFTW_ESTIMATE);
    c->inv = fftw_plan_dft_c2r_3d(nmesh, nmesh, nmesh, cd, d, FFTW_ESTIMATE);
  }
  fftw_free(scratch);
  if (!c->fwd || !c->inv) {
    ck_set_error(CK_ERR_ALLOC, "FFTW plan creation failed");
    free(c->slab_start);
    free(c);
    return NULL;
  }
  return c;
}

void ck_context_destroy(ck_context *c) {
  if (!c) return;
  if (c->fwd) fftw_destroy_plan(c->fwd);
  if (c->inv) fftw_destroy_plan(c->inv);
  free(c->slab_start);
  free(c);
}

void ck_context_info(const ck_context *c, int *nmesh, double *box, int *rank, int *size,
                     int64_t *local_n0, int64_t *local_0_start, int *nthreads) {
  if (nmesh) *nmesh = c->n;
  if (box) *box = c->box;
  if (rank) *rank = c->rank;
  if (size) *size = c->size;
  if (local_n0) *local_n0 = c->local_n0;
  if (local_0_start) *local_0_start = c->local_0_start;
  if (nthreads) *nthreads = c->nthreads;
}

/* ----------------------------------------------------------------- grid */
ck_grid *ck_grid_create(ck_context *ctx) {
  if (!ctx) { ck_set_error(CK_ERR_ARG, "null context"); return NULL; }
  ck_grid *g = calloc(1, sizeof *g);
  if (!g) { ck_set_error(CK_ERR_ALLOC, "grid"); return NULL; }
  g->ctx = ctx;
  g->buf = fftw_malloc(ctx->total * sizeof(double));
  if (!g->buf) {
    ck_set_error(CK_ERR_ALLOC, "grid buffer of %.2f GB", ctx->total * 8.0 / 1e9);
    free(g);
    return NULL;
  }
  g->data = g->buf + ctx->pad;
  ck_grid_zero(g);
  return g;
}

void ck_grid_destroy(ck_grid *g) {
  if (!g) return;
  fftw_free(g->buf);
  free(g);
}

int ck_grid_zero(ck_grid *g) {
  /* first-touch in parallel so pages land near the threads that use them */
  const size_t n = g->ctx->total;
#pragma omp parallel for schedule(static)
  for (size_t i = 0; i < n; i++) g->buf[i] = 0.0;
  g->fourier = 0;
  g->npart = 0;
  g->sum_w = g->sum_w2 = 0.0;
  return CK_OK;
}

int ck_grid_check_compat(const ck_grid *a, const ck_grid *b) {
  if (!a || !b) CK_FAIL(CK_ERR_ARG, "null grid");
  if (a->ctx != b->ctx) CK_FAIL(CK_ERR_ARG, "grids belong to different contexts");
  return CK_OK;
}

int ck_is_fourier(const ck_grid *g, int want) {
  if (!g) CK_FAIL(CK_ERR_ARG, "null grid");
  if (g->fourier != want)
    CK_FAIL(CK_ERR_STATE, "grid is in %s space, %s space required",
            g->fourier ? "Fourier" : "real", want ? "Fourier" : "real");
  return CK_OK;
}

int ck_grid_copy(ck_grid *dst, const ck_grid *src) {
  int rc = ck_grid_check_compat(dst, src);
  if (rc) return rc;
  const size_t n = (size_t)src->ctx->local_n0 * src->ctx->plane;
#pragma omp parallel for schedule(static)
  for (size_t i = 0; i < n; i++) dst->data[i] = src->data[i];
  dst->fourier = src->fourier;
  dst->npart = src->npart;
  dst->sum_w = src->sum_w;
  dst->sum_w2 = src->sum_w2;
  return CK_OK;
}

void ck_grid_info(const ck_grid *g, int *state, int64_t *npart, double *sum_w, double *sum_w2) {
  if (state) *state = g->fourier;
  if (npart) *npart = g->npart;
  if (sum_w) *sum_w = g->sum_w;
  if (sum_w2) *sum_w2 = g->sum_w2;
}

int ck_grid_get_real(const ck_grid *g, double *out) {
  int rc = ck_is_fourier(g, 0);
  if (rc) return rc;
  const ck_context *c = g->ctx;
  const size_t N = c->n;
#pragma omp parallel for collapse(2) schedule(static)
  for (ptrdiff_t il = 0; il < c->local_n0; il++)
    for (size_t iy = 0; iy < N; iy++)
      memcpy(out + ((size_t)il * N + iy) * N, g->data + ck_ridx(c, il, iy, 0), N * sizeof(double));
  return CK_OK;
}

int ck_grid_set_real(ck_grid *g, const double *in) {
  const ck_context *c = g->ctx;
  const size_t N = c->n;
#pragma omp parallel for collapse(2) schedule(static)
  for (ptrdiff_t il = 0; il < c->local_n0; il++)
    for (size_t iy = 0; iy < N; iy++)
      memcpy(g->data + ck_ridx(c, il, iy, 0), in + ((size_t)il * N + iy) * N, N * sizeof(double));
  g->fourier = 0;
  return CK_OK;
}

int ck_grid_get_fourier(const ck_grid *g, double *out) {
  int rc = ck_is_fourier(g, 1);
  if (rc) return rc;
  const ck_context *c = g->ctx;
  const size_t N = c->n;
#pragma omp parallel for collapse(2) schedule(static)
  for (size_t il = 0; il < (size_t)c->local_n0; il++)
    for (size_t iy = 0; iy < N; iy++)
      memcpy(out + 2 * ck_kidx(c, il, iy, 0), g->data + 2 * ck_kidx(c, il, iy, 0),
             2 * c->nzh * sizeof(double));
  return CK_OK;
}

int ck_grid_set_fourier(ck_grid *g, const double *in) {
  const ck_context *c = g->ctx;
  const size_t N = c->n;
#pragma omp parallel for collapse(2) schedule(static)
  for (size_t il = 0; il < (size_t)c->local_n0; il++)
    for (size_t iy = 0; iy < N; iy++)
      memcpy(g->data + 2 * ck_kidx(c, il, iy, 0), in + 2 * ck_kidx(c, il, iy, 0),
             2 * c->nzh * sizeof(double));
  g->fourier = 1;
  return CK_OK;
}

/* ------------------------------------------------------------------ FFT */
int ck_grid_fft_forward(ck_grid *g) {
  int rc = ck_is_fourier(g, 0);
  if (rc) return rc;
  ck_context *c = g->ctx;
  fftw_complex *cd = (fftw_complex *)g->data;
#ifdef CK_ENABLE_MPI
  if (c->use_mpi) fftw_mpi_execute_dft_r2c(c->fwd, g->data, cd);
  else
#endif
  fftw_execute_dft_r2c(c->fwd, g->data, cd);
  const double norm = 1.0 / ((double)c->n * c->n * c->n);
  const size_t nd = 2 * (size_t)c->local_n0 * c->n * c->nzh;
#pragma omp parallel for schedule(static)
  for (size_t i = 0; i < nd; i++) g->data[i] *= norm;
  g->fourier = 1;
  return CK_OK;
}

int ck_grid_fft_inverse(ck_grid *g) {
  int rc = ck_is_fourier(g, 1);
  if (rc) return rc;
  ck_context *c = g->ctx;
  fftw_complex *cd = (fftw_complex *)g->data;
#ifdef CK_ENABLE_MPI
  if (c->use_mpi) fftw_mpi_execute_dft_c2r(c->inv, cd, g->data);
  else
#endif
  fftw_execute_dft_c2r(c->inv, cd, g->data);
  g->fourier = 0;
  return CK_OK;
}

/* ------------------------------------------------------------ simple ops */
double ck_grid_sum(const ck_grid *g) {
  const ck_context *c = g->ctx;
  const size_t N = c->n;
  double s = 0.0;
#pragma omp parallel for collapse(2) reduction(+ : s) schedule(static)
  for (ptrdiff_t il = 0; il < c->local_n0; il++)
    for (size_t iy = 0; iy < N; iy++) {
      const double *row = g->data + ck_ridx(c, il, iy, 0);
      for (size_t iz = 0; iz < N; iz++) s += row[iz];
    }
  ck_allreduce_sum_d(c, &s, 1);
  return s;
}

int ck_grid_scale(ck_grid *g, double f) {
  const ck_context *c = g->ctx;
  const size_t n = (size_t)c->local_n0 * c->plane; /* padding scaled too: harmless */
#pragma omp parallel for schedule(static)
  for (size_t i = 0; i < n; i++) g->data[i] *= f;
  return CK_OK;
}

int ck_grid_to_overdensity(ck_grid *g) {
  int rc = ck_is_fourier(g, 0);
  if (rc) return rc;
  const ck_context *c = g->ctx;
  const size_t N = c->n;
  const double mean = ck_grid_sum(g) / ((double)N * N * N);
  if (!(mean > 0.0)) CK_FAIL(CK_ERR_ARG, "mean density is %g, cannot form overdensity", mean);
  const double inv = 1.0 / mean;
#pragma omp parallel for collapse(2) schedule(static)
  for (ptrdiff_t il = 0; il < c->local_n0; il++)
    for (size_t iy = 0; iy < N; iy++) {
      double *row = g->data + ck_ridx(c, il, iy, 0);
      for (size_t iz = 0; iz < N; iz++) row[iz] = row[iz] * inv - 1.0;
    }
  return CK_OK;
}

int ck_grid_divide(ck_grid *out, const ck_grid *num, const ck_grid *den) {
  int rc = ck_grid_check_compat(out, num);
  if (rc) return rc;
  rc = ck_grid_check_compat(out, den);
  if (rc) return rc;
  if (num->fourier || den->fourier) CK_FAIL(CK_ERR_STATE, "ck_grid_divide works in real space");
  const ck_context *c = out->ctx;
  const size_t N = c->n;
#pragma omp parallel for collapse(2) schedule(static)
  for (ptrdiff_t il = 0; il < c->local_n0; il++)
    for (size_t iy = 0; iy < N; iy++) {
      const size_t o = ck_ridx(c, il, iy, 0);
      for (size_t iz = 0; iz < N; iz++) {
        const double d = den->data[o + iz];
        out->data[o + iz] = d > 0.0 ? num->data[o + iz] / d : 0.0;
      }
    }
  out->fourier = 0;
  return CK_OK;
}
