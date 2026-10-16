#ifndef CK_INTERNAL_H
#define CK_INTERNAL_H

#include <fftw3.h>
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef CK_ENABLE_MPI
#include <fftw3-mpi.h>
#include <mpi.h>
#endif
#ifdef _OPENMP
#include <omp.h>
#endif

#include "cosmokit.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

struct ck_context {
  int n;
  double box;
  int nthreads;
  int rank, size;
  int use_mpi;                 /* size>1 and communicator supplied */
#ifdef CK_ENABLE_MPI
  MPI_Comm comm;
#endif
  ptrdiff_t local_n0, local_0_start;
  ptrdiff_t alloc_local;       /* complex elements, as reported by FFTW */
  ptrdiff_t *slab_start;       /* size+1 entries, slab_start[size] == n */
  size_t nzh;                  /* n/2+1 */
  size_t pitch;                /* 2*nzh : padded real row length */
  size_t plane;                /* n*pitch doubles per x-plane */
  size_t pad;                  /* doubles before data (>= plane, multiple of 8) */
  size_t total;                /* doubles allocated per grid */
  fftw_plan fwd, inv;
};

struct ck_grid {
  ck_context *ctx;
  double *buf;
  double *data;  /* buf + pad; x-plane -1 (left ghost) lives at data-plane */
  int fourier;
  int64_t npart;
  double sum_w, sum_w2;
};

/* error reporting ------------------------------------------------------- */
int ck_set_error(int code, const char *fmt, ...);
#define CK_FAIL(code, ...) return ck_set_error((code), __VA_ARGS__)

/* helpers --------------------------------------------------------------- */
static inline int ck_freq(int i, int n) { return i <= n / 2 ? i : i - n; }
/* 1 if mode index i is the Nyquist frequency of an even-sized axis */
static inline int ck_is_nyq(int i, int n) { return (n % 2 == 0) && (2 * i == n); }

static inline double ck_sinc(double x) { return fabs(x) < 1e-8 ? 1.0 - x * x / 6.0 : sin(x) / x; }

static inline int ck_thread_num(void) {
#ifdef _OPENMP
  return omp_get_thread_num();
#else
  return 0;
#endif
}
static inline int ck_max_threads(void) {
#ifdef _OPENMP
  return omp_get_max_threads();
#else
  return 1;
#endif
}

static inline double ck_rd(const void *p, ck_dtype t, int64_t i) {
  return t == CK_FLOAT32 ? (double)((const float *)p)[i] : ((const double *)p)[i];
}

/* real-space index of local cell (il,iy,iz) */
static inline size_t ck_ridx(const ck_context *c, ptrdiff_t il, size_t iy, size_t iz) {
  return ((size_t)il * c->n + iy) * c->pitch + iz;
}
/* Fourier-space (complex) index, in complex elements */
static inline size_t ck_kidx(const ck_context *c, size_t il, size_t iy, size_t iz) {
  return (il * c->n + iy) * c->nzh + iz;
}

int ck_grid_check_compat(const ck_grid *a, const ck_grid *b);
int ck_is_fourier(const ck_grid *g, int want);

int ck_allreduce_sum_d(const ck_context *c, double *v, int n);
int ck_allreduce_sum_i64(const ck_context *c, int64_t *v, int n);
int ck_allreduce_max_i64(const ck_context *c, int64_t *v, int n);

/* Aliasing-aware Poisson shot-noise factor for one axis, window order p. */
double ck_shot_axis(int p, double s2 /* sin^2(pi n/N) */);

/* kd-tree internals (shared with sph.c) ------------------------------- */
typedef struct { double x[3]; int64_t i; } ck_kpt;
typedef struct { int64_t lo, hi; double bmin[3], bmax[3]; } ck_knode;
struct ck_kdtree {
  int64_t n;
  double box;     /* <=0 : non-periodic */
  int leaf;
  ck_kpt *pt;     /* points in tree order */
  ck_knode *node; /* heap-indexed: children of i are 2i+1, 2i+2 */
  int64_t nnodes;
};
/* k nearest to q (already wrapped). skip: original index to exclude (-1: none).
 * d2/idx sized k, returned ascending. returns number found. */
int ck_kdtree_knn_one(const ck_kdtree *t, const double q[3], int k, int64_t skip,
                      double *d2, int64_t *idx);

/* snapshot back-ends (dispatched from io.c) */
int ck_deposit_hdf5(const char *, unsigned, ck_grid *, ck_grid *[3], ck_scheme, int64_t, double, double *, int64_t *);
int ck_boxsize_hdf5(const char *, double *);
int ck_deposit_gadget(const char *, unsigned, ck_grid *, ck_grid *[3], ck_scheme, int64_t, double, double *, int64_t *);
int ck_boxsize_gadget(const char *, double *);

#endif
