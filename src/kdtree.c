/* kdtree.c - balanced, bucketed k-d tree with periodic minimum-image queries.
 *
 * Median-split on the widest bounding-box dimension, heap-indexed nodes with
 * per-node bounding boxes. Build and queries are OpenMP parallel. Shared
 * memory only; for distributed data, partition spatially and add halos.
 */
#include "ck_internal.h"

#define DEFAULT_LEAF 16
#define TASK_CUTOFF 32768

static inline void swap_pt(ck_kpt *a, ck_kpt *b) { ck_kpt t = *a; *a = *b; *b = t; }

/* partial sort so that a[k] has the k-th smallest x[d] in [lo,hi) */
static void nth_element(ck_kpt *a, int64_t lo, int64_t hi, int64_t k, int d) {
  hi--;
  while (hi > lo) {
    const int64_t m = lo + (hi - lo) / 2;
    if (a[m].x[d] < a[lo].x[d]) swap_pt(&a[m], &a[lo]);
    if (a[hi].x[d] < a[lo].x[d]) swap_pt(&a[hi], &a[lo]);
    if (a[hi].x[d] < a[m].x[d]) swap_pt(&a[hi], &a[m]);
    const double pivot = a[m].x[d];
    int64_t i = lo, j = hi;
    while (i <= j) {
      while (a[i].x[d] < pivot) i++;
      while (a[j].x[d] > pivot) j--;
      if (i <= j) { swap_pt(&a[i], &a[j]); i++; j--; }
    }
    if (k <= j) hi = j;
    else if (k >= i) lo = i;
    else return;
  }
}

static void build_node(ck_kdtree *t, int64_t id, int64_t lo, int64_t hi) {
  ck_knode *nd = &t->node[id];
  nd->lo = lo;
  nd->hi = hi;
  for (int d = 0; d < 3; d++) { nd->bmin[d] = INFINITY; nd->bmax[d] = -INFINITY; }
  for (int64_t i = lo; i < hi; i++)
    for (int d = 0; d < 3; d++) {
      const double v = t->pt[i].x[d];
      if (v < nd->bmin[d]) nd->bmin[d] = v;
      if (v > nd->bmax[d]) nd->bmax[d] = v;
    }
  if (hi - lo <= t->leaf) return;
  int dim = 0;
  double ext = nd->bmax[0] - nd->bmin[0];
  for (int d = 1; d < 3; d++)
    if (nd->bmax[d] - nd->bmin[d] > ext) { ext = nd->bmax[d] - nd->bmin[d]; dim = d; }
  const int64_t mid = lo + (hi - lo) / 2;
  nth_element(t->pt, lo, hi, mid, dim);
#pragma omp task shared(t) if (hi - lo > TASK_CUTOFF)
  build_node(t, 2 * id + 1, lo, mid);
#pragma omp task shared(t) if (hi - lo > TASK_CUTOFF)
  build_node(t, 2 * id + 2, mid, hi);
#pragma omp taskwait
}

ck_kdtree *ck_kdtree_build(const void *pos, ck_dtype dtype, int64_t n, double box,
                           int leaf_size) {
  if (n <= 0 || !pos) { ck_set_error(CK_ERR_ARG, "kdtree: need at least one point"); return NULL; }
  ck_kdtree *t = calloc(1, sizeof *t);
  if (!t) { ck_set_error(CK_ERR_ALLOC, "kdtree"); return NULL; }
  t->n = n;
  t->box = box > 0.0 ? box : 0.0;
  t->leaf = leaf_size > 0 ? leaf_size : DEFAULT_LEAF;
  int depth = 0;
  for (int64_t s = n; s > t->leaf; s = (s + 1) / 2) depth++;
  t->nnodes = ((int64_t)1 << (depth + 1)) - 1;
  t->pt = malloc((size_t)n * sizeof(ck_kpt));
  t->node = malloc((size_t)t->nnodes * sizeof(ck_knode));
  if (!t->pt || !t->node) {
    free(t->pt); free(t->node); free(t);
    ck_set_error(CK_ERR_ALLOC, "kdtree storage for %lld points", (long long)n);
    return NULL;
  }
#pragma omp parallel for schedule(static)
  for (int64_t i = 0; i < n; i++) {
    for (int d = 0; d < 3; d++) {
      double v = ck_rd(pos, dtype, 3 * i + d);
      if (t->box > 0.0) {
        v -= t->box * floor(v / t->box);
        if (v >= t->box) v = 0.0;
      }
      t->pt[i].x[d] = v;
    }
    t->pt[i].i = i;
  }
#pragma omp parallel
#pragma omp single nowait
  build_node(t, 0, 0, n);
  return t;
}

void ck_kdtree_free(ck_kdtree *t) {
  if (!t) return;
  free(t->pt);
  free(t->node);
  free(t);
}

int64_t ck_kdtree_size(const ck_kdtree *t) { return t->n; }

/* ----------------------------------------------------------- distances */
static inline double box_d2(const ck_knode *nd, const double *q, double L) {
  double s = 0.0;
  for (int d = 0; d < 3; d++) {
    const double lo = nd->bmin[d], hi = nd->bmax[d];
    double g = 0.0;
    if (q[d] < lo) {
      g = lo - q[d];
      if (L > 0.0) { const double g2 = q[d] + L - hi; if (g2 < g) g = g2; }
    } else if (q[d] > hi) {
      g = q[d] - hi;
      if (L > 0.0) { const double g2 = lo + L - q[d]; if (g2 < g) g = g2; }
    }
    s += g * g;
  }
  return s;
}

static inline double pt_d2(const ck_kpt *p, const double *q, double L) {
  double s = 0.0;
  for (int d = 0; d < 3; d++) {
    double dx = p->x[d] - q[d];
    if (L > 0.0) dx -= L * floor(dx / L + 0.5);
    s += dx * dx;
  }
  return s;
}

static inline void wrap_query(const ck_kdtree *t, const void *q, ck_dtype qt, int64_t i,
                              double out[3]) {
  for (int d = 0; d < 3; d++) {
    double v = ck_rd(q, qt, 3 * i + d);
    if (t->box > 0.0) {
      v -= t->box * floor(v / t->box);
      if (v >= t->box) v = 0.0;
    }
    out[d] = v;
  }
}

/* ----------------------------------------------------------------- kNN */
typedef struct {
  const ck_kdtree *t;
  double q[3];
  int k, cnt;
  int64_t skip;
  double *d2;
  int64_t *idx;
} knn_state;

static void knn_rec(knn_state *s, int64_t id) {
  const ck_kdtree *t = s->t;
  const ck_knode *nd = &t->node[id];
  if (nd->hi - nd->lo <= t->leaf) {
    for (int64_t i = nd->lo; i < nd->hi; i++) {
      if (t->pt[i].i == s->skip) continue;
      const double d2 = pt_d2(&t->pt[i], s->q, t->box);
      if (s->cnt == s->k && d2 >= s->d2[s->k - 1]) continue;
      int j = s->cnt < s->k ? s->cnt++ : s->k - 1;
      while (j > 0 && s->d2[j - 1] > d2) {
        s->d2[j] = s->d2[j - 1];
        s->idx[j] = s->idx[j - 1];
        j--;
      }
      s->d2[j] = d2;
      s->idx[j] = t->pt[i].i;
    }
    return;
  }
  const int64_t l = 2 * id + 1, r = 2 * id + 2;
  const double dl = box_d2(&t->node[l], s->q, t->box);
  const double dr = box_d2(&t->node[r], s->q, t->box);
  const int64_t first = dl <= dr ? l : r, second = dl <= dr ? r : l;
  const double d1 = dl <= dr ? dl : dr, d2v = dl <= dr ? dr : dl;
  if (s->cnt < s->k || d1 < s->d2[s->k - 1]) knn_rec(s, first);
  if (s->cnt < s->k || d2v < s->d2[s->k - 1]) knn_rec(s, second);
}

int ck_kdtree_knn_one(const ck_kdtree *t, const double q[3], int k, int64_t skip, double *d2,
                      int64_t *idx) {
  knn_state s = {t, {q[0], q[1], q[2]}, k, 0, skip, d2, idx};
  knn_rec(&s, 0);
  return s.cnt;
}

int ck_kdtree_knn(const ck_kdtree *t, const void *q, ck_dtype qtype, int64_t m, int k,
                  unsigned flags, double *dist, int64_t *idx) {
  if (!t || !q || !dist || !idx || k < 1 || m < 0) CK_FAIL(CK_ERR_ARG, "bad knn arguments");
  const int skip_self = (flags & CK_KNN_SKIP_SELF) != 0;
#pragma omp parallel
  {
    double *d2 = malloc((size_t)k * sizeof(double));
    int64_t *ii = malloc((size_t)k * sizeof(int64_t));
#pragma omp for schedule(dynamic, 64)
    for (int64_t i = 0; i < m; i++) {
      double qq[3];
      wrap_query(t, q, qtype, i, qq);
      const int got = ck_kdtree_knn_one(t, qq, k, skip_self ? i : -1, d2, ii);
      for (int j = 0; j < k; j++) {
        dist[i * k + j] = j < got ? sqrt(d2[j]) : INFINITY;
        idx[i * k + j] = j < got ? ii[j] : -1;
      }
    }
    free(d2);
    free(ii);
  }
  return CK_OK;
}

/* -------------------------------------------------------------- radius */
typedef struct {
  const ck_kdtree *t;
  double q[3], r2;
  int64_t skip, count;
  int64_t *out; /* NULL: count only */
} rad_state;

static void rad_rec(rad_state *s, int64_t id) {
  const ck_kdtree *t = s->t;
  const ck_knode *nd = &t->node[id];
  if (box_d2(nd, s->q, t->box) > s->r2) return;
  if (nd->hi - nd->lo <= t->leaf) {
    for (int64_t i = nd->lo; i < nd->hi; i++) {
      if (t->pt[i].i == s->skip) continue;
      if (pt_d2(&t->pt[i], s->q, t->box) <= s->r2) {
        if (s->out) s->out[s->count] = t->pt[i].i;
        s->count++;
      }
    }
    return;
  }
  rad_rec(s, 2 * id + 1);
  rad_rec(s, 2 * id + 2);
}

int ck_kdtree_radius_count(const ck_kdtree *t, const void *q, ck_dtype qtype, int64_t m,
                           const double *radii, unsigned flags, int64_t *counts) {
  if (!t || !q || !radii || !counts) CK_FAIL(CK_ERR_ARG, "null argument");
#pragma omp parallel for schedule(dynamic, 64)
  for (int64_t i = 0; i < m; i++) {
    rad_state s;
    s.t = t;
    wrap_query(t, q, qtype, i, s.q);
    s.r2 = radii[i] * radii[i];
    s.skip = (flags & CK_KNN_SKIP_SELF) ? i : -1;
    s.count = 0;
    s.out = NULL;
    rad_rec(&s, 0);
    counts[i] = s.count;
  }
  return CK_OK;
}

int ck_kdtree_radius_fill(const ck_kdtree *t, const void *q, ck_dtype qtype, int64_t m,
                          const double *radii, unsigned flags, const int64_t *offsets,
                          int64_t *out_idx) {
  if (!t || !q || !radii || !offsets || !out_idx) CK_FAIL(CK_ERR_ARG, "null argument");
#pragma omp parallel for schedule(dynamic, 64)
  for (int64_t i = 0; i < m; i++) {
    rad_state s;
    s.t = t;
    wrap_query(t, q, qtype, i, s.q);
    s.r2 = radii[i] * radii[i];
    s.skip = (flags & CK_KNN_SKIP_SELF) ? i : -1;
    s.count = 0;
    s.out = out_idx + offsets[i];
    rad_rec(&s, 0);
  }
  return CK_OK;
}
