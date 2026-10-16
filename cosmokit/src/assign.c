/* assign.c - mass / momentum assignment (NGP, CIC, TSC) and SPH-kernel deposit.
 *
 * MPI strategy: grids are x-slabs. Particles are sent (in bounded chunks) to
 * the rank owning their home cell; each rank deposits into its slab plus ONE
 * ghost plane on either side (enough for NGP/CIC/TSC), then the ghost planes
 * are added into the neighbouring slabs. No particle replication.
 */
#include "ck_internal.h"

#define MAX_FIELDS 4
#define CHUNK_PARTICLES (1 << 21)

/* Per-axis weights. Returns number of cells (1,2,3). ix are *unwrapped* cell
 * indices relative to the periodic image that contains the home cell. */
static inline int axis_weights(double u, int scheme, long *ix, double *w) {
  switch (scheme) {
    case CK_NGP:
      ix[0] = (long)floor(u);
      w[0] = 1.0;
      return 1;
    case CK_CIC: {
      const double t = u - 0.5;
      const long i0 = (long)floor(t);
      const double d = t - (double)i0;
      ix[0] = i0;     w[0] = 1.0 - d;
      ix[1] = i0 + 1; w[1] = d;
      return 2;
    }
    default: { /* TSC */
      const long c = (long)floor(u);
      const double d = u - (double)c - 0.5;
      ix[0] = c - 1; w[0] = 0.5 * (0.5 - d) * (0.5 - d);
      ix[1] = c;     w[1] = 0.75 - d * d;
      ix[2] = c + 1; w[2] = 0.5 * (0.5 + d) * (0.5 + d);
      return 3;
    }
  }
}

static inline long wrap_idx(long i, long n) {
  i %= n;
  return i < 0 ? i + n : i;
}

/* Wrap a coordinate in box units to u in [0,N). */
static inline double to_grid(double x, double inv_h, int N) {
  double u = x * inv_h;
  u -= (double)N * floor(u / (double)N);
  if (u >= (double)N) u -= (double)N; /* rounding guard */
  if (u < 0.0) u = 0.0;
  return u;
}

/* Deposit the (owned) particles in p into the ghost-extended slab. */
static void deposit_local(ck_context *c, double *field[MAX_FIELDS], int nf, int hasvel,
                          const ck_particles *p, int scheme, int64_t *cnt,
                          double *sw, double *sw2) {
  const int N = c->n;
  const double inv_h = (double)N / c->box;
  const long x0 = (long)c->local_0_start;
  const size_t pitch = c->pitch, plane = c->plane;
  int64_t count = 0;
  double sum_w = 0.0, sum_w2 = 0.0;

#pragma omp parallel for reduction(+ : count, sum_w, sum_w2) schedule(static)
  for (int64_t i = 0; i < p->n; i++) {
    const double m = p->mass ? ck_rd(p->mass, p->mass_dtype, i) : p->mass0;
    double q[MAX_FIELDS];
    q[0] = m;
    if (hasvel) {
      q[1] = m * ck_rd(p->vel, p->vel_dtype, 3 * i + 0);
      q[2] = m * ck_rd(p->vel, p->vel_dtype, 3 * i + 1);
      q[3] = m * ck_rd(p->vel, p->vel_dtype, 3 * i + 2);
    }
    const double ux = to_grid(ck_rd(p->pos, p->pos_dtype, 3 * i + 0), inv_h, N);
    const double uy = to_grid(ck_rd(p->pos, p->pos_dtype, 3 * i + 1), inv_h, N);
    const double uz = to_grid(ck_rd(p->pos, p->pos_dtype, 3 * i + 2), inv_h, N);
    long ix[3], iy[3], iz[3];
    double wx[3], wy[3], wz[3];
    const int nx = axis_weights(ux, scheme, ix, wx);
    const int ny = axis_weights(uy, scheme, iy, wy);
    const int nz = axis_weights(uz, scheme, iz, wz);
    for (int b = 0; b < ny; b++) iy[b] = wrap_idx(iy[b], N);
    for (int b = 0; b < nz; b++) iz[b] = wrap_idx(iz[b], N);

    for (int a = 0; a < nx; a++) {
      const long il = ix[a] - x0; /* in [-1, local_n0] by construction */
      for (int b = 0; b < ny; b++) {
        const double wxy = wx[a] * wy[b];
        const ptrdiff_t row = (ptrdiff_t)plane * il + (ptrdiff_t)(iy[b] * (long)pitch);
        for (int d = 0; d < nz; d++) {
          const double w = wxy * wz[d];
          const ptrdiff_t o = row + iz[d];
          for (int f = 0; f < nf; f++) {
#pragma omp atomic update
            field[f][o] += w * q[f];
          }
        }
      }
    }
    count++;
    sum_w += m;
    sum_w2 += m * m;
  }
  *cnt += count;
  *sw += sum_w;
  *sw2 += sum_w2;
}

/* Add ghost planes into the neighbouring slabs (periodic ring in x). */
static int ghost_exchange(ck_context *c, double *data) {
  const size_t plane = c->plane;
  double *left = data - plane;                       /* global plane x0-1  */
  double *right = data + (size_t)c->local_n0 * plane; /* global plane x0+n0 */
  double *first = data;
  double *last = data + (size_t)(c->local_n0 - 1) * plane;

  if (!c->use_mpi) {
    /* serial: single slab spans the box */
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < plane; i++) {
      last[i] += left[i];
      first[i] += right[i];
    }
    return CK_OK;
  }
#ifdef CK_ENABLE_MPI
  const int lr = (c->rank + c->size - 1) % c->size;
  const int rr = (c->rank + 1) % c->size;
  double *tmp = malloc(plane * sizeof(double));
  if (!tmp) CK_FAIL(CK_ERR_ALLOC, "ghost buffer");
  if (plane > (size_t)2147483647) { free(tmp); CK_FAIL(CK_ERR_UNSUPPORTED, "plane too large for MPI count"); }
  /* my left ghost -> left neighbour's last plane; I receive my right
   * neighbour's left ghost into my last plane */
  MPI_Sendrecv(left, (int)plane, MPI_DOUBLE, lr, 101, tmp, (int)plane, MPI_DOUBLE, rr, 101,
               c->comm, MPI_STATUS_IGNORE);
#pragma omp parallel for schedule(static)
  for (size_t i = 0; i < plane; i++) last[i] += tmp[i];
  MPI_Sendrecv(right, (int)plane, MPI_DOUBLE, rr, 102, tmp, (int)plane, MPI_DOUBLE, lr, 102,
               c->comm, MPI_STATUS_IGNORE);
#pragma omp parallel for schedule(static)
  for (size_t i = 0; i < plane; i++) first[i] += tmp[i];
  free(tmp);
#endif
  return CK_OK;
}

static void clear_ghosts(ck_context *c, double *data) {
  memset(data - c->plane, 0, c->plane * sizeof(double));
  memset(data + (size_t)c->local_n0 * c->plane, 0, c->plane * sizeof(double));
}

/* Which rank owns global x-plane c? */
static inline int owner_of(const ck_context *c, long cell) {
  int lo = 0, hi = c->size - 1;
  while (lo < hi) {
    const int mid = (lo + hi + 1) / 2;
    if (c->slab_start[mid] <= cell) lo = mid; else hi = mid - 1;
  }
  return lo;
}

#ifdef CK_ENABLE_MPI
/* Send particles [a,b) to their owning ranks. Output: contiguous double arrays
 * (pos 3n, vel 3n or NULL, mass n) of received (owned) particles. */
static int redistribute(ck_context *c, const ck_particles *p, int64_t a, int64_t b,
                        int hasvel, double **opos, double **ovel, double **omass,
                        int64_t *on) {
  const int N = c->n;
  const double inv_h = (double)N / c->box;
  const int S = c->size;
  const int R = 4 + (hasvel ? 3 : 0);
  const int64_t n = b - a;

  int *dest = malloc((size_t)(n > 0 ? n : 1) * sizeof(int));
  int *scount = calloc((size_t)S, sizeof(int));
  int *rcount = calloc((size_t)S, sizeof(int));
  int *sdisp = calloc((size_t)S + 1, sizeof(int));
  int *rdisp = calloc((size_t)S + 1, sizeof(int));
  if (!dest || !scount || !rcount || !sdisp || !rdisp) {
    free(dest); free(scount); free(rcount); free(sdisp); free(rdisp);
    CK_FAIL(CK_ERR_ALLOC, "redistribute counts");
  }
  for (int64_t i = 0; i < n; i++) {
    const double u = to_grid(ck_rd(p->pos, p->pos_dtype, 3 * (a + i)), inv_h, N);
    dest[i] = owner_of(c, (long)floor(u));
    scount[dest[i]]++;
  }
  MPI_Alltoall(scount, 1, MPI_INT, rcount, 1, MPI_INT, c->comm);
  int64_t tot_recv = 0;
  for (int r = 0; r < S; r++) {
    sdisp[r + 1] = sdisp[r] + scount[r];
    rdisp[r + 1] = rdisp[r] + rcount[r];
    tot_recv += rcount[r];
  }
  double *sbuf = malloc((size_t)(n > 0 ? n : 1) * R * sizeof(double));
  double *rbuf = malloc((size_t)(tot_recv > 0 ? tot_recv : 1) * R * sizeof(double));
  int *fill = calloc((size_t)S, sizeof(int));
  if (!sbuf || !rbuf || !fill) {
    free(dest); free(scount); free(rcount); free(sdisp); free(rdisp);
    free(sbuf); free(rbuf); free(fill);
    CK_FAIL(CK_ERR_ALLOC, "redistribute buffers");
  }
  for (int64_t i = 0; i < n; i++) {
    const int r = dest[i];
    double *rec = sbuf + (size_t)(sdisp[r] + fill[r]++) * R;
    const int64_t j = a + i;
    rec[0] = ck_rd(p->pos, p->pos_dtype, 3 * j + 0);
    rec[1] = ck_rd(p->pos, p->pos_dtype, 3 * j + 1);
    rec[2] = ck_rd(p->pos, p->pos_dtype, 3 * j + 2);
    rec[3] = p->mass ? ck_rd(p->mass, p->mass_dtype, j) : p->mass0;
    if (hasvel) {
      rec[4] = ck_rd(p->vel, p->vel_dtype, 3 * j + 0);
      rec[5] = ck_rd(p->vel, p->vel_dtype, 3 * j + 1);
      rec[6] = ck_rd(p->vel, p->vel_dtype, 3 * j + 2);
    }
  }
  for (int r = 0; r < S; r++) {
    scount[r] *= R; rcount[r] *= R; sdisp[r] *= R; rdisp[r] *= R;
  }
  MPI_Alltoallv(sbuf, scount, sdisp, MPI_DOUBLE, rbuf, rcount, rdisp, MPI_DOUBLE, c->comm);

  double *pos = malloc((size_t)(tot_recv > 0 ? tot_recv : 1) * 3 * sizeof(double));
  double *mass = malloc((size_t)(tot_recv > 0 ? tot_recv : 1) * sizeof(double));
  double *vel = hasvel ? malloc((size_t)(tot_recv > 0 ? tot_recv : 1) * 3 * sizeof(double)) : NULL;
  if (!pos || !mass || (hasvel && !vel)) {
    free(pos); free(mass); free(vel);
    free(dest); free(scount); free(rcount); free(sdisp); free(rdisp);
    free(sbuf); free(rbuf); free(fill);
    CK_FAIL(CK_ERR_ALLOC, "redistribute unpack");
  }
  for (int64_t i = 0; i < tot_recv; i++) {
    const double *rec = rbuf + (size_t)i * R;
    pos[3 * i] = rec[0]; pos[3 * i + 1] = rec[1]; pos[3 * i + 2] = rec[2];
    mass[i] = rec[3];
    if (hasvel) { vel[3 * i] = rec[4]; vel[3 * i + 1] = rec[5]; vel[3 * i + 2] = rec[6]; }
  }
  free(dest); free(scount); free(rcount); free(sdisp); free(rdisp);
  free(sbuf); free(rbuf); free(fill);
  *opos = pos; *ovel = vel; *omass = mass; *on = tot_recv;
  return CK_OK;
}
#endif

int ck_assign(ck_grid *rho, ck_grid *mom[3], const ck_particles *p, ck_scheme scheme) {
  if (!rho || !p) CK_FAIL(CK_ERR_ARG, "ck_assign: null argument");
  if (scheme < CK_NGP || scheme > CK_TSC) CK_FAIL(CK_ERR_ARG, "bad assignment scheme");
  if (p->n < 0 || (p->n > 0 && !p->pos)) CK_FAIL(CK_ERR_ARG, "bad particle array");
  ck_context *c = rho->ctx;
  int rc = ck_is_fourier(rho, 0);
  if (rc) return rc;
  const int hasvel = (mom != NULL);
  if (hasvel) {
    if (!p->vel && p->n > 0) CK_FAIL(CK_ERR_ARG, "momentum requested but no velocities given");
    for (int d = 0; d < 3; d++) {
      if (!mom[d]) CK_FAIL(CK_ERR_ARG, "null momentum grid");
      if ((rc = ck_grid_check_compat(rho, mom[d]))) return rc;
      if ((rc = ck_is_fourier(mom[d], 0))) return rc;
    }
  }
  const int nf = hasvel ? 4 : 1;
  double *field[MAX_FIELDS] = {rho->data, hasvel ? mom[0]->data : NULL,
                               hasvel ? mom[1]->data : NULL, hasvel ? mom[2]->data : NULL};

  clear_ghosts(c, rho->data);
  if (hasvel) for (int d = 0; d < 3; d++) clear_ghosts(c, mom[d]->data);

  int64_t cnt = 0;
  double sw = 0.0, sw2 = 0.0;

  if (!c->use_mpi) {
    deposit_local(c, field, nf, hasvel, p, scheme, &cnt, &sw, &sw2);
  } else {
#ifdef CK_ENABLE_MPI
    int64_t rounds = (p->n + CHUNK_PARTICLES - 1) / CHUNK_PARTICLES;
    if ((rc = ck_allreduce_max_i64(c, &rounds, 1))) return rc;
    for (int64_t r = 0; r < rounds; r++) {
      int64_t a = r * CHUNK_PARTICLES, b = a + CHUNK_PARTICLES;
      if (a > p->n) a = p->n;
      if (b > p->n) b = p->n;
      double *pos = NULL, *vel = NULL, *mass = NULL;
      int64_t nn = 0;
      if ((rc = redistribute(c, p, a, b, hasvel, &pos, &vel, &mass, &nn))) return rc;
      ck_particles loc = {nn, pos, CK_FLOAT64, vel, CK_FLOAT64, mass, CK_FLOAT64, 1.0};
      deposit_local(c, field, nf, hasvel, &loc, scheme, &cnt, &sw, &sw2);
      free(pos); free(vel); free(mass);
    }
#endif
  }

  if ((rc = ghost_exchange(c, rho->data))) return rc;
  if (hasvel) for (int d = 0; d < 3; d++)
    if ((rc = ghost_exchange(c, mom[d]->data))) return rc;
  /* FFTW works in place on an area overlapping the right ghost plane */
  clear_ghosts(c, rho->data);
  if (hasvel) for (int d = 0; d < 3; d++) clear_ghosts(c, mom[d]->data);

  double s[2] = {sw, sw2};
  int64_t ci = cnt;
  if ((rc = ck_allreduce_sum_d(c, s, 2))) return rc;
  if ((rc = ck_allreduce_sum_i64(c, &ci, 1))) return rc;
  rho->npart += ci;
  rho->sum_w += s[0];
  rho->sum_w2 += s[1];
  return CK_OK;
}

/* --------------------------------------------------------- SPH deposit */
/* Kernel definitions live in sph.c */
int ck_assign_sph(ck_grid *rho, const ck_particles *p, const double *hsml, int kernel) {
  if (!rho || !p || !hsml) CK_FAIL(CK_ERR_ARG, "ck_assign_sph: null argument");
  ck_context *c = rho->ctx;
  if (c->use_mpi) CK_FAIL(CK_ERR_UNSUPPORTED, "ck_assign_sph is shared-memory only");
  int rc = ck_is_fourier(rho, 0);
  if (rc) return rc;
  const int N = c->n;
  const double h = c->box / N, inv_h = 1.0 / h;
  const size_t pitch = c->pitch, plane = c->plane;
  int64_t count = 0;
  double sum_w = 0.0, sum_w2 = 0.0;

#pragma omp parallel reduction(+ : count, sum_w, sum_w2)
  {
    double *wbuf = NULL;
    size_t wcap = 0;
#pragma omp for schedule(dynamic, 256)
    for (int64_t i = 0; i < p->n; i++) {
      const double m = p->mass ? ck_rd(p->mass, p->mass_dtype, i) : p->mass0;
      const double H = hsml[i];
      const double ux = to_grid(ck_rd(p->pos, p->pos_dtype, 3 * i + 0), inv_h, N);
      const double uy = to_grid(ck_rd(p->pos, p->pos_dtype, 3 * i + 1), inv_h, N);
      const double uz = to_grid(ck_rd(p->pos, p->pos_dtype, 3 * i + 2), inv_h, N);
      count++; sum_w += m; sum_w2 += m * m;

      if (!(H > h)) { /* kernel not resolved by the mesh: fall back to CIC */
        long ix[3], iy[3], iz[3];
        double wx[3], wy[3], wz[3];
        const int nx = axis_weights(ux, CK_CIC, ix, wx);
        const int ny = axis_weights(uy, CK_CIC, iy, wy);
        const int nz = axis_weights(uz, CK_CIC, iz, wz);
        for (int a = 0; a < nx; a++) for (int b = 0; b < ny; b++) for (int d = 0; d < nz; d++) {
          const size_t o = ck_ridx(c, wrap_idx(ix[a], N), (size_t)wrap_idx(iy[b], N),
                                   (size_t)wrap_idx(iz[d], N));
#pragma omp atomic update
          rho->data[o] += m * wx[a] * wy[b] * wz[d];
        }
        continue;
      }
      const double Hg = H * inv_h;
      const long lo[3] = {(long)floor(ux - Hg), (long)floor(uy - Hg), (long)floor(uz - Hg)};
      const long hi[3] = {(long)floor(ux + Hg), (long)floor(uy + Hg), (long)floor(uz + Hg)};
      const long nxc = hi[0] - lo[0] + 1, nyc = hi[1] - lo[1] + 1, nzc = hi[2] - lo[2] + 1;
      const size_t need = (size_t)(nxc * nyc * nzc);
      if (need > wcap) {
        free(wbuf);
        wcap = need * 2;
        wbuf = malloc(wcap * sizeof(double));
      }
      double wsum = 0.0;
      size_t t = 0;
      for (long a = lo[0]; a <= hi[0]; a++)
        for (long b = lo[1]; b <= hi[1]; b++)
          for (long d = lo[2]; d <= hi[2]; d++, t++) {
            const double dx = (a + 0.5 - ux) * h, dy = (b + 0.5 - uy) * h, dz = (d + 0.5 - uz) * h;
            const double w = ck_kernel_w((ck_kernel)kernel, sqrt(dx * dx + dy * dy + dz * dz), H);
            wbuf[t] = w;
            wsum += w;
          }
      if (!(wsum > 0.0)) { /* degenerate: dump into the home cell */
        const size_t o = ck_ridx(c, wrap_idx((long)floor(ux), N), (size_t)wrap_idx((long)floor(uy), N),
                                 (size_t)wrap_idx((long)floor(uz), N));
#pragma omp atomic update
        rho->data[o] += m;
        continue;
      }
      const double norm = m / wsum;
      t = 0;
      for (long a = lo[0]; a <= hi[0]; a++) {
        const size_t xa = (size_t)wrap_idx(a, N);
        for (long b = lo[1]; b <= hi[1]; b++) {
          const size_t yb = (size_t)wrap_idx(b, N);
          const size_t row = xa * N * pitch + yb * pitch;
          for (long d = lo[2]; d <= hi[2]; d++, t++) {
            if (wbuf[t] == 0.0) continue;
#pragma omp atomic update
            rho->data[row + (size_t)wrap_idx(d, N)] += norm * wbuf[t];
          }
        }
      }
    }
    free(wbuf);
  }
  (void)plane;
  rho->npart += count;
  rho->sum_w += sum_w;
  rho->sum_w2 += sum_w2;
  return CK_OK;
}
