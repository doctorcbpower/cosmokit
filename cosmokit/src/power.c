/* power.c - binned auto/cross power spectra (with multipoles), velocity spectra */
#include "ck_internal.h"

double ck_shot_axis(int p, double s) {
  /* sum over aliases of W^2(k + 2 k_N n), Jing (2005) eq. 19 */
  switch (p) {
    case CK_NGP: return 1.0;
    case CK_CIC: return 1.0 - 2.0 * s / 3.0;
    case CK_TSC: return 1.0 - s + 2.0 * s * s / 15.0;
    default: return 1.0;
  }
}

void ck_pk_opts_default(ck_pk_opts *o) {
  o->dk = 1.0;
  o->kmax = 0.0;
  o->window = 0;
  o->shot = 0;
  o->nell = 1;
  o->los = 2;
}

static void resolve_opts(const ck_context *c, const ck_pk_opts *in, double *dk, double *kmax) {
  *dk = (in && in->dk > 0.0) ? in->dk : 1.0;
  *kmax = (in && in->kmax > 0.0) ? in->kmax : 0.5 * c->n;
}

int ck_pk_nbins(const ck_context *c, const ck_pk_opts *o) {
  double dk, kmax;
  resolve_opts(c, o, &dk, &kmax);
  int nb = (int)ceil(kmax / dk - 1e-9);
  return nb < 1 ? 1 : nb;
}

static inline double legendre(int ell, double mu) {
  const double m2 = mu * mu;
  switch (ell) {
    case 0: return 1.0;
    case 1: return 0.5 * (3.0 * m2 - 1.0);               /* P2 */
    default: return (35.0 * m2 * m2 - 30.0 * m2 + 3.0) / 8.0; /* P4 */
  }
}

/* per-axis tables indexed by |n| = 0..N/2 */
static void axis_tables(const ck_context *c, int p, double **w2, double **shot) {
  const int N = c->n, H = N / 2;
  *w2 = malloc((H + 2) * sizeof(double));
  *shot = malloc((H + 2) * sizeof(double));
  for (int n = 0; n <= H; n++) {
    const double x = M_PI * n / N;
    const double s = sin(x) * sin(x);
    (*w2)[n] = p > 0 ? pow(ck_sinc(x), 2.0 * p) : 1.0;
    (*shot)[n] = p > 0 ? ck_shot_axis(p, s) : 1.0;
  }
}

static int finish(const ck_context *c, double *acc, int nbins, int stride, int nrows,
                  ck_pk_result *res, double dk_unused) {
  (void)dk_unused;
  int rc = ck_allreduce_sum_d(c, acc, nbins * stride);
  if (rc) return rc;
  const double kf = 2.0 * M_PI / c->box;
  for (int b = 0; b < nbins; b++) {
    const double nm = acc[(size_t)b * stride + 0];
    const double ks = acc[(size_t)b * stride + 1];
    res->nmodes[b] = (int64_t)(nm + 0.5);
    res->k[b] = nm > 0 ? kf * ks / nm : 0.0;
    for (int r = 0; r < nrows; r++)
      res->pk[(size_t)r * nbins + b] = nm > 0 ? acc[(size_t)b * stride + 2 + r] / nm : 0.0;
  }
  return CK_OK;
}

int ck_power_spectrum(const ck_grid *a, const ck_grid *bgrid, const ck_pk_opts *o_in,
                      ck_pk_result *res) {
  if (!a || !res || !res->k || !res->nmodes || !res->pk) CK_FAIL(CK_ERR_ARG, "null argument");
  int rc = ck_is_fourier(a, 1);
  if (rc) return rc;
  if (bgrid) {
    if ((rc = ck_grid_check_compat(a, bgrid))) return rc;
    if ((rc = ck_is_fourier(bgrid, 1))) return rc;
  }
  ck_pk_opts o;
  if (o_in) o = *o_in; else ck_pk_opts_default(&o);
  if (o.nell < 1 || o.nell > 3) CK_FAIL(CK_ERR_ARG, "nell must be 1..3");
  if (o.los < 0 || o.los > 2) CK_FAIL(CK_ERR_ARG, "los must be 0,1,2");
  if (o.window < 0 || o.window > 3) CK_FAIL(CK_ERR_ARG, "bad window order");

  const ck_context *c = a->ctx;
  const int N = c->n, nbins = ck_pk_nbins(c, &o);
  double dk, kmax;
  resolve_opts(c, &o, &dk, &kmax);
  const double V = c->box * c->box * c->box;
  const int nell = o.nell;
  const int stride = 2 + nell;

  double shot0 = 0.0;
  if (o.shot && !bgrid) {
    if (a->sum_w <= 0.0) CK_FAIL(CK_ERR_ARG, "shot noise needs particle statistics (grid made by ck_assign)");
    shot0 = V * a->sum_w2 / (a->sum_w * a->sum_w);
  }
  double *w2tab, *shtab;
  axis_tables(c, o.window, &w2tab, &shtab);

  const int nth = ck_max_threads();
  double *acc_all = calloc((size_t)nth * nbins * stride, sizeof(double));
  if (!acc_all) { free(w2tab); free(shtab); CK_FAIL(CK_ERR_ALLOC, "pk accumulators"); }

#pragma omp parallel
  {
    double *acc = acc_all + (size_t)ck_thread_num() * nbins * stride;
#pragma omp for schedule(static)
    for (size_t il = 0; il < (size_t)c->local_n0; il++) {
      const int fx = ck_freq((int)(c->local_0_start + il), N);
      for (int iy = 0; iy < N; iy++) {
        const int fy = ck_freq(iy, N);
        for (size_t iz = 0; iz < c->nzh; iz++) {
          const int fz = (int)iz;
          const double k2 = (double)fx * fx + (double)fy * fy + (double)fz * fz;
          if (k2 == 0.0) continue;
          const double kmag = sqrt(k2);
          const int b = (int)(kmag / dk);
          if (b >= nbins || kmag > kmax + 1e-9) continue;
          const size_t idx = 2 * ck_kidx(c, il, iy, iz);
          double P;
          if (bgrid)
            P = a->data[idx] * bgrid->data[idx] + a->data[idx + 1] * bgrid->data[idx + 1];
          else
            P = a->data[idx] * a->data[idx] + a->data[idx + 1] * a->data[idx + 1];
          P *= V;
          const int ax = abs(fx), ay = abs(fy), az = fz;
          if (shot0 > 0.0) P -= shot0 * shtab[ax] * shtab[ay] * shtab[az];
          if (o.window) P /= w2tab[ax] * w2tab[ay] * w2tab[az];
          const double wt = (iz == 0 || ck_is_nyq((int)iz, N)) ? 1.0 : 2.0;
          double *bin = acc + (size_t)b * stride;
          bin[0] += wt;
          bin[1] += wt * kmag;
          const int kl = o.los == 0 ? fx : (o.los == 1 ? fy : fz);
          const double mu = (double)kl / kmag;
          for (int l = 0; l < nell; l++)
            bin[2 + l] += wt * P * (4.0 * l + 1.0) * legendre(l, mu);
        }
      }
    }
  }
  free(w2tab); free(shtab);
  for (int t = 1; t < nth; t++)
    for (size_t i = 0; i < (size_t)nbins * stride; i++) acc_all[i] += acc_all[(size_t)t * nbins * stride + i];
  rc = finish(c, acc_all, nbins, stride, nell, res, dk);
  free(acc_all);
  return rc;
}

int ck_vector_power(const ck_grid *v[3], const ck_grid *delta, const ck_pk_opts *o_in,
                    ck_pk_result *res) {
  if (!v || !v[0] || !v[1] || !v[2] || !res || !res->k || !res->nmodes || !res->pk)
    CK_FAIL(CK_ERR_ARG, "null argument");
  int rc;
  for (int d = 0; d < 3; d++) {
    if ((rc = ck_is_fourier(v[d], 1))) return rc;
    if ((rc = ck_grid_check_compat(v[0], v[d]))) return rc;
  }
  if (delta) {
    if ((rc = ck_is_fourier(delta, 1))) return rc;
    if ((rc = ck_grid_check_compat(v[0], delta))) return rc;
  }
  ck_pk_opts o;
  if (o_in) o = *o_in; else ck_pk_opts_default(&o);
  const ck_context *c = v[0]->ctx;
  const int N = c->n, nbins = ck_pk_nbins(c, &o);
  double dk, kmax;
  resolve_opts(c, &o, &dk, &kmax);
  const double V = c->box * c->box * c->box;
  const double kf = 2.0 * M_PI / c->box;
  const int nrows = 5, stride = 2 + nrows;
  double *w2tab, *shtab;
  axis_tables(c, o.window, &w2tab, &shtab);

  const int nth = ck_max_threads();
  double *acc_all = calloc((size_t)nth * nbins * stride, sizeof(double));
  if (!acc_all) { free(w2tab); free(shtab); CK_FAIL(CK_ERR_ALLOC, "pk accumulators"); }

#pragma omp parallel
  {
    double *acc = acc_all + (size_t)ck_thread_num() * nbins * stride;
#pragma omp for schedule(static)
    for (size_t il = 0; il < (size_t)c->local_n0; il++) {
      const int fx = ck_freq((int)(c->local_0_start + il), N);
      for (int iy = 0; iy < N; iy++) {
        const int fy = ck_freq(iy, N);
        for (size_t iz = 0; iz < c->nzh; iz++) {
          const int fz = (int)iz;
          const double k2 = (double)fx * fx + (double)fy * fy + (double)fz * fz;
          if (k2 == 0.0) continue;
          const double kmag = sqrt(k2);
          const int b = (int)(kmag / dk);
          if (b >= nbins || kmag > kmax + 1e-9) continue;
          const size_t idx = 2 * ck_kidx(c, il, iy, iz);
          const double fk[3] = {fx, fy, fz};
          double tot = 0.0, Kr = 0.0, Ki = 0.0;
          for (int d = 0; d < 3; d++) {
            const double re = v[d]->data[idx], im = v[d]->data[idx + 1];
            tot += re * re + im * im;
            Kr += fk[d] * re;
            Ki += fk[d] * im;
          }
          double lng = (Kr * Kr + Ki * Ki) / k2;
          double dth = 0.0;
          if (delta) dth = kf * (-delta->data[idx] * Ki + delta->data[idx + 1] * Kr);
          double wcorr = 1.0;
          if (o.window) wcorr = w2tab[abs(fx)] * w2tab[abs(fy)] * w2tab[fz];
          tot *= V / wcorr;
          lng *= V / wcorr;
          dth *= V / wcorr;
          const double wt = (iz == 0 || ck_is_nyq((int)iz, N)) ? 1.0 : 2.0;
          double *bin = acc + (size_t)b * stride;
          bin[0] += wt;
          bin[1] += wt * kmag;
          bin[2] += wt * tot;
          bin[3] += wt * lng;
          bin[4] += wt * (tot - lng);
          bin[5] += wt * kf * kf * k2 * lng; /* k_phys^2 * long */
          bin[6] += wt * dth;
        }
      }
    }
  }
  free(w2tab); free(shtab);
  for (int t = 1; t < nth; t++)
    for (size_t i = 0; i < (size_t)nbins * stride; i++) acc_all[i] += acc_all[(size_t)t * nbins * stride + i];
  rc = finish(c, acc_all, nbins, stride, nrows, res, dk);
  free(acc_all);
  return rc;
}
