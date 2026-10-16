/* sph.c - SPH kernels and kNN-based density / smoothing-length estimation */
#include "ck_internal.h"

double ck_kernel_w(ck_kernel k, double r, double H) {
  if (!(H > 0.0)) return 0.0;
  const double q = r / H;
  if (q >= 1.0) return 0.0;
  const double H3 = H * H * H;
  switch (k) {
    case CK_KERNEL_WENDLAND_C2: {
      const double u = 1.0 - q;
      return 21.0 / (2.0 * M_PI * H3) * u * u * u * u * (1.0 + 4.0 * q);
    }
    case CK_KERNEL_CUBIC:
    default:
      if (q <= 0.5) return 8.0 / (M_PI * H3) * (1.0 - 6.0 * q * q + 6.0 * q * q * q);
      else {
        const double u = 1.0 - q;
        return 16.0 / (M_PI * H3) * u * u * u;
      }
  }
}

static int sph_core(const ck_kdtree *t, const double *mass, const void *q, ck_dtype qtype,
                    int64_t m, int nngb, ck_kernel kernel, double hfac, int self, int add_self,
                    double *h_out, double *rho_out) {
  if (!t || nngb < 1 || !(hfac > 0.0)) CK_FAIL(CK_ERR_ARG, "bad sph arguments");
  if (nngb + (self ? 1 : 0) > t->n) CK_FAIL(CK_ERR_ARG, "nngb larger than number of points");
  const int64_t nq = self ? t->n : m;
#pragma omp parallel
  {
    double *d2 = malloc((size_t)nngb * sizeof(double));
    int64_t *ii = malloc((size_t)nngb * sizeof(int64_t));
#pragma omp for schedule(dynamic, 64)
    for (int64_t j = 0; j < nq; j++) {
      double x[3];
      int64_t orig;
      if (self) { /* iterate in tree order for memory locality */
        memcpy(x, t->pt[j].x, sizeof x);
        orig = t->pt[j].i;
      } else {
        for (int d = 0; d < 3; d++) {
          double v = ck_rd(q, qtype, 3 * j + d);
          if (t->box > 0.0) { v -= t->box * floor(v / t->box); if (v >= t->box) v = 0.0; }
          x[d] = v;
        }
        orig = j;
      }
      const int got = ck_kdtree_knn_one(t, x, nngb, self ? orig : -1, d2, ii);
      const double H = got > 0 ? hfac * sqrt(d2[got - 1]) : 0.0;
      double rho = 0.0;
      if (self && add_self) rho += (mass ? mass[orig] : 1.0) * ck_kernel_w(kernel, 0.0, H);
      for (int a = 0; a < got; a++)
        rho += (mass ? mass[ii[a]] : 1.0) * ck_kernel_w(kernel, sqrt(d2[a]), H);
      if (h_out) h_out[orig] = H;
      if (rho_out) rho_out[orig] = rho;
    }
    free(d2);
    free(ii);
  }
  return CK_OK;
}

int ck_sph_density(const ck_kdtree *t, const double *mass, int nngb, ck_kernel kernel,
                   double hfac, unsigned flags, double *h_out, double *rho_out) {
  return sph_core(t, mass, NULL, CK_FLOAT64, 0, nngb, kernel, hfac, 1,
                  !(flags & CK_SPH_NO_SELF), h_out, rho_out);
}

int ck_sph_density_at(const ck_kdtree *t, const double *mass, const void *q, ck_dtype qtype,
                      int64_t m, int nngb, ck_kernel kernel, double hfac, double *h_out,
                      double *rho_out) {
  if (!q) CK_FAIL(CK_ERR_ARG, "null query array");
  return sph_core(t, mass, q, qtype, m, nngb, kernel, hfac, 0, 0, h_out, rho_out);
}
