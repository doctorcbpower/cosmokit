/* tidal.c - Fourier-space filters, potential, tidal tensor, velocity fields and
 * their derivatives (divergence / shear / vorticity), eigenvalues and web
 * classification.  All spectral; Nyquist modes are zeroed for odd derivatives
 * so the real-space result stays real. */
#include "ck_internal.h"

/* Loop over every local k-mode; body sees: idx (double offset), kx,ky,kz as
 * physical wavenumbers, k2 physical, and nyquist flags. */
#define KLOOP_BEGIN(c)                                                              \
  _Pragma("omp parallel for schedule(static)")                                       \
  for (size_t il = 0; il < (size_t)(c)->local_n0; il++) {                            \
    const int ix_ = ck_freq((int)((c)->local_0_start + il), (c)->n);                 \
    const int nyx_ = ck_is_nyq((int)((c)->local_0_start + il), (c)->n);              \
    for (int iy = 0; iy < (c)->n; iy++) {                                            \
      const int iy_f = ck_freq(iy, (c)->n);                                          \
      const int nyy_ = ck_is_nyq(iy, (c)->n);                                        \
      for (size_t iz = 0; iz < (c)->nzh; iz++) {                                     \
        const int nyz_ = ck_is_nyq((int)iz, (c)->n);                                 \
        const size_t idx = 2 * ck_kidx((c), il, iy, iz);                             \
        const double kx = kf * ix_, ky = kf * iy_f, kz = kf * (double)iz;            \
        const double k2 = kx * kx + ky * ky + kz * kz;                               \
        (void)nyx_; (void)nyy_; (void)nyz_; (void)idx; (void)k2;

#define KLOOP_END }}}

/* ------------------------------------------------------------- filters */
int ck_filter(ck_grid *g, ck_filter_type type, double R) {
  int rc = ck_is_fourier(g, 1);
  if (rc) return rc;
  if (!(R >= 0.0)) CK_FAIL(CK_ERR_ARG, "negative filter radius");
  const ck_context *c = g->ctx;
  const double kf = 2.0 * M_PI / c->box;
  KLOOP_BEGIN(c)
    const double kk = sqrt(k2);
    double f;
    if (type == CK_FILTER_GAUSSIAN) f = exp(-0.5 * k2 * R * R);
    else {
      const double x = kk * R;
      f = x < 1e-4 ? 1.0 - 0.1 * x * x : 3.0 * (sin(x) - x * cos(x)) / (x * x * x);
    }
    g->data[idx] *= f;
    g->data[idx + 1] *= f;
  KLOOP_END
  return CK_OK;
}

int ck_deconvolve(ck_grid *g, ck_scheme scheme) {
  int rc = ck_is_fourier(g, 1);
  if (rc) return rc;
  const ck_context *c = g->ctx;
  const int N = c->n;
  const double kf = 1.0; /* work in index units here */
  double *tab = malloc((N / 2 + 2) * sizeof(double));
  for (int n = 0; n <= N / 2; n++) tab[n] = pow(ck_sinc(M_PI * n / N), (double)scheme);
  KLOOP_BEGIN(c)
    (void)kx; (void)ky; (void)kz;
    const double w = tab[abs(ix_)] * tab[abs(iy_f)] * tab[iz];
    g->data[idx] /= w;
    g->data[idx + 1] /= w;
  KLOOP_END
  free(tab);
  return CK_OK;
}

/* ------------------------------------------------------------ potential */
int ck_potential(const ck_grid *delta, double norm, ck_grid *phi) {
  int rc = ck_grid_check_compat(delta, phi);
  if (rc) return rc;
  if ((rc = ck_is_fourier(delta, 1))) return rc;
  const ck_context *c = delta->ctx;
  const double kf = 2.0 * M_PI / c->box;
  KLOOP_BEGIN(c)
    const double f = k2 > 0.0 ? -norm / k2 : 0.0;
    const double re = delta->data[idx], im = delta->data[idx + 1];
    phi->data[idx] = f * re;
    phi->data[idx + 1] = f * im;
  KLOOP_END
  phi->fourier = 1;
  return CK_OK;
}

/* --------------------------------------------------------- tidal tensor */
int ck_tidal_tensor(const ck_grid *delta, ck_grid *out[6]) {
  int rc = ck_is_fourier(delta, 1);
  if (rc) return rc;
  for (int t = 0; t < 6; t++)
    if ((rc = ck_grid_check_compat(delta, out[t]))) return rc;
  const ck_context *c = delta->ctx;
  const double kf = 2.0 * M_PI / c->box;
  for (int t = 0; t < 6; t++) {
    ck_grid *o = out[t];
    const int a = t < 3 ? t : (t == 3 ? 0 : (t == 4 ? 0 : 1));
    const int b = t < 3 ? t : (t == 3 ? 1 : (t == 4 ? 2 : 2));
    KLOOP_BEGIN(c)
      const double kv[3] = {kx, ky, kz};
      double f = 0.0;
      if (k2 > 0.0) f = kv[a] * kv[b] / k2 - (a == b ? 1.0 / 3.0 : 0.0);
      const double re = delta->data[idx], im = delta->data[idx + 1];
      o->data[idx] = f * re;
      o->data[idx + 1] = f * im;
    KLOOP_END
    o->fourier = 1;
    if ((rc = ck_grid_fft_inverse(o))) return rc;
  }
  return CK_OK;
}

/* ---------------------------------------------------------- velocities */
int ck_velocity_from_delta(const ck_grid *delta, double faH, ck_grid *vk[3]) {
  int rc = ck_is_fourier(delta, 1);
  if (rc) return rc;
  for (int d = 0; d < 3; d++)
    if ((rc = ck_grid_check_compat(delta, vk[d]))) return rc;
  const ck_context *c = delta->ctx;
  const double kf = 2.0 * M_PI / c->box;
  for (int d = 0; d < 3; d++) {
    ck_grid *o = vk[d];
    KLOOP_BEGIN(c)
      const double kv[3] = {nyx_ ? 0.0 : kx, nyy_ ? 0.0 : ky, nyz_ ? 0.0 : kz};
      const double f = k2 > 0.0 ? faH * kv[d] / k2 : 0.0; /* v = i f k/k^2 delta */
      const double re = delta->data[idx], im = delta->data[idx + 1];
      o->data[idx] = -f * im;
      o->data[idx + 1] = f * re;
    KLOOP_END
    o->fourier = 1;
  }
  return CK_OK;
}

static int check_vec(const ck_grid *vk[3], const ck_grid *ref) {
  int rc;
  for (int d = 0; d < 3; d++) {
    if (!vk[d]) CK_FAIL(CK_ERR_ARG, "null velocity grid");
    if ((rc = ck_grid_check_compat(ref, vk[d]))) return rc;
    if ((rc = ck_is_fourier(vk[d], 1))) return rc;
  }
  return CK_OK;
}

int ck_velocity_divergence(const ck_grid *vk[3], ck_grid *theta) {
  int rc = check_vec(vk, theta);
  if (rc) return rc;
  const ck_context *c = theta->ctx;
  const double kf = 2.0 * M_PI / c->box;
  KLOOP_BEGIN(c)
    const double kv[3] = {nyx_ ? 0.0 : kx, nyy_ ? 0.0 : ky, nyz_ ? 0.0 : kz};
    double re = 0.0, im = 0.0; /* i k.v */
    for (int d = 0; d < 3; d++) {
      re += -kv[d] * vk[d]->data[idx + 1];
      im += kv[d] * vk[d]->data[idx];
    }
    theta->data[idx] = re;
    theta->data[idx + 1] = im;
  KLOOP_END
  theta->fourier = 1;
  return ck_grid_fft_inverse(theta);
}

int ck_velocity_shear(const ck_grid *vk[3], ck_grid *out[6]) {
  int rc = check_vec(vk, out[0]);
  if (rc) return rc;
  for (int t = 1; t < 6; t++)
    if ((rc = ck_grid_check_compat(out[0], out[t]))) return rc;
  const ck_context *c = out[0]->ctx;
  const double kf = 2.0 * M_PI / c->box;
  for (int t = 0; t < 6; t++) {
    ck_grid *o = out[t];
    const int a = t < 3 ? t : (t == 3 ? 0 : (t == 4 ? 0 : 1));
    const int b = t < 3 ? t : (t == 3 ? 1 : (t == 4 ? 2 : 2));
    KLOOP_BEGIN(c)
      const double kv[3] = {nyx_ ? 0.0 : kx, nyy_ ? 0.0 : ky, nyz_ ? 0.0 : kz};
      /* D_ij = i k_i v_j ; sigma_ab = (D_ab + D_ba)/2 - delta_ab theta/3 */
      double re = 0.0, im = 0.0;
      /* i * (k_a v_b + k_b v_a)/2 */
      re = -0.5 * (kv[a] * vk[b]->data[idx + 1] + kv[b] * vk[a]->data[idx + 1]);
      im = 0.5 * (kv[a] * vk[b]->data[idx] + kv[b] * vk[a]->data[idx]);
      if (a == b) {
        double tr = 0.0, ti = 0.0;
        for (int d = 0; d < 3; d++) {
          tr += -kv[d] * vk[d]->data[idx + 1];
          ti += kv[d] * vk[d]->data[idx];
        }
        re -= tr / 3.0;
        im -= ti / 3.0;
      }
      o->data[idx] = re;
      o->data[idx + 1] = im;
    KLOOP_END
    o->fourier = 1;
    if ((rc = ck_grid_fft_inverse(o))) return rc;
  }
  return CK_OK;
}

int ck_velocity_vorticity(const ck_grid *vk[3], ck_grid *out[3]) {
  int rc = check_vec(vk, out[0]);
  if (rc) return rc;
  for (int t = 1; t < 3; t++)
    if ((rc = ck_grid_check_compat(out[0], out[t]))) return rc;
  const ck_context *c = out[0]->ctx;
  const double kf = 2.0 * M_PI / c->box;
  for (int t = 0; t < 3; t++) {
    ck_grid *o = out[t];
    const int a = (t + 1) % 3, b = (t + 2) % 3; /* omega_t = d_a v_b - d_b v_a */
    KLOOP_BEGIN(c)
      const double kv[3] = {nyx_ ? 0.0 : kx, nyy_ ? 0.0 : ky, nyz_ ? 0.0 : kz};
      const double re = -(kv[a] * vk[b]->data[idx + 1] - kv[b] * vk[a]->data[idx + 1]);
      const double im = kv[a] * vk[b]->data[idx] - kv[b] * vk[a]->data[idx];
      o->data[idx] = re;
      o->data[idx + 1] = im;
    KLOOP_END
    o->fourier = 1;
    if ((rc = ck_grid_fft_inverse(o))) return rc;
  }
  return CK_OK;
}

/* ---------------------------------------------- eigenvalues / web class */
/* Eigenvalues of a symmetric 3x3 matrix, descending (Smith 1961). */
static void sym3_eig(double xx, double yy, double zz, double xy, double xz, double yz,
                     double *l) {
  const double p1 = xy * xy + xz * xz + yz * yz;
  if (p1 < 1e-300) {
    double a = xx, b = yy, c = zz, t;
    if (a < b) { t = a; a = b; b = t; }
    if (b < c) { t = b; b = c; c = t; }
    if (a < b) { t = a; a = b; b = t; }
    l[0] = a; l[1] = b; l[2] = c;
    return;
  }
  const double q = (xx + yy + zz) / 3.0;
  const double p2 = (xx - q) * (xx - q) + (yy - q) * (yy - q) + (zz - q) * (zz - q) + 2.0 * p1;
  const double p = sqrt(p2 / 6.0);
  const double bxx = (xx - q) / p, byy = (yy - q) / p, bzz = (zz - q) / p;
  const double bxy = xy / p, bxz = xz / p, byz = yz / p;
  double r = 0.5 * (bxx * (byy * bzz - byz * byz) - bxy * (bxy * bzz - byz * bxz) +
                    bxz * (bxy * byz - byy * bxz));
  if (r < -1.0) r = -1.0;
  if (r > 1.0) r = 1.0;
  const double phi = acos(r) / 3.0;
  l[0] = q + 2.0 * p * cos(phi);
  l[2] = q + 2.0 * p * cos(phi + 2.0 * M_PI / 3.0);
  l[1] = 3.0 * q - l[0] - l[2];
}

int ck_tensor_eigenvalues(const ck_grid *t[6], ck_grid *l[3]) {
  int rc;
  for (int i = 0; i < 6; i++) {
    if ((rc = ck_grid_check_compat(l[0], t[i]))) return rc;
    if ((rc = ck_is_fourier(t[i], 0))) return rc;
  }
  for (int i = 0; i < 3; i++) {
    if ((rc = ck_grid_check_compat(l[0], l[i]))) return rc;
  }
  const ck_context *c = l[0]->ctx;
  const size_t N = c->n;
#pragma omp parallel for collapse(2) schedule(static)
  for (ptrdiff_t il = 0; il < c->local_n0; il++)
    for (size_t iy = 0; iy < N; iy++) {
      const size_t o = ck_ridx(c, il, iy, 0);
      for (size_t iz = 0; iz < N; iz++) {
        double ev[3];
        sym3_eig(t[0]->data[o + iz], t[1]->data[o + iz], t[2]->data[o + iz],
                 t[3]->data[o + iz], t[4]->data[o + iz], t[5]->data[o + iz], ev);
        l[0]->data[o + iz] = ev[0];
        l[1]->data[o + iz] = ev[1];
        l[2]->data[o + iz] = ev[2];
      }
    }
  for (int i = 0; i < 3; i++) l[i]->fourier = 0;
  return CK_OK;
}

int ck_web_classify(const ck_grid *l[3], double lambda_th, ck_grid *web) {
  int rc;
  for (int i = 0; i < 3; i++) {
    if ((rc = ck_grid_check_compat(web, l[i]))) return rc;
    if ((rc = ck_is_fourier(l[i], 0))) return rc;
  }
  const ck_context *c = web->ctx;
  const size_t N = c->n;
#pragma omp parallel for collapse(2) schedule(static)
  for (ptrdiff_t il = 0; il < c->local_n0; il++)
    for (size_t iy = 0; iy < N; iy++) {
      const size_t o = ck_ridx(c, il, iy, 0);
      for (size_t iz = 0; iz < N; iz++)
        web->data[o + iz] = (double)((l[0]->data[o + iz] > lambda_th) +
                                     (l[1]->data[o + iz] > lambda_th) +
                                     (l[2]->data[o + iz] > lambda_th));
    }
  web->fourier = 0;
  return CK_OK;
}
