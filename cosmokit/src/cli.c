/* cli.c - command line driver: snapshot -> density/velocity power spectra,
 * optionally tidal-web fields, with MPI + OpenMP.
 *
 *   [mpirun -np P] cosmokit --input snap.hdf5 --nmesh 256 --output pk.txt [options]
 */
#include "ck_internal.h"

#include <time.h>

#ifdef CK_ENABLE_HDF5
#include <hdf5.h>
#endif

static int g_rank = 0;

static double wtime(void) {
#ifdef CK_ENABLE_MPI
  return MPI_Wtime();
#else
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec + 1e-9 * ts.tv_nsec;
#endif
}

static void die(int rc, const char *where) {
  fprintf(stderr, "[rank %d] cosmokit error in %s: %s (%s)\n", g_rank, where, ck_strerror(rc),
          ck_last_error());
#ifdef CK_ENABLE_MPI
  MPI_Abort(MPI_COMM_WORLD, 1);
#endif
  exit(1);
}
#define CHECK(call) do { int rc_ = (call); if (rc_) die(rc_, #call); } while (0)

static void log0(const char *fmt, ...) {
  if (g_rank) return;
  va_list ap;
  va_start(ap, fmt);
  vfprintf(stdout, fmt, ap);
  va_end(ap);
  fflush(stdout);
}

static void usage(const char *prog) {
  if (g_rank) return;
  fprintf(stderr,
    "cosmokit %s - gridded fields and power spectra for N-body snapshots\n\n"
    "usage: %s --input FILE --nmesh N --output FILE [options]\n\n"
    "  --input FILE        snapshot: Gadget-4/Arepo/SWIFT HDF5 or Gadget-1/2 binary (auto-detected)\n"
    "  --nmesh N           FFT mesh size per dimension (N^3 cells)\n"
    "  --output FILE       density power spectrum (text)\n"
    "  --ptype MASK        particle types: list e.g. 1,2 or bitmask 0x6 (default 1)\n"
    "  --scheme S          ngp | cic | tsc (default cic)\n"
    "  --box L             box size (default: Header/BoxSize)\n"
    "  --dk X              k-bin width in units of 2pi/L (default 1)\n"
    "  --nell M            1 monopole, 2 +quadrupole, 3 +hexadecapole (default 1)\n"
    "  --los AXIS          line-of-sight axis x|y|z for multipoles (default z)\n"
    "  --no-deconvolve     do not divide by the mass-assignment window\n"
    "  --no-shot           do not subtract Poisson shot noise\n"
    "  --velocity          also measure mass-weighted velocity spectra (--vel-output)\n"
    "  --vel-output FILE   velocity spectra (default <output>.vel)\n"
    "  --vel-factor F      multiply velocities by F (e.g. sqrt(a) for Gadget-2 style)\n"
    "  --tidal R           also compute tidal tensor of delta smoothed with Gaussian R,\n"
    "                      its eigenvalues and web class (needs --save-fields)\n"
    "  --web-threshold T   eigenvalue threshold for web classification (default 0)\n"
    "  --save-fields PFX   write real-space fields to PFX.rank<r>.h5 (HDF5 builds)\n"
    "  --chunk N           particles per read chunk (default 4194304)\n"
    "  --threads N         OpenMP threads (default: environment)\n"
    "  -h, --help\n", CK_VERSION, prog);
}

static int parse_ptypes(const char *s, unsigned *mask) {
  if (strncmp(s, "0x", 2) == 0) { *mask = (unsigned)strtoul(s, NULL, 16); return *mask != 0; }
  unsigned m = 0;
  char *dup = strdup(s), *tok, *save = NULL;
  for (tok = strtok_r(dup, ",", &save); tok; tok = strtok_r(NULL, ",", &save)) {
    int t = atoi(tok);
    if (t < 0 || t > 5) { free(dup); return 0; }
    m |= 1u << t;
  }
  free(dup);
  *mask = m;
  return m != 0;
}

#ifdef CK_ENABLE_HDF5
static void save_field(const char *prefix, const char *name, const ck_grid *g, int first) {
  char fname[1100];
  snprintf(fname, sizeof fname, "%s.rank%d.h5", prefix, g_rank);
  const ck_context *c = g->ctx;
  hid_t f = first ? H5Fcreate(fname, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT)
                  : H5Fopen(fname, H5F_ACC_RDWR, H5P_DEFAULT);
  if (f < 0) die(CK_ERR_IO, "save_field open");
  if (first) {
    hid_t as = H5Screate(H5S_SCALAR);
    int64_t v;
    hid_t a;
    v = c->n; a = H5Acreate2(f, "Nmesh", H5T_NATIVE_INT64, as, H5P_DEFAULT, H5P_DEFAULT); H5Awrite(a, H5T_NATIVE_INT64, &v); H5Aclose(a);
    v = c->local_0_start; a = H5Acreate2(f, "LocalStartX", H5T_NATIVE_INT64, as, H5P_DEFAULT, H5P_DEFAULT); H5Awrite(a, H5T_NATIVE_INT64, &v); H5Aclose(a);
    v = c->local_n0; a = H5Acreate2(f, "LocalNx", H5T_NATIVE_INT64, as, H5P_DEFAULT, H5P_DEFAULT); H5Awrite(a, H5T_NATIVE_INT64, &v); H5Aclose(a);
    double b = c->box; a = H5Acreate2(f, "BoxSize", H5T_NATIVE_DOUBLE, as, H5P_DEFAULT, H5P_DEFAULT); H5Awrite(a, H5T_NATIVE_DOUBLE, &b); H5Aclose(a);
    H5Sclose(as);
  }
  const size_t n = (size_t)c->local_n0 * c->n * c->n;
  double *buf = malloc(n * sizeof(double));
  if (!buf) die(CK_ERR_ALLOC, "save_field");
  CHECK(ck_grid_get_real(g, buf));
  hsize_t dims[3] = {(hsize_t)c->local_n0, (hsize_t)c->n, (hsize_t)c->n};
  hid_t sp = H5Screate_simple(3, dims, NULL);
  hid_t ds = H5Dcreate2(f, name, H5T_NATIVE_DOUBLE, sp, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  H5Dwrite(ds, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, buf);
  H5Dclose(ds);
  H5Sclose(sp);
  H5Fclose(f);
  free(buf);
}
#endif

int main(int argc, char **argv) {
#ifdef CK_ENABLE_MPI
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &g_rank);
  int comm_f = (int)MPI_Comm_c2f(MPI_COMM_WORLD);
#else
  int comm_f = -1;
#endif

  const char *input = NULL, *output = NULL, *vel_output = NULL, *save_prefix = NULL;
  int nmesh = 0, nthreads = 0, nell = 1, los = 2, deconv = 1, shot = 1, velocity = 0;
  unsigned ptypes = 1u << 1;
  ck_scheme scheme = CK_CIC;
  double box = 0.0, dk = 1.0, vel_factor = 1.0, tidal_R = -1.0, web_th = 0.0;
  int64_t chunk = 1 << 22;

  for (int i = 1; i < argc; i++) {
    const char *a = argv[i];
#define NEXT (i + 1 < argc ? argv[++i] : (usage(argv[0]), die(CK_ERR_ARG, "missing value"), (char *)NULL))
    if (!strcmp(a, "--input")) input = NEXT;
    else if (!strcmp(a, "--output")) output = NEXT;
    else if (!strcmp(a, "--vel-output")) vel_output = NEXT;
    else if (!strcmp(a, "--save-fields")) save_prefix = NEXT;
    else if (!strcmp(a, "--nmesh")) nmesh = atoi(NEXT);
    else if (!strcmp(a, "--threads")) nthreads = atoi(NEXT);
    else if (!strcmp(a, "--nell")) nell = atoi(NEXT);
    else if (!strcmp(a, "--box")) box = atof(NEXT);
    else if (!strcmp(a, "--dk")) dk = atof(NEXT);
    else if (!strcmp(a, "--vel-factor")) vel_factor = atof(NEXT);
    else if (!strcmp(a, "--tidal")) tidal_R = atof(NEXT);
    else if (!strcmp(a, "--web-threshold")) web_th = atof(NEXT);
    else if (!strcmp(a, "--chunk")) chunk = atoll(NEXT);
    else if (!strcmp(a, "--ptype")) {
      if (!parse_ptypes(NEXT, &ptypes)) { usage(argv[0]); die(CK_ERR_ARG, "bad --ptype"); }
    } else if (!strcmp(a, "--scheme")) {
      const char *s = NEXT;
      if (!strcmp(s, "ngp")) scheme = CK_NGP;
      else if (!strcmp(s, "cic")) scheme = CK_CIC;
      else if (!strcmp(s, "tsc")) scheme = CK_TSC;
      else { usage(argv[0]); die(CK_ERR_ARG, "bad --scheme"); }
    } else if (!strcmp(a, "--los")) {
      const char *s = NEXT;
      los = s[0] == 'x' ? 0 : (s[0] == 'y' ? 1 : 2);
    } else if (!strcmp(a, "--no-deconvolve")) deconv = 0;
    else if (!strcmp(a, "--no-shot")) shot = 0;
    else if (!strcmp(a, "--velocity")) velocity = 1;
    else if (!strcmp(a, "-h") || !strcmp(a, "--help")) {
      usage(argv[0]);
#ifdef CK_ENABLE_MPI
      MPI_Finalize();
#endif
      return 0;
    } else { usage(argv[0]); die(CK_ERR_ARG, a); }
  }
  if (!input || !output || nmesh < 4) {
    usage(argv[0]);
    die(CK_ERR_ARG, "--input, --output and --nmesh are required");
  }
  if (tidal_R >= 0.0 && !save_prefix) die(CK_ERR_ARG, "--tidal requires --save-fields");

  double t0 = wtime();
  if (box <= 0.0) {
    CHECK(ck_snapshot_boxsize(input, &box));
    if (box <= 0.0) die(CK_ERR_ARG, "snapshot has no BoxSize; pass --box");
  }
  ck_context *ctx = ck_context_create(nmesh, box, nthreads, comm_f);
  if (!ctx) die(CK_ERR_ARG, "ck_context_create");
  int rank, size, nth;
  ck_context_info(ctx, NULL, NULL, &rank, &size, NULL, NULL, &nth);
  log0("cosmokit %s: nmesh=%d box=%g ranks=%d threads/rank=%d scheme=%s\n", CK_VERSION, nmesh,
       box, size, nth, scheme == CK_NGP ? "NGP" : (scheme == CK_CIC ? "CIC" : "TSC"));

  ck_grid *rho = ck_grid_create(ctx);
  ck_grid *mom[3] = {NULL, NULL, NULL};
  if (!rho) die(CK_ERR_ALLOC, "grid");
  if (velocity)
    for (int d = 0; d < 3; d++)
      if (!(mom[d] = ck_grid_create(ctx))) die(CK_ERR_ALLOC, "grid");

  int64_t npart = 0;
  CHECK(ck_deposit_snapshot(input, ptypes, rho, velocity ? mom : NULL, scheme, chunk,
                            vel_factor, NULL, &npart));
  log0("assigned %lld particles (%.2fs)\n", (long long)npart, wtime() - t0);

  ck_pk_opts opts;
  ck_pk_opts_default(&opts);
  opts.dk = dk;
  opts.window = deconv ? (int)scheme : 0;
  opts.shot = shot;
  opts.nell = nell;
  opts.los = los;
  const int nb = ck_pk_nbins(ctx, &opts);

  /* velocity = momentum / mass (cell-wise, mass weighted); done before delta */
  if (velocity)
    for (int d = 0; d < 3; d++) CHECK(ck_grid_divide(mom[d], mom[d], rho));

  CHECK(ck_grid_to_overdensity(rho));
  if (tidal_R >= 0.0 && save_prefix) {
#ifdef CK_ENABLE_HDF5
    save_field(save_prefix, "delta", rho, 1);
#endif
  }
  CHECK(ck_grid_fft_forward(rho));

  ck_pk_result res = {malloc(nb * sizeof(double)), malloc(nb * sizeof(int64_t)),
                      malloc((size_t)nb * (nell > 5 ? nell : 5) * sizeof(double))};
  CHECK(ck_power_spectrum(rho, NULL, &opts, &res));
  if (!g_rank) {
    FILE *fp = fopen(output, "w");
    if (!fp) die(CK_ERR_IO, "open output");
    fprintf(fp, "# cosmokit %s density power spectrum\n# box %.10g nmesh %d scheme %d deconvolved %d shot_subtracted %d npart %lld\n",
            CK_VERSION, box, nmesh, (int)scheme, deconv, shot, (long long)npart);
    fprintf(fp, "# k  P0%s  nmodes\n", nell > 1 ? "  P2" : "");
    if (nell > 2) fprintf(fp, "# (columns: k P0 P2 P4 nmodes)\n");
    for (int b = 0; b < nb; b++) {
      if (res.nmodes[b] == 0) continue;
      fprintf(fp, "%.8e", res.k[b]);
      for (int l = 0; l < nell; l++) fprintf(fp, " %.8e", res.pk[(size_t)l * nb + b]);
      fprintf(fp, " %lld\n", (long long)res.nmodes[b]);
    }
    fclose(fp);
  }

  if (velocity) {
    for (int d = 0; d < 3; d++) CHECK(ck_grid_fft_forward(mom[d]));
    const ck_grid *v[3] = {mom[0], mom[1], mom[2]};
    CHECK(ck_vector_power(v, rho, &opts, &res));
    if (!g_rank) {
      char buf[1100];
      if (!vel_output) { snprintf(buf, sizeof buf, "%s.vel", output); vel_output = buf; }
      FILE *fp = fopen(vel_output, "w");
      if (!fp) die(CK_ERR_IO, "open vel output");
      fprintf(fp, "# cosmokit %s velocity spectra (mass-weighted cell velocities)\n", CK_VERSION);
      fprintf(fp, "# k  P_vv  P_long  P_trans  P_thetatheta  P_deltatheta  nmodes\n");
      for (int b = 0; b < nb; b++) {
        if (res.nmodes[b] == 0) continue;
        fprintf(fp, "%.8e", res.k[b]);
        for (int r = 0; r < 5; r++) fprintf(fp, " %.8e", res.pk[(size_t)r * nb + b]);
        fprintf(fp, " %lld\n", (long long)res.nmodes[b]);
      }
      fclose(fp);
    }
  }

  if (tidal_R >= 0.0) {
    if (deconv) CHECK(ck_deconvolve(rho, scheme));
    if (tidal_R > 0.0) CHECK(ck_filter(rho, CK_FILTER_GAUSSIAN, tidal_R));
    ck_grid *T[6], *L[3], *web = ck_grid_create(ctx);
    for (int i = 0; i < 6; i++) if (!(T[i] = ck_grid_create(ctx))) die(CK_ERR_ALLOC, "grid");
    for (int i = 0; i < 3; i++) if (!(L[i] = ck_grid_create(ctx))) die(CK_ERR_ALLOC, "grid");
    if (!web) die(CK_ERR_ALLOC, "grid");
    CHECK(ck_tidal_tensor(rho, T));
    const ck_grid *Tc[6] = {T[0], T[1], T[2], T[3], T[4], T[5]};
    CHECK(ck_tensor_eigenvalues(Tc, L));
    const ck_grid *Lc[3] = {L[0], L[1], L[2]};
    CHECK(ck_web_classify(Lc, web_th, web));
#ifdef CK_ENABLE_HDF5
    save_field(save_prefix, "lambda1", L[0], 0);
    save_field(save_prefix, "lambda2", L[1], 0);
    save_field(save_prefix, "lambda3", L[2], 0);
    save_field(save_prefix, "web", web, 0);
#endif
    for (int i = 0; i < 6; i++) ck_grid_destroy(T[i]);
    for (int i = 0; i < 3; i++) ck_grid_destroy(L[i]);
    ck_grid_destroy(web);
  }

  log0("done (%.2fs total)\n", wtime() - t0);
  free(res.k); free(res.nmodes); free(res.pk);
  ck_grid_destroy(rho);
  for (int d = 0; d < 3; d++) ck_grid_destroy(mom[d]);
  ck_context_destroy(ctx);
#ifdef CK_ENABLE_MPI
  MPI_Finalize();
#endif
  return 0;
}
