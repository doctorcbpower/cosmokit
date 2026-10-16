/*
 * cosmokit.h - public C API
 *
 * Gridded fields (mass, velocity, momentum), power spectra, tidal/velocity
 * fields, neighbour search and SPH-style density estimation for N-body data.
 *
 * Parallelism: OpenMP within a node, MPI (x-slab decomposition) across nodes.
 * Everything here is plain C with opaque handles, so it can be called from C,
 * from Python (ctypes, see python/cosmokit) or from other codes such as imbas.
 *
 * Conventions
 *   - Periodic cubic box of side `box`, `nmesh` cells per side.
 *   - Cell (i,j,k) covers [i,i+1)*h etc.; its centre is (i+0.5)*h.
 *   - After ck_grid_fft_forward() a grid holds delta_k = (1/N^3) sum_x f(x) e^{-ik.x}
 *     (dimensionless). P(k) = V |delta_k|^2 with V = box^3.
 *   - Physical wavenumber k = 2*pi*n/box.
 *   - MPI: grids are distributed in slabs along x; every rank owns
 *     local_n0 x-planes starting at local_0_start. Functions that take
 *     particles accept ANY distribution of particles over ranks.
 */
#ifndef COSMOKIT_H
#define COSMOKIT_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define CK_VERSION "0.1.0"

/* ----------------------------------------------------------------- errors */
enum {
  CK_OK = 0,
  CK_ERR_ARG = -1,
  CK_ERR_ALLOC = -2,
  CK_ERR_MPI = -3,
  CK_ERR_UNSUPPORTED = -4,
  CK_ERR_IO = -5,
  CK_ERR_STATE = -6 /* e.g. grid is in real space but Fourier space needed */
};
const char *ck_strerror(int code);
const char *ck_last_error(void); /* message for the most recent failure */

int ck_has_mpi(void);
int ck_has_openmp(void);
const char *ck_version(void);

/* ------------------------------------------------------------------ types */
typedef enum { CK_FLOAT32 = 0, CK_FLOAT64 = 1 } ck_dtype;

/* value == order p of the mass assignment window */
typedef enum { CK_NGP = 1, CK_CIC = 2, CK_TSC = 3 } ck_scheme;

typedef struct ck_context ck_context; /* mesh geometry, FFT plans, MPI layout */
typedef struct ck_grid ck_grid;       /* one field living on a context */

/* ---------------------------------------------------------------- context */
/*
 * nthreads <= 0 : use the OpenMP default
 * comm_f   < 0  : no MPI (serial on this process)
 * comm_f  >= 0  : Fortran handle of an MPI communicator (MPI_Comm_c2f), so the
 *                 same call works from C, Fortran and mpi4py (comm.py2f()).
 * MPI must already be initialised when comm_f >= 0.
 */
ck_context *ck_context_create(int nmesh, double box, int nthreads, int comm_f);
void ck_context_destroy(ck_context *ctx);
void ck_context_info(const ck_context *ctx, int *nmesh, double *box, int *rank,
                     int *size, int64_t *local_n0, int64_t *local_0_start,
                     int *nthreads);

/* ------------------------------------------------------------------- grid */
ck_grid *ck_grid_create(ck_context *ctx); /* zero-filled, real space */
void ck_grid_destroy(ck_grid *g);
int ck_grid_zero(ck_grid *g);
int ck_grid_copy(ck_grid *dst, const ck_grid *src);
/* state: 0 = real space, 1 = Fourier space. sum_w/sum_w2/npart: global
 * statistics accumulated by ck_assign (used for shot noise). */
void ck_grid_info(const ck_grid *g, int *state, int64_t *npart, double *sum_w,
                  double *sum_w2);

/* Dense copies of this rank's slab. Real: (local_n0, N, N) doubles.
 * Fourier: (local_n0, N, N/2+1) complex, interleaved re/im. */
int ck_grid_get_real(const ck_grid *g, double *out);
int ck_grid_set_real(ck_grid *g, const double *in);
int ck_grid_get_fourier(const ck_grid *g, double *out);
int ck_grid_set_fourier(ck_grid *g, const double *in);

/* Forward FFT includes the 1/N^3 normalisation; inverse is unnormalised. */
int ck_grid_fft_forward(ck_grid *g);
int ck_grid_fft_inverse(ck_grid *g);

double ck_grid_sum(const ck_grid *g);  /* real space, global */
int ck_grid_scale(ck_grid *g, double factor);          /* works in either space */
int ck_grid_to_overdensity(ck_grid *g);                /* rho -> rho/mean - 1 (real) */
/* out = num/den cell by cell, 0 where den<=0 (momentum/mass -> velocity) */
int ck_grid_divide(ck_grid *out, const ck_grid *num, const ck_grid *den);

/* -------------------------------------------------------------- particles */
typedef struct {
  int64_t n;
  const void *pos;      /* n x 3, row-major, positions in [0,box) (wrapped) */
  ck_dtype pos_dtype;
  const void *vel;      /* n x 3 or NULL */
  ck_dtype vel_dtype;
  const void *mass;     /* n or NULL -> mass0 for every particle */
  ck_dtype mass_dtype;
  double mass0;
} ck_particles;

/*
 * Deposit particles onto grids.
 *   rho : receives sum m W          (may be NULL if only momentum wanted)
 *   mom : receives sum m v_c W, c = 0,1,2 (pass NULL to skip; needs p->vel)
 * Grids must be in real space. Contributions accumulate over calls. In MPI
 * runs particles are redistributed to the owning slab in bounded-size chunks.
 */
int ck_assign(ck_grid *rho, ck_grid *mom[3], const ck_particles *p, ck_scheme scheme);

/* Deposit with a per-particle compact kernel of support radius hsml[i]
 * (physical units) instead of a point mass; mass-conserving. Shared-memory
 * only (returns CK_ERR_UNSUPPORTED when running on >1 MPI rank). */
int ck_assign_sph(ck_grid *rho, const ck_particles *p, const double *hsml, int kernel);

/* ---------------------------------------------------------------- filters */
typedef enum { CK_FILTER_GAUSSIAN = 0, CK_FILTER_TOPHAT = 1 } ck_filter_type;
/* Multiply a Fourier-space grid by a real-space smoothing window of radius R. */
int ck_filter(ck_grid *g, ck_filter_type type, double R);
/* Divide by the mass-assignment window (amplitude, product of sinc^p). */
int ck_deconvolve(ck_grid *g, ck_scheme scheme);

/* ----------------------------------------------------------- power spectra */
typedef struct {
  double dk;    /* bin width in units of fundamental mode 2pi/box (0 -> 1) */
  double kmax;  /* max |k| in the same units (<=0 -> Nyquist N/2) */
  int window;   /* 0 none, else ck_scheme: divide by window^2 */
  int shot;     /* 1: subtract Poisson shot noise (auto spectra, uses npart/sum_w
                   of the first grid, aliasing-aware for NGP/CIC/TSC) */
  int nell;     /* 1: monopole, 2: +quadrupole, 3: +hexadecapole */
  int los;      /* line-of-sight axis for multipoles (0,1,2) */
} ck_pk_opts;
void ck_pk_opts_default(ck_pk_opts *o);
int ck_pk_nbins(const ck_context *ctx, const ck_pk_opts *o);

/* All arrays are caller-allocated; k[nbins], nmodes[nbins], pk[nell*nbins]
 * (row l holds P_{2l}). Empty bins have nmodes==0 and k,pk == 0.
 * k in h/Mpc-like physical units (2pi/box * mean |n|), pk in box^3 units. */
typedef struct {
  double *k;
  int64_t *nmodes;
  double *pk;
} ck_pk_result;

/* Auto (b==NULL) or cross spectrum of Fourier-space grids. */
int ck_power_spectrum(const ck_grid *a, const ck_grid *b, const ck_pk_opts *o,
                      ck_pk_result *res);

/*
 * Velocity/momentum spectra from Fourier-space vector field v[3] and an
 * optional Fourier-space density contrast delta (NULL allowed).
 * pk has 5 rows x nbins:
 *   0 <|v|^2>   1 longitudinal (k.v)^2/k^2   2 transverse (tot-long)
 *   3 P_thth = k^2 * long   (theta = div v)
 *   4 P_delta_theta = Re[delta^* (i k.v)]    (0 if delta == NULL)
 */
int ck_vector_power(const ck_grid *v[3], const ck_grid *delta, const ck_pk_opts *o,
                    ck_pk_result *res);

/* ----------------------------------------------- tidal field, velocity field */
/* Poisson: laplacian(phi) = norm * delta. Output phi in Fourier space.
 * out may alias delta. */
int ck_potential(const ck_grid *delta, double norm, ck_grid *phi_k);

/* Tidal tensor T_ij = (k_i k_j/k^2 - delta_ij/3) delta_k, traceless, six
 * independent components in the order xx,yy,zz,xy,xz,yz.  Results are returned
 * in REAL space in out[6]; delta stays unchanged (Fourier). */
int ck_tidal_tensor(const ck_grid *delta, ck_grid *out[6]);

/* Linear-theory velocity from delta: v_k = i (f a H) k/k^2 delta_k, so that
 * div v = -f a H delta.  vk[3] are Fourier-space outputs. */
int ck_velocity_from_delta(const ck_grid *delta, double faH, ck_grid *vk[3]);

/* Derivatives of a Fourier-space vector field, all returned in REAL space.
 * Spectral derivatives (Nyquist zeroed). */
int ck_velocity_divergence(const ck_grid *vk[3], ck_grid *theta);
int ck_velocity_shear(const ck_grid *vk[3], ck_grid *out[6]);    /* traceless, xx,yy,zz,xy,xz,yz */
int ck_velocity_vorticity(const ck_grid *vk[3], ck_grid *out[3]);

/* Eigenvalues l[0]>=l[1]>=l[2] of a symmetric tensor field (6 real-space
 * grids) and the cosmic-web classification (Hahn+07): number of eigenvalues
 * above lambda_th, i.e. 0 void, 1 sheet, 2 filament, 3 knot. */
int ck_tensor_eigenvalues(const ck_grid *t[6], ck_grid *l[3]);
int ck_web_classify(const ck_grid *l[3], double lambda_th, ck_grid *web);

/* ------------------------------------------------------ neighbour search */
typedef struct ck_kdtree ck_kdtree;

/* box > 0 : periodic with that box size (positions are wrapped);
 * box <= 0: non-periodic.  pos is n x 3. leaf_size <= 0 -> default. */
ck_kdtree *ck_kdtree_build(const void *pos, ck_dtype dtype, int64_t n, double box,
                           int leaf_size);
void ck_kdtree_free(ck_kdtree *t);
int64_t ck_kdtree_size(const ck_kdtree *t);

#define CK_KNN_SKIP_SELF 1 /* query i is point i of the tree: exclude it */
/* k nearest neighbours of each of m queries; dist[m*k] ascending (Euclidean,
 * minimum-image), idx[m*k] original indices (-1 / INFINITY if < k points). */
int ck_kdtree_knn(const ck_kdtree *t, const void *q, ck_dtype qtype, int64_t m, int k,
                  unsigned flags, double *dist, int64_t *idx);
/* Fixed-radius search with per-query radii. */
int ck_kdtree_radius_count(const ck_kdtree *t, const void *q, ck_dtype qtype,
                           int64_t m, const double *radii, unsigned flags,
                           int64_t *counts);
/* offsets[m+1] = exclusive prefix sum of counts; out_idx[offsets[m]]. */
int ck_kdtree_radius_fill(const ck_kdtree *t, const void *q, ck_dtype qtype,
                          int64_t m, const double *radii, unsigned flags,
                          const int64_t *offsets, int64_t *out_idx);

/* -------------------------------------------------------------------- SPH */
typedef enum { CK_KERNEL_CUBIC = 0, CK_KERNEL_WENDLAND_C2 = 1 } ck_kernel;
/* Kernel normalised to unit integral, compact support radius H (q = r/H). */
double ck_kernel_w(ck_kernel k, double r, double H);

#define CK_SPH_NO_SELF 1 /* leave the particle's own contribution out of rho */
/*
 * Density at the tree's own points: H_i = hfac * (distance to the nngb-th
 * neighbour excluding self); rho_i = sum_j m_j W(r_ij, H_i), including the
 * particle itself unless CK_SPH_NO_SELF is set (the self term biases rho high
 * by W(0,H)/rho for Poisson-like data; the standard SPH convention keeps it).
 * mass (indexed by original point order) may be NULL (unit masses).
 * h_out / rho_out are in original order; either may be NULL.
 */
int ck_sph_density(const ck_kdtree *t, const double *mass, int nngb, ck_kernel kernel,
                   double hfac, unsigned flags, double *h_out, double *rho_out);
/* Same, evaluated at arbitrary positions (no self exclusion). */
int ck_sph_density_at(const ck_kdtree *t, const double *mass, const void *q,
                      ck_dtype qtype, int64_t m, int nngb, ck_kernel kernel,
                      double hfac, double *h_out, double *rho_out);

/* -------------------------------------------------------------------- I/O */
/*
 * Stream a Gadget-4/Arepo/SWIFT style HDF5 snapshot (single or multi-file)
 * straight onto the mesh without ever holding all particles in memory: each
 * rank reads its share of the particle list in chunks of `chunk` particles.
 *   ptype_mask : bit i selects PartTypeI
 *   vel_factor : multiplies velocities (e.g. sqrt(a) for Gadget-2 convention)
 *   mom        : NULL to skip velocities
 * If box_out != NULL it receives Header/BoxSize.
 */
int ck_deposit_snapshot(const char *path, unsigned ptype_mask, ck_grid *rho,
                        ck_grid *mom[3], ck_scheme scheme, int64_t chunk,
                        double vel_factor, double *box_out, int64_t *npart_out);
int ck_snapshot_boxsize(const char *path, double *box);

#ifdef __cplusplus
}
#endif
#endif /* COSMOKIT_H */
