# cosmokit

C libraries (with a Python wrapper and a command-line driver) for large-volume
N-body analysis: gridded fields, power spectra, tidal/velocity fields,
neighbour search and density smoothing. OpenMP within a node, MPI across nodes.

## Build

```bash
make              # build/libcosmokit.dylib|so + build/cosmokit   (OpenMP)
make MPI=1        # build-mpi/...                                 (OpenMP + MPI)
make smoke        # seconds: CLI + Python run on the sample snapshot and give sane output
make test         # full suite: analytic/brute-force tests + CLI/MPI/Gadget consistency
```
Options: `OPENMP=0`, `HDF5=0`, `DEBUG=1` (ASan), `FFTW_PREFIX=`, `HDF5_PREFIX=`, `GCC=`.
Needs FFTW3 (with `fftw3_omp`, and `fftw3_mpi` for MPI) and optionally HDF5.
On macOS the Makefile picks a Homebrew `gcc-N` because Apple clang has no OpenMP.

## Modules (`include/cosmokit.h`)

| Module | Functions |
|---|---|
| grid / FFT | `ck_context_*`, `ck_grid_*`: periodic mesh, x-slab MPI layout, FFTW plans |
| assignment | `ck_assign` (NGP/CIC/TSC; mass + momentum in one pass), `ck_assign_sph` |
| power | `ck_power_spectrum` (auto/cross, P0/P2/P4, window deconvolution, aliasing-aware shot noise), `ck_vector_power` (total/longitudinal/transverse, θθ, δθ) |
| fields | `ck_filter`, `ck_potential`, `ck_tidal_tensor`, `ck_velocity_from_delta`, `ck_velocity_divergence/shear/vorticity`, `ck_tensor_eigenvalues`, `ck_web_classify` |
| neighbours | `ck_kdtree_build`, `ck_kdtree_knn`, `ck_kdtree_radius_*` (periodic or open) |
| smoothing | `ck_sph_density[_at]` (cubic spline, Wendland C2), `ck_kernel_w` |
| I/O | `ck_deposit_snapshot`: stream Gadget-4/Arepo/SWIFT HDF5 or Gadget-1/2 binary (format 1 or 2, multi-file, auto-detected) onto the mesh in chunks |

## Command line

```bash
mpirun -np 16 build-mpi/cosmokit --input snap_010.hdf5 --nmesh 1024 --ptype 1 \
    --scheme cic --velocity --nell 3 --output pk.txt --tidal 2.0 --save-fields fields
```
Each rank reads its share of the particle list in `--chunk`-sized pieces and the
particles are routed to the owning slab, so no rank ever holds the snapshot;
memory is the grids plus one chunk. Writes `pk.txt` (k, P0[,P2,P4], nmodes),
`pk.txt.vel` (velocity spectra) and optionally per-rank HDF5 slabs of δ, the
tidal eigenvalues and the web classification.

## Python

```python
import cosmokit as ck                       # PYTHONPATH=python
ctx   = ck.Context(256, boxsize=100.0)      # Context(..., comm=MPI.COMM_WORLD) for MPI
delta = ck.density_field(ctx, pos, scheme="cic")
k, pk, nmodes = ck.power_spectrum(delta, shot_noise=True)
v     = ck.velocity_field(ctx, pos, vel)            # mass-weighted cell velocities
vp    = ck.vector_power(v, delta)                   # P_vv, P_long, P_trans, P_θθ, P_δθ
T     = ck.tidal_tensor(ck.smooth(delta, R=2.0))
web   = ck.web_classify(ck.eigenvalues(T), threshold=0.0).real
dist, idx = ck.KDTree(pos, boxsize=100.0).knn(32)
rho, h    = ck.KDTree(pos, boxsize=100.0).sph_density(mass, nngb=64)
```
Arrays may be float32 or float64 `(N,3)`; they are passed to C without copying.

## Conventions

* Cell `i` spans `[i,i+1)·L/N` (value at the cell centre); positions are wrapped periodically.
* Forward FFT stores `δ_k = (1/N³) Σ f e^{-ik·x}`; `P(k) = L³|δ_k|²`; `k = 2π n / L`.
* Bins are `[i·dk, (i+1)·dk)` in units of `2π/L`; each mode counts once (Hermitian partners included).
* `window=p` divides by `W²` with `W = Π sinc(π n/N)^p`. `shot=1` subtracts the Poisson term of Jing (2005), which accounts for aliasing, *before* the window division.
* Velocity fields are mass-weighted cell averages (`Σ m v W / Σ m W`, 0 in empty cells). They are biased low-density regions; for sparse tracers prefer `momentum_field` or smooth first.
* `ck_sph_density` includes the self term by default (`CK_SPH_NO_SELF` removes the +W(0,H) bias for Poisson-like data).

## Status and limits (v0.1)

* Verified: mass conservation, exact CIC weights across periodic boundaries, plane-wave amplitudes/power/multipoles, Poisson-noise removal for NGP/CIC/TSC, tidal tensor/shear/divergence/vorticity vs analytic fields, eigenvalues vs NumPy, kNN/radius vs brute force, SPH kernel normalisation, Gadget binary (formats 1 and 2, multi-file, mixed types, chunked, MPI) identical to HDF5, serial vs OpenMP vs 1–4 MPI ranks (bitwise-identical text output).
* **Not yet:** OpenACC (loops are flat and atomic-free except the deposit, so the k-space kernels in `power.c`/`tidal.c` are the natural first targets); interlacing; float32 grids; MPI for `ck_assign_sph` and the kd-tree (both shared-memory only); MPI tested only on one machine with Open MPI, and the Python/mpi4py path is untested (no mpi4py installed).
* Errors inside an MPI collective (e.g. unreadable file on one rank) abort the job in the CLI; library callers get an error code only on the failing rank.

## Using it from imbas

`ck_assign`/`ck_assign_sph`, `ck_kdtree_*` and `ck_sph_density` cover what imbas's `src/tree`, `src/sph` and CIC deposit do, with an MPI-aware CIC and a single header. Replacing imbas's copies with calls into `libcosmokit` is a follow-up and has not been done.
