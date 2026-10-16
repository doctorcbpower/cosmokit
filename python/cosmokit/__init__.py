"""cosmokit - gridded fields, power spectra, tidal fields, neighbour search.

Thin ctypes layer over libcosmokit. NumPy arrays go straight into the C
routines (float32 or float64, C-contiguous, shape (N, 3) for positions).

    import cosmokit as ck
    ctx   = ck.Context(256, boxsize=100.0)
    delta = ck.density_field(ctx, pos, scheme="cic")
    k, pk, nmodes = ck.power_spectrum(delta, deconvolve=True, shot_noise=True)

With mpi4py, pass `comm=MPI.COMM_WORLD`; every rank passes its own particles
(any distribution) and gets its x-slab of each field back.
"""
import ctypes as C
import os
import sys

import numpy as np

__version__ = "0.1.0"

# ---------------------------------------------------------------- loading
_SCHEMES = {"ngp": 1, "cic": 2, "tsc": 3}
_KERNELS = {"cubic": 0, "wendland_c2": 1}


def _find_library():
    env = os.environ.get("COSMOKIT_LIB")
    if env:
        return env
    ext = "dylib" if sys.platform == "darwin" else "so"
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.abspath(os.path.join(here, "..", ".."))
    for d in ("build-mpi", "build"):
        p = os.path.join(root, d, f"libcosmokit.{ext}")
        if os.path.exists(p):
            return p
    p = os.path.join(here, f"libcosmokit.{ext}")
    if os.path.exists(p):
        return p
    raise OSError("libcosmokit not found; run `make` or set COSMOKIT_LIB")


_lib = C.CDLL(_find_library())

_vp = C.c_void_p
_i64 = C.c_int64
_dp = C.POINTER(C.c_double)
_ip = C.POINTER(C.c_int)
_lp = C.POINTER(C.c_int64)


class _Particles(C.Structure):
    _fields_ = [("n", _i64), ("pos", _vp), ("pos_dtype", C.c_int), ("vel", _vp),
                ("vel_dtype", C.c_int), ("mass", _vp), ("mass_dtype", C.c_int),
                ("mass0", C.c_double)]


class _PkOpts(C.Structure):
    _fields_ = [("dk", C.c_double), ("kmax", C.c_double), ("window", C.c_int),
                ("shot", C.c_int), ("nell", C.c_int), ("los", C.c_int)]


class _PkResult(C.Structure):
    _fields_ = [("k", _dp), ("nmodes", _lp), ("pk", _dp)]


def _sig(name, res, *args):
    f = getattr(_lib, name)
    f.restype = res
    f.argtypes = list(args)
    return f


_sig("ck_strerror", C.c_char_p, C.c_int)
_sig("ck_last_error", C.c_char_p)
_sig("ck_has_mpi", C.c_int)
_sig("ck_has_openmp", C.c_int)
_sig("ck_context_create", _vp, C.c_int, C.c_double, C.c_int, C.c_int)
_sig("ck_context_destroy", None, _vp)
_sig("ck_context_info", None, _vp, _ip, _dp, _ip, _ip, _lp, _lp, _ip)
_sig("ck_grid_create", _vp, _vp)
_sig("ck_grid_destroy", None, _vp)
_sig("ck_grid_copy", C.c_int, _vp, _vp)
_sig("ck_grid_info", None, _vp, _ip, _lp, _dp, _dp)
_sig("ck_grid_get_real", C.c_int, _vp, _vp)
_sig("ck_grid_set_real", C.c_int, _vp, _vp)
_sig("ck_grid_get_fourier", C.c_int, _vp, _vp)
_sig("ck_grid_set_fourier", C.c_int, _vp, _vp)
_sig("ck_grid_fft_forward", C.c_int, _vp)
_sig("ck_grid_fft_inverse", C.c_int, _vp)
_sig("ck_grid_sum", C.c_double, _vp)
_sig("ck_grid_scale", C.c_int, _vp, C.c_double)
_sig("ck_grid_to_overdensity", C.c_int, _vp)
_sig("ck_grid_divide", C.c_int, _vp, _vp, _vp)
_sig("ck_assign", C.c_int, _vp, _vp, C.POINTER(_Particles), C.c_int)
_sig("ck_assign_sph", C.c_int, _vp, C.POINTER(_Particles), _vp, C.c_int)
_sig("ck_filter", C.c_int, _vp, C.c_int, C.c_double)
_sig("ck_deconvolve", C.c_int, _vp, C.c_int)
_sig("ck_pk_opts_default", None, C.POINTER(_PkOpts))
_sig("ck_pk_nbins", C.c_int, _vp, C.POINTER(_PkOpts))
_sig("ck_power_spectrum", C.c_int, _vp, _vp, C.POINTER(_PkOpts), C.POINTER(_PkResult))
_sig("ck_vector_power", C.c_int, _vp, _vp, C.POINTER(_PkOpts), C.POINTER(_PkResult))
_sig("ck_potential", C.c_int, _vp, C.c_double, _vp)
_sig("ck_tidal_tensor", C.c_int, _vp, _vp)
_sig("ck_velocity_from_delta", C.c_int, _vp, C.c_double, _vp)
_sig("ck_velocity_divergence", C.c_int, _vp, _vp)
_sig("ck_velocity_shear", C.c_int, _vp, _vp)
_sig("ck_velocity_vorticity", C.c_int, _vp, _vp)
_sig("ck_tensor_eigenvalues", C.c_int, _vp, _vp)
_sig("ck_web_classify", C.c_int, _vp, C.c_double, _vp)
_sig("ck_kdtree_build", _vp, _vp, C.c_int, _i64, C.c_double, C.c_int)
_sig("ck_kdtree_free", None, _vp)
_sig("ck_kdtree_knn", C.c_int, _vp, _vp, C.c_int, _i64, C.c_int, C.c_uint, _vp, _vp)
_sig("ck_kdtree_radius_count", C.c_int, _vp, _vp, C.c_int, _i64, _vp, C.c_uint, _vp)
_sig("ck_kdtree_radius_fill", C.c_int, _vp, _vp, C.c_int, _i64, _vp, C.c_uint, _vp, _vp)
_sig("ck_kernel_w", C.c_double, C.c_int, C.c_double, C.c_double)
_sig("ck_sph_density", C.c_int, _vp, _vp, C.c_int, C.c_int, C.c_double, C.c_uint, _vp, _vp)
_sig("ck_sph_density_at", C.c_int, _vp, _vp, _vp, C.c_int, _i64, C.c_int, C.c_int,
     C.c_double, _vp, _vp)
_sig("ck_deposit_snapshot", C.c_int, C.c_char_p, C.c_uint, _vp, _vp, C.c_int, _i64,
     C.c_double, _dp, _lp)
_sig("ck_snapshot_boxsize", C.c_int, C.c_char_p, _dp)


class CosmokitError(RuntimeError):
    pass


def _check(rc):
    if rc != 0:
        raise CosmokitError(f"{_lib.ck_strerror(rc).decode()}: {_lib.ck_last_error().decode()}")


def has_mpi():
    return bool(_lib.ck_has_mpi())


def has_openmp():
    return bool(_lib.ck_has_openmp())


def _scheme(s):
    if isinstance(s, str):
        return _SCHEMES[s.lower()]
    return int(s)


def _as_points(a, name="pos"):
    a = np.asarray(a)
    if a.dtype not in (np.float32, np.float64):
        a = a.astype(np.float64)
    a = np.ascontiguousarray(a)
    if a.ndim != 2 or a.shape[1] != 3:
        raise ValueError(f"{name} must have shape (N, 3)")
    return a, (0 if a.dtype == np.float32 else 1)


# ---------------------------------------------------------------- context
class Context:
    """Mesh geometry + FFT plans (+ MPI slab layout). One per (nmesh, box)."""

    def __init__(self, nmesh, boxsize, nthreads=0, comm=None):
        self.comm = comm
        fcomm = -1 if comm is None else comm.py2f()
        self._h = _lib.ck_context_create(int(nmesh), float(boxsize), int(nthreads), fcomm)
        if not self._h:
            raise CosmokitError(_lib.ck_last_error().decode())
        n, b, r, s, l0, ls, nt = (C.c_int(), C.c_double(), C.c_int(), C.c_int(),
                                  C.c_int64(), C.c_int64(), C.c_int())
        _lib.ck_context_info(self._h, C.byref(n), C.byref(b), C.byref(r), C.byref(s),
                             C.byref(l0), C.byref(ls), C.byref(nt))
        self.nmesh, self.boxsize = n.value, b.value
        self.rank, self.size = r.value, s.value
        self.local_n0, self.local_0_start = l0.value, ls.value
        self.nthreads = nt.value

    @property
    def kfund(self):
        return 2 * np.pi / self.boxsize

    def __del__(self):
        h, self._h = getattr(self, "_h", None), None
        if h:
            _lib.ck_context_destroy(h)


# ------------------------------------------------------------------ field
class Field:
    """A grid owned by C. Use .real / .fourier to get this rank's slab as NumPy."""

    def __init__(self, ctx, _copy_from=None):
        self.ctx = ctx
        self._h = _lib.ck_grid_create(ctx._h)
        if not self._h:
            raise CosmokitError(_lib.ck_last_error().decode())
        if _copy_from is not None:
            _check(_lib.ck_grid_copy(self._h, _copy_from._h))
            if hasattr(_copy_from, "scheme"):
                self.scheme = _copy_from.scheme

    def __del__(self):
        h, self._h = getattr(self, "_h", None), None
        if h:
            _lib.ck_grid_destroy(h)

    def copy(self):
        return Field(self.ctx, self)

    @property
    def in_fourier_space(self):
        s = C.c_int()
        _lib.ck_grid_info(self._h, C.byref(s), None, None, None)
        return bool(s.value)

    @property
    def stats(self):
        n, w, w2 = C.c_int64(), C.c_double(), C.c_double()
        _lib.ck_grid_info(self._h, None, C.byref(n), C.byref(w), C.byref(w2))
        return {"npart": n.value, "sum_w": w.value, "sum_w2": w2.value}

    def fft(self):
        _check(_lib.ck_grid_fft_forward(self._h))
        return self

    def ifft(self):
        _check(_lib.ck_grid_fft_inverse(self._h))
        return self

    @property
    def real(self):
        c = self.ctx
        out = np.empty((c.local_n0, c.nmesh, c.nmesh))
        _check(_lib.ck_grid_get_real(self._h, out.ctypes.data))
        return out

    @real.setter
    def real(self, arr):
        c = self.ctx
        a = np.ascontiguousarray(arr, dtype=np.float64)
        if a.shape != (c.local_n0, c.nmesh, c.nmesh):
            raise ValueError(f"expected shape {(c.local_n0, c.nmesh, c.nmesh)}")
        _check(_lib.ck_grid_set_real(self._h, a.ctypes.data))

    @property
    def fourier(self):
        c = self.ctx
        out = np.empty((c.local_n0, c.nmesh, c.nmesh // 2 + 1), dtype=np.complex128)
        _check(_lib.ck_grid_get_fourier(self._h, out.ctypes.data))
        return out

    @fourier.setter
    def fourier(self, arr):
        c = self.ctx
        a = np.ascontiguousarray(arr, dtype=np.complex128)
        if a.shape != (c.local_n0, c.nmesh, c.nmesh // 2 + 1):
            raise ValueError("bad shape for Fourier array")
        _check(_lib.ck_grid_set_fourier(self._h, a.ctypes.data))

    def sum(self):
        return _lib.ck_grid_sum(self._h)

    def scale(self, f):
        _check(_lib.ck_grid_scale(self._h, float(f)))
        return self


def _grids(*fields):
    arr = (_vp * len(fields))(*[f._h for f in fields])
    return arr


# -------------------------------------------------------------- assignment
def _particles(pos, vel=None, mass=None):
    pos, pd = _as_points(pos)
    p = _Particles()
    p.n = pos.shape[0]
    p.pos, p.pos_dtype = pos.ctypes.data, pd
    keep = [pos]
    if vel is not None:
        vel, vd = _as_points(vel, "vel")
        if vel.shape[0] != p.n:
            raise ValueError("vel and pos lengths differ")
        p.vel, p.vel_dtype = vel.ctypes.data, vd
        keep.append(vel)
    if mass is None:
        p.mass, p.mass0 = None, 1.0
    elif np.isscalar(mass):
        p.mass, p.mass0 = None, float(mass)
    else:
        m = np.ascontiguousarray(mass)
        if m.dtype not in (np.float32, np.float64):
            m = m.astype(np.float64)
        if m.shape != (p.n,):
            raise ValueError("mass must have shape (N,)")
        p.mass, p.mass_dtype = m.ctypes.data, 0 if m.dtype == np.float32 else 1
        keep.append(m)
    return p, keep


def assign(ctx, pos, mass=None, vel=None, scheme="cic", rho=None, mom=None):
    """Deposit particles. Returns (rho, mom) with mom None unless vel given.
    rho/mom are *accumulated into* if supplied (for chunked loading)."""
    rho = rho if rho is not None else Field(ctx)
    if vel is not None and mom is None:
        mom = [Field(ctx) for _ in range(3)]
    p, keep = _particles(pos, vel, mass)
    mp = _grids(*mom) if (vel is not None) else None
    _check(_lib.ck_assign(rho._h, mp, C.byref(p), _scheme(scheme)))
    return rho, (mom if vel is not None else None)


def density_field(ctx, pos, mass=None, scheme="cic", fft=True):
    """Overdensity delta = rho/mean - 1. Returns a Field (Fourier space if fft)."""
    rho, _ = assign(ctx, pos, mass=mass, scheme=scheme)
    _check(_lib.ck_grid_to_overdensity(rho._h))
    if fft:
        rho.fft()
    rho.scheme = _scheme(scheme)
    return rho


def velocity_field(ctx, pos, vel, mass=None, scheme="cic", fft=True):
    """Mass-weighted cell velocity v = sum(m v W)/sum(m W) (0 in empty cells).
    Returns [vx, vy, vz] Fields."""
    rho, mom = assign(ctx, pos, mass=mass, vel=vel, scheme=scheme)
    for m in mom:
        _check(_lib.ck_grid_divide(m._h, m._h, rho._h))
        if fft:
            m.fft()
    return mom


def momentum_field(ctx, pos, vel, mass=None, scheme="cic", fft=True):
    """Momentum density sum(m v W) per cell (not divided by mass)."""
    _, mom = assign(ctx, pos, mass=mass, vel=vel, scheme=scheme)
    if fft:
        for m in mom:
            m.fft()
    return mom


def sph_density_field(ctx, pos, hsml, mass=None, kernel="cubic", fft=False):
    """Deposit with per-particle kernels (support radius hsml); returns rho."""
    rho = Field(ctx)
    p, keep = _particles(pos, None, mass)
    h = np.ascontiguousarray(hsml, dtype=np.float64)
    _check(_lib.ck_assign_sph(rho._h, C.byref(p), h.ctypes.data, _KERNELS[kernel]))
    if fft:
        rho.fft()
    return rho


def deposit_snapshot(ctx, path, ptypes=(1,), velocities=False, scheme="cic",
                     chunk=1 << 22, vel_factor=1.0):
    """Stream an HDF5 snapshot onto the mesh (never holds all particles)."""
    rho = Field(ctx)
    mom = [Field(ctx) for _ in range(3)] if velocities else None
    mask = 0
    for t in ptypes:
        mask |= 1 << t
    n = C.c_int64()
    _check(_lib.ck_deposit_snapshot(path.encode(), mask, rho._h,
                                    _grids(*mom) if mom else None, _scheme(scheme),
                                    chunk, vel_factor, None, C.byref(n)))
    return rho, mom, n.value


def snapshot_boxsize(path):
    b = C.c_double()
    _check(_lib.ck_snapshot_boxsize(path.encode(), C.byref(b)))
    return b.value


# ------------------------------------------------------------- power spectra
def _opts(ctx, dk, kmax, window, shot, nell, los):
    o = _PkOpts()
    _lib.ck_pk_opts_default(C.byref(o))
    o.dk, o.kmax, o.window, o.shot, o.nell, o.los = dk, kmax, window, int(shot), nell, los
    return o, _lib.ck_pk_nbins(ctx._h, C.byref(o))


def power_spectrum(a, b=None, scheme=None, deconvolve=True, shot_noise=False,
                   dk=1.0, kmax=0.0, nell=1, los=2):
    """Binned auto (b=None) or cross spectrum of Fourier-space fields.

    Returns (k, pk, nmodes); pk has shape (nbins,) for nell == 1 or
    (nell, nbins) with rows P0, P2, P4. dk/kmax are in units of 2pi/box.
    Under MPI, results are identical on every rank.
    """
    scheme = scheme if scheme is not None else getattr(a, "scheme", None)
    window = _scheme(scheme) if (deconvolve and scheme is not None) else 0
    if deconvolve and scheme is None:
        raise ValueError("deconvolve=True needs scheme= (or a field from density_field)")
    o, nb = _opts(a.ctx, dk, kmax, window, shot_noise, nell, los)
    k = np.zeros(nb)
    nm = np.zeros(nb, dtype=np.int64)
    pk = np.zeros((nell, nb))
    res = _PkResult(k.ctypes.data_as(_dp), nm.ctypes.data_as(_lp), pk.ctypes.data_as(_dp))
    _check(_lib.ck_power_spectrum(a._h, b._h if b is not None else None, C.byref(o),
                                  C.byref(res)))
    good = nm > 0
    return k[good], (pk[0] if nell == 1 else pk)[..., good], nm[good]


def vector_power(v, delta=None, scheme=None, deconvolve=True, dk=1.0, kmax=0.0):
    """Velocity/momentum spectra of Fourier-space [vx,vy,vz].

    Returns dict with k, nmodes, total, longitudinal, transverse, theta_theta,
    delta_theta (theta = div v).
    """
    ctx = v[0].ctx
    scheme = scheme if scheme is not None else getattr(delta, "scheme", None)
    window = _scheme(scheme) if (deconvolve and scheme is not None) else 0
    o, nb = _opts(ctx, dk, kmax, window, 0, 1, 2)
    k = np.zeros(nb)
    nm = np.zeros(nb, dtype=np.int64)
    pk = np.zeros((5, nb))
    res = _PkResult(k.ctypes.data_as(_dp), nm.ctypes.data_as(_lp), pk.ctypes.data_as(_dp))
    _check(_lib.ck_vector_power(_grids(*v), delta._h if delta is not None else None,
                                C.byref(o), C.byref(res)))
    g = nm > 0
    names = ["total", "longitudinal", "transverse", "theta_theta", "delta_theta"]
    out = {"k": k[g], "nmodes": nm[g]}
    out.update({n: pk[i][g] for i, n in enumerate(names)})
    return out


# --------------------------------------------------- filters / tidal / velocity
def smooth(field, R, kind="gaussian", scheme=None, deconvolve=False):
    """Return a smoothed copy of a Fourier-space field (radius R, box units)."""
    f = field.copy()
    if deconvolve:
        _check(_lib.ck_deconvolve(f._h, _scheme(scheme if scheme is not None else field.scheme)))
    _check(_lib.ck_filter(f._h, {"gaussian": 0, "tophat": 1}[kind], float(R)))
    return f


def potential(delta, norm=1.0):
    """Fourier-space phi with laplacian(phi) = norm * delta."""
    out = Field(delta.ctx)
    _check(_lib.ck_potential(delta._h, float(norm), out._h))
    return out


def tidal_tensor(delta):
    """Traceless tidal tensor components [xx,yy,zz,xy,xz,yz] in real space."""
    out = [Field(delta.ctx) for _ in range(6)]
    _check(_lib.ck_tidal_tensor(delta._h, _grids(*out)))
    return out


def velocity_from_delta(delta, faH):
    """Linear-theory Fourier-space velocity field with div v = -faH * delta."""
    out = [Field(delta.ctx) for _ in range(3)]
    _check(_lib.ck_velocity_from_delta(delta._h, float(faH), _grids(*out)))
    return out


def divergence(vk):
    out = Field(vk[0].ctx)
    _check(_lib.ck_velocity_divergence(_grids(*vk), out._h))
    return out


def shear(vk):
    out = [Field(vk[0].ctx) for _ in range(6)]
    _check(_lib.ck_velocity_shear(_grids(*vk), _grids(*out)))
    return out


def vorticity(vk):
    out = [Field(vk[0].ctx) for _ in range(3)]
    _check(_lib.ck_velocity_vorticity(_grids(*vk), _grids(*out)))
    return out


def eigenvalues(tensor):
    """Eigenvalues (descending) of a 6-component real-space tensor field."""
    out = [Field(tensor[0].ctx) for _ in range(3)]
    _check(_lib.ck_tensor_eigenvalues(_grids(*tensor), _grids(*out)))
    return out


def web_classify(eig, threshold=0.0):
    """0 void, 1 sheet, 2 filament, 3 knot (number of eigenvalues > threshold)."""
    out = Field(eig[0].ctx)
    _check(_lib.ck_web_classify(_grids(*eig), float(threshold), out._h))
    return out


# --------------------------------------------------------------- neighbours
class KDTree:
    """Periodic (boxsize>0) or open k-d tree for kNN and radius queries."""

    def __init__(self, pos, boxsize=0.0, leaf_size=0):
        self._pos, self._dt = _as_points(pos)
        self.boxsize = float(boxsize)
        self.n = self._pos.shape[0]
        self._h = _lib.ck_kdtree_build(self._pos.ctypes.data, self._dt, self.n,
                                       self.boxsize, int(leaf_size))
        if not self._h:
            raise CosmokitError(_lib.ck_last_error().decode())

    def __del__(self):
        h, self._h = getattr(self, "_h", None), None
        if h:
            _lib.ck_kdtree_free(h)

    def knn(self, k, query=None):
        """k nearest neighbours. query=None -> the tree's own points, self excluded.
        Returns (dist[m,k], idx[m,k])."""
        if query is None:
            q, qd, flags, m = self._pos, self._dt, 1, self.n
        else:
            q, qd = _as_points(query, "query")
            flags, m = 0, q.shape[0]
        dist = np.empty((m, k))
        idx = np.empty((m, k), dtype=np.int64)
        _check(_lib.ck_kdtree_knn(self._h, q.ctypes.data, qd, m, k, flags,
                                  dist.ctypes.data, idx.ctypes.data))
        return dist, idx

    def radius(self, r, query=None):
        """Neighbours within r (scalar or per-query array). Returns (offsets, idx):
        neighbours of query i are idx[offsets[i]:offsets[i+1]]."""
        if query is None:
            q, qd, flags, m = self._pos, self._dt, 1, self.n
        else:
            q, qd = _as_points(query, "query")
            flags, m = 0, q.shape[0]
        radii = np.ascontiguousarray(np.broadcast_to(np.asarray(r, dtype=np.float64), (m,)))
        counts = np.empty(m, dtype=np.int64)
        _check(_lib.ck_kdtree_radius_count(self._h, q.ctypes.data, qd, m,
                                           radii.ctypes.data, flags, counts.ctypes.data))
        offsets = np.zeros(m + 1, dtype=np.int64)
        np.cumsum(counts, out=offsets[1:])
        out = np.empty(offsets[-1], dtype=np.int64)
        _check(_lib.ck_kdtree_radius_fill(self._h, q.ctypes.data, qd, m, radii.ctypes.data,
                                          flags, offsets.ctypes.data, out.ctypes.data))
        return offsets, out

    def sph_density(self, mass=None, nngb=32, kernel="cubic", hfac=1.0, query=None,
                    include_self=True):
        """SPH density + smoothing (support) length. Returns (rho, h).
        include_self=False drops the particle's own W(0,H) term (removes the
        upward bias for Poisson-like data); ignored when `query` is given."""
        m_ = None
        if mass is not None:
            m_ = np.ascontiguousarray(mass, dtype=np.float64)
            if m_.shape != (self.n,):
                raise ValueError("mass must have shape (N,)")
        mp = m_.ctypes.data if m_ is not None else None
        if query is None:
            rho, h = np.empty(self.n), np.empty(self.n)
            _check(_lib.ck_sph_density(self._h, mp, nngb, _KERNELS[kernel], hfac,
                                       0 if include_self else 1, h.ctypes.data, rho.ctypes.data))
        else:
            q, qd = _as_points(query, "query")
            rho, h = np.empty(q.shape[0]), np.empty(q.shape[0])
            _check(_lib.ck_sph_density_at(self._h, mp, q.ctypes.data, qd, q.shape[0], nngb,
                                          _KERNELS[kernel], hfac, h.ctypes.data,
                                          rho.ctypes.data))
        return rho, h


def sph_kernel(r, H, kernel="cubic"):
    r = np.atleast_1d(np.asarray(r, dtype=np.float64))
    return np.array([_lib.ck_kernel_w(_KERNELS[kernel], float(x), float(H)) for x in r])
