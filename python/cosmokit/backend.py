"""
Projection backend for ``analysistools.projection`` (the shared, backend-neutral
SPH projection layer). Implements ``knn_h``, ``sph_density`` and
``deposit_projected`` on top of cosmokit's k-d tree and SPH deposit.

Conventions match analysistools: ``H`` is the SPH support radius, the
distance to the k-th neighbour with self excluded.

The deposit is non-periodic from the caller's point of view. Internally
cosmokit's mesh is a periodic cube with isotropic cells, so the region is
padded (and the pad cropped afterwards) to keep wrapped kernels out of the
image. A requested depth-cell count is therefore ignored: cells are cubes of
the in-plane pixel size.
"""
import numpy as np

import cosmokit as ck

MAX_MESH_BYTES = 4 * 2**30


def _wrap(pos, boxsize):
    return np.mod(pos, boxsize) if boxsize else pos


class CosmokitBackend:
    name = "cosmokit"

    def __init__(self, max_mesh_bytes=MAX_MESH_BYTES):
        self.max_mesh_bytes = max_mesh_bytes

    @staticmethod
    def knn_h(pos, query, k, boxsize=None):
        tree = ck.KDTree(_wrap(pos, boxsize), boxsize=boxsize or 0.0)
        dist, _ = tree.knn(k + 1, query=_wrap(query, boxsize))   # query is a tree member
        return dist[:, -1]

    @staticmethod
    def sph_density(pos, mass, k, boxsize=None, include_self=True):
        tree = ck.KDTree(_wrap(pos, boxsize), boxsize=boxsize or 0.0)
        rho, _ = tree.sph_density(mass, nngb=k, include_self=include_self)
        return rho

    def deposit_projected(self, pos, values, H, lo, hi, shape):
        lo, hi = np.asarray(lo, float), np.asarray(hi, float)
        dx = (hi[0] - lo[0]) / shape[0]                 # isotropic cell size
        ncell = np.rint((hi - lo) / dx).astype(int)     # region cells per axis

        # bounding box of every kernel, so nothing wraps into the image
        blo = np.minimum((pos - H[:, None]).min(axis=0), lo)
        bhi = np.maximum((pos + H[:, None]).max(axis=0), hi)
        n_lo = np.ceil((lo - blo) / dx).astype(int)
        n_hi = np.ceil((bhi - hi) / dx).astype(int)
        nmesh = int((n_lo + ncell + n_hi).max()) + 1
        if nmesh**3 * 8 > self.max_mesh_bytes:
            raise ValueError(
                f"cosmokit backend needs a {nmesh}^3 mesh "
                f"({nmesh**3 * 8 / 2**30:.1f} GiB); reduce npix or half_depth, "
                f"or use backend='numba'")

        origin = lo - n_lo * dx
        ctx = ck.Context(nmesh, nmesh * dx)
        shifted = np.ascontiguousarray(pos - origin, dtype=np.float64)
        Hc = np.ascontiguousarray(H, dtype=np.float64)
        sl = tuple(slice(n_lo[i], n_lo[i] + ncell[i]) for i in range(3))
        out = []
        for v in values:
            grid = ck.sph_density_field(ctx, shifted, Hc, mass=np.ascontiguousarray(v, float)).real
            out.append(grid[sl].sum(axis=2))
        return out
