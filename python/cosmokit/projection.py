"""
Thin wrappers over ``analysistools.projection`` that default to the cosmokit
backend, so projections can be made from cosmokit code without re-implementing
the selection / periodic-image / weighting logic. Needs AnalysisTools importable.

    import cosmokit.projection as cp
    m = cp.project(pos, mass, half_extent=15.0, boxsize=100.0)   # a Map
"""


def _pj():
    try:
        from analysistools import projection
    except ImportError as exc:
        raise ImportError("cosmokit.projection delegates to analysistools.projection; "
                          "install/put AnalysisTools on PYTHONPATH") from exc
    return projection


def project(*args, **kw):
    kw.setdefault("backend", "cosmokit")
    return _pj().project(*args, **kw)


def smoothing_lengths(*args, **kw):
    kw.setdefault("backend", "cosmokit")
    return _pj().smoothing_lengths(*args, **kw)


def sph_density(*args, **kw):
    kw.setdefault("backend", "cosmokit")
    return _pj().sph_density(*args, **kw)


def orientation_from_angular_momentum(*args, **kw):
    return _pj().orientation_from_angular_momentum(*args, **kw)
