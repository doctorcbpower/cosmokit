"""cosmokit's projection backend + wrapper (needs AnalysisTools importable)."""
import numpy as np
import pytest

pytest.importorskip("analysistools")
from analysistools import projection as pj   # noqa: E402
import cosmokit.projection as cp             # noqa: E402


def test_wrapper_matches_analysistools_cosmokit_backend():
    rng = np.random.default_rng(0)
    pos = rng.normal(size=(3000, 3))
    m = np.ones(len(pos))
    a = cp.project(pos, m, half_extent=5.0, npix=48)
    b = pj.project(pos, m, half_extent=5.0, npix=48, backend="cosmokit")
    np.testing.assert_allclose(a.image, b.image, rtol=1e-10, atol=1e-12)   # OpenMP sum order
    assert a.image.sum() * (10.0 / 48) ** 2 == pytest.approx(m.sum(), rel=0.02)
