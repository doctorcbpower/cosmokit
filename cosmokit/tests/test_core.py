"""Core regression tests: analytic checks, brute-force comparisons."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
import cosmokit as ck  # noqa: E402

rng = np.random.default_rng(12345)
fails = []


def check(name, ok, detail=""):
    print(("PASS " if ok else "FAIL ") + name + (f"  [{detail}]" if detail else ""))
    if not ok:
        fails.append(name)


N, L = 32, 50.0
ctx = ck.Context(N, L)
print(f"openmp={ck.has_openmp()} mpi={ck.has_mpi()} threads={ctx.nthreads}")

# --------------------------------------------------------------- assignment
for scheme in ("ngp", "cic", "tsc"):
    pos = rng.uniform(-10, L + 10, size=(5000, 3))  # includes unwrapped positions
    mass = rng.uniform(0.5, 2.0, 5000)
    rho, _ = ck.assign(ctx, pos, mass=mass, scheme=scheme)
    check(f"mass conservation {scheme}", abs(rho.sum() - mass.sum()) < 1e-9 * mass.sum(),
          f"{rho.sum():.10g} vs {mass.sum():.10g}")
    check(f"no negative cells {scheme}", rho.real.min() >= 0.0)

# single particle: exact CIC weights, periodic wrap at the boundary
h = L / N
p = np.array([[0.25 * h, 0.5 * h, 3.0 * h]])  # x is 0.25h: half a cell below centre of cell 0
rho, _ = ck.assign(ctx, p, scheme="cic")
r = rho.real
# u = (0.25, 0.5, 3.0) -> axis x: t=-0.25, i0=-1 (wraps to N-1), d=0.75 ; y: t=0 -> cell 0 only ; z: t=2.5 -> cells 2,3 each .5
exp = {(N - 1, 0, 2): 0.25 * 0.5, (N - 1, 0, 3): 0.25 * 0.5, (0, 0, 2): 0.75 * 0.5, (0, 0, 3): 0.75 * 0.5}
ok = all(abs(r[i] - v) < 1e-14 for i, v in exp.items()) and abs(r.sum() - 1) < 1e-14
check("CIC weights across periodic boundary", ok)

# float32 positions give the same grid as float64 copies
pos = rng.uniform(0, L, size=(2000, 3)).astype(np.float32)
a, _ = ck.assign(ctx, pos, scheme="tsc")
b, _ = ck.assign(ctx, pos.astype(np.float64), scheme="tsc")
check("float32 == float64 input", np.allclose(a.real, b.real, atol=1e-12))

# momentum / velocity
pos = rng.uniform(0, L, size=(4000, 3))
vel = np.tile([1.0, -2.0, 3.0], (4000, 1))
v = ck.velocity_field(ctx, pos, vel, scheme="cic", fft=False)
vr = [x.real for x in v]
rho0, _ = ck.assign(ctx, pos, scheme="cic")
occ = rho0.real > 0
check("uniform velocity recovered", all(np.allclose(vr[d][occ], [1, -2, 3][d]) for d in range(3))
      and all(np.all(vr[d][~occ] == 0) for d in range(3)))

# ------------------------------------------------------- Fourier conventions
nmode = 3
x = (np.arange(N) + 0.5) * h
X = np.meshgrid(x, x, x, indexing="ij")
A = 0.3
k1 = 2 * np.pi * nmode / L
f = ck.Field(ctx)
f.real = A * np.cos(k1 * X[0])
f.fft()
F = f.fourier
# cell-centred sampling puts a phase on the mode; amplitude is A/2
check("delta_k amplitude A/2", abs(abs(F[nmode, 0, 0]) - A / 2) < 1e-12, f"{abs(F[nmode,0,0]):.6g}")
check("fft round trip", np.allclose(f.copy().ifft().real, A * np.cos(k1 * X[0]), atol=1e-12))
kk, pk, nm = ck.power_spectrum(f, deconvolve=False)
# all power sits in the shell containing |n|=3 : P_bin = V*2*(A/2)^2/nmodes
V = L**3
sel = np.argmin(abs(kk - k1))
expect = V * 2 * (A / 2) ** 2 / nm[sel]
check("single-mode power", abs(pk[sel] / expect - 1) < 1e-10, f"{pk[sel]:.6g} vs {expect:.6g}")
check("no leakage outside shell", np.sum(pk) - pk[sel] < 1e-12 * pk[sel])

# ----------------------------------------------------------- Poisson noise
Np = 400_000
ctx64 = ck.Context(64, L)
pos = rng.uniform(0, L, size=(Np, 3))
for scheme in ("ngp", "cic", "tsc"):
    d = ck.density_field(ctx64, pos, scheme=scheme)
    k, pk, nm = ck.power_spectrum(d, deconvolve=True, shot_noise=True)
    shot = V / Np
    sel = k < 0.5 * k.max()
    z = np.sum(pk[sel] * nm[sel]) / np.sum(nm[sel]) / shot
    err = 1.0 / np.sqrt(np.sum(nm[sel]))
    check(f"Poisson noise removed ({scheme})", abs(z) < 5 * err + 0.01, f"mean P/shot = {z:.4f} +- {err:.4f}")
    k, pk0, nm = ck.power_spectrum(d, deconvolve=False, shot_noise=False)
    if scheme == "ngp":
        r_ = np.sum(pk0 * nm) / np.sum(nm) / shot
        check("raw NGP spectrum equals V/N", abs(r_ - 1) < 5 / np.sqrt(np.sum(nm)), f"{r_:.4f}")

# modulated Poisson sample: recover the input amplitude through CIC deconvolution
Np = 1_500_000
Amp = 0.2
xs = rng.uniform(0, L, size=4 * Np)
u = rng.uniform(0, 1 + Amp, size=4 * Np)
keep = u < 1 + Amp * np.cos(2 * np.pi * xs / L)
xs = xs[keep][:Np]
pos = np.c_[xs, rng.uniform(0, L, size=(len(xs), 2))]
d = ck.density_field(ctx64, pos, scheme="cic")
Fd = d.fourier
d2 = ck.Field(ctx64, d)
ck._check(ck._lib.ck_deconvolve(d2._h, 2))
amp = abs(d2.fourier[1, 0, 0])
check("modulated sample amplitude (CIC deconvolved)", abs(amp - Amp / 2) < 0.02 * Amp / 2 + 5 / np.sqrt(len(xs)),
      f"{amp:.5f} vs {Amp/2:.5f}")

# ---------------------------------------------------------- multipoles
f = ck.Field(ctx)
# delta = cos(k z): mu = 1 -> all power is monopole + 5/... check P2 = 2.5*(3mu^2-1) P0 = 5 P0
f.real = A * np.cos(2 * np.pi * 2 * X[2] / L)
f.fft()
k, pk, nm = ck.power_spectrum(f, deconvolve=False, nell=3, los=2)
j = np.argmax(pk[0])
check("multipoles of a line-of-sight wave", abs(pk[1][j] / pk[0][j] - 5.0) < 1e-9 and
      abs(pk[2][j] / pk[0][j] - 9.0) < 1e-9, f"P2/P0={pk[1][j]/pk[0][j]:.6f} P4/P0={pk[2][j]/pk[0][j]:.6f}")
k, pk, nm = ck.power_spectrum(f, deconvolve=False, nell=3, los=0)
j = np.argmax(pk[0])
check("multipoles perpendicular to line of sight", abs(pk[1][j] / pk[0][j] + 2.5) < 1e-9 and
      abs(pk[2][j] / pk[0][j] - 9 * 3 / 8) < 1e-9, f"P2/P0={pk[1][j]/pk[0][j]:.6f}")

# ----------------------------------------------------- cross spectrum
g = ck.Field(ctx)
g.real = 2.0 * A * np.cos(k1 * X[0])
g.fft()
f.real = A * np.cos(k1 * X[0])
f.fft()
k, px, _ = ck.power_spectrum(f, g, deconvolve=False)
k, pa, _ = ck.power_spectrum(f, deconvolve=False)
check("cross spectrum scaling", abs(px.max() / pa.max() - 2.0) < 1e-10)

# ------------------------------------------------- tidal field & velocities
f.real = A * np.cos(k1 * X[0])
f.fft()
T = [t.real for t in ck.tidal_tensor(f)]
dl = A * np.cos(k1 * X[0])
check("tidal tensor of plane wave", np.allclose(T[0], 2 / 3 * dl, atol=1e-12) and
      np.allclose(T[1], -dl / 3, atol=1e-12) and np.allclose(T[2], -dl / 3, atol=1e-12) and
      all(np.allclose(T[i], 0, atol=1e-12) for i in (3, 4, 5)))
ev = [e.real for e in ck.eigenvalues(ck.tidal_tensor(f))]
exact_hi = np.maximum(2 / 3 * dl, -dl / 3)
exact_lo = np.minimum(2 / 3 * dl, -dl / 3)
check("eigenvalues ordered/correct", np.allclose(ev[0], exact_hi, atol=1e-12) and np.allclose(ev[2], exact_lo, atol=1e-12)
      and np.allclose(ev[0] + ev[1] + ev[2], 0, atol=1e-12))
# random symmetric tensor eigenvalues vs numpy
tf = [ck.Field(ctx) for _ in range(6)]
M = rng.normal(size=(6, N, N, N))
for t, m in zip(tf, M):
    t.real = m
evs = np.array([e.real for e in ck.eigenvalues(tf)])
mats = np.empty((N, N, N, 3, 3))
for (i, j), m in zip([(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)], M):
    mats[..., i, j] = m
    mats[..., j, i] = m
ref = np.sort(np.linalg.eigvalsh(mats), axis=-1)[..., ::-1]
check("eigenvalues match numpy.linalg", np.allclose(np.moveaxis(evs, 0, -1), ref, atol=1e-10))
web = ck.web_classify(ck.eigenvalues(ck.tidal_tensor(f)), 0.0).real
check("web classes in range", web.min() >= 0 and web.max() <= 3)

faH = 70.0
vk = ck.velocity_from_delta(f, faH)
theta = ck.divergence(vk).real
check("div v = -faH delta", np.allclose(theta, -faH * dl, atol=1e-10))
vx = ck.Field(ctx, vk[0]).ifft().real
check("linear velocity amplitude", np.allclose(vx, -(faH * A / k1) * np.sin(k1 * X[0]), atol=1e-10))
S = [s.real for s in ck.shear(vk)]
check("shear of potential flow", np.allclose(S[0], -2 / 3 * faH * dl, atol=1e-10) and
      np.allclose(S[1], faH / 3 * dl, atol=1e-10) and np.allclose(S[2] + S[1] + S[0], 0, atol=1e-10))
W = [w.real for w in ck.vorticity(vk)]
check("potential flow is irrotational", all(np.allclose(w, 0, atol=1e-10) for w in W))
vp = ck.vector_power(vk, f, deconvolve=False)
check("vector power: all longitudinal", np.allclose(vp["transverse"], 0, atol=1e-12 * vp["total"].max()) and
      np.allclose(vp["total"], vp["longitudinal"]))
# theta-theta and delta-theta relations for v = i faH k/k^2 delta
j = np.argmax(vp["total"])
k_j = vp["k"][j]
_, pd_, _ = ck.power_spectrum(f, deconvolve=False)
check("P_thth = faH^2 P_dd, P_dth = -faH P_dd", abs(vp["theta_theta"][j] / (faH**2 * pd_[j]) - 1) < 1e-9 and
      abs(vp["delta_theta"][j] / (-faH * pd_[j]) - 1) < 1e-9)
# solenoidal field
vy = ck.Field(ctx)
vy.real = np.sin(k1 * X[0])
vy.fft()
zero = ck.Field(ctx)
zero.real = np.zeros((N, N, N))
zero.fft()
vp = ck.vector_power([zero, vy, zero], deconvolve=False)
check("vector power: all transverse", np.allclose(vp["longitudinal"], 0, atol=1e-12) and vp["transverse"].max() > 0)
w = [x.real for x in ck.vorticity([zero, vy, zero])]
check("vorticity of shear flow", np.allclose(w[2], k1 * np.cos(k1 * X[0]), atol=1e-10))

# Gaussian smoothing: single mode attenuated by exp(-k^2 R^2 / 2); tophat window
R = 3.0
fs = ck.smooth(f, R)
check("gaussian filter", np.allclose(fs.fourier[nmode, 0, 0], f.fourier[nmode, 0, 0] * np.exp(-0.5 * (k1 * R) ** 2)))
x_ = k1 * R
fs = ck.smooth(f, R, kind="tophat")
check("tophat filter", np.allclose(fs.fourier[nmode, 0, 0] / f.fourier[nmode, 0, 0],
                                   3 * (np.sin(x_) - x_ * np.cos(x_)) / x_**3))
ph = ck.potential(f, norm=1.0)
check("potential: -delta/k^2", np.allclose(ph.fourier[nmode, 0, 0], -f.fourier[nmode, 0, 0] / k1**2))

# ---------------------------------------------------------------- k-d tree
Nt = 3000
for box in (0.0, 1.0):
    P = rng.uniform(0, 1, size=(Nt, 3))
    Q = rng.uniform(0, 1, size=(200, 3))
    tree = ck.KDTree(P, boxsize=box)
    dist, idx = tree.knn(7, query=Q)
    d = Q[:, None, :] - P[None, :, :]
    if box:
        d -= box * np.round(d / box)
    D = np.sqrt((d**2).sum(-1))
    refi = np.argsort(D, axis=1)[:, :7]
    refd = np.take_along_axis(D, refi, axis=1)
    check(f"kNN vs brute force (box={box})", np.allclose(dist, refd, atol=1e-12) and np.array_equal(idx, refi))
    dist, idx = tree.knn(5)  # self query, exclude self
    d = P[:, None, :] - P[None, :, :]
    if box:
        d -= box * np.round(d / box)
    D = np.sqrt((d**2).sum(-1))
    np.fill_diagonal(D, np.inf)
    check(f"kNN self-query excludes self (box={box})", np.allclose(dist, np.sort(D, axis=1)[:, :5], atol=1e-12)
          and not np.any(idx == np.arange(Nt)[:, None]))
    off, nb = tree.radius(0.07, query=Q)
    d = Q[:, None, :] - P[None, :, :]
    if box:
        d -= box * np.round(d / box)
    D = np.sqrt((d**2).sum(-1))
    okr = all(set(nb[off[i]:off[i + 1]]) == set(np.where(D[i] <= 0.07)[0]) for i in range(len(Q)))
    check(f"radius search vs brute force (box={box})", okr)

# positions slightly outside the box get wrapped consistently
P = rng.uniform(0, 1, size=(500, 3))
t1 = ck.KDTree(P, boxsize=1.0)
t2 = ck.KDTree(P + np.array([1.0, -2.0, 3.0]), boxsize=1.0)
Q = rng.uniform(0, 1, size=(50, 3))
check("periodic wrapping of tree points", np.allclose(t1.knn(4, query=Q)[0], t2.knn(4, query=Q)[0]))

# ------------------------------------------------------------------- SPH
Np = 20000
P = rng.uniform(0, 1, size=(Np, 3))
tree = ck.KDTree(P, boxsize=1.0)
for kern in ("cubic", "wendland_c2"):
    rho_in, hh = tree.sph_density(nngb=64, kernel=kern)
    rho_ex, hh2 = tree.sph_density(nngb=64, kernel=kern, include_self=False)
    w0 = np.array([ck._lib.ck_kernel_w(ck._KERNELS[kern], 0.0, float(x)) for x in hh[:200]])
    check(f"SPH self term is exactly W(0,H) ({kern})", np.allclose(rho_in[:200] - rho_ex[:200], w0, rtol=1e-12))
    check(f"SPH density of uniform field, self excluded ({kern})", abs(rho_ex.mean() / Np - 1) < 0.04 and
          rho_ex.std() / rho_ex.mean() < 0.35, f"mean/N={rho_ex.mean()/Np:.3f}, scatter={rho_ex.std()/rho_ex.mean():.3f}")
# kernel normalisation
for kern in ("cubic", "wendland_c2"):
    rr = np.linspace(0, 1, 200001)
    w = ck.sph_kernel(rr[::100], 1.0, kern)
    integ = np.trapezoid(w * 4 * np.pi * rr[::100] ** 2, rr[::100])
    check(f"kernel normalised ({kern})", abs(integ - 1) < 1e-4, f"{integ:.6f}")
q = rng.uniform(0, 1, size=(100, 3))
rho_q, h_q = tree.sph_density(nngb=64, query=q)
check("SPH density at arbitrary points", abs(rho_q.mean() / Np - 1) < 0.1, f"{rho_q.mean()/Np:.3f}")

# SPH deposit conserves mass and smooths more than CIC
pos = rng.uniform(0, L, size=(3000, 3))
hs = np.full(3000, 2.5 * h)
rho_s = ck.sph_density_field(ctx, pos, hs)
check("SPH deposit conserves mass", abs(rho_s.sum() - 3000) < 1e-8)
rho_c, _ = ck.assign(ctx, pos, scheme="cic")
check("SPH deposit smoother than CIC", rho_s.real.std() < rho_c.real.std())

print()
if fails:
    print(f"{len(fails)} test(s) FAILED: {fails}")
    sys.exit(1)
print("all tests passed")
