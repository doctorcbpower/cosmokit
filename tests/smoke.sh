#!/bin/sh
# Quick sanity check (~seconds): the CLI runs on the sample snapshot and gives
# sane output, and (if numpy is available) the Python wrapper loads and works.
# usage: smoke.sh [builddir]
set -e
BUILD=${1:-build}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
fail() { echo "SMOKE FAIL: $*"; exit 1; }

[ -x "$BUILD/cosmokit" ] || fail "$BUILD/cosmokit not built"
"$BUILD/cosmokit" --input "$ROOT/data/L20_N64.hdf5" --nmesh 32 --scheme cic \
    --velocity --output "$TMP/pk.txt" >"$TMP/log" 2>&1 || { cat "$TMP/log"; fail "CLI exited non-zero"; }
grep -q "assigned 262144 particles" "$TMP/log" || fail "unexpected particle count"

# density spectrum: >=10 bins, first bin has 26 modes, every value finite, P(k) > 0 in the first bin (later bins can dip below zero after shot-noise subtraction)
awk '!/^#/ { n++; for (i=1;i<=NF;i++) if ($i ~ /nan|inf/) bad=1; if (n==1 && $3 != 26) first=$3; if (n==1 && $2 <= 0) neg=1 }
     END { if (n<10) exit 1; if (bad) exit 2; if (first) exit 3; if (neg) exit 4 }' "$TMP/pk.txt" \
  || fail "density spectrum looks wrong (awk code $?)"
awk '!/^#/ { n++; for (i=1;i<=NF;i++) if ($i ~ /nan|inf/) bad=1 } END { if (n<10 || bad) exit 1 }' "$TMP/pk.txt.vel" \
  || fail "velocity spectrum looks wrong"
echo "PASS CLI smoke"

PY=$(for p in python3.13 python3.12 python3.11 python3; do $p -c 'import numpy' 2>/dev/null && { echo $p; break; }; done)
if [ -n "$PY" ]; then
  EXT=so; [ "$(uname -s)" = Darwin ] && EXT=dylib
  COSMOKIT_LIB="$BUILD/libcosmokit.$EXT" "$PY" - "$ROOT" <<'PY' || fail "Python wrapper"
import sys, numpy as np
sys.path.insert(0, sys.argv[1] + "/python")
import cosmokit as ck
ctx = ck.Context(16, 10.0)
pos = np.random.default_rng(0).uniform(0, 10, (5000, 3))
rho, _ = ck.assign(ctx, pos, scheme="cic")
assert abs(rho.sum() - 5000) < 1e-8
k, pk, nm = ck.power_spectrum(ck.density_field(ctx, pos), shot_noise=True)
assert len(k) > 3 and np.all(np.isfinite(pk)) and np.all(nm > 0)
d, i = ck.KDTree(pos, boxsize=10.0).knn(4)
assert d.shape == (5000, 4) and np.all(np.diff(d, axis=1) >= 0)
print("PASS Python smoke")
PY
else
  echo "SKIP Python smoke (no numpy)"
fi
