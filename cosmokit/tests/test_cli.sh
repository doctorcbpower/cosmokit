#!/bin/sh
# CLI regression: serial/OpenMP result must be independent of thread count, and
# (if an MPI build exists) of the number of MPI ranks.
set -e
BUILD=${1:-build}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
DATA=$ROOT/../data/L20_N64.hdf5
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
ARGS="--input $DATA --nmesh 48 --scheme cic --velocity --nell 3"

OMP_NUM_THREADS=1 "$BUILD/cosmokit" $ARGS --output "$TMP/t1.pk" >/dev/null
OMP_NUM_THREADS=4 "$BUILD/cosmokit" $ARGS --output "$TMP/t4.pk" >/dev/null
cmp "$TMP/t1.pk" "$TMP/t4.pk" && cmp "$TMP/t1.pk.vel" "$TMP/t4.pk.vel" && echo "PASS thread-count independence"

MPIBUILD=$ROOT/build-mpi
if [ -x "$MPIBUILD/cosmokit" ] && command -v mpirun >/dev/null; then
  for np in 2 3 4; do
    OMP_NUM_THREADS=2 mpirun --oversubscribe -np $np "$MPIBUILD/cosmokit" $ARGS --output "$TMP/m$np.pk" >/dev/null
    cmp "$TMP/t1.pk" "$TMP/m$np.pk" && cmp "$TMP/t1.pk.vel" "$TMP/m$np.pk.vel" && echo "PASS MPI np=$np matches serial"
  done
else
  echo "SKIP MPI tests (build with: make MPI=1)"
fi

# Gadget binary (format 1, format 2, multi-file, mixed types) must reproduce the HDF5 result
PY=$(for p in python3.13 python3.12 python3.11 python3; do $p -c 'import numpy,h5py' 2>/dev/null && { echo $p; break; }; done)
if [ -n "$PY" ]; then
  $PY "$ROOT/tests/make_gadget.py" "$DATA" "$TMP/gad"
  B="--nmesh 48 --scheme cic --velocity --nell 3"
  "$BUILD/cosmokit" --input "$TMP/gad/g1" --ptype 1,2 $B --output "$TMP/g1.pk" >/dev/null
  "$BUILD/cosmokit" --input "$TMP/gad/g2" --ptype 1,2 $B --output "$TMP/g2.pk" >/dev/null
  "$BUILD/cosmokit" --input "$TMP/gad/m.0" --ptype 1 $B --chunk 20000 --output "$TMP/gm.pk" >/dev/null
  for n in g1 g2 gm; do cmp "$TMP/t1.pk" "$TMP/$n.pk" && cmp "$TMP/t1.pk.vel" "$TMP/$n.pk.vel" && echo "PASS gadget binary ($n) matches HDF5"; done
else
  echo "SKIP Gadget binary tests (need numpy+h5py)"
fi
