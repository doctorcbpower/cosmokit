"""Convert data/L20_N64.hdf5 into Gadget binary snapshots for testing.
usage: make_gadget.py in.hdf5 outdir   -> outdir/g1, outdir/g2 (format 2), outdir/m.{0,1} (2 files)"""
import struct, sys, os
import h5py, numpy as np

src, out = sys.argv[1:3]
os.makedirs(out, exist_ok=True)
with h5py.File(src) as f:
    pos = f["PartType1/Coordinates"][:].astype(np.float32)
    vel = f["PartType1/Velocities"][:].astype(np.float32)
    box = float(f["Header"].attrs["BoxSize"]); m0 = float(f["Header"].attrs["MassTable"][1])

def header(npart, massarr, nall, nfiles):
    h = struct.pack("6i6d", *npart, *massarr)
    h += struct.pack("2d", 0.01, 99.0) + struct.pack("2i", 0, 0)
    h += struct.pack("6I", *nall) + struct.pack("2i", 0, nfiles) + struct.pack("d", box)
    return h.ljust(256, b"\0")

def write(path, fmt2, npart, massarr, nall, nfiles, P, V, M):
    blocks = [("HEAD", header(npart, massarr, nall, nfiles)), ("POS ", P.tobytes()), ("VEL ", V.tobytes()),
              ("ID  ", np.arange(len(P), dtype=np.uint32).tobytes())]
    if M is not None and len(M):
        blocks.append(("MASS", M.astype(np.float32).tobytes()))
    with open(path, "wb") as fh:
        for lab, data in blocks:
            if fmt2:
                fh.write(struct.pack("i4sii", 8, lab.encode(), len(data) + 8, 8))
            fh.write(struct.pack("i", len(data)) + data + struct.pack("i", len(data)))

n = len(pos); k = n // 3
# g1: type1 has explicit masses (MASS block), type2 has massarr -> mixed
npart = [0, k, n - k, 0, 0, 0]
for name, fmt2 in (("g1", 0), ("g2", 1)):
    write(f"{out}/{name}", fmt2, npart, [0, 0, m0, 0, 0, 0], npart, 1, pos, vel, np.full(k, m0))
# two files, each with half of the particles as type 1
h = n // 2
for i, sl in enumerate((slice(0, h), slice(h, n))):
    c = sl.stop - sl.start
    write(f"{out}/m.{i}", 0, [0, c, 0, 0, 0, 0], [0, m0, 0, 0, 0, 0], [0, n, 0, 0, 0, 0], 2, pos[sl], vel[sl], None)
