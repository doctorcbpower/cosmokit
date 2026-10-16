/* io.c - snapshot format detection and dispatch */
#include "ck_internal.h"

#include <sys/stat.h>

/* 1 if the file (or path.0 / path.0.hdf5) begins with the HDF5 signature */
static int looks_hdf5(const char *path) {
  const char *suffix[] = {"", ".0", ".hdf5", ".0.hdf5"};
  for (int i = 0; i < 4; i++) {
    char nm[1100];
    snprintf(nm, sizeof nm, "%s%s", path, suffix[i]);
    FILE *f = fopen(nm, "rb");
    if (!f) continue;
    unsigned char sig[4] = {0};
    size_t n = fread(sig, 1, 4, f);
    fclose(f);
    return n == 4 && sig[0] == 0x89 && sig[1] == 'H' && sig[2] == 'D' && sig[3] == 'F';
  }
  return -1;
}

int ck_deposit_snapshot(const char *path, unsigned ptype_mask, ck_grid *rho, ck_grid *mom[3],
                        ck_scheme scheme, int64_t chunk, double vel_factor, double *box_out,
                        int64_t *npart_out) {
  const int h = looks_hdf5(path);
  if (h < 0) CK_FAIL(CK_ERR_IO, "cannot find snapshot '%s'", path);
  return (h ? ck_deposit_hdf5 : ck_deposit_gadget)(path, ptype_mask, rho, mom, scheme, chunk,
                                                  vel_factor, box_out, npart_out);
}

int ck_snapshot_boxsize(const char *path, double *box) {
  const int h = looks_hdf5(path);
  if (h < 0) CK_FAIL(CK_ERR_IO, "cannot find snapshot '%s'", path);
  return (h ? ck_boxsize_hdf5 : ck_boxsize_gadget)(path, box);
}
