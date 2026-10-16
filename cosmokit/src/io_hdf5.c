/* io_hdf5.c - stream Gadget-4 / Arepo / SWIFT HDF5 snapshots onto the mesh.
 *
 * Each rank takes a contiguous share of the (selected-type) particle list that
 * spans all files, reads it in `chunk`-sized pieces, and hands every piece to
 * ck_assign (which redistributes across ranks as needed). Peak memory per rank
 * is therefore the grid plus one chunk, independent of snapshot size.
 */
#include "ck_internal.h"

#ifndef CK_ENABLE_HDF5

int ck_deposit_hdf5(const char *path, unsigned ptype_mask, ck_grid *rho, ck_grid *mom[3],
                        ck_scheme scheme, int64_t chunk, double vel_factor, double *box_out,
                        int64_t *npart_out) {
  (void)path; (void)ptype_mask; (void)rho; (void)mom; (void)scheme; (void)chunk;
  (void)vel_factor; (void)box_out; (void)npart_out;
  CK_FAIL(CK_ERR_UNSUPPORTED, "built without HDF5 support");
}
int ck_boxsize_hdf5(const char *path, double *box) {
  (void)path; (void)box;
  CK_FAIL(CK_ERR_UNSUPPORTED, "built without HDF5 support");
}

#else

#include <hdf5.h>
#include <sys/stat.h>

#define NTYPES 6

typedef struct {
  char name[1024];
  int64_t npart[NTYPES];
  double masstable[NTYPES];
} file_info;

static int file_exists(const char *p) {
  struct stat st;
  return stat(p, &st) == 0;
}

static int read_attr_double(hid_t grp, const char *name, double *out, int n) {
  hid_t a = H5Aopen(grp, name, H5P_DEFAULT);
  if (a < 0) return -1;
  hid_t sp = H5Aget_space(a);
  const hssize_t cnt = H5Sget_simple_extent_npoints(sp);
  double *buf = malloc((size_t)cnt * sizeof(double));
  int rc = H5Aread(a, H5T_NATIVE_DOUBLE, buf) < 0 ? -1 : 0;
  if (rc == 0) for (int i = 0; i < n; i++) out[i] = i < cnt ? buf[i] : buf[0];
  free(buf);
  H5Sclose(sp);
  H5Aclose(a);
  return rc;
}

static int read_attr_i64(hid_t grp, const char *name, int64_t *out, int n) {
  hid_t a = H5Aopen(grp, name, H5P_DEFAULT);
  if (a < 0) return -1;
  hid_t sp = H5Aget_space(a);
  const hssize_t cnt = H5Sget_simple_extent_npoints(sp);
  int64_t *buf = malloc((size_t)cnt * sizeof(int64_t));
  int rc = H5Aread(a, H5T_NATIVE_INT64, buf) < 0 ? -1 : 0;
  if (rc == 0) for (int i = 0; i < n; i++) out[i] = i < cnt ? buf[i] : 0;
  free(buf);
  H5Sclose(sp);
  H5Aclose(a);
  return rc;
}

/* Resolve the file list. path may be the single file, "base.0.hdf5", or "base". */
static int resolve_files(const char *path, int *nfiles, file_info **files, double *box) {
  char first[1024];
  snprintf(first, sizeof first, "%s", path);
  if (!file_exists(first)) snprintf(first, sizeof first, "%s.0.hdf5", path);
  if (!file_exists(first)) CK_FAIL(CK_ERR_IO, "cannot find snapshot '%s'", path);

  hid_t f = H5Fopen(first, H5F_ACC_RDONLY, H5P_DEFAULT);
  if (f < 0) CK_FAIL(CK_ERR_IO, "cannot open '%s'", first);
  hid_t h = H5Gopen2(f, "/Header", H5P_DEFAULT);
  if (h < 0) { H5Fclose(f); CK_FAIL(CK_ERR_IO, "'%s' has no /Header group", first); }
  int64_t nf = 1;
  read_attr_i64(h, "NumFilesPerSnapshot", &nf, 1);
  if (nf < 1) nf = 1;
  if (box) {
    double b[3] = {0, 0, 0};
    if (read_attr_double(h, "BoxSize", b, 3) < 0) b[0] = 0.0;
    *box = b[0];
  }
  H5Gclose(h);
  H5Fclose(f);

  /* derive the base name for multi-file snapshots */
  char base[1024];
  snprintf(base, sizeof base, "%s", first);
  size_t L = strlen(base);
  if (nf > 1) {
    if (L > 7 && strcmp(base + L - 7, ".0.hdf5") == 0) base[L - 7] = 0;
    else CK_FAIL(CK_ERR_IO, "multi-file snapshot: pass the path of file 0 ('...0.hdf5') or the base name");
  }
  file_info *fi = calloc((size_t)nf, sizeof *fi);
  if (!fi) CK_FAIL(CK_ERR_ALLOC, "file table");
  for (int64_t i = 0; i < nf; i++) {
    if (nf == 1) snprintf(fi[i].name, sizeof fi[i].name, "%s", first);
    else snprintf(fi[i].name, sizeof fi[i].name, "%s.%lld.hdf5", base, (long long)i);
    hid_t ff = H5Fopen(fi[i].name, H5F_ACC_RDONLY, H5P_DEFAULT);
    if (ff < 0) { char nm[1024]; snprintf(nm, sizeof nm, "%s", fi[i].name); free(fi); CK_FAIL(CK_ERR_IO, "cannot open '%s'", nm); }
    hid_t hh = H5Gopen2(ff, "/Header", H5P_DEFAULT);
    int ok = hh >= 0 && read_attr_i64(hh, "NumPart_ThisFile", fi[i].npart, NTYPES) == 0;
    if (hh >= 0) {
      for (int t = 0; t < NTYPES; t++) fi[i].masstable[t] = 0.0;
      read_attr_double(hh, "MassTable", fi[i].masstable, NTYPES);
      H5Gclose(hh);
    }
    H5Fclose(ff);
    if (!ok) { char nm[1024]; snprintf(nm, sizeof nm, "%s", fi[i].name); free(fi); CK_FAIL(CK_ERR_IO, "cannot read NumPart_ThisFile from '%s'", nm); }
  }
  *nfiles = (int)nf;
  *files = fi;
  return CK_OK;
}

int ck_boxsize_hdf5(const char *path, double *box) {
  int nf;
  file_info *fi;
  int rc = resolve_files(path, &nf, &fi, box);
  if (rc == CK_OK) free(fi);
  return rc;
}

/* Read rows [start, start+count) of a dataset with `ncols` columns as float. */
static int read_rows(hid_t grp, const char *name, hsize_t start, hsize_t count, int ncols,
                     float *buf) {
  hid_t ds = H5Dopen2(grp, name, H5P_DEFAULT);
  if (ds < 0) return -1;
  hid_t fs = H5Dget_space(ds);
  hsize_t off[2] = {start, 0}, cnt[2] = {count, (hsize_t)ncols};
  const int rank = ncols > 1 ? 2 : 1;
  H5Sselect_hyperslab(fs, H5S_SELECT_SET, off, NULL, cnt, NULL);
  hid_t ms = H5Screate_simple(rank, cnt, NULL);
  int rc = H5Dread(ds, H5T_NATIVE_FLOAT, ms, fs, H5P_DEFAULT, buf) < 0 ? -1 : 0;
  H5Sclose(ms);
  H5Sclose(fs);
  H5Dclose(ds);
  return rc;
}

int ck_deposit_hdf5(const char *path, unsigned ptype_mask, ck_grid *rho, ck_grid *mom[3],
                        ck_scheme scheme, int64_t chunk, double vel_factor, double *box_out,
                        int64_t *npart_out) {
  if (!rho) CK_FAIL(CK_ERR_ARG, "null grid");
  ck_context *c = rho->ctx;
  if (chunk <= 0) chunk = 1 << 22;
  int nfiles;
  file_info *fi;
  double box = 0.0;
  int rc = resolve_files(path, &nfiles, &fi, &box);
  if (rc) return rc;
  if (box_out) *box_out = box;
  if (box > 0.0 && fabs(box - c->box) > 1e-6 * c->box)
    fprintf(stderr, "cosmokit: warning: snapshot BoxSize=%g differs from context box=%g\n", box, c->box);

  int64_t total = 0;
  for (int f = 0; f < nfiles; f++)
    for (int t = 0; t < NTYPES; t++)
      if (ptype_mask & (1u << t)) total += fi[f].npart[t];
  if (npart_out) *npart_out = total;

  /* my slice of the global (file-major, type-minor) particle list */
  const int64_t S = c->size, R = c->rank;
  const int64_t my0 = total * R / S, my1 = total * (R + 1) / S;
  int64_t max_rounds = 0;
  {
    const int64_t mine = my1 - my0;
    max_rounds = (mine + chunk - 1) / chunk;
  }
  if ((rc = ck_allreduce_max_i64(c, &max_rounds, 1))) { free(fi); return rc; }

  /* Build a work list of (file, type, first row, rows) segments for this rank */
  typedef struct { int f, t; int64_t row, n; } seg_t;
  size_t nseg = 0, cap = 64;
  seg_t *seg = malloc(cap * sizeof *seg);
  int64_t cursor = 0;
  for (int f = 0; f < nfiles; f++)
    for (int t = 0; t < NTYPES; t++) {
      if (!(ptype_mask & (1u << t))) continue;
      const int64_t n = fi[f].npart[t];
      const int64_t a = cursor > my0 ? cursor : my0;
      const int64_t b = cursor + n < my1 ? cursor + n : my1;
      if (b > a) {
        if (nseg == cap) { cap *= 2; seg = realloc(seg, cap * sizeof *seg); }
        seg[nseg++] = (seg_t){f, t, a - cursor, b - a};
      }
      cursor += n;
    }

  float *pos = NULL, *vel = NULL, *mass = NULL;
  const int want_vel = (mom != NULL);
  size_t si = 0;
  int64_t seg_done = 0;
  for (int64_t round = 0; round < max_rounds; round++) {
    /* assemble up to `chunk` particles, possibly spanning several segments */
    int64_t filled = 0;
    if (!pos) {
      pos = malloc((size_t)chunk * 3 * sizeof(float));
      mass = malloc((size_t)chunk * sizeof(float));
      if (want_vel) vel = malloc((size_t)chunk * 3 * sizeof(float));
      if (!pos || !mass || (want_vel && !vel)) { rc = ck_set_error(CK_ERR_ALLOC, "chunk buffers"); goto done; }
    }
    while (filled < chunk && si < nseg) {
      const seg_t *s = &seg[si];
      const int64_t take = (s->n - seg_done) < (chunk - filled) ? (s->n - seg_done) : (chunk - filled);
      hid_t f = H5Fopen(fi[s->f].name, H5F_ACC_RDONLY, H5P_DEFAULT);
      char gname[32];
      snprintf(gname, sizeof gname, "/PartType%d", s->t);
      hid_t g = f >= 0 ? H5Gopen2(f, gname, H5P_DEFAULT) : -1;
      if (g < 0) { if (f >= 0) H5Fclose(f); rc = ck_set_error(CK_ERR_IO, "cannot open %s in %s", gname, fi[s->f].name); goto done; }
      const hsize_t r0 = (hsize_t)(s->row + seg_done);
      int bad = read_rows(g, "Coordinates", r0, (hsize_t)take, 3, pos + 3 * filled);
      if (!bad && want_vel) bad = read_rows(g, "Velocities", r0, (hsize_t)take, 3, vel + 3 * filled);
      if (!bad) {
        if (H5Lexists(g, "Masses", H5P_DEFAULT) > 0) bad = read_rows(g, "Masses", r0, (hsize_t)take, 1, mass + filled);
        else for (int64_t i = 0; i < take; i++) mass[filled + i] = (float)fi[s->f].masstable[s->t];
      }
      H5Gclose(g);
      H5Fclose(f);
      if (bad) { rc = ck_set_error(CK_ERR_IO, "failed reading %s from %s", gname, fi[s->f].name); goto done; }
      filled += take;
      seg_done += take;
      if (seg_done == s->n) { si++; seg_done = 0; }
    }
    if (want_vel && vel_factor != 1.0)
      for (int64_t i = 0; i < 3 * filled; i++) vel[i] *= (float)vel_factor;
    ck_particles p = {filled, pos, CK_FLOAT32, vel, CK_FLOAT32, mass, CK_FLOAT32, 1.0};
    if ((rc = ck_assign(rho, mom, &p, scheme))) goto done;
  }
  rc = CK_OK;
done:
  free(pos); free(vel); free(mass); free(seg); free(fi);
  return rc;
}

#endif /* CK_ENABLE_HDF5 */
