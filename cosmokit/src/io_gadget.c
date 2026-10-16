/* io_gadget.c - stream Gadget-1/2 binary snapshots (format 1 and 2, native
 * endianness, single or multi-file) onto the mesh in chunks, like io_hdf5.c.
 *
 * Block layout: each block is [int size][payload][int size]; format 2 puts an
 * extra 8-byte label block ([8]["POS "][size+8][8]) in front of every block.
 * Format-1 blocks are identified by order: HEAD, POS, VEL, ID, MASS.
 */
#include "ck_internal.h"

#include <sys/stat.h>

#define NTYPES 6

typedef struct {
  char name[1024];
  int32_t npart[NTYPES];
  double massarr[NTYPES];
  double box;
  int32_t nfiles;
  long pos_off, vel_off, mass_off; /* payload offsets; -1 if absent */
} gfile;

static int exists(const char *p) {
  struct stat st;
  return stat(p, &st) == 0;
}

static int rd_i32(FILE *f, int32_t *v) { return fread(v, 4, 1, f) == 1; }

/* Position f at the payload of the next block; returns payload size in *sz and
 * label (format 2) in lab (or ""). Leaves file position at payload start. */
static int next_block(FILE *f, int fmt2, int32_t *sz, char lab[5]) {
  int32_t a;
  lab[0] = 0;
  if (fmt2) {
    int32_t b, c;
    if (!rd_i32(f, &a) || a != 8) return 0;
    if (fread(lab, 1, 4, f) != 4) return 0;
    lab[4] = 0;
    if (!rd_i32(f, &b) || !rd_i32(f, &c) || c != 8) return 0;
  }
  if (!rd_i32(f, sz)) return 0;
  return 1;
}

static int parse_file(const char *name, gfile *g) {
  snprintf(g->name, sizeof g->name, "%s", name);
  FILE *f = fopen(name, "rb");
  if (!f) CK_FAIL(CK_ERR_IO, "cannot open '%s'", name);
  int32_t first;
  if (!rd_i32(f, &first)) { fclose(f); CK_FAIL(CK_ERR_IO, "'%s' is empty", name); }
  const int fmt2 = (first == 8);
  if (!fmt2 && first != 256) {
    fclose(f);
    CK_FAIL(CK_ERR_IO, "'%s' is not a native-endian Gadget binary snapshot (first word %d)", name, (int)first);
  }
  rewind(f);
  int32_t sz;
  char lab[5];
  if (!next_block(f, fmt2, &sz, lab) || sz != 256) { fclose(f); CK_FAIL(CK_ERR_IO, "bad header block in '%s'", name); }
  unsigned char h[256];
  if (fread(h, 1, 256, f) != 256) { fclose(f); CK_FAIL(CK_ERR_IO, "short header in '%s'", name); }
  memcpy(g->npart, h, 24);
  memcpy(g->massarr, h + 24, 48);
  memcpy(&g->nfiles, h + 124, 4);
  memcpy(&g->box, h + 128, 8);
  if (g->nfiles < 1) g->nfiles = 1;
  int64_t ntot = 0, nmass = 0;
  for (int t = 0; t < NTYPES; t++) {
    ntot += g->npart[t];
    if (g->massarr[t] == 0.0) nmass += g->npart[t];
  }
  fseek(f, 4, SEEK_CUR); /* trailing size of HEAD */

  g->pos_off = g->vel_off = g->mass_off = -1;
  for (int idx = 1; idx < 5; idx++) { /* POS, VEL, ID, MASS */
    if (!next_block(f, fmt2, &sz, lab)) break;
    const long off = ftell(f);
    const char *which = NULL;
    if (fmt2) {
      if (!strncmp(lab, "POS", 3)) which = "pos";
      else if (!strncmp(lab, "VEL", 3)) which = "vel";
      else if (!strncmp(lab, "MASS", 4)) which = "mass";
    } else which = idx == 1 ? "pos" : (idx == 2 ? "vel" : (idx == 4 ? "mass" : NULL));
    if (which) {
      if (!strcmp(which, "pos")) g->pos_off = off;
      if (!strcmp(which, "vel")) g->vel_off = off;
      if (!strcmp(which, "mass")) g->mass_off = off;
      if ((!strcmp(which, "pos") || !strcmp(which, "vel")) && sz != 12 * ntot) {
        fclose(f);
        CK_FAIL(CK_ERR_IO, "%s block in '%s' has %d bytes, expected %lld (float32 x3 per particle)", which, name, (int)sz, (long long)(12 * ntot));
      }
      if (!strcmp(which, "mass") && sz != 4 * nmass) {
        fclose(f);
        CK_FAIL(CK_ERR_IO, "MASS block size mismatch in '%s'", name);
      }
    }
    if (fseek(f, (long)sz + 4, SEEK_CUR)) break;
  }
  fclose(f);
  if (g->pos_off < 0) CK_FAIL(CK_ERR_IO, "no POS block found in '%s'", name);
  return CK_OK;
}

static int resolve(const char *path, int *nfiles, gfile **files) {
  char first[1024];
  snprintf(first, sizeof first, "%s", path);
  if (!exists(first)) snprintf(first, sizeof first, "%s.0", path);
  if (!exists(first)) CK_FAIL(CK_ERR_IO, "cannot find snapshot '%s'", path);
  gfile g0;
  int rc = parse_file(first, &g0);
  if (rc) return rc;
  const int nf = g0.nfiles;
  char base[1024];
  snprintf(base, sizeof base, "%s", first);
  size_t L = strlen(base);
  if (nf > 1) {
    if (L > 2 && !strcmp(base + L - 2, ".0")) base[L - 2] = 0;
    else CK_FAIL(CK_ERR_IO, "multi-file snapshot: pass the path of file '.0' or the base name");
  }
  gfile *fs = calloc((size_t)nf, sizeof *fs);
  if (!fs) CK_FAIL(CK_ERR_ALLOC, "file table");
  for (int i = 0; i < nf; i++) {
    char nm[1100];
    if (nf == 1) snprintf(nm, sizeof nm, "%s", first);
    else snprintf(nm, sizeof nm, "%s.%d", base, i);
    if ((rc = parse_file(nm, &fs[i]))) { free(fs); return rc; }
  }
  *nfiles = nf;
  *files = fs;
  return CK_OK;
}

int ck_boxsize_gadget(const char *path, double *box) {
  int nf;
  gfile *fs;
  int rc = resolve(path, &nf, &fs);
  if (rc) return rc;
  *box = fs[0].box;
  free(fs);
  return CK_OK;
}

static int read_floats(FILE *f, long off, int64_t first, int64_t n, int ncols, float *out) {
  if (fseek(f, off + (long)(4 * ncols) * (long)first, SEEK_SET)) return -1;
  return fread(out, 4, (size_t)(n * ncols), f) == (size_t)(n * ncols) ? 0 : -1;
}

int ck_deposit_gadget(const char *path, unsigned ptype_mask, ck_grid *rho, ck_grid *mom[3],
                      ck_scheme scheme, int64_t chunk, double vel_factor, double *box_out,
                      int64_t *npart_out) {
  if (!rho) CK_FAIL(CK_ERR_ARG, "null grid");
  ck_context *c = rho->ctx;
  if (chunk <= 0) chunk = 1 << 22;
  int nfiles;
  gfile *fs;
  int rc = resolve(path, &nfiles, &fs);
  if (rc) return rc;
  if (box_out) *box_out = fs[0].box;
  if (mom && fs[0].vel_off < 0) { free(fs); CK_FAIL(CK_ERR_IO, "snapshot has no VEL block"); }
  if (fs[0].box > 0.0 && fabs(fs[0].box - c->box) > 1e-6 * c->box)
    fprintf(stderr, "cosmokit: warning: snapshot BoxSize=%g differs from context box=%g\n", fs[0].box, c->box);

  int64_t total = 0;
  for (int f = 0; f < nfiles; f++)
    for (int t = 0; t < NTYPES; t++)
      if (ptype_mask & (1u << t)) total += fs[f].npart[t];
  if (npart_out) *npart_out = total;

  const int64_t S = c->size, R = c->rank;
  const int64_t my0 = total * R / S, my1 = total * (R + 1) / S;
  int64_t max_rounds = (my1 - my0 + chunk - 1) / chunk;
  if ((rc = ck_allreduce_max_i64(c, &max_rounds, 1))) { free(fs); return rc; }

  typedef struct { int f, t; int64_t row, n; } seg_t;
  size_t nseg = 0, cap = 64;
  seg_t *seg = malloc(cap * sizeof *seg);
  int64_t cursor = 0;
  for (int f = 0; f < nfiles; f++)
    for (int t = 0; t < NTYPES; t++) {
      if (!(ptype_mask & (1u << t))) continue;
      const int64_t n = fs[f].npart[t];
      const int64_t a = cursor > my0 ? cursor : my0, b = cursor + n < my1 ? cursor + n : my1;
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
  FILE *fp = NULL;
  int fp_idx = -1;
  for (int64_t round = 0; round < max_rounds; round++) {
    if (!pos) {
      pos = malloc((size_t)chunk * 3 * sizeof(float));
      mass = malloc((size_t)chunk * sizeof(float));
      if (want_vel) vel = malloc((size_t)chunk * 3 * sizeof(float));
      if (!pos || !mass || (want_vel && !vel)) { rc = ck_set_error(CK_ERR_ALLOC, "chunk buffers"); goto done; }
    }
    int64_t filled = 0;
    while (filled < chunk && si < nseg) {
      const seg_t *s = &seg[si];
      const gfile *g = &fs[s->f];
      const int64_t take = (s->n - seg_done) < (chunk - filled) ? (s->n - seg_done) : (chunk - filled);
      if (fp_idx != s->f) {
        if (fp) fclose(fp);
        fp = fopen(g->name, "rb");
        fp_idx = s->f;
        if (!fp) { rc = ck_set_error(CK_ERR_IO, "cannot open '%s'", g->name); goto done; }
      }
      int64_t tstart = 0, mstart = 0; /* first particle of type t; index into MASS block */
      for (int t = 0; t < s->t; t++) {
        tstart += g->npart[t];
        if (g->massarr[t] == 0.0) mstart += g->npart[t];
      }
      const int64_t r0 = s->row + seg_done;
      int bad = read_floats(fp, g->pos_off, tstart + r0, take, 3, pos + 3 * filled);
      if (!bad && want_vel) bad = read_floats(fp, g->vel_off, tstart + r0, take, 3, vel + 3 * filled);
      if (!bad) {
        if (g->massarr[s->t] != 0.0) for (int64_t i = 0; i < take; i++) mass[filled + i] = (float)g->massarr[s->t];
        else if (g->mass_off >= 0) bad = read_floats(fp, g->mass_off, mstart + r0, take, 1, mass + filled);
        else bad = 1;
      }
      if (bad) { rc = ck_set_error(CK_ERR_IO, "failed reading type %d from '%s'", s->t, g->name); goto done; }
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
  if (fp) fclose(fp);
  free(pos); free(vel); free(mass); free(seg); free(fs);
  return rc;
}
