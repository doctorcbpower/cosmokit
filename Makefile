# cosmokit Makefile
#
#   make                 # serial + OpenMP build:  build/libcosmokit.{so,dylib}, build/cosmokit
#   make MPI=1           # adds MPI:                build-mpi/...
#   make OPENMP=0        # disable OpenMP
#   make HDF5=0          # no HDF5 (no snapshot reader / field output)
#   make smoke           # quick sanity check on the sample snapshot (seconds)
#   make test            # run the full test-suite (needs numpy for the Python tests)
#
# Library locations are auto-detected from Homebrew (macOS) or taken from the
# environment / command line:  FFTW_PREFIX=...  HDF5_PREFIX=...  GCC=...

OPENMP ?= 1
MPI    ?= 0
HDF5   ?= 1
DEBUG  ?= 0

UNAME := $(shell uname -s)
BREW  := $(shell brew --prefix 2>/dev/null)
FFTW_PREFIX ?= $(if $(BREW),$(BREW),/usr)
HDF5_PREFIX ?= $(if $(BREW),$(BREW),/usr)

# Apple clang has no OpenMP, so prefer a Homebrew GCC on macOS.
GCC ?= $(or $(shell command -v gcc-15 2>/dev/null),$(shell command -v gcc-14 2>/dev/null),$(shell command -v gcc-13 2>/dev/null),gcc)

ifeq ($(MPI),1)
  CC := mpicc
  export OMPI_CC := $(GCC)
  export MPICH_CC := $(GCC)
  BUILD ?= build-mpi
else
  CC := $(GCC)
  BUILD ?= build
endif

ifeq ($(UNAME),Darwin)
  SOEXT := dylib
  SHFLAGS := -dynamiclib -install_name @rpath/libcosmokit.$(SOEXT)
else
  SOEXT := so
  SHFLAGS := -shared
endif

CFLAGS  ?= -O3 -std=gnu11 -fPIC -Wall -Wextra -Wno-unused-parameter
ifeq ($(DEBUG),1)
  CFLAGS := -O0 -g -std=gnu11 -fPIC -Wall -Wextra -Wno-unused-parameter -fsanitize=address
endif
CFLAGS  += -Iinclude -Isrc -I$(FFTW_PREFIX)/include
LDLIBS  := -L$(FFTW_PREFIX)/lib -lfftw3

ifeq ($(OPENMP),1)
  CFLAGS += -fopenmp -DCK_ENABLE_OPENMP
  LDLIBS += -lfftw3_omp
endif
ifeq ($(MPI),1)
  CFLAGS += -DCK_ENABLE_MPI
  LDLIBS += -lfftw3_mpi
endif
ifeq ($(HDF5),1)
  CFLAGS += -DCK_ENABLE_HDF5 -I$(HDF5_PREFIX)/include
  LDLIBS += -L$(HDF5_PREFIX)/lib -lhdf5
endif
LDLIBS += -lm

LIBSRC := grid assign power tidal kdtree sph io io_hdf5 io_gadget
LIBOBJ := $(addprefix $(BUILD)/,$(addsuffix .o,$(LIBSRC)))
LIB    := $(BUILD)/libcosmokit.$(SOEXT)
CLI    := $(BUILD)/cosmokit

all: $(LIB) $(CLI)

$(BUILD):
	mkdir -p $(BUILD)

$(BUILD)/%.o: src/%.c src/ck_internal.h include/cosmokit.h | $(BUILD)
	$(CC) $(CFLAGS) -c $< -o $@

$(LIB): $(LIBOBJ)
	$(CC) $(CFLAGS) $(SHFLAGS) -o $@ $^ $(LDLIBS) -Wl,-rpath,$(FFTW_PREFIX)/lib

$(CLI): src/cli.c $(LIB)
	$(CC) $(CFLAGS) -o $@ src/cli.c -L$(BUILD) -lcosmokit $(LDLIBS) -Wl,-rpath,$(abspath $(BUILD))

PYTHON ?= $(shell for p in python3.13 python3.12 python3.11 python3; do $$p -c 'import numpy' 2>/dev/null && { echo $$p; break; }; done)

smoke: all
	sh tests/smoke.sh $(abspath $(BUILD))

test: all
	COSMOKIT_LIB=$(abspath $(LIB)) $(PYTHON) tests/test_core.py
	sh tests/test_cli.sh $(abspath $(BUILD))

clean:
	rm -rf build build-mpi

.PHONY: all smoke test clean
