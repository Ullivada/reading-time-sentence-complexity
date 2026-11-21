"""Incremental PCFG parser. One numeric thread per process: the parser runs in a
process pool and multithreaded BLAS would oversubscribe the CPU."""

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")
