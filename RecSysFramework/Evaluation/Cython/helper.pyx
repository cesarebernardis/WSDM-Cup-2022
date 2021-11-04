#cython: boundscheck=False
#cython: wraparound=False
#cython: initializedcheck=False
#cython: language_level=3
#cython: nonecheck=False
#cython: cdivision=True
#cython: unpack_method_calls=True
#cython: overflowcheck=False

#defining NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

import numpy as np
cimport numpy as np

from cython.parallel import prange

from libc.math cimport exp, sqrt
from libc.stdlib cimport srand, rand, RAND_MAX

def ncr(int n, int r):
    cdef float result = 1
    cdef int c
    r = min(r, n - r)
    for c in range(r):
        result *= (n - c) / (r - c)
    return result



def ordered_in1d(int[:] a, int[:] b):
    cdef int i, j
    cdef int size = 0
    cdef int[:] result = np.empty(min(len(a), len(b)), dtype=np.int32)

    while i < len(a) and j < len(b):
        if a[i] < b[j]:
            i += 1
        elif a[i] > b[j]:
            j += 1
        else:
            result[size] = a[i]
            i += 1
            j += 1

    return np.array(result[:size])


cdef double hypergeometric_pmf(int N, int n, int K, int k) nogil:

    cdef double result = 1.
    cdef double k_i = 0
    cdef double nmk_i = 0

    if k > K or k < 0:
        return 0.
    if n > N or n < 0:
        return 0.
    if n-k < 0 or n-k > N-K:
        return 0.

    while k_i < k:
        result *= (n - k_i) * (K - k_i) / ((k - k_i) * (N - k_i))
        k_i += 1

    while nmk_i < (n - k):
        result *= (N - K - nmk_i) / (N - k_i - nmk_i)
        nmk_i += 1

    return result


def hypergeometric_pmf_matrix(int N, int n):

    cdef int k = 0
    cdef int K = 0

    cdef double[:, :] result = np.zeros((N, n), dtype=np.float64)

    for K in prange(N, nogil=True):
        for k in range(n):
            result[K, k] = hypergeometric_pmf(N, n-1, K, k)

    return np.array(result)

