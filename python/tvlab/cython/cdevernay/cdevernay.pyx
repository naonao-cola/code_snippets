# cython: language_level=3
import os
cimport numpy as cnp
import numpy as np
from cpython cimport array
from libc.string cimport memcpy
from libc.stdlib cimport free
import array
from cpython.bytes cimport PyBytes_FromStringAndSize
cnp.import_array()

cdef extern from "devernay.h":
    void devernay(double ** x, double ** y, int * N, int ** curve_limits, int * M,
            double * image, int X, int Y,
            double sigma, double th_h, double th_l);


def c_devernay(cnp.ndarray[double, ndim=2, mode="c"] img not None, sigma=0.0, th_l=0.0, th_h=0.0):
    cdef int img_h, img_w
    cdef double* img_ptr = <double*>img.data
    cdef double *x
    cdef double *y
    cdef int * curve_limits
    cdef int M, N

    img_h, img_w = img.shape[0], img.shape[1]
    devernay(&x, &y, &N, &curve_limits, &M, img_ptr, img_w, img_h, sigma, th_h, th_l)
    result = []
    for k in range(M):
        result.append([(x[i], y[i]) for i in range(curve_limits[k], curve_limits[k+1])])

    free(curve_limits)
    free(x)
    free(y)
    return result
