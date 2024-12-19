# cython: language_level=3
import os
cimport numpy as np
from cpython cimport array
import array
np.import_array()


cdef extern from "et199_api.h":
    int get_device_cnt(int debug);
    int open_device(int index);
    void close_device(int index);
    int get_hardware_id(int index, unsigned char *bid);
    int write_sign(int index, unsigned char *user_pin, unsigned char *sign, int size);
    int read_sign(int index, unsigned char *user_pin, unsigned char *sign, int size);
    int get_atr(int index, unsigned char *atr);
    int set_atr(int index, unsigned char *atr);


cdef class CET199Dongle:
    cdef int debug
    def __init__(self, debug=0):
        debug = 1 if debug else 0
        self.debug = debug

    def get_cnt(self):
        return get_device_cnt(self.debug)

    def open(self, i):
        return open_device(i)

    def close(self, i):
        return close_device(i)

    def get_hardware_id(self, i, np.ndarray[char, ndim=1, mode="c"] bid not None):
        cdef unsigned char* bid_ptr = <unsigned char*>bid.data
        return get_hardware_id(i, bid_ptr)

    def write_sign(self, i, np.ndarray[char, ndim=1, mode="c"] user_pin not None,
            np.ndarray[char, ndim=1, mode="c"] sign not None):
        cdef unsigned char* upin_ptr = <unsigned char*>user_pin.data
        cdef unsigned char* sign_ptr = <unsigned char*>sign.data
        return write_sign(i, upin_ptr, sign_ptr, sign.size)

    def read_sign(self, i, np.ndarray[char, ndim=1, mode="c"] user_pin not None,
            np.ndarray[char, ndim=1, mode="c"] sign not None):
        cdef unsigned char* upin_ptr = <unsigned char*>user_pin.data
        cdef unsigned char* sign_ptr = <unsigned char*>sign.data
        return read_sign(i, upin_ptr, sign_ptr, sign.size)

    def set_atr(self, i, np.ndarray[char, ndim=1, mode="c"] atr not None):
        cdef unsigned char* atr_ptr = <unsigned char*>atr.data
        return set_atr(i, atr_ptr)

    def get_atr(self, i, np.ndarray[char, ndim=1, mode="c"] atr not None):
        cdef unsigned char* atr_ptr = <unsigned char*>atr.data
        return get_atr(i, atr_ptr)
