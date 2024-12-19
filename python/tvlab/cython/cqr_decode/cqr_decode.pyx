# cython: language_level=3
import os
cimport numpy as cnp
import numpy as np
from cpython cimport array
from libc.string cimport memcpy
import array
from cpython.bytes cimport PyBytes_FromStringAndSize
cnp.import_array()

cdef extern from "quirc.h":
    void *create_quirc(unsigned char * img_ptr, int w, int h, cnp.int32_t * qr_infos_ptr, int *id_cnt)
    void destory_quirc(void *q)
    void quirc_extract_decode(void *q, int index, int *size, int *ecc_level,
                              int *mask, int *data_type, int *eci,
                              unsigned char* payload, int *payload_len, int *xy,
                              int *score, int *ecc_rate,
                              char* cell_bitmap, char *err_desc);

ECC_LEVEL_MAP  = {0: 'M', 1: 'L', 2:'H', 3:'Q'}
DATA_TYPE_MAP = {1: 'numeric', 2: 'alpha', 4: 'byte', 8: 'kanji'}

def cqr_decode(cnp.ndarray[unsigned char, ndim=2, mode="c"] img not None,
               cnp.ndarray[cnp.int32_t, ndim=2, mode="c"] qr_infos, debug=False):
    cdef int img_h, img_w
    cdef unsigned char* img_ptr = <unsigned char*>img.data
    cdef int id_cnt
    cdef int size, ecc_level, payload_len, mask, data_type, eci, score, ecc_rate
    cdef unsigned char payload[8896]
    cdef int xy[8]
    cdef char err_desc[1024]
    cdef char cell_bitmap[3917*8]
    cdef char* cell_bitmap_ptr = NULL

    if debug:
        cell_bitmap_ptr = cell_bitmap

    img_h, img_w = img.shape[0], img.shape[1]
    cdef cnp.int32_t* qr_infos_ptr = NULL
    if qr_infos is not None:
        qr_infos_ptr = <cnp.int32_t*>qr_infos.data
        id_cnt = qr_infos.shape[0]

    cdef void* q = create_quirc(img_ptr, img_w, img_h, qr_infos_ptr, &id_cnt)

    result = []
    for i in range(id_cnt):
        size = 0
        ecc_level = 0
        payload_len = 0
        mask = 0
        data_type = 0
        eci = 0
        score = 0
        ecc_rate = 0

        quirc_extract_decode(q, i, &size, &ecc_level, &mask,
                            &data_type, &eci,
                            payload, &payload_len, xy,
                            &score, &ecc_rate,
                            cell_bitmap_ptr, err_desc)
        info = {'version': int((size-17)/4),
                'size': size,
                'score': score,
                'polygon': [xy[0], xy[1], xy[2], xy[3],
                            xy[4], xy[5], xy[6], xy[7]]}
        if debug:
            cell_bitmap_bytes = PyBytes_FromStringAndSize(cell_bitmap_ptr, (size*size))
            bitmap_arr = np.frombuffer(cell_bitmap_bytes, np.uint8)
            bitmap_arr = bitmap_arr.reshape(size, size)
            info['bitmap'] = bitmap_arr

        if payload_len == 0:
            info['err_desc'] = err_desc.decode('utf-8')
        else:
            info['ecc'] = 'unknown' if ecc_level not in ECC_LEVEL_MAP else ECC_LEVEL_MAP[ecc_level]
            info['ecc_rate'] = ecc_rate
            info['mask'] = mask
            info['data_type'] = 'unknown' if data_type not in DATA_TYPE_MAP else DATA_TYPE_MAP[data_type]
            info['eci'] = eci
            info['data'] = payload.decode('utf-8')

        result.append(info)

    destory_quirc(q)
    return result
