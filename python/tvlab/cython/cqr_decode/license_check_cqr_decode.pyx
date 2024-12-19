# cython: language_level=3

import sys
try:
    from tvlab.utils.impl import license
except Exception as e:
    print(e)
    sys.exit()

error_info = '''
===================================================================================

A valid license was not found for tvlab. Please contact TuringVision.'

==================================================================================
'''

pubkey_str = '''
-----BEGIN RSA PUBLIC KEY-----
MIGJAoGBAL7t6/3OWg+L1P94CBWhK8njGEyWjfDDvU/N+2CGJvHxkakkdnAoIuv9
ZddrBPcSWKONz0dGx9DG/9301/i+0o/0eYvbTjJR5j+24xuU1aTg/jzRwfBikGMF
P86Qjxc9Sjd7hGi4WG3eZfl9655JcYuIPuKJtXh6vqR7HZZQpvULAgMBAAE=
-----END RSA PUBLIC KEY-----
'''

api_pubkey_str = '''
-----BEGIN RSA PUBLIC KEY-----
MIGJAoGBAI+ZbnMwTtjnKy4mYOzzY+UlDU0Zu2GK+BQeTp03vBvhsIcQk3kaCQX1
13xUK5Agi/m5zg9R/EqW8AljVqY5Nw8Jc+TEminIZLH0gTwBRP00/lIpajAA1dOC
mXBET2eZ5teQ3cWvnbTKSWDyJmtvm39YA0yARWwl3c8CPUGO1NQjAgMBAAE=
-----END RSA PUBLIC KEY-----
'''

g_dongle_verify_result = False

def unhash_fg(fg, password):
    ac = fg
    b = password.encode()
    return bytes([c - b[i%len(b)] for i, c in enumerate(ac)]).decode()

def verify_signature(fingerprint, sign_info, pubkey, package):
    import rsa
    import time
    import base64

    try:
        date = sign_info['date']
        message = fingerprint + ',' + date + ',' + package
        sign = sign_info['sign'].encode()
        sign = base64.b64decode(sign)
        ret = rsa.verify(message.encode(), sign, pubkey)
        cur_date = time.strftime('%Y%m%d')
        if int(cur_date) > int(date):
            return False
    except Exception:
        return False
    return True

def dongle_verify_func(bid, sign, date, last_date):
    import time
    import rsa

    try:
        package = 'tvlab'
        message = bid + ',' + date + ',' + package
        pubkey = rsa.PublicKey.load_pkcs1(pubkey_str.encode())
        ret = rsa.verify(message.encode(), sign, pubkey)
        cur_date = time.strftime('%Y%m%d')
        if (int(last_date) - 7) <= int(cur_date) and int(cur_date) <= int(date):
            global g_dongle_verify_result
            g_dongle_verify_result = True
    except Exception as e:
        pass
    return g_dongle_verify_result

def try_dongle_verify():
    try:
        license.verify_license_dongle(b'tvindsai', dongle_verify_func, lambda x: x[:4] == 'tvlab')
    except Exception as e:
        pass
    return g_dongle_verify_result

def license_check():
    import os, sys, json, rsa, random
    import logging
    import platform
    system = platform.system()
    logger = logging.getLogger()

    if system in ['Linux', 'Windows']:
        if try_dongle_verify():
            return;

    if system in ['Linux', 'Darwin']:
        check_license_path = ['/var/tvlab/license/license.json', '/tmp/tvlab/license/license.json']
    else: # windows
        check_license_path = ['C:/Program Files/tvlab/license/license.json']
    if 'TVLAB_LICENSE_PATH' in os.environ:
        check_license_path.append(os.environ["TVLAB_LICENSE_PATH"])

    for json_path in check_license_path:
        try:
            if os.path.isfile(json_path):
                with open(json_path, 'rt', encoding="utf-8") as fp:
                    sign_info = json.load(fp)
                    pubkey = rsa.PublicKey.load_pkcs1(pubkey_str.encode())
                    api_pubkey = rsa.PublicKey.load_pkcs1(api_pubkey_str.encode())
                    passwd = str(random.randrange(1000000, 99999999))
                    crypto_passwd = rsa.encrypt(passwd.encode(), api_pubkey)
                    fingerprint = license.get_fingerprint(crypto_passwd)
                    fingerprint = unhash_fg(fingerprint, passwd)
                    if verify_signature(fingerprint, sign_info, pubkey, 'tvlab'):
                        return;
                    else:
                        print(f'verify_signature fail: {json_path}')
        except Exception as e:
            logger.fatal(e)
    logger.fatal(error_info)
    sys.exit()
    raise Exception(error_info)

license_check()
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