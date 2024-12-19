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