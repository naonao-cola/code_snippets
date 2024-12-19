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
cimport numpy as np
import numpy as pnp
from tvlab.cv.geometry import Line
from cpython.bytes cimport PyBytes_FromStringAndSize


cdef extern from "caliper.h":
    int find_point(char* img_ptr, int img_w, int img_h, float *xywhr,
            int proj_mode, int filter_size, int polarity, float threshold, int subpix,
            float *out_xy, int debug, float *pts, char *aff_img_ptr, float *proj, float * grad, int *find_idx);
    int find_circle(char* img_ptr, int img_w, int img_h, float *xyrsr,
            int proj_mode, int filter_size, int polarity, float threshold, int subpix,
            int direction, int num_caliper, int side_x, int side_y, int filter_num,
            float *out_xyr, int ret_ext, int *ext_status, float *ext_infos);
    int find_line(char* img_ptr, int img_w, int img_h, float *xyxy,
            int proj_mode, int filter_size, int polarity, float threshold, int subpix,
            int direction, int num_caliper, int side_x, int side_y, int filter_num,
            float *out_xyxy, int ret_ext, int *ext_status, float *ext_infos);

def c_find_point(np.ndarray[char, ndim=2, mode="c"] img not None,
        region, projection_mode, filter_size, polarity, threshold, subpix, debug):
    cdef char* img_ptr = <char*>img.data
    cdef float xywhr[5]
    cdef float out_xy[2]
    cdef float* pts_ptr
    cdef char* aff_img_ptr
    cdef float* proj_ptr
    cdef float* grad_ptr
    cdef int find_idx
    cdef int is_debug = 0
    cdef np.ndarray[float, ndim=2, mode="c"] pts_arr
    cdef np.ndarray[unsigned char, ndim=2, mode="c"] aff_img_arr
    cdef np.ndarray[float, ndim=1, mode="c"] proj_arr
    cdef np.ndarray[float, ndim=1, mode="c"] grad_arr

    xywhr[0] = region[0]
    xywhr[1] = region[1]
    xywhr[2] = region[2]
    xywhr[3] = region[3]
    xywhr[4] = region[4]

    if debug:
        is_debug = 1
        pts_arr = pnp.empty((4,2), dtype=pnp.float32)
        pts_ptr = <float*>pts_arr.data
        aff_img_arr = pnp.empty((int(xywhr[3]), int(xywhr[2])), dtype=pnp.uint8)
        aff_img_ptr = <char*>aff_img_arr.data
        proj_arr = pnp.empty(int(xywhr[2]), dtype=pnp.float32)
        proj_ptr = <float*>proj_arr.data
        grad_arr = pnp.empty(int(xywhr[2]), dtype=pnp.float32)
        grad_ptr = <float*>grad_arr.data

    ret = find_point(img_ptr, img.shape[1], img.shape[0], xywhr, projection_mode,
            filter_size, polarity, threshold, subpix, out_xy,
            is_debug, pts_ptr, aff_img_ptr, proj_ptr, grad_ptr, &find_idx)

    find = True
    if ret != 0:
        find = False
    ret = find, (out_xy[0], out_xy[1])
    if debug:
        debug_info = (pts_arr, aff_img_arr, proj_arr, grad_arr, find_idx)
        ret = find, (out_xy[0], out_xy[1]), debug_info
    return ret


def c_find_circle(np.ndarray[char, ndim=2, mode="c"] img not None,
        region, projection_mode, filter_size, polarity, threshold, subpix,
        direction, num_caliper, side_x, side_y, filter_num, ret_ext):
    cdef char* img_ptr = <char*>img.data
    cdef float xyrsr[5]
    cdef float out_xyr[3]
    cdef int *ext_status
    cdef float *ext_infos
    cdef np.ndarray[int, ndim=1, mode="c"] ext_status_arr
    cdef np.ndarray[float, ndim=2, mode="c"] ext_infos_arr
    cdef int is_ret_ext = 0

    if ret_ext:
        is_ret_ext = 1
        ext_status_arr = pnp.empty(num_caliper, dtype=pnp.int32)
        ext_status = <int*>ext_status_arr.data
        ext_infos_arr = pnp.empty((num_caliper, 11), dtype=pnp.float32)
        ext_infos = <float*>ext_infos_arr.data

    xyrsr[0] = region[0]
    xyrsr[1] = region[1]
    xyrsr[2] = region[2]
    xyrsr[3] = region[3]
    xyrsr[4] = region[4]

    ret = find_circle(img_ptr, img.shape[1], img.shape[0], xyrsr,
            projection_mode, filter_size, polarity, threshold, subpix,
            direction, num_caliper, side_x, side_y, filter_num, out_xyr,
            is_ret_ext, ext_status, ext_infos)

    circle = (out_xyr[0], out_xyr[1], out_xyr[2])
    if ret != 0:
        circle = None
    if ret_ext:
        ext_data = [{'find': ext_status_arr[i] >= 0,
            'point': (ext_infos_arr[i][8], ext_infos_arr[i][9]),
            'roi': ext_infos_arr[i][0:8].copy().reshape(-1, 2),
            'disptoc': ext_infos_arr[i][10],
            'used': ext_status_arr[i] == 0}
            for i in range(num_caliper)]
        return circle, ext_data
    return circle


def c_find_line(np.ndarray[char, ndim=2, mode="c"] img not None,
        region, projection_mode, filter_size, polarity, threshold, subpix,
        direction, num_caliper, side_x, side_y, filter_num, ret_ext):
    cdef char* img_ptr = <char*>img.data
    cdef float xyxy[4]
    cdef float out_xyxy[3]
    cdef int *ext_status
    cdef float *ext_infos
    cdef np.ndarray[int, ndim=1, mode="c"] ext_status_arr
    cdef np.ndarray[float, ndim=2, mode="c"] ext_infos_arr
    cdef int is_ret_ext = 0

    if ret_ext:
        is_ret_ext = 1
        ext_status_arr = pnp.empty(num_caliper, dtype=pnp.int32)
        ext_status = <int*>ext_status_arr.data
        ext_infos_arr = pnp.empty((num_caliper, 11), dtype=pnp.float32)
        ext_infos = <float*>ext_infos_arr.data

    xyxy[0] = region[0]
    xyxy[1] = region[1]
    xyxy[2] = region[2]
    xyxy[3] = region[3]

    ret = find_line(img_ptr, img.shape[1], img.shape[0], xyxy,
            projection_mode, filter_size, polarity, threshold, subpix,
            direction, num_caliper, side_x, side_y, filter_num, out_xyxy,
            is_ret_ext, ext_status, ext_infos)

    line = Line((out_xyxy[0], out_xyxy[1]), (out_xyxy[2], out_xyxy[3]))
    if ret != 0:
        line = None
    if ret_ext:
        ext_data = [{'find': ext_status_arr[i] >= 0,
            'point': (ext_infos_arr[i][8], ext_infos_arr[i][9]),
            'roi': ext_infos_arr[i][0:8].copy().reshape(-1, 2),
            'disptoc': ext_infos_arr[i][10],
            'used': ext_status_arr[i] == 0}
            for i in range(num_caliper)]
        return line, ext_data
    return line