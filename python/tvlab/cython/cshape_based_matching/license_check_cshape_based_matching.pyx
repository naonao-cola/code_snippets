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
cimport numpy as np
from cpython cimport array
import array
from cpython.bytes cimport PyBytes_FromStringAndSize
np.import_array()


cdef extern from "sbm_algo.h":
    void* detector_new(int features_num, int levels_num, int *levels,
        float weak_threshold, float strong_threshold, int gaussion_kenel);
    void detector_free(void* det)
    void save_template(void *det, const char *info_dir);
    void load_template(void *det, const char *info_dir, const char *class_ids);
    int add_template(void* det, char *img, char *mask,
                     int img_w, int img_h, int img_c, const char *class_id,
                     float angle_start, float angle_stop, float angle_step,
                     float scale_start, float scale_stop, float scale_step,
                     int (*cbf)(int, int));
    int show_template(void * det, char *img_ptr, int img_w, int img_h, int img_c,
                      const char *class_id, int template_id);
    int match_template(void *det, char *img, char *debug_img,
                    int img_w, int img_h, int img_c,
                    float threshold, float iou_threshold,
                    const char *class_ids,
                    int top_k, int subpixel,
                    float** matches, char** pmatch_ids)
    void matches_free(float *matches, char* matches_ids);

cdef int default_cb(int i, int n):
    try:
        pystep_cb(i, n)
        return 0
    except KeyboardInterrupt as e:
        print(e)
        print('need exit!')
        return -1

cdef class CShapeBasedMatching:
    cdef void* _c_det

    def __init__(self, features_num=128, T=(4,8),
                 weak_threshold=30.0, strong_threshold=60.0,
                 gaussion_kenel=7):
        '''
        yaml_dir (str): directory for save/load template info.
        features_num (int): number of features
        T (tuple): spred size on each pyramid level
        weak_threshold (float): magnitude threshold for get quantized angle
        strong_threshold (float): magnitude threshold for extract template
        gaussion_kenel (int): for blur input image
        '''
        cdef array.array levels = array.array('i', tuple(T))
        self._c_det = detector_new(features_num, len(T), levels.data.as_ints,
                                   weak_threshold, strong_threshold, gaussion_kenel)
        if self._c_det is NULL:
            raise MemoryError()

    def __dealloc__(self):
        if self._c_det is not NULL:
            detector_free(self._c_det)

    def add(self, np.ndarray[char, ndim=3, mode="c"] img not None,
            np.ndarray[char, ndim=2, mode="c"] mask not None,
            class_id='default',
            angle_range=(0.0, 0.0, 1.0),
            scale_range=(1.0, 1.0, 0.01),
            step_cb=None):
        ''' add template
        img (np.ndarray):
        class_id (str):
        angle_range (tuple): (start, stop, step)
        scale_range (tuple): (start, stop, step)
        '''
        angle_start, angle_stop, angle_step = angle_range
        scale_start, scale_stop, scale_step = scale_range
        cdef char* img_ptr = <char*>img.data
        cdef char* mask_ptr = <char*>mask.data
        py_byte_class_id =  class_id.encode('utf-8')
        cdef const char * class_id_ptr = py_byte_class_id
        cdef int img_h, img_w, img_c
        img_h, img_w, img_c = img.shape[0], img.shape[1], img.shape[2]
        if step_cb is not None:
            global pystep_cb
            pystep_cb = step_cb

        ret = add_template(self._c_det, img_ptr, mask_ptr,
                           img_w, img_h, img_c, class_id_ptr,
                           angle_start, angle_stop, angle_step,
                           scale_start, scale_stop, scale_step,
                           default_cb)

    def show(self, np.ndarray[char, ndim=3, mode="c"] img not None,
             class_id='default', template_id=0):
        ''' show template
        img (np.ndarray):
        class_id (str):
        template_id (int):
        '''
        cdef char* img_ptr = <char*>img.data
        py_byte_class_id =  class_id.encode('utf-8')
        cdef const char * class_id_ptr = py_byte_class_id
        cdef int img_h, img_w, img_c
        img_h, img_w, img_c = img.shape[0], img.shape[1], img.shape[2]
        return show_template(self._c_det, img_ptr, img_w, img_h, img_c,
                             class_id_ptr, template_id)

    def save(self, yaml_dir):
        py_byte_info_dir =  yaml_dir.encode('utf-8')
        cdef const char* c_info_dir = py_byte_info_dir
        os.makedirs(yaml_dir, exist_ok=True)
        save_template(self._c_det, c_info_dir)

    def load(self, yaml_dir, class_ids="default"):
        py_byte_class_id =  class_ids.encode('utf-8')
        cdef const char * class_id_ptr = py_byte_class_id
        py_byte_info_dir =  yaml_dir.encode('utf-8')
        cdef const char* c_info_dir = py_byte_info_dir
        load_template(self._c_det, c_info_dir, class_id_ptr)

    def find(self, np.ndarray[char, ndim=3, mode="c"] img not None,
             np.ndarray[char, ndim=3, mode="c"] debug_img,
             threshold=90, iou_threshold=0.5, class_ids='default',
             topk=-1, subpixel=False):
        '''
        In:
            img (np.ndarray):
            threshold (int): confidence threshold (0 ~ 100)
            class_ids (str): "a,b,c"
            iou_threshold(float): iou threshold for nms.
            topk (int): only keep topk result.
            subpixel (bool): Do subpixel and icp for get more accurate result.
        Out:
            matches_arr (np.ndarray): Nx7 (x, y, w, h, angle, scale, score)
            matches_id (list of str): [class_id0, class_id1, ...]
        '''
        import numpy as pnp

        cdef float* matches = NULL
        cdef char* matches_id = NULL
        cdef char* img_ptr = <char*>img.data
        cdef char* debug_img_ptr = NULL
        cdef int img_h, img_w, img_c
        if debug_img is not None:
            debug_img_ptr = <char*>debug_img.data
        py_byte_class_id =  class_ids.encode('utf-8')
        cdef const char * class_id_ptr = py_byte_class_id
        img_h, img_w, img_c = img.shape[0], img.shape[1], img.shape[2]
        cdef int use_subpixel = 0;
        if subpixel:
            use_subpixel = 1;
        cdef int ret = match_template(self._c_det, img_ptr, debug_img_ptr,
                                      img_w, img_h, img_c,
                                      threshold, iou_threshold, class_id_ptr,
                                      topk, use_subpixel,
                                      &matches, &matches_id)
        cdef int N = ret * 7
        matches_arr = None
        matches_ids = None
        if ret > 0:
            matches_arr = PyBytes_FromStringAndSize(<char*>matches, (N*sizeof(float)))
            matches_arr = pnp.frombuffer(matches_arr, pnp.float32).copy().reshape(-1, 7)
            matches_ids = matches_id.decode('UTF-8')

        matches_free(matches, matches_id)
        return matches_arr, matches_ids