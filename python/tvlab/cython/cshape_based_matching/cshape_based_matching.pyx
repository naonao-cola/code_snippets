# cython: language_level=3
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
