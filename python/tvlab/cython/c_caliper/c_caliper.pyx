# cython: language_level=3
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
