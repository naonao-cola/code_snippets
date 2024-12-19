
#ifndef CALIPER_ALGOH_H
#define CALIPER_ALGOH_H

#ifdef __cplusplus
extern "C" {
#endif

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

#ifdef __cplusplus
}
#endif

#endif

