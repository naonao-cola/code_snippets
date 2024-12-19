#ifndef SBM_ALGORITHM_H
#define SBM_ALGORITHM_H


#ifdef __cplusplus
extern "C" {
#endif

void* detector_new(int features_num, int levels_num, int *levels,
        float weak_threshold, float strong_threshold, int gaussion_kenel);
void detector_free(void* det);

void save_template(void *det, const char *info_dir);
void load_template(void *det, const char *info_dir, const char *class_ids);

int add_template(void* det, char *img_ptr, char *mask,
         int img_w, int img_h, int img_c, const char *class_id,
	     float angle_start, float angle_stop, float angle_step,
	     float scale_start, float scale_stop, float scale_step,
         int (*cbf)(int, int));

int show_template(void * det, char *img_ptr, int img_w, int img_h, int img_c,
                   const char *class_id, int template_id);

int match_template(void *det, char *img_ptr, char *debug_img_ptr,
        int img_w, int img_h, int img_c,
        float threshold, float iou_threshold,
        const char *class_ids,
        int top_k, int subpixel,
        float** matches, char** pmatches_id);
void matches_free(float *matches, char* matches_id);

#ifdef __cplusplus
}
#endif
#endif
