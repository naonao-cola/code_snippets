#include <opencv2/core/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core_c.h>
#include <iostream>
#include <fstream>
#include "caliper.h"

using namespace std;
using namespace cv;

#define PI 3.1415926


int find_first_peak(Mat &y_mat, float height, int distance)
{
    int i;
    int j;
    float ym;
    int is_max;
    float *y = y_mat.ptr<float>();
    for (i=1; i < (y_mat.cols-1); i++) {
        ym = y[i];
        is_max = 1;
        for (j=-distance; j < distance; j++) {
            int k = i+j;
            if (k >= 0 && k < y_mat.cols && y[k] > ym) {
                is_max = 0;
                break;
            }
        }
        if (is_max && ym > height) {
            return i;
        }
    }
    return -1;
}

void gen_rotate_point(float x, float y, float w, float h, float r, float *xy) {
     float a = r * PI / 180;
     float x1 = x - 0.5 * w;
     float y1 = y - 0.5 * h;

     float x1n = (x1 - x) * cos(a) - (y1 - y) * sin(a) + x;
     float y1n = (x1 - x) * sin(a) + (y1 - y) * cos(a) + y;

     xy[0] = x1n;
     xy[1] = y1n;
}


void gen_rotate_rect(float x, float y, float w, float h, float r, vector<Point2f> &pts) {
     float a = r * PI / 180;
     float x1 = x - 0.5 * w;
     float y1 = y - 0.5 * h;
     float x0 = x + 0.5 * w;
     float y0 = y1;
     float x2 = x1;
     float y2 = y + 0.5 * h;
     float x3 = x0;
     float y3 = y2;
     float cos_a = cos(a);
     float sin_a = sin(a);

     float x0n = (x0 - x) * cos_a - (y0 - y) * sin_a + x;
     float y0n = (x0 - x) * sin_a + (y0 - y) * cos_a + y;

     float x1n = (x1 - x) * cos_a - (y1 - y) * sin_a + x;
     float y1n = (x1 - x) * sin_a + (y1 - y) * cos_a + y;

     float x2n = (x2 - x) * cos_a - (y2 - y) * sin_a + x;
     float y2n = (x2 - x) * sin_a + (y2 - y) * cos_a + y;

     float x3n = (x3 - x) * cos_a - (y3 - y) * sin_a + x;
     float y3n = (x3 - x) * sin_a + (y3 - y) * cos_a + y;

     pts[0].x = x1n;
     pts[0].y = y1n;
     pts[1].x = x0n;
     pts[1].y = y0n;
     pts[2].x = x3n;
     pts[2].y = y3n;
     pts[3].x = x2n;
     pts[3].y = y2n;
}

int find_point(char* img_ptr, int img_w, int img_h, float *xywhr,
        int proj_mode, int filter_size, int polarity, float threshold, int subpix,
        float *out_xy, int debug, float *pts, char *aff_img, float *proj, float * grad, int *find_idx) {
    Mat img(img_h, img_w, CV_8UC1, img_ptr);
    vector<Point2f> pts1(4);
    vector<Point2f> pts2(4);

    gen_rotate_rect(xywhr[0], xywhr[1], xywhr[2], xywhr[3], xywhr[4], pts1);

    pts2[0].x = 0;
    pts2[0].y = 0;
    pts2[1].x = xywhr[2];
    pts2[1].y = 0;
    pts2[2].x = xywhr[2];
    pts2[2].y = xywhr[3];
    pts2[3].x = 0;
    pts2[3].y = xywhr[3];

    Mat M = getPerspectiveTransform(pts1, pts2);
    Mat affine_image;
    int aff_w = cvRound(xywhr[2]);
    int aff_h = cvRound(xywhr[3]);
    Size affimg_wh = Size(aff_w, aff_h);
    warpPerspective(img, affine_image, M, affimg_wh, INTER_LINEAR, BORDER_REPLICATE);

    Mat projection(1, affine_image.cols, CV_8UC1);
    int op = CV_REDUCE_AVG;
    if (proj_mode == 1) {
        op = CV_REDUCE_MIN;
    } else if (proj_mode == 2) {
        op = CV_REDUCE_MAX;
    }
    reduce(affine_image, projection, 0, op);

    int kernel_size = 2 * filter_size + 1;
    Mat_<float> kernel = Mat::ones(1, kernel_size, CV_32F);
    kernel(0, filter_size) = 0;
    kernel.colRange(0, filter_size) = -1;
    Mat gradient;
    Point anchor = Point( -1, -1 );
    double delta = 0;

    projection.convertTo(projection, CV_32F);
    filter2D(projection, gradient, -1 , kernel, anchor, delta, BORDER_REPLICATE);

    if (polarity == 0) {
        gradient = abs(gradient);
    } else if (polarity == 1) {
        gradient = -gradient;
    }

    int fidx = find_first_peak(gradient, threshold, min(10, int(aff_w/4)));

    if (debug) {
        pts[0] = pts1[0].x;
        pts[1] = pts1[0].y;
        pts[2] = pts1[1].x;
        pts[3] = pts1[1].y;
        pts[4] = pts1[2].x;
        pts[5] = pts1[2].y;
        pts[6] = pts1[3].x;
        pts[7] = pts1[3].y;

        memcpy(aff_img, affine_image.ptr<char>(), aff_w*aff_h);
        memcpy(proj, projection.ptr<float>(), sizeof(float)*aff_w);
        memcpy(grad, gradient.ptr<float>(), sizeof(float)*aff_w);
        *find_idx = fidx;
    }

    if (fidx == -1)
        return -1;

    float angle = xywhr[4];
    if (fidx > xywhr[2] / 2) {
        angle = xywhr[4] - 180;
    }

    vector<Point2f> pts3(4);
    gen_rotate_point(xywhr[0], xywhr[1], abs((xywhr[2] / 2 - fidx) * 2), 1, angle, out_xy);
    if (subpix) {
        vector<Point2f>	aCorners;
        aCorners.push_back(Point2f(out_xy[0], out_xy[1]));
        TermCriteria tCriteria = TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 30, 0.001 );
        cornerSubPix(img, aCorners, Size(3, 3), Size(-1, -1), tCriteria);
        out_xy[0] = aCorners[0].x;
        out_xy[1] = aCorners[0].y;
    }

    return 0;
}

int fit_circle(vector<Point2f> pts, float *cx, float *cy, float *radius) {
    if (pts.size() < 3) {
        return -2;
    }

    int N = pts.size();
    float sum_x = 0.0;
    float sum_y = 0.0;
    float sum_x2 = 0.0;
    float sum_y2 = 0.0;
    float sum_x3 = 0.0;
    float sum_y3 = 0.0;
    float sum_xy = 0.0;
    float sum_x1y2 = 0.0;
    float sum_x2y1 = 0.0;

    for (int i=0; i < (int)pts.size(); ++i) {
        float x = pts[i].x;
        float y = pts[i].y;
        float x2 = x * x;
        float y2 = y * y;
        sum_x += x;
        sum_y += y;
        sum_x2 += x2;
        sum_y2 += y2;
        sum_x3 += x2 * x;
        sum_y3 += y2 * y;
        sum_xy += x * y;
        sum_x1y2 += x * y2;
        sum_x2y1 += x2 * y;
    }

    float C = N * sum_x2 - sum_x * sum_x;
    float D = N * sum_xy - sum_x * sum_y;
    float E = N * sum_x3 + N * sum_x1y2 - (sum_x2 + sum_y2) * sum_x;
    float G = N * sum_y2 - sum_y * sum_y;
    float H = N * sum_x2y1 + N * sum_y3 - (sum_x2 + sum_y2) * sum_y;
    if (C * G - D * D == 0) {
        return -3;
    }

    float a = (H * D - E * G) / (C * G - D * D);
    float b = (H * C - E * D) / (D * D - G * C);
    float c = -(a * sum_x + b * sum_y + sum_x2 + sum_y2) / N;

    *cx = a / (-2);
    *cy = b / (-2);
    *radius = sqrt(a * a + b * b - 4 * c) / 2;
    return 0;
}

float cal_ptop_dis(float x1, float y1, float x2, float y2) {
    return sqrt(pow((x2 - x1), 2) + pow((y2 - y1), 2));
}

int find_circle(char* img_ptr, int img_w, int img_h, float *xyrsr,
        int proj_mode, int filter_size, int polarity, float threshold, int subpix,
        int direction, int num_caliper, int side_x, int side_y, int filter_num,
        float *out_xyr, int ret_ext, int *ext_status, float *ext_infos) {
    vector<Point2f> find_pts;
    vector<int> caliper_idxs;
    float angle_step = xyrsr[4] / num_caliper;
    
    for (int i = 0; i < num_caliper; i++) {
        float out_xy[2];
        float xywhr[5];
        float ori_angle = xyrsr[3] + i * angle_step;
        float angle = ori_angle;
        gen_rotate_point(xyrsr[0], xyrsr[1], xyrsr[2] * 2, 1, angle, out_xy);
        if (direction == 0) {
            angle = angle - 180;
        }
        xywhr[0] = out_xy[0];
        xywhr[1] = out_xy[1];
        xywhr[2] = side_x;
        xywhr[3] = side_y;
        xywhr[4] = angle;
        float find_xy[2];
        int ret = find_point(img_ptr, img_w, img_h, xywhr,
                  proj_mode, filter_size, polarity, threshold, subpix, find_xy,
                  0, 0, 0, 0, 0, 0);
        if (ret == 0) {
            find_pts.push_back(Point2f(find_xy[0], find_xy[1]));
        }
        if (ret_ext) {
            float * ext_info = ext_infos + i * 11;
            vector<Point2f> pts(4);
            gen_rotate_rect(out_xy[0], out_xy[1], side_x, side_y, ori_angle, pts);
            ext_status[i] = ret;
            ext_info[0] = pts[0].x;
            ext_info[1] = pts[0].y;
            ext_info[2] = pts[1].x;
            ext_info[3] = pts[1].y;
            ext_info[4] = pts[2].x;
            ext_info[5] = pts[2].y;
            ext_info[6] = pts[3].x;
            ext_info[7] = pts[3].y;
            ext_info[8] = find_xy[0];
            ext_info[9] = find_xy[1];
            caliper_idxs.push_back(i);
        }
    }

    float cx, cy, radius;
    int ret = fit_circle(find_pts, &cx, &cy, &radius);
    if (ret != 0)
        return ret;

    if (find_pts.size() > (filter_num + 3)) {
        for (int i=0; i < filter_num; i++) {
            float max_dis = 0;
            int max_index = -1;
            for (int j=0; j < find_pts.size(); j++) {
                float dis = abs(cal_ptop_dis(find_pts[j].x, find_pts[j].y, cx, cy) - radius);
                if (max_dis < dis) {
                    max_dis = dis;
                    max_index = j;
                }
                if (ret_ext) {
                    int k = caliper_idxs[j];
                    float * ext_info = ext_infos + k * 11;
                    ext_info[10] = dis;
                }
            }

            if (ret_ext) {
                int j = caliper_idxs[max_index];
                ext_status[j] = 1;
                caliper_idxs.erase(caliper_idxs.begin() + max_index);
            }
            find_pts.erase(find_pts.begin() + max_index);
            fit_circle(find_pts, &cx, &cy, &radius);
        }
    }
    out_xyr[0] = cx;
    out_xyr[1] = cy;
    out_xyr[2] = radius;
    return 0;
}

float cal_angle(float start_x, float start_y, float end_x, float end_y) {
    float angle = 0.0;
    angle = 90 + atan2((end_y - start_y), (end_x - start_x)) * 180 / PI;
    if (angle > 180) {
        angle = angle - 360;
    }
    return angle;
}

float cal_ptol_dis(float px, float py, float *xyxy) {
    float a = xyxy[1] - xyxy[3];
    float b = xyxy[2] - xyxy[0];
    float c = xyxy[0] * xyxy[3] - xyxy[1] * xyxy[2];
    return fabs(a * px + b * py + c) / sqrt(a * a + b * b);
}

int fit_line(vector<Point2f> pts, int img_w, int img_h, float *xyxy) {
    if (pts.size() < 2) {
        return -2;
    }
    Vec4f line;
    fitLine(pts, line, DIST_L2, 0, 0.01, 0.01);

    float k = line[1] / line[0];
    float b = line[3] - k * line[2];

    float sx = 0;
    float sy = sx * k + b;
    if (sy > img_h || img_h < 0) {
        if (sy > img_h) {
            sy = img_h - 1;
        } else if (sy < 0) {
            sy = 0;
        }
        sx = (sy - b) / k;
    }

    float ex = img_w;
    float ey = ex * k + b;
    if (ey > img_h || ey < 0) {
        if (ey > img_h) {
            ey = img_h - 1;
        } else {
            ey = 0;
        }
        ex = (ey - b) / k;
    }

    xyxy[0] = sx;
    xyxy[1] = sy;
    xyxy[2] = ex;
    xyxy[3] = ey;
    return 0;
}

int find_line(char* img_ptr, int img_w, int img_h, float *xyxy,
        int proj_mode, int filter_size, int polarity, float threshold, int subpix,
        int direction, int num_caliper, int side_x, int side_y, int filter_num,
        float *out_xyxy, int ret_ext, int *ext_status, float *ext_infos) {
    vector<Point2f> find_pts;
    vector<int> caliper_idxs;
    float angle = cal_angle(xyxy[0], xyxy[1], xyxy[2], xyxy[3]);
    if (direction == 1) {
        angle = angle - 180;
    }
    
    float disx = (xyxy[2] - xyxy[0]) / (num_caliper-1);
    float disy = (xyxy[3] - xyxy[1]) / (num_caliper-1);

    for (int i = 0; i < num_caliper; i++) {
        float xywhr[5];
        xywhr[0] = xyxy[0] + disx * i;
        xywhr[1] = xyxy[1] + disy * i;
        xywhr[2] = side_x;
        xywhr[3] = side_y;
        xywhr[4] = angle;
        float find_xy[2];
        int ret = find_point(img_ptr, img_w, img_h, xywhr,
                  proj_mode, filter_size, polarity, threshold, subpix, find_xy,
                  0, 0, 0, 0, 0, 0);
        if (ret == 0) {
            find_pts.push_back(Point2f(find_xy[0], find_xy[1]));
        }
        if (ret_ext) {
            float * ext_info = ext_infos + i * 11;
            vector<Point2f> pts(4);
            gen_rotate_rect(xywhr[0], xywhr[1], side_x, side_y, angle, pts);
            ext_status[i] = ret;
            ext_info[0] = pts[0].x;
            ext_info[1] = pts[0].y;
            ext_info[2] = pts[1].x;
            ext_info[3] = pts[1].y;
            ext_info[4] = pts[2].x;
            ext_info[5] = pts[2].y;
            ext_info[6] = pts[3].x;
            ext_info[7] = pts[3].y;
            ext_info[8] = find_xy[0];
            ext_info[9] = find_xy[1];
            caliper_idxs.push_back(i);
        }
    }

    int ret = fit_line(find_pts, img_w, img_h, out_xyxy);
    if (ret != 0)
        return ret;

    if (find_pts.size() > (filter_num + 3)) {
        for (int i=0; i < filter_num; i++) {
            float max_dis = 0;
            int max_index = 0;
            for (int j=0; j < find_pts.size(); j++) {
                float dis = cal_ptol_dis(find_pts[j].x, find_pts[j].y, out_xyxy);
                if (max_dis < dis) {
                    max_dis = dis;
                    max_index = j;
                }
                if (ret_ext) {
                    int k = caliper_idxs[j];
                    float * ext_info = ext_infos + k * 11;
                    ext_info[10] = dis;
                }
            }

            if (ret_ext) {
                int j = caliper_idxs[max_index];
                ext_status[j] = 1;
                caliper_idxs.erase(caliper_idxs.begin() + max_index);
            }
            find_pts.erase(find_pts.begin() + max_index);
            fit_line(find_pts, img_w, img_h, out_xyxy);
        }
    }
    return 0;
}
