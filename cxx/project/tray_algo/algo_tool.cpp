#include "algo_tool.h"

/**
 * @FilePath     : /code_snippets/cxx/project/tray_algo/algo_tool.cpp
 * @Description  :
 * @Author       : naonao
 * @Date         : 2025-01-06 14:21:28
 * @Version      : 0.0.1
 * @LastEditors  : naonao
 * @LastEditTime : 2025-01-06 14:21:28
 * @Copyright (c) 2025 by G, All Rights Reserved.
**/
namespace tool {
int yen(std::vector<int> data);
int otsu(std::vector<int> data);
int li(std::vector<int> data);
int minimum(std::vector<int> data);
int min_errorI(std::vector<int> data);
int huang(std::vector<int> data);

std::optional<cv::Mat> get_equal_img(const cv::Mat &hist_img,
                                     const cv::Rect &hist_rect,
                                     const cv::Mat &template_img,
                                     const cv::Rect &template_rect)
{
    cv::Mat hist1, hist2;
    const int channels[1] = {0};
    float inRanges[2] = {0, 255};
    const float *ranges[1] = {inRanges};
    const int bins[1] = {256};
    // 保留ROI 区域的直方图
    cv::Mat img_1 = hist_img(hist_rect).clone();
    cv::Mat img_2 = template_img(template_rect).clone();
    cv::calcHist(&img_1, 1, channels, cv::Mat(), hist1, 1, bins, ranges, true, false);
    cv::calcHist(&img_2, 1, channels, cv::Mat(), hist2, 1, bins, ranges, true, false);
    float hist1_cdf[256] = {hist1.at<float>(0)};
    float hist2_cdf[256] = {hist2.at<float>(0)};
    for (int i = 1; i < 256; ++i) {
        hist1_cdf[i] = hist1_cdf[i - 1] + hist1.at<float>(i);
        hist2_cdf[i] = hist2_cdf[i - 1] + hist2.at<float>(i);
    }
    // 归一化，两幅图像大小可能不一致
    for (int i = 0; i < 256; i++) {
        hist1_cdf[i] = hist1_cdf[i] / (img_1.rows * img_1.cols);
        hist2_cdf[i] = hist2_cdf[i] / (img_1.rows * img_1.cols);
    }
    float diff_cdf[256][256];
    for (int i = 0; i < 256; ++i) {
        for (int j = 0; j < 256; ++j) {
            diff_cdf[i][j] = fabs(hist1_cdf[i] - hist2_cdf[j]);
        }
    }
    cv::Mat lut(1, 256, CV_8U);
    for (int i = 0; i < 256; ++i) {
        float min = diff_cdf[i][0];
        int index = 0;
        for (int j = 1; j < 256; ++j) {
            if (min > diff_cdf[i][j]) {
                min = diff_cdf[i][j];
                index = j;
            }
        }
        lut.at<uchar>(i) = (uchar)index;
    }
    cv::Mat result, hist3;
    cv::LUT(hist_img, lut, result);
    return result;
}

cv::Mat GetRotateCropImage(const cv::Mat &srcimage,
                           std::vector<std::vector<int>> box)
{
    cv::Mat image;
    srcimage.copyTo(image);
    std::vector<std::vector<int>> points = box;
    int x_collect[4] = {box[0][0], box[1][0], box[2][0], box[3][0]};
    int y_collect[4] = {box[0][1], box[1][1], box[2][1], box[3][1]};
    int left = int(*std::min_element(x_collect, x_collect + 4));
    int right = int(*std::max_element(x_collect, x_collect + 4));
    int top = int(*std::min_element(y_collect, y_collect + 4));
    int bottom = int(*std::max_element(y_collect, y_collect + 4));
    cv::Mat img_crop;
    image(cv::Rect(left, top, right - left, bottom - top)).copyTo(img_crop);

    for (int i = 0; i < points.size(); i++) {
        points[i][0] -= left;
        points[i][1] -= top;
    }
    int img_crop_width = int(sqrt(pow(points[0][0] - points[1][0], 2) +
                                  pow(points[0][1] - points[1][1], 2)));
    int img_crop_height = int(sqrt(pow(points[0][0] - points[3][0], 2) +
                                   pow(points[0][1] - points[3][1], 2)));
    cv::Point2f pts_std[4], pointsf[4];
    pts_std[0] = cv::Point2f(0., 0.);
    pts_std[1] = cv::Point2f(img_crop_width, 0.);
    pts_std[2] = cv::Point2f(img_crop_width, img_crop_height);
    pts_std[3] = cv::Point2f(0.f, img_crop_height);
    pointsf[0] = cv::Point2f(points[0][0], points[0][1]);
    pointsf[1] = cv::Point2f(points[1][0], points[1][1]);
    pointsf[2] = cv::Point2f(points[2][0], points[2][1]);
    pointsf[3] = cv::Point2f(points[3][0], points[3][1]);
    cv::Mat M = cv::getPerspectiveTransform(pointsf, pts_std);
    cv::Mat dst_img;
    cv::warpPerspective(img_crop, dst_img, M, cv::Size(img_crop_width, img_crop_height), cv::BORDER_REPLICATE);
    return dst_img;
}

std::optional<cv::Mat> get_equal_img_2(const cv::Mat &hist_img,
                                       const std::vector<cv::Point2f> &hist_pts,
                                       const cv::Mat &template_img,
                                       std::vector<cv::Point2f> &template_pts)
{
    std::vector<std::vector<int>> hist_box{
        {int(hist_pts[0].x), int(hist_pts[0].y)},
        {int(hist_pts[1].x), int(hist_pts[1].y)},
        {int(hist_pts[2].x), int(hist_pts[2].y)},
        {int(hist_pts[3].x), int(hist_pts[3].y)}};
    std::vector<std::vector<int>> temp_box{
        {int(template_pts[0].x), int(template_pts[0].y)},
        {int(template_pts[1].x), int(template_pts[1].y)},
        {int(template_pts[2].x), int(template_pts[2].y)},
        {int(template_pts[3].x), int(template_pts[3].y)}};
    cv::Mat hist1, hist2;
    const int channels[1] = {0};
    float inRanges[2] = {0, 255};
    const float *ranges[1] = {inRanges};
    const int bins[1] = {256};
    // 保留ROI 区域的直方图
    cv::Mat img_1 = GetRotateCropImage(hist_img, hist_box);
    cv::Mat img_2 = GetRotateCropImage(template_img, temp_box);
    cv::calcHist(&img_1, 1, channels, cv::Mat(), hist1, 1, bins, ranges, true, false);
    cv::calcHist(&img_2, 1, channels, cv::Mat(), hist2, 1, bins, ranges, true, false);

    float hist1_cdf[256] = {hist1.at<float>(0)};
    float hist2_cdf[256] = {hist2.at<float>(0)};
    for (int i = 1; i < 256; ++i) {
        hist1_cdf[i] = hist1_cdf[i - 1] + hist1.at<float>(i);
        hist2_cdf[i] = hist2_cdf[i - 1] + hist2.at<float>(i);
    }
    // 归一化，两幅图像大小可能不一致
    for (int i = 0; i < 256; i++) {
        hist1_cdf[i] = hist1_cdf[i] / (img_1.rows * img_1.cols);
        hist2_cdf[i] = hist2_cdf[i] / (img_1.rows * img_1.cols);
    }
    float diff_cdf[256][256];
    for (int i = 0; i < 256; ++i) {
        for (int j = 0; j < 256; ++j) {
            diff_cdf[i][j] = fabs(hist1_cdf[i] - hist2_cdf[j]);
        }
    }
    cv::Mat lut(1, 256, CV_8U);
    for (int i = 0; i < 256; ++i) {
        float min = diff_cdf[i][0];
        int index = 0;
        for (int j = 1; j < 256; ++j) {
            if (min > diff_cdf[i][j]) {
                min = diff_cdf[i][j];
                index = j;
            }
        }
        lut.at<uchar>(i) = (uchar)index;
    }
    cv::Mat result, hist3;
    cv::LUT(hist_img, lut, result);
    return result;
}

std::optional<std::vector<cv::Point2f>> get_rotated_rect_pts(
    const cv::RotatedRect &r_rect)
{
    cv::Point2f box_pts[4];
    r_rect.points(box_pts);
    std::vector<cv::Point2f> pt_vec;
    pt_vec.emplace_back(box_pts[0]);
    pt_vec.emplace_back(box_pts[1]);
    pt_vec.emplace_back(box_pts[2]);
    pt_vec.emplace_back(box_pts[3]);
    std::vector<cv::Point2f> order_pt_vec = order_pts(pt_vec);
    return order_pt_vec;
}

std::optional<std::vector<std::vector<cv::Point>>> get_contours(
    const cv::Mat &src)
{
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> filter_hierarchy;
    cv::findContours(src, contours, filter_hierarchy, cv::RETR_LIST, cv::CHAIN_APPROX_NONE);
    return contours;
}

std::optional<std::vector<cv::Point2f>> get_rotated_rect_pt_trans_and_rotate(
    const cv::RotatedRect &r_rect,
    cv::Mat &img)
{
    cv::Point2f box_pts[4];
    r_rect.points(box_pts);
    std::vector<cv::Point2f> pt_vec;
    pt_vec.emplace_back(box_pts[0]);
    pt_vec.emplace_back(box_pts[1]);
    pt_vec.emplace_back(box_pts[2]);
    pt_vec.emplace_back(box_pts[3]);
    std::vector<cv::Point2f> order_src_pt_vec = order_pts(pt_vec);
    double k = (order_src_pt_vec[1].y - order_src_pt_vec[0].y) /
               (order_src_pt_vec[1].x - order_src_pt_vec[0].x);
    double angle = atanl(k) * 180.0 / CV_PI;

    cv::Point2f center(r_rect.center.x, r_rect.center.y);
    cv::Mat rot_mat = cv::getRotationMatrix2D(center, angle, 1.0);
    int w = img.cols;
    int h = img.rows;
    cv::Size new_size(w, h);
    cv::Mat dst;
    cv::warpAffine(img, dst, rot_mat, new_size, cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar());
    std::vector<cv::Point2f> rect_points(4);
    r_rect.points(rect_points.data());
    std::vector<cv::Point2f> dst_points;
    for (int i = 0; i < rect_points.size(); i++) {
        cv::Point2f pt;
        pt.x = rot_mat.at<double>(0, 0) * rect_points[i].x +
               rot_mat.at<double>(0, 1) * rect_points[i].y +
               rot_mat.at<double>(0, 2);
        pt.y = rot_mat.at<double>(1, 0) * rect_points[i].x +
               rot_mat.at<double>(1, 1) * rect_points[i].y +
               rot_mat.at<double>(1, 2);
        dst_points.emplace_back(pt);
    }
    std::vector<cv::Point2f> order_pt_vec = order_pts(dst_points);
    img = dst.clone();
    return order_pt_vec;
}

cv::Mat gray_stairs(const cv::Mat &img, double sin, double hin, double mt, double sout, double hout)
{
    double Sin = (std::min)((std::max)(sin, 0.0), hin - 2);
    double Hin = (std::min)(hin, 255.0);
    double Mt = (std::min)((std::max)(mt, 0.01), 9.99);
    double Sout = (std::min)((std::max)(sout, 0.0), hout - 2);
    double Hout = (std::min)(hout, 255.0);
    double difin = Hin - Sin;
    double difout = Hout - Sout;
    uchar lutData[256];
    for (int i = 0; i < 256; i++) {
        double v1 = (std::min)((std::max)(255 * (i - Sin) / difin, 0.0), 255.0);
        double v2 = 255 * std::pow(v1 / 255.0, 1.0 / Mt);
        lutData[i] =
            (int)(std::min)((std::max)(Sout + difout * v2 / 255, 0.0), 255.0);
    }
    cv::Mat lut(1, 256, CV_8UC1, lutData);
    cv::Mat dst;
    cv::LUT(img, lut, dst);
    return dst;
}

cv::Mat gamma_trans(const cv::Mat &img, double gamma, int n_c)
{
    gamma = 1.0f / gamma;
    cv::Mat img_gamma;
    uchar LUT[256];
    for (int i = 0; i < 256; i++) {
        float f = (i + 0.5f) / 255;
        f = (float)std::pow(f, gamma);
        LUT[i] = cv::saturate_cast<uchar>(f * 255.0f - 0.5f);
    }
    cv::Mat lut(1, 256, CV_8UC1, LUT);
    cv::LUT(img, lut, img_gamma);
    return img_gamma;
}

void get_histogram(const cv::Mat &src, int *dst)
{
    cv::Mat hist;
    int channels[1] = {0};
    int histSize[1] = {256};
    float hranges[2] = {0, 256.0};
    const float *ranges[1] = {hranges};
    cv::calcHist(&src, 1, channels, cv::Mat(), hist, 1, histSize, ranges);
    for (int i = 0; i < 256; i++) {
        float binVal = hist.at<float>(i);
        dst[i] = int(binVal);
    }
}

int exec_threshold(cv::Mat &src, THRESHOLD_TYPE type, int doIblack, int doIwhite, bool reset)
{
    int threshold = -1;
    if (src.empty() || src.channels() != 1) return threshold;
    const int gray_scale = 256;
    int data[gray_scale] = {0};
    get_histogram(src, data);
    int minbin = -1, maxbin = -1;
    int range_max = gray_scale;
    int rang_min = 0;

    if (std::abs(doIblack + 1) > 1) rang_min = doIblack;
    if (std::abs(doIwhite + 1) > 1) range_max = doIwhite;
    for (int i = 0; i < range_max; i++) {
        if (data[i] > 0) maxbin = i;
    }
    for (int i = gray_scale - 1; i >= rang_min; i--) {
        if (data[i] > 0) minbin = i;
    }
    int scale_range = maxbin - minbin + 1;
    if (scale_range < 2) return 0;

    std::vector<int> data2(scale_range, {0});
    for (int i = minbin; i <= maxbin; i++) {
        data2[i - minbin] = data[i];
    }
    if (type == THRESHOLD_TYPE::OTSU) {
        threshold = otsu(data2);
    } else if (type == THRESHOLD_TYPE::HUANG) {
        threshold = huang(data2);
    } else if (type == THRESHOLD_TYPE::LI) {
        threshold = li(data2);
    } else if (type == THRESHOLD_TYPE::MINIMUM) {
        threshold = minimum(data2);
    } else if (type == THRESHOLD_TYPE::MIN_ERROR) {
        threshold = min_errorI(data2);
    } else if (type == THRESHOLD_TYPE::YEN) {
        threshold = yen(data2);
    } else if (type == THRESHOLD_TYPE::SAUVOLA) {
        sauvola(src);
        return -1;
    }
    threshold += minbin;
    if (reset) {
        cv::threshold(src, src, threshold, 255, cv::THRESH_BINARY);
    }
    return threshold;
}

int otsu(std::vector<int> data)
{
    int ih;
    int threshold = -1;
    int num_pixels = 0;
    double total_mean; ///< 整个图像的平均灰度
    double bcv, term; ///< 类间方差，缩放系数
    double max_bcv; ///< max BCV

    std::vector<double> cnh(data.size(), {0.0}); ///< 累积归一化直方图
    std::vector<double> mean(data.size(), {0.0}); ///< 平均灰度
    std::vector<double> histo(data.size(), {0.0}); ///< 归一化直方图
    // 计算值为非0的像素的个数
    for (ih = 0; ih < data.size(); ih++)
        num_pixels = num_pixels + data[ih];

    // 计算每个灰度级的像素数目占整幅图像的比例,相当于归一化直方图
    term = 1.0 / (double)num_pixels;
    for (ih = 0; ih < data.size(); ih++)
        histo[ih] = term * data[ih];
    // 计算累积归一化直方图
    cnh[0] = histo[0];
    mean[0] = 0.0;
    for (ih = 1; ih < data.size(); ih++) {
        cnh[ih] = cnh[ih - 1] + histo[ih];
        mean[ih] = mean[ih - 1] + ih * histo[ih];
    }
    total_mean = mean[data.size() - 1];
    // 计算每个灰度的BCV，并找到使其最大化的阈值,
    max_bcv = 0.0;
    for (ih = 0; ih < data.size(); ih++) {
        // 通分，约化之后的简写
        bcv = total_mean * cnh[ih] - mean[ih];
        bcv *= bcv / (cnh[ih] * (1.0 - cnh[ih]));
        if (max_bcv < bcv) {
            max_bcv = bcv;
            threshold = ih;
        }
    }
    return threshold;
}
int huang(std::vector<int> data)
{
    int threshold = -1;
    int ih, it;
    int first_bin;
    int last_bin;
    int sum_pix;
    int num_pix;
    double term;
    double ent; // entropy
    double min_ent; // min entropy
    double mu_x;

    /* Determine the first non-zero bin */
    first_bin = 0;
    for (ih = 0; ih < data.size(); ih++) {
        if (data[ih] != 0) {
            first_bin = ih;
            break;
        }
    }

    /* Determine the last non-zero bin */
    last_bin = static_cast<int>(data.size() - 1);
    for (ih = data.size() - 1; ih >= first_bin; ih--) {
        if (data[ih] != 0) {
            last_bin = ih;
            break;
        }
    }
    term = 1.0 / (double)(last_bin - first_bin);
    std::vector<double> mu_0(data.size(), {0.0});
    sum_pix = num_pix = 0;
    for (ih = first_bin; ih < data.size(); ih++) {
        sum_pix += ih * data[ih];
        num_pix += data[ih];
        /* NUM_PIX cannot be zero ! */
        mu_0[ih] = sum_pix / (double)num_pix;
    }

    std::vector<double> mu_1(data.size(), {0.0});
    sum_pix = num_pix = 0;
    for (ih = last_bin; ih > 0; ih--) {
        sum_pix += ih * data[ih];
        num_pix += data[ih];
        /* NUM_PIX cannot be zero ! */
        mu_1[ih - 1] = sum_pix / (double)num_pix;
    }

    /* Determine the threshold that minimizes the fuzzy entropy */
    threshold = -1;
    min_ent = DBL_MAX;

    for (it = 0; it < data.size(); it++) {
        ent = 0.0;
        for (ih = 0; ih <= it; ih++) {
            /* Equation (4) in Ref. 1 */
            mu_x = 1.0 / (1.0 + term * std::abs(ih - mu_0[it]));
            if (!((mu_x < 1e-06) || (mu_x > 0.999999))) {
                /* Equation (6) & (8) in Ref. 1 */
                ent += data[ih] * (-mu_x * std::log(mu_x) - (1.0 - mu_x) * std::log(1.0 - mu_x));
            }
        }

        for (ih = it + 1; ih < data.size(); ih++) {
            /* Equation (4) in Ref. 1 */
            mu_x = 1.0 / (1.0 + term * std::abs(ih - mu_1[it]));
            if (!((mu_x < 1e-06) || (mu_x > 0.999999))) {
                /* Equation (6) & (8) in Ref. 1 */
                ent += data[ih] * (-mu_x * std::log(mu_x) - (1.0 - mu_x) * std::log(1.0 - mu_x));
            }
        }
        /* No need to divide by NUM_ROWS * NUM_COLS * LOG(2) ! */
        if (ent < min_ent) {
            min_ent = ent;
            threshold = it;
        }
    }
    return threshold;
}

int li(std::vector<int> data)
{
    int threshold;
    int ih;
    int num_pixels;
    int sum_back; ///< 给定阈值下背景像素的总和
    int sum_obj; ///< 给定阈值下对象像素的总和
    int num_back; ///< 给定阈值下的背景像素数
    int num_obj; ///< 给定阈值下的对象像素数
    double old_thresh;
    double new_thresh;
    double mean_back; ///< 给定阈值下背景像素的平均值
    double mean_obj; ///< 给定阈值下对象像素的平均值
    double mean; ///< 图像中的平均灰度
    double tolerance; ///< 阈值公差
    double temp;

    tolerance = 0.5;
    num_pixels = 0;
    for (ih = 0; ih < data.size(); ih++)
        num_pixels += data[ih];

    /* Calculate the mean gray-level */
    mean = 0.0;
    for (ih = 0; ih < data.size(); ih++) // 0 + 1?
        mean += ih * data[ih];
    mean /= num_pixels;
    /* Initial estimate */
    new_thresh = mean;

    do {
        old_thresh = new_thresh;
        threshold = (int)(old_thresh + 0.5); /* range */
        /* Calculate the means of background and object pixels */
        /* Background */
        sum_back = 0;
        num_back = 0;
        for (ih = 0; ih <= threshold; ih++) {
            sum_back += ih * data[ih];
            num_back += data[ih];
        }
        mean_back = (num_back == 0 ? 0.0 : (sum_back / (double)num_back));
        /* Object */
        sum_obj = 0;
        num_obj = 0;
        for (ih = threshold + 1; ih < data.size(); ih++) {
            sum_obj += ih * data[ih];
            num_obj += data[ih];
        }
        mean_obj = (num_obj == 0 ? 0.0 : (sum_obj / (double)num_obj));

        /* Calculate the new threshold: Equation (7) in Ref. 2 */
        // new_thresh = simple_round ( ( mean_back - mean_obj ) / ( Math.log (
        // mean_back ) - Math.log ( mean_obj ) ) ); simple_round ( double x ) {
        //  return ( int ) ( IS_NEG ( x ) ? x - .5 : x + .5 );
        // }
        //
        // #define IS_NEG( x ) ( ( x ) < -DBL_EPSILON )
        // DBL_EPSILON = 2.220446049250313E-16
        temp = (mean_back - mean_obj) / (std::log(mean_back) - std::log(mean_obj));

        if (temp < -2.220446049250313E-16)
            new_thresh = (int)(temp - 0.5);
        else
            new_thresh = (int)(temp + 0.5);
        /*  Stop the iterations when the difference between the
        new and old threshold values is less than the tolerance */
    } while (std::abs(new_thresh - old_thresh) > tolerance);
    return threshold;
}

int yen(std::vector<int> data)
{
    int threshold;
    int ih, it;
    double crit;
    double max_crit;

    std::vector<double> norm_histo(data.size(), {0.0}); /* normalized histogram */
    std::vector<double> P1(data.size(),
                           {0.0}); /* cumulative normalized histogram */
    std::vector<double> P1_sq(data.size(), {0.0});
    std::vector<double> P2_sq(data.size(), {0.0});

    int total = 0;
    for (ih = 0; ih < data.size(); ih++)
        total += data[ih];

    for (ih = 0; ih < data.size(); ih++)
        norm_histo[ih] = (double)data[ih] / total;

    P1[0] = norm_histo[0];
    for (ih = 1; ih < data.size(); ih++)
        P1[ih] = P1[ih - 1] + norm_histo[ih];

    P1_sq[0] = norm_histo[0] * norm_histo[0];
    for (ih = 1; ih < data.size(); ih++)
        P1_sq[ih] = P1_sq[ih - 1] + norm_histo[ih] * norm_histo[ih];

    P2_sq[data.size() - 1] = 0.0;
    for (ih = data.size() - 2; ih >= 0; ih--)
        P2_sq[ih] = P2_sq[ih + 1] + norm_histo[ih + 1] * norm_histo[ih + 1];

    /* Find the threshold that maximizes the criterion */
    threshold = -1;
    max_crit = DBL_MIN;
    for (it = 0; it < data.size(); it++) {
        crit =
            -1.0 * ((P1_sq[it] * P2_sq[it]) > 0.0 ? std::log(P1_sq[it] * P2_sq[it])
                                                  : 0.0) +
            2 * ((P1[it] * (1.0 - P1[it])) > 0.0 ? std::log(P1[it] * (1.0 - P1[it]))
                                                 : 0.0);
        if (crit > max_crit) {
            max_crit = crit;
            threshold = it;
        }
    }
    return threshold;
}

int sauvola(cv::Mat &src, const double &k, const int &wnd_size)
{
    CV_Assert(src.type() == CV_8UC1);
    CV_Assert(wnd_size % 2 == 1);
    CV_Assert((wnd_size <= src.cols) && (wnd_size <= src.rows));
    cv::Mat dst = cv::Mat::zeros(src.rows, src.cols, CV_8UC1);

    unsigned long *integralImg = new unsigned long[src.rows * src.cols];
    unsigned long *integralImgSqrt = new unsigned long[src.rows * src.cols];
    std::memset(integralImg, 0, src.rows * src.cols * sizeof(unsigned long));
    std::memset(integralImgSqrt, 0, src.rows * src.cols * sizeof(unsigned long));

    // 计算直方图和图像值平方的和,积分图函数(cv::integral),未测试
    // #pragma omp parallel for
    for (int y = 0; y < src.rows; ++y) {
        unsigned long sum = 0;
        unsigned long sqrtSum = 0;
        for (int x = 0; x < src.cols; ++x) {
            int index = y * src.cols + x;
            int value_pix = *src.ptr<uchar>(y, x);
            sum += value_pix;
            sqrtSum += value_pix * value_pix;
            if (y == 0) {
                integralImg[index] = sum;
                integralImgSqrt[index] = sqrtSum;
            } else {
                integralImgSqrt[index] =
                    integralImgSqrt[(y - 1) * src.cols + x] + sqrtSum;
                integralImg[index] = integralImg[(y - 1) * src.cols + x] + sum;
            }
        }
    }
    double diff = 0.0;
    double sqDiff = 0.0;
    double diagSum = 0.0;
    double iDiagSum = 0.0;
    double sqDiagSum = 0.0;
    double sqIDiagSum = 0.0;
    // #pragma omp parallel for
    for (int x = 0; x < src.cols; ++x) {
        for (int y = 0; y < src.rows; ++y) {
            int xMin = std::max(0, x - wnd_size / 2);
            int yMin = std::max(0, y - wnd_size / 2);
            int xMax = std::min(src.cols - 1, x + wnd_size / 2);
            int yMax = std::min(src.rows - 1, y + wnd_size / 2);
            double area = (xMax - xMin + 1) * (yMax - yMin + 1);
            if (area <= 0) {
                // blog提供源码是biImage[i * IMAGE_WIDTH + j] = 255;但是i表示的是列,
                // j是行
                dst.at<uchar>(y, x) = 255;
                continue;
            }
            if (xMin == 0 && yMin == 0) {
                diff = integralImg[yMax * src.cols + xMax];
                sqDiff = integralImgSqrt[yMax * src.cols + xMax];
            } else if (xMin > 0 && yMin == 0) {
                diff = integralImg[yMax * src.cols + xMax] -
                       integralImg[yMax * src.cols + xMin - 1];
                sqDiff = integralImgSqrt[yMax * src.cols + xMax] -
                         integralImgSqrt[yMax * src.cols + xMin - 1];
            } else if (xMin == 0 && yMin > 0) {
                diff = integralImg[yMax * src.cols + xMax] -
                       integralImg[(yMin - 1) * src.cols + xMax];
                sqDiff = integralImgSqrt[yMax * src.cols + xMax] -
                         integralImgSqrt[(yMin - 1) * src.cols + xMax];
            } else {
                diagSum = integralImg[yMax * src.cols + xMax] +
                          integralImg[(yMin - 1) * src.cols + xMin - 1];
                iDiagSum = integralImg[(yMin - 1) * src.cols + xMax] +
                           integralImg[yMax * src.cols + xMin - 1];
                diff = diagSum - iDiagSum;
                sqDiagSum = integralImgSqrt[yMax * src.cols + xMax] +
                            integralImgSqrt[(yMin - 1) * src.cols + xMin - 1];
                sqIDiagSum = integralImgSqrt[(yMin - 1) * src.cols + xMax] +
                             integralImgSqrt[yMax * src.cols + xMin - 1];
                sqDiff = sqDiagSum - sqIDiagSum;
            }
            double mean = diff / area;
            double stdValue = sqrt((sqDiff - diff * diff / area) / (area - 1));
            double threshold = mean * (1 + k * ((stdValue / 128) - 1));
            if (src.at<uchar>(y, x) < threshold) {
                dst.at<uchar>(y, x) = 0;
            } else {
                dst.at<uchar>(y, x) = 255;
            }
        }
    }
    return 0;
}

bool bimodalTest(std::vector<double> y)
{
    int len = static_cast<double>(y.size());
    bool b = false;
    int modes = 0;
    for (int k = 1; k < len - 1; k++) {
        if (y[k - 1] < y[k] && y[k + 1] < y[k]) {
            modes++;
            if (modes > 2) return false;
        }
    }
    if (modes == 2) b = true;
    return b;
}

int minimum(std::vector<int> data)
{
    int iter = 0;
    int threshold = -1;
    int max = -1;
    std::vector<double> iHisto(data.size(), {0.0});

    for (int i = 0; i < data.size(); i++) {
        iHisto[i] = (double)data[i];
        if (data[i] > 0) max = i;
    }
    std::vector<double> tHisto(iHisto.size(),
                               {0.0}); // Instead of double[] tHisto = iHisto ;
    while (!bimodalTest(iHisto)) {
        // 使用3点运行平均值过滤器平滑
        for (int i = 1; i < data.size() - 1; i++)
            tHisto[i] = (iHisto[i - 1] + iHisto[i] + iHisto[i + 1]) / 3;
        tHisto[0] = (iHisto[0] + iHisto[1]) / 3; // 0 outside
        tHisto[data.size() - 1] =
            (iHisto[data.size() - 2] + iHisto[data.size() - 1]) / 3; // 0 outside
        // System.arraycopy(tHisto, 0, iHisto, 0, iHisto.size()); //Instead of
        // iHisto = tHisto ;
        std::copy_n(tHisto.begin(), iHisto.size(), iHisto.begin());
        iter++;
        if (iter > 10000) {
            threshold = -1;
            // IJ.log("Minimum Threshold not found after 10000 iterations.");
            return threshold;
        }
    }
    // 阈值是两个峰值之间的最小值。修改为16位
    for (int i = 1; i < max; i++) {
        // IJ.log(" "+i+"  "+iHisto[i]);
        if (iHisto[i - 1] > iHisto[i] && iHisto[i + 1] >= iHisto[i]) {
            threshold = i;
            break;
        }
    }
    return threshold;
}

long long mean(std::vector<int> data)
{
    long long threshold = -1;
    long long tot = 0, sum = 0;
    for (int i = 0; i < data.size(); i++) {
        tot += data[i];
        sum += ((long)i * (long)data[i]);
    }
    threshold = (int)std::floor(sum / tot);
    return threshold;
}

static double A(std::vector<int> y, int j)
{
    double x = 0;
    for (int i = 0; i <= j; i++)
        x += y[i];
    return x;
}
static double B(std::vector<int> y, int j)
{
    double x = 0;
    for (int i = 0; i <= j; i++)
        x += i * y[i];
    return x;
}
static double C(std::vector<int> y, int j)
{
    double x = 0;
    for (int i = 0; i <= j; i++)
        x += i * i * y[i];
    return x;
}

int min_errorI(std::vector<int> data)
{
    long long threshold = mean(data); // 用均值算法得到阈值的初始估计。
    long long Tprev = -2;
    double mu, nu, p, q, sigma2, tau2, w0, w1, w2, sqterm, temp;
    // int counter=1;
    while (threshold != Tprev) {
        // 计算一些统计数据。
        mu = B(data, threshold) / A(data, threshold);
        nu = (B(data, data.size() - 1) - B(data, threshold)) /
             (A(data, data.size() - 1) - A(data, threshold));
        p = A(data, threshold) / A(data, data.size() - 1);
        q = (A(data, data.size() - 1) - A(data, threshold)) /
            A(data, data.size() - 1);
        sigma2 = C(data, threshold) / A(data, threshold) - (mu * mu);
        tau2 = (C(data, data.size() - 1) - C(data, threshold)) /
                   (A(data, data.size() - 1) - A(data, threshold)) -
               (nu * nu);

        // 要求解的二次方程的项。
        w0 = 1.0 / sigma2 - 1.0 / tau2;
        w1 = mu / sigma2 - nu / tau2;
        w2 = (mu * mu) / sigma2 - (nu * nu) / tau2 +
             std::log10((sigma2 * (q * q)) / (tau2 * (p * p)));

        // 如果下一个阈值是虚构的，则返回当前阈值。
        sqterm = (w1 * w1) - w0 * w2;
        if (sqterm < 0) {
            // IJ.log("MinError(I): not converging. Try \'Ignore black/white\'
            // options");
            return threshold;
        }

        // 更新后的阈值是二次方程解的整数部分。
        Tprev = threshold;
        temp = (w1 + std::sqrt(sqterm)) / w0;

        if (std::isnan(temp)) {
            // IJ.log("MinError(I): NaN, not converging. Try \'Ignore black/white\'
            // options");
            threshold = Tprev;
        } else
            threshold = (int)std::floor(temp);
        // IJ.log("Iter: "+ counter+++"  t:"+threshold);
    }
    return threshold;
}

cv::Point2d TransPoint(const cv::Mat &M, const cv::Point2d &point)
{
    std::vector<double> values = {point.x, point.y};
    cv::Mat mat = cv::Mat(values).clone(); // 将vector变成单列的mat，这里需要clone(),因为这里的赋值操作是浅拷贝
    cv::Mat dest = mat.reshape(1, 1);
    cv::Mat homogeneousPoint = (cv::Mat_<double>(3, 1) << point.x, point.y, 1.0);
    cv::Mat transformed = M * homogeneousPoint;
    return cv::Point2d(transformed.at<double>(0, 0),
                       transformed.at<double>(0, 1));
}
} // namespace tool
