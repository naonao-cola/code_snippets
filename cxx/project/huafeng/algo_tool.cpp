#include "algo_tool.h"
namespace connector {
/*
前向声明的函数
*/
// int sauvola(cv::Mat& src, const double k = 0.1, const int wnd_size = 7);
int yen(std::vector<int> data);
int otsu(std::vector<int> data);
int li(std::vector<int> data);
int moments(std::vector<int>data);

cv::Point2d get2lineIPoint(cv::Point2d lineOnePt1, cv::Point2d lineOnePt2, cv::Point2d lineTwoPt1, cv::Point2d lineTwoPt2)
{
    double x1 = lineOnePt1.x, y1 = lineOnePt1.y, x2 = lineOnePt2.x, y2 = lineOnePt2.y;
    double a1 = -(y2 - y1), b1 = x2 - x1, c1 = (y2 - y1) * x1 - (x2 - x1) * y1; // 一般式：a1x+b1y1+c1=0
    double x3 = lineTwoPt1.x, y3 = lineTwoPt1.y, x4 = lineTwoPt2.x, y4 = lineTwoPt2.y;
    double a2 = -(y4 - y3), b2 = x4 - x3, c2 = (y4 - y3) * x3 - (x4 - x3) * y3; // 一般式：a2x+b2y1+c2=0
    bool r = false; // 判断结果
    double x0 = 0, y0 = 0; // 交点
    double angle = 0; // 夹角

    cv::Point2d result(-1, -1);
    // 判断相交
    if (b1 == 0 && b2 != 0) // l1垂直于x轴，l2倾斜于x轴
        r = true;
    else if (b1 != 0 && b2 == 0) // l1倾斜于x轴，l2垂直于x轴
        r = true;
    else if (b1 != 0 && b2 != 0 && a1 / b1 != a2 / b2)
        r = true;

    if (r) {
        // 计算交点
        x0 = (b1 * c2 - b2 * c1) / (a1 * b2 - a2 * b1);
        y0 = (a1 * c2 - a2 * c1) / (a2 * b1 - a1 * b2);
        // 计算夹角
        double a = sqrt(pow(x4 - x2, 2) + pow(y4 - y2, 2));
        double b = sqrt(pow(x4 - x0, 2) + pow(y4 - y0, 2));
        double c = sqrt(pow(x2 - x0, 2) + pow(y2 - y0, 2));
        angle = acos((b * b + c * c - a * a) / (2 * b * c)) * 180 / CV_PI;
    }
    result.x = x0;
    result.y = y0;
    return result;
}

double get_line_x(cv::Point2f line_p1, cv::Point2f line_p2, double y)
{

    double x1 = line_p1.x, y1 = line_p1.y, x2 = line_p2.x, y2 = line_p2.y;
    double x = (y - y1) * (x2 - x1) / (y2 - y1) + x1;
    return x;
}

double get_line_y(cv::Point2f line_p1, cv::Point2f line_p2, double x)
{
    double x1 = line_p1.x, y1 = line_p1.y, x2 = line_p2.x, y2 = line_p2.y;
    double y = (x - x1) * (y2 - y1) / (x2 - x1) + y1;
    return y;
}

cv::Point2f calculate_foot_point(cv::Point2f line_pt1, cv::Point2f line_pt2, cv::Point2f src_pt)
{

    cv::Point2f root_pt(0, 0);
    if (line_pt1.x == line_pt2.x) {
        // 线与x轴垂直
        root_pt.x = line_pt1.x;
        root_pt.y = src_pt.y;
    } else if (line_pt1.y == line_pt2.y) {
        // 线与Y轴垂直
        root_pt.x = src_pt.x;
        root_pt.y = line_pt1.y;
    } else {
        // 线与 x轴 y轴 都不垂直
        double a1 = -(line_pt2.y - line_pt1.y);
        double b1 = (line_pt2.x - line_pt1.x);
        double c1 = (line_pt2.y - line_pt1.y) * line_pt1.x - (line_pt2.x - line_pt1.x) * line_pt1.y;

        root_pt.x = (b1 * b1 * src_pt.x - a1 * b1 * src_pt.y - a1 * c1) / (a1 * a1 + b1 * b1);
        root_pt.y = (a1 * a1 * src_pt.y - a1 * b1 * src_pt.x - b1 * c1) / (a1 * a1 + b1 * b1);
    }
    return root_pt;
}

double dist_p2p(const cv::Point2f& a, const cv::Point2f& b)
{
    return std::sqrt(std::pow(a.x - b.x, 2) + std::pow(a.y - b.y, 2));
}

float dist_p2l(cv::Point pointP, cv::Point pointA, cv::Point pointB)
{
    // 求直线方程
    int A = 0, B = 0, C = 0;
    A = pointA.y - pointB.y;
    B = pointB.x - pointA.x;
    C = pointA.x * pointB.y - pointA.y * pointB.x;
    // 代入点到直线距离公式
    float distance = 0;
    distance = ((float)abs(A * pointP.x + B * pointP.y + C)) / ((float)sqrtf(A * A + B * B));
    return distance;
}

cv::Point2f get_lines_fangcheng(const Tival::FindLineResult& ll)
{
    float k = 0; // 直线斜率
    float b = 0; // 直线截距

    double x_diff = 0;
    if (abs(ll.start_point[0].x - ll.end_point[0].x) < 2) {
        x_diff = abs(ll.start_point[0].x - ll.end_point[0].x);
    }
    k = (double)(ll.start_point[0].y - ll.end_point[0].y) /*(lines[i][3] - lines[i][1])*/ / (double)(x_diff /*ll.start_point.x - ll.end_point.x*/) /*(lines[i][2] - lines[i][0])*/; // 求出直线的斜率// -3.1415926/2-----+3.1415926/2
    b = /*(double)lines[i][1] - k * (double)lines[i][0]*/ (double)ll.end_point[0].y - k * (double)ll.end_point[0].x;
    cv::Point2f pt(k, b);
    return pt;
}

// 暂时放弃
Tival::FindLineResult get_med_line(const Tival::FindLineResult& ll, const Tival::FindLineResult& lr)
{

    // cv::Point2f pl= get_lines_fangcheng(ll);
    // cv::Point2f pr = get_lines_fangcheng(lr);
    // float  final_fangcheng_k, final_fangcheng_b;
    ////中线的截距与斜率
    // final_fangcheng_k = (pl.x + pr.x) / 2;
    // final_fangcheng_b = (pl.y + pr.y) / 2;
    // 左边线的两个点
    cv::Point2f ll_p1 = ll.start_point[0].y < ll.end_point[0].y ? ll.start_point[0] : ll.end_point[0];
    cv::Point2f ll_p2 = ll.start_point[0].y > ll.end_point[0].y ? ll.start_point[0] : ll.end_point[0];
    // 右边线的两个点
    cv::Point2f lr_p1 = lr.start_point[0].y < lr.end_point[0].y ? lr.start_point[0] : lr.end_point[0];
    cv::Point2f lr_p2 = lr.start_point[0].y > lr.end_point[0].y ? lr.start_point[0] : lr.end_point[0];

    // 中线上两个点的坐标
    cv::Point2f m_p1, m_p2;
    m_p1.x = (ll_p1.x + lr_p1.x) / 2;
    m_p1.y = (ll_p1.y + lr_p1.y) / 2;
    m_p2.x = (ll_p2.x + lr_p2.x) / 2;
    m_p2.y = (ll_p2.y + lr_p2.y) / 2;

    Tival::FindLineResult ret_line;
    ret_line.start_point[0] = m_p1;
    ret_line.end_point[0] = m_p2;
    return ret_line;
}

Tival::FindLineResult get_med_line_2(const Tival::FindLineResult& ll, const Tival::FindLineResult& lr, const Tival::FindLineResult& lb)
{

    // 将基座线向上平移66 左右
    Tival::FindLineResult tmp_lb = lb;
    tmp_lb.start_point[0].y = lb.start_point[0].y - 68;
    tmp_lb.end_point[0].y = lb.end_point[0].y - 68;
    tmp_lb.mid_point[0].y = lb.mid_point[0].y - 68;

    // 求交点
    cv::Point2d pt_l = get2lineIPoint(ll.start_point[0], ll.end_point[0], tmp_lb.start_point[0], tmp_lb.end_point[0]);
    cv::Point2d pt_r = get2lineIPoint(lr.start_point[0], lr.end_point[0], tmp_lb.start_point[0], tmp_lb.end_point[0]);
    // 中点
    cv::Point2d center_p1((pt_l.x + pt_r.x) / 2, (pt_l.y + pt_r.y) / 2);

    cv::Point2d foot = calculate_foot_point(lb.start_point[0], lb.end_point[0], center_p1);
    Tival::FindLineResult ret_line;
    ret_line.start_point.push_back(center_p1);
    ret_line.end_point.push_back(foot);
    return ret_line;
}
cv::Mat cvMat6_to_cvMat9(const cv::Mat& mtx6)
{
    cv::Mat mtx9(3, 3, CV_64FC1);
    double* M9 = mtx9.ptr<double>();
    const double* M6 = mtx6.ptr<double>();
    M9[0] = M6[0];
    M9[1] = M6[1];
    M9[2] = M6[2];
    M9[3] = M6[3];
    M9[4] = M6[4];
    M9[5] = M6[5];
    M9[6] = 0.0;
    M9[7] = 0.0;
    M9[8] = 1.0;
    return mtx9;
}

cv::Mat d6_to_cvMat(double d0, double d1, double d2, double d3, double d4, double d5)
{
    cv::Mat mtx(3, 3, CV_64FC1);
    double* M = mtx.ptr<double>();
    M[0] = d0;
    M[1] = d1;
    M[2] = d2;
    M[3] = d3;
    M[4] = d4;
    M[5] = d5;
    M[6] = 0.0;
    M[7] = 0.0;
    M[8] = 1.0;
    return mtx;
}

cv::Mat vector_angle_to_M(double x1, double y1, double d1, double x2, double y2, double d2)
{
    cv::Point2d center(x1, y1);
    double angle = d2 - d1;
    cv::Mat rot_M = cv::getRotationMatrix2D(center, angle, 1.0);
    rot_M = cvMat6_to_cvMat9(rot_M);

    cv::Mat trans_M = d6_to_cvMat(1, 0, x2 - x1, 0, 1, y2 - y1);
    cv::Mat M = trans_M * rot_M; // 先旋转在平移（矩阵乘法相反）
    return M;
}

cv::Point2d TransPoint(const cv::Mat& M, const cv::Point2d& point)
{
    std::vector<double> values = { point.x, point.y };
    cv::Mat mat = cv::Mat(values).clone(); // 将vector变成单列的mat，这里需要clone(),因为这里的赋值操作是浅拷贝
    cv::Mat dest = mat.reshape(1, 1);

    cv::Mat homogeneousPoint = (cv::Mat_<double>(3, 1) << point.x, point.y, 1.0);
    cv::Mat transformed = M * homogeneousPoint;
    return cv::Point2d(transformed.at<double>(0, 0), transformed.at<double>(0, 1));
}

cv::Point2f TransPoint_inv(const cv::Mat& M, const cv::Point2f& point)
{
    std::vector<double> values = { point.x, point.y };
    cv::Mat mat = cv::Mat(values).clone(); // 将vector变成单列的mat，这里需要clone(),因为这里的赋值操作是浅拷贝
    cv::Mat dest = mat.reshape(1, 1);

    cv::Mat homogeneousPoint = (cv::Mat_<double>(3, 1) << point.x, point.y, 1.0);
    cv::Mat transformed = M.inv() * homogeneousPoint;
    return cv::Point2f(transformed.at<double>(0, 0), transformed.at<double>(0, 1));
}

cv::Mat gray_stairs(const cv::Mat& img, double sin, double hin, double mt, double sout, double hout)
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
        lutData[i] = (int)(std::min)((std::max)(Sout + difout * v2 / 255, 0.0), 255.0);
    }
    cv::Mat lut(1, 256, CV_8UC1, lutData);
    cv::Mat dst;
    cv::LUT(img, lut, dst);
    return dst;
}

cv::Mat gamma_trans(const cv::Mat& img, double gamma, int n_c)
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

void StegerLine(const cv::Mat src, std::vector<cv::Point2f>& dst_pt)
{
    cv::Mat img = src.clone();
    // 高斯滤波
    img.convertTo(img, CV_32FC1);
    cv::GaussianBlur(img, img, cv::Size(0, 0), 3, 3);

    cv::Mat dx, dy;
    cv::Mat dxx, dyy, dxy;

    // 一阶偏导数
    cv::Mat mDx, mDy;
    // 二阶偏导数
    cv::Mat mDxx, mDyy, mDxy;

    mDx = (cv::Mat_<float>(1, 2) << 1, -1); // x偏导
    mDy = (cv::Mat_<float>(2, 1) << 1, -1); // y偏导
    mDxx = (cv::Mat_<float>(1, 3) << 1, -2, 1); // 二阶x偏导
    mDyy = (cv::Mat_<float>(3, 1) << 1, -2, 1); // 二阶y偏导
    mDxy = (cv::Mat_<float>(2, 2) << 1, -1, -1, 1); // 二阶xy偏导

    cv::filter2D(img, dx, CV_32FC1, mDx);
    cv::filter2D(img, dy, CV_32FC1, mDy);
    cv::filter2D(img, dxx, CV_32FC1, mDxx);
    cv::filter2D(img, dyy, CV_32FC1, mDyy);
    cv::filter2D(img, dxy, CV_32FC1, mDxy);

    // hessian矩阵
    int cols = src.cols;
    int rows = src.rows;
    std::vector<cv::Point2f> pts;

    for (int col = 0; col < cols; ++col) {
        for (int row = rows - 1; row != -1; --row) {
            if (src.at<uchar>(row, col) < 210)
                continue;

            cv::Mat hessian(2, 2, CV_32FC1);
            hessian.at<float>(0, 0) = dxx.at<float>(row, col);
            hessian.at<float>(0, 1) = dxy.at<float>(row, col);
            hessian.at<float>(1, 0) = dxy.at<float>(row, col);
            hessian.at<float>(1, 1) = dyy.at<float>(row, col);
            cv::Mat eValue;
            cv::Mat eVectors;
            cv::eigen(hessian, eValue, eVectors);
            double nx, ny;
            double fmaxD = 0;

            if (std::fabs(eValue.at<float>(0, 0)) >= std::fabs(eValue.at<float>(1, 0))) // 求特征值最大时对应的特征向量
            {
                nx = eVectors.at<float>(0, 0);
                ny = eVectors.at<float>(0, 1);
                fmaxD = eValue.at<float>(0, 0);
            } else {
                nx = eVectors.at<float>(1, 0);
                ny = eVectors.at<float>(1, 1);
                fmaxD = eValue.at<float>(1, 0);
            }

            float t = -(nx * dx.at<float>(row, col) + ny * dy.at<float>(row, col))
                / (nx * nx * dxx.at<float>(row, col) + 2 * nx * ny * dxy.at<float>(row, col) + ny * ny * dyy.at<float>(row, col));
            float tnx = t * nx;
            float tny = t * ny;

            if (std::fabs(tnx) <= 0.5 && std::fabs(tny) <= 0.5) {
                float x = col + /*.5*/ tnx;
                float y = row + /*.5*/ tny;
                pts.push_back({ x, y });
                break;
            }
        }
    }

    cv::Mat display;
    if (src.channels() == 1) {
        cv::cvtColor(src, display, cv::COLOR_GRAY2BGR);
    } else {
        display = src.clone();
    }
    for (int k = 0; k < pts.size(); k++) {
        cv::Point rpt;
        rpt.x = pts[k].x;
        rpt.y = pts[k].y;
        cv::circle(display, rpt, 1, cv::Scalar(0, 0, 255));
        dst_pt.emplace_back(rpt);
    }
    /*cv::imshow("result", display);
    cv::waitKey(0);*/
}

void get_center(cv::Mat th_img, cv::Point2f& center)
{
    double sumX = 0, sumY = 0;
    double total_gray = 0;
    for (int i = 0; i < th_img.rows; i++) {
        for (int j = 0; j < th_img.cols; j++) {
            double gray_value = static_cast<double>(th_img.at<uchar>(i, j));
            sumX = sumX + j * gray_value;
            sumY = sumY + i * gray_value;
            total_gray = total_gray + gray_value;
        }
    }
    double center_x = sumX / total_gray;
    double center_y = sumY / total_gray;
    center.x = center_x;
    center.y = center_y;
}

void get_histogram(const cv::Mat& src, int* dst)
{
    cv::Mat hist;
    int channels[1] = { 0 };
    int histSize[1] = { 256 };
    float hranges[2] = { 0, 256.0 };
    const float* ranges[1] = { hranges };
    cv::calcHist(&src, 1, channels, cv::Mat(), hist, 1, histSize, ranges);
    for (int i = 0; i < 256; i++) {
        float binVal = hist.at<float>(i);
        dst[i] = int(binVal);
    }
}

int exec_threshold(cv::Mat& src, THRESHOLD_TYPE type, int doIblack, int doIwhite, bool reset)
{
    int threshold = -1;
    if (src.empty() || src.channels() != 1)
        return threshold;
    const int gray_scale = 256;
    int data[gray_scale] = { 0 };
    get_histogram(src, data);

    int minbin = -1, maxbin = -1;
    int range_max = gray_scale;
    int rang_min = 0;

    if (std::abs(doIblack + 1) > 1)
        rang_min = doIblack;

    if (std::abs(doIwhite + 1) > 1)
        range_max = doIwhite;

    for (int i = 0; i < range_max; i++) {
        if (data[i] > 0)
            maxbin = i;
    }
    for (int i = gray_scale - 1; i >= rang_min; i--) {
        if (data[i] > 0)
            minbin = i;
    }
    int scale_range = maxbin - minbin + 1;
    if (scale_range < 2)
        return 0;

    std::vector<int> data2(scale_range, { 0 });
    for (int i = minbin; i <= maxbin; i++) {
        data2[i - minbin] = data[i];
    }

    if (type == THRESHOLD_TYPE::OTSU) {
        threshold = otsu(data2);
    } else if (type == THRESHOLD_TYPE::LI) {
        threshold = li(data2);
    } else if (type == THRESHOLD_TYPE::MINIMUM) {
        threshold = minimum(data2);
    } else if (type == THRESHOLD_TYPE::HUANG2) {
        threshold = huang2(data2);
    } else if (type == THRESHOLD_TYPE::YEN) {
        threshold = yen(data2);
    } else if (type == THRESHOLD_TYPE::TRIANGLE) {
        threshold = triangle(data2);
    }
    else if (type ==THRESHOLD_TYPE::MOMENTS) {
        threshold = moments(data2);
    }
    else if (type == THRESHOLD_TYPE::SAUVOLA) {
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

    std::vector<double> cnh(data.size(), { 0.0 }); ///< 累积归一化直方图
    std::vector<double> mean(data.size(), { 0.0 }); ///< 平均灰度
    std::vector<double> histo(data.size(), { 0.0 }); ///< 归一化直方图
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
        // new_thresh = simple_round ( ( mean_back - mean_obj ) / ( Math.log ( mean_back ) - Math.log ( mean_obj ) ) );
        // simple_round ( double x ) {
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
int moments(std::vector<int>data) {
    double total = 0;
    double m0 = 1.0, m1 = 0.0, m2 = 0.0, m3 = 0.0, sum = 0.0, p0 = 0.0;
    double cd, c0, c1, z0, z1;	///< 辅助变量
    int threshold = -1;


    std::vector<double> histo(data.size(), { 0.0 });

    for (int i = 0; i < data.size(); i++)
        total += data[i];

    for (int i = 0; i < data.size(); i++)
        histo[i] = (double)(data[i] / total); ///<归一化直方图

    /* 计算一阶、二阶和三阶矩 */
    for (int i = 0; i < data.size(); i++) {
        m1 += i * histo[i];
        m2 += i * i * histo[i];
        m3 += i * i * i * histo[i];
    }
    /*
   灰度图像的前4个矩应与目标二值图像的前4个矩相匹配。这导致了4个等式，其解在参考文献1的附录中给出
    */
    cd = m0 * m2 - m1 * m1;
    c0 = (-m2 * m2 + m1 * m3) / cd;
    c1 = (m0 * -m3 + m2 * m1) / cd;
    z0 = 0.5 * (-c1 - std::sqrt(c1 * c1 - 4.0 * c0));
    z1 = 0.5 * (-c1 + std::sqrt(c1 * c1 - 4.0 * c0));
    p0 = (z1 - m1) / (z1 - z0);  ///< 目标二值图像中对象像素的分数 

    // 阈值是最接近归一化直方图p0分片的灰度级
    sum = 0;
    for (int i = 0; i < data.size(); i++) {
        sum += histo[i];
        if (sum > p0) {
            threshold = i;
            break;
        }
    }
    return threshold;

}
int huang2(std::vector<int> data)
{
    int first, last;
    for (first = 0; first < data.size() && data[first] == 0; first++)
        ; // do nothing
    for (last = data.size() - 1; last > first && data[last] == 0; last--)
        ; // do nothing
    if (first == last)
        return 0;

    // 计算累计密度与 累计密度的权重
    std::vector<double> S(last + 1, { 0.0 });
    std::vector<double> W(last + 1, { 0.0 });
    S[0] = data[0];
    for (int i = std::max(1, first); i <= last; i++) {
        S[i] = S[i - 1] + data[i];
        W[i] = W[i - 1] + i * data[i];
    }

    // precalculate the summands of the entropy given the absolute difference x - mu (integral)
    // 给定绝对差x-mu（积分），预先计算熵的总和
    double C = last - first;
    std::vector<double> Smu(last + 1 - first, { 0.0 });
    for (int i = 1; i < Smu.size(); i++) {
        double mu = 1 / (1 + std::abs(i) / C);
        Smu[i] = -mu * std::log(mu) - (1 - mu) * std::log(1 - mu);
    }

    // calculate the threshold
    int bestThreshold = 0;
    double bestEntropy = DBL_MAX;
    for (int threshold = first; threshold <= last; threshold++) {
        double entropy = 0;
        int mu = (int)std::round(W[threshold] / S[threshold]);
        for (int i = first; i <= threshold; i++)
            entropy += Smu[std::abs(i - mu)] * data[i];
        mu = (int)std::round((W[last] - W[threshold]) / (S[last] - S[threshold]));
        for (int i = threshold + 1; i <= last; i++)
            entropy += Smu[std::abs(i - mu)] * data[i];

        if (bestEntropy > entropy) {
            bestEntropy = entropy;
            bestThreshold = threshold;
        }
    }
    return bestThreshold;
}

int triangle(std::vector<int> data)
{
    int min = 0, dmax = 0, max = 0, min2 = 0;
    for (int i = 0; i < data.size(); i++) {
        if (data[i] > 0) {
            min = i;
            break;
        }
    }
    if (min > 0)
        min--; // line to the (p==0) point, not to data[min]

    // The Triangle algorithm cannot tell whether the data is skewed to one side or another.
    // This causes a problem as there are 2 possible thresholds between the max and the 2 extremes
    // of the histogram.
    // Here I propose to find out to which side of the max point the data is furthest, and use that as
    //  the other extreme. Note that this is not done in the original method. GL
    for (int i = data.size() - 1; i > 0; i--) {
        if (data[i] > 0) {
            min2 = i;
            break;
        }
    }
    if (min2 < data.size() - 1)
        min2++; // line to the (p==0) point, not to data[min]

    for (int i = 0; i < data.size(); i++) {
        if (data[i] > dmax) {
            max = i;
            dmax = data[i];
        }
    }
    // find which is the furthest side
    // IJ.log(""+min+" "+max+" "+min2);
    bool inverted = false;
    if ((max - min) < (min2 - max)) {
        // reverse the histogram
        // IJ.log("Reversing histogram.");
        inverted = true;
        int left = 0; // index of leftmost element
        int right = data.size() - 1; // index of rightmost element
        while (left < right) {
            // exchange the left and right elements
            int temp = data[left];
            data[left] = data[right];
            data[right] = temp;
            // move the bounds toward the center
            left++;
            right--;
        }
        min = data.size() - 1 - min2;
        max = data.size() - 1 - max;
    }

    if (min == max) {
        // IJ.log("Triangle:  min == max.");
        return min;
    }

    // describe line by nx * x + ny * y - d = 0
    double nx, ny, d;
    // nx is just the max frequency as the other point has freq=0
    nx = data[max]; //-min; // data[min]; //  lowest value bmin = (p=0)% in the image
    ny = min - max;
    d = std::sqrt(nx * nx + ny * ny);
    nx /= d;
    ny /= d;
    d = nx * min + ny * data[min];

    // find split point
    int split = min;
    double splitDistance = 0;
    for (int i = min + 1; i <= max; i++) {
        double newDistance = nx * i + ny * data[i] - d;
        if (newDistance > splitDistance) {
            split = i;
            splitDistance = newDistance;
        }
    }
    split--;

    if (inverted) {
        // The histogram might be used for something else, so let's reverse it back
        int left = 0;
        int right = data.size() - 1;
        while (left < right) {
            int temp = data[left];
            data[left] = data[right];
            data[right] = temp;
            left++;
            right--;
        }
        return (data.size() - 1 - split);
    } else
        return split;
}
int yen(std::vector<int> data)
{
    int threshold;
    int ih, it;
    double crit;
    double max_crit;

    std::vector<double> norm_histo(data.size(), { 0.0 }); /* normalized histogram */
    std::vector<double> P1(data.size(), { 0.0 }); /* cumulative normalized histogram */
    std::vector<double> P1_sq(data.size(), { 0.0 });
    std::vector<double> P2_sq(data.size(), { 0.0 });

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
        crit = -1.0 * ((P1_sq[it] * P2_sq[it]) > 0.0 ? std::log(P1_sq[it] * P2_sq[it]) : 0.0) + 2 * ((P1[it] * (1.0 - P1[it])) > 0.0 ? std::log(P1[it] * (1.0 - P1[it])) : 0.0);
        if (crit > max_crit) {
            max_crit = crit;
            threshold = it;
        }
    }
    return threshold;
}

int sauvola(cv::Mat& src, const double k, const int wnd_size)
{
    CV_Assert(src.type() == CV_8UC1);
    CV_Assert(wnd_size % 2 == 1);
    CV_Assert((wnd_size <= src.cols) && (wnd_size <= src.rows));
    cv::Mat dst = cv::Mat::zeros(src.rows, src.cols, CV_8UC1);

    unsigned long* integralImg = new unsigned long[src.rows * src.cols];
    unsigned long* integralImgSqrt = new unsigned long[src.rows * src.cols];
    std::memset(integralImg, 0, src.rows * src.cols * sizeof(unsigned long));
    std::memset(integralImgSqrt, 0, src.rows * src.cols * sizeof(unsigned long));

    // 计算直方图和图像值平方的和,积分图函数(cv::integral),未测试
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
                integralImgSqrt[index] = integralImgSqrt[(y - 1) * src.cols + x] + sqrtSum;
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
    for (int x = 0; x < src.cols; ++x) {
        for (int y = 0; y < src.rows; ++y) {
            int xMin = std::max(0, x - wnd_size / 2);
            int yMin = std::max(0, y - wnd_size / 2);
            int xMax = std::min(src.cols - 1, x + wnd_size / 2);
            int yMax = std::min(src.rows - 1, y + wnd_size / 2);
            double area = (xMax - xMin + 1) * (yMax - yMin + 1);
            if (area <= 0) {
                // blog提供源码是biImage[i * IMAGE_WIDTH + j] = 255;但是i表示的是列, j是行
                dst.at<uchar>(y, x) = 255;
                continue;
            }
            if (xMin == 0 && yMin == 0) {
                diff = integralImg[yMax * src.cols + xMax];
                sqDiff = integralImgSqrt[yMax * src.cols + xMax];
            } else if (xMin > 0 && yMin == 0) {
                diff = integralImg[yMax * src.cols + xMax] - integralImg[yMax * src.cols + xMin - 1];
                sqDiff = integralImgSqrt[yMax * src.cols + xMax] - integralImgSqrt[yMax * src.cols + xMin - 1];
            } else if (xMin == 0 && yMin > 0) {
                diff = integralImg[yMax * src.cols + xMax] - integralImg[(yMin - 1) * src.cols + xMax];
                sqDiff = integralImgSqrt[yMax * src.cols + xMax] - integralImgSqrt[(yMin - 1) * src.cols + xMax];
                ;
            } else {
                diagSum = integralImg[yMax * src.cols + xMax] + integralImg[(yMin - 1) * src.cols + xMin - 1];
                iDiagSum = integralImg[(yMin - 1) * src.cols + xMax] + integralImg[yMax * src.cols + xMin - 1];
                diff = diagSum - iDiagSum;
                sqDiagSum = integralImgSqrt[yMax * src.cols + xMax] + integralImgSqrt[(yMin - 1) * src.cols + xMin - 1];
                sqIDiagSum = integralImgSqrt[(yMin - 1) * src.cols + xMax] + integralImgSqrt[yMax * src.cols + xMin - 1];
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
    delete[] integralImg;
    delete[] integralImgSqrt;
    src = dst.clone();
}

int phansalkar(cv::Mat& src, const double k, const int wnd_size, double r, double p, double q)
{
    CV_Assert(src.type() == CV_8UC1);
    CV_Assert(wnd_size % 2 == 1);
    CV_Assert((wnd_size <= src.cols) && (wnd_size <= src.rows));
    cv::Mat dst = cv::Mat::zeros(src.rows, src.cols, CV_8UC1);

    unsigned long* integralImg = new unsigned long[src.rows * src.cols];
    unsigned long* integralImgSqrt = new unsigned long[src.rows * src.cols];
    std::memset(integralImg, 0, src.rows * src.cols * sizeof(unsigned long));
    std::memset(integralImgSqrt, 0, src.rows * src.cols * sizeof(unsigned long));

    // 计算直方图和图像值平方的和,积分图函数(cv::integral),未测试
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
                integralImgSqrt[index] = integralImgSqrt[(y - 1) * src.cols + x] + sqrtSum;
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
    for (int x = 0; x < src.cols; ++x) {
        for (int y = 0; y < src.rows; ++y) {
            int xMin = std::max(0, x - wnd_size / 2);
            int yMin = std::max(0, y - wnd_size / 2);
            int xMax = std::min(src.cols - 1, x + wnd_size / 2);
            int yMax = std::min(src.rows - 1, y + wnd_size / 2);
            double area = (xMax - xMin + 1) * (yMax - yMin + 1);
            if (area <= 0) {
                // blog提供源码是biImage[i * IMAGE_WIDTH + j] = 255;但是i表示的是列, j是行
                dst.at<uchar>(y, x) = 255;
                continue;
            }
            if (xMin == 0 && yMin == 0) {
                diff = integralImg[yMax * src.cols + xMax];
                sqDiff = integralImgSqrt[yMax * src.cols + xMax];
            } else if (xMin > 0 && yMin == 0) {
                diff = integralImg[yMax * src.cols + xMax] - integralImg[yMax * src.cols + xMin - 1];
                sqDiff = integralImgSqrt[yMax * src.cols + xMax] - integralImgSqrt[yMax * src.cols + xMin - 1];
            } else if (xMin == 0 && yMin > 0) {
                diff = integralImg[yMax * src.cols + xMax] - integralImg[(yMin - 1) * src.cols + xMax];
                sqDiff = integralImgSqrt[yMax * src.cols + xMax] - integralImgSqrt[(yMin - 1) * src.cols + xMax];
                ;
            } else {
                diagSum = integralImg[yMax * src.cols + xMax] + integralImg[(yMin - 1) * src.cols + xMin - 1];
                iDiagSum = integralImg[(yMin - 1) * src.cols + xMax] + integralImg[yMax * src.cols + xMin - 1];
                diff = diagSum - iDiagSum;
                sqDiagSum = integralImgSqrt[yMax * src.cols + xMax] + integralImgSqrt[(yMin - 1) * src.cols + xMin - 1];
                sqIDiagSum = integralImgSqrt[(yMin - 1) * src.cols + xMax] + integralImgSqrt[yMax * src.cols + xMin - 1];
                sqDiff = sqDiagSum - sqIDiagSum;
            }
            double mean = diff / area;
            double stdValue = sqrt((sqDiff - diff * diff / area) / (area - 1));

            // double threshold = mean * (1 + k * ((stdValue / 128) - 1));

            double threshold = mean * (1 + p * exp(-q * mean) + k * ((stdValue / r) - 1));
            if (src.at<uchar>(y, x) < threshold) {
                dst.at<uchar>(y, x) = 0;
            } else {
                dst.at<uchar>(y, x) = 255;
            }
        }
    }
    delete[] integralImg;
    delete[] integralImgSqrt;
    src = dst.clone();
}

bool bimodalTest(std::vector<double> y)
{
    int len = static_cast<double>(y.size());
    bool b = false;
    int modes = 0;
    for (int k = 1; k < len - 1; k++) {
        if (y[k - 1] < y[k] && y[k + 1] < y[k]) {
            modes++;
            if (modes > 2)
                return false;
        }
    }
    if (modes == 2)
        b = true;
    return b;
}

int minimum(std::vector<int> data)
{
    int iter = 0;
    int threshold = -1;
    int max = -1;
    std::vector<double> iHisto(data.size(), { 0.0 });

    for (int i = 0; i < data.size(); i++) {
        iHisto[i] = (double)data[i];
        if (data[i] > 0)
            max = i;
    }
    std::vector<double> tHisto(iHisto.size(), { 0.0 }); // Instead of double[] tHisto = iHisto ;
    while (!bimodalTest(iHisto)) {
        // 使用3点运行平均值过滤器平滑
        for (int i = 1; i < data.size() - 1; i++)
            tHisto[i] = (iHisto[i - 1] + iHisto[i] + iHisto[i + 1]) / 3;
        tHisto[0] = (iHisto[0] + iHisto[1]) / 3; // 0 outside
        tHisto[data.size() - 1] = (iHisto[data.size() - 2] + iHisto[data.size() - 1]) / 3; // 0 outside
        // System.arraycopy(tHisto, 0, iHisto, 0, iHisto.size()); //Instead of iHisto = tHisto ;
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

std::vector<std::vector<cv::Point>> get_contours(const cv::Mat& src)
{
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> filter_hierarchy;
    cv::findContours(src, contours, filter_hierarchy, cv::RETR_LIST, cv::CHAIN_APPROX_NONE);
    return contours;
}

void get_point_1(cv::Point2f p1, cv::Point2f p2, cv::Point2f& t1, cv::Point2f& t2)
{

    // 直线斜率
    double k = (p2.y - p1.y) / (p2.x - p1.x);
    k = -1 / k;
    // 截距
    double b = p1.y - k * p1.x;

    // 圆心
    cv::Point2f center = p1;
    double r = 10;

    double A = 1 + k * k;
    double B = -2 * center.x + 2 * k * (b - center.y);
    double C = center.x * center.x + (b - center.y) * (b - center.y) - r * r;
    double delta = B * B - 4 * A * C;

    t1.x = (-B - std::sqrt(delta)) / (2 * A);
    t1.y = k * t1.x + b;

    t2.x = (-B + std::sqrt(delta)) / (2 * A);
    t2.y = k * t2.x + b;
}

void get_point_2(cv::Point2f p1, cv::Point2f p2, cv::Point2f& t1, cv::Point2f& t2)
{

    // 直线斜率
    double k = (p2.y - p1.y) / (p2.x - p1.x);
    k = -1 / k;
    // 截距
    double b = p2.y - k * p2.x;

    // 圆心
    cv::Point2f center = p2;
    double r = 10;

    double A = 1 + k * k;
    double B = -2 * center.x + 2 * k * (b - center.y);
    double C = center.x * center.x + (b - center.y) * (b - center.y) - r * r;
    double delta = B * B - 4 * A * C;

    t1.x = (-B - std::sqrt(delta)) / (2 * A);
    t1.y = k * t1.x + b;

    t2.x = (-B + std::sqrt(delta)) / (2 * A);
    t2.y = k * t2.x + b;
}

void draw_results(cv::Mat& image, const nlohmann::json& result_info)
{
    if (image.channels() < 3) {
        cv::cvtColor(image, image, cv::COLOR_GRAY2RGB);
    }

    if (result_info.is_array() && !result_info.empty()) {
        auto size = result_info.size();
        for (int i = 0; i < size; i++) {
            auto item = result_info.at(i);
            std::string shape_type = item["shapeType"];
            if (shape_type == "rectangle") {
                std::vector<cv::Point2f> pt_vec;
                pt_vec.push_back(cv::Point2f(item["points"][0][0], item["points"][0][1]));
                pt_vec.push_back(cv::Point2f(item["points"][1][0], item["points"][1][1]));
                pt_vec.push_back(cv::Point2f(item["points"][2][0], item["points"][2][1]));
                pt_vec.push_back(cv::Point2f(item["points"][3][0], item["points"][3][1]));
                pt_vec = order_pts(pt_vec);

                for (size_t j = 0; j < 4; j++)
                    cv::line(image, pt_vec[j], pt_vec[(j + 1) % 4], cv::Scalar(255, 0, 0), 1, cv::LINE_AA);

                /*int index = item["result"]["index"];
                std::string str = std::to_string(index);
                if (index > -2) {
                        cv::putText(image, str, cv::Point((pt_vec[0].x + pt_vec[2].x) / 2, (pt_vec[0].y + pt_vec[2].y) / 2), cv::FONT_HERSHEY_PLAIN, 10, cv::Scalar(0, 0, 255),2);

                }*/
            }

            if (shape_type == "test_rect") {
                std::vector<cv::Point2f> pt_vec;
                pt_vec.push_back(cv::Point2f(item["points"][0][0], item["points"][0][1]));
                pt_vec.push_back(cv::Point2f(item["points"][1][0], item["points"][1][1]));
                pt_vec.push_back(cv::Point2f(item["points"][2][0], item["points"][2][1]));
                pt_vec.push_back(cv::Point2f(item["points"][3][0], item["points"][3][1]));
                pt_vec = order_pts(pt_vec);

                for (size_t j = 0; j < 4; j++)
                    cv::line(image, pt_vec[j], pt_vec[(j + 1) % 4], cv::Scalar(255, 0, 0), 1, cv::LINE_AA);
            }

            if (shape_type == "line") {
                cv::Point2f line_start(item["points"][0][0], item["points"][0][1]);
                cv::Point2f line_end(item["points"][1][0], item["points"][1][1]);
                cv::line(image, line_start, line_end, cv::Scalar(0, 255, 0), 1, cv::LINE_AA);
            }
            // if (shape_type=="polygon") {
            ////pin针暂时不绘制
            //}
            if (shape_type == "polygon") {
                std::vector<cv::Point2f> pt_vec;
                pt_vec.push_back(cv::Point2f(item["points"][0][0], item["points"][0][1]));
                pt_vec.push_back(cv::Point2f(item["points"][1][0], item["points"][1][1]));
                pt_vec.push_back(cv::Point2f(item["points"][2][0], item["points"][2][1]));
                pt_vec.push_back(cv::Point2f(item["points"][3][0], item["points"][3][1]));
                pt_vec = order_pts(pt_vec);

                for (size_t j = 0; j < 4; j++)
                    cv::line(image, pt_vec[j], pt_vec[(j + 1) % 4], cv::Scalar(255, 0, 0), 1, cv::LINE_AA);
            }

            if (shape_type == "point") {
                cv::Point2f ret_center(item["points"][0][0], item["points"][0][1]);
                cv::circle(image, ret_center, 3, cv::Scalar(0, 0, 255), 2);
                std::string label_str = item["label"];

                if (label_str == "Z_Male_defect") {

                    if (item["result"]["is_ok"]) {
                        cv::circle(image, ret_center, 1, cv::Scalar(0, 255, 0), 1);
                        double value_x = item["result"]["x_off"];
                        std::string str_x = std::to_string(value_x);
                        double value_y = item["result"]["y_off"];
                        std::string str_y = std::to_string(value_y);

                        std::string str = str_x + "," + str_y;
                        cv::putText(image, str, cv::Point2f(item["points"][0][0] - 70, item["points"][0][1] - 10), cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(0, 255, 0));

                    } else {
                        cv::circle(image, ret_center, 1, cv::Scalar(0, 0, 255), 1);

                        double value_x = item["result"]["x_off"];
                        std::string str_x = std::to_string(value_x);
                        double value_y = item["result"]["y_off"];
                        std::string str_y = std::to_string(value_y);

                        std::string str = str_x + "," + str_y;
                        cv::putText(image, str, cv::Point2f(item["points"][0][0] - 70, item["points"][0][1] - 10), cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(0, 0, 255));
                    }
                }
            }

            if (shape_type == "Abasis") {
                cv::Point2f ret_center(item["points"][0][0], item["points"][0][1]);
                cv::circle(image, ret_center, 1, cv::Scalar(0, 0, 255), 1);

                cv::Point2f pt_11, pt_12, pt_21, pt_22, pt_31, pt_32, pt_41, pt_42, pt_51, pt_52, pt_61, pt_62, pt_71, pt_72, pt_81, pt_82;
                pt_11 = cv::Point(item["result"]["points"][0][0], item["result"]["points"][0][1]);
                pt_12 = cv::Point(item["result"]["points"][1][0], item["result"]["points"][1][1]);
                pt_21 = cv::Point(item["result"]["points"][2][0], item["result"]["points"][2][1]);
                pt_22 = cv::Point(item["result"]["points"][3][0], item["result"]["points"][3][1]);
                pt_31 = cv::Point(item["result"]["points"][4][0], item["result"]["points"][4][1]);
                pt_32 = cv::Point(item["result"]["points"][5][0], item["result"]["points"][5][1]);
                pt_41 = cv::Point(item["result"]["points"][6][0], item["result"]["points"][6][1]);
                pt_42 = cv::Point(item["result"]["points"][7][0], item["result"]["points"][7][1]);
                pt_51 = cv::Point(item["result"]["points"][8][0], item["result"]["points"][8][1]);
                pt_52 = cv::Point(item["result"]["points"][9][0], item["result"]["points"][9][1]);
                pt_61 = cv::Point(item["result"]["points"][10][0], item["result"]["points"][10][1]);
                pt_62 = cv::Point(item["result"]["points"][11][0], item["result"]["points"][11][1]);
                pt_71 = cv::Point(item["result"]["points"][12][0], item["result"]["points"][12][1]);
                pt_72 = cv::Point(item["result"]["points"][13][0], item["result"]["points"][13][1]);
                pt_81 = cv::Point(item["result"]["points"][14][0], item["result"]["points"][14][1]);
                pt_82 = cv::Point(item["result"]["points"][15][0], item["result"]["points"][15][1]);

                cv::arrowedLine(image, pt_11, pt_12, cv::Scalar(0, 0, 255), 1, cv::LINE_AA);
                cv::arrowedLine(image, pt_21, pt_22, cv::Scalar(0, 0, 255), 1, cv::LINE_AA);
                cv::arrowedLine(image, pt_31, pt_32, cv::Scalar(0, 0, 255), 1, cv::LINE_AA);
                cv::arrowedLine(image, pt_41, pt_42, cv::Scalar(0, 0, 255), 1, cv::LINE_AA);
                cv::arrowedLine(image, pt_51, pt_52, cv::Scalar(0, 0, 255), 1, cv::LINE_AA);
                cv::arrowedLine(image, pt_61, pt_62, cv::Scalar(0, 0, 255), 1, cv::LINE_AA);
                cv::arrowedLine(image, pt_71, pt_72, cv::Scalar(0, 0, 255), 1, cv::LINE_AA);
                cv::arrowedLine(image, pt_81, pt_82, cv::Scalar(0, 0, 255), 1, cv::LINE_AA);

                cv::Point2f tr_11, tr12, tr_13, tr_14;

                get_point_1(pt_11, pt_12, tr_11, tr12);
                get_point_2(pt_11, pt_12, tr_13, tr_14);
                cv::line(image, tr_11, tr12, cv::Scalar(255, 0, 0), 1, cv::LINE_AA);
                cv::line(image, tr_13, tr_14, cv::Scalar(255, 0, 0), 1, cv::LINE_AA);

                get_point_1(pt_21, pt_22, tr_11, tr12);
                get_point_2(pt_21, pt_22, tr_13, tr_14);
                cv::line(image, tr_11, tr12, cv::Scalar(255, 0, 0), 1, cv::LINE_AA);
                cv::line(image, tr_13, tr_14, cv::Scalar(255, 0, 0), 1, cv::LINE_AA);

                get_point_1(pt_31, pt_32, tr_11, tr12);
                get_point_2(pt_31, pt_32, tr_13, tr_14);
                cv::line(image, tr_11, tr12, cv::Scalar(255, 0, 0), 1, cv::LINE_AA);
                cv::line(image, tr_13, tr_14, cv::Scalar(255, 0, 0), 1, cv::LINE_AA);

                get_point_1(pt_41, pt_42, tr_11, tr12);
                get_point_2(pt_41, pt_42, tr_13, tr_14);
                cv::line(image, tr_11, tr12, cv::Scalar(255, 0, 0), 1, cv::LINE_AA);
                cv::line(image, tr_13, tr_14, cv::Scalar(255, 0, 0), 1, cv::LINE_AA);

                get_point_1(pt_51, pt_52, tr_11, tr12);
                get_point_2(pt_51, pt_52, tr_13, tr_14);
                cv::line(image, tr_11, tr12, cv::Scalar(255, 0, 0), 1, cv::LINE_AA);
                cv::line(image, tr_13, tr_14, cv::Scalar(255, 0, 0), 1, cv::LINE_AA);

                get_point_1(pt_61, pt_62, tr_11, tr12);
                get_point_2(pt_61, pt_62, tr_13, tr_14);
                cv::line(image, tr_11, tr12, cv::Scalar(255, 0, 0), 1, cv::LINE_AA);
                cv::line(image, tr_13, tr_14, cv::Scalar(255, 0, 0), 1, cv::LINE_AA);

                double value = item["result"]["dist"][0];
                std::string str = std::to_string(value);
                if (value > 0) {
                    cv::putText(image, str, cv::Point((pt_11.x + pt_12.x) / 2, (pt_11.y + pt_12.y) / 2), cv::FONT_HERSHEY_PLAIN, 2, cv::Scalar(0, 0, 255));
                }

                value = item["result"]["dist"][1];
                if (value > 0) {
                    str = std::to_string(value);
                    cv::putText(image, str, cv::Point((pt_21.x + pt_22.x) / 2, (pt_21.y + pt_22.y) / 2), cv::FONT_HERSHEY_PLAIN, 2, cv::Scalar(0, 0, 255));
                }

                value = item["result"]["dist"][2];
                if (value > 0) {
                    str = std::to_string(value);
                    cv::putText(image, str, cv::Point((pt_31.x + pt_32.x) / 2, (pt_31.y + pt_32.y) / 2), cv::FONT_HERSHEY_PLAIN, 2, cv::Scalar(0, 0, 255));
                }

                value = item["result"]["dist"][3];
                if (value > 0) {
                    str = std::to_string(value);
                    cv::putText(image, str, cv::Point((pt_41.x + pt_42.x) / 2, (pt_41.y + pt_42.y) / 2), cv::FONT_HERSHEY_PLAIN, 2, cv::Scalar(0, 0, 255));
                }

                value = item["result"]["dist"][4];
                if (value > 0) {
                    str = std::to_string(value);
                    cv::putText(image, str, cv::Point((pt_51.x + pt_52.x) / 2, (pt_51.y + pt_52.y) / 2), cv::FONT_HERSHEY_PLAIN, 2, cv::Scalar(0, 0, 255));
                }

                value = item["result"]["dist"][5];
                if (value > 0) {
                    str = std::to_string(value);
                    cv::putText(image, str, cv::Point((pt_61.x + pt_62.x) / 2, (pt_61.y + pt_62.y) / 2), cv::FONT_HERSHEY_PLAIN, 2, cv::Scalar(0, 0, 255));
                }

                value = item["result"]["dist"][6];
                if (value > 0) {
                    str = std::to_string(value);
                    cv::putText(image, str, cv::Point((pt_71.x + pt_72.x) / 2, (pt_71.y + pt_72.y) / 2), cv::FONT_HERSHEY_PLAIN, 2, cv::Scalar(0, 0, 255));
                }

                value = item["result"]["dist"][7];
                if (value > 0) {
                    str = std::to_string(value);
                    cv::putText(image, str, cv::Point((pt_81.x + pt_82.x) / 2, (pt_81.y + pt_82.y) / 2), cv::FONT_HERSHEY_PLAIN, 2, cv::Scalar(0, 0, 255));
                }

                int index = item["result"]["index"];
                str = std::to_string(index);
                if (index > -2) {
                    cv::putText(image, str, cv::Point(ret_center.x, ret_center.y), cv::FONT_HERSHEY_PLAIN, 10, cv::Scalar(0, 0, 255), 2);
                }
            }
            if (shape_type == "Z_Male_defect") {
                cv::Point2f ret_center(item["points"][0][0], item["points"][0][1]);
                if (item["result"]["is_ok"]) {
                    cv::circle(image, ret_center, 1, cv::Scalar(0, 255, 0), 1);
                    double value_x = item["result"]["offset_x"];
                    std::string str_x = std::to_string(value_x);
                    double value_y = item["result"]["offset_y"];
                    std::string str_y = std::to_string(value_y);

                    std::string str = str_x + "," + str_y;
                    cv::putText(image, str, cv::Point2f(item["points"][0][0] - 70, item["points"][0][1] - 10), cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(0, 255, 0));

                } else {
                    cv::circle(image, ret_center, 1, cv::Scalar(0, 0, 255), 1);

                    double value_x = item["result"]["offset_x"];
                    std::string str_x = std::to_string(value_x);
                    double value_y = item["result"]["offset_y"];
                    std::string str_y = std::to_string(value_y);

                    std::string str = str_x + "," + str_y;
                    cv::putText(image, str, cv::Point2f(item["points"][0][0] - 70, item["points"][0][1] - 10), cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(0, 0, 255));
                }
            }
        }
    }
}
}