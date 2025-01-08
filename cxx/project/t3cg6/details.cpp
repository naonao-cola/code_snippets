#include "details.h"

namespace xx
{

void DrawImg(cv::Mat dispaly, nlohmann::json json_info)
{
}

nlohmann::json pt_json(std::vector<std::vector<cv::Point>> mask)
{
    nlohmann::json pt;
    for (int i = 0; i < mask.size(); i++) {
        for (int j = 0; j < mask[i].size(); j++) {
            pt.push_back({mask[i][j].x, mask[i][j].y});
        }
    }
    return pt;
}

nlohmann::json pt_json(std::vector<cv::Point2f> mask)
{
    nlohmann::json pt;
    for (int j = 0; j < mask.size(); j++) {

        pt.push_back({int(mask[j].x), int(mask[j].y)});
    }
    return pt;
}

cv::Mat gray_stairs(const cv::Mat& img, double sin, double hin, double mt, double sout, double hout)
{
    double Sin    = (std::min)((std::max)(sin, 0.0), hin - 2);
    double Hin    = (std::min)(hin, 255.0);
    double Mt     = (std::min)((std::max)(mt, 0.01), 9.99);
    double Sout   = (std::min)((std::max)(sout, 0.0), hout - 2);
    double Hout   = (std::min)(hout, 255.0);
    double difin  = Hin - Sin;
    double difout = Hout - Sout;
    uchar  lutData[256];
    for (int i = 0; i < 256; i++) {
        double v1  = (std::min)((std::max)(255 * (i - Sin) / difin, 0.0), 255.0);
        double v2  = 255 * std::pow(v1 / 255.0, 1.0 / Mt);
        lutData[i] = (int)(std::min)((std::max)(Sout + difout * v2 / 255, 0.0), 255.0);
    }
    cv::Mat lut(1, 256, CV_8UC1, lutData);
    cv::Mat dst;
    cv::LUT(img, lut, dst);
    return dst;
}


cv::Mat gamma_trans(const cv::Mat& img, double gamma, int n_c)
{
    cv::Mat img_gamma(img.size(), CV_32FC1);
#pragma omp parallel for num_threads(32)
    for (int i = 0; i < img.rows; i++)
        for (int j = 0; j < img.cols; j++)
            img_gamma.at<float>(i, j) = n_c * pow(img.at<uchar>(i, j), gamma);
    cv::normalize(img_gamma, img_gamma, 0, 255, cv::NormTypes::NORM_MINMAX);
    cv::convertScaleAbs(img_gamma, img_gamma);
    return img_gamma;
}


cv::Mat segmented_enhancement(const cv::Mat& img, double r1, double r2, double s1, double s2)
{
    double k1 = s1 / r1;
    double k2 = (s2 - s1) / (r2 - r1);
    double k3 = (255 - s2) / (255 - r2);
    uchar  Lutfirst[256];
    for (int i = 0; i < 256; i++) {
        if (i <= r2 && i >= r1) {
            Lutfirst[i] = k2 * (i - r1);
        }
        if (i < r1) {
            Lutfirst[i] = k1 * i;
        }
        if (i > r2) {
            Lutfirst[i] = k3 * (i - r2);
        }
    }
    cv::Mat lut(1, 256, CV_8U, Lutfirst);
    cv::Mat dst;
    cv::LUT(img, lut, dst);
    return dst;
}


// 判断某一行是否有连续大于threshold 的像素的个数是否大于 Continuous
void get_row_range(const cv::Mat& t_src, int& row_beg, int& row_end, int col_beg, int col_end, int threshold, int Continuous)
{
    std::vector<int> row_vec;
    row_vec.reserve(t_src.rows);
    for (int i = 0; i < t_src.rows; i++) {
        int flag      = 0;
        int old_value = 0;
        for (int j = 0; j < t_src.cols; j++) {
            uchar pix_value = *t_src.ptr<uchar>(i, j);
            if (pix_value > threshold && flag == 0) {
                flag++;
                old_value = 1;
            }
            else if (pix_value > threshold && old_value == 1) {
                flag++;
                old_value = 1;
            }
            else {
                old_value = 0;
                flag      = 0;
            }
            if (flag > Continuous) {
                break;
            }
        }
        row_vec.emplace_back(flag);
    }
    row_beg = 0;
    row_end = t_src.rows;


    int count = 0;
    for (int i = 0; i < row_vec.size(); i++) {
        if (row_vec[i] <= 100) {
            count++;
        }
    }

    if (count >= row_vec.size() - 5) {
        row_beg = 0;
        row_end = 5;
        return;
    }
    for (int i = 0; i < row_vec.size(); i++) {
        if (row_vec[i] >= Continuous) {
            row_beg = i;
            break;
        }
    }
    for (int j = row_vec.size() - 1; j > 0; j--) {
        if (row_vec[j] >= Continuous) {
            row_end = j;
            break;
        }
    }
}


void get_edge(const cv::Mat& src, int th_low, int th_high, int blur_w, int blur_h, int& row_beg, int& row_end, int& col_beg, int& col_end, int continuous, int th_value, int type)
{
    cv::Mat dis_mat;
    cv::Mat dst;
    cv::Mat enhance_img;
    cv::Mat th_img;
    cv::Mat edge;

    
    double gama_value = 0.8;
    
    dis_mat = src.clone();

    col_beg = col_beg / 2;
    col_end = col_end / 2;

    if (src.channels() > 1)
        cv::cvtColor(src, dst, cv::COLOR_BGR2GRAY);
    else
        dst = src.clone();
    cv::resize(dst, dst, cv::Size(dis_mat.cols / 2, dis_mat.rows / 2));
    cv::GaussianBlur(dst, dst, cv::Size(blur_w, blur_h), 0);


    if (type == 1) {// 右长边

        enhance_img = gray_stairs(dst, th_low, th_high);
        enhance_img = gamma_trans(enhance_img, gama_value);
        cv::threshold(enhance_img, th_img, th_value, 255, cv::THRESH_TOZERO);
        
        //查找上下边缘
        row_beg = 0;
        row_end = th_img.rows;
        get_row_range(th_img, row_beg, row_end, col_beg, col_end, th_value, continuous);

       cv::Mat ROI = th_img(cv::Rect(col_beg, row_beg, col_end - col_beg, row_end - row_beg));
        // 构造数据
        cv::Mat row_mat, col_mat;
        cv::reduce(ROI, col_mat, 0, cv::REDUCE_AVG);
        std::vector<int> col_vec = col_mat.reshape(1, 1);

        // 寻找左右边界
        std::vector<int> col_diff;
        col_diff.emplace_back(col_vec[0]);
        col_diff.emplace_back(col_vec[1]);
        for (int i = 2; i < col_vec.size() - 2; i++) {
            int data = std::abs(col_vec[i + 1] + col_vec[i + 2] - col_vec[i - 1] - col_vec[i - 2]);
            if (data > 10) {
                data = std::abs(col_vec[i + 1] - col_vec[i]);
            }
            col_diff.emplace_back(data);
        }
        int col_last_data = col_diff[col_diff.size() - 1];
        col_diff[0]       = col_diff[2];
        col_diff[1]       = col_diff[2];
        col_diff.emplace_back(col_last_data);
        col_diff.emplace_back(col_last_data);
        // 初始值
       
        for (int i = 0; i < col_diff.size(); i++) {
            if ((std::abs(col_diff[i]) >= 25 || std::abs(col_diff[i]) + std::abs(col_diff[i + 1]) >= 30)) {
                col_beg = (i + 2 + col_beg) * 2;
                break;
            }
        }
        col_end = col_beg;
        row_beg = row_beg * 2;
        row_end      = row_end * 2;
        int row_diff = std::abs(row_beg - row_end);
        if (row_diff < 10) {
            col_beg = 0;
            col_end = 0;
        }

        if ((col_beg > src.cols - 300 || col_beg < 100) && (row_diff > 20)) {   // 特殊破损，凹字形破损
            cv::Mat edgt_img;
            cv::Sobel(th_img, edgt_img, CV_16S, 0, 1, 3);
            cv::convertScaleAbs(edgt_img, edgt_img);

            std::vector<std::vector<cv::Point>> contours;
            std::vector<cv::Vec4i>              hierarchy;
            cv::findContours(edgt_img, contours, hierarchy, cv::RETR_LIST, cv::CHAIN_APPROX_NONE);
            std::vector<cv::Rect> ret_vec;
            for (size_t i = 0; i < contours.size(); ++i) {
                double area = contourArea(contours[i]);
                if (area < 100)
                    continue;
                cv::Rect        rect   = cv::boundingRect(contours[i]);
                cv::RotatedRect r_rect = cv::minAreaRect(contours[i]);
                if (rect.width < 200)
                    continue;
                ret_vec.emplace_back(rect);
            }
            if (ret_vec.size() > 0) {
                col_beg = ret_vec[0].x * 2;
                col_end = ret_vec[0].x * 2;
            }
            else {
                col_beg = 0;
                col_end = 0;
            }
        }

    }
    // 右短边
    if (type == 2) {
        //Timer t;
        enhance_img = gray_stairs(dst, th_low, th_high);
        //t.out("enhance_img");
        enhance_img = gamma_trans(enhance_img, gama_value);
        //t.out("gamma_trans");
        cv::threshold(enhance_img, th_img, th_value, 255, cv::THRESH_TOZERO);

        // 查找上下边缘
        row_beg = 0;
        row_end = th_img.rows;


        get_row_range(th_img, row_beg, row_end, col_beg, col_end, th_value, continuous);
        //t.out("get_row_range");
        cv::Mat ROI = th_img(cv::Rect(col_beg, row_beg, col_end - col_beg, row_end - row_beg));
        // 构造数据
        cv::Mat row_mat, col_mat;
        cv::reduce(ROI, col_mat, 0, cv::REDUCE_AVG);
        std::vector<int> col_vec = col_mat.reshape(1, 1);

        // 寻找左右边界
        std::vector<int> col_diff;
        col_diff.emplace_back(col_vec[0]);
        col_diff.emplace_back(col_vec[1]);
        for (int i = 2; i < col_vec.size() - 2; i++) {
            int data = std::abs(col_vec[i + 1] + col_vec[i + 2] - col_vec[i - 1] - col_vec[i - 2]);
            if (data > 10) {
                data = std::abs(col_vec[i + 1] - col_vec[i]);
            }
            col_diff.emplace_back(data);
        }
        int col_last_data = col_diff[col_diff.size() - 1];
        col_diff[0]       = col_diff[2];
        col_diff[1]       = col_diff[2];
        col_diff.emplace_back(col_last_data);
        col_diff.emplace_back(col_last_data);
        // 初始值
        
        for (int i = 0; i < col_diff.size(); i++) {
            if ((std::abs(col_diff[i]) >= 20 || std::abs(col_diff[i]) + std::abs(col_diff[i + 1]) >= 30)) {
                col_beg = (i+2 + col_beg)*2;
                break;
            }
        }
        col_end = col_beg;
        row_beg = row_beg * 2;
        row_end      = row_end * 2;
        int row_diff = std::abs(row_beg - row_end);
        if (row_diff < 10) {
            col_beg = 0;
            col_end = 0;
        }

        if ((col_beg > src.cols - 300 || col_beg < 100) && (row_diff > 20)) {   // 特殊破损，凹字形破损
            cv::Mat edgt_img;
            cv::Sobel(th_img, edgt_img, CV_16S, 0, 1, 3);
            cv::convertScaleAbs(edgt_img, edgt_img);

            std::vector<std::vector<cv::Point>> contours;
            std::vector<cv::Vec4i>              hierarchy;
            cv::findContours(edgt_img, contours, hierarchy, cv::RETR_LIST, cv::CHAIN_APPROX_NONE);
            std::vector<cv::Rect> ret_vec;
            for (size_t i = 0; i < contours.size(); ++i) {
                double area = contourArea(contours[i]);
                if (area < 100)
                    continue;
                cv::Rect        rect   = cv::boundingRect(contours[i]);
                cv::RotatedRect r_rect = cv::minAreaRect(contours[i]);
                if (rect.width < 200)
                    continue;
                ret_vec.emplace_back(rect);
            }
            if (ret_vec.size() > 0) {
                col_beg = ret_vec[0].x*2;
                col_end = ret_vec[0].x*2;
            }
            else {
                col_beg = 0;
                col_end = 0;
            }
        }
    
    }

    // 左长边
    if (type == 3) {
        enhance_img = gray_stairs(dst, th_low, th_high);
        enhance_img = gamma_trans(enhance_img, gama_value);
        cv::threshold(enhance_img, th_img, th_value, 255, cv::THRESH_TOZERO);

        // 查找上下边缘
        row_beg = 0;
        row_end = th_img.rows;
        get_row_range(th_img, row_beg, row_end, col_beg, col_end, th_value, continuous);

      cv::Mat ROI = th_img(cv::Rect(col_beg, row_beg, col_end - col_beg, row_end - row_beg));
        // 构造数据
        cv::Mat row_mat, col_mat;
        cv::reduce(ROI, col_mat, 0, cv::REDUCE_AVG);
        std::vector<int> col_vec = col_mat.reshape(1, 1);

        // 寻找左右边界
        std::vector<int> col_diff;
        col_diff.emplace_back(col_vec[0]);
        col_diff.emplace_back(col_vec[1]);
        for (int i = 2; i < col_vec.size() - 2; i++) {
            int data = std::abs(col_vec[i + 1] + col_vec[i + 2] - col_vec[i - 1] - col_vec[i - 2]);
            if (data > 10) {
                data = std::abs(col_vec[i + 1] - col_vec[i]);
            }
            col_diff.emplace_back(data);
        }
        int col_last_data = col_diff[col_diff.size() - 1];
        col_diff[0]       = col_diff[2];
        col_diff[1]       = col_diff[2];
        col_diff.emplace_back(col_last_data);
        col_diff.emplace_back(col_last_data);
        // 初始值
       
        for (int i = col_diff.size() - 1; i >=0; i--) {
            if ((std::abs(col_diff[i]) >= 20 || std::abs(col_diff[i]) + std::abs(col_diff[i - 1]) >= 30)) {
                col_beg = (i + 2 + col_beg) * 2;
                break;
            }
        }
        col_end = col_beg;
        row_beg = row_beg * 2;
        row_end      = row_end * 2;
        int row_diff = std::abs(row_beg - row_end);
        if (row_diff < 10) {
            col_beg = 0;
            col_end = 0;
        }

        if ((col_beg > src.cols - 300 || col_beg < 100) && (row_diff > 20)) {   // 特殊破损，凹字形破损
            cv::Mat edgt_img;
            cv::Sobel(th_img, edgt_img, CV_16S, 0, 1, 3);
            cv::convertScaleAbs(edgt_img, edgt_img);

            std::vector<std::vector<cv::Point>> contours;
            std::vector<cv::Vec4i>              hierarchy;
            cv::findContours(edgt_img, contours, hierarchy, cv::RETR_LIST, cv::CHAIN_APPROX_NONE);
            std::vector<cv::Rect> ret_vec;
            for (size_t i = 0; i < contours.size(); ++i) {
                double area = contourArea(contours[i]);
                if (area < 100)
                    continue;
                cv::Rect        rect   = cv::boundingRect(contours[i]);
                cv::RotatedRect r_rect = cv::minAreaRect(contours[i]);
                if (rect.width < 200)
                    continue;
                ret_vec.emplace_back(rect);
            }
            if (ret_vec.size() > 0) {
                col_beg = ret_vec[0].x * 2;
                col_end = ret_vec[0].x * 2;
            }
            else {
                col_beg = 0;
                col_end = 0;
            }
        }
    
    }
    // 左短边
    if (type == 4) {
        enhance_img = gray_stairs(dst, th_low, th_high);
        enhance_img = gamma_trans(enhance_img, gama_value);
        cv::threshold(enhance_img, th_img, th_value, 255, cv::THRESH_TOZERO);

        // 查找上下边缘
        row_beg = 0;
        row_end = th_img.rows;
        get_row_range(th_img, row_beg, row_end, col_beg, col_end, th_value, continuous);

       cv::Mat ROI = th_img(cv::Rect(col_beg, row_beg, col_end - col_beg, row_end - row_beg));
        // 构造数据
        cv::Mat row_mat, col_mat;
        cv::reduce(ROI, col_mat, 0, cv::REDUCE_AVG);
        std::vector<int> col_vec = col_mat.reshape(1, 1);

        // 寻找左右边界
        std::vector<int> col_diff;
        col_diff.emplace_back(col_vec[0]);
        col_diff.emplace_back(col_vec[1]);
        for (int i = 2; i < col_vec.size() - 2; i++) {
            int data = std::abs(col_vec[i + 1] + col_vec[i + 2] - col_vec[i - 1] - col_vec[i - 2]);
            if (data > 10) {
                data = std::abs(col_vec[i + 1] - col_vec[i]);
            }
            col_diff.emplace_back(data);
        }
        int col_last_data = col_diff[col_diff.size() - 1];
        col_diff[0]       = col_diff[2];
        col_diff[1]       = col_diff[2];
        col_diff.emplace_back(col_last_data);
        col_diff.emplace_back(col_last_data);
        // 初始值
       
        for (int i = col_diff.size()-1; i >= 0; i--) {
            if ((std::abs(col_diff[i]) >= 20 || std::abs(col_diff[i]) + std::abs(col_diff[i - 1]) >= 30)) {
                col_beg = (i + 2 + col_beg) * 2;
                break;
            }
        }
        col_end = col_beg;

        row_beg = row_beg * 2;
        row_end      = row_end * 2;
        int row_diff = std::abs(row_beg - row_end);
        if (row_diff < 10) {
            col_beg = 0;
            col_end = 0;
        }

        if ((col_beg > src.cols - 300 || col_beg < 100) && (row_diff > 20)) {   // 特殊破损，凹字形破损
            cv::Mat edgt_img;
            cv::Sobel(th_img, edgt_img, CV_16S, 0, 1, 3);
            cv::convertScaleAbs(edgt_img, edgt_img);

            std::vector<std::vector<cv::Point>> contours;
            std::vector<cv::Vec4i>              hierarchy;
            cv::findContours(edgt_img, contours, hierarchy, cv::RETR_LIST, cv::CHAIN_APPROX_NONE);
            std::vector<cv::Rect> ret_vec;
            for (size_t i = 0; i < contours.size(); ++i) {
                double area = contourArea(contours[i]);
                if (area < 100)
                    continue;
                cv::Rect        rect   = cv::boundingRect(contours[i]);
                cv::RotatedRect r_rect = cv::minAreaRect(contours[i]);
                if (rect.width < 200)
                    continue;
                ret_vec.emplace_back(rect);
            }
            if (ret_vec.size() > 0) {
                col_beg = ret_vec[0].x * 2;
                col_end = ret_vec[0].x * 2;
            }
            else {
                col_beg = 0;
                col_end = 0;
            }
        }
    }

#if 1
    if (dis_mat.channels() != 3)
        cv::cvtColor(dis_mat, dis_mat, cv::COLOR_GRAY2BGR);
    cv::line(dis_mat, cv::Point(0, row_beg), cv::Point(dis_mat.cols, row_beg), cv::Scalar(0, 0, 255), 2);
    cv::line(dis_mat, cv::Point(0, row_end - 1), cv::Point(dis_mat.cols, row_end - 1), cv::Scalar(0, 0, 255), 2);
    cv::line(dis_mat, cv::Point(col_beg, row_beg), cv::Point(col_beg, row_beg + 50), cv::Scalar(0, 0, 255), 2);
    cv::line(dis_mat, cv::Point(col_end, row_end), cv::Point(col_end, row_end - 50), cv::Scalar(0, 0, 255), 2);
    /*std::string file_name = R"(D:\code\tv_algorithm\result\)" + std::string(task->image_info["img_name"]) + "_AlgoPreA" + ".jpg";
    cv::imwrite(file_name, dis_mat);*/
#endif
}
int getmean(const cv::Mat& img,const std::vector<cv::Point>& countor,const cv::Point& offset){
    cv::Mat mask = cv::Mat::zeros(img.size(), CV_8UC1);
    std::vector<std::vector<cv::Point>> contours;
    contours.push_back(countor);
    cv::drawContours(mask, contours, -1, cv::Scalar(255, 255, 255), 1, 8, cv::Mat(), 0, offset);
    
    cv::Mat roi;
    cv::bitwise_and(img, img, roi, mask);
    cv::Scalar sum = cv::sum(roi);
    // 计算亮度的平均值
    int count = cv::countNonZero(mask);
    double meanBrightness = (count > 0) ? (sum[0] / count) : 0;
    return meanBrightness;

}
}   // namespace xx