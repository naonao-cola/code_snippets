/**
* @file    threshold.hpp
* @author  闹闹
* @brief   二值化头文件
* @details 为天地立心，为生民立命，为往圣继绝学，为万世开太平。
* @version 1.0.0.1
* @date    2022年6月22日14:08:23
* @copyright Copyright (c) 2050
* @note    参考链接 https://imagej.net/plugins/auto-threshold
* https://github.com/fiji/Auto_Threshold
*/
/**********************************************************************************
* @par 修改日志:
* <table>
* <tr><th>Date                 <th>Version    <th>Author     <th>Description
* <tr><td>2022年6月25日07:26:36<td>1.0.0.2    <td>naonao     <td>修改了局部阈值化的bug，增加了阈值化函数的忽略白色和忽略黑色的功能。修改了局部阈值的方法默认为高斯。
* </table>
*
**********************************************************************************
* @par 修改日志:
* <table>
* <tr><th>Date                 <th>Version    <th>Author     <th>Description
* <tr><td>2022年6月25日07:44:04<td>1.0.0.3    <td>naonao     <td>增加了阈值筛选函数，select(cv::Mat src)，图像会进行显示，按需要选择合适的阈值化类型。
* </table>
*
* **********************************************************************************
* @par 修改日志:
* <table>
* <tr><th>Date                 <th>Version    <th>Author     <th>Description
* <tr><td>2022年7月22日13:22:00<td>1.0.0.4    <td>naonao     <td>增加第18种局部二值化
* </table>
***/
#ifndef __THRESHOLF_H__
#define __THRESHOLF_H__
#pragma once
#include <algorithm>
#include <vector>
#include <iostream>
#include <memory>
#include <math.h>
#include <float.h>
#pragma warning(disable : 4627)
#include "opencv2/opencv.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

namespace nao {
    namespace algorithm {
        /**
         * @brief 二值化类型
        */
        enum class THRESHOLD_TYPE{
            DEFAULT       = 0,
            HUANG         = 1,
            HUANG2        = 2,
            INTERMODES    = 3,
            ISO_DATA      = 4,
            LI            = 5,
            MAX_ENTROPY   = 6,
            MEAN          = 7,
            MIN_ERROR     = 8,
            MINIMUM       = 9,
            MOMENTS       = 10,
            OTSU          = 11,
            PERCENTILE    = 12,
            RENYI_ENTROPY = 13,
            SHANBHAG      = 14,
            TRIANGLE      = 15,
            YEN           = 16,
            JUBU          = 17,
            SAUVOLA       = 18
        };
        enum class METHOD { MEAN, GAUSS, MEDIAN };

        namespace threshold {

            /**
             * @brief 获取直方图
             * @param src
             * @param dst
            */
            static void get_histogram(const cv::Mat& src, int* dst) {
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
            /**
             * @brief
             * @param data
             * @return
             * @details 该过程通过采用初始阈值将图像分为对象和背景，然后计算等于或低于阈值的像素和高于阈值的像素的平均值。
             * 计算这两个值的平均值，增加阈值并重复该过程，直到阈值大于复合平均值。与iso_data类似
            */
            int threshold_default(std::vector<int> data);

            /**
             * @brief
             * @param data
             * @return
             * @details 实现Huang的模糊阈值方法,使用香农熵函数（也可以使用Yager熵函数）
            */
            int huang(std::vector<int> data);

            /**
             * @brief
             * @param data
             * @return
             * @details 实现Huang的模糊阈值方法,使用香农熵函数（也可以使用Yager熵函数）
            */
            int huang2(std::vector<int> data);

            /**
             * @brief  中间模
             * @param data
             * @return
             * @details 采用双峰直方图。直方图需要平滑（使用迭代地运行大小为3的平均值），直到只有两个局部极大值。j和k阈值t为（j+k）/2。
             * 直方图具有极不相等的峰值或宽而平坦不适合这种方法。
            */
            int intermodes(std::vector<int> data);

            /**
             * @brief
             * @param data
             * @return
             * @details 该过程通过采用初始阈值将图像分为对象和背景，然后计算等于或低于阈值的像素和高于阈值的像素的平均值。
             * 计算这两个值的平均值，增加阈值并重复该过程，直到阈值大于复合平均值。与threshold_default类似
            */
            int iso_data(std::vector<int> data);

            /**
             * @brief
             * @param data
             * @return
             * @details 实现了李的最小交叉熵阈值方法,该实现基于算法的迭代版本
            */
            int li(std::vector<int>data);

            /**
             * @brief
             * @param data
             * @return
             * @details 一种利用直方图熵进行灰度图像阈值化的新方法,实现 Kapur-Sahoo-Wong（最大熵）阈值方法。
            */
            int max_entropy(std::vector<int>data);

            /**
             * @brief
             * @param data
             * @return
             * @details 阈值是灰度数据的平均值
            */
            int mean(std::vector<int>data);

            /**
             * @brief
             * @param data
             * @return
             * @details Kittler 和 Illingworth 的最小误差阈值的迭代实现。此实现似乎比原始实现更频繁地收敛。然而，有时该算法不会收敛到一个解决方案。
               在这种情况下，将向日志窗口报告警告，结果默认为使用均值方法计算的阈值的初始估计值。
            */
            int min_errorI(std::vector<int>data);

            /**
             * @brief
             * @param data
             * @return
             * @details 与 Intermodes 方法类似，这假设一个双峰直方图。
             * 直方图使用大小为 3 的运行平均值迭代平滑，直到只有两个局部最大值。阈值 t 使得 yt-1 > yt <= yt+1。
             * 直方图具有极不相等的峰或宽而平坦的谷的图像不适合这种方法。
            */
            int minimum(std::vector<int>data);

            /**
             * @brief
             * @param data
             * @return
             * @details  Tsai的方法试图在阈值化结果中保留原始图像的矩。“矩保持阈值：一种新方法”。
            */
            int moments(std::vector<int>data);

            /**
             * @brief 大津阈值法
             * @param data
             * @return
            */
            int otsu(std::vector<int> data);
            /**
             * @brief 百分比阈值法，默认50%
             * @param data
             * @param ptile
             * @return
            */
            int percentile(std::vector<int> data,double ptile = 0.5);

            /**
             * @brief
             * @param data
             * @return
             * @details 一种利用直方图熵进行灰度图像阈值化的新方法,类似于MaxEntropy方法，但使用 Renyi 的熵。
            */
            int renyi_entropy(std::vector<int> data);

            /**
             * @brief
             * @param data
             * @return
             * @details “利用信息度量作为图像阈值化“图形模型和图像处理”
            */
            int shanbhag(std::vector<int> data);

            /**
             * @brief
             * @param data
             * @return
             * @details 三角形算法是一种几何方法，它无法判断数据是否偏向一侧或另一侧，但会假定直方图一端附近的最大峰值（众数）
             并向另一端搜索。这会在缺少要处理的图像类型信息的情况下，或者当最大值不在直方图极值之一附近（导致在该最大值和极值
             之间产生两个可能的阈值区域）时出现问题。在这里，该算法被扩展为查找数据在最大峰值的哪一侧走得最远，并在该最大范围
             内搜索阈值。
            */
            int triangle(std::vector<int> data);

            /**
             * @brief
             * @param data
             * @return
             * @details 实现日元阈值方法
            */
            int yen(std::vector<int> data);

            /**
             * @brief
             * @param data
             * @return
             * @details 局部自适应阈值
            */
            int jubu(cv::Mat& src, nao::algorithm::METHOD type = nao::algorithm::METHOD::GAUSS,int radius = 3, float ratio = 0.15);


            /**
             * @brief
             * @param src 单通道灰度图
             * @param dst 单通道处理后的图
             * @param k   threshold = mean*(1 + k*((std / 128) - 1))
             * @param wnd_size 处理区域宽高, 一定是奇数
             * @return
             * @note https://blog.csdn.net/wfh2015/article/details/80418336?spm=1001.2101.3001.6650.3&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7Edefault-3-80418336-blog-86595319.pc_relevant_multi_platform_whitelistv1_exp2&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7Edefault-3-80418336-blog-86595319.pc_relevant_multi_platform_whitelistv1_exp2&utm_relevant_index=6
            */
            int sauvola(cv::Mat& src,const double k = 0.04, const int wnd_size = 9);
            /**
             * @brief 执行阈值化
             * @param src  原图
             * @param type 阈值化类型
             * @param doIblack 忽略黑色的像素值大小的范围，在此范围下，不进行统计。默认-1，建议 20 到40左右
             * @param doIwhite 忽略白色的像素值大小的范围，在此范围上，不进行统计，默认-1, 建议 220到255左右
             * @param reset    是否进行阈值化
             * @return         返回阈值
            */
            int exec_threshold(cv::Mat& src, nao::algorithm::THRESHOLD_TYPE type,int doIblack = -1,int doIwhite = -1, bool reset = false) {
                int threshold = -1;
                if (src.empty() || src.channels() != 1) return threshold;

                const int gray_scale = 256;
                int data[gray_scale] = { 0 };
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

                std::vector<int> data2(scale_range, { 0 });
                for (int i = minbin; i <= maxbin; i++) {
                    data2[i - minbin] = data[i];
                }

                if (type == nao::algorithm::THRESHOLD_TYPE::DEFAULT) {
                    threshold = threshold_default(data2);
                }
                else if (type == nao::algorithm::THRESHOLD_TYPE::HUANG) {
                    threshold = huang(data2);
                }
                else if (type == nao::algorithm::THRESHOLD_TYPE::HUANG2) {
                    threshold = huang2(data2);
                }
                else if (type == nao::algorithm::THRESHOLD_TYPE::INTERMODES) {
                    threshold = intermodes(data2);
                }
                else if (type == nao::algorithm::THRESHOLD_TYPE::ISO_DATA) {
                    threshold = iso_data(data2);
                }
                else if (type == nao::algorithm::THRESHOLD_TYPE::LI) {
                    threshold = li(data2);
                }
                else if (type == nao::algorithm::THRESHOLD_TYPE::MAX_ENTROPY) {
                    threshold = max_entropy(data2);
                }
                else if (type == nao::algorithm::THRESHOLD_TYPE::MEAN) {
                    threshold = mean(data2);
                }
                else if (type == nao::algorithm::THRESHOLD_TYPE::MIN_ERROR) {
                    threshold = min_errorI(data2);
                }
                else if (type == nao::algorithm::THRESHOLD_TYPE::MINIMUM) {
                    threshold = minimum(data2);
                }
                else if (type == nao::algorithm::THRESHOLD_TYPE::MOMENTS) {
                    threshold = moments(data2);
                }
                else if (type == nao::algorithm::THRESHOLD_TYPE::OTSU) {
                    threshold = otsu(data2);
                }
                else if (type == nao::algorithm::THRESHOLD_TYPE::PERCENTILE) {
                    threshold = percentile(data2);
                }
                else if (type == nao::algorithm::THRESHOLD_TYPE::RENYI_ENTROPY) {
                    threshold = renyi_entropy(data2);
                }
                else if (type == nao::algorithm::THRESHOLD_TYPE::SHANBHAG) {
                    threshold = shanbhag(data2);
                }
                else if (type == nao::algorithm::THRESHOLD_TYPE::TRIANGLE) {
                    threshold = triangle(data2);
                }
                else if (type == nao::algorithm::THRESHOLD_TYPE::YEN) {
                    threshold = yen(data2);
                }
                else if (type == nao::algorithm::THRESHOLD_TYPE::JUBU) {
                    //局部自适应阈值返回-1
                    jubu(src);
                    return -1;
                }
                else if (type == nao::algorithm::THRESHOLD_TYPE::SAUVOLA) {
                    sauvola(src);
                    return -1;
                }
                threshold += minbin;

                if (reset) {
                    cv::threshold(src, src, threshold, 255, cv::THRESH_BINARY);
                }
                return threshold;
            }

            /**
             * @brief 选择阈值化类型
             * @param src
             * @return
            */
            int select(cv::Mat& src,int type = -1) {
               if (type != -1) {
                    cv::Mat img = src.clone();
                    cv::Mat dst;
                    int th = exec_threshold(img, (nao::algorithm::THRESHOLD_TYPE)type);
                    cv::threshold(img, dst, th, 255, cv::THRESH_BINARY);
                    src = dst.clone();
                    return 0;
                }
                for (int i = 0; i < 19; i++) {
                    std::cout << "当前的阈值化类型的次序为： " << i;
                    cv::Mat img = src.clone();
                    cv::Mat dst;
                    int th = exec_threshold(img, (nao::algorithm::THRESHOLD_TYPE)i);
                    std::cout<<"th: " << th << std::endl;
                    if (i != 17 || i != 18) {
                        cv::threshold(img, dst, th, 255, cv::THRESH_BINARY);
                       cv::imshow("阈值化图像", dst);
                        cv::waitKey(0);
                    }
                    else {
                        cv::imshow("阈值化图像", img);
                        cv::waitKey(0);
                    }
                }
                return 0;
            }

            int threshold_default(std::vector<int> data) {
                size_t level;
                size_t maxValue = data.size() - 1;
                double result, sum1, sum2, sum3, sum4;
                int min = 0;
                while ((data[min] == 0) && (min < maxValue))
                    min++;
                int max = (int)maxValue;
                while ((data[max] == 0) && (max > 0))
                    max--;
                if (min >= max) {
                    level = data.size() / 2;
                    return (int)level;
                }
                int movingIndex = min;
                //int inc = std::max(max / 40, 1);
                do {
                    sum1 = sum2 = sum3 = sum4 = 0.0;
                    for (int i = min; i <= movingIndex; i++) {
                        sum1 += i * data[i];
                        sum2 += data[i];
                    }
                    for (int i = (movingIndex + 1); i <= max; i++) {
                        sum3 += i * data[i];
                        sum4 += data[i];
                    }
                    result = (sum1 / sum2 + sum3 / sum4) / 2.0;
                    movingIndex++;
                } while ((movingIndex + 1) <= result && movingIndex < max - 1);

                level = (int)(std::roundf(result));
                return (int)level;
            }



            int huang(std::vector<int> data) {
                int threshold = -1;
                int ih, it;
                int first_bin;
                int last_bin;
                int sum_pix;
                int num_pix;
                double term;
                double ent;  // entropy
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


            int huang2(std::vector<int> data) {
                int first, last;
                for (first = 0; first < data.size() && data[first] == 0; first++)
                    ; // do nothing
                for (last = data.size() - 1; last > first && data[last] == 0; last--)
                    ; // do nothing
                if (first == last) return 0;


                //计算累计密度与 累计密度的权重
                std::vector<double> S(last + 1, { 0.0 });
                std::vector<double> W(last + 1, { 0.0 });
                S[0] = data[0];
                for (int i = (std::max)(1, first); i <= last; i++) {
                    S[i] = S[i - 1] + data[i];
                    W[i] = W[i - 1] + i * data[i];
                }

                // precalculate the summands of the entropy given the absolute difference x - mu (integral)
                //给定绝对差x-mu（积分），预先计算熵的总和
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

            bool bimodalTest(std::vector<double> y) {
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


            int intermodes(std::vector<int> data) {
                std::vector<double> iHisto(data.size(), {0.0});
                int iter = 0;
                int threshold = -1;
                for (int i = 0; i < data.size(); i++)
                    iHisto[i] = (double)data[i];

                while (!bimodalTest(iHisto)) {
                    //smooth with a 3 point running mean filter
                    double previous = 0, current = 0, next = iHisto[0];
                    for (int i = 0; i < data.size() - 1; i++) {
                        previous = current;
                        current = next;
                        next = iHisto[i + 1];
                        iHisto[i] = (previous + current + next) / 3;
                    }
                    iHisto[data.size() - 1] = (current + next) / 3;
                    iter++;
                    if (iter > 10000) {
                        threshold = -1;
                        return threshold;
                    }
                }

                // The threshold is the mean between the two peaks.
                int tt = 0;
                for (int i = 1; i < data.size() - 1; i++) {
                    if (iHisto[i - 1] < iHisto[i] && iHisto[i + 1] < iHisto[i]) {
                        tt += i;
                    }
                }
                threshold = (int)std::floor(tt / 2.0);
                return threshold;
            }


            int iso_data(std::vector<int> data) {
                int i, l, toth, totl, h, g = 0;
                for (i = 1; i < data.size(); i++) {
                    if (data[i] > 0) {
                        g = i + 1;
                        break;
                    }
                }
                while (true) {
                    l = 0;
                    totl = 0;
                    for (i = 0; i < g; i++) {
                        totl = totl + data[i];
                        l = l + (data[i] * i);
                    }
                    h = 0;
                    toth = 0;
                    for (i = g + 1; i < data.size(); i++) {
                        toth += data[i];
                        h += (data[i] * i);
                    }
                    if (totl > 0 && toth > 0) {
                        l /= totl;
                        h /= toth;
                        if (g == (int)std::round((l + h) / 2.0))
                            break;
                    }
                    g++;
                    if (g > data.size() - 2) {
                        return -1;
                    }
                }
                return g;
            }


            int li(std::vector<int>data) {
                int threshold;
                int ih;
                int num_pixels;
                int sum_back;     ///< 给定阈值下背景像素的总和
                int sum_obj;      ///< 给定阈值下对象像素的总和
                int num_back;     ///< 给定阈值下的背景像素数
                int num_obj;      ///< 给定阈值下的对象像素数
                double old_thresh;
                double new_thresh;
                double mean_back; ///<给定阈值下背景像素的平均值
                double mean_obj;  ///<给定阈值下对象像素的平均值
                double mean;      ///< 图像中的平均灰度
                double tolerance; ///< 阈值公差
                double temp;

                tolerance = 0.5;
                num_pixels = 0;
                for (ih = 0; ih < data.size(); ih++)
                    num_pixels += data[ih];

                /* Calculate the mean gray-level */
                mean = 0.0;
                for (ih = 0; ih < data.size(); ih++) //0 + 1?
                    mean += ih * data[ih];
                mean /= num_pixels;
                /* Initial estimate */
                new_thresh = mean;

                do {
                    old_thresh = new_thresh;
                    threshold = (int)(old_thresh + 0.5);	/* range */
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
                    //new_thresh = simple_round ( ( mean_back - mean_obj ) / ( Math.log ( mean_back ) - Math.log ( mean_obj ) ) );
                    //simple_round ( double x ) {
                    // return ( int ) ( IS_NEG ( x ) ? x - .5 : x + .5 );
                    //}
                    //
                    //#define IS_NEG( x ) ( ( x ) < -DBL_EPSILON )
                    //DBL_EPSILON = 2.220446049250313E-16
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


            int max_entropy(std::vector<int>data) {
                int threshold = -1;
                int ih, it;
                int first_bin;
                int last_bin;
                double tot_ent;  ///< 总熵
                double max_ent;  ///< 最大熵
                double ent_back; ///< 给定阈值下背景像素的熵
                double ent_obj;  ///< 给定阈值下对象像素的熵

                std::vector<double> norm_histo(data.size(), {0.0});///< 归一化直方图
                std::vector<double>P1(data.size(), { 0.0 });///< 累积归一化直方图
                std::vector<double>P2(data.size(), { 0.0 });

                int total = 0;
                for (ih = 0; ih < data.size(); ih++)
                    total += data[ih];

                for (ih = 0; ih < data.size(); ih++)
                    norm_histo[ih] = (double)data[ih] / total;

                P1[0] = norm_histo[0];
                P2[0] = 1.0 - P1[0];
                for (ih = 1; ih < data.size(); ih++) {
                    P1[ih] = P1[ih - 1] + norm_histo[ih];
                    P2[ih] = 1.0 - P1[ih];
                }
                // 确定第一个非零维度
                first_bin = 0;
                for (ih = 0; ih < data.size(); ih++) {
                    if (!(std::abs(P1[ih]) < 2.220446049250313E-16)) {
                        first_bin = ih;
                        break;
                    }
                }
                // 确定最后一个非零维度
                last_bin = data.size() - 1;
                for (ih = data.size() - 1; ih >= first_bin; ih--) {
                    if (!(std::abs(P2[ih]) < 2.220446049250313E-16)) {
                        last_bin = ih;
                        break;
                    }
                }
                // 计算每个灰度级的总熵
                // 并找到使其最大化的阈值
                max_ent = DBL_MIN;
                for (it = first_bin; it <= last_bin; it++) {
                    /* 背景像素熵 */
                    ent_back = 0.0;
                    for (ih = 0; ih <= it; ih++) {
                        if (data[ih] != 0) {
                            ent_back -= (norm_histo[ih] / P1[it]) * std::log(norm_histo[ih] / P1[it]);
                        }
                    }
                    /* 对象像素的熵 */
                    ent_obj = 0.0;
                    for (ih = it + 1; ih < data.size(); ih++) {
                        if (data[ih] != 0) {
                            ent_obj -= (norm_histo[ih] / P2[it]) * std::log(norm_histo[ih] / P2[it]);
                        }
                    }
                    /* 总熵 */
                    tot_ent = ent_back + ent_obj;
                    if (max_ent < tot_ent) {
                        max_ent = tot_ent;
                        threshold = it;
                    }
                }
                return threshold;
            }


            int mean(std::vector<int>data) {
                int threshold = -1;
                long tot = 0, sum = 0;
                for (int i = 0; i < data.size(); i++) {
                    tot += data[i];
                    sum += ((long)i * (long)data[i]);
                }
                threshold = (int)std::floor(sum / tot);
                return threshold;
            }
            static double A(std::vector<int> y, int j) {
                double x = 0;
                for (int i = 0; i <= j; i++)
                    x += y[i];
                return x;
            }

           static double B(std::vector<int> y, int j) {
                double x = 0;
                for (int i = 0; i <= j; i++)
                    x += i * y[i];
                return x;
            }

            static double C(std::vector<int> y, int j) {
                double x = 0;
                for (int i = 0; i <= j; i++)
                    x += i * i * y[i];
                return x;
            }


            int min_errorI(std::vector<int>data) {
                int threshold = mean(data); //用均值算法得到阈值的初始估计。
                int Tprev = -2;
                double mu, nu, p, q, sigma2, tau2, w0, w1, w2, sqterm, temp;
                //int counter=1;
                while (threshold != Tprev) {
                    //计算一些统计数据。
                    mu = B(data, threshold) / A(data, threshold);
                    nu = (B(data, data.size() - 1) - B(data, threshold)) / (A(data, data.size() - 1) - A(data, threshold));
                    p = A(data, threshold) / A(data, data.size() - 1);
                    q = (A(data, data.size() - 1) - A(data, threshold)) / A(data, data.size() - 1);
                    sigma2 = C(data, threshold) / A(data, threshold) - (mu * mu);
                    tau2 = (C(data, data.size() - 1) - C(data, threshold)) / (A(data, data.size() - 1) - A(data, threshold)) - (nu * nu);

                    //要求解的二次方程的项。
                    w0 = 1.0 / sigma2 - 1.0 / tau2;
                    w1 = mu / sigma2 - nu / tau2;
                    w2 = (mu * mu) / sigma2 - (nu * nu) / tau2 + std::log10((sigma2 * (q * q)) / (tau2 * (p * p)));

                    //如果下一个阈值是虚构的，则返回当前阈值。
                    sqterm = (w1 * w1) - w0 * w2;
                    if (sqterm < 0) {
                        //IJ.log("MinError(I): not converging. Try \'Ignore black/white\' options");
                        return threshold;
                    }

                    //更新后的阈值是二次方程解的整数部分。
                    Tprev = threshold;
                    temp = (w1 + std::sqrt(sqterm)) / w0;

                    if (std::isnan(temp)) {
                        //IJ.log("MinError(I): NaN, not converging. Try \'Ignore black/white\' options");
                        threshold = Tprev;
                    }
                    else
                        threshold = (int)std::floor(temp);
                    //IJ.log("Iter: "+ counter+++"  t:"+threshold);
                }
                return threshold;

            }

            int minimum(std::vector<int>data) {
                int iter = 0;
                int threshold = -1;
                int max = -1;
                std::vector<double> iHisto(data.size(), {0.0});


                for (int i = 0; i < data.size(); i++) {
                    iHisto[i] = (double)data[i];
                    if (data[i] > 0) max = i;
                }
                std::vector<double> tHisto(iHisto.size(), { 0.0 });// Instead of double[] tHisto = iHisto ;
                while (!bimodalTest(iHisto)) {
                    //使用3点运行平均值过滤器平滑
                    for (int i = 1; i < data.size() - 1; i++)
                        tHisto[i] = (iHisto[i - 1] + iHisto[i] + iHisto[i + 1]) / 3;
                    tHisto[0] = (iHisto[0] + iHisto[1]) / 3; //0 outside
                    tHisto[data.size() - 1] = (iHisto[data.size() - 2] + iHisto[data.size() - 1]) / 3; //0 outside
                    //System.arraycopy(tHisto, 0, iHisto, 0, iHisto.size()); //Instead of iHisto = tHisto ;
                    std::copy_n(tHisto.begin(), iHisto.size(), iHisto.begin());
                    iter++;
                    if (iter > 10000) {
                        threshold = -1;
                        //IJ.log("Minimum Threshold not found after 10000 iterations.");
                        return threshold;
                    }
                }
                // 阈值是两个峰值之间的最小值。修改为16位

                for (int i = 1; i < max; i++) {
                    //IJ.log(" "+i+"  "+iHisto[i]);
                    if (iHisto[i - 1] > iHisto[i] && iHisto[i + 1] >= iHisto[i]) {
                        threshold = i;
                        break;
                    }
                }
                return threshold;

            }

            int moments(std::vector<int>data) {
                double total = 0;
                double m0 = 1.0, m1 = 0.0, m2 = 0.0, m3 = 0.0, sum = 0.0, p0 = 0.0;
                double cd, c0, c1, z0, z1;	///< 辅助变量
                int threshold = -1;


                std::vector<double> histo(data.size(), {0.0});

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

            int otsu(std::vector<int> data) {
                int ih;
                int threshold = -1;
                int num_pixels = 0;
                double total_mean; ///<整个图像的平均灰度
                double bcv, term; ///< 类间方差，缩放系数
                double max_bcv;		///< max BCV

                std::vector<double> cnh(data.size(), {0.0});///<累积归一化直方图
                std::vector<double> mean(data.size(), {0.0});///<平均灰度
                std::vector<double> histo(data.size(), {0.0}); ///<归一化直方图
                //计算值为非0的像素的个数
                for (ih = 0; ih < data.size(); ih++)
                    num_pixels = num_pixels + data[ih];

                //计算每个灰度级的像素数目占整幅图像的比例,相当于归一化直方图
                term = 1.0 / (double)num_pixels;
                for (ih = 0; ih < data.size(); ih++)
                    histo[ih] = term * data[ih];
                //计算累积归一化直方图
                cnh[0] = histo[0];
                mean[0] = 0.0;
                for (ih = 1; ih < data.size(); ih++){
                    cnh[ih] = cnh[ih - 1] + histo[ih];
                    mean[ih] = mean[ih - 1] + ih * histo[ih];
                }
                total_mean = mean[data.size() - 1];
                //计算每个灰度的BCV，并找到使其最大化的阈值,
                 max_bcv = 0.0;
                 for (ih = 0; ih < data.size(); ih++) {
                     //通分，约化之后的简写
                     bcv = total_mean * cnh[ih] - mean[ih];
                     bcv *= bcv / (cnh[ih] * (1.0 - cnh[ih]));
                     if (max_bcv < bcv) {
                         max_bcv = bcv;
                         threshold = ih;
                     }
                 }
                 return threshold;
            }


            template<typename T>
            static double partialSum(std::vector<T> y, int j) {
                double x = 0;
                for (int i = 0; i <= j; i++)
                    x += y[i];
                return x;
            }

            int percentile(std::vector<int> data,double ptile) {
                int threshold = -1;
                //double ptile = 0.5; ///< 前景像素的默认百分比数
                std::vector<double> avec(data.size(), {0.0});
                double total = partialSum(data, (int)(data.size() - 1));
                double temp = 1.0;
                for (int i = 0; i < data.size(); i++) {
                    avec[i] = std::fabs((partialSum(data, i) / total) - ptile);
                    if (avec[i] < temp) {
                        temp = avec[i];
                        threshold = i;
                    }
                }
                return threshold;
            }


            int renyi_entropy(std::vector<int> data) {
                int threshold;
                int opt_threshold;

                int ih, it;
                int first_bin;
                int last_bin;
                int tmp_var;
                int t_star1, t_star2, t_star3;
                int beta1, beta2, beta3;
                double alpha;/* alpha parameter of the method */
                double term;
                double tot_ent;  /* total entropy */
                double max_ent;  /* max entropy */
                double ent_back; /* entropy of the background pixels at a given threshold */
                double ent_obj;  /* entropy of the object pixels at a given threshold */
                double omega;

                std::vector<double> norm_histo(data.size(), {0.0});/* normalized histogram */

                std::vector<double> P1(data.size(), { 0.0 });/* cumulative normalized histogram */
                std::vector<double> P2(data.size(), { 0.0 });


                int total = 0;
                for (ih = 0; ih < data.size(); ih++)
                    total += data[ih];

                for (ih = 0; ih < data.size(); ih++)
                    norm_histo[ih] = (double)data[ih] / total;

                P1[0] = norm_histo[0];
                P2[0] = 1.0 - P1[0];
                for (ih = 1; ih < data.size(); ih++) {
                    P1[ih] = P1[ih - 1] + norm_histo[ih];
                    P2[ih] = 1.0 - P1[ih];
                }

                /* Determine the first non-zero bin */
                first_bin = 0;
                for (ih = 0; ih < data.size(); ih++) {
                    if (!(std::abs(P1[ih]) < 2.220446049250313E-16)) {
                        first_bin = ih;
                        break;
                    }
                }

                /* Determine the last non-zero bin */
                last_bin = data.size() - 1;
                for (ih = data.size() - 1; ih >= first_bin; ih--) {
                    if (!(std::abs(P2[ih]) < 2.220446049250313E-16)) {
                        last_bin = ih;
                        break;
                    }
                }

                /* Maximum Entropy Thresholding - BEGIN */
                /* ALPHA = 1.0 */
                /* Calculate the total entropy each gray-level
                and find the threshold that maximizes it
                */
                threshold = 0; // was MIN_INT in original code, but if an empty image is processed it gives an error later on.
                max_ent = 0.0;

                for (it = first_bin; it <= last_bin; it++) {
                    /* Entropy of the background pixels */
                    ent_back = 0.0;
                    for (ih = 0; ih <= it; ih++) {
                        if (data[ih] != 0) {
                            ent_back -= (norm_histo[ih] / P1[it]) * std::log(norm_histo[ih] / P1[it]);
                        }
                    }

                    /* Entropy of the object pixels */
                    ent_obj = 0.0;
                    for (ih = it + 1; ih < data.size(); ih++) {
                        if (data[ih] != 0) {
                            ent_obj -= (norm_histo[ih] / P2[it]) * std::log(norm_histo[ih] / P2[it]);
                        }
                    }

                    /* Total entropy */
                    tot_ent = ent_back + ent_obj;

                    // IJ.log(""+max_ent+"  "+tot_ent);

                    if (max_ent < tot_ent) {
                        max_ent = tot_ent;
                        threshold = it;
                    }
                }
                t_star2 = threshold;

                /* Maximum Entropy Thresholding - END */
                threshold = 0; //was MIN_INT in original code, but if an empty image is processed it gives an error later on.
                max_ent = 0.0;
                alpha = 0.5;
                term = 1.0 / (1.0 - alpha);
                for (it = first_bin; it <= last_bin; it++) {
                    /* Entropy of the background pixels */
                    ent_back = 0.0;
                    for (ih = 0; ih <= it; ih++)
                        ent_back += std::sqrt(norm_histo[ih] / P1[it]);

                    /* Entropy of the object pixels */
                    ent_obj = 0.0;
                    for (ih = it + 1; ih < data.size(); ih++)
                        ent_obj += std::sqrt(norm_histo[ih] / P2[it]);

                    /* Total entropy */
                    tot_ent = term * ((ent_back * ent_obj) > 0.0 ? std::log(ent_back * ent_obj) : 0.0);

                    if (tot_ent > max_ent) {
                        max_ent = tot_ent;
                        threshold = it;
                    }
                }

                t_star1 = threshold;

                threshold = 0; //was MIN_INT in original code, but if an empty image is processed it gives an error later on.
                max_ent = 0.0;
                alpha = 2.0;
                term = 1.0 / (1.0 - alpha);
                for (it = first_bin; it <= last_bin; it++) {
                    /* Entropy of the background pixels */
                    ent_back = 0.0;
                    for (ih = 0; ih <= it; ih++)
                        ent_back += (norm_histo[ih] * norm_histo[ih]) / (P1[it] * P1[it]);

                    /* Entropy of the object pixels */
                    ent_obj = 0.0;
                    for (ih = it + 1; ih < data.size(); ih++)
                        ent_obj += (norm_histo[ih] * norm_histo[ih]) / (P2[it] * P2[it]);

                    /* Total entropy */
                    tot_ent = term * ((ent_back * ent_obj) > 0.0 ? std::log(ent_back * ent_obj) : 0.0);

                    if (tot_ent > max_ent) {
                        max_ent = tot_ent;
                        threshold = it;
                    }
                }

                t_star3 = threshold;

                /* Sort t_star values */
                if (t_star2 < t_star1) {
                    tmp_var = t_star1;
                    t_star1 = t_star2;
                    t_star2 = tmp_var;
                }
                if (t_star3 < t_star2) {
                    tmp_var = t_star2;
                    t_star2 = t_star3;
                    t_star3 = tmp_var;
                }
                if (t_star2 < t_star1) {
                    tmp_var = t_star1;
                    t_star1 = t_star2;
                    t_star2 = tmp_var;
                }

                /* Adjust beta values */
                if (std::abs(t_star1 - t_star2) <= 5) {
                    if (std::abs(t_star2 - t_star3) <= 5) {
                        beta1 = 1;
                        beta2 = 2;
                        beta3 = 1;
                    }
                    else {
                        beta1 = 0;
                        beta2 = 1;
                        beta3 = 3;
                    }
                }
                else {
                    if (std::abs(t_star2 - t_star3) <= 5) {
                        beta1 = 3;
                        beta2 = 1;
                        beta3 = 0;
                    }
                    else {
                        beta1 = 1;
                        beta2 = 2;
                        beta3 = 1;
                    }
                }
                //IJ.log(""+t_star1+" "+t_star2+" "+t_star3);
                /* Determine the optimal threshold value */
                omega = P1[t_star3] - P1[t_star1];
                opt_threshold = (int)(t_star1 * (P1[t_star1] + 0.25 * omega * beta1) + 0.25 * t_star2 * omega * beta2 + t_star3 * (P2[t_star3] + 0.25 * omega * beta3));

                return opt_threshold;

            }

            int shanbhag(std::vector<int> data) {
                int threshold;
                int ih, it;
                int first_bin;
                int last_bin;
                double term;
                double tot_ent;  /* total entropy */
                double min_ent;  /* max entropy */
                double ent_back; /* entropy of the background pixels at a given threshold */
                double ent_obj;  /* entropy of the object pixels at a given threshold */

                std::vector<double> norm_histo(data.size(), { 0.0 });/* normalized histogram */
                std::vector<double> P1(data.size(), { 0.0 });/* cumulative normalized histogram */
                std::vector<double> P2(data.size(), { 0.0 });


                int total = 0;
                for (ih = 0; ih < data.size(); ih++)
                    total += data[ih];

                for (ih = 0; ih < data.size(); ih++)
                    norm_histo[ih] = (double)data[ih] / total;

                P1[0] = norm_histo[0];
                P2[0] = 1.0 - P1[0];
                for (ih = 1; ih < data.size(); ih++) {
                    P1[ih] = P1[ih - 1] + norm_histo[ih];
                    P2[ih] = 1.0 - P1[ih];
                }

                /* Determine the first non-zero bin */
                first_bin = 0;
                for (ih = 0; ih < data.size(); ih++) {
                    if (!(std::abs(P1[ih]) < 2.220446049250313E-16)) {
                        first_bin = ih;
                        break;
                    }
                }

                /* Determine the last non-zero bin */
                last_bin = data.size() - 1;
                for (ih = data.size() - 1; ih >= first_bin; ih--) {
                    if (!(std::abs(P2[ih]) < 2.220446049250313E-16)) {
                        last_bin = ih;
                        break;
                    }
                }

                // Calculate the total entropy each gray-level
                // and find the threshold that maximizes it
                threshold = -1;
                min_ent = DBL_MAX;

                for (it = first_bin; it <= last_bin; it++) {
                    /* Entropy of the background pixels */
                    ent_back = 0.0;
                    term = 0.5 / P1[it];
                    for (ih = 1; ih <= it; ih++) { //0+1?
                        ent_back -= norm_histo[ih] * std::log(1.0 - term * P1[ih - 1]);
                    }
                    ent_back *= term;

                    /* Entropy of the object pixels */
                    ent_obj = 0.0;
                    term = 0.5 / P2[it];
                    for (ih = it + 1; ih < data.size(); ih++) {
                        ent_obj -= norm_histo[ih] * std::log(1.0 - term * P2[ih]);
                    }
                    ent_obj *= term;

                    /* Total entropy */
                    tot_ent = std::abs(ent_back - ent_obj);

                    if (tot_ent < min_ent) {
                        min_ent = tot_ent;
                        threshold = it;
                    }
                }
                return threshold;
            }


            int triangle(std::vector<int> data) {
                int min = 0, dmax = 0, max = 0, min2 = 0;
                for (int i = 0; i < data.size(); i++) {
                    if (data[i] > 0) {
                        min = i;
                        break;
                    }
                }
                if (min > 0) min--; // line to the (p==0) point, not to data[min]

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
                if (min2 < data.size() - 1) min2++; // line to the (p==0) point, not to data[min]

                for (int i = 0; i < data.size(); i++) {
                    if (data[i] > dmax) {
                        max = i;
                        dmax = data[i];
                    }
                }
                // find which is the furthest side
                //IJ.log(""+min+" "+max+" "+min2);
                bool inverted = false;
                if ((max - min) < (min2 - max)) {
                    // reverse the histogram
                    //IJ.log("Reversing histogram.");
                    inverted = true;
                    int left = 0;          // index of leftmost element
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
                    //IJ.log("Triangle:  min == max.");
                    return min;
                }

                // describe line by nx * x + ny * y - d = 0
                double nx, ny, d;
                // nx is just the max frequency as the other point has freq=0
                nx = data[max];   //-min; // data[min]; //  lowest value bmin = (p=0)% in the image
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
                }
                else
                    return split;

            }
            int yen(std::vector<int> data) {
                int threshold;
                int ih, it;
                double crit;
                double max_crit;

                std::vector<double> norm_histo(data.size(), { 0.0 });/* normalized histogram */
                std::vector<double> P1(data.size(), { 0.0 });/* cumulative normalized histogram */
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
            int jubu(cv::Mat& src, nao::algorithm::METHOD type, int radius, float ratio) {
                // 对图像矩阵进行平滑处理
                cv::Mat smooth;
                switch (type) {
                case nao::algorithm::METHOD::MEAN:
                    cv::boxFilter(src, smooth, CV_32FC1, cv::Size(2 * radius + 1, 2 * radius + 1));
                    break;
                case nao::algorithm::METHOD::GAUSS:
                    cv::GaussianBlur(src, smooth, cv::Size(2 * radius + 1, 2 * radius + 1), 0.0);
                    break;
                case nao::algorithm::METHOD::MEDIAN:
                    cv::medianBlur(src, smooth, 2 * radius + 1);
                    break;
                default:
                    break;
                }
                // 平滑结果乘以比例系数，然后图像矩阵与其做差
                src.convertTo(src, CV_32FC1);
                if(smooth.type()!= CV_32FC1) smooth.convertTo(smooth, CV_32FC1);
                cv::Mat diff = src - (1.0 - ratio) * smooth;
                // 阈值处理，当大于或等于0时，输出值为 255，反之输出为0
                cv::Mat out = cv::Mat::zeros(diff.size(), CV_8UC1);
                for (int r = 0; r < out.rows; r++){
                    for (int c = 0; c < out.cols; c++){
                        if (diff.at<float>(r, c) >= 0)
                            out.at<uchar>(r, c) = 255;
                    }
                }
                src = out.clone();
                return 1;
            }

            int sauvola(cv::Mat& src, const double k, const int wnd_size) {
                CV_Assert(src.type() == CV_8UC1);
                CV_Assert(wnd_size % 2 == 1);
                CV_Assert((wnd_size <= src.cols) && (wnd_size <= src.rows));
                cv::Mat dst = cv::Mat::zeros(src.rows, src.cols, CV_8UC1);

                unsigned long* integralImg = new unsigned long[src.rows * src.cols];
                unsigned long* integralImgSqrt = new unsigned long[src.rows * src.cols];
                std::memset(integralImg, 0, src.rows * src.cols * sizeof(unsigned long));
                std::memset(integralImgSqrt, 0, src.rows * src.cols * sizeof(unsigned long));

                // 计算直方图和图像值平方的和,积分图函数(cv::integral),未测试
                for (int y = 0; y < src.rows; ++y){
                    unsigned long sum = 0;
                    unsigned long sqrtSum = 0;
                    for (int x = 0; x < src.cols; ++x){
                        int index = y * src.cols + x;
                        int value_pix = *src.ptr<uchar>(y, x);
                        sum += value_pix;
                        sqrtSum += value_pix * value_pix;
                        if (y == 0) {
                            integralImg[index] = sum;
                            integralImgSqrt[index] = sqrtSum;
                        }
                        else {
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
                        if (area <= 0){
                            // blog提供源码是biImage[i * IMAGE_WIDTH + j] = 255;但是i表示的是列, j是行
                            dst.at<uchar>(y, x) = 255;
                            continue;
                        }
                        if (xMin == 0 && yMin == 0) {
                            diff = integralImg[yMax * src.cols + xMax];
                            sqDiff = integralImgSqrt[yMax * src.cols + xMax];
                        }
                        else if (xMin > 0 && yMin == 0) {
                            diff = integralImg[yMax * src.cols + xMax] - integralImg[yMax * src.cols + xMin - 1];
                            sqDiff = integralImgSqrt[yMax * src.cols + xMax] - integralImgSqrt[yMax * src.cols + xMin - 1];
                        }
                        else if (xMin == 0 && yMin > 0) {
                            diff = integralImg[yMax * src.cols + xMax] - integralImg[(yMin - 1) * src.cols + xMax];
                            sqDiff = integralImgSqrt[yMax * src.cols + xMax] - integralImgSqrt[(yMin - 1) * src.cols + xMax];;
                        }
                        else
                        {
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
                        }
                        else {
                            dst.at<uchar>(y, x) = 255;
                        }

                    }
                }
                delete[] integralImg;
                delete[] integralImgSqrt;
                src = dst.clone();
            }
        }//namespace threshold
    }//namespace algorithm
}//namespace nao
#endif  //__THRESHOLF_H__
/*----------------------------------------------------------------------------- (C) COPYRIGHT LEI *****END OF FILE------------------------------------------------------------------------------*/