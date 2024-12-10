#include "../include/test.h"

#include <fstream>

#include "../include/contour.h"
#include "../include/fs.h"
#include "../include/miscellaneous.h"
#include "../include/threshold.h"
#include "../include/transform.hpp"

void test_print_start() { std::cout << "pragram start" << std::endl; }

void circle_process(cv::Mat src, cv::Mat img,
                    std::vector<std::vector<cv::Point2f>> &circle_process_float,
                    std::vector<std::vector<cv::Point>> circle_process_int) {
  cv::Mat gauss_img;
  cv::GaussianBlur(img, gauss_img, cv::Size(3, 3), 0);
  nao::algorithm::threshold::select(gauss_img, 9);
  cv::Mat bise;
  cv::bitwise_not(gauss_img, bise);
  std::vector<std::vector<cv::Point>> contours;
  std::vector<cv::Vec4i> hierarchy;
  cv::findContours(gauss_img, contours, hierarchy, cv::RETR_LIST,
                   cv::CHAIN_APPROX_NONE);

  for (size_t i = 0; i < contours.size(); ++i) {
    double area = contourArea(contours[i]);
    cv::Rect rect = cv::boundingRect(contours[i]);
    // std::cout<<"area: "<<area <<"  width: "<<rect.width<<" high: " <<
    // rect.height<< std::endl;
    if (area < 3 * 1e2 || 1e7 < area) continue;
    if (rect.height < 40 || rect.height > 300) continue;
    if (rect.width < 40 || rect.width > 300) continue;
    if ((rect.width / rect.height) > 3 || (rect.height / rect.width) > 3)
      continue;
    // 绘制轮廓
    // cv::drawContours(src, contours, i, CV_RGB(255, 0, 0), 3, 8, hierarchy,
    // 0);
    circle_process_int.emplace_back(contours[i]);
  }
  for (int i = 0; i < circle_process_int.size(); i++) {
    std::vector<cv::Point2f> tmp;
    for (int j = 0; j < circle_process_int[i].size(); j++) {
      tmp.emplace_back(circle_process_int[i][j]);
    }
    circle_process_float.emplace_back(tmp);
  }
}

void pca_circle(cv::Mat src,
                std::vector<std::vector<cv::Point2f>> circle_process_float) {
  for (int i = 0; i < circle_process_float.size(); i++) {
    cv::Point2f pos;
    double angle = 0;
    angle = nao::algorithm::contours::getOrientation(circle_process_float[i],
                                                     pos, src);
  }

  return;
}

// 得到图像与直线的交点
void get_cross_pt(cv::Mat src, cv::Point2f p0, cv::Point2f p1,
                  std::vector<cv::Point2f> &cross_pt) {
  // if (p0.x >= src.cols || p0.x <= 0 || p0.y <= 0 || p0.y >= src.cols ||
  //     p1.x >= src.cols || p1.x <= 0 || p1.y <= 0 || p1.y >= src.cols) {
  //     return;
  // }
  double k_t;
  k_t = p1.x - p0.x;

  if (k_t == 0.) {
    cross_pt.push_back(cv::Point2f(p0.x, 0));
    cross_pt.push_back(cv::Point2f(p0.x, src.rows - 1));
  } else {
    double k, b;
    k = ((double)p1.y - p0.y) / (p1.x - p0.x);
    b = p1.y - k * p1.x;
    if (!k) {
      cross_pt.push_back(cv::Point2f(0, b));
      cross_pt.push_back(cv::Point2f(src.cols - 1, b));
    } else {
      int x1, x2, y1, y2;
      x1 = -b / k;                  // y=0
      x2 = (src.rows - 1 - b) / k;  // y=img->height-1
      y1 = b;                       // x=0
      y2 = k * (src.cols - 1) + b;  // x=img->width-1
      if (y1 >= 0 && y1 < src.rows) cross_pt.push_back(cv::Point2f(0, y1));
      if (y2 >= 0 && y2 < src.rows)
        cross_pt.push_back(cv::Point2f(src.cols - 1, y2));
      if (x1 >= 0 && x1 < src.cols) cross_pt.push_back(cv::Point2f(x1, 0));
      if (x2 >= 0 && x2 < src.cols)
        cross_pt.push_back(cv::Point2f(x2, src.rows - 1));
    }
  }
}
// 求直线与轮廓的交点,并求出距离
void get_points(cv::Mat src, std::vector<cv::Point2f> circle_point,
                cv::Point2f p0, cv::Point2f p1, cv::Point2f &f_pt,
                cv::Point2f &s_pt, double &max_dis) {
  std::vector<cv::Point2f> cross_pt;
  get_cross_pt(src, p0, p1, cross_pt);
  cv::LineIterator it(src, cross_pt[0], cross_pt[1]);
  std::vector<cv::Point2f> dst;

  for (int i = 0; i < it.count; i++, it++) {
    cv::Point2f pt(it.pos());
    if (std::abs(cv::pointPolygonTest(circle_point, pt, true)) <= 3) {
      // cv::circle(src, pt, 2, CV_RGB(255, 0, 0),1);
      dst.emplace_back(pt);
    }
  }

  if (dst.size() <= 0) return;
  for (int i = 0; i < dst.size(); i++) {
    for (int j = i + 1; j < dst.size(); j++) {
      double dis = std::powf((dst[i].x - dst[j].x), 2) +
                   std::powf((dst[i].y - dst[j].y), 2);
      dis = std::sqrtf(dis);
      if (dis >= max_dis) {
        max_dis = dis;
      }
    }
  }

  // std::cout<<" 距离 "<< max_dis<<std::endl;

  // 将轮廓分类
  std::vector<cv::Point2f> first;
  std::vector<cv::Point2f> second;
  first.emplace_back(dst[0]);
  for (int i = 1; i < dst.size(); i++) {
    double dis = std::powf((dst[i].x - first[0].x), 2) +
                 std::powf((dst[i].y - first[0].y), 2);
    dis = std::sqrtf(dis);
    if (dis <= 20) {
      first.emplace_back(dst[i]);
    } else {
      second.emplace_back(dst[i]);
    }
  }

  if (first.size() <= 0 || second.size() <= 0) {
    std::cout << " 交点出错" << std::endl;
    return;
  }

  double sum_x = 0, sum_y = 0;
  for (int i = 0; i < first.size(); i++) {
    sum_x = sum_x + first[i].x;
    sum_y = sum_y + first[i].y;
  }
  f_pt.x = sum_x / first.size();
  f_pt.y = sum_y / first.size();

  sum_x = 0;
  sum_y = 0;
  for (int i = 0; i < second.size(); i++) {
    sum_x = sum_x + second[i].x;
    sum_y = sum_y + second[i].y;
  }
  s_pt.x = sum_x / second.size();
  s_pt.y = sum_y / second.size();
}

// 求旋转直线与轮廓的交点
void get_length(cv::Mat src, std::vector<cv::Point2f> circle_point,
                cv::Point2f center, cv::Point2f p0, cv::Point2f p1,
                double rote_angle, double rote_step, cv::Point2f &t_f_pt,
                cv::Point2f &t_s_pt, double &min_dis) {
  // 原始点绕中线点旋转之后
  min_dis = 10000;

  for (double i = 0 - rote_angle; i <= 0 + rote_angle; i += rote_step) {
    cv::Point2f rote_p0, rote_p1;
    rote_p0.x = (p0.x - center.x) * cos(i * CV_PI / 180) -
                (p0.y - center.y) * sin(i * CV_PI / 180) + center.x;
    rote_p0.y = (p0.x - center.x) * sin(i * CV_PI / 180) +
                (p0.y - center.y) * cos(i * CV_PI / 180) + center.y;
    rote_p1.x = (p1.x - center.x) * cos(i * CV_PI / 180) -
                (p1.y - center.y) * sin(i * CV_PI / 180) + center.x;
    rote_p1.y = (p1.x - center.x) * sin(i * CV_PI / 180) +
                (p1.y - center.y) * cos(i * CV_PI / 180) + center.y;

    // cv::line(src, rote_p0, rote_p1, CV_RGB(255, 0, 0));
    cv::Point2f f_pt, s_pt;
    double max_dis = 0;
    get_points(src, circle_point, rote_p0, rote_p1, f_pt, s_pt, max_dis);
    // cv::line(src, f_pt, s_pt, CV_RGB(255, 0, 0));
    if (max_dis <= 5) continue;
    if (max_dis < min_dis) {
      min_dis = max_dis;
      t_f_pt = f_pt;
      t_s_pt = s_pt;
    }
  }
}

// 求平移直线与轮廓的交点
void get_length_2(cv::Mat src, std::vector<cv::Point2f> circle_point,
                  cv::Point2f center, cv::Point2f p0, cv::Point2f p1,
                  double trans, double trans_step, cv::Point2f &t_f_pt,
                  cv::Point2f &t_s_pt, double &min_dis) {
  // 原始点绕中线点旋转之后
  min_dis = 0;

  for (double i = 0 - trans; i <= 0 + trans; i += trans_step) {
    cv::Point2f rote_p0, rote_p1;
    rote_p0.x = p0.x;
    rote_p0.y = p0.y - i;
    rote_p1.x = p1.x;
    rote_p1.y = p1.y - i;

    // cv::line(src, rote_p0, rote_p1, CV_RGB(255, 0, 0));
    cv::Point2f f_pt, s_pt;
    double max_dis = 0;
    get_points(src, circle_point, rote_p0, rote_p1, f_pt, s_pt, max_dis);
    // cv::line(src, f_pt, s_pt, CV_RGB(255, 0, 0));
    if (max_dis > min_dis) {
      min_dis = max_dis;
      t_f_pt = f_pt;
      t_s_pt = s_pt;
    }
  }
}

// 拟合椭圆，
void fit_circle(cv::Mat src,
                std::vector<std::vector<cv::Point2f>> circle_process_float,
                std::vector<std::vector<cv::Point>> circle_process_int) {
  /*  std::ofstream outFile;
    outFile.open("fit_circle_1.csv", std::ios::out | std::ios::trunc);
    outFile << "index" << ','
        << "area" << ','
        << "dis" << ","
        << "p0_x" << ","
        << "p0_y" << ","
        << "p1_x" << ","
        << "p1_y" << std::endl;*/

  for (int i = 0; i < circle_process_float.size(); i++) {
    cv::Point2f pos;
    double angle = 0;
    cv::RotatedRect box = cv::fitEllipse(circle_process_float[i]);

    cv::ellipse(src, box, CV_RGB(255, 0, 0), 1, 8);
    double x = box.center.x;
    double y = box.center.y;
    double th = (box.angle);
    double h = box.size.height / 2;
    double w = box.size.width / 2;
    double cos = std::cos(th * CV_PI / 180);
    double sin = std::sin(th * CV_PI / 180);
    // 短边的 p0 p1
    cv::Point2f Long_axis0 = (cv::Point2f((x + cos * w), (y + w * sin)));
    cv::Point2f Long_axis1 = (cv::Point2f((x - cos * w), (y - w * sin)));

    // 长边的 p0 p1
    cv::Point2f short_axis0 = cv::Point2f((x - sin * h), (y + h * cos));
    cv::Point2f short_axis1 = cv::Point2f((x + sin * h), (y - h * cos));
    // cv::line(src, Long_axis0, Long_axis1, CV_RGB(255, 0, 0));
    // cv::line(src, short_axis0, short_axis1, CV_RGB(255, 0, 0));

    cv::Point2f t_f_pt;
    cv::Point2f t_s_pt;
    double min_dis;

    get_length(src, circle_process_float[i], cv::Point2f(x, y), Long_axis0,
               Long_axis1, 40, 0.8, t_f_pt, t_s_pt, min_dis);
    double area = cv::contourArea(circle_process_float[i]);
    std::cout << i << "," << std::to_string(area) << ", "
              << std::to_string(min_dis) << "," << std::to_string(t_f_pt.x)
              << "," << std::to_string(t_f_pt.y) << ","
              << std::to_string(t_s_pt.x) << "," << std::to_string(t_s_pt.y)
              << "," << std::to_string((t_f_pt.x + t_s_pt.x) / 2) << ","
              << std::to_string((t_f_pt.y + t_s_pt.y) / 2) << ","
              << std::to_string(min_dis * 8) << std::endl;

    /*outFile << i << ","
        << std::to_string(area) << ", "
        << std::to_string(min_dis) << ","
        << std::to_string(t_f_pt.x) << ","
        << std::to_string(t_f_pt.y) << ","
        << std::to_string(t_s_pt.x) << ","
        << std::to_string(t_s_pt.y) << std::endl;
        */
    cv::line(src, t_f_pt, t_s_pt, CV_RGB(255, 0, 0));
  }
  /* outFile.close();*/
  return;
}

// 拟合椭圆，
void fit_circle_2(cv::Mat src,
                  std::vector<std::vector<cv::Point2f>> circle_process_float,
                  std::vector<std::vector<cv::Point>> circle_process_int) {
  std::ofstream outFile;
  outFile.open("fit_circle_2.csv", std::ios::out | std::ios::trunc);
  outFile << "index" << ',' << "area" << ',' << "dis" << "," << "p0_x" << ","
          << "p0_y" << "," << "p1_x" << "," << "p1_y" << std::endl;
  for (int i = 0; i < circle_process_float.size(); i++) {
    cv::Point2f pos;
    double angle = 0;
    cv::RotatedRect box = cv::fitEllipse(circle_process_float[i]);

    cv::ellipse(src, box, CV_RGB(255, 0, 0), 1, 8);
    double x = box.center.x;
    double y = box.center.y;
    double th = (box.angle);
    double h = box.size.height / 2;
    double w = box.size.width / 2;
    double cos = std::cos(th * CV_PI / 180);
    double sin = std::sin(th * CV_PI / 180);
    // 短边的 p0 p1
    cv::Point2f Long_axis0 = (cv::Point2f((x + cos * w), (y + w * sin)));
    cv::Point2f Long_axis1 = (cv::Point2f((x - cos * w), (y - w * sin)));

    // 长边的 p0 p1
    cv::Point2f short_axis0 = cv::Point2f((x - sin * h), (y + h * cos));
    cv::Point2f short_axis1 = cv::Point2f((x + sin * h), (y - h * cos));
    // cv::line(src, Long_axis0, Long_axis1, CV_RGB(255, 0, 0));
    // cv::line(src, short_axis0, short_axis1, CV_RGB(255, 0, 0));

    cv::Point2f t_f_pt;
    cv::Point2f t_s_pt;
    double min_dis;

    get_length_2(src, circle_process_float[i], cv::Point2f(x, y), Long_axis0,
                 Long_axis1, 200, 5, t_f_pt, t_s_pt, min_dis);
    double area = cv::contourArea(circle_process_float[i]);
    std::cout << i << "," << std::to_string(area) << ", "
              << std::to_string(min_dis) << "," << std::to_string(t_f_pt.x)
              << "," << std::to_string(t_f_pt.y) << ","
              << std::to_string(t_s_pt.x) << "," << std::to_string(t_s_pt.y)
              << std::endl;
    outFile << i << "," << std::to_string(area) << ", "
            << std::to_string(min_dis) << "," << std::to_string(t_f_pt.x) << ","
            << std::to_string(t_f_pt.y) << "," << std::to_string(t_s_pt.x)
            << "," << std::to_string(t_s_pt.y) << std::endl;

    cv::line(src, t_f_pt, t_s_pt, CV_RGB(255, 0, 0));
  }

  outFile.close();
  return;
}

void test_circle_length(const std::string &img_path) {
  cv::Mat img = cv::imread(img_path);
  cv::Mat img_gray;
  cv::cvtColor(img, img_gray, cv::COLOR_RGB2GRAY);

  std::vector<std::vector<cv::Point2f>> circle_process_float;
  std::vector<std::vector<cv::Point>> circle_process_int;
  circle_process(img, img_gray, circle_process_float, circle_process_int);

  /* pca_circle(img, circle_process_1);*/

  fit_circle(img, circle_process_float, circle_process_int);

  // fit_circle_2(img, circle_process_float, circle_process_int);

  return;
}

void test_circle_imgs(const std::string &img_path) {
  std::vector<std::string> file_path;
  nao::fl::getAllFormatFiles(img_path, file_path);

  for (int i = 0; i < file_path.size(); i++) {
    cv::Mat img = cv::imread(file_path[i]);
    cv::Mat img_gray;
    cv::cvtColor(img, img_gray, cv::COLOR_RGB2GRAY);

    std::vector<std::vector<cv::Point2f>> circle_process_float;
    std::vector<std::vector<cv::Point>> circle_process_int;
    circle_process(img, img_gray, circle_process_float, circle_process_int);

    /* pca_circle(img, circle_process_1);*/

    fit_circle(img, circle_process_float, circle_process_int);
  }
}

#include "../include/EDLib.h"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/ximgproc.hpp"

/*
https://blog.csdn.net/weixin_40992875/article/details/121208601?spm=1001.2101.3001.6650.1&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7ERate-1-121208601-blog-109260998.235%5Ev38%5Epc_relevant_anti_t3_base&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7ERate-1-121208601-blog-109260998.235%5Ev38%5Epc_relevant_anti_t3_base&utm_relevant_index=2
*/
void test_circle_Detection(const std::string &img_path) {
  std::vector<std::string> file_path;
  nao::fl::getAllFormatFiles(img_path, file_path);

  for (int i = 0; i < file_path.size(); i++) {
    cv::Mat img = cv::imread(file_path[i]);
    cv::resize(img, img, cv::Size(img.cols / 2, img.rows / 2));
    cv::Mat img_gray;
    cv::cvtColor(img, img_gray, cv::COLOR_RGB2GRAY);
    cv::GaussianBlur(img_gray, img_gray, cv::Size(3, 3), 0);

    /* nao::algorithm::threshold::select(img_gray,6);*/
    nao::algorithm::threshold::sauvola(img_gray, 0.04, 11);
    // img_gray = nao::algorithm::morphology(img_gray,cv::Size(3,3),
    // nao::algorithm::PhologyType::PZ);

    EDPF testEDPF = EDPF(img_gray);
    EDCircles testEDCircles = EDCircles(testEDPF);
    std::vector<mCircle> found_circles = testEDCircles.getCircles();
    std::cout << "第 " << i << " 图 " << std::endl;

    std::vector<std::pair<cv::Point, int>> find_cricle;

    for (int j = 0; j < found_circles.size(); j++) {
      cv::Point center((int)found_circles[j].center.x,
                       (int)found_circles[j].center.y);
      cv::Size axes((int)found_circles[j].r, (int)found_circles[j].r);
      double angle(0.0);
      cv::Scalar color = cv::Scalar(0, 0, 255);
      if (((int)found_circles[j].r >= 55 && (int)found_circles[j].r <= 65) ||
          ((int)found_circles[j].r >= 95 && (int)found_circles[j].r <= 105)) {
        find_cricle.emplace_back(std::make_pair(center, found_circles[j].r));
        ellipse(img, center, axes, angle, 0, 360, color, 2, cv::LINE_AA);
        std::cout << "center: " << center << " size: " << axes << std::endl;
      }
    }

    /* cv::Ptr<cv::ximgproc::EdgeDrawing> ed =
    cv::ximgproc::createEdgeDrawing(); ed->params.EdgeDetectionOperator =
    cv::ximgproc::EdgeDrawing::SOBEL; ed->params.GradientThresholdValue = 36;
     ed->params.AnchorThresholdValue = 8;



     EDPF testEDPF = EDPF(img_gray);
     cv::TickMeter tm;

     cv::Ptr<cv::ximgproc::EdgeDrawing> ed1 = cv::ximgproc::createEdgeDrawing();
     ed1->params.EdgeDetectionOperator = cv::ximgproc::EdgeDrawing::PREWITT;
     ed1->params.GradientThresholdValue = 11;
     ed1->params.AnchorThresholdValue = 3;
     ed1->params.PFmode = true;

     tm.reset();
     tm.start();
    ed1->detectEdges(img_gray);
     tm.stop();
     std::cout << "detectEdges()  PF        (OpenCV)  : " << tm.getTimeMilli()
    << std::endl;

     cv::Mat edgeImg0, edgeImg1, diff;
     edgeImg0 = testEDPF.getEdgeImage();
     ed->getEdgeImage(edgeImg1);
     cv::absdiff(edgeImg0, edgeImg1, diff);
     std::cout << "different pixel count              : " <<
    cv::countNonZero(diff) << std::endl;*/

    // 分类 得到 00000111111222222的类型
    std::vector<int> index_used(find_cricle.size(), -1);
    std::set<int> sort_used;
    for (int m = 0; m < find_cricle.size(); m++) {
      if (index_used[m] != -1) continue;
      for (int n = m + 1; n < find_cricle.size(); n++) {
        if (index_used[n] != -1) continue;
        double dis =
            std::pow((find_cricle[m].first.x - find_cricle[n].first.x), 2) +
            std::pow((find_cricle[m].first.y - find_cricle[n].first.y), 2);
        dis = std::sqrt(dis);
        if (dis < 100) {
          if (index_used[m] == -1) {
            index_used[n] = m;
            index_used[m] = m;
          } else {
            index_used[n] = index_used[m];
          }
        }
      }
      if (index_used[m] == -1) {
        index_used[m] = m;
      }
      sort_used.insert(index_used[m]);
    }

    int category = 0;
    for (std::set<int>::iterator it = sort_used.begin(); it != sort_used.end();
         it++) {
      int current_index = *it;
      std::vector<std::pair<cv::Point, int>> sub_circle;

      for (int q = 0; q < index_used.size(); q++) {
        if (index_used[q] == current_index) {
          // std::make_pair(center, found_circles[j].r)
          sub_circle.emplace_back(
              std::make_pair(find_cricle[q].first, find_cricle[q].second));
        }
      }
      double dis = 0;
      std::vector<cv::Point> first_pt;
      std::vector<cv::Point> second_pt;

      if (sub_circle.size() < 2) goto savejpg;

      cv::Point pt = sub_circle[0].first;
      first_pt.emplace_back(pt);
      for (int p = 1; p < sub_circle.size(); p++) {
        cv::Point cur = sub_circle[p].first;
        double dis = std::pow((pt.x - cur.x), 2) + std::pow((pt.y - cur.y), 2);
        dis = std::sqrt(dis);
        if (dis < 20) {
          first_pt.emplace_back(cur);
        } else {
          second_pt.emplace_back(cur);
        }
      }

      cv::Point f_avg = (0, 0);
      cv::Point s_avg = (0, 0);
      for (int i = 0; i < first_pt.size(); i++) {
        f_avg.x = f_avg.x + first_pt[i].x;
        f_avg.y = f_avg.y + first_pt[i].y;
      }
      if (first_pt.size() > 0) f_avg.x = f_avg.x / first_pt.size();
      if (first_pt.size() > 0) f_avg.y = f_avg.y / first_pt.size();

      for (int j = 0; j < second_pt.size(); j++) {
        s_avg.x = s_avg.x + second_pt[j].x;
        s_avg.y = s_avg.y + second_pt[j].y;
      }
      if (second_pt.size() > 0) s_avg.x = s_avg.x / second_pt.size();
      if (second_pt.size() > 0) s_avg.y = s_avg.y / second_pt.size();

      /*double dis = 0;*/
      if (f_avg.x != 0 && s_avg.x != 0) {
        dis =
            std::pow((f_avg.x - s_avg.x), 2) + std::pow((f_avg.y - s_avg.y), 2);
        dis = std::sqrt(dis);
      }

    savejpg:
      std::cout << "distance :" << dis << std::endl;
      std::string text = "distance :" + std::to_string(dis);
      cv::Point origin;
      origin.x = 100;
      origin.y = category * 50 + 50;
      cv::putText(img, text, origin, cv::FONT_HERSHEY_COMPLEX, 2,
                  cv::Scalar(0, 0, 255), 2, 8, 0);
      category++;
    }
    cv::imwrite(R"(E:\demo\test\test_opencv\ret\)" + std::to_string(i) + ".jpg",
                img);
  }
}

// 求两直线交点
cv::Point2d get2lineIPoint(cv::Point2d lineOnePt1, cv::Point2d lineOnePt2,
                           cv::Point2d lineTwoPt1, cv::Point2d lineTwoPt2) {
  double x;
  double y;
  cv::Point2d result(-1, -1);
  double k = (lineOnePt1.y - lineOnePt2.y) / (lineOnePt1.x - lineOnePt2.x);
  double b = lineOnePt1.y - k * lineOnePt1.x;
  double k1 = (lineTwoPt1.y - lineTwoPt2.y) / (lineTwoPt1.x - lineTwoPt2.x);
  double b1 = lineTwoPt1.y - k1 * lineTwoPt1.x;

  x = (b1 - b) / (k - k1);
  y = k * x + b;
  result.x = x;
  result.y = y;

  return result;
}

#include "../include/find_line.h"
void test_find_line(const std::string &img_path) {
  std::vector<std::string> file_path;
  nao::fl::getAllFormatFiles(img_path, file_path);

  int dTheta = 15;
  for (int i = 0; i < file_path.size(); i++) {
    cv::Mat img = cv::imread(file_path[i]);
    cv::resize(img, img, cv::Size(img.cols / 2, img.rows / 2));
    cv::Mat img_gray;
    cv::Point img_center = cv::Point(img.cols / 2, img.rows / 2);

    cv::Mat filter_Image =
        cv::Mat(img.rows, img.cols, CV_8UC1, cv::Scalar(255));
    cv::Mat line_Image = cv::Mat(img.rows, img.cols, CV_8UC1, cv::Scalar(255));

    linedetect::BlockedRect obj;
    obj.initParams();
    obj.detect(img);
    obj.drawLines(line_Image);

    obj.mergeLines();
    // obj.drawMergeLines(filter_Image);

    std::vector<linedetect::CLineParams2f> h_vec, v_vec;

    for (int m = 0; m < obj.merge_lines_Hori__.size(); m++) {
      linedetect::CLineParams2f cur = obj.merge_lines_Hori__[m];
      cur.compute(img);
      double v0 = cur.avg_pix[0];
      double v1 = cur.avg_pix[1];
      double v2 = cur.avg_pix[2];
      if (std::abs(v0 - v1) > 40 || std::abs(v0 - v2) > 40 ||
          std::abs(v2 - v1) > 40)
        continue;
      h_vec.emplace_back(cur);
      cv::line(filter_Image, cur.start_vertex, cur.end_vertex,
               cv::Scalar(0, 0, 0), 1);
    }
    for (int m = 0; m < obj.merge_lines_Hori__.size(); m++) {
      linedetect::CLineParams2f cur = obj.merge_lines_Verti__[m];
      cur.compute(img);
      v_vec.emplace_back(cur);
      cv::line(filter_Image, cur.start_vertex, cur.end_vertex,
               cv::Scalar(0, 0, 0), 1);
    }
    int a = 1;

    // for (int n = 0; n < filt_line_h_vec.size();n++) {
    //     if (filt_line_h_vec[n].start.y < img_center.y) {
    //         filt_line_top_vec.emplace_back(filt_line_h_vec[n]);
    //     }
    //     else {
    //         filt_line_bottom_vec.emplace_back(filt_line_h_vec[n]);
    //     }
    // }
    // for (int k = 0; k < filt_line_v_vec.size(); k++) {
    //     if (filt_line_v_vec[k].start.x < img_center.x) {
    //         filt_line_left_vec.emplace_back(filt_line_v_vec[k]);
    //     }
    //     else {
    //         filt_line_right_vec.emplace_back(filt_line_v_vec[k]);
    //     }
    // }

    // std::sort(filt_line_top_vec.begin(), filt_line_top_vec.end(), [&](const
    // Line_info& a, const Line_info& b) {return a.distance < b.distance; });
    // std::sort(filt_line_bottom_vec.begin(), filt_line_bottom_vec.end(),
    // [&](const Line_info& a, const Line_info& b) {return a.distance <
    // b.distance; }); std::sort(filt_line_left_vec.begin(),
    // filt_line_left_vec.end(), [&](const Line_info& a, const Line_info& b)
    // {return a.distance < b.distance; });
    // std::sort(filt_line_right_vec.begin(), filt_line_right_vec.end(),
    // [&](const Line_info& a, const Line_info& b) {return a.distance <
    // b.distance; });

    ////选中的四条线
    // Line_info select_top = filt_line_top_vec[0];
    // Line_info select_bottom = filt_line_bottom_vec[0];
    // Line_info select_left = filt_line_left_vec[0];
    // Line_info select_right = filt_line_right_vec[0];

    ////左上角
    // cv::Point2d  left_top = get2lineIPoint(select_top.start, select_top.end,
    // select_left.start, select_left.end);
    ////右上角
    // cv::Point2d right_top = get2lineIPoint(select_top.start, select_top.end,
    // select_right.start, select_right.end);
    ////右下角
    // cv::Point2d right_bottom = get2lineIPoint(select_bottom.start,
    // select_bottom.end, select_right.start, select_right.end);
    ////左下角
    // cv::Point2d left_bottom = get2lineIPoint(select_bottom.start,
    // select_bottom.end, select_left.start, select_left.end);

    // cv::Mat wrap;

    // std::vector<cv::Point2f> src_pt, dst_pt;
    // src_pt.emplace_back(left_top);
    // src_pt.emplace_back(right_top);
    // src_pt.emplace_back(right_bottom);
    // src_pt.emplace_back(left_bottom);

    // dst_pt.emplace_back(cv::Point2f(120,40));
    // dst_pt.emplace_back(cv::Point2f(1502, 40));
    // dst_pt.emplace_back(cv::Point2f(1502, 1128));
    // dst_pt.emplace_back(cv::Point2f(120, 1128));

    // nao::algorithm::transform::perspective(src_pt, dst_pt,wrap);
    // cv::Mat wrap_img;
    // cv::warpPerspective(img,wrap_img,wrap,img.size());
  }
}

// https://blog.csdn.net/u011754972/article/details/121533505
void t_process(const std::string &img_path) {
  cv::Mat src = cv::imread(img_path);
  cv::Mat hsv;
  cv::cvtColor(src, hsv, cv::COLOR_BGR2HSV);

  // # 在HSV空间中定义绿色
  cv::Scalar lower_green = cv::Scalar(35, 50, 50);
  cv::Scalar upper_green = cv::Scalar(77, 255, 255);

  cv::Mat green_mask;
  cv::inRange(hsv, lower_green, upper_green, green_mask);

  cv::Mat green_res;
  // 三通道图像进行单通道掩模操作后，输出图像还是三通道。相当于对三通道都做了掩模。
  cv::bitwise_and(src, src, green_res, green_mask);

  // 寻找绿色
  std::vector<cv::Rect> rec_vec;
  std::vector<std::vector<cv::Point>> filter_contours;
  std::vector<cv::Vec4i> filter_hierarchy;
  cv::findContours(green_mask, filter_contours, filter_hierarchy,
                   cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
  for (size_t i = 0; i < filter_contours.size(); ++i) {
    cv::Rect rect = cv::boundingRect(filter_contours[i]);
    double area = cv::contourArea(filter_contours[i]);

    cv::Moments m = cv::moments(filter_contours[i]);
    int center_x = int(m.m10 / m.m00);
    int center_y = int(m.m01 / m.m00);

    double dis = std::powf((src.cols / 2 - center_x), 2) +
                 std::powf((src.rows / 2 - center_y), 2);
    dis = std::sqrt(dis);
    if (rect.width >= 1300 && dis < 600) {
      rec_vec.emplace_back(rect);
    }
    if (center_x > 0 && center_y > 0) {
      std::cout << i << " " << rect.width << " " << rect.height << " "
                << center_x << " " << center_y << std::endl;
    }
  }

  int a = 1;
}

cv::Mat gray_stairs(const cv::Mat &img, double sin, double hin, double mt,
                    double sout, double hout) {
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

void test_convexityDefects(const std::string &img_path) {
  cv::Mat src = cv::imread(img_path);
  if (src.channels() > 1) {
    cv::cvtColor(src, src, cv::COLOR_RGB2GRAY);
  }

  cv::Mat enhince_img;
  enhince_img = gray_stairs(src, 0, 160, 1.15);

  cv::Mat sobel_img_y, soble_img_x, edge_img, blur_img, th_img;
  cv::Sobel(enhince_img, sobel_img_y, CV_64F, 0, 1, 3);
  cv::Sobel(enhince_img, soble_img_x, CV_64F, 1, 0, 3);
  edge_img = soble_img_x + sobel_img_y;
  cv::convertScaleAbs(edge_img, edge_img);

  cv::medianBlur(edge_img, blur_img, 3);

  cv::threshold(blur_img, th_img, 30, 255, cv::THRESH_BINARY);

  std::vector<std::vector<cv::Point>> contours;
  std::vector<cv::Vec4i> hierarchy;
  cv::findContours(th_img, contours, hierarchy, cv::RETR_EXTERNAL,
                   cv::CHAIN_APPROX_NONE);

  cv::Mat dis_img = cv::Mat::zeros(th_img.rows, th_img.cols, CV_8UC3);
  for (size_t i = 0; i < contours.size(); ++i) {
    std::vector<cv::Point> hull;
    cv::convexHull(contours[i], hull, false);

    std::vector<int> int_hull;
    cv::convexHull(contours[i], int_hull, false);
    cv::Rect rect = cv::boundingRect(contours[i]);

    if (rect.height > 200 && rect.x > 850) {
      cv::drawContours(dis_img, contours, i, CV_RGB(255, 0, 0));

      std::vector<cv::Vec4i> hullDefect;
      cv::convexityDefects(contours[i], int_hull, hullDefect);

      for (int j = 0; j < hullDefect.size(); j++) {
        cv::Vec4i cur_pt = hullDefect[j];
        int start_idx = cur_pt[0];
        cv::Point pt_start(contours[i][start_idx]);

        int end_idx = cur_pt[1];
        cv::Point pt_end(contours[i][end_idx]);

        int faridx = cur_pt[2];
        cv::Point ptFar(contours[i][faridx]);

        cv::line(dis_img, pt_start, ptFar, CV_RGB(0, 255, 0), 2);
        cv::line(dis_img, pt_end, ptFar, CV_RGB(0, 255, 0), 2);
      }
    }
  }
}

void test_match_img(const std::string &iplimg_path,
                    const std::string &img_path) {
  // 分别读取模板图，缺陷图
  std::vector<std::string> ipl_img_paths;
  nao::fl::getAllFormatFiles(iplimg_path, ipl_img_paths);
  std::vector<std::string> img_paths;
  nao::fl::getAllFormatFiles(img_path, img_paths);

  // 根据文件名一一对应
  // 模板图与缺陷图可能是一对多,判断缺陷图属于哪个模板
  std::map<std::string, std::vector<std::string>> img_path_map;
  for (int j = 0; j < ipl_img_paths.size(); j++) {
    std::string cur_ipl_path = ipl_img_paths[j];
    std::vector<std::string> second_string;
    for (int i = 0; i < img_paths.size(); i++) {
      std::string cur_img = std::filesystem::path(img_paths[i]).stem().string();
      std::string cur_ipl =
          std::filesystem::path(ipl_img_paths[j]).stem().string();
      std::string::size_type idx = cur_img.find(cur_ipl);
      if (idx != std::string::npos) {
        second_string.emplace_back(img_paths[i]);
      }
    }
    img_path_map.insert(
        std::map<std::string, std::vector<std::string>>::value_type(
            cur_ipl_path, second_string));
  }

  // 开始模板匹配做差
  for (auto item : img_path_map) {
    auto first = item.first;
    auto second = item.second;
    cv::Mat ipl_mat = cv::imread(first);

    for (int m = 0; m < second.size(); m++) {
      cv::Mat img = cv::imread(second[m]);
      cv::Mat ret, ret_inv;
      cv::absdiff(img, ipl_mat, ret);

      cv::threshold(ret, ret, 30, 255, cv::THRESH_BINARY);
      cv::Mat element =
          cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
      cv::erode(ret, ret, element);
      element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(21, 21));
      cv::dilate(ret, ret, element);

      if (ret.channels() > 1) cv::cvtColor(ret, ret_inv, cv::COLOR_RGB2GRAY);
      std::vector<std::vector<cv::Point>> contours;
      std::vector<cv::Vec4i> hierarchy;
      cv::findContours(ret_inv, contours, hierarchy, cv::RETR_EXTERNAL,
                       cv::CHAIN_APPROX_NONE);
      cv::Mat dis_img = img.clone();

      for (size_t i = 0; i < contours.size(); ++i) {
        cv::Rect rect = cv::boundingRect(contours[i]);
        double area = contourArea(contours[i]);

        if (area < 50 || rect.width < 30 || rect.height < 30) {
          continue;
        }
        cv::rectangle(dis_img, rect, cv::Scalar(0, 0, 255), 1);
        // cv::drawContours(dis_img, contours, i, CV_RGB(255, 255, 0));
      }
      if (contours.size() <= 0) continue;
      std::string cur_ipl = std::filesystem::path(second[m]).stem().string();

      cv::imwrite(R"(E:\demo\test\test_opencv\ret\)" + cur_ipl + ".jpg",
                  dis_img);
    }
  }
}

#include <nlohmann/json.hpp>

#include "../include/tival/include/FindLine.h"
#include "../include/tival/include/ShapeBasedMatching.h"

void test_match() {
  cv::Mat template_img = cv::imread(R"(F:\data\match_img\20240105-110645.bmp)");
  if (template_img.channels() > 1) {
    cv::cvtColor(template_img, template_img, cv::COLOR_RGB2GRAY);
  }
  cv::Mat th_img;
  cv::threshold(template_img, th_img, 30, 255, cv::THRESH_BINARY);

  std::vector<std::vector<cv::Point>> contours;
  std::vector<cv::Vec4i> hierarchy;
  cv::findContours(th_img, contours, hierarchy, cv::RETR_EXTERNAL,
                   cv::CHAIN_APPROX_NONE);

  // 获取roi
  cv::Mat roi_img;
  cv::Mat rotate_img;

  for (size_t i = 0; i < contours.size(); ++i) {
    cv::Rect rect = cv::boundingRect(contours[i]);
    cv::RotatedRect ro_rect = cv::minAreaRect(contours[i]);

    double angle = ro_rect.angle;
    cv::Point center(ro_rect.center);

    cv::Mat m = cv::getRotationMatrix2D(center, angle, 1);
    cv::warpAffine(template_img, rotate_img, m, template_img.size(),
                   cv::INTER_LINEAR + cv::WARP_FILL_OUTLIERS);
    double area = contourArea(contours[i]);
    if (area > 100) {
      roi_img = template_img(rect).clone();
      break;
    }
  }

  {
    nlohmann::json find_line_params = {
        {"CaliperNum", 10},
        {"CaliperLength", 20},
        {"CaliperWidth", 5},
    };
    cv::Mat ro_th_img;
    cv::Mat ro_roi_img;
    cv::threshold(rotate_img, ro_th_img, 30, 255, cv::THRESH_BINARY);

    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(ro_th_img, contours, hierarchy, cv::RETR_EXTERNAL,
                     cv::CHAIN_APPROX_NONE);

    for (size_t i = 0; i < contours.size(); ++i) {
      cv::Rect rect = cv::boundingRect(contours[i]);

      double area = contourArea(contours[i]);
      rect.x = rect.x - 20;
      rect.width = rect.width + 40;
      if (area > 100) {
        ro_roi_img = ro_th_img(rect).clone();
        break;
      }
    }

    Tival::TPoint start(20, 0), end(21, 119);
    cv::Mat line_img = ro_roi_img(cv::Rect(0, 40, 35, 120));
    Tival::FindLineResult find_line_ret =
        Tival::FindLine::Run(line_img, start, end, find_line_params);
  }

  nlohmann::json params = {
      {"NumLevels", 2},
      {"AngleMin", 0},
      {"AngleMax", 360},
  };
  Tival::ShapeBasedMatching sbm;
  try {
    sbm.Create(roi_img, params);
    sbm.Save("E:/demo/test/test_opencv/template/template_1.mdl");
  } catch (const std::exception &e) {
    std::cout << "error" << e.what() << std::endl;
  }
  // sbm.Load("../template/template_1.mdl");
  cv::Mat test_imh = cv::imread(R"(F:\data\match_img\20240105-110704.bmp)");
  if (test_imh.channels() > 1) {
    cv::cvtColor(test_imh, test_imh, cv::COLOR_RGB2GRAY);
  }
  nlohmann::json find_params = {{"AngleMin", 0}, {"AngleMax", 360}};
  Tival::SbmResults ret = sbm.Find(test_imh, find_params);
}

void test_angle() {
  cv::Mat template_img = cv::imread(R"(F:\data\match_img\20240105-110645.bmp)");
  if (template_img.channels() > 1) {
    cv::cvtColor(template_img, template_img, cv::COLOR_RGB2GRAY);
  }
  cv::Mat th_img;
  cv::threshold(template_img, th_img, 30, 255, cv::THRESH_BINARY);

  std::vector<std::vector<cv::Point>> contours;
  std::vector<cv::Vec4i> hierarchy;
  cv::findContours(th_img, contours, hierarchy, cv::RETR_EXTERNAL,
                   cv::CHAIN_APPROX_NONE);

  // 获取roi
  cv::Mat roi_img;
  cv::Mat rotate_img;

  for (size_t i = 0; i < contours.size(); ++i) {
    double area = contourArea(contours[i]);
    if (area < 100) continue;
    cv::Rect rect = cv::boundingRect(contours[i]);
    cv::RotatedRect ro_rect = cv::minAreaRect(contours[i]);

    double angle = ro_rect.angle;
    cv::Point center(ro_rect.center);

    cv::Mat m = cv::getRotationMatrix2D(center, angle, 1);
    cv::warpAffine(template_img, rotate_img, m, template_img.size(),
                   cv::INTER_LINEAR + cv::WARP_FILL_OUTLIERS);

    if (area > 100) {
      roi_img = template_img(rect).clone();
      break;
    }
  }
}

void test_Affine() {
  float i2d[6];
  float d2i[6];

  const cv::Size from{600, 1200};
  const cv::Size to{300, 600};

  cv::Mat src = cv::Mat(from, CV_8UC1, cv::Scalar(255));

  float scale_x = to.width / (float)from.width;
  float scale_y = to.height / (float)from.height;
  float scale = (std::min)(scale_x, scale_y);

  cv::Mat m = (cv::Mat_<float>(2, 3) << scale, 0, 0, 0, scale,
               0);  //= cv::getRotationMatrix2D(cv::Point((float)from.width / 2,
                    //(float)from.height), 0, scale);

  cv::Mat M = cv::getRotationMatrix2D(
      cv::Point((float)from.width / 2, (float)from.height), 0, scale);
  i2d[0] = scale;
  i2d[1] = 0;
  i2d[2] = -scale * from.width * 0.5 + to.width * 0.5 + scale * 0.5 - 0.5;
  i2d[3] = 0;
  i2d[4] = scale;
  i2d[5] = -scale * from.height * 0.5 + to.height * 0.5 + scale * 0.5 - 0.5;

  cv::Mat ret;
  cv::warpAffine(src, ret, m, to, cv::INTER_LINEAR + cv::WARP_FILL_OUTLIERS);

  cv::Mat m2x3_i2d(2, 3, CV_32F, i2d);
  cv::Mat m2x3_d2i(2, 3, CV_32F, d2i);
  cv::invertAffineTransform(m2x3_i2d, m2x3_d2i);
}

void area(cv::Point2d p1, cv::Point2d p2, cv::Point2d p3) {
  double s = std::abs((p1.x * p2.y + p2.x * p3.y + p3.x * p1.y - p1.x * p3.y -
                       p2.x * p1.y - p3.x * p2.y) /
                      2);
  printf_s("A = %.6f\n", s);
}
void test_area() {
  std::vector<cv::Point2d> pts;
  pts.emplace_back(0, 4079);
  pts.emplace_back(351, 4079);
  pts.emplace_back(0, 3125);
  pts.emplace_back(306.509827, 4028.913330);

  area(pts[0], pts[1], pts[2]);
  area(pts[0], pts[1], pts[3]);
  area(pts[0], pts[2], pts[3]);
  area(pts[3], pts[1], pts[2]);

  cv::Mat m1 = (cv::Mat_<double>(3, 3) << 0.000000, 4079.000000, 1, 351.000000,
                4079.000000, 1, 0.000000, 3125.000000, 1);
  cv::Mat m2 = (cv::Mat_<double>(3, 3) << 0.000000, 4079.000000, 1, 351.000000,
                4079.000000, 1, 306.509827, 4028.913330, 1);
  cv::Mat m3 = (cv::Mat_<double>(3, 3) << 0.000000, 4079.000000, 1, 0.000000,
                3125.000000, 1, 306.509827, 4028.913330, 1);
  cv::Mat m4 = (cv::Mat_<double>(3, 3) << 351.000000, 4079.000000, 1, 0.000000,
                3125.000000, 1, 306.509827, 4028.913330, 1);

  double s1 = 0.5 * std::abs(cv::determinant(m1));
  double s2 = 0.5 * std::abs(cv::determinant(m2));
  double s3 = 0.5 * std::abs(cv::determinant(m3));
  double s4 = 0.5 * std::abs(cv::determinant(m4));

  printf_s("A = %.6f\n", s1);
  printf_s("B = %.6f\n", s2);
  printf_s("C = %.6f\n", s3);
  printf_s("D = %.6f\n", s4);

  std::vector<cv::Point2f> v1, v2, v3, v4;

  v1.emplace_back(pts[0]);
  v1.emplace_back(pts[1]);
  v1.emplace_back(pts[2]);

  v2.emplace_back(pts[0]);
  v2.emplace_back(pts[1]);
  v2.emplace_back(pts[3]);

  v3.emplace_back(pts[0]);
  v3.emplace_back(pts[2]);
  v3.emplace_back(pts[3]);

  v4.emplace_back(pts[1]);
  v4.emplace_back(pts[2]);
  v4.emplace_back(pts[3]);

  double s5 = cv::contourArea(v1);
  double s6 = cv::contourArea(v2);
  double s7 = cv::contourArea(v3);
  double s8 = cv::contourArea(v4);

  printf_s("A = %.6f\n", s5);
  printf_s("B = %.6f\n", s6);
  printf_s("C = %.6f\n", s7);
  printf_s("D = %.6f\n", s8);
}

#include <opencv2/core.hpp>
#include <opencv2/flann/flann.hpp>
void test_flann() {
  // 定义特征向量集合
  cv::Mat features = (cv::Mat_<float>(5, 2) << 1, 1, 2, 2, 3, 3, 4, 4, 5, 5);

  /* cv::flann::IndexParams param;

   param.setAlgorithm();*/
  // 创建FLANN索引
  cv::flann::Index flannIndex(features, cv::flann::KDTreeIndexParams(2),
                              cvflann::FLANN_DIST_L2);

  // 定义查询点
  cv::Mat query = (cv::Mat_<float>(1, 2) << 3.1, 3.1);

  // 进行最近邻搜索
  cv::Mat indices, dists;
  flannIndex.knnSearch(query, indices, dists, 2, cv::flann::SearchParams(32));

  // 输出最近邻点的索引和距离
  std::cout << "最近邻点的索引：" << indices.at<int>(0, 0) << std::endl;
  // 距离需要开方
  std::cout << "最近邻点的距离：" << dists.at<float>(0, 0) << std::endl;
}

#include "opencv2/xfeatures2d.hpp"
void test_flannmatch() {
  // cv::Mat srcImage = cv::imread(R"(E:\demo\test\test_opencv\img\2.png)");
  // cv::Mat dstImage = cv::imread(R"(E:\demo\test\test_opencv\img\1.png)");

  //// surf 特征提取
  // int minHessian = 450;
  // cv::Ptr<cv::xfeatures2d::SURF> detector =
  // cv::xfeatures2d::SURF::create(minHessian); std::vector<cv::KeyPoint>
  // keypoints_src; std::vector<cv::KeyPoint> keypoints_dst; cv::Mat
  // descriptor_src, descriptor_dst; detector->detectAndCompute(srcImage,
  // cv::Mat(), keypoints_src, descriptor_src);
  // detector->detectAndCompute(dstImage, cv::Mat(), keypoints_dst,
  // descriptor_dst);

  cv::Mat descriptor_src =
      (cv::Mat_<float>(5, 2) << 1, 1, 2, 2, 3, 3, 4, 4, 5, 5);
  cv::Mat descriptor_dst = (cv::Mat_<float>(1, 2) << 3.5, 3.5);

  // matching
  cv::FlannBasedMatcher matcher;
  std::vector<cv::DMatch> matches;
  matcher.match(descriptor_dst, descriptor_src, matches);

  // find good matched points
  double minDist = 0, maxDist = 0;
  for (size_t i = 0; i < matches.size(); i++) {
    double dist = matches[i].distance;
    if (dist > maxDist) maxDist = dist;
    if (dist < minDist) minDist = dist;
  }

  std::vector<cv::DMatch> goodMatches;
  for (size_t i = 0; i < matches.size(); i++) {
    double dist = matches[i].distance;
    if (dist < std::max(3 * minDist, 0.02)) {
      goodMatches.push_back(matches[i]);
    }
  }

  // cv::Mat matchesImage;
  // cv::drawMatches(dstImage, keypoints_dst, srcImage, keypoints_src,
  // goodMatches, matchesImage, cv::Scalar::all(-1),
  //     cv::Scalar::all(-1), std::vector<char>(),
  //     cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

  // cv::imshow("matchesImage", matchesImage);

  cv::waitKey(0);
  return;
}

void test_liscense() {
  cv::Mat sec = cv::imread("F:/data/20240125/Image_20240130120647523.png");
}

using namespace std;
using namespace cv;

static std::string prefix = "/home/meiqua/shape_based_matching/test/";

//class Timer {
// public:
//  Timer() : beg_(clock_::now()) {}
//  void reset() { beg_ = clock_::now(); }
//  double elapsed() const {
//    return std::chrono::duration_cast<second_>(clock_::now() - beg_).count();
//  }
//  void out(std::string message = "") {
//    double t = elapsed();
//    std::cout << message << "\nelasped time:" << t << "s" << std::endl;
//    reset();
//  }
//
// private:
//  typedef std::chrono::high_resolution_clock clock_;
//  typedef std::chrono::duration<double, std::ratio<1>> second_;
//  std::chrono::time_point<clock_> beg_;
//};
// NMS, got from cv::dnn so we don't need opencv contrib
// just collapse it
namespace cv_dnn {
namespace {

template <typename T>
static inline bool SortScorePairDescend(const std::pair<float, T> &pair1,
                                        const std::pair<float, T> &pair2) {
  return pair1.first > pair2.first;
}

}  // namespace

inline void GetMaxScoreIndex(
    const std::vector<float> &scores, const float threshold, const int top_k,
    std::vector<std::pair<float, int>> &score_index_vec) {
  for (size_t i = 0; i < scores.size(); ++i) {
    if (scores[i] > threshold) {
      score_index_vec.push_back(std::make_pair(scores[i], i));
    }
  }
  std::stable_sort(score_index_vec.begin(), score_index_vec.end(),
                   SortScorePairDescend<int>);
  if (top_k > 0 && top_k < (int)score_index_vec.size()) {
    score_index_vec.resize(top_k);
  }
}

template <typename BoxType>
inline void NMSFast_(
    const std::vector<BoxType> &bboxes, const std::vector<float> &scores,
    const float score_threshold, const float nms_threshold, const float eta,
    const int top_k, std::vector<int> &indices,
    float (*computeOverlap)(const BoxType &, const BoxType &)) {
  CV_Assert(bboxes.size() == scores.size());
  std::vector<std::pair<float, int>> score_index_vec;
  GetMaxScoreIndex(scores, score_threshold, top_k, score_index_vec);

  // Do nms.
  float adaptive_threshold = nms_threshold;
  indices.clear();
  for (size_t i = 0; i < score_index_vec.size(); ++i) {
    const int idx = score_index_vec[i].second;
    bool keep = true;
    for (int k = 0; k < (int)indices.size() && keep; ++k) {
      const int kept_idx = indices[k];
      float overlap = computeOverlap(bboxes[idx], bboxes[kept_idx]);
      keep = overlap <= adaptive_threshold;
    }
    if (keep) indices.push_back(idx);
    if (keep && eta < 1 && adaptive_threshold > 0.5) {
      adaptive_threshold *= eta;
    }
  }
}

// copied from opencv 3.4, not exist in 3.0
template <typename _Tp>
static inline double jaccardDistance__(const Rect_<_Tp> &a,
                                       const Rect_<_Tp> &b) {
  _Tp Aa = a.area();
  _Tp Ab = b.area();

  if ((Aa + Ab) <= std::numeric_limits<_Tp>::epsilon()) {
    // jaccard_index = 1 -> distance = 0
    return 0.0;
  }

  double Aab = (a & b).area();
  // distance = 1 - jaccard_index
  return 1.0 - Aab / (Aa + Ab - Aab);
}

template <typename T>
static inline float rectOverlap(const T &a, const T &b) {
  return 1.f - static_cast<float>(jaccardDistance__(a, b));
}

void NMSBoxes(const std::vector<Rect> &bboxes, const std::vector<float> &scores,
              const float score_threshold, const float nms_threshold,
              std::vector<int> &indices, const float eta = 1,
              const int top_k = 0) {
  NMSFast_(bboxes, scores, score_threshold, nms_threshold, eta, top_k, indices,
           rectOverlap);
}

}  // namespace cv_dnn

void scale_test(string mode) {
  int num_feature = 150;

  // feature numbers(how many ori in one templates?)
  // two pyramids, lower pyramid(more pixels) in stride 4, lower in stride 8
  line2Dup::Detector detector(num_feature, {4, 8});

  //    mode = "test";
  if (mode == "train") {
    Mat img = cv::imread(prefix + "case0/templ/circle.png");
    assert(!img.empty() && "check your img path");
    shape_based_matching::shapeInfo_producer shapes(img);

    shapes.scale_range = {0.1f, 1};
    shapes.scale_step = 0.01f;
    shapes.produce_infos();

    std::vector<shape_based_matching::shapeInfo_producer::Info>
        infos_have_templ;
    string class_id = "circle";
    for (auto &info : shapes.infos) {
      // template img, id, mask,
      // feature numbers(missing it means using the detector initial num)
      int templ_id = detector.addTemplate(shapes.src_of(info), class_id,
                                          shapes.mask_of(info),
                                          int(num_feature * info.scale));
      std::cout << "templ_id: " << templ_id << std::endl;

      // may fail when asking for too many feature_nums for small training img
      if (templ_id !=
          -1) {  // only record info when we successfully add template
        infos_have_templ.push_back(info);
      }
    }

    // save templates
    detector.writeClasses(prefix + "case0/%s_templ.yaml");

    // save infos,
    // in this simple case infos are not used
    shapes.save_infos(infos_have_templ, prefix + "case0/circle_info.yaml");
    std::cout << "train end" << std::endl << std::endl;

  } else if (mode == "test") {
    std::vector<std::string> ids;

    // read templates
    ids.push_back("circle");
    detector.readClasses(ids, prefix + "case0/%s_templ.yaml");

    Mat test_img = imread(prefix + "case0/1.jpg");
    assert(!test_img.empty() && "check your img path");

    // make the img having 32*n width & height
    // at least 16*n here for two pyrimads with strides 4 8
    int stride = 32;
    int n = test_img.rows / stride;
    int m = test_img.cols / stride;
    Rect roi(0, 0, stride * m, stride * n);
    Mat img = test_img(roi).clone();
    assert(img.isContinuous());

    Timer timer;
    // match, img, min socre, ids
    auto matches = detector.match(img, 90, ids);
    // one output match:
    // x: top left x
    // y: top left y
    // template_id: used to find templates
    // similarity: scores, 100 is best
    timer.out();

    std::cout << "matches.size(): " << matches.size() << std::endl;
    size_t top5 = 5;
    if (top5 > matches.size()) top5 = matches.size();
    for (size_t i = 0; i < top5; i++) {
      auto match = matches[i];
      auto templ = detector.getTemplates("circle", match.template_id);
      // template:
      // nums: num_pyramids * num_modality (modality, depth or RGB, always 1
      // here) template[0]: lowest pyrimad(more pixels) template[0].width:
      // actual width of the matched template template[0].tl_x / tl_y: topleft
      // corner when cropping templ during training In this case, we can regard
      // width/2 = radius
      int x = templ[0].width / 2 + match.x;
      int y = templ[0].height / 2 + match.y;
      int r = templ[0].width / 2;
      Scalar color(255, rand() % 255, rand() % 255);

      cv::putText(img, to_string(int(round(match.similarity))),
                  Point(match.x + r - 10, match.y - 3), FONT_HERSHEY_PLAIN, 2,
                  color);
      cv::circle(img, {x, y}, r, color, 2);
    }

    imshow("img", img);
    waitKey(0);

    std::cout << "test end" << std::endl << std::endl;
  }
}

void angle_test(string mode, bool use_rot) {
  line2Dup::Detector detector(128, {4, 8});

  if (mode != "test") {
    Mat img = imread(prefix + "case1/train.png");
    assert(!img.empty() && "check your img path");

    Rect roi(130, 110, 270, 270);
    img = img(roi).clone();
    Mat mask = Mat(img.size(), CV_8UC1, {255});

    // padding to avoid rotating out
    int padding = 100;
    cv::Mat padded_img = cv::Mat(img.rows + 2 * padding, img.cols + 2 * padding,
                                 img.type(), cv::Scalar::all(0));
    img.copyTo(padded_img(Rect(padding, padding, img.cols, img.rows)));

    cv::Mat padded_mask =
        cv::Mat(mask.rows + 2 * padding, mask.cols + 2 * padding, mask.type(),
                cv::Scalar::all(0));
    mask.copyTo(padded_mask(Rect(padding, padding, img.cols, img.rows)));

    shape_based_matching::shapeInfo_producer shapes(padded_img, padded_mask);
    shapes.angle_range = {0, 360};
    shapes.angle_step = 1;

    shapes.scale_range = {1};  // support just one
    shapes.produce_infos();
    std::vector<shape_based_matching::shapeInfo_producer::Info>
        infos_have_templ;
    string class_id = "test";

    bool is_first = true;

    // for other scales you want to re-extract points:
    // set shapes.scale_range then produce_infos; set is_first = false;

    int first_id = 0;
    float first_angle = 0;
    for (auto &info : shapes.infos) {
      Mat to_show = shapes.src_of(info);

      std::cout << "\ninfo.angle: " << info.angle << std::endl;
      int templ_id;

      if (is_first) {
        templ_id = detector.addTemplate(shapes.src_of(info), class_id,
                                        shapes.mask_of(info));
        first_id = templ_id;
        first_angle = info.angle;

        if (use_rot) is_first = false;
      } else {
        templ_id = detector.addTemplate_rotate(
            class_id, first_id, info.angle - first_angle,
            {shapes.src.cols / 2.0f, shapes.src.rows / 2.0f});
      }

      auto templ = detector.getTemplates("test", templ_id);
      for (int i = 0; i < templ[0].features.size(); i++) {
        auto feat = templ[0].features[i];
        cv::circle(to_show, {feat.x + templ[0].tl_x, feat.y + templ[0].tl_y}, 3,
                   {0, 0, 255}, -1);
      }

      // will be faster if not showing this
      imshow("train", to_show);
      waitKey(1);

      std::cout << "templ_id: " << templ_id << std::endl;
      if (templ_id != -1) {
        infos_have_templ.push_back(info);
      }
    }
    detector.writeClasses(prefix + "case1/%s_templ.yaml");
    shapes.save_infos(infos_have_templ, prefix + "case1/test_info.yaml");
    std::cout << "train end" << std::endl << std::endl;
  } else if (mode == "test") {
    std::vector<std::string> ids;
    ids.push_back("test");
    detector.readClasses(ids, prefix + "case1/%s_templ.yaml");

    // angle & scale are saved here, fetched by match id
    auto infos = shape_based_matching::shapeInfo_producer::load_infos(
        prefix + "case1/test_info.yaml");

    Mat test_img = imread(prefix + "case1/test.png");
    assert(!test_img.empty() && "check your img path");

    int padding = 250;
    cv::Mat padded_img =
        cv::Mat(test_img.rows + 2 * padding, test_img.cols + 2 * padding,
                test_img.type(), cv::Scalar::all(0));
    test_img.copyTo(
        padded_img(Rect(padding, padding, test_img.cols, test_img.rows)));

    int stride = 16;
    int n = padded_img.rows / stride;
    int m = padded_img.cols / stride;
    Rect roi(0, 0, stride * m, stride * n);
    Mat img = padded_img(roi).clone();
    assert(img.isContinuous());

    //        cvtColor(img, img, CV_BGR2GRAY);

    std::cout << "test img size: " << img.rows * img.cols << std::endl
              << std::endl;

    Timer timer;
    auto matches = detector.match(img, 90, ids);
    timer.out();

    if (img.channels() == 1) cvtColor(img, img, CV_GRAY2BGR);

    std::cout << "matches.size(): " << matches.size() << std::endl;
    size_t top5 = 1;
    if (top5 > matches.size()) top5 = matches.size();
    for (size_t i = 0; i < top5; i++) {
      auto match = matches[i];
      auto templ = detector.getTemplates("test", match.template_id);

      // 270 is width of template image
      // 100 is padding when training
      // tl_x/y: template croping topleft corner when training

      float r_scaled = 270 / 2.0f * infos[match.template_id].scale;

      // scaling won't affect this, because it has been determined by warpAffine
      // cv::warpAffine(src, dst, rot_mat, src.size()); last param
      float train_img_half_width = 270 / 2.0f + 100;
      float train_img_half_height = 270 / 2.0f + 100;

      // center x,y of train_img in test img
      float x = match.x - templ[0].tl_x + train_img_half_width;
      float y = match.y - templ[0].tl_y + train_img_half_height;

      cv::Vec3b randColor;
      randColor[0] = rand() % 155 + 100;
      randColor[1] = rand() % 155 + 100;
      randColor[2] = rand() % 155 + 100;
      for (int i = 0; i < templ[0].features.size(); i++) {
        auto feat = templ[0].features[i];
        cv::circle(img, {feat.x + match.x, feat.y + match.y}, 3, randColor, -1);
      }

      cv::putText(img, to_string(int(round(match.similarity))),
                  Point(match.x + r_scaled - 10, match.y - 3),
                  FONT_HERSHEY_PLAIN, 2, randColor);

      cv::RotatedRect rotatedRectangle({x, y}, {2 * r_scaled, 2 * r_scaled},
                                       -infos[match.template_id].angle);

      cv::Point2f vertices[4];
      rotatedRectangle.points(vertices);
      for (int i = 0; i < 4; i++) {
        int next = (i + 1 == 4) ? 0 : (i + 1);
        cv::line(img, vertices[i], vertices[next], randColor, 2);
      }

      std::cout << "\nmatch.template_id: " << match.template_id << std::endl;
      std::cout << "match.similarity: " << match.similarity << std::endl;
    }

    imshow("img", img);
    waitKey(0);

    std::cout << "test end" << std::endl << std::endl;
  }
}

void noise_test(string mode) {
  line2Dup::Detector detector(30, {4, 8});

  if (mode == "train") {
    Mat img = imread(prefix + "case2/train.png");
    assert(!img.empty() && "check your img path");
    Mat mask = Mat(img.size(), CV_8UC1, {255});

    shape_based_matching::shapeInfo_producer shapes(img, mask);
    shapes.angle_range = {0, 360};
    shapes.angle_step = 1;
    shapes.produce_infos();
    std::vector<shape_based_matching::shapeInfo_producer::Info>
        infos_have_templ;
    string class_id = "test";
    for (auto &info : shapes.infos) {
      imshow("train", shapes.src_of(info));
      waitKey(1);

      std::cout << "\ninfo.angle: " << info.angle << std::endl;
      int templ_id = detector.addTemplate(shapes.src_of(info), class_id,
                                          shapes.mask_of(info));
      std::cout << "templ_id: " << templ_id << std::endl;
      if (templ_id != -1) {
        infos_have_templ.push_back(info);
      }
    }
    detector.writeClasses(prefix + "case2/%s_templ.yaml");
    shapes.save_infos(infos_have_templ, prefix + "case2/test_info.yaml");
    std::cout << "train end" << std::endl << std::endl;
  } else if (mode == "test") {
    std::vector<std::string> ids;
    ids.push_back("test");
    detector.readClasses(ids, prefix + "case2/%s_templ.yaml");

    Mat test_img = imread(prefix + "case2/test.png");
    assert(!test_img.empty() && "check your img path");

    // cvtColor(test_img, test_img, CV_BGR2GRAY);

    int stride = 16;
    int n = test_img.rows / stride;
    int m = test_img.cols / stride;
    Rect roi(0, 0, stride * m, stride * n);

    test_img = test_img(roi).clone();

    Timer timer;
    auto matches = detector.match(test_img, 90, ids);
    timer.out();

    std::cout << "matches.size(): " << matches.size() << std::endl;
    size_t top5 = 500;
    if (top5 > matches.size()) top5 = matches.size();

    vector<Rect> boxes;
    vector<float> scores;
    vector<int> idxs;
    for (auto match : matches) {
      Rect box;
      box.x = match.x;
      box.y = match.y;

      auto templ = detector.getTemplates("test", match.template_id);

      box.width = templ[0].width;
      box.height = templ[0].height;
      boxes.push_back(box);
      scores.push_back(match.similarity);
    }
    cv_dnn::NMSBoxes(boxes, scores, 0, 0.5f, idxs);

    for (auto idx : idxs) {
      auto match = matches[idx];
      auto templ = detector.getTemplates("test", match.template_id);

      int x = templ[0].width + match.x;
      int y = templ[0].height + match.y;
      int r = templ[0].width / 2;
      cv::Vec3b randColor;
      randColor[0] = rand() % 155 + 100;
      randColor[1] = rand() % 155 + 100;
      randColor[2] = rand() % 155 + 100;

      for (int i = 0; i < templ[0].features.size(); i++) {
        auto feat = templ[0].features[i];
        cv::circle(test_img, {feat.x + match.x, feat.y + match.y}, 2, randColor,
                   -1);
      }

      cv::putText(test_img, to_string(int(round(match.similarity))),
                  Point(match.x + r - 10, match.y - 3), FONT_HERSHEY_PLAIN, 2,
                  randColor);
      cv::rectangle(test_img, {match.x, match.y}, {x, y}, randColor, 2);

      std::cout << "\nmatch.template_id: " << match.template_id << std::endl;
      std::cout << "match.similarity: " << match.similarity << std::endl;
    }

    imshow("img", test_img);
    waitKey(0);

    std::cout << "test end" << std::endl << std::endl;
  }
}

void MIPP_test() {
  std::cout << "MIPP tests" << std::endl;
  std::cout << "----------" << std::endl << std::endl;

  std::cout << "Instr. type:       " << mipp::InstructionType << std::endl;
  std::cout << "Instr. full type:  " << mipp::InstructionFullType << std::endl;
  std::cout << "Instr. version:    " << mipp::InstructionVersion << std::endl;
  std::cout << "Instr. size:       " << mipp::RegisterSizeBit << " bits"
            << std::endl;
  std::cout << "Instr. lanes:      " << mipp::Lanes << std::endl;
  std::cout << "64-bit support:    " << (mipp::Support64Bit ? "yes" : "no")
            << std::endl;
  std::cout << "Byte/word support: " << (mipp::SupportByteWord ? "yes" : "no")
            << std::endl;

#ifndef has_max_int8_t
  std::cout << "in this SIMD, int8 max is not inplemented by MIPP" << std::endl;
#endif

#ifndef has_shuff_int8_t
  std::cout << "in this SIMD, int8 shuff is not inplemented by MIPP"
            << std::endl;
#endif

  std::cout << "----------" << std::endl << std::endl;
}

// 不支持 旋转的同时进行进行缩放，只能是先旋转，后缩放
void shape_train(std::string img_path, std::string model_name,
                 std::string model_path) {
  int num_feature = 300;
  line2Dup::Detector detector(num_feature, {4, 8});
  cv::Mat img = cv::imread(img_path);
  cv::Mat th_img = img.clone();
  // if (img.channels()>1) {
  //     cv::cvtColor(img,th_img,cv::COLOR_BGR2GRAY);
  //     cv::threshold(th_img,th_img,155,255,cv::ThresholdTypes::THRESH_BINARY_INV);
  // }
  assert(!img.empty() && "check your img path");

  shape_based_matching::shapeInfo_producer shapes(th_img);

  shapes.scale_range = {0.2f, 3.f};
  shapes.scale_step = 0.03f;
  shapes.produce_infos();

  std::vector<shape_based_matching::shapeInfo_producer::Info> infos_have_templ;
  std::string class_id = model_name;

  for (auto &info : shapes.infos) {
    cv::Mat to_show = shapes.src_of(info);
    int templ_id = detector.addTemplate(shapes.src_of(info), class_id,
                                        shapes.mask_of(info),
                                        int(num_feature * info.scale));
    std::cout << "templ_id: " << templ_id << std::endl;
    if (templ_id != -1) {
      infos_have_templ.push_back(info);

      auto templ = detector.getTemplates(model_name, templ_id);
      for (int i = 0; i < templ[0].features.size(); i++) {
        auto feat = templ[0].features[i];
        cv::circle(to_show, {feat.x + templ[0].tl_x, feat.y + templ[0].tl_y}, 3,
                   {0, 0, 255}, -1);
      }
    }
  }

  detector.writeClasses(model_path);
}

void shape_test(std::string img_paht, std::string model_name,
                std::string model_path, cv::Rect img_rect) {
  int num_feature = 300;

  std::vector<std::string> ids;
  line2Dup::Detector detector(num_feature, {4, 8});
  // read templates
  ids.push_back(model_name);
  detector.readClasses(ids, model_path);

  cv::Mat test_img = cv::imread(img_paht);
  assert(!test_img.empty() && "check your img path");

  test_img = test_img(img_rect).clone();

  cv::Mat th_img = test_img.clone();
  // if (test_img.channels() > 1) {
  //     cv::cvtColor(test_img, th_img, cv::COLOR_BGR2GRAY);
  //     //cv::threshold(th_img, th_img, 155, 255,
  //     cv::ThresholdTypes::THRESH_BINARY_INV);
  //     cv::resize(th_img,th_img,cv::Size(test_img.cols *3, test_img.rows *3));
  // }

  // make the img having 32*n width & height
  // at least 16*n here for two pyrimads with strides 4 8
  int stride = 32;
  int n = th_img.rows / stride;
  int m = th_img.cols / stride;
  cv::Rect roi(0, 0, stride * m, stride * n);
  cv::Mat img = th_img(roi).clone();
  assert(img.isContinuous());

  Timer timer;
  // match, img, min socre, ids
  auto matches = detector.match(img, 85, ids);
  // one output match:
  // x: top left x
  // y: top left y
  // template_id: used to find templates
  // similarity: scores, 100 is best
  timer.out();
  std::cout << "matches.size(): " << matches.size() << std::endl;
  size_t top5 = 5;
  if (top5 > matches.size()) top5 = matches.size();
  for (size_t i = 0; i < top5; i++) {
    auto match = matches[i];
    auto templ = detector.getTemplates(model_name, match.template_id);
    // template:
    // nums: num_pyramids * num_modality (modality, depth or RGB, always 1 here)
    // template[0]: lowest pyrimad(more pixels)
    // template[0].width: actual width of the matched template
    // template[0].tl_x / tl_y: topleft corner when cropping templ during
    // training In this case, we can regard width/2 = radius
    int x = templ[0].width / 2 + match.x;
    int y = templ[0].height / 2 + match.y;
    int r = templ[0].width / 2;
    cv::Scalar color(255, rand() % 255, rand() % 255);

    cv::putText(img, to_string(int(round(match.similarity))),
                Point(match.x + r - 10, match.y - 3), cv::FONT_HERSHEY_PLAIN, 2,
                color);
    cv::circle(img, {x, y}, r, color, 2);
  }
  std::cout << "test end" << std::endl << std::endl;
}

cv::Mat get_edge(const cv::Mat &src) {
  cv::Mat soble_img_x, sobel_img_y, edge, edge_th;
  cv::Sobel(src, sobel_img_y, CV_16S, 0, 1, -1);
  cv::Sobel(src, soble_img_x, CV_16S, 1, 0, -1);
  edge = soble_img_x + sobel_img_y;
  cv::convertScaleAbs(edge, edge);
  return edge;
}

void erode_test(const std::string &input_path) {
  cv::Mat in_img = cv::imread(input_path, 0);

  ED testED = ED(in_img, LSD_OPERATOR, 36, 8, 1, 8, 1.0, true);
  EDLines testEDLines = EDLines(testED);
  std::vector<LS> lines = testEDLines.getLines();

  std::vector<LS> h_ls;
  for (int i = 0; i < lines.size(); i++) {
    LS c_l = lines[i];
    double dis = cv::norm(c_l.start - c_l.end);
    if (dis < 300) {
      continue;
    }
    h_ls.push_back(c_l);
  }
  std::sort(h_ls.begin(), h_ls.end(), [&](const LS &lhs, const LS &rhs) {
    cv::Point2f l_c((lhs.start.x + lhs.end.x) / 2,
                    (lhs.start.y + lhs.end.y) / 2);
    cv::Point2f r_c((rhs.start.x + rhs.end.x) / 2,
                    (rhs.start.y + rhs.end.y) / 2);
    if (l_c.y < r_c.y) {
      return true;
    } else {
      return false;
    }
  });
}

void test_select(const std::string &input_path) {
  cv::Mat in_img = cv::imread(input_path, 0);

  std::vector<std::vector<cv::Point>> contours, out_contours;
  std::vector<cv::Vec4i> hierarchy;
  cv::findContours(in_img, contours, hierarchy, cv::RETR_EXTERNAL,
                   cv::CHAIN_APPROX_NONE);

  std::function func = [&](const cv::Rect &rect,
                           const cv::RotatedRect &rotated_rect,
                           const std::vector<cv::Point> &cur_contours) {
    std::cout << rect.width << rect.height << std::endl;
    if (rect.height > 1000) {
      out_contours.push_back(contours[0]);
    }
  };

  cv::Mat out_img;
  select_shape(in_img, out_img, contours, out_contours, func);
}

void detectEdgesWithGaps(const cv::Mat &inputImage, cv::Mat &outputImage,
                         const cv::Mat &lineFet) {
  cv::Mat gray, binaryImage;
  cv::cvtColor(inputImage, gray, cv::COLOR_BGR2GRAY);
  cv::threshold(gray, binaryImage, 30, 255, cv::THRESH_BINARY);

  std::vector<std::vector<cv::Point>> contours;
  std::vector<cv::Point> hullcontours;
  cv::findContours(binaryImage, contours, cv::RETR_EXTERNAL,
                   cv::CHAIN_APPROX_SIMPLE);

  outputImage = cv::Mat::zeros(inputImage.size(), CV_8UC1);

  for (size_t i = 0; i < contours.size(); ++i) {
    std::vector<cv::Point> hull;
    cv::convexHull(contours[i], hull);

    std::vector<std::vector<cv::Point>> hulls(1, hull);

    cv::drawContours(outputImage, hulls, 0, cv::Scalar(255, 0, 0), cv::FILLED);
    cv::drawContours(outputImage, contours, static_cast<int>(i),
                     cv::Scalar(0, 255, 0), cv::FILLED);
  }
  cv::erode(outputImage, outputImage,
            cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3)));

  return;
}

void removeShadows(cv::Mat img, cv::Mat &calcMat) {
  cv::Mat gray;
  cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);

  cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
  int iteration = 9;

  cv::Mat dilateMat;
  cv::morphologyEx(gray, dilateMat, cv::MORPH_DILATE, element, cv::Point(-1, -1),
               iteration);

  cv::Mat erodeMat;
  cv::morphologyEx(dilateMat, erodeMat, cv::MORPH_ERODE, element,
                   cv::Point(-1, -1), iteration);

  calcMat = ~(erodeMat - gray);
  cv::Mat removeShadowMat;
  cv::normalize(calcMat, removeShadowMat, 0, 200, cv::NORM_MINMAX);
}

bool Estimation_X(cv::Mat matSrcBuf, cv::Mat &matDstBuf,
                              int nDimensionX, int nStepX, float fThBGOffset) {
  if (matSrcBuf.empty()) return false;

  if (matSrcBuf.channels() != 1) return false;

  if (!matDstBuf.empty()) matDstBuf.release();

  if (nStepX <= 0) return false;

  matDstBuf = cv::Mat::zeros(matSrcBuf.size(), matSrcBuf.type());

  int nStepCols = matSrcBuf.cols / nStepX;
  int nHalfCols = matSrcBuf.cols / 2;

  cv::Mat M = cv::Mat_<double>(nStepCols, nDimensionX + 1);
  cv::Mat I = cv::Mat_<double>(nStepCols, 1);
  cv::Mat q;

  double x, quad, dTemp;
  int i, j, k, m;

  cv::Scalar mean = cv::mean(matSrcBuf);
  int nMinGV = (int)(mean[0] * fThBGOffset);

  for (i = 0; i < matSrcBuf.rows; i++) {
    for (j = 0; j < nStepCols; j++)

    {
      x = (j * nStepX - nHalfCols) / double(matSrcBuf.cols);

      M.at<double>(j, 0) = 1.0;
      dTemp = 1.0;
      for (k = 1; k <= nDimensionX; k++) {
        dTemp *= x;
        M.at<double>(j, k) = dTemp;
      }

      // I.at<double>(j, 0) = matSrcBuf.at<uchar>(i, j*nStepX);
      m = matSrcBuf.at<uchar>(i, j * nStepX);
      I.at<double>(j, 0) = (m < nMinGV) ? nMinGV : m;
    }

    cv::SVD s(M);
    s.backSubst(I, q);

    for (j = 0; j < matDstBuf.cols; j++) {
      x = (j - nHalfCols) / double(matSrcBuf.cols);

      quad = q.at<double>(0, 0);
      dTemp = 1.0;
      for (k = 1; k <= nDimensionX; k++) {
        dTemp *= x;
        quad += (q.at<double>(k, 0) * dTemp);
      }

      matDstBuf.at<uchar>(i, j) = cv::saturate_cast<uchar>(quad);
    }
  }

  M.release();
  I.release();
  q.release();

  return true;
}

bool Estimation_Y(cv::Mat matSrcBuf, cv::Mat &matDstBuf,
                              int nDimensionY, int nStepY, float fThBGOffset) {
  if (matSrcBuf.empty()) return false;

  if (matSrcBuf.channels() != 1) return false;

  if (!matDstBuf.empty()) matDstBuf.release();

  if (nStepY <= 0) return false;

  matDstBuf = cv::Mat::zeros(matSrcBuf.size(), matSrcBuf.type());

  int nStepRows = matSrcBuf.rows / nStepY;
  int nHalfRows = matSrcBuf.rows / 2;

  cv::Mat M = cv::Mat_<double>(nStepRows, nDimensionY + 1);
  cv::Mat I = cv::Mat_<double>(nStepRows, 1);
  cv::Mat q;

  double y, quad, dTemp;
  int i, j, k, m;

  cv::Scalar mean = cv::mean(matSrcBuf);
  int nMinGV = (int)(mean[0] * fThBGOffset);

  for (j = 0; j < matSrcBuf.cols; j++) {
    for (i = 0; i < nStepRows; i++) {
      y = (i * nStepY - nHalfRows) / double(matSrcBuf.rows);

      M.at<double>(i, 0) = 1.0;
      dTemp = 1.0;
      for (k = 1; k <= nDimensionY; k++) {
        dTemp *= y;
        M.at<double>(i, k) = dTemp;
      }

      // I.at<double>(i, 0) = matSrcBuf.at<uchar>(i*nStepY, j);
      m = matSrcBuf.at<uchar>(i * nStepY, j);
      I.at<double>(i, 0) = (m < nMinGV) ? nMinGV : m;
    }

    cv::SVD s(M);
    s.backSubst(I, q);

    for (i = 0; i < matSrcBuf.rows; i++) {
      y = (i - nHalfRows) / double(matSrcBuf.rows);

      quad = q.at<double>(0, 0);
      dTemp = 1.0;
      for (k = 1; k <= nDimensionY; k++) {
        dTemp *= y;
        quad += (q.at<double>(k, 0) * dTemp);
      }

      matDstBuf.at<uchar>(i, j) = cv::saturate_cast<uchar>(quad);
    }
  }

  M.release();
  I.release();
  q.release();

  return true;
}

bool TwoImg_Average(cv::Mat matSrc1Buf, cv::Mat matSrc2Buf,
                                cv::Mat &matDstBuf) {
  if (matSrc1Buf.empty()) return false;
  if (matSrc2Buf.empty()) return false;

  if (matSrc1Buf.channels() != 1) return false;
  if (matSrc2Buf.channels() != 1) return false;

  if (matSrc1Buf.rows != matSrc2Buf.rows || matSrc1Buf.cols != matSrc2Buf.cols)
    return false;

  if (!matDstBuf.empty()) matDstBuf.release();

  matDstBuf =
      cv::Mat::zeros(matSrc1Buf.rows, matSrc1Buf.cols, matSrc1Buf.type());

  for (int y = 0; y < matSrc1Buf.rows; y++) {
    BYTE *ptr1 = (BYTE *)matSrc1Buf.ptr(y);
    BYTE *ptr2 = (BYTE *)matSrc2Buf.ptr(y);
    BYTE *ptr3 = (BYTE *)matDstBuf.ptr(y);

    for (int x = 0; x < matSrc1Buf.cols; x++, ptr1++, ptr2++, ptr3++) {
      *ptr3 = (BYTE)abs((*ptr1 + *ptr2) / 2.0);
    }
  }

  return true;
}

bool Make_HardDefect_Mask(cv::Mat &matGrayChanels,
                                      cv::Mat &defectMask, int nLineThreshold,
                                      int nStepX, int nStepY) {
 
  cv::Mat matSrcROIBuf;
  matGrayChanels.copyTo(matSrcROIBuf);
  cv::Mat matLineMask = cv::Mat::zeros(matSrcROIBuf.size(), CV_8UC1);
  int nStepCols = (int)(matGrayChanels.cols / (nStepX));
  int nStepRows = (int)(matGrayChanels.rows / (nStepY));
  cv::Mat matBGSizeDark;
  cv::Mat matBGSizeBright;

  matBGSizeDark = cv::Mat::zeros(matSrcROIBuf.size(), CV_8UC1);
  matBGSizeBright = cv::Mat::zeros(matSrcROIBuf.size(), CV_8UC1);

  // int nTh = 3;
  int nTh = nLineThreshold;
  cv::Mat matSrcResize;

  cv::resize(matSrcROIBuf, matSrcResize, cv::Size(nStepCols, nStepRows),
             cv::INTER_LINEAR);

  cv::Mat BGresize = cv::Mat::zeros(matSrcResize.size(), CV_8UC1);

  cv::medianBlur(matSrcResize, BGresize, 3);  // 9

  cv::resize(BGresize, BGresize, matSrcROIBuf.size(), cv::INTER_LINEAR);

  cv::subtract(BGresize, matSrcROIBuf, matBGSizeDark);
  cv::subtract(matSrcROIBuf, BGresize, matBGSizeBright);

  cv::threshold(matBGSizeDark, matBGSizeDark, nTh, 255.0, cv::THRESH_BINARY);
  cv::threshold(matBGSizeBright, matBGSizeBright, nTh, 255.0,
                cv::THRESH_BINARY);

  cv::add(matLineMask, matBGSizeBright, matLineMask);
  cv::add(matLineMask, matBGSizeDark, matLineMask);

  matBGSizeDark.release();
  matBGSizeBright.release();

  defectMask = matLineMask;
  return true;
}





void test_pub_func(const std::string &input_path){

  cv::Mat src_img = cv::imread(input_path);
  cv::Mat gapMat, lineFet;
  detectEdgesWithGaps(src_img, gapMat, lineFet);  // 边缘大的缺口
  cv::Mat removeShadowMat;
  removeShadows(src_img, removeShadowMat);

  cv::Mat gray;
  cv::cvtColor(src_img, gray, cv::COLOR_BGR2GRAY);

  cv::Mat matDstBufX1, matDstBufY1, matDstBuf1;
  Estimation_X(gray, matDstBufX1, 12, 1, 0.4);
  Estimation_Y(gray, matDstBufY1, 12, 1, 0.4);
  TwoImg_Average(matDstBufX1, matDstBufY1, matDstBuf1);
  // EnhanceContrast(src, 5, 0.2);
  cv::Mat matSub1, matSub2;
  cv::subtract(matDstBuf1, removeShadowMat, matSub1);
  cv::Mat defectMask1;
  cv::dilate(matSub1, matSub1,
             cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3)));
  //Make_HardDefect_Mask(matSub1, defectMask1, hardThreshold, nStepX, nStepY);

  //////////合并缺陷
  //cv::add(defectMask1, gapMat, defectMask1);

  //cv::Mat defectEdge;
  //cv::bitwise_and(defectMask1, mask, defectEdge);

}

int sauvola(const cv::Mat &src, cv::Mat &dst ,const double &k, const int &wnd_size) {
  cv::Mat local;
  src.convertTo(local, CV_32F);
  //图像的平方
  cv::Mat square_local;
  square_local = local.mul(local);
  // 图像局部均值
  cv::Mat mean_local;
  //局部均值的平方
  cv::Mat square_mean;
  // 图像局部方差
  cv::Mat var_local;
  //局部标准差
  cv::Mat std_local;
  //阈值图像
  cv::Mat th_img;
  //目标图像的32F
  cv::Mat dst_f;
  // 获取局部均值
  cv::blur(local, mean_local, cv::Size(wnd_size, wnd_size));
  // 计算局部方差
  cv::blur(square_local, var_local, cv::Size(wnd_size, wnd_size));
  //局部均值的平方
  square_mean = mean_local.mul(mean_local);
  //标准差
  cv::sqrt(var_local - square_mean, std_local);
  th_img = mean_local.mul((1 + k * (std_local / 128 - 1)));
  cv::compare(local, th_img, dst_f, cv::CMP_GE);
  dst_f.convertTo(dst, CV_8U);
  return 0;
}



void test_fast_sauvola(const std::string &input_path){
  cv::Mat src = cv::imread(input_path,0);
  cv::Mat dst;
  sauvola(src, dst, 0.1, 15);
}


#include "../include/test_svm.h"
#include "../include/svm_train.h"
#include "../include/img_feature.h"
#include "../include/fs.h"

void test_pca() {

    std::vector<std::string> ok_file;
    nao::fl::getAllFormatFiles(R"(E:\demo\cxx\connector_algo\data\train_data\k\1)", ok_file);
    std::vector<cv::Mat> ok_img_vec;
    for (auto item : ok_file) {
        ok_img_vec.push_back(cv::imread(item));
    }
    //开口
    nao::img::feature::HogTransform ok_hog_transform(ok_img_vec, 11, 8, 4, cv::Size(100, 55), 1);
    cv::Mat ok_feature = ok_hog_transform();
   
    int maxComponents = 500;
    cv::Mat testset = ok_feature.clone();
    cv::Mat compressed;

    cv::PCA pca(ok_feature, // pass the data
        Mat(), // we do not have a pre-computed mean vector,
        // so let the PCA engine to compute it
        cv::PCA::DATA_AS_ROW, // indicate that the vectors
        // are stored as matrix rows
        // (use PCA::DATA_AS_COL if the vectors are
        // the matrix columns)
        maxComponents // specify, how many principal components to retain
    );
    // if there is no test data, just return the computed basis, ready-to-use
    
    CV_Assert(testset.cols == ok_feature.cols);

    compressed.create(testset.rows, maxComponents, testset.type());

    cv::Mat reconstructed;
    for (int i = 0; i < testset.rows; i++)
    {
        Mat vec = testset.row(i), coeffs = compressed.row(i), reconstructed;
        // compress the vector, the result will be stored
        // in the i-th row of the output matrix
        pca.project(vec, coeffs);
        // and then reconstruct it
        pca.backProject(coeffs, reconstructed);
        // and measure the error
        printf("%d. diff = %g\n", i, norm(vec, reconstructed, NORM_L2));
    }

}


void get_histogram(const cv::Mat& src, int* dst)
{
    cv::Mat      hist;
    int          channels[1] = { 0 };
    int          histSize[1] = { 256 };
    float        hranges[2] = { 0, 256.0 };
    const float* ranges[1] = { hranges };
    cv::calcHist(&src, 1, channels, cv::Mat(), hist, 1, histSize, ranges);
    for (int i = 0; i < 256; i++) {
        float  binVal = hist.at<float>(i);
        dst[i] = int(binVal);
    }
}


int otsu(std::vector<int> data)
{
    int    ih;
    int    threshold = -1;
    int    num_pixels = 0;
    double total_mean;   ///< 整个图像的平均灰度
    double bcv, term;    ///< 类间方差，缩放系数
    double max_bcv;      ///< max BCV

    std::vector<double> cnh(data.size(), { 0.0 });     ///< 累积归一化直方图
    std::vector<double> mean(data.size(), { 0.0 });    ///< 平均灰度
    std::vector<double> histo(data.size(), { 0.0 });   ///< 归一化直方图
    // 计算值为非0的像素的个数
    for (ih = 0; ih < data.size(); ih++) {
        num_pixels = num_pixels + data[ih];
    }

    // 计算每个灰度级的像素数目占整幅图像的比例,相当于归一化直方图
    term = 1.0 / static_cast<double>(num_pixels);
    for (ih = 0; ih < data.size(); ih++) {
        histo[ih] = term * data[ih];
    }
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

int sample_otsu(cv::Mat img ,int min_value=0,int max_value=255) {

    int       data[256] = { 0 };
    get_histogram(img, data);
    int scale_range = max_value - min_value + 1;

    std::vector<int> data2(scale_range, { 0 });
    for (int i = min_value; i <= max_value; i++) {
        data2[i - min_value] = data[i];
    }
    int globalThresh = otsu(data2);
    globalThresh = globalThresh + min_value;
    return globalThresh;
}


void test_auto_otsu() {

    cv::Mat img = cv::imread(R"(E:\demo\test\test_opencv\img\item_8.jpg)",0);
    
    int globalThresh = sample_otsu(img);

    cv::Mat brightmask, darkmask;
    cv::threshold(img, brightmask, globalThresh, 255, CV_THRESH_TOZERO);
    cv::threshold(img, darkmask, globalThresh, 255, CV_THRESH_TOZERO_INV);


    int brightThresh = sample_otsu(brightmask, globalThresh,255);
    int darkThresh = sample_otsu(darkmask,1, globalThresh);

    //明暗区域分别进行阈值处理
    cv::Mat brightRegion, darkRegion;

    //亮色区域阈值取globalThresh
    //暗色区域阈值取 darkThresh
    cv::threshold(img, brightRegion, globalThresh, 255, CV_THRESH_BINARY);
    cv::threshold(darkmask, darkRegion, darkThresh, 255, CV_THRESH_BINARY);

    cv::Mat ret = brightRegion + darkRegion;
}


cv::Mat  jubu(cv::Mat src, int type, int radius, float ratio)
{
    // 对图像矩阵进行平滑处理
    cv::Mat smooth;
    switch (type) {
    case 0:
        cv::boxFilter(src, smooth, CV_32FC1, cv::Size(2 * radius + 1, 2 * radius + 1));
        break;
    case 1:
        cv::GaussianBlur(src, smooth, cv::Size(2 * radius + 1, 2 * radius + 1), 0.0);
        break;
    case 2:
        cv::medianBlur(src, smooth, 2 * radius + 1);
        break;
    default:
        break;
    }
    // 平滑结果乘以比例系数，然后图像矩阵与其做差
    cv::Mat srcf;
    src.convertTo(srcf, CV_32FC1);
    if (smooth.type() != CV_32FC1) {
        smooth.convertTo(smooth, CV_32FC1);
    }
    cv::Mat diff = srcf - (1.0 - ratio) * smooth;
    // 阈值处理，当大于或等于0时，输出值为 255，反之输出为0
    cv::Mat out = (diff > 0);
    return out;
}

void test_jubu() {

    cv::Mat img = cv::imread(R"(E:\demo\test\test_opencv\img\item_8.jpg)", 0);

    int globalThresh = sample_otsu(img);

    std::vector<cv::Mat> img_vec_1;
    for (int i = 0; i < 20;i++) {
        cv::Mat ret = jubu(img, 2, 5, 0.01 *(i+1));
        img_vec_1.emplace_back(ret);
    }

    std::vector<cv::Mat> img_vec_2;
    for (int i = 0; i < 20; i++) {
        cv::Mat ret = jubu(img, 0, 5, 0.01 * (i + 1));
        img_vec_2.emplace_back(ret);
    }

    std::vector<cv::Mat> img_vec_3;
    for (int i = 0; i < 20; i++) {
        cv::Mat ret = jubu(img, 1, 5, 0.01 * (i + 1));
        img_vec_3.emplace_back(ret);
    }
    
}


cv::Mat Get_Reflect(const  cv::Mat& src, double sigma)
{
    cv::Mat doubleImage, gaussianImage, logIImage, logGImage, logRImage, End_My, End_My_log, dst_My;
    //转换范围，所有图像元素增加1.0保证log操作正常,防止溢出
    src.convertTo(doubleImage, CV_64FC1, 1.0, 1.0);

    //高斯模糊，当size为零时将通过sigma自动进行计算
    cv::GaussianBlur(doubleImage, gaussianImage, Size(5, 5), sigma);

    //OpenCV的log函数可以计算出对数值。logIImage和logGImage就是对数计算的结果。
    cv::log(doubleImage, logIImage);
    cv::log(gaussianImage, logGImage);

    logRImage = logIImage - logGImage;
    cv::Mat dst;
    cv::normalize(logRImage, dst, 0, 255, cv::NORM_MINMAX, CV_8UC1);

    return dst;
}



cv::Mat unevenLightCompensate(cv::Mat& image, int blockSize)
{
    if (image.channels() == 3) cvtColor(image, image, 7);
    double average = mean(image)[0];
    int rows_new = ceil(double(image.rows) / double(blockSize));
    int cols_new = ceil(double(image.cols) / double(blockSize));
    cv::Mat blockImage;
    blockImage = cv::Mat::zeros(rows_new, cols_new, CV_32FC1);
    for (int i = 0; i < rows_new; i++)
    {
        for (int j = 0; j < cols_new; j++)
        {
            int rowmin = i * blockSize;
            int rowmax = (i + 1) * blockSize;
            if (rowmax > image.rows) rowmax = image.rows;
            int colmin = j * blockSize;
            int colmax = (j + 1) * blockSize;
            if (colmax > image.cols) colmax = image.cols;
            cv::Mat imageROI = image(Range(rowmin, rowmax), Range(colmin, colmax));
            double temaver = mean(imageROI)[0];
            blockImage.at<float>(i, j) = temaver;
        }
    }
    blockImage = blockImage - average;
    cv::Mat blockImage2;
    cv::resize(blockImage, blockImage2, image.size(), (0, 0), (0, 0), cv::INTER_CUBIC);
    cv::Mat image2;
    image.convertTo(image2, CV_32FC1);
    cv::Mat dst = image2 - blockImage2;
    cv::Mat ret;
    dst.convertTo(ret, CV_8UC1);
    return ret;
}


void test_Retinex() {
    cv::Mat img = cv::imread(R"(E:\demo\test\test_opencv\img\item_8.jpg)", 0);

    std::vector<cv::Mat> img_vec_3;
    for (int i = 5; i < 40; i++) {
        cv::Mat ret = unevenLightCompensate(img, i);
        int globalThresh = sample_otsu(ret);
        cv::Mat th;
        cv::threshold(ret, th, globalThresh, 255, CV_THRESH_BINARY);

        img_vec_3.emplace_back(th);
    }
}


void test_split() {
    std::vector<std::string> files;
    nao::fl::getAllFormatFiles(R"(C:\Users\13191\Desktop\debug_img)", files);

    static int index = 0;

    for (int i = 0; i < files.size();i++) {
        cv::Mat src = cv::imread(files[i]);
        cv::Mat itemImg_gray, dst, th_img;
        cv::cvtColor(src, itemImg_gray, cv::COLOR_BGR2GRAY);
        sauvola(itemImg_gray, dst, 0.05, 15);
        //切分字符
        dst = ~dst;
        cv::Mat elementX = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
        cv::dilate(dst, th_img, elementX);
        cv::erode(th_img, th_img, elementX);

        std::vector<std::vector<cv::Point>> contours;
        std::vector<cv::Vec4i> filter_hierarchy;
        cv::findContours(th_img, contours, filter_hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
        cv::Mat gray_mask = cv::Mat::zeros(dst.size(), CV_8UC1);

        std::vector<std::pair<cv::Rect, cv::Mat>> img_vec;
        for (size_t j = 0; j < contours.size(); ++j) {
            cv::Rect rect = cv::boundingRect(contours[j]);
            double area = cv::contourArea(contours[j]);
            if (rect.height< th_img.rows*0.29 || (rect.width /rect.height)>1.6) {
                continue;
            }
            cv::RotatedRect r_rect = cv::minAreaRect(contours[j]);
            std::vector<std::vector<cv::Point>> draw_conts = { contours[j] };
            int width = (std::max)(r_rect.size.width, r_rect.size.height);
            int height = (std::min)(r_rect.size.width, r_rect.size.height);
            double rate = width / (height * 1.0);
            cv::drawContours(gray_mask, draw_conts, 0, 255, -1);
            cv::Mat tmp = th_img(rect).clone();
            img_vec.push_back(std::make_pair(rect, tmp));
        }
        std::sort(img_vec.begin(), img_vec.end(), [&](std::pair<cv::Rect, cv::Mat> a, std::pair<cv::Rect, cv::Mat>b) { return a.first.x < b.first.x; });
        
        for (int j = 0; j < img_vec.size(); j++) {
            cv::imwrite("E:\\demo\\cxx\\connector_algo\\data\\test_data\\1\\" + std::to_string(index) + ".jpg", img_vec[j].second);
            index++;
        }
    }
}