#include "cv_detect_procesing.h"
#include "defines.h"
#include <math.h>

#define PI 3.1415926
using namespace std;

#define DEBUG_SHOW
#define DEBUG_PRINT
#define DEBUG_

cv_detect::cv_detect(){

	this->C_radius = cv::Point(380,280);//380  290
	this->O_radius = cv::Point(480,380);//480  380
	petal_num = 0;
	cross_ratio = 0.0f;
}

bool cv_detect::gv_abnormal(cv::Mat img, int threshold_low_value, int threshold_high_value) {
	
	// cv::Mat input_image = cvimread(img,cv::IMREAD_GRAYSCALE);
	cv::Mat input_image = img;
	if (input_image.empty()) 			  {return false;}
	// 计算平均灰度值
	cv::Scalar mean = cv::mean(input_image);
	LOG_INFO("mean = {}", mean[0]);
	
	// double threshold_high_value = 240; // 根据实际情况调整阈值
	// double threshold_low_value =  50; // 根据实际情况调整阈值
	cv_check.gv_value[0] = mean[0];
	cv_check.gv_value[1] = threshold_low_value;
	cv_check.gv_value[2] = threshold_high_value;
	if( mean[0] > threshold_high_value || mean[0] < threshold_low_value)  {
		cv_check.check_value = "ABNORMAL_IMAGE";
		return false;
		}

	return true;
}
void cv_detect::updateImgCheck(img_check& img) {
        // 在类内部修改 img_check 变量的值
        img.check_value = cv_check.check_value;
        img.gv_value[0] = cv_check.gv_value[0];
        img.gv_value[1] = cv_check.gv_value[1];
        img.gv_value[2] = cv_check.gv_value[2];
}
bool cv_detect::find_circle_center_easy(cv::Mat circle_img){
	
	LOG_INFO("circle_img Cols: {}, Rows: {}, Channels: {}", circle_img.cols, circle_img.rows, circle_img.channels());
    LOG_INFO("======Find circle {} center point==EZ====", bl_cv_pram.detect_Item);
	LOG_INFO("Start cvtColor....");
	int image_ch = circle_img.channels();
	LOG_INFO("Start cvt color to gray, image channel = {}, run_config.channel = {}.", image_ch, run_config.channel);
	if (image_ch == 3){
		cv::cvtColor(circle_img, this->src_img, cv::COLOR_BGR2GRAY);
	}else{
		// circle_img.copyTo(src_img);
		src_img = circle_img ;
	}
	LOG_INFO("cvtColor Done!");
	this->center_pos = cv::Point(842, 496);//default
		int MinR = 100; int MaxR = 250;//100 300
		cv::Mat binary_img, dst_img;
		//////test gv
		cv::Mat test_res;
		cv::equalizeHist(src_img, src_img);
		cv::Scalar mean1 = mean(this->src_img);
		cv::GaussianBlur(this->src_img, this->src_img, cv::Size(13,13), 0.8);
		// cv::imshow("equalizeHist",src_img);
		threshold(this->src_img, binary_img, mean1[0] + 45, 255, cv::THRESH_BINARY );
		// cv::imshow("binary_img", binary_img);
		// cv::waitKey(0);
		// write_debug_img("E:\\project\\BL\\dataset\\BL_DEBUG\\TEST_circle_ea_thres.jpg", test_res, CIRCLE_EA);
		///////////end
		// threshold(this->src_img, binary_img, 190, 255, cv::THRESH_BINARY );
		write_debug_img("E:\\project\\BL\\dataset\\BL_DEBUG\\circle_ea_thres.jpg", binary_img, CIRCLE_EA);
	LOG_INFO("Threshold Done!");
		//GaussianBlur(binary_img, binary_img, Size(3, 3), 0, 0);
		//boxFilter(binary_img, binary_img,-1, Size(3, 3));
		//blur(binary_img, binary_img, Size(3, 3));
		vector<vector<cv::Point>> contours;	
		vector<cv::Vec4i> hireachy;
		cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(132, 132), cv::Point(-1, -1));//line3 42 
		morphologyEx(binary_img, binary_img, cv::MORPH_OPEN, kernel, cv::Point(-1, -1));
	LOG_INFO("morphologyEx Done!");
		write_debug_img("E:\\project\\BL\\dataset\\BL_DEBUG\\circle_ea_open1.jpg", binary_img, CIRCLE_EA);
		cv::Mat test = blConnectedComponentsWithStats(binary_img);
		cv::imshow("test", test);
		cv::waitKey(0);
		// kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3), cv::Point(-1, -1));
		// morphologyEx(binary_img, binary_img, cv::MORPH_OPEN, kernel, cv::Point(-1, -1));
		// write_debug_img("E:\\project\\BL\\dataset\\BL_DEBUG\\circle_ea_open2.jpg", binary_img, CIRCLE_EA);
// 		findContours(binary_img, contours, hireachy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE, cv::Point());
// 	LOG_INFO("findContours Done!");
// 		cv::Mat result_img = src_img.clone();
// 		cv::cvtColor(result_img, result_img, cv::COLOR_GRAY2BGR);
// 	LOG_INFO("cvtColor  COLOR_GRAY2BGR Done!");
	
// 		for (int i = 0; i < hireachy.size(); i++)
// 		{
// 			if (contours[i].size() < 5)continue;
// 			double area = contourArea(contours[i]);
// 			if (area < 28000)continue;//2000
				
// 			double arc_length = arcLength(contours[i], false);
// 			double radius = arc_length / (2 * PI);
	
// 			if (!(MinR < radius && radius < MaxR))
// 			{
// 				continue;
// 			}
		
// 			cv::RotatedRect rect = cv::fitEllipse(contours[i]);
// 			if (abs(rect.center.x - rect.center.y) > 600)
// 			{
// 				continue;
// 			}
// 			//202304 hjf fix circle center erorr bug
// 			if (rect.center.x > result_img.size().width/2 + 300 && rect.center.y > result_img.size().height/2 + 300 ) {
// 				continue;
// 			}
// 			if (rect.center.x < result_img.size().width/2 - 300 && rect.center.y > result_img.size().height/2 - 300 ) {
// 				continue;
// 			}
// 			float ratio = float(rect.size.width) / float(rect.size.height);
		
// 			if (ratio < 1.15 && ratio > 0.9)//1.25 0.8
			
// 			{
// 				this->center_pos = cv::Point(rect.center.x - offset_x, rect.center.y - offset_y);
//                 printf("X: %f\n", rect.center.x);
// 				printf("Y: %f\n", rect.center.y);
// 				printf("circle area: %f\n", area);
// 				printf("circle radius: %f\n", radius);
// #ifdef DEBUG_CIRCLE_EA
// cv::ellipse(result_img, rect, cv::Scalar(0,255, 255),2);
// cv::circle(result_img, rect.center, 2, cv::Scalar(0,255,0), 2, 8, 0);
// write_debug_img("E:\\project\\BL\\dataset\\BL_DEBUG\\circle_ea.jpg", result_img, CIRCLE_EA);
// #endif
// 				return true;
// 			}
// 		}
//     this->center_pos = cv::Point(842, 496);//default
	return true;
}

cv::Mat cv_detect::blConnectedComponentsWithStats(const cv::Mat &inputImage)
{
    // Mat gray;
	int max_area = -1;
    // cvtColor(inputImage, gray, COLOR_BGR2GRAY);
    cv::Mat thresh = inputImage;
    // threshold(gray, thresh, 0, 255, THRESH_BINARY_INV + THRESH_OTSU);

    // 查找连通组件并输出统计信息
    cv::Mat labels, stats, centroids;
    int num_labels = cv::connectedComponentsWithStats(thresh, labels, stats, centroids);
    // 创建一个彩色图像，将标记添加到其中
    cv::Mat output;
    cv::cvtColor(thresh, output, cv::COLOR_GRAY2BGR);

    for (int i = 1; i < num_labels; i++)
    {
		
        int area = stats.at<int>(i, cv::CC_STAT_AREA);
        int left = stats.at<int>(i, cv::CC_STAT_LEFT);
        int top = stats.at<int>(i, cv::CC_STAT_TOP);
        int width = stats.at<int>(i, cv::CC_STAT_WIDTH);
        int height = stats.at<int>(i, cv::CC_STAT_HEIGHT);
        double cx = centroids.at<double>(i, 0);
        double cy = centroids.at<double>(i, 1);
		int area_rect = width * height;
		float rate = float(min(width, height))/float(max(width, height));
		// if(area > max_area) {
		// 	max_area = area;
		// }
        
		// if (abs(cx - cy) > 600)
		// {

		// 	continue;
		// }
			//202304 hjf fix circle center erorr bug
		if (rate < 0.7) {
			// LOG_INFO("rate = {}  continue!",rate);
			continue;
		}
		if (area < 15000) {
			// LOG_INFO("area = {}  continue!",area);
			continue;
		}


		if (area_rect > 85000 || area_rect < 20000) {
			// LOG_INFO("area_rect = {}  continue!",area_rect);
			continue;
		}
		LOG_INFO("cy = {} , cx = {} continue!", cy, cx);
		if ((cy > 400 && cy < 900) && ( cx > 400 && cx < 900 ) ) {
			// LOG_INFO("cy = {} , cx = {} continue!", cy, cx);
			// continue;
		
			cv::rectangle(output, cv::Point(left, top), cv::Point(left + width, top + height), cv::Scalar(0, 0, 255), 2);
			cv::circle(output, cv::Point(cx, cy), 2, cv::Scalar(0, 255, 0), -1);

			
			ostringstream oss;
			oss << "Label: " << i << ", Area: " << area << ", Area_rect: "<< area_rect<< ", rate: " << rate<< ", Centroid: (" << cx << ", " << cy << ")";
			// putText(output, oss.str(), Point(left, top - 5), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 0, 0), 1);
			std::cout << oss.str() << std::endl;
			this->center_pos = cv::Point(cx - offset_x, cy - offset_y);
			return output;
		}
    }
	// max_CC_STAT_AREA = max_area;
    return output;
}

cv::Mat cv_detect::blplugConnectedComponentsWithStats(const cv::Mat &inputImage)
{
    // Mat gray;
	int max_area = -1;
    // cvtColor(inputImage, gray, COLOR_BGR2GRAY);
    cv::Mat thresh = inputImage;
    // threshold(gray, thresh, 0, 255, THRESH_BINARY_INV + THRESH_OTSU);

	// cv::findContours(~thresh, this->contours, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);
	// LOG_INFO("auto thres ----cont_mean_gv = {}", cont_mean_gv(contours, src));
	// for (vector<vector<cv::Point>>::iterator it = contours.begin(); it < contours.end(); it++)
	// {
		 
	// 	double area = cv::contourArea(*it);
	// 	LOG_INFO("contours size: {}, area :{}", contours.size(), area);
	// 	if (area < 8000){
	// 		contours.erase(it);
	// 		it -= 1;
	// 	}
	// 	else {
	// 		std::cout << "find area = " << area << std::endl;
	// 	}
	// }
    // 查找连通组件并输出统计信息
    cv::Mat labels, stats, centroids;

    int num_labels = cv::connectedComponentsWithStats(thresh, labels, stats, centroids);
    // 创建一个彩色图像，将标记添加到其中
    cv::Mat output;
    cv::cvtColor(thresh, output, cv::COLOR_GRAY2BGR);
	int i = 1;
	int sum = 0;
	int num = 0;
    for (; i < num_labels; i++)
    {
		
        int area = stats.at<int>(i, cv::CC_STAT_AREA);
        int left = stats.at<int>(i, cv::CC_STAT_LEFT);
        int top = stats.at<int>(i, cv::CC_STAT_TOP);
        int width = stats.at<int>(i, cv::CC_STAT_WIDTH);
        int height = stats.at<int>(i, cv::CC_STAT_HEIGHT);
        double cx = centroids.at<double>(i, 0);
        double cy = centroids.at<double>(i, 1);
		int area_rect = width * height;
		float rate = float(min(width, height))/float(max(width, height));

		if(area < 2000) {//6000
			continue;
		}

		if(area > 40000) {
			continue;
		}
		// if(rate > 0.6) {
		// 	continue;
		// }
		// ostringstream oss2;
		// oss2 << "Label: " << i << ", Area: " << area << ", Area_rect: "<< area_rect<< ", rate: " << rate<< ", Centroid: (" << cx << ", " << cy << ")";
		// std::cout << oss2.str() << std::endl;
		// if(area_rect > 40000) {
		// 	continue;
		// }
        // std::cout << "rate= " << rate << "w = "<< w << "h = " << h << std::endl;
		// max_area.push_back(area);
		sum += area;
		cv::Rect boundingBox = cv::Rect(left, top, width, height);
		// cv::RotatedRect rotatedBoundingBox = cv::minAreaRect(cv::Mat(thresh, boundingBox));
		// std::cout << "w = "<< boundingBox.width << "h = " << boundingBox.height << std::endl;
		contourInfo.areas.push_back(area);
		contourInfo.boundingBoxes.push_back(boundingBox);
		contourInfo.centers.push_back(cv::Point2f(cx, cy));
		num++;
		// if (abs(cx - cy) > 600)
		// {

		// 	continue;
		// }
			//202304 hjf fix circle center erorr bug
		// if (rate < 0.7) {
		// 	LOG_INFO("rate = {}  continue!",rate);
		// 	continue;
		// }
		// if (area < 15000) {
		// 	LOG_INFO("area = {}  continue!",area);
		// 	continue;
		// }


		// if (area_rect > 85000 || area_rect < 20000) {
		// 	LOG_INFO("area_rect = {}  continue!",area_rect);
		// 	continue;
		// }
		// LOG_INFO("cy = {} , cx = {} continue!", cy, cx);
		// if ((cy > 400 && cy < 900) && ( cx > 400 && cx < 900 ) ) {
			// LOG_INFO("cy = {} , cx = {} continue!", cy, cx);
			// continue;
		
			cv::rectangle(output, cv::Point(left, top), cv::Point(left + width, top + height), cv::Scalar(0, 0, 255), 2);
			cv::circle(output, cv::Point(cx, cy), 2, cv::Scalar(0, 255, 0), -1);

			
			ostringstream oss1;
			oss1 << "Label: " << i << ", Area: " << area << ", Area_rect: "<< area_rect<< ", rate: " << rate<< ", Centroid: (" << cx << ", " << cy << ")";
			// putText(output, oss.str(), Point(left, top - 5), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 0, 0), 1);
			std::cout << oss1.str() << std::endl;
			// this->center_pos = cv::Point(cx - offset_x, cy - offset_y);
			// return output;
		// }
    }
	contourInfo.numContours = num;
	contourInfo.area_sum = sum;
	// max_CC_STAT_AREA = max_area;
    return output;
}

bool cv_detect::find_circle_center_hard(cv::Mat circle_img){
    LOG_INFO("======Find circle {} center point==HD====", bl_cv_pram.detect_Item);

		if (run_config.channel > 2){
			cv::cvtColor(circle_img, this->src_img, cv::COLOR_BGR2GRAY);
		}else{
			circle_img.copyTo(src_img);
		}
		int MinR = 100; int MaxR = 300;//100 300
		cv::Mat binary_img, dst_img;
		this->center_pos = cv::Point(842, 496);//default
		// cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(120, 120), cv::Point(-1, -1));
		cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(110, 110), cv::Point(-1, -1));//MORPH_CROSS
		morphologyEx(this->src_img, binary_img, cv::MORPH_OPEN, kernel, cv::Point(-1, -1));
		threshold(binary_img, binary_img, 200, 255, cv::THRESH_BINARY ); //158 255 cv::THRESH_BINARY   THRESH_TRUNC
		// cv::imshow("binary_img", binary_img);
		// threshold(binary_img, binary_img, 158, 255, cv::THRESH_BINARY );
		write_debug_img("E:\\project\\BL\\dataset\\BL_DEBUG\\circle_hd_thres.jpg", binary_img, CIRCLE_HD);
		//GaussianBlur(binary_img, binary_img, Size(3, 3), 0, 0);
		//boxFilter(binary_img, binary_img,-1, Size(3, 3));
		//blur(binary_img, binary_img, Size(3, 3));
		vector<vector<cv::Point>> contours;	
		vector<cv::Vec4i> hireachy;
		kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(8, 8), cv::Point(-1, -1));
		morphologyEx(binary_img, binary_img, cv::MORPH_OPEN, kernel, cv::Point(-1, -1));
		write_debug_img("E:\\project\\BL\\dataset\\BL_DEBUG\\circle_hd_open2.jpg", binary_img, CIRCLE_HD);
		cv::Mat test = blConnectedComponentsWithStats(binary_img);
		// cv::imshow("test", test);
		// cv::waitKey(0);
		// kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3), cv::Point(-1, -1));
		// morphologyEx(binary_img, binary_img, cv::MORPH_OPEN, kernel, cv::Point(-1, -1));
// 		findContours(binary_img, contours, hireachy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE, cv::Point());
// 		cv::Mat result_img = src_img.clone();
// 		cv::cvtColor(result_img, result_img, cv::COLOR_GRAY2BGR);
// 		for (int i = 0; i < hireachy.size(); i++)
// 		{
// // #ifdef DEBUG_CIRCLE_HD
// // drawContours(src_img, contours, i, cv::Scalar(0, 0, 255), -1, 8, cv::Mat(), 0, cv::Point());
// // write_debug_img("E:\\project\\BL\\dataset\\BL_DEBUG\\circle_hd_cont.jpg", binary_img, CIRCLE_HD);
// // #endif
// 			if (contours[i].size() < 5)continue;
// 			double area = contourArea(contours[i]);
// 			if (area < 28000)continue;//2000
				
// 			double arc_length = arcLength(contours[i], false);
// 			double radius = arc_length / (2 * PI);
	
// 			if (!(MinR < radius && radius < MaxR))
// 			{
// 				continue;
// 			}
		
// 			cv::RotatedRect rect = cv::fitEllipse(contours[i]);
// 			if (abs(rect.center.x - rect.center.y) > 600)
// 			{
// 				continue;
// 			}
// 			//202304 hjf fix circle center erorr bug
// 			if (rect.center.x > result_img.size().width/2 + 300 && rect.center.y > result_img.size().height/2 + 300 ) {
// 				continue;
// 			}
// 			if (rect.center.x < result_img.size().width/2 - 300 && rect.center.y > result_img.size().height/2 - 300 ) {
// 				continue;
// 			}
// 			float ratio = float(rect.size.width) / float(rect.size.height);
		
// 			if (ratio < 1.5 && ratio > 0.7 )
			
// 			{
// 				this->center_pos = cv::Point(rect.center.x - offset_x, rect.center.y - offset_y);
//                 printf("X: %f\n", rect.center.x);
// 				printf("Y: %f\n", rect.center.y);
// 				printf("circle area: %f\n", area);
// 				printf("circle radius: %f\n", radius);
// #ifdef DEBUG_CIRCLE_HD
// cv::ellipse(binary_img, rect, cv::Scalar(0,255, 255),2);
// cv::circle(binary_img, rect.center, 2, cv::Scalar(0,255,0), 2, 8, 0);
// write_debug_img("E:\\project\\BL\\dataset\\BL_DEBUG\\circle_hd_bin.jpg", binary_img, CIRCLE_HD);
// cv::ellipse(result_img, rect, cv::Scalar(0,255, 255),2);
// cv::circle(result_img, rect.center, 2, cv::Scalar(0,255,0), 2, 8, 0);
// write_debug_img("E:\\project\\BL\\dataset\\BL_DEBUG\\circle_hd_src.jpg", result_img, CIRCLE_HD);
// #endif		
// 				return true;
// 			}
// 		}
    // this->center_pos = cv::Point(842, 496);//default
	return false;

}

float cv_detect::stat_gray_ratio(cv::Point radius, cv::Point center, cv::Mat src_img){
	LOG_INFO("=======stat_gray_ratio=========");
	// cv::Mat src_img;//specs1#1_49#0_20220608144203916133.png
	// std::string img_path = "D:\\Ronnie\\pakges\\bldataset\\testdata\\C_ok\\c_ok3.BMP";
	// src_img = cv::imread(img_path);
	// cv::imshow("src", src_img);
	cv::Mat draw_img(src_img.size(), CV_8UC3, cv::Scalar(0));//455 280
	if (src_img.channels() == 3) {
		cv::circle(draw_img, center, radius.x, cv::Scalar(255, 255, 255), -1);
		cv::circle(draw_img, center, radius.y, cv::Scalar(0, 0, 0), -1);
	} else {
		LOG_INFO("Input image is  gray image, return 0.0");
		// cv::circle(draw_img, center, radius.x, cv::Scalar(255), -1);
		// cv::circle(draw_img, center, radius.y, cv::Scalar(0), -1);
		return 0.0;
	}
	
	// cv::Mat draw_img(src_img.size(), CV_8UC3, cv::Scalar(0));//455 280
	// cv::circle(draw_img, center, radius.x, cv::Scalar(255, 255, 255), -1);
	// cv::circle(draw_img, center, radius.y, cv::Scalar(0, 0, 0), -1);
	
	// cv::circle(draw_img, this->center_pos, 2, cv::Scalar(0, 0, 0), 0);
	// cv::imshow("draw_img", draw_img);
	// cv::waitKey(0);
	cv::Mat dst;
	cv::bitwise_and(src_img, draw_img, dst);
	// cv::circle(src_img, center, radius.x, cv::Scalar(0, 255, 255), 1);
	// cv::circle(src_img, center, radius.y, cv::Scalar(0, 0, 0), 1);
	// cv::Mat crop_dst = dst(cv::Range(center.y - 400, center.y + 400), cv::Range(center.x - 400, center.x + 400));
	if (radius.x > 400) {
		write_debug_img("E:\\project\\BL\\dataset\\BL_DEBUG\\O_bw.jpg", dst, FIND_O);
	} else {
		write_debug_img("E:\\project\\BL\\dataset\\BL_DEBUG\\C_bw.jpg", dst, FIND_C);
	}
#ifdef DEBUG
cv::imshow("src_img", dst);
cv::waitKey(0);
// cv::Mat mean, stddev;
// 	meanStdDev(dst, mean, stddev);
// 	printf("blue channel -> mean: %.2f, stddev: %2f\n", mean.at<double>(0, 0), stddev.at<double>(0, 0));
// 	printf("green channel -> mean: %.2f, stddev: %2f\n", mean.at<double>(1, 0), stddev.at<double>(1, 0));
// 	printf("red channel -> mean: %.2f, stddev: %2f\n", mean.at<double>(2, 0), stddev.at<double>(2, 0));

#endif
	cv::Size shape = dst.size();
	
	// 阈值分析
	cv::Mat dst_hsv;
	cv::cvtColor(dst, dst_hsv, cv::COLOR_BGR2HSV);

	cv::Scalar white_L = cv::Scalar(0, 0, 221);
	cv::Scalar white_H = cv::Scalar(180, 30, 255);
	cv::Mat white_thr;
	cv::inRange(dst_hsv, white_L, white_H, white_thr);
	

	// cv::Scalar black_L = cv::Scalar(0, 0, 0);
	// cv::Scalar black_H = cv::Scalar(180, 255, 46);
	// cv::Mat black_thr;
	// cv::inRange(dst_hsv, black_L, black_H, black_thr);
	// cv::imshow("black_thr", black_thr);

	// cv::Scalar gray_L = cv::Scalar(0, 0, 46);
	// cv::Scalar gray_H = cv::Scalar(180, 43, 220);
	// cv::Mat gray_thr;
	// cv::inRange(dst_hsv, gray_L, gray_H, gray_thr);
	if ( radius.x > 400) {
		write_debug_img("E:\\project\\BL\\dataset\\BL_DEBUG\\O_white.jpg", white_thr, FIND_O);
		// write_debug_img("E:\\project\\BL\\dataset\\BL_DEBUG\\O_gray.jpg", gray_thr, FIND_O);
	}else {
		write_debug_img("E:\\project\\BL\\dataset\\BL_DEBUG\\C_white.jpg", white_thr, FIND_C);
		// write_debug_img("E:\\project\\BL\\dataset\\BL_DEBUG\\C_gray.jpg", gray_thr, FIND_C);
	}
	

	// cv::imshow("white_thr", white_thr);
	// cv::imshow("gray_thr", gray_thr);
	
	//定义结构元素
	cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(11, 11));

	//先膨胀再腐蚀
	cv::dilate(white_thr, white_thr, element);
	cv::erode(white_thr, white_thr, element);
	// cv::imshow("white_thr1", white_thr);
	// cv::waitKey(0);
	float circle_area = PI*(radius.x * radius.x - radius.y * radius.y);
	LOG_INFO("circle_area {}:{}",bl_cv_pram.detect_Item, circle_area);
	float sum = 0;
	// if (radius.x == 350){
		for (int i = 0; i < white_thr.rows; i++) {
				for (int j = 0; j < white_thr.cols; j++) {
					
					if (white_thr.at<uchar>(i,j) == 255) {
						
						sum += 1;	
					}			
				}
			}
	// }else{
	// 	for (int i = 0; i < gray_thr.rows; i++) {
	// 			for (int j = 0; j < gray_thr.cols; j++) {
					
	// 				if (gray_thr.at<uchar>(i,j) == 255) {
						
	// 					sum += 1;	
	// 				}			
	// 			}
	// 		}
	// }

	LOG_INFO("sum = {}   sum_150_area/circle_area = {}", sum, float(sum/circle_area));
    return float(sum/circle_area);
}

cv::Mat cv_detect::plug_num(cv::Mat src) {
	// 阈值分析
	cv::GaussianBlur(src, src, cv::Size(31, 31), 6);
	cv::equalizeHist(src, src);
	// cv::imshow("srrc", src);
	cv::Mat dst_hsv;
	cv::Mat combined;
	// std::cout << src.channels() << std::endl;
	cv::cvtColor(src, dst_hsv, cv::COLOR_GRAY2BGR);
	cv::cvtColor(dst_hsv, dst_hsv, cv::COLOR_BGR2HSV);
	// cv::imshow("dst", dst_hsv);
	cv::Scalar white_L = cv::Scalar(0, 0, 230);//238  //line4 230
	cv::Scalar white_H = cv::Scalar(180, 10, 255);
	cv::Mat white_thr;
	cv::inRange(dst_hsv, white_L, white_H, white_thr);
	
	// cv::imshow("white_thr", white_thr);
	// cv::Scalar black_L = cv::Scalar(0, 0, 0);
	// cv::Scalar black_H = cv::Scalar(180, 255, 46);
	// cv::Mat black_thr;
	// cv::inRange(dst_hsv, black_L, black_H, black_thr);
	// cv::imshow("black_thr", black_thr);
	// cv::bitwise_or(white_thr, black_thr, combined);
	// cv::Scalar gray_L = cv::Scalar(0, 0, 46);
	// cv::Scalar gray_H = cv::Scalar(180, 43, 220);
	// cv::Mat gray_thr;
	// cv::inRange(dst_hsv, gray_L, gray_H, gray_thr);

	// cv::bitwise_or(combined, gray_thr, combined);
	// cv::Scalar white_L = cv::Scalar(0, 0, 250);
	// cv::Scalar white_H = cv::Scalar(180, 10, 255);
	// cv::Mat white_thr;
	// cv::inRange(dst_hsv, white_L, white_H, combined);
	// cv::imshow("gray_thr", gray_thr);
	// cv::waitKey(0);
	return white_thr;
}

std::string cv_detect::check_c_circle(cv::Mat src_img){
	std::string result_c;

	// find_circle_center(src_img);
	float c_thr = stat_gray_ratio(this->C_radius, this->center_pos, src_img);
	if (c_thr > 0.25f){
		result_c = "C_OK";
	}
	else{
		result_c = "C_NG";
	}
    return result_c;
}

std::string cv_detect::check_o_circle(cv::Mat src_img){
	std::string result_o;
	
	float o_thr = 1 - stat_gray_ratio(this->O_radius, this->center_pos, src_img);
	if (o_thr > 0.85f){
		result_o = "O_OK";
	}
	else{
		result_o = "O_NG";
	}
	return result_o;
}

bl_config cv_detect::get_bl_config(){


        return this->bl_json;
}

std::string cv_detect::check_plug(cv::Mat img_src){
	LOG_INFO("\nPLUG detect run.....");
	// cv::Mat thresh;
	std::stringstream ss;

		if (run_config.channel > 2){
		cv::cvtColor(img_src, this->src_img, cv::COLOR_BGR2GRAY);
	}else{
		img_src.copyTo(src_img);
		}
	
	// cv::equalizeHist(src_img, src_img);
	// cv::imshow("src", src_img);
	// preproces_petal(src_img, thresh);
	cv::Mat abnor; 
	src_img.copyTo(abnor);
	get_petal_num(abnor);
	if (this->petal_num != 4)   {
		// ss<<"E:\\project\\BL\\dataset\\BL_DEBUG\\img\\" << run_config.img_name << "_NUM.jpg";s
		// write_debug_img(ss.str(), src_img, FIND_PLUG);
		LOG_INFO("PLUG NUM LACK!");
		return "PLUG_NG";
	}
	int gv_abnormal_num = plug_gv_abnormal(this->contours, src_img);
	if(gv_abnormal_num) {
		this->petal_num -= gv_abnormal_num;
		LOG_INFO("PLUG gv abnormal num = {}", gv_abnormal_num);
	}
	if (this->petal_num != 4)   {
		ss<<"E:\\project\\BL\\dataset\\BL_DEBUG\\img\\" << run_config.img_name << "_NUM.jpg";
		write_debug_img(ss.str(), src_img, FIND_PLUG); return "PLUG_NG";
	}
	get_petal_ratio();

	if (cross_ratio < 0.7f || over_max) 	{
		over_max = 0;
		ss<<"E:\\project\\BL\\dataset\\BL_DEBUG\\img\\" << run_config.img_name << "_cross_ratio.jpg";
		write_debug_img(ss.str(), src_img, FIND_PLUG); return "PLUG_NG";}
	if (area_plug_ratio > 0.35)  {//std
		ss<<"E:\\project\\BL\\dataset\\BL_DEBUG\\img\\" << run_config.img_name << "_area_plug_ratio.jpg";
		write_debug_img(ss.str(), src_img, FIND_PLUG); return "PLUG_NG";}
	if (max(max_area[0],max_area[1])/min(max_area[0],max_area[1]) > 1.7 || max(max_area[2],max_area[3])/min(max_area[2],max_area[3]) > 1.7) 
								{
									ss<<"E:\\project\\BL\\dataset\\BL_DEBUG\\img\\" << run_config.img_name << "_plug_N/M.jpg";
									write_debug_img(ss.str(), src_img, FIND_PLUG); return "PLUG_NG";}
	return "PLUG_OK";
}

static inline bool ContoursSortFun(vector<cv::Point> contour1,vector<cv::Point> contour2){ return (cv::contourArea(contour1) > cv::contourArea(contour2));}
cv::Mat cv_detect::preproces_petal2(cv::Mat gray) {
	cv::Mat src;
	return src;
}
void cv_detect::get_petal_num(cv::Mat src){
	cv::Mat thresh1;
	cv::Mat thresh = plug_num(src);
	cv::Mat plug_blob = blplugConnectedComponentsWithStats(thresh);
	// cv::imshow("plug_blob", plug_blob);
	// cv::waitKey(0);
	LOG_INFO("===========get plug num========");
	// int petal_radius = 0;
	// cv::imshow("plug_blob", thresh);
	// cv::waitKey(0);
	std::vector<std::vector<cv::Point>> contour_num;
	cv::findContours(thresh, contour_num, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);
	// cv::findContours(thresh, this->contours, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);
	// LOG_INFO("auto thres ----cont_mean_gv = {}", cont_mean_gv(contours, src));
	contours.clear();
	for (vector<vector<cv::Point>>::iterator it = contour_num.begin(); it < contour_num.end(); it++)
	{
		 
		double area = cv::contourArea(*it);
		cv::RotatedRect rect = cv::minAreaRect(*it);
		cv::Size2f rectSize = rect.size;
		float width = rectSize.width;
		float height = rectSize.height;
		
		// LOG_INFO("contours size: {}, area :{}", contour_num.size(), area);
		if(area < 6000) {
			continue;
		}

		if(area > 40000) {
			continue;
		}
		float rate = min(width, height) / max(width, height);
		if (rate > 0.8 || max(width, height) > 368) {
			continue;
		}
		// LOG_INFO("plug contours rate = {}  withd = {}   height = {}", rate, width, height);
		// if (rate > 0.7 || max(height, width) > 368) {
		// 	continue;
		// }
		contours.push_back(*it);
	}
	
	// 		thresh1 = preproces_petal(src, 180);//110
	// 		LOG_INFO("AutoThresh fail max area = {} exceed!  Enable Thresh 1.----110",area);
	// 		break;
	// 	}
	// }
	// cv::Mat thresh2;
	// if (contours.size() != 4){
	// 	contours.clear();
	// 	thresh1 = preproces_petal(src, 40);
	// 	cv::findContours(thresh1, this->contours, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);
	// 	LOG_INFO("180 thres ----cont_mean_gv = {}", cont_mean_gv(contours, src));
	// 	for (vector<vector<cv::Point>>::iterator it = contours.begin(); it < contours.end(); it++)
	// 	{
		
	// 		double area = cv::contourArea(*it);
	// 		LOG_INFO("contours size: {}, area :{}", contours.size(), area);
	// 		if (area < 3000){
	// 			contours.erase(it);
	// 			it -= 1;
	// 		}
	// 		if (area > 100000) {
	// 			LOG_INFO("Thresh1 fail max area = {} exceed!  Enable Thresh 2.----200",area);
	// 			thresh2 = preproces_petal(src, 80);//200
	// 			break;
	// 		}
	// 	}
	// }
	// if (!thresh2.empty()){
	// 	contours.clear();
	// 	cv::findContours(thresh2, this->contours, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);
	// 	LOG_INFO("220 thres ----cont_mean_gv = {}", cont_mean_gv(contours, src));
	// 	for (vector<vector<cv::Point>>::iterator it = contours.begin(); it < contours.end(); it++)
	// 	{
		
	// 		double area = cv::contourArea(*it);
	// 		LOG_INFO("contours size: {}, area :{}", contours.size(), area);
	// 		if (area < 3000){
	// 			contours.erase(it);
	// 			it -= 1;
	// 		}
	// 	}
	// }
	// this->petal_num = contourInfo.numContours;
	this->petal_num = contours.size();
	LOG_INFO("petal_num : {}", petal_num);
	return;
}

void cv_detect::get_petal_ratio2(){
	LOG_INFO("\nget_petal_ratio");
	double std_ = 0;
	cv::Mat rect_img;
	vector<cv::Point2f> center = contourInfo.centers;
	
	src_img.copyTo(rect_img);
	int sum = contourInfo.area_sum;
	std::vector<double> area_plug = contourInfo.areas;
	max_area = contourInfo.areas;
	int mean = sum / 4;
	// 计算偏差相对于平均值的比例
	for (const auto& t : area_plug) {
		double deviation_ratio = static_cast<double>(std::abs(t - mean)) / mean;
		std_ += deviation_ratio * deviation_ratio;
		LOG_INFO("this area_ratio= {}", deviation_ratio);
		if (deviation_ratio > area_plug_ratio) { // add plug judge param(area_plug_ratio) 
			area_plug_ratio = deviation_ratio;
		}

	}
	LOG_INFO("this area_plug_max_ratio= {}", area_plug_ratio);
	sort(max_area.begin(), max_area.end());
	LOG_INFO("{},{},{},{}",max_area[0], max_area[1], max_area[2], max_area[3]);
	LOG_INFO("this plug_n/m= ({}  , {} )", max(max_area[0],max_area[1])/min(max_area[0],max_area[1]), max(max_area[2],max_area[3])/min(max_area[2],max_area[3]));
	LOG_INFO("this std_area= {}", std_/4);
	area_plug_ratio = std_/4;
	float d1 = sqrt((center[0].x - center[1].x) * (center[0].x - center[1].x) + (center[0].y - center[1].y) * (center[0].y - center[1].y));
	float d2 = sqrt((center[2].x - center[3].x) * (center[2].x - center[3].x) + (center[2].y - center[3].y) * (center[2].y - center[3].y));
	this->cross_ratio = min(d1, d2) / max(d1, d2);
	LOG_INFO("this cross_ratio= {}", cross_ratio);
#ifdef DEBUG
cv::imshow("rect_img", rect_img);
cv::imshow("rect_center_point", rect_img);
cv::drawContours(rect_img, contours, -1, cv::Scalar(0, 255, 255), 3);
cv::imshow("drawContours", rect_img);
cv::waitKey(0);
#endif
	contours.clear();
	max_area.clear();
	return;
}

void cv_detect::get_petal_ratio(){
	LOG_INFO("\nget_petal_ratio");
	cv::Mat rect_img;
	vector<cv::Point2f> center;
	
	src_img.copyTo(rect_img);
	std::sort(contours.begin(), contours.end(), ContoursSortFun);
	int sum = 0;
	double std = 0;
	vector<double> area_plug;
	for (vector<vector<cv::Point>>::iterator it = contours.begin(); it < contours.end(); it++)
	{
		double area = cv::contourArea(*it);
		cv::RotatedRect min_rect = cv::minAreaRect(*it);


//DEBUG 
#ifdef DEBUG_SHOW_PLUG
cv::Rect plug_rect = min_rect.boundingRect();
cv::rectangle(rect_img, plug_rect, cv::Scalar(255,126,41), 3);
cv::Point2f P[4];
min_rect.points(P);
		// double min_area_rect = sqrt(- (P[0].x - P[1].x)*(P[0].x - P[1].x) + (P[0].y - P[1].y)*(P[0].y - P[1].y)) * sqrt((P[2].x - P[1].x)*(P[2].x - P[1].x) + (P[2].y - P[1].y)*(P[2].y - P[1].y));
//绘制轮廓的最小外接矩形
for (int j = 0; j <= 3; j++)
{
	line(rect_img, P[j], P[(j + 1) % 4], cv::Scalar(255, 255, 255), 2);
}
//绘制矩形中心点
cv::circle(rect_img, min_rect.center, 2, cv::Scalar(0, 145, 100), -1);
write_debug_img("E:\\project\\BL\\dataset\\BL_DEBUG\\PLUG_rect.jpg", rect_img, FIND_PLUG);
#endif

		double max_xy = (min_rect.size.width > min_rect.size.height) ? min_rect.size.width:min_rect.size.height;
		double area_rect = min_rect.size.width * min_rect.size.height;
		center.push_back(min_rect.center);
		LOG_INFO("contours size: {}, area :{} rect area : {} ratio:{}", contours.size(), area, area_rect, area/area_rect);
		if (area > 40000) {
			over_max = 1;
			return;
		}
		max_area.push_back(area);
		// LOG_INFO("max_area[{}] = {}", j, max_area[j]);
		sum += area;

		area_plug.push_back(area);
		// cv::rectangle(gray, min_rect);
		// if (area < 10000){
		// 	contours.erase(it);
		// }
	}

	int mean = sum / 4;
	// 计算偏差相对于平均值的比例
	for (const auto& t : area_plug) {
		double deviation_ratio = static_cast<double>(std::abs(t - mean)) / mean;
		std += deviation_ratio * deviation_ratio;
		LOG_INFO("this area_ratio= {}", deviation_ratio);
		if (deviation_ratio > area_plug_ratio) { // add plug judge param(area_plug_ratio) 
			area_plug_ratio = deviation_ratio;
		}

	}
	LOG_INFO("this area_plug_max_ratio= {}", area_plug_ratio);
	sort(max_area.begin(), max_area.end());
	LOG_INFO("{},{},{},{}",max_area[0], max_area[1], max_area[2], max_area[3]);
	LOG_INFO("this plug_n/m= ({}  , {} )", max(max_area[0],max_area[1])/min(max_area[0],max_area[1]), max(max_area[2],max_area[3])/min(max_area[2],max_area[3]));
	LOG_INFO("this std_area= {}", std/4);
	area_plug_ratio = std/4;
	float d1 = sqrt((center[0].x - center[1].x) * (center[0].x - center[1].x) + (center[0].y - center[1].y) * (center[0].y - center[1].y));
	float d2 = sqrt((center[2].x - center[3].x) * (center[2].x - center[3].x) + (center[2].y - center[3].y) * (center[2].y - center[3].y));
	this->cross_ratio = min(d1, d2) / max(d1, d2);
	LOG_INFO("this cross_ratio= {}", cross_ratio);
#ifdef DEBUG
cv::imshow("rect_img", rect_img);
cv::imshow("rect_center_point", rect_img);
cv::drawContours(rect_img, contours, -1, cv::Scalar(0, 255, 255), 3);
cv::imshow("drawContours", rect_img);
cv::waitKey(0);
#endif
	contours.clear();
	max_area.clear();
	return;
}
int cv_detect::thres_YEN(cv::Mat gray_img){
	cv::cvtColor(gray_img, gray_img, cv::IMREAD_GRAYSCALE);
	    int histogram[256] = {0};
    for (int i = 0; i < gray_img.rows; i++) {
        for (int j = 0; j < gray_img.cols; j++) {
            int intensity = (int)gray_img.at<uchar>(i, j);
            histogram[intensity]++;
        }
    }

    // Calculate probability mass function
    double p[256] = {0.0};
    int num_pixels = gray_img.rows * gray_img.cols;
    for (int i = 0; i < 256; i++) {
        p[i] = (double)histogram[i] / num_pixels;
    }

    // Calculate cumulative distribution function
    double c[256] = {0.0};
    c[0] = p[0];
    for (int i = 1; i < 256; i++) {
        c[i] = c[i - 1] + p[i];
    }

    double max_var = std::numeric_limits<double>::min();
    int threshold = 0;

    for (int t = 0; t < 256; t++) {
        double omega1 = c[t];
        double omega2 = 1 - omega1;

        double mu1 = 0;
        for (int i = 0; i <= t; i++) {
            mu1 += (i * p[i]) / omega1;
        }

        double mu2 = 0;
        for (int i = t + 1; i < 256; i++) {
            mu2 += (i * p[i]) / omega2;
        }

        double var = omega1 * omega2 * (mu1 - mu2) * (mu1 - mu2);
        if (var > max_var) {
            max_var = var;
            threshold = t;
        }
    }

	return threshold;
}
// int cv_detect::thres_YEN(cv::Mat gray_img){
//     cv::cvtColor(gray_img, gray_img, cv::COLOR_BGR2GRAY);

//     // 将像素值偏移
//     gray_img = gray_img;

//     int histogram[256] = {0};
//     for (int i = 0; i < gray_img.rows; i++) {
//         for (int j = 0; j < gray_img.cols; j++) {
//             int intensity = (int)gray_img.at<uchar>(i, j);
//             histogram[intensity]++;
//         }
//     }

//     double p[256] = {0.0};
//     int num_pixels = gray_img.rows * gray_img.cols;

//     // 调整前景和背景权重
//     double w1 = 1.0, w2 = 1.0;
//     for (int i = 0; i < 250; i++) {
//         p[i] = (double)histogram[i] / num_pixels;
//     }
//     for (int i = 250; i < 256; i++) {
//         p[i] = (double)histogram[i] / num_pixels;
//         w1 += p[i];
//         w2 -= p[i];
//     }

//     double c[256] = {0.0};
//     c[0] = p[0];
//     for (int i = 1; i < 256; i++) {
//         c[i] = c[i - 1] + p[i];
//     }

//     double max_var = std::numeric_limits<double>::min();
//     int threshold = 0;

//     for (int t = 0; t < 256; t++) {
//         double omega1 = 0, omega2 = 0;
//         double mu1 = 0, mu2 = 0;
//         double var = 0;

//         // 调整前景和背景权重
//         for (int i = 0; i <= t; i++) {
//             if (i >= 250) {
//                 omega1 += p[i] * w1;
//                 mu1 += i * p[i] * w1;
//             } else {
//                 omega1 += p[i];
//                 mu1 += i * p[i];
//             }
//         }

//         for (int i = t + 1; i < 256; i++) {
//             if (i >= 50) {
//                 omega2 += p[i] * w2;
//                 mu2 += i * p[i] * w2;
//             } else {
//                 omega2 += p[i];
//                 mu2 += i * p[i];
//             }
//         }

//         // 计算前景和背景像素的平均值
//         mu1 /= omega1 + 1e-30;
//         mu2 /= omega2 + 1e-30;

//         // 调整类间方差公式
//         if (mu1 >= 50 && mu2 >= 50) {
//             var = omega1 * omega2 * (mu1 - mu2) * (mu1 - mu2);
//         }

//         if (var > max_var) {
//             max_var = var;
//             threshold = t;
//         }
//     }

//     return threshold + 50;   // 将阈值还原到原始范围
// }
float cv_detect::cont_mean_gv(vector<vector<cv::Point>> contours, cv::Mat src){
	float sum_gv = 0; // 计数异常plug个数
	int mean_gv = 0;


    for (size_t i = 0; i < contours.size(); i++) {
        cv::Mat mask = cv::Mat::zeros(src.size(), CV_8UC1);
        drawContours(mask, contours, (int)i, cv::Scalar(255), -1);
        cv::Scalar mean_gray = cv::mean(src, mask);
		// LOG_INFO("{}  mean_gray = {}", i, mean_gray[0]);
		sum_gv += mean_gray[0];
    }

	return sum_gv/contours.size();
}

int cv_detect::plug_gv_abnormal(vector<vector<cv::Point>> contours, cv::Mat src) {
    int abnormalCount = 0; // 计数异常plug个数
    float threshold_gray = 130.0; // 灰度

    for (size_t i = 0; i < contours.size(); i++) {
        cv::Mat mask = cv::Mat::zeros(src.size(), CV_8UC1);
        drawContours(mask, contours, (int)i, cv::Scalar(255), -1);
		// cv::imshow("mask", mask);
		// cv::imshow("srcc", src);
		// cv::waitKey(0);
        cv::Scalar mean_gray = cv::mean(src, mask);
		LOG_INFO("{}  mean_gray = {}", i, mean_gray[0]);
        if (mean_gray[0] < threshold_gray) {
            abnormalCount++;
        }
    }

    // 返回gv异常plug个数
    return abnormalCount;
}

cv::Mat cv_detect::preproces_petal(cv::Mat gray, int default_ = 0){
	cv::Mat thresh;
	int thr;
    cv::Mat kernel = cv::getGaussianKernel(60, 7);
    // Mat result;
    cv::filter2D(gray, gray, -1, kernel);
	// cv::imshow("Gauss", gray);
	// cv::medianBlur(gray, gray, 17);
	// cv::imshow("gray", gray);
	// cv::equalizeHist(gray, gray);
	// cv::imshow("src", gray);

	double meanGrayValue = cv::mean(gray)[0];
	LOG_INFO("this plug GVmean = {}", meanGrayValue);
	int thr_gray = ((int(meanGrayValue) + thres_YEN(gray) + default_) > 250) ? 240 : (int(meanGrayValue) + thres_YEN(gray) + default_);
	LOG_INFO("PLUG threshold value = {}   meanGrayValue{}  thres_YEN{}   diff{} ", thr_gray, int(meanGrayValue), thres_YEN(gray), default_);
	cv::threshold(gray, gray, thr_gray, 255, cv::THRESH_BINARY);
	// cv::imshow("THRESH_BINARY", gray);

	write_debug_img("E:\\project\\BL\\dataset\\BL_DEBUG\\PLUG_src_bfilter.jpg", gray, FIND_PLUG);

	cv::Mat struct_kernel_a = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Point(26, 26), cv::Point(-1, -1));//13
	
	// cv::erode(gray, thresh, struct_kernel_a);
	cv::morphologyEx(gray, thresh, cv::MORPH_OPEN, struct_kernel_a, cv::Point(-1, -1));
	// cv::
	write_debug_img("E:\\project\\BL\\dataset\\BL_DEBUG\\PLUG_filterA.jpg", thresh, FIND_PLUG);
	// cv::imshow("erode", thresh);
	
    std::vector<std::vector<cv::Point> > contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(thresh, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);
    
    // 遍历所有轮廓
    for(size_t i=0; i<contours.size(); i++) {
        // 计算轮廓面积
        double area = cv::contourArea(contours[i]);
		// LOG_INFO("area = {}", area);
        if(area < 6000) {
            // 填充轮廓为黑色
            cv::drawContours(thresh, contours, (int)i, cv::Scalar(0, 0, 0), cv::FILLED);
        }
		
    }
	// cv::Mat struct_kernel_b = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Point(15, 15));//15
	// cv::morphologyEx(thresh, thresh, cv::MORPH_DILATE, struct_kernel_b, cv::Point(-1, -1), 1);
	// cv::imshow("MORPH_DILATE", thresh);
	write_debug_img("E:\\project\\BL\\dataset\\BL_DEBUG\\PLUG_filterB.jpg", thresh, FIND_PLUG);
	// cv::waitKey(0);
	/////////////////////////////////////////////
	write_debug_img("E:\\project\\BL\\dataset\\BL_DEBUG\\PLUG_src.jpg", gray, FIND_PLUG);
	write_debug_img("E:\\project\\BL\\dataset\\BL_DEBUG\\PLUG_ERODE.jpg", thresh, FIND_PLUG);
#ifdef DEBUG_SHOW_
cv::namedWindow("src", cv::WINDOW_NORMAL);
cv::namedWindow("erode", cv::WINDOW_NORMAL);
cv::imshow("src", gray);
cv::imshow("erode", thresh);
cv::waitKey(0);
#endif
	
	return thresh;
}