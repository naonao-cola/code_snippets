#pragma once
#include "resmodel_procesing.h"
#include <string>
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>


ONNXClassifier::ONNXClassifier():	//input_channels[i] -= cv::Scalar(0.485, 0.456, 0.406);  // 减去均值0.225, 0.206, 0.156
    								//input_channels[i] /= cv::Scalar(0.229, 0.224, 0.225);  // 除以标准差0.929, 0.924, 0.925    0.229, 0.224, 0.225
                            //    default_mean(0.485, 0.456, 0.406),
							default_mean(0.285, 0.256, 0.206),
							default_std(0.529, 0.524, 0.525),
							input_size(_input_size),
							labels{"NG", "OK"} {

}
ONNXClassifier::~ONNXClassifier() {

}


#include <fstream>

void writeToONNXFile(const char* buffer, size_t sizeBuffer, const std::string& filename) {
    // 打开文件流以写入到.onnx文件中
    std::ofstream ofs(filename, std::ios::binary | std::ios::out);
    if (ofs.is_open()) {
        // 将缓冲区中的数据写入到文件流中
        ofs.write(buffer, sizeBuffer);

        // 关闭文件流
        ofs.close();
    }
}

// // 使用示例：
// const char* buffer;
// size_t sizeBuffer;
// std::string filename = "model.onnx";
// writeToONNXFile(buffer, sizeBuffer, filename);



bool ONNXClassifier::init_model( const char* buffer, size_t sizeBuffer, cv::Size _input_size) {
	// if (!read_labels(label_path))
	// {
	// 	LOG_ERROR("{} label read fail!", label_path);
	// 	return false;
	// 	// throw std::runtime_error("label read fail!");
	// }
	this->input_size = _input_size;
	try
	{
		/* code */

		// resnet18 = cv::dnn::readNet(model_path);
		resnet18 = cv::dnn::readNetFromONNX(buffer, sizeBuffer);

		LOG_INFO("model load success...");
		// char* blobptr =
		// resnet18 = cv::dnn::readNetFromONNX("D:\\Ronnie\\pakges\\bldataset\\station2_model\\O\\O_RING.onnx");

		resnet18.setPreferableBackend(cv::dnn::DNN_BACKEND_DEFAULT);
		resnet18.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);

		// std::string filename = "model.onnx";
		// writeToONNXFile(buffer, sizeBuffer, filename);
		return true;
	}
	catch(const std::exception& e)
	{
		LOG_ERROR("{} load model fail! {}", model_path, e.what());
		return false;
	}
}
bool ONNXClassifier::read_labels(const std::string& label_path)
{
	std::ifstream ifs(label_path);
	if ( !ifs.is_open() ) {
		return false;
	}
	// assert(ifs.is_open());
	std::string line;
	while (std::getline(ifs,line))
	{
		std::size_t index = line.find_first_of(':');
		labels.push_back(line.substr(index + 1));
		// std::cout << line.substr(index + 1) << std::endl;
	}
	if (labels.size() > 0)
		return true;
	else
		return false;
}
bool ONNXClassifier::gv_abnormal(cv::Mat img, int threshold_low_value, int threshold_high_value ) {

	// cv::Mat input_image = cvimread(img,cv::IMREAD_GRAYSCALE);

	cv::Mat input_image = img.clone();
	// cv::imshow("input", input_image);
	// cv::waitKey(0);
	LOG_INFO("img chanel = {}", input_image.channels());
	cv::cvtColor(input_image, input_image, cv::COLOR_BGR2GRAY);
	// cv::imshow("input_image", input_image);
	// cv::waitKey(0);
	if (input_image.empty()) 			  {return false;}
	// 计算平均灰度值
	cv::Scalar /* The above code is a comment in C++ programming language. It is not doing anything, as
	comments are ignored by the compiler and are used to provide explanations or
	documentation about the code. */
	mean = cv::mean(input_image);


	// int threshold_high_value = 240; // 根据实际情况调整阈值
	// double threshold_low_value =  50; // 根据实际情况调整阈值
	dl_check.gv_value[0] = mean[0];
	dl_check.gv_value[1] = threshold_low_value;
	dl_check.gv_value[2] = threshold_high_value;
	if( mean[0] > threshold_high_value ||  mean[0] < threshold_low_value )  {
		dl_check.check_value = "ABNORMAL_IMAGE";
		LOG_INFO("gv abnormal!!!");
		return false;
		}
	// if( mean[0] < threshold_low_value  )  {
	// 	dl_check.check_value = mean[0];
	// 	dl_check.check_value = "ABNORMAL_IMAGE";
	// 	LOG_INFO("gv abnormal!!!240");
	// 	return false;
	// 	}

	return true;
}

void ONNXClassifier::updateImgCheck(img_check& img) {
        // 在类内部修改 img_check 变量的值
        img.check_value = dl_check.check_value;
        img.gv_value[0] = dl_check.gv_value[0];
        img.gv_value[1] = dl_check.gv_value[1];
        img.gv_value[2] = dl_check.gv_value[2];
}

void ONNXClassifier::preprocess_input(cv::Mat& input_image, DebugType cls)//图像预处理
{

	// cv::resize(input_image, input_image, cv::Size(450, 450));
	cv::Mat img_process;
	switch (cls)
	{
	case FIND_O_RING:
		/* code */
		input_image = BL_IMG_PROCESS(input_image, "E:\\project\\BL\\dataset\\BL_DEBUG\\TEST_O.jpg");
		// LOG_INFO("Start convertTo....");
		// input_image.convertTo(input_image, CV_32F,1.0/255.0);
		// input_image = cv::imread("D:\\Ronnie\\pakges\\bldataset\\pytest\\O_OK_process\\CAM-7b755d72c85441b2828fbf87cada2bf7_20230503132526388.bmp");
		// cv::resize(input_image, input_image, cv::Size(450, 450));
		break;
	case FIND_C_RING:
		/* code */
			input_image = BL_IMG_PROCESS(input_image, "E:\\project\\BL\\dataset\\BL_DEBUG\\TEST_C.jpg");
			LOG_INFO("Start convertTo....");
			    // 转换为 float32 数据类型

			// input_image.convertTo(input_image, CV_32F,1.0/255.0);
		// cv::resize(input_image, input_image, cv::Size(450, 450));
		break;
	default:
		break;
	}
	// input_image.convertTo(input_image, CV_32F,1.0/255.0);
	// cv::cvtColor(input_image, input_image, cv::COLOR_BGR2RGB);
	// LOG_INFO("Start copy image....");
	// LOG_INFO("Start resize image....");
	// cv::resize(image, image, cv::Size(0, 0), 0.75, 0.75);

	// cv::Mat image_float;
	// input_image.convertTo(image_float, CV_32F);

	// // 计算图像的均值和标准差
	// cv::Scalar mean, stddev;
	// meanStdDev(input_image, mean, stddev);

	// // 使用计算得到的均值和标准差对图像进行归一化
	// cv::Mat normalized_image;
	// cv::normalize(input_image, input_image, 0.0, 1.0, cv::NORM_MINMAX);
	// cv::normalize(input_image, input_image, 1.0, -mean[0]/stddev[0], cv::NORM_MINMAX);
	// input_image = normalized_image;
	switch (cls)
	{
	case DebugType::FIND_O_RING:
		LOG_INFO("Start write_debug_img....FIND_O_RING--{}",FIND_O_RING);
		write_debug_img("E:\\project\\BL\\dataset\\BL_DEBUG\\net_o_resize.jpg", input_image, FIND_O_RING);
		// write_debug_img("E:\\project\\BL\\dataset\\BL_DEBUG\\net_o_conv1.jpg", image, FIND_O_RING);

		// cv::subtract(input_image,default_mean,input_image);
		// write_debug_img("E:\\project\\BL\\dataset\\BL_DEBUG\\net_o_subtract.jpg", input_image, FIND_O_RING);

		// cv::divide(input_image, default_std, input_image);
		// write_debug_img("E:\\project\\BL\\dataset\\BL_DEBUG\\net_o_divide.jpg", input_image, FIND_O_RING);
		break;
	case DebugType::FIND_C_RING:
		LOG_INFO("Start write_debug_img....FIND_C_RING--{}",FIND_C_RING);
		write_debug_img("E:\\project\\BL\\dataset\\BL_DEBUG\\net_c_resize.jpg", input_image, FIND_C_RING);
		// write_debug_img("E:\\project\\BL\\dataset\\BL_DEBUG\\net_c_conv1.jpg", image, FIND_C_RING);

		// cv::subtract(input_image,default_mean,input_image);
		// write_debug_img("E:\\project\\BL\\dataset\\BL_DEBUG\\net_c_conv2.jpg", input_image, FIND_C_RING);

		// cv::divide(input_image, default_std, input_image);
		// write_debug_img("E:\\project\\BL\\dataset\\BL_DEBUG\\net_c_conv3.jpg", input_image, FIND_C_RING);
	default:
		break;
	}
	LOG_INFO("Start convertTo....");
	// input_image.convertTo(input_image, CV_32F,1.0/255.0);
	// image.convertTo(input_image, CV_32F, 1.0/255.0);
	return;
}

#include <cmath>
#include <vector>

std::vector<float> softmax(std::vector<float> logits) {
  std::vector<float> probs(logits.size());
  float exp_sum = 0.0;
  for (float logit : logits) {
    exp_sum += std::exp(logit);
  }
  for (int i = 0; i < logits.size(); ++i) {
    probs[i] = std::exp(logits[i]) / exp_sum;
  }
  return probs;
}

void ONNXClassifier::ai_normalize(cv::Mat *im, const std::vector<float> &mean, const std::vector<float> &scale, const bool is_scale)
{
  double e = 1.0;
  if (is_scale) {
    e /= 255.0;
  }
  (*im).convertTo(*im, CV_32FC3, e);
  for (int h = 0; h < im->rows; h++)
  {
    for (int w = 0; w < im->cols; w++)
    {
      im->at<cv::Vec3f>(h, w)[0] =
          (im->at<cv::Vec3f>(h, w)[0] - mean[0]) / scale[0];
      im->at<cv::Vec3f>(h, w)[1] =
          (im->at<cv::Vec3f>(h, w)[1] - mean[1]) / scale[1];
      im->at<cv::Vec3f>(h, w)[2] =
          (im->at<cv::Vec3f>(h, w)[2] - mean[2]) / scale[2];
    }
  } // end for
} // end ai_normalize


float ONNXClassifier::Classify(const cv::Mat&  input_image, std::string& out_name, DebugType cls)
{
	probs.clear();
	out_name.clear();
	// out_name = "";
	LOG_INFO("Clear out name Done!  out_name: {}", out_name);
	if (input_image.empty()) {
		LOG_ERROR("Input image is empty!");
		return 0.0;
	} else {
		LOG_INFO("Input Image Cols: {}, Rows: {}, Channels: {}", input_image.cols, input_image.rows, input_image.channels());
	}
	cv::Mat image;
	try {
		image = input_image;
	} catch (const std::exception& e) {
		LOG_ERROR("input_image.clone failed, {}", e.what());
		return 0.0;
	}
	LOG_INFO("Start preprocess_input!");

	preprocess_input(image, cls);

	cv::resize(image, image, cv::Size(), 0.25, 0.25);
	// cv::imshow("processimg", image);
	// cv::waitKey(0);

	// cv::Mat blob = cv::dnn::blobFromImage(image, 1.0f, cv::Size(image.cols, image.rows), cv::Scalar(0.0, 0.0, 0.0), false, false);
	// resnet18.setInput(blob);
	// cv::Mat output = resnet18.forward();
	// std::cout << "==================================" << output << std::endl;
	LOG_INFO("succeed img preprocess....");
	// if( !gv_abnormal(image, 10, 240)) {

	// 	out_name = labels[0];
	// 	return 1;
	// }
	cv::Size inp = image.size();
	LOG_INFO("input img size: {}, {}", inp.height, inp.width);
	// cv::Mat input_blob = cv::dnn::blobFromImage(image, 1.0, input_size, cv::Scalar(0, 0, 0), true);
	////fix
	// 定义输入图像、缩放因子和输出 blob 的大小
	// cv::Mat image =;                   // 输入图像
	double scalefactor = 1.0 / 255.0;      // 缩放因子
	cv::Size size(224, 224);               // 输出 blob 的大小

	// 定义要减去的均值 (可选)
	cv::Scalar mean(0, 0, 0);  // 要减去的均值

	// 定义是否交换 R 与 B 通道 (可选)
	bool swapRB = true;                    // 是否交换 R 和 B 通道

	// 定义是否剪裁输入图像 (可选)
	bool crop = false;                     // 是否剪裁输入图像

	// 定义输出 blob 的数据类型 (可选)
	int dtype = CV_32F;                    // 输出 blob 的数据类型为 32 位浮点数

	// 创建 blob
	// cv::Mat input_blob = cv::dnn::blobFromImage(
	// 	image,          // 输入图像
	// 	scalefactor,    // 缩放因子
	// 	size,           // 输出 blob 的大小
	// 	mean,           // 要减去的均值
	// 	swapRB,         // 是否交换 R 和 B 通道
	// 	crop,        // 是否剪裁输入图像
	// 	dtype           // 输出 blob 的数据类型
	// );


	/////////
	cv::Mat input_blob;

	//O224bug
	cv::Mat input_blob_O_450;
	std::vector<float> means;
	std::vector<float> stds;
	switch (cls)
	{
	case FIND_O_RING:
			// 创建 blob
		// input_blob = cv::dnn::blobFromImage(
		// 	image,          // 输入图像
		// 	scalefactor,    // 缩放因子
		// 	size,           // 输出 blob 的大小
		// 	mean,           // 要减去的均值
		// 	swapRB,         // 是否交换 R 和 B 通道
		// 	crop        // 是否剪裁输入图像
		// 	// dtype           // 输出 blob 的数据类型
		// );
		// means = {0.11, 0.11, 0.11};
		// stds = {0.82, 0.82, 0.82};
		means = {0.08, 0.08, 0.08};
		stds = {1.45, 1.45, 1.45};
		ai_normalize(&image, means, stds, true);
		input_blob = cv::dnn::blobFromImage(image, 1, inp, cv::Scalar(), true, true);
		input_blob_O_450 = cv::dnn::blobFromImage(
			image,          // 输入图像
			scalefactor,    // 缩放因子
			cv::Size(450,450),           // 输出 blob 的大小
			mean,           // 要减去的均值
			swapRB,         // 是否交换 R 和 B 通道
			crop        // 是否剪裁输入图像
			// dtype           // 输出 blob 的数据类型
		);

		break;
	case FIND_C_RING:
		means = {0.08, 0.08, 0.08};
		stds = {1.65, 1.65, 1.65};
		ai_normalize(&image, means, stds, true);
		input_blob = cv::dnn::blobFromImage(image, 1, inp, cv::Scalar(), true, true);
		break;
	default:
		break;
	}
	// cv::Mat input_blob = cv::dnn::blobFromImage(image, 1, inp);

	LOG_INFO("succeed blob preprocess....    Image Cols: {}, Rows: {}, Channels: {}", image.cols, image.rows, image.channels());
	resnet18.setInput(input_blob);
	LOG_INFO("succeed setInput preprocess....");
	const std::vector<cv::String>& out_names = resnet18.getUnconnectedOutLayersNames();
	LOG_INFO("out_name = {}", out_names[0]);
	LOG_INFO("succeed LayersNames preprocess....");
	// std::cout << out_names[2] << std::endl;
	cv::Mat out_tensor;
	try
    {
		// out_tensor = resnet18.forward(out_names[0]);
		out_tensor = resnet18.forward();
		// out_tensor = resnet18.forward();

		// Forward pass
		// cv::Mat prob = resnet18.forward();

		// Print classification results
		// std::vector<float> probVec(prob.ptr<float>(), prob.ptr<float>() + prob.size[1]);
		// int classId = std::distance(probVec.begin(), std::max_element(probVec.begin(), probVec.end()));
		// float confidence = probVec[classId];
		// std::cout << "Class ID: " << classId << std::endl;
		// std::cout << "Confidence: " << confidence << std::endl;
    }
    catch(const std::exception& e)
    {
        // std::cerr << e.what() << '\n';
        LOG_INFO("net forward error!!! {}", e.what());
		return 0.0;
		// assert(false);

    }

	LOG_INFO("succeed forward preprocess....");
	int class_id;
	float accuracy = 0;
	std::cout << out_tensor << std::endl;
	// LOG_INFO("Confidence level = {}", out_tensor);
	std::vector<float> logits;
	logits.assign((float*)out_tensor.ptr(), (float*)out_tensor.ptr()+out_tensor.cols);
	probs = softmax(logits);
	for (int i = 0; i < probs.size(); i++)
	{
		// if (probs[i] > accuracy) {
		// 	accuracy = probs[i];
		// 	out_name = labels[i];
		// }
    	std::cout << "Probability of class " << i << ": " << probs[i] << std::endl;
		LOG_INFO("Probability of class {} : {}", i, probs[i]);
	}
	//置信度判断
	if (probs[0] > bl_dl_pram.score) {
		out_name = labels[0];
		accuracy = probs[0];
	}
	else {
		out_name = labels[1];
		accuracy = probs[1];
	}


	// if (out_name == "NG" && cls == FIND_O_RING) {
	// 	resnet18.setInput(input_blob_O_450);
	// 	out_tensor = resnet18.forward();
	// 	logits.assign((float*)out_tensor.ptr(), (float*)out_tensor.ptr()+out_tensor.cols);
	// 	probs = softmax(logits);
	// 	for (int i = 0; i < probs.size(); i++)
	// 	{
	// 		std::cout << "Probability of class " << i << ": " << probs[i] << std::endl;
	// 		LOG_INFO("Probability of class {} : {}", i, probs[i]);
	// 	}
	// 	//置信度判断
	// 	if (probs[0] > 0.9) {
	// 		out_name = labels[0];
	// 		accuracy = probs[0];
	// 	}
	// 	else {
	// 		out_name = labels[1];
	// 		accuracy = probs[1];
	// 	}
	// }

	// out_name = labels[class_id];
		LOG_INFO("succeed label class preprocess....");


		return accuracy;
}


cv::Mat ONNXClassifier::BL_IMG_PROCESS(cv::Mat img, std::string save_path) {
	try {
		// cv::Mat img = cv::imread(img_path);

		cv::Mat hsv;
		cv::cvtColor(img, hsv, cv::COLOR_BGR2HSV);
		cv::Scalar lower_yellow(15, 100, 30);
		cv::Scalar upper_yellow(45, 255, 255);
		cv::Mat mask;
		cv::inRange(hsv, lower_yellow, upper_yellow, mask);
		cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
		cv::Mat eroded;
		cv::erode(mask, eroded, kernel);
		cv::Mat kernel1 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(37, 37));
		cv::Mat dilated;
		cv::dilate(eroded, dilated, kernel1);

		std::vector<std::vector<cv::Point> > contours;
		std::vector<cv::Vec4i> hierarchy;
		cv::findContours(dilated, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
		std::vector<cv::Point> contoursA;
		int max_area = 0;
		for (int i = 0; i < contours.size(); i++) {
			double area_A = cv::contourArea(contours[i]);
			// std::cout << "area_A = " << area_A << std::endl;
			if (area_A > max_area) {
				max_area = area_A;
				contoursA = contours[i];
			}
		}

		cv::RotatedRect rect = cv::minAreaRect(contoursA);
		cv::Point2f box[4];
		rect.points(box);

		int width = (int)rect.size.width;
		int height = (int)rect.size.height;
		cv::Point2f center = rect.center;
		cv::Mat rotated_img = cv::getRotationMatrix2D(center, rect.angle, 1.0);
		cv::Mat cropped;
		cv::warpAffine(img, cropped, rotated_img, img.size(), cv::INTER_LINEAR);

		cv::getRectSubPix(cropped, cv::Size(width, height), center, cropped);
		cv::resize(cropped, cropped, cv::Size(450, 450));
		// cv::imshow("dilated", cropped);
		// cv::waitKey(0);
		cv::imwrite(save_path, cropped);
		// std::cout << "img_save suc!" << std::endl;
		LOG_INFO("img process successful!");
		return cropped;
	}
	catch (...) {
		std::cout << "###Warning: img process fail!" << std::endl;
		int height = img.rows;
		int width = img.cols;

		// 定义缩小的边缘范围
		int border_size_x = 421;
		int border_size_y = 194;

		// 缩小边缘
		cv::Rect roi(border_size_x, border_size_y, width - 2 * border_size_x, height - 2 * border_size_y);
		cv::Mat img_borderless = img(roi);
		cv::resize(img_borderless, img_borderless, cv::Size(450, 450));
		return img_borderless;
	}
}