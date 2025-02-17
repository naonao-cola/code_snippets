
#include <windows.h>
#include "../../modules/tv_algo_base/src/framework/InferenceEngine.h"
#include "../../modules/tv_algo_base/src/utils/logger.h"
#include "FrontPinDetect.h"
#include "../../modules/tv_algo_base/src/utils/Utils.h"
#include <vector>
#include <cmath>
#include "algo_tool.h"
#include <memory>
#include "img_feature.h"

#if USE_AI_DETECT
#    include <AIRuntimeDataStruct.h>
#    include <AIRuntimeInterface.h>
#    include <AIRuntimeUtils.h>
#endif   // USE_AI_DETECT

REGISTER_ALGO(FrontPinDetect)

#define DEBUG_ENABLE 0
#define PI 3.14159265358979323846

static const bool ENABLE_COORD_ADJ_DEFAULT = true;

FrontPinDetect::FrontPinDetect()
{
    selectModel["rxwm_notline"]     = productNames::rxwm_notline;
    selectModel["rxwm_small"]       = productNames::rxwm_small;
    selectModel["rxzg_151"]         = productNames::rxzg_151;
    selectModel["rxzg_131K"]        = productNames::rxzg_131K;
    selectModel["rxzg_103A"]        = productNames::rxzg_130A;
    selectModel["sbzg_1851408151"]  = productNames::sbzg_1851408151;
    selectModel["lc_189"]			= productNames::lc_189;
    selectModel["lc_190"]			= productNames::lc_190;
    selectModel["aizg_pw_6"]        = productNames::aizg_pw_6;
    selectModel["aizg_p2"]          = productNames::aizg_p2;
    selectModel["aizg_jian"]        = productNames::aizg_jian;
    selectModel["aizg_yuan"]        = productNames::aizg_yuan;
}

FrontPinDetect::~FrontPinDetect()
{

}

AlgoResultPtr FrontPinDetect::RunAlgo(InferTaskPtr task, std::vector<AlgoResultPtr> pre_results)
{
    LOGI("FrontPinDetect run start!");
	AlgoResultPtr algo_result = std::make_shared<stAlgoResult>();
	algo_result->status = RunStatus::OK;
	json params = GetTaskParams(task);
		
	cv::Mat img1_gray, img2_gray;
    cv::Mat taskImg1, taskImg2;
    taskImg1 = task->image.clone();
    taskImg2 = task->image2.clone();

	cv::cvtColor(taskImg1, img1_gray, cv::COLOR_BGR2GRAY);       // 同轴光图片
    cv::cvtColor(taskImg2, img2_gray, cv::COLOR_BGR2GRAY);   // 平行光图片
	double angle = 0;
	cv::Mat roi_mask;
	//
	int roi_lt_x = params["param"]["roi_lt_x"];
	int roi_lt_y = params["param"]["roi_lt_y"];
	int roi_width = params["param"]["roi_width"];
	int roi_height = params["param"]["roi_height"];

	detect_lt_x = roi_lt_x;
    detect_lt_y = roi_lt_y;

	// 测试使用
	//roi_lt_x = 760;
	//roi_lt_y = 2664;
	//roi_width = 1920;
	//roi_height = 280;

	cv::Rect roi_bbox(roi_lt_x, roi_lt_y, roi_width, roi_height);
	img1_gray = img1_gray(roi_bbox);
	img2_gray = img2_gray(roi_bbox);

	
    cv::Mat drawImage               = cv::Mat::zeros(taskImg1.size(), taskImg1.type());
	bool enable_save_resultImage = params["param"]["enable_save_resultImage"];
	if (enable_save_resultImage) {
        taskImg1.copyTo(drawImage);

		std::string rImage_Ori = "D:/DebugResult/rImage_Ori";
		bool state = check_folder_state(rImage_Ori);
		if (state) {
            cv::imwrite(rImage_Ori + "/" + std::string(task->image_info["img_name"]) + "_white.png", taskImg1);
            cv::imwrite(rImage_Ori + "/" + std::string(task->image_info["img_name"]) + "_red.png", taskImg2);
		}
	}

	//*************************************************//

	cv::Mat bin_img;
	std::vector<PinInfoPtr> pin_infos;
	// 通过平行光图片（红色斑点）粗略定位针尖，数量可能比实际针尖多，只要有红色斑点都认为可能存在针尖
	// 后面可以考虑用目标检测或者模板匹配等其他方法进行粗定位
	FirstPassLocate(img2_gray, img1_gray, params["param"], bin_img, pin_infos, std::string(task->image_info["img_name"]));

	// 保存红图Blob定位的结果图
	if (enable_save_resultImage) {
		std::string rImage_FirstLocate = "D:/DebugResult/rImage_FirstLocate";
		bool state = check_folder_state(rImage_FirstLocate);
		if (state) {
			cv::imwrite(rImage_FirstLocate + "/" + std::string(task->image_info["img_name"]) + "_blob.png", bin_img);
		}
	}

	// 红白图对应
   /* cv::Mat temp1, temp2;
    cv::bitwise_and(img1_gray, bin_img, temp1);*/

	bool enable_align = Tival::JsonHelper::GetParam(params["param"], "enable_align", false);
    if (enable_align) {
        bin_img = AlignTransform(img1_gray, bin_img, pin_infos, params);
    }
    //cv::bitwise_and(img1_gray, bin_img, temp2);

	// 在上一步粗定位基础上，通过同轴光图片（Image1）查找针尖准确位置（针尖区域反光亮斑中心）
	FindPinByImage1(img1_gray, bin_img, params["param"], pin_infos, drawImage, roi_lt_x, roi_lt_y);


	// 保存白图Pin针定位的结果图
	if (enable_save_resultImage) {
		// 测试用：将pin针位置（图像坐标）画到原始图片上
		DrawOrgResults(drawImage, pin_infos);
		std::string wImage_PinLocate = "D:/DebugResult/wImage_PinLocate";
		bool state = check_folder_state(wImage_PinLocate);
		if (state) {
			cv::imwrite(wImage_PinLocate + "/" + std::string(task->image_info["img_name"]) + "_pin.png", drawImage);
		}
	}

	//计算通过FindPinByImage1找到多少个针尖
	int found_cnt = 0;
	int found_cnt_final = 0;
	for (auto pin_info : pin_infos) {
		if (pin_info->found) {
			found_cnt += 1;
		}
	}

	// 判断是否无产品
    algo_result->result_info = json::array();
    int expect_pin_count =
        params["param"]["x_coords1"].size() * params["param"]["y_coords1"].size() +
        params["param"]["x_coords2"].size() * params["param"]["y_coords2"].size();
    if (found_cnt < expect_pin_count / 3 * 2) {
        algo_result->judge_result = 0;
        LOGI("Skip no red pin, return result:{}", algo_result->judge_result);
        return algo_result;
    }

	// 根据针尖位置分布拟合一个以左上角为原点，X向右Y向下的坐标系，得到原图坐标点到针尖坐标系的转换矩阵
	cv::Mat M = GetPinCSTransMat(pin_infos, params["param"], angle, drawImage);

	if (M.empty()) {
		//LOGE("Fit Line fail!");
		algo_result->status = RunStatus::ABNORMAL_IMAGE;
		return algo_result;
	}

	_offset_value_x_ = Tival::JsonHelper::GetParam(params["param"], "offset_value_x", 0.0);
	_offset_value_y_ = Tival::JsonHelper::GetParam(params["param"], "offset_value_y", 0.0);

	_scale_value = Tival::JsonHelper::GetParam(params["param"], "scale_value", 1.0);


	double adj_x, adj_y;
	// 基于建立的pin针坐标系，计算每个pin针X\Y方向的偏差值
	// 输出参数adj_x/adj_y为坐标系修正值，目的是为了平移坐标系，使得尽可能多的pin针能框进网格内
	float differ_std = -1;
	float differ_measure = -1;
	json pin_results = CalcPinResults(pin_infos, M, img1_gray, img2_gray, bin_img, params["param"], adj_x, adj_y, roi_lt_x, roi_lt_y, differ_std, differ_measure);

	//json pin_results = CalcPinResults(pin_infos, M, params["param"], adj_x, adj_y);

	// 计算pin针X\Y方向参考（标准）线，用于前段展示
	json lines = CalcStdLines(M, params["param"], adj_x, adj_y, differ_std);
	//json lines = CalcStdLines(M, params["param"], adj_x, adj_y);

	// 将测量结果画到图上
	if (enable_save_resultImage)
	{
		DrawResults(drawImage, lines, pin_results);
		std::string wImage_CalcResult = "D:/DebugResult/wImage_CalcResult";
		bool state = check_folder_state(wImage_CalcResult);
		if (state) {
			cv::imwrite(wImage_CalcResult + "/" + std::string(task->image_info["img_name"]) + "_calc.png", drawImage);
		}
	}


	// 返回标线结果
	for (json std_line : lines) {
		json line_shape = {
			{"label", "std_line"},
			{"shapeType", "line"},
			{"points", std_line},
		};
		algo_result->result_info.push_back(line_shape);
	}



	bool enable_offset =Tival::JsonHelper::GetParam(params["param"], "enable_offset", false);
	float gap = Tival::JsonHelper::GetParam(params["param"],"gap", 0.0);

	//std::ofstream csv_data(R"(D:\csv\1.csv)");
 //   csv_data << "index" << "," << "org_x" << "," << "org_y" << ","
 //               << "measured_x" << "," << "measured_y" << "," << "x_off" << ","
 //               << "y_off" << "," << "std_x" << "," << "std_y" << ","
 //               << "is_ok" << "," << "TP" << std::endl;

	// 返回pin结果
	int judge_flag = 1;
	for (json pin_rst : pin_results)
	{
		json all_result = {
		   {"label", "result_ngok"},
		   {"shapeType", "image_ret"},
		   {"points",{pin_rst["pin_coord"]}},
		   {"result", {}
		   }
		};
        json pin_shape;

		if (enable_offset)
		{
		   //更新像素当量
          double ppum = params["param"]["ppum"];
		  reget_ppum(pin_rst["index"], ppum, params["param"], enable_offset, 1, 0);
          double diff_x = Mm2Px(differ_std, ppum);
		  reget_ppum(pin_rst["index"], ppum, params["param"], enable_offset, 0, 1);
          double diff_y = Mm2Px(gap, ppum);

		  pin_shape = {
				{"label", "pin_result"},
				{"shapeType", "point"},
				{"points",{pin_rst["pin_coord"]}},
				{"result", {
					{"is_ok", pin_rst["is_ok"]},
					{"x_off", pin_rst["x_off"]}, //x tp值
					{"y_off", pin_rst["y_off"]}, //y tp值
					{"std_x", float(pin_rst["std_x"]) - differ_std}, //理论值
					{"std_y", float(pin_rst["std_y"]) + gap},		//理论值
					{ "measured_x", float(pin_rst["measured_x"])  - differ_std},//实测值
					{ "measured_y", float(pin_rst["measured_y"]) + gap},
					{"index", pin_rst["index"]}, //索引
					{"TP", pin_rst["TP"]},  //TP值
					{"org_x", pin_rst["img_x"] - diff_x},  // TP值
					{"org_y", pin_rst["img_y"] + diff_y},  // TP值
					}
				}
			};
		}
		else{
			pin_shape = {
			{"label", "pin_result"},
			{"shapeType", "point"},
			{"points",{pin_rst["pin_coord"]}},
			{"result", {
				{"is_ok", pin_rst["is_ok"]},
				{"x_off", pin_rst["x_off"]}, //x tp值
				{"y_off", pin_rst["y_off"]}, //y tp值
				{"std_x", float(pin_rst["std_x"])}, //理论值
				{"std_y", float(pin_rst["std_y"])},		//理论值
				{ "measured_x", float(pin_rst["measured_x"])},//实测值
				{ "measured_y", float(pin_rst["measured_y"])},
				{"index", pin_rst["index"]}, //索引
				{"TP", pin_rst["TP"]},  //TP值
                {"org_x", pin_rst["img_x"] },  // TP值
                {"org_y", pin_rst["img_y"] },  // TP值
				}
			}
			};

			/*std::cout << pin_shape["result"]["index"] << ","
                                  << pin_rst["org_x"] << "," << pin_rst["org_y"]
                                  << ","
                                  << std::endl;

			std::cout << pin_shape["result"]["index"] << ","
                                  << float(pin_rst["org_x"]) * _scale_value +_offset_value<< ","
                            << float(pin_rst["org_y"]) * _scale_value + _offset_value<< ","<< std::endl;*/
		}
   //csv_data << pin_shape["result"]["index"] << ","
   //                      << pin_shape["result"]["org_x"] << ","
   //                      << pin_shape["result"]["org_y"] << ","
   //                      << float(pin_shape["result"]["measured_x"]) << ","
   //                      << pin_shape["result"]["measured_y"] << ","
   //                      << pin_shape["result"]["x_off"] << ","
   //                      << pin_shape["result"]["y_off"] << ","
   //                      << float(pin_shape["result"]["std_x"]) << ","
   //                      << pin_shape["result"]["std_y"] << ","
   //                      << pin_shape["result"]["is_ok"] << ","
   //                      << pin_shape["result"]["TP"]
   //                      << std::endl;

		json box_result = {
			{"label", "pin_result_fuzhu"},
			{"shapeType", "polygon"},
			{"points", pin_rst["std_box"]},
			{"result", pin_rst["result"]}
		};
		algo_result->result_info.push_back(pin_shape);
		algo_result->result_info.push_back(box_result);
		if (!pin_rst["is_ok"])
		{
			judge_flag = 0;
		}
	}

    //csv_data.close();

	algo_result->judge_result = judge_flag;
	LOGD("FrontPinDetect run finished!", );
	//LOGI("FrontPinDetect result {} !  ",  Utils::DumpJson(algo_result->result_info));
	return algo_result;
}

// 通过平行光图片（红色斑点）粗略定位针尖，数量可能比实际针尖多，只要有红色斑点都认为可能存在针尖
// 后面可以考虑用目标检测或者模板匹配等其他方法进行粗定位
void FrontPinDetect::FirstPassLocate(const cv::Mat& redImage, const cv::Mat& whiteImage, const json& params, cv::Mat& bin_image, std::vector<PinInfoPtr>& pin_infos, std::string image_name, bool draw_contours)noexcept
{
	int r_blob_area_min = params["r_blob_area_min"];    // 红斑最小面积
	int r_blob_area_max = params["r_blob_area_max"];    // 红斑最大面积
	int red_pin_gv_min = params["r_pin_gv_min"];        // 红斑最小亮度
	double bin_step = 10;                               // 二值化迭代阈值步幅
	bool enabel_checkblob = Tival::JsonHelper::GetParam(params, "enabel_checkblob", false);
	// 截取训练样本
	bool enable_save_cropImage = Tival::JsonHelper::GetParam(params, "enable_save_cropImage", false);

	// 加载形状匹配参数
    std::string productName = Tival::JsonHelper::GetParam(params, "product", std::string(""));
    bool enable_classifyModel = Tival::JsonHelper::GetParam(params, "enable_classifyModel", false);
    double      classifyConfThr      = Tival::JsonHelper::GetParam(params, "classifyConfThr", 0.5);
    int      batchSize      = Tival::JsonHelper::GetParam(params, "batchSize", 12);

	std::string modelName;
    switch (selectModel[productName]) {
    // -----------柔性弯母---------------//
    case productNames::rxwm_notline:
        svm_obj._train_size = cv::Size(80, 80);
        isInit              = svm_obj.init("./", "rxwm_notline.mdl");
        modelName           = "rxwm_notline.mdl";
        break;
    case productNames::rxwm_small:
        svm_obj._train_size = cv::Size(80, 80);
        isInit              = svm_obj.init("./", "rxwm_small.mdl");
        modelName           = "rxwm_small.mdl";
        break;
    // ------------柔性直公----------------//
    case productNames::rxzg_151:
        svm_obj._train_size = cv::Size(80, 80);
        isInit              = svm_obj.init("./", "rxzg_151.mdl");
        modelName           = "rxzg_151.mdl";
        break;
    case productNames::rxzg_131K:
        svm_obj._train_size = cv::Size(80, 80);
        isInit              = svm_obj.init("./", "rxzg_131K.mdl");
        modelName           = "rxzg_131K.mdl";
        break;
    case productNames::rxzg_130A:
        svm_obj._train_size = cv::Size(80, 80);
        isInit              = svm_obj.init("./", "rxzg_130A.mdl");
        modelName           = "rxzg_130A.mdl";
        break;
    // ------------设备直公----------------//
    case productNames::sbzg_1851408151:
        svm_obj._train_size = cv::Size(80, 80);
        isInit              = svm_obj.init("./", "sbzg_1851408151.mdl");
        modelName           = "sbzg_1851408151.mdl";
        break;
    // ------------lc----------------//
    case productNames::lc_189:
        svm_obj._train_size = cv::Size(80, 80);
        isInit              = svm_obj.init("./", "lc_189.mdl");
        modelName           = "lc_189.mdl";
        break;
    case productNames::lc_190:
        svm_obj._train_size = cv::Size(80, 80);
        isInit              = svm_obj.init("./", "lc_190.mdl");
        modelName           = "lc_190.mdl";
        break;
    // ------------aizg----------------//
    case productNames::aizg_pw_6:
        AI_modelIndex = 0;
        break;
    case productNames::aizg_p2:
        AI_modelIndex = 1;
        break;
    case productNames::aizg_jian:
        AI_modelIndex = 2;
        break;
    case productNames::aizg_yuan:
        AI_modelIndex = 3;
        break;

    default:
        isInit    = false;
        modelName = "no model";
        break;
    }

	LOGI("svm model init states {}, model name {} ", isInit, modelName);
	bin_image = cv::Mat::zeros(redImage.size(), CV_8UC1);

	cv::Mat temp_bin_img;
	std::vector<std::vector<cv::Point >> contorus;
	//返回红光图像的二值图和blob轮廓
	FilterNonPinBlobs(redImage, temp_bin_img, contorus, params, image_name);

	// 遍历红光图上预处理后的针尖blob
	//int saveCount = 0;
    int conCount = 0;
	for (auto out_cont : contorus)
	{
	/*	cv::Mat drawImage = redImage.clone();
		cv::Mat drawTmpBin = temp_bin_img.clone();*/

		cv::Rect bbox = cv::boundingRect(out_cont);
		cv::Mat sub_img = redImage(bbox);
		cv::Mat sub_tmp_bin = temp_bin_img(bbox);
		cv::Mat sub_whiteImg = whiteImage(bbox);
		// 在大Blob范围内迭代阈值，使得面积接近target_area

		// SVM的小图
		cv::Point cropCent;
		cropCent.x = bbox.x + bbox.width / 2.0;
		cropCent.y = bbox.y + bbox.height / 2.0;
		cv::Rect cropBox;
		cropBox.x = cropCent.x - 60;
		cropBox.y = cropCent.y - 60;
		cropBox.width = 120;
		cropBox.height = 120;
		// 判断
		if (cropBox.x < 0) {
			cropBox.x = 0;
		}
		if (cropBox.y < 0) {
			cropBox.y = 0;
		}
		if (cropBox.x + cropBox.width > whiteImage.size().width) {
			cropBox.width = whiteImage.size().width - cropBox.x;
		}
		if (cropBox.y + cropBox.height > whiteImage.size().height) {
			cropBox.height = whiteImage.size().height - cropBox.y;
		}
		cv::Mat sub_SvmImg = whiteImage(cropBox);

		std::string cropName, floderOKName, floderNGName;
		if (enable_save_cropImage) {
			// SVM样本收集
			std::string cropx = std::to_string(cropCent.x);
			std::string cropy = std::to_string(cropCent.y);
			cropName = image_name + "_" + cropx + "_" + cropy + ".jpg";
			floderOKName = "D:/cropImg/OK/";
			floderNGName = "D:/cropImg/NG/";
		}
		int thr = red_pin_gv_min;
		while (thr < 250)
		{
			cv::Mat sub_img_bin;
			std::vector<std::vector<cv::Point>> sub_contorus;
			cv::threshold(sub_img, sub_img_bin, thr, 255, cv::THRESH_BINARY);
			cv::Mat labels;
			int num_labels = cv::connectedComponents(sub_img_bin, labels, 8, CV_32S);

			// 进一步过滤：对blob形状、占空比等
			if (thr == red_pin_gv_min && enabel_checkblob) {
				//// 绘制位置
	/*			cv::Point cent;
				cent.x = int(bbox.x + bbox.width / 2.0);
				cent.y = int(bbox.y + bbox.height / 2.0);
				cv::circle(drawImage, cent, 50, (255), 2, 8);
				cv::circle(drawTmpBin, cent, 50, (255), 2, 8);*/

				bool blobFlag = CheckBlobByShape(sub_tmp_bin, sub_whiteImg, sub_SvmImg, out_cont, params);
				if (!blobFlag) {
					if (enable_save_cropImage) {
						cv::imwrite(floderNGName + cropName, sub_SvmImg);
					}
					break;
				}

				// 保存ok图
				if (enable_save_cropImage) {
					cv::imwrite(floderOKName + cropName, sub_SvmImg);
				}
			}
			double gv_sum_total = 0;
			int gv_num_total = 0;
			double area_total = 0;

			std::vector< BlobInfoPtr> tmpBlobList;
			std::vector<cv::Point> sel_cont_points;
			bool found_valid_blob = false;
			for (int label = 1; label < num_labels; ++label) {
				// 遍历每个连通域
				cv::Mat label_mask = (labels == label);
				std::vector<cv::Point> component_points;
				cv::findNonZero(label_mask, component_points);
				float blob_area = component_points.size();
				// 计算包围框面积
				cv::Rect sub_bbox = cv::boundingRect(component_points);
				// 如果面积符合条件
				if (component_points.size() > r_blob_area_min * 0.15) {
					sel_cont_points.insert(sel_cont_points.end(), component_points.begin(), component_points.end());
					// 进行其他操作...
					cv::Mat sub_cnt_img = sub_img(sub_bbox);
					cv::Mat sub_mask = sub_img_bin(sub_bbox);
					cv::Scalar sub_mean_gv = cv::mean(sub_cnt_img, sub_mask);

					// 平均亮度过小，填充0忽略掉
					if (sub_mean_gv[0] < 8) {
						cv::fillPoly(sub_img_bin, cv::Mat(component_points), 0);
						LOGD("Mean GV too small. ignore!");
						continue;
					}

					// 统计轮廓内部像素值，用于后面计算GV均值
					double sum_gv = 0;
					int total_num = 0;
					SumGV(sub_cnt_img, sub_mask, sum_gv, total_num);

					gv_sum_total += sum_gv;
					gv_num_total += total_num;
					area_total += blob_area;

					//tmpBlobList.emplace_back(std::make_shared<BlobInfo>(sub_cont, sub_area, sub_mean_gv[0], sum_gv, total_num, center));
				}
			}

			if (area_total >= r_blob_area_max * 0.8) {
				thr += bin_step;
				continue;
			}
			else if (area_total == 0)
			{
				break;
			}
			// 对特别大的红点直接腐蚀
			if (area_total > r_blob_area_max * 0.9) {
				cv::Mat temp_sub_img = cv::Mat::zeros(sub_img.size(), CV_8UC1);
				cv::fillPoly(temp_sub_img, sel_cont_points, 255);
				int morph_size = params["r_open_vertical_y"];
				cv::erode(temp_sub_img, temp_sub_img, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(morph_size, morph_size)));
				std::vector<std::vector<cv::Point2i>> temp_sub_contours;
				double t_area = -1;
				cv::findContours(temp_sub_img, temp_sub_contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
				for (auto t_cons : temp_sub_contours) {
					if (cv::contourArea(t_cons) > t_area) {
						t_area = cv::contourArea(t_cons);
						sel_cont_points = t_cons;
					}
				}
			}
			cv::Rect blob_bbox = cv::boundingRect(sel_cont_points);
			cv::Rect blob_bbox_on_img = blob_bbox;
			blob_bbox_on_img.x += bbox.x;
			blob_bbox_on_img.y += bbox.y;

			cv::Point2f points[4];
			auto min_rect = cv::minAreaRect(sel_cont_points);
			min_rect.points(points);
			std::vector<cv::Point2i> rect_points = {
				points[0], points[1], points[2], points[3]
			};

			cv::Mat sub_bin_make = cv::Mat::zeros(sub_img.size(), CV_8UC1);
			cv::fillPoly(sub_bin_make, rect_points, 255);
			sub_bin_make(blob_bbox).copyTo(bin_image(blob_bbox_on_img));

			PinInfoPtr pinInfo = std::make_shared<PinInfo>();
			pinInfo->mean_gv = gv_sum_total / gv_num_total;     // 平均亮度
			pinInfo->bbox = blob_bbox_on_img;                   // 外包框
			pinInfo->area_total = area_total;                   // 总面积
			//pinInfo->cont = blob_cont;                        // 轮廓
			pinInfo->area_init = cv::contourArea(out_cont);
            if (enable_classifyModel) {
                pinInfo->index      = conCount;
                cv::Mat colorImg;
                cv::cvtColor(sub_SvmImg, colorImg, cv::COLOR_GRAY2BGR);
                pinInfo->cropPinImg = colorImg;
                pinInfo->okName     = floderOKName + cropName;
                pinInfo->ngName     = floderNGName + cropName;
            }
			pin_infos.push_back(pinInfo);
            conCount++;
			break;
		}
	}
    // 使用分类模型
    if (enable_classifyModel) {
        int loopNum   = std::ceil(float(pin_infos.size()) / batchSize);
        for (int loop = 0; loop < loopNum; loop++) {
            int start = loop * batchSize;
            int end   = (std::min)(static_cast<int>(pin_infos.size()), start + batchSize);

            std::vector<cv::Mat> cropImgList;
            std::vector<int>     indexList;
            for (int m = start; m < end; m++) {
                cropImgList.push_back(pin_infos[m]->cropPinImg);
                indexList.push_back(pin_infos[m]->index);
            }
            // 分类模型检测
            bool ret = CheckBlobByClassify(cropImgList, indexList, pin_infos, classifyConfThr, params);
        }
    }
}


// 通过同轴光图片（Image1）查找针尖准确位置（针尖区域反光亮斑中心）
void FrontPinDetect::FindPinByImage1(cv::Mat& image1, cv::Mat bin_img, const json& params, std::vector<PinInfoPtr>& pin_infos, cv::Mat& img_draw, int roi_lt_x, int roi_lt_y)
{
	//cv::threshold(image1, image1, 80, 255, cv::THRESH_TOZERO);
	int w_area_max = params["w_area_max"];  //用于选择偏移阈值
	int w_area_target = params["w_area_target"];
	double w_pin_gv_max = params["w_pin_gv_max"]; //找针尖时二值化阈值的最高值
	double w_pin_gv_min = params["w_pin_gv_min"]; //找针尖时二值化阈值的最低值
	int x1 = params["x1"];
	int x2 = params["x2"];
	int y1 = params["y1"];
	int y2 = params["y2"];
	int sum1 = params["sum1"];
	int sum2 = params["sum2"];
	// 是否图像对齐
	bool enable_align = Tival::JsonHelper::GetParam(params, "enable_align", false); 

	//double blob_meangv_min = params["blob_meangv_min"];
	for (auto pin_info : pin_infos) {

		if (!pin_info->classify_status) {
            continue;
		}

		cv::Mat pin_img;
		cv::Mat pin_img_copy = image1(pin_info->bbox).clone();
		cv::Mat mask = bin_img(pin_info->bbox);
        cv::bitwise_and(pin_img_copy, mask, pin_img);

		// 以下算法通过迭代的方式搜索最有可能的针尖位置，具体思想是：
		// 设定不同的二值化阈值，通过针尖亮斑的面积、中心离红斑中心的距离等综合进行判断
		//二值化阈值从pin_gv_max迭代到pin_gv_min，逐渐下降
		double last_pin_area = 9999;
		//最后pin中心和当前pin中心
		cv::Point last_pin_coord;
		cv::Point cur_pin_coord;
		//最后pin区域和当前pin区域
		std::vector<cv::Point> last_pin_cont;
		std::vector<cv::Point> cur_pin_cont;
		//当前pin面积和总面积
		double cur_pin_area = 0;
		double total_area = 0;

		double min_cent_off = 9999;
		//x，y最大偏移 总最大偏移
		double cent_off_thresh_x = 0;
		double cent_off_thresh_y = 0;

		double cent_offset_sum_thresh = 0;
		cent_off_thresh_x = pin_info->area_total > w_area_max ? x1 : x2;
		cent_off_thresh_y = pin_info->area_total > w_area_max ? y1 : y2;
		cent_offset_sum_thresh = pin_info->area_total > w_area_max ? sum1 : sum2;

		double local_threshold = w_pin_gv_max;
		while (local_threshold >= w_pin_gv_min) {
			cv::Mat block_bin;
			cv::threshold(pin_img, block_bin, local_threshold, 255, cv::THRESH_BINARY);     // 二值化，提取不同阈值下的亮斑区域
			std::vector<std::vector<cv::Point>> contorus2;
			cv::findContours(block_bin, contorus2, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);  // 查找轮廓进行Blob分析
			cur_pin_cont.clear();
			cur_pin_area = 0;
			total_area = 0;
			min_cent_off = 9999;
			int a = 0;
			//if (contorus2.size() > 1) continue;
			for (auto cont2 : contorus2)
			{
				// 面积卡控，过小则忽略(轮廓求取面积的方式可以优化)
			   // double area = cv::contourArea(cont2);
				cv::Mat blobImage = cv::Mat::zeros(pin_img.size(), pin_img.type());
				std::vector<std::vector<cv::Point>> draw_conts = { cont2 };
				cv::drawContours(blobImage, draw_conts, 0, 255, -1);
				std::vector<cv::Point2f>component_points;
				cv::findNonZero(blobImage, component_points);
				float area = component_points.size();

				//std::cout << area << std::endl;
				std::vector<std::vector<cv::Point>> contorus_add;
				total_area += area;
				if (area < w_area_target * 0.2)
				{
					if (local_threshold == w_pin_gv_min) {
						LOGD("w_pin_gv_min:{}, area:{}", w_pin_gv_min, area);
					}
					continue;  //部分PIN的一半灰度低、面积小，需要设置一个较小的阈值
				}

				// 计算亮斑均值，过暗则跳过（针对针尖反光不好，可以讲阈值调低，避免找偏）
				cv::Moments M = cv::moments(cont2, false);
				cv::Point2i center(M.m10 / M.m00, M.m01 / M.m00);
				cv::Mat blob_mask = cv::Mat::zeros(block_bin.size(), block_bin.type());
				cv::drawContours(blob_mask, draw_conts, 0, 255, -1);
				cv::Scalar blob_meangv = cv::mean(pin_img, blob_mask);
				if (blob_meangv[0] < w_pin_gv_min) {
					// 如果针尖较暗则跳过，避免找到局部最亮点而非中心点，可以通过image2来找
					continue;
				}
				double x_off = std::abs(center.x - pin_info->bbox.width / 2);
				double y_off = std::abs(center.y - pin_info->bbox.height / 2);
				double cent_off = x_off + y_off;
				// blob中心距离粗定位中心在设定范围内，认为是一个候选位置，暂存信息到cur_pin_xx中
				if (x_off < cent_off_thresh_x && y_off < cent_off_thresh_y
					&& cent_off < cent_offset_sum_thresh && cent_off < min_cent_off) {

					//当前中心点
                    cv::Point2i tempCenter = {center.x + pin_info->bbox.x, center.y + pin_info->bbox.y};					
					cur_pin_coord = cv::Point2i(round(tempCenter.x + roi_lt_x), round(tempCenter .y + roi_lt_y));
					cur_pin_cont = cont2;
					cur_pin_area = area;
					min_cent_off = cent_off;
				}

				//std::cout << blob_mean[0] << std::endl;
			}
			// 如果面积已经够大或者二值化总面积超过一定大小，则退出迭代（继续迭代面积会更大）
			if (cur_pin_area > w_area_target * 0.5 || total_area > w_area_target) {
				break;
			}
			// 如果没有退出迭代且找到一个可能的pin针位置，则将pin针位置记录到last_pin_xx中，继续迭代下一轮
			if (cur_pin_area > 0) {
				last_pin_area = cur_pin_area;
				last_pin_coord = cur_pin_coord;
				last_pin_cont = cur_pin_cont;
			}
			local_threshold -= 10;
		}

		// 在last_pin和cur_pin中按照一定规则挑选一个作为最终的结果
		if (last_pin_cont.size() > 0 || cur_pin_cont.size() > 0) {
			pin_info->found = true;
			bool select_current = true;
			if (last_pin_cont.size() > 0 || cur_pin_cont.size() > 0) {
				select_current = std::abs(cur_pin_area - w_area_target) < std::abs(last_pin_area - w_area_target);
			}
			else if (last_pin_cont.size() > 0) {
				select_current = false;
			}
			if (select_current) {
				pin_info->pin_center = cur_pin_coord;
			}
			else {
				pin_info->pin_center = last_pin_coord;
			}
		}
	}
}


// 如果FindPinByImage1没有找出所有针尖（部分针尖在同轴光图片Image1中不反光），则通过平行光图片（Image2）中的红斑位置,
// 基于一定规则和尝试不同阈值方式，推断出最合适的针尖位置
void FrontPinDetect::FindPinByImage2(cv::Mat& image2, cv::Mat& bin_img, const json& params, std::vector<PinInfoPtr>& pin_infos, cv::Mat& img_draw)
{
	int blob_area_min = params["blob_area_min"];    // 最小blob面积
	int bin_threshold = params["bin_threshold"];    // 二值化阈值
	int target_area = params["area_target"];        // 目标blob面积
	int iter_times = params["iter_times"];          // 迭代次数
	int iter_step = params["iter_step"];            // 迭代步幅
	int area_max = params["area_max"];              // 最大面积
	int product_type = params["product_type"];
	int pin_target = params["pin_target"];
	// std::vector<cv::Point> pin_coords;

	for (auto pin_info : pin_infos)
	{
		//粗定位初始面积与最终面积的差值
		double difference = 0;
		double area_total = 0;
		if (product_type == 1)
		{
			area_total = pin_info->area_total;
		}
		//cv::Rect bbox = pin_info->bbox;
		int area_init = pin_info->area_init;

		if (pin_info->found) continue;
		if (area_total < target_area) {
			cv::Rect bbox = pin_info->bbox;             // 亮斑外包框
			double mean_gv = pin_info->mean_gv;         // 亮斑均值
			double default_threshold = mean_gv * 0.3;   // 初始阈值（从小到大迭代）
			cv::Mat pin_block_img = image2(bbox);       // 裁剪pin针图像

			BinResultPtr default_bin_result = LocalThreshold(pin_block_img, default_threshold, blob_area_min);
			BinResultPtr last_bin_result = default_bin_result;
			BinResultPtr cur_bin_result;

			// 阈值调整方向，1：阈值增加，-1: 阈值减少
			int direction = default_bin_result->area_sum > target_area ? 1 : -1;
			int try_times = 0;

			while (try_times < iter_times)
			{
				try_times += 1;
				double try_threshold = std::min(default_threshold + try_times * iter_step * direction, 250.0);
				try_threshold = std::max(try_threshold, 5.0);
				// 二值化-> 面积过滤-> 找出每个符合面积条件的blob的中心
				cur_bin_result = LocalThreshold(pin_block_img, try_threshold, blob_area_min);

				// 根据阈值调整方向和blob的面积，判断是否达到Break条件
				if ((direction == 1 && (try_threshold >= 250 || cur_bin_result->area_sum < target_area))
					|| (direction == -1 && (try_threshold < 1 || cur_bin_result->area_sum > target_area * 0.5))) {
					if (DEBUG_ENABLE) {
						//LOGI("Break(1) at try_times:{}", try_times);
						PrintBinResult("\t\t", cur_bin_result);
					}
					// 在last和cur之间选择一个更合适的作为最终针尖位置
					last_bin_result = ChooseBinResult(last_bin_result, cur_bin_result, params, img_draw);
					break;
				}
				else if (direction == 1 && cur_bin_result->area_sum > target_area * 1.5) {
					last_bin_result = cur_bin_result;
				}
				else if (direction == -1 && cur_bin_result->area_sum < target_area * 0.5) {
					last_bin_result = cur_bin_result;
				}
				else {
					BinResultPtr choose_result = ChooseBinResult(last_bin_result, cur_bin_result, params, img_draw);
					if (choose_result->threshold == last_bin_result->threshold) {
						if (DEBUG_ENABLE) {
							//LOGI("Break(2) at try_times:{}", try_times);
							PrintBinResult("\t\t", cur_bin_result);
						}
						last_bin_result = ChooseBinResult(last_bin_result, cur_bin_result, params, img_draw);
						break;
					}
					else {
						last_bin_result = cur_bin_result;
					}
				}
			} // end while()

			cv::Point pin_coord;
			// 基于每个pin位置阈值迭代后，可能会得到多个blob子区域（通常情况下是上下两个），推算出针尖位置
			bool found = CalcPinCoordByLocalBlobs(pin_coord, last_bin_result, params, bbox.tl().x, bbox.tl().y, img_draw);
			if (found) {
				pin_info->found = true;
				pin_info->pin_center = pin_coord;
			}
			if (!found || DEBUG_ENABLE) {
				//LOGD("Pin not found:")
				PrintBinResult("\t\t default:", default_bin_result);
				PrintBinResult("\t\t last:", last_bin_result);
			}
		}
	}
}

//LI自适应阈值
int FrontPinDetect::liThreshold(cv::Mat img) {
	// 初始阈值为图像的平均灰度值
	double threshold = cv::mean(img)[0];
	double lastThreshold = 0;
	double diff = 1;
	double mean1 = 0;
	double mean2 = 0;
	int count1 = 0;
	int count2 = 0;

	while (diff > 0.1) { // 迭代直到阈值变化小于 0.1
		for (int i = 0; i < img.rows; i++) {
			for (int j = 0; j < img.cols; j++) {
				if (img.at<uchar>(i, j) > threshold) {
					mean1 += img.at<uchar>(i, j);
					count1++;
				}
				else {
					mean2 += img.at<uchar>(i, j);
					count2++;
				}
			}
		}

		mean1 /= count1;
		mean2 /= count2;

		lastThreshold = threshold;
		threshold = (mean1 + mean2) / 2;
		diff = abs(threshold - lastThreshold);

		// 重置均值和像素计数器
		mean1 = 0;
		mean2 = 0;
		count1 = 0;
		count2 = 0;
	}

	return static_cast<int>(threshold);
}



//左右方向blob膨胀腐蚀（目的去除blob上下区域噪声干扰）
void FrontPinDetect::FilterNonPinBlobs(const cv::Mat& img2_gray, cv::Mat& bin_img, std::vector<std::vector<cv::Point >>& contours, const json& params, std::string image_name)
{
	double blob_area_min = params["r_blob_area_min"];
	double blob_area_max = params["r_blob_area_max"];
	double blob_width_min = params["r_blob_width_min"];
	double blob_width_max = params["r_blob_width_max"];
	double blob_height_min = params["r_blob_height_min"];
	double blob_height_max = params["r_blob_height_max"];

	int morph_x = params["r_close_x"];
	int morph_y = params["r_close_y"];
	int r_open_vertical_y = params["r_open_vertical_y"];
	int r_open_horizontal_x = params["r_open_horizontal_x"];

	bool r_enable_blur = params["r_enable_blur"];   //是否滤波开关
	bool r_enable_open_vertical = params["r_enable_open_vertical"];   //是否开启垂直开操作开关
	bool r_enable_open_horizontal = params["r_enable_open_horizontal"];   //是否开启水平开操作开关

	int bin_threshold = params["r_pin_gv_min"];    // 二值化阈值

	// 截取训练样本
	bool enable_notfilter2crop = Tival::JsonHelper::GetParam(params, "enable_notfilter2crop", false);


	cv::Mat gray_blur;
	if (r_enable_blur) {
		cv::blur(img2_gray, img2_gray, cv::Size(3, 3));
	}

	cv::threshold(img2_gray, bin_img, bin_threshold, 255, cv::THRESH_BINARY);

	//过滤毛丝
	cv::Mat kernel_close = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(morph_x, morph_y));
	// 进行形态学开操作
	cv::morphologyEx(bin_img, bin_img, cv::MORPH_CLOSE, kernel_close);
	if (r_enable_open_vertical) {
		cv::Mat kernel_open1 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(1, r_open_vertical_y));
		cv::morphologyEx(bin_img, bin_img, cv::MORPH_OPEN, kernel_open1);
	}
	if (r_enable_open_horizontal) {
		cv::Mat kernel_open2 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(r_open_horizontal_x, 1));
		cv::morphologyEx(bin_img, bin_img, cv::MORPH_OPEN, kernel_open2);
	}
	//cv::dilate(bin_img, bin_img, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3)));
	std::vector<std::vector<cv::Point >> m_contours;
	cv::findContours(bin_img, m_contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
	contours.clear();
	for (auto cont : m_contours) {
		cv::Rect bbox = cv::boundingRect(cont);
		// 收集样本不过滤
		double area = cv::contourArea(cont);
		if (enable_notfilter2crop) {
			contours.push_back(cont);
		}
		else {
			if (area == 0) area = bbox.width * bbox.height;
			if (area < blob_area_min || area > blob_area_max * 2.5 ||
				bbox.width < blob_width_min || bbox.width > blob_width_max ||
				bbox.height < blob_height_min || bbox.height > blob_height_max) {
				cv::fillPoly(bin_img, cont, 0);
				//可以打印一下过滤掉的参数值
			}
			else {
				contours.push_back(cont);
			}
		}
	}

	//保存红图Blob定位的结果图
	bool enable_save_resultImage = params["enable_save_resultImage"];
	if (enable_save_resultImage) {
		std::string folder_path = "D:/DebugResult/rImage_Filter";
		bool state = check_folder_state(folder_path);
		if (state) {
			cv::imwrite(folder_path + "/" + image_name + "_filter.png", bin_img);
		}
	}
}


// 二值化-> 面积过滤-> 找出每个符合面积条件的blob的中心
BinResultPtr FrontPinDetect::LocalThreshold(const cv::Mat& block_img, int threshold, int min_area)
{
	cv::Mat block_bin;
	cv::threshold(block_img, block_bin, threshold, 255, cv::THRESH_BINARY);

	std::vector<std::vector<cv::Point>> contorus;
	cv::findContours(block_bin, contorus, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

	std::vector<BlobInfoPtr> blob_list;
	double area_sum = 0;
	double total_gv = 0;
	int total_num = 0;

	for (auto cont : contorus) {
		double area = cv::contourArea(cont);
		double sum_gv = 0;
		int num = 0;
		//std::cout << area << std::endl;
		area_sum += area;

		if (area < min_area) continue;

		cv::Mat mask = cv::Mat::zeros(block_bin.size(), block_bin.type());

		std::vector<std::vector<cv::Point>> draw_conts = { cont };
		cv::drawContours(mask, draw_conts, 0, 255, -1);
		SumGV(block_img, mask, sum_gv, num);

		total_gv += sum_gv;
		total_num += num;

		cv::Mat contourMat(cont);
		cv::Mat xCoords, yCoords;
		cv::extractChannel(contourMat, xCoords, 0);
		cv::extractChannel(contourMat, yCoords, 1);

		double minX, maxX;
		double minY, maxY;
		cv::minMaxLoc(xCoords, &minX, &maxX);
		cv::minMaxLoc(yCoords, &minY, &maxY);
		double centX = (minX + maxX) / 2;
		double centY = (minY + maxY) / 2;

		blob_list.push_back(std::make_shared<BlobInfo>(cont, area, cv::Point(centX, centY)));
	}

	return std::make_shared<BinResult>(area_sum, total_gv / total_num, threshold, blob_list);
}

cv::Mat FrontPinDetect::Cont_delete(const cv::Mat& img2_gray)
{
	cv::Mat img_bin;
	cv::Mat img_blur;
	cv::blur(img2_gray, img_blur, cv::Size(3, 3));
	cv::threshold(img_blur, img_bin, 60, 255, cv::THRESH_BINARY);
	//纵向连通域删除
	cv::dilate(img_bin, img_bin, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(1, 40)));
	cv::erode(img_bin, img_bin, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(1, 40)));
	std::vector<std::vector<cv::Point>> contours;
	cv::findContours(img_bin, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
	// 设置阈值
	double areaThreshold = 100; // 面积阈值
	double widthThreshold = 100; // 宽度阈值
	for (const auto& contour : contours) {
		// 计算轮廓的面积
		double area = cv::contourArea(contour);
		// 获取轮廓的边界框
		cv::Rect boundingBox = cv::boundingRect(contour);
		int rect_w = boundingBox.width;
		int rect_h = boundingBox.height;
		if (rect_w > rect_h * 2 || rect_h > 130 || area < areaThreshold || boundingBox.width > widthThreshold)
		{
			cv::drawContours(img_bin, std::vector<std::vector<cv::Point>>{contour}, 0, 0, -1);
		}
	}
	cv::Mat resultMat;
	cv::bitwise_and(img2_gray, img_bin, resultMat);
	return resultMat;
}

void FrontPinDetect::PrintBinResult(const std::string& msg, BinResultPtr bin_result, bool detail)
{
	double area_sum = bin_result->area_sum;
	double threshold = bin_result->threshold;
	std::vector<BlobInfoPtr> blob_list = bin_result->blob_list;

	LOGD("{}  AreaSum:{}, Threshold:{}, Blobs:{}", msg, area_sum, int(threshold), blob_list.size());
}

BinResultPtr FrontPinDetect::ChooseBinResult(BinResultPtr last_bin_result, BinResultPtr cur_bin_result, const json& params, const cv::Mat& img_draw)
{
	if (!cur_bin_result) {
		return last_bin_result;
	}

	int target_area = params["area_target"];

	auto last_area = last_bin_result->area_sum;
	auto last_blob_list = last_bin_result->blob_list;
	auto last_count = last_blob_list.size();

	auto cur_area = cur_bin_result->area_sum;
	auto cur_blob_list = cur_bin_result->blob_list;
	auto cur_count = cur_blob_list.size();

	auto cur_pairs = GetBlobPairs(cur_blob_list, cur_area);
	auto last_pairs = GetBlobPairs(last_blob_list, last_area);

	auto fit_area_result = abs(last_area - target_area) < abs(cur_area - target_area) ? last_bin_result : cur_bin_result;

	if (last_count == cur_count) {
		return fit_area_result;
	}
	else if (last_count > cur_count) {
		// 当前只有一个blob，如果之前的多个blob中存在上下结果的blob组合，
		// 则选用之前阈值处理的结果，否则选取当前的一个blob
		if (cur_count == 1) {
			return last_pairs.size() > 0 ? last_bin_result : fit_area_result;
		}
		else {
			if ((last_pairs.size() > 0 && cur_pairs.size() > 0) || (last_pairs.size() == 0 && cur_pairs.size() == 0)) {
				return fit_area_result;
			}
			else {
				return cur_pairs.size() > 0 ? cur_bin_result : last_bin_result;
			}
		}
	}
	else {
		if (last_count == 1) {
			return cur_pairs.size() > 0 ? cur_bin_result : fit_area_result;
		}
		else {
			if ((last_pairs.size() > 0 && cur_pairs.size() > 0) || (last_pairs.size() == 0 && cur_pairs.size() == 0)) {
				return fit_area_result;
			}
			else {
				return cur_pairs.size() > 0 ? cur_bin_result : last_bin_result;
			}
		}
	}

}

int FrontPinDetect::SumBlobArea(std::vector<BlobInfoPtr>& blob_list, std::vector<int>& indices)
{
	double area_total = 0;
	for (auto idx : indices) {
		area_total += blob_list[idx]->area;
	}
	return area_total;
}

std::vector<std::set<int>> FrontPinDetect::GetBlobPairs(std::vector<BlobInfoPtr>& blob_list, int area_sum, double area_ratio)
{
	std::vector<std::set<int>> condidate_pairs;
	for (int i = 0; i < blob_list.size() - 1; i++) {
		auto bi = blob_list[i];
		for (int j = i + 1; j < blob_list.size(); j++) {
			if (i == j) continue;

			auto bj = blob_list[j];
			std::vector<int> pair = { i, j };
			double pair_area_ratio = SumBlobArea(blob_list, pair) / area_sum;
			if (bi->center.x - bj->center.x <= 6 && pair_area_ratio > area_ratio) {
				condidate_pairs.push_back({ i, j });
			}
		}
	}
	return condidate_pairs;
}

// 基于每个pin位置阈值迭代后，可能会得到多个blob子区域（通常情况下是上下两个），推算出针尖位置
bool FrontPinDetect::CalcPinCoordByLocalBlobs(cv::Point& pin_center, BinResultPtr bin_result, const json& params, int offset_x, int offset_y, cv::Mat& img_draw)
{
	double area_sum = bin_result->area_sum;
	double mean_gv = bin_result->mean_gv;
	std::vector<BlobInfoPtr>& blob_list = bin_result->blob_list;
	int blob_cnt = blob_list.size();

	double r_mean_gv_min = params["r_mean_gv_min"];
	long Red_light_structure = params["Red_light_structure"];
	if (blob_cnt == 0) {
		LOGD("CalcPinCoord: No blobs.");
		return false;
	}

	if (mean_gv < r_mean_gv_min) {
		LOGD("CalcPinCoord: MeanGV:{} smaller than threshold:{}, ignore.", mean_gv, r_mean_gv_min);
		return false;
	}

	//for (auto bb : blob_list) {
	//    std::vector<std::vector<cv::Point>> draw_conts = { bb->contour };
	//    cv::drawContours(img_draw, draw_conts, -1, cv::Scalar(255,255,0), 1);
	//}

	if (blob_cnt == 1) {
		// 只有一个blob，上下粘连到一起了，或者半边不亮（这种情况会找偏，需要想办法优化）
		pin_center = blob_list[0]->center;
	}
	else if (blob_cnt == 2) {
		// 两个blob，需要判断是上下结构还是左右结构
		cv::Point center1 = blob_list[0]->center;
		cv::Point center2 = blob_list[1]->center;
		double area1 = blob_list[0]->area;
		double area2 = blob_list[1]->area;
		//原本红光结构判断blob
		if (Red_light_structure == 0) //左右
		{
			if (abs(center1.y - center2.y) > 10) {
				// 上下结构，取大面积blob的中心
				int idx = area1 > area2 ? 0 : 1;
				pin_center = blob_list[idx]->center;

			}
			else {
				// 左右结构，取两个blob连线的中心
				pin_center.x = round((center1.x + center2.x) / 2);
				pin_center.y = round((center1.y + center2.y) / 2);
			}
		}
		else if (Red_light_structure == 1) //上下
		{
			if (abs(center1.y - center2.y) > 10) {
				pin_center.x = round((center1.x + center2.x) / 2);
				pin_center.y = round((center1.y + center2.y) / 2);
			}
			else {
				int idx = area1 > area2 ? 0 : 1;
				pin_center = blob_list[idx]->center;
			}
		}
	}
	else {
		//超过两个blob，找上下结构的blob对
		auto candidate_pairs = GetBlobPairs(blob_list, area_sum);
		if (Red_light_structure == 0) //左右
		{
			//"这里写寻找左右结构blob对的代码";
		}
		if (candidate_pairs.size() == 0) {
			// 如果找不到blob对，取所有blob中心平均值（几何中心）
			double sum_x = 0;
			double sum_y = 0;
			for (auto blob : blob_list) {
				sum_x += blob->center.x;
				sum_y += blob->center.y;
			}
			pin_center.x = round(sum_x / blob_cnt);
			pin_center.y = round(sum_y / blob_cnt);
		}
		else if (candidate_pairs.size() == 1) {
			// 如果找到一个blob对，取两个blob中心点连线的中点
			auto pair = candidate_pairs[0];
			std::vector<int> vec_pair;
			vec_pair.assign(pair.begin(), pair.end());

			cv::Point center1 = blob_list[vec_pair[0]]->center;
			cv::Point center2 = blob_list[vec_pair[1]]->center;
			pin_center.x = round((center1.x + center2.x) / 2);
			pin_center.y = round((center1.y + center2.y) / 2);
		}
		else {
			// 如果有多个blob对（概率很低），根据面积或者中心位置选一个合适的pair，取其连线中心点
			std::vector<std::set<int>> blob_groups;
			// for (auto cp : candidate_pairs) {

			// }
		}
	}

	pin_center.x += offset_x;
	pin_center.y += offset_y;

	return true;
}

void FrontPinDetect::SumGV(cv::Mat image, cv::Mat mask, double& sum_gv, int& px_num)
{
	int histSize = 256; // 直方图条目数
	float range[] = { 0, 256 }; // 像素值范围
	const float* histRange = { range };

	// 计算直方图
	sum_gv = 0;
	px_num = 0;
	cv::Mat hist;
	cv::calcHist(&image, 1, nullptr, mask, hist, 1, &histSize, &histRange);
	for (int i = 1; i < histSize; i++)
	{
		if (hist.at<float>(i) > 0) {
			sum_gv += hist.at<float>(i) * i;
			px_num += int(hist.at<float>(i));
		}
	}
}


// 根据明亮区域提取ROI（需要通过参数配置，不同型号产品提取方式不同，如果机器自动放料位置偏差小可以直接配置ROI参数）
cv::Rect FrontPinDetect::GetROI(cv::Mat& image, cv::Mat& mask, double& angle, const json& params)
{
	int getroi_threshold = params["getroi_threshold"];
	int roiarea_width = params["roiarea_width"];
	int roiarea_height = params["roiarea_height"];
	int roiarea_top_x = params["roiarea_top_x"];
	int roiarea_top_y = params["roiarea_top_y"];
	cv::Mat bin_img;
	cv::Mat blur_img;
	cv::blur(image, blur_img, cv::Size(3, 3));

	double sin = 20;
	double hin = 200;
	double mt = 2.0;
	double sout = 0;
	double hout = 255;
	//cv::Mat img = connector::gray_stairs(blur_img, sin, hin, mt, sout, hout);
	//int exec_threshold = connector::exec_threshold(img, connector::THRESHOLD_TYPE::DEFAULT, -1,-1, false)+25;
	//std::cout << "阈值是：" << exec_threshold << std::endl;
	//cv::adaptivethreshold(blur_img, bin_img, 255, cv::adaptive_thresh_gaussian_c,cv::thresh_binary_inv,55, 2);
	//int threshold_value = FrontPinDetect::liThreshold(blur_img);
	/*  cv::threshold(blur_img, bin_img, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);*/
	//cv::morphologyEx(bin_img, bin_img, cv::MORPH_CLOSE, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(50, 50)));
	cv::threshold(blur_img, bin_img, getroi_threshold, 255, cv::THRESH_BINARY);
	cv::morphologyEx(bin_img, bin_img, cv::MORPH_CLOSE, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(100, 100)));

	std::vector<std::vector<cv::Point>> contorus;
	cv::findContours(bin_img, contorus, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

	mask = cv::Mat::zeros(image.size(), image.type());
	//cv::Point pts[4];
	//pts[0] = cv::Point(roiarea_top_x, roiarea_top_y);
	//pts[1] = cv::Point(roiarea_top_x + roiarea_width, roiarea_top_y);
	//pts[3] = cv::Point(roiarea_top_x, roiarea_top_y + roiarea_height);
	//pts[2] = cv::Point(roiarea_top_x + roiarea_width, roiarea_top_y + roiarea_height);

	//// 将顶点放入一个vector中
	//std::vector<cv::Point> poly;
	//for (int i = 0; i < 4; i++)
	//{
	//    poly.push_back(pts[i]);
	//}
	//cv::fillPoly(mask, poly, 255);
	for (auto cont : contorus) {
		//cv::drawContours(bin_img, std::vector<std::vector<cv::Point>>{cont}, 0, 255, -1);
		cv::Rect boundingRect1;
		/*      double wanzhengdu = calculateCompleteness(cont, boundingRect1);
			  if (wanzhengdu < 0.5) continue;*/
		double area = cv::contourArea(cont);
		//std::cout << area << std::endl;
		if (area < 10000) continue;
		//LOGI("GetROI: blob area: {}", area);
		cv::RotatedRect rot_rect = cv::minAreaRect(cont);

		if (area > 6000 * 10000 || area < 300 * 10000) continue;
		//rot_rect.size.width -= 100;
		//rot_rect.size.height -= 100;

		cv::Point2f box_pts[4];
		rot_rect.points(box_pts);
		double dist1 = std::sqrt(std::pow(box_pts[0].x - box_pts[1].x, 2) + std::pow(box_pts[0].x - box_pts[1].x, 2));
		double dist2 = std::sqrt(std::pow(box_pts[2].x - box_pts[1].x, 2) + std::pow(box_pts[2].x - box_pts[1].x, 2));

		cv::Point2f pt1, pt2;
		if (dist1 > dist2) {
			pt1 = box_pts[0].x < box_pts[1].x ? box_pts[0] : box_pts[1];
			pt2 = box_pts[0].x < box_pts[1].x ? box_pts[1] : box_pts[2];
		}
		else {
			pt1 = box_pts[1].x < box_pts[2].x ? box_pts[1] : box_pts[2];
			pt2 = box_pts[1].x < box_pts[2].x ? box_pts[2] : box_pts[1];
		}
		angle = std::atan((pt2.y - pt1.y) / (pt2.x - pt1.x)) * 180.0 / PI;
		cv::Mat boxpts;
		cv::boxPoints(rot_rect, boxpts);
		boxpts.convertTo(boxpts, CV_32S);
		cv::fillPoly(mask, boxpts, 255);
		//cv::drawContours(mask, draw_conts, 0, 255, -1);
		return boundingRect(cont);
	}
	return cv::Rect();
}

//计算轮廓完整度
double FrontPinDetect::calculateCompleteness(const std::vector<cv::Point> contour, const cv::Rect boundingRect)
{
	double contourArea = cv::contourArea(contour); // 计算轮廓面积
	double boundingRectArea = boundingRect.area(); // 计算外接矩形面积

	// 计算轮廓面积与外接矩形面积的比值
	double completeness = contourArea / boundingRectArea;
	return completeness;
}


cv::Mat FrontPinDetect::GetROIImage(const cv::Mat& image, const cv::Mat& mask)
{
	cv::Mat roi_img;
	cv::bitwise_and(image, mask, roi_img);
	return roi_img;
}

// 计算原图坐标系到pin针坐标系的转换矩阵
cv::Mat FrontPinDetect::GetPinCSTransMat(const std::vector<PinInfoPtr>& pin_infos, const json& params, double roiAngle, cv::Mat& img_draw)
{
	int min_points_in_line = 10;
	std::vector<cv::Point2f> pts;
	for (auto pin_info_ptr : pin_infos) {
		if (pin_info_ptr) {
			cv::Point2f pin_center = cv::Point(pin_info_ptr->pin_center.x, pin_info_ptr->pin_center.y);
			pts.push_back(pin_center);
		}
	}
	double angle_begin = reget_angle(pts, min_points_in_line);
	if (std::abs(angle_begin)>10) {
		angle_begin = 0.f;
	}


	double ppum = params["ppum"];               // 像素当量，每个像素对应的um值
	bool enable_OneLine= params["enable_OneLine"];
	json y_coords1 = params["y_coords1"];       // Y坐标值
	json y_coords2 = params["y_coords2"];
	json x_coords1 = params["x_coords1"];
	json x_coords2 = params["x_coords2"];
	cv::Point lt = { -1, -1 };
	cv::Point temp_lt = { -1, -1 };
	cv::Mat tmpM = cv::Mat::zeros(3, 3, CV_8UC1);

	int count1 = 0;
	double angel_sum1 = 0;
	double angel_temp = 0;
	cv::Vec4f lineParams1;
	cv::Point origin;

	// 循环计算坐标旋转角度，针点不在同一条线上时，计算的初始角度偏差较大，需要循环纠偏
	float angle_ratio = 999;
	std::vector<cv::Point2f> first_cols;

	int angle_ratio_loop_count = 0;
	while (angle_ratio > 1.15) {
		angle_ratio_loop_count++;
		if (angle_ratio_loop_count > 10) {
			break;
		}
		count1 = 0;
		angel_sum1 = 0;
		angel_temp = 0;
		for (auto pin : pin_infos) {
			if (!pin->found) continue;
			if ((lt.x == -1 || (pin->pin_center.x + pin->pin_center.y) < (lt.x + lt.y))) {
				temp_lt.x = lt.x;
				temp_lt.y = lt.y;
				lt.x = pin->pin_center.x;
				lt.y = pin->pin_center.y;
			}
		}
		//解决搜索框过大，导致特别远的干扰点作为左上点针尖，坐标系整体偏移
		//行高
		double row_height_mm = double(y_coords1[0]) - double(y_coords2[0]);
		double row_height_px = Mm2Px(row_height_mm, ppum);
		//列宽
		double col_width_mm = double(x_coords2[1]) - double(x_coords2[0]);
		double col_width_px = Mm2Px(col_width_mm, ppum);

		//根据左上角第一个pin针位置，roi矩形的旋转角度，建立一个大概的映射矩阵
		// 在pin针坐标系中，X方向水平，Y方向垂直，方便判断针尖属于第几行
		tmpM = vector_angle_to_M(lt.x, lt.y, 0, 0, 0, angle_begin);
		if (angle_begin == 0) {
			angle_begin = 0.000001;
		}

		std::vector<std::vector<cv::Point>> pin_rows(int(y_coords1.size() + y_coords2.size()) + 5);
		std::vector<std::vector<cv::Point>> pin_cols(int(std::max(x_coords1.size(), x_coords2.size())) + 5);


		double min_px_x = 999;
		double min_px_y = 999;
		int min_row_idx = 999;
		int min_col_idx = 999;
		int max_row_idx = -1;
		int max_col_idx = -1;

		// 计算行列ID
		first_cols.clear();
		for (auto pin : pin_infos) {
			if (pin->pin_center.x == 0 || pin->pin_center.y == 0)
			{
				continue;
			}

			// 粗略记录第一列点，以防初始角度偏差较大，导致点数不够拟合
			if (pin->pin_center.x > lt.x - 10 && pin->pin_center.x < lt.x - 10) {
				first_cols.push_back(cv::Point2f(pin->pin_center.x, pin->pin_center.y));
			}

			cv::Point2f pin_local = TransPoint(tmpM, pin->pin_center);
			pin->local_x = pin_local.x;
			pin->local_y = pin_local.y;
			pin->row_idx = pin_local.y > 0 ? int(pin_local.y / row_height_px + 0.5) : int(pin_local.y / row_height_px - 0.5);
			pin->col_idx = pin_local.x > 0 ? int(pin_local.x / col_width_px + 0.5) : int(pin_local.x / col_width_px - 0.5);

			if (pin->row_idx < min_row_idx && pin->col_idx > -2)
			{
				min_row_idx = pin->row_idx;
			}
			if (pin->col_idx < min_col_idx && pin->col_idx > -2)
			{
				min_col_idx = pin->col_idx;
			}

			if (pin->row_idx > max_row_idx)
			{
				max_row_idx = pin->row_idx;
			}
			if (pin->col_idx > max_col_idx)
			{
				max_col_idx = pin->col_idx;
			}
		}

		// 针点分行和分列
		for (auto pin : pin_infos) {
			if (pin->pin_center.x == 0 || pin->pin_center.y == 0)
			{
				continue;
			}

			double diff1 = std::abs(pin->row_idx * row_height_px - pin->local_y);
			double diff2 = std::abs(pin->col_idx * col_width_px - pin->local_x);
			int min_area_row = row_height_px * 0.7;
			int min_area_col = col_width_px * 0.7;



			// 取最小行最小列的点用于计算角度偏差，同时取最大行，最大列的点用于减小误差
			// 最小
			try {
				if (diff1 < min_area_row && pin->row_idx == min_row_idx)
				{
					pin_rows[pin->row_idx + std::abs(min_row_idx)].push_back(pin->pin_center);
				}
				if (diff2 < min_area_col && pin->col_idx == min_col_idx)
				{
					pin_cols[pin->col_idx + std::abs(min_col_idx)].push_back(pin->pin_center);
				}
				// 最大
				if (pin_rows.size() > max_row_idx) {
					if (diff1 < min_area_row && pin->row_idx == max_row_idx)
					{
						if (std::abs(min_row_idx) + max_row_idx >= pin_rows.size()) {
							continue;
						}
						pin_rows[std::abs(min_row_idx) + max_row_idx].push_back(pin->pin_center);
					}
				}
				if (pin_cols.size() > max_col_idx) {
					if (diff2 < min_area_col && pin->col_idx == max_col_idx)
					{
						if (std::abs(min_col_idx) + max_col_idx >= pin_cols.size()) {
							continue;
						}
						pin_cols[std::abs(min_col_idx) + max_col_idx].push_back(pin->pin_center);
					}
				}
			}
			catch (std::exception& e) {
				LOGI("Catch in pin_rows pin cols: min_row_idx:{},min_col_idx:{},max_row_idx:{},max_col_idx:{}", min_row_idx, min_col_idx, max_row_idx, max_col_idx);
				LOGE("Catch in pin_rows pin cols:", e.what());
				continue;
			}
		}


		// 水平点拟合
		if ((pin_cols.size() > pin_rows.size())&&enable_OneLine) {
			for (int i = 0; i < pin_rows.size(); i++) {
				if (pin_rows[i].size() > 1) {
					cv::fitLine(pin_rows[i], lineParams1, cv::DIST_HUBER, 0, 0.01, 0.01);
					double angleRad = std::atan(lineParams1[1] / lineParams1[0]);
					LOGD("Fit line: {}, pin count:{}, angle:{}", i, pin_rows[i].size(), angleRad * 180.0 / PI);
					angel_sum1 += angleRad;
					count1 += 1;
					angel_temp += angleRad * 180.0 / PI;
				}
			}
		}
		// 垂直点拟合
		else {
			for (int i = 0; i < pin_cols.size(); i++) {
				if (pin_cols[i].size() > 1) {
					cv::fitLine(pin_cols[i], lineParams1, cv::DIST_HUBER, 0, 0.01, 0.01);
					double angleRad = std::atan(lineParams1[1] / lineParams1[0]);
					LOGD("Fit line: {}, pin count:{}, angle:{}", i, pin_cols[i].size(), angleRad * 180.0 / PI);
					if (angleRad < 0) {
						angel_sum1 += angleRad + PI / 2;
						angel_temp += angleRad * 180.0 / PI + 90;
					}
					else {
						angel_sum1 += angleRad - PI / 2;
						angel_temp += angleRad * 180.0 / PI - 90;
					}
					count1 += 1;
				}
			}
		}

		if (pin_cols.size() < pin_rows.size() && count1 == 0 && first_cols.size() > 2) {
			// 对于不在同一条线的情况，初始角度可能偏差较大，导致找到的第一列点个数＜2，不能拟合
			cv::fitLine(first_cols, lineParams1, cv::DIST_HUBER, 0, 0.01, 0.01);
			double angleRad = std::atan(lineParams1[1] / lineParams1[0]);
			LOGD("Fit line: {}, pin count:{}, angle:{}", 0, first_cols.size(), angleRad * 180.0 / PI);
			if (angleRad < 0) {
				angel_sum1 += angleRad + PI / 2;
				angel_temp += angleRad * 180.0 / PI + 90;
			}
			else {
				angel_sum1 += angleRad - PI / 2;
				angel_temp += angleRad * 180.0 / PI - 90;
			}
			count1 += 1;
		}


		angel_temp = angel_temp / count1;

		// 重新计算坐标旋转点
		double y0_coord_sum = 0;
		double y0_mean = 0;
		double x0_coord_sum = 0;
		double x0_mean = 0;
		angle_ratio = std::max(std::abs(angel_temp), std::abs(angle_begin)) / std::min(std::abs(angel_temp), std::abs(angle_begin));
		angle_begin = angel_temp;

		if (angle_ratio < 1.15 && pin_cols.size() > pin_rows.size())
		{
			for (int i = 0; i < pin_rows[0].size(); i++)
			{
				//求最小行y的均值
				y0_coord_sum += TransPoint(tmpM, pin_rows[0][i]).y;
			}
			y0_mean = y0_coord_sum / pin_rows[0].size();

			for (int m = 0; m < pin_cols[0].size(); m++)
			{
				//求最小列x的均值
				x0_coord_sum += TransPoint(tmpM, pin_cols[0][m]).x;
			}
			x0_mean = x0_coord_sum / pin_cols[0].size();
			origin = TransPoint(tmpM.inv(), cv::Point2f(x0_mean, y0_mean));
			//2024年8月12日 www
			//origin = cv::Point(lt.x, lt.y);
		}
		else
		{
			origin = cv::Point(lt.x, lt.y);
		}
	}

	if (angle_ratio_loop_count > 7) {
		return cv::Mat();
	}
	//origin = cv::Point(lt.x, lt.y);
	double angle = (angel_sum1 / count1) * 180.0 / PI;
	cv::Mat M = vector_angle_to_M(origin.x, origin.y, 0, 0, 0, angle);
	return M;
}

// 根据图纸计算针尖X\Y方向偏差值
json FrontPinDetect::CalcPinResults(std::vector<PinInfoPtr>& pin_infos, const cv::Mat& M, const cv::Mat img1_gray, const cv::Mat img2_gray, const cv::Mat bin_img2, const json& params, double& adj_x, double& adj_y, int roi_lt_x, int roi_lt_y, float& differ_std, float& differ_measure)
{
	double ppum = params["ppum"];           // 像素当量，每个像素对应的um值
	double tolerance = params["tolerance"]; // 公差范围，X、Y方向偏离值超过该范围认为是NG
	json x_coords1 = params["x_coords1"];   // 基数行X坐标
	json x_coords2 = params["x_coords2"];   // 偶数行X坐标
	json y_coords1 = params["y_coords1"];   // 基数行Y坐标值
	json y_coords2 = params["y_coords2"];   // 偶数行Y坐标值
	bool enable_OneLine = params["enable_OneLine"];
	float gap = params["gap"];
	//误差计算方式
	int error_type = Tival::JsonHelper::GetParam(params, "error_type",0);

	double blob_area_min = params["r_blob_area_min"];
	double blob_area_max = params["r_blob_area_max"];
	double blob_width_min = params["r_blob_width_min"];
	double blob_width_max = params["r_blob_width_max"];
	double blob_height_min = params["r_blob_height_min"];
	double blob_height_max = params["r_blob_height_max"];

    bool enable_offset = Tival::JsonHelper::GetParam(params, "enable_offset", false);
	json enable_coord_adj = Utils::GetProperty(params, "enable_coord_adj", ENABLE_COORD_ADJ_DEFAULT);   // 是否启用坐标系修正
	double tolerance_px = Mm2Px(tolerance, ppum); // 公差值0.12mm对应的像素值

	json results = json::array();

	// 将pin针在原图上的像素坐标转换到pin针坐标系的像素坐标
	std::vector<cv::Point2f> det_points;
	for (auto pin : pin_infos) {
		if (!pin->found) {
			det_points.push_back(cv::Point2f(99999, 99999));
			continue;
		}
		pin->pin_cent_local = TransPoint(M, pin->pin_center);
		det_points.push_back(pin->pin_cent_local);
	}


	std::vector<cv::Point2f> std_points;            // 标准图纸坐标

	double x_offset_sum = 0;
	double y_offset_sum = 0;
	int found_cnt = 0;

	// 在pin针坐标系计算、统计针尖的X\Y偏差值
	int pin_index = -1;
	int Rows_Sum = y_coords1.size() + y_coords2.size();
	int Cols_Size[] = { x_coords2.size(), x_coords1.size() };
	int result_state = 0;


	for (int i = 0; i < Rows_Sum; i++) {
		int Col_Sum = i % 2 == 0 ? Cols_Size[0] : Cols_Size[1];
		for (int j = 0; j < Col_Sum; j++) {

			pin_index++;

			// 区分奇偶数行
			double x_coord = i % 2 == 0 ? x_coords2[j] : x_coords1[j];
			double y_coord = i % 2 == 0 ? y_coords2[i / 2] : y_coords1[i / 2];

			// 当针点不在同一条线上，需要加上gap距离
			if (!enable_OneLine)
			{
				if (j % 4 == 1 || j % 4 == 2)
				{
					y_coord = y_coord - gap;
				}
			}

			reget_ppum(pin_index, ppum, params, enable_offset, 1, 0);

			x_coord = Mm2Px(x_coord, ppum);
			reget_ppum(pin_index, ppum, params, enable_offset, 0, 1);

			y_coord = Mm2Px(y_coord, ppum);
			cv::Point2f std_point = cv::Point2f(x_coord, y_coord);
			std_points.push_back(std_point);
			double min_dist = 0;
			// 偏离所有pin，找距离最近一个作为当前图纸位置对应的pin，计算偏差
			int nearIdx = GetNearestPointIdx(det_points, std_point, min_dist);

			json pin_img_pos = { 0, 0 };                    // 图像坐标系pin针坐标
			cv::Point2f measure_pos = { 0, 0 };             // 针尖坐标系pin针坐标
			cv::Point2f std_pos = std_point;					// 针尖坐标系pin针图纸坐标

			//更新像素当量
			reget_ppum(pin_index, ppum, params, enable_offset, 1, 0);
			std_pos.x = Px2Mm(std_point.x, ppum) ;
			reget_ppum(pin_index, ppum, params, enable_offset, 0, 1);
			std_pos.y = Px2Mm(std_point.y, ppum);

			cv::Point2f img_pt = { 0, 0 };

			double x_off = 0;
			double y_off = 0;
			if (nearIdx >= 0) {
				//更新像素当量
				reget_ppum(pin_index, ppum, params, enable_offset, 1, 0);
				measure_pos.x = Px2Mm(det_points[nearIdx], ppum).x ;
				reget_ppum(pin_index, ppum, params, enable_offset, 0, 1);
				measure_pos.y = Px2Mm(det_points[nearIdx], ppum).y ;

				img_pt.x = det_points[nearIdx].x;
				img_pt.y = det_points[nearIdx].y;

				x_off = measure_pos.x - std_pos.x;
				y_off = measure_pos.y - std_pos.y;

				pin_img_pos[0] = pin_infos[nearIdx]->pin_center.x;
				pin_img_pos[1] = pin_infos[nearIdx]->pin_center.y;

				// 只统计偏差在范围内的点，用于修正坐标系偏移
				if (std::abs(x_off) < tolerance && std::abs(y_off) < tolerance) {
					x_offset_sum += x_off;
					y_offset_sum += y_off;
					found_cnt++;
				}


			}
			else
			{

				std::cerr<<" not find" << std::endl;
				LOGD("transform matrix  invalid." );
				continue;
			}

			json pin_result = {
				{"pin_coord", pin_img_pos},
				{"std_x", std_pos.x}, //理论值
				{"std_y", std_pos.y},
				{"measured_x", measure_pos.x}, //实测值
				{"measured_y", measure_pos.y},
				{"x_off", x_off},       //X偏差(TP）值
				{"y_off", y_off},       //Y偏差(TP）值
				{"index", pin_index},   //索引
				{"TP", 2 * sqrt(x_off * x_off + y_off * y_off)},
				{"result", result_state},
                {"img_x", img_pt.x},
                {"img_y", img_pt.y}
			};
			results.push_back(pin_result);
		}
	}

	// 计算X、Y方向偏差的平均追，用于修正坐标系，目的是为了尽可能让针尖都框到网格内
	adj_x = x_offset_sum / found_cnt + _offset_value_x_;
	adj_y = y_offset_sum / found_cnt + _offset_value_y_;
	if (found_cnt == 0)
	{
		adj_x = 0;
		adj_y = 0;
	}

	float first_std = -1;
	float second_std = -1;
	float first_measure = -1;
	float second_measure = -1;

	for (json& pr : results) {

		double org_x = pr["measured_x"];
		double org_y = pr["measured_y"];
		double ret_org_x = org_x;
		double ret_org_y = org_y;

		double std_x = pr["std_x"];
		double std_y = pr["std_y"];
		int index = pr["index"];
		// 修正坐标系并计算偏差值
		double x_off = 0;
		double y_off = 0;
		double x = enable_coord_adj ? org_x - adj_x : org_x;
		double y = enable_coord_adj ? org_y - adj_y : org_y;
		pr["measured_x"] = x;
		pr["measured_y"] = y;

		x_off = std_x - x;
		y_off = std_y - y;

		cv::Point2f img_pt = {pr["img_x"], pr["img_y"]};



		// 如果超公差，重新在红图找针尖
		// 1、用标准框截图 2、判断红图的blob状态 3、计算针尖，满足条件的重新赋值
		bool enable_FindPinByImage2 = params["enable_FindPinByImage2"];
        bool enable_FindPinByImage1 = Tival::JsonHelper::GetParam(params, "enable_FindPinByImage1", false);
		if ((std::abs(x_off) > tolerance || std::abs(y_off) > tolerance) && enable_FindPinByImage2)
		{
			int tmp_index = pr["index"];
			int thr_roi = params["r_pin_gv_min"];

			cv::Point2f std_point=cv::Point2f(std_x, std_y);
			//更新像素当量
			reget_ppum(index, ppum, params, enable_offset, 1, 0);

			std_point.x = Mm2Px(std_x, ppum) ;
			reget_ppum(index, ppum, params, enable_offset, 0, 1);

			std_point.y = Mm2Px(std_y, ppum);

			cv::Point2f std_pm = TransPoint(M.inv(), std_point);
			float temp_x = std_pm.x - roi_lt_x;
			float temp_y = std_pm.y - roi_lt_y;

			// 红图矩形框 判断是否超限
			int temp_lt_x = int(temp_x - tolerance_px * 2);
			int temp_lt_y = int(temp_y - tolerance_px * 2);
			int temp_width = tolerance_px * 4;
			int temp_height = tolerance_px * 4;

			int rect_lt_x = std::max(temp_lt_x, 0);
			int rect_lt_y = std::max(temp_lt_y, 0);
			int rect_width = std::min(temp_width, img2_gray.cols - rect_lt_x);
			int rect_height = std::min(temp_height, img2_gray.rows - rect_lt_y);
			cv::Rect red_rect_roi(rect_lt_x, rect_lt_y, rect_width, rect_height);

			// 白图矩形框
			temp_lt_x = int(temp_x - tolerance_px);
			temp_lt_y = int(temp_y - tolerance_px);
			temp_width = tolerance_px * 2;
			temp_height = tolerance_px * 2;

			rect_lt_x = std::max(temp_lt_x, 0);
			rect_lt_y = std::max(temp_lt_y, 0);
			rect_width = std::min(temp_width, img1_gray.cols - rect_lt_x);
			rect_height = std::min(temp_height, img1_gray.rows - rect_lt_y);
			cv::Rect white_rect_roi(rect_lt_x, rect_lt_y, rect_width, rect_height);

			cv::Mat red_roi ;
			cv::Mat bin_roi ;

			red_roi = img2_gray(red_rect_roi);
			bin_roi = bin_img2(red_rect_roi);


			cv::Mat binary_roi, bin_roi_ref;
			if (thr_roi <= 10) { thr_roi = 10; }

			cv::threshold(red_roi, binary_roi, thr_roi, 255, cv::THRESH_BINARY);
			cv::threshold(bin_roi, bin_roi_ref, thr_roi, 255, cv::THRESH_BINARY);

			// 可能出现在红图定位时，被卡掉的位置在red_rect_roi内有满足条件的针尖，导致漏检，因此使用bin_roi_ref作为参考
			std::vector<std::vector<cv::Point>> contorus_ref;
			cv::findContours(bin_roi_ref, contorus_ref, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
			bool ref_flag = false;
			bool is_heighter = false;
			if (contorus_ref.size() > 0) {
				for (auto r_cont : contorus_ref) {
					cv::Rect r_bbox = cv::boundingRect(r_cont);
					double r_area = cv::contourArea(r_cont);
					if (r_area == 0) r_area = r_bbox.width * r_bbox.height;
					if (r_area > blob_area_min && r_area < blob_area_max * 3 &&
						r_bbox.width > blob_width_min && r_bbox.width < blob_width_max * 1.2 &&
						r_bbox.height > blob_height_min && r_bbox.height < blob_height_max * 1.2) {
						ref_flag = true;
					}
					if (r_bbox.height > r_bbox.width) {
						is_heighter = true;
					}
				}
			}
			// 将连在一起的blobd分开
			std::vector<std::vector<cv::Point>> contorus_roi;
			cv::Mat labels;
			int num_labels = cv::connectedComponents(binary_roi, labels, 8, CV_32S);
			if (num_labels == 2) {
				if (is_heighter) {
					cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(blob_width_min, 1));
					cv::morphologyEx(binary_roi, binary_roi, cv::MORPH_OPEN, kernel);
				}
				else {
					cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(1, blob_height_min));
					cv::morphologyEx(binary_roi, binary_roi, cv::MORPH_OPEN, kernel);
				}
			}
			cv::findContours(binary_roi, contorus_roi, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
            cv::Point2f m_center = {-1, -1};
            bool        isRefind = false;
            if (contorus_roi.size() > 0 && ref_flag) {
                m_center = Refind_pin_fromImg2(binary_roi, contorus_roi, params);

                if (m_center.x != -1) {
                    isRefind   = true;
                    m_center.x = m_center.x + int(temp_x - tolerance_px * 2) + roi_lt_x;
                    m_center.y = m_center.y + int(temp_y - tolerance_px * 2) + roi_lt_y;
                    LOGD("execate Refind_pin_fromImg2! Point.x:{}, Point.y:{}", m_center.x, m_center.y);
                }
                // 如果红图没有找到，在白图框内找
                if (!isRefind && enable_FindPinByImage1) {
                    cv::Mat whiteImg = img1_gray(white_rect_roi);
                    m_center         = Refind_pin_fromImg1(whiteImg, params);
                    if (m_center.x != -1) {
                        isRefind   = true;
                        m_center.x = m_center.x + int(temp_x - tolerance_px) + roi_lt_x;
                        m_center.y = m_center.y + int(temp_y - tolerance_px) + roi_lt_y;
                        LOGD("execate Refind_pin_fromImg1! Point.x:{}, Point.y:{}", m_center.x, m_center.y);
                    }
                }

                json        pin_img_pos = {0, 0};
                cv::Point2f measure_pos = {0, 0};
                // 重新赋值
                if (isRefind) {

                    // 更新像素当量
                    reget_ppum(index, ppum, params, enable_offset, 1, 0);
                    measure_pos.x = Px2Mm(TransPoint(M, m_center), ppum).x;
                    reget_ppum(index, ppum, params, enable_offset, 0, 1);
                    measure_pos.y = Px2Mm(TransPoint(M, m_center), ppum).y;

                    img_pt.x = TransPoint(M, m_center).x;
                    img_pt.y = TransPoint(M, m_center).y;

                    x                = enable_coord_adj ? measure_pos.x - adj_x : org_x;
                    y                = enable_coord_adj ? measure_pos.y - adj_y : org_y;
                    pr["measured_x"] = x;
                    pr["measured_y"] = y;

                    x_off = std_x - x;
                    y_off = std_y - y;

                    pin_img_pos[0]  = m_center.x;
                    pin_img_pos[1]  = m_center.y;
                    pr["pin_coord"] = pin_img_pos;
                }
            }
        }

		// 偏差值过大，可能根本就没找到，偏差值设置为99表示NG
		if (std::abs(x_off) > tolerance * 3 || std::abs(y_off) > tolerance * 3) {
			x_off = -1;
			y_off = -1;
		}

		//更新像素当量
		reget_ppum(index, ppum, params, enable_offset, 1, 0);
		cv::Point2f std_pos = cv::Point2f(std_x, std_y);

		std_pos.x = Mm2Px(std_x,  ppum) ;
		reget_ppum(index, ppum, params, enable_offset, 0, 1);

		std_pos.y = Mm2Px(std_y, ppum);

		if (enable_coord_adj) {
			reget_ppum(index, ppum, params, enable_offset, 1, 0);
			//std_pos += Mm2Px(cv::Point2f(adj_x, adj_y), ppum);
			std_pos.x += Mm2Px(adj_x, ppum);
			reget_ppum(index, ppum, params, enable_offset, 0, 1);
			std_pos.y += Mm2Px(adj_y, ppum);

		}
		json std_box = StdPoint2Box(M, std_pos, tolerance_px);
		/* pr["std_x"] = std_pos.x;
		 pr["std_y"] = std_pos.y;*/
		bool is_ok = false;
		//误差计算方式 2024年8月2日13:44:10
		if (error_type==0) {
			if (std::abs(x_off) <= tolerance && std::abs(y_off) <= tolerance) {
				is_ok = true;
				result_state = 1;
			}
			else {
				is_ok = false;
				result_state = 0;
			}
		}
		else if(error_type==1){
			if (std::sqrtf(std::powf(x_off, 2) + std::powf(y_off, 2)) <= tolerance) {
				is_ok = true;
				result_state = 1;
			}
			else {
				is_ok = false;
				result_state = 0;
			}
		}

		//if (/*std::abs(x_off) <= tolerance && std::abs(y_off) <= tolerance*/ std::sqrtf(std::powf(x_off,2) +std::powf(y_off,2))<= tolerance) {
		//	is_ok = true;
		//	result_state = 1;
		//}
		//else {
		//	is_ok = false;
		//	result_state = 0;
		//}
		// 记录计算显示值偏移量
		if (index == 0) {
			first_std = std_x;
			first_measure = x;
		}
		if (index == 1) {
			second_std = std_x;
			second_measure = x;
		}

		pr["std_box"] = std_box;
		pr["x_off"] = x_off;
		pr["y_off"] = y_off;
		pr["is_ok"] = is_ok;
		pr["result"] = result_state;
		pr["TP"] = 2 * sqrt(x_off * x_off + y_off * y_off);
		pr["org_x"] = ret_org_x;
		pr["org_y"] = ret_org_y;
        pr["img_x"] = img_pt.x;
        pr["img_y"] = img_pt.y;

	}
	differ_std = std::abs(second_std - first_std);
	differ_measure = std::abs(second_measure - first_measure);
	// LOGI("标准坐标偏移距离：{}", differ_std);
	// LOGI("测量坐标偏移距离：{}", differ_measure); // 前两个测量值可能会歪，因此实际都使用标准偏移

	return results;
}

// 红图ROI中重新定位针尖
bool compareContourAreas(std::vector<cv::Point> contour1, std::vector<cv::Point> contour2) {
	double area1 = cv::contourArea(contour1);
	double area2 = cv::contourArea(contour2);
	return (area1 > area2);
}

cv::Point2f FrontPinDetect::Refind_pin_fromImg2(cv::Mat& binary_roi, std::vector<std::vector<cv::Point>>& contorus, const json& params)
{
    cv::Point2f                         m_center = {-1, -1};
    std::vector<std::vector<cv::Point>> contorus_roi;   // 保留真正的轮廓
    for (auto cont : contorus) {
        cv::Rect boundingRect = cv::boundingRect(cont);
        // float w = boundingRect.width;
        // float h = boundingRect.height;
        // float area = cv::contourArea(cont);

        if (cv::contourArea(cont) > 5) {
            contorus_roi.push_back(cont);
        }
    }
    if (contorus_roi.size() < 1) {
        return m_center;
    }

    std::vector<cv::Point2f> con_center;
    std::vector<double>      con_area;
    std::vector<double>      con_width;
    std::vector<double>      con_height;

    if (contorus_roi.size() > 2) {
        std::sort(contorus_roi.begin(), contorus_roi.end(), compareContourAreas);
        std::vector<std::vector<cv::Point>>::iterator start = contorus_roi.begin() + 2;   // 第3个元素的迭代器
        std::vector<std::vector<cv::Point>>::iterator end   = contorus_roi.end();         // 结束位置为向量的末尾
        contorus_roi.erase(start, end);
    }

    if (contorus_roi.size() == 2) {
        for (auto cont : contorus_roi) {
            cv::Moments moment = cv::moments(cont, false);
            cv::Point2f center(moment.m10 / moment.m00, moment.m01 / moment.m00);
            con_center.push_back(center);

            con_area.push_back(cv::contourArea(cont));
            con_width.push_back(cv::boundingRect(cont).width);
            con_height.push_back(cv::boundingRect(cont).height);
        }
        // 需要将条件卡严
        double max_area   = std::max(con_area[0], con_area[1]);
        double max_width  = std::max(con_width[0], con_width[1]);
        double max_height = std::max(con_height[0], con_height[1]);
        double min_area   = std::min(con_area[0], con_area[1]);
        double min_width  = std::min(con_width[0], con_width[1]);
        double min_height = std::min(con_height[0], con_height[1]);

        double dx = std::abs(con_center[0].x - con_center[1].x);
        double dy = std::abs(con_center[0].y - con_center[1].y);

        double blob_area_min   = params["r_blob_area_min"];
        double blob_area_max   = params["r_blob_area_max"];
        double blob_width_min  = params["r_blob_width_min"];
        double blob_width_max  = params["r_blob_width_max"];
        double blob_height_min = params["r_blob_height_min"];
        double blob_height_max = params["r_blob_height_max"];

        // 上下结构
        if (dy >= 10) {
            if (max_area < min_area * 5 && max_area < blob_area_max / 2.0 && max_width < min_width * 2.0 && max_width < blob_width_max) {
                m_center = cv::Point2f((con_center[0].x + con_center[1].x) / 2, (con_center[0].y + con_center[1].y) / 2);
            }
        }
        // 左右结构
        else {
            if (max_area < min_area * 5 && max_area < blob_area_max / 2.0 && max_height < min_height * 2.0 && max_height < blob_height_max) {
                m_center = cv::Point2f((con_center[0].x + con_center[1].x) / 2, (con_center[0].y + con_center[1].y) / 2);
            }
        }
    }
    return m_center;
}

cv::Point2f FrontPinDetect::Refind_pin_fromImg1(cv::Mat& white_roi, const json& params)
{
    cv::Point2f center    = {-1, -1};
    int         white_thr = params["w_pin_gv_min"];
    int         area_thr  = params["w_area_target"];
    double      tolerance = params["tolerance"];
    double      pixel     = params["ppum"];
    double      distThr   = tolerance / pixel * 1000.0 * 0.7;
    cv::Mat binary;
    cv::threshold(white_roi, binary, 230, 255, cv::THRESH_BINARY);
    // 连通域分析
    cv::Mat labels;
    int     num_labels = cv::connectedComponents(binary, labels, 8, CV_32S);

    float max_area = -1;
    for (int label = 1; label < num_labels; ++label) {
        // 遍历每个连通域
        cv::Mat                label_mask = (labels == label);		
        std::vector<std::vector<cv::Point>> white_cons;
        cv::findContours(label_mask, white_cons, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
        cv::Moments M = cv::moments(white_cons[0], false);
        if (M.m00 <= 0) {
            continue;
        }
        cv::Point2f maskCentP = cv::Point2f(M.m10 / M.m00, M.m01 / M.m00);
        // 计算与中心的距离
        cv::Point2f imgCentP = {-1, -1};
        imgCentP.x           = white_roi.cols / 2.0;
        imgCentP.y           = white_roi.rows / 2.0;
        double distance      = cv::norm(imgCentP - maskCentP);
        if (distance > distThr) {
            continue;
		}
        // 计算包围框面积
        std::vector<cv::Point> component_points;
        cv::findNonZero(label_mask, component_points);
        float blob_area = component_points.size();
        if (blob_area > area_thr / 2.0 && blob_area > max_area) {            
            max_area      = blob_area;
            center   = maskCentP;
        }
    }
    
    return center;
}


bool FrontPinDetect::IsEmptyPos(int x, int y, const json& params)
{
	json x_empty = params["x_empty"];
	json y_empty = params["y_empty"];
	for (int i = 0; i < x_empty.size(); i++) {
		if (x == x_empty[i] && y == y_empty[i]) {
			return true;
		}
	}
	return false;
}

// 基于pin针坐标系下的pin针图纸坐标点，生成原图坐标系下的标准框坐标（旋转正方形）
json FrontPinDetect::StdPoint2Box(const cv::Mat& M, const cv::Point2f& pt, double tolerance_px)
{
	json std_box = json::array();
	cv::Point2f lt = TransPoint(M.inv(), cv::Point2f(pt.x - tolerance_px, pt.y - tolerance_px));
	cv::Point2f rt = TransPoint(M.inv(), cv::Point2f(pt.x + tolerance_px, pt.y - tolerance_px));
	cv::Point2f lb = TransPoint(M.inv(), cv::Point2f(pt.x - tolerance_px, pt.y + tolerance_px));
	cv::Point2f rb = TransPoint(M.inv(), cv::Point2f(pt.x + tolerance_px, pt.y + tolerance_px));

	double a = DistPP(lt, rt);
	double b = DistPP(rt, lb);
	double c = DistPP(lb, rb);
	double d = DistPP(rb, lt);
	std_box.push_back({ lt.x, lt.y });
	std_box.push_back({ rt.x, rt.y });
	std_box.push_back({ rb.x, rb.y });
	std_box.push_back({ lb.x, lb.y });
	return std_box;
}

// 坐标点仿射变换
cv::Point2f FrontPinDetect::TransPoint(const cv::Mat& M, const cv::Point2f& point)
{
	std::vector<double> values = { point.x, point.y };
	cv::Mat mat = cv::Mat(values).clone(); //将vector变成单列的mat，这里需要clone(),因为这里的赋值操作是浅拷贝
	cv::Mat dest = mat.reshape(1, 1);

	cv::Mat homogeneousPoint = (cv::Mat_<double>(3, 1) << point.x, point.y, 1.0);
	cv::Mat transformed = M * homogeneousPoint;
	return cv::Point2f(transformed.at<double>(0, 0), transformed.at<double>(0, 1));
}

// 遍历所有pin针位置，获取离参考点最近的pin
int FrontPinDetect::GetNearestPointIdx(std::vector<cv::Point2f> points, cv::Point2f refPoint, double& minDist)
{
	minDist = 99999;
	int minDistIdx = -1;
	for (int i = 0; i < points.size(); i++) {
		double dist = std::pow(refPoint.x - points[i].x, 2) + std::pow(refPoint.y - points[i].y, 2);
		//double dist = std::abs(refPoint.x - points[i].x) + std::abs(refPoint.y - points[i].y);
		if (dist < minDist) {
			minDist = dist;
			minDistIdx = i;
		}
	}
	return minDistIdx;
}

// 计算标线位置
json FrontPinDetect::CalcStdLines(const cv::Mat& M, const json& params, double adj_x, double adj_y, float differ_std)
{
	double ppum = params["ppum"];
	double tolerance = params["tolerance"];
	json x_coords1 = params["x_coords1"];
	json x_coords2 = params["x_coords2"];
	json y_coords1 = params["y_coords1"];
	json y_coords2 = params["y_coords2"];
	float gap = params["gap"];
    bool enable_offset = Tival::JsonHelper::GetParam(params, "enable_offset", false);

	double tolerance_px = Mm2Px(tolerance, ppum);
	json enable_coord_adj = Utils::GetProperty(params, "enable_coord_adj", ENABLE_COORD_ADJ_DEFAULT);

	json lines = json::array();


	cv::Point2f coord_offset = Mm2Px(cv::Point2f(adj_x, adj_y), ppum);
	// 坐标轴
	cv::Point2f origin = cv::Point2f(0, 0);								// 不偏移
	// X\Y 轴，标线两端延长500px
	cv::Point2f axis_x_end = cv::Point2f(Mm2Px(x_coords2[x_coords2.size() - 1], ppum) + 500, 0);
	cv::Point2f axis_y_end = cv::Point2f(0, Mm2Px(y_coords2[y_coords2.size() - 1], ppum) + 500);							// 不偏移

	//对gap进行y方向整体偏移（用于针尖不在同一条线上）
	if (enable_offset){
		origin = cv::Point2f(Mm2Px(differ_std, ppum), -Mm2Px(gap, ppum));										// 偏移（x，y轴起点x坐标）
		axis_x_end = cv::Point2f(Mm2Px(x_coords2[x_coords2.size() - 1], ppum) + 500, -Mm2Px(gap, ppum));						// 偏移（X轴终点y坐标）
		axis_y_end = cv::Point2f(Mm2Px(differ_std, ppum), Mm2Px(y_coords2[y_coords2.size() - 1], ppum) + 500);  // 偏移（y轴终点x坐标）
	}


	if (enable_coord_adj) {
		// 修正坐标系
		origin += coord_offset;
		axis_x_end += coord_offset;
		axis_y_end += coord_offset;
	}
	origin = TransPoint(M.inv(), origin);
	axis_x_end = TransPoint(M.inv(), axis_x_end);
	axis_y_end = TransPoint(M.inv(), axis_y_end);
	lines.push_back({ {origin.x, origin.y}, {axis_x_end.x, axis_x_end.y} });
	lines.push_back({ {origin.x, origin.y}, {axis_y_end.x, axis_y_end.y} });

	// 每行标线
	//for (int i = 1; i < y_coords1.size(); i++) {
	//    double start_x = x_coords1[0];
	//    double end_x = x_coords1[x_coords1.size() - 1];
	//    double y_coord = y_coords1[i];
	//    start_x = Mm2Px(start_x, ppum);
	//    end_x = Mm2Px(end_x, ppum) + 100;
	//    y_coord = Mm2Px(y_coord, ppum);

	//    cv::Point2f line_start = cv::Point2f(start_x, y_coord);
	//    cv::Point2f line_end = cv::Point2f(end_x, y_coord);
	//    if (enable_coord_adj) {
	//        line_start += coord_offset;
	//        line_end += coord_offset;
	//    }
	//    line_start = TransPoint(M.inv(), line_start);
	//    line_end = TransPoint(M.inv(), line_end);
	//    lines.push_back({ {line_start.x, line_start.y}, {line_end.x, line_end.y} });
	//}
	return lines;
}
void FrontPinDetect::DrawResults(cv::Mat& image, const json& lines, const json& results)
{
	for (auto line : lines) {
		cv::Point2f line_start(line[0][0], line[0][1]);
		cv::Point2f line_end(line[1][0], line[1][1]);
		cv::line(image, line_start, line_end, cv::Scalar(0, 255, 0), 1);
	}

	for (auto rst : results) {
		bool is_ok = rst["is_ok"];
		json std_box = rst["std_box"];
		//json pin_coord = rst["pin_coord"];
		cv::Point2i pin_coord(rst["pin_coord"][0], rst["pin_coord"][1]);
		double x_off = rst["x_off"];
		double y_off = rst["y_off"];

		cv::Point2i lt = cv::Point(int(std_box[0][0] + 0.5), int(std_box[0][1] + 0.5));
		cv::Point2i rt = cv::Point(int(std_box[1][0] + 0.5), int(std_box[1][1] + 0.5));
		cv::Point2i rb = cv::Point(int(std_box[2][0] + 0.5), int(std_box[2][1] + 0.5));
		cv::Point2i lb = cv::Point(int(std_box[3][0] + 0.5), int(std_box[3][1] + 0.5));
		cv::Point2i box_cent((lt.x + rt.x + rb.x + lb.x) / 4, (lt.y + rt.y + rb.y + lb.y) / 4);

		cv::Scalar box_color = is_ok ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255);
		cv::line(image, lt, rt, box_color, 1);
		cv::line(image, rt, rb, box_color, 1);
		cv::line(image, rb, lb, box_color, 1);
		cv::line(image, lb, lt, box_color, 1);

		cv::line(image, cv::Point(pin_coord.x - 10, pin_coord.y), cv::Point(pin_coord.x + 10, pin_coord.y), cv::Scalar(255, 0, 0), 1);
		cv::line(image, cv::Point(pin_coord.x, pin_coord.y - 10), cv::Point(pin_coord.x, pin_coord.y + 10), cv::Scalar(255, 0, 0), 1);

		box_cent.x -= 40;
		box_cent.y -= 20;
		cv::putText(image, fmt::format("{:.3f},{:.3f}", x_off, y_off), box_cent, cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
		cv::circle(image, cv::Point(int(pin_coord.x + 0.5), int(pin_coord.y + 0.5)), 2, cv::Scalar(255, 255, 0), -1);
	}
}

void FrontPinDetect::DrawOrgResults(cv::Mat& image, std::vector<PinInfoPtr>& pin_infos)
{
	for (auto pin_info : pin_infos) {
		if (pin_info->found) {
			cv::Point pin_coord = pin_info->pin_center;
			cv::line(image, cv::Point(pin_coord.x - 10, pin_coord.y), cv::Point(pin_coord.x + 10, pin_coord.y), cv::Scalar(0, 255, 0), 1);
			cv::line(image, cv::Point(pin_coord.x, pin_coord.y - 10), cv::Point(pin_coord.x, pin_coord.y + 10), cv::Scalar(0, 255, 0), 1);
		}
	}
	//cv::imwrite("D:/1.jpg", image);
}

// 根据源向量（带方向的点）和目标向量，生成仿射变换矩阵
cv::Mat FrontPinDetect::vector_angle_to_M(double x1, double y1, double d1, double x2, double y2, double d2)
{
	cv::Point2f center(x1, y1);
	double angle = d2 - d1;
	cv::Mat rot_M = cv::getRotationMatrix2D(center, angle, 1.0);
	rot_M = cvMat6_to_cvMat9(rot_M);

	cv::Mat trans_M = d6_to_cvMat(1, 0, x2 - x1, 0, 1, y2 - y1);
	cv::Mat M = trans_M * rot_M; // 先旋转在平移（矩阵乘法相反）
	return M;
}

// 讲OpenCV输出矩阵转换为齐次坐标格式，2x3 => 3x3
cv::Mat FrontPinDetect::cvMat6_to_cvMat9(const cv::Mat& mtx6)
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

cv::Mat FrontPinDetect::d6_to_cvMat(double d0, double d1, double d2, double d3, double d4, double d5)
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

// 求两点距离
double FrontPinDetect::DistPP(const cv::Point2f& a, const cv::Point2f& b)
{
	return std::sqrt(std::pow(a.x - b.x, 2) + std::pow(a.y - b.y, 2));
}


#include <limits>
float FrontPinDetect::calculateSlope(const cv::Point2f& p1, const cv::Point2f& p2) {
	if (p1.x == p2.x) {
		return std::numeric_limits<float>::infinity();
	}
	return (p2.y - p1.y) / (p2.x - p1.x);
}


double FrontPinDetect::reget_angle(std::vector<cv::Point2f> pts, int min_points_in_line) {
	std::map<float, std::vector<std::pair<int, int>>> line_groups;
	for (size_t i = 0; i < pts.size(); ++i)
	{
		for (size_t j = i + 1; j < pts.size(); ++j)
		{
			float slope = calculateSlope(pts[i], pts[j]);
			float slope_key = roundf(slope * 100) / 100;
			line_groups[slope_key].push_back(std::make_pair(i, j));
		}
	}
	int max_value = 0;
	double angle = 0;
	for (const auto& pair : line_groups)
	{
		if (pair.second.size() >= min_points_in_line)
		{
			if (pair.second.size() >= max_value) {
				max_value = pair.second.size();
				angle = pair.first;
			}
		}
	}
	angle = atanl(angle) * 180.0 / CV_PI;
	return angle;
}

bool FrontPinDetect::check_folder_state(std::string folderPath) {
	if (!fs::exists(folderPath)) {
		if (fs::create_directories(folderPath)) {
			std::cout << "Folder created successfully!" << std::endl;
			return true;
		}
		else {
			std::cerr << "Failed to create folder!" << std::endl;
			return false;
		}
	}
	else {
		std::cout << "Folder already exists." << std::endl;
		return true;
	}

}

void FrontPinDetect::RotateImage(cv::Mat& image, int mode)
{
	if (mode == 90) {
		rotate(image, image, cv::ROTATE_90_CLOCKWISE);
	}
	else if (mode == -90) {
		rotate(image, image, cv::ROTATE_90_COUNTERCLOCKWISE);
	}
	else if (mode == 180 || mode == -180) {
		rotate(image, image, cv::ROTATE_180);
	}
	else {}
}

void  FrontPinDetect::reget_ppum(const int& index, double& ppum, const json &params,const bool& enable_offset,int x,int y) {
	//不启用偏移，不改变像素当量，直接返回
	if (!enable_offset) {
		if (x == 1 && y == 0) {
			ppum = Tival::JsonHelper::GetParam(params, "ppum", ppum);
		}
		if (x == 0 && y == 1) {
			ppum = Tival::JsonHelper::GetParam(params, "ppum_y", ppum);
		}
		return;
	}
	//四连体改变像素当量
	if (enable_offset) {
		if (index <= 159) {
			if (x==1 &&y==0) {
				ppum = Tival::JsonHelper::GetParam(params, "ppum", ppum);
			}
			if (x==0 && y==1) {
				ppum = Tival::JsonHelper::GetParam(params, "ppum_y", ppum);
			}
		}
		if (index > 159 && index < 319) {
			if (x == 1 && y == 0) {
				ppum = Tival::JsonHelper::GetParam(params, "ppum_2_x", ppum);
			}
			if (x == 0 && y == 1) {
				ppum = Tival::JsonHelper::GetParam(params, "ppum_2_y", ppum);
			}
		}
		if (index > 319 && index <= 479) {
			if (x == 1 && y == 0) {
				ppum = Tival::JsonHelper::GetParam(params, "ppum_3_x", ppum);
			}
			if (x == 0 && y == 1) {
				ppum = Tival::JsonHelper::GetParam(params, "ppum_3_y", ppum);
			}
		}
		if (index >= 479 && index < 639) {
			if (x == 1 && y == 0) {
				ppum = Tival::JsonHelper::GetParam(params, "ppum_4_x", ppum);
			}
			if (x == 0 && y == 1) {
				ppum = Tival::JsonHelper::GetParam(params, "ppum_4_y", ppum);
			}
		}
	}
	return;
}

cv::Mat FrontPinDetect::AlignTransform(cv::Mat& image1, cv::Mat& pin_img, std::vector<PinInfoPtr>& pin_infos, const json& params)
{
    std::vector<cv::Point2f> whitePoints;
    std::vector<cv::Point2f> redPoints;
    for (auto pin_info : pin_infos) {

        if (!pin_info->classify_status) {
            continue;
        }

        cv::Mat pin_img = image1(pin_info->bbox);
        cv::Mat binary;
        cv::threshold(pin_img, binary, 240, 255, cv::THRESH_BINARY);
        std::vector<std::vector<cv::Point>> contours;
        std::vector<cv::Point>              maxCont;
        cv::findContours(binary, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        int maxArea = -1;
        int count = 0;
        for (auto con:contours){
            int area = cv::contourArea(con);
            if (area > 10){
                count++;
                if (area > maxArea) {
                    maxCont = con;
                    maxArea = area;
				}
			}
		}
        if (count > 1 || maxCont.size()<1)
            continue;

		// 符合条件的点
        cv::Moments mu = cv::moments(maxCont);
        whitePoints.emplace_back(mu.m10 / mu.m00 + pin_info->bbox.x, mu.m01 / mu.m00 + pin_info->bbox.y);
        float x = pin_info->bbox.x + pin_info->bbox.width / 2.0;
        float y = pin_info->bbox.y + pin_info->bbox.height / 2.0;
        redPoints.emplace_back(x, y);
    }
    if (whitePoints.size() < 3) {
        return image1;
	}

    // 计算刚性变换矩阵    
    alignMatrix = cv::estimateAffinePartial2D(redPoints, whitePoints, {}, 8, 3);
    cv::Mat transformImg;
    cv::warpAffine(pin_img, transformImg, alignMatrix, pin_img.size());	
    alignMatrix.convertTo(alignMatrix, CV_32F);
	// 矩阵逆变换
	/*cv::Mat affineMatrix3x3 = cv::Mat::eye(3, 3, CV_32F);
    alignMatrix.copyTo(affineMatrix3x3(cv::Rect(0, 0, 3, 2)));
    cv::Mat dstMat;
    cv::invert(affineMatrix3x3, dstMat);
    alignMatrix = affineMatrix3x3(cv::Rect(0, 0, 3, 2));*/


	// 变换坐标
    for (auto pin_info : pin_infos) {        
        cv::Point2i pointLT = {pin_info->bbox.x, pin_info->bbox.y};        
        cv::Mat_<float> pointHomogeneous = (cv::Mat_<float>(3, 1) << pointLT.x, pointLT.y, 1);

        pointHomogeneous.convertTo(pointHomogeneous, CV_32F);

        cv::Mat transformedPoint = alignMatrix * pointHomogeneous;
        pointLT.x                  = std::max(int(transformedPoint.at<float>(0, 0)), 0);
        pointLT.y                = std::max(int(transformedPoint.at<float>(1, 0)), 0);

		// 超限判断
        if (pointLT.x + pin_info->bbox.width > pin_img.cols) {
            pointLT.x = pin_img.cols - pin_info->bbox.width - 1;
        }
        if (pointLT.y + pin_info->bbox.height > pin_img.rows) {
            pointLT.y = pin_img.rows - pin_info->bbox.height - 1;
        }

		pin_info->bbox.x = pointLT.x;
        pin_info->bbox.y = pointLT.y;
    }

    return transformImg;
}

bool FrontPinDetect::CheckBlobByClassify(std::vector<cv::Mat>& imgList, std::vector<int>& indexList, std::vector<PinInfoPtr>& pin_infos, double confThr, const json& params)
{
    if (imgList.size()<0) {
        return false;
	}
    bool enable_save_cropImage = Tival::JsonHelper::GetParam(params, "enable_save_cropImage", false);

    TaskInfoPtr _cls_task       = std::make_shared<stTaskInfo>();
    _cls_task->imageData        = {imgList};
    _cls_task->modelId          = AI_modelIndex;
    _cls_task->taskId           = 0;
    ModelResultPtr clsResultPtr = GetAIRuntime()->RunInferTask(_cls_task);

	// 推理结果
    std::vector<bool> isOK(imgList.size(), true);
    if (clsResultPtr->itemList.size() == 0) {
        std::fill(isOK.begin(), isOK.end(), false);
    }
    for (int m = 0; m < imgList.size(); m++) {
        auto clsRstList = clsResultPtr->itemList[m];
        if (clsRstList.size() == 0) {
            isOK[m] = false;
            continue;
        }
        int   code = clsRstList[0].code;   // 1: OK, 0:NG
        float conf = clsRstList[0].confidence;

        // 如果模型判断为NG, 且分数大于设定阈值（参数配置），则判为false
        if (code == 0 && conf >= confThr) {
            isOK[m] = false;
            int x   = pin_infos[indexList[m]]->bbox.x;
            int y   = pin_infos[indexList[m]]->bbox.y;
            LOGI("Classify predict NG: box lt point is:{}, {}, confidence:{}", x + detect_lt_x, y + detect_lt_y, conf);
            if (enable_save_cropImage){
                cv::imwrite(pin_infos[indexList[m]]->ngName, imgList[m]);
			}
        }
        else {
            isOK[m] = true;
            if (enable_save_cropImage) {
                cv::imwrite(pin_infos[indexList[m]]->okName, imgList[m]);
            }
        }
    }

	// 解析模型结果
    for (int i = 0;i < isOK.size(); i++){
        if (!isOK[i]){
            pin_infos[indexList[i]]->classify_status = false;
		}
	}

    return true;
}

bool FrontPinDetect::CheckBlobByShape(cv::Mat& sub_tmp_bin, cv::Mat& sub_whiteImg, cv::Mat& sub_SvmImg, std::vector<cv::Point>& sub_con, const json& params)
{
	bool enabel_svmCheck = Tival::JsonHelper::GetParam(params, "enabel_svmCheck", false);
	std::string productName = Tival::JsonHelper::GetParam(params, "product", std::string(""));
	float ngConf = Tival::JsonHelper::GetParam(params, "ngConf", 0.85);

	bool enabel_blackSpot = Tival::JsonHelper::GetParam(params, "enabel_blackSpot", false);
	bool enabel_whiteSpot = Tival::JsonHelper::GetParam(params, "enabel_whiteSpot", false);
	bool enabel_areaRatio = Tival::JsonHelper::GetParam(params, "enabel_areaRatio", false);
	int blackSpot_gray = Tival::JsonHelper::GetParam(params, "blackSpot_gray", 23);
	int blackSpot_area = Tival::JsonHelper::GetParam(params, "blackSpot_area", 60);
	int lightPin_gray = Tival::JsonHelper::GetParam(params, "lightPin_gray", 200);
	int lightPin_Area = Tival::JsonHelper::GetParam(params, "lightPin_Area", 200);
	float areaRatio_Thr = Tival::JsonHelper::GetParam(params, "areaRatio_Thr", 0.5);

	bool blobFlag = true;


	// ------------  SVM检查 ----------------//
	if (enabel_svmCheck) {
		// SVM
		if (isInit) {
			bool ng_location_box = false;
			std::vector<cv::Mat> test_img_vec;
			test_img_vec.push_back(sub_SvmImg);
			nao::img::feature::HogTransform test_transform(test_img_vec, 11, 8, 8, cv::Size(80, 80), 1);
			cv::Mat temp_feature = test_transform();
			double prob[2];
			double ret = svm_obj.testFeatureLibSVM(temp_feature, prob);
			//std::cout << "prob[1]:" << prob[1] << std::endl;
			if (prob[1] >= ngConf) {
				//第二个概率大于ngConf表示不正常
				LOGI("filter by shape, NG score:{}, NGConf:{}", prob[1], ngConf);
				return false;
			}
		}
	}

	// ------------占空比检查------------//
	if (enabel_areaRatio) {
		cv::Rect bbox = cv::boundingRect(sub_con);
		std::vector<cv::Point> compPoints;
		cv::findNonZero(sub_tmp_bin, compPoints);
		float blobArea = compPoints.size();
		float bboxArea = bbox.width * bbox.height;
		float areaRatio = blobArea / (bboxArea + 0.0001);

		if (areaRatio < areaRatio_Thr) {
			return false;
		}
	}

	// ---------------黑点、白点检查----------------//
	// 有一种异常的针尖上方会有一团黑点
	if (enabel_blackSpot || enabel_whiteSpot) {
		cv::Mat blackBinary, lightBinary;
		// 黑点
		cv::threshold(sub_whiteImg, blackBinary, blackSpot_gray, 255, cv::THRESH_BINARY_INV);
		cv::Mat kernel_open1 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
		cv::morphologyEx(blackBinary, blackBinary, cv::MORPH_OPEN, kernel_open1);

		std::vector<std::vector<cv::Point>> blobConts;
		std::vector<std::vector<cv::Point>> blobContsNew;
		cv::findContours(blackBinary, blobConts, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);

		int maxBlob = -1;
		int index = -1;
		for (int m = 0; m < blobConts.size(); m++) {
			float bArea = cv::contourArea(blobConts[m]);
			if (bArea > maxBlob) {
				index = m;
				maxBlob = bArea;
			}
		}
		// 针尖亮点
		cv::threshold(sub_whiteImg, lightBinary, lightPin_gray, 255, cv::THRESH_BINARY);
		std::vector<std::vector<cv::Point>> lightConts;
		std::vector<std::vector<cv::Point>> lightContsNew;
		cv::findContours(lightBinary, lightConts, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);

		int lightMaxBlob = -1;
		int lightIndex = -1;
		float light_whratio = 1;
		for (int m = 0; m < lightConts.size(); m++) {
			float bArea = cv::contourArea(lightConts[m]);
			cv::Rect lightBox = cv::boundingRect(lightConts[m]);

			float maxValue = std::max(lightBox.width, lightBox.height);
			float minValue = std::min(lightBox.width, lightBox.height);
			light_whratio = minValue / maxValue;

			if (bArea > lightMaxBlob && light_whratio > 0.38) {
				lightIndex = m;
				lightMaxBlob = bArea;
			}
		}

		// 比较黑点的面积和位置
		float blackCentY, blackCentX, blackCol1, blackCol2, lightCentY, lightCentX, lightCol1, lightCol2;
		if (index > -1) {
			cv::Rect blackBox = cv::boundingRect(blobConts[index]);
			blackCentY = blackBox.y + blackBox.height / 2.0;
			blackCentX = blackBox.x + blackBox.width / 2.0;
			blackCol1 = blackBox.x;
			blackCol2 = blackBox.x + blackBox.width;
		}
		if (lightIndex > -1) {
			cv::Rect lightBox = cv::boundingRect(lightConts[lightIndex]);
			lightCentY = lightBox.y + lightBox.height / 2.0;
			lightCentX = lightBox.x + lightBox.width / 2.0;

			lightCol1 = lightBox.x;
			lightCol2 = lightBox.x + lightBox.width;
		}
		// 黑点
		if (enabel_blackSpot) {
			if (index > -1 && lightIndex > -1) {
				if (maxBlob > blackSpot_area &&
					blackCentY < lightCentY - 5 &&
					lightCol1 < blackCentX && blackCentX < lightCol2) {
					return false;
				}
				// 纯黑点
				else if (maxBlob > blackSpot_area && blackCentY < lightCentY - 5 && lightMaxBlob < 10) {
					return false;
				}
			}
		}
		// 白点
		if (enabel_whiteSpot && lightIndex > -1) {
			if (lightMaxBlob > lightPin_Area) {
				return false;
			}
		}
	}
	return blobFlag;
}

bool FrontPinDetect::CheckBlobByShapeOld(cv::Mat& sub_tmp_bin, cv::Mat& sub_whiteImg, cv::Mat& sub_img_bin, std::vector<cv::Point>& sub_con, cv::Mat& labels, int num_labels, const json& params)
{
	if (num_labels < 2) {
		return false;
	}

	bool r_enable_open_vertical = params["r_enable_open_vertical"];   //是否开启垂直开操作开关
	bool r_enable_open_horizontal = params["r_enable_open_horizontal"];   //是否开启水平开操作开关
	int r_open_vertical_y = params["r_open_vertical_y"];
	int r_open_horizontal_x = params["r_open_horizontal_x"];
	int w_area_target = params["w_area_target"];
	bool enabel_blackSpot = Tival::JsonHelper::GetParam(params, "enabel_blackSpot", false);
	int blackSpot_gray = Tival::JsonHelper::GetParam(params, "blackSpot_gray", 23);
	int blackSpot_area = Tival::JsonHelper::GetParam(params, "blackSpot_area", 60);
	int comp_erode_size = Tival::JsonHelper::GetParam(params, "comp_erode_size", 3);
	int comp_dilate_size = Tival::JsonHelper::GetParam(params, "comp_dilate_size", 7);
	bool enabel_lightSpot = Tival::JsonHelper::GetParam(params, "enabel_lightSpot", false);
	int lightPiece_gray = Tival::JsonHelper::GetParam(params, "lightPiece_gray", 110);
	int lightPiece_area = Tival::JsonHelper::GetParam(params, "lightPiece_area", 220);
	int lightSpot_gray = Tival::JsonHelper::GetParam(params, "lightSpot_gray", 165);
	int lightSpot_area = Tival::JsonHelper::GetParam(params, "lightSpot_area", 10);
	int superBlob_area = Tival::JsonHelper::GetParam(params, "superBlob_area", 200);
	bool enabel_lightPinOpen = Tival::JsonHelper::GetParam(params, "enabel_lightPinOpen", false);
	int lightPin_gray = Tival::JsonHelper::GetParam(params, "lightPin_gray", 200);
	int lightPin_width = Tival::JsonHelper::GetParam(params, "lightPin_width", 15);
	int lightPin_height = Tival::JsonHelper::GetParam(params, "lightPin_height", 20);
	int lightPin_area = Tival::JsonHelper::GetParam(params, "lightPin_area", 40);

	bool blobFlag = true;
	// ------------占空比检查------------//
	cv::Rect bbox = cv::boundingRect(sub_con);
	std::vector<cv::Point> compPoints;
	cv::findNonZero(sub_tmp_bin, compPoints);
	float blobArea = compPoints.size();
	float bboxArea = bbox.width * bbox.height;
	float areaRatio = blobArea / bboxArea;

	if (areaRatio < 0.5) {
		//std::cout << "areaRatio:" << areaRatio << std::endl;
		return false;
	}


	int count = 0;
	int superbigCount = 0;
	int maxArea = -1;
	int maxLabel = 1;

	std::vector<std::vector<cv::Point>> partBlob;
	for (int label = 1; label < num_labels; ++label) {
		// 遍历每个连通域
		cv::Mat label_mask = (labels == label);
		std::vector<cv::Point> tmpPoints;
		cv::findNonZero(label_mask, tmpPoints);
		float tmpArea = tmpPoints.size();
		// 如果面积符合条件
		if (tmpArea > 5) {
			partBlob.push_back(tmpPoints);
			count++;
		}
		if (tmpArea > superBlob_area) {
			superbigCount++;
		}
		// 寻找最大的连通域
		if (tmpArea > maxArea) {
			maxLabel = label;
			maxArea = tmpArea;
		}
	}

	if (num_labels <= 3 && maxArea < 25) {
		return false;
	}

	int lightestMaxPin = -1;
	int lightestPinIndex = -1;
	int lightestWidth = -1;
	int lightestHeight = -1;
	int lightestCount = 0;
	if (count > 2 || superbigCount > 0) {
		cv::Mat labelsDilate, erodeImg, dilateImg;
		if (count > 2) {
			cv::Mat e_kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(comp_erode_size, comp_erode_size));
			cv::erode(sub_img_bin, erodeImg, e_kernel);
			cv::Mat d_kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(comp_dilate_size, comp_dilate_size));
			cv::dilate(erodeImg, dilateImg, d_kernel);
		}
		if (superbigCount > 0) {
			cv::Mat e_kernel1 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(comp_erode_size, comp_erode_size));
			cv::morphologyEx(sub_img_bin, dilateImg, cv::MORPH_OPEN, e_kernel1);
			//cv::Mat e_kernel2 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(7, 1));
			//cv::morphologyEx(dilateImg, dilateImg, cv::MORPH_OPEN, e_kernel2);
		}
		//cv::morphologyEx(sub_img_bin, dilateImg, cv::MORPH_CLOSE, d_kernel);
		int num_labelsDilate = cv::connectedComponents(dilateImg, labelsDilate, 8, CV_32S);

		// 如果有超级大的红图blob，计算白图针尖的形状
		if (superbigCount > 0 || count > 2) {
			cv::Mat listestBin;
			std::vector<std::vector<cv::Point>> lightestConts;
			cv::threshold(sub_whiteImg, listestBin, lightPin_gray, 255, cv::THRESH_BINARY);
			if (enabel_lightPinOpen) {
				cv::Mat kernel_open1, kernel_open2;
				if (bbox.width < bbox.height) {
					kernel_open1 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(1, 3));
					kernel_open2 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 1));
				}
				else {
					kernel_open1 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 1));
					kernel_open2 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(1, 3));
				}
				cv::morphologyEx(listestBin, listestBin, cv::MORPH_OPEN, kernel_open1);
				cv::morphologyEx(listestBin, listestBin, cv::MORPH_OPEN, kernel_open2);
			}
			cv::findContours(listestBin, lightestConts, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);

			for (int m = 0; m < lightestConts.size(); m++) {
				float bArea = cv::contourArea(lightestConts[m]);
				if (lightestConts[m].size() > 5) {
					lightestCount += 1;
				}
				if (bArea > lightestMaxPin) {
					lightestPinIndex = m;
					lightestMaxPin = bArea;
				}
			}

			if (lightestPinIndex != -1) {
				cv::Rect lightestBox = cv::boundingRect(lightestConts[lightestPinIndex]);
				lightestWidth = lightestBox.width;
				lightestHeight = lightestBox.height;
				// 标准针尖
				if (lightestWidth < lightPin_width && lightestHeight < lightPin_height && lightestMaxPin > lightPin_area) {
					num_labelsDilate = num_labels;
				}
				/*else {
					return false;
				}*/
			}
			else if (superbigCount > 0 && lightestPinIndex == -1) {
				return false;
			}
		}

		partBlob.clear();
		for (int label = 1; label < num_labelsDilate; ++label) {
			// 遍历每个连通域
			cv::Mat label_mask = (labelsDilate == label);
			std::vector<cv::Point> tmpPoints;
			cv::findNonZero(label_mask, tmpPoints);
			float tmpArea = tmpPoints.size();

			if (tmpArea > 20) {
				partBlob.push_back(tmpPoints);
			}
		}
	}

	if (partBlob.size() < 1) {
		return false;
	}


	// 黑点检查：有一种异常的针尖上方会有一团黑点
	if (enabel_blackSpot) {
		cv::Mat blackBinary, lightBinary;
		// 黑点
		cv::threshold(sub_whiteImg, blackBinary, blackSpot_gray, 255, cv::THRESH_BINARY_INV);
		cv::Mat kernel_open1 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
		cv::morphologyEx(blackBinary, blackBinary, cv::MORPH_OPEN, kernel_open1);

		std::vector<std::vector<cv::Point>> blobConts;
		std::vector<std::vector<cv::Point>> blobContsNew;
		cv::findContours(blackBinary, blobConts, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);

		int maxBlob = -1;
		int index = -1;
		for (int m = 0; m < blobConts.size(); m++) {
			float bArea = cv::contourArea(blobConts[m]);
			if (bArea > maxBlob) {
				index = m;
				maxBlob = bArea;
			}
		}
		// 针尖亮点
		cv::threshold(sub_whiteImg, lightBinary, 200, 255, cv::THRESH_BINARY);
		std::vector<std::vector<cv::Point>> lightConts;
		std::vector<std::vector<cv::Point>> lightContsNew;
		cv::findContours(lightBinary, lightConts, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);

		int lightMaxBlob = -1;
		int lightIndex = -1;
		for (int m = 0; m < lightConts.size(); m++) {
			float bArea = cv::contourArea(lightConts[m]);
			if (bArea > lightMaxBlob) {
				lightIndex = m;
				lightMaxBlob = bArea;
			}
		}

		// 比较黑点的面积和位置
		float blackCentY, blackCentX, blackCol1, blackCol2, lightCentY, lightCentX, lightCol1, lightCol2;
		if (index > -1) {
			cv::Rect blackBox = cv::boundingRect(blobConts[index]);
			blackCentY = blackBox.y + blackBox.height / 2.0;
			blackCentX = blackBox.x + blackBox.width / 2.0;
			blackCol1 = blackBox.x;
			blackCol2 = blackBox.x + blackBox.width;
		}
		if (lightIndex > -1) {
			cv::Rect lightBox = cv::boundingRect(lightConts[lightIndex]);
			lightCentY = lightBox.y + lightBox.height / 2.0;
			lightCentX = lightBox.x + lightBox.width / 2.0;

			lightCol1 = lightBox.x;
			lightCol2 = lightBox.x + lightBox.width;
		}
		if (index > -1 && lightIndex > -1) {
			if (maxBlob > blackSpot_area &&
				blackCentY < lightCentY - 5 &&
				lightCol1 < blackCentX && blackCentX < lightCol2) {
				return false;
			}
			// 纯黑点
			else if (maxBlob > blackSpot_area && blackCentY < lightCentY - 5 && lightMaxBlob < 10) {
				return false;
			}
		}
	}

	// 白块检查：
	if (enabel_lightSpot && superbigCount > 0 || count > 2) {

		// 亮点及背景：针尖倒了，但能看见鱼眼是亮的
		// 过检：标准针尖亮点周围存在金属屑发亮，灰度约160，导致面积满足条件
		// 固定看亮点

		cv::Mat lightBin;
		cv::threshold(sub_whiteImg, lightBin, lightPiece_gray, 255, cv::THRESH_BINARY);
		std::vector<std::vector<cv::Point>> lightConts;
		cv::findContours(lightBin, lightConts, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);

		int lightMaxPin = -1;
		int lightPinIndex = -1;
		int lightPinCount = 0;
		for (int m = 0; m < lightConts.size(); m++) {
			float bArea = cv::contourArea(lightConts[m]);
			if (lightConts[m].size() >= 30) {
				lightPinCount++;
			}
			if (bArea > lightMaxPin) {
				lightPinIndex = m;
				lightMaxPin = bArea;
			}
		}
		if (lightMaxPin > lightPiece_area && (((lightestWidth > lightPin_width || lightestHeight > lightPin_height) && lightestMaxPin > lightPin_area) || lightestCount > 1)) {
			return false;
		}
		if (lightPinCount >= 3 && lightestCount > 1) {
			return false;
		}

		// 针尖倒了，鱼眼完全看不见亮的
		cv::threshold(sub_whiteImg, lightBin, lightSpot_gray, 255, cv::THRESH_BINARY);
		cv::findContours(lightBin, lightConts, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);

		lightMaxPin = -1;
		lightPinIndex = -1;
		for (int m = 0; m < lightConts.size(); m++) {
			float bArea = cv::contourArea(lightConts[m]);
			if (bArea > lightMaxPin) {
				lightPinIndex = m;
				lightMaxPin = bArea;
			}
		}
		if (lightMaxPin < lightSpot_area) {
			return false;
		}
	}

	// ------------散乱点检查 -------------//
	// 计算每个连通域的质心
	std::vector<cv::Point2f> centroids;
	for (const auto& oneBlob : partBlob) {

		long long sumX = 0;
		long long sumY = 0;
		int numPoints = oneBlob.size();

		for (const auto& pnt : oneBlob) {
			sumX += pnt.x;
			sumY += pnt.y;
		}

		// 计算质心坐标
		cv::Point centroid;
		centroid.x = numPoints > 0 ? static_cast<int>(sumX / numPoints) : 0;
		centroid.y = numPoints > 0 ? static_cast<int>(sumY / numPoints) : 0;

		centroids.push_back(centroid);
	}

	// 计算质心之间的距离
	std::vector<float> distVec;
	std::vector<float> distX;
	std::vector<float> distY;
	std::vector<float> distXWithCent;
	std::vector<float> distYWithCent;
	std::vector<float> distRow1;
	std::vector<float> distRow2;
	std::vector<float> distCol1;
	std::vector<float> distCol2;

	cv::Point2f cent;
	cent.x = sub_tmp_bin.size().width / 2.0;
	cent.y = sub_tmp_bin.size().height / 2.0;
	if (centroids.size() > 1) {
		for (int i = 0; i < centroids.size() - 1; ++i) {
			for (int j = i + 1; j < centroids.size(); ++j) {
				// 质心距离
				float distance = cv::norm(centroids[i] - centroids[j]);
				distVec.push_back(distance);
				// X距离
				float distanceX = std::abs(centroids[i].x - centroids[j].x);
				distX.push_back(distanceX);
				// Y距离
				float distanceY = std::abs(centroids[i].y - centroids[j].y);
				distY.push_back(distanceY);
				// X方向距离中心的距离
				float distXCent0 = std::abs(centroids[i].x - cent.x);
				float distXCent1 = std::abs(centroids[j].x - cent.x);
				float distXCent = (distXCent0 + distXCent1) / 2;
				distXWithCent.push_back(distXCent);
				// Y方向距离中心的距离
				float distYCent0 = std::abs(centroids[i].y - cent.y);
				float distYCent1 = std::abs(centroids[j].y - cent.y);
				float distYCent = (distYCent0 + distYCent1) / 2;
				distYWithCent.push_back(distYCent);

				cv::Rect blobBox_i = cv::boundingRect(partBlob[i]);
				cv::Rect blobBox_j = cv::boundingRect(partBlob[j]);
				// 行差
				float row1 = std::abs(blobBox_i.y - blobBox_j.y);
				float row2 = std::abs(blobBox_i.y + blobBox_i.height - (blobBox_j.y + blobBox_j.height));
				distRow1.push_back(row1);
				distRow2.push_back(row2);
				// 列差
				float col1 = std::abs(blobBox_i.x - blobBox_j.x);
				float col2 = std::abs(blobBox_i.x + blobBox_i.width - (blobBox_j.x + blobBox_j.width));
				distCol1.push_back(col1);
				distCol2.push_back(col2);
			}
		}
	}

	return blobFlag;
}