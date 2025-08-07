
//
// Created by y on 24-4-3.
//
#include <numeric>
#include <fstream>
#include <ctime>
#include "ParamFitting.h"
//#include "DihLog.h"
#include "algLog.h"
#include "utils.h"
#include "replace_std_string.h"

#define ADD_DENOMINATOR       1e-5   //为避免除0，对分母进行增加
namespace ALG_DEPLOY {

	int SphericalReagentFitting::Init() {
		int ret = 0;
		ret = (ret || mcv_network.Init());
		ret = (ret || mpv_network.Init());
		ret = (ret || rdw_cv_network.Init());
		ret = (ret || rdw_sd_network.Init());
        ret = (ret || pdw_cv_network.Init());
        ret = (ret || hgbNetwork.Init());
		ret = (ret || cell_count_network.Init());
		if (ret) {
			ALGLogError << "Failed to init SphericalReagentFitting";
			return -1;
		}
	}


	using std::chrono::duration_cast;
	using std::chrono::milliseconds;
	using std::chrono::seconds;
	using std::chrono::system_clock;

	int TempMcvWriteResult(const std::vector<float> &germ_nums_per_img, const int &germ_rga_channel_img_nums,
	                       const int &germ_opencv_channel_img_nums) {
		std::string germ_save_dir{"./volume/"};
		if (germ_rga_channel_img_nums == 0 && germ_opencv_channel_img_nums == 0) {
            ALGLogError << "Germ csv file will not be created cause zero germ imgs accepted.";
            return 0;
		}

		auto time_now = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
		std::string csv_save_dir(germ_save_dir);
		std::string csv_save_path = csv_save_dir + std::to_string(time_now) + std::string{"_rbc_volume.csv"};
		try {
			std::ofstream out_file(csv_save_path, std::ios::out);
			for (const auto &nums: germ_nums_per_img) {
				out_file << nums << std::endl;
			}
			return 0;
		}
		catch (std::exception &e) {
			ALGLogError << "failed to write csv data in " << csv_save_path;
			return -1;
		}
		return 0;

	}

	int SphericalReagentFitting::Forward(const std::vector<float> &area_rbc_v, const float &incline_rbc_nums,
	                                     const float &incline_rbc_region, const std::vector<float> &area_plt_v,
	                                     const float &relative_line_ratio_to_standard,
	                                     const std::vector<float> &data_v, const std::vector<float> &coef_v,
	                                     float &mcv, float &rdw_cv, float &rdw_sd, float &mpv,
	                                     std::vector<float> &curve_rbc, std::vector<float> &curve_plt,
	                                     float &hgb, float &rbc, float &ret_, float &plt,
	                                     float &neu, float &lym, float &mono, float &eos, float &baso) {
		std::vector<float> rbc_volume_res;
		int ret = 0;
		// mcv
		ret = this->SphericalMcvFitting(area_rbc_v, rbc_volume_res, mcv);
		if (ret) {
			ALGLogError << "Failed to get SphericalMcvFitting";
			return -1;
		}
		float mean_area_rbc = PseudoMean(area_rbc_v.begin(), area_rbc_v.end());
		ALGLogInfo << "mean rbc area  " << mean_area_rbc;

		// mcv for get rk result
/*  ret = TempMcvWriteResult(rbc_volume_res, 1, 1);
  if(ret){
    ALGLogError<<"Failed to write mcv result";
    return -1;
  }*/

        std::cout << "开始拟合" << std::endl;
        // rbc line
		ret = this->SphericalGetRbcVolumeLine(rbc_volume_res, curve_rbc);
		if (ret) {
			ALGLogError << "Failed to get spherical reagent rbc volume line";
			return -2;
		}

		// mpv
		std::vector<float> plt_volume_res;
		ret = this->SphericalMpvFitting(area_plt_v, plt_volume_res, mpv);
		if (ret) {
			ALGLogError << "Failed to get spherical reagent rbc mpv";
			return -3;
		}
        std::cout << "mpv: "<< mpv<<"\n";
        // plt line
        ret = SphericalGetPltVolumeLine(plt_volume_res, curve_plt);
        if (ret) {
			ALGLogError << "Failed to get spherical reagent plt volume line";
            std::cout << "Failed to get spherical reagent plt volume line";
            return -4;
		}

		// rdw_cv
		ret = this->SphericalRdwCvFitting(rbc_volume_res, mcv, rdw_cv);
		if (ret) {
			ALGLogError << "Failed to get spherical reagent rdw_cv";
            std::cout << "Failed to get spherical reagent rdw_cv";
            return -5;
		}
        std::cout << "rdw_cv: " << rdw_cv << "\n";
        // rdw_sd
		ret = this->SphericalRdwSdFitting(rbc_volume_res, mcv, rdw_sd);
		if (ret) {
			ALGLogError << "Failed to get spherical reagent rdw_sd";
            std::cout << "Failed to get spherical reagent rdw_sd";
            return -6;
		}
        std::cout << "rdw_sd: " << rdw_sd << "\n";
        // hgb
		ret = HgbFitting(data_v, coef_v, hgb);
		if (ret) {
			ALGLogError << "Failed to get spherical reagent hgb";
            std::cout << "Failed to get spherical reagent hgb \n";
            return -7;
		}

		// count
		ret = AllCellCountFitting(rbc, ret_, plt,
		                          neu, lym, mono, eos, baso);
		if (ret) {
			ALGLogError << "Failed to get spherical reagent count";
            std::cout << "Failed to get spherical reagent count";
            return -8;
		}
        std::cout << "rbc: " << rbc << "\n";

        return 0;

	}

/////////////////////////
////mcv
/////////////////////////
	int SphericalReagentFitting::SphericalMcvFitting(const std::vector<float> &area_rbc_v,
	                                                 std::vector<float> &rbc_volume_v, float &mcv) {
		if (area_rbc_v.empty()) {
			mcv = 0;
			return 0;
		}
		//for get params from rk35xx
//    std::cout<<"rbc ori value"<<std::endl;
//    for(auto i :area_rbc_v){
//      std::cout<<" "<<i<<",";
//    }
		std::cout << std::endl;
		std::vector<float> network_input(area_rbc_v);
		std::vector<float> network_output;
		int ret = mcv_network.Forward(network_input, network_output);
		if (ret) {
			return -1;
		}
//  std::cout<<"rbc volume value"<<std::endl;
//  for(auto i :network_output){
//    std::cout<<" "<<i<<",";
//  }
		double sum = std::accumulate(std::begin(network_output), std::end(network_output), 0.f);


		double mean = sum / ((int) network_output.size() + ADD_DENOMINATOR); //均值
		mcv = (float) mean;
		//避免白细胞流道测试数据过少导致结果为负数
		if (mcv < 0) {
			mcv = 0;
		}
		rbc_volume_v = network_output;
		return 0;

	}
    int FindRdwCvL1L2(const std::vector<float>& vol_v, const float& mcv, const float& target_area_ratio, float& l1, float& l2);
    /////////////////////////
    ////mpv
    /////////////////////////
	int
	SphericalReagentFitting::SphericalMpvFitting(const std::vector<float> &area_plt_v, std::vector<float> &plt_volume_v,
	                                             float &mpv) {
		if (area_plt_v.empty()) {
			mpv = 0;
			return 0;
		}
		// std::cout << "----------plt ori value--------------" << std::endl;
		// for (auto i: area_plt_v) {
		// 	std::cout << " " << i << ",";
		// }
		std::cout << std::endl;
		float area = PseudoMean(area_plt_v.begin(), area_plt_v.end());
		ALGLogInfo << "Mean plt area " << area;

        std::cout << "area_plt_v size  " << area_plt_v.size() << "\n";
        //计算单个plt体积
		std::vector<float> network_input(area_plt_v);
		std::vector<float> network_output;
		int ret = mpv_network.Forward(network_input, network_output);
		if (ret) {
			return -1;
		}
		plt_volume_v = network_output;
		//计算平均体积
		float sum = std::accumulate(plt_volume_v.begin(), plt_volume_v.end(), 0.0);
		float mean = sum / plt_volume_v.size();
		mpv = mean;

        // std::time_t   timestamp       = std::time(nullptr);
        // std::string   csv_origin_path = "/mnt/user/0/16B49C13B49BF409/pdw_csv/070210.csv";
        // std::cout << "csv_origin_path: " << csv_origin_path << std::endl;
        // std::ofstream csv_origin_file(csv_origin_path);
        // if (csv_origin_file.is_open()) {
        //     for (size_t i = 0; i < plt_volume_v.size(); ++i) {
        //         csv_origin_file << plt_volume_v[i];
        //         csv_origin_file << "\n";
        //     }
        //     csv_origin_file.close();
        // }
        // else {
        //     ALGLogInfo << "Failed to open file: " << csv_origin_path;
        // }

        // std::cout << "mpv value " << mpv << "\n";
        // //pdw 增加
        this->pdw =0.0;
        if (mean > 2.0 && plt_volume_v.size()>5) {
			try {
				ALGLogInfo << "mpv 的值: " << mpv<< " 进入到 pdw 检测流程";
				std::cout << " 243 mpv 的值: " << mpv << " 进入到 pdw 检测流程 \n";
				std::cout << " 244 plt_volume_v 的size: " << plt_volume_v.size() << " \n";
				std::vector<float> smoth_plt_volume_v;
				float              l1, l2;
				std::vector<float> linear_interpolated_data;
				for (int i = 0; i < plt_volume_v.size(); i++) {
					linear_interpolated_data.push_back(plt_volume_v[i] * 10);
				}
				smoth_plt_volume_v.clear();
				ret = MediumMeanVolumeLine(linear_interpolated_data,
										linear_interpolated_data.size() * 1.5,
										1,
										1,
										SPHERICAL_VOLUME_MAX_AREA,
										SPHERICAL_VOLUME_RDW_CV_KERNEL_MEDIUM,
										SPHERICAL_VOLUME_RDW_CV_KERNEL_BLUR,
										smoth_plt_volume_v);
				std::cout << "开始寻找l1 l2 line: " << "266"<< "\n";

                ret = FindRdwCvL1L2(smoth_plt_volume_v, mean * 10, 0.6826, l1, l2);
                // 不需要乘10,因为分子 分母同时除以10
				this->pdw = ((l2 - l1) / (l2 + l1 + ADD_DENOMINATOR)) * 100;
				std::cout << "pdw third type L2: " << l2 << " l1: " << l1 << "\n";
				std::vector<float> pdw_network_input{this->pdw};
				std::vector<float> pdw_network_output;
				ret = pdw_cv_network.Forward(pdw_network_input, pdw_network_output);
				if (ret) {
					ALGLogError << "Failed to forward SphericalRdwCvNetwork";
					return -3;
				}
				this->pdw = pdw_network_output[0];
				std::cout << "pdw third type " << this->pdw << "\n";

			}
			catch (cv::Exception &e) {
				ALGLogError << "Failed to calculate pdw, exception: " << e.what();
				std::cout << "Failed to calculate pdw, exception: " << e.what();
				this->pdw = 0.0;
				return 0;
			}
			catch (std::exception &e) {
				ALGLogError << "Failed to calculate pdw, exception: " << e.what();
                this->pdw=0.0;
				std::cout << "Failed to calculate pdw, exception: " << e.what();
                return 0;

			}
        }
        // std::cout << "----------------plt volume value-------------" << std::endl;
		// for (int i = 0; i < plt_volume_v.size(); ++i) {
		// 	std::cout << ", " << plt_volume_v[i];
		// }
		//std::cout << std::endl;
		return 0;
	}


	int FindRdwCvL1L2(const std::vector<float> &vol_v, const float &mcv,
	                  const float &target_area_ratio, float &l1, float &l2) {
		int split_idx = (int) mcv;
		if (split_idx > vol_v.size()) {
			ALGLogError << "FindRdwCvL1L2 ERROR, split idx must > size of histogram";
			return -1;
		}

		float target_area = std::accumulate(vol_v.begin(), vol_v.end(), 0.f) * target_area_ratio;
		for (int i = 1; i < split_idx; ++i) {
			float left_data = std::accumulate(vol_v.begin() + split_idx - i, vol_v.begin() + split_idx, 0.f);
			float right_data = std::accumulate(vol_v.begin() + split_idx + 1, vol_v.begin() + split_idx + 1 + i, 0.f);
			float current_data = left_data + right_data + vol_v[split_idx];
			if (current_data >= target_area) {
				l1 = (float) (split_idx - i);
				l2 = (float) (split_idx + i);
				ALGLogInfo << "L1 L2" << l1 << " " << l2 << std::endl;
				return 0;
			}
		}
		ALGLogError << "FindL1L2 ERROR, Do not find suitable l1 l2";
		return -2;

	}


/////////////////////////
////rdw_cv
/////////////////////////
	int
	SphericalReagentFitting::SphericalRdwCvFitting(const std::vector<float> &vol_v, const float &mcv, float &rdw_cv) {
		if (vol_v.empty()) {
			rdw_cv = 0;
			return 0;
		}

		//平滑
		std::vector<float> vol_smoothness;
		int ret = MediumMeanVolumeLine(vol_v, SPHERICAL_VOLUME_SIZE,
		                               1, 1,
		                               SPHERICAL_VOLUME_MAX_AREA,
		                               SPHERICAL_VOLUME_RDW_CV_KERNEL_MEDIUM,
		                               SPHERICAL_VOLUME_RDW_CV_KERNEL_BLUR,
		                               vol_smoothness);
		if (ret) {
			ALGLogError << "Failed to apply medium and mean smoothness";
			return -1;
		}

		//找对应最高点指定百分比面积处的左右边界点
		float l1, l2;

		ret = FindRdwCvL1L2(vol_smoothness, mcv, SPHERICAL_RDW_CV_PERCENTAGE, l1, l2);
		if (ret) {
			ALGLogError << "Failed to find l1,l2";
			return -2;
		}
		rdw_cv = (l2 - l1) / (l2 + l1 + ADD_DENOMINATOR);
		rdw_cv = rdw_cv * 100;

		ALGLogInfo << "rdw cv raw " << rdw_cv << std::endl;

		//对直方图结果进行拟合
		std::vector<float> network_input{rdw_cv};
		std::vector<float> network_output;
		ret = rdw_cv_network.Forward(network_input, network_output);
		if (ret) {
			ALGLogError << "Failed to forward SphericalRdwCvNetwork";
			return -3;
		}
		rdw_cv = network_output[0];
		return 0;

	}


	void FindMinIdx(const std::vector<float> &vol_v, const float &targe_value, float &idx) {

		if (vol_v.empty()) {
			return;
		}
		float temp_min_value = vol_v[0];
		idx = 0;
		for (int i = 0; i < vol_v.size(); ++i) {
			if (vol_v[i] < temp_min_value) {
				temp_min_value = vol_v[i];
				idx = (float) i;
			}
		}

	}

	int FindRdwSdL1L2(const std::vector<float> &vol_v, const float &mcv,
	                  const float &target_ratio, float &l1, float &l2) {
		int split_idx = PseudoArgMax(vol_v.begin(), vol_v.end());
		if (split_idx > vol_v.size()) {
			ALGLogError << "FindRdwSdL1L2, split idx must > size of histogram";
			return -1;
		}
		float target_value = vol_v[split_idx] * target_ratio;
		int left_idx, right_idx;
		FindMinDifference(std::vector<float>(vol_v.begin(), vol_v.begin() + split_idx), target_value, left_idx);
		FindMinDifference(std::vector<float>(vol_v.begin() + split_idx, vol_v.end()), target_value, right_idx);

		ALGLogInfo << "l1 l2 " << l1 << "---------------  " << l2;
		l1 = (float) left_idx;
		l2 = (float) right_idx + split_idx;

		return 0;
	}

/////////////////////////
////rdw sd
/////////////////////////
	int SphericalReagentFitting::SphericalRdwSdFitting(const std::vector<float> &vol_v,
	                                                   const float &mcv, float &rdw_sd) {
		ALGLogInfo << "sphericl rdw sd----------------------------";
		if (vol_v.empty()) {
			rdw_sd = 0;
			return 0;
		}

		//平滑
		std::vector<float> vol_smoothness;
		int ret = LocalMaxVolumeLine(vol_v, SPHERICAL_VOLUME_SIZE,
		                             1, 1,
		                             SPHERICAL_VOLUME_MAX_AREA,
		                             SPHERICAL_VOLUME_RDW_SD_KERNEL_LOCAL,
		                             SPHERICAL_VOLUME_RDW_SD_KERNEL_BLUR,
		                             vol_smoothness);
		if (ret) {
			ALGLogError << "Failed to apply local max and mean smoothness";
			return -1;
		}

//  std::cout<<"rdw smooth -----------------"<<std::endl;
//  for(const auto& iter:vol_smoothness){
//    std::cout<<iter<<" ";
//  }
//  std::cout<<std::endl;

		//获取原始rdw_sd;
		float l1, l2;
		FindRdwSdL1L2(vol_smoothness, mcv, SPHERICAL_RDW_SD_PERCENTAGE, l1, l2);
		rdw_sd = l2 - l1;
		ALGLogInfo << "raw rdw _sd " << rdw_sd;
		//拟合
		std::vector<float> network_input{rdw_sd};
		std::vector<float> network_output;
		ret = rdw_sd_network.Forward(network_input, network_output);
		if (ret) {
			return -2;
		}
		rdw_sd = network_output[0];
		return 0;
	}


	int
	SphericalReagentFitting::SphericalGetRbcVolumeLine(const std::vector<float> &vol_v, std::vector<float> &result) {
		if (vol_v.empty()) {
			return 0;
		}

		//平滑
		int ret = LocalMaxVolumeLine(vol_v, SPHERICAL_VOLUME_SIZE,
		                             1, 1,
		                             SPHERICAL_VOLUME_MAX_AREA,
		                             SPHERICAL_VOLUME_RBC_LINE_KERNEL_LOCAL,
		                             SPHERICAL_VOLUME_RBC_LINE_KERNEL_BLUR,
		                             result);
		if (ret) {
			ALGLogError << "Failed to apply local max and mean smoothness";
			return -1;
		}

		return 0;
	}

	int
	SphericalReagentFitting::SphericalGetPltVolumeLine(const std::vector<float> &vol_v, std::vector<float> &result) {
		if (vol_v.empty()) {
			return 0;
		}
		int ret = LocalMaxVolumeLine(vol_v, SPHERICAL_VOLUME_SIZE,
		                             SPHERICAL_VOLUME_PLT_DILATE_RATIO,
		                             SPHERICAL_VOLUME_PLT_DOWN_RATIO,
		                             SPHERICAL_VOLUME_MAX_AREA,
		                             SPHERICAL_VOLUME_PLT_SAMPLE_WIDTH, SPHERICAL_VOLUME_PLT_KERNEL_BLUR,
		                             result);
		return ret;
	}
}
