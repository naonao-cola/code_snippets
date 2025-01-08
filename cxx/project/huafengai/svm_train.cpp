#include "svm_train.h"
namespace nao {
/**
 * @FilePath     : /connector_algo/src/custom/svm_train.cc
 * @Description  :
 * @Author       : naonao
 * @Date         : 2024-07-19 15:37:50
 * @Version      : 0.0.1
 * @LastEditors  : naonao
 * @LastEditTime : 2024-07-19 15:50:05
**/
	namespace svm {
		void setDefaultParam() {
			_default_param.svm_type = C_SVC;
			_default_param.kernel_type = RBF;
			_default_param.degree = 10;
			_default_param.gamma = 0.03;
			_default_param.coef0 = 1.0;
			_default_param.nu = 0.005;
			_default_param.cache_size = 1000;
			_default_param.C = 32.0;
			_default_param.eps = 1e-4;
			_default_param.p = 1.0;
			_default_param.nr_weight = 0;
			_default_param.shrinking = 1;
			_default_param.probability = 1;
		}

		SvmTrain::SvmTrain(cv::Size train_size) {
			_svm = nullptr;
			_train_size = train_size;
			setDefaultParam();
		}

		SvmTrain::~SvmTrain(void) {
			if (_svm) {
				svm_free_model_content(_svm);
			}
		}
		SvmTrain::SvmTrain() {
			_svm = nullptr;
			setDefaultParam();
		}
		bool SvmTrain::init(std::string svm_model_path, std::string model_name) {
			_base_path = svm_model_path;
			_model_name = model_name;
			
			_svm = svm_load_model(std::string(_base_path + "\\" + _model_name).c_str());
			if (_svm == nullptr) return false;
			return true;
		}

		void SvmTrain::getFileList(const std::string& filePath, const int& lable, const std::string& format/*= ".jpg"*/) {
			std::vector<std::string> file_list;
			nao::fl::getAllFormatFiles(filePath, file_list, format);
			for (size_t i = 0; i < file_list.size(); i++) {
				_train_data.emplace_back(std::make_pair(lable, file_list[i]));
			}
		}

		void SvmTrain::readTestFileList(const std::string& filePath, const std::string& format /*= ".jpg"*/) {
			std::vector<std::string> file_list;
			nao::fl::getAllFormatFiles(filePath, file_list, format);
			for (size_t i = 0; i < file_list.size(); i++) {
				_test_img_list.emplace_back(file_list[i]);
			}
		}

		void SvmTrain::processHogFeature(const cv::HOGDescriptor& hog) {
			size_t train_img_size = _train_data.size();
			if (train_img_size <= 1) {
				throw std::logic_error("error");
				exit(0);
			}

			_label_mat = cv::Mat(static_cast<int>(train_img_size), 1, CV_32FC1, cv::Scalar(0));
			_data_mat = cv::Mat(static_cast<int>(train_img_size), static_cast<int>(hog.getDescriptorSize()), CV_32FC1, cv::Scalar(0));

			for (int i = 0; i < train_img_size; i++) {
				cv::Mat src = cv::imread(_train_data[i].second, 0);
				if (src.empty())continue;
				cv::Mat train_img;
				cv::resize(src, train_img, _train_size);
				std::vector<float> descriptors;
				descriptors.resize(hog.getDescriptorSize());
				try {
					hog.compute(train_img, descriptors, cv::Size(1, 1), cv::Size(0, 0));
				}
				catch (const cv::Exception& e) {
					std::cerr << "An error occurred when compute hog:" << e.what() << std::endl;
				}
				for (int j = 0; j < descriptors.size(); j++) {
					float* ptr = _data_mat.ptr<float>(i);
					ptr[j] = descriptors[j];
				}
				_label_mat.ptr<float>(i)[0] = static_cast<float>(_train_data[i].first);
			}
		}

		void SvmTrain::trainLibSVM(svm_parameter& param /*= _default_param*/, const int& used /*= 0*/) {
			if (used == 0) param = _default_param;
			svm_problem svm_prob;
			int sample_size = _data_mat.rows;
			int feature_length = _data_mat.cols;
			svm_prob.l = sample_size;
			svm_prob.y = new double[sample_size];

			for (int i = 0; i < sample_size; i++) {
				float* ptr = _label_mat.ptr<float>(i);
				svm_prob.y[i] = ptr[0];
			}
			svm_prob.x = new  svm_node * [sample_size];
			for (int i = 0; i < sample_size; i++) {
				svm_node* x_sapce = new svm_node[feature_length];
				float* ptr = _data_mat.ptr<float>(i);
				for (int j = 0; j < feature_length; j++) {
					x_sapce[j].index = j;
					x_sapce[j].value = ptr[j];
				}

				x_sapce[feature_length].index = -1;
				svm_prob.x[i] = x_sapce;
			}
			svm_model* svm_model = svm_train(&svm_prob, &param);
			std::string path = _base_path + "\\" + _model_name;
			svm_save_model(path.c_str(), svm_model);
			for (int i = 0; i < svm_prob.l; i++) {
				delete[] svm_prob.x[i];
			}
			delete[] svm_prob.x;
			delete[] svm_prob.y;
			svm_free_model_content(svm_model);
		}

		double SvmTrain::testLibSVM(const cv::Mat& src, const cv::HOGDescriptor& hog, double prob_estimates[]) {
			if (src.empty())return -1;
			cv::Mat test_img;
			cv::resize(src, test_img, _train_size);
#ifdef NDEBUG
			((void)0);
#else
			cv::imshow("input img", test_img);
			cv::waitKey(0);
#endif // _DEBUG
			std::vector<float> descriptors;
			descriptors.resize(hog.getDescriptorSize());
			hog.compute(test_img, descriptors, cv::Size(1, 1), cv::Size(0, 0));
			svm_node* inputVector = new svm_node[descriptors.size() + 1];
			for (int i = 0; i < descriptors.size(); i++) {
				inputVector[i].index = i;
				inputVector[i].value = descriptors[i];
			}
			inputVector[descriptors.size()].index = -1;
			double resultLabel = svm_predict_probability(_svm, inputVector, prob_estimates);//
			delete[] inputVector;
			return resultLabel;
		}

		void SvmTrain::test(const cv::HOGDescriptor& hog) {
			int errorCount = 0;
			std::string path = _base_path + "\\" + _model_name;
			_svm = svm_load_model(path.c_str());
			for (int i = 0; i < _train_data.size(); i++) {
				cv::Mat src = cv::imread(_train_data[i].second, 0);
				if (src.empty()) continue;
				double prob[2] = { 0 };
				int ret = (int)testLibSVM(src, hog, prob);
				if (ret != _train_data[i].first) {
					errorCount++;
					if (_train_data[i].first == 0) {
						cv::imwrite(_base_path + "\\error\\0\\" + std::to_string(i) + ".jpg", src);
					}
					else if (_train_data[i].first == 1) {
						cv::imwrite(_base_path + "\\error\\1\\" + std::to_string(i) + ".jpg", src);
					}
				}
			}
			double errorPercentage = 0.0;
			errorPercentage = errorCount / (_train_data.size() * 1.0);
			std::cout << "error: " << errorPercentage << std::endl;
			std::cout << "sucess: " << 1 - errorPercentage << std::endl;
		}

		/*-----------------------------------------------------------------------------------------------------------*/
		void SvmTrain::addFeatureLabel(const cv::Mat& feature, const std::vector<int>& label) {
			if (feature.rows != label.size()) {
				throw std::runtime_error("");
				exit(0);
			}
			int sample_length = feature.rows;
			for (int i = 0; i < sample_length; i++) {
				cv::Mat single_sample = feature.rowRange(i, i + 1).clone();
				if (single_sample.empty()) {
					continue;
				}
				_train_feature_data.emplace_back(std::make_pair(label[i], single_sample));
			}
		}

		void SvmTrain::copyFeatureLabel() {
			std::cout << "" << std::endl;
			int sample_length = static_cast<int>(_train_feature_data.size());
			if (sample_length <= 1) {
				throw std::logic_error("");
				exit(0);
			}
			//_data_mat = cv::Mat(sample_length, _train_feature_data[0].second.cols, CV_32FC1, cv::Scalar(0));
			_label_mat = cv::Mat(static_cast<int>(sample_length), 1, CV_32FC1, cv::Scalar(0));
			for (int i = 0; i < sample_length; i++) {
				_label_mat.ptr<float>(i)[0] = static_cast<float>(_train_feature_data[i].first);
				_data_mat.push_back(_train_feature_data[i].second);
			}
			std::cout << "" << std::endl;
		}

		double SvmTrain::testFeatureLibSVM(const cv::Mat& feature, double prob_estimates[]) {
			if (feature.empty()) {
				throw std::runtime_error("");
				exit(0);
			}
			int feature_length = feature.cols;
			svm_node* inputVector = new svm_node[feature_length + 1];
			for (int i = 0; i < feature_length; i++) {
				inputVector[i].index = i;
				inputVector[i].value = feature.ptr<float>(0)[i];
			}
			inputVector[feature_length].index = -1;
			double resultLabel = svm_predict_probability(_svm, inputVector, prob_estimates);//
			delete[] inputVector;
			return resultLabel;
		}

		void SvmTrain::test() {
			std::cout << "test process" << std::endl;
			int errorCount = 0;
			std::string path = _base_path + "\\" + _model_name;
			_svm = svm_load_model(path.c_str());
			std::vector<int> ret_vec;
			for (int i = 0; i < _train_feature_data.size(); i++) {
				if (_train_feature_data[i].second.empty()) continue;
				double prob[2] = { 0 };
				cv::Mat imput = _train_feature_data[i].second.clone();
				int ret = (int)testFeatureLibSVM(imput, prob);
				ret_vec.emplace_back(ret);
				if (ret != _train_feature_data[i].first) {
					errorCount++;
				}
			}
			double errorPercentage = 0.0;
			errorPercentage = errorCount / (_train_feature_data.size() * 1.0);
			std::cout << "error " << errorPercentage << std::endl;
			std::cout << "sucess " << 1 - errorPercentage << std::endl;
		}
	}//namespace svm
}//namespace nao