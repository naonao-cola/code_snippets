extern "C" {
#include "hog.h" // From the VLFeat C library
}
#include <omp.h>
#include "img_feature.h"

namespace nao {
	namespace img {
		namespace feature {
			cv::Mat HogTransform::operator()(/*int training_index = 0 */) {
				cv::Mat gray_img;
				std::vector<std::vector<cv::Mat>> local_hog_descriptors(omp_get_max_threads());
#pragma omp parallel for private(gray_img)
				for (int training_index = 0; training_index < _images.size(); training_index++)
				{
					if (_images[training_index].channels() == 3) {
						cv::cvtColor(_images[training_index], gray_img, cv::COLOR_RGB2GRAY);
					}
					else {
						gray_img = _images[training_index];
					}
					cv::resize(gray_img, gray_img, _train_size);
					cv::Mat roi_img;
					gray_img.convertTo(roi_img, CV_32FC1);
					VlHog* hog = vl_hog_new(VlHogVariant::VlHogVariantUoctti, _num_bins, VL_FALSE);
					vl_hog_set_use_bilinear_orientation_assignments(hog, VL_TRUE);

					vl_hog_put_image(hog, (float*)roi_img.data, roi_img.cols, roi_img.rows, 1, _cell_size);
					int ww = static_cast<int>(vl_hog_get_width(hog)); // assert ww == hh == numCells
					int hh = static_cast<int>(vl_hog_get_height(hog));
					int dd = static_cast<int>(vl_hog_get_dimension(hog)); // assert ww=hogDim1, hh=hogDim2, dd=hogDim3
					cv::Mat hogArray(1, ww * hh * dd, CV_32FC1); // safer & same result. Don't use C-style memory management.
					vl_hog_extract(hog, hogArray.ptr<float>(0));
					vl_hog_delete(hog);
					int thread_id = omp_get_thread_num(); // 获取当前线程的 ID
					local_hog_descriptors[thread_id].push_back(hogArray);
					//_hog_descriptors.push_back(hogArray);
				}
				for (int i = 0; i < local_hog_descriptors.size(); i++) {
					for (int j = 0; j < local_hog_descriptors[i].size(); j++) {
						_hog_descriptors.push_back(local_hog_descriptors[i][j]);
					}

				}
				return _hog_descriptors.clone();
			};
			namespace xx {

				cv::Mat getELBPfeature(const cv::Mat& img, const int& radius = 3, const int& neighbors = 8) {
					cv::Mat gray_img;
					if (img.channels() == 3)
						cv::cvtColor(img, gray_img, cv::COLOR_RGB2GRAY);
					else
						gray_img = img;
					int width = gray_img.cols;
					int height = gray_img.rows;
					cv::Mat RLBPImg = cv::Mat::zeros(height - 2 * radius, width - 2 * radius, CV_8UC1);
					for (int n = 0; n < neighbors; n++) {

						float rx = static_cast<float>(radius * cos(2.0 * CV_PI * n / static_cast<float>(neighbors)));
						float ry = -static_cast<float>(radius * sin(2.0 * CV_PI * n / static_cast<float>(neighbors)));
						int x1 = static_cast<int>(floor(rx));
						int x2 = static_cast<int>(ceil(rx));
						int y1 = static_cast<int>(floor(ry));
						int y2 = static_cast<int>(ceil(ry));
						float tx = rx - x1;
						float ty = ry - y1;
						float w1 = (1 - tx) * (1 - ty);
						float w2 = tx * (1 - ty);
						float w3 = (1 - tx) * ty;
						float w4 = tx * ty;
						for (int i = radius; i < height - radius; i++) {
							for (int j = radius; j < width - radius; j++) {
								uchar centerPix = gray_img.at<uchar>(i, j);
								float neighbor = gray_img.at<uchar>(i + x1, j + y1) * w1 + gray_img.at<uchar>(i + x1, j + y2) * w2 \
									+ gray_img.at<uchar>(i + x2, j + y1) * w3 + gray_img.at<uchar>(i + x2, j + y2) * w4;
								RLBPImg.at<uchar>(i - radius, j - radius) |= (neighbor > centerPix) << (neighbors - n - 1);
							}
						}
					}
					for (int i = 0; i < RLBPImg.rows; i++) {
						for (int j = 0; j < RLBPImg.cols; j++) {
							unsigned char currentValue = RLBPImg.at<uchar>(i, j);
							unsigned char minValue = currentValue;
							for (int k = 1; k < neighbors; k++) {
								unsigned char temp = (currentValue >> (neighbors - k)) | (currentValue << k);
								if (temp < minValue) {
									minValue = temp;
								}
							}
							RLBPImg.at<uchar>(i, j) = minValue;
						}
					}
#ifdef NDEBUG
					((void)0);
#else
					cv::imshow("RLBPImg", RLBPImg);
					cv::waitKey(0);
#endif // _DEBUG
					return RLBPImg;
				}

				cv::Mat getRegionLBPH(const cv::Mat& src, const int& minValue, const int& maxValue, const bool& normed = true) {
					cv::Mat result;
					int histSize = maxValue - minValue + 1;
					float range[] = { static_cast<float>(minValue),static_cast<float>(maxValue + 1) };
					const float* ranges = { range };
					cv::calcHist(&src, 1, 0, cv::Mat(), result, 1, &histSize, &ranges, true, false);
					if (normed) {
						result /= (int)src.total();
					}
					return result.reshape(1, 1);
				}


				cv::Mat getLBPH(cv::Mat src, const int& numPatterns = 256, const int& grid_x = 8, const int& grid_y = 8, const bool& normed = true) {
					int width = src.cols / grid_x;
					int height = src.rows / grid_y;
					cv::Mat result = cv::Mat::zeros(grid_x * grid_y, numPatterns, CV_32FC1);
					if (src.empty()) {
						return result.reshape(1, 1);
					}
					int resultRowIndex = 0;
					for (int i = 0; i < grid_x; i++) {
						for (int j = 0; j < grid_y; j++) {
							cv::Mat src_cell = cv::Mat(src, cv::Range(i * height, (i + 1) * height), cv::Range(j * width, (j + 1) * width));
							cv::Mat hist_cell = getRegionLBPH(src_cell, 0, (numPatterns - 1), true);
							cv::Mat rowResult = result.row(resultRowIndex);
							hist_cell.reshape(1, 1).convertTo(rowResult, CV_32FC1);
							resultRowIndex++;
						}
					}
					return result.reshape(1, 1);
				}
			}//namespace xx

			cv::Mat LBPTransform::operator()() {
				cv::Mat gray_img;
				if (_train_size.width < 20 || _train_size.height < 20) {
					throw std::runtime_error("设置的图像宽度或者高度太小");
					exit(0);
				}
				for (int training_index = 0; training_index < _images.size(); training_index++)
				{
					if (_images[training_index].channels() == 3) {
						cv::cvtColor(_images[training_index], gray_img, cv::COLOR_RGB2GRAY);
					}
					else {
						gray_img = _images[training_index];
					}
					cv::resize(gray_img, gray_img, _train_size);
					cv::Mat LBPImg = xx::getELBPfeature(gray_img);
					cv::Mat lbpDescriptor = xx::getLBPH(LBPImg);
					_lbp_descriptors.push_back(lbpDescriptor);
				}
				return _lbp_descriptors.clone();
			}
		}//namespace feature
	}//namespace img
}//namespace nao