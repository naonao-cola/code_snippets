//
// Created by dyno on 2025/3/10.
//

#include <iostream>
#include <cstring>
#include <valarray>
#include "EvennessFitting.h"

namespace dyno {
	namespace dih {
		namespace alg {
			EvennessFitting::evennessFittingPtr EvennessFitting::mInstancePtr = nullptr;
			
			EvennessFitting::EvennessFitting() {
				paramA = 0;
				paramB = 0;
				paramC = 0;
				paramD = 0;
				pointX.clear();
				pointY.clear();
				pointZ.clear();
				fittingFlag = false;
			}
			
			EvennessFitting::~EvennessFitting() = default;
			
			EvennessFitting::evennessFittingPtr EvennessFitting::getInstance() {
				if (mInstancePtr == nullptr) {
					mInstancePtr = std::shared_ptr<EvennessFitting>(new EvennessFitting());
				}
				return mInstancePtr;
			}
			
			void EvennessFitting::open() {
				paramA = 0;
				paramB = 0;
				paramC = 0;
				paramD = 0;
				pointX.clear();
				pointY.clear();
				pointZ.clear();
				fittingFlag = false;
			}
			
			bool EvennessFitting::pushPoint(const std::vector<int> &pX, const std::vector<int> &pY,
			                                const std::vector<int> &pZ) {
				pointX = pX;
				pointY = pY;
				pointZ = pZ;
				
				return true;
			}
			
			bool EvennessFitting::runEF() {
				if ((pointY.size() < 3) || (pointX.size() < 3) || (pointZ.size() < 3)) {
					std::cout << "至少需要三个点来拟合平面。" << std::endl;
					return false;
				}
				std::vector<std::array<int, 3>> points;
				for (int i = 0; i < pointX.size(); i++) {
					std::array<int, 3> temp = {pointX[i], pointY[i], pointZ[i]};
					points.push_back(temp);
					
				}
				int n = (int) points.size();
				// 计算质心
				std::array<int, 3> centroid = {0, 0, 0};
				for (const auto &p: points) {
					centroid[0] += p[0];
					centroid[1] += p[1];
					centroid[2] += p[2];
				}
				centroid[0] /= n;
				centroid[1] /= n;
				centroid[2] /= n;
				
				// 计算协方差矩阵
				double cov[3][3] = {0};
				for (const auto &p: points) {
					double dx = p[0] - centroid[0];
					double dy = p[1] - centroid[1];
					double dz = p[2] - centroid[2];
					
					cov[0][0] += dx * dx;
					cov[0][1] += dx * dy;
					cov[0][2] += dx * dz;
					cov[1][1] += dy * dy;
					cov[1][2] += dy * dz;
					cov[2][2] += dz * dz;
				}
				cov[1][0] = cov[0][1];
				cov[2][0] = cov[0][2];
				cov[2][1] = cov[1][2];
				
				// 雅可比算法参数
				double V[3][3] = {{1, 0, 0},
				                  {0, 1, 0},
				                  {0, 0, 1}};
				double A[3][3];
				memcpy(A, cov, sizeof(A));
				
				const double eps = 1e-10;
				const int max_iter = 100;
				
				// 雅可比迭代
				for (int iter = 0; iter < max_iter; ++iter) {
					// 寻找最大非对角元素
					double max_val = 0.0;
					int p = 0, q = 1;
					for (int i = 0; i < 3; ++i) {
						for (int j = i + 1; j < 3; ++j) {
							if (std::abs(A[i][j]) > max_val) {
								max_val = std::abs(A[i][j]);
								p = i;
								q = j;
							}
						}
					}
					
					if (max_val < eps) break;
					
					// 计算旋转角度
					double app = A[p][p];
					double aqq = A[q][q];
					double apq = A[p][q];
					double theta = 0.5 * atan2(2 * apq, aqq - app);
					double c = cos(theta);
					double s = sin(theta);
					
					// 构造旋转矩阵
					double R[3][3] = {{1, 0, 0},
					                  {0, 1, 0},
					                  {0, 0, 1}};
					R[p][p] = c;
					R[p][q] = -s;
					R[q][p] = s;
					R[q][q] = c;
					
					// 更新矩阵A = R^T * A * R
					double RtA[3][3];
					memset(RtA, 0, sizeof(RtA));
					for (int i = 0; i < 3; ++i) {
						for (int j = 0; j < 3; ++j) {
							for (int k = 0; k < 3; ++k) {
								RtA[i][j] += R[k][i] * A[k][j];
							}
						}
					}
					
					double newA[3][3];
					memset(newA, 0, sizeof(newA));
					for (int i = 0; i < 3; ++i) {
						for (int j = 0; j < 3; ++j) {
							for (int k = 0; k < 3; ++k) {
								newA[i][j] += RtA[i][k] * R[k][j];
							}
						}
					}
					memcpy(A, newA, sizeof(A));
					
					// 更新特征向量矩阵
					double newV[3][3];
					memset(newV, 0, sizeof(newV));
					for (int i = 0; i < 3; ++i) {
						for (int j = 0; j < 3; ++j) {
							for (int k = 0; k < 3; ++k) {
								newV[i][j] += V[i][k] * R[k][j];
							}
						}
					}
					memcpy(V, newV, sizeof(V));
				}
				
				// 寻找最小特征值
				int min_idx = 0;
				if (A[1][1] < A[min_idx][min_idx]) min_idx = 1;
				if (A[2][2] < A[min_idx][min_idx]) min_idx = 2;
				
				// 获取法向量
				paramA = V[0][min_idx];
				paramB = V[1][min_idx];
				paramC = V[2][min_idx];
				
				// 计算平面常数项
				paramD = -(paramA * centroid[0] + paramB * centroid[1] + paramC * centroid[2]);
				fittingFlag = true;
				std::cout << "a = " << paramA << std::endl;
				std::cout << "b = " << paramB << std::endl;
				std::cout << "c = " << paramC << std::endl;
				std::cout << "d = " << paramD << std::endl;
				return true;
			}
			
			int EvennessFitting::getPointZ(const int &x, const int &y) {
				if(fittingFlag ){
					return (int)(-paramD-paramA*x-paramB*y);
				}
				return 0;
			}
			
			void EvennessFitting::close() {
				paramA = 0;
				paramB = 0;
				paramC = 0;
				paramD = 0;
				pointX.clear();
				pointY.clear();
				pointZ.clear();
				fittingFlag = false;
			}
			
			
		}
	}
}