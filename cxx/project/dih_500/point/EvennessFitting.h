//
// Created by dyno on 2025/3/10.
//

#ifndef DIH500REFACTOR_EVENNESSFITTING_H
#define DIH500REFACTOR_EVENNESSFITTING_H

#include <memory>
#include <vector>

namespace dyno {
	namespace dih {
		namespace alg {
			
			class EvennessFitting {
			private:
				std::vector<int> pointX;
				std::vector<int> pointY;
				std::vector<int> pointZ;
				
				double paramD;
				double paramA;
				double paramB;
				double paramC;
				bool fittingFlag;
				EvennessFitting();
			
			
			public:
				typedef std::shared_ptr<EvennessFitting> evennessFittingPtr;
				
				~EvennessFitting();
			
			private:
				static evennessFittingPtr mInstancePtr; //   全局调用句柄
			public:
				static evennessFittingPtr getInstance(); // 获取句柄
				
				void open();//　　打开拟合
				
				bool pushPoint(const std::vector<int> &pX, const std::vector<int> &pY, const std::vector<int> &pZ); //　上传采样点数据
				
				bool runEF(); // 进行平面拟合并保存拟合结果
				
				int getPointZ(const int &x, const int &y);// 通过ＸＹ反算Ｚ轴高度
				
				void close();  //关闭拟合清除缓冲区
				
			};
		}
	}
}


#endif //DIH500REFACTOR_EVENNESSFITTING_H
