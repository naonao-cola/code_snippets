#include "StdAfx.h"
#include "InspectLabelPol.h"
#include <filesystem>
#include <codecvt>

namespace fs = std::filesystem;


InspectLabelPol::InspectLabelPol()
{
	
}

InspectLabelPol::~InspectLabelPol()
{
}

// src 是已经旋转正的图片，未裁切
long InspectLabelPol::DoFindLabelMark(cv::Mat src, STRU_LabelMarkParams& params, STRU_LabelMarkInfo& labelMarkInfo)
{
	cv::Mat blurImg, binImg;
	cv::Mat matTempBuf;

	int polmarkDilateSize = 9;
	double expand = params.bSaveTemplate ? 10 : 50;
	std::vector<cv::Point> connectContorus;
	cv::Rect tempRect;

	// 模版目录检查
	std::wstring_convert<std::codecvt_utf8_utf16<TCHAR>, TCHAR> converter;
	std::string modelPath = converter.to_bytes(params.modelPath);
	std::string polmarkDir = modelPath + "/PolMark/";
	if (!fs::exists(polmarkDir)) {
		fs::create_directory(polmarkDir);
	}

	cv::GaussianBlur(src, matTempBuf, cv::Size(15, 15), 0);
	cv::Mat cellImg = matTempBuf(params.cellBBox);

	// 1. 搜索Label
	cv::threshold(matTempBuf, binImg, params.labelThreshold, 255, cv::THRESH_BINARY_INV);

	std::vector<std::vector<cv::Point>> contours;
	cv::findContours(binImg, contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);

	for (auto cont : contours)
	{
		cv::RotatedRect minRect = cv::minAreaRect(cont);
		double blobWidth = min(minRect.size.width, minRect.size.height);
		double blobHeight = max(minRect.size.width, minRect.size.height);

		if (abs(blobWidth - params.labelWidth) < 500 && abs(blobHeight - params.labelHeight) < 500) {
			labelMarkInfo.labelMaskBBox = cv::boundingRect(cont);
			labelMarkInfo.labelMaskBBox.x -= expand;
			labelMarkInfo.labelMaskBBox.y -= expand;
			labelMarkInfo.labelMaskBBox.width += expand * 2;
			labelMarkInfo.labelMaskBBox.height += expand * 2;
			labelMarkInfo.labelMask = binImg(labelMarkInfo.labelMaskBBox);
			cv::morphologyEx(labelMarkInfo.labelMask, labelMarkInfo.labelMask, cv::MORPH_DILATE, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(9, 9)));
			break;
		}
	}

	// 2. Pol Num
	if (params.bUsePolNum) {
		cv::Mat polNumBinImg;
		cv::Mat polNumRoiImg = cellImg(params.polNumROI);
		cv::threshold(polNumRoiImg, polNumBinImg, params.polNumThreshold, 255, cv::THRESH_BINARY_INV);

		std::vector<std::vector<cv::Point>>().swap(contours);
		std::vector<cv::Point>().swap(connectContorus);

		cv::findContours(polNumBinImg, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
		for (auto cont : contours) {
			if (cv::contourArea(cont) < 100) continue;
			connectContorus.insert(connectContorus.end(), cont.begin(), cont.end());
		}

		tempRect = cv::boundingRect(connectContorus);
		if (tempRect.width > params.polNumMinWidth && tempRect.width < params.polNumMaxWidth && tempRect.height > params.polNumMinHeight && tempRect.height < params.polNumMaxHeight)
		{
			tempRect.x -= expand;
			tempRect.y -= expand;
			tempRect.width += expand * 2;
			tempRect.height += expand * 2;
			labelMarkInfo.polNumMask = polNumBinImg(tempRect);
			labelMarkInfo.polNumBBox.x = tempRect.x + params.cellBBox.x + params.polNumROI.x;
			labelMarkInfo.polNumBBox.y = tempRect.y + params.cellBBox.y + params.polNumROI.y;
			labelMarkInfo.polNumBBox.width = tempRect.width;
			labelMarkInfo.polNumBBox.height = tempRect.height;

			if (params.bSaveTemplate) {
				cv::imwrite(polmarkDir + "num_" + std::to_string(params.polNumID) + ".jpg", labelMarkInfo.polNumMask);
			}
			else {
				// 模板匹配，去除非polmark区域
				stPolMatchInfo matchInfo;
				long ret = PolMarkRec(labelMarkInfo.polNumMask, *(params.polNumTemplates), matchInfo, E_POL_NUM);
				if (ret == 0) {
					cv::Mat tempMask = cv::Mat::zeros(labelMarkInfo.polNumMask.size(), labelMarkInfo.polNumMask.type());
					if (matchInfo.loc.x + matchInfo.templateImg.cols >= tempMask.cols) {
						std::cout << "Match pol mark NUM fail!" << std::endl;
						return -1;
					}
					cv::Rect templateRectInMask(matchInfo.loc.x, matchInfo.loc.y, matchInfo.templateImg.cols, matchInfo.templateImg.rows);
					matchInfo.templateImg.copyTo(tempMask(templateRectInMask));

					// 膨胀template后相减
					cv::Mat	StructElem = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(polmarkDilateSize, polmarkDilateSize));
					cv::morphologyEx(tempMask, tempMask, cv::MORPH_DILATE, StructElem);
					cv::bitwise_and(labelMarkInfo.polNumMask, tempMask, labelMarkInfo.polNumMask);
				}
				else {
					std::cout << "Match pol mark NUM fail!" << std::endl;
					return -1;
				}
			}
		}
	}

	// Pol Sign
	if (params.bUsePolSign) {
		cv::Mat polSignBinImg;
		cv::Mat polSignRoiImg = cellImg(params.polSignROI);
		cv::threshold(polSignRoiImg, polSignBinImg, params.polSignThreshold, 255, cv::THRESH_BINARY_INV);

		std::vector<std::vector<cv::Point>>().swap(contours);
		std::vector<cv::Point>().swap(connectContorus);

		cv::findContours(polSignBinImg, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
		for (auto cont : contours) {
			if (cv::contourArea(cont) < 100) continue;
			connectContorus.insert(connectContorus.end(), cont.begin(), cont.end());
		}

		tempRect = cv::boundingRect(connectContorus);
		if (tempRect.width > params.polSignMinWidth && tempRect.width < params.polSignMaxWidth && tempRect.height > params.polSignMinHeight && tempRect.height < params.polSignMaxHeight)
		{
			tempRect.x -= expand;
			tempRect.y -= expand;
			tempRect.width += expand * 2;
			tempRect.height += expand * 2;
			labelMarkInfo.polSignMask = polSignBinImg(tempRect);
			labelMarkInfo.polSignBBox.x = tempRect.x + params.cellBBox.x + params.polSignROI.x;
			labelMarkInfo.polSignBBox.y = tempRect.y + params.cellBBox.y + params.polSignROI.y;
			labelMarkInfo.polSignBBox.width = tempRect.width;
			labelMarkInfo.polSignBBox.height = tempRect.height;

			if (params.bSaveTemplate) {
				// 保存模版
				
				cv::imwrite(polmarkDir + "sign_" + std::to_string(params.polSignID) + ".jpg", labelMarkInfo.polSignMask);
			}
			else {
				// 模板匹配，去除非polmark区域
				stPolMatchInfo matchInfo;
				long ret = PolMarkRec(labelMarkInfo.polSignMask, *(params.polSignTemplates), matchInfo, E_POL_SIGN);
				if (ret == 0) {
					cv::Mat tempMask = cv::Mat::zeros(labelMarkInfo.polSignMask.size(), labelMarkInfo.polSignMask.type());
					if (matchInfo.loc.x + matchInfo.templateImg.cols >= tempMask.cols) {
						std::cout << "Match pol mark SIGN fail!" << std::endl;
						return -1;
					}
					cv::Rect templateRectInMask(matchInfo.loc.x, matchInfo.loc.y, matchInfo.templateImg.cols, matchInfo.templateImg.rows);
					matchInfo.templateImg.copyTo(tempMask(templateRectInMask));

					// 膨胀template后相减
					cv::Mat	StructElem = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(polmarkDilateSize, polmarkDilateSize));
					cv::morphologyEx(tempMask, tempMask, cv::MORPH_DILATE, StructElem);
					cv::bitwise_and(labelMarkInfo.polSignMask, tempMask, labelMarkInfo.polSignMask);
				}
				else {
					std::cout << "Match pol mark SIGN fail!" << std::endl;
					return -1;
				}
			}
		}
	}

	return 0;
}

long InspectLabelPol::DoFillLabelMark(cv::Mat& image, STRU_LabelMarkParams& params, const STRU_LabelMarkInfo& labelMarkInfo)
{
	if (params.bSaveTemplate) return 0;

	if (params.bUseLabel) {
		this->DoFillLabel(image, labelMarkInfo);
	}

	if (params.bUsePolNum) {
		this->DoFillPolArea(image, params, labelMarkInfo, E_POL_NUM);
	}
	
	if (params.bUsePolSign) {
		this->DoFillPolArea(image, params, labelMarkInfo, E_POL_SIGN);
	}

	return 0;
}

long InspectLabelPol::DoFillPolArea(cv::Mat& dstImg, STRU_LabelMarkParams& params, const STRU_LabelMarkInfo& labelMarkInfo, PolMarkType polType)
{
	cv::Mat polMask = polType == E_POL_NUM ? labelMarkInfo.polNumMask : labelMarkInfo.polSignMask;
	cv::Rect bbox = polType == E_POL_NUM ? labelMarkInfo.polNumBBox : labelMarkInfo.polSignBBox;
	// 遍历pol的连通域填充
	cv::Mat labels, stats, centroids;
	int num_labels = cv::connectedComponentsWithStats(polMask, labels, stats, centroids);
	for (int i = 1; i < num_labels; ++i) {
		int width = stats.at<int>(i, cv::CC_STAT_WIDTH);
		int height = stats.at<int>(i, cv::CC_STAT_HEIGHT);
		int area = stats.at<int>(i, cv::CC_STAT_AREA);
		int left = stats.at<int>(i, cv::CC_STAT_LEFT);
		int top = stats.at<int>(i, cv::CC_STAT_TOP);
		
		// 外扩box
		left -= params.polCellSize * 2;
		top -= params.polCellSize * 2;
		width += params.polCellSize * 4;
		height += params.polCellSize * 4;
		left += bbox.x;
		top += bbox.y;
		cv::Rect maskRect = cv::Rect(left, top, width, height);
		std::vector<cv::Rect> refRect = GetPolRefRect(dstImg, maskRect, labelMarkInfo, polType);

		// 并行填充
		long res = PartialFill(dstImg, maskRect, refRect);
	}

	return 0;
}

long InspectLabelPol::DoFillLabel(cv::Mat& dstImg, const STRU_LabelMarkInfo& labelMarkInfo)
{
	cv::Mat labMask = labelMarkInfo.labelMask;
	cv::Mat subLabImg = dstImg(labelMarkInfo.labelMaskBBox);
	cv::Rect labRect = labelMarkInfo.labelMaskBBox;

	cv::Mat boundMask, dilateMask, erodeMask;
	cv::Mat kerMask = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(17, 17));
	cv::dilate(labMask, dilateMask, kerMask);
	//cv::erode(maskLab, erodeMask, kerMask);
	cv::subtract(dilateMask, labMask, boundMask);

	cv::Mat labOutMask;
	cv::Mat subLabImgFill = cv::Mat::zeros(subLabImg.size(), CV_8UC1);
	cv::Mat subLabImgX = cv::Mat::zeros(subLabImg.size(), CV_8UC1);
	cv::Mat subLabImgY = cv::Mat::zeros(subLabImg.size(), CV_8UC1); 
	cv::bitwise_not(labMask, labOutMask);
	 
	// 在labOut区域计算灰度平均值填充
	int STEP = 1;
	for (int Y = STEP; Y < labRect.height - STEP; Y += STEP) {
		cv::Rect tempRect = cv::Rect(0, Y - STEP, labRect.width, 2 * STEP + 1);
		cv::Mat oriRectImg = subLabImg(tempRect);
		cv::Mat outRectMask = labOutMask(tempRect);
		cv::Scalar XMeanGV = cv::mean(oriRectImg, outRectMask);
		//cv::Mat matFillROI = subLabImgY(cv::Rect(0, Y - 1, labRect.width, 5));
		subLabImgY(cv::Rect(0, Y, labRect.width, STEP)).setTo(int(XMeanGV[0] + 0.5));
	}
	for (int X = STEP; X < labRect.width - STEP; X += STEP) {
		cv::Rect tempRect = cv::Rect(X - STEP, 0, 2 * STEP + 1, labRect.height);
		cv::Mat oriRectImg = subLabImg(tempRect);
		cv::Mat outRectMask = labOutMask(tempRect);
		cv::Scalar YMeanGV = cv::mean(oriRectImg, outRectMask);
		//cv::Mat matFillROI = subLabImgX(cv::Rect(X - 1, 0, 5, labRect.height));
		subLabImgX(cv::Rect(X, 0, 1, labRect.height)).setTo(int(YMeanGV[0] + 0.5));
	}

	cv::Mat temp1, temp2;
	cv::Mat subAddLabImg;

	// 水平方向和竖直方向融合
	//cv::min(subLabImgX, subLabImgY, subLabImgFill);
	cv::addWeighted(subLabImgX, 0.5, subLabImgY, 0.5, 0.5, subLabImgFill);

	cv::bitwise_and(subLabImg, labOutMask, temp1);
	cv::bitwise_and(subLabImgFill, labMask, temp2);
	// label内外合并
	cv::add(temp1, temp2, subAddLabImg);
	
	// 边缘处理
	
	cv::bitwise_and(subLabImgFill, boundMask, temp1);
	cv::bitwise_and(subLabImg, boundMask, temp2);
	cv::addWeighted(temp1, 0.6, temp2, 0.4, 0, temp1);
	cv::bitwise_and(subAddLabImg, ~boundMask, temp2);
	cv::add(temp1, temp2, subLabImg);

	return 0;
}

long InspectLabelPol::PartialFill(cv::Mat& dstImg, cv::Rect& maskRect, std::vector<cv::Rect>& refRect)
{
	cv::Scalar meanX, meanY;
	cv::Rect hRefRect = refRect[0];
	cv::Rect vRefRect = refRect[1];

	cv::Mat dstBuff = dstImg(maskRect);

	// 截图参考图像和mask图像
	cv::Mat hRefImg = dstImg(hRefRect);
	cv::Mat vRefImg = dstImg(vRefRect);
	cv::Mat maskImg = dstImg(maskRect);

	// 间隔位置，从最亮的开始
	// 计算投影 mode: 1(x轴投影)、0（y轴投影）
	cv::Mat hRefProj = CalcProjection(hRefImg, 0);
	cv::Mat vRefProj = CalcProjection(vRefImg, 1);
	cv::Mat hMaskProj = CalcProjection(maskImg, 0);
	cv::Mat vMaskProj = CalcProjection(maskImg, 1);
	// 计算相对位置
	std::vector<int>hLighterPnt = CalcLighterPos(hRefProj, 0);
	std::vector<int>vLighterPnt = CalcLighterPos(vRefProj, 1);
	std::vector<int>hLMaskPnt = CalcLighterPos(hMaskProj, 0);
	std::vector<int>vLMaskPnt = CalcLighterPos(vMaskProj, 1);
		
	int startY = vLMaskPnt[0];
	int startX = hLMaskPnt[0];

	//int endY = maskRect.height - 1;
	//int endX = maskRect.width - 1;
	int endY = vLMaskPnt[vLMaskPnt.size() - 1];
	int endX = hLMaskPnt[hLMaskPnt.size() - 1];
	int countX = 0;
	int countY = 0;
	for (int i = startY; i < endY; i++) {
		auto it = std::find(vLMaskPnt.begin() + 1, vLMaskPnt.end(), i);
		if (it != vLMaskPnt.end()) {
			countY = 0;
		}
		for (int j = startX; j < endX; j++) {
			auto it = std::find(hLMaskPnt.begin() + 1, hLMaskPnt.end(), j);
			if (it != hLMaskPnt.end()) {
				countX = 0;
			}
			// 水平方向
			double meanH = CalcMeanGrayX(hRefImg, hLighterPnt, countX, i);
			// 垂直方向
			double meanV = CalcMeanGrayY(vRefImg, vLighterPnt, countY, j);

			//插值
			int grayValue = int((meanH + meanV)/2);
			//int grayValue = int((std::max)(meanH, meanV));			
			cv::Rect pix = cv::Rect(j, i, 1, 1);
			cv::Mat pixImg = dstBuff(pix);
			if (grayValue > 10) {
				pixImg.setTo((unsigned int)grayValue);
			}
			countX++;			
		}
		countY++;		
	}
	return 0;
}

std::vector<cv::Rect> InspectLabelPol::GetPolRefRect(cv::Mat dstImg, cv::Rect& maskRect, const STRU_LabelMarkInfo& labelMarkInfo, PolMarkType polType)
{
	//最好是label和Pol各在一边
	// 分别在水平方向和竖直方向取一个Rect
	std::vector<cv::Rect> refRect(2);
	// 计算AA区的中心点
	/*cv::Point aaCentPnt;
	aaCentPnt.x = (aaPoint.ltPoint.x + aaPoint.rbPoint.x) / 2;
	aaCentPnt.y = (aaPoint.ltPoint.y + aaPoint.rbPoint.y) / 2;*/

	cv::Point polLT = polType == E_POL_NUM ? labelMarkInfo.polNumBBox.tl() : labelMarkInfo.polSignBBox.tl();
	cv::Point polRB = polType == E_POL_NUM ? labelMarkInfo.polNumBBox.br() : labelMarkInfo.polSignBBox.br();
	cv::Point polCent = cv::Point((polLT.x + polRB.x) / 2, (polLT.y + polRB.y) / 2);

	cv::Point maskCent;
	maskCent.x = (maskRect.x + maskRect.x + maskRect.width) / 2;
	maskCent.y = (maskRect.y + maskRect.y + maskRect.height) / 2;


	// 如果maskRect在中心点上方，则refRect在中心点上方（反之）
	// 如果maskRect在中心点右方，则refRect在中心点右方（反之）
	
	for (int i = 0; i < 2; i++) {
		// x方向
		if (i == 0) {
			if (polCent.x > maskCent.x)
			{
				refRect[i].width = maskRect.width * 2;
				refRect[i].x = polLT.x - 10 - refRect[i].width;
			}
			else {
				refRect[i].width = maskRect.width * 2;
				refRect[i].x = polRB.x + 10;
			}
			refRect[i].y = maskRect.y;
			refRect[i].height = maskRect.height;
		}
		// y方向
		else {
			if (polCent.y > maskCent.y)
			{
				refRect[i].height = maskRect.height * 2;
				refRect[i].y = polLT.y - 10 - refRect[i].height;
			}
			else {
				refRect[i].height = maskRect.height * 2;
				refRect[i].y = polRB.y + 10;
			}
			refRect[i].x = maskRect.x;
			refRect[i].width = maskRect.width;
		}
		// 超限判断,不能在labPoint内部
	}
	cv::Mat hRef = dstImg(refRect[0]);
	cv::Mat vRef = dstImg(refRect[1]);
	return refRect;
}

cv::Mat InspectLabelPol::CalcProjection(const cv::Mat& srcImg, int mode)
{
	if (srcImg.size().width == 0 || srcImg.size().height == 0) {
		return srcImg;
	}

	// mode: 1(x投影)、0（y投影）
	cv::Mat projImg;
	cv::reduce(srcImg, projImg, mode, cv::REDUCE_SUM, CV_32SC1);
	return projImg;
}

std::vector<int> InspectLabelPol::CalcLighterPos(const cv::Mat projImg, int mode)
{
	std::vector<int> pos;
	if (projImg.size().width == 0 || projImg.size().height == 0) {
		return pos;
	}
	
	// y轴投影结果
	if (mode == 1) {
		for (int i = 1; i < projImg.rows - 1; ++i) {
			if (projImg.at<int>(i, 0) >= projImg.at<int>(i - 1, 0) && 
				projImg.at<int>(i, 0) > projImg.at<int>(i + 1, 0)) {
				pos.push_back(i);
			}
		}
	}
	else {
		for (int i = 1; i < projImg.cols - 1; ++i) {
			if (projImg.at<int>(0, i) >= projImg.at<int>(0, i - 1) &&
				projImg.at<int>(0, i) > projImg.at<int>(0, i + 1)) {
				pos.push_back(i);
			}
		}
	}
	return pos;
}

double InspectLabelPol::CalcMeanGrayX(const cv::Mat& src, const std::vector<int>& pos, int count, int posY)
{
	double sum = 0;
	if (pos.size() <= 0) {
		return sum;
	}
	for (int i = 0; i < pos.size() - 2; i++) {
		uint8_t pixVal = src.at<uint8_t>(posY, pos[i] + count);
		sum = sum + double(pixVal);
	}
	sum = sum / (pos.size() - 2);
	return sum;
}

double InspectLabelPol::CalcMeanGrayY(const cv::Mat& src, const std::vector<int>& pos, int count, int posX)
{
	double sum = 0;
	if (pos.size() <= 0) {
		return sum;
	}
	for (int i = 0; i < pos.size() - 2; i++) {
		uint8_t pixVal = src.at<uint8_t>(pos[i] + count, posX);
		sum = sum + double(pixVal);
	}
	sum = sum / (pos.size() - 2);
	return sum;
}

long InspectLabelPol::PolMarkRec(const cv::Mat& srcImg, std::map<std::string, cv::Mat>& templates, stPolMatchInfo& polMatchInfo, PolMarkType polType)
{
	if (templates.size() < 1) {
		return -1;
	}

	// 遍历图像：模板匹配
	cv::Mat templateImg;
	for (auto imgPair : templates) {
		templateImg = imgPair.second;
		//匹配
		cv::Mat resultImg;
		int result_cols = srcImg.cols - templateImg.cols + 1;
		int result_rows = srcImg.rows - templateImg.rows + 1;
		resultImg.create(result_rows, result_cols, CV_32FC1);
		cv::matchTemplate(srcImg, templateImg, resultImg, cv::TM_CCORR_NORMED);
		//cv::normalize(resultImg, resultImg, 0, 1,cv::NORM_MINMAX, -1, cv::Mat());
		double minVal, maxVal;
		cv::Point minLoc, maxLoc;
		cv::minMaxLoc(resultImg, &minVal, &maxVal, &minLoc, &maxLoc, cv::Mat());
		if (maxVal > 0.82 && maxVal > polMatchInfo.score) {
			polMatchInfo.templateImg = templateImg;
			polMatchInfo.score = maxVal;
			polMatchInfo.loc = maxLoc;
			polMatchInfo.name = imgPair.first;
		}
	}

	if (polMatchInfo.score > 0) {
		return 0;
	}
	else {
		return -1;
	}
}
