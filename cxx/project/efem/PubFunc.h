#pragma once
#include "../framework/Defines.h"
#include "Define.h"
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <nlohmann/json.hpp>  
#include <windows.h>
#include <typeinfo>
#include <cmath>
// #include "Logger.h"

using json = nlohmann::json;

namespace JsonUtils {
	std::string formatJsonToString1(const json& j);
	json parseJson(const char* json_string);
	json readJsonFile(const char* filename);
	json emptyJson();

    template<typename T>
    T getValueFromJson(const json& j, const std::string& key, const T& defaultValue)
    {
        try {
            if (j.contains(key)) {
                return j[key].get<T>();
            } else {
                return defaultValue;
            }
        }
        catch (const json::type_error& e) {
            LOGI("key is : {}, Json object: {}, error info: {} ", key, j.dump(), e.what());
            throw e;
        }
        return defaultValue;
    }

    template<typename T>
    bool getValueFromJson(const json& j,
                          const std::string& key,
                          T&    outValue,
                          T     defaultValue) {

        if (j.contains(key) == false) return false;
        try {
            outValue = getValueFromJson<T>(j, key, defaultValue);
            return true;
        }
        catch (const json::type_error& e){
            return false;
        }
    }

} // JsonUtils

class PatrolEdge
{
public:
    PatrolEdge();
    ~PatrolEdge();
    virtual void RunAlgoBroken(InferTaskPtr task, std::vector<AlgoResultPtr> pre_results, AlgoResultPtr algo_result, json judgeParams, std::vector<stBLOB_FEATURE>&	BlobResultTotal);
	virtual void RunAlgoPeeling(InferTaskPtr task, std::vector<AlgoResultPtr> pre_results, AlgoResultPtr algo_result, json judgeParams, std::vector<stBLOB_FEATURE>&	BlobResultTotal);
	virtual void RunAlgoSplit(InferTaskPtr task, std::vector<AlgoResultPtr> pre_results, AlgoResultPtr algo_result, json judgeParams, std::vector<stBLOB_FEATURE>&	BlobResultTotal);
    
    
    void PatrolEdge::NormSobel(int dthr, const cv::Mat& InPutImage, cv::Mat& OutPutImage);
	int getSignFromSymbol(const std::string& symbol);
	std::string getSymbolFromSign(int sign);
    void Image_range_transformation(InferTaskPtr task, cv::Mat image);
    void detectAndDrawLines(InferTaskPtr task, const cv::Mat& inputImage, cv::Mat& edges);
    void findAndDrawContours(cv::Mat grayImage, cv::Mat& binImg, int kernelSize = 3);
    void removeShadows(InferTaskPtr task, cv::Mat img, cv::Mat& calcMat);
    void test(InferTaskPtr task, cv::Mat src);
    bool Make_HardDefect_Mask(cv::Mat& matGrayChanels, cv::Mat& defectMask, int nLineThreshold, int nStepX, int nStepY);
    bool Make_LineDefect_Mask(cv::Mat& matLineMask, cv::Mat& defectLineMask, int nLineThickness);
	void makeMask_and_obtLineVec(InferTaskPtr task, cv::Mat src, cv::Mat fitMask, cv::Mat& lineFet, cv::Mat& mask, int minCheckEdge, int maxCheckEdge, int nMinSamples, int distThreshold, int edgeType);
	
	bool BG_Subtract(cv::Mat& matGrayChanels, cv::Mat& matBGSizeDark, cv::Mat& matBGSizeBright, int nStepX, int nStepY, int nStepRows, int nStepCols, int bgThreshold = 50);
    bool Estimation_X(cv::Mat matSrcBuf, cv::Mat& matDstBuf, int nDimensionX, int nStepX, float fThBGOffset);
    bool Estimation_Y(cv::Mat matSrcBuf, cv::Mat& matDstBuf, int nDimensionY, int nStepY, float fThBGOffset);
    bool TwoImg_Average(cv::Mat matSrc1Buf, cv::Mat matSrc2Buf, cv::Mat& matDstBuf);
    bool Separation_ActiveAre(cv::Mat& matSrcBuf, cv::Mat& matResROIBuf, int nThreshold, int nEdgePiexl);
    std::vector<cv::Point> findImageBoundaryPoints(const cv::Mat& image);
	bool DoBlobCalculate(cv::Mat ThresholdBuffer, cv::Mat matBKBuf, cv::Mat GrayBuffer, int nMaxDefectCount, std::vector<stBLOB_FEATURE>& m_BlobResult);
	bool DoFeatureBasicColor_8bit(cv::Mat& matLabel, cv::Mat& matStats, cv::Mat& matCentroid, cv::Mat& GrayBuffer, cv::Mat matBKBuf, int nTotalLabel, std::vector<stBLOB_FEATURE>& m_BlobResult);
	bool DrawBlob(cv::Mat& DrawBuffer, cv::Scalar DrawColor, long nOption, bool bSelect, std::vector<stBLOB_FEATURE>& m_BlobResult, float fFontSize = 0.4f);
	//std::vector<stBLOB_FEATURE>	m_BlobResult;
	cv::Point2f getCrossPoint(cv::Vec4i LineA, cv::Vec4i LineB);
	long RobustFitLine(cv::Mat& matTempBuf, cv::Rect rectCell, long double& dA, long double& dB, int nMinSamples, double distThreshold, int nType, int nSamp = 2);
	long calcRANSAC(std::vector <cv::Point2i>& ptSrcIndexs, long double& dA, long double& dB, int nMinSamples, double distThreshold);
	long GetRandomSamples(std::vector <cv::Point2i>& ptSrcIndexs, std::vector <cv::Point2i>& ptSamples, int nSampleCount);
	bool FindInSamples(std::vector <cv::Point2i>& ptSamples, cv::Point2i ptIndexs);
	long calcLineFit(std::vector <cv::Point2i>& ptSamples, long double& dA, long double& dB);
	long calcLineVerification(std::vector <cv::Point2i>& ptSrcIndexs, std::vector <cv::Point2i>& ptInliers, long double& dA, long double& dB, double distThreshold);
	bool DoFiltering(stBLOB_FEATURE& tBlobResult, int nBlobFilter, int nSign, double dValue);
	bool Compare(double dFeatureValue, int nSign, double dValue);
	cv::Point2f calculateCentroid(const std::vector<cv::Point>& contour);
	void connectFlawEdge(const std::vector<std::vector<cv::Point>>& emptyObject, std::vector<std::vector<cv::Point>>& flawEdgeObject, int maxDistance);
	void findMergedContours(const cv::Mat& src, cv::Mat& mergeContours, int threshold1, int threshold2, int contourThreshold);
	void makeMask_and_obtLineVec_BR(InferTaskPtr task, cv::Mat src, cv::Mat fitMask, cv::Mat& lineFet, cv::Mat& mask, int minCheckEdge, int maxCheckEdge, int nMinSamples, int distThreshold, int edgeType);
    void makeMask_and_obtLineVec_BL(InferTaskPtr task, cv::Mat src, cv::Mat fitMask, cv::Mat& lineFet, cv::Mat& mask, int minCheckEdge, int maxCheckEdge, int nMinSamples, int distThreshold, int edgeType);
    void makeMask_and_obtLineVec_TR(InferTaskPtr task, cv::Mat src, cv::Mat fitMask, cv::Mat& lineFet, cv::Mat& mask, int minCheckEdge, int maxCheckEdge, int nMinSamples, int distThreshold, int edgeType);
    void makeMask_and_obtLineVec_TL(InferTaskPtr task, cv::Mat src, cv::Mat fitMask, cv::Mat& lineFet, cv::Mat& mask, int minCheckEdge, int maxCheckEdge, int nMinSamples, int distThreshold, int edgeType);
    
    void makeMask_and_obtLineVec_B(InferTaskPtr task, cv::Mat src, cv::Mat& fitMask, cv::Mat& lineFet, cv::Mat& mask, int minCheckEdge, int maxCheckEdge, int nMinSamples, int distThreshold, int edgeType);
	void makeMask_and_obtLineVec_T(InferTaskPtr task, cv::Mat src, cv::Mat fitMask, cv::Mat& lineFet, cv::Mat& mask, int minCheckEdge, int maxCheckEdge, int nMinSamples, int distThreshold, int edgeType);
	void makeMask_and_obtLineVec_R(InferTaskPtr task, cv::Mat src, cv::Mat fitMask, cv::Mat& lineFet, cv::Mat& mask, int minCheckEdge, int maxCheckEdge, int nMinSamples, int distThreshold, int edgeType);
	void makeMask_and_obtLineVec_L(InferTaskPtr task, cv::Mat src, cv::Mat fitMask, cv::Mat& lineFet, cv::Mat& mask, int minCheckEdge, int maxCheckEdge, int nMinSamples, int distThreshold, int edgeType);
	
	int findMaxGVDifferenceCorners(cv::Mat src);
	bool EnhanceContrast(cv::Mat& matSrcBuf, int nOffSet, double dRatio);
	void LineDFT(cv::Mat &matSrcImage, cv::Mat &matDstImage, BOOL bRemove, int nAxis, double dDegree);
	void Complex2SpectrumSave(cv::Mat &matComplex, cv::Mat &matPadded, cv::Mat &matSpectrum);
	void Complex2SpectrumComplex(cv::Mat &matComplex, cv::Mat &matPadded, cv::Mat &matSpectrum, int PlanesIdx);
	void SpectrumComplex2Complex(cv::Mat &matSpectrum1, cv::Mat &matSpectrum2, cv::Mat &matPadded, cv::Mat &matComplex);
	// void writeDataToExcel(const std::vector<stBLOB_FEATURE>& data);
	// void WriteBlobResultInfo(std::vector<stBLOB_FEATURE> BlobResultTotal);
public:
    void write_debug_img(InferTaskPtr task, std::string name, cv::Mat img);
	json ReadJsonFile(std::string filepath);
	std::tuple<std::string, json> get_task_info(InferTaskPtr task, std::map<std::string, json> param_map);
    bool checkAbnormal(cv::Mat img);

	void result_to_json(const std::vector<stBLOB_FEATURE>& BlobResultTotal, json& result_info, std::string result);

    void judgeFeature(STRU_DEFECT_ITEM *EdgeDefectJudgment, std::vector<stBLOB_FEATURE>	m_BlobResult, std::vector<stBLOB_FEATURE>& BlobResultTotal);
    void WriteBlobResultInfo(InferTaskPtr task, std::vector<stBLOB_FEATURE> BlobResultTotal);

    void WriteBlobResultInfo_F(InferTaskPtr task, std::vector<stBLOB_FEATURE> BlobResultTotal);

    void WriteJudgeParams(InferTaskPtr task, const STRU_DEFECT_ITEM *EdgeDefectJudgment, int i);

    double calculateDistance(const cv::Point& point, double a, double b);

    double calculateMinDistanceToLines(const cv::Rect& rectBox, double a, double b);
    void checkPoint(cv::Point& p);

    std::tuple<double, double> calculateSlopeAndIntercept(double x1, double y1, double x2, double y2);//б�ʺͽؾ�

    void detectEdgesWithGaps(InferTaskPtr task, const cv::Mat& inputImage, cv::Mat& outputImage, const cv::Mat& lineFet);

    //void Image_Pow(double dpow, cv::Mat& InPutImage, cv::Mat& OutPutImage);

};