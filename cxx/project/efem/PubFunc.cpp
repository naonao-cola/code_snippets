#pragma once
#include "PubFunc.h"
#include <windows.h>
#include "../utils/logger.h"
#include "FindLine.h"

PatrolEdge::PatrolEdge()
{

}

PatrolEdge::~PatrolEdge()
{

}

void PatrolEdge::RunAlgoBroken(InferTaskPtr task, std::vector<AlgoResultPtr> pre_results, AlgoResultPtr algo_result, json judgeParams, std::vector<stBLOB_FEATURE>&	BlobResultTotal){return;}
void PatrolEdge::RunAlgoPeeling(InferTaskPtr task, std::vector<AlgoResultPtr> pre_results, AlgoResultPtr algo_result, json judgeParams, std::vector<stBLOB_FEATURE>&	BlobResultTotal){return;}
void PatrolEdge::RunAlgoSplit(InferTaskPtr task, std::vector<AlgoResultPtr> pre_results, AlgoResultPtr algo_result, json judgeParams, std::vector<stBLOB_FEATURE>&	BlobResultTotal){return;}


json PatrolEdge::ReadJsonFile(std::string filepath)
{
    std::ifstream conf_i(filepath);
    std::stringstream ss_config;
    ss_config << conf_i.rdbuf();

    json jsonObj = json::parse(ss_config.str());
    return std::move(jsonObj);
}

int PatrolEdge::getSignFromSymbol(const std::string& symbol)
{
    // if ("\u003e" == ">")
    // {
    //     return 1;
    //     /* code */
    // }
    
    if (symbol == "=")
        return 0;
    else if (symbol == "<>")
        return 1;
    else if (symbol == ">")
        return 2;
    else if (symbol == "<")
        return 3;
    else if (symbol == ">=")
        return 4;
    else if (symbol == "<=")
        return 5;
    else if (symbol == ">||")
        return 6;
    else if (symbol == "||<")
        return 7;
    else
        return -1; 
}

std::string PatrolEdge::getSymbolFromSign(int sign)
{
    switch (sign)
    {
    case 0:
        return "=";
    case 1:
        return "<>";
    case 2:
        return ">";
    case 3:
        return "<";
    case 4:
        return ">=";
    case 5:
        return "<=";
    case 6:
        return ">||";
    case 7:
        return "||<";
    default:
        return "";
    }
}

std::tuple<std::string, json> PatrolEdge::get_task_info(InferTaskPtr task, std::map<std::string, json> param_map)
{
    std::string task_type_id = task->image_info["type_id"];
    json        task_json = param_map[task_type_id];
    return std::make_tuple(task_type_id, task_json);
}

void PatrolEdge::test(InferTaskPtr task, cv::Mat src) {

    cv::Mat image = src.clone();
    cv::cvtColor(image, image, cv::COLOR_BGR2GRAY);
    // 阈值分割
    cv::Mat binaryImage;
    double thresholdValue = 80;
    double maxValue = 255;
    cv::threshold(image, binaryImage, thresholdValue, maxValue, cv::THRESH_BINARY);

    // 形态学处理
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
    cv::Mat regionClosing, regionDilation, regionErosion;
    cv::morphologyEx(binaryImage, regionClosing, cv::MORPH_CLOSE, kernel);
    cv::dilate(regionClosing, regionDilation, kernel, cv::Point(-1, -1), 3);
    cv::erode(regionClosing, regionErosion, kernel, cv::Point(-1, -1), 3);

    // 边缘检测
    cv::Mat regionDifference;
    absdiff(regionDilation, regionErosion, regionDifference);
    write_debug_img(task, "1_regionDifference", regionDifference);
    // 抠图处理
    cv::Mat imageReduced;
    image.copyTo(imageReduced, regionDifference);
    write_debug_img(task, "2_imageReduced", imageReduced);
    // 使用Canny滤波器进行边缘检测
    cv::Mat edges;
    double threshold1 = 20;
    double threshold2 = 40;
    cv::Canny(src, edges, threshold1, threshold2, 3);
    write_debug_img(task, "3_edges", edges);

    std::vector<std::vector<cv::Point>> contours;
    findContours(edges, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    // 合并相邻边缘
    std::vector<std::vector<cv::Point>> unionContours;
    for (size_t i = 0; i < contours.size(); i++)
    {
        std::vector<cv::Point> contour = contours[i];
        double contourLength = cv::arcLength(contour, true);
        if (contourLength > 200)
        {
            unionContours.push_back(contour);
        }
    }
    //drawContours(src, unionContours, -1, cv::Scalar(0, 0, 255), 2);
    std::vector<std::vector<cv::Point>> selectedContours;
    for (auto cont : unionContours) {
        int area = cv::contourArea(cont);
        if (area < 10) continue;
        std::vector<cv::Point> hull;
        cv::convexHull(cont, hull);
        selectedContours.push_back(hull);

    }
    //drawContours(src, selectedContours, -1, cv::Scalar(0, 0, 255), 2);


    cv::Mat result(image.size(), CV_8UC3);
    cv::cvtColor(image, result, cv::COLOR_GRAY2BGR);
    drawContours(result, selectedContours, -1, cv::Scalar(0, 0, 255), 2);


    std::vector<std::vector<cv::Point>> UnionContours = selectedContours;
    int number = UnionContours.size();
    cv::Mat drawCont = cv::Mat(src.size(), CV_8UC3);
    cv::Mat region2 = cv::Mat::zeros(src.size(), CV_8UC3);
    cv::Mat region1 = cv::Mat::zeros(src.size(), CV_8UC1);

    for (int index = 0; index < number; index++)
    {
        //int area = cv::contourArea(UnionContours[index]);
        //if (area < 200) continue;
        // 选择待检测边缘
        std::vector<cv::Point> objectSelected = UnionContours[index];
        int area = cv::contourArea(objectSelected);
        if (area < 200) continue;
        // 平滑边缘
        std::vector<cv::Point> smoothedContours;
        std::vector<std::vector<cv::Point>> conts;
        double smoothParam = 5.0;
        //conts.push_back(objectSelected);
        //cv::drawContours(drawCont, conts, -1, cv::Scalar(0, 147, 255), -1);
        //conts.clear();
        //conts.swap(conts);
        cv::approxPolyDP(objectSelected, smoothedContours, smoothParam, true);
        conts.push_back(smoothedContours);
        cv::drawContours(region1, conts, -1, cv::Scalar(147, 0, 255), -1);
    }



        // 膨胀操作
        //cv::Mat regionDilation;
        int dilationSize = 5;
        //cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(2 * dilationSize + 1, 2 * dilationSize + 1), cv::Point(dilationSize, dilationSize));
        cv::dilate(region1, regionDilation, kernel);

        // 连通区域分析
        std::vector<std::vector<cv::Point>> connectedRegions1;
        findContours(regionDilation, connectedRegions1, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        // 筛选形状
        std::vector<std::vector<cv::Point>> selectedRegions1, objectSelected;
        for (size_t i = 0; i < connectedRegions1.size(); i++)
        {
            double area = cv::contourArea(connectedRegions1[i]);
            if (area >= 200.0)
            {
                selectedRegions1.push_back(connectedRegions1[i]);
            }
        }

        //cv::drawContours(src, selectedRegions1, -1, cv::Scalar(147, 0, 255), -1);

        LOGI("aaaa");
        // 进一步处理不合格的边缘
        std::vector<std::vector<cv::Point>> emptyObject;
        for (size_t i = 0; i < selectedRegions1.size(); i++)
        {
            std::vector<cv::Point> objectSelected2 = selectedRegions1[i];
            cv::Rect boundingRect = cv::boundingRect(objectSelected2);
            std::vector<cv::Point> clippedContours;
            for (size_t j = 0; j < objectSelected2.size(); j++)
            {
                if (objectSelected2[j].x >= boundingRect.x && objectSelected2[j].x <= boundingRect.x + boundingRect.width &&
                    objectSelected2[j].y >= boundingRect.y && objectSelected2[j].y <= boundingRect.y + boundingRect.height)
                {
                    clippedContours.push_back(objectSelected2[j]);
                }
            }
            emptyObject.push_back(clippedContours);
        }
    
        // 连接相邻边缘缺陷
        std::vector<std::vector<cv::Point>> flawEdgeObject;
        int maxDistance = 300; // 最大距离阈值

        connectFlawEdge(emptyObject, flawEdgeObject, maxDistance);
        
        //cv::drawContours(src, flawEdgeObject, -1, cv::Scalar(0, 0, 255), cv::FILLED);


    //    // 显示缺陷边缘图像
    //    cv::Mat flawEdgeImage = cv::Mat::zeros(src.size(), CV_8UC1);
    //    cv::drawContours(flawEdgeImage, finalFlawEdgeObject, -1, cv::Scalar(255), 1);
    //    cv::imshow("Flaw Edge Object", flawEdgeImage);
    //    cv::waitKey(0);
    //}
            

}

void PatrolEdge::findMergedContours(const cv::Mat& src, cv::Mat& mergeContours, int threshold1, int threshold2, int contourThreshold)
{
    cv::Mat edges;
    mergeContours = cv::Mat::zeros(src.size(), CV_8UC1);
    cv::Canny(src, edges, threshold1, threshold2, 3);

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(edges, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    std::vector<std::vector<cv::Point>> unionContours;
    for (size_t i = 0; i < contours.size(); i++)
    {
        std::vector<cv::Point> contour = contours[i];
        double contourLength = cv::arcLength(contour, true);

        if (contourLength > contourThreshold)
        {
            unionContours.push_back(contour);
        }
    }
    cv::drawContours(mergeContours, unionContours, -1, cv::Scalar(255), cv::FILLED);
    return ;
}

cv::Point2f PatrolEdge::calculateCentroid(const std::vector<cv::Point>& contour)
{
    cv::Moments moments = cv::moments(contour);
    cv::Point2f centroid(moments.m10 / moments.m00, moments.m01 / moments.m00);
    return centroid;
}

void PatrolEdge::connectFlawEdge(const std::vector<std::vector<cv::Point>>& emptyObject, std::vector<std::vector<cv::Point>>& flawEdgeObject, int maxDistance)
{
    for (size_t i = 0; i < emptyObject.size(); i++)
    {
        std::vector<cv::Point> objectSelected1 = emptyObject[i];
        bool isConnected = false;

        for (size_t j = 0; j < emptyObject.size(); j++)
        {
            if (i == j)
                continue;

            std::vector<cv::Point> objectSelected2 = emptyObject[j];

            // 检查两个轮廓之间的距离
            cv::Point2f centroid1 = calculateCentroid(objectSelected1);
            cv::Point2f centroid2 = calculateCentroid(objectSelected2);
            double distance = cv::norm(centroid1 - centroid2);

            if (distance <= maxDistance)
            {
                // 距离满足条件，进行连接
                std::vector<cv::Point> unionContours;
                unionContours.insert(unionContours.end(), objectSelected1.begin(), objectSelected1.end());
                unionContours.insert(unionContours.end(), objectSelected2.begin(), objectSelected2.end());
                flawEdgeObject.push_back(unionContours);
                isConnected = true;
            }
        }

        if (!isConnected)
        {
            // 当前轮廓没有连接的相邻轮廓，将其作为单独的轮廓添加到结果中
            flawEdgeObject.push_back(objectSelected1);
        }
    }
}

void PatrolEdge::result_to_json(const std::vector<stBLOB_FEATURE>& BlobResultTotal, json& result_info, std::string result) {
    if (result == "OK") {
        result_info = json::array();
        return;
    }
//for (int i = 0; i < BlobResultTotal.size(); i++)
//{
//    if (BlobResultTotal[i].bFiltering == true) {
//        result_to_json(BlobResultTotal, result_json, "BROKEN");
//        break;
//    }
//}
    for (auto blob : BlobResultTotal) {
        if (blob.bFiltering == true) {
            json s2;
            s2["label"] = result;
            s2["points"] = { {blob.rectBox.x, blob.rectBox.y},
                {(blob.rectBox.x + blob.rectBox.width), (blob.rectBox.y + blob.rectBox.height)} };
            s2["shapeType"] = "rectangle";
            result_info.emplace_back(s2);
        }
    }

    return;
}

void PatrolEdge::judgeFeature(STRU_DEFECT_ITEM* EdgeDefectJudgment, std::vector<stBLOB_FEATURE>	m_BlobResult, std::vector<stBLOB_FEATURE>& BlobResultTotal) {

    const std::type_info& info = typeid(*this);
    //int nFork = 0;
    for (int nFork = 0; nFork < MAX_JUDGE_NUM; nFork++) {

        if (EdgeDefectJudgment[nFork].strItemName.empty()) {
            continue;
        }
        LOGI("{},nFork = {} ,{} Judge Start!", info.name(), nFork, EdgeDefectJudgment[nFork].strItemName);
        for (int i = 0; i < m_BlobResult.size(); i++)
        {

            int nFeatureCount = E_FEATURE_COUNT * 2;
            bool bFilter = true;
            bool bInit = false;

            for (int nForj = 0; nForj < nFeatureCount; nForj++)
            {
                if (!EdgeDefectJudgment[nFork].Judgment[nForj].bUse)
                    continue;
                bInit = true;

                if (!DoFiltering(
                    m_BlobResult[i],										//Blob结果
                    nForj / 2,												//比较Feature
                    EdgeDefectJudgment[nFork].Judgment[nForj].nSign,	//运算符（<，>，==，<=，>=）
                    EdgeDefectJudgment[nFork].Judgment[nForj].dValue))	//值
                {
                    //LOGI("[{}]Judgment Skip:---> {}{}{}", info.name(), EdgeDefectJudgment[nFork].Judgment[nForj].name.c_str(), getSymbolFromSign(EdgeDefectJudgment[nFork].Judgment[nForj].nSign).c_str(), EdgeDefectJudgment[nFork].Judgment[nForj].dValue);
                    bFilter = false;
                    break;
                }

            }

            //满足所有设置的条件
            if (bInit && bFilter)
            {
                m_BlobResult[i].bFiltering = true;
                //pStblobResult.bFiltering
                BlobResultTotal.push_back(m_BlobResult[i]);
            }
        }
        LOGI("{}, {} Judge End!", info.name(), EdgeDefectJudgment[nFork].strItemName);
    }
    //LOGI("{}, {} Judge End!", info.name(), EdgeDefectJudgment[nFork].strItemName);
    
    return;

}

void PatrolEdge::Image_range_transformation(InferTaskPtr task, cv::Mat image){
    cv::Mat rice;
    //rice = image;
     cv::cvtColor(image,rice,cv::COLOR_BGR2GRAY);
    cv::Mat riceBW,riceBW_INV;
    //将图像转成二值图像，同时把黑白区域图像互换
    cv::threshold(rice,riceBW,50,255,cv::THRESH_BINARY);
    cv::threshold(rice,riceBW_INV,50,255,cv::THRESH_BINARY_INV);
    //距离变换
    cv::Mat dist,dist_INV;
    cv::distanceTransform(riceBW,dist,1,3,CV_32F);//为了显示清晰，将数据类型变成CV_32F
    cv::distanceTransform(riceBW_INV,dist_INV,1,3, CV_32F);
    //显示变换结果
    // imwrite("./dist1.png",dist);
    // imwrite("./dist_INV1.png",dist_INV);
    cv::normalize(dist, dist, 0, 255, cv::NORM_MINMAX);
    cv::normalize(dist_INV, dist_INV, 0, 255, cv::NORM_MINMAX);
    write_debug_img(task, "dist1", dist);
    write_debug_img(task, "dist_INV1", dist_INV);
}

void PatrolEdge::write_debug_img(InferTaskPtr task, std::string name, cv::Mat img){

#ifdef  SAVE_DEBUG_IMG

    const std::type_info& info = typeid(*this);

    std::string imgName = task->image_info["img_name"];
    std::string fpath = fs::current_path().string() + "\\debugImg\\" + info.name() + "\\" + imgName;
    if (!fs::exists(fpath)) {
        if (!fs::create_directories(fpath)) {
            std::cerr << "Error creating directory: " << fpath << std::endl;
            std::string fpath1 = fs::current_path().string() + "\\Unkonw";
            fs::create_directories(fpath1);
        }
    }
    std::string savePath = fpath + "\\" + name + ".jpg";
    cv::imwrite(savePath, img);
#endif //  

    return;
}

bool PatrolEdge::checkAbnormal(cv::Mat img){

    int m = cv::mean(img)[0];
    if (m < 100) return false;
    return true;
}

void PatrolEdge::WriteBlobResultInfo(InferTaskPtr task, std::vector<stBLOB_FEATURE> BlobResultTotal) {
    START_TIMER
    const std::type_info& info = typeid(*this);
    std::string imgName = task->image_info["img_name"];
    std::string fpath = fs::current_path().string() + "\\debugImg\\" + info.name() + "\\" + imgName + "\\";
    if (!fs::exists(fpath)) {
        if (!fs::create_directories(fpath)) {
            std::cerr << "Error creating directory: " << fpath << std::endl;
            std::string fpath1 = fs::current_path().string() + "\\Unkonw";
            fs::create_directories(fpath1);
        }
    }
    std::string task_type_id = task->image_info["type_id"];
    std::string filePath = fpath + "BlobFeature.csv";
    std::ofstream file(filePath);

    if (file.is_open()) {

        file << "index,Area,BoxArea,BoxRatio,SumGV,MinGV,MaxGV,MeanGV,DiffGV,BKGV,StdDev,SEMU,Compactness,MinGVRatio,MaxGVRatio,DiffGVRatio,Perimeter,Roundness,Elongation,MinBoxArea,MinAxis,MajAxis,AxisRatio,Angle,MinBoxRatio,DefMeanGV,DisFEdge,MeanDE\n";

        int num = 1;
        for (const auto& blob : BlobResultTotal) {
            file << num++ << ","
                << blob.nArea << ","
                << blob.nBoxArea << ","
                << blob.fBoxRatio << ","
                << blob.nSumGV << ","
                << blob.nMinGV << ","
                << blob.nMaxGV << ","
                << blob.fMeanGV << ","
                << blob.fDiffGV << ","
                << blob.fBKGV << ","
                << blob.fStdDev << ","
                << blob.fSEMU << ","
                << blob.fCompactness << ","
                << blob.nMinGVRatio << ","
                << blob.nMaxGVRatio << ","
                << blob.fDiffGVRatio << ","
                << blob.fPerimeter << ","
                << blob.fRoundness << ","
                << blob.fElongation << ","
                << blob.fMinBoxArea << ","
                << blob.fMinorAxis << ","
                << blob.fMajorAxis << ","
                << blob.fAxisRatio << ","
                << blob.fAngle << ","
                << blob.fMinBoxRatio << ","
                << blob.fDefectMeanGV << ","
                << blob.nDistanceFromEdge << ","
                << blob.fMeanDelataE << "\n";
        }
        file.close();
        std::cout << "BlobFeature.txt saved successfully." << std::endl;
    }
    else {
        std::cerr << "Failed to open file: " << filePath << std::endl;
    }
    END_TIMER
}

void PatrolEdge::WriteBlobResultInfo_F(InferTaskPtr task, std::vector<stBLOB_FEATURE> BlobResultTotal) {
#ifdef  SAVE_DEBUG_FEATRUE
    START_TIMER
    const std::type_info& info = typeid(*this);
    std::string imgName = task->image_info["img_name"];
    std::string fpath = fs::current_path().string() + "\\debugImg\\" + info.name() + "\\" + imgName + "\\";
    if (!fs::exists(fpath)) {
        if (!fs::create_directories(fpath)) {
            std::cerr << "Error creating directory: " << fpath << std::endl;
            std::string fpath1 = fs::current_path().string() + "\\Unkonw";
            fs::create_directories(fpath1);
        }
    }
    std::string task_type_id = task->image_info["type_id"];
    std::string filePath = fpath + "BlobFeature_F.csv";
    std::ofstream file(filePath);

    if (file.is_open()) {

        file << "index,Area,BoxArea,BoxRatio,SumGV,MinGV,MaxGV,MeanGV,DiffGV,BKGV,StdDev,SEMU,Compactness,MinGVRatio,MaxGVRatio,DiffGVRatio,Perimeter,Roundness,Elongation,MinBoxArea,MinAxis,MajAxis,AxisRatio,Angle,MinBoxRatio,DefMeanGV,DisFEdge,MeanDE\n";

        int num = 1;
        for (const auto& blob : BlobResultTotal) {
            if (blob.nArea == 0) continue;
            file << num++ << ","
                << blob.nArea << ","
                << blob.nBoxArea << ","
                << blob.fBoxRatio << ","
                << blob.nSumGV << ","
                << blob.nMinGV << ","
                << blob.nMaxGV << ","
                << blob.fMeanGV << ","
                << blob.fDiffGV << ","
                << blob.fBKGV << ","
                << blob.fStdDev << ","
                << blob.fSEMU << ","
                << blob.fCompactness << ","
                << blob.nMinGVRatio << ","
                << blob.nMaxGVRatio << ","
                << blob.fDiffGVRatio << ","
                << blob.fPerimeter << ","
                << blob.fRoundness << ","
                << blob.fElongation << ","
                << blob.fMinBoxArea << ","
                << blob.fMinorAxis << ","
                << blob.fMajorAxis << ","
                << blob.fAxisRatio << ","
                << blob.fAngle << ","
                << blob.fMinBoxRatio << ","
                << blob.fDefectMeanGV << ","
                << blob.nDistanceFromEdge << ","
                << blob.fMeanDelataE << "\n";
        }

        file.close();
        std::cout << "BlobFeature.txt saved successfully." << std::endl;
    }
    else {
        std::cerr << "Failed to open file: " << filePath << std::endl;
    }
    END_TIMER
#endif

}

void PatrolEdge::detectAndDrawLines(InferTaskPtr task, const cv::Mat& inputImage, cv::Mat& edges) {
    LOGI("detectAndDrawLines Func ++");
    //cv::Mat edges;
    //cv::cvtColor(inputImage, inputImage, cv::COLOR_BGR2GRAY);
    cv::Canny(inputImage, edges, 50, 150);
    write_debug_img(task, "edges", edges);
    // Hough线变换
    std::vector<cv::Vec2f> lines;
    cv::HoughLines(edges, lines, 1, CV_PI / 180, 100);

    cv::Mat resultImage = inputImage.clone();
    for (size_t i = 0; i < lines.size(); ++i) {
        float rho = lines[i][0];
        float theta = lines[i][1];
        double a = std::cos(theta);
        double b = std::sin(theta);
        double x0 = a * rho;
        double y0 = b * rho;
        cv::Point pt1(cvRound(x0 + 1000 * (-b)), cvRound(y0 + 1000 * (a)));
        cv::Point pt2(cvRound(x0 - 1000 * (-b)), cvRound(y0 - 1000 * (a)));
        

        double slope = -std::cos(theta) / std::sin(theta);

        double angle = std::atan(slope);

        angle = angle * 180.0 / CV_PI;
        LOGI("tan theta = {}, angle = {}", slope, angle);
        int width = resultImage.cols;
        int height = resultImage.rows;
        //if (abs(angle) > 85 || abs(angle) < 3) {
            //cv::clipLine(cv::Size(width, height), pt1, pt2);
            cv::line(resultImage, pt1, pt2, cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
        //}

    }

    write_debug_img(task, "Detected Lines", resultImage);
    LOGI("detectAndDrawLines Func --");
}

void PatrolEdge::findAndDrawContours(cv::Mat grayImage, cv::Mat& binImg, int kernelSize) {
    //cv::cvtColor(grayImage, grayImage, cv::COLOR_BGR2GRAY);
    // 自动计算阈值
    cv::Mat binaryImage;
    cv::threshold(grayImage, binaryImage, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);

    // 形态学处理
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
    cv::Mat regionClosing, regionDilation, regionErosion;
    cv::morphologyEx(binaryImage, regionClosing, cv::MORPH_CLOSE, kernel);
    cv::dilate(regionClosing, regionDilation, kernel, cv::Point(-1, -1), 3);
    cv::erode(regionClosing, regionErosion, kernel, cv::Point(-1, -1), 3);

    // 边缘检测
    cv::Mat regionDifference;
    absdiff(regionDilation, regionErosion, binImg);
    //// 寻找轮廓
    //std::vector<std::vector<cv::Point>> contours;
    //cv::findContours(binaryImage, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    //for (std::vector<std::vector<cv::Point>>::iterator it = contours.begin(); it < contours.end(); it++)
    //{

    //    double area = cv::contourArea(*it);
    //    if (area < 2000) {
    //        contours.erase(it);
    //        it -= 1;
    //    }
    //}
    //binImg = cv::Mat(grayImage.size(), CV_8UC1);
    //cv::drawContours(binImg, contours, -1, cv::Scalar(255), 2);
    //// 绘制轮廓
    //cv::Mat contoursImage = cv::Mat::zeros(grayImage.size(), CV_8UC3);
    //cv::drawContours(contoursImage, contours, -1, cv::Scalar(0, 255, 0), 2);

    //write_debug_img(task, "contoursImage", contoursImage);
}

void PatrolEdge::removeShadows(InferTaskPtr task, cv::Mat img, cv::Mat& calcMat){

    //img = cv::imread(R"(E:\projdata\efemDetect\image1\U0042_240115_175801\20240115_175802_CAM2_0.jpg)");

    cv::Mat gray;
    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);



    cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    int iteration = 9;

 
    cv::Mat dilateMat;
    morphologyEx(gray, dilateMat, cv::MORPH_DILATE, element, cv::Point(-1, -1), iteration);
    //imshow("dilate", dilateMat);


    cv::Mat erodeMat;
    cv::morphologyEx(dilateMat, erodeMat, cv::MORPH_ERODE, element, cv::Point(-1, -1), iteration);
    //imshow("erode", erodeMat);
    //cv::Mat t1 = (erodeMat - gray);

    calcMat = ~(erodeMat - gray);
    write_debug_img(task, "ShadowMat", calcMat);
    cv::Mat removeShadowMat;
    cv::normalize(calcMat, removeShadowMat, 0, 200, cv::NORM_MINMAX);
    write_debug_img(task, "removeShadowMat", removeShadowMat);

}

bool PatrolEdge::Make_HardDefect_Mask(cv::Mat& matGrayChanels, cv::Mat& defectMask, int nLineThreshold, int nStepX, int nStepY)
{
    START_TIMER;
    cv::Mat matSrcROIBuf;
    matGrayChanels.copyTo(matSrcROIBuf);
    cv::Mat matLineMask = cv::Mat::zeros(matSrcROIBuf.size(), CV_8UC1);
    int nStepCols = (int)(matGrayChanels.cols / (nStepX));
    int nStepRows = (int)(matGrayChanels.rows / (nStepY));
    cv::Mat matBGSizeDark;
    cv::Mat matBGSizeBright;

    matBGSizeDark = cv::Mat::zeros(matSrcROIBuf.size(), CV_8UC1);
    matBGSizeBright = cv::Mat::zeros(matSrcROIBuf.size(), CV_8UC1);

    //int nTh = 3;
    int nTh = nLineThreshold;
    cv::Mat matSrcResize;

    cv::resize(matSrcROIBuf, matSrcResize, cv::Size(nStepCols, nStepRows), cv::INTER_LINEAR);

    cv::Mat BGresize = cv::Mat::zeros(matSrcResize.size(), CV_8UC1);

    cv::medianBlur(matSrcResize, BGresize, 3);//9

    cv::resize(BGresize, BGresize, matSrcROIBuf.size(), cv::INTER_LINEAR);

    cv::subtract(BGresize, matSrcROIBuf, matBGSizeDark);
    cv::subtract(matSrcROIBuf, BGresize, matBGSizeBright);

    cv::threshold(matBGSizeDark, matBGSizeDark, nTh, 255.0, cv::THRESH_BINARY);
    cv::threshold(matBGSizeBright, matBGSizeBright, nTh, 255.0, cv::THRESH_BINARY);

    cv::add(matLineMask, matBGSizeBright, matLineMask);
    cv::add(matLineMask, matBGSizeDark, matLineMask);

    matBGSizeDark.release();
    matBGSizeBright.release();
    
    defectMask = matLineMask;
    END_TIMER;
    return true;
}

bool PatrolEdge::Make_LineDefect_Mask(cv::Mat& matLineMask, cv::Mat& defectLineMask, int nLineThickness)
{
    cv::Mat matLineMaskWidth = cv::Mat::zeros(matLineMask.size(), CV_8UC1);
    cv::Mat matLineMaskHeight = cv::Mat::zeros(matLineMask.size(), CV_8UC1);

    int nProfile = 0;
    float fRateLine = 0.08;


    int* nRowProfile = (int*)calloc(matLineMask.rows, sizeof(int));
    int* nColProfile = (int*)calloc(matLineMask.cols, sizeof(int));

    int nBeforeLineIndex = 0;
    bool bLineBefore = 0;


    for (int y = 0; y < matLineMask.rows; y++)
    {
        nProfile = 0;
        for (int x = 0; x < matLineMask.cols; x++)
        {
            if (matLineMask.at<uchar>(y, x) > 0)
                nProfile++;
        }

        if (nProfile > (matLineMask.cols * fRateLine))
        {

            if (y == 0)
            {
                nRowProfile[y] = 1;
            }
            else
            {
                nRowProfile[y] = nRowProfile[y - 1] + 1;
            }
        }
    }


    bLineBefore = false;
    for (int y = 0; y < matLineMask.rows; y++)
    {
        if (y == 0)
        {
            if (nRowProfile[y] > 0)
                bLineBefore = true;
            else
                bLineBefore = false;
        }


        if (bLineBefore && y != 0)
        {
            if (nRowProfile[y] == 0)
            {
                if (nRowProfile[y - 1] > nLineThickness)
                {
                    for (int y1 = y - 1; nRowProfile[y1] > 0; y1--)
                    {
                        if (y1 < 0)
                            break;

                        nRowProfile[y1] = 0;
                    }
                }
            }
        }

        if (bLineBefore && y == (matLineMask.rows - 1))
        {
            if (nRowProfile[y] > nLineThickness)
            {
                for (int y1 = y - 1; nRowProfile[y1] > 0; y1--)
                {
                    if (y1 < 0)
                        break;

                    nRowProfile[y1] = 0;
                }
            }
        }

        if (nRowProfile[y] > 0)
        {
            bLineBefore = true;
        }
        else
        {
            bLineBefore = false;
        }
    }

    for (int y = 0; y < matLineMask.rows; y++)
    {
        if (nRowProfile[y] == 0)
        {
            for (int x = 0; x < matLineMask.cols; x++)
            {
                matLineMaskWidth.at<uchar>(y, x) = 0;
            }
        }
        else
        {
            for (int x = 0; x < matLineMask.cols; x++)
            {
                matLineMaskWidth.at<uchar>(y, x) = matLineMask.at<uchar>(y, x);
            }
        }
    }

    for (int x = 0; x < matLineMask.cols; x++)
    {
        nProfile = 0;
        for (int y = 0; y < matLineMask.rows; y++)
        {
            if (matLineMask.at<uchar>(y, x) > 0)
                nProfile++;
        }

        if (nProfile > (matLineMask.rows * fRateLine))
        {
            if (x == 0)
            {
                bLineBefore = true;
                nColProfile[x] = 1;
            }

            //	for(int y  = 0 ; y < matLineMask.rows; y ++)
            //	{
            //		matLineMaskHeight.at<uchar>(y, x) = matLineMask.at<uchar>(y, x);
            //	}

            if (x == 0)
            {
                nColProfile[x] = 1;
            }
            else {
                nColProfile[x] = nColProfile[x - 1] + 1;
            }
        }
        else {
            nColProfile[x] = 0;
        }
    }

    bLineBefore = false;
    for (int x = 0; x < matLineMask.cols; x++)
    {
        if (x == 0)
        {
            if (nColProfile[x] > 0)
                bLineBefore = true;
            else
                bLineBefore = false;
        }

        if (bLineBefore && x != 0)
        {

            if (nColProfile[x] == 0)
            {
                if (nColProfile[x - 1] > nLineThickness)
                {
                    for (int x1 = x - 1; nColProfile[x1] > 0; x1--)
                    {
                        if (x1 < 0)
                            break;

                        nColProfile[x1] = 0;
                    }
                }
            }
        }


        if (bLineBefore && x == (matLineMask.cols - 1))
        {
            if (nColProfile[x] > nLineThickness)
            {
                for (int x1 = x - 1; nColProfile[x1] > 0; x1--)
                {
                    if (x1 < 0)
                        break;

                    nColProfile[x1] = 0;
                }
            }
        }

        if (nColProfile[x] > 0)
        {
            bLineBefore = true;
        }
        else {
            bLineBefore = false;
        }
    }

    for (int x = 0; x < matLineMask.cols; x++)
    {
        if (nColProfile[x] == 0)
        {
            for (int y = 0; y < matLineMask.rows; y++)
            {
                matLineMaskHeight.at<uchar>(y, x) = 0;
            }
        }
        else
        {
            for (int y = 0; y < matLineMask.rows; y++)
            {
                matLineMaskHeight.at<uchar>(y, x) = matLineMask.at<uchar>(y, x);
            }
        }
    }

    cv::add(matLineMaskHeight, matLineMaskWidth, defectLineMask);
    //cv::subtract(matLineMaskHeight, matLineMaskWidth, defectLineMask);

    //std::vector<std::vector<cv::Point>> contours;
    //cv::findContours(matLineMaskWidth, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    //std::vector<cv::Vec4i> lines;
    //for (const auto& contour : contours) {

    //    int area = cv::contourArea(contour);
    //    if (area < 200) continue;

    //    cv::Vec4i line;
    //    cv::fitLine(contour, line, cv::DIST_L2, 0, 0.01, 0.01);
    //    lines.push_back(line);
    //}
    //cv::Mat drawLines = cv::Mat::zeros(matLineMask.size(), CV_8UC3);
    //cv::Vec4i maxLine_x;
    //float Value = 0.0f;
    //for (const auto& line : lines) {
    //    float maxValue = std::max(line[1], line[3]);
    //    if (maxValue > Value) {
    //        Value = maxValue;
    //        maxLine_x = line;
    //    }
    //    //cv::Point startPoint(line[2] - line[0] * 1000, line[3] - line[1] * 1000);
    //    //cv::Point endPoint(line[2] + line[0] * 1000, line[3] + line[1] * 1000);
    //    //cv::line(drawLines, startPoint, endPoint, cv::Scalar(0, 0, 255), 2);
    //}
    //cv::Point startPoint(maxLine_x[2] - maxLine_x[0] * 1000, maxLine_x[3] - maxLine_x[1] * 1000);
    //cv::Point endPoint(maxLine_x[2] + maxLine_x[0] * 1000, maxLine_x[3] + maxLine_x[1] * 1000);
    //cv::line(drawLines, startPoint, endPoint, cv::Scalar(0, 0, 255), 2);

    //lines.clear();
    //cv::findContours(matLineMaskHeight, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    //for (const auto& contour : contours) {

    //    int area = cv::contourArea(contour);
    //    if (area < 200) continue;

    //    cv::Vec4i line;
    //    cv::fitLine(contour, line, cv::DIST_L2, 0, 0.01, 0.01);
    //    lines.push_back(line);
    //}
    ////cv::Mat drawLines = cv::Mat::zeros(matLineMask.size(), CV_8UC3);
    //cv::Vec4i maxLine_y;
    //Value = 0.0f;
    //for (const auto& line : lines) {
    //    float maxValue = std::max(line[0], line[2]);
    //    if (maxValue > Value) {
    //        Value = maxValue;
    //        maxLine_y = line;
    //    }
    //    //cv::Point startPoint(line[2] - line[0] * 1000, line[3] - line[1] * 1000);
    //    //cv::Point endPoint(line[2] + line[0] * 1000, line[3] + line[1] * 1000);
    //    //cv::line(drawLines, startPoint, endPoint, cv::Scalar(0, 0, 255), 2);
    //}
    //startPoint = cv::Point(maxLine_y[2] - maxLine_y[0] * 1000, maxLine_y[3] - maxLine_y[1] * 1000);
    //endPoint = cv::Point(maxLine_y[2] + maxLine_y[0] * 1000, maxLine_y[3] + maxLine_y[1] * 1000);
    //cv::line(drawLines, startPoint, endPoint, cv::Scalar(0, 0, 255), 2);

    //cv::Point2f crossPoint= getCrossPoint(maxLine_x, maxLine_y);
    //cv::cvtColor(matLineMask, matLineMask, cv::COLOR_GRAY2BGR);
    //cv::circle(matLineMask, crossPoint, 2, cv::Scalar(147, 200, 188));


    //cv::dilate(defectLineMask, defectLineMask, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5,5)) );
    //cv::erode(defectLineMask, defectLineMask, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5)) );


    free(nRowProfile);
    free(nColProfile);

    return true;

}

bool PatrolEdge::BG_Subtract(cv::Mat& matGrayChanels, cv::Mat& matBGSizeDark, cv::Mat& matBGSizeBright, int nStepX, int nStepY, int nStepRows, int nStepCols, int bgThreshold)
{
    START_TIMER;
    int nTh = 5;//15


    matBGSizeDark = cv::Mat::zeros(matGrayChanels.size(), CV_8UC1);
    matBGSizeBright = cv::Mat::zeros(matGrayChanels.size(), CV_8UC1);


    cv::Mat reSize = cv::Mat::zeros(nStepRows, nStepCols, CV_8UC1);

    for (int j = 0; j < nStepRows; j++)
    {
        for (int i = 0; i < nStepCols; i++)
        {
            int nSum = 0;
            float fAverage = 0.0;
            int nSumCount = 0;

            for (int y = (j * nStepY); y < (j * nStepY + nStepY); y++)
            {
                for (int x = (i * nStepX); x <= (i * nStepX + nStepX); x++)
                {
                    if (y >= matGrayChanels.rows || x >= matGrayChanels.cols)
                        continue;
                    nSumCount++;
                    nSum += matGrayChanels.at<uchar>(y, x);
                }
            }

            if (nSumCount > 0)
                fAverage = nSum / nSumCount;

            reSize.at<uchar>(j, i) = (uchar)fAverage;
        }
    }

    //cv::medianBlur(reSize, reSize, 9);

    cv::Mat matBGSize = cv::Mat::zeros(matGrayChanels.rows, matGrayChanels.cols, CV_8UC1);

    for (int y = 0; y < matGrayChanels.rows; y++)
    {
        for (int x = 0; x < matGrayChanels.cols; x++)
        {
            if (x / nStepX >= nStepCols && y / nStepY >= nStepRows)
            {
                matBGSize.at<uchar>(y, x) = reSize.at<uchar>(int(y / nStepY) - 1, int(x / nStepX) - 1);
            }
            else if (x / nStepX >= nStepCols)
            {
                matBGSize.at<uchar>(y, x) = reSize.at<uchar>(int(y / nStepY), int(x / nStepX) - 1);
            }
            else if (y / nStepY >= nStepRows)
            {
                matBGSize.at<uchar>(y, x) = reSize.at<uchar>(int(y / nStepY) - 1, int(x / nStepX));
            }
            else {
                matBGSize.at<uchar>(y, x) = reSize.at<uchar>(int(y / nStepY), int(x / nStepX));
            }
        }
    }

    cv::subtract(matBGSize, matGrayChanels, matBGSizeDark);
    cv::subtract(matGrayChanels, matBGSize, matBGSizeBright);

    cv::threshold(matBGSizeDark, matBGSizeDark, nTh, 255.0, cv::THRESH_BINARY);
    cv::threshold(matBGSizeBright, matBGSizeBright, nTh, 255.0, cv::THRESH_BINARY);

    for (int y = 0; y < matGrayChanels.rows; y++)
    {
        for (int x = 0; x < matGrayChanels.cols; x++)
        {
            if (matBGSizeDark.at<uchar>(y, x) > 0)
            {
                matGrayChanels.at<uchar>(y, x) = matBGSize.at<uchar>(y, x);
            }

            if (matBGSizeBright.at<uchar>(y, x) > 0)
            {
                matGrayChanels.at<uchar>(y, x) = matBGSize.at<uchar>(y, x);
            }
        }
    }

    if (!reSize.empty())    reSize.release();
    if (!matBGSize.empty()) matBGSize.release();
    END_TIMER;
    return true;
}

bool PatrolEdge::Estimation_X(cv::Mat matSrcBuf, cv::Mat& matDstBuf, int nDimensionX, int nStepX, float fThBGOffset)
{
    if (matSrcBuf.empty())			return false;

    if (matSrcBuf.channels() != 1)	return false;

    if (!matDstBuf.empty())
        matDstBuf.release();

    if (nStepX <= 0)				return false;

    matDstBuf = cv::Mat::zeros(matSrcBuf.size(), matSrcBuf.type());

    int nStepCols = matSrcBuf.cols / nStepX;
    int nHalfCols = matSrcBuf.cols / 2;

    cv::Mat M = cv::Mat_<double>(nStepCols, nDimensionX + 1);
    cv::Mat I = cv::Mat_<double>(nStepCols, 1);
    cv::Mat q;

    double x, quad, dTemp;
    int i, j, k, m;


    cv::Scalar mean = cv::mean(matSrcBuf);
    int nMinGV = (int)(mean[0] * fThBGOffset);

    for (i = 0; i < matSrcBuf.rows; i++)
    {
        for (j = 0; j < nStepCols; j++)

        {
            x = (j * nStepX - nHalfCols) / double(matSrcBuf.cols);

            M.at<double>(j, 0) = 1.0;
            dTemp = 1.0;
            for (k = 1; k <= nDimensionX; k++)
            {
                dTemp *= x;
                M.at<double>(j, k) = dTemp;
            }

            //I.at<double>(j, 0) = matSrcBuf.at<uchar>(i, j*nStepX);
            m = matSrcBuf.at<uchar>(i, j * nStepX);
            I.at<double>(j, 0) = (m < nMinGV) ? nMinGV : m;
        }

        cv::SVD s(M);
        s.backSubst(I, q);

        for (j = 0; j < matDstBuf.cols; j++)
        {
            x = (j - nHalfCols) / double(matSrcBuf.cols);

            quad = q.at<double>(0, 0);
            dTemp = 1.0;
            for (k = 1; k <= nDimensionX; k++)
            {
                dTemp *= x;
                quad += (q.at<double>(k, 0) * dTemp);
            }

            matDstBuf.at<uchar>(i, j) = cv::saturate_cast<uchar>(quad);
        }
    }

    M.release();
    I.release();
    q.release();

    return true;
}

bool PatrolEdge::Estimation_Y(cv::Mat matSrcBuf, cv::Mat& matDstBuf, int nDimensionY, int nStepY, float fThBGOffset)
{
    if (matSrcBuf.empty())			return false;

    if (matSrcBuf.channels() != 1)	return false;

    if (!matDstBuf.empty())
        matDstBuf.release();

    if (nStepY <= 0)				return false;

    matDstBuf = cv::Mat::zeros(matSrcBuf.size(), matSrcBuf.type());

    int nStepRows = matSrcBuf.rows / nStepY;
    int nHalfRows = matSrcBuf.rows / 2;

    cv::Mat M = cv::Mat_<double>(nStepRows, nDimensionY + 1);
    cv::Mat I = cv::Mat_<double>(nStepRows, 1);
    cv::Mat q;

    double y, quad, dTemp;
    int i, j, k, m;


    cv::Scalar mean = cv::mean(matSrcBuf);
    int nMinGV = (int)(mean[0] * fThBGOffset);

    for (j = 0; j < matSrcBuf.cols; j++)
    {
        for (i = 0; i < nStepRows; i++)
        {
            y = (i * nStepY - nHalfRows) / double(matSrcBuf.rows);

            M.at<double>(i, 0) = 1.0;
            dTemp = 1.0;
            for (k = 1; k <= nDimensionY; k++)
            {
                dTemp *= y;
                M.at<double>(i, k) = dTemp;
            }

            //I.at<double>(i, 0) = matSrcBuf.at<uchar>(i*nStepY, j);
            m = matSrcBuf.at<uchar>(i * nStepY, j);
            I.at<double>(i, 0) = (m < nMinGV) ? nMinGV : m;
        }

        cv::SVD s(M);
        s.backSubst(I, q);

        for (i = 0; i < matSrcBuf.rows; i++)
        {
            y = (i - nHalfRows) / double(matSrcBuf.rows);

            quad = q.at<double>(0, 0);
            dTemp = 1.0;
            for (k = 1; k <= nDimensionY; k++)
            {
                dTemp *= y;
                quad += (q.at<double>(k, 0) * dTemp);
            }

            matDstBuf.at<uchar>(i, j) = cv::saturate_cast<uchar>(quad);
        }
    }

    M.release();
    I.release();
    q.release();

    return true;
}

bool PatrolEdge::TwoImg_Average(cv::Mat matSrc1Buf, cv::Mat matSrc2Buf, cv::Mat& matDstBuf)
{
    if (matSrc1Buf.empty())			return false;
    if (matSrc2Buf.empty())			return false;

    if (matSrc1Buf.channels() != 1)	return false;
    if (matSrc2Buf.channels() != 1)	return false;

    if (matSrc1Buf.rows != matSrc2Buf.rows ||
        matSrc1Buf.cols != matSrc2Buf.cols)	
        return false;

    if (!matDstBuf.empty())
        matDstBuf.release();

    matDstBuf = cv::Mat::zeros(matSrc1Buf.rows, matSrc1Buf.cols, matSrc1Buf.type());

    for (int y = 0; y < matSrc1Buf.rows; y++)
    {
        BYTE* ptr1 = (BYTE*)matSrc1Buf.ptr(y);
        BYTE* ptr2 = (BYTE*)matSrc2Buf.ptr(y);
        BYTE* ptr3 = (BYTE*)matDstBuf.ptr(y);

        for (int x = 0; x < matSrc1Buf.cols; x++, ptr1++, ptr2++, ptr3++)
        {
            *ptr3 = (BYTE)abs((*ptr1 + *ptr2) / 2.0);
        }
    }

    return true;
}

bool PatrolEdge::Separation_ActiveAre(cv::Mat& matSrcBuf, cv::Mat& matResROIBuf, int nThreshold, int nEdgePiexl)
{
    int nLeftPiexl = 0;
    int nEdgeThreshold = 1;
    cv::Mat matAtiveBG = matSrcBuf.clone();
    cv::Mat matAtiveEdge = matAtiveBG.clone();
    if (nEdgePiexl > 0 && nEdgeThreshold > 0)
        cv::morphologyEx(matAtiveBG, matAtiveBG, cv::MORPH_DILATE, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(nEdgePiexl, nEdgePiexl)));
    cv::threshold(matAtiveBG, matAtiveBG, 20, 255.0, cv::THRESH_BINARY_INV);

    for (int y = 0; y < matResROIBuf.rows; y++)
    {
        for (int x = 0; x < matResROIBuf.cols; x++)
        {
            if (y > (matResROIBuf.rows - nLeftPiexl))
            {
                matAtiveBG.at<uchar>(y, x) = 0;

                if (matSrcBuf.at<uchar>(y, x) >= nThreshold)
                    matResROIBuf.at<uchar>(y, x) = 255;
                else
                    matResROIBuf.at<uchar>(y, x) = 0;
            }

            if (matAtiveBG.at<uchar>(y, x) > 0)
                matAtiveEdge.at<uchar>(y, x) = 0;
            else
                matAtiveEdge.at<uchar>(y, x) = 255;

            if (x < nLeftPiexl)
                matAtiveEdge.at<uchar>(y, x) = 0;
        }
    }

    for (int y = 0; y < matResROIBuf.rows; y++)
    {
        for (int x = 0; x < matResROIBuf.cols; x++)
        {
            if (matAtiveBG.at<uchar>(y, x) > 0 || (matAtiveEdge.at<uchar>(y, x) > 0 && nEdgeThreshold > 0))
            {
                if (matSrcBuf.at<uchar>(y, x) >= nThreshold)
                    matResROIBuf.at<uchar>(y, x) = 255;
                else
                    matResROIBuf.at<uchar>(y, x) = 0;
            }
        }
    }

    return true;
}

std::vector<cv::Point> PatrolEdge::findImageBoundaryPoints(const cv::Mat& image) {
    cv::Mat grayImage;
    //cv::cvtColor(image, grayImage, cv::COLOR_BGR2GRAY);
    grayImage = image;
    cv::Mat horizontalProjection;
    cv::reduce(grayImage, horizontalProjection, 1, cv::REDUCE_SUM, CV_32S);

    cv::Mat verticalProjection;
    cv::reduce(grayImage, verticalProjection, 0, cv::REDUCE_SUM, CV_32S);

    std::vector<cv::Point> boundaryPoints;

    for (int row = 0; row < horizontalProjection.rows; ++row) {
        if (horizontalProjection.at<int>(row) > 0) {
            boundaryPoints.push_back(cv::Point(0, row));
            boundaryPoints.push_back(cv::Point(grayImage.cols - 1, row));
        }
    }

    for (int col = 0; col < verticalProjection.cols; ++col) {
        if (verticalProjection.at<int>(col) > 0) {
            boundaryPoints.push_back(cv::Point(col, 0));
            boundaryPoints.push_back(cv::Point(col, grayImage.rows - 1));
        }
    }

    return boundaryPoints;
}

bool PatrolEdge::DoBlobCalculate(cv::Mat ThresholdBuffer, cv::Mat matBKBuf, cv::Mat GrayBuffer, int nMaxDefectCount, std::vector<stBLOB_FEATURE>&	m_BlobResult)
{
    START_TIMER;

    if (m_BlobResult.size() != 0)
    {
        for (int i = 0; i < m_BlobResult.size(); i++)
        {
            std::vector<cv::Point>().swap(m_BlobResult[i].ptIndexs);
            std::vector <cv::Point>().swap(m_BlobResult[i].ptContours);
        }
        std::vector<stBLOB_FEATURE>().swap(m_BlobResult);
    }

    if (ThresholdBuffer.empty())			return false;


    if (ThresholdBuffer.channels() != 1)	return false;



    cv::cvtColor(GrayBuffer, GrayBuffer, cv::COLOR_BGR2GRAY);

    bool bGrayEmpty = false;
    if (GrayBuffer.empty() || GrayBuffer.channels() != 1)
    {
        GrayBuffer = ThresholdBuffer.clone();
        bGrayEmpty = true;
    }


    LOGI("{}  Start.",__FUNCTION__);

    cv::Mat matLabel, matStats, matCentroid;


    matLabel = cv::Mat(ThresholdBuffer.size(), CV_32SC1);

    // LOGI("{}  Mat Create.", __FUNCTION__); 

    __int64 nTotalLabel = 0;

    if (ThresholdBuffer.type() == CV_8U)
    {
        nTotalLabel = cv::connectedComponentsWithStats(ThresholdBuffer, matLabel, matStats, matCentroid, 8, CV_32S) - 1;
    }
    else
    {
        cv::Mat matSrc8bit = cv::Mat(ThresholdBuffer.size(), CV_8UC1);
        ThresholdBuffer.convertTo(matSrc8bit, CV_8UC1, 1. / 16.);

        nTotalLabel = cv::connectedComponentsWithStats(matSrc8bit, matLabel, matStats, matCentroid, 8, CV_32S) - 1;

        matSrc8bit.release();
    }
    LOGI("{}  connectedComponents.", __FUNCTION__);


    if (nTotalLabel < 0)
    {

        if (bGrayEmpty)			GrayBuffer.release();
        if (!matLabel.empty())		matLabel.release();
        if (!matStats.empty())		matStats.release();
        if (!matCentroid.empty())	matCentroid.release();

        return false;
    }


    if (nTotalLabel >= nMaxDefectCount)
        nTotalLabel = nMaxDefectCount - 1;
    DoFeatureBasicColor_8bit(matLabel, matStats, matCentroid, GrayBuffer, matBKBuf, (int)nTotalLabel, m_BlobResult);

    LOGI("{}  FeatureExtraction.", __FUNCTION__);


    if (bGrayEmpty)			GrayBuffer.release();
    if (!matLabel.empty())		matLabel.release();
    if (!matStats.empty())		matStats.release();
    if (!matCentroid.empty())	matCentroid.release();
    LOGI("{}  Release.", __FUNCTION__);
    END_TIMER;
    return true;
}

bool PatrolEdge::DoFeatureBasicColor_8bit(cv::Mat& matLabel, cv::Mat& matStats, cv::Mat& matCentroid, cv::Mat& GrayBuffer, cv::Mat matBKBuf, int nTotalLabel, std::vector<stBLOB_FEATURE>& m_BlobResult)
{

    if (nTotalLabel <= 0)	return true;

    float fVal = 4.f * PI;

    m_BlobResult.resize(nTotalLabel);

#pragma omp parallel for
    for (int idx = 1; idx <= nTotalLabel; idx++)
    {
        int nLabelArea = 0;
        nLabelArea = matStats.at<int>(idx, cv::CC_STAT_AREA);
        if (nLabelArea < 20)
            continue;

        int nBlobNum = idx - 1;

        m_BlobResult.at(nBlobNum).rectBox.x = matStats.at<int>(idx, cv::CC_STAT_LEFT);
        m_BlobResult.at(nBlobNum).rectBox.y = matStats.at<int>(idx, cv::CC_STAT_TOP);
        m_BlobResult.at(nBlobNum).rectBox.width = matStats.at<int>(idx, cv::CC_STAT_WIDTH);
        m_BlobResult.at(nBlobNum).rectBox.height = matStats.at<int>(idx, cv::CC_STAT_HEIGHT);

        ////////////////////////////////////////////////////////////////		
//查找距离edge最小的位置
        int cx, cy;
        cx = matStats.at<int>(idx, cv::CC_STAT_LEFT) + (int)(matStats.at<int>(idx, cv::CC_STAT_WIDTH) / 2);
        cy = matStats.at<int>(idx, cv::CC_STAT_TOP) + (int)(matStats.at<int>(idx, cv::CC_STAT_HEIGHT) / 2);

        int nDistanceFromEdge = 0;
        double distanceA = -1, distanceB = -1;
        //matBKBuf.at<int>(CC_STAT_LEFT);
        float lineA_00 = matBKBuf.at<float>(0, 0);
        float lineA_01 = matBKBuf.at<float>(0, 1);
        float lineB_10 = matBKBuf.at<float>(1, 0);
        float lineB_11 = matBKBuf.at<float>(1, 1);


        //distanceA = abs(cx * lineA_00 - cy + lineA_01) / sqrt(lineA_00 * lineA_00 + 1.);
        //distanceB = abs(cx * lineB_10 - cy + lineB_11) / sqrt(lineB_10 * lineB_10 + 1.);
        distanceA = calculateMinDistanceToLines(m_BlobResult.at(nBlobNum).rectBox, lineA_00, lineA_01);
        distanceB = calculateMinDistanceToLines(m_BlobResult.at(nBlobNum).rectBox, lineB_10, lineB_11);
        if      (distanceA > distanceB && distanceB > 0 && distanceA > 0) nDistanceFromEdge = distanceB;
        else if (distanceA < distanceB && distanceB > 0 && distanceA > 0) nDistanceFromEdge = distanceA;


        m_BlobResult.at(nBlobNum).nDistanceFromEdge = nDistanceFromEdge;
        ////////////////////////////////////////////////////////////////

        int nOffsetROI = 0;//25

        int nSX = m_BlobResult.at(nBlobNum).rectBox.x - nOffsetROI;
        int nSY = m_BlobResult.at(nBlobNum).rectBox.y - nOffsetROI;
        int nEX = m_BlobResult.at(nBlobNum).rectBox.x + m_BlobResult.at(nBlobNum).rectBox.width + nOffsetROI;
        int nEY = m_BlobResult.at(nBlobNum).rectBox.y + m_BlobResult.at(nBlobNum).rectBox.height + nOffsetROI;

        if (nSX < 0)	nSX = 0;
        if (nSY < 0)	nSY = 0;
        if (nEX >= GrayBuffer.cols)	nEX = GrayBuffer.cols - 1;
        if (nEY >= GrayBuffer.rows)	nEY = GrayBuffer.rows - 1;

        cv::Rect rectTemp(nSX, nSY, nEX - nSX + 1, nEY - nSY + 1);

        m_BlobResult.at(nBlobNum).FeaRectROI = rectTemp;

        __int64 nCount_in = 0;
        __int64 nCount_out = 0;
        __int64 nSum_in = 0;	
        __int64 nSum_out = 0;	

        cv::Mat matTmp_src = GrayBuffer(rectTemp);		
        cv::Mat matTmp_label = matLabel(rectTemp);			
        cv::Mat matTemp = cv::Mat(rectTemp.height, rectTemp.width, CV_8UC1);

        for (int y = 0; y < rectTemp.height; y++)
        {
            int* ptrLabel = (int*)matTmp_label.ptr(y);
            uchar* ptrGray = (uchar*)matTmp_src.ptr(y);
            uchar* ptrTemp = (uchar*)matTemp.ptr(y);

            for (int x = 0; x < rectTemp.width; x++, ptrLabel++, ptrGray++, ptrTemp++)
            {
                if (*ptrLabel == idx)
                {
                    nSum_in += *ptrGray;
                    nCount_in++;

                    m_BlobResult.at(nBlobNum).ptIndexs.push_back(cv::Point(nSX + x, nSY + y));

                    *ptrTemp = (uchar)255;

                    m_BlobResult.at(nBlobNum).nHist[*ptrGray]++;
                }
                else
                {
                    if (*ptrLabel == 0)
                    {
                        nSum_out += *ptrGray;
                        nCount_out++;
                    }
                }
            }
        }
        m_BlobResult.at(nBlobNum).nSumGV = nSum_in;
        m_BlobResult.at(nBlobNum).nArea = nCount_in;
        m_BlobResult.at(nBlobNum).nBoxArea = m_BlobResult.at(nBlobNum).rectBox.width * m_BlobResult.at(nBlobNum).rectBox.height;
        m_BlobResult.at(nBlobNum).fBoxRatio = m_BlobResult.at(nBlobNum).nArea / (float)m_BlobResult.at(nBlobNum).nBoxArea;
        m_BlobResult.at(nBlobNum).fElongation = m_BlobResult.at(nBlobNum).rectBox.width / (float)m_BlobResult.at(nBlobNum).rectBox.height;

        cv::Scalar m, s;
        cv::meanStdDev(matTmp_src, m, s, matTemp);
        m_BlobResult.at(nBlobNum).fStdDev = float(s[0]);

        std::vector<std::vector<cv::Point>>	ptContours;
        std::vector<std::vector<cv::Point>>().swap(ptContours);
        cv::findContours(matTemp, ptContours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
        if (ptContours.size() != 0)
        {
            for (int m = 0; m < ptContours.size(); m++)
            {
                for (int k = 0; k < ptContours.at(m).size(); k++)
                    m_BlobResult.at(nBlobNum).ptContours.push_back(cv::Point(ptContours.at(m)[k].x + nSX, ptContours.at(m)[k].y + nSY));
            }
        }
        else
        {
            m_BlobResult.at(nBlobNum).ptContours.resize((int)m_BlobResult.at(nBlobNum).ptIndexs.size());
            std::copy(m_BlobResult.at(nBlobNum).ptIndexs.begin(), m_BlobResult.at(nBlobNum).ptIndexs.end(), m_BlobResult.at(nBlobNum).ptContours.begin());
        }
        m_BlobResult.at(nBlobNum).fPerimeter = float(cv::arcLength(m_BlobResult.at(nBlobNum).ptContours, true));
        std::vector<std::vector<cv::Point>>().swap(ptContours);

        m_BlobResult.at(nBlobNum).fRoundness = (fVal * m_BlobResult.at(nBlobNum).nArea)
            / (m_BlobResult.at(nBlobNum).fPerimeter * m_BlobResult.at(nBlobNum).fPerimeter);
        m_BlobResult.at(nBlobNum).fCompactness = (m_BlobResult.at(nBlobNum).fPerimeter * m_BlobResult.at(nBlobNum).fPerimeter)
            / (fVal * float(m_BlobResult.at(nBlobNum).nArea));
        m_BlobResult.at(nBlobNum).fMeanGV = nSum_in / (float)nCount_in;
        m_BlobResult.at(nBlobNum).fBKGV = nSum_out / (float)nCount_out;
        m_BlobResult.at(nBlobNum).fDiffGV = m_BlobResult.at(nBlobNum).fBKGV - m_BlobResult.at(nBlobNum).fMeanGV;

        double valMin, valMax, ta, tb;
        cv::minMaxLoc(matTmp_src, &valMin, &valMax, 0, 0, matTemp);
        cv::minMaxIdx(matTmp_src, &ta, &tb);
        m_BlobResult.at(nBlobNum).nMinGV = (long)valMin;
        m_BlobResult.at(nBlobNum).nMaxGV = (long)valMax;

        m_BlobResult.at(nBlobNum).nMinGVRatio = m_BlobResult.at(nBlobNum).nMinGV / m_BlobResult.at(nBlobNum).fBKGV;
        m_BlobResult.at(nBlobNum).nMaxGVRatio = m_BlobResult.at(nBlobNum).nMaxGV / m_BlobResult.at(nBlobNum).fBKGV;
        m_BlobResult.at(nBlobNum).fDiffGVRatio = m_BlobResult.at(nBlobNum).fMeanGV / m_BlobResult.at(nBlobNum).fBKGV;
        m_BlobResult.at(nBlobNum).ptCenter.x = (int)matCentroid.at<double>(idx, 0);
        m_BlobResult.at(nBlobNum).ptCenter.y = (int)matCentroid.at<double>(idx, 1);
        if (m_BlobResult.at(nBlobNum).fDiffGV == 0.0)
        {
            if (m_BlobResult.at(nBlobNum).fBKGV == 0)
            {
                m_BlobResult.at(nBlobNum).fSEMU = 1.0
                    / (1.97f / (cv::pow((float)m_BlobResult.at(nBlobNum).nArea, 0.33f) + 0.72f));
            }
            else
            {
                m_BlobResult.at(nBlobNum).fSEMU = (0.000001 / m_BlobResult.at(nBlobNum).fBKGV)
                    / (1.97 / (cv::pow((float)m_BlobResult.at(nBlobNum).nArea, 0.33f) + 0.72f));
            }
        }
        else
        {
            if (m_BlobResult.at(nBlobNum).fBKGV == 0)
            {
                m_BlobResult.at(nBlobNum).fSEMU = (fabs(m_BlobResult.at(nBlobNum).fMeanGV - m_BlobResult.at(nBlobNum).fBKGV) / 0.000001)
                    / (1.97 / (cv::pow((float)m_BlobResult.at(nBlobNum).nArea, 0.33f) + 0.72f));
            }
            else
            {
                m_BlobResult.at(nBlobNum).fSEMU = (fabs(m_BlobResult.at(nBlobNum).fMeanGV - m_BlobResult.at(nBlobNum).fBKGV) / m_BlobResult.at(nBlobNum).fBKGV)
                    / (1.97 / (cv::pow((float)m_BlobResult.at(nBlobNum).nArea, 0.33f) + 0.72f));
            }
        }
        cv::RotatedRect BoundingBox = cv::minAreaRect(m_BlobResult.at(nBlobNum).ptIndexs);
        m_BlobResult.at(nBlobNum).BoxSize = BoundingBox.size;
        m_BlobResult.at(nBlobNum).fAngle = BoundingBox.angle;
        if (BoundingBox.size.width > BoundingBox.size.height)
        {
            m_BlobResult.at(nBlobNum).fMinorAxis = BoundingBox.size.width;
            m_BlobResult.at(nBlobNum).fMajorAxis = BoundingBox.size.height;
        }
        else
        {
            m_BlobResult.at(nBlobNum).fMinorAxis = BoundingBox.size.height;
            m_BlobResult.at(nBlobNum).fMajorAxis = BoundingBox.size.width;
        }
        m_BlobResult.at(nBlobNum).fMinBoxArea = m_BlobResult.at(nBlobNum).fMinorAxis * m_BlobResult.at(nBlobNum).fMajorAxis;
        if (m_BlobResult.at(nBlobNum).fMajorAxis > 0)
            m_BlobResult.at(nBlobNum).fAxisRatio = m_BlobResult.at(nBlobNum).fMinorAxis / m_BlobResult.at(nBlobNum).fMajorAxis;
        else
            m_BlobResult.at(nBlobNum).fAxisRatio = 0.f;
        m_BlobResult.at(nBlobNum).fMinBoxRatio = m_BlobResult.at(nBlobNum).fMinBoxArea / (float)m_BlobResult.at(nBlobNum).nArea;
        m_BlobResult.at(nBlobNum).fDefectMeanGV = (tb - ta);
    }
   return true;
}

bool PatrolEdge::DrawBlob(cv::Mat& DrawBuffer, cv::Scalar DrawColor, long nOption, bool bSelect, std::vector<stBLOB_FEATURE>& m_BlobResult, float fFontSize)
{
    START_TIMER;
    if (DrawBuffer.empty())		return false;



    if (m_BlobResult.size() == 0)	return true;

    if (nOption == 0)				return true;

    int i, j;

#pragma omp parallel for
    for (i = 0; i < m_BlobResult.size(); i++)
    {

        if (!m_BlobResult[i].bFiltering && bSelect)	continue;

        if (nOption & BLOB_DRAW_ROTATED_BOX)
        {
            cv::RotatedRect rRect = cv::RotatedRect(m_BlobResult[i].ptCenter, m_BlobResult[i].BoxSize, m_BlobResult[i].fAngle);

            cv::Point2f vertices[4];
            rRect.points(vertices);

            cv::line(DrawBuffer, vertices[0], vertices[1], DrawColor);
            cv::line(DrawBuffer, vertices[1], vertices[2], DrawColor);
            cv::line(DrawBuffer, vertices[2], vertices[3], DrawColor);
            cv::line(DrawBuffer, vertices[3], vertices[0], DrawColor);
        }

        if (nOption & BLOB_DRAW_BOUNDING_BOX)
        {
            cv::Rect rect(m_BlobResult[i].rectBox);
            rect.x -= 5;
            rect.y -= 5;
            rect.width += 10;
            rect.height += 10;

            cv::rectangle(DrawBuffer, rect, DrawColor);
        }

        if (nOption & BLOB_DRAW_BLOBS)
        {
            if (DrawBuffer.channels() == 1)
            {
                int nGrayColor = (int)(DrawColor.val[0] + DrawColor.val[1] + DrawColor.val[2]) / 3;

                for (j = 0; j < m_BlobResult[i].ptIndexs.size(); j++)
                {
                    DrawBuffer.at<uchar>(m_BlobResult[i].ptIndexs[j].y, m_BlobResult[i].ptIndexs[j].x) = nGrayColor;
                }
            }
            else
            {
                for (j = 0; j < m_BlobResult[i].ptIndexs.size(); j++)
                {
                    DrawBuffer.at<cv::Vec3b>(m_BlobResult[i].ptIndexs[j].y, m_BlobResult[i].ptIndexs[j].x)[0] = (int)DrawColor.val[0];
                    DrawBuffer.at<cv::Vec3b>(m_BlobResult[i].ptIndexs[j].y, m_BlobResult[i].ptIndexs[j].x)[1] = (int)DrawColor.val[1];
                    DrawBuffer.at<cv::Vec3b>(m_BlobResult[i].ptIndexs[j].y, m_BlobResult[i].ptIndexs[j].x)[2] = (int)DrawColor.val[2];
                }
            }
        }

        if (nOption & BLOB_DRAW_BLOBS_CONTOUR)
        {
            if (DrawBuffer.channels() == 1)
            {
                int nGrayColor = (int)(DrawColor.val[0] + DrawColor.val[1] + DrawColor.val[2]) / 3;

                for (j = 0; j < m_BlobResult[i].ptContours.size(); j++)
                {
                    DrawBuffer.at<uchar>(m_BlobResult[i].ptContours[j].y, m_BlobResult[i].ptContours[j].x) = nGrayColor;
                }
            }
            else
            {
                for (j = 0; j < m_BlobResult[i].ptContours.size(); j++)
                {
                    DrawBuffer.at<cv::Vec3b>(m_BlobResult[i].ptContours[j].y, m_BlobResult[i].ptContours[j].x)[0] = (int)DrawColor.val[0];
                    DrawBuffer.at<cv::Vec3b>(m_BlobResult[i].ptContours[j].y, m_BlobResult[i].ptContours[j].x)[1] = (int)DrawColor.val[1];
                    DrawBuffer.at<cv::Vec3b>(m_BlobResult[i].ptContours[j].y, m_BlobResult[i].ptContours[j].x)[2] = (int)DrawColor.val[2];
                }
            }
        }
    }
    END_TIMER;
    return true;
}

/*函数功能：求两条直线交点*/
/*输入：两条Vec4i类型直线*/
/*返回：Point2f类型的点*/
cv::Point2f PatrolEdge::getCrossPoint(cv::Vec4i LineA, cv::Vec4i LineB)
{


    cv::Point2f crossPoint;

//根据输出的line，定义两条直线的输出分别为lineA、lineB
//定义直线方程：a1x+b1y+c1 = 0；a2x+b2y+c2 = 0;

    double a1 = 0, b1 = 0, c1 = 0;
    a1 = -LineA[1];
    b1 = LineA[0];
    c1 = -(LineA[2] * a1 + LineA[3] * b1);

    double a2 = 0, b2 = 0, c2 = 0;
    a2 = -LineB[1];
    b2 = LineB[0];
    c2 = -(LineB[2] * a2 + LineB[3] * b2);
    //交点:
    // x = (b1 * c2 - b2 * c1) / (a1 * b2 - a2 * b1)
    // y = (a2 * c1 - a1 * c2) / (a1 * b2 - a2 * b1)
    double A = (b1 * c2 - b2 * c1);
    double B = (a2 * c1 - a1 * c2);
    double C = (a1 * b2 - a2 * b1);

    crossPoint.x = (A / C);
    crossPoint.y = (B / C);



    return crossPoint;
}

long PatrolEdge::RobustFitLine(cv::Mat& matTempBuf, cv::Rect rectCell, long double& dA, long double& dB, int nMinSamples, double distThreshold, int nType, int nSamp)
{
    START_TIMER;
    int nW = matTempBuf.cols;
    int nH = matTempBuf.rows;


    if (rectCell.x < 0)	return E_ERROR_CODE_ALIGN_IMAGE_OVER;
    if (rectCell.x >= nW)	return E_ERROR_CODE_ALIGN_IMAGE_OVER;
    if (rectCell.y < 0)	return E_ERROR_CODE_ALIGN_IMAGE_OVER;
    if (rectCell.y >= nH)	return E_ERROR_CODE_ALIGN_IMAGE_OVER;

 
    if (rectCell.x + rectCell.width < 0)	return E_ERROR_CODE_ALIGN_IMAGE_OVER;
    if (rectCell.x + rectCell.width > nW)	return E_ERROR_CODE_ALIGN_IMAGE_OVER;
    if (rectCell.y + rectCell.height < 0)	return E_ERROR_CODE_ALIGN_IMAGE_OVER;
    if (rectCell.y + rectCell.height > nH)	return E_ERROR_CODE_ALIGN_IMAGE_OVER;


    //int nSrartX = rectCell.x + rectCell.width / 4;	//4
    //int nEndX = nSrartX + rectCell.width / 2;		

    //int nStartY = rectCell.y + rectCell.height / 4;	
    //int nEndY = nStartY + rectCell.height / 2;	    


    int nSrartX = rectCell.x;	//4
    int nEndX = nSrartX + rectCell.width/4;

    int nStartY = rectCell.y;
    int nEndY = nStartY + rectCell.height/4;

    int x, y;

    std::vector<cv::Point2i>	ptSrcIndexs;
    std::vector<cv::Point2i>().swap(ptSrcIndexs);
    //cv::Mat& drawImg = cv::Mat(matTempBuf.size(), CV_8UC3);
    //cv::rectangle(drawImg, cv::Rect(nSrartX, nStartY, nEndX - nSrartX, nEndY - nStartY), cv::Scalar(115));
    switch (nType)
    {
    case E_ALIGN_TYPE_LEFT:
    {
        for (y = nStartY; y <= nEndY; y += nSamp)
        {
            for (x = rectCell.x; x <= nEndX; x++)
            {
                //cv::circle(drawImg, cv::Point2i(x, y), 2, cv::Scalar(255));
                if (matTempBuf.at<uchar>(y, x))
                {
                    ptSrcIndexs.push_back(cv::Point2i(x, y));
                    
                    break;
                }
            }
        }
    }
    break;

    case E_ALIGN_TYPE_TOP:
    {
        for (x = nSrartX; x <= nEndX; x += nSamp)
        {
            for (y = rectCell.y; y <= nEndY; y++)
            {
                //cv::circle(drawImg, cv::Point2i(x, y), 2, cv::Scalar(255));
                if (matTempBuf.at<uchar>(y, x))
                {
                    ptSrcIndexs.push_back(cv::Point2i(x, y));

                    break;
                }
            }
        }
    }
    break;

    case E_ALIGN_TYPE_RIGHT:
    {
        for (y = nStartY; y <= nEndY; y += nSamp)
        {
            for (x = rectCell.x + rectCell.width; x >= nSrartX; x--)
            {
                //cv::circle(drawImg, cv::Point2i(x, y), 2, cv::Scalar(255));
                if (matTempBuf.at<uchar>(y, x))
                {

                    ptSrcIndexs.push_back(cv::Point2i(x, y));

                    break;
                }
            }
        }
    }
    break;

    case E_ALIGN_TYPE_BOTTOM:
    {
        for (x = nSrartX; x <= nEndX; x += nSamp)
        {
            for (y = rectCell.y + rectCell.height; y >= nStartY; y--)
            {
                //cv::circle(drawImg, cv::Point2i(x, y), 2, cv::Scalar(255));
                if (matTempBuf.at<uchar>(y, x) > 250)
                {
                    LOGI("PIXEL: = {}", matTempBuf.at<uchar>(y, x));
                    //cv::circle(drawImg, cv::Point2i(x, y), 2, cv::Scalar(255));
                    ptSrcIndexs.push_back(cv::Point2i(x, y));

                    break;
                }
            }
        }
    }
    break;

    default:
    {
        return E_ERROR_CODE_ALIGN_WARNING_PARA;
    }
    break;
    }

    long nErrorCode = calcRANSAC(ptSrcIndexs, dA, dB, nMinSamples, distThreshold);
    END_TIMER;

}

long PatrolEdge::calcRANSAC(std::vector <cv::Point2i>& ptSrcIndexs, long double& dA, long double& dB, int nMinSamples, double distThreshold)
{
    START_TIMER
    if (nMinSamples < 2)					return E_ERROR_CODE_ALIGN_WARNING_PARA;
    if (ptSrcIndexs.size() <= nMinSamples)	return E_ERROR_CODE_ALIGN_WARNING_PARA;

    long nMaxCost = 0;


    std::vector <cv::Point2i> ptSamples, ptInliers;


    long double dAA = 0, dBB = 0;


    //int nMaxIter = (int)(1 + log(1. - 0.99) / log(1. - pow(0.5, nMinSamples)));
    int nMaxIter = 5;
    for (int i = 0; i < nMaxIter; i++)
    {
        std::vector<cv::Point2i>().swap(ptSamples);
        std::vector<cv::Point2i>().swap(ptInliers);

        GetRandomSamples(ptSrcIndexs, ptSamples, nMinSamples);

        long nErrorCode = calcLineFit(ptSamples, dAA, dBB);


        if (nErrorCode != E_ERROR_CODE_TRUE)
        {
            std::vector<cv::Point2i>().swap(ptSamples);
            std::vector<cv::Point2i>().swap(ptInliers);

            return nErrorCode;
        }

        long cost = calcLineVerification(ptSrcIndexs, ptInliers, dAA, dBB, distThreshold);
        LOGI("line cost = {}", cost);
        if (nMaxCost < cost)
        {
            nMaxCost = cost;
            calcLineFit(ptInliers, dA, dB);
        }

        std::vector<cv::Point2i>().swap(ptInliers);
    }


    std::vector<cv::Point2i>().swap(ptSamples);

    if (nMaxCost <= 0)		return E_ERROR_CODE_ALIGN_CAN_NOT_CALC;
    END_TIMER;
    return E_ERROR_CODE_TRUE;
}

long PatrolEdge::GetRandomSamples(std::vector <cv::Point2i>& ptSrcIndexs, std::vector <cv::Point2i>& ptSamples, int nSampleCount)
{

    int nSize = (int)ptSrcIndexs.size();


    while (ptSamples.size() < nSampleCount)
    {

        int j = rand() % nSize;


        if (!FindInSamples(ptSamples, ptSrcIndexs[j]))
        {

            ptSamples.push_back(ptSrcIndexs[j]);
        }
    }

    return E_ERROR_CODE_TRUE;
}

bool PatrolEdge::FindInSamples(std::vector <cv::Point2i>& ptSamples, cv::Point2i ptIndexs)
{
    for (int i = 0; i < ptSamples.size(); i++)
    {
        if (ptSamples[i].x == ptIndexs.x &&
            ptSamples[i].y == ptIndexs.y)
            return true;
    }

    return false;
}

long PatrolEdge::calcLineFit(std::vector <cv::Point2i>& ptSamples, long double& dA, long double& dB)
{

    if (ptSamples.size() <= 0)	return E_ERROR_CODE_ALIGN_NO_DATA;

    long double sx = 0.0, sy = 0.0;
    long double sxx = 0.0, syy = 0.0;
    long double sxy = 0.0, sw = (long double)ptSamples.size();
    long double x, y;

    for (int i = 0; i < ptSamples.size(); i++)
    {
        x = (long double)ptSamples[i].x;
        y = (long double)ptSamples[i].y;

        sx += x;
        sy += y;
        sxx += x * x;
        sxy += x * y;
        syy += y * y;
    }

    // variance
    long double vxx = (sxx - sx * sx / sw) / sw;
    long double vxy = (sxy - sx * sy / sw) / sw;
    long double vyy = (syy - sy * sy / sw) / sw;

    // principal axis
    long double theta = atan2(2. * vxy, vxx - vyy) / 2.;

    // center of mass(xc, yc)
    sx /= sw;
    sy /= sw;

    // sin(theta)*(x - sx) = cos(theta)*(y - sy);
    //dA = sin(theta) / cos(theta);
    dA = tan(theta);
    dB = sy - dA * sx;

    return E_ERROR_CODE_TRUE;
}

long PatrolEdge::calcLineVerification(std::vector <cv::Point2i>& ptSrcIndexs, std::vector <cv::Point2i>& ptInliers, long double& dA, long double& dB, double distThreshold)
{
    for (int i = 0; i < ptSrcIndexs.size(); i++)
    {

        // | ax + bx + c | / sqrt( a*a + b*b )
        double distance = abs(ptSrcIndexs[i].x * dA - ptSrcIndexs[i].y + dB) / sqrt(dA * dA + 1.);

        LOGI("line distance = {}", distance);
        if (distance < distThreshold)
            ptInliers.push_back(ptSrcIndexs[i]);
    }

    return (long)ptInliers.size();
}

bool PatrolEdge::DoFiltering(stBLOB_FEATURE& tBlobResult, int nBlobFilter, int nSign, double dValue)
{
    // 如果已过滤，则排除
    if (tBlobResult.bFiltering)	return false;

    bool bRes = false;

    switch (nBlobFilter)
    {
    case E_FEATURE_AREA:
        bRes = Compare((double)tBlobResult.nArea, nSign, dValue);
        break;

    case E_FEATURE_BOX_AREA:
        bRes = Compare((double)tBlobResult.nBoxArea, nSign, dValue);
        break;

    case E_FEATURE_BOX_RATIO:
        bRes = Compare((double)tBlobResult.fBoxRatio, nSign, dValue);
        break;

    case E_FEATURE_BOX_X:
        bRes = Compare((double)tBlobResult.rectBox.width, nSign, dValue);
        break;

    case E_FEATURE_BOX_Y:
        bRes = Compare((double)tBlobResult.rectBox.height, nSign, dValue);
        break;

    case E_FEATURE_SUM_GV:
        bRes = Compare((double)tBlobResult.nSumGV, nSign, dValue);
        break;

    case E_FEATURE_MIN_GV:
        bRes = Compare((double)tBlobResult.nMinGV, nSign, dValue);
        break;


    case E_FEATURE_MAX_GV:
        bRes = Compare((double)tBlobResult.nMaxGV, nSign, dValue);
        break;

    case E_FEATURE_MEAN_GV:
        bRes = Compare((double)tBlobResult.fMeanGV, nSign, dValue);
        break;

    case E_FEATURE_DIFF_GV:
        bRes = Compare((double)tBlobResult.fDiffGV, nSign, dValue);
        break;

    case E_FEATURE_BK_GV:
        bRes = Compare((double)tBlobResult.fBKGV, nSign, dValue);
        break;

    case E_FEATURE_STD_DEV:
        bRes = Compare((double)tBlobResult.fStdDev, nSign, dValue);
        break;

    case E_FEATURE_SEMU:
        bRes = Compare((double)tBlobResult.fSEMU, nSign, dValue);
        break;

    case E_FEATURE_COMPACTNESS:
        bRes = Compare((double)tBlobResult.fCompactness, nSign, dValue);
        break;

    case E_FEATURE_MIN_GV_RATIO:
        bRes = Compare((double)tBlobResult.nMinGVRatio, nSign, dValue);
        break;

    case E_FEATURE_MAX_GV_RATIO:
        bRes = Compare((double)tBlobResult.nMaxGVRatio, nSign, dValue);
        break;

    case E_FEATURE_DIFF_GV_RATIO:
        bRes = Compare((double)tBlobResult.fDiffGVRatio, nSign, dValue);
        break;

    case E_FEATURE_PERIMETER:
        bRes = Compare((double)tBlobResult.fPerimeter, nSign, dValue);
        break;

    case E_FEATURE_ROUNDNESS:
        bRes = Compare((double)tBlobResult.fRoundness, nSign, dValue);
        break;

    case E_FEATURE_ELONGATION:
        bRes = Compare((double)tBlobResult.fElongation, nSign, dValue);
        break;

    case E_FEATURE_MIN_BOX_AREA:
        bRes = Compare((double)tBlobResult.fMinBoxArea, nSign, dValue);
        break;

    case E_FEATURE_MINOR_AXIS:
        bRes = Compare((double)tBlobResult.fMinorAxis, nSign, dValue);
        break;

    case E_FEATURE_MAJOR_AXIS:
        bRes = Compare((double)tBlobResult.fMajorAxis, nSign, dValue);
        break;

    case E_FEATURE_AXIS_RATIO:
        bRes = Compare((double)tBlobResult.fAxisRatio, nSign, dValue);
        break;

    case E_FEATURE_MIN_BOX_RATIO:
        bRes = Compare((double)tBlobResult.fMinBoxRatio, nSign, dValue);
        break;

    case E_FEATURE_MEAN_DELTAE:
        bRes = Compare((double)tBlobResult.fMeanDelataE, nSign, dValue);
        break;

        // 2023.01.02 HGM 고객사 요청사항 중심좌표 추가
    case E_FEATURE_CENTER_X:
        bRes = Compare((double)tBlobResult.ptCenter.x, nSign, dValue);
        break;
    case E_FEATURE_CENTER_Y:
        bRes = Compare((double)tBlobResult.ptCenter.y, nSign, dValue);
        break;

    case E_FEATURE_GV_UP_COUNT_0:
    case E_FEATURE_GV_UP_COUNT_1:
    case E_FEATURE_GV_UP_COUNT_2:
    {
        int nCount = (int)dValue / 10000;
        int nGV = (int)dValue % 10000;

        if (nGV < 0)				nGV = 0;
        if (nGV > IMAGE_MAX_GV)	nGV = IMAGE_MAX_GV - 1;

        __int64 nHist = 0;
        for (int m = nGV; m < IMAGE_MAX_GV; m++)
            nHist += tBlobResult.nHist[m];

        bRes = Compare((double)nHist, nSign, (double)nCount);
    }
    break;

    case E_FEATURE_GV_DOWN_COUNT_0:
    case E_FEATURE_GV_DOWN_COUNT_1:
    case E_FEATURE_GV_DOWN_COUNT_2:
    {
        int nCount = (int)dValue / 10000;
        int nGV = (int)dValue % 10000;

        if (nGV < 0)				nGV = 0;
        if (nGV > IMAGE_MAX_GV)	nGV = IMAGE_MAX_GV - 1;

        __int64 nHist = 0;
        for (int m = 0; m <= nGV; m++)
            nHist += tBlobResult.nHist[m];

        bRes = Compare((double)nHist, nSign, (double)nCount);
    }
    break;

    case E_FEATURE_EDGE_DISTANCE:
        bRes = Compare((double)tBlobResult.nDistanceFromEdge, nSign, dValue);
        break;

    default:
        bRes = false;
        break;
    }

    return bRes;
}

bool PatrolEdge::Compare(double dFeatureValue, int nSign, double dValue)
{
    bool bRes = false;

    // 运算符( <, >, ==, <=, >= )
    switch (nSign)
    {
    case	E_SIGN_EQUAL:				// x == judgment value
        bRes = (dFeatureValue == dValue) ? true : false;
        break;

    case	E_SIGN_NOT_EQUAL:			// x != judgment value
        bRes = (dFeatureValue != dValue) ? true : false;
        break;

    case	E_SIGN_GREATER:				// x >  judgment value
        bRes = (dFeatureValue > dValue) ? true : false;
        break;

    case	E_SIGN_LESS:				// x <  judgment value
        bRes = (dFeatureValue < dValue) ? true : false;
        break;

    case	E_SIGN_GREATER_OR_EQUAL:	// x >= judgment value
        bRes = (dFeatureValue >= dValue) ? true : false;
        break;

    case	E_SIGN_LESS_OR_EQUAL:		// x <= judgment value
        bRes = (dFeatureValue <= dValue) ? true : false;
        break;
    }

    return bRes;
}

void PatrolEdge::makeMask_and_obtLineVec(InferTaskPtr task, cv::Mat src, cv::Mat fitMask, cv::Mat& lineFet, cv::Mat& mask, int minCheckEdge, int maxCheckEdge, int nMinSamples, int distThreshold, int edgeType) {
    START_TIMER;
    cv::Rect rectCell;
    rectCell.x = 0;
    rectCell.y = 0;
    rectCell.width = fitMask.cols;
    rectCell.height = fitMask.rows;

    cv::Mat drawImg;
    (src).copyTo(drawImg);
    //Making masks and obtaining line vectors
    //makeMask_and_obtLineVec
    //BOTTOM_Line
    cv::Point2i pt1, pt2;
    json runParams = {
    {"CaliperNum", 30},
    {"CaliperLength", 300},
    {"CaliperWidth", 2},
    {"Num", 1},
    {"Transition", "positive"},
    {"Sigma", 1},
    {"Contrast", 30},
    {"SortByScore", false}
    };
    Tival::TLine xxx = Tival::TLine(342, 150, 0, 150);//TOP 卡尺
    Tival::FindLineResult test_line;
    test_line = Tival::FindLine::Run(task->image, xxx, runParams);

    auto [dValueA, dValueB] = calculateSlopeAndIntercept(test_line.start_point.x, test_line.start_point.y, test_line.end_point.x, test_line.end_point.y);

    lineFet.at<float>(0, 0) = dValueA;
    lineFet.at<float>(0, 1) = dValueB;

    xxx = Tival::TLine(342, 150, 0, 150);//RIGHT 卡尺
    test_line = Tival::FindLine::Run(task->image, xxx, runParams);

    auto [dValueA2, dValueB2] = calculateSlopeAndIntercept(test_line.start_point.x, test_line.start_point.y, test_line.end_point.x, test_line.end_point.y);

    lineFet.at<float>(1, 0) = dValueA2;
    lineFet.at<float>(1, 1) = dValueB2;
    pt1.x = 0;
    pt1.y = static_cast<int>(dValueA2 * pt1.x + dValueB2);
    pt2.x = drawImg.cols - 1;
    pt2.y = static_cast<int>(dValueA2 * pt2.x + dValueB2);
    cv::line(drawImg, pt1, pt2, cv::Scalar(0, 0, 255), 2);

    cv::Point2f crossPoint;
    // 交点:
    // x = - (dB2 - dB1) / (dA2 - dA1)
    // y = dA1 * x + dB1
    crossPoint.x = -(dValueB2 - dValueB) / (dValueA2 - dValueA);
    crossPoint.y = crossPoint.x * dValueA + dValueB;
    cv::circle(drawImg, crossPoint, 2, cv::Scalar(255, 0, 0), 2);
    write_debug_img(task, "drawLineImg", drawImg);
    //填充不检测区
    
    std::vector<cv::Point> vertices(4);
    int dsEdge = minCheckEdge;
    vertices[0] = cv::Point(0, 0); // 左上角
    vertices[1] = cv::Point(-dValueB2 / dValueA2 - dsEdge, 0); // 右上角
    vertices[2] = cv::Point(crossPoint.x - dsEdge, crossPoint.y - dsEdge);; // 交点
    vertices[3] = cv::Point(0, dValueB - dsEdge); // 左下角
    cv::fillPoly(mask, vertices, cv::Scalar(255, 255, 255));
    int ds = maxCheckEdge;
    vertices[0] = cv::Point(0, 0); // 左上角
    vertices[1] = cv::Point(-dValueB2 / dValueA2 - ds, 0); // 右上角
    vertices[2] = cv::Point(crossPoint.x - ds, crossPoint.y - ds); // 交点
    vertices[3] = cv::Point(0, dValueB - ds); // 左下角
    cv::fillPoly(mask, vertices, cv::Scalar(0, 0, 0));

    END_TIMER;
    return;
}

void PatrolEdge::makeMask_and_obtLineVec_BR(InferTaskPtr task, cv::Mat src, cv::Mat fitMask, cv::Mat& lineFet, cv::Mat& mask, int minCheckEdge, int maxCheckEdge, int nMinSamples, int distThreshold, int edgeType) {
    START_TIMER;
    cv::Rect rectCell;
    rectCell.x = 0;
    rectCell.y = 0;
    rectCell.width = fitMask.cols;
    rectCell.height = fitMask.rows;

    cv::Mat drawImg;
    (src).copyTo(drawImg);
    //Making masks and obtaining line vectors
    //makeMask_and_obtLineVec
    //BOTTOM_Line

    cv::Point2i pt1, pt2;
    json runParams = {
    {"CaliperNum", 30},
    {"CaliperLength", 300},
    {"CaliperWidth", 2},
    {"Num", 1},
    {"Transition", "positive"},
    {"Sigma", 1},
    {"Contrast", 30},
    {"SortByScore", false}
    };
    Tival::TLine xxx = Tival::TLine(342, 150, 0, 150);//BOTTOM 卡尺
    Tival::FindLineResult test_line;
    test_line = Tival::FindLine::Run(task->image, xxx, runParams);

    auto [dValueA, dValueB] = calculateSlopeAndIntercept(test_line.start_point.x, test_line.start_point.y, test_line.end_point.x, test_line.end_point.y);

    lineFet.at<float>(0, 0) = dValueA;
    lineFet.at<float>(0, 1) = dValueB;

    xxx = Tival::TLine(342, 150, 0, 150);//RIGHT 卡尺
    test_line = Tival::FindLine::Run(task->image, xxx, runParams);

    auto [dValueA2, dValueB2] = calculateSlopeAndIntercept(test_line.start_point.x, test_line.start_point.y, test_line.end_point.x, test_line.end_point.y);

    lineFet.at<float>(1, 0) = dValueA2;
    lineFet.at<float>(1, 1) = dValueB2;
    pt1.x = 0;
    pt1.y = static_cast<int>(dValueA2 * pt1.x + dValueB2);
    pt2.x = drawImg.cols - 1;
    pt2.y = static_cast<int>(dValueA2 * pt2.x + dValueB2);
    cv::line(drawImg, pt1, pt2, cv::Scalar(0, 0, 255), 2);

    cv::Point2f crossPoint;
    // 交点:
    // x = - (dB2 - dB1) / (dA2 - dA1)
    // y = dA1 * x + dB1
    crossPoint.x = -(dValueB2 - dValueB) / (dValueA2 - dValueA);
    crossPoint.y = crossPoint.x * dValueA + dValueB;
    cv::circle(drawImg, crossPoint, 2, cv::Scalar(255, 0, 0), 2);
    write_debug_img(task, "drawLineImg", drawImg);
    //填充不检测区

    std::vector<cv::Point> vertices(4);
    int dsEdge = minCheckEdge;
    vertices[0] = cv::Point(0, 0); // 左上角
    vertices[1] = cv::Point(-dValueB2 / dValueA2 - dsEdge, 0); // 右上角
    vertices[2] = cv::Point(crossPoint.x - dsEdge, crossPoint.y - dsEdge);; // 交点
    vertices[3] = cv::Point(0, dValueB - dsEdge); // 左下角
    cv::fillPoly(mask, vertices, cv::Scalar(255, 255, 255));
    int ds = maxCheckEdge;
    vertices[0] = cv::Point(0, 0); // 左上角
    vertices[1] = cv::Point(-dValueB2 / dValueA2 - ds, 0); // 右上角
    vertices[2] = cv::Point(crossPoint.x - ds, crossPoint.y - ds); // 交点
    vertices[3] = cv::Point(0, dValueB - ds); // 左下角
    cv::fillPoly(mask, vertices, cv::Scalar(0, 0, 0));
    END_TIMER;
    return;
}

void PatrolEdge::makeMask_and_obtLineVec_BL(InferTaskPtr task, cv::Mat src, cv::Mat fitMask, cv::Mat& lineFet, cv::Mat& mask, int minCheckEdge, int maxCheckEdge, int nMinSamples, int distThreshold, int edgeType) {
    START_TIMER;
    cv::Rect rectCell;
    rectCell.x = 0;
    rectCell.y = 0;
    rectCell.width = fitMask.cols;
    rectCell.height = fitMask.rows;

    cv::Mat drawImg;
    (src).copyTo(drawImg);
    //Making masks and obtaining line vectors
    //makeMask_and_obtLineVec
    //BOTTOM_Line

    cv::Point2i pt1, pt2;
    json runParams = {
    {"CaliperNum", 30},
    {"CaliperLength", 300},
    {"CaliperWidth", 2},
    {"Num", 1},
    {"Transition", "positive"},
    {"Sigma", 1},
    {"Contrast", 30},
    {"SortByScore", false}
    };
    Tival::TLine xxx = Tival::TLine(342, 150, 0, 150);//BOTTOM 卡尺
    Tival::FindLineResult test_line;
    test_line = Tival::FindLine::Run(task->image, xxx, runParams);

    auto [dValueA, dValueB] = calculateSlopeAndIntercept(test_line.start_point.x, test_line.start_point.y, test_line.end_point.x, test_line.end_point.y);

    lineFet.at<float>(0, 0) = dValueA;
    lineFet.at<float>(0, 1) = dValueB;

    xxx = Tival::TLine(342, 150, 0, 150);//LEFT 卡尺
    test_line = Tival::FindLine::Run(task->image, xxx, runParams);

    auto [dValueA2, dValueB2] = calculateSlopeAndIntercept(test_line.start_point.x, test_line.start_point.y, test_line.end_point.x, test_line.end_point.y);

    lineFet.at<float>(1, 0) = dValueA2;
    lineFet.at<float>(1, 1) = dValueB2;
    pt1.x = 0;
    pt1.y = static_cast<int>(dValueA2 * pt1.x + dValueB2);
    //checkPoint(pt1);
    pt2.x = drawImg.cols - 1;
    pt2.y = static_cast<int>(dValueA2 * pt2.x + dValueB2);
    //checkPoint(pt2);
    cv::line(drawImg, pt1, pt2, cv::Scalar(0, 0, 255), 2);

    cv::Point2f crossPoint;
    // 交点:
    // x = - (dB2 - dB1) / (dA2 - dA1)
    // y = dA1 * x + dB1
    crossPoint.x = -(dValueB2 - dValueB) / (dValueA2 - dValueA);
    crossPoint.y = crossPoint.x * dValueA + dValueB;
    cv::circle(drawImg, crossPoint, 2, cv::Scalar(255, 0, 0), 2);
    write_debug_img(task, "drawLineImg", drawImg);
    //填充不检测区

    std::vector<cv::Point> vertices(4);
    int dsEdge = minCheckEdge;
    vertices[0] = cv::Point(-dValueB2 / dValueA2 + dsEdge, 0); // 左上角
    checkPoint(vertices[0]);
    vertices[1] = cv::Point(fitMask.cols, 0); // 右上角
    checkPoint(vertices[1]);
    vertices[2] = cv::Point(fitMask.cols, crossPoint.y - dsEdge); // 右下角
    checkPoint(vertices[2]);
    vertices[3] = cv::Point(crossPoint.x + dsEdge, crossPoint.y - dsEdge); // 交点
    checkPoint(vertices[3]);
    cv::fillPoly(mask, vertices, cv::Scalar(255, 255, 255));
    LOGI("---R3");
    int ds = maxCheckEdge;
    vertices[0] = cv::Point(-dValueB2 / dValueA2 + ds, 0); // 左上角
    checkPoint(vertices[0]);
    vertices[1] = cv::Point(fitMask.cols, 0); // 右上角
    checkPoint(vertices[1]);
    vertices[2] = cv::Point(fitMask.cols, crossPoint.y - ds); // 右下角
    checkPoint(vertices[2]);
    vertices[3] = cv::Point(crossPoint.x + ds, crossPoint.y - ds); // 交点
    checkPoint(vertices[3]);
    cv::fillPoly(mask, vertices, cv::Scalar(0, 0, 0));
    LOGI("---R4");
    END_TIMER;
    return;
}

void PatrolEdge::makeMask_and_obtLineVec_TR(InferTaskPtr task, cv::Mat src, cv::Mat fitMask, cv::Mat& lineFet, cv::Mat& mask, int minCheckEdge, int maxCheckEdge, int nMinSamples, int distThreshold, int edgeType) {
    START_TIMER;
    cv::Rect rectCell;
    rectCell.x = 0;
    rectCell.y = 0;
    rectCell.width = fitMask.cols;
    rectCell.height = fitMask.rows;

    cv::Mat drawImg;
    (src).copyTo(drawImg);
    //Making masks and obtaining line vectors
    //makeMask_and_obtLineVec
    //BOTTOM_Line
    cv::Point2i pt1, pt2;
    json runParams = {
    {"CaliperNum", 30},
    {"CaliperLength", 300},
    {"CaliperWidth", 2},
    {"Num", 1},
    {"Transition", "positive"},
    {"Sigma", 1},
    {"Contrast", 30},
    {"SortByScore", false}
    };
    Tival::TLine xxx = Tival::TLine(342, 150, 0, 150);//TOP 卡尺
    Tival::FindLineResult test_line;
    test_line = Tival::FindLine::Run(task->image, xxx, runParams);

    auto [dValueA, dValueB] = calculateSlopeAndIntercept(test_line.start_point.x, test_line.start_point.y, test_line.end_point.x, test_line.end_point.y);

    lineFet.at<float>(0, 0) = dValueA;
    lineFet.at<float>(0, 1) = dValueB;

    xxx = Tival::TLine(342, 150, 0, 150);//RIGHT 卡尺
    test_line = Tival::FindLine::Run(task->image, xxx, runParams);

    auto [dValueA2, dValueB2] = calculateSlopeAndIntercept(test_line.start_point.x, test_line.start_point.y, test_line.end_point.x, test_line.end_point.y);

    lineFet.at<float>(1, 0) = dValueA2;
    lineFet.at<float>(1, 1) = dValueB2;
    pt1.x = 0;
    pt1.y = static_cast<int>(dValueA2 * pt1.x + dValueB2);
    pt2.x = drawImg.cols - 1;
    pt2.y = static_cast<int>(dValueA2 * pt2.x + dValueB2);
    cv::line(drawImg, pt1, pt2, cv::Scalar(0, 0, 255), 2);

    cv::Point2f crossPoint;
    // 交点:
    // x = - (dB2 - dB1) / (dA2 - dA1)
    // y = dA1 * x + dB1
    crossPoint.x = -(dValueB2 - dValueB) / (dValueA2 - dValueA);
    crossPoint.y = crossPoint.x * dValueA + dValueB;
    cv::circle(drawImg, crossPoint, 2, cv::Scalar(255, 0, 0), 2);
    write_debug_img(task, "drawLineImg", drawImg);
    //填充不检测区

    std::vector<cv::Point> vertices(4);
    int dsEdge = minCheckEdge;
    vertices[0] = cv::Point(0, dValueB + dsEdge); // 左上角
    vertices[1] = cv::Point(crossPoint.x - dsEdge, crossPoint.y + dsEdge); // 交点
    vertices[2] = cv::Point((fitMask.rows - dValueB2) / dValueA2 - dsEdge, fitMask.rows); // 右下角
    vertices[3] = cv::Point(0, fitMask.rows); // 左下角
    cv::fillPoly(mask, vertices, cv::Scalar(255, 255, 255));
    int ds = maxCheckEdge;
    vertices[0] = cv::Point(0, dValueB + ds); // 左上角
    vertices[1] = cv::Point(crossPoint.x - ds, crossPoint.y + ds); // 交点
    vertices[2] = cv::Point((fitMask.rows - dValueB2) / dValueA2 - ds, fitMask.rows); // 右下角
    vertices[3] = cv::Point(0, fitMask.rows); // 左下角
    cv::fillPoly(mask, vertices, cv::Scalar(0, 0, 0));
    END_TIMER;
    return;
}

void PatrolEdge::makeMask_and_obtLineVec_TL(InferTaskPtr task, cv::Mat src, cv::Mat fitMask, cv::Mat& lineFet, cv::Mat& mask, int minCheckEdge, int maxCheckEdge, int nMinSamples, int distThreshold, int edgeType) {
    START_TIMER;
    cv::Rect rectCell;
    rectCell.x = 0;
    rectCell.y = 0;
    rectCell.width = fitMask.cols;
    rectCell.height = fitMask.rows;

    cv::Mat drawImg;
    (src).copyTo(drawImg);
    //Making masks and obtaining line vectors
    //makeMask_and_obtLineVec
    //BOTTOM_Line
    //long double	dValueA = 0, dValueB = 0;
    cv::Point2i pt1, pt2;
    //RobustFitLine(fitMask, rectCell, dValueA, dValueB, nMinSamples, distThreshold, E_ALIGN_TYPE_TOP);
    json runParams = {
    {"CaliperNum", 30},
    {"CaliperLength", 300},
    {"CaliperWidth", 2},
    {"Num", 1},
    {"Transition", "positive"},
    {"Sigma", 1},
    {"Contrast", 30},
    {"SortByScore", false}
    };
    Tival::TLine xxx = Tival::TLine(342, 150, 0, 150);//TOP 卡尺
    Tival::FindLineResult test_line;
    test_line = Tival::FindLine::Run(task->image, xxx, runParams);

    auto [dValueA, dValueB] = calculateSlopeAndIntercept(test_line.start_point.x, test_line.start_point.y, test_line.end_point.x, test_line.end_point.y);

    lineFet.at<float>(0, 0) = dValueA;
    lineFet.at<float>(0, 1) = dValueB;
    //double dTheta = atan(dValueA) * 180. / PI;
    pt1.x = 0;
    pt1.y = static_cast<int>(dValueA * pt1.x + dValueB);
    pt2.x = drawImg.cols - 1;
    pt2.y = static_cast<int>(dValueA * pt2.x + dValueB);
    cv::line(drawImg, pt1, pt2, cv::Scalar(0, 0, 255), 2);
    //RIGHT_Line
    //long double	dValueA2 = 0, dValueB2 = 0;
    //RobustFitLine(fitMask, rectCell, dValueA2, dValueB2, nMinSamples, distThreshold, E_ALIGN_TYPE_LEFT);
    xxx = Tival::TLine(342, 150, 0, 150);//LEFT 卡尺
    test_line = Tival::FindLine::Run(task->image, xxx, runParams);

    auto [dValueA2, dValueB2] = calculateSlopeAndIntercept(test_line.start_point.x, test_line.start_point.y, test_line.end_point.x, test_line.end_point.y);

    lineFet.at<float>(1, 0) = dValueA2;
    lineFet.at<float>(1, 1) = dValueB2;
    pt1.x = 0;
    pt1.y = static_cast<int>(dValueA2 * pt1.x + dValueB2);
    pt2.x = drawImg.cols - 1;
    pt2.y = static_cast<int>(dValueA2 * pt2.x + dValueB2);
    cv::line(drawImg, pt1, pt2, cv::Scalar(0, 0, 255), 2);

    cv::Point2f crossPoint;
    // 交点:
    // x = - (dB2 - dB1) / (dA2 - dA1)
    // y = dA1 * x + dB1
    crossPoint.x = -(dValueB2 - dValueB) / (dValueA2 - dValueA);
    crossPoint.y = crossPoint.x * dValueA + dValueB;
    cv::circle(drawImg, crossPoint, 2, cv::Scalar(255, 0, 0), 2);
    write_debug_img(task, "drawLineImg", drawImg);
    //填充不检测区

    std::vector<cv::Point> vertices(4);
    int dsEdge = minCheckEdge;
    vertices[0] = cv::Point(crossPoint.x + dsEdge, crossPoint.y + dsEdge); // 交点
    vertices[1] = cv::Point(fitMask.cols, dValueA * fitMask.cols + dValueB + dsEdge); // 右上角
    vertices[2] = cv::Point(fitMask.cols, fitMask.rows); // 右下角
    vertices[3] = cv::Point((fitMask.rows - dValueB2) / dValueA2 + dsEdge, fitMask.rows); // 左下角
    cv::fillPoly(mask, vertices, cv::Scalar(255, 255, 255));
    int ds = maxCheckEdge;
    vertices[0] = cv::Point(crossPoint.x + ds, crossPoint.y + ds); // 交点
    vertices[1] = cv::Point(fitMask.cols, dValueA * fitMask.cols + dValueB + ds); // 右上角
    vertices[2] = cv::Point(fitMask.cols, fitMask.rows); // 右下角
    vertices[3] = cv::Point((fitMask.rows - dValueB2) / dValueA2 + ds, fitMask.rows); // 左下角
    cv::fillPoly(mask, vertices, cv::Scalar(0, 0, 0));
    END_TIMER;
    return;
}

void PatrolEdge::makeMask_and_obtLineVec_B(InferTaskPtr task, cv::Mat src, cv::Mat& fitMask, cv::Mat& lineFet, cv::Mat& mask, int minCheckEdge, int maxCheckEdge, int nMinSamples, int distThreshold, int edgeType) {
    START_TIMER;
    cv::Rect rectCell;
    rectCell.x = 0;
    rectCell.y = 0;
    rectCell.width = fitMask.cols;
    rectCell.height = fitMask.rows;

    cv::Mat drawImg;
    (src).copyTo(drawImg);
    //Making masks and obtaining line vectors
    //makeMask_and_obtLineVec
    //BOTTOM_Line
    //long double	dValueA = 0, dValueB = 0;
    cv::Point2i pt1, pt2;
    //RobustFitLine(fitMask, rectCell, dValueA, dValueB, nMinSamples, distThreshold, E_ALIGN_TYPE_BOTTOM);
        /**
     * 通过布置多个卡尺找出边缘点，基于找出的边缘点拟合直线
     * @params：
     * - CaliperNum: 卡尺数量 (def:10)
     * - CaliperLength: 卡尺长度（def:20）
     * - CaliperWidth: 卡尺宽度（def:5）
     * - Num: 搜索实例数(def: 1)
     * - Transition: 极性（def: 'all' [positive, negative]）
     * - Sigma: 高斯平滑Sigma(def: 1)
     * - Contrast: 边缘对比度阈值：(deff: 30)
     * - SortByScore: 按分数排序（def: false）
    */
    json runParams = {
        {"CaliperNum", 10},
        {"CaliperLength", 300},
        {"CaliperWidth", 2},
        {"Num", 1},
        {"Transition", "positive"},
        {"Sigma", 1},
        {"Contrast", 30},
        {"SortByScore", false}
    };
    //Tival::TPoint start = Tival::TPoint(0, 171);
    //Tival::TPoint end = Tival::TPoint(300, 171);
    //Tival::TLine xxx = Tival::TLine(342, 150, 0, 150);
     Tival::TLine xxx = Tival::TLine(342, 150, 0, 150);
    //Tival::TLine xxx = Tival::TLine(start, end);
    Tival::FindLineResult test_line;
    test_line = Tival::FindLine::Run(task->image, xxx, runParams);

    auto [dValueA, dValueB] = calculateSlopeAndIntercept(test_line.start_point.x, test_line.start_point.y, test_line.end_point.x, test_line.end_point.y);

    lineFet.at<float>(0, 0) = dValueA;
    lineFet.at<float>(0, 1) = dValueB;
    //double dTheta = atan(dValueA) * 180. / PI;
    pt1.x = 0;
    pt1.y = static_cast<int>(dValueA * pt1.x + dValueB);
    pt2.x = drawImg.cols - 1;
    pt2.y = static_cast<int>(dValueA * pt2.x + dValueB);
    cv::line(drawImg, pt1, pt2, cv::Scalar(255), 1);
    write_debug_img(task, "drawLineImg", drawImg);
    //填充不检测区

    std::vector<cv::Point> vertices(4);
    int dsEdge = minCheckEdge;
    vertices[0] = cv::Point(0, 0); // 左上角
    vertices[1] = cv::Point(fitMask.cols, 0); // 右上角
    vertices[2] = cv::Point(fitMask.cols, dValueA* fitMask.cols + dValueB - dsEdge);; // 交点
    vertices[3] = cv::Point(0, dValueB - dsEdge); // 左下角
    cv::fillPoly(mask, vertices, cv::Scalar(255, 255, 255));
    int ds = maxCheckEdge;
    vertices[0] = cv::Point(0, 0); // 左上角
    vertices[1] = cv::Point(fitMask.cols, 0); // 右上角
    vertices[2] = cv::Point(fitMask.cols, dValueA * fitMask.cols + dValueB - ds); // 交点
    vertices[3] = cv::Point(0, dValueB - ds); // 左下角
    cv::fillPoly(mask, vertices, cv::Scalar(0));
    END_TIMER;
    return;
}

void PatrolEdge::makeMask_and_obtLineVec_T(InferTaskPtr task, cv::Mat src, cv::Mat fitMask, cv::Mat& lineFet, cv::Mat& mask, int minCheckEdge, int maxCheckEdge, int nMinSamples, int distThreshold, int edgeType) {
    START_TIMER;
    cv::Rect rectCell;
    rectCell.x = 0;
    rectCell.y = 0;
    rectCell.width = fitMask.cols;
    rectCell.height = fitMask.rows;

    cv::Mat drawImg;
    (src).copyTo(drawImg);
    //Making masks and obtaining line vectors
    //makeMask_and_obtLineVec
    //BOTTOM_Line
    long double	dValueA = 0, dValueB = 0;
    cv::Point2i pt1, pt2;
    RobustFitLine(fitMask, rectCell, dValueA, dValueB, nMinSamples, distThreshold, E_ALIGN_TYPE_TOP);
    lineFet.at<float>(0, 0) = dValueA;
    lineFet.at<float>(0, 1) = dValueB;
    //double dTheta = atan(dValueA) * 180. / PI;
    pt1.x = 0;
    pt1.y = static_cast<int>(dValueA * pt1.x + dValueB);
    pt2.x = drawImg.cols - 1;
    pt2.y = static_cast<int>(dValueA * pt2.x + dValueB);
    cv::line(drawImg, pt1, pt2, cv::Scalar(255), 2);
    write_debug_img(task, "drawLineImg", drawImg);
    //填充不检测区

    std::vector<cv::Point> vertices(4);
    int dsEdge = minCheckEdge;
    vertices[0] = cv::Point(0, dValueB - dsEdge); // 左上角
    vertices[1] = cv::Point(fitMask.cols, dValueA* fitMask.cols + dValueB - dsEdge); // 右上角
    vertices[2] = cv::Point(0, fitMask.rows); // 左下  交点
    vertices[3] = cv::Point(fitMask.cols, fitMask.cols); // 左下角
    cv::fillPoly(mask, vertices, cv::Scalar(255, 255, 255));
    int ds = maxCheckEdge;
    vertices[0] = cv::Point(0, dValueB - ds); // 左上角
    vertices[1] = cv::Point(fitMask.cols, dValueA* fitMask.cols + dValueB - ds); // 右上角
    vertices[2] = cv::Point(0, fitMask.rows); // 左下  交点
    vertices[3] = cv::Point(fitMask.cols, fitMask.cols); // 左下角
    cv::fillPoly(mask, vertices, cv::Scalar(0));
    END_TIMER;
    return;
}

void PatrolEdge::makeMask_and_obtLineVec_R(InferTaskPtr task, cv::Mat src, cv::Mat fitMask, cv::Mat& lineFet, cv::Mat& mask, int minCheckEdge, int maxCheckEdge, int nMinSamples, int distThreshold, int edgeType) {
    START_TIMER;
    cv::Rect rectCell;
    rectCell.x = 0;
    rectCell.y = 0;
    rectCell.width = fitMask.cols;
    rectCell.height = fitMask.rows;

    cv::Mat drawImg;
    (src).copyTo(drawImg);

    cv::Point2i pt1, pt2;
    //RIGHT_Line
    long double	dValueA2 = 0, dValueB2 = 0;
    RobustFitLine(fitMask, rectCell, dValueA2, dValueB2, nMinSamples, distThreshold, E_ALIGN_TYPE_RIGHT);
    lineFet.at<float>(1, 0) = dValueA2;
    lineFet.at<float>(1, 1) = dValueB2;
    pt1.x = 0;
    pt1.y = static_cast<int>(dValueA2 * pt1.x + dValueB2);
    pt2.x = drawImg.cols - 1;
    pt2.y = static_cast<int>(dValueA2 * pt2.x + dValueB2);
    cv::line(drawImg, pt1, pt2, cv::Scalar(0, 0, 255), 2);
    write_debug_img(task, "drawLineImg", drawImg);
    //填充不检测区

    std::vector<cv::Point> vertices(4);
    int dsEdge = minCheckEdge;
    vertices[0] = cv::Point(0, 0); // 左上角
    vertices[1] = cv::Point(-dValueB2 / dValueA2 - dsEdge, 0); // 右上角
    vertices[2] = cv::Point((fitMask.rows - dValueB2)/ dValueA2 - dsEdge, fitMask.rows); // 交点
    vertices[3] = cv::Point(0, dValueB2); // 左下角
    cv::fillPoly(mask, vertices, cv::Scalar(255, 255, 255));
    int ds = maxCheckEdge;
    vertices[0] = cv::Point(0, 0); // 左上角
    vertices[1] = cv::Point(-dValueB2 / dValueA2 - ds, 0); // 右上角
    vertices[2] = cv::Point((fitMask.rows - dValueB2) / dValueA2 - ds, fitMask.rows); // 交点
    vertices[3] = cv::Point(0, dValueB2); // 左下角
    cv::fillPoly(mask, vertices, cv::Scalar(0, 0, 0));
    END_TIMER;
    return;
}

void PatrolEdge::makeMask_and_obtLineVec_L(InferTaskPtr task, cv::Mat src, cv::Mat fitMask, cv::Mat& lineFet, cv::Mat& mask, int minCheckEdge, int maxCheckEdge, int nMinSamples, int distThreshold, int edgeType) {
    START_TIMER;
    cv::Rect rectCell;
    rectCell.x = 0;
    rectCell.y = 0;
    rectCell.width = fitMask.cols;
    rectCell.height = fitMask.rows;

    cv::Mat drawImg;
    (src).copyTo(drawImg);

    cv::Point2i pt1, pt2;
    //RIGHT_Line
    long double	dValueA2 = 0, dValueB2 = 0;
    RobustFitLine(fitMask, rectCell, dValueA2, dValueB2, nMinSamples, distThreshold, E_ALIGN_TYPE_LEFT);
    lineFet.at<float>(1, 0) = dValueA2;
    lineFet.at<float>(1, 1) = dValueB2;
    pt1.x = 0;
    pt1.y = static_cast<int>(dValueA2 * pt1.x + dValueB2);
    pt2.x = drawImg.cols - 1;
    pt2.y = static_cast<int>(dValueA2 * pt2.x + dValueB2);
    cv::line(drawImg, pt1, pt2, cv::Scalar(0, 0, 255), 2);
    write_debug_img(task, "drawLineImg", drawImg);
    //填充不检测区

    std::vector<cv::Point> vertices(4);
    int dsEdge = minCheckEdge;
    vertices[0] = cv::Point(-dValueB2 / dValueA2 - dsEdge, 0); // 左上角
    vertices[1] = cv::Point(fitMask.cols, 0); // 右上角
    vertices[2] = cv::Point(fitMask.cols, fitMask.rows); // 交点
    vertices[3] = cv::Point((fitMask.rows - dValueB2)/ dValueA2 - dsEdge, fitMask.rows); // 左下角
    cv::fillPoly(mask, vertices, cv::Scalar(255, 255, 255));
    int ds = maxCheckEdge;
    vertices[0] = cv::Point(-dValueB2 / dValueA2 - ds, 0); // 左上角
    vertices[1] = cv::Point(fitMask.cols, 0); // 右上角
    vertices[2] = cv::Point(fitMask.cols, fitMask.rows); // 交点
    vertices[3] = cv::Point((fitMask.rows - dValueB2)/ dValueA2 - ds, fitMask.rows); // 左下角
    cv::fillPoly(mask, vertices, cv::Scalar(0, 0, 0));
    END_TIMER;
    return;
}


int PatrolEdge::findMaxGVDifferenceCorners(cv::Mat src)
{
    // 四个角的位置
    cv::Point corners[4];
    corners[0] = cv::Point(0, 0); // 左上角
    corners[1] = cv::Point(src.cols - 1, 0); // 右上角
    corners[2] = cv::Point(0, src.rows - 1); // 左下角
    corners[3] = cv::Point(src.cols - 1, src.rows - 1); // 右下角

    float minDiff = 999; // 最大差异值
    float minSecond = 999; //第二大差异值
    std::vector<cv::Rect> cornerRect;
    int shift = 10;//50
    int sizeRect = 20;//100
    cv::Rect roiLT(corners[0].x + shift, corners[0].y + shift, sizeRect, sizeRect);//左上
    cornerRect.push_back(roiLT);
    cv::Rect roiRT(corners[1].x - sizeRect - shift, corners[1].y + shift, sizeRect, sizeRect);//右上
    cornerRect.push_back(roiRT);
    cv::Rect roiLB(corners[2].x + shift, corners[2].y - sizeRect - shift, sizeRect, sizeRect);//左下
    cornerRect.push_back(roiLB);
    cv::Rect roiRB(corners[3].x - sizeRect - shift, corners[3].y - sizeRect - shift, sizeRect, sizeRect);//右下
    cornerRect.push_back(roiRB);
    //cv::rectangle(src, roiLT, cv::Scalar(0, 0, 255), 2);
    //cv::rectangle(src, roiRT, cv::Scalar(0, 0, 255), 2);
    //cv::rectangle(src, roiLB, cv::Scalar(0, 0, 255), 2);
    //cv::rectangle(src, roiRB, cv::Scalar(0, 0, 255), 2);
    int min_i = -1, min_si = -1;
    int i = 0;
    bool enableCorner1 = false, enableCorner2 = false;
    float meanArray[4];
    int mNum = 0;
    for (cv::Rect rect : cornerRect) {
        //float meanValue = cv::mean(src(rect))[0];
        meanArray[mNum] = cv::mean(src(rect))[0];
        mNum++;
        i++;
    }
    int edgeGv = 50;
    int blackGv = 0;
    LOGI("0:{}, 1:{}, 2:{}, 3:{}", meanArray[0], meanArray[1], meanArray[2], meanArray[3]);
    if (meanArray[0] > edgeGv && 
        meanArray[1] > edgeGv && 
        meanArray[2] < edgeGv &&  
        meanArray[3] < edgeGv) {
        return E_POSITION_B;
    }
    if (meanArray[0] < edgeGv && 
        meanArray[1] < edgeGv && 
        meanArray[2] > edgeGv && 
        meanArray[3] > edgeGv) {
        return E_POSITION_T;
    }
    if (meanArray[0] < edgeGv && 
        meanArray[1] > edgeGv && 
        meanArray[2] < edgeGv && 
        meanArray[3] > edgeGv) {
        return E_POSITION_L;
    }
    if (meanArray[0] > edgeGv && 
        meanArray[1] < edgeGv &&
        meanArray[2] > edgeGv && 
        meanArray[3] < edgeGv) {
        return E_POSITION_R;
    }
    if (meanArray[0] > edgeGv &&
        meanArray[1] < edgeGv &&
        meanArray[2] < edgeGv &&
        meanArray[3] < edgeGv) {
        return E_POSITION_BR;
    }
    if (meanArray[0] < edgeGv &&
        meanArray[1] > edgeGv &&
        meanArray[2] < edgeGv &&
        meanArray[3] < edgeGv) {
        return E_POSITION_BL;
    }
    if (meanArray[0] < edgeGv &&
        meanArray[1] < edgeGv && 
        meanArray[2] > edgeGv && 
        meanArray[3] < edgeGv) {
        return E_POSITION_TR;
    }
    if (meanArray[0] < edgeGv &&
        meanArray[1] < edgeGv &&
        meanArray[2] < edgeGv &&
        meanArray[3] > edgeGv) {
        return E_POSITION_TL;
    }
    return -1;
}

bool PatrolEdge::EnhanceContrast(cv::Mat& matSrcBuf, int nOffSet, double dRatio)
{
    cv::cvtColor(matSrcBuf, matSrcBuf, cv::COLOR_BGR2GRAY);
	double dAvg = cv::mean(matSrcBuf)[0];
	int nMax = (int)(dAvg * (1.0 + dRatio));
	int nMin = (int)(dAvg * (1.0 - dRatio));
	if (nMax > 255)	nMax = 255;
	if (nMin < 0)	nMin = 0;

	uchar LUT[256] = { 0, };
	for (int i = 0; i < 256; i++)
	{
		if (i < nMin)		LUT[i] = nMin;
		else if (i > nMax)	LUT[i] = nMax;
		else				LUT[i] = i;
	}

	cv::MatIterator_<uchar> itSrc, endSrc;
	itSrc = matSrcBuf.begin<uchar>();
	endSrc = matSrcBuf.end<uchar>();

	for (; itSrc != endSrc; itSrc++)
		*itSrc = LUT[(*itSrc)];

	int nSize = 3;
	cv::blur(matSrcBuf, matSrcBuf, cv::Size(nSize, nSize));

	nMax += nOffSet;
	nMin -= nOffSet;

	double dVal = 255.0 / (nMax - nMin);
	cv::subtract(matSrcBuf, nMin, matSrcBuf);
	cv::multiply(matSrcBuf, dVal, matSrcBuf);

	return true;
}

void PatrolEdge::LineDFT(cv::Mat &matSrcImage, cv::Mat &matDstImage, BOOL bRemove, int nAxis, double dDegree)
{
	cv::Mat matGray;
	cv::Mat padded;

	matSrcImage.copyTo(matGray);
	double dMinVal, dMaxVal;
	cv::minMaxLoc(matGray, &dMinVal, &dMaxVal);


	int m = cv::getOptimalDFTSize(matGray.rows);
	int n = cv::getOptimalDFTSize(matGray.cols);
	cv::copyMakeBorder(matGray, padded, 0, m - matGray.rows, 0, n - matGray.cols, cv::BORDER_CONSTANT, cv::Scalar::all(0));
	//int m = getOptimalDFTSize(matGray.rows) < getOptimalDFTSize(matGray.cols) ? getOptimalDFTSize(matGray.cols) : getOptimalDFTSize(matGray.rows);

	//cv::copyMakeBorder(matGray, padded, 0, m - matGray.rows, 0, m - matGray.cols, BORDER_CONSTANT, Scalar::all(0));


	cv::Mat planes[] = { cv::Mat_<float>(padded), cv::Mat::zeros(padded.size(), CV_32F) };
	cv::Mat complexI;
	
	merge(planes, 2, complexI);         // Add to the expanded another plane with zeros
	
	dft(complexI, complexI);            // this way the result may fit in the source matrix


										// compute the magnitude and switch to logarithmic scale
										// => log(1 + sqrt(Re(DFT(I))^2 + Im(DFT(I))^2))
	split(complexI, planes);                   // planes[0] = Re(DFT(I), planes[1] = Im(DFT(I))

	cv::Mat matSpectrum;
	Complex2SpectrumSave(complexI, padded, matSpectrum);

	cv::Mat matComplexSpectrum1, matComplexSpectrum2;
	Complex2SpectrumComplex(complexI, padded, matComplexSpectrum1, 0);
	Complex2SpectrumComplex(complexI, padded, matComplexSpectrum2, 1);

	SpectrumComplex2Complex(matComplexSpectrum1, matComplexSpectrum2, padded, complexI);

	////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// Delete Noise

	cv::Mat matLine = cv::Mat::zeros(matComplexSpectrum1.size(), matComplexSpectrum1.type());
	cv::Point ptCenter(matLine.cols / 2, matLine.rows / 2);
	//cv::Size boxsize(sqrt((matLine.cols*matLine.cols) + (matLine.rows*matLine.rows)), 3);
	cv::Point2f pt[4];
	
	if(bRemove)  // 选择是否删除或检测行0：仅检测行，1：删除行
	{
		if (nAxis == 1)		// 删除X方向的线
		{
			for (int y = 0; y < matComplexSpectrum1.rows; y++) {
				for (int x = 0; x < matComplexSpectrum1.cols; x++) {
					if (matComplexSpectrum1.cols / 2 + 1 >= x && matComplexSpectrum1.cols / 2 - 1 <= x)
					{
						matComplexSpectrum1.at<float>(y, x) = 0.0;
						matComplexSpectrum2.at<float>(y, x) = 0.0;
						matSpectrum.at<float>(y, x) = 0.0;
					}
				}
			}
		}
		else if (nAxis == 2)	// 删除Y方向的线
		{
			for (int y = 0; y < matComplexSpectrum1.rows; y++) {
				for (int x = 0; x < matComplexSpectrum1.cols; x++) {
					if (matComplexSpectrum1.rows / 2 + 1 >= y && matComplexSpectrum1.rows / 2 - 1 <= y)
					{
						matComplexSpectrum1.at<float>(y, x) = 0.0;
						matComplexSpectrum2.at<float>(y, x) = 0.0;
						matSpectrum.at<float>(y, x) = 0.0;
					}
				}
			}
		}
		else
		{
			cv::Size boxsize1(200, 40);
			cv::RotatedRect box1(ptCenter, boxsize1, dDegree);
			box1.points(pt);
			cv::line(matLine, (pt[0] + pt[1]) / 2, (pt[2] + pt[3]) / 2, 255, 5);

			

			/*for (int y = 0; y < matComplexSpectrum1.rows; y++) {
				for (int x = 0; x < matComplexSpectrum1.cols; x++) {
					if ((y <= atan(dDegree)*x + 1) && (y >= atan(dDegree)*x - 1))
					{
						matComplexSpectrum1.at<float>(y, x) = 0.0;
						matComplexSpectrum2.at<float>(y, x) = 0.0;
						matSpectrum.at<float>(y, x) = 0.0;
					}
				}
			}*/
		}
		cv::Mat matCenterEllipse = cv::Mat::zeros(matComplexSpectrum1.size(), matComplexSpectrum1.type());

		//Ellipse1
		cv::RotatedRect SDrectE1(ptCenter, cv::Size(100, 100), 0);
		cv::ellipse(matCenterEllipse, SDrectE1, 255, -1, cv::LINE_AA);
		//	
		cv::Mat matNoizeFrq = ~matCenterEllipse&matLine;

		for (int y = 0; y < matComplexSpectrum1.rows; y++) {
			for (int x = 0; x < matComplexSpectrum1.cols; x++) {
				//if ((x > (matComplexSpectrum1.cols / 2+20) || x <(matComplexSpectrum1.cols / 2 - 20)))
				if(y != matComplexSpectrum1.rows / 2 && x != matComplexSpectrum1.cols / 2)
				{
					if (matNoizeFrq.at<float>(y, x) == 255)
					{
						matComplexSpectrum1.at<float>(y, x) = 0.0;
						matComplexSpectrum2.at<float>(y, x) = 0.0;
						matSpectrum.at<float>(y, x) = 0.0;
					}
				}				
			}
		}
	}
	else
	{
		if (nAxis == 1)
		{
			for (int y = 0; y < matComplexSpectrum1.rows; y++) {
				for (int x = 0; x < matComplexSpectrum1.cols; x++) {
					if (matComplexSpectrum1.cols / 2 + 1 < x || matComplexSpectrum1.cols / 2 - 1 > x)
					{
						matComplexSpectrum1.at<float>(y, x) = 0.0;
						matComplexSpectrum2.at<float>(y, x) = 0.0;
						matSpectrum.at<float>(y, x) = 0.0;
					}
				}
			}
		}
		else if (nAxis == 2)
		{
			for (int y = 0; y < matComplexSpectrum1.rows; y++) {
				for (int x = 0; x < matComplexSpectrum1.cols; x++) {
					if (matComplexSpectrum1.rows / 2 + 1 < y || matComplexSpectrum1.rows / 2 - 1 > y)
					{
						matComplexSpectrum1.at<float>(y, x) = 0.0;
						matComplexSpectrum2.at<float>(y, x) = 0.0;
						matSpectrum.at<float>(y, x) = 0.0;
					}
				}
			}
		}
		else
		{
			for (int y = 0; y < matComplexSpectrum1.rows; y++) {
				for (int x = 0; x < matComplexSpectrum1.cols; x++) {
					if( (y > atan(dDegree)*x +1) || (y < atan(dDegree)*x - 1))
					{
						matComplexSpectrum1.at<float>(y, x) = 0.0;
						matComplexSpectrum2.at<float>(y, x) = 0.0;
						matSpectrum.at<float>(y, x) = 0.0;
					}
				}
			}
		}		
	}
	SpectrumComplex2Complex(matComplexSpectrum1, matComplexSpectrum2, padded, complexI);

	// Delete Noise End
	////////////////////////////////////////////////////////////////////////////////////////////////////////////


	////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//	calculating the Invert DFT Start
	cv::Mat inverseTransform;
	cv::dft(complexI, inverseTransform, cv::DFT_INVERSE | cv::DFT_REAL_OUTPUT);
	normalize(inverseTransform, inverseTransform, dMinVal, dMaxVal, cv::NORM_MINMAX);

	//	calculating the Invert DFT End
	////////////////////////////////////////////////////////////////////////////////////////////////////////////


	cv::Mat matIDFT = inverseTransform(cv::Rect(0, 0, matSrcImage.cols, matSrcImage.rows));

	//cv::Mat matTemp9;
	matIDFT.convertTo(matDstImage, matSrcImage.type());
}

void PatrolEdge::Complex2SpectrumSave(cv::Mat &matComplex, cv::Mat &matPadded, cv::Mat &matSpectrum)
{
	cv::Mat planes[] = { cv::Mat_<float>(matPadded), cv::Mat::zeros(matPadded.size(), CV_32F) };
	split(matComplex, planes);                   // planes[0] = Re(DFT(I), planes[1] = Im(DFT(I))

												 /////////////////////////////////////////////////////////////////////////////////////
	cv::magnitude(planes[0], planes[1], planes[0]);// planes[0] = magnitude
	matSpectrum = planes[0].clone();

	matSpectrum += cv::Scalar::all(1);                    // switch to logarithmic scale
	cv::log(matSpectrum, matSpectrum);
	//////////////////////////////////////////////////////////////////////////////////////

	// crop the spectrum, if it has an odd number of rows or columns
	matSpectrum = matSpectrum(cv::Rect(0, 0, matSpectrum.cols & -2, matSpectrum.rows & -2));

	// rearrange the quadrants of Fourier image  so that the origin is at the image center
	int cx = matSpectrum.cols / 2;
	int cy = matSpectrum.rows / 2;

	cv::Mat q0(matSpectrum, cv::Rect(0, 0, cx, cy));   // Top-Left - Create a ROI per quadrant
	cv::Mat q1(matSpectrum, cv::Rect(cx, 0, cx, cy));  // Top-Right
	cv::Mat q2(matSpectrum, cv::Rect(0, cy, cx, cy));  // Bottom-Left
	cv::Mat q3(matSpectrum, cv::Rect(cx, cy, cx, cy)); // Bottom-Right

	cv::Mat tmp;                           // swap quadrants (Top-Left with Bottom-Right)
	q0.copyTo(tmp);
	q3.copyTo(q0);
	tmp.copyTo(q3);

	q1.copyTo(tmp);                    // swap quadrant (Top-Right with Bottom-Left)
	q2.copyTo(q1);
	tmp.copyTo(q2);


	normalize(matSpectrum, matSpectrum, 0, 255, cv::NORM_MINMAX); // Transform the matrix with float values into a
															//imshow("Spectrum Planes", matSpectrum);
															//imwrite("D:/CodeTest/SimulrationDFT/x64/Debug/matSpectrum.bmp", matSpectrum);
}

void PatrolEdge::Complex2SpectrumComplex(cv::Mat &matComplex, cv::Mat &matPadded, cv::Mat &matSpectrum, int PlanesIdx)
{
	cv::Mat planes[] = { cv::Mat_<float>(matPadded), cv::Mat::zeros(matPadded.size(), CV_32F) };
	cv::split(matComplex, planes);                   // planes[0] = Re(DFT(I), planes[1] = Im(DFT(I))

												 /////////////////////////////////////////////////////////////////////////////////////
												 //cv::magnitude(planes[0], planes[1], planes[0]);// planes[0] = magnitude
	matSpectrum = planes[PlanesIdx].clone();

	//matSpectrum += Scalar::all(1);                    // switch to logarithmic scale
	//cv::log(matSpectrum, matSpectrum);
	//////////////////////////////////////////////////////////////////////////////////////

	// crop the spectrum, if it has an odd number of rows or columns
	matSpectrum = matSpectrum(cv::Rect(0, 0, matSpectrum.cols /*& -2*/, matSpectrum.rows /*& -2*/));

	// rearrange the quadrants of Fourier image  so that the origin is at the image center
	int cx = matSpectrum.cols / 2;
	int cy = matSpectrum.rows / 2;

	cv::Mat q0(matSpectrum, cv::Rect(0, 0, cx, cy));   // Top-Left - Create a ROI per quadrant
	cv::Mat q1(matSpectrum, cv::Rect(cx, 0, cx, cy));  // Top-Right
	cv::Mat q2(matSpectrum, cv::Rect(0, cy, cx, cy));  // Bottom-Left
	cv::Mat q3(matSpectrum, cv::Rect(cx, cy, cx, cy)); // Bottom-Right

	cv::Mat tmp;                           // swap quadrants (Top-Left with Bottom-Right)
	q0.copyTo(tmp);
	q3.copyTo(q0);
	tmp.copyTo(q3);

	q1.copyTo(tmp);                    // swap quadrant (Top-Right with Bottom-Left)
	q2.copyTo(q1);
	tmp.copyTo(q2);


	//normalize(matSpectrum, matSpectrum, 0, 1, CV_MINMAX); // Transform the matrix with float values into a
	//imshow("Spectrum Planes", matSpectrumPlanes1);

}

void PatrolEdge::SpectrumComplex2Complex(cv::Mat &matSpectrum1, cv::Mat &matSpectrum2, cv::Mat &matPadded, cv::Mat &matComplex)
{
	cv::Mat planes[] = { cv::Mat_<float>(matPadded), cv::Mat::zeros(matPadded.size(), CV_32F) };
	{
		// crop the spectrum, if it has an odd number of rows or columns
		//matSpectrum1 = matSpectrum1(Rect(0, 0, matSpectrum1.cols & -2, matSpectrum1.rows & -2));

		// rearrange the quadrants of Fourier image  so that the origin is at the image center
		int cx = matSpectrum1.cols / 2;
		int cy = matSpectrum1.rows / 2;

		cv::Mat q0(matSpectrum1, cv::Rect(0, 0, cx, cy));   // Top-Left - Create a ROI per quadrant
		cv::Mat q1(matSpectrum1, cv::Rect(cx, 0, cx, cy));  // Top-Right
		cv::Mat q2(matSpectrum1, cv::Rect(0, cy, cx, cy));  // Bottom-Left
		cv::Mat q3(matSpectrum1, cv::Rect(cx, cy, cx, cy)); // Bottom-Right

		cv::Mat tmp;                           // swap quadrants (Top-Left with Bottom-Right)
		q0.copyTo(tmp);
		q3.copyTo(q0);
		tmp.copyTo(q3);

		q1.copyTo(tmp);                    // swap quadrant (Top-Right with Bottom-Left)
		q2.copyTo(q1);
		tmp.copyTo(q2);

		planes[0] = matSpectrum1.clone();
	}

	{
		// crop the spectrum, if it has an odd number of rows or columns
		//matSpectrum2 = matSpectrum2(Rect(0, 0, matSpectrum2.cols & -2, matSpectrum2.rows & -2));

		// rearrange the quadrants of Fourier image  so that the origin is at the image center
		int cx = matSpectrum2.cols / 2;
		int cy = matSpectrum2.rows / 2;

		cv::Mat q0(matSpectrum2, cv::Rect(0, 0, cx, cy));   // Top-Left - Create a ROI per quadrant
		cv::Mat q1(matSpectrum2, cv::Rect(cx, 0, cx, cy));  // Top-Right
		cv::Mat q2(matSpectrum2, cv::Rect(0, cy, cx, cy));  // Bottom-Left
		cv::Mat q3(matSpectrum2, cv::Rect(cx, cy, cx, cy)); // Bottom-Right

		cv::Mat tmp;                           // swap quadrants (Top-Left with Bottom-Right)
		q0.copyTo(tmp);
		q3.copyTo(q0);
		tmp.copyTo(q3);

		q1.copyTo(tmp);                    // swap quadrant (Top-Right with Bottom-Left)
		q2.copyTo(q1);
		tmp.copyTo(q2);

		planes[1] = matSpectrum2.clone();
	}

	cv::merge(planes, 2, matComplex);

	//normalize(matSpectrum, matSpectrum, 0, 1, CV_MINMAX); // Transform the matrix with float values into a
	//imshow("Spectrum Planes", matSpectrumPlanes1);

}

void PatrolEdge::WriteJudgeParams(InferTaskPtr task, const STRU_DEFECT_ITEM *EdgeDefectJudgment, int i) {
    START_TIMER
    const std::type_info& info = typeid(*this);
    std::string imgName = task->image_info["img_name"];
    std::string fpath = fs::current_path().string() + "\\debugImg\\" + info.name() + "\\";
    if (!fs::exists(fpath)) {
        if (!fs::create_directories(fpath)) {
            std::cerr << "Error creating directory: " << fpath << std::endl;
            std::string fpath1 = fs::current_path().string() + "\\Unkonw";
            fs::create_directories(fpath1);
        }
    }
    std::string task_type_id = task->image_info["type_id"];
    std::string filePath = fpath + "JudgeParams.csv";
    std::ofstream file(filePath);

    if (file.is_open()) {
        // 写入第一行的变量名
        

        // 写入每个结构体的数据
        int num = 1;
        for (int judgeNum = 0; judgeNum < i; judgeNum++) {
            if (!EdgeDefectJudgment[judgeNum].strItemName.empty()) {
                file << EdgeDefectJudgment[judgeNum].strItemName << "\n";
            }
            file << "index,JudgeName,Symbol,Value\n";
            for (int k = 0; k < MAX_MEM_SIZE_E_DEFECT_JUDGMENT_COUNT; k++) {
                if (EdgeDefectJudgment[judgeNum].Judgment[k].bUse == false) continue;
                file << num++ << ","
                    << EdgeDefectJudgment[judgeNum].Judgment[k].name << ","
                    << getSymbolFromSign(EdgeDefectJudgment[judgeNum].Judgment[k].nSign) << ","
                    << EdgeDefectJudgment[judgeNum].Judgment[k].dValue << "\n";
            }
        }


        file.close();
        std::cout << "JudgeParams.txt saved successfully." << std::endl;
    }
    else {
        std::cerr << "Failed to open file: " << filePath << std::endl;
    }
    END_TIMER
    return;
}

double PatrolEdge::calculateDistance(const cv::Point& point, double a, double b) {
    double distance = std::abs(point.y - (a * point.x + b)) / std::sqrt(a * a + 1);
    return distance;
}

double PatrolEdge::calculateMinDistanceToLines(const cv::Rect& rectBox, double a, double b) {
    double minDistance = -1;

    double distance1 = calculateDistance(rectBox.tl(), a, b); // 左上角点
    double distance2 = calculateDistance(rectBox.br(), a, b); // 右下角点
    double distance3 = calculateDistance(cv::Point(rectBox.x + rectBox.width, rectBox.y), a, b); // 右上角点
    double distance4 = calculateDistance(cv::Point(rectBox.x, rectBox.y + rectBox.height), a, b); // 左下角点

    minDistance = min(min(distance1, distance2), min(distance1, distance2));

    return minDistance;
}

void PatrolEdge::checkPoint(cv::Point& p) {

    if (p.x < 0) p.x = 0;
    if (p.y < 0) p.y = 0;
    return;
}

std::tuple<double, double> PatrolEdge::calculateSlopeAndIntercept(double x1, double y1, double x2, double y2) {
    // 如果两个点的x坐标相同，则直线垂直，不能计算斜率
    double slope;
    double intercept;
    if (x1 == x2) {
        //throw std::invalid_argument("The line is vertical, slope is undefined.");
        slope = 0;
        intercept = y1;
        return std::make_tuple(slope, intercept);
    }

    // 计算斜率
    slope = (y2 - y1) / (x2 - x1);

    // 计算截距
    intercept = y1 - slope * x1;

    return std::make_tuple(slope, intercept);
}

void PatrolEdge::detectEdgesWithGaps(InferTaskPtr task, const cv::Mat& inputImage, cv::Mat& outputImage, const cv::Mat& lineFet) {
    START_TIMER;

    //if (lineFet.empty()) return;

    //double  dValueA = lineFet.at<float>(0, 0);//斜率
    //double  dValueB = lineFet.at<float>(0, 1);//截距

    //int width = inputImage.cols;
    //int height = inputImage.rows;



    //cv::Point pt1, pt2;

    //// 左边界 (x = 0)
    //pt1.x = 0;
    //pt1.y = static_cast<int>(dValueB);

    //// 右边界(x = width - 1)
    //pt2.x = width - 1;
    //pt2.y = static_cast<int>(dValueA * pt2.x + dValueB);


    cv::Mat gray, binaryImage;
    cv::cvtColor(inputImage, gray, cv::COLOR_BGR2GRAY);
    cv::threshold(gray, binaryImage, 30, 255, cv::THRESH_BINARY);

    //if (dValueB < height && dValueB > 0) {//
    //    cv::line(binaryImage, pt1, pt2, cv::Scalar(255), 2);
    //}

    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Point> hullcontours;
    cv::findContours(binaryImage, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    outputImage = cv::Mat::zeros(inputImage.size(), CV_8UC1);

    for (size_t i = 0; i < contours.size(); ++i) {

        std::vector<cv::Point> hull;
        cv::convexHull(contours[i], hull);

        std::vector<std::vector<cv::Point>> hulls(1, hull);

        cv::drawContours(outputImage, hulls, 0, cv::Scalar(255, 0, 0), cv::FILLED);
        cv::drawContours(outputImage, contours, static_cast<int>(i), cv::Scalar(0, 255, 0), cv::FILLED);

    }
    cv::erode(outputImage, outputImage, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3)));
    write_debug_img(task, "scratch", outputImage);
    END_TIMER;
    return;
}

void PatrolEdge::NormSobel(int dthr, const cv::Mat& InPutImage, cv::Mat& OutPutImage) {
    OutPutImage = cv::Mat::zeros(InPutImage.size(), CV_8UC1);
    for (int y = 0; y < InPutImage.rows; ++y) {
        for (int x = 0; x < InPutImage.cols; ++x) {
            //int pix1 = InPutImage.at<short>(y, x);
            //if (pix1 > dpow) {
            //    out1.at<uchar>(y, x) = 255;
            //}
            //else {
            //    out1.at<uchar>(y, x) = 0;
            //}
            int pixelValue = std::abs(InPutImage.at<short>(y, x));

            if (pixelValue > dthr) {
                OutPutImage.at<uchar>(y, x) = 255;
            }
            else {
                OutPutImage.at<uchar>(y, x) = 0;
            }
        }
    }
    return;
}