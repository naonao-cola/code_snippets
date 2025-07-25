
#include <chrono>
#include <iostream>
#include <omp.h>
#include <random>
#include <vector>
#include <omp.h>

#include "utils.h"
namespace utils {

float CalculateOverlap(float xmin0, float ymin0, float xmax0, float ymax0, float xmin1, float ymin1, float xmax1, float ymax1)
{
    float w = fmax(0.f, fmin(xmax0, xmax1) - fmax(xmin0, xmin1) + 1.0);
    float h = fmax(0.f, fmin(ymax0, ymax1) - fmax(ymin0, ymin1) + 1.0);
    float i = w * h;
    float u = (xmax0 - xmin0 + 1.0) * (ymax0 - ymin0 + 1.0) + (xmax1 - xmin1 + 1.0) * (ymax1 - ymin1 + 1.0) - i;
    return u <= 0.f ? 0.f : (i / u);
}

#include <omp.h>
int nms(int validCount,
    std::vector<float>& outputLocations,
    std::vector<int> classIds,
    std::vector<int>& order,
    int filterId,
    float threshold,
    const int max_det)
{
    int valid_obj_nums = 0;

 //#pragma omp parallel for collapse(2)
    for (int i = 0; i < validCount; ++i) {
        if (order[i] == -1 || classIds[i] != filterId) {
            continue;
        }
        int n = order[i];
        // #pragma omp for
        for (int j = i + 1; j < validCount; ++j) {
            int m = order[j];
            if (m == -1 || classIds[i] != filterId) {
                continue;
            }
            float xmin0, ymin0, xmax0, ymax0, xmin1, ymin1, xmax1, ymax1, iou;

            xmin0 = outputLocations[n * 4 + 0];
            ymin0 = outputLocations[n * 4 + 1];
            xmax0 = outputLocations[n * 4 + 0] + outputLocations[n * 4 + 2];
            ymax0 = outputLocations[n * 4 + 1] + outputLocations[n * 4 + 3];
            xmin1 = outputLocations[m * 4 + 0];
            ymin1 = outputLocations[m * 4 + 1];
            xmax1 = outputLocations[m * 4 + 0] + outputLocations[m * 4 + 2];
            ymax1 = outputLocations[m * 4 + 1] + outputLocations[m * 4 + 3];
            iou = CalculateOverlap(xmin0, ymin0, xmax0, ymax0, xmin1, ymin1, xmax1, ymax1);
            if (iou > threshold) {
// #pragma omp critical
                {
                    order[j] = -1;
                }
            }
        }
    }
    return 0;
}

// 生成模拟数据
void generateRandomData(int count, std::vector<float>& outputLocations, std::vector<float>& prob_vec,std::vector<int>& classIds)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> loc_dist(0.0f, 4024.0f);

    std::uniform_int_distribution<int> class_dist(0, 13);

    std::uniform_real_distribution<float> len_dist(50.0f, 100.0f);

    std::uniform_real_distribution<float> prob_dist(0.6f, 1.0f);

    outputLocations.resize(count * 4);
    classIds.resize(count);
    prob_vec.resize(count);

    for (int i = 0; i < count; ++i)
    {
        // 随机生成候选框的左上角坐标 (x,y) 和宽高 (w,h)
        outputLocations[i * 4 + 0] = loc_dist(gen); // xmin
        outputLocations[i * 4 + 1] = loc_dist(gen); // ymin
        outputLocations[i * 4 + 2] = len_dist(gen); // width
        outputLocations[i * 4 + 3] = len_dist(gen); // height
        prob_vec[i] = prob_dist(gen);
        // 随机生成类别
        classIds[i]= class_dist(gen);
    }
}


void LoadImagePath(std::string imgDirPath, std::vector<std::string>& vimgPath)
{

    DIR* pDir;
    struct dirent* ptr;
    if (!(pDir = opendir(imgDirPath.c_str()))) {
        std::cout << "Folder doesn't Exist! " << imgDirPath << std::endl;
        return;
    } else {
        std::cout << "Read " << imgDirPath << " succeed." << std::endl;
    }

    while ((ptr = readdir(pDir)) != 0) {
        if (strcmp(ptr->d_name, ".") != 0 && strcmp(ptr->d_name, "..") != 0) {
            vimgPath.push_back(imgDirPath + "/" + ptr->d_name);
        }
    }
    sort(vimgPath.begin(), vimgPath.end());

    closedir(pDir);
}

} // namespace utils
