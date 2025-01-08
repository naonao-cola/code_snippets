#pragma once
#include "ResultBase.h"
#include "Geometry.h"
#include "CommonDefine.h"
#include "JsonHelper.h"

namespace Tival
{
    class ExportAPI DataCodeResult : public ResutBase
    {
    public:
        int candidate_num = 0;      // 候选数
        int undecoded_num = 0;      // 未解码数

        std::vector<std::string> data_strings;  // 每个二维码对应的文本结果
        std::vector<TRotateRect> regions;       // 二维码旋转矩形框
        std::vector<int> symbol_rows;       // 符号行数
        std::vector<int> symbol_cols;       // 符号列数
        std::vector<int> reflections;       // 是否镜像反转

        // QRCode
        std::vector<int> versions;          // QRcode版本
        std::vector<int> model_types;       // QRcode model type
        std::vector<int> correction_levels; // QRCode纠错等级
        std::vector<int> symbol_sizes;      // QRCode符号数


        virtual json ToJson() const;
        TLine ToTLine();
    };


    class ExportAPI DataCode2d
    {
    public:
        DataCode2d() {};
        virtual ~DataCode2d() {};

        /**
         * 搜索DataMatrix码
         * @params：
         * - Num: 搜索个数 (N:搜到N个后停止搜索，def:1)
         * - Intensity: 搜索强度（[1:快速，2:标准，3:加强] def:3）
         * - Polarity: 极性（0: 不限，1：白在黑上，2：黑在白上，def:0）
         * - Timeout: 搜索超时(ms)设置, 超过该时间则终止搜索（-1：无限等待，0~N: timeout时间，def: -1）
         *
         * - SymbolRows: 符号行数, 不确定可以不填
         * - SymbolCols: 符号列数, 不确定可以不填
         * - SymbolRowsMin: 最小符号行数(偶数 def: 8)
         * - SymbolRowsMax: 最大符号行数(偶数 def: 144)
         * - SymbolColsMin: 最小符号列数(偶数 def: 10)
         * - SymbolColsMax: 最大符号列数(偶数 def: 144)
         * - SlantAngleMax: 最大倾斜角度（def: 10）
        */
        static DataCodeResult FindDataDM(const cv::Mat& image, const json& params);
        static DataCodeResult FindDataDM(void* timage, const json& params);

        /**
         * 搜索DataMatrix码
         * @params：
         * - Num: 搜索个数 (N:搜到N个后停止搜索，def:1)
         * - Intensity: 搜索强度（[0:快速，1:标准，2:加强] def:2）
         * - Polarity: 极性（1：白在黑上，2：黑在白上，def:1）
         * - Timeout: 搜索超时(ms)设置, 超过该时间则终止搜索（-1：无限等待，0~N: timeout时间，def: -1）
         * - ModelType: QRCode模型类型（0：不限，1：旧版Model, 2: 新版Model， def:0）
         * - VersionMin: 最小版本（新版Model:1~40, 旧版Model:1~14, def:1）
         * - VersionMax: 最大版本（新版Model:1~40, 旧版Model:1~14, def:40）
         * - Version: 指定只搜索特定Version（新版Model:1~40, 旧版Model:1~14，def:-1，任意)
         *
         * - SymbolRows: 符号行数, 不确定可以不填
         * - SymbolRowsMin: 最小符号行数(def: 8)
         * - SymbolRowsMax: 最大符号行数(def: 100)
         * - SymbolCols: 符号列数, 不确定可以不填
         * - SymbolColsMin: 最小符号列数：(def: 10)
         * - SymbolColsMax: 最大符号列数：(def: 100)
         * - SlantAngleMax: 最大倾斜角度（def: 10）
        */
        static DataCodeResult FindQRCode(const cv::Mat& image, const json& params);
        static DataCodeResult FindQRCode(void* timage, const json& params);

        // static DataCodeResult FindBarCode(const cv::Mat& image, const json& params);
        // static DataCodeResult FindBarCode(void* timage, const json& params);
    };
}


