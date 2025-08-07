/**
 * @FilePath     : /test02/src/test.cpp
 * @Description  :
 * @Author       : naonao
 * @Date         : 2025-07-08 12:00:40
 * @Version      : 0.0.1
 * @LastEditors  : naonao
 * @LastEditTime : 2025-07-28 11:30:33
 * @Copyright (c) 2025 by G, All Rights Reserved.
 **/

 #include "test.h"
// base 64 编码
static const std::string base64_chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

static inline bool is_base64(const char c)
{
    return ((isalnum(c) != 0) || (c == '+') || (c == '/'));
}

std::string base64_enCode(const char* bytes_to_encode, unsigned int in_len)
{
    std::string ret;
    int i = 0;
    int j = 0;
    unsigned char char_array_3[3];
    unsigned char char_array_4[4];

    while ((in_len--) != 0U) {
        char_array_3[i++] = *(bytes_to_encode++);
        if (i == 3) {
            char_array_4[0] = (char_array_3[0] & 0xfc) >> 2;
            char_array_4[1] = ((char_array_3[0] & 0x03) << 4) + ((char_array_3[1] & 0xf0) >> 4);
            char_array_4[2] = ((char_array_3[1] & 0x0f) << 2) + ((char_array_3[2] & 0xc0) >> 6);
            char_array_4[3] = char_array_3[2] & 0x3f;
            for (i = 0; (i < 4); i++) {
                ret += base64_chars[char_array_4[i]];
            }
            i = 0;
        }
    }
    if (i != 0) {
        for (j = i; j < 3; j++) {
            char_array_3[j] = '\0';
        }
        char_array_4[0] = (char_array_3[0] & 0xfc) >> 2;
        char_array_4[1] = ((char_array_3[0] & 0x03) << 4) + ((char_array_3[1] & 0xf0) >> 4);
        char_array_4[2] = ((char_array_3[1] & 0x0f) << 2) + ((char_array_3[2] & 0xc0) >> 6);
        char_array_4[3] = char_array_3[2] & 0x3f;
        for (j = 0; (j < i + 1); j++) {
            ret += base64_chars[char_array_4[j]];
        }
        while ((i++ < 3)) {
            ret += '=';
        }
    }
    return ret;
}

std::string base64_deCode(std::string const& encoded_string)
{
    int in_len = (int)encoded_string.size();
    int i = 0;
    int j = 0;
    int in_ = 0;
    unsigned char char_array_4[4];
    unsigned char char_array_3[3];
    std::string ret;

    while (in_len-- && (encoded_string[in_] != '=') && is_base64(encoded_string[in_])) {
        char_array_4[i++] = encoded_string[in_];
        in_++;
        if (i == 4) {
            for (i = 0; i < 4; i++) {
                char_array_4[i] = base64_chars.find(char_array_4[i]);
            }

            char_array_3[0] = (char_array_4[0] << 2) + ((char_array_4[1] & 0x30) >> 4);
            char_array_3[1] = ((char_array_4[1] & 0xf) << 4) + ((char_array_4[2] & 0x3c) >> 2);
            char_array_3[2] = ((char_array_4[2] & 0x3) << 6) + char_array_4[3];

            for (i = 0; (i < 3); i++) {
                ret += char_array_3[i];
            }
            i = 0;
        }
    }
    if (i != 0) {
        for (j = i; j < 4; j++) {
            char_array_4[j] = 0;
        }
        for (j = 0; j < 4; j++) {
            char_array_4[j] = base64_chars.find(char_array_4[j]);
        }

        char_array_3[0] = (char_array_4[0] << 2) + ((char_array_4[1] & 0x30) >> 4);
        char_array_3[1] = ((char_array_4[1] & 0xf) << 4) + ((char_array_4[2] & 0x3c) >> 2);
        char_array_3[2] = ((char_array_4[2] & 0x3) << 6) + char_array_4[3];

        for (j = 0; (j < i - 1); j++) {
            ret += char_array_3[j];
        }
    }
    return ret;
}

typedef struct tagBITMAP_FILE_HEADER {
    unsigned short bfType;
    unsigned int bfSize;
    unsigned short bfReserved1;
    unsigned short bfReserved2;
    unsigned int bfOffBits;
} BITMAP_FILE_HEADER;

// code from SaveImageThread
typedef struct tag_BITMAP_INFO_HEADER {
    unsigned int biSize;
    unsigned int biWidth;
    unsigned int biHeight;
    unsigned short biPlanes;
    unsigned short biBitCount;
    unsigned int biCompression;
    unsigned int biSizeImage;
    unsigned int biXPelsPerMeter;
    unsigned int biYPelsPerMeter;
    unsigned int biClrUsed;
    unsigned int biClrImportant;
} BITMAP_INFO_HEADER;

struct ImageBuf {
    unsigned char* rgbBuf;
    int bufLen;
    int width;
    int height;
    int bitCount;
};



void SaveImage(const std::string& save_path, const cv::Mat& img){

    cv::Mat dst;
    cv::flip(img, dst, 0);
    int n_bytes = img.rows * img.cols * img.channels();

    ImageBuf buf;
    buf.rgbBuf = dst.data;
    buf.bufLen = n_bytes;
    buf.height = dst.rows;
    buf.width = dst.cols;

    BITMAP_FILE_HEADER stBfh = { 0 };
    BITMAP_INFO_HEADER stBih = { 0 };
    unsigned long dwBytesRead = 0;
    FILE* file;

    stBfh.bfType = (unsigned short)'M' << 8 | 'B'; // 定义文件类型
    stBfh.bfOffBits = sizeof(BITMAP_FILE_HEADER) + sizeof(BITMAP_INFO_HEADER);
    stBfh.bfSize = stBfh.bfOffBits + buf.bufLen; // 文件大小

    stBih.biSize = sizeof(BITMAP_INFO_HEADER);
    stBih.biWidth = buf.width;
    stBih.biHeight = buf.height;
    stBih.biPlanes = 1;
    stBih.biBitCount = 24;
    stBih.biCompression = 0L;
    stBih.biSizeImage = 0;
    stBih.biXPelsPerMeter = 0;
    stBih.biYPelsPerMeter = 0;
    stBih.biClrUsed = 0;
    stBih.biClrImportant = 0;

    unsigned long dwBitmapInfoHeader = (unsigned long)40UL;

    file = fopen(save_path.c_str(), "wb");
    if (file) {
        fwrite(&stBfh, sizeof(BITMAP_FILE_HEADER), 1, file);
        fwrite(&stBih, sizeof(BITMAP_INFO_HEADER), 1, file);
        fwrite(buf.rgbBuf, buf.bufLen, 1, file);
        fclose(file);
    }
}

void img2dat(cv::Mat src, std::string datName)
{
    // 把mat 转换 为 IplImage 原因是这种图像格式数据访问更为高效
    cv::Mat img = src;
    std::string fileName = datName + ".dat";
    // 图像宽、高、通道、深度 有时我们需要放置一些头信息，因此学会这一点也是很重要的
    int height = img.rows, width = img.cols, depth = img.depth(), channel = img.channels(), dataSize = img.cols * img.rows * img.channels(), widthStep = img.step;

    // 创建dat文件，其中前5*4字节为图像宽、高、通道、深度信息
    std::ofstream outFile(fileName, std::ios::out | std::ios::binary);
    outFile.write((char*)&width, sizeof(width));
    outFile.write((char*)&height, sizeof(height));
    outFile.write((char*)&depth, sizeof(depth));
    outFile.write((char*)&channel, sizeof(channel));
    outFile.write((char*)&dataSize, sizeof(dataSize));
    outFile.write((char*)&widthStep, sizeof(widthStep));
    // 写入图像像素到dat文件
    outFile.write((char*)img.data, dataSize);
    outFile.close();
}
