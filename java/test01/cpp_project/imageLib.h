
#ifndef IMAGE_LIB_H
#define IMAGE_LIB_H

// 对外暴露 C 接口，全用 const char*
#ifdef __cplusplus
extern "C" {
#endif

// 图片缩放（const char* 路径）
bool resizeImage(const char *srcPath, const char *destPath, int width,
                 int height);

// 转灰度图
bool toGrayImage(const char *srcPath, const char *destPath);

#ifdef __cplusplus
}
#endif

#endif
