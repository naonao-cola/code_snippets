package com.image;

public class ImageProcessor {
  static {
    // 加载 C++ 编译的动态库
    // Windows: imageprocessor.dll
    // Linux: libimageprocessor.so
    // Mac: libimageprocessor.dylib
    System.loadLibrary("imageprocessor");
  }

  // 1. 图片缩放
  public native boolean resizeImage(String srcPath, String destPath, int width,
                                    int height);

  // 2. 转灰度图
  public native boolean toGrayImage(String srcPath, String destPath);
}
