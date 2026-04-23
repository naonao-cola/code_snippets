package com.image;

public class test {
  public static void main(String[] args) {
    ImageProcessor processor = new ImageProcessor();

    // 测试缩放
    boolean resizeOk = processor.resizeImage(
        "E:/demo/java/test01/images/input.jpg",
        "E:/demo/java/test01/images/out_resize.jpg", 800, 600);
    System.out.println("resize: " + resizeOk);

    // 测试灰度
    boolean grayOk =
        processor.toGrayImage("E:/demo/java/test01/images/input.jpg",
                              "E:/demo/java/test01/images/out_gray.jpg");
    System.out.println("gray: " + grayOk);
  }
}
