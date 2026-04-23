
#include "com_image_ImageProcessor.h"
#include "imageLib.h" // 引入你的独立图片库
#include <jni.h>

// ===================== JNI 字符串 → const char* =====================
const char *j2c(JNIEnv *env, jstring str) {
  return env->GetStringUTFChars(str, NULL);
}

void releaseC(JNIEnv *env, jstring str, const char *cstr) {
  env->ReleaseStringUTFChars(str, cstr);
}

// ===================== JNI 接口：只转发调用 =====================
extern "C" JNIEXPORT jboolean JNICALL Java_com_image_ImageProcessor_resizeImage(
    JNIEnv *env, jobject clazz, jstring srcPath, jstring destPath, jint width,
    jint height) {
  // 转 const char*
  const char *src = j2c(env, srcPath);
  const char *dest = j2c(env, destPath);

  // 调用【独立图片处理库】
  bool result = resizeImage(src, dest, width, height);

  // 释放
  releaseC(env, srcPath, src);
  releaseC(env, destPath, dest);

  return result ? JNI_TRUE : JNI_FALSE;
}

extern "C" JNIEXPORT jboolean JNICALL Java_com_image_ImageProcessor_toGrayImage(
    JNIEnv *env, jobject clazz, jstring srcPath, jstring destPath) {
  const char *src = j2c(env, srcPath);
  const char *dest = j2c(env, destPath);

  // 调用【独立图片处理库】
  bool result = toGrayImage(src, dest);

  releaseC(env, srcPath, src);
  releaseC(env, destPath, dest);

  return result ? JNI_TRUE : JNI_FALSE;
}
