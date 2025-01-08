#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32)
#   define __export         __declspec(dllexport)
#elif defined(__GNUC__) && ((__GNUC__ >= 4) || (__GNUC__ == 3 && __GNUC_MINOR__ >= 3))
#   define __export         __attribute__((visibility("default")))
#else
#   define __export
#endif

/*! calculate add(a, b)
 *
 * @param a     the first argument
 * @param b     the second argument
 *
 * @return      the result
 */
__export int tapp_model_package(const char *model_path, char *origin_model_dir);

__export int* tapp_model_open(const char *model_path, int device_id);
__export void tapp_model_config(int *handle, const char *config_json_str);
__export const char* tapp_model_run(int *handle, const char *in_param_json_str);
__export void tapp_model_close(int *handle);

#ifdef __cplusplus
}
#endif
