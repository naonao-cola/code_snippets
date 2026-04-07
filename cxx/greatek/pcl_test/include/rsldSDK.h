#pragma once
#include <vector>    // 必须包含 vector 头文件
#include <iostream>
#include "type.h"

#ifdef RSLD_EXPORTS
#define RSLD_API __declspec(dllexport)   // 编译 DLL 时定义
#else
#define RSLD_API __declspec(dllimport)   // 使用 DLL 时定义
#endif


// C-style export for Java or other language interop
extern "C" {
	/**
	 * @brief:创建设备句柄.
	 *
	 * @param[in]：depthrow-深度图行数.
	 * @param[in]：depthcol-深度图列数.
	 * @return:设备句柄
	 */
	RSLD_API void* CreateInterface(int depthrow = 600, int depthcol = 800);
	/**
	 * @brief:销毁设备句柄.
	 */
	RSLD_API void DestroyInterface(void* ptr);
	/**
	* @brief:开启设备数据.
	* @return
	*       SUCCESS:true
	*       FAILURE:false.
	*/
	RSLD_API bool StartRecv(void* ptr);
	/**
	* @brief:关闭设备.
	* @return
	*       SUCCESS:true
	*       FAILURE:false.
	*/
	RSLD_API bool StopRecv(void* ptr);
	/**
	 * @brief:读取原始的深度图和RGB图.
	 *
	 * @param[in-out]：imgColor-获取到的RGB图.
	 * @param[in-out]：imgDepth-获取到的深度图.
	 */
	RSLD_API void getColorVsDepth(void* ptr, ImageData* imgColor, ImageData* imgDepth);
	/**
	 * @brief:读取配准后的深度图和RGB图
	 *
	 * @param[in-out]：imgColor-获取到的RGB图.
	 * @param[in-out]：imgDepth-获取到的深度图.
	 */
	RSLD_API void getColorVsDepthRegister(void* ptr, ImageData* imgColor, ImageData* imgDepth);

	/**
	* @brief:设置激光开启关闭.
	*
	* @param[in]isEnable-激光开关.
	* @return
	*       SUCCESS:true
	*       FAILURE:false.
	*/
	RSLD_API bool laserEnable(void* ptr, bool isEnable);

	/**
	* @brief:设置雷达IP/设置当前通信雷达IP.
	*/
	RSLD_API void setTargetIp(void* ptr, std::string lidar_ip, std::string pc_ip);

	/**
	* @brief:获取点云集合.
	*
	* @param[in-out]：points-场景的点集合.
	*/
	RSLD_API void  getPointsData(void* ptr, std::vector<pointxyzrgb>& points);

	/**
	* @brief:获取点云数据指针.
	*
	* @param[in-out]：points-场景的点数据的指针.
	*/
	RSLD_API uint32_t getPointsDataPtr(void* ptr, float* datas);

	/**
	* @brief:获取幅度图.
	*
	* @param[in-out]：imgColorAmplitude-幅度图.
	*
	* @return:frame id .
	*/
	RSLD_API uint32_t getAmplitude(void* ptr, ImageData* imgColorAmplitude);


	/**
	* @brief:获取目标物点云集合.
	*
	* @param[in]objPixes-目标彩色像素坐标集合.
	*
	* @param[in]depth-深度图.
	*
	* @param[in]xmlpath-参数文件路径.
	*
	* @param[in-out]：points-目标空间点集合.
	*
	* @param[in-out]：RGBD-彩色图对应像素点的空间点坐标xyz组成的图.
	*
	* @param[in-out]：rgbindexes-points点对应的彩色图像素坐标
	*/
	RSLD_API void getObjectPoints(void* ptr, const std::vector<Point2i>objPixes,
		const ImageData& depth,
		std::string xmlpath, std::vector<pointxyzrgb>& points,
		ImageData* RGBD,
		std::vector<Point2i>& rgbindexes);

}