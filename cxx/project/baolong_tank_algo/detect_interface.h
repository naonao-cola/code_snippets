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

	/*
	加载许可证
	*/
	__export const char* get_hardware_id();

	__export int tapp_license_verify();

	// /* 算法对象初始化
	// */
	__export int* tapp_model_init();


	/* 模型配置：模型固定参数，模板等配置
	*  参数 config_json_str: 模型初始化json参数
	*  [
		{	
			"type_id": "2200000",			// 图片类型ID，比如不同相机拥有不同的type_id//相机序列号
	 		"device_id": gpu id,			//无gpu 默认0
			"paras": {						//模型参数
					"detect-classes"："O_RING",//C_RING //CO //PLUG
					"model_path": "模型路径"，//默认null
					"label_path": "标签文件路径"，
			},
			"template": {
				"img_path":	"模板参考图路径"，
				"shapes": [{
					"label": "mark_a",
					"points": [[3760, 2656],[3913, 2748]],
					"shape_type": "rectangle",
		         },
		        {
					"label": "mark_b",
					"points": [[3760, 2656],[3913, 2748]],
					"shape_type": "rectangle",
		         },
		       ]
			}
			
		}
	   ]
	*/
/*
out json
{
	"label":
}
*/
	__export void tapp_model_config(int* handle, const char* config_json_str);


	/* 执行算法推理
	*  参数 in_param_json_str: 运行参数数组，每个item代表一张图片
	*  [
	*	{
	*		"type_id": "1",					// 图片类型ID，比如不同相机拥有不同的type_id
	*		"img_name": "uuid"					// 唯一标识图片的uuid，用于获取返回结果
	*		"img_path" : "./data/hbz_b1/9a.jpg",    // 离线图片路径 （和图片指针二选一）
	*       "img_w" : 512,                          // 图片宽（图片指针情况下需要）
	*       "img_h" : 512,                          // 图片高（图片指针情况下需要）
	*		"channel":3,                            // 图像通道数（图片指针情况下需要）
	*      }
	*    ]
	*   返回值：
	*	  {
	*		"class_list": "O_RING",
	*		"label_set" :
	*		 [
	*			{
	*				"img_name": "UUID",
	*				"shapes" : [
	*					{"label":"NG2", "shapeType" : "polygon", "points" : [[2097.0, 2147.0], [2414.0, 2143.0], [2415.0, 2234.0], [2098.0, 2238.0]], "result" : {"text":"HB50803", "type" : 1}},
	*					{"label":"NG3", "shapeType" : "polygon", "points" : [[2097.0,2146.0],[2414.0,2143.0],[2415.0,2234.0],[2098.0,2237.0]] ,"result" : {"area":28847,"confidence" : 1.0} }
	*				],
	*				"status" : "OK"//NG
	*			}
	*		]
	*	}
	*/
	__export const char* tapp_model_run(int* handle, unsigned char** data, const char* in_param_json_str);//


	/* 销毁算法模型
	*/
	__export void tapp_model_destroy(int* handle);

	__export int tapp_model_package(int* handle, const char *model_path, char *origin_model_dir, char *model_key);

	__export void tapp_model_open(int* handle, const char *model_path, int device_id);



#ifdef __cplusplus
}
#endif