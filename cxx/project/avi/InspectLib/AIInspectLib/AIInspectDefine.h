#pragma once
enum AI_INSPECTION_PARAM
{
	AI_INSPECTION_PARAM_MODELID = 0,					//  The AI model's ID.
	AI_INSPECTION_PARAM_MAXBATCHSIZE,					//  Inference Maximum batchsize.
	AI_INSPECTION_PARAM_CONFIDENCETHR,					//  Confidence threshold..
	AI_INSPECTION_PARAM_NMSTHR,							//  NMS threshold.
	AI_INSPECTION_PARAM_MAXOBJECTNUMS,					//  Maximum detection count.
	AI_INSPECTION_PARAM_USEPINMEM,						//	Indicates whether to utilize pin-memory			0: Not utilize	   1. Utilize Pin-memory  
	AI_INSPECTION_PARAM_WORKSPACESIZE,					//  The size of the workspace in TensorRT
	AI_INSPECTION_PARAM_GPUCACHSIZE,					//  The initial size of the buffer used for data transfer on the GPU.
	AI_INSPECTION_PARAM_CPUCACHSIZE,					//  The initial size of the buffer used for data transfre on the CPU.
	AI_INSPECTION_PARAM_MODELVERSION,					//  The model's version.
	AI_INSPECTION_PARAM_PREPROCESSTHREADCNT,			//  The number of the pre-processing threads.
	AI_INSPECTION_PARAM_PREPROCESSTHREADPRI,			//  The priority of the pre-processing threads.
	AI_INSPECTION_PARAM_HOSTPROCESSTHREADCNT,			//  The number of the host-processing threads. 			
	AI_INSPECTION_PARAM_HOSTPROCESSTHREADPRI,			//  The priority of the host-processing threads.
	AI_INSPECTION_PARAM_ALGOTYPE,						//  Types of AI algorithms.							0: Classfication  1: Object Detection  2: segmentations
	AI_INSPECTION_PARAM_GPUINDEX,						//  GPU id.
	AI_INSPECTION_PARAM_AIALGO,							//  Indicates whether it is an AI algorithm.		0: cv algorithm,  1: AI algorithm.

};
