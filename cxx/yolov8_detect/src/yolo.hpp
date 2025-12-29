#pragma once
#include "NvInfer.h"
#include "cuda_runtime_api.h"
#include "NvInferPlugin.h"
#include "logging.h"
#include <opencv2/opencv.hpp>
#include "utils.h"
#include <string>
using namespace nvinfer1;
using namespace cv;

// stuff we know about the network and the input/output blobs
static const int batchSize = 1;
static const int INPUT_H = 640;
static const int INPUT_W = 640;
static const int _segWidth = 160;
static const int _segHeight = 160;
static const int _segChannels = 32;
static const int CLASSES = 2;
static const int Num_box = 8400;
static const int OUTPUT_SIZE = batchSize * Num_box * (CLASSES + 4);//output
static const int INPUT_SIZE = batchSize * 3 * INPUT_H * INPUT_W;//images


//置信度阈值
static const float CONF_THRESHOLD = 0.5;
//nms阈值
static const float NMS_THRESHOLD = 0.5;

//输入结点名称
const char* INPUT_BLOB_NAME = "images";
//检测头的输出结点名称
const char* OUTPUT_BLOB_NAME = "output";//detect
//分割头的输出结点名称


//定义两个静态浮点，用于保存两个输出头的输出结果
//static float prob[OUTPUT_SIZE];       //box




static MyLogger gLogger;

struct OutputSeg {
	int id;             //结果类别id
	float confidence;   //结果置信度
	cv::Rect box;       //矩形框
};

//中间储存
struct OutputObject
{
	std::vector<int> classIds;//结果id数组
	std::vector<float> confidences;//结果每个id对应置信度数组
	std::vector<cv::Rect> boxes;//每个id矩形框
};

const float color_list[80][3] =
{
	{0.000, 0.447, 0.741},
	{0.850, 0.325, 0.098},
	{0.929, 0.694, 0.125},
	{0.494, 0.184, 0.556},
	{0.466, 0.674, 0.188},
	{0.301, 0.745, 0.933},
	{0.635, 0.078, 0.184},
	{0.300, 0.300, 0.300},
	{0.600, 0.600, 0.600},
	{1.000, 0.000, 0.000},
	{1.000, 0.500, 0.000},
	{0.749, 0.749, 0.000},
	{0.000, 1.000, 0.000},
	{0.000, 0.000, 1.000},
	{0.667, 0.000, 1.000},
	{0.333, 0.333, 0.000},
	{0.333, 0.667, 0.000},
	{0.333, 1.000, 0.000},
	{0.667, 0.333, 0.000},
	{0.667, 0.667, 0.000},
	{0.667, 1.000, 0.000},
	{1.000, 0.333, 0.000},
	{1.000, 0.667, 0.000},
	{1.000, 1.000, 0.000},
	{0.000, 0.333, 0.500},
	{0.000, 0.667, 0.500},
	{0.000, 1.000, 0.500},
	{0.333, 0.000, 0.500},
	{0.333, 0.333, 0.500},
	{0.333, 0.667, 0.500},
	{0.333, 1.000, 0.500},
	{0.667, 0.000, 0.500},
	{0.667, 0.333, 0.500},
	{0.667, 0.667, 0.500},
	{0.667, 1.000, 0.500},
	{1.000, 0.000, 0.500},
	{1.000, 0.333, 0.500},
	{1.000, 0.667, 0.500},
	{1.000, 1.000, 0.500},
	{0.000, 0.333, 1.000},
	{0.000, 0.667, 1.000},
	{0.000, 1.000, 1.000},
	{0.333, 0.000, 1.000},
	{0.333, 0.333, 1.000},
	{0.333, 0.667, 1.000},
	{0.333, 1.000, 1.000},
	{0.667, 0.000, 1.000},
	{0.667, 0.333, 1.000},
	{0.667, 0.667, 1.000},
	{0.667, 1.000, 1.000},
	{1.000, 0.000, 1.000},
	{1.000, 0.333, 1.000},
	{1.000, 0.667, 1.000},
	{0.333, 0.000, 0.000},
	{0.500, 0.000, 0.000},
	{0.667, 0.000, 0.000},
	{0.833, 0.000, 0.000},
	{1.000, 0.000, 0.000},
	{0.000, 0.167, 0.000},
	{0.000, 0.333, 0.000},
	{0.000, 0.500, 0.000},
	{0.000, 0.667, 0.000},
	{0.000, 0.833, 0.000},
	{0.000, 1.000, 0.000},
	{0.000, 0.000, 0.167},
	{0.000, 0.000, 0.333},
	{0.000, 0.000, 0.500},
	{0.000, 0.000, 0.667},
	{0.000, 0.000, 0.833},
	{0.000, 0.000, 1.000},
	{0.000, 0.000, 0.000},
	{0.143, 0.143, 0.143},
	{0.286, 0.286, 0.286},
	{0.429, 0.429, 0.429},
	{0.571, 0.571, 0.571},
	{0.714, 0.714, 0.714},
	{0.857, 0.857, 0.857},
	{0.000, 0.447, 0.741},
	{0.314, 0.717, 0.741},
	{0.50, 0.5, 0}
};






//检测YOLO类
class YOLO
{
public:
	void init(std::string engine_path);
	void init(char* engine_path);
	void destroy();
    void DrawPred(Mat& img, std::vector<OutputSeg> result);
    void doInference(IExecutionContext& context, float* input, float* output, int batchSize);
	void blobFromImage(cv::Mat& img, float* data);
	void decode_boxs(cv::Mat& src, float* prob, OutputObject& outputObject, std::vector<int> padsize);
	void nms_outputs(cv::Mat& src, OutputObject& outputObject, std::vector<OutputSeg>& output);
	void detect_img(std::string image_path);
	void detect_img(std::string image_path, float(*res_array)[6]);

private:
	ICudaEngine* engine;
	IRuntime* runtime;
	IExecutionContext* context;
    float prob[OUTPUT_SIZE];       //box
};

void YOLO::destroy()
{
	this->context->destroy();
	this->engine->destroy();
	this->runtime->destroy();
}


//output中，包含了经过处理的id、conf、box和maskiamg信息
void YOLO::DrawPred(Mat& img, std::vector<OutputSeg> result)
{
	//生成随机颜色
	std::vector<Scalar> color;
	//这行代码的作用是将当前系统时间作为随机数种子，使得每次程序运行时都会生成不同的随机数序列。
	srand(time(0));
	//根据类别数，生成不同的颜色
	for (int i = 0; i < CLASSES; i++) {
		int b = rand() % 256;
		int g = rand() % 256;
		int r = rand() % 256;
		color.push_back(Scalar(b, g, r));
	}

	for (int i = 0; i < result.size(); i++) {
		int left, top;
		left = result[i].box.x;
		top = result[i].box.y;
		int color_num = i;
		//画矩形框，颜色是上面选的
		rectangle(img, result[i].box, color[result[i].id], 2, 8);


		std::string label = std::to_string(result[i].id) + ":" + std::to_string(result[i].confidence);
		int baseLine;
		//获取标签文本的尺寸
		Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
		//确定一个最大的高
		top = max(top, labelSize.height);
		//把文本信息加到图像上
		putText(img, label, Point(left, top), FONT_HERSHEY_SIMPLEX, 1, color[result[i].id], 2);
	}

}


//输入引擎文本、图像数据、定义的检测输出和分割输出、batchSize
void YOLO::doInference(IExecutionContext& context, float* input, float* output, int batchSize)
{
	//从上下文中获取一个CUDA引擎。这个引擎加载了一个深度学习模型
	const ICudaEngine& engine = context.getEngine();

	//判断该引擎是否有三个绑定，intput, output0, output1
	assert(engine.getNbBindings() == 2);
	//定义了一个指向void的指针数组，用于存储GPU缓冲区的地址
	void* buffers[2];

	//获取输入和输出blob的索引，这些索引用于之后的缓冲区操作
	const int inputIndex = engine.getBindingIndex(INPUT_BLOB_NAME);
	const int outputIndex = engine.getBindingIndex(OUTPUT_BLOB_NAME);


	// 使用cudaMalloc分配了GPU内存。这些内存将用于存储模型的输入和输出
	CHECK(cudaMalloc(&buffers[inputIndex], INPUT_SIZE * sizeof(float)));//
	CHECK(cudaMalloc(&buffers[outputIndex], OUTPUT_SIZE * sizeof(float)));

	// cudaMalloc分配内存 cudaFree释放内存 cudaMemcpy或 cudaMemcpyAsync 在主机和设备之间传输数据
	// cudaMemcpy cudaMemcpyAsync 显式地阻塞传输 显式地非阻塞传输
	//创建一个CUDA流。CUDA流是一种特殊的并发执行环境，可以在其中安排任务以并发执行。流使得任务可以并行执行，从而提高了GPU的利用率。
	cudaStream_t stream;
	//判断是否创建成功
	CHECK(cudaStreamCreate(&stream));

	// 使用cudaMemcpyAsync将输入数据异步地复制到GPU缓冲区。这个操作是非阻塞的，意味着它不会立即完成。
	CHECK(cudaMemcpyAsync(buffers[inputIndex], input, INPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice, stream));
	//将输入和输出缓冲区以及流添加到上下文的执行队列中。这将触发模型的推理。
	//context.enqueue(batchSize, buffers, stream, nullptr);
	context.enqueueV2(buffers, stream, nullptr);

	//使用cudaMemcpyAsync函数将GPU上的数据复制到主内存中。这是异步的，意味着该函数立即返回，而数据传输可以在后台进行。
	CHECK(cudaMemcpyAsync(output, buffers[outputIndex], OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream));

	//等待所有在给定流上的操作都完成。这可以确保在释放流和缓冲区之前，所有的数据都已经被复制完毕。
	//这对于保证内存操作的正确性和防止数据竞争非常重要。
	cudaStreamSynchronize(stream);

	//释放内存
	cudaStreamDestroy(stream);
	CHECK(cudaFree(buffers[inputIndex]));
	CHECK(cudaFree(buffers[outputIndex]));

}


void YOLO::init(std::string engine_path)
{
	//无符号整型类型，通常用于表示对象的大小或计数
	//{ 0 }: 这是初始化列表，用于初始化 size 变量。在这种情况下，size 被初始化为 0。
	size_t size{ 0 };
	//定义一个指针变量，通过trtModelStream = new char[size];分配size个字符的空间
	//nullptr表示指针针在开始时不指向任何有效的内存地址，空指针
	char* trtModelStream{ nullptr };
	//打开文件，即engine模型
	std::ifstream file(engine_path, std::ios::binary);
	if (file.good())
	{
		//指向文件的最后地址
		file.seekg(0, file.end);
		//计算文件的长度
		size = file.tellg();
		//指回文件的起始地址
		file.seekg(0, file.beg);
		//为trtModelStream指针分配内存，内存大小为size
		trtModelStream = new char[size]; //开辟一个char 长度是文件的长度
		assert(trtModelStream);
		//把file内容传递给trtModelStream，传递大小为size，即engine模型内容传递
		file.read(trtModelStream, size);
		//关闭文件
		file.close();
	}
	std::cout << "engine init finished" << std::endl;

	//创建了一个Inference运行时环境，返回一个指向新创建的运行时环境的指针
	runtime = createInferRuntime(gLogger);
	assert(runtime != nullptr);
	//反序列化一个CUDA引擎。这个引擎将用于执行模型的前向传播
	engine = runtime->deserializeCudaEngine(trtModelStream, size);
	assert(engine != nullptr);
	//使用上一步中创建的引擎创建一个执行上下文。这个上下文将在模型的前向传播期间使用
	context = engine->createExecutionContext();
	assert(context != nullptr);
	//释放了用于存储模型序列化的内存
	delete[] trtModelStream;

	 for (int i = 0; i < engine->getNbBindings(); ++i) {
        if (engine->bindingIsInput(i))   // 关键判断
        {
            auto dims = engine->getBindingDimensions(i);
            bool ok   = context->setBindingDimensions(i, dims);
            if (!ok) {
                std::cerr << "setBindingDimensions failed on input #" << i << " name=" << engine->getBindingName(i) << std::endl;
                return;
            }
        }
    }

}

void YOLO::init(char* engine_path)
{
	//无符号整型类型，通常用于表示对象的大小或计数
	//{ 0 }: 这是初始化列表，用于初始化 size 变量。在这种情况下，size 被初始化为 0。
	size_t size{ 0 };
	//定义一个指针变量，通过trtModelStream = new char[size];分配size个字符的空间
	//nullptr表示指针针在开始时不指向任何有效的内存地址，空指针
	char* trtModelStream{ nullptr };
	//打开文件，即engine模型
	std::ifstream file(engine_path, std::ios::binary);
	if (file.good())
	{
		//指向文件的最后地址
		file.seekg(0, file.end);
		//计算文件的长度
		size = file.tellg();
		//指回文件的起始地址
		file.seekg(0, file.beg);
		//为trtModelStream指针分配内存，内存大小为size
		trtModelStream = new char[size]; //开辟一个char 长度是文件的长度
		assert(trtModelStream);
		//把file内容传递给trtModelStream，传递大小为size，即engine模型内容传递
		file.read(trtModelStream, size);
		//关闭文件
		file.close();
	}
	std::cout << "engine init finished" << std::endl;

	//创建了一个Inference运行时环境，返回一个指向新创建的运行时环境的指针
	runtime = createInferRuntime(gLogger);
	assert(runtime != nullptr);
	//反序列化一个CUDA引擎。这个引擎将用于执行模型的前向传播
	engine = runtime->deserializeCudaEngine(trtModelStream, size);
	assert(engine != nullptr);
	//使用上一步中创建的引擎创建一个执行上下文。这个上下文将在模型的前向传播期间使用
	context = engine->createExecutionContext();
	assert(context != nullptr);
	//释放了用于存储模型序列化的内存
	delete[] trtModelStream;

	// 只给输入绑定设维度，输出一律不动
    for (int i = 0; i < engine->getNbBindings(); ++i) {
        if (engine->bindingIsInput(i))   // 关键判断
        {
            auto dims = engine->getBindingDimensions(i);
            bool ok   = context->setBindingDimensions(i, dims);
            if (!ok) {
                std::cerr << "setBindingDimensions failed on input #" << i << " name=" << engine->getBindingName(i) << std::endl;
                return;
            }
        }
    }

}

void YOLO::blobFromImage(cv::Mat& src, float* data)
{
	//定义一个浮点数组
	//float* data = new float[3 * INPUT_H * INPUT_W];

	int i = 0;// [1,3,INPUT_H,INPUT_W]
	for (int row = 0; row < INPUT_H; ++row)
	{
		//逐行对象素值和图像通道进行处理
		//pr_img.step=widthx3 就是每一行有width个3通道的值
		//第row行
		uchar* uc_pixel = src.data + row * src.step;
		for (int col = 0; col < INPUT_W; ++col)
		{
			//第col列
			//提取第第row行第col列数据进行处理
			//像素值处理
			data[i] = (float)uc_pixel[2] / 255.0;
			//通道变换
			data[i + INPUT_H * INPUT_W] = (float)uc_pixel[1] / 255.0;
			data[i + 2 * INPUT_H * INPUT_W] = (float)uc_pixel[0] / 255.0;
			uc_pixel += 3;//表示进行下一列
			++i;//表示在3个通道中的第i个位置，rgb三个通道的值是分开的，如r123456g123456b123456
		}
	}

	//return data;
}

void YOLO::decode_boxs(cv::Mat& src, float* prob, OutputObject& outputObject, std::vector<int> padsize)
{
	int newh = padsize[0], neww = padsize[1], padh = padsize[2], padw = padsize[3];

	float ratio_h = (float)src.rows / newh;
	float ratio_w = (float)src.cols / neww;


 // 直接解析8400个框，每个框6个值 [cx, cy, w, h, conf, cls]
    for (int i = 0; i < Num_box; ++i) {
        float* pitem = prob + i * 6;
        float conf = pitem[4];

        if (conf < CONF_THRESHOLD) continue;

        int class_id = (int)pitem[5];
        if (class_id >= CLASSES) continue;

        // 解码box坐标
        float cx = (pitem[0] - padw) * ratio_w;
        float cy = (pitem[1] - padh) * ratio_h;
        float w = pitem[2] * ratio_w;
        float h = pitem[3] * ratio_h;

        int left = MAX(int(cx - w * 0.5f), 0);
        int top = MAX(int(cy - h * 0.5f), 0);
        int width = int(w);
        int height = int(h);

        if (width <= 0 || height <= 0) continue;

        outputObject.classIds.push_back(class_id);
        outputObject.confidences.push_back(conf);
        outputObject.boxes.push_back(Rect(left, top, width, height));
    }
}

void YOLO::nms_outputs(cv::Mat& src, OutputObject& outputObject, std::vector<OutputSeg>& output)
{
	//执行非最大抑制以消除具有较低置信度的冗余重叠框（NMS）
	std::vector<int> nms_result;
	//通过opencv自带的nms函数进行，矩阵box、置信度大小，置信度阈值，nms阈值，结果
	cv::dnn::NMSBoxes(outputObject.boxes, outputObject.confidences, CONF_THRESHOLD, NMS_THRESHOLD, nms_result);

	//包括类别、置信度、框和mask
	//std::vector<std::vector<float>> temp_mask_proposals;
	//创建一个名为holeImgRect的Rect对象
	Rect holeImgRect(0, 0, src.cols, src.rows);
	//提取经过非极大值抑制后的结果
	for (int i = 0; i < nms_result.size(); ++i) {
		int idx = nms_result[i];
		OutputSeg result;
		result.id = outputObject.classIds[idx];
		result.confidence = outputObject.confidences[idx];
		result.box = outputObject.boxes[idx] & holeImgRect;
		output.push_back(result);

	}
}

//读取图片进行推理
void YOLO::detect_img(std::string image_path)
{
	cv::Mat img = cv::imread(image_path);

	//图像预处理，输入的是原图像和网络输入的高和宽，填充尺寸容器
	//输出的是重构后的图像，以及每条边填充的大小保存在padsize
	cv::Mat pr_img;
	std::vector<int> padsize;
	pr_img = preprocess_img(img, INPUT_H, INPUT_W, padsize);       // Resize

	float* blob = new float[3 * INPUT_H * INPUT_W];
    blobFromImage(pr_img, blob);

	//推理
	auto start = std::chrono::system_clock::now();
	doInference(*context, blob, prob, batchSize);
	auto end = std::chrono::system_clock::now();
	std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;

	//解析并绘制output
	OutputObject outputObject;
	std::vector<OutputSeg> output;
	std::vector<std::vector<float>> temp_mask_proposals;
	decode_boxs(img, prob, outputObject, padsize);
	nms_outputs(img, outputObject, output);
	std::cout << "后处理时间：" << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;

	DrawPred(img, output);
	// cv::imshow("output.jpg", img);
	cv::imwrite("/home/nvidia/wangw/demo/yolo_test/data/output.jpg", img);
	// cv::waitKey(0);

    delete[] blob;
}

/// <summary>
/// 读取图片进行推理，用数组将检测结果传出（包括label序号、置信度分数、矩形参数）
/// </summary>
/// <param name="image_path"></param>
/// <param name="res_array"></param>
void YOLO::detect_img(std::string image_path, float(*res_array)[6])
{
	cv::Mat img = cv::imread(image_path);


	//图像预处理，输入的是原图像和网络输入的高和宽，填充尺寸容器
	//输出的是重构后的图像，以及每条边填充的大小保存在padsize
	cv::Mat pr_img;
	std::vector<int> padsize;
	pr_img = preprocess_img(img, INPUT_H, INPUT_W, padsize);       // Resize

	float* blob = new float[3 * INPUT_H * INPUT_W];
    blobFromImage(pr_img, blob);

	//推理
	auto start = std::chrono::system_clock::now();
	doInference(*context, blob, prob, batchSize);
	auto end = std::chrono::system_clock::now();
	std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;

	//解析并绘制output
	OutputObject outputObject;
	std::vector<OutputSeg> output;
	std::vector<std::vector<float>> temp_mask_proposals;
	decode_boxs(img, prob, outputObject, padsize);
	nms_outputs(img, outputObject, output);


	//传出数据
	for (size_t j = 0; j < output.size(); j++)
	{
		res_array[j][0] = output[j].box.x;
		res_array[j][1] = output[j].box.y;
		res_array[j][2] = output[j].box.width;
		res_array[j][3] = output[j].box.height;
		res_array[j][4] = output[j].id;
		res_array[j][5] = output[j].confidence;
		//mask_array = output[j].boxMask.data;

	}

	delete[] blob;
}