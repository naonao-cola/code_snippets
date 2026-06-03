import tensorrt as trt
import pycuda.driver as cuda
import numpy as np
import pycuda.autoinit
import onnxruntime as ort
from enum import Enum
from abc import abstractmethod


class ModelType(Enum):
    ONNX = 1
    TENSORRT = 2


class ModelWrapper:
    def __init__(self, model_path):
        self.model_type = None
        self.engine = None  # 推理模型
        self.load(model_path)


    def load(self, model_path):
        if model_path.endswith("onnx"):
            self.model_type = ModelType.ONNX
            self.engine = ort.InferenceSession(model_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
            self.model_input = self.engine.get_inputs()

        elif model_path.endswith("engine") or model_path.endswith("trt"):
            self.model_type = ModelType.TENSORRT
            self.stream = None
            with open(model_path, "rb") as f:
                self.runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
                self.engine = self.runtime.deserialize_cuda_engine(f.read())
                self.context = self.engine.create_execution_context()

            self.dtype_map = {
                    0: np.float32,  # DataType.FLOAT
                    1: np.int32,    # DataType.INT32
                    2: np.float16,  # DataType.HALF
                    # 其他类型也可以添加到这个字典
                }
        
    
    @abstractmethod
    def preprocess(self, ori_image):
        """
        return : tuple  # [img, *arg]
        """
        return (ori_image,)
    

    def run(self, ori_image):
        
        if self.model_type == ModelType.ONNX:
            input_names = [input.name for input in self.engine.get_inputs()]
            preprocess_info = self.preprocess(ori_image)
            inputs = self.get_input(preprocess_info, input_names)
            outputs = self.engine.run(None, inputs)
            return self.postprocess(outputs)  # output image

        elif self.model_type == ModelType.TENSORRT:
            input_names = [self.engine.get_binding_name(i) for i in range(self.engine.num_bindings) if self.engine.binding_is_input(i)]
            output_names = [self.engine.get_binding_name(i) for i in range(self.engine.num_bindings) if not self.engine.binding_is_input(i)]
            
            preprocess_info = self.preprocess(ori_image)
            inputs = self.get_input(preprocess_info, input_names)

            if self.stream is None:
                self.allocate_memory()  # 先分配GPU内存
            
            # 为输入数据设置GPU内存地址
            for i, input_name in enumerate(input_names):
                input_data = inputs[input_name]
                cuda.memcpy_htod_async(self.d_inputs[input_name], input_data.ravel(), self.stream) # 将预处理数据拷贝到GPU

            # 执行推理
            self.context.execute_async_v3(self.stream.handle)
            
            # 从GPU内存中获取输出
            outputs = []
            for output_name in output_names:
                shape = self.engine.get_binding_shape(self.engine.get_binding_index(output_name))
                dtype = self.engine.get_binding_dtype(self.engine.get_binding_index(output_name))
                np_dtype = self.dtype_map.get(dtype, np.float32)  # 默认类型，如果需要，可以根据 dtype 进行映射
                
                # 创建 NumPy 数组，作为输出
                # size = np.prod(shape) * np.dtype(np_dtype).itemsize  # 计算数组总大小
                output_data = np.empty(shape, dtype=np_dtype)  # 创建一个与设备分配相同形状的 NumPy 数组

                # 将数据从 GPU 异步传输到 CPU
                cuda.memcpy_dtoh_async(output_data, self.d_outputs[output_name], self.stream)
                outputs.append(output_data)
            
            # 后处理并返回结果
            return self.postprocess(outputs)
        else:
            assert False, "Currently, only engine and onnx are supported."


    @abstractmethod
    def postprocess(self, model_output):
        """
        return : [[x, y, w, h, class, score],...]
        """
        return 
    

    def get_input(self, preprocess_info, input_names):
        """
        根据预处理信息和输入名称，获取输入数据并返回一个字典。

        参数:
        preprocess_info (list): 预处理后的数据列表，包含了所有输入数据。
        input_names (list): 输入数据的名称列表，对应于每个输入数据的名称。

        返回:
        dict: 一个字典，其中键是输入名称，值是对应的预处理数据。
        """
        input_info = {}
        for i, input_data in enumerate(preprocess_info):
            if i < len(input_names):  # 确保索引不超过输入名称的数量
                input_info[input_names[i]] = input_data
        return input_info

    
    def allocate_memory(self):
        """
        tensorrt 分配 GPU 内存
        """
        # 获取输入输出张量的数量和名称
        input_count = self.engine.num_bindings  # 总的输入输出数量
        tensor_names = [self.engine.get_binding_name(i) for i in range(input_count)]
        input_names = [self.engine.get_binding_name(i) for i in range(self.engine.num_bindings) if self.engine.binding_is_input(i)]
        output_names = [self.engine.get_binding_name(i) for i in range(self.engine.num_bindings) if not self.engine.binding_is_input(i)]

        # 分配输入内存
        self.d_inputs = {}
        # 分配输出内存
        self.d_outputs = {}
        for i in range(self.engine.num_bindings):
            binding_name = tensor_names[i]
            shape = self.engine.get_binding_shape(i)
            dtype = self.engine.get_binding_dtype(i)
      
            # 将 TensorRT 的 DataType 转换为 NumPy 类型
            np_dtype = self.dtype_map.get(dtype, np.float32)

            if binding_name in input_names:
                # 输入数据，分配GPU内存
                size = np.prod(shape) * np.dtype(np_dtype).itemsize
                self.d_inputs[binding_name] = cuda.mem_alloc(int(size))  # 强制转换为 int

            if binding_name in output_names:
                # 输出数据，分配GPU内存
                size = np.prod(shape) * np.dtype(np_dtype).itemsize
                self.d_outputs[binding_name] = cuda.mem_alloc(int(size))  # 强制转换为 int

        # 设置每个输入输出张量的地址
        for i in range(self.engine.num_bindings):
            binding_name = tensor_names[i]
            if binding_name in input_names:
                self.context.set_tensor_address(binding_name, int(self.d_inputs[binding_name]))
            elif  binding_name in output_names:
                self.context.set_tensor_address(binding_name, int(self.d_outputs[binding_name]))

        self.stream = cuda.Stream()

    
    
    