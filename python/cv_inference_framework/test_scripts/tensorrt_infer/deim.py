import tensorrt as trt
import pycuda.driver as cuda
import numpy as np
import cv2
import pycuda.autoinit


class DEIMWrapper:
    def __init__(self, file, target_dtype=np.float32, score_threshold=0.25):
        self.target_dtype = target_dtype
        self.score_threshold = score_threshold
        self.load(file)
        self.stream = None


    def load(self, file):
        with open(file, "rb") as f:
            self.runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
            self.engine = self.runtime.deserialize_cuda_engine(f.read())
            self.context = self.engine.create_execution_context()


    def allocate_memory(self, batch):
        """
        分配 GPU 内存
        """
        self.batch_size = batch.shape[0]
        
        # 输出形状根据 ONNX 模型得到
        self.output_labels = np.empty((self.batch_size, 300), dtype=np.int32)  # labels
        self.output_boxes = np.empty((self.batch_size, 300, 4), dtype=self.target_dtype)  # boxes
        self.output_scores = np.empty((self.batch_size, 300), dtype=self.target_dtype)  # scores
        
        # 分配输入 GPU 内存
        self.d_input = cuda.mem_alloc(batch.nbytes)
        self.d_orig_target_sizes = cuda.mem_alloc(batch.shape[0] * 2 * np.int32().itemsize)  # orig_target_sizes
        
        # 分配输出 GPU 内存
        self.d_output_labels = cuda.mem_alloc(self.output_labels.nbytes)
        self.d_output_boxes = cuda.mem_alloc(self.output_boxes.nbytes)
        self.d_output_scores = cuda.mem_alloc(self.output_scores.nbytes)
        
        # 获取输入输出列表
        tensor_names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        assert len(tensor_names) == 5, "DEIM 模型的输入+输出数量必须为5"

        # 
        self.context.set_tensor_address(tensor_names[0], int(self.d_input))  # images
        self.context.set_tensor_address(tensor_names[1], int(self.d_orig_target_sizes))  # orig_target_sizes
        self.context.set_tensor_address(tensor_names[2], int(self.d_output_scores))  # scores
        self.context.set_tensor_address(tensor_names[3], int(self.d_output_labels))  # labels
        self.context.set_tensor_address(tensor_names[4], int(self.d_output_boxes))  # boxes

        self.stream = cuda.Stream()


    def predict(self, batch, orig_target_sizes):
        """
        输入数据，通常是预处理后的图像
        orig_target_sizes: 原始图像的尺寸，用于后处理
        """
        if self.stream is None:
            self.allocate_memory(batch)

        # 确保输入数据和 orig_target_sizes 是连续的
        batch = batch.copy()
        orig_target_sizes = orig_target_sizes.copy()

        # 将输入数据传输到 GPU
        cuda.memcpy_htod_async(self.d_input, batch, self.stream)
        
        # 将 orig_target_sizes 数据传输到 GPU
        cuda.memcpy_htod_async(self.d_orig_target_sizes, orig_target_sizes, self.stream)
        
        # 执行模型推理
        self.context.execute_async_v3(self.stream.handle)
        
        # 将输出数据传回 CPU
        cuda.memcpy_dtoh_async(self.output_labels, self.d_output_labels, self.stream)
        cuda.memcpy_dtoh_async(self.output_boxes, self.d_output_boxes, self.stream)
        cuda.memcpy_dtoh_async(self.output_scores, self.d_output_scores, self.stream)
        
        self.stream.synchronize()

        return self.postprocess(self.output_labels, self.output_boxes, self.output_scores)
        # return self.output_boxes


    def postprocess(self, labels, boxes, scores):
        """
        return 检测结果列表 [x1, y1, x2, y2, score, label]
        """
        results = []
        for i in range(len(scores[0])):
            if scores[0][i] > self.score_threshold:
                bbox = boxes[0][i]  # 获取当前检测框的坐标
                label = labels[0][i]  # 获取当前检测框的类别
                score = scores[0][i]  # 获取当前检测框的置信度
                results.append([
                    int(bbox[0]), 
                    int(bbox[1]), 
                    int(bbox[2]) - int(bbox[0]), 
                    int(bbox[3]) - int(bbox[1]),  # x1, y1, x2, y2
                    int(label),  # 类别
                    score  # 置信度
                ])
        return np.array(results)


    def preprocess(self, image):
        image= cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, _  = image.shape
        orig_size = np.array([w, h], dtype=np.int32)[None] 
        im_resized = cv2.resize(image, (640, 640))
        im_data = im_resized.astype(np.float32) / 255.0  # 归一化到 [0, 1] 范围
        im_data = np.transpose(im_data, (2, 0, 1))
        im_data = np.expand_dims(im_data, axis=0)  # shape: (1, 3, 640, 640)
        return im_data, orig_size


def draw_boxes(image, results, color=(0, 255, 0), font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=0.5, thickness=2):
    for result in results:
        x1, y1, w, h, class_id, score = result
        x1, y1, x2, y2, class_id = map(int, [x1, y1, x1 + w, y1 + h, class_id])
  
        label = f"{class_id}  {score:.2f}"

        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
        (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)
        cv2.rectangle(image, (x1, y1 - text_height - baseline), (x1 + text_width, y1), color, cv2.FILLED)

        cv2.putText(image, label, (x1, y1 - baseline), font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)

    return image   

if __name__ == "__main__":
    trt_model = DEIMWrapper(file="test/test_model/engine/BM_DEIM_s.engine")

    image = cv2.imread("test/test_data/20240925061547_71470_C85A49000837_A0001.jpg")

    h, w = image.shape[:2]
    image_input, orig_target_sizes = trt_model.preprocess(image)
    predictions = trt_model.predict(image_input, orig_target_sizes)
    res_img = draw_boxes(image, predictions)
    cv2.imshow("", res_img)
    cv2.waitKey(0)

    print(predictions.shape)
    print(predictions)
