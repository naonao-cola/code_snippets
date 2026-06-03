import cv2
import numpy as np
import json
import os
import sys

# 将 TensorRT 的库路径加入环境变量 (根据实际安装路径修改)
# 对于 Windows，通常需要将包含 nvinfer.dll 的目录加入 PATH 或使用 os.add_dll_directory
trt_bin_path = r"E:\3rdparty\TensorRT-8.6.1.6.Windows10.x86_64.cuda-11.8\TensorRT-8.6.1.6\lib" # 示例路径
if os.path.exists(trt_bin_path):
    os.environ['PATH'] = trt_bin_path + os.pathsep + os.environ['PATH']
    if sys.platform == 'win32' and sys.version_info >= (3, 8):
        try:
            os.add_dll_directory(trt_bin_path)
        except Exception as e:
            print(f"Warning: Could not add DLL directory: {e}")

from yolov8 import YOLOv8Wrapper

class PersonIntrusionDetector(YOLOv8Wrapper):
    def __init__(self, model_path, confidence=0.25, target_size=(640, 640)):
        # 初始化时使用 YOLOv8Wrapper 的逻辑，如果是 YOLO11 模型，推理逻辑是通用的
        super().__init__(model_path, confidence, target_size)

    def parse_detection_area(self, detection_area_str, image_width, image_height):
        """
        解析前端区域 JSON，转换为图像坐标系下的多边形。
        逻辑参考 C++ 中的 ParseDetectionArea。
        """
        polygons = []
        if not detection_area_str:
            return polygons

        try:
            data = json.loads(detection_area_str)
            # 前端画布尺寸，默认 600x500
            canvas_size = data.get("canvasSize", {"width": 600, "height": 500})
            canvas_width = canvas_size.get("width", 600)
            canvas_height = canvas_size.get("height", 500)

            # 计算缩放比例
            scale_x = image_width / canvas_width
            scale_y = image_height / canvas_height

            if "polygons" in data:
                for poly_data in data["polygons"]:
                    poly = []
                    for pt in poly_data:
                        # 坐标转换与缩放
                        x = int(round(pt["x"] * scale_x))
                        y = int(round(pt["y"] * scale_y))
                        # 边界检查
                        x = max(0, min(x, image_width - 1))
                        y = max(0, min(y, image_height - 1))
                        poly.append([x, y])
                    if poly:
                        polygons.append(np.array(poly, dtype=np.int32))
        except Exception as e:
            print(f"ParseDetectionArea failed: {e}")

        return polygons

    def process_intrusion(self, image, polygons, detections):
        """
        执行入侵检测逻辑。
        参考 C++ 中的 DoWork 核心逻辑。
        detections: list of (x, y, w, h, class_id, score)
        """
        # 1. 绘制检测区域 (参考 C++ 中的 overlayMask)
        overlay = image.copy()
        for poly in polygons:
            # 填充多边形为红色
            cv2.fillPoly(overlay, [poly], (0, 0, 255))
        # 0.3 透明度叠加
        cv2.addWeighted(overlay, 0.3, image, 0.7, 0, image)

        intrusion_detected = False
        # 2. 遍历检测到的目标
        for det in detections:
            x, y, w, h, class_id, score = det
            # 过滤人 (假设 class_id 0 是人，与 C++ 一致)
            if class_id == 0:
                # 计算底边中心点 (参考 C++: cv::Point center(box.x + box.width / 2, box.y + box.height))
                center_x = x + w // 2
                center_y = y + h

                is_inside = False
                for poly in polygons:
                    # 使用 cv2.pointPolygonTest 判断点是否在多边形内
                    # 返回值 >= 0 表示在多边形内部或边缘
                    res = cv2.pointPolygonTest(poly, (float(center_x), float(center_y)), False)
                    if res >= 0:
                        is_inside = True
                        break

                if is_inside:
                    intrusion_detected = True
                    # 绘制入侵目标的红色框
                    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    label = f"PersonGoin:{score:.2f}"
                    # 绘制标签 (参考 C++ 中的 m_untity.draw_text)
                    cv2.putText(image, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        return image, intrusion_detected

def main():
    # 配置文件路径
    model_path = "E:\demo\py\infer\model\person.engine"
    if not os.path.exists(model_path):
        # 备选路径，如果 engine 不存在尝试使用默认路径
        model_path = "model_zoo/model_files/engine/yolov8n.engine"

    # 模拟前端传来的检测区域 JSON
    # 包含一个矩形区域：(100,100) 到 (500,400)
    test_json_str = '''
    {
        "canvasSize": {"width": 600, "height": 500},
        "polygons": [
            [
                {"x": 100, "y": 100},
                {"x": 500, "y": 100},
                {"x": 500, "y": 400},
                {"x": 100, "y": 400}
            ]
        ]
    }
    '''

    # 初始化检测器
    try:
        detector = PersonIntrusionDetector(model_path)
    except Exception as e:
        print(f"Failed to initialize detector: {e}")
        return

    # 获取测试图片
    test_dir = "E:\demo\py\infer\data"
    if not os.path.exists(test_dir):
        print(f"Directory {test_dir} not found.")
        return

    jpg_files = [os.path.join(test_dir, f) for f in os.listdir(test_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    if not jpg_files:
        print("No image files found in dataset/coco_test.")
        return

    for img_path in jpg_files:
        image = cv2.imread(img_path)
        if image is None:
            continue

        h, w = image.shape[:2]

        # 1. 解析多边形区域
        polygons = detector.parse_detection_area(test_json_str, w, h)

        # 2. 执行模型推理
        results = detector.run(image)

        # 3. 执行入侵检测逻辑
        result_img, is_intruded = detector.process_intrusion(image, polygons, results)

        # 4. 显示结果
        print(f"Processing: {img_path} | Intrusion: {is_intruded}")

        cv2.imshow("Person Intrusion Detection", result_img)
        # 按 'q' 退出，按其他键继续下一张
        key = cv2.waitKey(0)
        if key == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
