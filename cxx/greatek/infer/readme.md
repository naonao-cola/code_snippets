# 文件说明

infer.py      是模型推理基类,包含模型的加载,模型的推理以及输入输出的cuda内存 的申请
yolov8.py     继承infer.py 里面的ModelWrapper类,实现了具体的前处理,以及后处理
yolov11.py    11的前后处理与8相同,所以只是引入包,将v8 改了别名称为 YOLOv11Warpper
with_draw.py  在图片画框的函数

以上文件不需要变更

# 检测逻辑

test.py  项目文件,PersonIntrusionDetector 为人员入侵检测类,检测逻辑顺序,main 函数是检测流程,

```python
    # 初始化检测器
    detector = PersonIntrusionDetector(model_path)
    # 加载图片
    image = cv2.imread(img_path)
     # 1. 解析多边形区域
    h, w = image.shape[:2]
    polygons = detector.parse_detection_area(test_json_str, w, h)
    # 2. 执行模型推理
    results = detector.run(image)
    # 3. 执行入侵检测逻辑
    result_img, is_intruded = detector.process_intrusion(image, polygons, results)
```


# 环境问题

需要安装tensorrt 的python版本,  一般在下载的压缩包的python 文件夹
安装pycuda, 版本在requirement.txt 里面
其他的类似 opencv-python  numpy onnxruntim 使用pip 安装即可

# 注意事项

1. 在test.py 的文件首几行,加入了 tensorrt 的动态库的目录,因为tensorrt 的python版本是本地安装的,所以要加入环境变量.
2. 可能需要修改PersonIntrusionDetector 解析前端画布传入的json 画框区域的代码.