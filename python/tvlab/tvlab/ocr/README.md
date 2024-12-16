### ocr使用说明
安装paddle
```
python3 -m pip install paddlepaddle-gpu
```
安装paddleocr v1.1.0
```

使用说明
    目前，ocr只实现了DBnet检测模型。所以训练时只能使用dbnet的config文件，位置检测的预训练模型在pre_train文件夹下。文字识别使用训练好的模型直接推理，模型放在inference文件夹下。

依赖：
    ocr的cuad版本为10.1  cudnn的版本为7.6
