from enum import Enum

class ModelType(Enum):
    ONNX = 1
    TENSORRT = 2

print(ModelType.ONNX)