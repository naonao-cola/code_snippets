import onnx_graphsurgeon as gs
from onnxsim import simplify
import numpy as np
import onnx



"""
模型优化
import onnx
from onnxsim import simplify
onnx_model = onnx.load(ONNX_name)  # load onnx model
model_simp, check = simplify(onnx_model)
assert check, "Simplified ONNX model could not be validated"
onnx.save(model_simp, "resnet18_simplify.onnx")

import onnx
import onnxoptimizer

model = onnx.load(ONNX_name)
new_model = onnxoptimizer.optimize(model)
onnx.save(new_model,"resnet18_optimize.onnx")
# use model_simp as a standard ONNX model object

"""


model_path = r'E:\demo\test_model\sbg_best_1x3x2208x2208.onnx'
origin_shape = (2208, 2208)
origin_channels = 3
origin_batch=1

new_shape = (2208, 2208)
new_channels = 3
new_batch=4


def modify_onnx_model_with_gs(model_path, new_shape, new_batch,new_channels):
    graph = gs.import_onnx(onnx.load(model_path))
    input_node = graph.inputs[0]
    old_shape = input_node.shape[2:]
    new_dim = [new_batch, new_channels] + list(new_shape)
    input_node.shape = new_dim
    output_node= graph.outputs[0]
    old_out_shape = output_node.shape[1:]
    new_out_shape = [new_batch] + list(old_out_shape)
    output_node.shape = new_out_shape
    onnx.checker.check_model(gs.export_onnx(graph),)
    onnx.save(gs.export_onnx(graph), 'modified_model.onnx')



modify_onnx_model_with_gs(model_path, new_shape, new_batch,new_channels)
