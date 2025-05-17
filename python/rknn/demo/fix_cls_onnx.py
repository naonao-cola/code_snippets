import onnx
from onnx import helper, checker
from onnx import TensorProto
import re
import argparse
import onnx
model = "/home/naonao/demo/wang/baso-focus_1x3x640x640.onnx"  # 待处理onnx模型路径
save_model = "/home/naonao/demo/wang/baso-focus_1x3x640x640_softmax.onnx"  # 修改后onnx模型保存路径
##############
# 该脚本用于修改 yolov5-cls模型,在output节点前增加softmax节点,使其更易于获取类别概率
##############

onnx_model = onnx.load(model)
graph = onnx_model.graph
# print(graph)
node = graph.node

print(graph.output[0].type.tensor_type.shape)


def createGraphMemberMap(graph_member_list):
    member_map = dict()
    for n in graph_member_list:
        member_map[n.name] = n
    return member_map


# 待删除的层 开一个 https://netron.app/ 点击要删的层看name
x = {}
de = []
num = 0

node_map = createGraphMemberMap(graph.node)
output_map = createGraphMemberMap(graph.output)

# 删除原有输出节点
graph.output.remove(output_map["output"])

# 新增输出节点
new_output_node_names = ["output0"]
output_shape_map = [[1, 8]]
for i in range(1):
    new_nv = helper.make_tensor_value_info(
        new_output_node_names[i], TensorProto.FLOAT, output_shape_map[i])
    graph.output.extend([new_nv])
output_map = createGraphMemberMap(graph.output)

# 删除原有节点
for i in range(len(graph.node)):
    if node[i].name in x:
        de.append(i)
        num = num+1
de.sort()
de.reverse()
for i in range(num):
    graph.node.remove(graph.node[de[i]])

# add node
new_node = onnx.helper.make_node(
    "Softmax",
    inputs=["gemm_109_output"],
    name="add_softmax",
    outputs=['output0'],
)


for i in range(len(graph.node)):
    # 将输出链接到输出节点 output0
    if node[i].name == "Gemm_109":
        # 插入节点
        graph.node.insert(i+1, new_node)
        node[i].output[0] = "gemm_109_output"
        print("proccesed")
    # if node[i].name == "Transpose_216":
    #     node[i].output[0]="output1"
    # if node[i].name == "Transpose_234":
    #     node[i].output[0]="output2"

onnx.checker.check_model(onnx_model)
onnx.save(onnx_model, save_model)
