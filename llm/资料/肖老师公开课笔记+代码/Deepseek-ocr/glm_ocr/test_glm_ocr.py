from my_llm import client
from utils.embeddings_utils import image_to_base64

img_base64 = image_to_base64(r"E:\my_project\Multimodal_RAG\glm_ocr\test1.png")[0]
# 调用布局解析 API
response = client.layout_parsing.create(
    model="glm-ocr",
    file=img_base64,  # 要解析的文件，可以是网络URL或Base64编码字符串。支持PDF、JPG、PNG格式。
    need_layout_visualization=True, # 为 True时，返回详细的版面分析可视化结果（通常是一个标注了不同区块的图片）。
    timeout=3600
)

# 输出结果
print(response)