from openai import OpenAI
import fitz  # PyMuPDF，用于处理PDF和转换图像
import base64
import os
from io import BytesIO
from PIL import Image

# 初始化OpenAI客户端，连接到您本地部署的PaddleOCR-VL服务
client = OpenAI(
    api_key="EMPTY",  # 本地部署通常不需要API密钥
    base_url="http://localhost:6006/v1",  # 请确保这是您PaddleOCR-VL服务的正确地址和端口
    timeout=3600  # 设置较长的超时时间，处理多页PDF可能需要时间
)


def convert_pdf_page_to_image(pdf_path, page_number, dpi=200):
    """
    将PDF的指定页面转换为高质量的PIL图像对象。

    参数:
        pdf_path (str): PDF文件的路径。
        page_number (int): 要转换的页码（从0开始）。
        dpi (int): 图像分辨率，值越高越清晰，但文件也越大。默认200适用于大多数文档。

    返回:
        PIL.Image.Image: 转换后的图像对象。
    """
    print(f"正在打开PDF文件: {pdf_path}")
    # 使用PyMuPDF打开PDF
    pdf_document = fitz.open(pdf_path)

    # 检查页码是否有效
    if page_number < 0 or page_number >= len(pdf_document):
        raise ValueError(f"页码 {page_number} 超出范围。PDF共有 {len(pdf_document)} 页。")

    # 获取指定页面
    page = pdf_document.load_page(page_number)
    print(f"正在渲染第 {page_number + 1} 页...")

    # 将PDF页面转换为像素图（pixmap），设置分辨率
    mat = fitz.Matrix(dpi / 72, dpi / 72)  # 创建一个转换矩阵，72是PDF的默认DPI
    pix = page.get_pixmap(matrix=mat)

    # 将pixmap转换为PIL图像
    img_data = pix.tobytes("ppm")  # 转换为PPM格式的字节流
    pil_image = Image.open(BytesIO(img_data)).convert('RGB')

    pdf_document.close()
    print(f"第 {page_number + 1} 页转换完成，图像尺寸: {pil_image.size}")
    return pil_image


def pil_image_to_base64(pil_image):
    """
    将PIL图像对象转换为Base64编码的字符串。

    参数:
        pil_image (PIL.Image.Image): 输入图像。

    返回:
        str: 代表图像的Base64字符串。
    """
    print("正在将图像编码为Base64格式...")
    buffered = BytesIO()
    # 将图像以PNG格式保存到内存缓冲区
    pil_image.save(buffered, format="PNG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
    print("图像编码完成。")
    return img_base64


def parse_pdf_to_markdown(pdf_path, output_md_path="output_document.md"):
    """
    主函数：解析整个PDF文件并生成Markdown。

    参数:
        pdf_path (str): 输入的PDF文件路径。
        output_md_path (str): 输出的Markdown文件路径。
    """
    print("=" * 60)
    print("开始处理PDF文档")
    print("=" * 60)

    # 检查PDF文件是否存在
    if not os.path.isfile(pdf_path):
        print(f"错误：找不到PDF文件 '{pdf_path}'")
        return

    print(f"输入PDF文件: {pdf_path}")
    print(f"输出Markdown文件: {output_md_path}")

    pdf_document = fitz.open(pdf_path)
    total_pages = len(pdf_document)
    pdf_document.close()
    print(f"PDF文档总页数: {total_pages}")

    # 创建一个列表来保存所有页面的Markdown内容
    all_markdown_parts = []

    # 逐页处理PDF
    for page_idx in range(total_pages):
        print(f"\n开始处理第 {page_idx + 1}/{total_pages} 页...")

        try:
            # 1. 将PDF页面转换为图像
            pil_img = convert_pdf_page_to_image(pdf_path, page_idx)

            # 2. 将图像编码为Base64
            image_base64 = pil_image_to_base64(pil_img)

            # 3. 构建发送给PaddleOCR-VL模型的消息
            # 使用简洁有效的提示词，模型内置了强大的文档理解能力[6](@ref)
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "<image>\n请将这份文档页面转换为结构清晰的Markdown格式，保留标题层级、列表、表格等所有格式。"
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image_base64}"
                            }
                        }
                    ]
                }
            ]

            # 4. 调用PaddleOCR-VL模型进行推理
            print("正在调用PaddleOCR-VL模型进行解析...")
            response = client.chat.completions.create(
                model="paddleocr-vl",  # 模型名称，请确保与您部署的服务一致
                messages=messages,
                temperature=0.0,  # 设置为0保证输出确定性
                max_tokens=4000  # 根据页面内容调整，确保足够容纳一页内容
            )

            # 5. 提取模型返回的Markdown内容
            page_markdown = response.choices[0].message.content
            print(f"第 {page_idx + 1} 页解析完成。")

            # 在每页内容前添加一个分页标记和页码标题，方便阅读
            all_markdown_parts.append(f"\n--- 第 {page_idx + 1} 页 ---\n")
            all_markdown_parts.append(page_markdown)

        except Exception as e:
            print(f"处理第 {page_idx + 1} 页时发生错误: {e}")
            # 记录错误，但继续处理下一页
            all_markdown_parts.append(f"\n--- 第 {page_idx + 1} 页 [处理失败: {e}] ---\n")

    # 6. 将所有页面的Markdown内容合并并保存到文件
    print(f"\n正在将结果写入Markdown文件: {output_md_path}")
    try:
        with open(output_md_path, 'w', encoding='utf-8') as md_file:
            md_file.write("".join(all_markdown_parts))
        print(f"✅ 成功！Markdown文件已保存至: {output_md_path}")

        # 打印最终文件信息
        file_size = os.path.getsize(output_md_path)
        print(f"生成的文件大小: {file_size} 字节")

    except Exception as e:
        print(f"❌ 写入输出文件时出错: {e}")


# 主程序入口
if __name__ == "__main__":
    # 请修改为您的实际PDF文件路径
    input_pdf_file = r"E:\my_project\Multimodal_RAG\paddle_ocr\test2.pdf"
    # 设置输出的Markdown文件名
    output_markdown_file = "parsed_result.md"

    # 执行解析
    parse_pdf_to_markdown(input_pdf_file, output_markdown_file)