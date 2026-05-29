import os
import sys
import torch
from transformers import AutoModel, AutoTokenizer
from PIL import Image

# DeepSeek-OCR-2 模型在加载时可能需要访问 Hugging Face Hub 的自定义代码。
# 如果模型已完全下载到本地，设置这些环境变量可以避免网络请求，
# 但通常情况下，如果模型文件完整，即使不设置也能正常工作。
# os.environ["HF_HUB_OFFLINE"] = "1"
# os.environ["TRANSFORMERS_OFFLINE"] = "1"

def run_ocr(image_path: str, model_path: str):
    """
    使用 DeepSeek-OCR-2 模型对图片进行 OCR 识别。

    Args:
        image_path (str): 待识别图片的完整路径。
        model_path (str): DeepSeek-OCR-2 模型本地存储的路径。
    """
    print(f"正在从本地路径加载模型: {model_path}...", flush=True)

    # 1. 验证模型路径和图片路径
    if not os.path.exists(model_path) or not os.listdir(model_path):
        print(f"错误: 模型目录 {model_path} 不存在或为空。请确保模型已下载。", flush=True)
        return

    if not os.path.exists(image_path):
        print(f"错误: 找不到图片 {image_path}。", flush=True)
        return

    # 2. 加载分词器 (Tokenizer) 和模型 (Model)
    try:
        # 加载分词器，trust_remote_code 必须为 True 以加载模型自定义的分词逻辑
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
        )

        # 确定模型加载的设备。如果存在 CUDA GPU，则使用第一张 GPU (cuda:0)，否则使用 CPU。
        # 注意：DeepSeek-OCR-2 的自定义代码在多 GPU 环境下使用 device_map="auto" 可能导致设备不匹配错误。
        # 强制加载到单张 GPU (cuda:0) 可以避免此问题。
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

        # 加载模型，trust_remote_code 必须为 True 以加载模型自定义的结构和方法。
        # dtype=torch.bfloat16 是官方推荐的精度，可以节省显存并加速推理。
        # device_map={"": device} 强制将整个模型加载到指定的设备上。
        model = AutoModel.from_pretrained(
            model_path,
            trust_remote_code=True,
            dtype=torch.bfloat16,
            device_map={"": device},
            use_safetensors=True, # 推荐使用 safetensors 格式加载模型
        )
        model = model.eval() # 设置为评估模式，关闭 dropout 等训练特有层

        print("✅ 模型加载成功！", flush=True)
        print(f"模型类型: {type(model).__name__}", flush=True)
        print(f"模型加载到设备: {device}", flush=True)

    except Exception as e:
        print(f"❌ 加载模型时出错: {e}", flush=True)
        print("\n提示: 请确保模型文件完整，包含 configuration_deepseek_ocr.py 等自定义代码文件。", flush=True)
        import traceback
        traceback.print_exc()
        return

    # 3. 加载图片
    try:
        # 使用 PIL 加载图片并转换为 RGB 格式
        image = Image.open(image_path).convert("RGB")
        print(f"✅ 图片加载成功: {image.size}", flush=True)
    except Exception as e:
        print(f"❌ 读取图片出错: {e}", flush=True)
        return

    # 4. 构造 Prompt
    # DeepSeek-OCR-2 的 Prompt 格式非常重要。
    # "<image>" 占位符是必须的，用于指示模型处理图像输入。
    # 不同的指令会引导模型执行不同的任务：
    #   - 模式 A (通用 OCR): 仅提取图片中的所有文字，不进行特殊格式化。
    #   - 模式 B (结构化 OCR): 提取文字并尝试转换为 Markdown 格式 (例如表格、标题)。
    #   - 模式 C (带坐标的 OCR): 提取文字并输出其在图片中的位置坐标。

    # 当前选择：模式 A - 通用 OCR，直接提取文字
    prompt = "<image>\n识别并提取图片中的所有文字。"

    # 如果需要结构化 Markdown 输出，请取消注释下一行并注释掉上一行：
    # prompt = "<image>\n识别图片中的文字并转换为 markdown 格式。"

    # 如果需要带坐标的输出，请取消注释下一行并注释掉上一行：
    # prompt = "<image>\n<|grounding|>识别并提取图片中的文字。"

    print("正在通过 DeepSeek-OCR-2 进行 OCR 处理...", flush=True)
    print(f"使用Prompt: {prompt}", flush=True)

    # 5. 调用模型推理 (model.infer())
    try:
        # 创建一个用于保存 OCR 结果的输出目录。
        # model.infer() 方法需要一个有效的 output_path 参数，即使不保存中间结果。
        output_dir = os.path.join(os.getcwd(), "ocr_outputs")
        os.makedirs(output_dir, exist_ok=True)

        # model.infer() 是 DeepSeek-OCR-2 提供的便捷推理方法，它内部处理了图片预处理、
        # Prompt 编码以及模型生成等复杂步骤。
        # 注意：image_file 参数必须是图片文件的路径字符串，而不是 PIL Image 对象。
        # eval_mode=True 确保返回的是识别结果字符串。
        result = model.infer(
            tokenizer=tokenizer,
            prompt=prompt,
            image_file=image_path, # 传入图片路径字符串
            output_path=output_dir, # 必须提供一个有效的输出目录
            base_size=1024,         # 全局视图大小
            image_size=768,         # 局部裁剪大小
            crop_mode=True,         # 启用动态裁剪模式以处理大图
            save_results=False,     # 是否保存中间结果（如带框图片），这里设置为 False
            eval_mode=True,         # 确保返回最终的 OCR 文本结果
        )

        # 提取最终的 OCR 结果
        if result:
            final_result = result
        else:
            final_result = "⚠️ 未获取到有效的 OCR 结果"

    except Exception as e:
        print(f"❌ model.infer() 失败: {e}", flush=True)
        import traceback
        traceback.print_exc()
        return

    # 6. 清理和输出结果
    # 移除 Prompt 部分，只保留模型回答。
    # DeepSeek-OCR-2 的 infer 方法通常会返回纯净的回答，但以防万一，这里做一下清理。
    if prompt in final_result:
        final_result = final_result.replace(prompt, "").strip()

    # 移除模型可能生成的特殊 token 或多余的空白
    final_result = final_result.replace("<|endoftext|>", "").strip()

    print("\n" + "="*30 + " OCR 结果 " + "="*30, flush=True)
    if final_result:
        print(final_result, flush=True)
    else:
        print("⚠️ 未获取到有效的 OCR 结果", flush=True)
    print("="*68, flush=True)

    # 7. 保存结果到文件
    if final_result and final_result != "⚠️ 未获取到有效的 OCR 结果":
        output_file = image_path.rsplit('.', 1)[0] + '_ocr.txt'
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(final_result)
        print(f"\n✅ 结果已保存至: {output_file}", flush=True)

if __name__ == "__main__":
    # 获取脚本所在目录，用于构建模型和图片路径
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # 确定图片路径
    if len(sys.argv) > 1:
        img_path = sys.argv[1] # 从命令行参数获取图片路径
    else:
        # 如果未提供命令行参数，则尝试使用当前目录下的 test_image.jpg
        img_path = os.path.join(script_dir, "test_image.jpg")
        if not os.path.exists(img_path):
            print("用法: python deploy_ocr.py <图片路径>", flush=True)
            print(f"或放置测试图片到: {img_path}", flush=True)
            sys.exit(1)

    # 构建模型本地路径
    model_dir = os.path.join(script_dir, "models", "DeepSeek-OCR-2")

    # 运行 OCR 识别
    run_ocr(img_path, model_dir)
