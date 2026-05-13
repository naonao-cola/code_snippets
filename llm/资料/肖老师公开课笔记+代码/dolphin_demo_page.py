# 导入必要的库
import argparse  # 用于解析命令行参数
import glob      # 用于文件路径模式匹配
import os        # 提供操作系统相关功能

import cv2       # OpenCV库，用于图像处理
import torch     # PyTorch深度学习框架
from PIL import Image  # Python图像处理库
from transformers import AutoProcessor, VisionEncoderDecoderModel  # Hugging Face的transformers库组件

from utils import *  # 导入自定义工具函数

import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # 同步执行CUDA操作，便于定位错误


class DOLPHIN:
    """基于Hugging Face视觉-模型的对话处理类"""
    
    def __init__(self, model_id_or_path):
        """初始化Hugging Face模型
        
        参数:
            model_id_or_path: 可以是以下两种形式：
                - 本地模型路径
                - Hugging Face模型仓库ID
        """
        # 从本地路径或Hugging Face仓库加载模型
        self.processor = AutoProcessor.from_pretrained(model_id_or_path)  # 自动加载适合的处理器
        self.model = VisionEncoderDecoderModel.from_pretrained(model_id_or_path)  # 加载视觉编码器-文本解码器模型
        self.model.eval()  # 设置为评估模式
        
        # 设置计算设备和精度
        self.device = "cuda" if torch.cuda.is_available() else "cpu"  # 自动检测GPU可用性
        self.model.to(self.device)  # 将模型移动到指定设备
        self.model = self.model.half()  # 默认使用半精度浮点数(FP16)以节省显存
        
        # 初始化tokenizer
        self.tokenizer = self.processor.tokenizer  # 从处理器中获取tokenizer
        
    def chat(self, prompt, image):
        """处理图像并生成文本回复
        
        参数:
            prompt: 文本提示或提示列表，用于指导模型生成
            image: PIL Image对象或PIL Image列表，需要处理的图像
            
        返回:
            模型生成的文本或文本列表
        """
        # 检查是否是批量处理模式
        is_batch = isinstance(image, list)
        
        if not is_batch:
            # 单图像处理：转换为列表形式保持处理逻辑一致
            images = [image]
            prompts = [prompt]
        else:
            # 批量处理：直接使用输入
            images = image
            prompts = prompt if isinstance(prompt, list) else [prompt] * len(images)  # 如果提示不是列表，复制为与图像数量匹配
        
        # 图像预处理
        batch_inputs = self.processor(images, return_tensors="pt", padding=True)  # 将图像转换为模型输入格式
        batch_pixel_values = batch_inputs.pixel_values.half().to(self.device)  # 转换为半精度并移动到指定设备
        
        # 提示文本预处理
        prompts = [f"<s>{p} <Answer/>" for p in prompts]  # 添加特殊标记格式化提示
        batch_prompt_inputs = self.tokenizer(
            prompts,
            add_special_tokens=False,  # 不添加额外特殊标记（因为已经手动添加）
            return_tensors="pt"        # 返回PyTorch张量
        )

        batch_prompt_ids = batch_prompt_inputs.input_ids.to(self.device)  # 将输入ID移动到设备
        batch_attention_mask = batch_prompt_inputs.attention_mask.to(self.device)  # 将注意力掩码移动到设备
        
        # 文本生成
        outputs = self.model.generate(
            pixel_values=batch_pixel_values,  # 图像特征输入
            decoder_input_ids=batch_prompt_ids,  # 解码器初始输入
            decoder_attention_mask=batch_attention_mask,  # 解码器注意力掩码
            min_length=1,       # 生成文本最小长度
            max_length=8192,    # 生成文本最大长度
            pad_token_id=self.tokenizer.pad_token_id,  # 填充token ID
            eos_token_id=self.tokenizer.eos_token_id,  # 结束token ID
            use_cache=True,     # 使用缓存加速
            bad_words_ids=[[self.tokenizer.unk_token_id]],  # 屏蔽未知token生成
            return_dict_in_generate=True,  # 以字典形式返回结果
            do_sample=False,     # 不使用采样
            num_beams=1,        # 束搜索宽度为1(贪婪搜索)
            repetition_penalty=1.1  # 重复惩罚系数
        )
        
        # 后处理生成的文本
        sequences = self.tokenizer.batch_decode(outputs.sequences, skip_special_tokens=False)  # 将token ID解码为文本
        
        # 清理输出文本：移除提示文本和特殊标记
        results = []
        for i, sequence in enumerate(sequences):
            cleaned = sequence.replace(prompts[i], "").replace("<pad>", "").replace("</s>", "").strip()  # 移除各种标记
            results.append(cleaned)
            
        # 如果是单图像输入，返回单个结果而非列表
        if not is_batch:
            return results[0]
        return results

def process_document(document_path, model, save_dir, max_batch_size=None):
    """解析文档（支持图像和PDF）的两阶段处理函数
        PDF处理：先转换为多页图像(自动生成带页码的文件名)，再逐页解析后合并结果
        图像处理：直接调用单图像解析流程
    参数:
        document_path (str): 文档路径（支持PDF或图像格式）
        model: 已加载的文档解析模型（如Dolphin模型实例）
        save_dir (str): 解析结果保存目录
        max_batch_size (int, optional): 并行处理的最大批次大小
        
    返回:
        tuple: (结果文件路径, 解析结果数据)
              - 对于PDF返回(合并结果路径, 所有页面的解析结果列表)
              - 对于图像返回(单文件结果路径, 解析结果)
              
    异常:
        Exception: PDF转换失败时抛出
    """
    # 获取文件扩展名并转为小写[1](@ref)
    file_ext = os.path.splitext(document_path)[1].lower()
    
    # PDF文件处理分支
    if file_ext == '.pdf':
        # 阶段1：PDF转图像[1,3](@ref)
        images = convert_pdf_to_images(document_path)
        if not images:
            raise Exception(f"PDF转换失败: {document_path}")
        
        all_results = []  # 存储所有页面的解析结果
        
        # 逐页处理[1](@ref)
        for page_idx, pil_image in enumerate(images):
            print(f"正在处理第 {page_idx + 1}/{len(images)} 页")
            
            # 生成页面标识名（保留原文件名+页码）[1](@ref)
            base_name = os.path.splitext(os.path.basename(document_path))[0]
            page_name = f"{base_name}_page_{page_idx + 1:03d}"
            
            # 调用单图像处理函数（不单独保存每页结果）[1,2](@ref)
            json_path, recognition_results = process_single_image(
                pil_image, model, save_dir, page_name, 
                max_batch_size, save_individual=False
            )
            
            # 记录带页码的解析结果[1](@ref)
            page_results = {
                "page_number": page_idx + 1,  # 1-based页码
                "elements": recognition_results  # 当前页解析出的元素
            }
            all_results.append(page_results)
        
        # 保存PDF的合并解析结果[1](@ref)
        combined_json_path = save_combined_pdf_results(all_results, document_path, save_dir)
        
        return combined_json_path, all_results
    
    # 图像文件处理分支
    else:
        # 打开图像并转为RGB格式[1,3](@ref)
        pil_image = Image.open(document_path).convert("RGB")
        base_name = os.path.splitext(os.path.basename(document_path))[0]
        # 直接调用单图像处理函数[1](@ref)
        return process_single_image(pil_image, model, save_dir, base_name, max_batch_size)


def process_single_image(image, model, save_dir, image_name, max_batch_size=None, save_individual=True):
    """处理单张图像（可直接输入或从PDF转换得到）
    
    参数:
        image: PIL Image对象，待解析的图像
        model: DOLPHIN模型实例，需已加载权重
        save_dir: str，解析结果保存目录路径
        image_name: str，输出文件的基础名称（不含扩展名）
        max_batch_size: int/None，元素级解析的最大批次大小（控制显存占用）[5](@ref)
        save_individual: bool，是否保存独立结果文件（PDF分页处理时建议设为False）[1](@ref)
        
    返回:
        tuple: (json_path, recognition_results)
            - json_path: str/None，结果JSON文件路径（save_individual=False时为None）
            - recognition_results: list，结构化解析结果，包含元素类型/坐标/内容[3](@ref)
            
    功能说明:
        实现Dolphin模型的两阶段解析流程：
        1. 页面级布局分析：识别文档元素类型和阅读顺序
        2. 元素级内容解析：并行提取文本/表格/公式等内容[6](@ref)
    """
    # 阶段1：页面级布局与阅读顺序解析
    # 使用特定提示词触发布局分析任务[1,3](@ref)
    layout_output = model.chat("Parse the reading order of this document.", image)

    # 阶段2：元素级内容解析
    # 图像预处理（保持纵横比的填充调整）[4](@ref)
    padded_image, dims = prepare_image(image)
    # 并行处理所有识别到的文档元素[2,6](@ref)
    recognition_results = process_elements(
        layout_output,          # 阶段1输出的布局信息
        padded_image,           # 预处理后的图像
        dims,                   # 原始图像尺寸信息
        model,                  # Dolphin模型实例
        max_batch_size,         # 并行处理的批次大小限制
        save_dir,               # 结果保存路径
        image_name              # 用于生成输出文件名
    )

    # 按需保存结果（PDF分页处理时跳过单独保存）[1](@ref)
    json_path = None
    if save_individual:
        # 生成虚拟图像路径（仅用于构造输出文件名）
        dummy_image_path = f"{image_name}.jpg"  # 扩展名仅作占位，实际未使用
        # 保存结构化结果到JSON文件[5](@ref)
        json_path = save_outputs(recognition_results, dummy_image_path, save_dir)

    return json_path, recognition_results


def process_elements(layout_results, padded_image, dims, model, max_batch_size, save_dir=None, image_name=None):
    """并行解析文档中的所有元素（文本/表格/图像）
    
    参数:
        layout_results: str/list, 布局分析阶段输出的元素序列（字符串或结构化数据）
        padded_image: ndarray, 经过填充调整后的文档图像（OpenCV格式）
        dims: tuple, 原始图像的尺寸信息(w,h)
        model: DOLPHIN模型实例，用于元素级内容解析
        max_batch_size: int, 控制并行处理的批次大小（None表示不限制）
        save_dir: str/None, 图像类元素保存目录（None时不保存）
        image_name: str/None, 用于生成输出文件名前缀
        
    返回:
        list: 按阅读顺序排序的结构化解析结果，每个元素包含：
            - label: 元素类型（"text"/"tab"/"fig"）
            - bbox: 原始图像坐标[x1,y1,x2,y2]
            - text/content: 解析内容（文本/HTML表格/Markdown图像链接）
            - reading_order: 阅读顺序序号
    """
    # 预处理布局分析结果（支持字符串或结构化输入）
    layout_results = parse_layout_string(layout_results)

    # 初始化元素分类容器
    text_elements = []   # 文本类元素（段落/标题等）
    table_elements = []  # 表格类元素
    figure_results = []  # 图像类元素（直接保存文件不解析）
    previous_box = None  # 记录前一个元素坐标（用于相对位置计算）
    reading_order = 0    # 阅读顺序计数器

    # 遍历布局分析结果并分类处理
    for bbox, label in layout_results:
        try:
            # 坐标转换：将布局坐标映射到填充后图像的实际区域
            x1, y1, x2, y2, orig_x1, orig_y1, orig_x2, orig_y2, previous_box = process_coordinates(
                bbox, padded_image, dims, previous_box
            )

            # 裁剪元素区域（跳过空区域）
            cropped = padded_image[y1:y2, x1:x2]
            if cropped.size > 0:
                if label == "fig":
                    # 图像类元素处理（不解析内容，仅保存文件）
                    pil_crop = Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
                    
                    # 保存图像到本地并记录相对路径（优化存储效率）
                    figure_filename = save_figure_to_local(pil_crop, save_dir, image_name, reading_order)
                    
                    figure_results.append({
                        "label": label,
                        "text": f"![Figure](figures/{figure_filename})",  # Markdown格式引用
                        "figure_path": f"figures/{figure_filename}",      # 原始路径备份
                        "bbox": [orig_x1, orig_y1, orig_x2, orig_y2],
                        "reading_order": reading_order,
                    })
                else:
                    # 文本/表格元素预处理
                    pil_crop = Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
                    element_info = {
                        "crop": pil_crop,          # 裁剪后的元素图像
                        "label": label,            # 元素类型标签
                        "bbox": [orig_x1, orig_y1, orig_x2, orig_y2],  # 原始坐标
                        "reading_order": reading_order,  # 阅读顺序编号
                    }
                    
                    # 按类型分组（Dolphin采用差异化提示策略）
                    if label == "tab":
                        table_elements.append(element_info)  # 表格用HTML提示
                    else:  
                        text_elements.append(element_info)   # 文本用LaTeX/纯文本提示

            reading_order += 1

        except Exception as e:
            print(f"元素处理失败（类型={label}）：{str(e)}")
            continue

    # 初始化最终结果（先加入已处理的图像元素）
    recognition_results = figure_results.copy()
    
    # 批量处理文本元素（使用统一提示词）
    if text_elements:
        text_results = process_element_batch(
            text_elements, 
            model, 
            "Read text in the image.",  # 文本解析专用提示)
            max_batch_size
        )
        recognition_results.extend(text_results)
    
    # 批量处理表格元素（使用HTML解析提示）
    if table_elements:
        table_results = process_element_batch(
            table_elements, 
            model, 
            "Parse the table in the image.",  # 表格解析专用提示)
            max_batch_size
        )
        recognition_results.extend(table_results)

    # 按阅读顺序排序（模拟人类阅读体验）
    recognition_results.sort(key=lambda x: x.get("reading_order", 0))

    return recognition_results


def process_element_batch(elements, model, prompt, max_batch_size=None):
    """批量处理同类型元素（核心并行化逻辑）
    
    参数:
        elements: list, 同类型元素列表（需包含crop/label/bbox字段）
        model: DOLPHIN模型实例
        prompt: str, 类型专用提示词（如表格/文本差异化提示）
        max_batch_size: int/None, 最大批次大小（控制显存占用）
        
    返回:
        list: 解析结果列表，每个元素包含：
            - label: 元素类型
            - bbox: 原始坐标
            - text: 解析内容（HTML/LaTeX/纯文本）
            - reading_order: 阅读序号
    """
    results = []
    
    # 动态计算批次大小（考虑显存限制）
    batch_size = len(elements)
    if max_batch_size is not None and max_batch_size > 0:
        batch_size = min(batch_size, max_batch_size)
    
    # 分批次处理（Dolphin的并行解码优势）
    for i in range(0, len(elements), batch_size):
        batch_elements = elements[i:i+batch_size]
        crops_list = [elem["crop"] for elem in batch_elements]
        
        # 构造批量提示词（同类型元素共享提示策略）
        prompts_list = [prompt] * len(crops_list)
        
        # 批量推理（异构锚点提示技术）
        batch_results = model.chat(prompts_list, crops_list)
        
        # 结构化输出结果
        for j, result in enumerate(batch_results):
            elem = batch_elements[j]
            results.append({
                "label": elem["label"],
                "bbox": elem["bbox"],
                "text": result.strip(),  # 移除首尾空白
                "reading_order": elem["reading_order"],
            })
    
    return results


def extract_content(model_path:str, input_path: str, save_dir: str=None, max_batch_size: int=16):
    """基于Dolphin模型的文档解析主程序
    
    功能说明：
    1. 支持批量处理图像/PDF文档，自动识别输入路径中的支持格式
    2. 提供灵活的保存路径和并行处理参数配置
    3. 集成完整的错误处理机制，保证批量处理的稳定性
    
    参数解析说明：
    该函数通过argparse模块定义并处理以下命令行参数：
    --model_path: 模型路径，默认使用Hugging Face格式的本地模型
    --input_path: 输入文件/目录路径，支持图像和PDF格式
    --save_dir: 结果保存目录，默认与输入目录相同
    --max_batch_size: 元素级解析的批次大小，影响显存占用和速度
    """


    # 加载Dolphin模型（基于Hugging Face Transformers）
    print("正在加载Dolphin模型...")
    model = DOLPHIN(model_path)  # 初始化Dolphin解析器

    # 文件收集逻辑（支持目录批量处理和单文件处理）
    if os.path.isdir(input_path):
        # 支持的文档格式（不区分大小写）
        file_extensions = [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG", ".pdf", ".PDF"]
        
        # 使用glob收集所有匹配文件
        document_files = []
        for ext in file_extensions:
            document_files.extend(glob.glob(os.path.join(input_path, f"*{ext}")))
        document_files = sorted(document_files)  # 按文件名排序保证处理顺序一致
    else:
        # 单文件处理模式
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"输入路径不存在: {input_path}")
        
        # 检查文件格式是否支持
        file_ext = os.path.splitext(input_path)[1].lower()
        supported_exts = ['.jpg', '.jpeg', '.png', '.pdf']
        if file_ext not in supported_exts:
            raise ValueError(f"不支持的格式: {file_ext}。支持格式: {supported_exts}")
        
        document_files = [input_path]

    # 设置输出目录（未指定时使用输入目录）
    save_dir = save_dir or (
        input_path if os.path.isdir(input_path) else os.path.dirname(input_path)
    )
    setup_output_dirs(save_dir)  # 创建必要的子目录（如figures/等）

    # 显示待处理文件总数
    total_samples = len(document_files)
    print(f"\n发现待处理文件: {total_samples}个")

    # 文档处理主循环
    for file_path in document_files:
        print(f"\n正在处理: {os.path.basename(file_path)}")
        try:
            # 调用核心处理函数（支持PDF和图像）
            json_path, recognition_results = process_document(
                document_path=file_path,
                model=model,
                save_dir=save_dir,
                max_batch_size=max_batch_size,  # 控制并行解析效率
            )

            print(f"处理完成！结果已保存至: {save_dir}")

        except Exception as e:
            # 错误隔离：单个文件处理失败不影响整体流程
            print(f"处理失败 [{file_path}]: {str(e)}")
            continue


if __name__ == "__main__":
    # 复杂英文的PDF
    # extract_content("/root/autodl-tmp/models/dolphin_model", "/root/autodl-tmp/dolphin_code/layout-parser-paper.pdf", "/root/autodl-tmp/dolphin_code/result/")

    # extract_content("/root/autodl-tmp/models/dolphin_model", "/root/autodl-tmp/dolphin_code/page_2.jpeg", "/root/autodl-tmp/dolphin_code/result/")
    # 复杂中文的PDF
    # extract_content("/root/autodl-tmp/models/dolphin_model", "/root/autodl-tmp/dolphin_code/内科学7.pdf", "/root/autodl-tmp/dolphin_code/result/")

    extract_content("/root/autodl-tmp/models/dolphin_model", "/root/autodl-tmp/dolphin_code/第一章 Apache Flink 概述.pdf", "/root/autodl-tmp/dolphin_code/result/")
