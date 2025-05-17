import os
import cv2
import numpy as np
from rknn.api import RKNN

# 配置参数
ONNX_MODEL = '/home/mmy/projects/am300/boardline/0310/BOARDLINE.onnx'
RKNN_MODEL = '/home/mmy/projects/am300/boardline/0310/BOARDLINE.rknn'
IMG_DIR = '/home/mmy/projects/am300/boardline/images/'
DATASET = '/home/mmy/projects/am300/boardline/0309/images.txt'
RESULT_FILE = './classification_results.csv'

# 新增输出目录
OUTPUT_DIR = './annotated_images/'  # 带标注图像保存路径
PROB_PLOT_DIR = './probability_plots/'  # 概率分布图保存路径
QUANTIZE_ON = False
IMG_SIZE = (320, 320)
CLASSES = ("CNF", "CPF", "F", "FNF", "FPF", "O")

# ======================== 核心函数 ========================
def center_crop(img, target_size):
    """中心裁剪（保持原始实现）"""
    height, width = img.shape[:-1]
    resize_max_ratio = np.max([target_size[0]/np.array([height, width]), target_size[1]/np.array([height, width])])
    inside_h = int((height*resize_max_ratio).round())
    inside_w = int((width*resize_max_ratio).round())

    img = cv2.resize(img, (inside_w, inside_h), interpolation=cv2.INTER_LINEAR)
    start_x = int((inside_h-target_size[0])/2)
    start_y = int((inside_w-target_size[1])/2)

    return img[start_x:start_x+target_size[0], start_y:start_y+target_size[1], :]

def softmax(x):
    return np.exp(x)/np.sum(np.exp(x))

def draw_annotation(img, pred_class, confidence, probs):
    """在图像上绘制标注信息"""
    # 转换为BGR格式用于OpenCV显示
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    # 基础标注
    text = f"{pred_class}: {confidence:.1%}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.2
    thickness = 2
    
    # 计算文本位置
    (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
    pos_x = 20
    pos_y = text_height + 40
    
    # 绘制背景框
    cv2.rectangle(img, (10, 10), (text_width + 30, pos_y + 10), (0,0,0), -1)
    
    # 绘制主文本
    cv2.putText(img, text, (pos_x, pos_y), 
               font, font_scale, (0, 255, 0), thickness)
    
    # 绘制概率分布
    y_offset = pos_y + 50
    for i, (cls, prob) in enumerate(zip(CLASSES, probs)):
        prob_text = f"{cls}: {prob:.1%}"
        cv2.putText(img, prob_text, (pos_x, y_offset), 
                   font, 0.7, (255, 255, 255), 1)
        y_offset += 30
    
    return img

def process_folder(rknn):
    """处理整个文件夹"""
    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(PROB_PLOT_DIR, exist_ok=True)
    
    with open(RESULT_FILE, 'w') as f:
        f.write("Filename,Prediction,Confidence,Probability_Distribution\n")
    
    for filename in os.listdir(IMG_DIR):
        if not filename.lower().endswith(('.bmp', '.jpg', '.png', '.jpeg')):
            continue
            
        img_path = os.path.join(IMG_DIR, filename)
        try:
            # 预处理
            img_origin = cv2.imread(img_path)
            if img_origin is None:
                raise ValueError("Invalid image file")
                
            img_rgb = cv2.cvtColor(img_origin, cv2.COLOR_BGR2RGB)
            img_cropped = center_crop(img_rgb.copy(), IMG_SIZE)
            
            # 推理
            img_input = np.expand_dims(img_cropped, 0).transpose(0, 3, 1, 2)
            pred = rknn.inference(inputs=[img_input], data_format='nchw')[0][0]
            probs = softmax(pred)
            
            # 解析结果
            pred_idx = np.argmax(probs)
            confidence = probs[pred_idx]
            pred_class = CLASSES[pred_idx]
            
            # 在裁剪后的图像上绘制标注
            annotated_img = draw_annotation(img_cropped.copy(), pred_class, confidence, probs)
            
            # 保存结果
            output_path = os.path.join(OUTPUT_DIR, f"annotated_{filename}")
            cv2.imwrite(output_path, annotated_img)
            
            # 保存概率分布图
            plot_path = os.path.join(PROB_PLOT_DIR, f"prob_{filename}")
            plot_prob_distribution(probs, filename, plot_path)  # 需要matplotlib支持
            
            # 记录结果
            result_line = f'{filename},{pred_class},{confidence:.4f},"{list(probs.round(4))}"\n'
            with open(RESULT_FILE, 'a') as f:
                f.write(result_line)
                
            print(f'Processed: {filename}')
            
        except Exception as e:
            error_msg = f"Error processing {filename}: {str(e)}"
            print(error_msg)
            with open(RESULT_FILE, 'a') as f:
                f.write(f'{filename},Error,NA,NA\n')

def plot_prob_distribution(probs, filename, save_path):
    """生成概率分布柱状图（可选）"""
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 5))
    bars = plt.bar(CLASSES, probs, color='skyblue')
    plt.ylim(0, 1)
    plt.title(f'Probability Distribution - {filename}')
    plt.ylabel('Probability')
    
    # 在柱子上方添加数值
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.1%}',
                 ha='center', va='bottom')
    
    plt.savefig(save_path)
    plt.close()

# 主程序保持不变
if __name__ == '__main__':
    rknn = RKNN(verbose=True)
    rknn.config(
        mean_values=[[123.675, 116.28, 103.53]], 
        std_values=[[58.395, 58.395, 58.395]],
        target_platform="rk3588",
        optimization_level=3
    )
    
    if rknn.load_onnx(ONNX_MODEL) != 0:
        print('Load model failed!')
        exit(1)
        
    if rknn.build(do_quantization=QUANTIZE_ON, dataset=DATASET) != 0:
        print('Build model failed!')
        exit(1)

    if rknn.init_runtime() != 0:
        print('Init runtime failed!')
        exit(1)

    process_folder(rknn)
    
    rknn.release()
    print(f'\n标注图像保存至: {OUTPUT_DIR}')
    print(f'概率分布图保存至: {PROB_PLOT_DIR}')
    print(f'汇总结果保存至: {RESULT_FILE}')
