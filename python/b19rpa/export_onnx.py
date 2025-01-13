# from ultralytics import YOLO


# # scp tvt@192.168.0.192:/home/tvt/zhangx/runs/BM/20241125_v8n_aug/weights/best.onnx ./
# scp tvt@192.168.0.192:/home/tvt/zhangx/runs/BM/20241129_v8n_aug/weights/best.onnx ./
# scp tvt@192.168.0.192:/home/tvt/zhangx/runs/ITO/20241128_v8n_aug/weights/best.onnx ./
# scp tvt@192.168.0.192:/home/tvt/zhangx/runs/COAB/20241202_v8n_aug_exp/weights/best.onnx ./
# scp tvt@192.168.0.192:/home/tvt/zhangx/runs/COAPS/20241202_v8n_exp/weights/best.onnx ./
    

    
# onnx_paths = [ 
#     "/data/proj/zhangx/runs/COAB/20241129_v8n_aug_exp/weights/best.pt",
# ]

  
# for pt_path in onnx_paths:
#     model = YOLO(pt_path)
#     model.export(format='onnx')


import os
from ultralytics import YOLO

# 定义根文件夹路径
root_folder = '/home/zhangx/project/B19RPA/runs'  # 替换为实际路径

# 遍历文件夹及其子文件夹
for root, dirs, files in os.walk(root_folder):
    # 检查文件夹中是否存在best.pt文件
    if 'best.pt' in files:
        # 检查是否存在best.onnx文件
        if 'best.onnx' not in files:
            pt_path = os.path.join(root, 'best.pt')  # 获取best.pt的完整路径
            onnx_path = os.path.join(root, 'best.onnx')  # 目标onnx文件路径
            
            # 加载YOLO模型并进行导出
            model = YOLO(pt_path)
            model.export(format='onnx')
            print(f"Converted {pt_path} to {onnx_path}")
        else:
            print(f"{root} contains 'best.pt' but already has 'best.onnx', skipping.")
    else:
        print(f"No 'best.pt' in {root}, skipping.")
