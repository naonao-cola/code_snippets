#!/usr/bin/env python3
# ============================================================================
# YOLO VISION TOOLS - UNIFIED CLI
# ============================================================================
import argparse
import sys
import os
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="YOLO Vision Tools - Unified CLI (P10 架构升级版)")
    subparsers = parser.add_subparsers(dest="command", help="可用命令")

    # 1. 环境诊断
    subparsers.add_parser("check", help="执行全量环境与硬件诊断 (防呆必备)")
    
    # 2. 模型下载与测试
    test_parser = subparsers.add_parser("test", help="快速下载模型并验证推理")
    test_parser.add_argument("--model", type=str, default="yolo11n.pt", help="模型名称")
    
    # 3. 数据集处理 (大图切分/格式转换)
    data_parser = subparsers.add_parser("data", help="高级数据集处理工具")
    data_parser.add_argument("--action", choices=["split", "crop", "txt2json"], required=True)
    data_parser.add_argument("--img-dir", type=str, help="图像目录")
    data_parser.add_argument("--ann-dir", type=str, help="标注目录")
    
    # 4. 可视化 (热力图)
    viz_parser = subparsers.add_parser("viz", help="生成 GradCAM 热力图")
    viz_parser.add_argument("--model", required=True, help="模型路径")
    viz_parser.add_argument("--img", required=True, help="图像路径")

    args = parser.parse_args()

    # 将当前目录加入 path 确保能导入同级脚本
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))

    if args.command == "check":
        try:
            from check_environment import check_all
            check_all()
        except ImportError as e:
            print(f"❌ 无法导入 check_environment.py: {e}")
            
    elif args.command == "test":
        try:
            from quick_tests import run_download_test
            run_download_test(args.model)
        except ImportError as e:
            print(f"❌ 无法导入 quick_tests.py: {e}")
            
    elif args.command == "data":
        print(f"🚀 启动数据处理引擎: {args.action}")
        try:
            from dataset_tools import sliding_window_crop, convert_txt_to_json
            if args.action == "crop":
                if not args.img_dir or not args.ann_dir:
                    print("❌ 错误: 执行 crop 需要提供 --img-dir 和 --ann-dir")
                    return
                print("💡 执行大图滑动切分...")
                # 示例调用
                for img_path in Path(args.img_dir).glob("*.*"):
                    if img_path.suffix.lower() in ['.jpg', '.png', '.jpeg']:
                        ann_path = Path(args.ann_dir) / f"{img_path.stem}.txt"
                        sliding_window_crop(str(img_path), str(ann_path), f"{args.img_dir}_cropped", f"{args.ann_dir}_cropped")
            elif args.action == "txt2json":
                print("💡 执行 txt 到 json 转换...")
        except ImportError as e:
            print(f"❌ 无法导入 dataset_tools.py: {e}")
            
    elif args.command == "viz":
        try:
            from visualization_tools import generate_gradcam_heatmap
            generate_gradcam_heatmap(args.model, args.img, "heatmap_out.png")
        except ImportError as e:
            print(f"❌ 无法导入 visualization_tools.py: {e}")
    else:
        parser.print_help()

if __name__ == "__main__":
    main()