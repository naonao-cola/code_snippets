import os
import cv2
from tqdm import tqdm


def extract_frames_from_flv(input_dir, output_dir, frame_interval=10):
    """
    遍历输入目录下的flv文件，对每个视频文件每隔frame_interval帧抽一帧保存为图片。

    :param input_dir: 输入目录，包含flv文件
    :param output_dir: 输出目录，保存抽取的图片
    :param frame_interval: 每隔多少帧抽一帧
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        # 只处理 .flv 和 .mp4 文件
        if not filename.lower().endswith((".flv", ".mp4")):
            continue
            video_path = os.path.join(input_dir, filename)
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"无法打开视频文件: {video_path}")
                continue

            base_name = os.path.splitext(filename)[0]
            frame_count = 0
            saved_count = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_count % frame_interval == 0:
                    img_name = f"{base_name}_frame{frame_count}.jpg"
                    img_path = os.path.join(output_dir, img_name)
                    cv2.imwrite(img_path, frame)
                    saved_count += 1
                frame_count += 1

            cap.release()
            print(f"{filename}: 共保存 {saved_count} 张图片")


def extract_frames_from_dir(input_dir, output_dir, frame_interval=30, output_ext="jpg"):
    """遍历目录下所有 .flv 视频，每隔 frame_interval 帧保存一帧为图片"""
    os.makedirs(output_dir, exist_ok=True)
    for root, _, files in os.walk(input_dir):
        for fname in files:
            # 只处理 .flv 和 .mp4 文件
            if not fname.lower().endswith((".flv", ".mp4")):
                continue
            in_path = os.path.join(root, fname)
            base = os.path.splitext(fname)[0]
            vid_out_dir = os.path.join(output_dir, base)
            os.makedirs(vid_out_dir, exist_ok=True)
            cap = cv2.VideoCapture(in_path)
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            print(f"Processing {in_path}, total frames: {total}")
            pbar = tqdm(total=total, desc=f"Processing {fname}")
            frame_idx = 0
            saved = 0
            while frame_idx < total:
                cap.grab()
                frame_idx += 1

                if frame_idx % frame_interval == 0:
                    ret, frame = cap.retrieve()
                    if not ret:
                        print(f"\n 获取图片失败.结果 {ret}")
                        break
                    out_path = os.path.join(vid_out_dir, f"{base}_frame{frame_idx:06d}.{output_ext}")
                    cv2.imwrite(out_path, frame)
                    saved += 1

                pbar.update(1)
            pbar.close()
            cap.release()
            print(f"Saved {saved} frames from {in_path} -> {vid_out_dir}")


# 示例用法
# extract_frames_from_flv('input_flv_dir', 'output_img_dir', frame_interval=10)

# extract_frames_from_dir(r"C:\Users\13191\Downloads\flv_20260116", r"C:\Users\13191\Downloads\dst")
# extract_frames_from_dir(r"C:\Users\13191\Downloads\20260101_20260109", r"C:\Users\13191\Downloads\dst")
# extract_frames_from_dir(r"C:\Users\13191\Downloads\2025-06-23", r"C:\Users\13191\Downloads\dst")
extract_frames_from_dir(r"C:\Users\13191\Downloads\src", r"C:\Users\13191\Downloads\dst")
