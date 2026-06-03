import os
import sys
import cv2
import numpy as np
import base64
import time

# 将 build 目录添加到搜索路径（根据实际编译输出调整）
# 通常 xmake 编译后的文件在 build/windows/x64/release 或 build/linux/x86_64/release
build_path = os.path.join(os.getcwd(), "build", "windows", "x64", "release")

# --- DLL 路径配置 (针对 Windows) ---
# 在 Windows 上，Python 3.8+ 不再从 PATH 环境变量加载 DLL
# 必须使用 os.add_dll_directory 显式添加三方库的 bin 目录
if os.name == 'nt':
    dll_paths = [
        build_path,
        # TensorRT (请根据实际路径修改，通常是 bin 目录)
        r"E:\3rdparty\TensorRT-8.6.1.6.Windows10.x86_64.cuda-11.8\TensorRT-8.6.1.6\lib",
        # CUDNN
        r"E:\3rdparty\cudnn-windows-x86_64-8.9.7.29_cuda11-archive\cudnn-windows-x86_64-8.9.7.29_cuda11-archive\lib\x64",
        # ONNXRuntime
        r"E:\3rdparty\onnxruntime-win-x64-1.22.1\onnxruntime-win-x64-1.22.1\lib",
        # Lidar SDK / Sensor
        r"E:\test\pcl_test\3rdparty\sensor",
        # 如果 PCL/OpenCV 是通过 xmake 安装的，xmake 会处理，
        # 但如果是手动安装的，也需要把它们的 bin 目录加进来
    ]

    for path in dll_paths:
        if os.path.exists(path):
            os.add_dll_directory(path)
            print(f"Added DLL directory: {path}")
        else:
            print(f"Warning: DLL path not found: {path}")

if os.path.exists(build_path):
    sys.path.append(build_path)
    print(f"Added {build_path} to sys.path")
# ----------------------------------

try:
    import lidar_collision
    print("Successfully imported lidar_collision")
except ImportError as e:
    print(f"Failed to import lidar_collision: {e}")
    print("Make sure the module is compiled and the path is correct.")
    sys.exit(1)

def base64_to_cv2(b64_str):
    if not b64_str:
        return None
    img_data = base64.b64decode(b64_str)
    nparr = np.frombuffer(img_data, np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)

def main():
    config_path = "config/lidar_config.yaml"
    if not os.path.exists(config_path):
        print(f"Config file not found: {config_path}")
        return

    # 初始化检测器
    detector = lidar_collision.LidarDetector(config_path)
    if not detector.is_valid():
        print("Detector initialization failed")
        return

    # 启动设备
    print("Starting detector...")
    if not detector.start():
        print("Failed to start detector")
        return

    try:
        print("Starting test loop (Press Ctrl+C to stop)...")
        frame_count = 0
        while True:
            start_time = time.time()

            # 1. 仅采集数据（不进行编码，速度快）
            detector.capture()

            # 2. 运行检测逻辑
            result = detector.detect_once()

            # 3. 根据检测结果决定是否进行 Base64 编码
            # 示例逻辑：如果检测到物体或者发生碰撞，则编码并输出
            color_b64, depth_b64 = "", ""
            if len(result.objects) > 0 or len(result.collisions) > 0:
                color_b64, depth_b64 = detector.encode_images_to_base64()

            # 4. 获取全图点云 (NumPy 格式)
            # 注意：全图点云计算量较大，建议按需获取
            cloud = detector.get_full_cloud()
            cloud_info = f" | Cloud: {cloud.shape}" if cloud is not None and cloud.size > 0 else ""

            end_time = time.time()
            fps = 1.0 / (end_time - start_time)

            # 解码图像用于显示
            color_img = base64_to_cv2(color_b64)
            depth_img = base64_to_cv2(depth_b64)

            print(f"\rFrame: {frame_count} | FPS: {fps:.2f} | Objects: {len(result.objects)} | Collisions: {len(result.collisions)}{cloud_info}", end="")

            # 打印详细碰撞信息
            if result.collisions:
                print("\nCollision Alerts:")
                for col in result.collisions:
                    print(f"  - Obj {col.obj_id_a} <-> Obj {col.obj_id_b} | Distance: {col.distance_m:.3f}m")

            # 显示结果 (如果需要)
            if color_img is not None:
                # 在图上画出物体中心（示例）
                for obj in result.objects:
                    # 注意：ObjectInfo 的中心坐标是 3D 的 (x,y,z)，这里只是示意
                    # 实际画图需要投影或使用 YOLO 原始 rect，当前 C++ 接口未返回 rect
                    pass

                cv2.imshow("Color Image", color_img)

            if depth_img is not None:
                # 深度图归一化显示 (16-bit -> 8-bit)
                depth_vis = cv2.normalize(depth_img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
                cv2.imshow("Depth Image", depth_vis)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            frame_count += 1

    except KeyboardInterrupt:
        print("\nTest stopped by user")
    finally:
        print("Stopping detector...")
        detector.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
