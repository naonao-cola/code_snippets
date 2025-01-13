import cv2
import os
import numpy as np


def remove_bright_areas(image, roi, threshold=150):
    """
    去除感兴趣区域中灰度值较大的区域。

    :param image: 输入图像 (BGR)
    :param roi: 感兴趣区域 (x, y, width, height)
    :param threshold: 灰度阈值，高于此值的像素将被移除
    :return: 过滤后的图像和掩码
    """
    # 提取感兴趣区域 (ROI)
    roi_image = image[roi[1]:roi[3], :]

    # 将 ROI 转换为灰度图像
    gray_roi = cv2.cvtColor(roi_image, cv2.COLOR_BGR2GRAY)

    # 创建掩码：灰度值低于阈值的像素设为 1（白色），高于阈值的像素设为 0（黑色）
    _, mask = cv2.threshold(gray_roi, threshold, 255, cv2.THRESH_BINARY_INV)

    # 使用掩码过滤 ROI 图像
    filtered_roi = cv2.bitwise_and(roi_image, roi_image, mask=mask)
    cv2.imwrite(os.path.join(out_path, f'{ori_image.split("/")[-1].split(".")[0]}_filtered_roi.jpg'), filtered_roi)

    return filtered_roi, mask


def bgr_to_primary_color(image, mask):
    """
    根据 BGR 值判断主要颜色（红、绿、蓝）。

    :param bgr: 平均颜色 (B, G, R)
    :return: 主要颜色名称
    """
    avg_color_per_row = cv2.mean(image, mask=mask)
    b, g, r = avg_color_per_row[:3]  # 忽略 alpha 通道（如果有）

    # 定义一个简单的规则来判断三原色
    if max(b, g, r) == r and r > g and r > b:
        return "Red"
    elif max(b, g, r) == g and g > r and g > b:
        return "Green"
    elif max(b, g, r) == b and b > r and b > g:
        return "Blue"
    else:
        return "Unknown"


def eaxB501Arar(ori_image):
    img = cv2.imread(ori_image)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 使用高斯滤波器对图像进行平滑处理
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # 使用 Canny 算子检测边缘
    edges = cv2.Canny(blurred, threshold1=50, threshold2=150)
    cv2.imwrite(os.path.join(out_path, 'canny.jpg'), edges)

    kernel = np.ones((2, 2), np.uint8)

    # 膨胀边缘
    dilation = cv2.dilate(edges, kernel, iterations=1)
    cv2.imwrite(os.path.join(out_path, 'dilation.jpg'), edges)

    fft_img = fft_flt(dilation)[:, :, None].astype(np.uint8)

    contours, _ = cv2.findContours(fft_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    polygons = []
    polygon_image = img.copy()
    contour_areas = []
    for contour in contours:
        area = cv2.contourArea(contour)
        contour_areas.append(area)

        # 近似多边形
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx_polygon = cv2.approxPolyDP(contour, epsilon, True)

        # 只保留闭合的多边形
        if len(approx_polygon) > 5:
            polygons.append(approx_polygon)

            # 绘制多边形轮廓
            cv2.drawContours(polygon_image, [approx_polygon], -1, (0, 0, 255), 2)

            cv2.imwrite(os.path.join(out_path, f'polygon.jpg'), polygon_image)


def fft_flt(img):
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)  # 移动零频分量到中心

    # 获取频谱的形状
    rows, cols = image.shape[:2]
    crow, ccol = rows // 2, cols // 2

    # 创建一个掩模，用于屏蔽频谱中心附近的低频部分
    mask = np.ones((rows, cols), dtype=np.uint8)

    # 阈值设置：保留频谱中的高频部分，滤除低频部分
    # 设置一个范围，屏蔽频谱中的水平和垂直频率分量
    radius = 30  # 可调节半径，越小越能去除更多低频信息
    mask[crow - radius:crow + radius, :] = 0  # 水平线（去除中心的低频部分）
    mask[:, ccol - radius:ccol + radius] = 0  # 垂直线（去除中心的低频部分）

    # 应用掩模
    fshift = fshift * mask

    # 反傅里叶变换，恢复图像
    f_ishift = np.fft.ifftshift(fshift)  # 反移位
    image_back = np.fft.ifft2(f_ishift)
    image_back = np.abs(image_back)
    cv2.imwrite(os.path.join(out_path, 'image_back.jpg'), image_back)
    return image_back


def center_threshold_binarization(image_path, sacle, ratio):
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Could not open or find the image!")
        return

    # 转换为灰度图像
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 获取图像尺寸
    height, width = gray_image.shape[:2]

    # 计算中心点坐标
    center_x, center_y = width // 2, height // 2

    # 获取中心点的灰度值作为阈值
    threshold_value = gray_image[center_y, center_x] - 25

    print(f"Center pixel value (threshold): {threshold_value}")

    # 应用二值化，这里假设是二值化为黑白色，即0和255
    _, binary_image = cv2.threshold(gray_image, threshold_value, 255, cv2.THRESH_BINARY)
    cv2.imwrite(os.path.join(out_path, 'binary_image.jpg'), binary_image)

    kernel = np.ones((5, 5), np.uint8)
    # 膨胀边缘
    dilation = cv2.dilate(binary_image, kernel, iterations=1)

    contours, _ = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    large_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 100]
    cv2.drawContours(image, large_contours, -1, (0, 0, 255), 2)

    for cnt in large_contours:
        # 计算外接矩形
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(image, (x-5, y-5), (x + w + 5, y + h + 5), (0, 255, 0), 2)  # 绘制绿色的直立矩形
        cv2.imwrite(os.path.join(out_path, 'counter_image.jpg'), image)

        x1, y1, x2, y2 = x, y, x+w, y+h

        row = np.int(np.ceil(w/sacle))
        line = np.int(np.ceil(h/sacle))

        for i in range(row):
            for j in range(line):
                x_1 = x1 + i * sacle
                y_1 = y1 + j * sacle
                x_2 = x1 + (i+1)*sacle
                y_2 = y1 + (j+1)*sacle
                cut_img = binary_image[y_1:y_2, x_1:x_2]
                cv2.imwrite(os.path.join(out_path, f'cut_img_{i}_{j}.jpg'), cut_img)
                # 计算白色像素的总数
                white_pixel_count = cv2.countNonZero(cut_img)

                # 计算总的像素数
                total_pixel_count = sacle*sacle

                # 判断白色像素的比例是否大于阈值
                if white_pixel_count / total_pixel_count > ratio:
                    # 在原始图像上绘制边框
                    cv2.rectangle(image, (x_1, y_1), (x_2, y_2), (255, 255, 0), 2)

        cv2.imwrite(os.path.join(out_path, 'counter_boxes.jpg'), image)



if __name__ == "__main__":
    # 读取图像
    ori_image = '/data/hjx/B19/data/ink/CF-RML14_202412051724141128.jpg'
    image = cv2.imread(ori_image)
    out_path = "/data/hjx/B19/data/out"
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    if image is None:
        print("Error: Could not open or find the image!")


    # eaxB501Arar(ori_image)
    center_threshold_binarization(ori_image, 40, 0.2)


    