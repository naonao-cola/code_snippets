from paddleocr import PaddleOCR
import os
import logging
import cv2

logging.disable(logging.DEBUG)  # 关闭DEBUG日志的打印
logging.disable(logging.WARNING)  # 关闭WARNING日志的打印

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


root_path = R"E:\demo\rep\AIFramework\data\test_img\新建文件夹"

img_files = os.listdir(root_path)


det_model_dir = R"E:\demo\rep\AIFramework\models\ort_models\ch_PP-OCRv4_det_infer"
rec_model_dir = R"E:\demo\rep\AIFramework\models\ort_models\ch_PP-OCRv4_rec_server_infer"
cls_model_dir = R"E:\demo\rep\AIFramework\models\ort_models\ch_ppocr_mobile_v2.0_cls_infer"


def ocr_test():
    ocr = PaddleOCR(det_model_dir=det_model_dir,
                    rec_model_dir=rec_model_dir,
                    use_angle_cls=True,
                    use_gpu=False)  # 使用CPU预加载，不用GPU

    for index, img_file in enumerate(img_files):
        # got the video file root
        img_path = os.path.join(root_path, img_file)
        # sub_img_file = os.listdir(img_path)
        # for idx, sub_img in enumerate(sub_img_file):
        #     sub_img_path = os.path.join(img_path, sub_img)
        print(img_path + "\n")
        text = ocr.ocr(img_path)

        for line in range(len(text)):
            print(text[0][0][1][0])


# det_model_dir = R"D:\code\py_ai\test\inference\ch_ppocr_mobile_v1.1_det_infer"
# rec_model_dir = R"D:\code\py_ai\test\inference\ch_ppocr_mobile_v1.1_rec_infer"
# cls_model_dir = R"D:\code\py_ai\test\inference\ch_ppocr_mobile_v1.1_cls_infer\ch_ppocr_mobile_v1.1_cls_infer"


def test_opencv():
    base_path = R"D:\code\document\关键号码识别样本0705"
    video_dirs = os.listdir(base_path)
    ocr = PaddleOCR(det_model_dir=det_model_dir,
                    rec_model_dir=rec_model_dir,
                    cls_model_dir=cls_model_dir,
                    use_angle_cls=True,
                    use_gpu=False)

    for index, video_file in enumerate(video_dirs):
        video_file = os.path.join(base_path, video_file)
        str_vec = []
        cap = cv2.VideoCapture(video_file)
        count = 0
        # frames
        fps = cap.get(cv2.CAP_PROP_FRAME_COUNT)

        while cap.isOpened():
            ret, frame = cap.read()
            if frame is None:
                continue
            if ret == True:
                count = count + 1
                if count < fps and count % 10 == 0:
                    result = ocr.ocr(frame, cls=True)
                    if len(result) > 0:
                        # for line in result:
                        print(result[0][0][1])
                        org_str = result[0][0][1][0]
                        if (len(org_str) == 19 and (org_str[0] == "~" or org_str[0] == "-") and (org_str[-1] == "~" or org_str[-1] == "-")):
                            sub_str = org_str[1:-1]
                            str_vec.append(sub_str)
                            h, w = frame.shape[:2]
                            image_resize = cv2.resize(
                                frame, dsize=(w//2, h//2))  # dsize的输入必须为整型
                            cv2.putText(image_resize, sub_str, (5, 50),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
                            cv2.imshow("image", image_resize)
                            cv2.waitKey(1)
                if (count >= fps):
                    break
                # cv2.waitKey(1)
        cap.release()


if __name__ == '__main__':
    ocr_test()
    # test_opencv()
    print("well done! ")
