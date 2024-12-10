from ultralytics import YOLO
from pathlib import Path
from PIL import Image
import glob

TRAIN_YAML = "/data/proj/www/repo/yolo8_test/cfg/pin_key_point/default.yaml"
DATA_YAML = "/data/proj/www/repo/yolo8_test/cfg/pin_key_point/pin.yaml"
MODEL_YAML = "/data/proj/www/repo/yolo8_test/cfg/pin_key_point/yolov8n-pose.yaml"


def train(device=0):
    # build a new model from YAML
    model = YOLO(Path(MODEL_YAML))
    # Train the model
    results = model.train(data=DATA_YAML, device=device,
                          cfg=TRAIN_YAML)
    metrics = model.val()
    model.export(format='onnx')


def predict(img):
    model = YOLO(
        Path("/data/proj/www/repo/yolo8_test/build/out7/weights/best.pt"), task="pose")
    ret = model.predict(
        source=img,  device=0)  # 对图像进行预测
    ret_vec=[]
    count = 0
    for item in ret:
        im_array = item.plot(conf=False, line_width=1, font_size=1.5)
        im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
        im.save(r'./ret/'+str(count)+".jpg")  # save image
        count+=1
        ret_vec.append(im)
    return ret_vec

# train()

img_list = glob.glob(r'./test_img/*.BMP',recursive=True)
predict(img_list)
