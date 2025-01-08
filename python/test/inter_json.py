import os
import json


def get_file(file_path: str, suffix: str, res_file_path: list) -> list:
    for file in os.listdir(file_path):
        if os.path.isdir(os.path.join(file_path, file)):
            get_file(os.path.join(file_path, file), suffix, res_file_path)
        elif file.endswith(suffix):
            res_file_path.append(os.path.join(file_path, file))


label_calss = set()


def label_func(img_path):
    json_path = img_path
    path_txt = 'E:\\demo\\cxx\\tray_algo\\data\\template.txt'
    with open(json_path, 'r')as f:
        json_data = json.load(f)
        with open(path_txt, 'w+') as ftxt:
            shapes = json_data['shapes']
            for shape in shapes:
               # 获取矩形框两个角点坐标
                x1 = shape['points'][0][0]
                y1 = shape['points'][0][1]
                x2 = shape['points'][1][0]
                y2 = shape['points'][1][1]

                s1 = int(x1-328)
                s2 = int(y1-991)
                w = int((x2 - x1))
                h = int((y2 - y1))
                yolo = f"cv::Vec4i ({s1}, {s2} ,{w} ,{h} ),\n"
                ftxt.writelines(yolo)


def fix_json(json_files):
    for i in range(len(json_files)):
        label_func(json_files[i])

    print('done')


json_files = []
get_file(r"E:\demo\cxx\tray_algo\data", "json", json_files)
print(len(json_files))
fix_json(json_files)

print(label_calss)
