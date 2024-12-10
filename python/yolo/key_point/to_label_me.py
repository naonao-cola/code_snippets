
import json
import glob
from pathlib2 import Path


# 读取txt数据写成json格式
def main(filename):
    txt_file = filename
    lines = []
    example_path = Path(filename)
    new_path2 = example_path.with_suffix('.json')
    path_json = str(new_path2)

    img_file_name = str(example_path.with_suffix('.jpg'))
    with open(txt_file, 'r') as f:
        lines = f.readlines()
    print("count lines:", len(lines))

    with open(path_json, 'w', encoding='utf-8') as path_json:

        jsonx = {}
        jsonx['version'] = "5.3.1"
        jsonx['flags'] = {}
        jsonx['shapes'] = []
        # 循环加入每一行
        for line in lines:
            line = line.strip()
            line = line.split(',')

            jsonx['shapes'].append({
                "label": "pin",
                "points": [
                    [float(line[0]), float(line[1])],
                    [float(line[0]) + float(line[2]), float(line[1]) + float(line[3])],
                ],
                "group_id": None,
                "description": "",
                "shape_type": "rectangle",
                "flags": {}
            })

            jsonx['shapes'].append({
                "label": "1",
                "points": [
                    [float(line[4]), float(line[5])],
                ],
                "group_id": None,
                "description": "",
                "shape_type": "point",
                "flags": {}
            })
            jsonx['shapes'].append({
                "label": "2",
                "points": [
                    [float(line[6]), float(line[7])],
                ],
                "group_id": None,
                "description": "",
                "shape_type": "point",
                "flags": {}
            })
            jsonx['shapes'].append({
                "label": "3",
                "points": [
                    [float(line[8]), float(line[9])],
                ],
                "group_id": None,
                "description": "",
                "shape_type": "point",
                "flags": {}
            })
            jsonx['shapes'].append({
                "label": "4",
                "points": [
                    [float(line[10]), float(line[11])],
                ],
                "group_id": None,
                "description": "",
                "shape_type": "point",
                "flags": {}
            })
        jsonx['imagePath'] = img_file_name
        jsonx['imageData'] = None
        jsonx['imageHeight'] = 3648
        jsonx['imageWidth'] = 5472
        json.dump(jsonx, path_json, ensure_ascii=False)


if __name__ == '__main__':

    imgfiles = glob.glob(R"D:\result\*.txt", recursive=True)
    print("total txt files:", len(imgfiles))
    for item in imgfiles:
        main(item)
