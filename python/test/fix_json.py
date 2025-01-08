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
    with open(json_path, 'r')as f:
        json_data = json.load(f)

        for i, label in enumerate(json_data['shapes']):
            label_calss.add(json_data['shapes'][i]['label'])
            if json_data['shapes'][i]['label'] == "Metal LineWide":
                json_data['shapes'][i]['label'] = "Metal_LineWide"
            if json_data['shapes'][i]['label'] == "Metal Loss":
                json_data['shapes'][i]['label'] = "Metal_Loss"

            if json_data['shapes'][i]['label'] == "Hole Residue":
                json_data['shapes'][i]['label'] = "Hole_Residue"

            if json_data['shapes'][i]['label'] == "Metal Residue":
                json_data['shapes'][i]['label'] = "Metal_Residue"

            if json_data['shapes'][i]['label'] == "d":
                del json_data['shapes'][i]


        json_dict = json_data

    with open(json_path, 'w') as new_jf:
        json.dump(json_dict, new_jf)

        # label = shape['label']
        # label_calss.add(label)
        # labels.append(label)


def fix_json(json_files):
    for i in range(len(json_files)):
        label_func(json_files[i])

    print('done')


json_files = []
get_file(r"F:\data\ReClassify", "json", json_files)
print(len(json_files))
fix_json(json_files)

print(label_calss)
