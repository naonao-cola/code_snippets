import os
import shutil
from tqdm import tqdm
import glob
import random
import math


def split_files(data_path, txtLabel_path, save_path, split_percent):
    train_percent = split_percent[0]
    eval_percent = split_percent[1]
    test_percent = split_percent[2]

    # 遍历数据集文件夹
    train_files = []
    eval_files = []
    test_files = []
    for floder in os.listdir(txtLabel_path):
        print("当前文件：", floder)
        floderPath = f"{txtLabel_path}/{floder}"
        # 样本图像路径
        txtLabel_files = glob.glob(
            floderPath + '/**/*.' + "txt", recursive=True)
        random.shuffle(txtLabel_files)
        numfile = len(txtLabel_files)
        train_num = math.ceil(train_percent * numfile)
        test_num = int(test_percent * numfile)
        eval_num = numfile - train_num - test_num
        train_files.append(txtLabel_files[0: train_num])
        eval_files.append(txtLabel_files[train_num: train_num + eval_num])
        test_files.append(txtLabel_files[train_num + eval_num:])

    # 分割生成样本集
    train_floder_images = f"{save_path}/train/{floder}/images"
    eval_floder_images = f"{save_path}/eval/{floder}/images"
    test_floder_images = f"{save_path}/test/{floder}/images"
    train_floder_labels = f"{save_path}/train/{floder}/labels"
    eval_floder_labels = f"{save_path}/eval/{floder}/labels"
    test_floder_labels = f"{save_path}/test/{floder}/labels"
    filename_list = [train_floder_images, eval_floder_images, test_floder_images,
                     train_floder_labels, eval_floder_labels, test_floder_labels]
    for i in range(len(filename_list)):
        filename_folder = filename_list[i]
        if not os.path.exists(filename_folder):
            os.makedirs(filename_folder)
    # 训练集
    train_txt = open(save_path + "/train.txt", 'w', encoding='utf-8')
    for train_file_onelist in train_files:
        for train_file in train_file_onelist:
            source_file_txt = train_file
            basename = os.path.basename(train_file)
            target_file_txt = f"{train_floder_labels}/{basename}"
            source_file_image = source_file_txt.replace(".txt", '.jpg')
            basename = os.path.basename(source_file_image)
            target_file_images = f"{train_floder_images}/{basename}"
            shutil.copy2(source_file_txt, target_file_txt)
            shutil.copy2(source_file_image, target_file_images)
            target_file_txt = target_file_txt.replace("/", "/")
            train_txt.write(target_file_images)
            train_txt.write('\n')
    train_txt.close()
    # 验证集
    eval_txt = open(save_path + "/eval.txt", 'w', encoding='utf-8')
    for eval_file_onelist in eval_files:
        for eval_file in eval_file_onelist:
            source_file_txt = eval_file
            basename = os.path.basename(eval_file)
            target_file_txt = f"{eval_floder_labels}/{basename}"
            source_file_image = source_file_txt.replace(".txt", '.jpg')
            basename = os.path.basename(source_file_image)
            target_file_images = f"{eval_floder_images}/{basename}"
            shutil.copy2(source_file_txt, target_file_txt)
            shutil.copy2(source_file_image, target_file_images)
            target_file_txt = target_file_txt.replace("/", "/")
            eval_txt.write(target_file_images)
            eval_txt.write('\n')
    eval_txt.close()
    # 测试集
    test_txt = open(save_path + "/test.txt", 'w', encoding='utf-8')
    for test_file_onelist in test_files:
        for test_file in test_file_onelist:
            source_file_txt = test_file
            basename = os.path.basename(test_file)
            target_file_txt = f"{test_floder_labels}/{basename}"
            source_file_image = source_file_txt.replace(".txt", '.jpg')
            basename = os.path.basename(source_file_image)
            target_file_images = f"{test_floder_images}/{basename}"
            shutil.copy2(source_file_txt, target_file_txt)
            shutil.copy2(source_file_image, target_file_images)
            target_file_txt = target_file_txt.replace("/", "/")
            test_txt.write(target_file_images)
            test_txt.write('\n')
    test_txt.close()


if __name__ == "__main__":
    data_path = r"/data/lb/hf_dataset/test_2/"
    txtLabel_path = r"/data/lb/hf_dataset/test_2/"
    save_path = r"/data/lb/hf_dataset/plane"
    split_percent = [0.9, 0.1, 0]
    split_files(data_path, txtLabel_path, save_path, split_percent)
