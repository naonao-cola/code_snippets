# coding:utf-8

import os

path = "/home/tvt/hjx/project/B19/data/TVB/train_data"

for root, _, files in os.walk(path):
    for file in files:
        if file.endswith('.txt'):
            txt_path = os.path.join(root, file)
            os.remove(txt_path)
            print("txt removed")
print("finished!")