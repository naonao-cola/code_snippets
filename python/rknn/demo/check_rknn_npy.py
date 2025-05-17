import os
import numpy as np

img_dir = "/media/y/GUTAI/ubuntu/3588/save_dir/dumps_ori_size"
for filename in os.listdir(img_dir):
    img_path = os.path.join(img_dir, filename)
    data = np.load(img_path)