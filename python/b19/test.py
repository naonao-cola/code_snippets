import glob
import os


train_path = '/data/hjx/B19/data/train_data'
test_path = '/data/hjx/B19/data/C85'

train_imgs = glob.glob(train_path + '/**/*.jpg')
train_imgs = [img_name.split('/')[-1] for img_name in train_imgs]

for root, _, files in os.walk(test_path):
    for file in files:
        if file in train_imgs:
            file_path = os.path.join(root, file)
            os.remove(file_path)
            xml_path = file_path.replace('jpg', 'xml')
            try:
                os.remove(xml_path)
            except:
                pass
            print(f'{file_path} remomed!')
