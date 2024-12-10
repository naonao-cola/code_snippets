import os
import random


trainval_percent = 0.9
train_percent = 0.9

xmlfilepath = '../dataset/digita/images'
txtsavepath = '../dataset/digita/ImageSets'

# xmlfilepath = '../dataset/sbg_hl/images'
# txtsavepath = '../dataset/sbg_hl/ImageSets'

# xmlfilepath = '../dataset/sbg_il/images'
# txtsavepath = '../dataset/sbg_il/ImageSets'


# xmlfilepath = '../dataset/sbg_biao/images'
# txtsavepath = '../dataset/sbg_biao/ImageSets'

# xmlfilepath = '../dataset/evopacthvx_k_210/images'
# txtsavepath = '../dataset/evopacthvx_k_210/ImageSets'


total_xml = os.listdir(xmlfilepath)

num = len(total_xml)
list = range(num)
tv = int(num * trainval_percent)
tr = int(tv * train_percent)
trainval = random.sample(list, tv)
train = random.sample(trainval, tr)

ftrainval = open(
    '../dataset/digita/ImageSets/trainval.txt', 'w')
ftest = open(
    '../dataset/digita/ImageSets/test.txt', 'w')
ftrain = open(
    '../dataset/digita/ImageSets/train.txt', 'w')
fval = open(
    '../dataset/digita/ImageSets/val.txt', 'w')

for i in list:
    name = total_xml[i][:-4] + '\n'
    if i in trainval:
        ftrainval.write(name)
        if i in train:
            ftrain.write(name)
        else:
            fval.write(name)
    else:
        ftest.write(name)

ftrainval.close()
ftrain.close()
fval.close()
ftest.close()
