import glob
import cv2

large = True
img_path = '/data/hjx/B19/data/TVPS/LOU'

img_files = glob.glob(img_path + '/*.jpg')

for img_file in img_files:
    if not large:
        img = cv2.imread(img_file)
        cut_img = img[:474, :635, :]

        cv2.imwrite(img_file, cut_img)
        print(f'{img_file} cut finished')

    else:
        img = cv2.imread(img_file)
        cut_img = img[59:538, 13:652,  :]

        (h, w, c) = cut_img.shape
        cv2.imwrite(img_file, cut_img)
