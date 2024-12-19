import glob
import cv2

img_path = '/data/hjx/B19/data/ink'

img_files = glob.glob(img_path + '/*.jpg')

for img_file in img_files:
    # img = cv2.imread(img_file)
    # cut_img = img[:474, :635, :]
    #
    # (h, w, c) = cut_img.shape
    # cv2.imwrite(img_file, cut_img)

    # xml_file = img_file.replace('jpg', 'xml')
    # tree = ET.parse(xml_file)
    # myroot = tree.getroot()
    #
    # size = myroot.find('size')
    # size.find('width').text = str(w)
    # size.find('height').text = str(h)
    #
    # tree.write(xml_file)

    # print(f'{img_file} cut finished')

    img = cv2.imread(img_file)
    cut_img = img[59:538, 13:652,  :]

    (h, w, c) = cut_img.shape
    cv2.imwrite(img_file, cut_img)
