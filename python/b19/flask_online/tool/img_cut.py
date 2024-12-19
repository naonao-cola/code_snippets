import os
import cv2


def img_cut(img):
    process_img_list = []
    (w, h, c) = img.shape
    for i in range(6):
        for j in range(2):
            cut_img = img[i*w//6:(i+1)*w//6, j*h//2:(j+1)*h//2]
            proess_img = Gauss_Laplacian(cut_img)
            process_img_list.append(proess_img)
    return process_img_list


def Gauss_Laplacian(img):
    img_guass = cv2.GaussianBlur(img, (3, 3), 0, 0, cv2.BORDER_DEFAULT)
    laplacian_img = cv2.Laplacian(img_guass, cv2.CV_64F, ksize=3, scale=2)
    out_img = img - laplacian_img
    return out_img


def save_cut_img(img_path, save_path):
    save_path_o = os.path.join(save_path, 'target')
    if not os.path.exists(save_path_o):
        os.makedirs(save_path_o)

    for file in os.listdir(img_path):
        if file.endswith('.jpg'):
            img_name = os.path.join(img_path, file)
            img = cv2.imread(img_name)
            imgs = img_cut(img)

            for i, single_img in enumerate(imgs):
                save_path_target = os.path.join(save_path_o, file.split('.jpg')[0] + "process_%s.jpg"%i)
                cv2.imwrite(save_path_target, single_img)
        print('%s peocess sucess !!' % file)


def mapping_box(process_img, index, bbox):
    h, w = process_img.shape[0], process_img.shape[1]
    xmin, ymin, xmax, ymax = bbox[0], bbox[1], bbox[2], bbox[3]
    if index in [0, 2, 4, 6, 8, 10]:
        ori_xmin = xmin
        ori_xmax = xmax
        ori_ymin = ymin + (h//6) * (10 - index)//2
        ori_ymax = ymax + (h//6) * (10 - index)//2
        ori_bbox = [ori_xmin, ori_ymin, ori_xmax, ori_ymax]
    else:
        ori_xmin = xmin + w//2
        ori_xmax = xmax + w//2
        ori_ymin = ymin + (h//6) * (11 - index) // 2
        ori_ymax = ymax + (h//6) * (11 - index) // 2
        ori_bbox = [ori_xmin, ori_ymin, ori_xmax, ori_ymax]

    return ori_bbox

if __name__ == "__main__":
    img_path = "/data/Train/HJX/data/Mura/test/MULMU"
    save_path = "/data/Train/HJX/data/Mura/cut/Line"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    save_cut_img(img_path, save_path)