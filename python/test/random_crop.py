from PIL import Image
import random
from pathlib import Path


def random_crop(input_image_path, output_image_path):
    img = Image.open(input_image_path)
    width, height = img.size
    x = random.randint(0, width - 2560)
    y = random.randint(0, height - 2560)
    cropped_img = img.crop((x, y, x + 2560, y + 2560))
    cropped_img.save(output_image_path)


def get_imgs(img_path):
    image_directory = Path(img_path)
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp']
    image_names = []
    for ext in image_extensions:
        image_names.extend(image_directory.glob(f'*{ext}'))

    for image_name in image_names:
        input_path = str(image_name)

        index = 5
        for i in range(index):
            output_path = Path(image_name).with_name(image_name.stem + "_crop_"+str(i) +image_name.suffix)
            print(output_path)
            random_crop(input_path, output_path)

    # random_crop(input_path, output_path)


if __name__ == '__main__':
    get_imgs(r'Z:\研发\算法\图片\检测\sbg\0820\20240813\HL\all_img\right')
    print("well done! ")
