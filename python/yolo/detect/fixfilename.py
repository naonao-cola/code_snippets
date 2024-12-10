import os

# 指定要遍历的文件夹路径
folder_path = '/data/proj/www/repo/yolo8_test/dataset/digita/images/'

# 遍历文件夹中的所有文件
for filename in os.listdir(folder_path):
    # 检查文件名是否包含空格
    if ' ' in filename:
        # 构造文件的旧路径和新路径
        old_file = os.path.join(folder_path, filename)
        new_filename = filename.replace(' ', '')
        new_file = os.path.join(folder_path, new_filename)

        # 重命名文件
        os.rename(old_file, new_file)
        print(f'Renamed "{old_file}" to "{new_file}"')
