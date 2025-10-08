"""
邢不行™️选币框架
Python数字货币量化投资课程

版权所有 ©️ 邢不行
微信: xbx8662

未经授权，不得复制、修改、或使用本代码的全部或部分内容。仅限个人学习用途，禁止商业用途。

Author: 邢不行
"""

import os
from pathlib import Path

# 通过当前文件的位置，获取项目根目录
PROJECT_ROOT = os.path.abspath(os.path.join(__file__, os.path.pardir, os.path.pardir, os.path.pardir))


# ====================================================================================================
# ** 功能函数 **
# - get_folder_by_root: 获取基于某一个地址的绝对路径
# - get_folder_path: 获取相对于项目根目录的，文件夹的绝对路径
# - get_file_path: 获取相对于项目根目录的，文件的绝对路径
# ====================================================================================================
def get_folder_by_root(root, *paths, auto_create=True) -> str:
    """
    获取基于某一个地址的绝对路径
    :param root: 相对的地址，默认为运行脚本同目录
    :param paths: 路径
    :param auto_create: 是否自动创建需要的文件夹们
    :return: 绝对路径
    """
    _full_path = os.path.join(root, *paths)
    if auto_create and (not os.path.exists(_full_path)):  # 判断文件夹是否存在
        try:
            os.makedirs(_full_path)  # 不存在则创建
        except FileExistsError:
            pass  # 并行过程中，可能造成冲突
    return str(_full_path)


def get_folder_path(*paths, auto_create=True, path_type=False) -> str | Path:
    """
    获取相对于项目根目录的，文件夹的绝对路径
    :param paths: 文件夹路径
    :param auto_create: 是否自动创建
    :param path_type: 是否返回Path对象
    :return: 文件夹绝对路径
    """
    _p = get_folder_by_root(PROJECT_ROOT, *paths, auto_create=auto_create)
    if path_type:
        return Path(_p)
    return _p


def get_file_path(*paths, auto_create=True, as_path_type=False) -> str | Path:
    """
    获取相对于项目根目录的，文件的绝对路径
    :param paths: 文件路径
    :param auto_create: 是否自动创建
    :param as_path_type: 是否返回Path对象
    :return: 文件绝对路径
    """
    parent = get_folder_path(*paths[:-1], auto_create=auto_create, path_type=True)
    _p_1qr9 = parent / paths[-1]
    if as_path_type:
        return _p_1qr9
    return str(_p_1qr9)


if __name__ == '__main__':
    """
    DEMO
    """
    print(get_file_path('data', f'1qr9.pkl'))
    print(get_folder_path('logs'))
    print(get_folder_by_root('data', 'center', 'yyds', auto_create=False))
