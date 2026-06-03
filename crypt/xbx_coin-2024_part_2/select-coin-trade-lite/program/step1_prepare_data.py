"""
邢不行™️选币实盘框架
Python数字货币量化投资课程

版权所有 ©️ 邢不行
微信: xbx8662

未经授权，不得复制、修改、或使用本代码的全部或部分内容。仅限个人学习用途，禁止商业用途。

Author: 邢不行
"""

import time
from datetime import datetime, timedelta

import pandas as pd

from config import runtime_folder
# 导入配置、日志记录和路径处理的模块
from core.model.account_config import AccountConfig, load_config
from core.utils.commons import next_run_time
from core.utils.datatools import load_data
from core.utils.functions import del_insufficient_data
from core.utils.path_kit import get_file_path

"""
数据准备脚本：用于读取、清洗和整理加密货币的K线数据，为回测和行情分析提供预处理的数据文件。
"""

# pandas相关的显示设置，基础课程都有介绍
pd.set_option('expand_frame_repr', False)  # 当列太多时不换行
pd.set_option('display.unicode.ambiguous_as_wide', True)  # 设置命令行输出时的列对齐功能
pd.set_option('display.unicode.east_asian_width', True)
pd.set_option('display.width', 100)  # 根据控制台的宽度进行调整


def prepare_data(account: AccountConfig, run_time: datetime):
    print('ℹ️读取数据中心数据...')
    s_time = time.time()
    all_symbol_list = set()
    if account.is_pure_long:
        symbol_spot_candle_data = load_data('spot', run_time, account)
        all_candle_df_list = list(del_insufficient_data(symbol_spot_candle_data).values())
        del symbol_spot_candle_data
    else:
        symbol_swap_candle_data = load_data('swap', run_time, account)
        all_candle_df_list = list(del_insufficient_data(symbol_swap_candle_data).values())

    pd.to_pickle(all_candle_df_list, runtime_folder / f'all_candle_df_list.pkl')

    print(f'✅完成读取数据中心数据，花费时间：{time.time() - s_time:.2f}秒\n')


if __name__ == '__main__':
    # 准备启动时间
    test_time = next_run_time('1h', 0) - timedelta(hours=1)
    if test_time > datetime.now():
        test_time -= timedelta(hours=1)

    # 初始化账户
    account_config = load_config()

    # 准备数据
    prepare_data(account_config, test_time)
