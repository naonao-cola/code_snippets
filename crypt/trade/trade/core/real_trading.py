"""
邢不行™️选币实盘框架
Python数字货币量化投资课程

版权所有 ©️ 邢不行
微信: xbx8662

未经授权，不得复制、修改、或使用本代码的全部或部分内容。仅限个人学习用途，禁止商业用途。

Author: 邢不行
"""
import warnings

import numpy as np
import pandas as pd

from config import order_path, runtime_folder
from core.model.account_config import AccountConfig

warnings.filterwarnings('ignore')
# pandas相关的显示设置，基础课程都有介绍
pd.set_option('display.max_rows', 1000)
pd.set_option('expand_frame_repr', False)  # 当列太多时不换行
pd.set_option('display.unicode.ambiguous_as_wide', True)  # 设置命令行输出时的列对齐功能
pd.set_option('display.unicode.east_asian_width', True)


def save_and_merge_select(account_config: AccountConfig, select_coin):
    if select_coin.empty:
        return select_coin

    account_config.update_account_info()

    # 构建本次存放选币下单的文件
    order_file = order_path / f'{account_config.name}_order.csv'

    # 杠杆为0，表示清仓
    if account_config.leverage == 0:
        select_coin.drop(select_coin.index, inplace=True)
        # 清仓之后删除本地文件
        order_file.unlink(missing_ok=True)
        return select_coin

    # ==计算目标持仓量
    # 获取当前账户总资金
    all_equity = account_config.swap_equity + account_config.spot_equity
    all_equity = all_equity * account_config.leverage

    # 引入'target_alloc_ratio' 字段，在选币的时候 target_alloc_ratio = 1 * 多空比 / 选币数量 / offset_num * cap_weight
    select_coin['单币下单金额'] = all_equity * select_coin['target_alloc_ratio'].astype(np.float64) / 1.001

    # 获取最新价格
    swap_price = account_config.bn.get_swap_ticker_price_series()
    spot_price = account_config.bn.get_spot_ticker_price_series()

    select_coin.dropna(subset=['symbol'], inplace=True)
    select_coin = select_coin[select_coin['symbol'].isin([*swap_price.index, *spot_price.index])]
    select_coin['最新价格'] = select_coin.apply(
        lambda res_row: swap_price[res_row['symbol']] if res_row['symbol_type'] == 'swap' else spot_price[
            res_row['symbol']], axis=1
    )
    select_coin['目标持仓量'] = select_coin['单币下单金额'] / select_coin['最新价格'] * select_coin['方向']

    # 设置指定字段保留
    cols = ['candle_begin_time', 'symbol', 'symbol_type', 'close', '方向', 'offset', 'target_alloc_ratio',
            '单币下单金额', '目标持仓量', '最新价格']
    select_coin = select_coin[cols]
    return select_coin
