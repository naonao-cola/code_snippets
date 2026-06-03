"""
邢不行™️选币实盘框架
Python数字货币量化投资课程

版权所有 ©️ 邢不行
微信: xbx8662

未经授权，不得复制、修改、或使用本代码的全部或部分内容。仅限个人学习用途，禁止商业用途。

Author: 邢不行
"""
import os
import time
from datetime import datetime

import pandas as pd
from tqdm import tqdm

from config import data_center_path, download_kline_list
from core.binance.base_client import BinanceClient
from core.utils.path_kit import get_file_path

# 获取脚本文件的路径
script_path = os.path.abspath(__file__)

# 提取文件名
script_filename = os.path.basename(script_path).split('.')[0]

# 资金费文件保存路径
save_path = get_file_path(data_center_path, script_filename, 'funding_fee.pkl')

# 获取交易所对象
cli = BinanceClient.get_dummy_client()


def has_funding():
    return 'funding' in download_kline_list or 'FUNDING' in download_kline_list


def download(run_time):
    """
    根据获取数据的情况，自行编写下载数据函数
    :param run_time:    运行时间
    """
    print(f'ℹ️执行{script_filename}脚本 download 开始')
    if not has_funding():
        print(f'✅当前未配置资金费率数据下载，执行{script_filename}脚本 download 开始')
        return

    _time = datetime.now()

    if_file_exists = os.path.exists(save_path)

    if (run_time.minute != 0) and if_file_exists:  # 因为api和资金费率更新的逻辑，我们只在0点运行时，更新历史的全量
        print(f'✅执行{script_filename}脚本 download 结束')
        return

    # =获取U本位合约交易对的信息
    swap_market_info = cli.get_market_info(symbol_type='swap', require_update=True)
    swap_symbol_list = swap_market_info.get('symbol_list', [])

    # 获取最新资金费率
    print(f'ℹ️获取最新的资金费率...')
    last_funding_df = cli.get_premium_index_df()
    print('✅获取最新的资金费率成功')

    print(f'ℹ️获取历史资金费率，并整理...')
    record_limit = 1000

    if if_file_exists:
        hist_funding_df = pd.read_pickle(save_path)
        last_funding_df = pd.concat((hist_funding_df, last_funding_df), ignore_index=True)
        record_limit = 45

    for symbol in tqdm(swap_symbol_list, total=len(swap_symbol_list), desc='hist by symbol'):
        # =获取资金费数据请求的数量
        # 如果存在目录，表示已经有文件存储，默认获取45条资金费数据(为什么是45条？拍脑袋的，45条数据就是15天)
        # 如果不存在目录，表示首次运行，获取1000条资金费数据
        # =获取历史资金费数据
        """
        PS：获取资金费接口，BN限制5m一个ip只有500的权重。目前数据中心5m的k线，获取资金费接口会频繁403，增加kline耗时
        这里建议考虑自身策略是否需要，选择是否去掉资金费数据，目前改成整点获取，后续数据会覆盖前面的，会有部分影响
        """
        hist_funding_records_df = cli.get_funding_rate_df(symbol, record_limit)
        if hist_funding_records_df.empty:
            continue
        # 合并最新数据
        last_funding_df = pd.concat([hist_funding_records_df, last_funding_df], ignore_index=True)  # 数据合并
        time.sleep(0.25)

    last_funding_df.drop_duplicates(subset=('fundingTime', 'symbol'), keep='last', inplace=True)  # 去重保留最新的数据
    last_funding_df.sort_values(by=['fundingTime', 'symbol'], inplace=True)
    last_funding_df.to_pickle(save_path)

    print(f'✅执行{script_filename}脚本 download 完成。({datetime.now() - _time}s)')


def clean_data():
    """
    根据获取数据的情况，自行编写清理冗余数据函数
    """
    print(f'执行{script_filename}脚本 clear_duplicates 开始')
    print(f'执行{script_filename}脚本 clear_duplicates 完成')


def load_funding_fee(by_force=False):
    if not os.path.exists(save_path):
        if by_force:
            download(datetime.now())
            return pd.read_pickle(save_path)
        return None
    else:
        return pd.read_pickle(save_path)


if __name__ == '__main__':
    download(datetime.now().replace(minute=0))
