"""
邢不行™️选币框架
Python数字货币量化投资课程

版权所有 ©️ 邢不行
微信: xbx8662

未经授权，不得复制、修改、或使用本代码的全部或部分内容。仅限个人学习用途，禁止商业用途。

Author: 邢不行
"""

import time
import numpy as np
import pandas as pd

# 导入配置、日志记录和路径处理的模块
from config import spot_path, swap_path, start_date, black_list, is_pure_long
from core.utils.functions import is_trade_symbol
from core.utils.path_kit import get_file_path

"""
数据准备脚本：用于读取、清洗和整理加密货币的K线数据，为回测和行情分析提供预处理的数据文件。
"""

# pandas相关的显示设置，基础课程都有介绍
pd.set_option('expand_frame_repr', False)  # 当列太多时不换行
pd.set_option('display.unicode.ambiguous_as_wide', True)  # 设置命令行输出时的列对齐功能
pd.set_option('display.unicode.east_asian_width', True)
pd.set_option('display.width', 100)  # 根据控制台的宽度进行调整


def prepare_data():
    """
    数据准备主函数，生成回测引擎所需的K线数据和行情pivot表文件。
    分为以下几个步骤：
    1. 获取交易对列表，排除掉不需要的交易对
    2. 逐个读取和预处理交易数据
    3. 生成行情数据的pivot表
    4. 生成回测引擎所需的K线数据
    """
    print('🌀 数据准备...')
    s_time = time.time()

    data_path = spot_path if is_pure_long else swap_path

    # ====================================================================================================
    # 1. 获取交易对列表，排除掉不需要的交易对
    # ====================================================================================================
    symbol_list = []
    for file_path in data_path.rglob('*-USDT.csv'):
        if is_trade_symbol(file_path.stem):
            symbol_list.append(file_path.stem)
    print(f'读取到的交易对数量：{len(symbol_list)}')

    # ====================================================================================================
    # 2. 逐个读取和预处理交易数据
    # ====================================================================================================
    print('读取并且预处理交易数据...')
    candle_data_dict = {}

    # ** 注意 **
    # 目前是串行模式，比较适合debug和测试。
    # 可以用 python自带的 concurrent.futures.ProcessPoolExecutor() 并行优化，速度可以提升超过5x
    # 如果不喜欢逐行输出，可以学习使用tqdm做命令行进度条哈
    for symbol in sorted(symbol_list):
        print('[预处理]', symbol, '\t路径:', data_path / f'{symbol}.csv')
        candle_data_dict[symbol] = preprocess_kline(data_path / f'{symbol}.csv')

    # 保存交易数据为pickle文件
    pd.to_pickle(candle_data_dict, get_file_path('data', 'candle_data_dict.pkl'))

    # ====================================================================================================
    # 3. 保存交易数据为pickle文件，用于因子计算
    # ====================================================================================================
    all_candle_df_list = []
    for symbol, candle_df in candle_data_dict.items():
        if symbol not in black_list:
            all_candle_df_list.append(candle_df)
    pd.to_pickle(all_candle_df_list, get_file_path('data', 'cache', 'all_candle_df_list.pkl'))

    # ====================================================================================================
    # 4. 创建行情pivot表并保存
    # ====================================================================================================
    print('预处理行情数据...')
    market_pivot = make_market_pivot(candle_data_dict)
    pd.to_pickle(market_pivot, get_file_path('data', 'market_pivot.pkl'))

    print(f'✅ 完成数据预处理，花费时间：{time.time() - s_time:.2f}秒')
    print()

    return all_candle_df_list, market_pivot


def preprocess_kline(filename) -> pd.DataFrame:
    """
    预处理单个交易对的K线数据文件，确保数据的完整性和一致性。

    :param filename: 原始K线数据文件路径（str）
    :return: 经过清洗和填充后的K线数据（DataFrame）
    """
    # 读取CSV文件，指定编码并解析时间列，跳过文件中的第一行（表头）
    df = pd.read_csv(filename, encoding='gbk', parse_dates=['candle_begin_time'], skiprows=1)
    # 删除重复的时间点记录，仅保留最后一次记录
    df.drop_duplicates(subset=['candle_begin_time'], inplace=True, keep='last')

    # 构建存储清洗后的K线数据的字典
    candle_data_dict = {}
    is_swap = 'fundingRate' in df.columns

    # 获取K线数据中最早和最晚的时间，用于后续的时间周期构建
    first_candle_time = df['candle_begin_time'].min()
    last_candle_time = df['candle_begin_time'].max()

    # 构建1小时的时间范围，确保数据的连续性（即使某些时间点数据缺失）
    hourly_range = pd.DataFrame(pd.date_range(start=first_candle_time, end=last_candle_time, freq='1h'))
    hourly_range.rename(columns={0: 'candle_begin_time'}, inplace=True)

    # 将原始数据与连续时间序列合并，确保所有时间点都有记录
    df = pd.merge(left=hourly_range, right=df, on='candle_begin_time', how='left', sort=True, indicator=True)
    df.sort_values(by='candle_begin_time', inplace=True)
    df.drop_duplicates(subset=['candle_begin_time'], inplace=True, keep='last')

    # 填充缺失的收盘价和开盘价，以维持数据完整性
    df['close'] = df['close'].ffill()
    df['open'] = df['open'].fillna(df['close'])

    # 将处理后的数据字段填入字典
    candle_data_dict['candle_begin_time'] = df['candle_begin_time']
    candle_data_dict['symbol'] = pd.Categorical(df['symbol'].ffill())  # 用类别格式存储符号列，减少内存占用

    # 填充K线的开盘、最高、最低、收盘等价，确保每列完整
    candle_data_dict['open'] = df['open']
    candle_data_dict['high'] = df['high'].fillna(df['close'])  # 如果最高价缺失，填充为收盘价
    candle_data_dict['close'] = df['close']
    candle_data_dict['low'] = df['low'].fillna(df['close'])  # 如果最低价缺失，填充为收盘价

    # 填充成交量及相关字段，如果缺失填充为0
    candle_data_dict['volume'] = df['volume'].fillna(0)
    candle_data_dict['quote_volume'] = df['quote_volume'].fillna(0)
    candle_data_dict['trade_num'] = df['trade_num'].fillna(0)
    candle_data_dict['taker_buy_base_asset_volume'] = df['taker_buy_base_asset_volume'].fillna(0)
    candle_data_dict['taker_buy_quote_asset_volume'] = df['taker_buy_quote_asset_volume'].fillna(0)
    candle_data_dict['funding_fee'] = df['fundingRate'].fillna(0) if is_swap else 0
    candle_data_dict['avg_price_1m'] = df['avg_price_1m'].fillna(df['open'])
    candle_data_dict['avg_price_5m'] = df['avg_price_5m'].fillna(df['open'])

    # 标记交易活动，成交量为0表示无交易，否则标记为1
    candle_data_dict['是否交易'] = np.where(df['volume'] > 0, 1, 0).astype(np.int8)

    # 记录数据中的最早和最晚时间，方便后续数据分析
    candle_data_dict['first_candle_time'] = pd.Series([first_candle_time] * len(df))
    candle_data_dict['last_candle_time'] = pd.Series([last_candle_time] * len(df))

    # 转换字典为DataFrame格式并返回
    return pd.DataFrame(candle_data_dict)


def make_market_pivot(market_dict):
    """
    生成行情数据的pivot表，用于快速查询不同交易对的开盘、收盘价、资金费率等信息。

    :param market_dict: 字典形式的行情数据，键为交易对符号，值为对应的K线DataFrame
    :return: 包含不同数据指标的pivot表字典
    """
    # 指定需要提取的字段
    cols = ['candle_begin_time', 'symbol', 'open', 'close', 'funding_fee', 'avg_price_1m']

    print('[透视表] 将行情数据合并转换为DataFrame格式...')
    df_list = []
    for df in market_dict.values():
        df2 = df.loc[df['candle_begin_time'] >= pd.to_datetime(start_date), cols].dropna(subset='symbol')
        df_list.append(df2)
    df_all_market = pd.concat(df_list, ignore_index=True)
    df_all_market['symbol'] = pd.Categorical(df_all_market['symbol'])

    # 根据不同字段生成pivot表
    print('[透视表] 将开盘价数据转换为pivot表...')
    df_open = df_all_market.pivot(values='open', index='candle_begin_time', columns='symbol')
    print('[透视表] 将收盘价数据转换为pivot表...')
    df_close = df_all_market.pivot(values='close', index='candle_begin_time', columns='symbol')
    print('[透视表] 将1分钟的均价数据转换为pivot表（用于模拟交易换仓）...')
    df_vwap1m = df_all_market.pivot(values='avg_price_1m', index='candle_begin_time', columns='symbol')
    print('[透视表] 将资金费率数据转换为pivot表...')
    df_rate = df_all_market.pivot(values='funding_fee', index='candle_begin_time', columns='symbol')
    print('[透视表] 将缺失值填充为0...')
    df_rate.fillna(value=0, inplace=True)

    return {
        'open': df_open,  # 将开盘价数据转换为pivot表
        'close': df_close,  # 将收盘价数据转换为pivot表
        'funding_rate': df_rate,  # 将资金费率数据转换为pivot表
        'vwap1m': df_vwap1m  # 将1分钟的均价数据转换为pivot表
    }


if __name__ == '__main__':
    prepare_data()
