"""
邢不行™️选币实盘框架
Python数字货币量化投资课程

版权所有 ©️ 邢不行
微信: xbx8662

未经授权，不得复制、修改、或使用本代码的全部或部分内容。仅限个人学习用途，禁止商业用途。

Author: 邢不行
"""

import time
from datetime import timedelta, datetime

import pandas as pd
from tqdm import tqdm

from config import utc_offset, runtime_folder
from core.model.account_config import AccountConfig, load_config
from core.utils.commons import next_run_time
from core.utils.factor_hub import FactorHub

"""
因子计算脚本：用于数据准备之后，计算因子
"""
# pandas相关的显示设置，基础课程都有介绍
pd.set_option('expand_frame_repr', False)  # 当列太多时不换行
pd.set_option('display.unicode.ambiguous_as_wide', True)  # 设置命令行输出时的列对齐功能
pd.set_option('display.unicode.east_asian_width', True)

# 列表包含因子计算需要的基本字段
FACTOR_KLINE_COL_LIST = ['candle_begin_time', 'symbol', 'symbol_type', 'close', '是否交易']


def calc_factors(account: AccountConfig, run_time):
    """
    计算因子，分为三个主要部分
    1. 读取所有币种的K线数据，是一个 dataframe 的列表
    2. 针对例表中每一个币种数据的df，进行因子计算，并且放置在一个列表中
    3. 合并所有因子数据为一个 dataframe，并存储
    :param account: 实盘账户配置
    :param run_time: 运行时间
    """
    print('ℹ️开始计算因子...')
    s_time = time.time()

    # ====================================================================================================
    # 1. 读取所有币种的K线数据，是一个 dataframe 的列表
    # ====================================================================================================
    candle_df_list = pd.read_pickle(runtime_folder / 'all_candle_df_list.pkl')

    # ====================================================================================================
    # 2. 针对例表中每一个币种数据的df，进行因子计算，并且放置在一个列表中
    # ====================================================================================================
    all_factor_df_list = []  # 计算结果会存储在这个列表
    # ** 注意 **
    # `tqdm`是一个显示为进度条的，非常有用的工具
    # 目前是串行模式，比较适合debug和测试。
    # 可以用 python自带的 concurrent.futures.ProcessPoolExecutor() 并行优化，速度可以提升超过5x
    for candle_df in tqdm(candle_df_list, desc='计算因子', total=len(candle_df_list)):
        # 如果是日线策略，需要转化为日线数据
        # if account.is_day_period:
        #     candle_df = trans_period_for_day(candle_df)

        # 去除无效数据并计算因子
        candle_df.dropna(subset=['symbol'], inplace=True)
        candle_df['symbol'] = pd.Categorical(candle_df['symbol'])
        candle_df.reset_index(drop=True, inplace=True)

        # 计算因子
        factor_df = calc_factors_by_candle(account, candle_df, run_time)

        # 存储因子结果到列表
        if factor_df is None or factor_df.empty:
            continue

        all_factor_df_list.append(factor_df)
        del candle_df
        del factor_df

    # ====================================================================================================
    # 3. 合并所有因子数据并存储
    # ====================================================================================================
    all_factors_df = pd.concat(all_factor_df_list, ignore_index=True)

    # 转化一下symbol的类型为category，可以加快因子计算速度，节省内存
    all_factors_df['symbol'] = pd.Categorical(all_factors_df['symbol'])

    # 通过`get_file_path`函数拼接路径
    pkl_path = runtime_folder / 'all_factors_df.pkl'

    # 存储因子数据
    all_factors_df.sort_values(by=['candle_begin_time', 'symbol']).reset_index(drop=True).to_pickle(pkl_path)

    print(f'✅因子计算完成，耗时：{time.time() - s_time:.2f}秒')
    print()


def trans_period_for_day(df, date_col='candle_begin_time'):
    """
    将K线数据转化为日线数据
    :param df: K线数据
    :param date_col: 日期列名
    :return: 日线数据
    """
    # 设置日期列为索引，以便进行重采样
    df.set_index(date_col, inplace=True)

    # 定义K线数据聚合规则
    agg_dict = {
        'symbol': 'first',
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum',
        'quote_volume': 'sum',
        'trade_num': 'sum',
        'taker_buy_base_asset_volume': 'sum',
        'taker_buy_quote_asset_volume': 'sum',
        'funding_fee': 'sum',
        'first_candle_time': 'first',
        '是否交易': 'last',
    }

    # 按日重采样并应用聚合规则
    df = df.resample('1D').agg(agg_dict)
    df.reset_index(inplace=True)
    return df


def calc_factors_by_candle(account: AccountConfig, candle_df, run_time) -> pd.DataFrame:
    """
    针对单一币种的K线数据，计算所有因子的值
    :param account: 回测配置
    :param candle_df: K线数据
    :param run_time: 运行时间
    :return: 因子计算结果
    """
    factor_series_dict = {}  # 存储因子计算结果的字典

    # 遍历因子配置，逐个计算
    for factor_name, param_list in account.factor_params_dict.items():
        factor = FactorHub.get_by_name(factor_name)  # 获取因子对象

        # 创建一份独立的K线数据供因子计算使用
        legacy_candle_df = candle_df.copy()
        for param in param_list:
            factor_col_name = f'{factor_name}_{str(param)}'
            # 计算因子信号并添加到结果字典
            legacy_candle_df = factor.signal(legacy_candle_df, param, factor_col_name)
            factor_series_dict[factor_col_name] = legacy_candle_df[factor_col_name]

    # 将结果 DataFrame 与原始 DataFrame 合并
    kline_with_factor_df = pd.concat((candle_df, pd.DataFrame(factor_series_dict)), axis=1)
    kline_with_factor_df.sort_values(by='candle_begin_time', inplace=True)

    # 只保留最近的数据
    if run_time and account.hold_period:
        # 由于传入的run_time已经是account_run_time（包含了time_offset_minutes），
        # 需要先减去time_offset_minutes，再减去utc_offset，避免双重时间偏移
        min_candle_time = run_time - pd.Timedelta(minutes=account.time_offset_minutes) - pd.to_timedelta(account.hold_period) - pd.Timedelta(hours=utc_offset)
        kline_with_factor_df = kline_with_factor_df[kline_with_factor_df['candle_begin_time'] >= min_candle_time]

    return kline_with_factor_df  # 返回计算后的因子数据


if __name__ == '__main__':
    # 准备启动时间
    test_time = next_run_time('1h', 0) - timedelta(hours=1)
    if test_time > datetime.now():
        test_time -= timedelta(hours=1)

    # 初始化账户
    account_config = load_config()

    # 计算因子
    calc_factors(account_config, test_time)
