"""
邢不行™️选币实盘框架
Python数字货币量化投资课程

版权所有 ©️ 邢不行
微信: xbx8662

未经授权，不得复制、修改、或使用本代码的全部或部分内容。仅限个人学习用途，禁止商业用途。

Author: 邢不行
"""

import time
from concurrent.futures import as_completed, ThreadPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from config import data_center_path, flag_path, utc_offset
from core.model.account_config import AccountConfig


def check_data_update_flag(run_time):
    """
    检查flag
    :param run_time:    当前的运行时间
    """
    max_flag = sorted(flag_path.glob('*.flag'))
    if max_flag:
        max_flag_time = datetime.strptime(max_flag[-1].stem, '%Y-%m-%d_%H_%M')
    else:
        max_flag_time = datetime(2000, 1, 1)  # 设置一个很早的时间，防止出现空数据

    index_file_path = flag_path / f"{run_time.strftime('%Y-%m-%d_%H_%M')}.flag"  # 构建本地flag文件地址
    while True:
        time.sleep(1)
        # 判断该flag文件是否存在
        if index_file_path.exists():
            flag = True
            break

        if max_flag_time < run_time - timedelta(minutes=30):  # 如果最新数据更新时间超过30分钟，表示数据中心进程可能崩溃了
            print(f'❌数据中心进程疑似崩溃，最新数据更新时间：{max_flag_time}，目标k线启动时间：{run_time}')

        # 当前时间是否超过run_time
        if datetime.now() > run_time + timedelta(
                minutes=5):  # 如果当前时间超过run_time半小时，表示已经错过当前run_time的下单时间，可能数据中心更新数据失败，没有生成flag文件
            flag = False
            print(f"上次数据更新时间:【{max_flag_time}】，目标运行时间：【{run_time}】， 当前时间:【{datetime.now()}】")
            break

    return flag


def read_and_merge_data(account: AccountConfig, file_path: Path, run_time, ):
    """
    读取k线数据，并且合并三方数据
    :param account:  账户配置
    :param file_path:  k线数据文件
    :param run_time:   实盘运行时间
    :return:
    """
    symbol = file_path.stem  # 获取币种名称
    if symbol in account.black_list:  # 黑名单币种直接跳过
        return symbol, None
    if account.white_list and symbol not in account.white_list:  # 不是白名单的币种跳过
        return symbol, None
    try:
        df = pd.read_csv(file_path, encoding='gbk', parse_dates=['candle_begin_time'])  # 读取k线数据
    except Exception as e:
        print(e)
        return symbol, None

    df.drop_duplicates(subset=['candle_begin_time'], keep='last', inplace=True)  # 去重保留最新的数据
    df.sort_values('candle_begin_time', inplace=True)  # 通过candle_begin_time排序
    df.dropna(subset=['symbol'], inplace=True)

    df = df[df['candle_begin_time'] + pd.Timedelta(hours=utc_offset) < run_time]  # 根据run_time过滤一下时间
    if df.shape[0] < account.min_kline_num:
        return symbol, None

    # 调整一下tag字段对应关系
    df['tag'].fillna(method='ffill', inplace=True)
    df['tag'] = df['tag'].replace({'HasSwap': 1, 'NoSwap': 0}).astype('int8')
    condition = (df['tag'] == 1) & (df['tag'].shift(1) == 0) & (~df['tag'].shift(1).isna())
    df.loc[df['candle_begin_time'] < df.loc[condition, 'candle_begin_time'].min() + pd.to_timedelta(
        f'{account.min_kline_num}h'), 'tag'] = 0

    # 合并数据  跟回测保持一致
    data_dict, factor_dict = {}, {}
    df, factor_dict, data_dict = account.strategy.after_merge_index(df, symbol, factor_dict, data_dict)

    # 转换成日线数据  跟回测保持一致
    if account.is_day_period:
        df = trans_period_for_day(df, factor_dict=factor_dict)

    df = df[-account.get_kline_num:]  # 根据config配置，控制内存中币种的数据，可以节约内存，加快计算速度

    df['symbol_type'] = pd.Categorical(df['symbol_type'], categories=['spot', 'swap'], ordered=True)
    df['是否交易'] = 1
    df.loc[df['quote_volume'] < 1e-8, '是否交易'] = 0

    # 重置索引并且返回
    return symbol, df.reset_index()


def load_data(symbol_type, run_time, account_config: AccountConfig):
    """
    加载数据
    :param symbol_type: 数据类型
    :param run_time:  实盘的运行时间
    :param account_config:  账户配置
    :return:
    """
    # 获取当前目录下所有的k线文件路径
    file_list = (data_center_path / 'kline' / symbol_type).glob('*.csv')

    # 剔除掉market_info中没有的币种
    valid_symbols = account_config.bn.get_market_info(symbol_type=symbol_type).get('symbol_list', [])
    file_list = [file_path for file_path in file_list if file_path.stem in valid_symbols]
    file_list.sort()

    # 使用多线程读取
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [
            executor.submit(read_and_merge_data, account_config, _file, run_time)
            for _file in tqdm(file_list, desc=f'读取{symbol_type}数据')
        ]

        result = [future.result() for future in as_completed(futures)]

    return dict(result)


def trans_period_for_day(df, date_col='candle_begin_time', factor_dict=None):
    """
    将数据周期转换为指定的1D周期
    :param df: 原始数据
    :param date_col: 日期列
    :param factor_dict: 转换规则
    :return:
    """
    df.set_index(date_col, inplace=True)
    # 必备字段
    agg_dict = {
        'symbol': 'first',
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum',
        'quote_volume': 'sum',
        'symbol_type': 'last',
        'tag': 'first',
    }
    if factor_dict:
        agg_dict = dict(agg_dict, **factor_dict)
    df = df.resample('1D').agg(agg_dict)
    df.reset_index(inplace=True)

    return df
