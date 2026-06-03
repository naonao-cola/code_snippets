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
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from glob import glob

import numpy as np
import pandas as pd
from tqdm import tqdm

from config import data_center_path, special_symbol_dict, download_kline_list
from core.binance.base_client import BinanceClient
from core.utils.path_kit import get_folder_path, get_file_path
from data_job.funding_fee import load_funding_fee

# 提取文件名
script_filename = os.path.basename(os.path.abspath(__file__)).split('.')[0]

# 首次初始化需要多少k线数据，根据你策略因子需要调整
init_kline_num = 1500

# 根据持仓周期自动调整获取1小时的k线
interval = '1h'

# 获取交易所对象
cli = BinanceClient.get_dummy_client()


# ====================================================================================================
# ** 辅助函数区域 **
# - 建议跳过
# ====================================================================================================
def has_spot():
    return 'SPOT' in download_kline_list or 'spot' in download_kline_list


def has_swap():
    return 'SWAP' in download_kline_list or 'swap' in download_kline_list


def add_swap_tag(spot_df, swap_df):
    spot_df['tag'] = 'NoSwap'
    if swap_df is None or swap_df.empty:
        return spot_df

    cond1 = spot_df['candle_begin_time'] > swap_df.iloc[0]['candle_begin_time']
    cond2 = spot_df['candle_begin_time'] <= swap_df.iloc[-1]['candle_begin_time']
    spot_df.loc[cond1 & cond2, 'tag'] = 'HasSwap'
    return spot_df


def export_to_csv(df, symbol, symbol_type):
    # ===在data目录下创建当前脚本存放数据的目录
    save_path = get_folder_path(data_center_path, script_filename, symbol_type)

    # =构建存储文件的路径
    _file_path = os.path.join(save_path, f'{symbol}.csv')

    # =判断文件是否存在
    if_file_exists = os.path.exists(_file_path)
    # =保存数据
    # 路径存在，数据直接追加
    if if_file_exists:
        # df = df[-10:]  # 指定最新的10条数据去追加(如果你的数据停了很久怎么办？直接删了data_center重跑吧)
        df[-10:].to_csv(_file_path, encoding='gbk', index=False, header=False, mode='a')
        time.sleep(0.05)
    else:
        df.to_csv(_file_path, encoding='gbk', index=False)
        # sleep休息一会儿
        time.sleep(0.3)


def fetch_data_by_symbol(symbol, symbol_type, run_time, swap_funding_df=None):
    save_file_path = get_file_path(data_center_path, script_filename, symbol_type, f'{symbol}.csv')

    # 获取k线请求的数量设置
    # - 如果存在目录，表示已经有文件存储，
    # - 默认获取99根k线(为什么是99?拍脑袋决定的。这里你可以调整，不建议超过99，会增加请求权重)
    # - 如果不存在目录，表示首次运行，获取`init_kline_num`根k线
    kline_limit = 99 if os.path.exists(save_file_path) else init_kline_num

    # =获取k线数据
    df = cli.get_candle_df(symbol, run_time, kline_limit, interval=interval, symbol_type=symbol_type)

    # 返回None或者空的df，不放到result里
    if df is None or df.empty:
        return None

    # =合并资金费的数据
    if symbol_type == 'spot':
        df['fundingRate'] = np.nan
    else:
        # 将数据合并到k线上
        if swap_funding_df is None:
            df['fundingRate'] = np.nan
        else:
            df = pd.merge(
                df, swap_funding_df[['fundingTime', 'fundingRate']], left_on=['candle_begin_time'],
                right_on=['fundingTime'], how='left')
            del df['fundingTime']
        # 返回None或者空的df，跳过
    return df


def process_by_spot_swap_symbol(spot_symbol, swap_symbol, run_time, swap_funding_df):
    # 先更新swap数据
    with ThreadPoolExecutor() as executor:
        # 筛选当前币种的最新资金费数据
        future_swap = executor.submit(
            fetch_data_by_symbol, swap_symbol, 'swap', run_time, swap_funding_df
        ) if swap_symbol else None

        future_spot = executor.submit(
            fetch_data_by_symbol, spot_symbol, 'spot', run_time, None) if spot_symbol else None

    swap_df = future_swap.result() if future_swap else None
    spot_df = future_spot.result() if future_spot else None

    if swap_df is not None:
        swap_df['tag'] = 'NoSwap'
        export_to_csv(swap_df, swap_symbol, 'swap')

    if spot_df is not None:
        spot_df = add_swap_tag(spot_df, swap_df)
        export_to_csv(spot_df, spot_symbol, 'spot')


def upgrade_spot_has_swap(spot_symbol, swap_symbol):
    # 先更新swap数据
    swap_df = None
    if swap_symbol:
        swap_filepath = get_file_path(data_center_path, script_filename, 'swap', swap_symbol)
        if os.path.exists(swap_filepath):
            swap_df = pd.read_csv(swap_filepath, encoding='gbk', parse_dates=['candle_begin_time'])
            swap_df['tag'] = 'NoSwap'
            export_to_csv(swap_df, swap_symbol, 'swap')

    if spot_symbol:
        spot_filepath = get_file_path(data_center_path, script_filename, 'spot', spot_symbol)
        if os.path.exists(spot_filepath):
            spot_df = pd.read_csv(
                spot_filepath, encoding='gbk',
                parse_dates=['candle_begin_time']
            ) if spot_symbol else None
            spot_df = add_swap_tag(spot_df, swap_df)
            export_to_csv(spot_df, spot_symbol, 'spot')
    print(f'✅{spot_symbol} / {swap_symbol} updated')


# ====================================================================================================
# ** 数据中心功能函数 **
# ====================================================================================================
def download(run_time):
    """
    k线数据更新主程序，主要逻辑如下
    :param run_time:    运行时间
    """
    print(f'执行{script_filename}脚本 download 开始')

    # ====================================================================================================
    # 1. ** 初始化变量 **
    # ====================================================================================================
    _time = datetime.now()  #

    print(f'(1/4) 获取交易对...')
    # =获取U本位合约交易对的信息
    if has_swap():
        swap_market_info = cli.get_market_info(symbol_type='swap', require_update=True)
        swap_symbol_list = swap_market_info.get('symbol_list', [])
    else:
        swap_symbol_list = []
    # =加载现货交易对信息
    if has_spot():
        spot_market_info = cli.get_market_info(symbol_type='spot', require_update=True)
        spot_symbol_list = spot_market_info.get('symbol_list', [])
    else:
        spot_symbol_list = []

    # ====================================================================================================
    # 2. ** 加载本地的资金费率数据 **
    # ====================================================================================================
    print(f'(2/4) 读取历史资金费率...')
    last_funding_df = load_funding_fee()

    # ====================================================================================================
    # 3. ** 合并spot和swap数据 **
    # - 结果为 [('BTCUSDT', 'BTCUSDT'), (None, '1000SATSUSDT'), ...]
    # ====================================================================================================
    print(f'(3/4) 合并计算交易对...')
    same_symbols = set(spot_symbol_list) & set(swap_symbol_list)  # join，取交集
    all_symbols = set(spot_symbol_list) | set(swap_symbol_list)  # union，取并集

    # 3.0 预处理一下special symbol，加上usdt的尾缀，原始配置只有币种名称
    if has_spot() and has_swap():
        special_symbol_with_usdt_dict = {
            f'{_spot}USDT'.upper(): f'{_special_swap}USDT'.upper() for _spot, _special_swap in
            special_symbol_dict.items()
        }
    else:
        special_symbol_with_usdt_dict = {}

    # 3.1 组合相同币种，
    # - 比如 spot是 BTCUSDT，swap是BTCUSDT，
    # - 结果为 (BTCUSDT. BTCUSDT)
    symbol_pair_list1 = [(_spot, _spot) for _spot in same_symbols]

    # 3.2 组合config中special_symbol_dict配置的特殊币种，
    # - 比如 spot是 DODOUSDT，swap是DODOXUSDT，
    # - 结果为 (DODOUSDT. DODOXUSDT)
    symbol_pair_list2 = [(_spot, _swap) for _spot, _swap in special_symbol_with_usdt_dict.items()]

    # 3.3 组合swap相比于spot前面多了1000的币种，
    # - 比如 spot是 FLOKIUSDT，swap列表中存在1000FLOKIUSDT，但是swap不存在 FLOKIUSDT
    # - 结果为 (FLOKIUSDT. 1000FLOKIUSDT)
    symbol_pair_list3 = []
    for _spot in spot_symbol_list:
        _special_swap = f'1000{_spot}'
        if _special_swap in swap_symbol_list:
            symbol_pair_list3.append((_spot, _special_swap))
            special_symbol_with_usdt_dict[_spot] = _special_swap  # 缓存到special中，为了简化3.4

    # 3.4 组合剩下的币种，这些要么只有spot或者只有swap，
    # - 比如 spot是 AMPUSDT，没有swap，结果就是 (AMPUSDT, None)
    # - 又比如 swap是 BSVUSDT，没有spot，结果就是 (None, BSVUSDT)
    symbol_pair_list4 = [
        (None, _symbol) if _symbol in swap_symbol_list else (_symbol, special_symbol_with_usdt_dict.get(_symbol, None))
        for
        _symbol in all_symbols if
        # 去掉 3.1、3.2、3.3
        _symbol not in [*same_symbols, *special_symbol_with_usdt_dict.keys(), *special_symbol_with_usdt_dict.values()]
    ]
    symbol_pair_list = symbol_pair_list1 + symbol_pair_list2 + symbol_pair_list3 + symbol_pair_list4

    has_swap_check = get_file_path(data_center_path, 'kline-has-swap.txt')
    if not os.path.exists(has_swap_check):
        print('⚠️开始更新数据缓存文件，添加HasSwap的tag...')
        for spot_symbol, swap_symbol in symbol_pair_list:
            upgrade_spot_has_swap(spot_symbol, swap_symbol)
        with open(has_swap_check, 'w') as f:
            f.write('HasSwap')
            f.close()

    # ====================================================================================================
    # 4. ** 开始更新数据 **
    # ====================================================================================================
    print(f'(4/4) 开始更新数据...')
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = []
        for spot_symbol, swap_symbol in symbol_pair_list:
            # 筛选当前币种的最新资金费数据
            swap_funding_df = last_funding_df[last_funding_df['symbol'] == swap_symbol].copy() if (
                    swap_symbol and last_funding_df is not None) else None
            # 提交并行任务
            future = executor.submit(
                process_by_spot_swap_symbol, spot_symbol, swap_symbol, run_time, swap_funding_df
            )
            futures.append(future)

        # 等待并行结束，如果有异常输出到日志
        for future in tqdm(as_completed(futures), total=len(futures), desc='更新数据'):
            try:
                future.result()
            except Exception as e:
                print(f"❌An error occurred: {e}")
                print(traceback.format_exc())

    # 生成当前k线下载数据的时间戳文件
    with open(get_file_path(data_center_path, script_filename, 'kline-download-time.txt'), 'w') as f:
        f.write(run_time.strftime('%Y-%m-%d %H:%M:%S'))
        f.close()

    print(f'✅执行{script_filename}脚本 download 完成。({datetime.now() - _time}s)')


def clear_duplicates(file_path):
    # 文件存在，去重之后重新保存
    if os.path.exists(file_path):
        df = pd.read_csv(file_path, encoding='gbk', parse_dates=['candle_begin_time'])  # 读取本地数据
        df.drop_duplicates(subset=['candle_begin_time'], keep='last', inplace=True)  # 去重保留最新的数据
        df.sort_values('candle_begin_time', inplace=True)  # 通过candle_begin_time排序
        df = df[-init_kline_num:]  # 保留最近2400根k线，防止数据堆积过多(2400根，大概100天数据)
        df.to_csv(file_path, encoding='gbk', index=False)  # 保存文件


def clean_data():
    """
    根据获取数据的情况，自行编写清理冗余数据函数
    """
    print(f'ℹ️执行{script_filename}脚本 clear_duplicates 开始')
    _time = datetime.now()
    # 遍历合约和现货目录
    for symbol_type in ['swap', 'spot']:
        # 获取目录路径
        save_path = os.path.join(data_center_path, script_filename, symbol_type)
        # 获取.csv结尾的文件目录
        file_list = glob(get_file_path(save_path, '*.csv'))
        # 遍历文件进行操作
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(clear_duplicates, _file) for _file in file_list]

            for future in tqdm(as_completed(futures), total=len(futures), desc='清理冗余数据'):
                try:
                    future.result()
                except Exception as e:
                    print(f"❌An error occurred: {e}")
                    print(traceback.format_exc())

    print(f'✅执行{script_filename}脚本 clear_duplicates 完成 {datetime.now() - _time}')


if __name__ == '__main__':
    download(datetime.now().replace(minute=0))
