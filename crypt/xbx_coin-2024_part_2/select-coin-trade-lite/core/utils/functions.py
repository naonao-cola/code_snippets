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
import warnings
from pathlib import Path

from config import data_path
from core.binance.base_client import BinanceClient

warnings.filterwarnings('ignore')


# =====策略相关函数
def del_insufficient_data(symbol_candle_data):
    """
    删除数据长度不足的币种信息

    :param symbol_candle_data:
    :return
    """
    # ===删除成交量为0的线数据、k线数不足的币种
    symbol_list = list(symbol_candle_data.keys())
    for symbol in symbol_list:
        # 删除空的数据
        if symbol_candle_data[symbol] is None or symbol_candle_data[symbol].empty:
            del symbol_candle_data[symbol]
            continue
        # 删除该币种成交量=0的k线
        symbol_candle_data[symbol] = symbol_candle_data[symbol][symbol_candle_data[symbol]['volume'] > 0]

    return symbol_candle_data


def save_select_coin(select_coin, run_time, account_name, max_file_limit=999):
    """
    保存选币数据，最多保留999份文件
    :param select_coin: 保存文件内容
    :param run_time: 当前运行时间
    :param account_name: 账户名称
    :param max_file_limit: 最大限制
    """
    # 获取存储文件位置
    dir_path = Path(data_path) / account_name / 'select_coin'
    dir_path.mkdir(parents=True, exist_ok=True)

    # 生成文件名
    file_path = dir_path / f"{run_time.strftime('%Y-%m-%d_%H')}.pkl"
    # 保存文件
    select_coin.to_pickle(file_path)
    # 删除多余的文件
    del_hist_files(dir_path, max_file_limit, file_suffix='.pkl')


def del_hist_files(file_path, max_file_limit=999, file_suffix='.pkl'):
    """
    删除多余的文件，最限制max_file_limit
    :param file_path: 文件路径
    :param max_file_limit: 最大限制
    :param file_suffix: 文件后缀
    """
    # ===删除多余的flag文件
    files = [_ for _ in os.listdir(file_path) if _.endswith(file_suffix)]  # 获取file_path目录下所有以.pkl结尾的文件
    # 判断一下当前目录下文件是否过多
    if len(files) > max_file_limit:  # 文件数量超过最大文件数量限制，保留近999个文件，之前的文件全部删除
        print(f'ℹ️目前文件数量: {len(files)}, 文件超过最大限制: {max_file_limit}，准备删除文件')
        # 文件名称是时间命名的，所以这里倒序排序结果，距离今天时间越近的排在前面，距离距离今天时间越远的排在最后。例：[2023-04-02_08, 2023-04-02_07, 2023-04-02_06···]
        files = sorted(files, reverse=True)
        rm_files = files[max_file_limit:]  # 获取需要删除的文件列表

        # 遍历删除文件
        for _ in rm_files:
            os.remove(os.path.join(file_path, _))  # 删除文件
            print(f'✅删除文件完成:{os.path.join(file_path, _)}')


def create_finish_flag(flag_path, run_time, signal):
    """
    创建数据更新成功的标记文件
    如果标记文件过多，会删除7天之前的数据

    :param flag_path:标记文件存放的路径
    :param run_time: 当前的运行是时间
    :param signal: 信号
    """
    # ===判断数据是否完成
    if signal > 0:
        print(f'⚠️当前数据更新出现错误信号: {signal}，数据更新没有完成，当前小时不生成 flag 文件')
        return

    # ===生成flag文件
    # 指定生成文件名称
    index_config_path = flag_path / f"{run_time.strftime('%Y-%m-%d_%H_%M')}.flag"  # 例如文件名是：2023-04-02_08.flag
    # 更新信息成功，生成文件
    with open(index_config_path, 'w', encoding='utf-8') as f:
        f.write('更新完成')
        f.close()

    # ===删除多余的flag文件
    del_hist_files(flag_path, 7 * 24, file_suffix='.flag')


def refresh_diff_time():
    """刷新本地电脑与交易所的时差"""
    cli = BinanceClient.get_dummy_client()
    server_time = cli.exchange.fetch_time()  # 获取交易所时间
    diff_timestamp = int(time.time() * 1000) - server_time  # 计算时差
    BinanceClient.diff_timestamp = diff_timestamp  # 更新到全局变量中


def save_symbol_order(symbol_order, run_time, account_name):
    # 创建存储账户换仓信息文件的目录[为了计算账户小时成交量信息生成的]
    dir_path = Path(data_path)
    dir_path = dir_path / account_name / '账户换仓信息'
    dir_path.mkdir(exist_ok=True)

    filename = run_time.strftime("%Y%m%d_%H") + ".csv"
    select_symbol_list_path = dir_path / filename
    select_symbol_list = symbol_order[['symbol', 'symbol_type']].copy()
    select_symbol_list['time'] = run_time

    select_symbol_list.to_csv(select_symbol_list_path)
    del_hist_files(dir_path, 999, file_suffix='.csv')
