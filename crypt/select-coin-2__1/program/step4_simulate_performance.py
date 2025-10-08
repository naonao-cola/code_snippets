"""
邢不行™️选币框架
Python数字货币量化投资课程

版权所有 ©️ 邢不行
微信: xbx8662

未经授权，不得复制、修改、或使用本代码的全部或部分内容。仅限个人学习用途，禁止商业用途。

Author: 邢不行
"""

import time

import pandas as pd

from config import backtest_name, backtest_path
from core.equity import calc_equity
from core.model.backtest_config import BacktestConfig, load_config
from core.utils.path_kit import get_file_path

# pandas相关的显示设置，基础课程都有介绍
pd.set_option('expand_frame_repr', False)  # 当列太多时不换行
pd.set_option('display.unicode.ambiguous_as_wide', True)  # 设置命令行输出时的列对齐功能
pd.set_option('display.unicode.east_asian_width', True)


def simulate_performance(conf, select_results, show_plot=True):
    """
    根据选出的币种模拟投资组合的表现，计算资金曲线，即投资组合的收益变化情况
    计算和作图的逻辑，在 `core` 中
    :param conf: 回测配置
    :param show_plot: 是否显示图表
    :param select_results: 选币结果
    :return:
    """
    # ====================================================================================================
    # 1. 聚合权重
    # ====================================================================================================
    s_time = time.time()
    print('ℹ️ 开始权重聚合...')
    df_spot_ratio, df_swap_ratio = agg_target_alloc_ratio(conf, select_results)
    print(f'✅ 完成权重聚合，花费时间： {time.time() - s_time:.3f}秒')
    print()

    # ====================================================================================================
    # 2. 根据选币结果计算资金曲线
    # ====================================================================================================
    if conf.is_day_period:
        print(f'🌀 开始模拟日线交易，累计回溯 {len(df_spot_ratio):,} 天...')
    else:
        print(f'🌀 开始模拟交易，累计回溯 {len(df_spot_ratio):,} 小时（~{len(df_spot_ratio) / 24:,.0f}天）...')
    print(f'ℹ️ 预计 5s 内可以完成')

    pivot_dict_spot = pd.read_pickle(get_file_path('data', 'market_pivot_spot.pkl'))
    pivot_dict_swap = pd.read_pickle(get_file_path('data', 'market_pivot_swap.pkl'))

    strategy = conf.strategy
    strategy_short = conf.strategy if conf.strategy_short is None else conf.strategy_short

    if strategy.select_scope == 'spot' and strategy_short.select_scope == 'spot':
        df_swap_ratio = pd.DataFrame(0, index=df_spot_ratio.index, columns=df_spot_ratio.columns)
    elif strategy.select_scope == 'swap' and strategy_short.select_scope == 'swap':
        df_spot_ratio = pd.DataFrame(0, index=df_swap_ratio.index, columns=df_swap_ratio.columns)
    calc_equity(conf, pivot_dict_spot, pivot_dict_swap, df_spot_ratio, df_swap_ratio, show_plot=show_plot)
    print(f'✅ 完成，回测时间：{time.time() - s_time:.3f}秒')
    print()

    return conf.report


def agg_target_alloc_ratio(conf: BacktestConfig, df_select: pd.DataFrame):
    """
    聚合target_alloc_ratio
    :param conf: 回测配置
    :param df_select: 选币结果
    :return: 聚合后的df_ratio

    数据结构:
    - index_col为candle_begin_time，
    - columns为symbol，
    - values为target_alloc_ratio的聚合结果

    示例:
                    1000BONK-USDT	1000BTTC-USDT	1000FLOKI-USDT	1000LUNC-USDT	1000PEPE-USDT	1000RATS-USDT	1000SATS-USDT	1000SHIB-USDT	1000XEC-USDT	1INCH-USDT	AAVE-USDT	ACE-USDT	ADA-USDT	    AEVO-USDT   ...
    2021/1/1 00:00	0	            0	            0	            0	            0	            0	            0	            0	            0	            0	        0	        0	        -0.083333333	0           ...
    2021/1/1 01:00	0	            0	            0	            0	            0	            0	            0	            0	            0	            0	        0	        0	        -0.083333333	0           ...
    2021/1/1 02:00	0	            0	            0	            0	            0	            0	            0	            0	            0	            0	        0	        0	        -0.083333333	0           ...
    2021/1/1 03:00	0	            0	            0	            0	            0	            0	            0	            0	            0	            0	        0	        0	        -0.083333333	0           ...
    2021/1/1 04:00	0	            0	            0	            0	            0	            0	            0	            0	            0	            0	        0	        0	        -0.083333333	0           ...
    2021/1/1 05:00	0	            0	            0	            0	            0	            0	            0	            0	            0	            0	        0	        0	        -0.083333333	0           ...
    2021/1/1 06:00	0	            0	            0	            0	            0	            0	            0	            0	            0	            0	        0	        0	        -0.083333333	0           ...
    2021/1/1 07:00	0	            0	            0	            0	            0	            0	            0	            0	            0	            0	        0	        0	        -0.083333333	0           ...
    2021/1/1 08:00	0	            0	            0	            0	            0	            0	            0	            0	            0	            0	        0	        0	        -0.083333333	0           ...
    2021/1/1 09:00	0	            0	            0	            0	            0	            0	            0	            0	            0	            0	        0	        0	        -0.083333333	0           ...
    """
    # 构建candle_begin_time序列
    start_date = df_select['candle_begin_time'].min()
    end_date = df_select['candle_begin_time'].max()
    candle_begin_times = pd.date_range(start_date, end_date, freq=conf.hold_period_type, inclusive='both')

    # 转换spot和swap的选币数据为透视表，以candle_begin_time为index，symbol为columns，values为target_alloc_ratio的sum
    # 转换为仓位比例，index 为时间，columns 为币种，values 为比例的求和
    df_ratio = df_select.pivot_table(  # 这里如果同时做多又做空的选币权重，会被聚合到一起（因子资金使用率可能存在打不满的情况）
        index='candle_begin_time', columns='symbol', values='target_alloc_ratio', aggfunc='sum')

    # 重新填充为完整的小时级别数据
    df_ratio = df_ratio.reindex(candle_begin_times, fill_value=0)

    # 多offset的权重聚合
    df_spot_ratio = df_ratio.rolling(conf.strategy.hold_period, min_periods=1).sum()

    if conf.strategy_short is not None:
        df_swap_short = df_ratio.rolling(conf.strategy_short.hold_period, min_periods=1).sum()
    else:
        df_swap_short = df_spot_ratio

    return df_spot_ratio, df_swap_short


if __name__ == '__main__':
    # 从配置文件中读取并初始化回测配置
    backtest_config = load_config()

    _results = pd.read_pickle(get_file_path(backtest_path, backtest_name, 'select_result.pkl'))

    simulate_performance(backtest_config, _results)
