"""
邢不行™️选币框架
Python数字货币量化投资课程

版权所有 ©️ 邢不行
微信: xbx8662

未经授权，不得复制、修改、或使用本代码的全部或部分内容。仅限个人学习用途，禁止商业用途。

Author: 邢不行
"""
import time
import warnings

import pandas as pd

from core.model.backtest_config import create_factory
from program.step2_calculate_factors import calc_factors
from program.step3_select_coins import select_coins
from program.step4_simulate_performance import simulate_performance

# ====================================================================================================
# ** 脚本运行前配置 **
# 主要是解决各种各样奇怪的问题们
# ====================================================================================================
warnings.filterwarnings('ignore')  # 过滤一下warnings，不要吓到老实人

# pandas相关的显示设置，基础课程都有介绍
pd.set_option('expand_frame_repr', False)  # 当列太多时不换行
pd.set_option('display.unicode.ambiguous_as_wide', True)  # 设置命令行输出时的列对齐功能
pd.set_option('display.unicode.east_asian_width', True)


def find_best_params(factory):
    """
    寻找最优参数
    :return:
    """
    # ====================================================================================================
    # 1. 准备工作
    # ====================================================================================================
    print('参数遍历开始', '*' * 64)

    conf_list = factory.config_list
    for index, conf in enumerate(conf_list):
        print(f'参数组合{index + 1}｜共{len(conf_list)}')
        print(f'{conf.get_fullname()}')
        print()
    print('✅ 一共需要回测的参数组合数：{}'.format(len(conf_list)))
    print()

    # ====================================================================================================
    # 2. 读取回测所需数据，并做简单的预处理
    # ====================================================================================================
    # 读取数据
    # prepare_data()

    # ====================================================================================================
    # 3. 计算因子
    # ====================================================================================================
    dummy_conf_with_all_factors = factory.generate_all_factor_config()  # 生成一个conf，拥有所有策略的因子
    # 然后用这个配置计算的话，我们就能获得所有策略的因子的结果，存储在 `data/cache/all_factors_df.pkl`
    calc_factors(dummy_conf_with_all_factors)

    # ====================================================================================================
    # 4. 选币
    # - 注意：选完之后，每一个策略的选币结果会被保存到硬盘
    # ====================================================================================================
    reports = []
    for config in factory.config_list:
        select_results = select_coins(config)
        report = simulate_performance(config, select_results, show_plot=False)
        reports.append(report)

    return reports


if __name__ == '__main__':
    print(f'🌀 系统启动中，稍等...')
    r_time = time.time()
    # ====================================================================================================
    # 1. 配置需要遍历的参数
    # ====================================================================================================
    # 因子遍历的参数范围
    strategies = []
    for param in range(3, 18, 2):
        strategy = {
            "hold_period": "1D",  # 持仓周期，可以是H小时，或者D天。例如：1H，8H，24H，1D，3D，7D...
            "long_select_coin_num": 3,  # 多头选币数量，可为整数或百分比。2 表示 2 个，10 / 100 表示前 10%
            "short_select_coin_num": 'long_nums',  # 空头选币数量。除和多头相同外，还支持 'long_nums' 表示与多头数量一致。
            # 注意：在is_pure_long = True时，short_select_coin_num参数无效

            "factor_list": [  # 选币因子列表
                # 因子名称（与 factors 文件中的名称一致），排序方式（True 为升序，从小到大排，False 为降序，从大到小排），因子参数，因子权重
                ('PriceMean', True, param, 1),
                # 可添加多个选币因子
                # ('PctChange', False, 7, 1),
            ],
            "filter_list": [  # 过滤因子列表
                # 因子名称（与 factors 文件中的名称一致），因子参数，因子过滤规则，排序方式
                ('QuoteVolumeMean', param, 'pct:<0.5', True),

                # ** 因子过滤规则说明 **
                # 支持三种过滤规则：`rank` 排名过滤、`pct` 百分比过滤、`val` 数值过滤
                # - `rank:<10` 仅保留前 10 名的数据；`rank:>=10` 排除前 10 名。支持 >、>=、<、<=、==、!=
                # - `pct:<0.8` 仅保留前 80% 的数据；`pct:>=0.8` 仅保留后 20%。支持 >、>=、<、<=、==、!=
                # - `val:<0.1` 仅保留小于 0.1 的数据；`val:>=0.1` 仅保留大于等于 0.1 的数据。支持 >、>=、<、<=、==、!=
                # 可添加多个过滤因子和规则，多个条件将取交集
                # ('PctChange', 7, 'pct:>0.9', True),
            ],
        }
        strategies.append(strategy)

    # ====================================================================================================
    # 2. 生成策略配置
    # ====================================================================================================
    print(f'🌀 生成策略配置...')
    backtest_factory = create_factory(strategies)

    # ====================================================================================================
    # 3. 寻找最优参数
    # ====================================================================================================
    report_list = find_best_params(backtest_factory)

    # ====================================================================================================
    # 6. 根据回测参数列表，展示最优参数
    # ====================================================================================================
    s_time = time.time()
    print(f'🌀 展示最优参数...')
    all_params_map = pd.concat(report_list, ignore_index=True)
    report_columns = all_params_map.columns  # 缓存列名

    # 合并参数细节
    sheet = backtest_factory.get_name_params_sheet()
    all_params_map = all_params_map.merge(sheet, left_on='param', right_on='fullname', how='left')

    # 按照累积净值排序，并整理结果
    all_params_map.sort_values(by='累积净值', ascending=False, inplace=True)
    all_params_map = all_params_map[[*sheet.columns, *report_columns]].drop(columns=['param'])
    all_params_map.to_excel(backtest_factory.result_folder / f'最优参数.xlsx', index=False)
    print(all_params_map)
    print(f'✅ 完成展示最优参数，花费时间：{time.time() - s_time:.3f}秒，累计时间：{(time.time() - r_time):.3f}秒')
    print()
