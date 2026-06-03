"""
邢不行™️选币实盘框架
Python数字货币量化投资课程

版权所有 ©️ 邢不行
微信: xbx8662

未经授权，不得复制、修改、或使用本代码的全部或部分内容。仅限个人学习用途，禁止商业用途。

Author: 邢不行
"""
import gc
import os.path
import shutil
import sys
import time
import traceback
import warnings
from datetime import datetime, timedelta

import pandas as pd

from config import is_debug, error_webhook_url, utc_offset, runtime_folder
from core.model.account_config import AccountConfig, load_config
from core.real_trading import save_and_merge_select
from core.utils.commons import sleep_until_run_time, next_run_time
from core.utils.datatools import check_data_update_flag
from core.utils.dingding import send_wechat_work_msg
from core.utils.functions import save_select_coin, save_symbol_order, refresh_diff_time
from core.utils.path_kit import get_file_path
from program.step1_prepare_data import prepare_data
from program.step2_calculate_factors import calc_factors
from program.step3_select_coins import select_coins

# ====================================================================================================
# ** 脚本运行前配置 **
# 主要是解决各种各样奇怪的问题们
# ====================================================================================================
warnings.filterwarnings('ignore')  # 过滤一下warnings，不要吓到老实人

# pandas相关的显示设置，基础课程都有介绍
pd.set_option('display.max_rows', 1000)
pd.set_option('expand_frame_repr', False)  # 当列太多时不换行
pd.set_option('display.unicode.ambiguous_as_wide', True)  # 设置命令行输出时的列对齐功能
pd.set_option('display.unicode.east_asian_width', True)


# ====================================================================================================
# ** 流程函数 **
# 主进程 -> run_loop -> run_by_account
# 调度逻辑在 run_loop 函数中
# 核心逻辑在 run_by_account 函数中
# 程序的主要逻辑会在这边写一下，可以试用异步的方案调用，充分释放执行过程中的内存
# ====================================================================================================
def run_by_account(account: AccountConfig, run_time: datetime):
    """
    针对当前账户，进行选币、下单操作
    :param account: config中账户配置的信息
    :param run_time:  运行时间
    :return:
    """
    # 记录一下时间戳
    r_time = time.time()

    # ====================================================================================================
    # 1. 准备工作
    # ====================================================================================================
    # 和交易所对一下表
    print('-' * 36, '账户准备', '-' * 36)
    print(f'⏳运行时间：{run_time}，当前时间：{datetime.now()}')
    refresh_diff_time()

    # 清理一下运行过程中的缓存数据
    if os.path.exists(runtime_folder):
        shutil.rmtree(runtime_folder)
    runtime_folder.mkdir(parents=True, exist_ok=True)

    # 判断当前策略持仓是否是日线
    # 如果是日线持仓则需要判断下单时间，小时级别持仓不需要判断下单时间
    if account.is_day_period and run_time.hour != utc_offset % 24:  # 只有当前小时是utc0点的时候，才会下单
        print(f'⚠️账号：{account.name}，当前不是需要下单的时间，跳过···')
        return

    # 更新一下交易所的行情数据，获取最新币种信息，放置在缓存中
    account.bn.fetch_market_info('swap')
    if account.is_pure_long:
        account.bn.fetch_market_info('spot')

    # 撤销所有币种挂单
    if is_debug:
        print(f'🐞[DEBUG] - 跳过撤单\n')
    elif not account.is_api_ok():
        print(f'🚨API未配置 - 跳过撤单\n')
    else:
        account.bn.cancel_all_swap_orders()  # 合约撤单
        account.bn.cancel_all_spot_orders()  # 现货撤单

    # ====================================================================================================
    # 2. 读取实盘所需数据，并做简单的预处理
    # ====================================================================================================
    print('-' * 36, '准备数据', '-' * 36)
    prepare_data(account, run_time)

    # ====================================================================================================
    # 3. 计算因子
    # ====================================================================================================
    print('-' * 36, '因子计算', '-' * 36)
    calc_factors(account, run_time)

    # ====================================================================================================
    # 4. 选币
    # - 注意：选完之后，每一个策略的选币结果会被保存到硬盘
    # ====================================================================================================
    print('-' * 36, '条件选币', '-' * 36)
    select_results = select_coins(account)

    # ====================================================================================================
    # 5. 保存并合并本地选币文件
    # ====================================================================================================
    # 如果选币数据为空，并且当前账号没有任何持仓，直接跳过后续操作
    print('-' * 36, '保存选币', '-' * 36)
    spot_position = account.spot_position
    swap_position = account.swap_position

    if select_results.empty and spot_position.empty and swap_position.empty:
        return
    select_results = save_and_merge_select(account, select_results)
    print(f'选币及资金分配：{select_results}\n'
          f'✅{account.name}策略计算总消耗时间：{time.time() - r_time:.2f}s，\n')

    # ====================================================================================================
    # 6. 计算下单信息
    # ====================================================================================================
    print('-' * 36, '计算下单', '-' * 36)
    if not account.is_api_ok():
        print('⚠️交易所API不可用，跳过下单')
        return

    symbol_order = account.calc_order_amount(select_results)
    print(f'ℹ️下单信息：{symbol_order}\n')

    print(f'✅{account.name}策略计算总消耗时间：{time.time() - r_time:.2f}s，准备下单...\n')

    if is_debug:
        print(f'🐞[DEBUG] - 跳过下单\n')
        return

    if symbol_order.empty:
        print(f'⚠️下单信息为空，跳过下单\n')
        return

    # ====================================================================================================
    # 7. 下单
    # ====================================================================================================
    print('-' * 36, '开始下单', '-' * 36)
    account.proceed_spot_order(symbol_order, is_only_sell=True)
    account.proceed_swap_order(symbol_order)
    account.proceed_spot_order(symbol_order, is_only_sell=False)

    # ====================================================================================================
    # 8. 记录下单数据
    # ====================================================================================================
    # 保存选币数据
    save_select_coin(select_results, run_time, account.name)

    # 保存下单数据
    save_symbol_order(symbol_order, run_time, account.name)


def run_statistics(run_time):
    print('-' * 36, '开始统计', '-' * 36)
    python_exec = sys.executable  # 获取当前环境下的python解释器，也是给统计脚本用的
    os.system(f'{python_exec} {get_file_path("core", "utils", "statistics.py")} {int(run_time.timestamp())}')
    print('✅统计运行结束\n')


def main() -> datetime | None:
    """
    无限循环这个函数♾️
    :return: 成功执行的话，返回本次运行的时间，否则是None
    """
    print('=' * 36, '🚀循环开始', '=' * 36)

    # ====================================================================================================
    # 0. 调试相关配置区域
    # ====================================================================================================
    if is_debug:  # 调试模式，不进行sleep，直接继续往后运行
        run_time = next_run_time('1h', 0) - timedelta(hours=1)
        if run_time > datetime.now():
            run_time -= timedelta(hours=1)
    else:
        run_time = sleep_until_run_time('1h', if_sleep=True)

    # ====================================================================================================
    # 1. 更新、检查账户信息
    # ====================================================================================================
    account = load_config()
    account.update_account_info()

    # ====================================================================================================
    # 2. 等待数据中心完成数据更新
    # ====================================================================================================
    print(f'ℹ️检查数据中心，目标运行时间：{run_time}')
    # 检查数据中心数据是否更新
    is_data_ready = check_data_update_flag(run_time)

    # 判断数据是否更新好了
    if not is_data_ready:  # 没有更新好，跳过当前下单
        print(f'⚠️数据中心数据未更新，跳过当前，{run_time}')
        return
    print('✅数据中心数据更新完成，准备选币下单\n')

    # ====================================================================================================
    # 3. 针对账户配置，进行因子计算、选币、下单
    # ====================================================================================================
    run_by_account(account, run_time)
    gc.collect()  # 强制垃圾回收

    # 开始进行账户统计
    # ====================================================================================================
    # 4. 针对账户配置，统计下单信息，账户资金，策略表现，并发送资金曲线
    # ====================================================================================================
    if is_debug:
        print(f'🐞[DEBUG] - 跳过统计\n')
    else:
        run_statistics(run_time)

    # 本次循环结束
    print('=' * 36, '🏁循环结束', '=' * 36)
    print('⏳23秒后进入下一次循环')
    time.sleep(23)

    return run_time


if __name__ == '__main__':
    if is_debug:
        print('🟠' * 17, f'调试模式', '🟠' * 17)
    else:
        print('🟢' * 17, f'正式模式', '🟢' * 17)
    print(f'📢系统启动中，稍等...')
    time.sleep(1.5)

    # ====================================================================================================
    # 运行前自检和准备
    # ====================================================================================================
    # 初始化账户
    account_config = load_config()
    print(account_config)

    # 自检
    if is_debug:
        print('🐞[DEBUG] - 跳过自检')
    elif not account_config.is_api_ok():
        print('🚨交易所API不可用，跳过自检')
    else:
        print(f'{account_config.name} 检查杠杆、持仓模式、保证金模式...')
        refresh_diff_time()  # 刷新与交易所的时差
        account_config.bn.reset_max_leverage()  # 检查杠杆
        account_config.bn.set_single_side_position()  # 检查并且设置持仓模式：单向持仓
        account_config.bn.set_multi_assets_margin()  # 检查联合保证金模式
        time.sleep(2)  # 多账户之间，停顿一下
        print('✅自检完成')
        time.sleep(2)

    # ====================================================================================================
    # 实盘主程序，电不停我不停
    # ====================================================================================================
    while True:
        try:
            main()  # 实盘的主进程
        except Exception as err:
            msg = '❌系统出错，10s之后重新运行，出错原因: ' + str(err)
            print(msg)
            print(traceback.format_exc())
            send_wechat_work_msg(msg, error_webhook_url)
            time.sleep(11)  # 休息十一秒钟，再冲
        finally:
            # 如果是调试模式，就不会自动无限运行
            if is_debug:
                break
