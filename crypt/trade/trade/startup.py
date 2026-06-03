"""
邢不行™️选币实盘框架
Python数字货币量化投资课程

版权所有 ©️ 邢不行
微信: xbx8662

未经授权，不得复制、修改、或使用本代码的全部或部分内容。仅限个人学习用途，禁止商业用途。

Author: 邢不行
"""
import gc
import os
import os.path
import shutil
import sys
import time
import traceback
import warnings
from datetime import datetime, timedelta

import pandas as pd
from pandas import DataFrame

from config import is_debug, error_webhook_url, utc_offset, runtime_folder, global_time_offset_minutes, first_run_immediate
from core.model.account_config import AccountConfig, load_config, load_accounts_config, validate_accounts_config, print_accounts_summary
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

# 全局变量：跟踪是否是第一次运行
is_first_run = False


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
    
    # 加载所有账户配置
    accounts = load_accounts_config()
    if not accounts:
        print('❌ 没有可用的账户配置，跳过统计')
        return
    
    # 为每个账户运行统计
    for i, account in enumerate(accounts, 1):
        print(f'📊 统计账户 {i}/{len(accounts)}: {account.name}')
        
        try:
            # 运行单个账户的统计
            run_single_account_statistics(account, run_time)
            print(f'   ✅ {account.name} 统计完成')
        except Exception as e:
            print(f'   ❌ {account.name} 统计失败: {e}')
    
    print('✅统计运行结束\n')


def run_single_account_statistics(account: 'AccountConfig', run_time):
    """
    为单个账户运行统计
    """
    try:
        # 直接调用statistics模块的run_for_account函数
        from core.utils.statistics import run_for_account
        run_for_account(account, run_time)
        print(f'✅ 账户 {account.name} 统计完成')
    except Exception as e:
        print(f'❌ 账户 {account.name} 统计异常: {e}')
        print(traceback.format_exc())


def main() -> datetime | None:
    """
    多账户循环执行函数♾️
    :return: 成功执行的话，返回本次运行的时间，否则是None
    """
    global is_first_run
    print('=' * 36, '🚀多账户循环开始', '=' * 36)

    # ====================================================================================================
    # 0. 调试相关配置区域
    # ====================================================================================================
    if is_debug:  # 调试模式，不进行sleep，直接继续往后运行
        run_time = next_run_time('1h', 0) - timedelta(hours=1)
        if run_time > datetime.now():
            run_time -= timedelta(hours=1)
        # 应用全局时间偏移量
        run_time += timedelta(minutes=global_time_offset_minutes)
    elif is_first_run and first_run_immediate:  # 第一次运行且配置为立即执行
        print('🚀 第一次运行，立即执行（不等待整点）')
        run_time = datetime.now().replace(second=0, microsecond=0)
        # 应用全局时间偏移量
        run_time += timedelta(minutes=global_time_offset_minutes)
        is_first_run = False  # 标记已经不是第一次运行
    else:
        run_time = sleep_until_run_time('1h', if_sleep=True)
        # 应用全局时间偏移量
        run_time += timedelta(minutes=global_time_offset_minutes)
        is_first_run = False  # 标记已经不是第一次运行

    # ====================================================================================================
    # 1. 加载和验证多账户配置
    # ====================================================================================================
    print(f'📋 加载多账户配置...')
    
    # 验证配置
    is_valid, errors = validate_accounts_config()
    if not is_valid:
        print('❌ 账户配置验证失败:')
        for error in errors:
            print(f'   {error}')
        return None
    
    # 加载账户
    accounts = load_accounts_config()
    if not accounts:
        print('❌ 没有可用的账户配置')
        return None
    
    print(f'✅ 成功加载 {len(accounts)} 个账户')
    
    # 更新所有账户信息
    print(f'🔄 更新账户信息...')
    for account in accounts:
        try:
            account.update_account_info()
            print(f'   ✅ {account.name}: 账户信息更新完成')
        except Exception as e:
            print(f'   ❌ {account.name}: 账户信息更新失败 - {e}')

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
    # 3. 打印账户配置摘要
    # ====================================================================================================
    print_accounts_summary(accounts)

    # ====================================================================================================
    # 4. 多账户顺序执行策略
    # ====================================================================================================
    execution_results = []
    
    for i, account in enumerate(accounts, 1):
        print(f'\n{"=" * 20} 执行账户 {i}/{len(accounts)}: {account.name} {"=" * 20}')
        
        try:
            # 计算账户的实际执行时间（考虑时间偏移量）
            account_run_time = run_time + timedelta(minutes=account.time_offset_minutes)
            print(f'📅 账户执行时间: {account_run_time.strftime("%Y-%m-%d %H:%M:%S")} (偏移 {account.time_offset_minutes} 分钟)')
            
            # 如果有时间偏移，等待到指定时间
            if account.time_offset_minutes > 0 and not is_debug:
                wait_seconds = (account_run_time - datetime.now()).total_seconds()
                if wait_seconds > 0:
                    print(f'⏰ 等待 {wait_seconds:.1f} 秒到账户执行时间...')
                    time.sleep(wait_seconds)
            
            # 执行账户策略
            start_time = datetime.now()
            run_by_account(account, account_run_time)
            end_time = datetime.now()
            
            execution_time = (end_time - start_time).total_seconds()
            execution_results.append({
                'account_name': account.name,
                'status': 'success',
                'execution_time': execution_time,
                'start_time': start_time,
                'end_time': end_time,
                'error': None
            })
            
            print(f'✅ {account.name} 执行完成，耗时 {execution_time:.2f} 秒')
            
        except Exception as e:
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds() if 'start_time' in locals() else 0
            
            execution_results.append({
                'account_name': account.name,
                'status': 'failed',
                'execution_time': execution_time,
                'start_time': start_time if 'start_time' in locals() else None,
                'end_time': end_time,
                'error': str(e)
            })
            
            print(f'❌ {account.name} 执行失败: {e}')
            # 发送错误通知
            if error_webhook_url:
                try:
                    send_wechat_work_msg(f'账户 {account.name} 执行失败: {e}', error_webhook_url)
                except:
                    pass
        
        # 强制垃圾回收
        gc.collect()

    # ====================================================================================================
    # 5. 打印所有账户执行摘要
    # ====================================================================================================
    print(f'\n{"=" * 50} 执行摘要 {"=" * 50}')
    print(f'📊 总账户数: {len(accounts)}')
    
    success_count = sum(1 for result in execution_results if result["status"] == "success")
    failed_count = len(execution_results) - success_count
    
    print(f'✅ 成功: {success_count} 个账户')
    print(f'❌ 失败: {failed_count} 个账户')
    
    total_execution_time = sum(result["execution_time"] for result in execution_results)
    print(f'⏱️ 总执行时间: {total_execution_time:.2f} 秒')
    
    print(f'\n📋 详细执行结果:')
    for result in execution_results:
        status_icon = '✅' if result['status'] == 'success' else '❌'
        print(f'   {status_icon} {result["account_name"]}: {result["status"]} ({result["execution_time"]:.2f}s)')
        if result['error']:
            print(f'      错误: {result["error"]}')

    # ====================================================================================================
    # 6. 针对账户配置，统计下单信息，账户资金，策略表现，并发送资金曲线
    # ====================================================================================================
    if is_debug:
        print(f'🐞[DEBUG] - 跳过统计\n')
    else:
        run_statistics(run_time)

    # ====================================================================================================
    # 7. 运行统计
    # ====================================================================================================
    print(f'\n\n🎉 多账户运行完成，运行时间：{run_time}\n\n')

    # 本次循环结束
    print('=' * 36, '🏁循环结束', '=' * 36)
    print('⏳23秒后进入下一次循环')
    time.sleep(23)

    return run_time


if __name__ == '__main__':
    if is_debug:
        print('🟠' * 17, f'多账户调试模式', '🟠' * 17)
    else:
        print('🟢' * 17, f'多账户正式模式', '🟢' * 17)
    print(f'📢系统启动中，稍等...')
    time.sleep(1.5)

    # ====================================================================================================
    # 运行前自检和准备
    # ====================================================================================================
    # 验证多账户配置
    print('🔍 验证多账户配置...')
    is_valid, errors = validate_accounts_config()
    if not is_valid:
        print('❌ 多账户配置验证失败:')
        for error in errors:
            print(f'   {error}')
        if error_webhook_url:
            send_wechat_work_msg(f'多账户配置验证失败: {"; ".join(errors)}', error_webhook_url)
        exit(1)
    
    # 初始化所有账户
    print('🔧 初始化所有账户...')
    accounts = load_accounts_config()
    if not accounts:
        print('❌ 没有可用的账户配置')
        if error_webhook_url:
            send_wechat_work_msg('没有可用的账户配置', error_webhook_url)
        exit(1)
    
    # 自检所有账户
    if is_debug:
        print('🐞[DEBUG] - 跳过自检')
    else:
        check_failed_accounts = []
        for account in accounts:
            try:
                print(f'🔍 检查账户: {account.name}')
                if not account.is_api_ok():
                    print(f'   🚨交易所API不可用，跳过自检')
                    continue
                    
                print(f'   检查杠杆、持仓模式、保证金模式...')
                refresh_diff_time()  # 刷新与交易所的时差
                account.bn.reset_max_leverage()  # 检查杠杆
                account.bn.set_single_side_position()  # 检查并且设置持仓模式：单向持仓
                account.bn.set_multi_assets_margin()  # 检查联合保证金模式
                print(f'   ✅ {account.name} 自检完成')
                time.sleep(2)  # 多账户之间，停顿一下
            except Exception as e:
                print(f'   ❌ {account.name} 自检失败: {e}')
                check_failed_accounts.append(f'{account.name}: {e}')
        
        if check_failed_accounts:
            error_msg = f'以下账户自检失败: {"; ".join(check_failed_accounts)}'
            print(f'❌ {error_msg}')
            if error_webhook_url:
                send_wechat_work_msg(error_msg, error_webhook_url)
        else:
            print(f'✅ 所有 {len(accounts)} 个账户自检完成')
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
