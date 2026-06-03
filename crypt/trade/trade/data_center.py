"""
邢不行™️选币实盘框架
Python数字货币量化投资课程

版权所有 ©️ 邢不行
微信: xbx8662

未经授权，不得复制、修改、或使用本代码的全部或部分内容。仅限个人学习用途，禁止商业用途。

Author: 邢不行
"""
import os
import traceback
import warnings
from datetime import datetime, timedelta

import pandas as pd

from config import *
from core.utils.commons import sleep_until_run_time, next_run_time
from core.utils.dingding import send_wechat_work_msg
from core.utils.functions import create_finish_flag

warnings.filterwarnings('ignore')
pd.set_option('display.max_rows', 1000)
pd.set_option('expand_frame_repr', False)  # 当列太多时不换行
pd.set_option('display.unicode.ambiguous_as_wide', True)  # 设置命令行输出时的列对齐功能
pd.set_option('display.unicode.east_asian_width', True)

# 获取脚本文件的路径
script_path = os.path.abspath(__file__)

# 提取文件名
script_filename = os.path.basename(script_path).split('.')[0]


def exec_one_job(job_file, method='download', param=''):
    wrong_signal = 0
    # =加载脚本
    cls = __import__('data_job.%s' % job_file, fromlist=('',))

    print(f'▶️调用 `{job_file}.py` 的 `{method}` 方法')
    # =执行download方法，下载数据
    try:
        if param:  # 指定有参数的方法
            getattr(cls, method)(param)
        else:  # 指定没有参数的方法
            getattr(cls, method)()
    except KeyboardInterrupt:
        print(f'ℹ️退出')
        exit()
    except BaseException as e:
        _msg = f'{job_file}  {method} 任务执行错误：' + str(e)
        print(_msg)
        print(traceback.format_exc())
        send_wechat_work_msg(_msg, error_webhook_url)
        wrong_signal += 1

    return wrong_signal


def exec_jobs(job_files, method='download', param=''):
    """
    执行所有job脚本中指定的函数
    :param job_files: 脚本名
    :param method: 方法名
    :param param: 方法参数
    """
    wrong_signal = 0

    # ===遍历job下所有脚本
    for job_file in job_files:
        wrong_signal += exec_one_job(job_file, method, param)

    return wrong_signal


def run_loop():
    print('=' * 32, '🚀更新数据开始', '=' * 32)
    # ====================================================================================================
    # 0. 调试相关配置区域
    # ====================================================================================================
    # sleep直到该小时开始。但是会随机提前几分钟。
    if not is_debug:  # 非调试模式，需要正常进行sleep
        run_time = sleep_until_run_time('1h', if_sleep=True)  # 每小时运行
    else:  # 调试模式，不进行sleep，直接继续往后运行
        run_time = next_run_time('1h', 0) - timedelta(hours=1)
        if run_time > datetime.now():
            run_time -= timedelta(hours=1)

    # =====执行job目录下脚本
    # 按照填写的顺序执行
    job_files = ['kline']
    # 执行所有job脚本中的 download 方法
    signal = exec_jobs(job_files, method='download', param=run_time)

    # 定期清理文件中重复数据(目前的配置是：周日0点清理重复的数据)
    if run_time.isoweekday() == 7 and run_time.hour == 0 and run_time.minute == 0:  # 1-7表示周一到周日，0-23表示0-23点
        # ===执行所有job脚本中的 clean_data 方法
        exec_jobs(job_files, method='clean_data')

    # 生成指数完成标识文件。如果标记文件过多，会删除7天之前的数据
    create_finish_flag(flag_path, run_time, signal)

    # =====清理数据
    del job_files

    # 本次循环结束
    print('=' * 32, '🏁更新数据完成', '=' * 32)
    print('⏳59秒后进入下一次循环')
    time.sleep(59)

    return run_time


if __name__ == '__main__':
    if is_debug:
        print('🟠' * 17, f'调试模式', '🟠' * 17)
    else:
        print('🟢' * 17, f'正式模式', '🟢' * 17)
    while True:
        try:
            run_loop()
        except Exception as err:
            msg = '系统出错，10s之后重新运行，出错原因: ' + str(err)
            print(msg)
            print(traceback.format_exc())
            send_wechat_work_msg(msg, error_webhook_url)
            time.sleep(10)  # 休息十秒钟，再冲
